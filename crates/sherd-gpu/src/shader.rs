//! The plumbing every kernel in this crate shares: a compiled pipeline, its explicit layout, and
//! the two host-side packings the WGSL files expect.
//!
//! It is a separate module for one reason found in task G1 and repeated here: **the shader module
//! and the pipeline are built once, at executor construction, and never inside a timed region.**
//! Metal compiles a shader on first pipeline creation, and G1 §5.1 measured what happens when that
//! lands inside the measurement — a kernel that runs at 14 ns/query reported 197 ns/query and the
//! GPU looked 3.4× *slower* than the CPU. [`Kernel::build`] is therefore called from
//! [`GpuExecutor::new`](crate::GpuExecutor::new) and the per-batch path only creates buffers and
//! submits.
//!
//! The layout is always explicit (`BindGroupLayout` + `PipelineLayout`), never
//! `get_bind_group_layout`: an auto layout is exclusive to one pipeline and holds only the
//! bindings that entry point happens to use (E7 §7.3).

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, ShaderStages,
};

use crate::GpuError;
use crate::buffers::Dispatch;
use crate::device::Gpu;

/// How many buffers of each kind one bind group of this crate's kernels holds.
///
/// The wgpu defaults are the budget (D §6.8: the kernels are written to the defaults, not to what
/// this adapter happens to offer): **8 storage buffers** and 12 uniform buffers per compute stage.
/// Eight is the binding count `coarse.wgsl` and `icp.wgsl` are shaped around — it is why the
/// target points and their normals travel interleaved in one array rather than as two, and why the
/// grid's slots and counts do too.
pub const MAX_STORAGE_BINDINGS: usize = 8;

/// A compiled compute pipeline with the layout its bind groups are built against.
#[derive(Debug)]
pub struct Kernel {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
    label: &'static str,
}

impl Kernel {
    /// Compiles `source` and builds the pipeline for `entry_point`.
    ///
    /// `uniforms` binding slots come first, then `storage` slots; the WGSL file's `@binding`
    /// numbers are exactly `0..uniforms` and `uniforms..uniforms + storage.len()`. `storage` says
    /// which of those are read-only.
    ///
    /// # Panics
    ///
    /// If more than [`MAX_STORAGE_BINDINGS`] storage buffers are asked for — a portable build has
    /// eight, and a kernel that wants nine has to pack instead (D §6.8).
    #[must_use]
    pub fn build(
        gpu: &Gpu,
        label: &'static str,
        source: &str,
        entry_point: &str,
        uniforms: usize,
        storage: &[bool],
    ) -> Self {
        assert!(
            storage.len() <= MAX_STORAGE_BINDINGS,
            "{label}: {} storage buffers, the portable limit is {MAX_STORAGE_BINDINGS} (D §6.8)",
            storage.len()
        );
        let module = gpu.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let mut entries: Vec<BindGroupLayoutEntry> =
            (0..uniforms).map(|i| uniform_entry(u32::try_from(i).unwrap_or(0))).collect();
        for (i, &read_only) in storage.iter().enumerate() {
            let binding = u32::try_from(uniforms + i).unwrap_or(0);
            entries.push(storage_entry(binding, read_only));
        }
        let layout = gpu.device().create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline_layout = gpu.device().create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = gpu.device().create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout, label }
    }

    /// A bind group over `resources`, in binding order.
    #[must_use]
    pub fn bind(&self, gpu: &Gpu, resources: &[BindingResource<'_>]) -> BindGroup {
        let entries: Vec<BindGroupEntry<'_>> = resources
            .iter()
            .enumerate()
            .map(|(i, resource)| BindGroupEntry {
                binding: u32::try_from(i).unwrap_or(0),
                resource: resource.clone(),
            })
            .collect();
        gpu.device().create_bind_group(&BindGroupDescriptor {
            label: Some(self.label),
            layout: &self.layout,
            entries: &entries,
        })
    }

    /// Submits one dispatch and waits for it.
    ///
    /// `submit` + `poll(Wait)` and nothing else — the pipeline is already compiled and the buffers
    /// are already resident, which is what makes a wall-clock measurement around this call the
    /// kernel's own (E7 §7.1, G1 §5.1).
    pub fn dispatch(&self, gpu: &Gpu, bind: &BindGroup, grid: Dispatch) -> Result<(), GpuError> {
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(self.label) });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(self.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, 1);
        }
        gpu.queue().submit(Some(encoder.finish()));
        gpu.wait()
    }
}

/// A uniform block, mapped at creation.
#[must_use]
pub fn uniform<T: bytemuck::Pod>(gpu: &Gpu, label: &str, value: &T) -> Buffer {
    use wgpu::util::DeviceExt;
    gpu.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// A read-only uniform binding at `binding`.
#[must_use]
pub fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A storage binding at `binding`, read-only or read-write.
#[must_use]
pub fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// `vec3<f32>` arrays padded to the `vec4<f32>` the kernels bind (D §6.3).
#[must_use]
pub fn pad4(points: &[[f32; 3]]) -> Vec<[f32; 4]> {
    points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect()
}

/// Two `f64` arrays of the same length narrowed and interleaved into one `vec4<f32>` array:
/// element `2k` is `a[k]`, element `2k + 1` is `b[k]`.
///
/// The interleaving is not a micro-optimisation, it is the binding budget: a portable compute
/// stage has [`MAX_STORAGE_BINDINGS`] storage buffers and the coarse kernel needs the grid's three
/// arrays, the target cloud, its normals, the source cloud, its normals, the poses and one output
/// — nine as separate bindings, seven interleaved this way.
#[must_use]
#[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
pub fn interleave(a: &[[f64; 3]], b: &[[f64; 3]]) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(a.len() * 2);
    for (p, q) in a.iter().zip(b) {
        out.push([p[0] as f32, p[1] as f32, p[2] as f32, 0.0]);
        out.push([q[0] as f32, q[1] as f32, q[2] as f32, 0.0]);
    }
    out
}

/// [`interleave`] over an already-narrowed first array — the grid's own `f32` points, so that the
/// kernel queries exactly the coordinates the grid was built from.
#[must_use]
#[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
pub fn interleave_narrowed(a: &[[f32; 3]], b: &[[f64; 3]]) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(a.len() * 2);
    for (p, q) in a.iter().zip(b) {
        out.push([p[0], p[1], p[2], 0.0]);
        out.push([q[0] as f32, q[1] as f32, q[2] as f32, 0.0]);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "a device packing is exact or it is wrong")]

    use super::{interleave, interleave_narrowed, pad4};

    /// The two packings are the layouts the WGSL files index: `2k` and `2k + 1`.
    #[test]
    fn the_interleaved_layout_is_point_then_normal() {
        let p = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let n = vec![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let packed = interleave(&p, &n);
        assert_eq!(packed.len(), 4);
        assert_eq!(packed[0], [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(packed[1], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(packed[2], [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(packed[3], [0.0, 1.0, 0.0, 0.0]);

        let narrowed = vec![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(interleave_narrowed(&narrowed, &n), packed);
        assert_eq!(pad4(&narrowed)[1], [4.0, 5.0, 6.0, 0.0]);

        // A shorter second array truncates rather than panics: the caller has checked the lengths
        // and the zip is what enforces it.
        assert_eq!(interleave(&p, &n[..1]).len(), 2);
    }
}
