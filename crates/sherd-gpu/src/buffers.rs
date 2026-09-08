//! Buffer creation, batch sizing and dispatch shape (D §6.3, D §6.4).
//!
//! Four rules, three of them measured rather than assumed.
//!
//! * **Batches are chunked to the storage-buffer binding cap.** D §6.3 sizes a batch so that the
//!   grids plus the candidates fit in one binding; the *portable* cap is the wgpu default of
//!   128 MB, not the 4 GiB this Metal adapter happens to offer (E7 §2). [`Chunking`] computes the
//!   split, and it is arithmetic, so it is tested everywhere.
//! * **Dispatches are 2-D beyond 65 535 workgroups.** `max_compute_workgroups_per_dimension` is
//!   65 535 both on this adapter and in the wgpu defaults; E7 §2 hit the wall at 96 000 workgroups
//!   and had to recompute the linear index as `gid.x + gid.y · wg_x`. [`Dispatch`] is that
//!   fallback, and every kernel in this crate takes its `wg_x` as a uniform rather than deriving
//!   it, because the CPU mirror has to agree on the shape (E7 §3).
//! * **Uploads are mapped at creation on unified memory, staged on discrete.** E7 §2 measured
//!   4.6–4.9 GB/s up and 2.5–2.9 GB/s back on this integrated part, both driver-overhead-bound
//!   rather than bandwidth-bound, and both far more than D §6.3 needs. **The discrete path is
//!   untestable here** — this machine exposes one integrated Metal adapter and no software
//!   fallback at all (E7 §6) — so [`upload`] carries both and only one of them has ever run.
//! * **Readback goes through a `MAP_READ` buffer, once per batch.** Volumes are tiny: candidate
//!   states and scores, a few hundred KB.

use bytemuck::Pod;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, CommandEncoderDescriptor};

use crate::GpuError;
use crate::device::{Gpu, MAX_WORKGROUPS_PER_DIM, WORKGROUP};

/// How one array is split so that no binding exceeds the cap (D §6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunking {
    /// Elements in every chunk but possibly the last.
    pub per_chunk: usize,
    /// How many chunks the array becomes.
    pub chunks: usize,
    /// The total the split covers.
    pub total: usize,
}

impl Chunking {
    /// Splits `total` elements of `stride` bytes into chunks of at most `cap` bytes.
    ///
    /// `cap` is the binding cap of the device, and one element never spans two chunks. A stride
    /// larger than the cap is not representable and gives one element per chunk, which is what a
    /// caller then has to notice.
    #[must_use]
    pub fn of(total: usize, stride: usize, cap: u64) -> Self {
        let stride = stride.max(1);
        let per_chunk = usize::try_from(cap / stride as u64).unwrap_or(usize::MAX).max(1);
        let chunks = total.div_ceil(per_chunk);
        Self { per_chunk, chunks, total }
    }

    /// The element range of chunk `i`.
    #[must_use]
    pub fn range(&self, i: usize) -> std::ops::Range<usize> {
        let start = (i * self.per_chunk).min(self.total);
        let end = (start + self.per_chunk).min(self.total);
        start..end
    }

    /// Every chunk's range, in order.
    pub fn ranges(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.chunks).map(|i| self.range(i))
    }
}

/// The workgroup grid one dispatch asks for, with E7 §2's 2-D fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// Workgroups along x.
    pub x: u32,
    /// Workgroups along y; `1` until x would exceed the per-dimension limit.
    pub y: u32,
    /// The workgroup count the kernel is *meant* to run — `x·y` may be larger, and the kernel
    /// discards the overhang by comparing its linear index against this.
    pub wanted: u32,
}

impl Dispatch {
    /// The grid for `items` invocations at [`WORKGROUP`] lanes each.
    #[must_use]
    pub fn for_items(items: usize) -> Self {
        Self::for_workgroups(u32::try_from(items.div_ceil(WORKGROUP as usize)).unwrap_or(u32::MAX))
    }

    /// The grid for a given workgroup count.
    ///
    /// Beyond `max_compute_workgroups_per_dimension` the count is folded into two dimensions and
    /// the kernel recomputes its linear index as `gid.x + gid.y · wg_x` — which is why `x` is
    /// passed to the shader as a uniform and never derived there (E7 §3: the shape is data).
    #[must_use]
    pub fn for_workgroups(workgroups: u32) -> Self {
        if workgroups <= MAX_WORKGROUPS_PER_DIM {
            return Self { x: workgroups.max(1), y: 1, wanted: workgroups };
        }
        let x = MAX_WORKGROUPS_PER_DIM;
        let y = workgroups.div_ceil(x);
        Self { x, y, wanted: workgroups }
    }

    /// Whether the grid had to become two-dimensional.
    #[must_use]
    pub fn is_2d(&self) -> bool {
        self.y > 1
    }
}

/// Uploads a `Pod` slice as a storage buffer.
///
/// Mapped at creation, which on a unified-memory adapter is the whole of the upload (E7 §2). On a
/// discrete adapter `create_buffer_init` stages internally through `queue.write_buffer`; that path
/// exists, is what D §6.3 specifies, and **has never run on this machine**, which has one
/// integrated Metal adapter and no software fallback (E7 §6).
pub fn upload<T: Pod>(gpu: &Gpu, label: &str, data: &[T]) -> Buffer {
    gpu.device().create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    })
}

/// A buffer the kernels write and the host reads back.
pub fn output(gpu: &Gpu, label: &str, bytes: u64) -> Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Copies a device buffer into host memory through a `MAP_READ` staging buffer.
///
/// One readback per batch, which is what D §6.3's "readback volume is tiny" assumes.
pub fn read_back<T: Pod>(
    gpu: &Gpu,
    what: &'static str,
    buffer: &Buffer,
    len: usize,
) -> Result<Vec<T>, GpuError> {
    let elements = len * std::mem::size_of::<T>();
    let bytes = elements as u64;
    let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes.max(4),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        gpu.device().create_command_encoder(&CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    gpu.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    gpu.wait()?;
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(GpuError::Readback { what, message: e.to_string() }),
        Err(e) => return Err(GpuError::Readback { what, message: e.to_string() }),
    }
    let view = slice
        .get_mapped_range()
        .map_err(|e| GpuError::Readback { what, message: e.to_string() })?;
    let out = bytemuck::cast_slice::<u8, T>(&view[..elements]).to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Chunking, Dispatch};
    use crate::device::{DEFAULT_BINDING_CAP, MAX_WORKGROUPS_PER_DIM, WORKGROUP};

    /// A batch is split so that no chunk exceeds the binding cap, every element is in exactly one
    /// chunk, and the ranges are contiguous and in order.
    #[test]
    fn a_batch_is_chunked_to_the_binding_cap() {
        // 16-byte `vec4<f32>` points against the portable 128 MB cap: 8 388 608 per binding.
        let split = Chunking::of(20_000_000, 16, DEFAULT_BINDING_CAP);
        assert_eq!(split.per_chunk, 8_388_608);
        assert_eq!(split.chunks, 3);
        let ranges: Vec<_> = split.ranges().collect();
        assert_eq!(ranges[0], 0..8_388_608);
        assert_eq!(ranges[2], 16_777_216..20_000_000);
        assert_eq!(ranges.iter().map(std::iter::ExactSizeIterator::len).sum::<usize>(), 20_000_000);
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "contiguous");
        }

        // Everything that fits is one chunk; nothing at all is no chunk.
        assert_eq!(Chunking::of(1000, 16, DEFAULT_BINDING_CAP).chunks, 1);
        assert_eq!(Chunking::of(0, 16, DEFAULT_BINDING_CAP).chunks, 0);
        assert_eq!(Chunking::of(0, 16, DEFAULT_BINDING_CAP).range(0), 0..0);
        // An element larger than the cap still yields one element per chunk rather than none.
        let huge = Chunking::of(4, 1 << 30, 1024);
        assert_eq!((huge.per_chunk, huge.chunks), (1, 4));
    }

    /// The dispatch stays one-dimensional up to the per-dimension limit and folds into two beyond
    /// it — the wall E7 §2 hit at 96 000 workgroups.
    #[test]
    fn the_dispatch_folds_into_two_dimensions_past_the_limit() {
        let small = Dispatch::for_workgroups(1000);
        assert_eq!((small.x, small.y, small.wanted), (1000, 1, 1000));
        assert!(!small.is_2d());

        let edge = Dispatch::for_workgroups(MAX_WORKGROUPS_PER_DIM);
        assert_eq!((edge.x, edge.y), (MAX_WORKGROUPS_PER_DIM, 1));
        assert!(!edge.is_2d());

        // E7's own case: 4096 poses × 6000 points at 256 lanes is 96 000 workgroups.
        let e7 = Dispatch::for_items(4096 * 6000);
        assert_eq!(e7.wanted, 96_000);
        assert_eq!((e7.x, e7.y), (MAX_WORKGROUPS_PER_DIM, 2));
        assert!(e7.is_2d());
        assert!(u64::from(e7.x) * u64::from(e7.y) >= u64::from(e7.wanted), "the grid covers it");

        // Items round up to whole workgroups, and one item still gets one workgroup.
        assert_eq!(Dispatch::for_items(WORKGROUP as usize + 1).wanted, 2);
        assert_eq!(Dispatch::for_items(1).wanted, 1);
        assert_eq!(Dispatch::for_items(0).x, 1, "an empty dispatch is still a legal grid");
    }
}
