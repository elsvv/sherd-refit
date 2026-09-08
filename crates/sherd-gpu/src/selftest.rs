//! D §6.8's self-test, as E7 §8 corrects it.
//!
//! A GPU that has never been checked is not a backend, it is a hope. Before `Backend::Auto` may
//! pick this device, and before `--backend gpu` may accept it, four things are measured on it:
//!
//! | check | what it proves | expectation |
//! |---|---|---|
//! | adapters and limits | the device can run D §6's kernels at all | the wgpu defaults of [`Requirements`](crate::device::Requirements) |
//! | a fixed-order reduction of 1e7 `f32` terms | the shape D §6.4 specifies survives the compiler | **bit-identical** to the CPU mirror (E7 §3) |
//! | D §6.2's bounded-NN kernel on a synthetic cloud | the grid, the traversal order and the tie rule agree | 0 differing neighbours, `max |Δd|` ≤ 1e-6 of the cloud (E7 §5.1: 56 in 24.6 M, 2.4e-7) |
//! | host wall time of that kernel, GPU against all CPU cores | whether the GPU is worth using | ≥ 1.5× for `Backend::Auto` (D §6.8) |
//!
//! Three corrections from E7 are built in rather than commented on:
//!
//! * **The timing is host wall time**, `submit` + `poll(Wait)`, never a timestamp query: on this
//!   Metal driver `TIMESTAMP_QUERY` is advertised, resolves without error and returns nonsense —
//!   0.000 ms for a kernel whose wall time is 4.7 ms (E7 §7.1). The batch is sized so that the
//!   ≈ 0.5 ms submit cost cannot dominate.
//! * **An explicit `BindGroupLayout` and `PipelineLayout`, never `get_bind_group_layout`.** Auto
//!   layouts are exclusive to one pipeline and only contain the bindings that entry point uses
//!   (E7 §7.3).
//! * **The throughput row is only meaningful in a release build.** The GPU side is a compiled
//!   kernel either way, while the CPU side is `sherd-core` — a workspace member, so `-O0` in a
//!   debug build (the `opt-level = 2` of `[profile.dev.package."*"]` covers dependencies, not
//!   members). Measured on this machine: 22.8 ms release against 131.2 ms debug for the same
//!   384 000 queries, i.e. a ratio of 4.2× or 12.6× depending only on how the *CPU* was compiled.
//!   `Backend::Auto` therefore decides on what the release binary measures; a debug run will
//!   over-report the GPU, and the row prints both wall times so that is visible rather than
//!   hidden inside the ratio.
//! * **A failure here means the CPU with no second opinion.** This machine exposes one Metal
//!   adapter and no software fallback of any kind (E7 §6), so there is no third implementation to
//!   break the tie. The report says which check failed, and by how much.
//!
//! # What the reduction's CPU mirror is
//!
//! [`reduce_mirror`] is `kernels/reduce.wgsl` transcribed: the same workgroup count, the same lane
//! stride, the same loop bound and the same 256 → 1 tree, single-threaded and in `f32`. E7 §3's
//! finding is that this is bit-identical on Metal in stock configuration, and the self-test
//! asserts the 32 bits rather than a tolerance — which is what makes a future kernel mismatch a
//! signal instead of a tolerance argument.

use std::time::{Duration, Instant};

use sherd_core::spatial::grid::HashGrid;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderStages,
};

use crate::GpuError;
use crate::buffers::{self, Dispatch};
use crate::device::{AdapterEntry, Gpu, Requirements, WORKGROUP};

/// D §6.8's threshold for `Backend::Auto`: the GPU must beat the whole CPU by this much.
pub const AUTO_SPEEDUP: f64 = 1.5;

/// Terms in the reduction check — E7 §3's own 1e7.
pub const REDUCE_TERMS: usize = 10_000_000;

/// Workgroups the reduction runs in, D §6.4's own 256.
pub const REDUCE_WORKGROUPS: usize = 256;

/// Points in each of the two synthetic clouds of the NN check (E7 §5's 6000).
pub const NN_POINTS: usize = 6000;

/// Poses in the NN check.
///
/// E7 §5 measured 64 poses at 4.70 ms and 12.2 ns/query, already past the ≈ 0.5 ms submit cost
/// that dominates a single-pose dispatch — enough for the throughput number to mean something and
/// small enough that a self-test at every run start costs milliseconds.
pub const NN_POSES: usize = 64;

/// E7 §5's radius on E7 §5's cloud.
pub const NN_RADIUS: f32 = 0.03;

/// The largest neighbour-distance disagreement the NN check tolerates, in units of the cloud.
///
/// E7 measured 2.4e-7 over 24.6 M queries in stock configuration, twice; 1e-6 is four times that
/// and still 100× inside the tightest distance tolerance of D §10.2 (1e-4 `t`).
pub const NN_MAX_DELTA: f32 = 1e-6;

/// The largest share of queries whose *neighbour choice* may differ between the two executors.
///
/// **This is not a tolerance that was widened to let a run pass.** E7 §5.1 measured the rate in
/// stock configuration and called it inherent: `2.3e-6` of queries pick a different point when two
/// candidates are within a ULP of each other, because the pose transform `R·p + τ` is contracted
/// into FMAs on Metal and the two transformed points already differ (E7 §4.2). D §7's "ties → the
/// lowest index" rule cannot fix it, since the tie is not exact on one of the two sides. `1e-5` is
/// four times the measured rate.
///
/// The *count* is therefore not the check. The check is that every disagreement is a genuine
/// near-tie: the two candidates must be the same distance from the query to within
/// [`NN_MAX_DELTA`]. A neighbour that is actually wrong shows a large `|Δd|` and fails whatever
/// the count is.
pub const NN_MAX_DIFFERING_SHARE: f64 = 1e-5;

/// One check's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// What was measured.
    pub name: &'static str,
    /// Whether it held.
    pub passed: bool,
    /// The measurement, in the units the row's own documentation gives.
    pub detail: String,
}

/// What the self-test found (D §6.8).
#[derive(Clone, Debug)]
pub struct SelfTest {
    /// The adapter it ran on.
    pub adapter: AdapterEntry,
    /// One row per check, in the order they ran.
    pub checks: Vec<Check>,
    /// GPU wall time of the NN batch (`submit` + `poll(Wait)`), E7 §7.1's host timing.
    pub gpu: Duration,
    /// The same batch on the CPU, over the whole rayon pool.
    pub cpu: Duration,
    /// `cpu / gpu` on that batch — what [`AUTO_SPEEDUP`] is compared against.
    pub speedup: f64,
    /// Queries the throughput figures are over.
    pub queries: usize,
}

impl SelfTest {
    /// Whether every check held.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// The checks that did not hold, as the error message lists them.
    #[must_use]
    pub fn failures(&self) -> Vec<String> {
        self.checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| format!("{}: {}", c.name, c.detail))
            .collect()
    }

    /// Nanoseconds of GPU wall time per bounded-NN query — E7 §5's own unit.
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "a nanosecond count printed to one decimal")]
    pub fn ns_per_query(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        self.gpu.as_secs_f64() * 1e9 / self.queries as f64
    }

    /// Runs D §6.8's self-test on an open device.
    #[allow(clippy::cast_precision_loss, reason = "counts printed to one or two decimals")]
    pub fn run(gpu: &Gpu) -> Result<Self, GpuError> {
        let mut checks = Vec::new();

        // 1. Limits. The device would not have opened below them, so this row records what was
        //    granted rather than re-deciding it.
        let limits = gpu.limits();
        let unmet = Requirements::default().unmet(limits);
        checks.push(Check {
            name: "limits",
            passed: unmet.is_empty(),
            detail: if unmet.is_empty() {
                format!(
                    "workgroup {} lanes, {} B shared, {} MB binding, {} workgroups/dim",
                    limits.max_compute_invocations_per_workgroup,
                    limits.max_compute_workgroup_storage_size,
                    limits.max_storage_buffer_binding_size / (1024 * 1024),
                    limits.max_compute_workgroups_per_dimension,
                )
            } else {
                unmet.join("; ")
            },
        });

        // 2. E7 §3's reduction: the same schedule on both sides, and the same 32 bits.
        let terms = reduce_terms(REDUCE_TERMS);
        let want = reduce_mirror(&terms, REDUCE_WORKGROUPS);
        let got = reduce_on_device(gpu, &terms, REDUCE_WORKGROUPS)?;
        checks.push(Check {
            name: "reduction",
            passed: got.to_bits() == want.to_bits(),
            detail: format!(
                "{REDUCE_TERMS} terms, {REDUCE_WORKGROUPS} workgroups: gpu {got:e} \
                 (0x{:08x}), cpu {want:e} (0x{:08x})",
                got.to_bits(),
                want.to_bits()
            ),
        });

        // 3 and 4. E7 §5's bounded-NN kernel: agreement and throughput on the same batch.
        let (target, source, poses) = nn_cloud(NN_POINTS, NN_POSES);
        let grid = HashGrid::of_f32(&target, NN_RADIUS).expect("a non-empty synthetic cloud");
        let queries = NN_POSES * NN_POINTS;

        // The GPU figure is the dispatch alone (E7 §7.1); the CPU figure is the same batch over
        // the whole rayon pool, warmed the same way so that neither side pays a first-touch cost
        // the other does not.
        let (device_out, gpu_time) = nn_on_device(gpu, &grid, &source, &poses)?;
        let _ = nn_on_host(&grid, &source, &poses);
        let started = Instant::now();
        let host_out = nn_on_host(&grid, &source, &poses);
        let cpu_time = started.elapsed();

        let mut differing = 0_usize;
        let mut misses = 0_usize;
        let mut max_delta = 0.0_f32;
        let mut max_tie_delta = 0.0_f32;
        for (device, host) in device_out.iter().zip(&host_out) {
            // A hit on one side and a miss on the other is never a tie: it is a wrong answer.
            if (device.0 == u32::MAX) != (host.0 == u32::MAX) {
                misses += 1;
                continue;
            }
            if device.0 == u32::MAX {
                continue;
            }
            let delta = (device.1.sqrt() - host.1.sqrt()).abs();
            max_delta = max_delta.max(delta);
            if device.0 != host.0 {
                differing += 1;
                max_tie_delta = max_tie_delta.max(delta);
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "a query count printed as a share")]
        let share = differing as f64 / queries as f64;
        checks.push(Check {
            name: "bounded nn",
            // The count is a *rate* against E7 §5.1's measured 2.3e-6; the substantive check is
            // that every disagreement is a near-tie and that neither side found a neighbour the
            // other missed.
            passed: misses == 0
                && max_delta <= NN_MAX_DELTA
                && max_tie_delta <= NN_MAX_DELTA
                && share <= NN_MAX_DIFFERING_SHARE,
            detail: format!(
                "{queries} queries, {} occupied cells, {} per cell at most: {differing} differing \
                 neighbours ({share:.2e}, limit {NN_MAX_DIFFERING_SHARE:.0e}), each a tie to \
                 {max_tie_delta:e}; {misses} hit/miss disagreements; max |Δd| {max_delta:e} \
                 (limit {NN_MAX_DELTA:e})",
                grid.occupied(),
                grid.max_per_cell(),
            ),
        });

        let speedup = if gpu_time.as_secs_f64() > 0.0 {
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        } else {
            0.0
        };
        checks.push(Check {
            name: "throughput",
            // Not a pass criterion: a slow GPU is a reason not to use it, not a broken one.
            // `Selection` reads the number; this row records it.
            passed: true,
            detail: format!(
                "gpu {:.2} ms ({:.1} ns/query), cpu {:.2} ms over {} threads: {speedup:.2}x",
                gpu_time.as_secs_f64() * 1e3,
                gpu_time.as_secs_f64() * 1e9 / queries as f64,
                cpu_time.as_secs_f64() * 1e3,
                rayon::current_num_threads(),
            ),
        });

        Ok(Self {
            adapter: gpu.entry().clone(),
            checks,
            gpu: gpu_time,
            cpu: cpu_time,
            speedup,
            queries,
        })
    }
}

/// What `Backend::Auto` decided, and why (D §6.8).
#[derive(Clone, Debug)]
pub struct Selection {
    /// The adapter, when one was opened.
    pub adapter: Option<AdapterEntry>,
    /// The self-test, when it ran.
    pub selftest: Option<SelfTest>,
    /// Whether the GPU executor will be used.
    pub use_gpu: bool,
    /// One sentence naming the deciding fact.
    pub reason: String,
}

impl Selection {
    /// D §6.8's rule, applied to a self-test that has already run.
    ///
    /// `has_kernels` is what makes this honest in phase 2a: the GPU executor exists, the device
    /// works and the self-test passes, but the four `Executor` methods are still the CPU's
    /// (phases 2b and 2c), so `Auto` must not claim a GPU run. When the kernels land, the flag
    /// becomes `true` and this rule is unchanged.
    #[must_use]
    pub fn decide(selftest: SelfTest, has_kernels: bool) -> Self {
        let adapter = Some(selftest.adapter.clone());
        if !selftest.passed() {
            let reason =
                format!("the self-test failed ({}); using the CPU", selftest.failures().join("; "));
            return Self { adapter, selftest: Some(selftest), use_gpu: false, reason };
        }
        if !has_kernels {
            let reason = format!(
                "the self-test passed at {:.2}x on {}, but phase 2a's GPU executor has no kernels \
                 yet (D §12: 2b and 2c) and routes every method to the CPU; using the CPU",
                selftest.speedup, selftest.adapter.name
            );
            return Self { adapter, selftest: Some(selftest), use_gpu: false, reason };
        }
        if selftest.adapter.is_software() {
            let reason = format!(
                "{} is a software implementation of the API, not a GPU; using the CPU",
                selftest.adapter.name
            );
            return Self { adapter, selftest: Some(selftest), use_gpu: false, reason };
        }
        if selftest.speedup < AUTO_SPEEDUP {
            let reason = format!(
                "the GPU is {:.2}x the CPU on the self-test batch, below the {AUTO_SPEEDUP}x \
                 D §6.8 asks for; using the CPU",
                selftest.speedup
            );
            return Self { adapter, selftest: Some(selftest), use_gpu: false, reason };
        }
        let reason = format!(
            "the self-test passed and the GPU is {:.2}x the CPU on its batch, at or above the \
             {AUTO_SPEEDUP}x of D §6.8",
            selftest.speedup
        );
        Self { adapter, selftest: Some(selftest), use_gpu: true, reason }
    }

    /// The decision when no adapter could be opened at all.
    #[must_use]
    pub fn no_gpu(error: &GpuError) -> Self {
        Self { adapter: None, selftest: None, use_gpu: false, reason: error.to_string() }
    }
}

/// E7 §3's array: uniform in `[−1, 1)` with every 997th scaled by 1e5, so that the summation order
/// genuinely matters.
///
/// The stream is a fixed xorshift rather than a seeded RNG so that this file and its WGSL twin can
/// be reasoned about together, and so that two runs of the self-test compare the same numbers.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "24 bits into an f32 mantissa")]
pub fn reduce_terms(n: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 * (1.0 / 16_777_216.0);
            let value = unit * 2.0 - 1.0;
            if i % 997 == 0 { value * 1e5 } else { value }
        })
        .collect()
}

/// `kernels/reduce.wgsl` transcribed: the same striding, the same loop bound, the same tree.
///
/// Single-threaded and in `f32`, because the point is the *order*, not the speed.
#[must_use]
pub fn reduce_mirror(terms: &[f32], workgroups: usize) -> f32 {
    let partials = reduce_pass(terms, workgroups);
    // The host repeats the pass over the partials, in one workgroup — which is what
    // `reduce_on_device` submits as its second dispatch.
    reduce_pass(&partials, 1)[0]
}

/// One pass of the reduction: `workgroups` partials, each the workgroup's own 256 → 1 tree.
fn reduce_pass(terms: &[f32], workgroups: usize) -> Vec<f32> {
    let lanes = WORKGROUP as usize;
    let block = terms.len().div_ceil(workgroups.max(1));
    (0..workgroups)
        .map(|group| {
            let start = group * block;
            let end = (start + block).min(terms.len());
            let mut scratch = [0.0_f32; WORKGROUP as usize];
            for (lane, slot) in scratch.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                let mut i = start + lane;
                while i < end {
                    acc += terms[i];
                    i += lanes;
                }
                *slot = acc;
            }
            let mut stride = lanes / 2;
            while stride > 0 {
                for lane in 0..stride {
                    scratch[lane] += scratch[lane + stride];
                }
                stride /= 2;
            }
            scratch[0]
        })
        .collect()
}

/// The self-test's synthetic input: a target cloud, a source cloud and one `[R | tau]` per pose.
pub type NnCloud = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[[f32; 4]; 3]>);

/// E7 §5's synthetic data: two clouds on a warped unit sheet — a stand-in for a fracture surface —
/// and small random rigid perturbations as poses.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "24 bits into an f32 mantissa")]
pub fn nn_cloud(points: usize, poses: usize) -> NnCloud {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 * (1.0 / 16_777_216.0)
    };
    let sheet = |u: f32, v: f32| [u, v, 0.15 * (u * 6.3).sin() * (v * 4.1).cos()];
    let target: Vec<[f32; 3]> = (0..points).map(|_| sheet(next(), next())).collect();
    let source: Vec<[f32; 3]> = (0..points).map(|_| sheet(next(), next())).collect();
    let poses = (0..poses)
        .map(|_| {
            // A small rotation about z and a small translation: enough to move the queries around
            // the grid without emptying the hit rate.
            let angle = (next() - 0.5) * 0.05;
            let (s, c) = (angle.sin(), angle.cos());
            let t = [(next() - 0.5) * 0.02, (next() - 0.5) * 0.02, (next() - 0.5) * 0.02];
            [[c, -s, 0.0, t[0]], [s, c, 0.0, t[1]], [0.0, 0.0, 1.0, t[2]]]
        })
        .collect();
    (target, source, poses)
}

/// The NN batch on the CPU, over the whole rayon pool: `HashGrid`'s own traversal, which is the
/// mirror `kernels/nn.wgsl` was written from.
#[must_use]
pub fn nn_on_host(
    grid: &HashGrid,
    source: &[[f32; 3]],
    poses: &[[[f32; 4]; 3]],
) -> Vec<(u32, f32)> {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    (0..poses.len() * source.len())
        .into_par_iter()
        .map(|id| {
            let (pose, k) = (id / source.len(), id % source.len());
            let (r, p) = (&poses[pose], source[k]);
            let q =
                std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + r[i][3]);
            grid.nearest_within(&q).map_or((u32::MAX, -1.0), |(j, d)| (j, d * d))
        })
        .collect()
}

/// `kernels/reduce.wgsl` on the device: one dispatch of `workgroups`, then one of 1.
fn reduce_on_device(gpu: &Gpu, terms: &[f32], workgroups: usize) -> Result<f32, GpuError> {
    let module = gpu.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("reduce"),
        source: wgpu::ShaderSource::Wgsl(include_str!("kernels/reduce.wgsl").into()),
    });
    // An explicit layout, never `get_bind_group_layout`: auto layouts are exclusive to one
    // pipeline and hold only the bindings that entry point uses (E7 §7.3).
    let layout = gpu.device().create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("reduce"),
        entries: &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false)],
    });
    let pipeline_layout = gpu.device().create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("reduce"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = gpu.device().create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("reduce"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("reduce"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut input = terms.to_vec();
    let mut groups = workgroups;
    loop {
        let block = input.len().div_ceil(groups.max(1));
        let dispatch = Dispatch::for_workgroups(u32::try_from(groups).unwrap_or(u32::MAX));
        let params = ReduceParams {
            n: u32::try_from(input.len()).unwrap_or(u32::MAX),
            block: u32::try_from(block).unwrap_or(u32::MAX),
            workgroups: u32::try_from(groups).unwrap_or(u32::MAX),
            wg_x: dispatch.x,
        };
        let params_buffer = uniform(gpu, "reduce params", &params);
        let terms_buffer = buffers::upload(gpu, "reduce terms", &input);
        let out_buffer = buffers::output(gpu, "reduce partials", (groups * 4) as u64);
        let bind = gpu.device().create_bind_group(&BindGroupDescriptor {
            label: Some("reduce"),
            layout: &layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: terms_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor { label: Some("reduce") });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }
        gpu.queue().submit(Some(encoder.finish()));
        gpu.wait()?;
        input = buffers::read_back::<f32>(gpu, "reduce partials", &out_buffer, groups)?;
        if groups == 1 {
            return Ok(input[0]);
        }
        groups = 1;
    }
}

/// `kernels/nn.wgsl` on the device, over the whole batch in one dispatch.
///
/// Returns the answers and **the wall time of the dispatch alone**: `submit` + `poll(Wait)`, with
/// the shader module, the pipeline and every buffer already built, and after one warm-up
/// dispatch. That separation is the point. Metal compiles a shader on first use, and a timing
/// that includes the compile reports the compiler rather than the kernel — the first version of
/// this function did, and it made a kernel E7 measured at 12 ns/query look like 197 ns/query and
/// the GPU look 3.4× *slower* than the CPU. E7 §7.1 is explicit that the measurement is host wall
/// time around the submission, because this driver's timestamp queries return nonsense.
fn nn_on_device(
    gpu: &Gpu,
    grid: &HashGrid,
    source: &[[f32; 3]],
    poses: &[[[f32; 4]; 3]],
) -> Result<(Vec<(u32, f32)>, Duration), GpuError> {
    let module = gpu.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nn"),
        source: wgpu::ShaderSource::Wgsl(include_str!("kernels/nn.wgsl").into()),
    });
    let layout = gpu.device().create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("nn"),
        entries: &[
            uniform_entry(0),
            uniform_entry(1),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
            storage_entry(7, true),
            storage_entry(8, false),
            storage_entry(9, false),
        ],
    });
    let pipeline_layout = gpu.device().create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("nn"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = gpu.device().create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("nn"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("nearest"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let queries = poses.len() * source.len();
    let dispatch = Dispatch::for_items(queries);
    // `GridHeader` is `#[repr(C)]` and `Pod` on the core side precisely so that it is the WGSL
    // block: `vec4<f32>` (origin, `w` = 1/r) then `cap`, `n` and two words of padding (D §6.2).
    let header = grid.header();
    let params = NnParams {
        poses: u32::try_from(poses.len()).unwrap_or(u32::MAX),
        points: u32::try_from(source.len()).unwrap_or(u32::MAX),
        radius: grid.radius(),
        wg_x: dispatch.x,
    };
    let params_buffer = uniform(gpu, "nn params", &params);
    let header_buffer = uniform(gpu, "nn grid", &header);
    let slots = buffers::upload(gpu, "nn slots", grid.slots());
    let counts = buffers::upload(gpu, "nn counts", grid.counts());
    let indices = buffers::upload(gpu, "nn sorted_idx", grid.sorted_idx());
    let target = buffers::upload(gpu, "nn target", &pad4(grid.points()));
    let source_buffer = buffers::upload(gpu, "nn source", &pad4(source));
    let poses_buffer = buffers::upload(gpu, "nn poses", &flatten(poses));
    let found = buffers::output(gpu, "nn found", (queries * 4) as u64);
    let distance2 = buffers::output(gpu, "nn d2", (queries * 4) as u64);

    let bind = gpu.device().create_bind_group(&BindGroupDescriptor {
        label: Some("nn"),
        layout: &layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: header_buffer.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: slots.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: counts.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: indices.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: target.as_entire_binding() },
            BindGroupEntry { binding: 6, resource: source_buffer.as_entire_binding() },
            BindGroupEntry { binding: 7, resource: poses_buffer.as_entire_binding() },
            BindGroupEntry { binding: 8, resource: found.as_entire_binding() },
            BindGroupEntry { binding: 9, resource: distance2.as_entire_binding() },
        ],
    });

    let submit = |label: &str| -> Result<(), GpuError> {
        let mut encoder =
            gpu.device().create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }
        gpu.queue().submit(Some(encoder.finish()));
        gpu.wait()
    };
    // Warm-up: the shader is compiled and the buffers are resident after this one.
    submit("nn warmup")?;
    let started = Instant::now();
    submit("nn")?;
    let elapsed = started.elapsed();

    let idx = buffers::read_back::<u32>(gpu, "nn found", &found, queries)?;
    let d2 = buffers::read_back::<f32>(gpu, "nn d2", &distance2, queries)?;
    Ok((idx.into_iter().zip(d2).collect(), elapsed))
}

/// `vec3<f32>` arrays padded to the `vec4<f32>` the kernels bind (D §6.3).
fn pad4(points: &[[f32; 3]]) -> Vec<[f32; 4]> {
    points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect()
}

/// Poses flattened into the three `vec4<f32>` rows per pose the kernel indexes.
fn flatten(poses: &[[[f32; 4]; 3]]) -> Vec<[f32; 4]> {
    poses.iter().flat_map(|rows| rows.iter().copied()).collect()
}

/// `kernels/reduce.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    n: u32,
    block: u32,
    workgroups: u32,
    wg_x: u32,
}

/// `kernels/nn.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NnParams {
    poses: u32,
    points: u32,
    radius: f32,
    wg_x: u32,
}

/// A uniform block, mapped at creation.
fn uniform<T: bytemuck::Pod>(gpu: &Gpu, label: &str, value: &T) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    gpu.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// A read-only uniform binding at `binding`.
fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
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
fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
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

#[cfg(test)]
mod tests {
    use super::{
        AUTO_SPEEDUP, Check, NN_POSES, NN_RADIUS, Selection, SelfTest, nn_cloud, nn_on_host,
        reduce_mirror, reduce_terms,
    };
    use crate::device::AdapterEntry;
    use sherd_core::spatial::grid::HashGrid;
    use std::time::Duration;

    fn adapter(kind: &str) -> AdapterEntry {
        AdapterEntry {
            index: 0,
            backend: "Metal".to_owned(),
            name: "Apple M2 Pro".to_owned(),
            device_type: kind.to_owned(),
            vendor: 0,
            device: 0,
            driver: String::new(),
            driver_info: String::new(),
        }
    }

    fn selftest(passed: bool, speedup: f64, kind: &str) -> SelfTest {
        SelfTest {
            adapter: adapter(kind),
            checks: vec![Check { name: "reduction", passed, detail: "measured".to_owned() }],
            gpu: Duration::from_millis(5),
            cpu: Duration::from_millis(40),
            speedup,
            queries: NN_POSES * 6000,
        }
    }

    /// The reduction mirror is deterministic, is not a plain left-to-right sum (so the schedule
    /// genuinely matters), and reproduces the same bits on every call.
    #[test]
    fn the_reduction_mirror_is_the_schedule_and_not_a_plain_sum() {
        let terms = reduce_terms(200_000);
        let tree = reduce_mirror(&terms, 256);
        assert_eq!(tree.to_bits(), reduce_mirror(&terms, 256).to_bits(), "deterministic");

        let mut naive = 0.0_f32;
        for &t in &terms {
            naive += t;
        }
        assert_ne!(
            tree.to_bits(),
            naive.to_bits(),
            "if the two agreed, the check would not be testing the schedule"
        );
        // Both are the same sum to f32's own precision; the point is the last bits, not the value.
        let exact: f64 = terms.iter().map(|&t| f64::from(t)).sum();
        assert!((f64::from(tree) - exact).abs() < exact.abs() * 1e-3 + 1e3, "{tree} vs {exact}");

        // The workgroup count is data: a different count is a different (still valid) schedule.
        assert_ne!(reduce_mirror(&terms, 128).to_bits(), tree.to_bits());
    }

    /// The self-test's synthetic cloud is E7 §5's shape: a sheet dense enough that most queries
    /// hit, and a grid whose cells hold a handful of points each.
    #[test]
    fn the_self_test_cloud_is_the_one_e7_measured() {
        let (target, source, poses) = nn_cloud(2000, 8);
        assert_eq!((target.len(), source.len(), poses.len()), (2000, 2000, 8));
        let grid = HashGrid::of_f32(&target, NN_RADIUS).expect("a non-empty cloud");
        assert!(grid.occupied() > 100, "the sheet spreads over many cells: {}", grid.occupied());
        assert!(grid.max_per_cell() < 100, "and none of them is the whole cloud");

        let out = nn_on_host(&grid, &source, &poses);
        assert_eq!(out.len(), 8 * 2000);
        let hits = out.iter().filter(|(j, _)| *j != u32::MAX).count();
        assert!(hits * 2 > out.len(), "most queries find a neighbour: {hits} of {}", out.len());
        // Every reported distance is inside the radius, and a miss reports −1.
        for &(j, d2) in &out {
            if j == u32::MAX {
                assert!(d2 < 0.0);
            } else {
                assert!(d2 <= NN_RADIUS * NN_RADIUS + 1e-9, "{d2}");
            }
        }
    }

    /// D §6.8's `Backend::Auto` rule, every branch, without an adapter.
    #[test]
    fn the_auto_rule_needs_a_pass_a_gpu_and_one_and_a_half_times() {
        // Phase 2a: everything passes and it still says CPU, because there are no kernels.
        let phase2a = Selection::decide(selftest(true, 8.0, "IntegratedGpu"), false);
        assert!(!phase2a.use_gpu);
        assert!(phase2a.reason.contains("no kernels yet"), "{}", phase2a.reason);

        // With kernels: a passing self-test above the threshold picks the GPU.
        let picked = Selection::decide(selftest(true, 8.0, "IntegratedGpu"), true);
        assert!(picked.use_gpu, "{}", picked.reason);
        assert!(picked.reason.contains("8.00x"), "{}", picked.reason);

        // Below the threshold it does not.
        let slow = Selection::decide(selftest(true, AUTO_SPEEDUP - 0.01, "IntegratedGpu"), true);
        assert!(!slow.use_gpu && slow.reason.contains("below"), "{}", slow.reason);
        let exactly = Selection::decide(selftest(true, AUTO_SPEEDUP, "IntegratedGpu"), true);
        assert!(exactly.use_gpu, "the rule is `at or above`: {}", exactly.reason);

        // A failed check always wins, and a software adapter is never preferred to the CPU.
        let failed = Selection::decide(selftest(false, 100.0, "IntegratedGpu"), true);
        assert!(!failed.use_gpu && failed.reason.contains("self-test failed"), "{}", failed.reason);
        let software = Selection::decide(selftest(true, 100.0, "Cpu"), true);
        assert!(!software.use_gpu && software.reason.contains("software"), "{}", software.reason);

        // And with no adapter at all the reason is the error the operator has to act on.
        let none = Selection::no_gpu(&crate::GpuError::NoAdapter);
        assert!(!none.use_gpu && none.adapter.is_none());
        assert!(none.reason.contains("no GPU adapter"), "{}", none.reason);
    }

    /// The failure list is what the `--backend gpu` error prints, and `passed` is the conjunction.
    #[test]
    fn the_self_test_reports_every_check_that_failed() {
        let mut test = selftest(true, 4.0, "IntegratedGpu");
        assert!(test.passed() && test.failures().is_empty());
        assert!(test.ns_per_query() > 0.0);
        test.checks.push(Check {
            name: "bounded nn",
            passed: false,
            detail: "3 differing neighbours".to_owned(),
        });
        assert!(!test.passed());
        assert_eq!(test.failures(), vec!["bounded nn: 3 differing neighbours".to_owned()]);
    }
}
