//! The device side of `Executor::coarse_scores` (D §6.4 step 2, D §6.5).
//!
//! One dispatch scores a chunk of hypotheses against one breakline: one workgroup per pose, 64
//! lanes striding over the probe, the hash grid of D §6.2 underneath. What comes back is an
//! **integer agreement count** per pose, and the score is the same `f64` division the CPU
//! executor does — `f64::from(agree) / n_points`, never a reciprocal (step C1 measured that
//! `k · (1/60) ≠ k / 60`). So the two executors' scores are bit-identical wherever their counts
//! are, and the cross-check compares integers.
//!
//! # What is chunked, and against what
//!
//! Two limits, both D §6.3's and neither this adapter's:
//!
//! * **≤ 20 M point-queries per dispatch** (D §6.4 step 2, ≈ 10 ms of GPU time), so that no
//!   submission can approach a TDR timeout on a driver that has one;
//! * **≤ 128 MB per storage binding**, the wgpu default. The pose array is the one that can reach
//!   it — 150 000 hypotheses × 48 B is 7 MB, so in practice it never does, but the split is
//!   applied rather than assumed.
//!
//! Beyond 65 535 workgroups the dispatch folds into two dimensions and the kernel recomputes its
//! linear index from `params.wg_x` (E7 §2). At 60 probe points a chunk is 333 333 poses, so the
//! 2-D path is reached by the pose count and not by the query count.

use std::time::{Duration, Instant};

use sherd_core::executor::batch::CoarseBatch;
use sherd_core::spatial::grid::HashGrid;
use wgpu::BindingResource;

use crate::GpuError;
use crate::buffers::{self, Chunking, Dispatch};
use crate::device::{DEFAULT_BINDING_CAP, Gpu};
use crate::shader::{Kernel, interleave, interleave_narrowed, uniform};

/// D §6.4 step 2's cap on one dispatch.
pub const MAX_POINT_QUERIES: usize = 20_000_000;

/// The smallest batch worth a dispatch, in point-queries — measured, like `icp::MIN_CANDIDATES`.
///
/// A dispatch costs a submission, a grid build, an upload and a readback whatever it computes, and
/// the grid and the target cloud are rebuilt for every call. `crates/sherd-gpu/tests/adapter.rs`
/// prints the table this comes from, over a 6 000-point breakline:
///
/// | poses × points | queries | ratio |
/// |---|---|---|
/// | 1 × 60 | 60 | 2.5× (both sides microseconds) |
/// | 64 × 60 | 3 840 | 1.4× |
/// | 64 × 800 | 51 200 | 0.61× |
/// | 1 000 × 60 | 60 000 | 0.41× |
/// | 1 000 × 800 | 800 000 | 2.53× |
/// | 40 000 × 60 | 2 400 000 | 1.80× |
/// | 40 000 × 800 | 32 000 000 | 3.36× |
///
/// R §5.2's own batch is tens of thousands of hypotheses on sixty points — 1.5 to 2.4 M queries,
/// far above the line. R §5.4's re-score is a few hundred poses on `brk_sub`, which lands near it
/// and goes either way by collection.
pub const MIN_QUERIES: usize = 200_000;

/// The WGSL source: D §6.2's grid, then the kernel that queries it.
const SOURCE: &str =
    concat!(include_str!("kernels/grid.wgsl"), include_str!("kernels/coarse.wgsl"));

/// `kernels/coarse.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CoarseParams {
    lo: [f32; 4],
    hi: [f32; 4],
    poses: u32,
    points: u32,
    radius: f32,
    normal_agree: f32,
    wg_x: u32,
    first_pose: u32,
    pad: [u32; 2],
}

/// The compiled coarse-score pipeline.
///
/// Built once, in [`GpuExecutor::new`](crate::GpuExecutor::new): Metal compiles a shader on first
/// pipeline creation and G1 §5.1 measured what that does to a timed region.
#[derive(Debug)]
pub struct CoarseKernel {
    kernel: Kernel,
}

/// What one `coarse_scores` submission measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoarseRun {
    /// Dispatches submitted.
    pub dispatches: usize,
    /// Wall time of the dispatches alone — `submit` + `poll(Wait)`, no upload and no readback.
    pub gpu: Duration,
    /// Point-queries the run made: poses × probe points.
    pub queries: usize,
}

impl CoarseKernel {
    /// Compiles the kernel.
    #[must_use]
    pub fn build(gpu: &Gpu) -> Self {
        // Bindings: 0 params (uniform), 1 grid header (uniform), then the seven storage arrays.
        Self {
            kernel: Kernel::build(
                gpu,
                "coarse",
                SOURCE,
                "coarse",
                2,
                &[true, true, true, true, true, true, false],
            ),
        }
    }

    /// The agreement count of every pose of `batch`, or `None` when there is nothing to score on
    /// the device.
    ///
    /// `None` is not a failure: an empty probe, an empty breakline or a radius the grid cannot be
    /// built at are the cases the CPU executor answers with zeros, and the caller falls through to
    /// it rather than inventing an answer here.
    /// `force` ignores [`MIN_QUERIES`] — the cross-check harness's switch, never a run's.
    pub fn run(
        &self,
        gpu: &Gpu,
        batch: &CoarseBatch<'_>,
        force: bool,
    ) -> Result<Option<(Vec<u32>, CoarseRun)>, GpuError> {
        let poses = batch.poses.len();
        let points = batch.points.len();
        if poses == 0 || points == 0 || batch.target.points.is_empty() {
            return Ok(None);
        }
        if batch.target.normals.len() != batch.target.points.len() || batch.normals.len() != points
        {
            return Ok(None);
        }
        // Too small to be worth a submission and a readback (see `MIN_QUERIES`).
        if !force && poses * points < MIN_QUERIES {
            return Ok(None);
        }
        let Some(grid) = HashGrid::build(batch.target.points, batch.radius) else {
            return Ok(None);
        };

        // The device side of the batch (D §6.3): the grid's three arrays, the target cloud and the
        // probe interleaved with their normals, and the poses as three `vec4<f32>` rows each.
        let target = interleave_narrowed(grid.points(), batch.target.normals);
        let probe = interleave(batch.points, batch.normals);
        let device_poses = batch.poses.device();
        let rows: Vec<[f32; 4]> = device_poses.iter().flat_map(|p| p.rows).collect();

        // The breakline's box, from the grid's own `f32` points, so that the kernel's reject is
        // computed against the coordinates it queries.
        let mut box_lo = [f32::INFINITY; 3];
        let mut box_hi = [f32::NEG_INFINITY; 3];
        for p in grid.points() {
            for k in 0..3 {
                box_lo[k] = box_lo[k].min(p[k]);
                box_hi[k] = box_hi[k].max(p[k]);
            }
        }

        let header = uniform(gpu, "coarse grid", &grid.header());
        let slots = buffers::upload(gpu, "coarse slots", grid.slots());
        let counts = buffers::upload(gpu, "coarse counts", grid.counts());
        let indices = buffers::upload(gpu, "coarse sorted_idx", grid.sorted_idx());
        let cloud = buffers::upload(gpu, "coarse target", &target);
        let probe_buffer = buffers::upload(gpu, "coarse probe", &probe);
        let out = buffers::output(gpu, "coarse agree", (poses * 4) as u64);

        // Two caps, whichever bites first (D §6.3, D §6.4 step 2).
        let by_queries = (MAX_POINT_QUERIES / points).max(1);
        let by_binding = Chunking::of(poses, 3 * 16, DEFAULT_BINDING_CAP).per_chunk;
        let split = Chunking::of(poses, 1, by_queries.min(by_binding) as u64);

        let mut run = CoarseRun { dispatches: 0, gpu: Duration::ZERO, queries: poses * points };
        for range in split.ranges() {
            let chunk = &rows[range.start * 3..range.end * 3];
            let chunk_poses = range.end - range.start;
            let grid_shape = Dispatch::for_workgroups(u32::try_from(chunk_poses).unwrap_or(1));
            let params = CoarseParams {
                lo: [box_lo[0], box_lo[1], box_lo[2], 0.0],
                hi: [box_hi[0], box_hi[1], box_hi[2], 0.0],
                poses: u32::try_from(chunk_poses).unwrap_or(u32::MAX),
                points: u32::try_from(points).unwrap_or(u32::MAX),
                radius: narrow(batch.radius),
                normal_agree: narrow(batch.normal_agree),
                wg_x: grid_shape.x,
                first_pose: u32::try_from(range.start).unwrap_or(0),
                pad: [0; 2],
            };
            let params_buffer = uniform(gpu, "coarse params", &params);
            let pose_buffer = buffers::upload(gpu, "coarse poses", chunk);
            let bind = self.kernel.bind(
                gpu,
                &[
                    params_buffer.as_entire_binding(),
                    header.as_entire_binding(),
                    slots.as_entire_binding(),
                    counts.as_entire_binding(),
                    indices.as_entire_binding(),
                    cloud.as_entire_binding(),
                    probe_buffer.as_entire_binding(),
                    pose_buffer.as_entire_binding(),
                    BindingResource::Buffer(out.as_entire_buffer_binding()),
                ],
            );
            let started = Instant::now();
            self.kernel.dispatch(gpu, &bind, grid_shape)?;
            run.gpu += started.elapsed();
            run.dispatches += 1;
        }
        let counts = buffers::read_back::<u32>(gpu, "coarse agree", &out, poses)?;
        Ok(Some((counts, run)))
    }
}

/// The `f32` a threshold becomes on the way to the device (D §6.8: `Scales` are computed in `f64`
/// and passed as `f32`).
#[inline]
#[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
fn narrow(x: f64) -> f32 {
    x as f32
}

/// The score the host computes from a device count — the CPU executor's own division.
///
/// A `f64` division by the probe size, never a multiplication by its reciprocal: `1/60` has no
/// exact double and step C1 measured 52 of one pot_G pair's 40 029 hypotheses landing one ULP
/// away when the mean was a reciprocal. Doing it here rather than in the kernel is what makes the
/// two executors' scores bit-identical wherever their counts agree.
#[must_use]
pub fn score_of(agree: u32, points: usize) -> f64 {
    if points == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss, reason = "the probe is sixty points")]
    let n = points as f64;
    f64::from(agree) / n
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the point of `score_of` is which double it produces")]

    use super::{MAX_POINT_QUERIES, MIN_QUERIES, score_of};

    /// The host-side division is the CPU executor's, to the bit — including the cases where a
    /// reciprocal would differ.
    #[test]
    fn the_score_is_a_division_and_not_a_reciprocal() {
        for k in 0_u32..=60 {
            assert_eq!(score_of(k, 60).to_bits(), (f64::from(k) / 60.0).to_bits());
        }
        // The two disagree for real k, which is why the kernel returns a count.
        let reciprocal = 1.0_f64 / 60.0;
        let differing = (0_u32..=60).filter(|&k| f64::from(k) * reciprocal != f64::from(k) / 60.0);
        assert!(differing.count() > 0, "if these agreed the distinction would be academic");
        assert_eq!(score_of(0, 0), 0.0, "an empty probe scores zero rather than NaN");
    }

    /// D §6.4 step 2's cap is a query count, so the pose chunk depends on the probe size, and so
    /// does the floor under which a batch is the CPU's.
    #[test]
    fn the_dispatch_cap_is_twenty_million_point_queries() {
        assert_eq!(MAX_POINT_QUERIES / 60, 333_333);
        assert_eq!(MAX_POINT_QUERIES / 800, 25_000);
        // R §5.2 scores tens of thousands of hypotheses on sixty points and is never near the
        // floor; one pose on sixty points is, and belongs on the CPU.
        const { assert!(60 < MIN_QUERIES && 25_000 * 60 > MIN_QUERIES) };
        // The measured cells: 1 000 × 60 is 0.41× and stays on the CPU; 1 000 × 800 is 2.53× and
        // goes to the device.
        const { assert!(1_000 * 60 < MIN_QUERIES && 1_000 * 800 >= MIN_QUERIES) };
        const { assert!(MIN_QUERIES < MAX_POINT_QUERIES) };
    }
}
