//! The device side of `Executor::icp_rung` (D §6.4 steps 4 and 5, D §6.5).
//!
//! One dispatch runs **every iteration** of one rung for a batch of candidates: one workgroup of
//! 256 lanes per candidate, the hash grid of D §6.2 for the correspondence search, a fixed-order
//! tree over the 29 accumulators and invocation 0 doing the 6×6 LDLT or the 3×3 Umeyama. What
//! crosses the bus is the initial poses in and sixteen words per candidate out.
//!
//! # The shifted frame
//!
//! The kernel is `f32` (D §6.8), and E5 measured what `f32` point loops do in the *scans'* own
//! coordinates: the median stage-2 pose on pot G moves **9.8 t**. The scans sit 100–150 units from
//! the origin, where `f32` resolves 1e-5, while a fragment's own extent is tens of units — so this
//! module hands the kernel both clouds translated to their own centroids, in `f64`, and translates
//! the answer back. That is D §7's last row: *"Shrinking coordinates for an `f32` path means
//! translating both clouds by −c — a rigid change of frame R §7 is equivariant under — not
//! re-parameterising the Jacobian alone."*
//!
//! The equivariance is exact for R §5.4's estimator and **not** for R §5.6's, and the kernel's
//! header says why: Umeyama reads its translation off the two means, which move with the clouds,
//! while the point-to-plane step reads its translation off a linearisation about the origin of
//! whatever frame it is in. The kernel therefore adds `(R − I − ω̂)·c_t` to the point-to-plane
//! update, which makes the shifted composition the world composition exactly. Without it the path
//! would be D §7's `Assembly::Centred`, measured at 0.044 t at the median and 0.61 t at p90 on
//! terracotta — outside D §10.2's 0.01 t row.
//!
//! # What is chunked
//!
//! * **≤ 512 candidates per dispatch**, D §6.4's own bound for Windows TDR;
//! * **the correspondence array**, `candidates × n_src` words, against the 128 MB binding cap.
//!
//! # What is delegated
//!
//! An empty cloud, a non-positive or non-finite radius, a point-to-plane rung whose target has no
//! normals, and a source too large for `u32` indexing. Each is a case the CPU executor answers
//! exactly (`register` returns the unmoved pose, or converges immediately on an empty
//! correspondence set), and reproducing that on the device would buy nothing.

use std::time::{Duration, Instant};

use sherd_core::executor::batch::IcpBatch;
use sherd_core::matching::icp::{Estimation, Registration};
use sherd_core::spatial::grid::HashGrid;
use wgpu::BindingResource;

use crate::GpuError;
use crate::buffers::{self, Dispatch};
use crate::device::{DEFAULT_BINDING_CAP, Gpu};
use crate::shader::{Kernel, interleave_narrowed, uniform};

/// D §6.4's cap on one dispatch, for Windows TDR.
pub const MAX_CANDIDATES: usize = 512;

/// The smallest batch worth a dispatch, in candidates — **measured, not chosen**.
///
/// D §6.4 puts one workgroup on one candidate, so a rung of one candidate is one workgroup on a
/// sixteen-core GPU while the CPU has ten cores and SIMD. `crates/sherd-gpu/tests/adapter.rs`
/// prints the crossover table this comes from, on a 30-iteration point-to-plane rung:
///
/// | candidates | 2 000 points | 12 000 points |
/// |---|---|---|
/// | 1 | 0.32× | 0.72× |
/// | 4 | 0.30× | 0.72× |
/// | 16 | 0.85× | 1.71× |
/// | 64 | 1.55× | 2.57× |
/// | 256 | 2.24× | 4.40× |
///
/// R §9's refinement and every single-pose caller are the 0.3× row, and before this threshold
/// existed they made a `--backend gpu` terracotta run's refine stage **2.6× slower than the CPU's**
/// (1.89 s against 0.74 s) — the whole GPU gain of the matching stage, given back by a batch of
/// one.
pub const MIN_CANDIDATES: usize = 16;

/// The smallest batch worth a dispatch, in candidate-points.
///
/// The table above is not a function of the candidate count alone: sixteen candidates over 2 000
/// points is 0.85× and over 12 000 is 1.71×, so the threshold takes the product as well. 100 000
/// puts the 16 × 2 000 cell on the CPU and the 64 × 2 000 and 16 × 12 000 cells on the device,
/// which is what the measurements say.
pub const MIN_WORK: usize = 100_000;

/// Words the kernel keeps per candidate: twelve of the pose, then count, error², iterations and
/// the convergence flag.
pub const STATE_WORDS: usize = 16;

/// The WGSL source: D §6.2's grid, then the rung that queries it.
const SOURCE: &str = concat!(include_str!("kernels/grid.wgsl"), include_str!("kernels/icp.wgsl"));

/// `kernels/icp.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct IcpParams {
    centre: [f32; 4],
    candidates: u32,
    n_src: u32,
    radius: f32,
    max_iter: u32,
    wg_x: u32,
    first: u32,
    pad: [u32; 2],
}

/// The two compiled rung pipelines, one per R §7 estimator.
#[derive(Debug)]
pub struct IcpKernel {
    point: Kernel,
    plane: Kernel,
}

/// What one `icp_rung` submission measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct IcpRun {
    /// Dispatches submitted.
    pub dispatches: usize,
    /// Wall time of the dispatches alone — `submit` + `poll(Wait)`.
    pub gpu: Duration,
    /// Candidate-iterations the run could have made: candidates × `max_iteration`.
    pub iterations: usize,
}

impl IcpKernel {
    /// Compiles both entry points.
    #[must_use]
    pub fn build(gpu: &Gpu) -> Self {
        // Bindings: 0 params, 1 grid header (both uniform), then slots, counts, sorted_idx, the
        // interleaved target cloud, the source, the candidate state and the correspondences.
        let storage = [true, true, true, true, true, false, false];
        Self {
            point: Kernel::build(gpu, "icp point", SOURCE, "point_to_point", 2, &storage),
            plane: Kernel::build(gpu, "icp plane", SOURCE, "point_to_plane", 2, &storage),
        }
    }

    /// One rung for every candidate of `batch`, or `None` when the device cannot answer it.
    #[allow(clippy::too_many_lines, reason = "one submission: the frame, the buffers, the chunks")]
    /// `force` ignores [`MIN_CANDIDATES`] and [`MIN_WORK`] — the cross-check harness's switch,
    /// never a run's.
    pub fn run(
        &self,
        gpu: &Gpu,
        batch: &IcpBatch<'_>,
        force: bool,
    ) -> Result<Option<(Vec<Registration>, IcpRun)>, GpuError> {
        let n = batch.len();
        let n_src = batch.source.len();
        let radius = batch.options.max_correspondence_distance;
        if n == 0 || n_src == 0 || batch.target.is_empty() || !(radius.is_finite() && radius > 0.0)
        {
            return Ok(None);
        }
        if u32::try_from(n_src).is_err() {
            return Ok(None);
        }
        // Too small to be worth a dispatch: the CPU wins below the measured crossover, and a
        // wrong answer to that question costs more than the kernel gains (see `MIN_CANDIDATES`).
        if !force && (n < MIN_CANDIDATES || n * n_src < MIN_WORK) {
            return Ok(None);
        }
        let plane = batch.options.estimation == Estimation::PointToPlane;
        if plane && !batch.target.has_normals() {
            return Ok(None);
        }

        // The shifted frame: both clouds to their own centroids, in `f64`, once for the batch.
        let centre_t = batch.target.centroid();
        let centre_s = centroid(batch.source);
        let shifted_target = shifted(batch.target.points(), centre_t);
        let normals: Vec<[f64; 3]> = if plane {
            batch.target.normals().to_vec()
        } else {
            vec![[0.0; 3]; shifted_target.len()]
        };
        let Some(grid) = HashGrid::of_f32(&shifted_target, narrow(radius)) else {
            return Ok(None);
        };
        // D §1's ceiling, before any buffer exists (see `coarse::device_bytes`).
        let Some(_held) = gpu.allocations().reserve(device_bytes(&grid, n_src, n)) else {
            return Ok(None);
        };
        let cloud = interleave_narrowed(grid.points(), &normals);
        let source: Vec<[f32; 4]> = batch
            .source
            .iter()
            .map(|p| {
                [
                    narrow(p[0] - centre_s[0]),
                    narrow(p[1] - centre_s[1]),
                    narrow(p[2] - centre_s[2]),
                    0.0,
                ]
            })
            .collect();

        let header = uniform(gpu, "icp grid", &grid.header());
        let slots = buffers::upload(gpu, "icp slots", grid.slots());
        let counts = buffers::upload(gpu, "icp counts", grid.counts());
        let indices = buffers::upload(gpu, "icp sorted_idx", grid.sorted_idx());
        let cloud_buffer = buffers::upload(gpu, "icp target", &cloud);
        let source_buffer = buffers::upload(gpu, "icp source", &source);

        // D §6.4's TDR bound, and the binding cap on the correspondence array.
        let by_binding = usize::try_from(DEFAULT_BINDING_CAP / (4 * n_src as u64)).unwrap_or(1);
        let per_dispatch = MAX_CANDIDATES.min(by_binding).max(1);
        let corres =
            buffers::output(gpu, "icp corres", (per_dispatch.min(n) as u64) * (n_src as u64) * 4);

        let kernel = if plane { &self.plane } else { &self.point };
        let mut run = IcpRun {
            dispatches: 0,
            gpu: Duration::ZERO,
            iterations: n * batch.options.max_iteration,
        };
        let staging = buffers::staging(gpu, (n * STATE_WORDS * 4) as u64);
        let mut encoder = kernel.encoder(gpu);
        let mut held = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + per_dispatch).min(n);
            let chunk = end - start;
            let initial: Vec<f32> = batch.inits[start..end]
                .iter()
                .flat_map(|init| shifted_state(init, &centre_s, &centre_t))
                .collect();
            let state = buffers::upload(gpu, "icp state", &initial);
            let shape = Dispatch::for_workgroups(u32::try_from(chunk).unwrap_or(1));
            let params = IcpParams {
                centre: [narrow(centre_t[0]), narrow(centre_t[1]), narrow(centre_t[2]), 0.0],
                candidates: u32::try_from(chunk).unwrap_or(u32::MAX),
                n_src: u32::try_from(n_src).unwrap_or(u32::MAX),
                radius: narrow(radius),
                max_iter: u32::try_from(batch.options.max_iteration).unwrap_or(u32::MAX),
                wg_x: shape.x,
                first: 0,
                pad: [0; 2],
            };
            let params_buffer = uniform(gpu, "icp params", &params);
            let bind = kernel.bind(
                gpu,
                &[
                    params_buffer.as_entire_binding(),
                    header.as_entire_binding(),
                    slots.as_entire_binding(),
                    counts.as_entire_binding(),
                    indices.as_entire_binding(),
                    cloud_buffer.as_entire_binding(),
                    source_buffer.as_entire_binding(),
                    BindingResource::Buffer(state.as_entire_buffer_binding()),
                    BindingResource::Buffer(corres.as_entire_buffer_binding()),
                ],
            );
            // Every chunk's dispatch goes into one command buffer, in order — the passes of a
            // command buffer execute in the order they were recorded, which is what lets the
            // chunks share one correspondence array — and each writes its own slice of the one
            // staging buffer the host maps at the end.
            kernel.record(&mut encoder, &bind, shape);
            encoder.copy_buffer_to_buffer(
                &state,
                0,
                &staging,
                (start * STATE_WORDS * 4) as u64,
                (chunk * STATE_WORDS * 4) as u64,
            );
            held.push((params_buffer, state, bind));
            run.dispatches += 1;
            start = end;
        }
        let started = Instant::now();
        let words = buffers::submit_and_read::<f32>(
            gpu,
            "icp state",
            encoder.finish(),
            staging,
            n * STATE_WORDS,
        )?;
        run.gpu = started.elapsed();
        drop(held);
        let out: Vec<Registration> = (0..n)
            .map(|c| {
                registration(
                    &words[c * STATE_WORDS..(c + 1) * STATE_WORDS],
                    &centre_s,
                    &centre_t,
                    n_src,
                )
            })
            .collect();
        Ok(Some((out, run)))
    }
}

/// What one call of this kernel puts on the device (D §1, `--gpu-memory`).
///
/// The grid's three arrays, the interleaved target cloud, the source, one correspondence word per
/// source point per candidate of the largest dispatch, and the candidate state twice — once on the
/// device, once in the staging buffer it is copied into.
fn device_bytes(grid: &HashGrid, n_src: usize, candidates: usize) -> u64 {
    let cell = |n: usize, stride: usize| (n as u64) * (stride as u64);
    let per_dispatch = MAX_CANDIDATES.min(candidates).max(1);
    cell(grid.slots().len(), 16)
        + cell(grid.counts().len(), 4)
        + cell(grid.sorted_idx().len(), 4)
        + cell(grid.points().len(), 32)
        + cell(n_src, 16)
        + cell(per_dispatch, 4) * (n_src as u64)
        + cell(candidates, 4 * STATE_WORDS) * 2
        + 256
}

/// The poses the device actually starts from, back in world coordinates: `init` through the
/// shifted `f32` state and out again.
///
/// This is the **control** the cross-check needs. A rung that ends somewhere else on the GPU has
/// two possible reasons — the kernel's `f32` arithmetic, or the ladder amplifying the `f32`
/// rounding of its *starting pose* — and only one of them is the kernel's. Feeding these poses to
/// the CPU executor answers the second question with the reference implementation's own `f64`
/// arithmetic, so the difference that is left is the kernel's.
///
/// It is deliberately not "`init` rounded through `f32`": the shift of [`shifted_state`] is what
/// buys this path its precision, and the rounding a `--backend gpu` run actually applies to a pose
/// is the rounding of the *shifted* translation, which is a few tens of units rather than 150.
#[must_use]
pub fn device_round_trip(
    inits: &[sherd_core::matching::icp::Pose],
    source: &[[f64; 3]],
    target: &sherd_core::matching::icp::IcpTarget,
) -> Vec<sherd_core::matching::icp::Pose> {
    let centre_t = target.centroid();
    let centre_s = centroid(source);
    inits
        .iter()
        .map(|init| {
            let words = shifted_state(init, &centre_s, &centre_t);
            let mut back = [0.0_f32; STATE_WORDS];
            back[..12].copy_from_slice(&words[..12]);
            registration(&back, &centre_s, &centre_t, source.len().max(1)).transform
        })
        .collect()
}

/// The mean of a point set, summed in index order (D §7).
fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0; 3];
    }
    let mut sum = [0.0; 3];
    for p in points {
        for (out, value) in sum.iter_mut().zip(p) {
            *out += value;
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "cloud sizes are far below 2^53")]
    let n = points.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// A cloud translated by `−centre`, narrowed once.
#[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
fn shifted(points: &[[f64; 3]], centre: [f64; 3]) -> Vec<[f32; 3]> {
    points
        .iter()
        .map(|p| [(p[0] - centre[0]) as f32, (p[1] - centre[1]) as f32, (p[2] - centre[2]) as f32])
        .collect()
}

/// The sixteen words one candidate starts with: the pose in the shifted frame, then four zeros.
///
/// `T'` maps the shifted source to the shifted target: `p' = R(p − c_s) + (R c_s + τ − c_t)`. The
/// translation is formed in `f64` here, where `R c_s` and `c_t` are both ≈ 150 units and their
/// difference is the small number the kernel needs; doing it on the device would be the one
/// `f32` cancellation this whole change of frame exists to avoid.
fn shifted_state(
    init: &sherd_core::matching::icp::Pose,
    centre_s: &[f64; 3],
    centre_t: &[f64; 3],
) -> [f32; STATE_WORDS] {
    let mut words = [0.0_f32; STATE_WORDS];
    for i in 0..3 {
        let mut tau = init[(i, 3)] - centre_t[i];
        for (j, &c) in centre_s.iter().enumerate() {
            words[i * 4 + j] = narrow(init[(i, j)]);
            tau += init[(i, j)] * c;
        }
        words[i * 4 + 3] = narrow(tau);
    }
    words
}

/// The candidate's answer, back in world coordinates.
///
/// `τ_world = τ' − R c_s + c_t`, in `f64`, undoing [`shifted_state`]. The rotation is the `f32`
/// the kernel produced, widened exactly; the fitness and the RMSE are recomputed here from the
/// integer count and the accumulated `error²` with the CPU executor's own arithmetic —
/// `|C| / n_source` and `sqrt(error² / |C|)` — so that the only difference from the reference is
/// the `f32` sum the device made, not a second division.
fn registration(
    words: &[f32],
    centre_s: &[f64; 3],
    centre_t: &[f64; 3],
    n_src: usize,
) -> Registration {
    let mut transform = sherd_core::matching::icp::Pose::identity();
    for i in 0..3 {
        let mut tau = f64::from(words[i * 4 + 3]) + centre_t[i];
        for (j, &c) in centre_s.iter().enumerate() {
            let r = f64::from(words[i * 4 + j]);
            transform[(i, j)] = r;
            tau -= r * c;
        }
        transform[(i, 3)] = tau;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the kernel writes an integer count as an exactly representable f32"
    )]
    let correspondences = words[12].max(0.0) as usize;
    let error2 = f64::from(words[13]);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the iteration count is bounded by `max_iteration`"
    )]
    let iterations = words[14].max(0.0) as usize;
    #[allow(clippy::cast_precision_loss, reason = "cloud sizes are far below 2^53")]
    let (n_corres, n_source) = (correspondences as f64, n_src as f64);
    let (fitness, inlier_rmse) = if correspondences == 0 {
        (0.0, 0.0)
    } else {
        (n_corres / n_source, (error2 / n_corres).sqrt())
    };
    Registration {
        transform,
        fitness,
        inlier_rmse,
        correspondences,
        iterations,
        converged: words[15] != 0.0,
    }
}

/// The `f32` a threshold or a coordinate becomes on the way to the device.
#[inline]
#[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
fn narrow(x: f64) -> f32 {
    x as f32
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "a change of frame is exact or it is wrong")]

    use super::{
        MAX_CANDIDATES, MIN_CANDIDATES, MIN_WORK, STATE_WORDS, centroid, registration, shifted,
        shifted_state,
    };
    use sherd_core::matching::icp::{Pose, Rotation, Translation, homogeneous};

    /// The change of frame is a round trip: a pose narrowed into the shifted frame and read back
    /// comes out where it went in, to `f32` of the *shifted* magnitudes rather than the world's.
    ///
    /// That is the whole point of the shift. A pose whose translation is 150 units from the origin
    /// carries 1e-5 of `f32` resolution there; expressed about the two centroids the same pose is
    /// a few units and carries 1e-6.
    #[test]
    fn the_shift_is_a_round_trip_at_the_shifted_magnitudes() {
        let angle = 0.37_f64;
        let (s, c) = angle.sin_cos();
        let r = Rotation::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
        let centre_s = [141.0, -128.0, 88.0];
        let centre_t = [139.5, -130.25, 90.5];
        // A pose an ICP rung is actually handed: one that already lands the source cloud roughly
        // on the target. That is the precondition the shift needs — an arbitrary pose leaves
        // `R c_s + τ` far from `c_t` and gains nothing, and R §5.4's rungs start from R §5.1's
        // hypotheses, which align the two breaklines by construction.
        let source_centre = Translation::new(centre_s[0], centre_s[1], centre_s[2]);
        let target_centre = Translation::new(centre_t[0], centre_t[1], centre_t[2]);
        let tau = target_centre - r * source_centre + Translation::new(1.5, -0.75, 0.25);
        let init = homogeneous(&r, &tau);

        let words = shifted_state(&init, &centre_s, &centre_t);
        // The shifted translation is small: that is what buys the precision.
        let shifted_tau = (0..3).map(|i| f64::from(words[i * 4 + 3]).abs()).fold(0.0_f64, f64::max);
        assert!(
            shifted_tau < 60.0,
            "the shifted translation is a few tens of units: {shifted_tau}"
        );

        let mut back = [0.0_f32; STATE_WORDS];
        back[..12].copy_from_slice(&words[..12]);
        back[12] = 1000.0;
        back[13] = 40.0;
        back[14] = 7.0;
        back[15] = 1.0;
        let out = registration(&back, &centre_s, &centre_t, 4000);
        for i in 0..3 {
            for j in 0..4 {
                let delta = (out.transform[(i, j)] - init[(i, j)]).abs();
                assert!(delta < 2e-5, "({i}, {j}) moved by {delta}");
            }
        }
        assert_eq!(out.correspondences, 1000);
        assert_eq!(out.fitness, 0.25);
        assert_eq!(out.inlier_rmse, (40.0_f64 / 1000.0).sqrt());
        assert_eq!(out.iterations, 7);
        assert!(out.converged);

        // An identity pose about equal centroids is the identity again, exactly.
        let identity = shifted_state(&Pose::identity(), &[0.0; 3], &[0.0; 3]);
        assert_eq!(&identity[..12], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    /// The centroid is D §7's: summed in index order, then divided once.
    #[test]
    fn the_centroid_is_summed_in_index_order() {
        assert_eq!(centroid(&[]), [0.0; 3]);
        let points = vec![[1.0, 2.0, 3.0], [3.0, 4.0, 9.0]];
        assert_eq!(centroid(&points), [2.0, 3.0, 6.0]);
        let narrowed = shifted(&points, [2.0, 3.0, 6.0]);
        assert_eq!(narrowed, vec![[-1.0, -1.0, -3.0], [1.0, 1.0, 3.0]]);
    }

    /// D §6.4's TDR bound is 512 candidates, the state is sixteen words, and the crossover the
    /// adapter test measured is what decides a batch of one.
    #[test]
    fn the_dispatch_bounds_are_the_designs() {
        assert_eq!(MAX_CANDIDATES, 512);
        assert_eq!(STATE_WORDS, 16);
        // R §9's refinement and every single-pose caller: one candidate, always the CPU.
        const { assert!(1 < MIN_CANDIDATES) };
        // The measured cells: 16 × 2 000 is 0.85× and stays on the CPU; 16 × 12 000 is 1.71× and
        // 64 × 2 000 is 1.55×, and both go to the device.
        const { assert!(16 * 2_000 < MIN_WORK) };
        const { assert!(16 * 12_000 >= MIN_WORK && 64 * 2_000 >= MIN_WORK) };
    }
}
