//! The device side of `Executor::icp_rung` (D §6.4 steps 4 and 5, D §6.5).
//!
//! One dispatch runs **every iteration** of one rung for a batch of candidates: one workgroup of
//! 256 lanes per candidate, the hash grid of D §6.2 for the correspondence search, a fixed-order
//! tree over the 29 accumulators and invocation 0 doing the 6×6 LDLT or the 3×3 Umeyama. What
//! crosses the bus is the initial poses in and seventeen words per candidate out.
//!
//! # What is checked before an answer is believed
//!
//! Task H1 measured (`notes/2026-09-09-h1-wd1.md`) that on this Metal adapter a command buffer the
//! driver aborts — which is what a second process driving the same device produces, several times
//! a minute — resolves its fence and its `map_async` as **success**: `wgpu-hal`'s Metal `Fence`
//! counts `MTLCommandBufferStatus::Error` as completion, and the signal handler is on the last
//! command buffer of a submission rather than on the compute pass that failed. So `Ok` from the
//! pipeline says the host did not fail; it does not say the device ran.
//!
//! [`registration`] therefore refuses a state block it cannot vouch for, and one refused block
//! sends the **whole batch** to the CPU executor and increments `MethodStats::corrupt`. The
//! receipts are the [`buffers::SENTINEL`](crate::buffers::SENTINEL) fill of the staging buffer —
//! an unwritten word comes back as a NaN rather than as a plausible zero or as the previous
//! call's answer — and a per-call **nonce** in word 16 the kernel must give back as `nonce + 1`.
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

use std::sync::atomic::{AtomicU32, Ordering};
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

/// The size policy of D §6.4, as one predicate: is a rung of `candidates` candidates over
/// `points` source points worth a dispatch?
///
/// [`IcpKernel::run`] applies it and [`STAGE2_ON_DEVICE`] reads it; the cross-check harness
/// reports what it decided rather than inferring it from two constants.
#[must_use]
pub const fn on_device(candidates: usize, points: usize) -> bool {
    candidates >= MIN_CANDIDATES && candidates * points >= MIN_WORK
}

/// R §5.6's stage-2 candidate count in a production run — `Params::stage2`, the `--candidates`
/// flag's default (R §1.1).
///
/// Named here so that the line below is a statement about the production policy and not about a
/// number that happens to be small. The unit test at the bottom of this file asserts that it *is*
/// `Params::default().stage2`, so a change to R §1.1 breaks the build rather than moving a
/// criterion quietly.
pub const STAGE2_CANDIDATES: usize = 10;

/// **Policy: R §5.6's stage-2 ladder is answered by the CPU, on every collection and every
/// adapter** (audit §A.2.4, D §12's 2b row).
///
/// Ten candidates is under [`MIN_CANDIDATES`], so no stage-2 rung of a production run can reach
/// the device whatever its source cloud holds — but that was a *consequence* of two measured
/// thresholds until this constant existed, and a future tuning of either could have moved a
/// verification criterion without anyone deciding to. The exit criterion of D §12's 2b row is
/// stated at this policy: the device does R §5.2's coarse score and R §5.4's two stage-1
/// breakline rungs, and `gpu-check` prints `cpu by policy` on every stage-2 row rather than a
/// deviation nothing in a run would ever produce.
///
/// The f32/f64 divergence this policy leaves outside the criterion is measured and understood
/// (D §6.7): it is a stopping-rule discontinuity, not a precision shortfall, and task W's
/// double-single experiment moved none of it.
pub const STAGE2_ON_DEVICE: bool = STAGE2_CANDIDATES >= MIN_CANDIDATES;

const _: () = assert!(
    !STAGE2_ON_DEVICE,
    "R §5.6's stage 2 is the CPU's by policy: `STAGE2_CANDIDATES` must stay under `MIN_CANDIDATES`"
);

/// Words the kernel keeps per candidate: twelve of the pose, then count, error², iterations, the
/// convergence flag and the nonce of [`NONCE`].
pub const STATE_WORDS: usize = 17;

/// The per-call nonce every state block carries in and out (task H1, audit §A.1.6).
///
/// The host writes `n` into word 16 of every candidate's initial state and the kernel writes
/// `n + 1` back. Values are exact `f32` integers below `2^23`, so both the addition and the
/// comparison are exact; the counter wraps there, which means a *stale* block old enough to have
/// travelled 8 388 600 calls round the counter would pass — a run makes a few thousand.
static NONCE: AtomicU32 = AtomicU32::new(0);

/// The nonce of the next call.
fn next_nonce() -> f32 {
    let n = NONCE.fetch_add(1, Ordering::Relaxed) % 8_388_600;
    #[allow(clippy::cast_precision_loss, reason = "n < 2^23 is exact in f32")]
    let nonce = (n + 1) as f32;
    nonce
}

/// How far from orthonormal a rotation the device wrote may be before it is refused: `max
/// |RᵀR − I|` over the nine entries, and `|det R − 1|`.
///
/// The audit's number, and far looser than the `f32` rounding of a rung that ran. On the seven
/// development collections it fires **once**, and not on a corrupt readback at all: `synthetic_20`
/// has one stage-1 candidate whose covariance comes out rank-deficient, and
/// `kernels/icp.wgsl`'s `umeyama_rotation` completes a rank-1 `U` to zero columns rather than to
/// an orthonormal basis, so it returns a matrix that is 1.0 from orthonormal with `det = 0`. That
/// is a real defect of the kernel (task H1, `notes/2026-09-09-h1-wd1.md` §4, H1-D1) that this
/// check found on its first production run; until it is fixed the batch is answered by the CPU
/// executor, which is the reference implementation's answer to it.
pub const ORTHONORMAL_TOLERANCE: f64 = 1e-3;

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

/// What [`IcpKernel::run`] decided about one batch.
#[derive(Debug)]
pub enum IcpAnswer {
    /// The device answered, and every state block passed [`registration`]'s checks.
    Device(Vec<Registration>, IcpRun),
    /// The batch is not the device's — an empty cloud, a rung under the size thresholds, a
    /// point-to-plane rung with no target normals, a reservation the budget refused. The CPU
    /// executor answers it and the call is counted as delegated, exactly as before.
    Cpu,
    /// The device was asked and what came back could not be believed: the whole batch goes to the
    /// CPU and is counted as `corrupt` as well as delegated. The string names the first candidate
    /// that failed, the check it failed and the census of the readback.
    Corrupt(String),
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

    /// One rung for every candidate of `batch`, or the reason the CPU has to answer it.
    #[allow(clippy::too_many_lines, reason = "one submission: the frame, the buffers, the chunks")]
    /// `force` ignores [`MIN_CANDIDATES`] and [`MIN_WORK`] — the cross-check harness's switch,
    /// never a run's.
    pub fn run(&self, gpu: &Gpu, batch: &IcpBatch<'_>, force: bool) -> Result<IcpAnswer, GpuError> {
        let n = batch.len();
        let n_src = batch.source.len();
        let radius = batch.options.max_correspondence_distance;
        if n == 0 || n_src == 0 || batch.target.is_empty() || !(radius.is_finite() && radius > 0.0)
        {
            return Ok(IcpAnswer::Cpu);
        }
        if u32::try_from(n_src).is_err() {
            return Ok(IcpAnswer::Cpu);
        }
        // Too small to be worth a dispatch: the CPU wins below the measured crossover, and a
        // wrong answer to that question costs more than the kernel gains (see `MIN_CANDIDATES`).
        if !force && !on_device(n, n_src) {
            return Ok(IcpAnswer::Cpu);
        }
        let plane = batch.options.estimation == Estimation::PointToPlane;
        if plane && !batch.target.has_normals() {
            return Ok(IcpAnswer::Cpu);
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
            return Ok(IcpAnswer::Cpu);
        };
        // D §1's ceiling, before any buffer exists (see `coarse::device_bytes`).
        let Some(_held) = gpu.allocations().reserve(device_bytes(&grid, n_src, n)) else {
            return Ok(IcpAnswer::Cpu);
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

        let nonce = next_nonce();
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
                .flat_map(|init| shifted_state(init, &centre_s, &centre_t, nonce))
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
        let mut out = Vec::with_capacity(n);
        for c in 0..n {
            let block = &words[c * STATE_WORDS..(c + 1) * STATE_WORDS];
            match registration(
                block,
                &centre_s,
                &centre_t,
                n_src,
                Check { nonce, max_iter: batch.options.max_iteration },
            ) {
                Ok(registration) => out.push(registration),
                // One bad block condemns the batch: the readback is one copy of one buffer, so a
                // block the device did not write says nothing about the blocks beside it, and the
                // CPU executor's answer to the whole batch is the only one that is known to be
                // R §7's.
                Err(why) => {
                    return Ok(IcpAnswer::Corrupt(format!(
                        "candidate {c} of {n}: {why} (it reports {} correspondences of {n_src} \
                         after {} iterations); {}; readback {}",
                        block[12],
                        block[14],
                        receipts(&words, n, nonce),
                        buffers::census(&words)
                    )));
                }
            }
        }
        Ok(IcpAnswer::Device(out, run))
    }
}

/// Which candidates of a refused readback carry this call's receipt, which still carry the nonce
/// the host uploaded, and which carry neither.
///
/// This is the audit's experiment E6 in the form the measurement made possible
/// (`notes/2026-09-09-h1-wd1.md`). E6 asks for a per-iteration trace, to tell an aborted command
/// buffer (A.1.3) from a driver that corrupted a reduction mid-kernel (A.1.4): *"garbage from
/// iteration 0 is A.1.3; a one-ULP drift from iteration k is A.1.4."* A candidate whose word 16
/// still holds the nonce **the host wrote** has not reached `store_state` at all, and one whose
/// word 16 holds `nonce + 1` has finished the whole rung; a drift would be a third class, a
/// finished candidate whose answer is wrong, and every refusal task H1 measured had none.
#[allow(
    clippy::float_cmp,
    reason = "the nonce is an exact f32 integer and the receipt is exactly it, or it is not"
)]
fn receipts(words: &[f32], n: usize, nonce: f32) -> String {
    let (mut done, mut untouched, mut neither) = (0, 0, 0);
    for c in 0..n {
        match words.get(c * STATE_WORDS + 16) {
            Some(&w) if w == nonce + 1.0 => done += 1,
            Some(&w) if w == nonce => untouched += 1,
            _ => neither += 1,
        }
    }
    format!(
        "{done} of {n} candidates finished the rung, {untouched} were never started, \
         {neither} carry neither receipt"
    )
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
            let words = shifted_state(init, &centre_s, &centre_t, 0.0);
            pose_of(&words[..12], &centre_s, &centre_t)
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
    nonce: f32,
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
    words[16] = nonce;
    words
}

/// What [`registration`] has to be told before it can vouch for a state block.
#[derive(Clone, Copy, Debug)]
pub struct Check {
    /// The nonce this call wrote into word 16 of every block; the kernel owes `nonce + 1`.
    pub nonce: f32,
    /// `ICPConvergenceCriteria::max_iteration_`: a rung cannot have applied more.
    pub max_iter: usize,
}

/// The candidate's answer, back in world coordinates — or the check it failed.
///
/// `τ_world = τ' − R c_s + c_t`, in `f64`, undoing [`shifted_state`]. The rotation is the `f32`
/// the kernel produced, widened exactly; the fitness and the RMSE are recomputed here from the
/// integer count and the accumulated `error²` with the CPU executor's own arithmetic —
/// `|C| / n_source` and `sqrt(error² / |C|)` — so that the only difference from the reference is
/// the `f32` sum the device made, not a second division.
///
/// # What is checked, and why each check is here
///
/// Each check exists because a *readback the device never wrote* has a characteristic shape (task
/// H1, `notes/2026-09-09-h1-wd1.md`), and every one of them is a property the kernel's output has
/// by construction — with the single measured exception in the third row.
///
/// | check | what it catches |
/// |---|---|
/// | every word finite | the [`SENTINEL`](crate::buffers::SENTINEL) fill of a staging buffer whose copy never ran, and any NaN the arithmetic could not have made |
/// | word 16 is `nonce + 1` | a block from a **previous call** — a recycled allocation, or, which is what this machine actually produces, a state buffer whose dispatch was aborted and which therefore still holds the nonce the host uploaded rather than its successor |
/// | rotation orthonormal to [`ORTHONORMAL_TOLERANCE`] with `det > 0` | a page of zeros, which would otherwise decode as a plausible "0 iterations, not converged" answer at the target centroid — **and** H1-D1, the rank-deficient `umeyama_rotation` of [`ORTHONORMAL_TOLERANCE`]'s own note |
/// | `0 ≤ count ≤ n_src`, integral | a word that is not the count the kernel writes |
/// | `error² ≥ 0` | the same, on the residual |
/// | `0 ≤ iterations ≤ max_iter`, integral | a rung that cannot have run |
/// | the convergence flag is 0 or 1 | the same, on the flag |
#[allow(
    clippy::float_cmp,
    reason = "every comparison here is of a word the kernel writes exactly or does not write"
)]
fn registration(
    words: &[f32],
    centre_s: &[f64; 3],
    centre_t: &[f64; 3],
    n_src: usize,
    check: Check,
) -> Result<Registration, String> {
    if words.len() < STATE_WORDS {
        return Err(format!("{} words of {STATE_WORDS}", words.len()));
    }
    if let Some(k) = words.iter().position(|w| !w.is_finite()) {
        return Err(format!("word {k} is {} and not finite", words[k]));
    }
    if words[16] != check.nonce + 1.0 {
        return Err(format!(
            "nonce {} came back as {} rather than {}",
            check.nonce,
            words[16],
            check.nonce + 1.0
        ));
    }
    let transform = pose_of(words, centre_s, centre_t);
    if let Some(defect) = orthonormality(&transform) {
        return Err(format!(
            "the rotation is {defect:.3e} from orthonormal, over {ORTHONORMAL_TOLERANCE:.0e}"
        ));
    }
    #[allow(clippy::cast_precision_loss, reason = "cloud sizes are far below 2^53")]
    let source = n_src as f64;
    let correspondences = count_word(words[12], source, "the correspondence count")?;
    if words[13] < 0.0 {
        return Err(format!("the squared error is {}", words[13]));
    }
    #[allow(clippy::cast_precision_loss, reason = "max_iteration is a small integer")]
    let iterations = count_word(words[14], check.max_iter as f64, "the iteration count")?;
    if words[15] != 0.0 && words[15] != 1.0 {
        return Err(format!("the convergence flag is {}", words[15]));
    }
    let error2 = f64::from(words[13]);
    #[allow(clippy::cast_precision_loss, reason = "cloud sizes are far below 2^53")]
    let n_corres = correspondences as f64;
    let (fitness, inlier_rmse) = if correspondences == 0 {
        (0.0, 0.0)
    } else {
        (n_corres / source, (error2 / n_corres).sqrt())
    };
    Ok(Registration {
        transform,
        fitness,
        inlier_rmse,
        correspondences,
        iterations,
        converged: words[15] != 0.0,
    })
}

/// A word the kernel wrote as `f32(k)` for an integer `k` in `0 ..= limit`, back as a `usize`.
fn count_word(word: f32, limit: f64, what: &str) -> Result<usize, String> {
    let value = f64::from(word);
    if value < 0.0 || value > limit || value.fract() != 0.0 {
        return Err(format!("{what} is {word}, not an integer in 0..={limit}"));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "checked against `limit` immediately above"
    )]
    Ok(value as usize)
}

/// How far `pose`'s rotation is from orthonormal, or `None` when it is inside
/// [`ORTHONORMAL_TOLERANCE`] and properly oriented.
///
/// `max |RᵀR − I|` over the nine entries, and the determinant, both in `f64` over the widened
/// `f32` the device wrote.
fn orthonormality(pose: &sherd_core::matching::icp::Pose) -> Option<f64> {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let mut dot = 0.0;
            for k in 0..3 {
                dot += pose[(k, i)] * pose[(k, j)];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    let det = pose.fixed_view::<3, 3>(0, 0).determinant();
    if worst > ORTHONORMAL_TOLERANCE || (det - 1.0).abs() > ORTHONORMAL_TOLERANCE {
        return Some(worst.max((det - 1.0).abs()));
    }
    None
}

/// The twelve pose words back in world coordinates, with nothing checked — the arithmetic
/// [`registration`] and [`device_round_trip`] share.
fn pose_of(
    words: &[f32],
    centre_s: &[f64; 3],
    centre_t: &[f64; 3],
) -> sherd_core::matching::icp::Pose {
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
    transform
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
        Check, MAX_CANDIDATES, MIN_CANDIDATES, ORTHONORMAL_TOLERANCE, STAGE2_CANDIDATES,
        STAGE2_ON_DEVICE, STATE_WORDS, centroid, next_nonce, on_device, registration, shifted,
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

        let nonce = 41.0;
        let words = shifted_state(&init, &centre_s, &centre_t, nonce);
        assert_eq!(words[16], nonce, "the nonce goes out in word 16");
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
        back[16] = nonce + 1.0;
        let check = Check { nonce, max_iter: 30 };
        let out =
            registration(&back, &centre_s, &centre_t, 4000, check).expect("a well-formed block");
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
        let identity = shifted_state(&Pose::identity(), &[0.0; 3], &[0.0; 3], 1.0);
        assert_eq!(&identity[..12], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    /// A state block the device never wrote is refused, in each of the three shapes task H1
    /// measured it in (`notes/2026-09-09-h1-wd1.md`, audit §A.1.6).
    ///
    /// This is the test the audit's §B.1 asks for. Every one of these blocks decodes into a
    /// perfectly plausible `Registration` under the old decoder — a zero block into "0 iterations,
    /// not converged, at the target centroid", a stale block into the *previous* call's pose — and
    /// the pipeline has no way to tell either from an answer.
    #[test]
    fn a_block_the_device_did_not_write_is_refused() {
        let centre_s = [141.0, -128.0, 88.0];
        let centre_t = [139.5, -130.25, 90.5];
        let nonce = 77.0;
        let check = Check { nonce, max_iter: 30 };
        let good = |nonce: f32| {
            let mut w = [0.0_f32; STATE_WORDS];
            w[0] = 1.0;
            w[5] = 1.0;
            w[10] = 1.0;
            w[12] = 12.0;
            w[13] = 3.0;
            w[14] = 9.0;
            w[15] = 1.0;
            w[16] = nonce + 1.0;
            w
        };
        let refusal = |w: &[f32; STATE_WORDS]| {
            registration(w, &centre_s, &centre_t, 4000, check).expect_err("refused")
        };
        assert!(registration(&good(nonce), &centre_s, &centre_t, 4000, check).is_ok());

        // A fresh allocation the copy never reached.
        let zero = [0.0_f32; STATE_WORDS];
        let why = refusal(&zero);
        assert!(why.contains("nonce"), "{why}");

        // A staging buffer still holding `buffers::SENTINEL`, and a single NaN word.
        let sentinel = [f32::from_bits(crate::buffers::SENTINEL); STATE_WORDS];
        assert!(refusal(&sentinel).contains("not finite"));
        let mut nan = good(nonce);
        nan[3] = f32::NAN;
        assert!(refusal(&nan).contains("not finite"));

        // A recycled buffer holding the **previous** call's answer: every word plausible, every
        // check but the nonce satisfied.
        let stale = good(nonce - 1.0);
        let why = refusal(&stale);
        assert!(why.contains("nonce 77 came back as 77 rather than 78"), "{why}");

        // The rest of the checks, one block each.
        let mut skew = good(nonce);
        skew[1] = 0.01;
        assert!(refusal(&skew).contains("orthonormal"), "1e-2 is over the 1e-3 tolerance");
        let mut mirrored = good(nonce);
        mirrored[10] = -1.0;
        assert!(refusal(&mirrored).contains("orthonormal"), "a reflection is not a rotation");
        let mut counted = good(nonce);
        counted[12] = 4001.0;
        assert!(refusal(&counted).contains("correspondence count"));
        counted[12] = 12.5;
        assert!(refusal(&counted).contains("correspondence count"), "not an integer");
        let mut residual = good(nonce);
        residual[13] = -1.0;
        assert!(refusal(&residual).contains("squared error"));
        let mut iterated = good(nonce);
        iterated[14] = 31.0;
        assert!(refusal(&iterated).contains("iteration count"), "over max_iter");
        let mut flagged = good(nonce);
        flagged[15] = 2.0;
        assert!(refusal(&flagged).contains("convergence flag"));
        assert!(registration(&good(nonce)[..12], &centre_s, &centre_t, 4000, check).is_err());

        // A rotation inside the tolerance is not refused: the check is for a block that was never
        // written, not for the kernel's own `f32` rounding.
        let mut rounded = good(nonce);
        #[allow(clippy::cast_possible_truncation, reason = "the tolerance is a small f64")]
        let inside = (ORTHONORMAL_TOLERANCE / 10.0) as f32;
        rounded[1] = inside;
        assert!(registration(&rounded, &centre_s, &centre_t, 4000, check).is_ok());
    }

    /// The nonce is an exact `f32` integer, never zero, and moves on every call.
    #[test]
    fn the_nonce_is_exact_and_never_repeats_inside_a_run() {
        let a = next_nonce();
        let b = next_nonce();
        assert!(a >= 1.0 && b >= 1.0, "zero is never issued: {a}, {b}");
        assert_ne!(a, b);
        for n in [a, b] {
            assert_eq!(n.fract(), 0.0);
            assert_eq!(n + 1.0 - 1.0, n, "the increment the kernel makes is exact");
        }
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
        // Sixteen words of answer and one of nonce (task H1).
        assert_eq!(STATE_WORDS, 17);
        // R §9's refinement and every single-pose caller: one candidate, always the CPU.
        const { assert!(1 < MIN_CANDIDATES) };
        // The measured cells: 16 × 2 000 is 0.85× and stays on the CPU; 16 × 12 000 is 1.71× and
        // 64 × 2 000 is 1.55×, and both go to the device.
        const { assert!(!on_device(16, 2_000)) };
        const { assert!(on_device(16, 12_000) && on_device(64, 2_000)) };
    }

    /// The stage-2 policy is a statement about R §1.1's own parameter, not about a number that
    /// happens to be small (audit §A.2.4).
    ///
    /// If `--candidates`' default ever rises to sixteen, this test fails and the exit criterion of
    /// D §12's 2b row has to be restated before the build goes green again.
    #[test]
    fn stage_two_is_the_cpus_by_policy_and_the_policy_is_r_1_1s_own_number() {
        assert_eq!(
            usize::try_from(sherd_core::Params::default().stage2).expect("a small count"),
            STAGE2_CANDIDATES,
            "`STAGE2_CANDIDATES` is R §1.1's `--candidates` default"
        );
        const { assert!(!STAGE2_ON_DEVICE) };
        // And it is the candidate count that decides it, whatever the fracture cloud holds.
        const { assert!(!on_device(STAGE2_CANDIDATES, 1_000_000)) };
    }
}
