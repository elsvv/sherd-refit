//! [`GpuExecutor`] — the device side of D §6.1's four inner loops.
//!
//! # Which of the four run on the device
//!
//! | method | kernel | phase | state |
//! |---|---|---|---|
//! | `coarse_scores` | `kernels/coarse.wgsl`, one workgroup per pose | 2b | **on the device** |
//! | `icp_rung` | `kernels/icp.wgsl`, one workgroup per candidate, every iteration inside | 2b | **on the device**, both estimators |
//! | `bounded_distance` | BVH closest point with `r_max` | 2c | delegated to the CPU |
//! | `inside` | AABB reject → parity rays → depth | 2c | delegated to the CPU |
//!
//! Nothing here silently pretends to be a GPU result. Every method that falls through to
//! [`CpuExecutor`](sherd_core::executor::CpuExecutor) counts the call, per method, and
//! `sherd-refit-rs gpu-check` prints `delegated` for those rows rather than a deviation of zero.
//! A batch the kernels cannot form — an empty probe, a breakline the grid cannot be built over, a
//! rung with no target normals — is counted the same way and answered by the CPU: it is a
//! fall-through, not a failure.
//!
//! # What a device error does
//!
//! It falls back to the CPU for that batch and is counted, rather than aborting the run. D §6.8
//! says a device loss mid-run "falls back to the CPU for the remaining blocks and is recorded in
//! the report", and this is the per-batch form of that: the answer is still the reference
//! implementation's, and [`MethodStats::host_errors`] says how often the device could not give
//! one.
//!
//! **`host_errors` is named for what it can see.** It counts a `Result::Err` on this side of the
//! bus — a map that failed, a poll that failed — and task H1 measured that a command buffer the
//! driver *aborts* is none of those: `wgpu-hal`'s Metal fence resolves an aborted command buffer
//! as success and wgpu-core never looks at the status, so the host is told the submission ran and
//! reads whatever the readback buffer holds. [`MethodStats::corrupt`] is the counter that sees
//! that one: the decoders refuse a readback that fails their validity checks, the whole batch goes
//! to the CPU, and the run's device lines print how often.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use sherd_core::executor::batch::{CoarseBatch, DistBatch, IcpBatch, InsideBatch, InsideOutcome};
use sherd_core::executor::{CPU, Executor};
use sherd_core::matching::icp::Registration;

use crate::coarse::{CoarseAnswer, CoarseKernel, score_of};
use crate::device::Gpu;
use crate::icp::{IcpAnswer, IcpKernel};
use crate::selftest::SelfTest;

/// What one `Executor` method did, over the life of the executor.
#[derive(Debug, Default)]
pub struct MethodStats {
    /// Calls that reached this method.
    pub calls: AtomicU64,
    /// Calls answered by the CPU implementation — because the method has no kernel yet, because
    /// the batch had nothing the device could do with it, or because the device failed.
    pub delegated: AtomicU64,
    /// Calls answered on the device.
    pub on_device: AtomicU64,
    /// Dispatches submitted.
    pub dispatches: AtomicU64,
    /// Wall time of the dispatches alone — `submit` + `poll(Wait)`, no upload and no readback.
    pub gpu_nanos: AtomicU64,
    /// The method's own unit of work: point-queries for the coarse score, candidate-iterations
    /// for the ICP rung.
    pub items: AtomicU64,
    /// Errors **on this side of the bus** — a failed map or poll — that sent a batch to the CPU.
    /// A command buffer the driver aborted is not one of them; see [`MethodStats::corrupt`].
    pub host_errors: AtomicU64,
    /// Readbacks the decoder refused, each of which sent its **whole batch** to the CPU (task H1).
    pub corrupt: AtomicU64,
}

impl MethodStats {
    fn call(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn delegate(&self) {
        self.delegated.fetch_add(1, Ordering::Relaxed);
    }

    /// A delegated call, with the work it carried — how big the batches the CPU is answering are
    /// is the measurement that decides whether a kernel for them would pay (task G3, item 2).
    fn delegate_items(&self, items: usize) {
        self.delegate();
        self.items.fetch_add(items as u64, Ordering::Relaxed);
    }

    fn device(&self, dispatches: usize, gpu: Duration, items: usize) {
        self.on_device.fetch_add(1, Ordering::Relaxed);
        self.dispatches.fetch_add(dispatches as u64, Ordering::Relaxed);
        self.gpu_nanos
            .fetch_add(u64::try_from(gpu.as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
        self.items.fetch_add(items as u64, Ordering::Relaxed);
    }

    fn error(&self) {
        self.host_errors.fetch_add(1, Ordering::Relaxed);
        self.delegate();
    }

    /// A readback the decoder refused: the batch is the CPU's, and it is counted twice over —
    /// once as a delegation, because that is what happened to the work, and once as `corrupt`,
    /// because a delegation for this reason is not the same event as a batch that was never the
    /// device's.
    fn corrupt(&self) {
        self.corrupt.fetch_add(1, Ordering::Relaxed);
        self.delegate();
    }

    /// `(calls, delegated, on_device, dispatches, gpu, items, host_errors, corrupt)`.
    #[must_use]
    pub fn snapshot(&self) -> MethodSnapshot {
        MethodSnapshot {
            calls: self.calls.load(Ordering::Relaxed),
            delegated: self.delegated.load(Ordering::Relaxed),
            on_device: self.on_device.load(Ordering::Relaxed),
            dispatches: self.dispatches.load(Ordering::Relaxed),
            gpu: Duration::from_nanos(self.gpu_nanos.load(Ordering::Relaxed)),
            items: self.items.load(Ordering::Relaxed),
            host_errors: self.host_errors.load(Ordering::Relaxed),
            corrupt: self.corrupt.load(Ordering::Relaxed),
        }
    }
}

/// A readable copy of one [`MethodStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MethodSnapshot {
    /// Calls that reached the method.
    pub calls: u64,
    /// Calls the CPU answered.
    pub delegated: u64,
    /// Calls the device answered.
    pub on_device: u64,
    /// Dispatches submitted.
    pub dispatches: u64,
    /// Wall time of the dispatches alone.
    pub gpu: Duration,
    /// The method's own unit of work.
    pub items: u64,
    /// Errors on the host side of the bus — a failed map or poll.
    pub host_errors: u64,
    /// Readbacks the decoder refused (task H1).
    pub corrupt: u64,
}

/// Per-method counters for the whole run (the note's timing table, `gpu-check`'s rows).
#[derive(Debug, Default)]
pub struct Stats {
    /// `coarse_scores`.
    pub coarse: MethodStats,
    /// `icp_rung`.
    pub icp: MethodStats,
    /// `bounded_distance`.
    pub distance: MethodStats,
    /// `inside`.
    pub inside: MethodStats,
}

impl Stats {
    /// How many calls of each method fell through to the CPU — `(coarse, icp, distance, inside)`.
    #[must_use]
    pub fn delegated_counts(&self) -> (u64, u64, u64, u64) {
        (
            self.coarse.delegated.load(Ordering::Relaxed),
            self.icp.delegated.load(Ordering::Relaxed),
            self.distance.delegated.load(Ordering::Relaxed),
            self.inside.delegated.load(Ordering::Relaxed),
        )
    }

    /// Total wall time spent inside a dispatch, over every method.
    #[must_use]
    pub fn gpu_busy(&self) -> Duration {
        [&self.coarse, &self.icp, &self.distance, &self.inside]
            .into_iter()
            .map(|m| Duration::from_nanos(m.gpu_nanos.load(Ordering::Relaxed)))
            .sum()
    }

    /// One line per method, for a log or a note.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        [
            ("coarse", &self.coarse),
            ("icp", &self.icp),
            ("distance", &self.distance),
            ("inside", &self.inside),
        ]
        .into_iter()
        .map(|(name, stats)| {
            let s = stats.snapshot();
            format!(
                "{name}: {} calls, {} on device, {} delegated ({} host errors, {} corrupt \
                 readbacks), {} dispatches, {:.1} ms gpu, {} items",
                s.calls,
                s.on_device,
                s.delegated,
                s.host_errors,
                s.corrupt,
                s.dispatches,
                s.gpu.as_secs_f64() * 1e3,
                s.items,
            )
        })
        .collect()
    }
}

/// The compiled kernels, built once with the executor.
#[derive(Debug)]
struct Kernels {
    coarse: CoarseKernel,
    icp: IcpKernel,
}

/// The wgpu implementation of [`Executor`].
#[derive(Debug)]
pub struct GpuExecutor {
    gpu: Arc<Gpu>,
    selftest: SelfTest,
    kernels: Kernels,
    stats: Stats,
    force: AtomicBool,
}

impl GpuExecutor {
    /// Whether the matching stage's two kernels are behind the trait.
    ///
    /// True since task G2: `coarse_scores` and `icp_rung` are `kernels/coarse.wgsl` and
    /// `kernels/icp.wgsl`, and `--backend gpu` runs them. `bounded_distance` and `inside` are
    /// phase 2c's and delegate.
    pub const HAS_KERNELS: bool = true;

    /// Whether `Backend::Auto` may **pick** this executor — which is a different question, and on
    /// this machine the answer is no.
    ///
    /// D §6.8's rule reads the self-test's ratio, and the self-test measures E7 §5's bounded-NN
    /// kernel on an idle device: 3–6× the whole ten-core CPU. That number is real and it is not
    /// the matching stage's.
    ///
    /// **Task G2 measured the stage at 1.0× and read the cause as an idle device. Task G3 built
    /// D §6.4's software pipeline and measured the cause, and it is a different one.** With one
    /// submitting thread and the pool deep enough to cover its waits, the matching stage over the
    /// seven development collections, three interleaved runs each, is
    ///
    /// | terracotta | pot_A | pot_B | pot_C | pot_G | pot_H | synthetic_20 |
    /// |---|---|---|---|---|---|---|
    /// | 1.11× | 1.25× | 1.40× | 1.13× | 1.04× | 1.06× | 1.08× |
    ///
    /// — better than G2's 1.0× and still under D §6.8's 1.5× bar. What stops it is not the queue
    /// any more: on `synthetic_20` the device now has work outstanding for 9.1 s of a 13.4 s
    /// stage (G3's figure; task G4's query ceiling then halved it on purpose and task W measured
    /// **5.44 s of 11.34 s, 48 %**, on the tuned tree — V6-D6). It is that **the device and the
    /// ten cores share one envelope**. The same 330
    /// submissions of the same work take
    ///
    /// | `--threads` | 1 | 2 | 4 | 6 | 9 |
    /// |---|---|---|---|---|---|
    /// | device outstanding | 5.66 s | 5.67 s | 6.08 s | 7.62 s | 9.06 s |
    /// | matching stage | 36.41 s | 23.69 s | 13.84 s | **11.86 s** | 13.38 s |
    ///
    /// so the device loses **1.60×** of its own throughput as the CPU fills up, and the stage has
    /// a minimum at six workers rather than at nine. That is an integrated part sharing power and
    /// memory bandwidth with the cores it is supposed to be running alongside, and no scheduler
    /// fixes it.
    ///
    /// **Task G4 tuned the two thresholds that decide how much of that envelope the device is
    /// given** — `coarse::MAX_QUERIES` and the cap on `device_slack` below — and re-measured the
    /// same seven collections the same way:
    ///
    /// | terracotta | pot_A | pot_B | pot_C | pot_G | pot_H | synthetic_20 |
    /// |---|---|---|---|---|---|---|
    /// | 1.07× | 1.26× | **1.43×** | 1.13× | 1.09× | 1.10× | **1.28×** |
    ///
    /// `synthetic_20`, the collection where the device has real work to do, moves 1.08 → 1.28×;
    /// the range is 1.07–1.43× against G3's 1.04–1.40×. That is [`selftest::STAGE_SPEEDUP`], and
    /// it is what `info` and this constant quote.
    ///
    /// So `Auto` keeps the CPU and says why, `--backend gpu` runs the kernels for anyone measuring
    /// them, and this constant flips when a *measured* 1.5× exists — which on this machine would
    /// take a kernel for R §6 that does not have to share, or a discrete GPU that does not share
    /// at all. D §6.8's bar is 1.5×; the honest measurement is 1.07–1.43×.
    pub const AUTO_ELIGIBLE: bool = false;

    /// How many worker threads the matching stage runs beyond `--threads`: **half as many again**
    /// (D §6.4, `Executor::device_slack`).
    ///
    /// A worker that hands a batch to `pipeline::Submitter` waits on a channel, and a waiting
    /// worker is a block that is not being prepared. Measured on `synthetic_20`, matching-stage
    /// seconds, two runs each, with the submitting thread already in place:
    ///
    /// | `--threads` | pool without slack | pool with it | matching, no slack | with slack |
    /// |---|---|---|---|---|
    /// | 1 | 1 | 2 | 71.5 s | **37.1 s (1.93×)** |
    /// | 4 | 4 | 6 | 18.7 s | **13.7 s (1.36×)** |
    /// | 9 (the default here) | 9 | 14 | 13.0 s | 13.1 s (1.00×) |
    ///
    /// So it pays exactly where the pool is smaller than the machine and is free where the pool
    /// already fills it — which is what a knob for *covering a wait* should do. Nine of ten cores
    /// is this machine's default and the row that gains nothing; a four-core laptop is the row
    /// that gains a third.
    ///
    /// A fraction rather than a constant, because what it covers scales with the number of
    /// workers. Half is the measured shape: at `--threads 9` the pool then holds 14, and the
    /// profile that motivated it showed 4.9 of ten cores working with 18.
    ///
    /// **Task G4 measured the other end of the same curve and put a cap on it.** G3 measured the
    /// slack at `--threads` 1, 4 and 9 — the rows above — and read "1.00× at nine" as *free*. It
    /// is not free; it is the top of a hill. Sweeping the slack alone at the default
    /// `--threads 9`, three warm runs each on `synthetic_20`, matching-stage medians:
    ///
    /// | slack | 0 | 1 | 2 | 3 | **5 (G3's)** | 8 |
    /// |---|---|---|---|---|---|---|
    /// | pool | 9 | **10** | 11 | 12 | 14 | 17 |
    /// | matching | 12.84 s | **12.43 s** | 12.03 s | 12.87 s | **13.80 s** | 14.05 s |
    /// | device outstanding | 8.3 s | 8.2 s | 8.4 s | 9.3 s | 9.7 s | 10.0 s |
    ///
    /// The minimum is broad — anything from 10 to 12 workers is inside the run-to-run spread —
    /// and 14 is outside it, by 11 %. The reason is task G3 §5's envelope read from the pool's
    /// side: a worker waiting on the device is not using a core, so a *few* extra workers cost
    /// nothing and cover the wait; but once the pool is deeper than the machine, the extra ones
    /// are not covering a wait at all, they are adding a concurrent batch to a device that gets
    /// slower the busier the cores beside it are.
    ///
    /// So the fraction stays and [`Executor::device_slack`] caps its result at one worker per core
    /// plus one waiting on the device: `min(⌈threads/2⌉, cores + 1 − threads)`, never below one.
    /// That reproduces every row G3 measured — `--threads 1` still gets a pool of 2 and
    /// `--threads 4` a pool of 6 on this ten-core machine — and takes the default from 14 workers
    /// to 11, which is where the table above has its minimum.
    pub const SLACK_NUMERATOR: usize = 1;
    /// The denominator of [`GpuExecutor::SLACK_NUMERATOR`]'s fraction.
    pub const SLACK_DENOMINATOR: usize = 2;

    /// Wraps an open device whose self-test has already run, compiling the kernels.
    ///
    /// The compilation happens **here**, not on the first batch: Metal compiles a shader when the
    /// pipeline is created, and G1 §5.1 measured a 14 ns/query kernel reporting 197 ns/query
    /// because the compile was inside the timed region.
    #[must_use]
    pub fn new(gpu: Arc<Gpu>, selftest: SelfTest) -> Self {
        let kernels = Kernels { coarse: CoarseKernel::build(&gpu), icp: IcpKernel::build(&gpu) };
        Self { gpu, selftest, kernels, stats: Stats::default(), force: AtomicBool::new(false) }
    }

    /// The device.
    #[must_use]
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The self-test this executor was admitted on.
    #[must_use]
    pub fn selftest(&self) -> &SelfTest {
        &self.selftest
    }

    /// The per-method counters.
    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Run the kernels whatever the size thresholds say — **for the cross-check harness only**.
    ///
    /// `coarse::MIN_QUERIES` and `icp::MIN_CANDIDATES` are a *policy*: they send a batch the CPU
    /// answers faster to the CPU. D §10.4 layer 3 is about the *kernel*, and on a small collection
    /// the policy would send every rung to the CPU and the cross-check would print a deviation of
    /// zero it had not measured — the exact failure task G1 built the `delegated` label to avoid.
    /// Measured on terracotta: all 24 of the four pairs' rungs are under the threshold, because
    /// its breakline cloud is under 400 points.
    ///
    /// A run never sets this. `sherd-refit-rs gpu-check` does, and prints which rows reached the
    /// device either way.
    pub fn force_device(&self, force: bool) {
        self.force.store(force, Ordering::Relaxed);
    }

    /// Whether the size thresholds are being ignored.
    #[must_use]
    pub fn forced(&self) -> bool {
        self.force.load(Ordering::Relaxed)
    }

    /// How many calls have fallen through to the CPU, per method.
    #[must_use]
    pub fn delegated(&self) -> (u64, u64, u64, u64) {
        self.stats.delegated_counts()
    }
}

impl Executor for GpuExecutor {
    fn device_slack(&self) -> usize {
        let threads = rayon::current_num_threads();
        let wanted = (threads * Self::SLACK_NUMERATOR).div_ceil(Self::SLACK_DENOMINATOR);
        // The slack covers a wait, and one worker per core plus one waiting on the device is as
        // far as that goes; past it the extra workers are not covering a wait, they are adding a
        // concurrent batch to a device that gets slower the busier the cores are (see
        // `SLACK_NUMERATOR`).
        let cores =
            std::thread::available_parallelism().map_or(threads, std::num::NonZero::get).max(1);
        wanted.min((cores + 1).saturating_sub(threads)).max(1)
    }

    fn name(&self) -> &'static str {
        // R §6's two methods are still the CPU's, and the name reaches the log lines.
        "gpu (matching on device, verification on cpu)"
    }

    fn coarse_scores(&self, batch: &CoarseBatch<'_>) -> Vec<f64> {
        self.stats.coarse.call();
        match self.kernels.coarse.run(&self.gpu, batch, self.forced()) {
            Ok(CoarseAnswer::Device(counts, run)) => {
                self.stats.coarse.device(run.dispatches, run.gpu, run.queries);
                counts.into_iter().map(|k| score_of(k, batch.points.len())).collect()
            }
            Ok(CoarseAnswer::Cpu) => {
                self.stats.coarse.delegate();
                CPU.coarse_scores(batch)
            }
            Ok(CoarseAnswer::Corrupt(why)) => {
                tracing::warn!(%why, "coarse_scores refused a device readback: the batch is the CPU's");
                self.stats.coarse.corrupt();
                CPU.coarse_scores(batch)
            }
            Err(e) => {
                tracing::warn!(error = %e, "coarse_scores fell back to the CPU");
                self.stats.coarse.error();
                CPU.coarse_scores(batch)
            }
        }
    }

    fn icp_rung(&self, batch: &IcpBatch<'_>) -> Vec<Registration> {
        self.stats.icp.call();
        match self.kernels.icp.run(&self.gpu, batch, self.forced()) {
            Ok(IcpAnswer::Device(out, run)) => {
                self.stats.icp.device(run.dispatches, run.gpu, run.iterations);
                out
            }
            Ok(IcpAnswer::Cpu) => {
                self.stats.icp.delegate();
                CPU.icp_rung(batch)
            }
            Ok(IcpAnswer::Corrupt(why)) => {
                tracing::warn!(%why, "icp_rung refused a device readback: the batch is the CPU's");
                self.stats.icp.corrupt();
                CPU.icp_rung(batch)
            }
            Err(e) => {
                tracing::warn!(error = %e, "icp_rung fell back to the CPU");
                self.stats.icp.error();
                CPU.icp_rung(batch)
            }
        }
    }

    fn bounded_distance(&self, batch: &DistBatch<'_>) -> Vec<f64> {
        // R §6.1 is still the CPU's, and task G3 measured why rather than assuming it: the size of
        // the batches that arrive here is the whole argument, so they are counted.
        self.stats.distance.call();
        self.stats.distance.delegate_items(batch.points.len());
        CPU.bounded_distance(batch)
    }

    fn inside(&self, batch: &InsideBatch<'_>) -> Vec<InsideOutcome> {
        self.stats.inside.call();
        self.stats.inside.delegate_items(batch.points.len());
        CPU.inside(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuExecutor, Stats};
    use std::time::Duration;

    /// Phase 2b's flag is what `Backend::Auto` reads, and the two matching kernels are in.
    #[test]
    fn the_kernels_exist_and_auto_still_says_cpu() {
        const { assert!(GpuExecutor::HAS_KERNELS, "task G2 puts coarse and icp on the device") };
        const {
            assert!(
                !GpuExecutor::AUTO_ELIGIBLE,
                "measured: matching is 1.07-1.43x, under D §6.8's 1.5x bar"
            );
        }
    }

    /// The slack is capped at the cores the pool has not already claimed (task G4).
    ///
    /// A unit test of the arithmetic rather than of `device_slack` itself, because that reads
    /// rayon's current pool and this machine's core count; the rule is the same expression.
    #[test]
    fn the_slack_covers_a_wait_and_never_oversubscribes_the_machine() {
        fn slack(threads: usize, cores: usize) -> usize {
            let wanted =
                (threads * GpuExecutor::SLACK_NUMERATOR).div_ceil(GpuExecutor::SLACK_DENOMINATOR);
            wanted.min((cores + 1).saturating_sub(threads)).max(1)
        }
        // Task G3's three measured rows on this ten-core machine are reproduced exactly.
        assert_eq!(1 + slack(1, 10), 2, "--threads 1: G3 measured 1.93x from a pool of 2");
        assert_eq!(4 + slack(4, 10), 6, "--threads 4: G3 measured 1.36x from a pool of 6");
        // ... and the default, which G3 left at 14 and G4 measured as 11 % past the minimum.
        assert_eq!(9 + slack(9, 10), 11, "--threads 9: ten cores busy and one worker waiting");
        // A pool already at or past the machine gets one waiter and no more.
        assert_eq!(10 + slack(10, 10), 11);
        assert_eq!(16 + slack(16, 10), 17, "--threads over the core count is the caller's choice");
        // A machine whose core count cannot be read is not a reason to return zero.
        assert_eq!(slack(1, 1), 1);
    }

    /// The counters are per method; a host error and a refused readback each count as a
    /// delegation as well as themselves — the run still gets the CPU's answer.
    #[test]
    fn the_counters_are_per_method_and_an_error_is_also_a_delegation() {
        let stats = Stats::default();
        assert_eq!(stats.delegated_counts(), (0, 0, 0, 0));
        stats.coarse.call();
        stats.coarse.device(2, Duration::from_millis(5), 1_200_000);
        stats.icp.call();
        stats.icp.error();
        stats.inside.call();
        stats.inside.delegate();
        assert_eq!(stats.delegated_counts(), (0, 1, 0, 1));

        let coarse = stats.coarse.snapshot();
        assert_eq!((coarse.calls, coarse.on_device, coarse.dispatches), (1, 1, 2));
        assert_eq!(coarse.items, 1_200_000);
        assert_eq!(stats.gpu_busy(), Duration::from_millis(5));
        assert_eq!(stats.icp.snapshot().host_errors, 1);
        assert_eq!(stats.icp.snapshot().corrupt, 0);
        stats.coarse.corrupt();
        assert_eq!(stats.coarse.snapshot().corrupt, 1);
        assert_eq!(stats.delegated_counts(), (1, 1, 0, 1), "a corrupt readback is a delegation");
        assert_eq!(stats.lines().len(), 4);
        let expected = "coarse: 1 calls, 1 on device, 1 delegated (0 host errors, 1 corrupt \
                        readbacks)";
        assert!(stats.lines()[0].starts_with(expected), "{:?}", stats.lines());
    }
}
