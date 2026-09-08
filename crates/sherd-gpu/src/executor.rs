//! [`GpuExecutor`] — the device side of D §6.1's four inner loops.
//!
//! # Which of the four run on the device
//!
//! | method | kernel | phase | state |
//! |---|---|---|---|
//! | `coarse_scores` | `kernels/coarse.wgsl`, one workgroup per pose | 2b | **on the device** |
//! | `icp_rung` | `kernels/icp.wgsl`, one workgroup per candidate, every iteration inside | 2b | delegated to the CPU (next commit) |
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
//! implementation's, and [`Stats::errors`] says how often the device could not give one.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sherd_core::executor::batch::{CoarseBatch, DistBatch, IcpBatch, InsideBatch, InsideOutcome};
use sherd_core::executor::{CPU, Executor};
use sherd_core::matching::icp::Registration;

use crate::coarse::{CoarseKernel, score_of};
use crate::device::Gpu;
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
    /// Device errors that sent a batch to the CPU.
    pub errors: AtomicU64,
}

impl MethodStats {
    fn call(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn delegate(&self) {
        self.delegated.fetch_add(1, Ordering::Relaxed);
    }

    fn device(&self, dispatches: usize, gpu: Duration, items: usize) {
        self.on_device.fetch_add(1, Ordering::Relaxed);
        self.dispatches.fetch_add(dispatches as u64, Ordering::Relaxed);
        self.gpu_nanos
            .fetch_add(u64::try_from(gpu.as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
        self.items.fetch_add(items as u64, Ordering::Relaxed);
    }

    fn error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        self.delegate();
    }

    /// `(calls, delegated, on_device, dispatches, gpu, items, errors)`.
    #[must_use]
    pub fn snapshot(&self) -> MethodSnapshot {
        MethodSnapshot {
            calls: self.calls.load(Ordering::Relaxed),
            delegated: self.delegated.load(Ordering::Relaxed),
            on_device: self.on_device.load(Ordering::Relaxed),
            dispatches: self.dispatches.load(Ordering::Relaxed),
            gpu: Duration::from_nanos(self.gpu_nanos.load(Ordering::Relaxed)),
            items: self.items.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
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
    /// Device errors.
    pub errors: u64,
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
                "{name}: {} calls, {} on device, {} delegated ({} device errors), {} dispatches, \
                 {:.1} ms gpu, {} items",
                s.calls,
                s.on_device,
                s.delegated,
                s.errors,
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
}

/// The wgpu implementation of [`Executor`].
#[derive(Debug)]
pub struct GpuExecutor {
    gpu: Arc<Gpu>,
    selftest: SelfTest,
    kernels: Kernels,
    stats: Stats,
}

impl GpuExecutor {
    /// Whether the matching stage's two kernels are behind the trait.
    ///
    /// This is what `Backend::Auto` reads (`Selection::decide`). It is a single flag rather than
    /// one per method because the decision it feeds is a whole-run one: the coarse score is 27 %
    /// of `synthetic_20`'s matching core-seconds (E2's profile) and the ICP ladders are another
    /// 27 %, so a build with only the coarse kernel cannot clear D §6.8's 1.5× bar by Amdahl's law
    /// however fast that kernel is — `1/(1 − 0.27)` is 1.37×. It flips when `icp_rung` lands.
    pub const HAS_KERNELS: bool = false;

    /// Wraps an open device whose self-test has already run, compiling the kernels.
    ///
    /// The compilation happens **here**, not on the first batch: Metal compiles a shader when the
    /// pipeline is created, and G1 §5.1 measured a 14 ns/query kernel reporting 197 ns/query
    /// because the compile was inside the timed region.
    #[must_use]
    pub fn new(gpu: Arc<Gpu>, selftest: SelfTest) -> Self {
        let kernels = Kernels { coarse: CoarseKernel::build(&gpu) };
        Self { gpu, selftest, kernels, stats: Stats::default() }
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

    /// How many calls have fallen through to the CPU, per method.
    #[must_use]
    pub fn delegated(&self) -> (u64, u64, u64, u64) {
        self.stats.delegated_counts()
    }
}

impl Executor for GpuExecutor {
    fn name(&self) -> &'static str {
        // Not "gpu" while a method of the four is still the CPU's, and the name reaches the log.
        if Self::HAS_KERNELS { "gpu" } else { "gpu (coarse on device, the rest on cpu)" }
    }

    fn coarse_scores(&self, batch: &CoarseBatch<'_>) -> Vec<f64> {
        self.stats.coarse.call();
        match self.kernels.coarse.run(&self.gpu, batch) {
            Ok(Some((counts, run))) => {
                self.stats.coarse.device(run.dispatches, run.gpu, run.queries);
                counts.into_iter().map(|k| score_of(k, batch.points.len())).collect()
            }
            Ok(None) => {
                self.stats.coarse.delegate();
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
        self.stats.icp.delegate();
        CPU.icp_rung(batch)
    }

    fn bounded_distance(&self, batch: &DistBatch<'_>) -> Vec<f64> {
        // Phase 2c: the BVH kernels. Counted, not pretended.
        self.stats.distance.call();
        self.stats.distance.delegate();
        CPU.bounded_distance(batch)
    }

    fn inside(&self, batch: &InsideBatch<'_>) -> Vec<InsideOutcome> {
        self.stats.inside.call();
        self.stats.inside.delegate();
        CPU.inside(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuExecutor, Stats};
    use std::time::Duration;

    /// Phase 2b's flag is what `Backend::Auto` reads, and the two matching kernels are in.
    #[test]
    fn the_auto_flag_waits_for_both_matching_kernels() {
        const {
            assert!(
                !GpuExecutor::HAS_KERNELS,
                "the coarse kernel alone is 27 % of matching: 1.37x by Amdahl, under D §6.8's 1.5x"
            );
        }
    }

    /// The counters are per method, and a device error counts as a delegation as well as an
    /// error — the run still gets the CPU's answer.
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
        assert_eq!(stats.icp.snapshot().errors, 1);
        assert_eq!(stats.lines().len(), 4);
        assert!(
            stats.lines()[0].starts_with("coarse: 1 calls, 1 on device"),
            "{:?}",
            stats.lines()
        );
    }
}
