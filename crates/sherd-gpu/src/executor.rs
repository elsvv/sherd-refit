//! [`GpuExecutor`] — the device side of D §6.1's four inner loops.
//!
//! # Phase 2a: the boundary, not the kernels
//!
//! Every method here **routes to [`CpuExecutor`](sherd_core::executor::CpuExecutor)** and records
//! that it did. That is the honest state of task G1: D §12 puts the hash grid and `icp_rung` in
//! phase 2b and the BVH kernels in 2c, and a method that quietly returned the CPU's answer while
//! calling itself a GPU result would make every later cross-check vacuous. So:
//!
//! * [`GpuExecutor::HAS_KERNELS`] is `false`, and
//!   [`Selection::decide`](crate::selftest::Selection::decide) reads it: `Backend::Auto` will not
//!   pick this executor however fast the self-test says the device is, and it says why.
//! * [`GpuExecutor::delegated`] counts the calls that fell through, per method, so
//!   `sherd-refit-rs gpu-check` can label a row `delegated` instead of printing a deviation of
//!   zero.
//! * `--backend gpu` still opens the device, runs the self-test and fails loudly when either step
//!   fails (D §6.8) — that path is real, and it is what phase 2b will hang its kernels on.
//!
//! # What each method will become
//!
//! | method | kernel | phase | batch already carries |
//! |---|---|---|---|
//! | `coarse_scores` | D §6.5's `coarse_main`, one workgroup per pose | 2b | [`CoarseBatch::device_grid`], `device_points`, `Poses::device` |
//! | `icp_rung` | D §6.5's rung, one workgroup per candidate, all iterations in the kernel | 2b | [`IcpBatch::device_grid`], `device_source`, `device_inits` |
//! | `bounded_distance` | BVH closest point with `r_max` | 2c | `device_points`, and the fragment slot's BVH |
//! | `inside` | AABB reject → parity rays → depth | 2c | `device_points`, and the fragment slot's BVH |
//!
//! The batches already produce every device array those kernels bind, and the slot table already
//! decides what is resident, which is the point of building them before the kernels exist.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use sherd_core::executor::batch::{CoarseBatch, DistBatch, IcpBatch, InsideBatch, InsideOutcome};
use sherd_core::executor::{CPU, Executor};
use sherd_core::matching::icp::Registration;

use crate::device::Gpu;
use crate::selftest::SelfTest;

/// How many calls of each method fell through to the CPU.
#[derive(Debug, Default)]
pub struct Delegated {
    /// `coarse_scores`.
    pub coarse: AtomicU64,
    /// `icp_rung`.
    pub icp: AtomicU64,
    /// `bounded_distance`.
    pub distance: AtomicU64,
    /// `inside`.
    pub inside: AtomicU64,
}

impl Delegated {
    /// `(coarse, icp, distance, inside)`.
    #[must_use]
    pub fn counts(&self) -> (u64, u64, u64, u64) {
        (
            self.coarse.load(Ordering::Relaxed),
            self.icp.load(Ordering::Relaxed),
            self.distance.load(Ordering::Relaxed),
            self.inside.load(Ordering::Relaxed),
        )
    }
}

/// The wgpu implementation of [`Executor`].
#[derive(Debug)]
pub struct GpuExecutor {
    gpu: Arc<Gpu>,
    selftest: SelfTest,
    delegated: Delegated,
}

impl GpuExecutor {
    /// Whether the four `Executor` methods have kernels behind them yet.
    ///
    /// `false` in phase 2a. Every consumer of this crate that has to decide something — the
    /// `Backend::Auto` rule, the `gpu-check` table, the log line a run prints — reads this rather
    /// than assuming, so that flipping it in phase 2b turns the whole path on in one place.
    pub const HAS_KERNELS: bool = false;

    /// Wraps an open device whose self-test has already run.
    #[must_use]
    pub fn new(gpu: Arc<Gpu>, selftest: SelfTest) -> Self {
        Self { gpu, selftest, delegated: Delegated::default() }
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

    /// How many calls have fallen through to the CPU, per method.
    #[must_use]
    pub fn delegated(&self) -> &Delegated {
        &self.delegated
    }
}

impl Executor for GpuExecutor {
    fn name(&self) -> &'static str {
        // Not "gpu": in phase 2a every method is the CPU's, and the name reaches the log lines.
        if Self::HAS_KERNELS { "gpu" } else { "gpu (delegating to cpu)" }
    }

    fn coarse_scores(&self, batch: &CoarseBatch<'_>) -> Vec<f64> {
        self.delegated.coarse.fetch_add(1, Ordering::Relaxed);
        CPU.coarse_scores(batch)
    }

    fn icp_rung(&self, batch: &IcpBatch<'_>) -> Vec<Registration> {
        self.delegated.icp.fetch_add(1, Ordering::Relaxed);
        CPU.icp_rung(batch)
    }

    fn bounded_distance(&self, batch: &DistBatch<'_>) -> Vec<f64> {
        self.delegated.distance.fetch_add(1, Ordering::Relaxed);
        CPU.bounded_distance(batch)
    }

    fn inside(&self, batch: &InsideBatch<'_>) -> Vec<InsideOutcome> {
        self.delegated.inside.fetch_add(1, Ordering::Relaxed);
        CPU.inside(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::{Delegated, GpuExecutor};
    use std::sync::atomic::Ordering;

    /// Phase 2a's flag is what every other decision reads; when it flips, the name does too.
    #[test]
    fn phase_two_a_has_no_kernels_and_says_so() {
        const { assert!(!GpuExecutor::HAS_KERNELS, "task G1 builds the boundary, not the kernels") }
    }

    /// The delegation counters are per method and start at zero.
    #[test]
    fn the_delegation_counters_are_per_method() {
        let counts = Delegated::default();
        assert_eq!(counts.counts(), (0, 0, 0, 0));
        counts.coarse.fetch_add(3, Ordering::Relaxed);
        counts.inside.fetch_add(1, Ordering::Relaxed);
        assert_eq!(counts.counts(), (3, 0, 0, 1));
    }
}
