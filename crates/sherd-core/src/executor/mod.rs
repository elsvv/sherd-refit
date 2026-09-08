//! The executor split (D §6).
//!
//! Four inner loops — the coarse breakline score, one rung of the ICP, the bounded
//! point-to-surface distance and the inside test — are where the matcher spends its time, and they
//! are the only places that get a second implementation. They sit behind [`Executor`]:
//! [`CpuExecutor`] here (rayon, and the reference implementation of every method), `GpuExecutor` in
//! `sherd-gpu` (WGSL, phase 2b), with identical [`batch`] structs. Everything else — hypotheses,
//! NMS, seam and continuity, the scoring arithmetic, assembly — is written once and runs on the
//! CPU. D §6.1's fifth method, `cone_cast`, is R §3.4.3's and belongs to preprocessing; it joins
//! the trait with the kernel that needs it (phase 2b, optional).
//!
//! # What a batch is, and what it is not
//!
//! A batch is the whole of one kernel's input — the fixed cloud with its tree or BVH, the moving
//! points, the poses, the thresholds — so a GPU submission can be formed from it without reaching
//! back into the pipeline. It is **not** a narrowing: the slices are `f64` and the structures are
//! the CPU's own, because the CPU executor must keep computing what phase 1 verified, bit for bit,
//! and experiment E5 measured `f32` point loops far outside D §10.2. The device layouts of D §6.3
//! — SoA `vec4<f32>`, `bytemuck::Pod`, D §6.2's hash grid — are produced *from* a batch, by the
//! executor that needs them, on the way to the device.
//!
//! # Backends and the engine
//!
//! [`Backend`] is what a run asks for on the command line and what `report.json` records.
//! [`Engine`] is what the pipeline passes down: the executor that was actually chosen, together
//! with D §7's two numerical knobs, which travel to the same places.

use std::fmt;
use std::str::FromStr;

pub mod batch;
pub mod cpu;

pub use batch::{
    CoarseBatch, DistBatch, DistReduce, IcpBatch, InsideBatch, InsideOutcome, PoseGpu, Poses,
};
pub use cpu::CpuExecutor;

use crate::matching::icp::{Numerics, Registration};

/// Which executor a run asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Use the GPU when an adapter exists, the self-test passes and it is at least 1.5× faster
    /// than the CPU on this machine; otherwise the CPU (D §6.8).
    #[default]
    Auto,
    /// Always the CPU.
    Cpu,
    /// The GPU, and fail if there is none — the way a benchmark asks for it.
    Gpu,
}

impl Backend {
    /// The spelling used on the command line and in `report.json`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Returned when `--backend` is given something else.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownBackend(pub String);

impl fmt::Display for UnknownBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown backend `{}`, expected auto, cpu or gpu", self.0)
    }
}

impl std::error::Error for UnknownBackend {}

impl FromStr for Backend {
    type Err = UnknownBackend;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(UnknownBackend(other.to_owned())),
        }
    }
}

/// The five inner loops of D §6.1, as both executors implement them.
///
/// `CpuExecutor` is the reference implementation of every method; a second implementation is
/// correct when it agrees with this one inside D §10.2's tolerances **and** takes the same
/// decisions on the development sets (D §6.7, corrected by E7 §8: for anything with a multiply-add
/// in it, "the same bits" is not available on Metal and was never the contract).
///
/// Implementations must be `Send + Sync`: a batch can be submitted from any thread of the pool,
/// and the pipeline holds one executor for the whole run.
pub trait Executor: Send + Sync + fmt::Debug {
    /// The name the report and the log lines use.
    fn name(&self) -> &'static str;

    /// R §5.2's coarse score and R §5.4's re-score: one score per pose (D §6.5's `coarse_main`).
    fn coarse_scores(&self, batch: &CoarseBatch<'_>) -> Vec<f64>;

    /// One rung of R §7's ICP — up to `max_iteration` iterations — for every candidate of the
    /// batch, in batch order.
    fn icp_rung(&self, batch: &IcpBatch<'_>) -> Vec<Registration>;

    /// R §6.1's point-to-triangle-set distance, exact below the window and `+∞` above it; one
    /// value per point, or the batch's minimum, as [`DistReduce`] asks.
    fn bounded_distance(&self, batch: &DistBatch<'_>) -> Vec<f64>;

    /// R §6.4's inside test, with the depth of every point that is inside.
    fn inside(&self, batch: &InsideBatch<'_>) -> Vec<InsideOutcome>;
}

/// The process-wide CPU executor.
///
/// It is stateless, so one value serves every caller and [`Engine::REFERENCE`] can borrow it for
/// `'static`.
pub static CPU: CpuExecutor = CpuExecutor;

/// What the pipeline hands to every stage that runs a kernel: the executor and D §7's numerics.
///
/// The two travel together because they reach the same functions and because they answer the same
/// question — *which arithmetic ran this rung* — from two sides. `Engine` is `Copy`, so passing it
/// down costs a pointer and a two-byte enum pair.
#[derive(Clone, Copy, Debug)]
pub struct Engine<'e> {
    /// The executor every batch goes to.
    pub exec: &'e dyn Executor,
    /// D §7's two knobs: the scalar of the ICP point loops and the frame of the normal equations.
    pub numerics: Numerics,
}

impl Engine<'static> {
    /// The CPU executor under the reference numerics — what every parity row of D §10.2 is
    /// measured with, and what a caller with no reason to choose otherwise wants.
    pub const REFERENCE: Self = Self { exec: &CPU, numerics: Numerics::REFERENCE };

    /// The CPU executor under other numerics (experiment E5's instrument).
    #[must_use]
    pub const fn cpu(numerics: Numerics) -> Self {
        Self { exec: &CPU, numerics }
    }
}

impl<'e> Engine<'e> {
    /// An engine over a chosen executor.
    #[must_use]
    pub const fn new(exec: &'e dyn Executor, numerics: Numerics) -> Self {
        Self { exec, numerics }
    }

    /// The same engine under other numerics.
    #[must_use]
    pub const fn with(self, numerics: Numerics) -> Self {
        Self { exec: self.exec, numerics }
    }
}

impl Default for Engine<'static> {
    fn default() -> Self {
        Self::REFERENCE
    }
}

#[cfg(test)]
mod tests {
    use super::Backend;

    #[test]
    fn spellings_round_trip() {
        for b in [Backend::Auto, Backend::Cpu, Backend::Gpu] {
            assert_eq!(b.to_string().parse::<Backend>(), Ok(b));
        }
        assert_eq!(Backend::default(), Backend::Auto);
        assert!("metal".parse::<Backend>().is_err());
        assert_eq!(
            "metal".parse::<Backend>().unwrap_err().to_string(),
            "unknown backend `metal`, expected auto, cpu or gpu"
        );
    }
}
