//! The executor split (D §6).
//!
//! Five inner loops — coarse scores, one ICP rung, bounded point-to-surface distance, the inside
//! test and the cone cast — are the only places the pipeline spends its time, and the only ones
//! that get a second implementation. They will sit behind an `Executor` trait: `CpuExecutor`
//! here (rayon), `GpuExecutor` in `sherd-gpu` (WGSL, phase 2), with identical batch structs.
//! Everything else — hypotheses, NMS, seam and continuity, the scoring arithmetic, assembly —
//! is written once and runs on the CPU.
//!
//! The trait and the batch types arrive with the CPU implementation in phase 1c; what phase 1a
//! needs already is the way a run names its backend, because the CLI takes `--backend` and the
//! report records what actually ran (D §4.3).

use std::fmt;
use std::str::FromStr;

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
