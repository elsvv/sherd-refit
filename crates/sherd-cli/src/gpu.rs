//! `gpu-check`'s adapter between `clap` and `sherd-gpu` (D §10.4 layer 3). `--backend`'s
//! resolution lives in `sherd-backend`, where the desktop app links it too (A §3.6).

use anyhow::Result;
// Only the CPU-only `check` bails; with the feature on, the failure comes back from `crosscheck`.
#[cfg(not(feature = "gpu"))]
use anyhow::bail;
pub(crate) use sherd_backend::{info_lines, resolve, selftest_lines};

/// `sherd-refit-rs gpu-check`: the cross-check table as lines, and how many rows failed.
///
/// The harness itself moved into `sherd_gpu::crosscheck` in task H2 (audit §B.7); what is left
/// here is the adapter between `clap` and it. It returns *lines* rather than rows so that a build
/// without the crate has no row type to mirror.
#[cfg(feature = "gpu")]
pub(crate) fn check(
    stage: &str,
    set: Option<&std::path::Path>,
    fixture: Option<&std::path::Path>,
    adapter: Option<&str>,
    pairs: usize,
    chaos: bool,
    force_device: bool,
) -> Result<(Vec<String>, usize)> {
    let rows =
        sherd_gpu::crosscheck::check(stage, set, fixture, adapter, pairs, chaos, force_device)?;
    let failed = rows.iter().filter(|row| row.failed()).count();
    Ok((sherd_gpu::crosscheck::table(&rows), failed))
}

/// [`check`] for a build without the `gpu` feature.
#[cfg(not(feature = "gpu"))]
pub(crate) fn check(
    _stage: &str,
    _set: Option<&std::path::Path>,
    _fixture: Option<&std::path::Path>,
    _adapter: Option<&str>,
    _pairs: usize,
    _chaos: bool,
    _force_device: bool,
) -> Result<(Vec<String>, usize)> {
    bail!("gpu-check: this binary was built without the `gpu` feature (D §2)")
}
