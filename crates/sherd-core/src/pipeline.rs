//! The run: discovery, preprocessing, matching, assembly, refinement, outputs (D §5).
//!
//! One process, one rayon pool. Preprocessing is a `par_iter` over fragments bounded by a
//! memory-aware semaphore (a scan of `f` faces reserves `60 MB + 110 B·f` from `--memory-budget`,
//! default half of physical RAM). Matching walks the same 3×3 blocks of the collection order the
//! reference walks, so the pair order — and with it every seeded draw — is the reference's;
//! candidates inside a pair run in parallel and are collected by index, so nothing depends on the
//! schedule. Cancellation is an `AtomicBool` checked between units of work; progress is a
//! callback. Matching, assembly, refinement and the outputs are filled in in phase 1d.
//!
//! Step S4 fills in the first stage: [`preprocess`], which is what `sherd-refit-rs segment`
//! drives — R §3.1–3.3 then, R §3.1–3.4 since step B1. The memory-aware semaphore is not part of it yet — that needs the per-fragment
//! high-water mark measured rather than guessed, which belongs with the memory work of phase 1e —
//! so the fan-out is a plain `par_iter` over the collection, with `--threads` sizing the pool.

use std::path::Path;

use rayon::prelude::*;

use crate::collection::Entry;
use crate::error::Result;
use crate::fragment::{Fragment, cache};

/// What preprocessing one fragment produced.
#[derive(Debug)]
pub struct Preprocessed {
    /// The fragment, at the state R §3.4 leaves it in (the match arrays follow).
    pub fragment: Fragment,
    /// Whether it came from the cache rather than from the file (R §3.7).
    pub cached: bool,
    /// Wall-clock seconds the fragment took, cache hit or not.
    pub seconds: f64,
}

/// R §3.1–3.4 for a whole collection, in parallel, through the fragment cache (R §3.7, D §5
/// stage 2).
///
/// `out_dir` is the run's output directory: caches are written to `<out_dir>/cache/<name>.sherd`
/// and read from there when they describe the same file at the same `target_faces`. `None`
/// bypasses the cache entirely, which is what the parity harness wants and what `--no-cache`
/// gives.
///
/// Results come back in **collection order**, whatever order the pool finished them in, and every
/// fragment's own computation is single-threaded apart from the ray loops of R §3.2 and R §3.4.3
/// and the radius queries of R §3.4.2, all of which are indexed collects, so the result does not
/// depend on the thread count. A fragment that fails to load takes its error into the
/// result vector instead of aborting the collection: one unreadable file among a hundred scans is
/// a warning, not the end of the run.
pub fn preprocess(
    entries: &[Entry],
    target_faces: usize,
    out_dir: Option<&Path>,
) -> Vec<Result<Preprocessed>> {
    entries
        .par_iter()
        .map(|entry| {
            let started = std::time::Instant::now();
            let cache_path = out_dir.map(|dir| cache::cache_path(dir, &entry.name));
            let (fragment, cached) = Fragment::load_or_build(
                &entry.path,
                target_faces,
                &entry.name,
                cache_path.as_deref(),
            )?;
            Ok(Preprocessed { fragment, cached, seconds: started.elapsed().as_secs_f64() })
        })
        .collect()
}

/// Sizes the process-wide rayon pool (D §5's `--threads`); `0` leaves it at one per core.
///
/// Returns an error string when the pool has already been built, which can only happen if this is
/// called twice or after something else has already used rayon.
pub fn set_threads(threads: usize) -> std::result::Result<(), String> {
    if threads == 0 {
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{preprocess, set_threads};
    use crate::collection::Entry;

    #[test]
    fn an_unreadable_file_fails_only_itself() {
        let dir = std::env::temp_dir().join(format!("sherd-preprocess-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let broken = dir.join("broken.ply");
        std::fs::write(&broken, b"ply\nformat ascii 1.0\nend_header\n").unwrap();
        let entries = vec![Entry { path: broken, name: "broken".to_owned() }];
        let results = preprocess(&entries, 200_000, None);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "a mesh with no triangles is R §3.1's error case");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zero_threads_leaves_the_pool_alone() {
        assert!(set_threads(0).is_ok());
    }
}
