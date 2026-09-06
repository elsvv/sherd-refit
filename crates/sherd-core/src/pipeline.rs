//! The run: discovery, preprocessing, matching, assembly, refinement, outputs (D §5).
//!
//! One process, one rayon pool. Preprocessing is a `par_iter` over fragments bounded by a
//! memory-aware semaphore (a scan of `f` faces reserves `60 MB + 110 B·f` from `--memory-budget`,
//! default half of physical RAM). Matching walks the same 3×3 blocks of the collection order the
//! reference walks, so the pair order — and with it every seeded draw — is the reference's;
//! candidates inside a pair run in parallel and are collected by index, so nothing depends on the
//! schedule. Cancellation is an `AtomicBool` checked between units of work; progress is a
//! callback. Filled in in phase 1d.
