//! The Rust side of the parity harness (D §10.1): `--dump-fixtures DIR` writes the same layout of
//! `.npy` and JSON files at the same stage boundaries as the Python sink
//! (`sherd_refit/fixture.py`), so that `tools/compare_fixtures.py REF CAND` can judge a Rust run
//! against a Python one. Reading fixtures — the injected mode — lives in the `sherd-parity`
//! crate and is done (`sherd-refit-rs parity`).
//!
//! **This module is still empty, and `--dump-fixtures` is not a flag yet.** It used to say
//! "filled in by plan step S4", which was wrong: S4's row of the plan names the fixture *reader*
//! and the parity CLI, not the writer, and both of those are in `sherd-parity`. The writer is
//! wanted when a Rust run has stages of its own worth dumping — a Python-side comparison of the
//! port's segmentation and match arrays — so it lands with phase 1d's `run`, alongside the
//! `--dump-fixtures` flag that drives it (phase-1a verification, finding F10).
