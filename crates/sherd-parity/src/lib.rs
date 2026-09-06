//! `sherd-parity` — the Rust side of the parity harness (D §10).
//!
//! `tools/dump_fixtures.py` writes, for one collection, every stage boundary of the Python
//! pipeline as `.npy` arrays and JSON scalars, plus a `manifest.json` that hashes each file
//! (D §10.1, `docs/superpowers/notes/2026-09-06-p0-fixtures.md`). This crate reads such a dump:
//!
//! * **injected mode** — a Rust stage is fed the Python stage's inputs, so the comparison is
//!   exact where the reference is deterministic;
//! * **native mode** — the Rust stage runs on Rust's own upstream results, and the comparison is
//!   statistical, within the tolerances of D §10.2.
//!
//! Phase 1a step S1 provides the dump's layout and its manifest, which is enough to check that a
//! fixture is intact and to read the run's parameters. The array reader (`npyz`) and the stage
//! runners behind `sherd-refit-rs parity --stage …` follow in plan step S4.

pub mod layout;
pub mod manifest;

pub use layout::FixtureDir;
pub use manifest::{Collection, FileEntry, Manifest, Pairs};
