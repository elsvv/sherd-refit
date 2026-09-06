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
//! Phase 1a step S1 provided the dump's layout and its manifest, which is enough to check that a
//! fixture is intact and to read the run's parameters. Step S4 added the array reader ([`npy`],
//! over `npyz`), the comparison vocabulary ([`report`]) and the stage runners behind
//! `sherd-refit-rs parity --stage …` ([`stages`]) for the three stages the port computes so far:
//! `load`, `thickness` and `working mesh`. The later rows of D §10.2's table join them with the
//! stages they judge, in phases 1b–1d.

pub mod layout;
pub mod manifest;
pub mod npy;
pub mod report;
pub mod stages;

pub use layout::FixtureDir;
pub use manifest::{Collection, FileEntry, Manifest, Pairs};
pub use report::{Check, Mode, StageReport};
pub use stages::Stage;
