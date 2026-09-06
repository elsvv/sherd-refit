//! `sherd-core` — the sherd-refit algorithm in Rust: everything except the GPU executor
//! (`sherd-gpu`, phase 2) and the Python bindings (`sherd-py`, phase 3).
//!
//! # Where the truth lives
//!
//! The algorithm is frozen and documented outside this crate; nothing here is invented:
//!
//! * `docs/superpowers/specs/2026-09-06-algorithm-reference.md` (**R**) — the algorithm stage by
//!   stage, as the Python at commit `9d4b9d3` computes it. Where R and the Python differ, the
//!   Python wins; items marked *PMC* in R are the places where this port is allowed to differ,
//!   each one to be re-verified against the parity gates of R §13.
//! * `docs/superpowers/specs/2026-09-06-rust-core-design.md` (**D**) — this workspace's layout
//!   (§2), the pinned crates and the experiments behind them (§3), the data model (§4), the
//!   executor split (§6), determinism policy (§7) and the parity tolerances (§10.2).
//!
//! Every module below names the R sections it implements and the phase-1 step that fills it in
//! (`docs/superpowers/plans/2026-09-06-rust-core-phase0-1a.md`).
//!
//! # Module map (D §2)
//!
//! | module | R sections | filled in |
//! |---|---|---|
//! | [`io`] | §3.1, §11 | S2 |
//! | [`mesh`] | §3.1 (clean, components) S2; §3.3 (the rest) S3 | S2, S3 |
//! | [`fragment`] | §3.2, §3.4–3.7 | S3, S4, phase 1b |
//! | [`spatial`] | §3.2, §3.4.3, §6.1, §6.4 | phase 1b |
//! | [`matching`] | §4–§7 | phase 1c |
//! | [`assembly`] | §8 | phase 1d |
//! | [`refine`], [`report`], [`render`], [`pipeline`] | §9, §11, §2 | phase 1d |
//!
//! # State
//!
//! Phase 1a step S1 was the scaffold: the cross-cutting types ([`Vec3f`], [`Params`],
//! [`WorkingMesh`], [`Pose`], [`Error`]), the seeded RNG and the module tree. Step S2 added the
//! load stage — every reader of [`io`], the PLY writer of R §11.4, and R §3.1's cleaning and
//! largest-component passes in [`mesh`]. The remaining algorithm modules are documented but
//! empty; they are filled in step by step, each step gated on the fixtures under `fixtures/` and
//! on `tools/compare_fixtures.py`.

pub mod assembly;
pub mod error;
pub mod executor;
pub mod fixture;
pub mod fragment;
pub mod io;
pub mod matching;
pub mod mesh;
pub mod params;
pub mod pipeline;
pub mod refine;
pub mod render;
pub mod report;
pub mod rng;
pub mod spatial;
pub mod types;
pub mod vec3;

pub use error::{Error, Result};
pub use executor::Backend;
pub use mesh::Mesh;
pub use params::Params;
pub use types::{Cloud, FaceLabel, FragId, Pose, SourceRef, WorkingMesh};
pub use vec3::Vec3f;

/// Version of this crate, reported as `core_version` in the cache metadata and in the `engine`
/// key of `report.json` (D §4.3).
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The frozen algorithm this port reproduces: the date of the algorithm reference and the commit
/// of the Python it was written from (D §4.3). Any algorithmic change bumps this string and
/// invalidates every cache file that carries an older one.
pub const ALGO_REF: &str = "2026-09-06/9d4b9d3";

/// Layout version of `<out>/cache/<name>.sherd` (D §4.2). Bumped when the tensor set or the
/// metadata keys change, independently of [`ALGO_REF`].
pub const CACHE_VERSION: u32 = 1;
