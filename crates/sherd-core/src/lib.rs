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
//! | [`collection`] | §2 | S4 |
//! | [`io`] | §3.1, §11 | S2 |
//! | [`mesh`] | §3.1 (clean, components) S2; §3.3 (the rest) S3; §3.4.6 islands B1 | S2, S3, B1 |
//! | [`fragment`] | §3.2, §3.4–3.7 | S3 (mesh), S4 (cache), B1 (segmentation), B2 (breaklines), B3 (samples) |
//! | [`spatial`] | §3.2, §3.4.1–3.4.3, §6.1, §6.4 | B1 (rays, KD-tree), phase 1c (inside test) |
//! | [`matching`] | §1.2, §4–§7 | C1 (scales, hypotheses, coarse, NMS), C2 (ICP and the two ladders), phase 1c (the rest) |
//! | [`assembly`] | §8 | phase 1d |
//! | [`refine`], [`report`], [`render`], [`pipeline`] | §9, §11, §2 | phase 1d |
//!
//! # State
//!
//! Phase 1a step S1 was the scaffold: the cross-cutting types ([`Vec3f`], [`Params`],
//! [`WorkingMesh`], [`Pose`], [`Error`]), the seeded RNG and the module tree. Step S2 added the
//! load stage — every reader of [`io`], the PLY writer of R §11.4, and R §3.1's cleaning and
//! largest-component passes in [`mesh`]. Step S3 added the rest of the preprocessing up to the
//! working mesh: face geometry and `res`, edge adjacency and `closed_enough`, the adaptive face
//! budget with `meshopt`'s decimation, Taubin smoothing, the wall thickness of R §3.2, and
//! [`fragment::Fragment::from_mesh_file`], which runs all of it in the reference's order. Step S4
//! added the fragment cache ([`fragment::cache`], D §4.2), collection discovery ([`collection`],
//! R §2) and, in `sherd-parity`, the reader for the Python fixtures and the stage runners behind
//! `sherd-refit-rs parity`. Step B1 opened phase 1b with R §3.4's shell/fracture segmentation
//! ([`fragment::segment`]) and the two spatial structures it runs on
//! ([`spatial::bvh`], [`spatial::kdtree`]); its labels are the `labels` tensor of the cache and
//! the `segmentation` row of the parity table. Step B2 added R §3.5.3–3.5.5's breaklines and
//! frames ([`fragment::breakline`]) and step B3 the sampled match arrays of R §3.5.1–3.5.2 and
//! §3.5.6 with the runtime [`MatchData`](fragment::samples::MatchData) of R §3.6
//! ([`fragment::samples`]), which completes R §3: the ten tensors of a fragment cache and the
//! `breakline` and `samples` rows of the parity table. Step C1 opened phase 1c with the first
//! half of a pair: the resolved distances of R §1.2 ([`matching::scales`]), the frame-pair
//! hypotheses of R §5.1 ([`matching::hypotheses`]), the coarse breakline score of R §5.2
//! ([`matching::coarse`]), the non-maximum suppression of R §5.3 ([`matching::nms`]) and the
//! [`Pair`](matching::pair::Pair) of R §4.2 that drives them — with the `hypotheses`, `coarse` and
//! `nms` rows of the parity table. Step C2 added the refinement: Open3D's `registration_icp` as
//! R §7 freezes it ([`matching::icp`]) and the two ladders it drives ([`matching::ladder`]) — the
//! breakline rungs of R §5.4 with their re-score, the suppression of R §5.5, and the four
//! registration and fracture rungs of R §5.6 — with the `stage1` and `stage2` rows of the parity
//! table. The remaining algorithm modules are documented but empty; they
//! are filled in step by step, each step gated on the fixtures under `fixtures/` and on
//! `tools/compare_fixtures.py`.

pub mod assembly;
pub mod collection;
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

pub use collection::Entry;
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
///
/// `2`: step B1 added `labels u8[m]` (R §3.4). `3`: step B2 added the five breakline tensors
/// `brk_P`, `brk_ns`, `brk_nf`, `brk_f` (f32 `[k, 3]`) and `brk_sub` (u32 `[j]`) of R §3.5.3–3.5.5,
/// with their `brk_params` in the metadata. `4`: step B3 added the five sampled ones — `S`,
/// `Pf` (f32 `[n, 3]`), `sp`, `fp`, `margin_idx` (u32 `[n]`) of R §3.5.1–3.5.2 and §3.5.6 — with
/// their `md_params` and the two face areas of R §3.4. `5`: task T1 added `brk_valid` (bool `[k]`),
/// R §3.5.4's frame-validity mask, which used to be recomputed from the narrowed frames and is now
/// the one the subset was filtered with (defect D3 of the phase-1b verification). Every tensor the
/// port reads back must be in the file, so an older cache is refused and its fragment recomputed,
/// rather than being read back half empty.
pub const CACHE_VERSION: u32 = 5;
