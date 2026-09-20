# Desktop App, Milestone 1 — the core's session API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a finished match reusable — save it, reassemble from it under new constraints in under a second, refine and write outputs from the result — while the CLI's output stays byte for byte what it is.

**Architecture:** `pipeline::run_with` stays the orchestrator. Its duplicated "apply constraints → assemble → object round" block becomes `assemble_stage`, its tail becomes `write_outputs`. A new `sherd_core::session` module adds `MatchState` (a JSON snapshot taken before the last `constraints::apply`) and `reassemble`, which mirrors the candidate layout a full run under the same constraints would have. Backend resolution moves from the CLI into a new `sherd-backend` crate so the app can link it.

**Tech Stack:** Rust 2024, MSRV 1.89, `serde_json` (already in the workspace, `float_roundtrip` on), `rayon`, `nalgebra`. No new third-party dependency.

**Spec:** `docs/superpowers/specs/2026-09-20-desktop-app-design.md` §3 (cited as `A §3`). This plan is milestone 1 of A §13; milestones 2–6 get their own plans once this one's interfaces exist.

## Global Constraints

- **CLI output byte for byte.** The gate is `cargo test -p sherd-cli --test run_cli`. No task may change what `sherd-refit-rs run` writes; every new `RunOptions` field defaults to today's behaviour and the CLI never sets it.
- **Verification budget (A §11, the user's instruction of 2026-09-20).** Run **only** the commands a task lists. While editing: `cargo check -p <crate>`. No `cargo test --workspace`, no clippy, no release builds, no `parity` inside a task — Task 8 runs the milestone gate once. Every automated run uses `fixtures/slab/input` (two fragments, `pieceA` and `pieceB`); never `input/karas_reduced` or `input/sfspp`.
- **Workspace lints** (`Cargo.toml`): `missing_docs`, `unreachable_pub`, clippy `pedantic` are warnings that Task 8 turns into errors; `todo!`, `unimplemented!`, `dbg!` and `unsafe` are denied. Every `pub` and `pub(crate)` item gets a doc comment; items not reachable from outside the crate are `pub(crate)`, not `pub`.
- **Do not touch** `sherd_refit/` (the frozen Python reference), `crates/sherd-parity/`, `fixtures/`, or any file under `crates/sherd-core/src/export/viewer.html`.
- **Doc-comment voice:** the tree explains *why*, citing `R §n` (algorithm reference), `D §n` (`2026-09-06-rust-core-design.md`) and now `A §n`. Match it; no banner comments, no restating the code.
- **Commits:** branch `desktop-app`. Subject `M1.<task>: <what changed, as a sentence>`, body optional, last line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **The slab's tier behaviour** (from `crates/sherd-cli/tests/run_cli.rs`, const `AMBIGUOUS`): with `Params::default()` (`tiers: None`) the slab's one join is *accepted* and assembled; with `tiers: Some(Thresholds::default())` it is only **probable** and nothing is assembled; with `rival_refused: false` it is **confirmed**. The tests below rely on all three.

## File Structure

| file | responsibility |
|---|---|
| `crates/sherd-core/src/pipeline.rs` (modify) | orchestration; gains `assemble_stage`, `write_outputs`, `Finished`, `Assembled`; `RunOptions::{match_state, excluded}`; several private items become `pub(crate)` |
| `crates/sherd-core/src/session.rs` (create) | `MatchState` + save/load, `load_fragments`, `reassemble`, `Reassembled`, `refine_poses`, `write_reviewed` |
| `crates/sherd-core/src/matching/pair.rs` (modify) | `Candidate` gains serde derives through a matrix adapter |
| `crates/sherd-core/src/tiers.rs` (modify) | serde + `PartialEq` on `TierReport`, `Probes` and what they contain; `classify_watched` |
| `crates/sherd-core/src/error.rs` (modify) | `Error::State` |
| `crates/sherd-core/src/collection.rs` (modify) | `discover_excluding` |
| `crates/sherd-core/src/lib.rs` (modify) | `pub mod session;` |
| `crates/sherd-core/tests/session.rs` (create) | every test of this milestone that needs a slab run |
| `crates/sherd-backend/` (create) | `resolve`, `Resolved`, `info_lines`, `selftest_lines`, moved from the CLI |
| `crates/sherd-cli/src/gpu.rs`, `Cargo.toml` (modify) | keeps `check`; re-exports the rest from `sherd-backend` |

---

### Task 1: `assemble_stage` and `write_outputs` — two extractions from `run_with`

A pure refactor. Nothing a run writes may change.

**Files:**
- Modify: `crates/sherd-core/src/pipeline.rs` (`run_with`, lines ≈ 421–1027)

**Interfaces:**
- Consumes: nothing new.
- Produces (all `pub(crate)` in `crate::pipeline`):
  - `struct Assembled { assembly: Assembly, constraints: Option<constraints::Report>, objects: Option<objects::ObjectReport> }`
  - `fn assemble_stage(exec: &dyn Executor, fragments: &[Fragment], names: &[String], pieces: &[Piece<'_>], candidates: &mut [Candidate], tiers: Option<&mut TierReport>, plan: Option<&Resolved>, params: &Params, started: Option<Instant>, stages: &mut StageLog) -> Assembled`
  - `struct Finished<'a>` (fields below) and `fn write_outputs(out_dir: &Path, done: &Finished<'_>, options: &RunOptions, written: Vec<PathBuf>) -> Result<Vec<PathBuf>>`
  - `StageLog` becomes `pub(crate)` (its fields and methods too).

- [ ] **Step 1: Record the green baseline**

Run: `cargo test -p sherd-cli --test run_cli 2>&1 | tail -5`
Expected: `test result: ok.` Note the wall time; Step 5 must show the same tests passing.

- [ ] **Step 2: Extract `assemble_stage`**

Add below `pinned_candidate` in `pipeline.rs`:

```rust
/// What R §8 and the two passes around it leave behind.
#[derive(Debug)]
pub(crate) struct Assembled {
    /// R §8's assembly, after roadmap item 4's one object round.
    pub(crate) assembly: crate::assembly::Assembly,
    /// What `constraints.json` did, when there was one.
    pub(crate) constraints: Option<constraints::Report>,
    /// Roadmap item 4's report, when `Params::objects` is set.
    pub(crate) objects: Option<objects::ObjectReport>,
}

/// `constraints::apply`, R §8 and the object round — the block `run_with` runs after every
/// matching pass, and the whole of what [`crate::session::reassemble`] runs (A §3.1).
///
/// `started` is the first pass's clock: R §11.2's `timings["assembly"]` covers building the pieces
/// as well, so the caller starts it and this function only closes it, between R §8 and the object
/// round, where `run_with` always closed it. `None` records nothing.
#[allow(clippy::too_many_arguments, reason = "R §8's inputs, one argument each")]
pub(crate) fn assemble_stage(
    exec: &dyn Executor,
    fragments: &[Fragment],
    names: &[String],
    pieces: &[Piece<'_>],
    candidates: &mut [Candidate],
    mut tiers: Option<&mut crate::tiers::TierReport>,
    plan: Option<&Resolved>,
    params: &Params,
    started: Option<Instant>,
    stages: &mut StageLog,
) -> Assembled {
    let gate = if params.tiers.is_some() { Gate::Confirmed } else { Gate::Accepted };
    let constraints = plan.map(|r| {
        constraints::apply(r, names, candidates, tiers.as_deref_mut(), gate == Gate::Confirmed)
    });
    let mut assembly =
        assemble_under(exec, pieces, candidates, params, gate, plan, params.objects.as_ref());
    if let Some(started) = started {
        stages.finish("assembly", started.elapsed().as_secs_f64());
    }
    let objects = object_round(
        fragments,
        names,
        candidates,
        &mut assembly,
        pieces,
        ObjectRound { params, gate, constraints: plan, tiers, exec, stages },
    );
    Assembled { assembly, constraints, objects }
}
```

In `run_with`, replace **both** occurrences of the block (first pass ≈ lines 668–730: `let gate = …`, `let mut honoured = …apply…`, `let mut assembly = assemble_under(…)`, `stages.finish("assembly", …)`, `let mut object_pass = object_round(…)`; second pass ≈ lines 771–797) with calls:

```rust
    // first pass — `started`, `samples`, `scenes` and `pieces` are built exactly as before, above
    let Assembled { mut assembly, constraints: mut honoured, objects: mut object_pass } =
        assemble_stage(
            engine.exec, &fragments, &names, &pieces, &mut candidates, tiered.as_mut(),
            plan.as_ref(), params, Some(started), &mut stages,
        );
```

```rust
    // second pass, after `agree_pass`
        Assembled { assembly, constraints: honoured, objects: object_pass } = assemble_stage(
            engine.exec, &fragments, &names, &pieces, &mut candidates, tiered.as_mut(),
            plan.as_ref(), params, None, &mut stages,
        );
```

The one reordering: `constraints::apply` used to run just before `let started = Instant::now()` and now runs just after the pieces are built. It reads and writes only `candidates` and `tiered`, which building the pieces does not touch. Delete `gate` from `run_with` if nothing else reads it (`cargo check` will say).

Make `StageLog`, its three fields and its three methods `pub(crate)`.

Run: `cargo check -p sherd-core`
Expected: no errors.

- [ ] **Step 3: Extract `write_outputs`**

Add below `assemble_stage`:

```rust
/// Everything R §11's writers and `export/` read, borrowed from whoever finished an assembly —
/// `run_with`, or [`crate::session::write_reviewed`] (A §3.1).
#[derive(Debug)]
pub(crate) struct Finished<'a> {
    /// The input directory, for the collection's title.
    pub(crate) input: &'a Path,
    /// The collection.
    pub(crate) fragments: &'a [Fragment],
    /// Its names, by [`FragId`].
    pub(crate) names: &'a [String],
    /// Every candidate, as the assembly saw them.
    pub(crate) candidates: &'a [Candidate],
    /// R §8's result.
    pub(crate) assembly: &'a crate::assembly::Assembly,
    /// The tier report, when the tier pass ran.
    pub(crate) tiers: Option<&'a crate::tiers::TierReport>,
    /// What the constraints did.
    pub(crate) constraints: Option<&'a constraints::Report>,
    /// The review images' index.
    pub(crate) review: Option<&'a crate::review::ReviewIndex>,
    /// Roadmap item 4's report.
    pub(crate) objects: Option<&'a objects::ObjectReport>,
    /// One recentred world pose per fragment.
    pub(crate) poses: &'a [Matrix4<f64>],
    /// The collection's median wall thickness.
    pub(crate) thickness: f64,
    /// R §11.2's `timings`.
    pub(crate) timings: &'a Timings,
    /// `report.json`'s `memory` block.
    pub(crate) memory: Option<&'a MemoryReport>,
}

/// R §11's five writers and `export/`'s files. `written` carries what the caller wrote already
/// (the measurement, the review images) because `README.txt` lists every file of the folder.
pub(crate) fn write_outputs(
    out_dir: &Path,
    done: &Finished<'_>,
    options: &RunOptions,
    mut written: Vec<PathBuf>,
) -> Result<Vec<PathBuf>> {
    // body: see below
    Ok(written)
}
```

Body: **move** `run_with`'s lines from `let stats: Vec<FragmentStats> = …` through `written.push(crate::export::readme::write_readme(…)?);` (≈ 912–1002) into it, unchanged except for these substitutions:

| in `run_with` | in `write_outputs` |
|---|---|
| `fragments`, `names`, `candidates`, `poses`, `thickness` | `done.fragments`, `done.names`, `done.candidates`, `done.poses`, `done.thickness` |
| `assembly.used` / `.rejected` / `.groups` / `.order` | `done.assembly.…` |
| `tiered.as_ref()` | `done.tiers` |
| `tiered.is_some()` | `done.tiers.is_some()` |
| `honoured.as_ref()` | `done.constraints` |
| `review.as_ref()` | `done.review` |
| `object_pass.as_ref()` | `done.objects` |
| `&stages.timings` | `done.timings` |
| `stages.memory().as_ref()` | `done.memory` |
| `params` | `&options.params` |
| `input` | `done.input` |

`run_with` keeps `let started = Instant::now();`, builds `written` from `measured` and `review_files` as today, then:

```rust
    let memory = stages.memory();
    let written = write_outputs(
        out_dir,
        &Finished {
            input,
            fragments: &fragments,
            names: &names,
            candidates: &candidates,
            assembly: &assembly,
            tiers: tiered.as_ref(),
            constraints: honoured.as_ref(),
            review: review.as_ref(),
            objects: object_pass.as_ref(),
            poses: &poses,
            thickness,
            timings: &stages.timings,
            memory: memory.as_ref(),
        },
        options,
        written,
    )?;
```

followed by the unchanged `stages.finish("output", …)`, the `tracing::info!` and `Ok(RunSummary { … })`.

Run: `cargo check -p sherd-core`
Expected: no errors.

- [ ] **Step 4: Format**

Run: `cargo fmt -p sherd-core`

- [ ] **Step 5: The byte-for-byte gate**

Run: `cargo test -p sherd-cli --test run_cli 2>&1 | tail -5`
Expected: `test result: ok.` with the same number of tests as Step 1.

- [ ] **Step 6: Commit**

```bash
git add crates/sherd-core/src/pipeline.rs
git commit -m "M1.1: run_with's assembly block and its writers become two functions a second caller can use"
```

---

### Task 2: `MatchState` — what a match leaves behind, saved and read back exactly

**Files:**
- Create: `crates/sherd-core/src/session.rs`
- Create: `crates/sherd-core/tests/session.rs`
- Modify: `crates/sherd-core/src/lib.rs`, `src/error.rs`, `src/matching/pair.rs`, `src/tiers.rs`, `src/pipeline.rs`

**Interfaces:**
- Consumes: Task 1's `assemble_stage` call sites in `run_with`.
- Produces:
  - `sherd_core::session::MatchState { names: Vec<String>, params: Params, thickness: f64, resolution: f64, pairs: usize, skipped_pairs: usize, backend: String, candidates: Vec<Candidate>, tiers: Option<TierReport> }` with `save(&self, path: &Path) -> Result<()>` and `load(path: &Path) -> Result<Self>`; `#[derive(Clone, Debug, PartialEq)]`
  - `sherd_core::session::{STATE_FORMAT: &str = "sherd-match-state", STATE_VERSION: u32 = 1}`
  - `RunOptions::match_state: Option<PathBuf>` (default `None`)
  - `Error::State { path: PathBuf, message: String }`
  - `Candidate`, `TierReport`, `Probes`: `Serialize + Deserialize`; `TierReport: PartialEq`

- [ ] **Step 1: Write the failing tests**

Create `crates/sherd-core/tests/session.rs`:

```rust
//! A §3: a match saved by a run, read back, and reassembled. Every run here is `fixtures/slab`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sherd_core::Params;
use sherd_core::pipeline::{self, RunOptions, RunSummary};
use sherd_core::session::MatchState;

fn slab() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("session-{tag}"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// One slab run without refinement, meshes or previews, leaving `match.state` in its folder.
fn slab_run(tag: &str, params: Params) -> (RunSummary, PathBuf) {
    let out = scratch(tag);
    let state = out.join("match.state");
    let options = RunOptions {
        params,
        refine: false,
        preview: false,
        write_meshes: false,
        match_state: Some(state.clone()),
        ..RunOptions::default()
    };
    (pipeline::run(&slab(), &out, &options).expect("the slab runs"), state)
}

/// The run most tests read: `Params::default()`, so R §6.5's own gate, and the slab assembled.
fn accepted_run() -> &'static (RunSummary, PathBuf) {
    static RUN: OnceLock<(RunSummary, PathBuf)> = OnceLock::new();
    RUN.get_or_init(|| slab_run("accepted", Params::default()))
}

#[test]
fn a_saved_match_reads_back_exactly() {
    let (summary, path) = accepted_run();
    let state = MatchState::load(path).expect("the state loads");
    assert_eq!(state.names, summary.names);
    assert_eq!(state.candidates.len(), summary.candidates.len());
    assert_eq!(state.candidates[0].transform, summary.candidates[0].transform);
    assert_eq!(state.thickness.to_bits(), summary.thickness.to_bits());

    let again = path.with_extension("again");
    state.save(&again).expect("the state saves");
    assert_eq!(MatchState::load(&again).expect("and loads"), state);
    assert_eq!(std::fs::read(&again).unwrap(), std::fs::read(path).unwrap());
}

#[test]
fn a_state_of_another_version_or_format_is_refused() {
    let dir = scratch("refused");
    for (name, body) in [
        ("version", r#"{"format":"sherd-match-state","version":999}"#),
        ("format", r#"{"format":"something-else","version":1}"#),
        ("garbage", "not json at all"),
    ] {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        let err = MatchState::load(&path).expect_err("refused");
        assert!(matches!(err, sherd_core::Error::State { .. }), "{name}: {err}");
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sherd-core --test session 2>&1 | tail -5`
Expected: compile error — `session` and `match_state` do not exist.

- [ ] **Step 3: `Error::State`**

In `crates/sherd-core/src/error.rs`, add to `enum Error` after `Write`:

```rust
    /// A saved match ([`MatchState`](crate::session::MatchState)) that is not one, is of another
    /// version, or does not describe the collection it is being used with (A §3.2).
    #[error("{path}: {message}")]
    State {
        /// The state file.
        path: PathBuf,
        /// What is wrong with it.
        message: String,
    },
```

- [ ] **Step 4: Serde on `Candidate`, `TierReport`, `Probes`**

`crates/sherd-core/src/matching/pair.rs` — `Candidate` holds a `Matrix4<f64>` and `nalgebra`'s `serde` feature is off. Add, above `Candidate`:

```rust
/// A pose as `report.json` and every other file of this project writes one: four rows of four
/// (README, «Одно соглашение о матрице»).
pub(crate) mod rows {
    use nalgebra::Matrix4;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<S: Serializer>(m: &Matrix4<f64>, s: S) -> Result<S::Ok, S::Error> {
        let rows: [[f64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]));
        rows.serialize(s)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Matrix4<f64>, D::Error> {
        let rows = <[[f64; 4]; 4]>::deserialize(d)?;
        Ok(Matrix4::from_fn(|r, c| rows[r][c]))
    }
}
```

and change the struct's derive and field:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    // …
    #[serde(with = "rows")]
    pub transform: Matrix4<f64>,
```

`crates/sherd-core/src/tiers.rs` — `TierReport` (≈ line 903) becomes `#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]`; `Probes` (≈ line 214) gains `Serialize, Deserialize`. Run `cargo check -p sherd-core`: for every type the compiler then names as missing `Serialize`/`Deserialize`/`PartialEq` (expected: `ScoreRow`, `Research`, possibly `RivalKind`), add the missing derive to that type. Any of them that holds a `Matrix4<f64>` takes `#[serde(with = "crate::matching::pair::rows")]` on that field. Do **not** add, remove or change any `#[serde(...)]` attribute already present — `report.json`'s bytes depend on them.

- [ ] **Step 5: `session.rs` with `MatchState`**

Create `crates/sherd-core/src/session.rs`:

```rust
//! A match that outlives its run (A §3): saved by `run_with`, read back by a review session, and
//! reassembled under the reviewer's decisions without matching a pair again.
//!
//! Matching is 94 % of a run (997 of 1066 s on the 155-fragment `karas`); R §8's assembly is
//! 0.14 s of it. A decision about one join therefore costs a reassembly, provided the candidate
//! list the assembly was built from is still there — which is what [`MatchState`] is.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::matching::pair::Candidate;
use crate::params::Params;
use crate::tiers::TierReport;

/// The `format` tag of a state file.
pub const STATE_FORMAT: &str = "sherd-match-state";
/// The version this build writes and the only one it reads.
pub const STATE_VERSION: u32 = 1;

/// The candidate list and the tier report **as they stood before the last
/// `constraints::apply`**, with what R §8 needs to be run over them again.
///
/// Before, not after: `apply` promotes a `must_join` in place and the object round demotes in
/// place, so the lists a finished run holds already carry that run's constraints. Reassembling
/// under *other* constraints has to start from the match's own opinion.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchState {
    /// The collection's names, in R §2's order; a [`FragId`](crate::FragId) indexes them.
    pub names: Vec<String>,
    /// Every threshold the match ran with.
    pub params: Params,
    /// The collection's median wall thickness (R §4.1).
    pub thickness: f64,
    /// The collection's median working-mesh edge.
    pub resolution: f64,
    /// Pairs matched.
    pub pairs: usize,
    /// Pairs R §4.1's wall-ratio filter skipped.
    pub skipped_pairs: usize,
    /// D §4.3's `backend` label of the run that matched — `cpu`, or `gpu:Apple M2 Pro`.
    pub backend: String,
    /// Every candidate of every pair, in pair order, pinned poses last.
    pub candidates: Vec<Candidate>,
    /// The tier report over exactly that list, when the tier pass ran.
    pub tiers: Option<TierReport>,
}

/// The file's shape: the tag and the version first, so that a reader can refuse before it parses
/// 35 MB of candidates it would misread.
#[derive(Serialize)]
struct FileRef<'a> {
    format: &'a str,
    version: u32,
    names: &'a [String],
    params: &'a Params,
    thickness: f64,
    resolution: f64,
    pairs: usize,
    skipped_pairs: usize,
    backend: &'a str,
    candidates: &'a [Candidate],
    tiers: Option<&'a TierReport>,
}

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
}

#[derive(Deserialize)]
struct FileOwned {
    names: Vec<String>,
    params: Params,
    thickness: f64,
    resolution: f64,
    pairs: usize,
    skipped_pairs: usize,
    backend: String,
    candidates: Vec<Candidate>,
    tiers: Option<TierReport>,
}

impl MatchState {
    /// Writes the state as JSON (A §3.2).
    ///
    /// JSON and not a compact binary format: `Scores`, `Evidence`, `Probes` and `Params` are
    /// `report.json`'s own types and use `skip_serializing_if`, which a format that is not
    /// self-describing cannot read back. `serde_json`'s `float_roundtrip` makes every `f64` exact.
    ///
    /// # Errors
    ///
    /// [`Error::Write`] when the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = FileRef {
            format: STATE_FORMAT,
            version: STATE_VERSION,
            names: &self.names,
            params: &self.params,
            thickness: self.thickness,
            resolution: self.resolution,
            pairs: self.pairs,
            skipped_pairs: self.skipped_pairs,
            backend: &self.backend,
            candidates: &self.candidates,
            tiers: self.tiers.as_ref(),
        };
        let json =
            serde_json::to_vec(&file).map_err(|e| Error::write(path, std::io::Error::other(e)))?;
        std::fs::write(path, json).map_err(|e| Error::write(path, e))
    }

    /// Reads a state back, bit for bit what [`save`](Self::save) was given.
    ///
    /// # Errors
    ///
    /// [`Error::State`] for a file that is not a state, is of another version, or does not parse.
    pub fn load(path: &Path) -> Result<Self> {
        let refuse = |message: String| Error::State { path: path.to_owned(), message };
        let bytes = std::fs::read(path).map_err(|e| refuse(e.to_string()))?;
        let header: Header =
            serde_json::from_slice(&bytes).map_err(|e| refuse(format!("not a match state: {e}")))?;
        if header.format != STATE_FORMAT {
            return Err(refuse(format!("not a match state: format `{}`", header.format)));
        }
        if header.version != STATE_VERSION {
            return Err(refuse(format!(
                "match state version {} — this build reads version {STATE_VERSION}; run the \
                 collection again",
                header.version
            )));
        }
        let f: FileOwned = serde_json::from_slice(&bytes).map_err(|e| refuse(e.to_string()))?;
        Ok(Self {
            names: f.names,
            params: f.params,
            thickness: f.thickness,
            resolution: f.resolution,
            pairs: f.pairs,
            skipped_pairs: f.skipped_pairs,
            backend: f.backend,
            candidates: f.candidates,
            tiers: f.tiers,
        })
    }
}
```

In `crates/sherd-core/src/lib.rs` add `pub mod session;` after `pub mod rng;`.

- [ ] **Step 6: The snapshot in `run_with`**

`crates/sherd-core/src/pipeline.rs`:

1. `RunOptions` gains, after `viewer_faces`:

```rust
    /// Where to save the match for a later [`reassemble`](crate::session::reassemble) (A §3.1), or
    /// `None` — the default, and what the CLI always passes — for a run that keeps nothing.
    pub match_state: Option<PathBuf>,
```

and `match_state: None,` in `Default`.

2. Immediately **before each** of the two `assemble_stage` calls:

```rust
    // A §3.1: the lists as the match left them, before this run's own constraints touch them.
    let mut snapshot = options.match_state.as_ref().map(|_| (candidates.clone(), tiered.clone()));
```

(second site: `snapshot = options.match_state.as_ref().map(|_| (candidates.clone(), tiered.clone()));` — the second pass's list replaces the first's.)

3. Right after `let used: Vec<(FragId, FragId)> = …`:

```rust
    if let (Some(path), Some((candidates, tiers))) = (&options.match_state, snapshot.take()) {
        crate::session::MatchState {
            names: names.clone(),
            params: *params,
            thickness,
            resolution,
            pairs: pairs.len(),
            skipped_pairs: skipped,
            backend: options.backend_label(),
            candidates,
            tiers,
        }
        .save(path)?;
        tracing::info!(out = %path.display(), "match state saved");
    }
```

- [ ] **Step 7: Run the tests**

Run: `cargo test -p sherd-core --test session 2>&1 | tail -5`
Expected: `2 passed`. If `a_saved_match_reads_back_exactly` fails on the final `assert_eq!` of states, a type in the tree has a non-`Option` field behind `skip_serializing_if` with no `#[serde(default)]`; add `default` to that one attribute (it changes reading only, never the bytes written).

- [ ] **Step 8: Commit**

```bash
cargo fmt -p sherd-core
git add crates/sherd-core
git commit -m "M1.2: MatchState -- the match as it stood before the run's own constraints, saved as JSON and read back exactly"
```

---

### Task 3: `reassemble` — R §8 again, under other constraints, without matching

**Files:**
- Modify: `crates/sherd-core/src/session.rs`, `crates/sherd-core/src/pipeline.rs` (visibility only), `crates/sherd-core/tests/session.rs`

**Interfaces:**
- Consumes: `pipeline::{assemble_stage, Assembled, StageLog, pinned_candidate, preprocess_collection}` (make the last two `pub(crate)`), `MatchState`.
- Produces:
  - `session::load_fragments(entries: &[Entry], target_faces: usize, cache_dir: Option<&Path>, budget: Budget, seed: u64, watch: &Watch) -> Result<Vec<Fragment>>`
  - `session::Reassembled { assembly: Assembly, candidates: Vec<Candidate>, tiers: Option<TierReport>, constraints: Option<constraints::Report>, objects: Option<ObjectReport>, poses: Vec<Matrix4<f64>>, used: Vec<(FragId, FragId)> }` — `poses` are R §8.2-recentred and **not** refined
  - `session::reassemble(engine: Engine<'_>, fragments: &[Fragment], state: &MatchState, constraints: Option<&Constraints>) -> Result<Reassembled>`

- [ ] **Step 1: Write the failing tests**

Append to `crates/sherd-core/tests/session.rs`:

```rust
use sherd_core::assembly::constraints::Constraints;
use sherd_core::executor::Engine;
use sherd_core::memory::Budget;
use sherd_core::progress::Watch;
use sherd_core::session::{self, Reassembled};
use sherd_core::tiers::{Thresholds, Tier};

fn fragments(params: &Params) -> Vec<sherd_core::fragment::Fragment> {
    let entries = sherd_core::collection::discover(slab()).expect("the slab is found");
    session::load_fragments(
        &entries,
        RunOptions::default().target_faces,
        None,
        Budget::default_for_machine(),
        params.seed,
        &Watch::default(),
    )
    .expect("the slab preprocesses")
}

fn reassembled(state: &MatchState, constraints: Option<&str>) -> Reassembled {
    let parsed: Option<Constraints> =
        constraints.map(|json| serde_json::from_str(json).expect("valid constraints"));
    session::reassemble(Engine::REFERENCE, &fragments(&state.params), state, parsed.as_ref())
        .expect("reassembles")
}

#[test]
fn reassembling_with_no_constraints_is_the_run_before_refinement() {
    let (summary, path) = accepted_run();
    let state = MatchState::load(path).unwrap();
    let out = reassembled(&state, None);
    assert_eq!(out.assembly.groups, summary.groups);
    assert_eq!(out.used, summary.used);
    assert_eq!(out.poses, summary.poses, "bit for bit: the run had `refine: false`");
}

#[test]
fn a_rejected_join_splits_its_group() {
    let (summary, path) = accepted_run();
    assert!(summary.groups.iter().any(|g| g.len() == 2), "the slab assembles under R §6.5's gate");
    let state = MatchState::load(path).unwrap();
    let out = reassembled(
        &state,
        Some(r#"{"version":1,"must_not_join":[["pieceA","pieceB"]]}"#),
    );
    assert!(out.assembly.groups.iter().all(|g| g.len() == 1));
    assert!(out.used.is_empty());
}

#[test]
fn an_accepted_probable_join_is_placed_at_the_pose_that_was_accepted() {
    // The shipped tier rule confirms nothing on this slab (`run_cli.rs`, `AMBIGUOUS`): its one
    // join is probable, and a run places nothing. That is the app's review case exactly.
    let params = Params { tiers: Some(Thresholds::default()), ..Params::default() };
    let (summary, path) = slab_run("probable", params);
    assert!(summary.groups.iter().all(|g| g.len() == 1), "nothing is confirmed, nothing placed");
    let state = MatchState::load(&path).unwrap();

    // The reviewer accepts the pair's *second* accepted pose where there is one, to show that the
    // pose they chose is the pose that is placed, not the pair's best.
    let accepted: Vec<_> = state.candidates.iter().filter(|c| c.accepted).collect();
    let chosen = accepted.get(1).or(accepted.first()).expect("the slab has an accepted candidate");
    let m = chosen.transform;
    let rows: Vec<String> = (0..4)
        .map(|r| format!("[{:?},{:?},{:?},{:?}]", m[(r, 0)], m[(r, 1)], m[(r, 2)], m[(r, 3)]))
        .collect();
    let json = format!(
        r#"{{"version":1,"must_join":[{{"a":"{}","b":"{}","pose":[{}]}}]}}"#,
        state.names[chosen.a as usize],
        state.names[chosen.b as usize],
        rows.join(",")
    );

    let out = reassembled(&state, Some(&json));
    assert!(out.assembly.groups.iter().any(|g| g.len() == 2), "the accepted join is placed");
    let placed = out.candidates.last().expect("the pinned candidate is appended last");
    assert_eq!(placed.transform, chosen.transform, "the pose is the one that was accepted");
    assert_eq!(placed.scores, chosen.scores, "R §6 was not run again: the match had scored it");
    assert_eq!(placed.tier, Tier::Confirmed);
    assert_eq!(
        out.candidates.iter().filter(|c| (c.a, c.b) == (chosen.a, chosen.b)).count(),
        1,
        "the pair's matched candidates are gone, as in a full run that pins the pair"
    );
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sherd-core --test session 2>&1 | tail -5`
Expected: compile error — `load_fragments`, `reassemble`, `Reassembled` do not exist.

- [ ] **Step 3: Visibility**

In `crates/sherd-core/src/pipeline.rs` change `fn pinned_candidate` and `fn preprocess_collection` to `pub(crate) fn`.

- [ ] **Step 4: Implement**

Append to `crates/sherd-core/src/session.rs` (and extend its `use` block with what the compiler asks for: `nalgebra::Matrix4`, `rayon::prelude::*`, `crate::assembly::{Assembly, Piece, recenter}`, `crate::assembly::constraints::{self, Constraints}`, `crate::collection::Entry`, `crate::executor::Engine`, `crate::fragment::Fragment`, `crate::memory::Budget`, `crate::objects::ObjectReport`, `crate::pipeline`, `crate::progress::Watch`, `crate::tiers::Tier`, `crate::types::FragId`):

```rust
/// The collection's fragments through R §3.7's cache — what `run_with` itself starts from, for a
/// session that starts from a saved match instead of a run.
///
/// # Errors
///
/// The first fragment that fails to preprocess fails the call, as it fails a run.
pub fn load_fragments(
    entries: &[Entry],
    target_faces: usize,
    cache_dir: Option<&Path>,
    budget: Budget,
    seed: u64,
    watch: &Watch,
) -> Result<Vec<Fragment>> {
    pipeline::preprocess_collection(entries, target_faces, cache_dir, budget, seed, watch)
}

/// What [`reassemble`] produced: R §8's result and the lists it was built from.
#[derive(Debug)]
pub struct Reassembled {
    /// R §8's assembly after the object round.
    pub assembly: Assembly,
    /// The candidate list the assembly read — the state's, with each pinned pair's candidates
    /// replaced by its pinned one, and with what `apply` and the object round did to the bands.
    pub candidates: Vec<Candidate>,
    /// The tier report over exactly that list.
    pub tiers: Option<TierReport>,
    /// What each constraint did.
    pub constraints: Option<constraints::Report>,
    /// Roadmap item 4's report.
    pub objects: Option<ObjectReport>,
    /// One world pose per fragment after R §8.2's recentring. **Not refined** (R §9).
    pub poses: Vec<Matrix4<f64>>,
    /// The joins the assembly took, in the order it took them.
    pub used: Vec<(FragId, FragId)>,
}

/// R §8 over a saved match, under `constraints`, without matching anything (A §3.1).
///
/// The result is the one a **full run under the same constraints** would assemble from, given the
/// same match. A full run does not match a pair that `must_join` pins with a pose: the pair's only
/// candidate is the pinned one, appended after every matched candidate. This does the same to the
/// saved list — the pair's matched candidates and their tier rows go, the pinned one is appended.
/// When the pinned pose is bit for bit the pose of a candidate the match scored, that candidate's
/// scores and evidence are kept and R §6 is not run again; any other pose is scored by
/// `pinned_candidate`, as the full run would score it.
///
/// The tier pass is not repeated. A decision changes what is placed, not what the engine believed
/// about the pairs nobody decided on.
///
/// # Errors
///
/// [`Error::State`] when `fragments` is not the collection the state was matched on, and whatever
/// `constraints::resolve` refuses (an unknown name, a pair in two lists, a pose that is not rigid).
pub fn reassemble(
    engine: Engine<'_>,
    fragments: &[Fragment],
    state: &MatchState,
    constraints: Option<&Constraints>,
) -> Result<Reassembled> {
    if !fragments.iter().map(|f| f.name.as_str()).eq(state.names.iter().map(String::as_str)) {
        return Err(Error::State {
            path: std::path::PathBuf::new(),
            message: "the fragments are not the collection this match was made on".to_owned(),
        });
    }
    let params = &state.params;
    let plan = constraints.map(|file| constraints::resolve(file, &state.names)).transpose()?;

    let mut candidates = state.candidates.clone();
    let mut tiers = state.tiers.clone();
    if let Some(plan) = &plan {
        for forced in &plan.forced {
            let Some(pose) = forced.pose else { continue };
            let pair = constraints::key(forced.a, forced.b);
            let scored = candidates
                .iter()
                .position(|c| constraints::key(c.a, c.b) == pair && c.transform == pose);
            let pinned = match scored {
                Some(i) => Candidate { accepted: true, ..candidates[i] },
                None => {
                    pipeline::pinned_candidate(engine, fragments, forced.a, forced.b, &pose, params)
                }
            };
            let row = scored.and_then(|i| {
                tiers.as_ref().map(|t| (t.evidence[i].clone(), t.probes[i].clone()))
            });
            let keep: Vec<bool> =
                candidates.iter().map(|c| constraints::key(c.a, c.b) != pair).collect();
            retain(&mut candidates, &keep);
            if let Some(report) = tiers.as_mut() {
                retain(&mut report.tiers, &keep);
                retain(&mut report.evidence, &keep);
                retain(&mut report.probes, &keep);
                let (evidence, probes) = row.unwrap_or((None, None));
                report.tiers.push(pinned.tier);
                report.evidence.push(evidence);
                report.probes.push(probes);
            }
            candidates.push(pinned);
        }
    }

    let samples: Vec<Vec<[f64; 3]>> = fragments.iter().map(|f| f.samples.surface_f64()).collect();
    let scenes: Vec<_> = fragments.par_iter().map(Fragment::surface_scene_arc).collect();
    let piece = |with_mesh: bool| -> Vec<Piece<'_>> {
        fragments
            .iter()
            .zip(&samples)
            .zip(&scenes)
            .map(|((f, s), scene)| Piece {
                thick: f.thick,
                res: f.res(),
                watertight: f.watertight,
                mesh: if with_mesh { scene.as_deref() } else { None },
                s_pen: s,
            })
            .collect()
    };
    let pieces = piece(true);
    let mut stages = pipeline::StageLog::default();
    let pipeline::Assembled { assembly, constraints: report, objects } = pipeline::assemble_stage(
        engine.exec,
        fragments,
        &state.names,
        &pieces,
        &mut candidates,
        tiers.as_mut(),
        plan.as_ref(),
        params,
        None,
        &mut stages,
    );
    let used = assembly.used.iter().map(|&i| (candidates[i].a, candidates[i].b)).collect();
    let poses = recenter(&assembly.poses, &piece(false), &assembly.groups);
    Ok(Reassembled { assembly, candidates, tiers, constraints: report, objects, poses, used })
}

/// `Vec::retain` by a mask computed once, so that three parallel vectors lose the same rows.
fn retain<T>(items: &mut Vec<T>, keep: &[bool]) {
    let mut at = 0;
    items.retain(|_| {
        at += 1;
        keep[at - 1]
    });
}
```

`Forced::pose` is an `Option<Matrix4<f64>>`; if `Forced` is not `Copy`, write `let Some(pose) = forced.pose.as_ref().copied() else { continue };`. `run_with` passes `mesh: None` pieces to `recenter` (its `placed` list); `piece(false)` is that list.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p sherd-core --test session 2>&1 | tail -8`
Expected: `5 passed`.

If `reassembling_with_no_constraints…` differs in `poses` only: `run_with` recentres with pieces whose `mesh` is `None` — check `piece(false)` is what reaches `recenter`. If `an_accepted_probable_join…` fails on `placed.transform`, the constraint file's pose convention is not rows; read `MustJoin`'s deserialiser in `assembly/constraints.rs` and write the test's JSON in its convention — do not change the engine.

- [ ] **Step 6: Commit**

```bash
cargo fmt -p sherd-core
git add crates/sherd-core
git commit -m "M1.3: reassemble -- R §8 over a saved match under other constraints, laid out as a full run would lay it out"
```

---

### Task 4: `refine_poses` and `write_reviewed` — what a review session does after reassembling

**Files:**
- Modify: `crates/sherd-core/src/session.rs`, `crates/sherd-core/src/pipeline.rs` (visibility), `crates/sherd-core/tests/session.rs`

**Interfaces:**
- Consumes: `pipeline::{refine, write_outputs, Finished}` (`refine` becomes `pub(crate)`), `Reassembled`, `MatchState`.
- Produces:
  - `session::refine_poses(engine: Engine<'_>, fragments: &[Fragment], groups: &[Vec<FragId>], poses: &[Matrix4<f64>], used: &[(FragId, FragId)], params: &Params, budget: Budget, watch: &Watch) -> Result<Vec<Matrix4<f64>>>` — R §9 over **the groups it is given** (pass only the unrefined ones), then R §8.2's recentring is the caller's
  - `session::write_reviewed(out_dir: &Path, input: &Path, fragments: &[Fragment], state: &MatchState, done: &Reassembled, poses: &[Matrix4<f64>], options: &RunOptions) -> Result<Vec<PathBuf>>`

Task 5 adds the `watch` plumbing inside `refine`; here `refine_poses` accepts `watch` and ignores it, so the signature does not change later.

- [ ] **Step 1: Write the failing test**

Append to `crates/sherd-core/tests/session.rs`:

```rust
#[test]
fn a_reviewed_assembly_is_written_by_the_runs_own_writers() {
    let params = Params { tiers: Some(Thresholds::default()), ..Params::default() };
    let (_, path) = slab_run("reviewed", params);
    let state = MatchState::load(&path).unwrap();
    let chosen = *state.candidates.iter().find(|c| c.accepted).unwrap();
    let m = chosen.transform;
    let rows: Vec<String> = (0..4)
        .map(|r| format!("[{:?},{:?},{:?},{:?}]", m[(r, 0)], m[(r, 1)], m[(r, 2)], m[(r, 3)]))
        .collect();
    let json = format!(
        r#"{{"version":1,"must_join":[{{"a":"pieceA","b":"pieceB","pose":[{}]}}]}}"#,
        rows.join(",")
    );
    let frags = fragments(&state.params);
    let parsed: Constraints = serde_json::from_str(&json).unwrap();
    let done = session::reassemble(Engine::REFERENCE, &frags, &state, Some(&parsed)).unwrap();

    let refined = session::refine_poses(
        Engine::REFERENCE,
        &frags,
        &done.assembly.groups,
        &done.assembly.poses,
        &done.used,
        &state.params,
        Budget::default_for_machine(),
        &Watch::default(),
    )
    .expect("R §9 runs");
    assert_eq!(refined.len(), frags.len());

    let out = scratch("reviewed-out");
    let options = RunOptions {
        params: state.params,
        preview: false,
        write_meshes: false,
        ..RunOptions::default()
    };
    let written =
        session::write_reviewed(&out, &slab(), &frags, &state, &done, &done.poses, &options)
            .expect("the writers run");
    for file in ["transforms.json", "report.json", "report.md", "transforms.csv", "joins.csv"] {
        assert!(written.iter().any(|p| p.ends_with(file)), "{file} is written");
    }
    let report = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(report.contains("## Constraints"), "the reviewer's decision is in the report");
    assert!(report.contains("pieceA"));
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `cargo test -p sherd-core --test session a_reviewed 2>&1 | tail -5`
Expected: compile error — `refine_poses`, `write_reviewed` do not exist.

- [ ] **Step 3: Implement**

In `pipeline.rs` make `fn refine` `pub(crate)`. Append to `session.rs` (add `std::path::PathBuf`, `crate::pipeline::RunOptions`, `crate::report::Timings` to the imports):

```rust
/// R §9's full-resolution refinement over `groups` — every group of the assembly, or only the
/// ones a review left unrefined (A §8.4). Reads the source scans. The result is not recentred.
///
/// # Errors
///
/// A source scan that cannot be read.
#[allow(clippy::too_many_arguments, reason = "R §9's inputs, one argument each")]
pub fn refine_poses(
    engine: Engine<'_>,
    fragments: &[Fragment],
    groups: &[Vec<FragId>],
    poses: &[Matrix4<f64>],
    used: &[(FragId, FragId)],
    params: &Params,
    budget: Budget,
    watch: &Watch,
) -> Result<Vec<Matrix4<f64>>> {
    let _ = watch; // progress arrives with task M1.5
    pipeline::refine(engine, fragments, groups, poses, used, params, budget)
}

/// R §11's writers and `export/` over a reviewed assembly: the files `sherd-refit-rs run` writes,
/// with the reviewer's decisions in `report.md`'s `## Constraints` (A §9.1).
///
/// `poses` are the caller's — recentred, refined or not — and `options` carries the output
/// switches (`write_meshes`, `placed_all`, `merged_meshes`, `viewer`, `preview`) and the backend
/// label. `timings` and `memory` are empty: this is not a run.
///
/// # Errors
///
/// Whatever a writer fails on.
pub fn write_reviewed(
    out_dir: &Path,
    input: &Path,
    fragments: &[Fragment],
    state: &MatchState,
    done: &Reassembled,
    poses: &[Matrix4<f64>],
    options: &RunOptions,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir).map_err(|e| Error::write(out_dir, e))?;
    let timings = Timings::default();
    pipeline::write_outputs(
        out_dir,
        &pipeline::Finished {
            input,
            fragments,
            names: &state.names,
            candidates: &done.candidates,
            assembly: &done.assembly,
            tiers: done.tiers.as_ref(),
            constraints: done.constraints.as_ref(),
            review: None,
            objects: done.objects.as_ref(),
            poses,
            thickness: state.thickness,
            timings: &timings,
            memory: None,
        },
        options,
        Vec::new(),
    )
}
```

`write_outputs` takes its thresholds from `options.params`; callers pass `params: state.params`, as the test does.

- [ ] **Step 4: Run the test**

Run: `cargo test -p sherd-core --test session a_reviewed 2>&1 | tail -5`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
cargo fmt -p sherd-core
git add crates/sherd-core
git commit -m "M1.4: a reviewed assembly is refined and written by the run's own stages"
```

---

### Task 5: progress in the two stages that stood still — `tiers` and `refine`

**Files:**
- Modify: `crates/sherd-core/src/tiers.rs`, `crates/sherd-core/src/pipeline.rs`, `crates/sherd-core/src/session.rs`, `crates/sherd-core/tests/session.rs`

**Interfaces:**
- Consumes: `progress::Watch`.
- Produces: `tiers::classify_watched(engine, fragments, candidates, params, thresholds, watch: &Watch) -> TierReport`; `tiers::classify` keeps its signature and calls it with `&Watch::default()`. Stage names reported: `"tiers"`, `"refine"` (beside today's `"preprocess"`, `"matching"`).

- [ ] **Step 1: Write the failing test**

Append to `crates/sherd-core/tests/session.rs`:

```rust
#[derive(Debug, Default)]
struct Recorder(std::sync::Mutex<Vec<(String, usize, usize)>>);

impl sherd_core::progress::Progress for Recorder {
    fn advance(&self, stage: &str, done: usize, total: usize) {
        self.0.lock().unwrap().push((stage.to_owned(), done, total));
    }
}

#[test]
fn the_tier_pass_and_the_refinement_report_their_progress() {
    let recorder = std::sync::Arc::new(Recorder::default());
    let out = scratch("progress");
    let options = RunOptions {
        // `rival_refused: false` is the one conjunct the slab needs off to confirm its join
        // (`run_cli.rs`, `AMBIGUOUS`) — and R §9 only runs when something was assembled.
        params: Params {
            tiers: Some(Thresholds { rival_refused: false, ..Thresholds::default() }),
            ..Params::default()
        },
        preview: false,
        write_meshes: false,
        watch: Watch { cancel: None, progress: Some(recorder.clone()) },
        ..RunOptions::default()
    };
    pipeline::run(&slab(), &out, &options).expect("the slab runs");
    let seen = recorder.0.lock().unwrap();
    for stage in ["preprocess", "matching", "tiers", "refine"] {
        let last = seen.iter().filter(|(s, ..)| s == stage).max_by_key(|(_, done, _)| *done);
        let (_, done, total) = last.unwrap_or_else(|| panic!("`{stage}` reported nothing"));
        assert!(*total > 0 && done == total, "`{stage}` ended at {done} of {total}");
    }
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `cargo test -p sherd-core --test session the_tier_pass 2>&1 | tail -5`
Expected: FAIL — `` `tiers` reported nothing ``.

- [ ] **Step 3: `tiers`**

In `crates/sherd-core/src/tiers.rs`:
- rename the body of `classify` to `classify_watched` with one more parameter `watch: &Watch`, documented `/// [`classify`] under D §5's progress callback (A §3.4): one report per candidate probed.`; `classify` becomes a one-line call with `&Watch::default()`.
- `classify` calls `probe(engine, fragments, candidates, params)`; give `probe` the same `watch` parameter. Inside it, the probes of the accepted candidates are computed by a `par_iter` (≈ lines 1045–1070). Before that `par_iter` add

```rust
    let probed = std::sync::atomic::AtomicUsize::new(0);
    let to_probe = candidates.iter().filter(|c| c.accepted).count();
```

  and as the **last statement inside the closure that computes one accepted candidate's `Probes`** (not in the branch that returns `None` for a candidate that is not probed):

```rust
        watch.advance(
            "tiers",
            probed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
            to_probe,
        );
```

  If `probe` has two such `par_iter`s over the accepted candidates, the counter goes in the one that computes the resamples and redraws (the expensive one); `to_probe` must equal the number of times that closure runs, so derive it from the same filter that closure uses.
- In `pipeline.rs`, `tier_pass` and `agree_pass` call `classify`: give `tier_pass` a `watch: &Watch` parameter, call `classify_watched(…, watch)`, and pass `&options.watch` from both call sites in `run_with`. `agree_pass` already receives `options`; use `&options.watch` for its `classify` call too.

- [ ] **Step 4: `refine`**

In `pipeline.rs`, `refine` (≈ line 1616) calls `fracture_clouds`, whose `par_iter` over fragments reads one source scan per job — that is R §9's cost. Add `watch: &Watch` to both signatures. In `fracture_clouds`, before the `par_iter`:

```rust
    let built = AtomicUsize::new(0);
    let to_build = in_group.iter().filter(|&&inside| inside).count();
```

(use the name the function already has for its "is this fragment in a group" flags), and as the last statement of the closure's branch that builds a cloud:

```rust
            watch.advance("refine", built.fetch_add(1, Ordering::Relaxed) + 1, to_build);
```

`run_with` passes `&options.watch`; in `session.rs`, `refine_poses` drops its `let _ = watch;` line and passes `watch` through.

- [ ] **Step 5: Run the test, then the whole session file**

Run: `cargo test -p sherd-core --test session 2>&1 | tail -5`
Expected: `7 passed`.

- [ ] **Step 6: Commit**

```bash
cargo fmt -p sherd-core
git add crates/sherd-core
git commit -m "M1.5: the tier pass and R §9 report their progress -- the two stages of every run where the bar stood still"
```

---

### Task 6: discovery with exclusions

**Files:**
- Modify: `crates/sherd-core/src/collection.rs`, `crates/sherd-core/src/pipeline.rs`

**Interfaces:**
- Produces: `collection::discover_excluding(dir: impl AsRef<Path>, excluded: &BTreeSet<String>) -> Result<Vec<Entry>>`; `RunOptions::excluded: BTreeSet<String>` (default empty).

- [ ] **Step 1: Write the failing test**

In `crates/sherd-core/src/collection.rs`, inside the existing `#[cfg(test)] mod tests`:

```rust
    /// A §5.1: an excluded scan is left out by **name**, after the names are made, so that
    /// leaving one out renames nothing else.
    #[test]
    fn an_excluded_fragment_is_left_out_and_the_others_keep_their_names() {
        let dir = std::env::temp_dir().join(format!("sherd-excluding-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["FY234009.ply", "FY249010.ply", "FY234011.ply"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        let all: Vec<String> = super::discover(&dir).unwrap().into_iter().map(|e| e.name).collect();
        let excluded = std::collections::BTreeSet::from(["FY234009".to_owned()]);
        let kept: Vec<String> =
            super::discover_excluding(&dir, &excluded).unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(all.len(), 3);
        assert_eq!(kept, all.into_iter().filter(|n| n != "FY234009").collect::<Vec<_>>());
        std::fs::remove_dir_all(&dir).ok();
    }
```

- [ ] **Step 2: Run it to see it fail**

Run: `cargo test -p sherd-core --lib collection 2>&1 | tail -5`
Expected: compile error — `discover_excluding` does not exist.

- [ ] **Step 3: Implement**

In `collection.rs`, below `discover`:

```rust
/// [`discover`] without the fragments named in `excluded` (A §5.1).
///
/// The names are made over the **whole** directory first and the exclusion is applied to them, so
/// leaving a scan out never renames another — R §2 disambiguates equal stems, and a name that
/// moved would orphan every decision and every cache keyed by it. A name in `excluded` that is
/// not in the directory is ignored: the file it named may simply be gone.
///
/// # Errors
///
/// [`discover`]'s.
pub fn discover_excluding(
    dir: impl AsRef<Path>,
    excluded: &std::collections::BTreeSet<String>,
) -> Result<Vec<Entry>> {
    Ok(discover(dir)?.into_iter().filter(|entry| !excluded.contains(&entry.name)).collect())
}
```

In `pipeline.rs`: `RunOptions` gains

```rust
    /// Fragments left out of the run by name (A §5.1); empty — the default — is every scan of the
    /// input directory, which is all the CLI ever asks for.
    pub excluded: BTreeSet<String>,
```

with `excluded: BTreeSet::new(),` in `Default`, and `run_with`'s first line becomes
`let entries = collection::discover_excluding(input, &options.excluded)?;`.

- [ ] **Step 4: Run the test**

Run: `cargo test -p sherd-core --lib collection 2>&1 | tail -5`
Expected: all `collection` tests pass.

- [ ] **Step 5: Commit**

```bash
cargo fmt -p sherd-core
git add crates/sherd-core
git commit -m "M1.6: a run can leave fragments out by name, and leaving one out renames no other"
```

---

### Task 7: `sherd-backend` — `--backend` resolution where the app can link it

A move. `crates/sherd-cli/src/gpu.rs` holds `info_lines`, `selftest_lines`, `Resolved` (with its `impl`), `resolve` (two `cfg` variants) and `check` (two `cfg` variants). Everything but `check` moves.

**Files:**
- Create: `crates/sherd-backend/Cargo.toml`, `crates/sherd-backend/src/lib.rs`
- Modify: `Cargo.toml` (workspace), `crates/sherd-cli/Cargo.toml`, `crates/sherd-cli/src/gpu.rs`

**Interfaces:**
- Produces: `sherd_backend::{resolve(backend: Backend, adapter: Option<&str>, memory: Option<f64>) -> anyhow::Result<Resolved>, Resolved { backend, engine, reason, adapter, executor (gpu only) }, info_lines() -> Vec<String>, selftest_lines(adapter: Option<&str>) -> Vec<String>}` — all `pub`, bodies unchanged.

- [ ] **Step 1: The crate**

`crates/sherd-backend/Cargo.toml`:

```toml
[package]
name = "sherd-backend"
description = "What `--backend auto|cpu|gpu` resolves to: shared by the CLI and the desktop app"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
repository.workspace = true
publish.workspace = true

# The same optional feature `sherd-cli` has (D §2): without it there is no wgpu in the tree, and
# `resolve` still answers — `cpu`, or an error for `--backend gpu`.
[features]
default = ["gpu"]
gpu = ["dep:sherd-gpu"]

[dependencies]
sherd-core.workspace = true
sherd-gpu = { workspace = true, optional = true }
anyhow.workspace = true
tracing.workspace = true

[lints]
workspace = true
```

Workspace `Cargo.toml`: add `"crates/sherd-backend"` to `members` and, under `# --- workspace members`, `sherd-backend = { path = "crates/sherd-backend", default-features = false }`.

- [ ] **Step 2: Move the code**

`git mv` is not possible for a partial move; create `crates/sherd-backend/src/lib.rs` with `gpu.rs`'s module doc comment (reworded from "What the CLI does about…" to "What a front end does about `--backend` and `--gpu-adapter` (D §6.8, D §9): the CLI and the desktop app (A §3.6) resolve it here, identically."), its `use` lines, and `info_lines`, `selftest_lines`, `Resolved` + `impl Resolved`, and both `resolve` variants — bodies and doc comments unchanged, every `pub(crate)` → `pub`. Drop `tracing` from the manifest if the moved code does not use it (`cargo check` will warn of an unused dependency only under `-D warnings`; check with `grep -n tracing crates/sherd-backend/src/lib.rs`).

`crates/sherd-cli/src/gpu.rs` keeps its two `check` variants and the imports they need, and starts with:

```rust
//! `gpu-check`'s adapter between `clap` and `sherd-gpu` (D §10.4 layer 3). `--backend`'s
//! resolution lives in `sherd-backend`, where the desktop app links it too (A §3.6).

pub(crate) use sherd_backend::{info_lines, resolve, selftest_lines};
```

(`Resolved` is re-exported too only if `main.rs` names the type; `grep -n 'gpu::Resolved' crates/sherd-cli/src/main.rs`.)

`crates/sherd-cli/Cargo.toml`:

```toml
[features]
default = ["gpu"]
gpu = ["dep:sherd-gpu", "sherd-backend/gpu"]

[dependencies]
sherd-backend.workspace = true
```

- [ ] **Step 3: Both shapes compile, and the CLI's own tests of `resolve` pass**

Run: `cargo check -p sherd-cli && cargo check -p sherd-cli --no-default-features`
Expected: no errors in either.

Run: `cargo test -p sherd-cli --bin sherd-refit-rs 2>&1 | tail -5`
Expected: `test result: ok.` (these include the two tests at the bottom of `main.rs` that call `gpu::resolve`).

- [ ] **Step 4: Commit**

```bash
cargo fmt -p sherd-backend -p sherd-cli
git add Cargo.toml Cargo.lock crates/sherd-backend crates/sherd-cli
git commit -m "M1.7: --backend is resolved in sherd-backend, which the desktop app can link and the CPU-only CLI still builds against"
```

---

### Task 8: the milestone gate, once

Nothing is written here except fixes for what the gate finds.

**Files:** whatever the gate names.

- [ ] **Step 1: Format and lints, both shapes** (CI's `check` job, `.github/workflows/rust.yml`)

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings
```

Expected: clean. `--locked` fails if `Cargo.lock` was not committed in Task 7 — commit it. Typical findings: a missing doc comment on a `pub(crate)` item, `unreachable_pub`, `too_many_arguments` on `assemble_stage` (already allowed with a reason) or `refine_poses`.

- [ ] **Step 2: The byte-for-byte gate and this milestone's tests**

```bash
cargo test -p sherd-cli 2>&1 | tail -5
cargo test -p sherd-core --test session 2>&1 | tail -5
cargo test -p sherd-core --lib 2>&1 | tail -5
```

Expected: all `ok`. `cargo test --workspace` is **not** run: `sherd-parity` and `sherd-gpu` were not touched, and CI runs the whole suite on four platforms.

- [ ] **Step 3: Parity, one stage, as a smoke test of the refactor**

`parity --stage all` over eight dumps is CI's job. Locally, the one dump that is in the repository:

```bash
cargo run -q -p sherd-cli -- parity --fixtures fixtures/slab/dump --input fixtures/slab/input \
    --stage assembly --stage outputs
```

Expected: zero failures on both stages — the two stages whose code Task 1 moved.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A crates Cargo.toml Cargo.lock
git commit -m "M1.8: the milestone's gate -- fmt, clippy in both shapes, the CLI's byte-for-byte tests, parity on the slab"
```

(skip if Steps 1–3 found nothing).

---

## What the next plan can rely on

| need of milestone 2 (`sherd-app-core`) | provided by |
|---|---|
| run a collection, keep its match | `pipeline::run_with` with `RunOptions { match_state: Some(..), excluded, write_meshes: false, preview: false, watch, .. }` |
| open a review session | `session::load_fragments` + `session::MatchState::load` |
| one decision → a draft assembly | `session::reassemble` with a `Constraints` built from the decisions |
| «Уточнить позы» | `session::refine_poses` over the unrefined groups, then `assembly::recenter` |
| Export | `session::write_reviewed` |
| progress for every stage of an app run | `Watch` — `preprocess`, `matching`, `tiers`, `refine` |
| `auto` / `cpu` / `gpu`, adapters, self-test | `sherd_backend::{resolve, info_lines, selftest_lines}` |

Not provided here, by design: `PairDetail`'s seam data (it reads `review.rs`'s internals and is planned with the review screen, milestone 5), per-file progress of the mesh writers (milestone 6, with Export), display meshes and thumbnails (milestone 2's `Prepare`, over the public `export::scene::display_mesh` and `render`; A §3.5).
