# Desktop App, Milestone 2 — `sherd-app-core`: workspace, protocol, worker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything the desktop app does that is not a window: a workspace on disk, a worker process that prepares a collection and runs it while reporting progress and obeying cancel, and the host side that spawns it and keeps the run's files — proven end to end, headless, on `fixtures/slab`.

**Architecture:** A new crate `crates/sherd-app-core` with no Tauri in it. `protocol` is the JSON-lines contract; `worker::serve(stdin, stdout)` is the engine role of the app's one binary (A §2.1) and gets a tiny bin target of its own so tests can spawn it; `host` spawns a worker, turns its stdout into events and its stderr into `engine.log`, and does the run's bookkeeping (`run.json`, `assembly.json`). `workspace`, `run`, `snapshot`, `decisions` are the on-disk model of A §4 and A §8.1. One addition to `sherd-core`: a single fragment as GLB.

**Tech Stack:** Rust 2024 (MSRV 1.89 — `std::fs::File::try_lock` is used for the workspace lock), `serde`/`serde_json`, `thiserror`, `tracing` + `tracing-subscriber`, `rayon`, `chrono 0.4.45` (new: local time for run ids). Spec: `docs/superpowers/specs/2026-09-20-desktop-app-design.md` (`A §n`).

## Global Constraints

- **Toolchain.** Bare `cargo` on this machine is Homebrew's 1.88 and fails the MSRV. Start **every** shell that runs cargo with
  `export PATH="$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin:$PATH"`.
- **Verification budget (A §11).** Run **only** the commands a task lists, plus `cargo check -p <crate>` while editing. No `cargo test --workspace`, no clippy, no release builds inside a task; Task 9 is the gate. The only collection anything runs on is `fixtures/slab/input` (fragments `pieceA`, `pieceB`). `cargo test -p sherd-cli --test run_cli` takes 4–7 minutes: run it in the background, and only where a task lists it.
- **The CLI's output stays byte for byte.** Task 1 is the only task that touches `sherd-core`.
- **Workspace lints** apply to the new crate (`[lints] workspace = true`): every `pub`/`pub(crate)` item documented, `pub(crate)` for what is not reachable from outside, no `todo!`/`unimplemented!`/`dbg!`/`unsafe`. Doc comments say **why** and cite `A §n`, `D §n`, `R §n`.
- **No panics on data that came from a file or a pipe** outside tests: return an error.
- **Ownership rule (A §2.1).** The worker writes engine artefacts (`cache/`, `fragments/`, everything `run_with` writes into the run folder, `candidates.json`). The host writes workspace state (`sherd-workspace.json`, `run.json`, `assembly.json`, `engine.log`). Every state file goes through `atomic::write_json`.
- **Tests use the CPU** (`BackendChoice::Cpu`): CI runners have no GPU.
- **New dependencies** are pinned exactly in the root `Cargo.toml` `[workspace.dependencies]` with a one-line reason, like every dependency there.
- **Do not touch** `sherd_refit/`, `crates/sherd-parity/`, `fixtures/`, `crates/sherd-core/src/export/viewer.html`.
- **Commits:** branch `desktop-app`, subject `M2.<task>: <sentence>`, trailer as your own system instructions give it.
- **The slab under the shipped rule** confirms nothing: with tiers on its one join is *probable* and no group is assembled (`crates/sherd-cli/tests/run_cli.rs`, const `AMBIGUOUS`). Tests below expect exactly that.

## What milestone 1 provides (real signatures, tree at `1906e46`)

```rust
// sherd_core::pipeline
pub fn run_with(input: &Path, out_dir: &Path, options: &RunOptions, engine: Engine<'_>) -> Result<RunSummary>;
pub fn set_threads(threads: usize) -> std::result::Result<(), String>;
pub struct RunOptions { pub target_faces: usize, pub params: Params, pub preview: bool, pub refine: bool,
    pub write_meshes: bool, pub cache: Option<PathBuf>, pub workers: usize, pub backend: Backend,
    pub adapter: Option<String>, pub memory: Budget, pub watch: Watch, pub review_images: bool,
    pub match_state: Option<PathBuf>, pub excluded: BTreeSet<String>, /* … */ }   // Clone + Default
pub struct RunSummary { pub names: Vec<String>, pub poses: Vec<Matrix4<f64>>, pub groups: Vec<Vec<FragId>>,
    pub used: Vec<(FragId, FragId)>, pub candidates: Vec<Candidate>, pub pairs: usize,
    pub skipped_pairs: usize, pub timings: Timings, /* … */ }
// sherd_core::session
pub fn load_fragments(entries: &[Entry], target_faces: usize, cache_dir: Option<&Path>, budget: Budget, seed: u64, watch: &Watch) -> Result<Vec<Fragment>>;
pub struct MatchState { pub names, pub params, pub candidates: Vec<Candidate>, pub tiers: Option<TierReport>, /* … */ }  // ::load(&Path)
// sherd_core::progress
pub struct Watch { pub cancel: Option<Cancel>, pub progress: Option<Arc<dyn Progress>> }
pub trait Progress: Send + Sync + Debug { fn advance(&self, stage: &str, done: usize, total: usize); }
// stages reported: "preprocess", "matching", "tiers", "refine"
// sherd_core::collection
pub fn discover(dir) -> Result<Vec<Entry>>;  pub fn discover_excluding(dir, &BTreeSet<String>) -> Result<Vec<Entry>>;  // Entry { path, name }
// sherd_core::export::scene
pub fn face_targets(areas: &[f64], total: usize) -> Vec<usize>;  pub fn display_mesh(mesh: &Mesh, target: usize) -> DisplayMesh;  pub const DEFAULT_FACES: usize = 600_000;
// sherd_core::io
pub fn read_mesh(path: impl AsRef<Path>) -> Result<Mesh>;      // Mesh { v: Vec<[f64;3]>, f: Vec<[u32;3]>, colors: Option<Vec<[u8;3]>> }
// sherd_core::render
pub struct Splat { pub points, pub normals: Vec<[f64;3]>, pub paint: Paint }  pub enum Paint { Uniform([f64;3]), PerPoint(Vec<[f64;3]>) }
pub fn principal_views(points: &[[f64;3]]) -> [View; 4];  pub fn render_views(meshes: &[Splat], views: &[View], width: usize, height: usize) -> Rgb;  // Rgb::write_png(path)
// sherd_core::memory
pub struct Budget;  Budget::default_for_machine(), Budget::gigabytes(f64);  pub struct MemorySemaphore;  ::new(Budget), .acquire(u64) -> Permit;  pub fn scan_faces(&Path) -> Option<u64>;  pub fn reservation(faces: u64) -> u64;
// sherd_core::report
pub struct FragmentStats { name, faces, orig_faces, orig_vertices, thickness, thickness_mode, resolution, watertight, extent, area, fracture_area_fraction }  // Serialize; FragmentStats::of(&Fragment)
// sherd_backend
pub fn resolve(backend: Backend, adapter: Option<&str>, memory: Option<f64>) -> anyhow::Result<Resolved>;  // Resolved { backend, engine: Engine<'static>, reason, adapter, .. }
pub fn info_lines() -> Vec<String>;  pub fn selftest_lines(adapter: Option<&str>) -> Vec<String>;
// sherd_core: CORE_VERSION, GIT_COMMIT, ALGO_REF;  Backend { Auto, Cpu, Gpu } (FromStr, as_str; no serde)
```

The CLI's defaults — which the app's «Стандарт» preset must equal — are `Params { tiers: Some(Thresholds::default()), objects: Some(ObjectParams::default()), ..Params::default() }` (`crates/sherd-cli/src/main.rs`, `fn params`).

## File Structure

| file | responsibility |
|---|---|
| `crates/sherd-core/src/export/scene.rs` (modify) | `fragment_glb`: one display mesh as a GLB |
| `crates/sherd-app-core/Cargo.toml`, `src/lib.rs` | the crate; `AppError` |
| `src/atomic.rs` | `write_json`, `read_json`: temp file + rename |
| `src/workspace.rs` | `sherd-workspace.json`, the lock, the folder's paths |
| `src/snapshot.rs` | what the input folder held, and how two snapshots differ (A §4 «Stale») |
| `src/run.rs` | `run.json`, run ids, the lifecycle, interrupted runs |
| `src/decisions.rs` | `decisions.json` and decisions → `Constraints` (A §8.1–8.2) |
| `src/protocol.rs` | `Job`, `Request`, `Event`, `RunSpec` and the DTOs — the whole wire contract |
| `src/worker/mod.rs` | `serve`: the loop, the emitter, cancel on request and on EOF, panics → `Failed` |
| `src/worker/prepare.rs` | the `Prepare` job |
| `src/worker/run.rs` | the `Run` job and `candidates.json` |
| `src/bin/sherd-engine-worker.rs` | `fn main` = `worker::serve(stdin, stdout)`; what tests spawn |
| `src/host.rs` | spawn a worker, its events, `engine.log`, the run's bookkeeping |
| `tests/worker_e2e.rs` | Prepare → Run on the slab; cancel; host goes away |

---

### Task 1: `fragment_glb` — one display mesh as a GLB

**Files:**
- Modify: `crates/sherd-core/src/export/scene.rs`

**Interfaces:**
- Produces: `pub fn fragment_glb(name: &str, mesh: &DisplayMesh) -> Vec<u8>` — a complete glTF 2.0 binary with one scene, one node named `name` (identity transform) and one mesh, in the **original file's coordinates**; `COLOR_0` and material 0 when `mesh.colors` is `Some`, the plain-clay material otherwise; an empty `Vec` is never returned — a mesh with no faces gives a scene with a node and no mesh.

- [ ] **Step 1: Write the failing test**

In `scene.rs`'s `mod tests`, beside `the_groups_stand_apart_and_the_file_reads_back_as_gltf`:

```rust
    /// A §3.5: the app keeps one display mesh per fragment and lays a run over them as matrices,
    /// so a fragment is a file of its own, in its scan's coordinates.
    #[test]
    fn one_fragment_reads_back_as_gltf_in_its_own_coordinates() {
        let mesh = display_mesh(&tetra_of(2.0), MAX_FACES);
        let bytes = fragment_glb("FY234001", &mesh);
        let gltf = gltf::Gltf::from_slice(&bytes).expect("a valid GLB");
        let node = gltf.nodes().next().expect("one node");
        assert_eq!(node.name(), Some("FY234001"));
        assert_eq!(node.transform().matrix(), gltf::scene::Transform::Decomposed {
            translation: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3],
        }.matrix());
        let primitive = node.mesh().expect("a mesh").primitives().next().expect("a primitive");
        let count = primitive.get(&gltf::Semantic::Positions).expect("positions").count();
        assert_eq!(count, mesh.positions.len());
        assert_eq!(primitive.get(&gltf::Semantic::Colors(0)).is_some(), mesh.colors.is_some());
    }
```

- [ ] **Step 2: Run it to see it fail**

Run: `cargo test -p sherd-core --lib export::scene 2>&1 | tail -5`
Expected: compile error — `fragment_glb` not found.

- [ ] **Step 3: Implement**

`mesh_entries`' loop body builds one mesh entry. Move it, unchanged, into

```rust
/// One glTF mesh, its arrays appended to `buffers`. Material 0 is the scan's colours, 1 the plain
/// clay a colourless scan is drawn in.
fn mesh_entry(name: &str, mesh: &DisplayMesh, buffers: &mut Buffers) -> Value
```

(from `let (lo, hi) = extent(…)` to the `json!({ "name": …, "primitives": […] })` it pushes, with `ex.names[n]` → `name`), and have `mesh_entries` call `entries.push(mesh_entry(&ex.names[n], mesh, buffers));`. Then:

```rust
/// One fragment's display mesh as a GLB of its own, in its original file's coordinates (A §3.5).
///
/// `scene.glb` bakes a run into the file: a node per fragment carrying that run's matrix. The app
/// keeps the meshes and the runs apart — the meshes once per workspace, a run as matrices over
/// them — so that a draft reassembly (A §8.4) moves meshes and regenerates nothing.
pub fn fragment_glb(name: &str, mesh: &DisplayMesh) -> Vec<u8> {
    let mut buffers = Buffers::default();
    let mut node = json!({ "name": name });
    let mut meshes = Vec::new();
    if !mesh.faces.is_empty() {
        meshes.push(mesh_entry(name, mesh, &mut buffers));
        node["mesh"] = json!(0);
    }
    // glTF indexes materials by position: a colourless mesh asks for material 1, so both exist.
    let materials =
        vec![material("scan colours", [1.0, 1.0, 1.0]), material("clay", [0.52, 0.33, 0.22])];
    let bin_length = buffers.bin.len();
    let mut doc = json!({
        "asset": { "version": "2.0", "generator": format!("sherd-refit {CORE_VERSION} ({GIT_COMMIT})") },
        "scene": 0,
        "scenes": [{ "name": name, "nodes": [0] }],
        "nodes": [node],
        "materials": materials,
    });
    if !meshes.is_empty() {
        doc["meshes"] = json!(meshes);
        doc["accessors"] = json!(buffers.accessors);
        doc["bufferViews"] = json!(buffers.views);
        doc["buffers"] = json!([{ "byteLength": bin_length }]);
    }
    container(&doc, buffers.bin)
}
```

- [ ] **Step 4: Run the scene tests**

Run: `cargo test -p sherd-core --lib export::scene 2>&1 | tail -5`
Expected: all pass, the new one included. (`scene.glb`'s own bytes are covered by the existing tests and by Task 9's `run_cli`.)

- [ ] **Step 5: Commit**

```bash
cargo fmt -p sherd-core
git add crates/sherd-core/src/export/scene.rs
git commit -m "M2.1: one fragment's display mesh is a GLB of its own -- the app keeps meshes and runs apart"
```

---

### Task 2: the crate, atomic files, and the workspace

**Files:**
- Create: `crates/sherd-app-core/Cargo.toml`, `src/lib.rs`, `src/atomic.rs`, `src/workspace.rs`
- Modify: root `Cargo.toml`

**Interfaces:**
- Produces:
  - `sherd_app_core::{AppError, Result}`
  - `atomic::{write_json<T: Serialize>(path: &Path, value: &T) -> Result<()>, read_json<T: DeserializeOwned>(path: &Path) -> Result<T>}`
  - `workspace::{Workspace, WorkspaceFile, InputRef, WORKSPACE_FILE, WORKSPACE_VERSION}` with `Workspace::{create(root), open(root), root(), file(), input() -> Option<PathBuf>, set_input(&mut self, dir), set_excluded(&mut self, name, excluded: bool), set_last_spec(&mut self, spec: serde_json::Value), cache_dir(), fragments_dir(), runs_dir(), run_dir(id), exports_dir()}`

`last_spec` is a `serde_json::Value` here so that this task does not depend on Task 5's `RunSpec`; Task 5 leaves it a `Value` — the workspace does not need to understand it.

- [ ] **Step 1: The manifest**

Root `Cargo.toml`: add `"crates/sherd-app-core"` to `members`; under `# --- storage, fixtures, output` add

```toml
chrono = { version = "=0.4.45", default-features = false, features = [
    "clock",
    "std",
] } # local wall-clock time for the app's run ids and run.json; nothing in the core reads a clock
```

and under `# --- workspace members`: `sherd-backend` is already there; add `sherd-app-core = { path = "crates/sherd-app-core" }`.

`crates/sherd-app-core/Cargo.toml`:

```toml
[package]
name = "sherd-app-core"
description = "The desktop app without its window: workspace, worker protocol, run bookkeeping"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
repository.workspace = true
publish.workspace = true

# The same optional feature the CLI has (D §2), passed through to where `--backend` is resolved.
[features]
default = ["gpu"]
gpu = ["sherd-backend/gpu"]

[[bin]]
name = "sherd-engine-worker"
path = "src/bin/sherd-engine-worker.rs"

[dependencies]
sherd-core.workspace = true
sherd-backend.workspace = true
anyhow.workspace = true
chrono.workspace = true
rayon.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true

[lints]
workspace = true
```

Create `src/bin/sherd-engine-worker.rs` now as a stub the crate needs in order to build, and fill it in Task 5:

```rust
//! The engine role of the app's binary, as a binary of its own (A §2.1): what the headless tests
//! spawn, and what the Tauri shell reproduces by running itself with `--engine-worker`.

fn main() {}
```

- [ ] **Step 2: Write the failing tests**

`src/workspace.rs` ends with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-ws-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_workspace_is_a_folder_that_reopens_as_it_was_left() {
        let root = scratch("reopen").join("karas");
        let scans = scratch("reopen-scans");
        {
            let mut ws = Workspace::create(&root).unwrap();
            ws.set_input(&scans).unwrap();
            ws.set_excluded("FY234009", true).unwrap();
        }
        let ws = Workspace::open(&root).unwrap();
        assert_eq!(ws.input().unwrap(), scans.canonicalize().unwrap());
        assert!(ws.file().excluded.contains("FY234009"));
        assert_eq!(ws.run_dir("2026-09-20_1412"), root.join("runs").join("2026-09-20_1412"));
    }

    /// A §4: the input is found by its path relative to the workspace first, so that a workspace
    /// and its scans moved together to another disk still open.
    #[test]
    fn the_input_is_found_again_after_both_folders_moved_together() {
        let base = scratch("moved");
        let (root, scans) = (base.join("a/karas"), base.join("a/scans"));
        std::fs::create_dir_all(&scans).unwrap();
        {
            let mut ws = Workspace::create(&root).unwrap();
            ws.set_input(&scans).unwrap();
        }
        std::fs::rename(base.join("a"), base.join("b")).unwrap();
        let ws = Workspace::open(&base.join("b/karas")).unwrap();
        assert_eq!(ws.input().unwrap(), base.join("b/scans").canonicalize().unwrap());
    }

    #[test]
    fn a_second_open_of_the_same_workspace_is_refused() {
        let root = scratch("locked").join("karas");
        let _first = Workspace::create(&root).unwrap();
        assert!(matches!(Workspace::open(&root), Err(AppError::Locked { .. })));
    }

    #[test]
    fn a_folder_that_is_not_a_workspace_and_a_newer_workspace_are_both_refused() {
        let empty = scratch("not-one");
        assert!(matches!(Workspace::open(&empty), Err(AppError::NotAWorkspace { .. })));
        let newer = scratch("newer");
        std::fs::write(newer.join(WORKSPACE_FILE), r#"{"version":99,"excluded":[]}"#).unwrap();
        assert!(matches!(Workspace::open(&newer), Err(AppError::Version { found: 99, .. })));
    }
}
```

`src/atomic.rs` ends with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_replaced_whole_and_leaves_no_temporary_behind() {
        let dir = std::env::temp_dir().join(format!("sherd-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        write_json(&path, &vec![1, 2, 3]).unwrap();
        write_json(&path, &vec![4]).unwrap();
        assert_eq!(read_json::<Vec<u32>>(&path).unwrap(), vec![4]);
        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, [std::ffi::OsString::from("state.json")]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
```

- [ ] **Step 3: Implement**

`src/lib.rs`:

```rust
//! The desktop app without its window (A §2): the workspace on disk, the protocol between the
//! window's process and the engine's, the worker that speaks it, and the host that keeps a run's
//! files. Nothing here depends on Tauri, which is what lets all of it be tested headless.

pub mod atomic;
pub mod workspace;

use std::path::PathBuf;

/// What this crate fails with.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// A file or folder of the workspace could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The path.
        path: PathBuf,
        /// The OS's word.
        source: std::io::Error,
    },
    /// A state file does not parse.
    #[error("{path}: {source}")]
    Json {
        /// The file.
        path: PathBuf,
        /// serde's word.
        source: serde_json::Error,
    },
    /// The folder holds no `sherd-workspace.json`.
    #[error("{path} is not a sherd-refit workspace")]
    NotAWorkspace {
        /// The folder.
        path: PathBuf,
    },
    /// Another running app has the workspace open (A §10).
    #[error("{path} is open in another window of the app")]
    Locked {
        /// The workspace.
        path: PathBuf,
    },
    /// A state file written by a newer app.
    #[error("{path} was written by a newer version of the app (format {found}, this build reads {expected})")]
    Version {
        /// The file.
        path: PathBuf,
        /// Its version.
        found: u32,
        /// Ours.
        expected: u32,
    },
    /// The engine's own error.
    #[error(transparent)]
    Core(#[from] sherd_core::Error),
    /// The worker could not be started or spoken to.
    #[error("the engine process: {0}")]
    Worker(String),
}

impl AppError {
    /// An [`AppError::Io`] about `path`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io { path: path.into(), source }
    }
}

/// `Result` with [`AppError`].
pub type Result<T> = std::result::Result<T, AppError>;
```

`src/atomic.rs`:

```rust
//! State files are replaced whole or not at all (A §2.1): `decisions.json` is written on every
//! click of a review, and a crash between two clicks must leave the previous file, not half of
//! the next one.

use std::io::Write;
use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{AppError, Result};

/// Writes `value` as pretty JSON to a temporary neighbour of `path` and renames it over `path`.
///
/// # Errors
///
/// [`AppError::Io`] when the folder cannot be written.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_vec_pretty(value)
        .map_err(|source| AppError::Json { path: path.to_owned(), source })?;
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(format!(".{}.tmp", std::process::id()));
    let temporary = path.with_file_name(name);
    let written = (|| {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&json)?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)
    })();
    written.map_err(|source| {
        std::fs::remove_file(&temporary).ok();
        AppError::io(path, source)
    })
}

/// Reads a JSON state file.
///
/// # Errors
///
/// [`AppError::Io`] when it cannot be read, [`AppError::Json`] when it does not parse.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|source| AppError::io(path, source))?;
    serde_json::from_slice(&bytes).map_err(|source| AppError::Json { path: path.to_owned(), source })
}
```

`src/workspace.rs` (above the tests):

```rust
//! A workspace (A §4): a plain folder holding `sherd-workspace.json`, one collection, its cache,
//! its display meshes and the history of its runs. The scans themselves are linked, never copied
//! — 155 of them are 10 GB.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{AppError, Result, atomic};

/// The file that makes a folder a workspace.
pub const WORKSPACE_FILE: &str = "sherd-workspace.json";
/// The file an open workspace holds a lock on.
pub const LOCK_FILE: &str = "sherd-workspace.lock";
/// The format this build writes, and the newest it reads.
pub const WORKSPACE_VERSION: u32 = 1;

/// Where the scans are.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRef {
    /// As the user picked it.
    pub absolute: PathBuf,
    /// From the workspace folder, when there is such a path: tried first, so that a workspace and
    /// its scans moved together — another disk, a colleague's machine — still open.
    pub relative: Option<PathBuf>,
}

/// `sherd-workspace.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceFile {
    /// [`WORKSPACE_VERSION`].
    pub version: u32,
    /// The input folder, once one is linked.
    #[serde(default)]
    pub input: Option<InputRef>,
    /// Fragments left out of every run, by name (A §5.1). The files are untouched.
    #[serde(default)]
    pub excluded: BTreeSet<String>,
    /// The last launch sheet, as the app wrote it; the workspace does not read it.
    #[serde(default)]
    pub last_spec: Option<serde_json::Value>,
}

/// An open workspace. Holding one holds its lock; dropping it releases the lock.
#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
    file: WorkspaceFile,
    _lock: File,
}

impl Workspace {
    /// Makes `root` a workspace. The folder may exist; it may not already be a workspace.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`]; [`AppError::Locked`] when it is one already and is open elsewhere.
    pub fn create(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root).map_err(|e| AppError::io(root, e))?;
        if root.join(WORKSPACE_FILE).exists() {
            return Self::open(root);
        }
        let lock = lock(root)?;
        let file = WorkspaceFile {
            version: WORKSPACE_VERSION,
            input: None,
            excluded: BTreeSet::new(),
            last_spec: None,
        };
        let ws = Self { root: root.to_owned(), file, _lock: lock };
        ws.save()?;
        Ok(ws)
    }

    /// Opens a workspace and takes its lock.
    ///
    /// # Errors
    ///
    /// [`AppError::NotAWorkspace`], [`AppError::Locked`], [`AppError::Version`] for a file a newer
    /// app wrote, [`AppError::Json`].
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join(WORKSPACE_FILE);
        if !path.is_file() {
            return Err(AppError::NotAWorkspace { path: root.to_owned() });
        }
        let lock = lock(root)?;
        let file: WorkspaceFile = atomic::read_json(&path)?;
        if file.version > WORKSPACE_VERSION {
            return Err(AppError::Version { path, found: file.version, expected: WORKSPACE_VERSION });
        }
        Ok(Self { root: root.to_owned(), file, _lock: lock })
    }

    /// The folder.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The file as it stands.
    pub fn file(&self) -> &WorkspaceFile {
        &self.file
    }

    /// The input folder if it can be found now: by its relative path first, then its absolute one.
    /// `None` both when no input is linked and when it is linked but missing (A §5, «input
    /// missing»); [`file`](Self::file) tells the two apart.
    pub fn input(&self) -> Option<PathBuf> {
        let input = self.file.input.as_ref()?;
        let relative = input.relative.as_ref().map(|r| self.root.join(r));
        relative
            .into_iter()
            .chain(std::iter::once(input.absolute.clone()))
            .find(|p| p.is_dir())
            .and_then(|p| p.canonicalize().ok())
    }

    /// Links the input folder.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`] when `dir` is not a folder that can be resolved.
    pub fn set_input(&mut self, dir: &Path) -> Result<()> {
        let absolute = dir.canonicalize().map_err(|e| AppError::io(dir, e))?;
        let root = self.root.canonicalize().map_err(|e| AppError::io(&self.root, e))?;
        self.file.input = Some(InputRef { relative: relative_to(&root, &absolute), absolute });
        self.save()
    }

    /// Leaves a fragment out of every run, or takes it back in (A §5.1).
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn set_excluded(&mut self, name: &str, excluded: bool) -> Result<()> {
        if excluded {
            self.file.excluded.insert(name.to_owned());
        } else {
            self.file.excluded.remove(name);
        }
        self.save()
    }

    /// Remembers the launch sheet.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn set_last_spec(&mut self, spec: serde_json::Value) -> Result<()> {
        self.file.last_spec = Some(spec);
        self.save()
    }

    /// `cache/`: R §3.7's fragment cache, shared by every run.
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    /// `fragments/`: thumbnails, display meshes and `index.json`.
    pub fn fragments_dir(&self) -> PathBuf {
        self.root.join("fragments")
    }

    /// `runs/`.
    pub fn runs_dir(&self) -> PathBuf {
        self.root.join("runs")
    }

    /// `runs/<id>/`.
    pub fn run_dir(&self, id: &str) -> PathBuf {
        self.runs_dir().join(id)
    }

    /// `exports/`.
    pub fn exports_dir(&self) -> PathBuf {
        self.root.join("exports")
    }

    fn save(&self) -> Result<()> {
        atomic::write_json(&self.root.join(WORKSPACE_FILE), &self.file)
    }
}

/// The OS's advisory lock on [`LOCK_FILE`]: released when the process ends, however it ends, so
/// there is no stale lock to explain to anyone.
fn lock(root: &Path) -> Result<File> {
    let path = root.join(LOCK_FILE);
    let file = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| AppError::io(&path, e))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => Err(AppError::Locked { path: root.to_owned() }),
        Err(std::fs::TryLockError::Error(e)) => Err(AppError::io(&path, e)),
    }
}

/// `target` as seen from `base`, both absolute; `None` when they share no root (two Windows
/// drives).
fn relative_to(base: &Path, target: &Path) -> Option<PathBuf> {
    let (base, target): (Vec<Component<'_>>, Vec<Component<'_>>) =
        (base.components().collect(), target.components().collect());
    let shared = base.iter().zip(&target).take_while(|(a, b)| a == b).count();
    if shared == 0 {
        return None;
    }
    let mut out = PathBuf::new();
    for _ in shared..base.len() {
        out.push("..");
    }
    for part in &target[shared..] {
        out.push(part);
    }
    Some(out)
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p sherd-app-core --lib 2>&1 | tail -6`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
cargo fmt -p sherd-app-core
git add Cargo.toml Cargo.lock crates/sherd-app-core
git commit -m "M2.2: sherd-app-core -- a workspace is a locked folder that finds its scans again after both moved"
```

---

### Task 3: the input's snapshot, and `run.json`

**Files:**
- Create: `crates/sherd-app-core/src/snapshot.rs`, `src/run.rs`
- Modify: `src/lib.rs` (`pub mod snapshot; pub mod run;`)

**Interfaces:**
- Produces:
  - `snapshot::{FileStamp { name: String, file: String, size: u64, mtime_ms: i64 }, InputSnapshot { files: Vec<FileStamp>, excluded: BTreeSet<String> }, StaleDiff { added, removed, changed, excluded_added, excluded_removed: Vec<String> }}`, `snapshot::scan(input: &Path, excluded: &BTreeSet<String>) -> Result<InputSnapshot>`, `snapshot::diff(then: &InputSnapshot, now: &InputSnapshot) -> StaleDiff`, `StaleDiff::is_empty()`
  - `run::{RunFile, RunStatus, FailKind, EngineInfo, StageTime, RunCounts, RUN_FILE, RUN_VERSION}`; `RunFile::{new(id, spec: serde_json::Value, input: InputSnapshot, created: String), load(dir), save(dir)}`; `run::list(runs_dir: &Path) -> Result<Vec<RunFile>>` newest first; `run::mark_interrupted(runs_dir: &Path, finished: &str) -> Result<Vec<String>>`; `run::new_id(existing: &[String], now: chrono::DateTime<chrono::Local>) -> String`; `run::timestamp(now) -> String` (RFC 3339)
  - `FailKind` is `serde(rename_all = "snake_case")`: `Input, TooFew, Disk, Gpu, Protocol, Internal, Crashed, Cancelled` (A §10)

- [ ] **Step 1: Write the failing tests**

`src/snapshot.rs` tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str, files: &[(&str, &[u8])]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-snap-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in files {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
        dir
    }

    /// A §4 «Stale»: names, sizes, mtimes and the exclusions — and what changed is *said*.
    #[test]
    fn what_changed_in_the_input_is_named() {
        let input = dir("diff", &[("a.ply", b"1"), ("b.ply", b"22"), ("c.ply", b"333")]);
        let then = scan(&input, &BTreeSet::new()).unwrap();
        assert_eq!(then.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
        assert!(diff(&then, &then).is_empty());

        std::fs::remove_file(input.join("a.ply")).unwrap();
        std::fs::write(input.join("b.ply"), b"4444").unwrap();
        std::fs::write(input.join("d.ply"), b"5").unwrap();
        let now = scan(&input, &BTreeSet::from(["c".to_owned()])).unwrap();
        let d = diff(&then, &now);
        assert_eq!((d.added, d.removed, d.changed), (vec!["d".to_owned()], vec!["a".to_owned()], vec!["b".to_owned()]));
        assert_eq!(d.excluded_added, ["c"]);
        assert!(d.excluded_removed.is_empty());
    }
}
```

`src/run.rs` tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn runs(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-runs-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn a_run(id: &str) -> RunFile {
        RunFile::new(id, serde_json::json!({}), InputSnapshot::default(), "2026-09-20T14:12:00+03:00".to_owned())
    }

    #[test]
    fn run_ids_are_local_minutes_and_a_second_run_in_the_same_minute_gets_a_suffix() {
        let now = chrono::Local.with_ymd_and_hms(2026, 9, 20, 14, 12, 40).unwrap();
        assert_eq!(new_id(&[], now), "2026-09-20_1412");
        let taken = ["2026-09-20_1412".to_owned(), "2026-09-20_1412-2".to_owned()];
        assert_eq!(new_id(&taken, now), "2026-09-20_1412-3");
    }

    /// A §4: a `running` with no live worker behind it is a run the app did not see the end of.
    #[test]
    fn a_run_left_running_is_marked_interrupted_and_the_list_is_newest_first() {
        let dir = runs("lifecycle");
        let mut done = a_run("2026-09-19_0900");
        done.status = RunStatus::Done;
        done.save(&dir.join(&done.id)).unwrap();
        a_run("2026-09-20_1412").save(&dir.join("2026-09-20_1412")).unwrap();
        std::fs::create_dir_all(dir.join("not-a-run")).unwrap();

        assert_eq!(mark_interrupted(&dir, "2026-09-20T15:00:00+03:00").unwrap(), ["2026-09-20_1412"]);
        let listed = list(&dir).unwrap();
        assert_eq!(listed.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["2026-09-20_1412", "2026-09-19_0900"]);
        assert_eq!(listed[0].status, RunStatus::Interrupted);
        assert_eq!(listed[0].finished.as_deref(), Some("2026-09-20T15:00:00+03:00"));
        assert_eq!(listed[1].status, RunStatus::Done);
    }

    #[test]
    fn a_failure_is_stored_with_its_kind() {
        let json = serde_json::to_value(RunStatus::Failed { kind: FailKind::Gpu, message: "device lost".into() }).unwrap();
        assert_eq!(json, serde_json::json!({ "state": "failed", "kind": "gpu", "message": "device lost" }));
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sherd-app-core --lib 2>&1 | tail -5`
Expected: compile errors — the modules do not exist.

- [ ] **Step 3: Implement `snapshot.rs`**

```rust
//! What the input folder held when a run was made, and how that differs from now (A §4, «Stale»).
//!
//! Names, sizes and modification times, never content: the input is gigabytes, and a file changed
//! under the same size and mtime is still caught by R §3.7's cache validation at the next
//! `Prepare`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::{AppError, Result};

/// One scan as the snapshot saw it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStamp {
    /// The fragment's name (R §2) — what every decision, cache and display mesh is keyed by.
    pub name: String,
    /// The file's name in the input folder.
    pub file: String,
    /// Bytes.
    pub size: u64,
    /// Modification time, milliseconds since the epoch; 0 where the file system has none.
    pub mtime_ms: i64,
}

/// The input as it stood.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSnapshot {
    /// Every scan of the folder, the excluded ones included, in R §2's order.
    pub files: Vec<FileStamp>,
    /// The workspace's exclusions at the time.
    pub excluded: BTreeSet<String>,
}

/// How two snapshots differ, by fragment name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleDiff {
    /// Scans that are there now and were not.
    pub added: Vec<String>,
    /// Scans that were there and are not.
    pub removed: Vec<String>,
    /// Scans whose size or modification time changed.
    pub changed: Vec<String>,
    /// Fragments excluded since.
    pub excluded_added: Vec<String>,
    /// Fragments taken back in since.
    pub excluded_removed: Vec<String>,
}

impl StaleDiff {
    /// Whether nothing differs — the run is current.
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// The input folder now.
///
/// # Errors
///
/// [`AppError::Core`] when the folder cannot be listed, [`AppError::Io`] for a file's metadata.
pub fn scan(input: &Path, excluded: &BTreeSet<String>) -> Result<InputSnapshot> {
    let mut files = Vec::new();
    for entry in sherd_core::collection::discover(input)? {
        let meta = std::fs::metadata(&entry.path).map_err(|e| AppError::io(&entry.path, e))?;
        let mtime_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .and_then(|d| i64::try_from(d.as_millis()).ok())
            .unwrap_or(0);
        files.push(FileStamp {
            name: entry.name,
            file: entry.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
            size: meta.len(),
            mtime_ms,
        });
    }
    Ok(InputSnapshot { files, excluded: excluded.clone() })
}

/// What changed from `then` to `now`.
pub fn diff(then: &InputSnapshot, now: &InputSnapshot) -> StaleDiff {
    let by_name = |s: &InputSnapshot| -> BTreeMap<String, (u64, i64)> {
        s.files.iter().map(|f| (f.name.clone(), (f.size, f.mtime_ms))).collect()
    };
    let (old, new) = (by_name(then), by_name(now));
    StaleDiff {
        added: new.keys().filter(|n| !old.contains_key(*n)).cloned().collect(),
        removed: old.keys().filter(|n| !new.contains_key(*n)).cloned().collect(),
        changed: new
            .iter()
            .filter(|(n, stamp)| old.get(*n).is_some_and(|was| was != *stamp))
            .map(|(n, _)| n.clone())
            .collect(),
        excluded_added: now.excluded.difference(&then.excluded).cloned().collect(),
        excluded_removed: then.excluded.difference(&now.excluded).cloned().collect(),
    }
}
```

(The test rewrites `b.ply` with a different **size**, so it does not depend on the file system's mtime resolution.)

- [ ] **Step 4: Implement `run.rs`**

```rust
//! `run.json` (A §4): what a run was asked, on what input, how it ended. Written by the host —
//! the worker never touches it (A §2.1) — so that a worker that dies still leaves a run the
//! history can explain.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sherd_core::Params;

use crate::snapshot::InputSnapshot;
use crate::{AppError, Result, atomic};

/// The file's name inside a run folder.
pub const RUN_FILE: &str = "run.json";
/// The format this build writes.
pub const RUN_VERSION: u32 = 1;

/// Why a job failed (A §10). The window turns a kind into a sentence and an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailKind {
    /// A scan cannot be read or parsed.
    Input,
    /// Fewer than two fragments.
    TooFew,
    /// A write failed, or there is no space.
    Disk,
    /// The adapter or the device.
    Gpu,
    /// The window and the worker do not speak the same protocol.
    Protocol,
    /// A panic in the engine; the backtrace is in `engine.log`.
    Internal,
    /// The process ended without saying how.
    Crashed,
    /// The user's button, or the host going away.
    Cancelled,
}

/// Where a run is in its life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RunStatus {
    /// A worker is on it — or was, when the app last wrote.
    Running,
    /// Finished.
    Done,
    /// Stopped by the user.
    Cancelled,
    /// Ended badly.
    Failed {
        /// Why, by class.
        kind: FailKind,
        /// Why, in the engine's words.
        message: String,
    },
    /// Found `running` at start-up with no worker behind it.
    Interrupted,
}

/// What matched: D §4.3's engine block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    /// `sherd-core`'s version.
    pub core_version: String,
    /// The algorithm reference it implements.
    pub algo_ref: String,
    /// The commit it was built from.
    pub commit: String,
    /// `cpu`, or `gpu:<adapter>`.
    pub backend: String,
}

/// One stage's wall clock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageTime {
    /// R §11.2's stage name.
    pub stage: String,
    /// Seconds.
    pub seconds: f64,
}

/// What a finished run found — the numbers the history and the mode tabs show.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunCounts {
    /// Fragments in the run.
    pub fragments: usize,
    /// Pairs matched.
    pub pairs: usize,
    /// Pairs R §4.1's wall-ratio filter skipped.
    pub skipped_pairs: usize,
    /// Candidates scored.
    pub candidates: usize,
    /// Confirmed joins (with the tier off: accepted ones).
    pub confirmed: usize,
    /// Probable joins.
    pub probable: usize,
    /// Groups of two or more.
    pub groups: usize,
    /// Fragments in no group.
    pub unassembled: usize,
    /// R §11.2's timings, in the order the stages finished.
    pub timings: Vec<StageTime>,
}

/// `run.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunFile {
    /// [`RUN_VERSION`].
    pub version: u32,
    /// The folder's name.
    pub id: String,
    /// RFC 3339, local offset.
    pub created: String,
    /// When it ended, however it ended.
    #[serde(default)]
    pub finished: Option<String>,
    /// Where it is in its life.
    pub status: RunStatus,
    /// The launch sheet, as the window sent it.
    pub spec: serde_json::Value,
    /// Every threshold it resolved to; filled when the run ends.
    #[serde(default)]
    pub params: Option<Params>,
    /// The input it ran on (A §4, «Stale»).
    pub input: InputSnapshot,
    /// What matched.
    #[serde(default)]
    pub engine: Option<EngineInfo>,
    /// What it found.
    #[serde(default)]
    pub counts: Option<RunCounts>,
    /// The run whose decisions it started from (A §8.5).
    #[serde(default)]
    pub carried_from: Option<String>,
}

impl RunFile {
    /// A run that is about to start.
    pub fn new(id: &str, spec: serde_json::Value, input: InputSnapshot, created: String) -> Self {
        Self {
            version: RUN_VERSION,
            id: id.to_owned(),
            created,
            finished: None,
            status: RunStatus::Running,
            spec,
            params: None,
            input,
            engine: None,
            counts: None,
            carried_from: None,
        }
    }

    /// Reads `<dir>/run.json`.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`], [`AppError::Json`], [`AppError::Version`].
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(RUN_FILE);
        let run: Self = atomic::read_json(&path)?;
        if run.version > RUN_VERSION {
            return Err(AppError::Version { path, found: run.version, expected: RUN_VERSION });
        }
        Ok(run)
    }

    /// Writes `<dir>/run.json`, creating the folder.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| AppError::io(dir, e))?;
        atomic::write_json(&dir.join(RUN_FILE), self)
    }
}

/// Every run of the workspace, newest first. A folder without a readable `run.json` is not a run
/// and is skipped: the history must open even when one folder is damaged.
///
/// # Errors
///
/// [`AppError::Io`] when `runs/` exists and cannot be listed. A missing `runs/` is no runs.
pub fn list(runs_dir: &Path) -> Result<Vec<RunFile>> {
    if !runs_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut runs: Vec<RunFile> = std::fs::read_dir(runs_dir)
        .map_err(|e| AppError::io(runs_dir, e))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| RunFile::load(&entry.path()).ok())
        .collect();
    runs.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(runs)
}

/// Marks every run still `running` as interrupted (A §4). Called once when a workspace opens,
/// before any worker is spawned, so a `running` on disk can only be one nobody is running.
///
/// # Errors
///
/// [`AppError::Io`].
pub fn mark_interrupted(runs_dir: &Path, finished: &str) -> Result<Vec<String>> {
    let mut marked = Vec::new();
    for mut run in list(runs_dir)? {
        if run.status == RunStatus::Running {
            run.status = RunStatus::Interrupted;
            run.finished = Some(finished.to_owned());
            run.save(&runs_dir.join(&run.id))?;
            marked.push(run.id);
        }
    }
    Ok(marked)
}

/// A run's id: local time to the minute, `-2`, `-3`, … when the minute is taken (A §4).
pub fn new_id(existing: &[String], now: chrono::DateTime<chrono::Local>) -> String {
    let base = now.format("%Y-%m-%d_%H%M").to_string();
    if !existing.contains(&base) {
        return base;
    }
    (2..).map(|n| format!("{base}-{n}")).find(|id| !existing.contains(id)).unwrap_or(base)
}

/// `now` as `run.json` writes a time.
pub fn timestamp(now: chrono::DateTime<chrono::Local>) -> String {
    now.to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p sherd-app-core --lib 2>&1 | tail -5`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
cargo fmt -p sherd-app-core
git add crates/sherd-app-core
git commit -m "M2.3: run.json and the input's snapshot -- a run says what it ran on, and a stale one says what changed"
```

---

### Task 4: `decisions.json`, and decisions → constraints

**Files:**
- Create: `crates/sherd-app-core/src/decisions.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `decisions::{DecisionsFile { version: u32, decisions: Vec<Decision> }, Decision { a, b: String, verdict: Verdict, pose: Option<[[f64; 4]; 4]>, source: Option<String>, bulk: bool, at: String, carried_from: Option<String> }, Verdict::{Accept, Reject}, DECISIONS_FILE, DECISIONS_VERSION}`; `DecisionsFile::{load_or_default(dir: &Path) -> Result<Self>, save(&self, dir: &Path) -> Result<()>, set(&mut self, decision: Decision), clear(&mut self, a: &str, b: &str)}`; `decisions::to_constraints(file: &DecisionsFile, names: &[String]) -> Result<(Option<Constraints>, Vec<Decision>)>` — the second value is what was dropped.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const POSE: [[f64; 4]; 4] =
        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, -2.25], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]];

    fn decision(a: &str, b: &str, verdict: Verdict) -> Decision {
        Decision {
            a: a.to_owned(),
            b: b.to_owned(),
            verdict,
            pose: (verdict == Verdict::Accept).then_some(POSE),
            source: Some("probable".to_owned()),
            bulk: false,
            at: "2026-09-20T14:31:07+03:00".to_owned(),
            carried_from: None,
        }
    }

    /// A §8.1: the key is the unordered pair, and a later decision about it replaces the earlier.
    #[test]
    fn a_pair_has_one_decision_whichever_way_round_it_is_named() {
        let mut file = DecisionsFile::default();
        file.set(decision("pieceA", "pieceB", Verdict::Reject));
        file.set(decision("pieceB", "pieceA", Verdict::Accept));
        assert_eq!(file.decisions.len(), 1);
        assert_eq!(file.decisions[0].verdict, Verdict::Accept);
        file.clear("pieceA", "pieceB");
        assert!(file.decisions.is_empty());
    }

    /// A §8.2: accept is `must_join` with the pose, reject is `must_not_join`, and a decision about
    /// a fragment that is gone is dropped and **said** — `constraints::resolve` fails the whole
    /// run on an unknown name, so the filter is the app's duty.
    #[test]
    fn decisions_become_the_engines_constraints_and_the_orphans_are_reported() {
        let mut file = DecisionsFile::default();
        file.set(decision("pieceA", "pieceB", Verdict::Accept));
        file.set(decision("pieceA", "pieceC", Verdict::Reject));
        file.set(decision("pieceA", "gone", Verdict::Reject));
        let names = ["pieceA", "pieceB", "pieceC"].map(str::to_owned);
        let (constraints, dropped) = to_constraints(&file, &names).unwrap();
        let json = serde_json::to_value(constraints.expect("two decisions survive")).unwrap();
        assert_eq!(json["version"], 1);
        assert_eq!(json["must_join"][0]["a"], "pieceA");
        assert_eq!(json["must_join"][0]["pose"][1][3], -2.25);
        assert_eq!(json["must_not_join"], serde_json::json!([["pieceA", "pieceC"]]));
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].b, "gone");
        // and the engine itself accepts what was built
        sherd_core::assembly::constraints::resolve(
            &serde_json::from_value(json).unwrap(),
            &names,
        )
        .expect("the engine resolves it");
    }

    #[test]
    fn no_decisions_are_no_constraints() {
        assert_eq!(to_constraints(&DecisionsFile::default(), &[]).unwrap().0, None);
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sherd-app-core --lib decisions 2>&1 | tail -5`
Expected: compile error.

- [ ] **Step 3: Implement**

```rust
//! The reviewer's decisions (A §8.1) and what the engine is told about them (A §8.2).

use std::path::Path;

use serde::{Deserialize, Serialize};
use sherd_core::assembly::constraints::Constraints;

use crate::{AppError, Result, atomic};

/// The file's name inside a run folder.
pub const DECISIONS_FILE: &str = "decisions.json";
/// The format this build writes.
pub const DECISIONS_VERSION: u32 = 1;

/// What the reviewer said about a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Place it, at [`Decision::pose`].
    Accept,
    /// Never place this pair — the pair as a whole, which is `must_not_join`'s meaning.
    Reject,
}

/// One decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// One fragment, by name.
    pub a: String,
    /// The other.
    pub b: String,
    /// What was decided.
    pub verdict: Verdict,
    /// The accepted candidate's pose, four rows of four, `b` into `a`'s frame **as the candidate
    /// names them**. Carried here so that a decision outlives the `match.state` it was made on.
    #[serde(default)]
    pub pose: Option<[[f64; 4]; 4]>,
    /// The band the candidate was in when it was decided on.
    #[serde(default)]
    pub source: Option<String>,
    /// Made by «Принять все оставшиеся вероятные» (A §8.4), and undone with it.
    #[serde(default)]
    pub bulk: bool,
    /// RFC 3339.
    pub at: String,
    /// The run it was carried over from (A §8.5).
    #[serde(default)]
    pub carried_from: Option<String>,
}

/// `decisions.json`: the decisions as they stand. Undo and redo are the window's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionsFile {
    /// [`DECISIONS_VERSION`].
    pub version: u32,
    /// One per pair.
    pub decisions: Vec<Decision>,
}

impl Default for DecisionsFile {
    fn default() -> Self {
        Self { version: DECISIONS_VERSION, decisions: Vec::new() }
    }
}

fn same_pair(d: &Decision, a: &str, b: &str) -> bool {
    (d.a == a && d.b == b) || (d.a == b && d.b == a)
}

impl DecisionsFile {
    /// `<dir>/decisions.json`, or no decisions when the run has none yet.
    ///
    /// # Errors
    ///
    /// [`AppError::Json`], [`AppError::Version`].
    pub fn load_or_default(dir: &Path) -> Result<Self> {
        let path = dir.join(DECISIONS_FILE);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let file: Self = atomic::read_json(&path)?;
        if file.version > DECISIONS_VERSION {
            return Err(AppError::Version { path, found: file.version, expected: DECISIONS_VERSION });
        }
        Ok(file)
    }

    /// Writes `<dir>/decisions.json`.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn save(&self, dir: &Path) -> Result<()> {
        atomic::write_json(&dir.join(DECISIONS_FILE), self)
    }

    /// Decides a pair, replacing whatever was decided about it before.
    pub fn set(&mut self, decision: Decision) {
        self.clear(&decision.a.clone(), &decision.b.clone());
        self.decisions.push(decision);
    }

    /// Takes a pair's decision back.
    pub fn clear(&mut self, a: &str, b: &str) {
        self.decisions.retain(|d| !same_pair(d, a, b));
    }
}

/// The engine's `constraints.json` for these decisions, and the decisions that could not be
/// carried because a fragment they name is not in `names`.
///
/// Built as the documented JSON and parsed by the engine's own deserialiser, so that this crate
/// depends on the file format the README describes and not on the shape of the engine's enums.
///
/// # Errors
///
/// [`AppError::Json`] if the engine refuses the JSON, which would be a bug here.
pub fn to_constraints(
    file: &DecisionsFile,
    names: &[String],
) -> Result<(Option<Constraints>, Vec<Decision>)> {
    let known = |d: &Decision| names.contains(&d.a) && names.contains(&d.b);
    let (kept, dropped): (Vec<&Decision>, Vec<&Decision>) = file.decisions.iter().partition(|d| known(d));
    let dropped: Vec<Decision> = dropped.into_iter().cloned().collect();
    if kept.is_empty() {
        return Ok((None, dropped));
    }
    let must_join: Vec<serde_json::Value> = kept
        .iter()
        .filter(|d| d.verdict == Verdict::Accept)
        .map(|d| match d.pose {
            Some(pose) => serde_json::json!({ "a": d.a, "b": d.b, "pose": pose }),
            None => serde_json::json!([d.a, d.b]),
        })
        .collect();
    let must_not_join: Vec<serde_json::Value> = kept
        .iter()
        .filter(|d| d.verdict == Verdict::Reject)
        .map(|d| serde_json::json!([d.a, d.b]))
        .collect();
    let json = serde_json::json!({ "version": 1, "must_join": must_join, "must_not_join": must_not_join });
    let constraints = serde_json::from_value(json)
        .map_err(|source| AppError::Json { path: DECISIONS_FILE.into(), source })?;
    Ok((Some(constraints), dropped))
}
```

If the engine's `Constraints` requires `same_object`/`different_object` keys to be present, add them as empty arrays to the `json!` — check `crates/sherd-core/src/assembly/constraints.rs` for `#[serde(default)]` on those fields.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p sherd-app-core --lib decisions 2>&1 | tail -5`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
cargo fmt -p sherd-app-core
git add crates/sherd-app-core
git commit -m "M2.4: a reviewer's decisions, and the constraints the engine is given for them -- orphans dropped and said"
```

---

### Task 5: the protocol, and a worker that answers `Info`

**Files:**
- Create: `crates/sherd-app-core/src/protocol.rs`, `src/worker/mod.rs`
- Modify: `src/lib.rs`, `src/bin/sherd-engine-worker.rs`

**Interfaces:**
- Produces (`protocol`, everything `Clone + Debug + PartialEq + Serialize + Deserialize`):
  - `PROTOCOL: u32 = 1`
  - `enum Job { Info { adapter: Option<String>, selftest: bool }, Prepare(PrepareJob), Run(RunJob) }` — `#[serde(tag = "job", rename_all = "snake_case")]`
  - `struct PrepareJob { workspace: PathBuf, input: PathBuf, excluded: BTreeSet<String>, target_faces: usize, seed: u64, memory_gb: Option<f64>, workers: usize }`
  - `struct RunJob { workspace: PathBuf, input: PathBuf, run_id: String, excluded: BTreeSet<String>, spec: RunSpec, constraints: Option<Constraints> }`
  - `enum Request { Cancel }` — `#[serde(tag = "request", rename_all = "snake_case")]`
  - `enum BackendChoice { Auto, Cpu, Gpu }`, `enum Preset { Standard, Thorough, Custom }` (snake_case)
  - `struct RunSpec { preset, backend: BackendChoice, adapter: Option<String>, gpu_memory_gb: Option<f64>, seed: u64, target_faces: usize, tiers: bool, agree_seeds: u32, thick_ratio: f64, min_tight: f64, max_gap: f64, max_pen: f64, min_seam: f64, workers: usize, memory_gb: Option<f64> }` with `Default` (= the CLI's defaults) and `fn params(&self) -> Params`
  - `enum Event { Hello { protocol, core_version, algo_ref, commit }, Info { backends: Vec<String>, selftest: Vec<String> }, Stage { name: String }, Progress { stage: String, done: usize, total: usize }, FragmentReady(FragmentInfo), Assembly(AssemblyDto), Done { counts: Option<RunCounts>, engine: Option<EngineInfo>, params: Option<Params> }, Failed { kind: FailKind, message: String } }` — `#[serde(tag = "event", rename_all = "snake_case")]`
  - `struct FragmentInfo { name, file: String, size: u64, mtime_ms: i64, stats: FragmentStats, warnings: Vec<Warning>, coloured: bool, display_faces: usize }`
  - `enum Warning { ThicknessOutlier { thickness: f64, median: f64 }, NotWatertight }` — `#[serde(tag = "warning", rename_all = "snake_case")]`
  - `struct AssemblyDto { groups: Vec<GroupDto>, poses: BTreeMap<String, [[f64; 4]; 4]>, joins: Vec<JoinDto>, unplaced: Vec<UnplacedDto> }`, `GroupDto { members: Vec<String>, refined: bool }`, `JoinDto { a, b: String }`, `UnplacedDto { a, b, reason: String }`
- Produces (`worker`): `pub fn serve(input: impl BufRead + Send + 'static, output: impl Write + Send + 'static) -> i32` (exit code: 0 done, 1 failed, 2 cancelled); `pub(crate) struct Emitter` (`emit(&self, &Event)`, `Clone`); `pub(crate) struct Context { emitter: Emitter, cancel: Cancel, watch: Watch }`; `pub(crate) fn fail_kind(error: &sherd_core::Error) -> FailKind`.

`Stage` is emitted by the worker's `Progress` implementation when a stage's first report arrives; the spec's `index`/`of` are dropped — the set of stages varies with the options (`screen`, `second_pass`), so the window owns the strip and lights a stage by name. Update A §2.2's event list in the same commit: `Stage { name }`.

- [ ] **Step 1: Write the failing tests**

`src/protocol.rs` tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_and_an_event_are_one_tagged_json_line_each() {
        let job = Job::Info { adapter: None, selftest: false };
        let line = serde_json::to_string(&job).unwrap();
        assert_eq!(line, r#"{"job":"info","adapter":null,"selftest":false}"#);
        assert_eq!(serde_json::from_str::<Job>(&line).unwrap(), job);

        let event = Event::Progress { stage: "matching".into(), done: 7, total: 190 };
        assert_eq!(
            serde_json::to_string(&event).unwrap(),
            r#"{"event":"progress","stage":"matching","done":7,"total":190}"#
        );
        assert_eq!(serde_json::to_string(&Request::Cancel).unwrap(), r#"{"request":"cancel"}"#);
    }

    /// A §7.4: «Стандарт» is the CLI's defaults, threshold for threshold.
    #[test]
    fn the_standard_preset_is_the_command_lines_defaults() {
        use sherd_core::objects::ObjectParams;
        use sherd_core::tiers::Thresholds;
        let cli = Params {
            tiers: Some(Thresholds::default()),
            objects: Some(ObjectParams::default()),
            ..Params::default()
        };
        assert_eq!(RunSpec::default().params(), cli);
        let thorough = RunSpec { agree_seeds: 2, ..RunSpec::default() }.params();
        assert_eq!(thorough.tiers.unwrap().agree_seeds, 2);
        assert_eq!(RunSpec { tiers: false, ..RunSpec::default() }.params().tiers, None);
    }
}
```

`src/worker/mod.rs` tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn events(sink: &Sink) -> Vec<Event> {
        String::from_utf8(sink.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).expect("every line is an event"))
            .collect()
    }

    #[test]
    fn a_worker_says_hello_first_answers_info_and_ends_with_done() {
        let sink = Sink::default();
        let job = r#"{"job":"info","adapter":null,"selftest":false}"#;
        let code = serve(std::io::Cursor::new(format!("{job}\n")), sink.clone());
        let seen = events(&sink);
        assert!(matches!(&seen[0], Event::Hello { protocol, .. } if *protocol == PROTOCOL));
        assert!(matches!(&seen[1], Event::Info { backends, .. } if !backends.is_empty()));
        assert!(matches!(seen.last(), Some(Event::Done { .. })));
        assert_eq!(code, 0);
    }

    #[test]
    fn a_line_that_is_not_a_job_is_a_protocol_failure_and_not_a_panic() {
        let sink = Sink::default();
        let code = serve(std::io::Cursor::new("make me a sandwich\n".to_owned()), sink.clone());
        let seen = events(&sink);
        assert!(matches!(seen.last(), Some(Event::Failed { kind: FailKind::Protocol, .. })));
        assert_eq!(code, 1);
    }

    #[test]
    fn progress_is_throttled_but_a_stage_always_starts_and_ends_on_the_wire() {
        let sink = Sink::default();
        let progress = WorkerProgress::new(Emitter::new(sink.clone()));
        for done in 1..=1000 {
            sherd_core::progress::Progress::advance(&progress, "matching", done, 1000);
        }
        let seen = events(&sink);
        assert_eq!(seen[0], Event::Stage { name: "matching".into() });
        assert_eq!(seen.last(), Some(&Event::Progress { stage: "matching".into(), done: 1000, total: 1000 }));
        assert!(seen.len() < 50, "1000 reports in a few milliseconds are a handful of lines, not 1000");
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sherd-app-core --lib 2>&1 | tail -5`
Expected: compile errors.

- [ ] **Step 3: Implement `protocol.rs`**

The types exactly as the Interfaces block lists them, each with a doc comment, plus:

```rust
impl Default for RunSpec {
    fn default() -> Self {
        let (p, t, o) = (Params::default(), Thresholds::default(), RunOptions::default());
        Self {
            preset: Preset::Standard,
            backend: BackendChoice::Auto,
            adapter: None,
            gpu_memory_gb: None,
            seed: p.seed,
            target_faces: o.target_faces,
            tiers: true,
            agree_seeds: t.agree_seeds,
            thick_ratio: p.thick_ratio,
            min_tight: p.min_tight,
            max_gap: p.max_gap,
            max_pen: p.max_pen,
            min_seam: p.min_seam,
            workers: 0,
            memory_gb: None,
        }
    }
}

impl RunSpec {
    /// Every threshold of the run: the CLI's defaults (`crates/sherd-cli/src/main.rs`, `params`)
    /// with the launch sheet's eleven on top (A §7.4).
    pub fn params(&self) -> Params {
        Params {
            seed: self.seed,
            thick_ratio: self.thick_ratio,
            min_tight: self.min_tight,
            max_gap: self.max_gap,
            max_pen: self.max_pen,
            min_seam: self.min_seam,
            tiers: self
                .tiers
                .then(|| Thresholds { agree_seeds: self.agree_seeds, ..Thresholds::default() }),
            objects: Some(ObjectParams::default()),
            ..Params::default()
        }
    }

    /// The engine's `Backend` for the choice.
    pub fn backend(&self) -> sherd_core::Backend {
        match self.backend {
            BackendChoice::Auto => sherd_core::Backend::Auto,
            BackendChoice::Cpu => sherd_core::Backend::Cpu,
            BackendChoice::Gpu => sherd_core::Backend::Gpu,
        }
    }
}
```

Check the field names against `Params` (`crates/sherd-core/src/params.rs`) and `Thresholds`; `seed`'s type is whatever `Params::seed` is. Re-export `FailKind`, `RunCounts`, `EngineInfo`, `StageTime` from `crate::run` so the protocol has one home: `pub use crate::run::{EngineInfo, FailKind, RunCounts, StageTime};`.

- [ ] **Step 4: Implement `worker/mod.rs`**

```rust
//! The engine role of the app's binary (A §2.1): one job in, events out, one line of JSON each.
//!
//! Why a process: a run is 17–80 minutes, holds ~3 GB and drives `wgpu` on drivers nobody has
//! tested. An abort must end a run and not the window, the memory must go back to the OS, and a
//! hard kill must exist behind the cooperative cancel.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sherd_core::progress::{Cancel, Progress, Watch};

use crate::protocol::{Event, FailKind, Job, PROTOCOL, Request};

/// Where events go. One line each, flushed, under a lock: reports come from rayon's workers.
#[derive(Clone)]
pub(crate) struct Emitter(Arc<Mutex<Box<dyn Write + Send>>>);

impl std::fmt::Debug for Emitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Emitter")
    }
}

impl Emitter {
    pub(crate) fn new(output: impl Write + Send + 'static) -> Self {
        Self(Arc::new(Mutex::new(Box::new(output))))
    }

    /// Writes one event. A host that has gone away is not an error worth stopping for here: its
    /// going away closes stdin, which is what stops the job.
    pub(crate) fn emit(&self, event: &Event) {
        let Ok(mut line) = serde_json::to_vec(event) else { return };
        line.push(b'\n');
        if let Ok(mut out) = self.0.lock() {
            let _ = out.write_all(&line).and_then(|()| out.flush());
        }
    }
}

/// D §5's `Progress`, onto the wire. A stage's first report also announces the stage; after
/// that at most one line per [`WorkerProgress::EVERY`] per stage, and always the last one —
/// 11 000 pairs are 11 000 reports, and the window needs ten a second.
#[derive(Debug)]
pub(crate) struct WorkerProgress {
    emitter: Emitter,
    last: Mutex<HashMap<String, Instant>>,
}

impl WorkerProgress {
    const EVERY: Duration = Duration::from_millis(100);

    pub(crate) fn new(emitter: Emitter) -> Self {
        Self { emitter, last: Mutex::new(HashMap::new()) }
    }
}

impl Progress for WorkerProgress {
    fn advance(&self, stage: &str, done: usize, total: usize) {
        let now = Instant::now();
        let Ok(mut last) = self.last.lock() else { return };
        let due = match last.get(stage) {
            None => {
                self.emitter.emit(&Event::Stage { name: stage.to_owned() });
                true
            }
            Some(at) => done >= total || now.duration_since(*at) >= Self::EVERY,
        };
        if due {
            last.insert(stage.to_owned(), now);
            self.emitter.emit(&Event::Progress { stage: stage.to_owned(), done, total });
        }
    }
}

/// What a job is handed.
#[derive(Debug)]
pub(crate) struct Context {
    pub(crate) emitter: Emitter,
    pub(crate) cancel: Cancel,
    pub(crate) watch: Watch,
}

/// How a job ended, other than well.
#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) kind: FailKind,
    pub(crate) message: String,
}

impl Failure {
    pub(crate) fn new(kind: FailKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl From<sherd_core::Error> for Failure {
    fn from(error: sherd_core::Error) -> Self {
        Self { kind: fail_kind(&error), message: error.to_string() }
    }
}

/// A §10's class for an engine error.
pub(crate) fn fail_kind(error: &sherd_core::Error) -> FailKind {
    use sherd_core::Error as E;
    match error {
        E::Cancelled => FailKind::Cancelled,
        E::Write { .. } => FailKind::Disk,
        E::Read { .. } | E::UnsupportedFormat { .. } | E::EmptyMesh { .. } => FailKind::Input,
        _ => FailKind::Internal,
    }
}

/// Serves one job: `Hello`, the job's events, then `Done` or `Failed`. Returns the process's exit
/// code — 0 done, 1 failed, 2 cancelled.
///
/// After the job line the input is read on a thread of its own: a `cancel` request raises D §5's
/// flag, and so does the end of the input — a window that died leaves no 3 GB orphan (A §2.1).
pub fn serve(mut input: impl BufRead + Send + 'static, output: impl Write + Send + 'static) -> i32 {
    let emitter = Emitter::new(output);
    emitter.emit(&Event::Hello {
        protocol: PROTOCOL,
        core_version: sherd_core::CORE_VERSION.to_owned(),
        algo_ref: sherd_core::ALGO_REF.to_owned(),
        commit: sherd_core::GIT_COMMIT.to_owned(),
    });
    let mut line = String::new();
    let job: Job = match input.read_line(&mut line).map(|_| serde_json::from_str(line.trim())) {
        Ok(Ok(job)) => job,
        Ok(Err(e)) => return fail(&emitter, &Failure::new(FailKind::Protocol, format!("not a job: {e}"))),
        Err(e) => return fail(&emitter, &Failure::new(FailKind::Protocol, e.to_string())),
    };

    let cancel = Cancel::new();
    let flag = cancel.clone();
    std::thread::spawn(move || {
        for line in input.lines() {
            match line.as_deref().map(str::trim).map(serde_json::from_str::<Request>) {
                Ok(Ok(Request::Cancel)) => flag.cancel(),
                Ok(Err(_)) => {} // an unknown request is ignored, not fatal: the window may be newer
                Err(_) => break,
            }
        }
        flag.cancel(); // end of input: the host is gone
    });

    let watch = Watch {
        cancel: Some(cancel.clone()),
        progress: Some(Arc::new(WorkerProgress::new(emitter.clone()))),
    };
    let context = Context { emitter: emitter.clone(), cancel, watch };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dispatch(job, &context)));
    match outcome {
        Ok(Ok(done)) => {
            emitter.emit(&done);
            0
        }
        Ok(Err(failure)) => fail(&emitter, &failure),
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "the engine panicked".to_owned());
            fail(&emitter, &Failure::new(FailKind::Internal, message))
        }
    }
}

fn fail(emitter: &Emitter, failure: &Failure) -> i32 {
    emitter.emit(&Event::Failed { kind: failure.kind, message: failure.message.clone() });
    if failure.kind == FailKind::Cancelled { 2 } else { 1 }
}

/// Runs the job; the `Event` it returns is the job's `Done`.
fn dispatch(job: Job, context: &Context) -> Result<Event, Failure> {
    match job {
        Job::Info { adapter, selftest } => {
            context.emitter.emit(&Event::Info {
                backends: sherd_backend::info_lines(),
                selftest: if selftest { sherd_backend::selftest_lines(adapter.as_deref()) } else { Vec::new() },
            });
            Ok(Event::Done { counts: None, engine: None, params: None })
        }
        Job::Prepare(_) | Job::Run(_) => {
            Err(Failure::new(FailKind::Protocol, "this build of the worker does not run this job yet"))
        }
    }
}
```

The `Prepare`/`Run` arm is replaced by Tasks 6 and 7; it is a real answer until then, not a placeholder — a worker asked for a job it cannot do says so.

`src/bin/sherd-engine-worker.rs`:

```rust
//! The engine role of the app's binary, as a binary of its own (A §2.1): what the headless tests
//! spawn, and what the Tauri shell reproduces by running itself with `--engine-worker`.

fn main() {
    // The engine's log goes to stderr, which the host appends to the run's `engine.log`; stdout
    // is the protocol's and nothing else may write to it.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("sherd=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_target(false)
        .init();
    let code = sherd_app_core::worker::serve(std::io::stdin().lock(), std::io::stdout());
    std::process::exit(code);
}
```

`StdinLock<'static>` is `BufRead` but not `Send`; if the compiler refuses it, pass `std::io::BufReader::new(std::io::stdin())` instead.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p sherd-app-core --lib 2>&1 | tail -5`
Expected: `17 passed` (12 before, 2 protocol, 3 worker).

- [ ] **Step 6: Commit**

```bash
cargo fmt -p sherd-app-core
git add crates/sherd-app-core docs/superpowers/specs/2026-09-20-desktop-app-design.md
git commit -m "M2.5: the protocol, and a worker that says hello, answers info, throttles progress and stops when its host is gone"
```

---

### Task 6: the `Prepare` job

**Files:**
- Create: `crates/sherd-app-core/src/worker/prepare.rs`
- Modify: `src/worker/mod.rs` (dispatch)

**Interfaces:**
- Consumes: `session::load_fragments`, `export::scene::{display_mesh, face_targets, fragment_glb, DEFAULT_FACES}`, `io::read_mesh`, `render`, `memory`, `report::FragmentStats`, `snapshot::scan`.
- Produces: `pub(crate) fn prepare(job: &PrepareJob, context: &Context) -> Result<Event, Failure>`; on disk, under `<workspace>/`: `cache/<name>.sherd`, `fragments/<name>.glb` (display mesh, source coordinates, scan colours), `fragments/<name>.seg.glb` (working mesh, grey shell and red fracture as vertex colours — «Излом красным», A §7.3), `fragments/<name>.png` (320×240 thumbnail), `fragments/index.json` = `Vec<FragmentInfo>` in R §2's order. Stage names reported: `"preprocess"` (the engine's), `"display"` (one per fragment, here). One `FragmentReady` per fragment.
- `pub const INDEX_FILE: &str = "index.json"; pub const THICKNESS_OUTLIER: f64 = 0.4;`

The flow: (1) `sherd_core::pipeline::set_threads(job.workers)`; (2) `discover_excluding` → `load_fragments` into `cache/` (progress `preprocess`, cancellable); fewer than two fragments → `Failure { kind: TooFew }`; (3) the collection's median thickness; (4) a `par_iter` over the fragments, each under the cancel check and the memory semaphore: read the source, `display_mesh` at its `face_targets` share of `DEFAULT_FACES`, write the three files **unless they exist and `index.json`'s stamp for the fragment (size, mtime) is unchanged**, emit `FragmentReady`, advance `"display"`; (5) write `index.json` atomically; `Done`.

- [ ] **Step 1: Implement**

```rust
//! `Prepare` (A §2.2, A §3.5): the collection preprocessed into the workspace's cache — which is
//! the first stage of any run, so a run then starts at matching — and, from the same pass, what
//! the window shows of the input: a thumbnail, a display mesh and the warnings.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use sherd_core::export::scene::{self, DisplayMesh};
use sherd_core::fragment::Fragment;
use sherd_core::memory::{self, Budget, MemorySemaphore};
use sherd_core::render::{self, Paint, Splat};
use sherd_core::report::FragmentStats;

use super::{Context, Failure};
use crate::protocol::{Event, FailKind, FragmentInfo, PrepareJob, Warning};
use crate::{atomic, snapshot};

/// `fragments/index.json`.
pub const INDEX_FILE: &str = "index.json";
/// The README's own threshold: a wall more than 40 % off the collection's median is worth a word.
pub const THICKNESS_OUTLIER: f64 = 0.4;

const THUMB: (usize, usize) = (320, 240);
const CLAY: [f64; 3] = [0.80, 0.62, 0.46];
const SHELL: [u8; 3] = [204, 204, 204];
const FRACTURE: [u8; 3] = [230, 51, 51];
```

then `prepare` following the flow above, with these pieces:

```rust
fn budget(gb: Option<f64>) -> Budget {
    gb.map_or_else(Budget::default_for_machine, Budget::gigabytes)
}

fn io_failure(path: &Path, e: &dyn std::fmt::Display) -> Failure {
    Failure::new(FailKind::Disk, format!("{}: {e}", path.display()))
}

/// The thumbnail: the fragment's own surface samples (R §3.5.1), which are uniform over the
/// surface — a display mesh's vertices are not — through the renderer the engine's previews use.
fn thumbnail(fragment: &Fragment, path: &Path) -> Result<(), Failure> {
    let points = fragment.samples.surface_f64();
    let normals: Vec<[f64; 3]> = fragment
        .samples
        .sp
        .iter()
        .map(|&face| fragment.mesh.face_normals[face as usize].to_f64())
        .collect();
    let views = render::principal_views(&points);
    let splat = Splat { points, normals, paint: Paint::Uniform(CLAY) };
    render::render_views(&[splat], &views[..1], THUMB.0, THUMB.1)
        .write_png(path)
        .map_err(|e| io_failure(path, &e))
}

/// The working mesh with R §3.4's labels as vertex colours: a vertex is red when most of the
/// faces around it are fracture. Vertex colours, because a simplifier keeps vertices and makes
/// new faces.
fn segmentation_mesh(fragment: &Fragment, target: usize) -> DisplayMesh {
    let mesh = &fragment.mesh;
    let mut votes = vec![(0_u32, 0_u32); mesh.v.len()];
    for (face, label) in mesh.f.iter().zip(&fragment.labels) {
        for &v in face {
            let slot = &mut votes[v as usize];
            slot.1 += 1;
            slot.0 += u32::from(label.is_fracture());
        }
    }
    let colors = votes.iter().map(|&(red, all)| if 2 * red > all { FRACTURE } else { SHELL }).collect();
    let as_mesh = sherd_core::Mesh {
        v: mesh.v.iter().map(|p| p.to_f64()).collect(),
        f: mesh.f.clone(),
        colors: Some(colors),
    };
    scene::display_mesh(&as_mesh, target)
}
```

`display_mesh` turns `colors` into **linear** 16-bit values (it treats them as sRGB scan colours); that is what glTF's `COLOR_0` wants and what the window's renderer expects, so nothing is undone here.

The per-fragment job, inside the `par_iter` (collect `Result<FragmentInfo, Failure>` by index):

```rust
    context.watch.check()?;                       // Error::Cancelled -> Failure via From
    let stamp = &stamps[n];                       // snapshot::scan(&job.input, &job.excluded), filtered to the run's fragments by name
    let (glb, seg, png) = (dir.join(format!("{name}.glb")), dir.join(format!("{name}.seg.glb")), dir.join(format!("{name}.png")));
    let fresh = previous.get(name).is_some_and(|p: &FragmentInfo| p.size == stamp.size && p.mtime_ms == stamp.mtime_ms)
        && glb.is_file() && seg.is_file() && png.is_file();
    let (coloured, display_faces) = if fresh {
        (previous[name].coloured, previous[name].display_faces)
    } else {
        let permit = semaphore.acquire(memory::scan_faces(&entry.path).map_or(0, memory::reservation));
        let source = sherd_core::io::read_mesh(&entry.path)?;
        let display = scene::display_mesh(&source, targets[n]);
        drop(source);
        drop(permit);
        std::fs::write(&glb, scene::fragment_glb(name, &display)).map_err(|e| io_failure(&glb, &e))?;
        std::fs::write(&seg, scene::fragment_glb(name, &segmentation_mesh(fragment, targets[n])))
            .map_err(|e| io_failure(&seg, &e))?;
        thumbnail(fragment, &png)?;
        (display.colors.is_some(), display.faces.len())
    };
```

`targets` = `scene::face_targets(&areas, scene::DEFAULT_FACES)` with `areas[n]` = the sum of `fragment.mesh.face_areas` as `f64` — the same expression `run_with` uses for `scene.glb`. `previous` is `index.json` read into a `BTreeMap<String, FragmentInfo>` when it exists and parses, empty otherwise. Warnings: `ThicknessOutlier { thickness, median }` when `(thickness / median - 1.0).abs() > THICKNESS_OUTLIER`; `NotWatertight` when `!fragment.watertight`. After each fragment: `context.emitter.emit(&Event::FragmentReady(info.clone()))` and `context.watch.advance("display", done.fetch_add(1, Ordering::Relaxed) + 1, fragments.len())`.

In `worker/mod.rs`: `mod prepare; pub use prepare::{INDEX_FILE, THICKNESS_OUTLIER};` and `Job::Prepare(job) => prepare::prepare(&job, context),`.

- [ ] **Step 2: Compile**

Run: `cargo check -p sherd-app-core`
Expected: no errors. (The job is exercised end to end by Task 8's test, on the slab, through a real process; a second, in-process slab run here would buy nothing that one does not.)

- [ ] **Step 3: Commit**

```bash
cargo fmt -p sherd-app-core
git add crates/sherd-app-core
git commit -m "M2.6: Prepare -- the cache a run starts from, and from the same pass a thumbnail, two display meshes and the warnings"
```

---

### Task 7: the `Run` job, and `candidates.json`

**Files:**
- Create: `crates/sherd-app-core/src/worker/run.rs`
- Modify: `src/worker/mod.rs` (dispatch), `src/protocol.rs` (`CandidateRow`)

**Interfaces:**
- Consumes: `pipeline::{run_with, set_threads, RunOptions}`, `session::MatchState`, `sherd_backend::resolve`, `RunSpec::{params, backend}`.
- Produces: `pub(crate) fn run(job: &RunJob, context: &Context) -> Result<Event, Failure>`; files in `<workspace>/runs/<run_id>/`: everything `run_with` writes with meshes, previews and review images off, `match.state`, and `candidates.json`.
- `protocol::CandidateRow { a, b: String, pose: [[f64; 4]; 4], score: f64, scores: Scores, tier: Tier, used: bool, evidence: Option<Evidence> }` (`Scores`, `Tier`, `Evidence` are the engine's serialisable types); `pub const CANDIDATES_FILE: &str = "candidates.json"; pub const MATCH_STATE_FILE: &str = "match.state"; pub const REJECTED_PER_FRAGMENT: usize = 10;`
- `pub(crate) fn assembly_dto(names: &[String], groups: &[Vec<FragId>], poses: &[Matrix4<f64>], used: &[(FragId, FragId)], refined: bool) -> AssemblyDto`

- [ ] **Step 1: Write the failing test for the index's selection rule**

In `run.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use sherd_core::tiers::Tier;

    /// A §4: every confirmed and probable join, and per fragment its ten best rejected — a
    /// fragment's worksheet, not the 54 000 rows of `report.json`.
    #[test]
    fn the_index_keeps_every_live_candidate_and_each_fragments_best_rejected() {
        // (a, b, score, tier): fragment 0 has 12 rejected candidates against partners 1..=12
        let mut rows: Vec<(u32, u32, f64, Tier)> =
            (1..=12).map(|b| (0, b, f64::from(b), Tier::Rejected)).collect();
        rows.push((0, 13, 0.5, Tier::Probable));
        rows.push((1, 2, 0.1, Tier::Confirmed));
        let kept = select(&rows, 10);
        let kept_rows: Vec<_> = kept.iter().map(|&i| rows[i]).collect();
        assert!(kept_rows.contains(&(0, 13, 0.5, Tier::Probable)));
        assert!(kept_rows.contains(&(1, 2, 0.1, Tier::Confirmed)));
        // fragment 0's ten best rejected are scores 12 down to 3; but (0,1) and (0,2) are also
        // among fragment 1's and 2's own best, so they stay: the index is a union of worksheets.
        let rejected_of_zero: Vec<f64> =
            kept_rows.iter().filter(|r| r.3 == Tier::Rejected).map(|r| r.2).collect();
        assert_eq!(rejected_of_zero.len(), 12);
        assert!(kept.windows(2).all(|w| w[0] < w[1]), "in candidate order, each once");
        assert_eq!(select(&rows, 0).len(), 2, "with no worksheet only the live candidates stay");

        // Where the cap bites: twelve rejected poses of ONE pair are in both fragments' lists, and
        // both lists keep the same ten best.
        let one_pair: Vec<(u32, u32, f64, Tier)> =
            (1..=12).map(|k| (0, 1, f64::from(k), Tier::Rejected)).collect();
        let kept: Vec<f64> = select(&one_pair, 10).iter().map(|&i| one_pair[i].2).collect();
        assert_eq!(kept, (3..=12).map(f64::from).collect::<Vec<_>>());
    }
}
```

with the rule as a pure function over `(a, b, score, tier)` so it needs no engine types to test:

```rust
/// Which candidates the index keeps, as ascending indices: every confirmed and probable one, and
/// for each fragment the `per_fragment` best-scoring rejected candidates it takes part in.
pub(crate) fn select(rows: &[(u32, u32, f64, Tier)], per_fragment: usize) -> Vec<usize> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut keep: BTreeSet<usize> = BTreeSet::new();
    let mut rejected: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &(a, b, _, tier)) in rows.iter().enumerate() {
        if tier == Tier::Rejected {
            rejected.entry(a).or_default().push(i);
            rejected.entry(b).or_default().push(i);
        } else {
            keep.insert(i);
        }
    }
    for list in rejected.values_mut() {
        list.sort_by(|&x, &y| rows[y].2.total_cmp(&rows[x].2).then(x.cmp(&y)));
        keep.extend(list.iter().take(per_fragment));
    }
    keep.into_iter().collect()
}
```

Run: `cargo test -p sherd-app-core --lib worker::run 2>&1 | tail -5` — expected: compile error first, then `1 passed` once the function is in.

- [ ] **Step 2: Implement `run`**

```rust
//! `Run` (A §2.2): the whole pipeline into the run's folder, keeping the match (A §3.1) and
//! writing the window's index of it — and none of the heavy outputs, which Export writes on
//! demand (A §4): a run here is tens of megabytes, which is what makes a history affordable.
```

The function:

1. `pipeline::set_threads(job.spec.workers)` — an `Err` here means the pool exists already; ignore it (the worker is a fresh process, and in-process tests share a pool).
2. `let resolved = sherd_backend::resolve(job.spec.backend(), job.spec.adapter.as_deref(), job.spec.gpu_memory_gb).map_err(|e| Failure::new(FailKind::Gpu, format!("{e:#}")))?;`
3. `let dir = job.workspace.join("runs").join(&job.run_id);` — `create_dir_all`, `Disk` on failure.
4. Options:

```rust
    let options = RunOptions {
        target_faces: job.spec.target_faces,
        params: job.spec.params(),
        preview: false,
        write_meshes: false,
        review_images: false,
        cache: Some(job.workspace.join("cache")),
        workers: job.spec.workers,
        backend: resolved.backend,
        adapter: resolved.adapter.clone(),
        memory: job.spec.memory_gb.map_or_else(Budget::default_for_machine, Budget::gigabytes),
        watch: context.watch.clone(),
        constraints: job.constraints.clone(),
        match_state: Some(dir.join(MATCH_STATE_FILE)),
        excluded: job.excluded.clone(),
        ..RunOptions::default()
    };
    let summary = pipeline::run_with(&job.input, &dir, &options, resolved.engine)?;
```

   `run_with` fails with a read error naming the folder when fewer than two meshes are found; map that one case to `TooFew` by checking `discover_excluding(&job.input, &job.excluded)?.len() < 2` **before** the call.
5. `candidates.json`: load the `MatchState` just saved (it carries the tier **evidence**, which `RunSummary` does not); build `rows` from `summary.candidates` (their `tier` is the final one — after constraints and the object round — and `used` comes from `summary.used`); `select(&rows, REJECTED_PER_FRAGMENT)`; write `Vec<CandidateRow>` with `atomic::write_json`. The evidence of candidate `i` is `state.tiers.as_ref().and_then(|t| t.evidence.get(i).cloned().flatten())` **only when** `state.candidates.len() == summary.candidates.len()`; a run with pinned constraints has the same length by construction (A §3.1), but if they differ, write `evidence: None` rather than misattribute a row.
6. Emit `Event::Assembly(assembly_dto(&summary.names, &summary.groups, &summary.poses, &summary.used, options.refine))`.
7. Return `Event::Done { counts: Some(counts), engine: Some(EngineInfo { core_version, algo_ref, commit, backend: options.backend_label() }), params: Some(options.params) }` with `counts` from the summary: `confirmed`/`probable` by counting `Tier` over `summary.candidates` (with the tier off, `confirmed` = accepted), `groups` = groups of two or more, `unassembled` = fragments in groups of one, `timings` from `summary.timings` in order (iterate the `Ordered<f64>`; if it has no public iterator, serialise it to a `serde_json::Value` map — it serialises as an object in finish order).

`assembly_dto`: `poses` keyed by name with rows `std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))`; `unplaced` empty for a run (it is `reassemble`'s, milestone 5).

In `worker/mod.rs`: `mod run; pub use run::{CANDIDATES_FILE, MATCH_STATE_FILE, REJECTED_PER_FRAGMENT};` and `Job::Run(job) => run::run(&job, context),`.

- [ ] **Step 3: Run the crate's tests**

Run: `cargo test -p sherd-app-core --lib 2>&1 | tail -5`
Expected: `18 passed`.

- [ ] **Step 4: Commit**

```bash
cargo fmt -p sherd-app-core
git add crates/sherd-app-core
git commit -m "M2.7: Run -- the pipeline into a run folder with its match kept and the window's index of it, and nothing heavy"
```

---

### Task 8: the host, and the whole thing end to end

**Files:**
- Create: `crates/sherd-app-core/src/host.rs`, `crates/sherd-app-core/tests/worker_e2e.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
  - `host::WorkerCommand { program: PathBuf, args: Vec<String> }`
  - `host::Worker::{spawn(command: &WorkerCommand, job: &Job, log: Option<&Path>) -> Result<Worker>, cancel(&mut self), kill(&mut self), next(&mut self) -> Option<HostEvent>}`; `enum HostEvent { Event(Event), Exited { code: Option<i32> } }`
  - `host::Outcome { Done { counts, engine, params }, Failed { kind, message } }` and `host::drive(worker: &mut Worker, on_event: impl FnMut(&Event)) -> Outcome` — consumes events until the process has exited; a process that exits without `Done`/`Failed` is `Failed { kind: Crashed, message: "exit code …" }`; a `Hello` with another `protocol` is `Failed { kind: Protocol }` and the worker is killed
  - `host::start_run(ws: &Workspace, command: &WorkerCommand, spec: &RunSpec, constraints: Option<Constraints>, carried_from: Option<String>, now: chrono::DateTime<chrono::Local>) -> Result<(RunFile, Worker)>` — scans the input, picks the id, writes `run.json` (`running`), spawns
  - `host::finish_run(ws: &Workspace, run: &mut RunFile, outcome: &Outcome, now) -> Result<()>` and `host::save_assembly(run_dir: &Path, assembly: &AssemblyDto) -> Result<()>` (`ASSEMBLY_FILE = "assembly.json"`)
  - `host::prepare_job(ws: &Workspace, spec: &RunSpec) -> Result<Job>`

`Worker::spawn`: `Command::new(&command.program).args(&command.args)` with all three stdio piped; on Windows add `.creation_flags(0x0800_0000)` (`CREATE_NO_WINDOW`) under `#[cfg(windows)]` with `use std::os::windows::process::CommandExt;`. Write the job line to stdin and **keep** stdin (dropping it cancels). One thread reads stdout lines → `serde_json::from_str::<Event>` → channel (`std::sync::mpsc`); a line that does not parse is appended to the log and skipped. One thread copies stderr to `log` (append mode), or discards it when `log` is `None`. `next` blocks on the channel; when the channel closes it `wait()`s the child and yields `Exited` once, then `None`. `cancel` writes `{"request":"cancel"}\n`; `kill` kills and waits.

- [ ] **Step 1: Write the end-to-end tests**

`crates/sherd-app-core/tests/worker_e2e.rs`:

```rust
//! A §11: the worker driven headless, through a real process and real pipes, on `fixtures/slab`.

use std::path::{Path, PathBuf};

use sherd_app_core::host::{self, Outcome, Worker, WorkerCommand};
use sherd_app_core::protocol::{BackendChoice, Event, FailKind, FragmentInfo, RunSpec};
use sherd_app_core::run::{RunFile, RunStatus};
use sherd_app_core::workspace::Workspace;

fn slab() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
}

fn command() -> WorkerCommand {
    WorkerCommand { program: env!("CARGO_BIN_EXE_sherd-engine-worker").into(), args: Vec::new() }
}

fn workspace(tag: &str) -> Workspace {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("e2e-{tag}"));
    std::fs::remove_dir_all(&root).ok();
    let mut ws = Workspace::create(&root).expect("a workspace");
    ws.set_input(&slab()).expect("the slab is linked");
    ws
}

fn cpu() -> RunSpec {
    RunSpec { backend: BackendChoice::Cpu, ..RunSpec::default() }
}

fn now() -> chrono::DateTime<chrono::Local> {
    chrono::Local::now()
}

#[test]
fn a_collection_is_prepared_and_run_and_the_workspace_holds_what_a_4_says() {
    let ws = workspace("prepare-run");

    // Prepare
    let job = host::prepare_job(&ws, &cpu()).unwrap();
    let mut worker = Worker::spawn(&command(), &job, None).unwrap();
    let mut ready: Vec<FragmentInfo> = Vec::new();
    let outcome = host::drive(&mut worker, |event| {
        if let Event::FragmentReady(info) = event {
            ready.push(info.clone());
        }
    });
    assert!(matches!(outcome, Outcome::Done { .. }), "{outcome:?}");
    ready.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(ready.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["pieceA", "pieceB"]);
    for name in ["pieceA", "pieceB"] {
        for file in [format!("{name}.glb"), format!("{name}.seg.glb"), format!("{name}.png")] {
            assert!(ws.fragments_dir().join(&file).is_file(), "{file}");
        }
        assert!(ws.cache_dir().join(format!("{name}.sherd")).is_file());
    }
    let index: Vec<FragmentInfo> =
        sherd_app_core::atomic::read_json(&ws.fragments_dir().join("index.json")).unwrap();
    assert_eq!(index.len(), 2);
    assert!(index[0].stats.faces > 0 && index[0].display_faces > 0);

    // Run
    let (mut run, mut worker) = host::start_run(&ws, &command(), &cpu(), None, None, now()).unwrap();
    let dir = ws.run_dir(&run.id);
    assert_eq!(RunFile::load(&dir).unwrap().status, RunStatus::Running);
    let mut stages: Vec<String> = Vec::new();
    let mut matching_ended = false;
    let outcome = host::drive(&mut worker, |event| match event {
        Event::Stage { name } => stages.push(name.clone()),
        Event::Progress { stage, done, total } if stage == "matching" => matching_ended = done == total,
        Event::Assembly(assembly) => host::save_assembly(&dir, assembly).unwrap(),
        _ => {}
    });
    host::finish_run(&ws, &mut run, &outcome, now()).unwrap();

    let run = RunFile::load(&dir).unwrap();
    assert_eq!(run.status, RunStatus::Done, "{outcome:?}");
    assert!(matching_ended, "matching reported its last pair");
    assert!(stages.iter().any(|s| s == "matching") && stages.iter().any(|s| s == "tiers"));
    assert_eq!(run.input.files.len(), 2);
    let counts = run.counts.expect("a finished run has counts");
    assert_eq!((counts.fragments, counts.pairs), (2, 1));
    // the shipped rule confirms nothing on this slab: its join is probable and nothing is placed
    assert_eq!((counts.confirmed, counts.groups, counts.unassembled), (0, 0, 2));
    assert!(counts.probable >= 1);
    assert_eq!(run.engine.unwrap().backend, "cpu");
    for file in ["match.state", "candidates.json", "assembly.json", "report.json", "transforms.json", "engine.log"] {
        assert!(dir.join(file).is_file(), "{file}");
    }
    for heavy in ["placed", "scene.glb", "viewer.html", "review"] {
        assert!(!dir.join(heavy).exists(), "a run in the app writes no {heavy}");
    }
}

#[test]
fn a_cancelled_run_ends_as_cancelled() {
    let ws = workspace("cancel");
    let (mut run, mut worker) = host::start_run(&ws, &command(), &cpu(), None, None, now()).unwrap();
    worker.cancel();
    let outcome = host::drive(&mut worker, |_| {});
    assert!(matches!(outcome, Outcome::Failed { kind: FailKind::Cancelled, .. }), "{outcome:?}");
    host::finish_run(&ws, &mut run, &outcome, now()).unwrap();
    assert_eq!(RunFile::load(&ws.run_dir(&run.id)).unwrap().status, RunStatus::Cancelled);
}

/// A §2.1: a window that dies leaves no orphan — the end of stdin is a cancel.
#[test]
fn a_worker_whose_host_goes_away_stops_by_itself() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let ws = workspace("orphan");
    let job = host::prepare_job(&ws, &cpu()).unwrap();
    let mut child = Command::new(command().program)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{}", serde_json::to_string(&job).unwrap()).unwrap();
    drop(stdin);
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(2), "cancelled, by the end of its input");
}
```

Why the cancel test is not a race: the host writes the job line and the cancel line back to back before the worker has read either; the worker's reader thread finds the cancel already buffered, while the main thread still has to resolve a backend, list the folder and preprocess two scans before `match_all` checks the flag ahead of the slab's one pair.

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sherd-app-core --test worker_e2e 2>&1 | tail -5`
Expected: compile error — `host` does not exist.

- [ ] **Step 3: Implement `host.rs`**

As the Interfaces block specifies. `start_run`:

```rust
    let input = ws.input().ok_or_else(|| AppError::Worker("the input folder is not available".into()))?;
    let snapshot = snapshot::scan(&input, &ws.file().excluded)?;
    let existing: Vec<String> = run::list(&ws.runs_dir())?.into_iter().map(|r| r.id).collect();
    let id = run::new_id(&existing, now);
    let mut file = RunFile::new(&id, serde_json::to_value(spec).map_err(json)?, snapshot, run::timestamp(now));
    file.carried_from = carried_from;
    file.save(&ws.run_dir(&id))?;
    let job = Job::Run(RunJob { workspace: ws.root().to_owned(), input, run_id: id.clone(),
        excluded: ws.file().excluded.clone(), spec: spec.clone(), constraints });
    let worker = Worker::spawn(command, &job, Some(&ws.run_dir(&id).join(ENGINE_LOG)))?;
```

`finish_run` maps `Outcome::Done` → `RunStatus::Done` with `counts`, `engine`, `params`; `Failed { kind: Cancelled }` → `RunStatus::Cancelled`; any other `Failed` → `RunStatus::Failed { kind, message }`; sets `finished`, saves. `prepare_job` builds a `PrepareJob` from the workspace and the spec (`target_faces`, `seed`, `memory_gb`, `workers`).

- [ ] **Step 4: Run the end-to-end tests**

Run: `cargo test -p sherd-app-core --test worker_e2e 2>&1 | tail -8`
Expected: `3 passed`. They spawn real processes that each preprocess the slab; allow a few minutes on a cold build. If `a_cancelled_run_ends_as_cancelled` finishes as `Done`, the cancel was read too late: check that `Worker::spawn` writes the job line and that `cancel()` is called before `drive` — do not add sleeps.

- [ ] **Step 5: Commit**

```bash
cargo fmt -p sherd-app-core
git add crates/sherd-app-core
git commit -m "M2.8: the host -- a worker's events, its log and its run's files; the slab prepared, run, cancelled and orphaned, headless"
```

---

### Task 9: the milestone gate, once

- [ ] **Step 1: Format and lints, both shapes**

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings
cargo check -p sherd-app-core --no-default-features
```

Expected: clean. Fix what is found in the code it is found in; an `#[allow]` needs a `reason`.

- [ ] **Step 2: This milestone's tests, and the one core test file Task 1 touched**

```bash
cargo test -p sherd-app-core 2>&1 | tail -12
cargo test -p sherd-core --lib export 2>&1 | tail -4
```

Expected: all `ok`.

- [ ] **Step 3: The CLI's byte-for-byte gate** (Task 1 moved code on `scene.glb`'s path). Run in the background — 4–7 minutes:

```bash
cargo test -p sherd-cli --test run_cli 2>&1 | tail -5
```

Expected: `12 passed; 0 failed; 2 ignored`.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A crates Cargo.toml Cargo.lock
git commit -m "M2.9: the milestone's gate -- fmt, clippy in both shapes, the app core's tests, the CLI's byte-for-byte tests"
```

---

## What the next plan can rely on

| need of milestone 3 (the Tauri shell and the Вход mode) | provided by |
|---|---|
| create / open a workspace, link the input, exclude a fragment | `workspace::Workspace` |
| the input's file list at once, and «устарел» | `snapshot::{scan, diff}` |
| prepare with progress, thumbnails and warnings arriving one by one | `host::prepare_job` + `host::Worker` + `Event::{Stage, Progress, FragmentReady}`; `fragments/index.json` for a workspace reopened |
| show one fragment in 3D, scan colours or «излом красным» | `fragments/<name>.glb`, `fragments/<name>.seg.glb` |
| run history and interrupted runs | `run::{list, mark_interrupted}` |
| the engine role of the app's own binary | `worker::serve` behind `--engine-worker` |
| TypeScript types | `protocol` — `ts-rs` derives are added with the frontend, milestone 3 |
