# Desktop App, Milestone 3 — the shell, the frame and the «Вход» mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A window. Create or open a workspace, link a folder of scans, watch it being prepared with thumbnails arriving one by one, look at any fragment in 3D — its scan or its fracture faces in red — read what is wrong with it, and leave it out of the assembly.

**Architecture:** `apps/desktop/src-tauri` is a thin Tauri 2 crate over `sherd-app-core`: commands return one `WorkspaceView` DTO built in `sherd-app-core::view` (no Tauri there, so it is unit-tested), jobs are `host::Worker`s driven on a thread that forwards every `Event` to the window. The same executable is the worker when started with `--engine-worker`. The frontend is React + TypeScript; its types are generated from Rust by `ts-rs`; the 3D viewer is a React-free three.js class; status is **derived** by a pure function. With no Tauri around it (a plain browser) the frontend runs on a mock IPC layer, which is how its screens are looked at without a test harness.

**Tech Stack:** Tauri `=2.11.6` (`tauri-build =2.6.3`, `tauri-plugin-dialog =2.7.3`, `tauri-plugin-opener =2.5.5`), `ts-rs =12.0.1`; Node 24, pnpm 9; `react`/`react-dom 19.3.0`, `typescript 7.0.2`, `vite 8.3.0`, `@vitejs/plugin-react 6.1.1`, `tailwindcss`/`@tailwindcss/vite 4.3.3`, `three 0.186.0` (`@types/three 0.186.0`), `zustand 5.0.15`, `i18next 26.4.2`, `react-i18next 17.0.14`, `@tanstack/react-virtual 3.14.13`, `lucide-react 1.47.0`, `clsx 2.1.1`, `@tauri-apps/api 2.11.1`, `@tauri-apps/cli 2.11.5`, `@tauri-apps/plugin-dialog 2.7.3`, `@tauri-apps/plugin-opener 2.5.5`, `vitest 5.0.1`, `eslint 10.11.0`, `typescript-eslint 8.70.0`. Spec: `docs/superpowers/specs/2026-09-20-desktop-app-design.md` (`A §n`); the screens the user chose: `docs/superpowers/specs/desktop-app-mockups/{main-layout,states-and-input}.html` — **open them; variant A of `main-layout.html` with both side panes collapsible is the frame, and part 3 of `states-and-input.html` is the «Вход» mode.**

## Global Constraints

- **Toolchain.** Every shell that runs cargo starts with `export PATH="$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin:$PATH"` (bare `cargo` is Homebrew's 1.88 and fails the MSRV). JavaScript uses **pnpm**, run from `apps/desktop/`. Exact versions as listed above, no `^`.
- **The Tauri crate is not a member of the root Cargo workspace.** Root `Cargo.toml` gets `exclude = ["apps/desktop/src-tauri"]`; the crate has its own empty `[workspace]` table, its own `Cargo.lock`, and `apps/desktop/src-tauri/.cargo/config.toml` points `build.target-dir` at the root `target/` so the engine is not compiled twice. Reason: the repository's CI builds `--workspace` on Linux runners that have no WebKit; the core's CI must not start depending on a GUI toolkit.
- **The engine is optimised in the app's dev profile** (`[profile.dev.package.sherd-core]`/`sherd-gpu` `opt-level = 3`): `tauri dev` on a real collection at `opt-level = 0` would be unusable.
- **Verification budget (A §11).** Run only what a task lists. No Playwright, no WebDriver, no component tests: Vitest on pure logic only. No `cargo test --workspace`, no clippy inside a task — Task 9 is the gate. Nothing automated runs on anything but `fixtures/slab`.
- **No panics on data from outside** (files, IPC arguments, events) in Rust; in TypeScript no non-null assertions on IPC data and no `any`.
- **Every user-visible string** goes through i18next, with both `ru.json` and `en.json` filled. Russian is the primary wording (the mock-ups' own words); English is a translation, not a placeholder.
- **Ownership (A §2.1)** as before: only host-side code writes workspace state.
- **The window never reads `report.json` or `match.state`**, never loads a source scan, and gets files only through Tauri's asset protocol scoped to the open workspace.
- **Render on demand.** The viewer draws when something changed, never in a free-running loop: the engine may be using the same GPU.
- **Commits:** branch `desktop-app`, subject `M3.<task>: <sentence>`, trailer as your own system instructions give it.
- **Do not touch** `sherd_refit/`, `crates/sherd-parity/`, `fixtures/`, `crates/sherd-core/`.

## What milestone 2 provides (tree at `fc252db`)

```rust
// sherd_app_core
pub enum AppError { Io{path,source}, Json{path,source}, NotAWorkspace{path}, Locked{path}, Version{path,found,expected}, Core(sherd_core::Error), Worker(String) }
// workspace
Workspace::{create(&Path), open(&Path), root(), file() -> &WorkspaceFile, input() -> Option<PathBuf>, set_input(&mut self, &Path), set_excluded(&mut self, &str, bool), set_last_spec(&mut self, serde_json::Value), cache_dir(), fragments_dir(), runs_dir(), run_dir(id), exports_dir()}
WorkspaceFile { version, input: Option<InputRef{absolute, relative}>, excluded: BTreeSet<String>, last_spec: Option<Value> }
// snapshot
scan(input: &Path, excluded: &BTreeSet<String>) -> Result<InputSnapshot>;  diff(then, now) -> StaleDiff;  FileStamp { name, file, size, mtime_ms }
// run
RunFile { version, id, created, finished, status: RunStatus, spec: Value, params: Option<Params>, input: InputSnapshot, engine, counts, carried_from };  list(&Path) -> Result<Vec<RunFile>>;  mark_interrupted(&Path, finished: &str) -> Result<Vec<String>>;  timestamp(now) -> String
RunStatus = Running | Done | Cancelled | Failed{kind, message} | Interrupted      // serde tag = "state", snake_case
// protocol
Job, PrepareJob, RunJob, Request, RunSpec (Default = the CLI's defaults), BackendChoice, Preset,
Event = Hello{..} | Info{backends, selftest} | Stage{name} | Progress{stage, done, total} | FragmentReady(FragmentInfo) | Assembly(AssemblyDto) | Done{counts, engine, params} | Failed{kind, message}   // serde tag = "event", snake_case
FragmentInfo { name, file, size, mtime_ms, stats: FragmentStats, warnings: Vec<Warning>, coloured, display_faces };  Warning = ThicknessOutlier{thickness, median} | NotWatertight   // tag = "warning"
// worker
pub fn serve(input: impl BufRead + Send + 'static, output: impl Write + Send + 'static) -> i32;   INDEX_FILE = "index.json"
// host
WorkerCommand { program, args };  Worker::{spawn(&WorkerCommand, &Job, log: Option<&Path>), canceller() -> Canceller, kill(), next()};  Canceller::cancel(&self)  // Clone + Send + Sync
Outcome = Done{counts, engine, params} | Failed{kind, message};  drive(&mut Worker, impl FnMut(&Event)) -> Outcome;  prepare_job(&Workspace, &RunSpec) -> Result<Job>
```

On disk after `Prepare`: `<ws>/fragments/<name>.glb` (scan colours or clay, source coordinates), `<name>.seg.glb` (grey shell, red fracture, as vertex colours), `<name>.png` (320×240, dark background), `index.json` = `Vec<FragmentInfo>`. Stages a `Prepare` reports: `preprocess`, `display`.

## File Structure

```
Cargo.toml                                   + exclude
crates/sherd-app-core/
  Cargo.toml                                 + feature `ts`, optional ts-rs
  src/view.rs                        NEW     WorkspaceView and how it is built; `prepared`
  src/worker/mod.rs                          + serve_stdio()
  src/{protocol,run,snapshot,decisions,host}.rs   + cfg_attr(feature = "ts", derive(TS))
apps/desktop/
  package.json  pnpm-lock.yaml  tsconfig.json  vite.config.ts  eslint.config.js  index.html
  public/mock/                               pieceA.glb, pieceA.seg.glb, pieceA.png (from the slab)
  src/
    main.tsx  App.tsx  styles.css
    ipc/bindings/*.ts              GENERATED by ts-rs, committed
    ipc/{api.ts, tauri.ts, mock.ts, index.ts}
    state/{status.ts, status.test.ts, workspace.ts, jobs.ts, ui.ts}
    i18n/{index.ts, ru.json, en.json}
    viewer/{FragmentViewer.ts, matrix.ts, matrix.test.ts, FragmentView.tsx}
    app/{Welcome.tsx, Frame.tsx, TopBar.tsx, StatusLine.tsx, Banner.tsx, shortcuts.ts}
    modes/input/{InputLeft.tsx, InputRight.tsx, InputCentre.tsx, format.ts, format.test.ts}
    ui/{Button.tsx, Chip.tsx, IconButton.tsx, Meter.tsx}
  src-tauri/
    Cargo.toml  Cargo.lock  build.rs  tauri.conf.json  .cargo/config.toml
    capabilities/default.json  icons/
    src/{main.rs, state.rs, error.rs, recent.rs, commands.rs, jobs.rs}
```

---

### Task 1: `WorkspaceView`, and TypeScript types from Rust

**Files:**
- Create: `crates/sherd-app-core/src/view.rs`
- Modify: `crates/sherd-app-core/Cargo.toml`, `src/lib.rs`, `src/protocol.rs`, `src/run.rs`, `src/snapshot.rs`, `src/decisions.rs`, `src/host.rs`, `src/worker/mod.rs`, `src/bin/sherd-engine-worker.rs`; root `Cargo.toml` (`ts-rs` in `[workspace.dependencies]`)
- Generated: `apps/desktop/src/ipc/bindings/*.ts`

**Interfaces:**
- Produces:
  - `view::{WorkspaceView { root: String, name: String, input: InputView, excluded: Vec<String>, files: Vec<FileStamp>, fragments: Vec<FragmentInfo>, prepared: bool, runs: Vec<RunView>, fragments_dir: String, job: Option<JobView> }, InputView { linked: bool, path: Option<String>, available: bool }, RunView { run: RunFile, stale: StaleDiff }, JobView { kind: JobKind, run_id: Option<String> }, JobKind::{Prepare, Run}}` — all `Clone + Debug + PartialEq + Serialize + Deserialize`
  - `view::build(ws: &Workspace, job: Option<JobView>) -> Result<WorkspaceView>`; `view::prepared(files: &[FileStamp], excluded: &BTreeSet<String>, fragments: &[FragmentInfo]) -> bool`
  - `worker::serve_stdio() -> i32` — initialises `tracing` to stderr and serves stdin/stdout; the bin target and the Tauri binary both call it
  - feature `ts` on `sherd-app-core`; `#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]` on: `RunSpec, BackendChoice, Preset, Event, FragmentInfo, Warning, AssemblyDto, GroupDto, JoinDto, UnplacedDto, CandidateRow, FailKind, RunStatus, RunFile, EngineInfo, StageTime, RunCounts, FileStamp, InputSnapshot, StaleDiff, DecisionsFile, Decision, Verdict, Outcome` and the five `view` types

- [ ] **Step 1: Write the failing tests** (in `view.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FragmentInfo;

    fn stamp(name: &str, size: u64) -> FileStamp {
        FileStamp { name: name.into(), file: format!("{name}.ply"), size, mtime_ms: 1 }
    }

    fn info(name: &str, size: u64) -> FragmentInfo {
        let stats: sherd_core::report::FragmentStats = serde_json::from_value(serde_json::json!({
            "name": name, "faces": 1, "orig_faces": 1, "orig_vertices": 1, "thickness": 1.0,
            "thickness_mode": 1.0, "resolution": 1.0, "watertight": true, "extent": [1.0, 1.0, 1.0],
            "area": 1.0, "fracture_area_fraction": 0.1
        }))
        .unwrap();
        FragmentInfo {
            name: name.into(), file: format!("{name}.ply"), size, mtime_ms: 1, stats,
            warnings: Vec::new(), coloured: false, display_faces: 1,
        }
    }

    /// A §5: «preparing» ends when every scan that takes part has an up-to-date row — an excluded
    /// scan needs none, a changed scan's old row does not count, and no scans is not prepared.
    #[test]
    fn prepared_means_every_included_scan_has_a_current_row() {
        let files = [stamp("a", 10), stamp("b", 20)];
        let none = BTreeSet::new();
        assert!(prepared(&files, &none, &[info("a", 10), info("b", 20)]));
        assert!(!prepared(&files, &none, &[info("a", 10)]), "b has no row");
        assert!(!prepared(&files, &none, &[info("a", 10), info("b", 21)]), "b changed since");
        assert!(prepared(&files, &BTreeSet::from(["b".to_owned()]), &[info("a", 10)]));
        assert!(!prepared(&[], &none, &[]), "nothing to prepare is not «prepared»");
    }

    #[test]
    fn a_view_of_a_fresh_workspace_says_no_input_and_survives_a_missing_one() {
        let root = std::env::temp_dir().join(format!("sherd-view-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let scans = root.join("scans");
        std::fs::create_dir_all(&scans).unwrap();
        std::fs::write(scans.join("a.ply"), b"x").unwrap();
        let mut ws = crate::workspace::Workspace::create(&root.join("karas")).unwrap();

        let empty = build(&ws, None).unwrap();
        assert_eq!(empty.name, "karas");
        assert_eq!(empty.input, InputView { linked: false, path: None, available: false });

        ws.set_input(&scans).unwrap();
        let linked = build(&ws, None).unwrap();
        assert!(linked.input.linked && linked.input.available);
        assert_eq!(linked.files.len(), 1);
        assert!(!linked.prepared && linked.fragments.is_empty() && linked.runs.is_empty());

        std::fs::remove_dir_all(&scans).unwrap();
        let missing = build(&ws, None).unwrap();
        assert!(missing.input.linked && !missing.input.available, "A §5: input missing, not an error");
        assert!(missing.files.is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    /// The mirror exists because `FragmentStats` is the engine's type and cannot derive `TS` here;
    /// this is what keeps the mirror honest.
    #[cfg(feature = "ts")]
    #[test]
    fn the_typescript_mirror_of_fragment_stats_has_the_engines_fields() {
        let real = serde_json::to_value(info("a", 1).stats).unwrap();
        let mirror = serde_json::to_value(crate::protocol::FragmentStatsTs::default()).unwrap();
        let keys = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&real), keys(&mirror));
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sherd-app-core --lib view 2>&1 | tail -5`
Expected: compile error.

- [ ] **Step 3: Implement `view.rs`**

```rust
//! What the window is told about a workspace, in one value (A §5): the frontend derives every
//! state it shows from this and from the job's events, so it never has a second opinion to keep
//! in step with the first.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::protocol::FragmentInfo;
use crate::run::{self, RunFile};
use crate::snapshot::{self, FileStamp, InputSnapshot, StaleDiff};
use crate::worker::INDEX_FILE;
use crate::workspace::Workspace;
use crate::{Result, atomic};
```

`build`: `name` = the root's file name; `input.linked` = `ws.file().input.is_some()`, `available`/`path` from `ws.input()`; `files` = `snapshot::scan(&input, &excluded)?.files` when available, empty otherwise; `fragments` = `index.json` read with `atomic::read_json` when the file exists **and parses**, empty otherwise (a damaged index is re-made by the next `Prepare`, not an error to show); `prepared` as below; `runs` = `run::list(&ws.runs_dir())?`, each with `stale` = `snapshot::diff(&run.input, &now)` where `now` is the current `InputSnapshot` — and an empty diff when the input is not available (A §5: a missing input is its own state, not staleness); `fragments_dir` = the path as a string; `job` as given.

```rust
/// Whether the input needs no `Prepare`: every scan that takes part has a row for the file as it
/// is now. No scans at all is `false` — there is nothing to show as ready.
pub fn prepared(files: &[FileStamp], excluded: &BTreeSet<String>, fragments: &[FragmentInfo]) -> bool {
    let mut included = files.iter().filter(|f| !excluded.contains(&f.name)).peekable();
    included.peek().is_some()
        && included.all(|f| {
            fragments.iter().any(|i| i.name == f.name && i.size == f.size && i.mtime_ms == f.mtime_ms)
        })
}
```

- [ ] **Step 4: `serve_stdio`**

Move the body of `src/bin/sherd-engine-worker.rs`'s `main` into `worker::serve_stdio() -> i32` (tracing to stderr with `EnvFilter`, no ANSI, then `serve(BufReader::new(stdin()), stdout())`), documented as "the engine role of a binary (A §2.1): the bin target of this crate and the app's own executable under `--engine-worker` are this one call". The bin's `main` becomes `std::process::exit(sherd_app_core::worker::serve_stdio())`.

- [ ] **Step 5: The `ts` feature**

Root `Cargo.toml`, under `# --- storage, fixtures, output`:

```toml
ts-rs = { version = "=12.0.1", features = [
    "serde-json-impl",
] } # TypeScript types of the app's wire contract, generated by a test; optional, the engine never sees it
```

`crates/sherd-app-core/Cargo.toml`: `ts = ["dep:ts-rs"]` under `[features]`, `ts-rs = { workspace = true, optional = true }` under `[dependencies]`.

On every type the Interfaces block lists: `#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]`. Fields whose type is the engine's:

| field | attribute |
|---|---|
| `FragmentInfo::stats` | `#[cfg_attr(feature = "ts", ts(as = "FragmentStatsTs"))]` |
| `RunFile::params`, `Event::Done::params`, `Outcome::Done::params` | `#[cfg_attr(feature = "ts", ts(type = "Record<string, unknown> | null"))]` |
| `CandidateRow::scores` | `ts(type = "Record<string, number | boolean | null>")` |
| `CandidateRow::tier` | `ts(type = "\"confirmed\" | \"probable\" | \"rejected\"")` (`Tier` is `serde(rename_all = "lowercase")`) |
| `CandidateRow::evidence` | `ts(type = "Record<string, unknown> | null")` — milestone 5 gives it a real type |

and in `protocol.rs`:

```rust
/// `sherd_core::report::FragmentStats`, field for field, for the TypeScript bindings only: the
/// engine's type cannot derive `TS` from here. `view`'s tests compare the two.
#[cfg(feature = "ts")]
#[derive(Default, serde::Serialize, ts_rs::TS)]
#[ts(export, rename = "FragmentStats")]
pub struct FragmentStatsTs {
    name: String, faces: u64, orig_faces: u64, orig_vertices: u64, thickness: f64,
    thickness_mode: f64, resolution: f64, watertight: bool, extent: [f64; 3], area: f64,
    fracture_area_fraction: f64,
}
```

(one field per line with a doc comment each, in the engine's order). `u64` and `usize` export as `bigint` by default in ts-rs; every integer this contract carries is a count far below 2^53 and arrives through `JSON.parse` as a `number`, so put `#[cfg_attr(feature = "ts", ts(type = "number"))]` on each `u64`/`usize`/`i64` field, or set `TS_RS_LARGE_INT=number` in Step 6's command if ts-rs 12 supports it (`cargo doc -p ts-rs --open` is not needed — read `~/.cargo/registry/src/*/ts-rs-12.0.1/README.md`). The generated files must contain no `bigint`.

- [ ] **Step 6: Generate the bindings**

```bash
mkdir -p apps/desktop/src/ipc/bindings
TS_RS_EXPORT_DIR="$PWD/apps/desktop/src/ipc/bindings" cargo test -p sherd-app-core --features ts --lib 2>&1 | tail -5
ls apps/desktop/src/ipc/bindings | head -40
grep -l bigint apps/desktop/src/ipc/bindings/*.ts
```

Expected: tests pass (the three of this task and ts-rs's own `export_bindings_*`), one `.ts` file per exported type, and the `grep` prints nothing. `Event.ts` must be a discriminated union on `event`, `RunStatus.ts` on `state`, `Warning.ts` on `warning`.

- [ ] **Step 7: The default build is untouched**

Run: `cargo test -p sherd-app-core --lib 2>&1 | tail -3`
Expected: `21 passed` (19 + the two that need no feature).

- [ ] **Step 8: Commit**

```bash
cargo fmt -p sherd-app-core
git add Cargo.toml Cargo.lock crates/sherd-app-core apps/desktop/src/ipc/bindings
git commit -m "M3.1: what the window is told about a workspace, in one value -- and its TypeScript types, generated from the Rust ones"
```

---

### Task 2: the two scaffolds — a Tauri crate that can be the worker, and a frontend that builds

**Files:**
- Create: everything under `apps/desktop/` that the File Structure lists for this task: `package.json`, `tsconfig.json`, `vite.config.ts`, `eslint.config.js`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/styles.css`, `src-tauri/{Cargo.toml, build.rs, tauri.conf.json, .cargo/config.toml, capabilities/default.json, icons/, src/main.rs}`, `apps/desktop/.gitignore`
- Modify: root `Cargo.toml` (`exclude`), root `.gitignore`

**Interfaces:**
- Produces: `pnpm dev` (Vite on port 1420), `pnpm build`, `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm tauri …`; a binary `sherd-desktop` that is the engine worker under `--engine-worker` and the app otherwise.

- [ ] **Step 1: Root manifest**

In `[workspace]` of the root `Cargo.toml`:

```toml
# The desktop shell is a workspace of its own (apps/desktop/src-tauri/Cargo.toml): `--workspace`
# here is what CI builds on Linux runners with no WebKit, and the core's CI must not come to
# depend on a GUI toolkit. It shares this `target/` through its own .cargo/config.toml.
exclude = ["apps/desktop/src-tauri"]
```

Root `.gitignore`: add `node_modules/` and `apps/desktop/dist/`.

- [ ] **Step 2: The Tauri crate**

`apps/desktop/src-tauri/Cargo.toml`:

```toml
[package]
name = "sherd-desktop"
description = "sherd-refit's desktop app: a Tauri 2 shell over sherd-app-core"
version = "0.1.0"
edition = "2024"
rust-version = "1.89"
publish = false

# A workspace of its own; see `exclude` in the repository's root Cargo.toml.
[workspace]

[build-dependencies]
tauri-build = { version = "=2.6.3", features = [] }

[dependencies]
sherd-app-core = { path = "../../../crates/sherd-app-core" }
tauri = { version = "=2.11.6", features = ["protocol-asset"] }
tauri-plugin-dialog = "=2.7.3"
tauri-plugin-opener = "=2.5.5"
serde = { version = "1.0.229", features = ["derive"] }
serde_json = { version = "1.0.151", features = ["float_roundtrip"] }
chrono = { version = "=0.4.45", default-features = false, features = ["clock", "std"] }
tracing = "0.1.44"

# The root workspace's two profile decisions, repeated because profiles are per workspace — and
# one more: the engine itself is optimised in dev, or `tauri dev` on a real collection would match
# pairs at opt-level 0.
[profile.release]
lto = "thin"
codegen-units = 1
debug = "line-tables-only"

[profile.dev.package."*"]
opt-level = 2

[profile.dev.package.sherd-core]
opt-level = 3

[profile.dev.package.sherd-gpu]
opt-level = 3

[lints.rust]
missing_docs = "warn"
missing_debug_implementations = "warn"
unreachable_pub = "warn"
unsafe_code = "deny"
rust_2018_idioms = { level = "warn", priority = -1 }

[lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
todo = "deny"
unimplemented = "deny"
dbg_macro = "deny"
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
doc_markdown = "allow"
needless_pass_by_value = "allow" # Tauri commands take their arguments by value
```

`.cargo/config.toml`:

```toml
# One `target/` for the repository: the engine is compiled once, not once per workspace.
[build]
target-dir = "../../../target"
```

`build.rs`: `fn main() { tauri_build::build() }`.

`tauri.conf.json`:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Sherd Refit",
  "version": "0.1.0",
  "identifier": "io.github.elsvv.sherd-refit",
  "build": {
    "beforeDevCommand": "pnpm dev",
    "devUrl": "http://localhost:1420",
    "beforeBuildCommand": "pnpm build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      { "label": "main", "title": "Sherd Refit", "width": 1360, "height": 860, "minWidth": 960, "minHeight": 600, "dragDropEnabled": true }
    ],
    "security": {
      "csp": "default-src 'self'; img-src 'self' asset: http://asset.localhost blob: data:; connect-src 'self' ipc: http://ipc.localhost asset: http://asset.localhost; style-src 'self' 'unsafe-inline'; worker-src 'self' blob:",
      "assetProtocol": { "enable": true, "scope": [] }
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": ["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.icns", "icons/icon.ico"]
  }
}
```

The asset scope is empty on purpose: `commands.rs` (Task 3) allows exactly the open workspace at runtime.

`capabilities/default.json`:

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "What the one window may ask of the shell",
  "windows": ["main"],
  "permissions": ["core:default", "dialog:allow-open", "opener:allow-reveal-item-in-dir"]
}
```

`src/main.rs`:

```rust
//! sherd-refit's desktop app (A §2): one executable, two roles. Started with `--engine-worker`
//! it is the engine — one job on stdin, events on stdout — and never opens a window; started any
//! other way it is the window, and spawns itself for every job.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// The argument that makes this process the engine.
pub const ENGINE_WORKER: &str = "--engine-worker";

fn main() {
    // Before anything of Tauri's: the worker must not initialise a GUI toolkit it never shows.
    if std::env::args().nth(1).as_deref() == Some(ENGINE_WORKER) {
        std::process::exit(sherd_app_core::worker::serve_stdio());
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("the window could not be opened");
}
```

Icons: `tauri icon` needs a 1024×1024 source. Write one with Python's standard library (no PIL) — a clay disc on a dark rounded square is enough for now — to `src-tauri/icons/source.png`, then `pnpm tauri icon src-tauri/icons/source.png`:

```python
import struct, zlib
S = 1024
def px(x, y):
    dx, dy = x - S / 2, y - S / 2
    inside = abs(dx) < 440 and abs(dy) < 440
    disc = dx * dx + dy * dy < 300 * 300
    return (201, 138, 88, 255) if disc else (38, 38, 42, 255) if inside else (0, 0, 0, 0)
rows = b"".join(b"\x00" + bytes(c for x in range(S) for c in px(x, y)) for y in range(S))
def chunk(tag, data): return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", S, S, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
open("apps/desktop/src-tauri/icons/source.png", "wb").write(png)
```

- [ ] **Step 3: The frontend scaffold**

`package.json` (`"type": "module"`, `"private": true`, `"packageManager": "pnpm@9.12.1"`), scripts:

```json
"scripts": {
  "dev": "vite",
  "build": "tsc --noEmit -p tsconfig.json && vite build",
  "typecheck": "tsc --noEmit -p tsconfig.json",
  "lint": "eslint src",
  "test": "vitest run",
  "tauri": "tauri"
}
```

with the dependencies of the Tech Stack line at exactly those versions (`dependencies`: react, react-dom, three, zustand, i18next, react-i18next, @tanstack/react-virtual, lucide-react, clsx, @tauri-apps/api, @tauri-apps/plugin-dialog, @tauri-apps/plugin-opener; `devDependencies`: the rest, plus `@types/react 19.3.0`, `@types/react-dom 19.3.0`, `@types/three 0.186.0`).

`tsconfig.json`: `"strict": true, "noUncheckedIndexedAccess": true, "exactOptionalPropertyTypes": true, "noImplicitOverride": true, "verbatimModuleSyntax": true, "moduleResolution": "bundler", "module": "ESNext", "target": "ES2022", "jsx": "react-jsx", "lib": ["ES2022", "DOM", "DOM.Iterable"], "resolveJsonModule": true, "skipLibCheck": true, "types": ["vite/client"]`, `"include": ["src"]`.

`vite.config.ts`: plugins `react()` and `tailwindcss()`; `server: { port: 1420, strictPort: true }`, `clearScreen: false`, `build: { target: "es2022", chunkSizeWarningLimit: 1200 }`, and Vitest's `test: { environment: "node", include: ["src/**/*.test.ts"] }`.

`eslint.config.js`: `typescript-eslint`'s `strictTypeChecked` over `src/**/*.{ts,tsx}` with `parserOptions.projectService: true`, ignoring `src/ipc/bindings/**`; rules `@typescript-eslint/no-explicit-any: error`, `@typescript-eslint/no-non-null-assertion: error`.

`src/styles.css` — Tailwind 4 and the app's tokens; every colour in the app is one of these:

```css
@import "tailwindcss";

:root {
  --bg: #f4f4f6;  --panel: #ffffff;  --panel-2: #ececef;  --border: #d4d4da;
  --text: #1c1c1f;  --muted: #6e6e76;  --accent: #0a6ee0;  --accent-text: #ffffff;
  --ok: #2f9e55;  --warn: #d98a0b;  --danger: #d6372f;  --clay: #c98a58;
  --viewport: #26262a;  --viewport-text: #b9b9c2;  --select: rgba(10, 110, 224, 0.12);
  color-scheme: light;
}
:root[data-theme="dark"] {
  --bg: #1b1b1e;  --panel: #242428;  --panel-2: #2e2e33;  --border: #3a3a41;
  --text: #ececf1;  --muted: #94949d;  --accent: #3b8ff5;  --accent-text: #ffffff;
  --ok: #3fb568;  --warn: #f0a530;  --danger: #f0574e;  --clay: #d29a6b;
  --viewport: #161618;  --viewport-text: #8d8d97;  --select: rgba(59, 143, 245, 0.18);
  color-scheme: dark;
}
@theme inline {
  --color-bg: var(--bg);  --color-panel: var(--panel);  --color-panel-2: var(--panel-2);
  --color-border: var(--border);  --color-text: var(--text);  --color-muted: var(--muted);
  --color-accent: var(--accent);  --color-accent-text: var(--accent-text);  --color-ok: var(--ok);
  --color-warn: var(--warn);  --color-danger: var(--danger);  --color-clay: var(--clay);
  --color-viewport: var(--viewport);  --color-viewport-text: var(--viewport-text);  --color-select: var(--select);
}
html, body, #root { height: 100%; }
body { background: var(--bg); color: var(--text); font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; overflow: hidden; user-select: none; -webkit-user-select: none; }
```

The 3D viewport is dark in both themes (thumbnails are rendered on `#292929`, and scans read best against dark).

`src/App.tsx` for now renders the product name centred on `bg-bg`; `src/main.tsx` mounts it in `StrictMode` and imports `./styles.css`.

- [ ] **Step 4: Build both**

```bash
cd apps/desktop && pnpm install && pnpm build && pnpm lint
cd src-tauri && cargo check 2>&1 | tail -3
```

Expected: `vite build` writes `dist/`; `cargo check` finishes (the first one compiles Tauri — several minutes; use `timeout 600000`, and `run_in_background` if it does not fit).

- [ ] **Step 5: The binary is a worker when asked to be**

```bash
cd apps/desktop/src-tauri && cargo build 2>&1 | tail -2
echo '{"job":"info","adapter":null,"selftest":false}' | ../../../target/debug/sherd-desktop --engine-worker | head -3
```

Expected: three JSON lines — `{"event":"hello",…}`, `{"event":"info",…}`, `{"event":"done",…}` — and no window.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml .gitignore apps/desktop
git commit -m "M3.2: a Tauri crate that is the engine under --engine-worker, and a frontend that builds -- outside the core's workspace, inside its target/"
```

(`pnpm-lock.yaml`, `src-tauri/Cargo.lock` and the generated icons are committed; `node_modules/`, `dist/` and `src-tauri/gen/` are not — add `gen/` to `apps/desktop/.gitignore` if Tauri creates it.)

---

### Task 3: the shell's commands — workspaces, the input, exclusions, recent

**Files:**
- Create: `apps/desktop/src-tauri/src/{state.rs, error.rs, recent.rs, commands.rs}`
- Modify: `src/main.rs`

**Interfaces:**
- Produces Tauri commands (names as the frontend invokes them), each `Result<T, CommandError>`:

| command | arguments | returns |
|---|---|---|
| `app_info` | — | `AppInfo { version, core_version, commit }` |
| `recent_list` | — | `Vec<RecentEntry { path, name, opened_at, available }>` |
| `workspace_create` | `path: String` | `WorkspaceView` |
| `workspace_open` | `path: String` | `WorkspaceView` |
| `workspace_close` | — | `()` |
| `workspace_view` | — | `WorkspaceView` |
| `input_link` | `path: String` | `WorkspaceView` |
| `fragment_exclude` | `name: String, excluded: bool` | `WorkspaceView` |

- `CommandError { kind: String, message: String }` — kinds: `locked`, `not_a_workspace`, `version`, `io`, `json`, `engine`, `worker`, `busy`, `no_workspace`
- `state::AppState { workspace: Mutex<Option<Workspace>>, job: Mutex<Option<JobSlot>> }`, `state::JobSlot { kind: JobKind, run_id: Option<String>, canceller: Canceller }`
- `recent::{load(config_dir: &Path) -> Vec<RecentEntry>, touch(config_dir: &Path, root: &Path, now: &str)}` — `recent.json` in Tauri's `app_config_dir`, newest first, at most 12, written with `sherd_app_core::atomic`

Opening a workspace: `Workspace::open` → `run::mark_interrupted(&ws.runs_dir(), &run::timestamp(Local::now()))` (A §4: no worker of ours exists yet, so a `running` on disk is nobody's) → `app.asset_protocol_scope().allow_directory(ws.root(), true)` → `recent::touch` → store → `view::build`. Opening while another workspace is open closes that one first (dropping it releases its lock); refuse with `busy` while a job is running. `workspace_create` on a folder that already is a workspace opens it.

- [ ] **Step 1: Write the failing test** (in `recent.rs`; pure functions, no Tauri runtime)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recent_list_is_newest_first_without_duplicates_and_says_what_is_gone() {
        let dir = std::env::temp_dir().join(format!("sherd-recent-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let (a, b) = (dir.join("ws/a"), dir.join("ws/b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        touch(&dir, &a, "2026-09-20T10:00:00+03:00");
        touch(&dir, &b, "2026-09-20T11:00:00+03:00");
        touch(&dir, &a, "2026-09-20T12:00:00+03:00");
        std::fs::remove_dir_all(&b).unwrap();
        let list = load(&dir);
        assert_eq!(list.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!((list[0].available, list[1].available), (true, false));
        assert_eq!(list[0].opened_at, "2026-09-20T12:00:00+03:00");
        std::fs::remove_dir_all(&dir).ok();
    }
}
```

- [ ] **Step 2: Implement** `error.rs` (`CommandError` + `From<AppError>` mapping each variant to its kind, `Core` → `engine`), `recent.rs`, `state.rs`, `commands.rs`; register in `main.rs` with `.manage(AppState::default())` and `tauri::generate_handler![…]`. A damaged or missing `recent.json` is an empty list, never an error.

- [ ] **Step 3: Test and check**

```bash
cd apps/desktop/src-tauri && cargo test 2>&1 | tail -4 && cargo check 2>&1 | tail -2
```

Expected: `1 passed`; no warnings.

- [ ] **Step 4: Commit**

```bash
git add apps/desktop/src-tauri
git commit -m "M3.3: the shell's commands -- a workspace opened is locked, scoped for the asset protocol, remembered, and its orphaned runs marked"
```

---

### Task 4: jobs in the shell — `Prepare` with its events in the window, and cancel

**Files:**
- Create: `apps/desktop/src-tauri/src/jobs.rs`
- Modify: `src/commands.rs`, `src/main.rs`

**Interfaces:**
- Produces commands `prepare_start() -> Result<(), CommandError>`, `job_cancel() -> Result<(), CommandError>`; events to the window: `engine:event` with payload `{ job: JobKind, run_id: string | null, event: Event }`, `engine:finished` with payload `{ job: JobKind, run_id: string | null, outcome: Outcome, view: WorkspaceView }`.
- `jobs::worker_command() -> Result<WorkerCommand, CommandError>` = `current_exe()` + `["--engine-worker"]`.

`prepare_start`: `busy` if a job is in the slot; `no_workspace` if none is open; build the job with `host::prepare_job(ws, &spec)` where `spec` is `ws.file().last_spec` parsed as a `RunSpec` when it parses and `RunSpec::default()` otherwise; `Worker::spawn(&worker_command()?, &job, Some(&ws.root().join("prepare.log")))`; put `JobSlot { kind: Prepare, run_id: None, canceller: worker.canceller() }` in the slot; then on a **named thread** (`std::thread::Builder::new().name("sherd-job".into())`):

```rust
    let outcome = host::drive(&mut worker, |event| {
        let _ = app.emit("engine:event", EngineEvent { job: JobKind::Prepare, run_id: None, event: event.clone() });
    });
    // The slot is emptied before the window is told, so a `prepare_start` issued from the
    // window's `finished` handler cannot find the job it just saw end still "busy".
    *state.job.lock()… = None;
    let view = state.workspace.lock()….as_ref().map(|ws| view::build(ws, None));
    let _ = app.emit("engine:finished", EngineFinished { job, run_id, outcome, view });
```

Every `lock()` handles poisoning by returning a `CommandError { kind: "worker" }` or, on the job thread, by ending quietly — no `unwrap`. `job_cancel` calls `canceller.cancel()` on a clone taken under the lock; it is `Ok(())` when there is no job. While a job runs, `workspace_view` and the commands that return a view fill `job: Some(JobView { .. })`. `workspace_close` and `workspace_open` answer `busy` during a job.

`Event` values arrive at up to ten a second per stage (the worker throttles); forwarding each is fine.

- [ ] **Step 1: Implement**, then `cd apps/desktop/src-tauri && cargo check 2>&1 | tail -2` — expected: no warnings.

- [ ] **Step 2: The shell prepares the slab, without a window**

There is no headless Tauri runtime to drive, and `host::drive` over a real worker process is already covered by `sherd-app-core`'s end-to-end test. What this task adds is the binary being its own worker, so prove that:

```bash
cd apps/desktop/src-tauri && cargo build 2>&1 | tail -1
WS="$(mktemp -d)/ws" && mkdir -p "$WS"
printf '%s\n' "{\"job\":\"prepare\",\"workspace\":\"$WS\",\"input\":\"$PWD/../../../fixtures/slab/input\",\"excluded\":[],\"target_faces\":200000,\"seed\":0,\"memory_gb\":null,\"workers\":0}" \
  | ../../../target/debug/sherd-desktop --engine-worker | grep -c '"event":"fragment_ready"'
ls "$WS/fragments"
```

Expected: `2`, and `index.json`, `pieceA.glb`, `pieceA.png`, `pieceA.seg.glb`, `pieceB.*`. (If `PrepareJob`'s field names differ, take them from `crates/sherd-app-core/src/protocol.rs`.)

- [ ] **Step 3: Commit**

```bash
git add apps/desktop/src-tauri
git commit -m "M3.4: a job in the shell -- the app spawns itself as the engine, forwards its events to the window and can stop it"
```

---

### Task 5: the frontend's foundation — IPC, the mock, stores, derived status, i18n

**Files:**
- Create: `apps/desktop/src/ipc/{api.ts, tauri.ts, mock.ts, index.ts}`, `src/state/{status.ts, status.test.ts, workspace.ts, jobs.ts, ui.ts}`, `src/i18n/{index.ts, ru.json, en.json}`, `apps/desktop/public/mock/*`

**Interfaces:**
- `ipc/api.ts`:

```ts
import type { Event as EngineEvent } from "./bindings/Event";
import type { Outcome } from "./bindings/Outcome";
import type { WorkspaceView } from "./bindings/WorkspaceView";
import type { JobKind } from "./bindings/JobKind";

export interface AppInfo { version: string; core_version: string; commit: string }
export interface RecentEntry { path: string; name: string; opened_at: string; available: boolean }
export interface CommandError { kind: string; message: string }
export interface EngineEventPayload { job: JobKind; run_id: string | null; event: EngineEvent }
export interface EngineFinishedPayload { job: JobKind; run_id: string | null; outcome: Outcome; view: WorkspaceView | null }
export type Unlisten = () => void;

export interface Api {
  appInfo(): Promise<AppInfo>;
  recentList(): Promise<RecentEntry[]>;
  workspaceCreate(path: string): Promise<WorkspaceView>;
  workspaceOpen(path: string): Promise<WorkspaceView>;
  workspaceClose(): Promise<void>;
  workspaceView(): Promise<WorkspaceView>;
  inputLink(path: string): Promise<WorkspaceView>;
  fragmentExclude(name: string, excluded: boolean): Promise<WorkspaceView>;
  prepareStart(): Promise<void>;
  jobCancel(): Promise<void>;
  pickFolder(title: string): Promise<string | null>;
  /** A URL the window can fetch for a file inside the open workspace. */
  assetUrl(path: string): string;
  onEngineEvent(handler: (payload: EngineEventPayload) => void): Promise<Unlisten>;
  onEngineFinished(handler: (payload: EngineFinishedPayload) => void): Promise<Unlisten>;
  onFolderDropped(handler: (path: string) => void): Promise<Unlisten>;
}
export function isCommandError(e: unknown): e is CommandError { … }   // an object with string `kind` and `message`
```

- `ipc/index.ts`: `export const api: Api = "__TAURI_INTERNALS__" in window ? tauriApi : mockApi;`
- `ipc/tauri.ts`: `invoke` with snake_case command names, `listen("engine:event" | "engine:finished")`, `open({ directory: true, title })` from `@tauri-apps/plugin-dialog`, `convertFileSrc`, and `getCurrentWebview().onDragDropEvent` (on `"drop"`, the first path).
- `ipc/mock.ts`: an in-memory workspace for a plain browser — twelve fragments (`FY234001`…`FY234012`) built from one template `FragmentInfo`, two of them with a `thickness_outlier`, one `not_watertight`; `prepareStart` plays `stage`/`progress`/`fragment_ready` events on timers over ~6 s and then `finished`; `assetUrl(path)` maps any `*.seg.glb` → `/mock/pieceA.seg.glb`, `*.glb` → `/mock/pieceA.glb`, `*.png` → `/mock/pieceA.png`; `pickFolder` resolves to `/mock/scans/karas_reduced`. Copy the three files from `target/tmp/e2e-prepare-run/fragments/` into `apps/desktop/public/mock/` (if that folder is gone, `cargo test -p sherd-app-core --test worker_e2e a_collection` recreates it).
- `state/status.ts`:

```ts
export type Status =
  | { kind: "empty" }
  | { kind: "input_missing" }
  | { kind: "preparing" }
  | { kind: "unprepared" }
  | { kind: "ready" }
  | { kind: "running"; runId: string }
  | { kind: "current"; runId: string }
  | { kind: "stale"; runId: string; diff: StaleDiff }
  | { kind: "draft"; runId: string }
  | { kind: "failed"; runId: string; failKind: FailKind; message: string }
  | { kind: "cancelled"; runId: string }
  | { kind: "interrupted"; runId: string };

export function deriveStatus(view: WorkspaceView, selectedRunId: string | null, hasUnrefinedGroups: boolean): Status
```

  Precedence, first match wins: no input linked → `empty`; `view.job?.kind === "prepare"` → `preparing`; `view.job?.kind === "run"` → `running` (its `run_id`); the selected run (or none): with a run selected — `failed`/`cancelled`/`interrupted` by its status, a `running` with no job → `interrupted`, `done` → `stale` when its diff is not empty **and the input is available**, else `draft` when `hasUnrefinedGroups`, else `current`; with none selected — input not available → `input_missing`, not `prepared` → `unprepared`, else `ready`. A selected finished run stays viewable when the input is missing (A §5), which is why `input_missing` is only the no-run answer.
- `state/workspace.ts` (Zustand): `{ view: WorkspaceView | null, recent: RecentEntry[], error: CommandError | null, selectedRunId: string | null }` + actions wrapping the API (each sets `view` from the returned view, or `error`), `refresh()`, and `applyFragmentReady(info)` which upserts into `view.fragments` by name.
- `state/jobs.ts`: `{ stage: string | null, progress: Record<string, { done: number; total: number }>, startedAt: number | null, lastFailure: { kind: FailKind; message: string } | null }`, fed by `onEngineEvent`/`onEngineFinished`; `finished` replaces the workspace store's view with the payload's.
- `state/ui.ts`: `{ mode: "input" | "assembly" | "review", leftOpen, rightOpen, selectedFragment: string | null, inputFilter: "all" | "warnings" | "excluded", inputLayout: "grid" | "list", fragmentView: "scan" | "seg", wireframe: boolean, theme: "system" | "light" | "dark", language: "ru" | "en" }`, with `theme` and `language` persisted in `localStorage` and `theme` applied as `data-theme` on `<html>` (`system` follows `prefers-color-scheme`).
- `i18n/index.ts`: i18next + react-i18next, resources from the two JSON files, `lng` from the UI store (default: `ru` when `navigator.language` starts with `ru`, else `en`), `fallbackLng: "ru"`, `interpolation.escapeValue: false`.

- [ ] **Step 1: Write the failing test** `state/status.test.ts` — one `it` per row of A §5, over a `view()` factory that builds a `WorkspaceView` with overrides:

```ts
import { describe, expect, it } from "vitest";
import { deriveStatus } from "./status";
// view(overrides), run(id, status, stale?) helpers build minimal valid values

describe("deriveStatus (A §5)", () => {
  it("is empty with no input linked", () => { expect(deriveStatus(view({ input: { linked: false, path: null, available: false } }), null, false).kind).toBe("empty"); });
  it("is preparing while a prepare job runs, whatever else is true", () => { … });
  it("is running while a run job runs, and names the run", () => { … });
  it("is input_missing when the folder is gone and no run is selected", () => { … });
  it("is unprepared when a scan has no current row", () => { … });
  it("is ready when prepared and nothing is selected", () => { … });
  it("is current for a finished run whose input has not changed", () => { … });
  it("is stale, with the diff, when the input changed since", () => { … });
  it("is current, not stale, for a finished run while the input is missing", () => { … });
  it("is draft when a group of the current assembly is not refined", () => { … });
  it("is failed with its kind and message", () => { … });
  it("is cancelled", () => { … });
  it("is interrupted, also for a run left `running` with no job behind it", () => { … });
});
```

Write every body; each asserts `kind` and the fields that kind carries.

- [ ] **Step 2: Run it to see it fail** — `pnpm test` → the module does not exist.

- [ ] **Step 3: Implement** everything in the Interfaces block. `ru.json`/`en.json` start with the keys this task needs (`status.*` for the twelve kinds, `error.*` for the nine `CommandError` kinds, `stage.preprocess`, `stage.display`) — later tasks add theirs. Russian wording from the mock-ups: `status.empty` «Нет входных данных», `status.preparing` «Подготовка входа», `status.ready` «Готов к сборке», `status.stale` «Результат устарел», `error.locked` «Этот воркспейс уже открыт в другом окне приложения», `stage.preprocess` «Предобработка сканов», `stage.display` «Превью и модели».

- [ ] **Step 4: Verify**

```bash
cd apps/desktop && pnpm test 2>&1 | tail -6 && pnpm typecheck && pnpm lint
```

Expected: 13 tests pass; no type or lint errors.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop
git commit -m "M3.5: the window's foundation -- typed IPC with a mock behind it, stores, i18n, and a status that is derived and never stored"
```

---

### Task 6: the frame — welcome, top bar, three panes that collapse, the status line

**Files:**
- Create: `apps/desktop/src/app/{Welcome.tsx, Frame.tsx, TopBar.tsx, StatusLine.tsx, Banner.tsx, shortcuts.ts}`, `src/ui/{Button.tsx, Chip.tsx, IconButton.tsx, Meter.tsx}`
- Modify: `src/App.tsx`, `src/i18n/*.json`

**Interfaces:**
- `App` shows `Welcome` when `view === null`, `Frame` otherwise; subscribes once to the engine events and to window `focus` (→ `refresh()`, which is how «устарел» is noticed, A §4).
- `Frame` lays out `TopBar` / (`left` | `centre` | `right`) / `StatusLine`, taking the three panes from the active mode; in this milestone only `input` supplies them, `assembly` and `review` tabs are rendered **disabled** with a tooltip «Появится после первой сборки».
- `shortcuts.ts`: `useShortcuts()` — `Mod+B` toggles the left pane, `Mod+Alt+B` the right, `1`/`2`/`3` the modes (ignored when disabled), `F` fits the viewer, all ignored while an `<input>` has focus. `Mod` is `metaKey` on macOS, `ctrlKey` elsewhere (`navigator.platform`).

**What it looks like** — `docs/superpowers/specs/desktop-app-mockups/main-layout.html`, variant A, is the reference; measurements:

- Top bar 40 px, `bg-panel-2`, bottom border. Left to right: workspace name with a chevron (menu: «Открыть другой…», «Закрыть», language ru/en, theme system/light/dark) · run selector chip (in this milestone: «Прогонов ещё нет», disabled) · flexible space · three mode tabs as chips with their status (`Вход · 155 · 11 предупр.`; the active one `bg-accent text-accent-text`) · flexible space · two pane toggles (lucide `PanelLeft`, `PanelRight`, pressed state when open) · primary action.
- Primary action by status: `empty` → «Выбрать папку со сканами»; `preparing` → «Отменить» (danger outline); `unprepared` → «Подготовить»; `ready` → «Собрать…» **disabled**, tooltip «Сборка — в следующей версии»; `input_missing` → «Указать папку заново».
- Left pane 300 px (Вход: 360 px), right pane 260 px, each with a 1 px border, `bg-panel`, collapsing to zero width with no animation; with both collapsed the centre fills the window (the user's explicit request).
- Status line 26 px, `bg-panel-2`, `text-muted`: the status's sentence on the left; while a job runs, the stage's name, `done / total`, a 160 px `Meter`, elapsed time `m:ss`; right: counts (`155 фрагментов · 11 предупр. · 1 исключён`).
- `Banner` sits under the top bar for `input_missing` (danger), `stale` (warn, listing the diff: «+3 файла, −1, 2 изменены»), a job failure (danger, the message, «Показать лог» is milestone 4), and a `CommandError` (dismissible).
- `Welcome`: centred column 520 px — product name, one sentence («Сборка 3D-сканов керамических фрагментов»), two buttons «Создать воркспейс…» and «Открыть…» (both pick a folder), then «Недавние» — rows of name, path in `text-muted`, relative date; an unavailable one is dimmed and says «папка недоступна». Errors show inline under the buttons in the error's own words (`error.<kind>`).
- Empty state of `Frame` (status `empty`): the centre shows a dashed drop zone — «Перетащите сюда папку со сканами или выберите её» with the supported formats (`.ply .obj .stl .off`) — and both the button and a dropped folder call `inputLink`.

`ui/` components are small and unstyled beyond the tokens: `Button({ variant: "primary" | "ghost" | "danger", size })`, `Chip({ active, tone: "default" | "warn" | "danger", disabled })`, `IconButton({ pressed, label })` (the label is its `aria-label` and tooltip), `Meter({ value, max })` (`role="progressbar"`). Every interactive element is a real `<button>` with a visible focus ring.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify** — `cd apps/desktop && pnpm typecheck && pnpm lint && pnpm test 2>&1 | tail -3 && pnpm build 2>&1 | tail -3`. Expected: clean, 13 tests, a build.
- [ ] **Step 3: Look at it.** `pnpm dev`, open `http://localhost:1420` in a browser: the mock's welcome screen; «Создать воркспейс…» → the frame with the drop zone; the toggles and `Mod+B` collapse the panes to a full-window centre; the theme and language switch. Fix what is broken; do not write a test for it.
- [ ] **Step 4: Commit**

```bash
git add apps/desktop
git commit -m "M3.6: the frame -- welcome, a top bar that says where the workspace stands, three panes of which two collapse, a status line"
```

---

### Task 7: the viewer — one fragment in 3D

**Files:**
- Create: `apps/desktop/src/viewer/{FragmentViewer.ts, matrix.ts, matrix.test.ts, FragmentView.tsx}`

**Interfaces:**
- `matrix.ts`: `export function rowsToMatrix4(rows: readonly (readonly number[])[]): THREE.Matrix4` — throws a `RangeError` unless 4×4.
- `FragmentViewer` (no React):

```ts
export class FragmentViewer {
  constructor(canvas: HTMLCanvasElement);
  /** Shows the GLB at `url`, or nothing for `null`. Resolves when it is on screen; a newer call wins over an older one still loading. */
  show(url: string | null): Promise<void>;
  setWireframe(on: boolean): void;
  fit(): void;
  dispose(): void;
}
```

- `FragmentView({ url, wireframe, fitSignal })` — a `<canvas>` filling its parent; owns one `FragmentViewer` for its lifetime; shows a centred `text-viewport-text` line while loading and the error's message if the load fails.

- [ ] **Step 1: Write the failing test** `matrix.test.ts`

```ts
import { describe, expect, it } from "vitest";
import { Vector3 } from "three";
import { rowsToMatrix4 } from "./matrix";

// README, «Одно соглашение о матрице»: files carry M by rows, p' = M·p on column vectors.
// THREE.Matrix4.set() takes rows; .fromArray() and .elements are column-major. This is the one
// place the two meet, and the classic place to get a mirrored assembly.
describe("rowsToMatrix4", () => {
  const rows = [
    [0, -1, 0, 10],
    [1, 0, 0, 20],
    [0, 0, 1, 30],
    [0, 0, 0, 1],
  ];
  it("moves a point as M·p does", () => {
    const p = new Vector3(1, 2, 3).applyMatrix4(rowsToMatrix4(rows));
    expect([p.x, p.y, p.z]).toEqual([8, 21, 33]); // (0·1 − 1·2 + 10, 1·1 + 20, 3 + 30)
  });
  it("keeps the translation in the fourth column", () => {
    expect(Array.from(rowsToMatrix4(rows).elements.slice(12, 15))).toEqual([10, 20, 30]);
  });
  it("refuses what is not 4×4", () => {
    expect(() => rowsToMatrix4([[1, 0, 0, 0]])).toThrow(RangeError);
  });
});
```

- [ ] **Step 2: Run it to see it fail**, then implement `matrix.ts` with `new Matrix4().set(...)` over the sixteen numbers row by row.

- [ ] **Step 3: Implement `FragmentViewer`.**

- `WebGLRenderer({ canvas, antialias: true, powerPreference: "low-power" })`, `outputColorSpace = SRGBColorSpace`, `setPixelRatio(Math.min(devicePixelRatio, 2))`, clear colour from the CSS variable `--viewport`.
- A `PerspectiveCamera(35°)`; `OrbitControls` (from `three/examples/jsm/controls/OrbitControls.js`) with damping **off**; `HemisphereLight(0xffffff, 0x3a3a40, 1.1)` plus a `DirectionalLight(0xffffff, 1.6)` that is a **child of the camera**, so the relief of a fracture reads from every angle.
- `GLTFLoader` (`three/examples/jsm/loaders/GLTFLoader.js`). The files carry two materials (vertex-coloured white, plain clay) and double-sided, non-metallic settings; use them as loaded.
- **Render on demand:** one `render()` scheduled through `requestAnimationFrame` by a `dirty` flag, requested on controls `change`, after a load, on resize, on `setWireframe`. No loop.
- `fit()`: bounding sphere of the object → target = its centre, distance = `r / sin(fov/2) · 1.15`, `near = r / 100`, `far = r · 100` (scans are in millimetres with coordinates in the hundreds; fixed planes would clip them), looking along `(1, -1, 0.8)` with `up = (0, 0, 1)`.
- A `ResizeObserver` on the canvas's parent sets the size and the aspect.
- An LRU of the last 8 loaded scenes keyed by URL, so that stepping through neighbours is instant; an evicted or disposed scene has every geometry and material `dispose()`d. `show()` keeps a load counter so a slow earlier load cannot replace a later one.
- `dispose()` stops the observer, disposes the controls, the cache and the renderer, and calls `renderer.forceContextLoss()`.

- [ ] **Step 4: Verify** — `pnpm test 2>&1 | tail -3 && pnpm typecheck && pnpm lint`. Expected: 16 tests.
- [ ] **Step 5: Commit**

```bash
git add apps/desktop
git commit -m "M3.7: one fragment in 3D -- drawn when something changed and not otherwise, and the one place a file's matrix meets three.js"
```

---

### Task 8: the «Вход» mode

**Files:**
- Create: `apps/desktop/src/modes/input/{InputLeft.tsx, InputCentre.tsx, InputRight.tsx, format.ts, format.test.ts}`
- Modify: `src/app/Frame.tsx`, `src/App.tsx`, `src/i18n/*.json`

**Interfaces:**
- `format.ts`: `formatBytes(n: number, lang: "ru" | "en"): string` (`9.8 ГБ` / `9.8 GB`, one decimal from MB up), `formatCount(n: number, lang): string` (`1 495 166` with a narrow no-break space in `ru`, `1,495,166` in `en`), `formatPercent(f: number): string` (`0.1209 → "12 %"`), `warningText(w: Warning, t): string`.
- The reference is part 3 of `docs/superpowers/specs/desktop-app-mockups/states-and-input.html`.

**Left** (360 px): header — the input path (middle-ellipsised), `155 файлов · 9.8 ГБ`; filter chips «Все», «С предупреждениями N», «Исключённые N»; a grid/list toggle. The list is **virtualised** with `@tanstack/react-virtual` (rows of as many 104 px cells as fit; 155 is nothing, a museum box of 2 000 is not). A cell: the thumbnail (`api.assetUrl(`${view.fragments_dir}/${name}.png`)`, `loading="lazy"`, on `bg-viewport`) or a pulsing placeholder while the fragment has no row yet; the name underneath, middle-ellipsised; a `⚠` badge in `text-warn` when it has warnings; struck through at 40 % opacity when excluded; selected = 2 px `accent` outline. The source of rows is `view.files` (so every scan is there from the first second), joined with `view.fragments` by name. Arrow keys move the selection; `Enter` does nothing.

**Centre:** `FragmentView` for the selected fragment — `${name}.glb` or `${name}.seg.glb` by the toggle — with a toolbar of chips over the viewport's top-left: «Скан», «Излом красным», «Сетка», «Вписать»; with nothing selected, or a fragment not prepared yet, a quiet line in `text-viewport-text` saying which.

**Right** (260 px), for the selected fragment: a key–value list — файл, размер, граней в скане, граней в рабочем меше, толщина стенки, замкнут (да/нет), излом (% площади), габариты; then one block per warning, bordered `warn`:
- `thickness_outlier`: «Толщина стенки {{thickness}} при медиане коллекции {{median}} ({{percent}}). Возможно, это дно, ручка или фрагмент другого сосуда.»
- `not_watertight`: «Меш не замкнут. Стыки этого фрагмента программа не сможет подтвердить сама — их придётся принимать вручную.» (R §6.4; README, «правило замкнутости»)

and a ghost button «Исключить из сборки» / «Вернуть в сборку» calling `fragmentExclude`. Under it, in `text-muted`: «Файл не изменяется: это пометка в воркспейсе.»

**Auto-prepare** (in `App.tsx`): when the status becomes `unprepared`, the input is available and no job runs, call `prepareStart()` — once per distinct `view.files` signature (names+sizes+mtimes joined), so a failed prepare is not retried in a loop; a failure shows the `Banner` with «Повторить подготовку». `fragment_ready` events call `applyFragmentReady`, which is what makes thumbnails appear one by one.

- [ ] **Step 1: Write the failing test** `format.test.ts`

```ts
import { describe, expect, it } from "vitest";
import { formatBytes, formatCount, formatPercent } from "./format";

describe("format", () => {
  it("writes sizes the way the language does", () => {
    expect(formatBytes(9_800_000_000, "ru")).toBe("9.8 ГБ");
    expect(formatBytes(9_800_000_000, "en")).toBe("9.8 GB");
    expect(formatBytes(512, "en")).toBe("512 B");
    expect(formatBytes(1_127_755, "ru")).toBe("1.1 МБ");
  });
  it("groups thousands", () => {
    expect(formatCount(1_495_166, "ru")).toBe("1 495 166");
    expect(formatCount(1_495_166, "en")).toBe("1,495,166");
  });
  it("rounds a fraction to whole percent", () => {
    expect(formatPercent(0.1209)).toBe("12 %");
  });
});
```

(decimal units, as Finder and Explorer show them; `formatCount` must not depend on the runtime's ICU data — build it by hand.)

- [ ] **Step 2: Run it to see it fail, implement `format.ts`, then the three panes and the wiring.**
- [ ] **Step 3: Verify** — `cd apps/desktop && pnpm test 2>&1 | tail -3 && pnpm typecheck && pnpm lint && pnpm build 2>&1 | tail -2`. Expected: 19 tests; clean.
- [ ] **Step 4: Look at it, in the browser and in the app.**
  - Browser: `pnpm dev` → create → pick the folder: twelve placeholders fill in one by one over a few seconds while the status line counts; a fragment shows in 3D; «Излом красным» switches to the red-and-grey mesh; «Сетка»; excluding strikes the cell through and moves it to the filter.
  - App: `pnpm tauri dev` (first build: several minutes), create a workspace in a temp folder, link `fixtures/slab/input`: two thumbnails arrive, both fragments show in 3D with their fracture faces red in «Излом красным». Close and reopen the app: the workspace is in «Недавние» and opens prepared, with no second `Prepare`.
  If `pnpm tauri dev` cannot open a window in your environment, say so in your report rather than guessing; the browser check stands.
- [ ] **Step 5: Commit**

```bash
git add apps/desktop
git commit -m "M3.8: the Вход mode -- every scan from the first second, its thumbnail when it is ready, its fracture in red, what is wrong with it, and a way to leave it out"
```

---

### Task 9: the milestone gate, once

- [ ] **Step 1: Rust**

```bash
cargo fmt --all --check && (cd apps/desktop/src-tauri && cargo fmt --check)
cargo clippy -p sherd-app-core --all-targets --features ts --locked -- -D warnings
cargo clippy -p sherd-app-core --all-targets --locked -- -D warnings
(cd apps/desktop/src-tauri && cargo clippy --all-targets --locked -- -D warnings && cargo test 2>&1 | tail -3)
cargo test -p sherd-app-core 2>&1 | grep 'test result'
```

Expected: clean; the shell's one test and `sherd-app-core`'s 21 + 3 pass.

- [ ] **Step 2: The bindings are what the Rust types say**

```bash
TS_RS_EXPORT_DIR="$PWD/apps/desktop/src/ipc/bindings" cargo test -p sherd-app-core --features ts --lib 2>&1 | tail -2
git status --short apps/desktop/src/ipc/bindings
```

Expected: nothing changed. A diff means a Rust type moved without its bindings: commit the regenerated files.

- [ ] **Step 3: Frontend**

```bash
cd apps/desktop && pnpm install --frozen-lockfile && pnpm lint && pnpm typecheck && pnpm test 2>&1 | tail -4 && pnpm build 2>&1 | tail -3
```

- [ ] **Step 4: The app bundles.** A §11 asks for a bundle at this milestone; a **debug** bundle proves the packaging without the release profile's LTO build, which milestone 6 does once. In the background:

```bash
cd apps/desktop && pnpm tauri build --debug 2>&1 | tail -6
```

Expected: a `.app` (and a `.dmg`) under `target/debug/bundle/`. If the `.dmg` step fails for a reason that is the machine's (no GUI session for `hdiutil`/AppleScript), re-run with `--bundles app` and say so in the report.

- [ ] **Step 5: README.** Add a short section «Настольное приложение (в разработке)» after «Rust-ядро»: what it is in two sentences, and how to run it from source — the PATH line, `cd apps/desktop && pnpm install && pnpm tauri dev` — and that `pnpm dev` alone opens the interface in a browser on mock data.

- [ ] **Step 6: Commit**

```bash
git add -A apps/desktop crates/sherd-app-core README.md Cargo.toml Cargo.lock
git commit -m "M3.9: the milestone's gate -- fmt and clippy for the shell and the app core, bindings in step, the frontend's checks, a debug bundle"
```

---

## What the next plan can rely on

| need of milestone 4 (run, progress, history, «Сборка») | provided by |
|---|---|
| start a run from the window | `host::start_run`/`finish_run`/`save_assembly` + the job thread of `jobs.rs`, which already forwards events and frees the slot |
| stages, a bar, time left | `engine:event` → `state/jobs.ts` (`progress` by stage, `startedAt`) |
| the run selector, stale banners, failed runs | `WorkspaceView.runs` with each run's `StaleDiff`; `deriveStatus` already answers every row of A §5 |
| many fragments in one scene under matrices | `viewer/matrix.ts`, the LRU and on-demand rendering of `FragmentViewer`; `fragments/<name>.glb` in source coordinates |
| a place for «Сборка»'s panes | `Frame`'s mode slot; the disabled tabs |
