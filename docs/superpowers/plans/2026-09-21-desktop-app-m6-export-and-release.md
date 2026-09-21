# Desktop App, Milestone 6 — export, Blender, settings, the release build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the work out of the app: «Экспорт» writes the reviewed assembly with the engine's own writers — the result folder a CLI run writes, or its tables and report alone; «Открыть в Blender» puts a group or the whole assembly into Blender from the original scans, named and placed; settings hold what belongs to the machine; and the app builds as a release bundle on macOS and Windows, locally and in CI.

**Architecture:** `Export` is one more request of the review session (it already holds the fragments and the match): reassemble under the decisions, refine whatever group is not refined, `session::write_reviewed` with the chosen switches into a folder the user picked. The Blender script is a pure function of the assembly, the input snapshot and two choices, tested against a golden file and compiled by Python; launching Blender is the shell's. Settings are a JSON file in the app's config directory, read by the shell and offered to the window. CI gains a workflow of its own for the app, separate from the core's.

**Tech Stack:** as milestone 5, plus Rust `fs4` (free disk space; latest `0.x` that builds on 1.89, pinned exactly). Spec: `docs/superpowers/specs/2026-09-20-desktop-app-design.md` (`A §9`, `A §10`, `A §11`).

## Global Constraints

- **Toolchain.** `export PATH="$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin:$PATH"` before anything that runs cargo. pnpm 9 from `apps/desktop/`; exact versions.
- **Verification budget (A §11).** Run only what a task lists. No Playwright/WebDriver/component tests. No clippy inside a task (Task 8 is the gate). Only `fixtures/slab`. `cargo test -p sherd-cli --test run_cli` (4–7 min, background) runs once, in Task 8, because Task 2 touches `sherd-core`.
- **The CLI's output stays byte for byte.** Task 2 is the only task that touches `crates/sherd-core`: it adds progress reports and nothing else.
- **Looking is part of building**, as before: `node tools/look.mjs` in the **background** with its output in a file, killed if silent for 90 s; start every script with `{"goto":"/"},{"storage":{"sherd.language":"ru","sherd.theme":"light"}}`; read every PNG. Work in small steps: no tool call writes more than ~150 lines.
- **Blender is not installed on the development machine.** Nothing may claim to have run Blender. The script is verified by its golden test and by `python3 -m py_compile`; its behaviour inside Blender is listed under "What is left for a person".
- **No panics on data from outside**; strict TypeScript; every string through i18next, `ru`/`en` key parity; `src/ipc/contract.test.ts` stays green for every new command.
- **Ownership (A §2.1).** The worker writes what `Export` writes (it is the engine's output); the host writes settings and state.
- **Nothing is pushed, published or signed.** The CI workflow file is committed, not run; signing and notarisation are sub-project 4.
- **Commits:** branch `desktop-app`, subject `M6.<task>: <sentence>`, trailer as your own system instructions give it.
- **Do not touch** `crates/sherd-parity`, `sherd_refit/`, `fixtures/`.

## Where things are (tree at `b15416c`)

`crates/sherd-app-core/src/worker/review.rs` (the session: `Reassemble`, `PairDetail`, `Refine`, `Close`; the baseline; `merge_refined` in `src/review.rs`), `protocol.rs` (`Request`, `Event`, DTOs), `host.rs` (`Worker`, `Requester`, `Canceller`, `carry`), `apps/desktop/src-tauri/src/{jobs.rs, commands.rs, state.rs}` (the session slot, `review_*` commands), `apps/desktop/src/{state/review.ts, modes/, app/, viewer/}`, `apps/desktop/src/ipc/contract.test.ts`. `session::write_reviewed(out_dir, input, fragments, &MatchState, &Reassembled, poses, &RunOptions) -> Result<Vec<PathBuf>>` writes what `options`' switches say: `write_meshes` (`placed/` and, with `viewer`, `scene.glb` + `viewer.html`), `placed_all`, `merged_meshes`, `preview`; the tables, `report.*`, `transforms.*` and `README.txt` always.

---

### Task 1: what milestone 5's reviews left open

**Files:** `crates/sherd-app-core/src/worker/{review.rs, mod.rs}`, `apps/desktop/src-tauri/src/{commands.rs, jobs.rs}`, `apps/desktop/src/{state/review.ts, modes/assembly/AssemblyRight.tsx, viewer/AssemblyViewer.ts, ipc/mock.ts}`

- `worker/review.rs`: after `Refine`, R §8.2's `recenter` is applied **only to the groups that were just refined**; a group copied from the baseline keeps its poses bit for bit (a second recentring moves it by a rounding error, and "refined groups stay as they were" is the rule). Extend the end-to-end test: two `Refine`s in a row return identical poses for the already-refined group.
- `worker/mod.rs`: requests that arrive during a `Prepare` or a `Run` are dropped by the reader, as the doc comment says (drop the receiver before dispatch for those two kinds).
- `commands.rs`: `review_apply` is `#[tauri::command(async)]` — it fsyncs a file and must not run on the main thread; `jobs.rs`: `start_review` for a run whose session is **closing** waits for the slot and opens a new one instead of answering "already open".
- The window: a ghost is withdrawn when the reassembly places its fragment; `setGhost` shows the anchor's group for as long as the ghost is up when that group is hidden («без пары» is hidden by default, and an unpaired fragment is exactly whose candidates get hovered); `review.ts`' `assembly` branch ignores an answer whose `run_id` is not the open session's; the mock answers `reviewApply` only after it said `ready`.
- [ ] Implement; `cargo test -p sherd-app-core 2>&1 | grep 'test result'`; the shell's `cargo check && cargo test`; `pnpm test && pnpm typecheck && pnpm lint`; one look at a hovered candidate of an unpaired fragment (the ghost is visible).
- [ ] Commit — `M6.1: what milestone 5's reviews left open -- a refined group that is not recentred twice, a ghost that can be seen and goes away, and four smaller things`

---

### Task 2: progress from the engine's writers

**Files:** `crates/sherd-core/src/pipeline.rs` (`write_outputs`), `crates/sherd-core/src/report.rs` (`write_placed_selected`) and `src/export/scene.rs` (`write_scene`) only if the counter has to live inside their parallel loops; `crates/sherd-core/tests/session.rs`

A §3.4 left the output stage's progress to this milestone. `write_outputs` reports `watch.advance("output", done, total)` where `total` is the number of **mesh files** it is going to write (`placed/*.ply`, plus one for `scene.glb` when the viewer is on) and `done` counts them as they finish — from whichever thread finished one, as `preprocess` does. With no meshes to write it reports nothing. `RunOptions::watch` is already in `options`. No file's bytes change; nothing is reordered.

- [ ] **Failing test** in `tests/session.rs`: `write_reviewed` of the slab's accepted join with `write_meshes: true, viewer: true` under a recording `Progress` ends the stage `output` at `done == total == 3` (two placed meshes and the scene); with `write_meshes: false` the stage is never reported.
- [ ] Implement; `cargo test -p sherd-core --test session 2>&1 | tail -4` → 10 passed.
- [ ] Commit — `M6.2: the writers say how far they are -- one report per mesh file, which is where an export's time goes`

---

### Task 3: `Export` in the review session

**Files:** `crates/sherd-app-core/src/{protocol.rs, worker/review.rs}`, `tests/worker_e2e.rs`, `Cargo.toml` (`fs4`), root `Cargo.toml`

**Interfaces:**
- `Request::Export { decisions: DecisionsFile, what: ExportWhat, dest: PathBuf }`; `enum ExportWhat { Folder { placed_all: bool, merged_meshes: bool, previews: bool }, Tables }` (`serde(tag = "kind", rename_all = "snake_case")`); `Event::Exported { dest: PathBuf, files: Vec<String> /* relative to dest */, bytes: u64 }`.
- `Folder` = `write_meshes: true, viewer: true` plus the three switches; `Tables` = `write_meshes: false, viewer: false, preview: false` — the tables, `report.md`/`report.json`, `transforms.*`, `README.txt`.
- Before writing: reassemble under the decisions; **refine** every unrefined group of two or more (as `Refine` does, becoming the new baseline, and answering with an `Assembly` event first so the window's state is the exported one); refuse (`RequestFailed`) when `dest` exists and is not an empty directory, when the input folder is not available for a `Folder` export (it reads the scans), or when the free space at `dest` (`fs4::available_space`) is under the estimate — for `Folder`: the sum of the source files' sizes of the fragments that will be placed × 1.5, for `Tables`: 64 MB. The message says the two numbers.
- The backend recorded is the match's (`write_reviewed` does that); progress is the stage `output` of Task 2, after `refine`'s when refinement ran.

- [ ] **End to end**, appended to `tests/worker_e2e.rs`: review session on the slab, `Export` with one accepted decision and `Folder { false, false, false }` into a fresh directory → `Assembly` (refined), progress for `output`, `Exported` whose `files` include `placed/pieceA.ply`, `placed/pieceB.ply`, `scene.glb`, `viewer.html`, `transforms.csv`, `report.md`, `README.txt`; `report.md` contains `## Constraints` and both names; a second `Export` into the same, now non-empty directory is a `RequestFailed` and the session goes on (`Close` → `Done`). `Tables` into another directory writes no `placed/` and no `scene.glb`.
- [ ] `cargo test -p sherd-app-core 2>&1 | grep 'test result'`; regenerate the bindings.
- [ ] Commit — `M6.3: Export -- the reviewed assembly, refined first, written by the engine's own writers into a folder that was empty and has room`

---

### Task 4: the Blender script

**Files:** `crates/sherd-app-core/src/blender.rs` (new), `src/blender_golden.py` (the golden file, beside it), `src/lib.rs`

**Interfaces:**
```rust
pub enum Resolution { Full, Display }            // the original scans, or fragments/<name>.glb
pub struct BlenderFragment { pub name: String, pub source: PathBuf, pub display: PathBuf, pub pose: [[f64; 4]; 4] }
pub struct BlenderGroup { pub index: usize, pub label: String, pub fragments: Vec<BlenderFragment> }
/// The Python that builds the scene. Pure: the same input gives the same text.
pub fn script(title: &str, groups: &[BlenderGroup], resolution: Resolution) -> String
pub fn groups_of(assembly: &AssemblyDto, snapshot: &InputSnapshot, input: &Path, fragments_dir: &Path, scope: Scope) -> Vec<BlenderGroup>   // Scope::{All, Group(usize)}; singleton groups are left out of All
pub fn find_blender(override_path: Option<&Path>) -> Option<PathBuf>
```
What the script does (A §9.2), written for Blender 4.0+ and refusing to run on anything older with a message box:
- a new empty scene's worth of state is **not** assumed: it creates a top collection named `title` and works inside it, deleting nothing of the user's;
- per group a child collection `label` («Группа 0»…) and, when there is more than one group, an empty parent per group placed on a row — each group's offset along X is the running sum of the previous groups' bounding widths plus 20 % (computed in the script from the imported objects' bounds, so the Rust side needs no geometry);
- per fragment: `Full` → import the source by extension with Blender's native importers and **explicit axes** — `bpy.ops.wm.ply_import(filepath=…, forward_axis='Y', up_axis='Z')`, `bpy.ops.wm.obj_import(…, forward_axis='Y', up_axis='Z')`, `bpy.ops.wm.stl_import(…, forward_axis='Y', up_axis='Z')` — so that no importer default rotates anything, then `obj.matrix_world = M` with `M` the pose's rows (`mathutils.Matrix`); an `.off` source, which Blender cannot read, and every fragment under `Display` → `bpy.ops.import_scene.gltf(filepath=…)`, whose importer converts glTF's Y-up to Blender's Z-up **in the vertex data** — `(x, y, z) → (x, −z, y)` — while our GLBs hold the scan's own coordinates unconverted, so the pose is post-multiplied by the inverse of that conversion: `obj.matrix_world = M @ Matrix(((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1)))` (the constant has a unit test: it maps `(x, −z, y)` back to `(x, y, z)`);
- objects are named after fragments; no scale is applied — the units stay the scan's, and the script's last lines print that and the number of objects; a file that fails to import is reported at the end, not fatal;
- paths are written as Python raw strings with backslashes and quotes escaped (a Windows path, a path with a quote or a non-ASCII name must survive — tested).

`find_blender`: the override if it is a file; else macOS `/Applications/Blender.app/Contents/MacOS/Blender` and `~/Applications/…`; Windows every `C:\Program Files\Blender Foundation\Blender *\blender.exe`, newest first; Linux `blender` on `PATH`.

- [ ] **Failing tests:** the golden test (two groups, one `.off`, a path with a space, a quote and «керамика» in it — `assert_eq!(script(…), include_str!("blender_golden.py"))`, with an `UPDATE_GOLDEN=1` escape that rewrites the file), `groups_of` (scope, singletons left out, a fragment missing from the snapshot skipped), the axis constant, path escaping, `find_blender(Some(nonexistent))` falls through.
- [ ] Implement; `cargo test -p sherd-app-core --lib blender 2>&1 | tail -3`; **`python3 -m py_compile crates/sherd-app-core/src/blender_golden.py`** must pass (the golden file is real Python).
- [ ] Commit — `M6.4: Open in Blender, as a script -- the original scans imported with explicit axes, placed by their matrices, a collection per group`

---

### Task 5: settings, and the shell's side of export and Blender

**Files:** `apps/desktop/src-tauri/src/{settings.rs (new), commands.rs, jobs.rs, main.rs}`, `capabilities/default.json`

**Interfaces:**
- `settings.json` in the app's config dir: `Settings { version, backend: BackendChoice, memory_gb: Option<f64>, workers: usize, blender_path: Option<PathBuf> }` (TS-exported through a mirror or by living in `sherd-app-core::settings` — prefer the latter, with its load/save and a test: a missing or damaged file is `Default`); commands `settings_get() -> Settings`, `settings_set(settings: Settings) -> Settings`. The launch sheet's defaults come from the settings when the workspace has no last spec; `Prepare` uses `memory_gb` and `workers`.
- `export_start(what: ExportWhat, dest: String) -> Result<(), CommandError>` — opens the review session for the selected run if none is open, sends `Export` with the file's decisions; the `Exported`/`RequestFailed` events reach the window as the session's events do. `export_default_dest(what) -> String` = `<workspace>/exports/<YYYY-MM-DD_HHMM>_<folder|tables>`.
- `blender_open(run_id: String, scope: ScopeDto, resolution: ResolutionDto) -> Result<BlenderOutcome, CommandError>` with `BlenderOutcome { launched: bool, script: String, blender: Option<String> }`: writes `exports/<stamp>_blender/open_in_blender.py`, and when `find_blender(settings.blender_path)` finds one, spawns it **detached** (`blender --python <script>`, stdio null, not waited for; `CREATE_NO_WINDOW` is *not* set — Blender is a GUI app) — `#[tauri::command(async)]`.
- `reveal(path: String)` through `tauri-plugin-opener`'s `reveal_item_in_dir`, refused for a path outside the open workspace and outside the last export's `dest`.
- [ ] Implement; `cargo check && cargo test` in the shell; `cargo test -p sherd-app-core --lib settings`; bindings regenerated; `pnpm test` (the contract test will fail until Task 6 adds the window's calls — that is expected here; say so, and do not weaken it).
- [ ] Commit — `M6.5: settings that belong to the machine, and the shell's side of export and of Blender`

---

### Task 6: the window — «Экспорт», «Открыть в Blender», «Настройки»

**Files:** `apps/desktop/src/app/{ExportDialog.tsx, SettingsDialog.tsx, TopBar.tsx}`, `src/modes/assembly/AssemblyRight.tsx`, `src/state/{export.ts, settings.ts}`, `src/ipc/{api.ts, tauri.ts, mock.ts}`, `src/i18n/*.json`

- **«Экспорт»** (top bar; enabled for a `done` run) opens a dialog: what — «Папка результата» («всё, что пишет командная строка: таблицы, отчёт, scene.glb и viewer.html, меши собранных фрагментов в полном разрешении») with three checkboxes («меши и несобранных фрагментов», «по одному слитому мешу на группу», «картинки-превью групп»), or «Только таблицы и отчёт»; where — the default destination with «Выбрать…» (a directory picker); a line that the export reflects the review («2 принято · 1 отклонено — попадут в раздел Constraints отчёта») and that unrefined poses are refined first («≈30 с»). «Экспортировать» → progress in the dialog (refine, then `output` with its counts) → «Готово: 273 файла, 212 МБ» with «Показать в папке»; a refusal (folder not empty, no room, input missing) is shown in the dialog in a sentence, and the dialog stays open.
- **«Открыть в Blender»**: in the Export menu («вся сборка») and in the group inspector («эта группа»), each with «в полном разрешении» / «облегчённые модели»; the result is a toast — «Blender запущен» or «Blender не найден: скрипт сохранён» with «Показать в папке» and «Указать путь к Blender…» (opens settings).
- **«Настройки»** (workspace menu, and `Mod+,`): язык, тема (both already in the UI store — move their controls here and leave the menu's shortcuts), вычисления по умолчанию, лимит памяти, потоки, путь к Blender with «Найти автоматически»; «Снимок PNG» in «Сборка» now goes through a save dialog in Tauri and stays a download link in the browser.
- The mock plays an export (2 s of progress, then `exported` with plausible files) and a Blender launch (`launched: false`).
- [ ] Implement; `pnpm test && pnpm typecheck && pnpm lint && pnpm build`; look: the export dialog (both kinds), its progress, its result, its refusal; the Blender toast; the settings dialog; `en`/dark once.
- [ ] Commit — `M6.6: Экспорт, Открыть в Blender, Настройки -- the work leaves the app, and the machine's choices have a place`

---

### Task 7: CI for the app, and the README

**Files:** `.github/workflows/desktop.yml` (new), `README.md`, `docs/superpowers/specs/2026-09-20-desktop-app-design.md` (status line)

- `desktop.yml`, triggered on pushes and pull requests that touch `apps/desktop/**`, `crates/sherd-app-core/**`, `crates/sherd-backend/**` or `crates/sherd-core/**`: job **frontend** (ubuntu-latest: pnpm 9, Node 24, `pnpm install --frozen-lockfile`, `lint`, `typecheck`, `test`, `build`); job **bindings** (ubuntu-latest, the repository's Rust toolchain: regenerate with `--features ts`, fail if `git status --porcelain apps/desktop/src/ipc/bindings` is not empty); job **bundle** on `macos-14` and `windows-latest`: `pnpm install --frozen-lockfile`, `pnpm tauri build` (unsigned), upload `target/release/bundle/**` as an artifact. No Linux bundle (not a target, A §0). No secrets, no signing. Follow the conventions of the existing `.github/workflows/rust.yml` (toolchain install, `Swatinem/rust-cache`, `--locked`).
- README, «Настольное приложение»: what it does, mode by mode, in the words a museum colleague reads; how to build it from source; where a workspace keeps what; what is not there yet (инвентарные номера, `.tfm`, подпись сборок).
- The spec's status line: «implemented, milestones 1–6», with the two manual acceptances still open.
- [ ] Validate the workflow's YAML (`python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/desktop.yml'))"` if PyYAML is there; otherwise read it twice) — it cannot be run here.
- [ ] Commit — `M6.7: CI for the app -- its checks, its bindings in step, an unsigned bundle on macOS and Windows -- and a README a colleague can read`

---

### Task 8: the milestone gate, and the release build

- [ ] **Rust** — fmt for both workspaces; `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings`; `cargo clippy -p sherd-app-core --all-targets --features ts --locked -- -D warnings`; the shell's clippy and tests; `cargo test -p sherd-app-core`; `cargo test -p sherd-core --test session`.
- [ ] **The CLI's byte-for-byte gate**, in the background: `cargo test -p sherd-cli --test run_cli 2>&1 | tail -5`.
- [ ] **Bindings in step; frontend** — as in milestone 5's gate.
- [ ] **The whole walk, looked at**, welcome → prepare → run → «Сборка» → «Ревью» → export → settings, `ru`/light and `en`/dark.
- [ ] **The release build**, once, in the background (LTO — 10–25 minutes): `cd apps/desktop && pnpm tauri build 2>&1 | tail -8`. Then start `target/release/bundle/macos/Sherd Refit.app/Contents/MacOS/sherd-desktop` for six seconds and stop it: it must stay up and write nothing to stderr; and `echo '{"job":"info","adapter":null,"selftest":false}' | <that binary> --engine-worker` answers `hello`, `info`, `done`. Report the bundle's size and path. If the `.dmg` step needs a GUI session the machine does not give, build with `--bundles app` and say so.
- [ ] Commit — `M6.8: the milestone's gate -- and a release bundle that starts, and is its own engine`

---

## What is left for a person

1. **The two manual acceptances of A §11**, in the real window (`cd apps/desktop && pnpm tauri dev`, or the release `.app`): a workspace on `input/karas_reduced`, a GPU run with its time left and a cancel; a review with a dozen decisions, «Принять все оставшиеся», «Уточнить позы», an export, a second run with the decisions carried over.
2. **Blender**: install Blender 4.x, press «Открыть в Blender» on a group and on the whole assembly, in both resolutions: the fragments must stand as they do in the app's «Сборка», named, a collection per group. The axis handling of the display GLBs is reasoned, not observed.
3. **Windows**: nobody has run the app there; CI builds it once the branch is pushed.
4. Sub-projects 3 (inventory numbers, `.tfm`, `placed/` per group) and 4 (signing, notarisation, auto-update) — after the call with the museum.
