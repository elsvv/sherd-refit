# sherd-refit — desktop app: foundation and join review

**Date:** 2026-09-20. **Status:** design, no code. Implements the `app` row of D §12
(`2026-09-06-rust-core-design.md`, cited as `D`; the algorithm reference is cited as `R`). This is
sub-project 1+2 of four; §12 names the other two and what is deliberately left to them.

Sources: the user's brief of 2026-09-20, the museum chat of 2026-09-09…16 (requests quoted in §1),
and a read of the tree at `871b3ae`.

## 0. Decisions in one table

| topic | decision |
|---|---|
| shell | Tauri 2; targets macOS (arm64, x86_64) and Windows x64. Linux must keep building, is not a target |
| process model | one executable, two roles: the UI process, and the same binary re-launched with `--engine-worker` for every engine job. JSON lines over stdio. One worker at a time |
| crates | new `crates/sherd-app-core` (workspace model, protocol, worker loop; **no Tauri dependency**); `apps/desktop/src-tauri` (thin shell); `apps/desktop/src` (frontend) |
| core changes | `run_with`'s assembly and output stages extracted; a `session` module with `MatchState` save/load and `reassemble`; exclusions in discovery; progress in the silent stages; backend resolution moved out of the CLI into a new `sherd-backend` crate. **CLI output stays byte for byte** |
| workspace | a plain folder holding `sherd-workspace.json`; one collection per workspace; input linked by path, never copied; `runs/<id>/` history; heavy exports only on demand |
| review | decisions per fragment pair (`accept` with pose / `reject`), saved on every click; **instant draft reassembly** from a warm worker session; full-resolution refinement on a button and before any export |
| decisions → engine | `accept` = `must_join` with pose, `reject` = `must_not_join` (the existing `assembly::constraints`); carried into the next full run by default |
| frontend | React + TypeScript + Vite, Tailwind + shadcn/ui, Zustand, TanStack Virtual, i18next (ru, en); protocol types generated from Rust with `ts-rs` |
| viewer | three.js in a React-free `viewer/` module; per-fragment display meshes loaded once per workspace, a run is only matrices and groups on top of them |
| layout | three panes (tree · 3D · inspector), both side panes collapsible as in VS Code, three modes: Вход · Сборка · Ревью |
| export | the core's own writers, fed the reviewed assembly: the CLI's result folder whole or in parts; **Open in Blender** by generated script over the original scans |

## 1. Goals, requests, non-goals

The CLI works and nobody but its author can use it. The app is for the museum's staff: open a
folder of scans, see what is in it, run the assembly with a progress bar that means something,
look at the result, and — the screen the app exists for (D §12) — settle the joins the engine was
not sure about.

Requests from the chat this spec answers:

| who, when | request | where |
|---|---|---|
| Mary, 09-09 | the fragment's number under the cursor | §7.2 (carried over from `viewer.html`) |
| nadia, 09-11 | "можно как-то вручную подтверждать комбинации, которые он называет вероятными?" | §8 |
| nadia, 09-11 | "можно ему разрешить соединить всё, что он считает вероятным?" | §8.4 |
| nadia, 09-11 | could not open the assembly in Blender; wants groups pre-sorted for import | §9.2 |
| Mary, 09-16 | a duplicated scan (234009 / 249010) must be left out | §5.1 |
| user, 09-20 | workspaces, input preview, clear states, progress with time left, a better result preview | §5–§7 |

Left to later sub-projects (§12): inventory numbers, `.tfm`, per-group `placed/` folders, signing
and auto-update.

Non-goals: any change to the algorithm or its thresholds; placing a fragment by hand with the
mouse (a join the engine never proposed cannot be accepted in this version); comparing two runs
side by side; several collections in one workspace; cloud or sharing features.

## 2. Architecture

```
crates/
  sherd-core       + `session`: a saved match, and reassembly from it (§3)
  sherd-gpu        unchanged
  sherd-backend    NEW: `--backend` resolution and the self-test, moved from sherd-cli (§3.6)
  sherd-cli        same behaviour
  sherd-app-core   NEW: workspace model, protocol, worker loop, decisions → constraints,
                   status derivation inputs, ETA calibration, Blender script. No Tauri.
apps/desktop/
  src-tauri/       Tauri 2 shell: commands, events, worker spawn, asset scope
  src/             React frontend
```

### 2.1 Two roles, one binary

`main()` looks at its arguments before anything of Tauri is initialised. With `--engine-worker` it
runs `sherd_app_core::worker::serve(stdin, stdout)` and exits; otherwise it starts the app.

- **UI process.** Owns the workspace's state files, spawns the worker
  (`std::env::current_exe()`, `CREATE_NO_WINDOW` on Windows), forwards its events to the frontend,
  captures its stderr. Never loads a mesh or a fragment.
- **Worker.** Links `sherd-core` and `sherd-gpu`. Reads one `Job`, and for a review session further
  `Request`s, as JSON lines on stdin; writes `Event`s as JSON lines on stdout. stdin reaching EOF
  raises the `Cancel` flag and the worker exits at the next unit of work: a UI that dies leaves no
  3 GB orphan.

Why a process and not a thread: a run is 17–80 minutes, holds ~3 GB, and drives `wgpu` on
machines whose drivers nobody has tested. An abort (OOM, device loss inside a driver, stack
overflow) must end a run, not the window; the memory must go back to the OS when the run ends; and
a hard kill must exist behind the cooperative cancel. Why not the CLI as a sidecar: two binaries
to ship and sign, and the CLI's log is prose, not a protocol.

**Ownership rule.** The worker writes engine artefacts (`cache/`, `fragments/`, `match.state`,
`report.*`, `transforms.*`, `joins.csv`, `candidates.json`, everything under `exports/`). The UI
process writes workspace state (`sherd-workspace.json`, `run.json`, `decisions.json`,
`assembly.json`, `engine.log`). Every state file is written to a temporary name and renamed.

### 2.2 Protocol (`sherd_app_core::protocol`)

Serde enums, `#[derive(TS)]` for the frontend's bindings. The first event of every worker is
`Hello { protocol: u32, core_version, algo_ref, commit }`; a UI that reads another `protocol`
fails the job with `kind: protocol`.

| job | what it does | ends with |
|---|---|---|
| `Prepare { workspace, input, excluded, target_faces, seed, memory_budget, workers }` | preprocess every fragment into `cache/`; write `fragments/<name>.png`, `<name>.glb` and `index.json` | `Done` |
| `Run { workspace, run_id, spec, constraints }` | the full pipeline into `runs/<run_id>/`, meshes, previews and review images off; writes `match.state` and `candidates.json` | `Assembly` with every group `refined`, then `Done { summary }` |
| `Review { workspace, run_id }` | loads the fragment cache and `match.state`, then serves requests until `Close` or EOF | — |

Requests inside a review session:

| request | answer | cost on `karas` (155 fragments) |
|---|---|---|
| `Reassemble { decisions }` | `Assembly { groups: [{members, refined}], poses, joins, unplaced: [{a, b, reason}] }` | < 1 s |
| `PairDetail { a, b, pose }` | `PairDetail { points_b, class_b, seam_voxels, metrics }` — what `review.rs` draws, as data | < 1 s |
| `Refine { decisions }` | progress, then `Assembly` with every group `refined` | ≈ 15 s on a whole collection, less for a few groups; reads the source scans |
| `Export { decisions, what, dest }` | progress, then `Exported { files }` | ≈ 15–60 s |
| `Cancel`, `Close` | | |

Events common to all jobs: `Stage { name }`, `Progress { stage, done, total }`,
`FragmentReady { name, stats, warnings, … }` (Prepare only), `Failed { kind, message }`. The
engine's log is not an event: the worker's `tracing` output goes to stderr, which the host appends
to the run's `engine.log` — so a worker's last words survive a worker that dies mid-sentence.
`Stage` carries no `index`/`of`: the set of stages varies with
the options (`screen`, `second_pass`), so the window owns the strip and lights a stage by name when
its first report arrives.

`RunSpec` is the subset of `RunOptions` the app sets (§7.4); everything else is
`RunOptions::default()`.

## 3. Changes outside the app

Every one of them leaves the CLI's output byte for byte what it is; `crates/sherd-cli/tests/run_cli.rs`
and `parity --stage all` are the gate.

### 3.1 `sherd_core::session` — a saved match, and reassembly from it

`pipeline::run_with` is one 700-line function and stays the orchestrator. Two pieces of it become
functions, because a second caller needs exactly them:

- **`assemble_stage`** — `constraints::apply` → `assemble_under` → the object round. `run_with`
  has this block twice already (first pass, and again after R §8.1's second pass);
- **`write_outputs`** — R §11's writers and `export/`, the tail of `run_with`.

`session` is built on those two:

```
MatchState                                   the candidate list and the tier report as they stood
                                             before the last `constraints::apply`, with names,
                                             `Params`, thickness, resolution, backend label
RunOptions::match_state: Option<PathBuf>     `run_with` saves the state there; the CLI never sets it
load_fragments(entries, …) -> Vec<Fragment>  the cache-backed preprocessing `run_with` uses
reassemble(engine, fragments, &MatchState, Option<&Constraints>) -> Reassembled
refine_poses(…) -> Vec<Matrix4<f64>>         R §9 over the groups it is given
write_reviewed(out_dir, …)                   `write_outputs` over a `Reassembled`
```

Matching itself is not extracted: the app's `Run` job calls `run_with`, and nothing else needs the
match without the run around it.

**`reassemble` mirrors a full run under the same constraints.** A full run does not match a pair
pinned by `must_join` with a pose: the pair's only candidate is the pinned one, appended after every
matched candidate. `reassemble` reproduces that layout — it removes the pair's matched candidates
(and their rows of the tier report), and appends the pinned one. When the pinned pose is bit for bit
the pose of a candidate the match already scored — the case for every decision made in the app —
that candidate's scores are reused and R §6 is not run again; otherwise `pinned_candidate` scores
it, as the full run would. `must_not_join` is left to `assemble_under`'s veto, as in a full run whose
pair did get matched. The tier pass is not repeated: a decision changes what is placed, not what the
engine believed.

### 3.2 `MatchState::save` / `load`

`runs/<id>/match.state` is **JSON** through `serde_json` (whose `float_roundtrip` feature, already
on in this workspace, reads every `f64` back bit-identical), with a `format` tag and a `version`
number; another version is refused with a typed error, never misread. Not a compact binary format,
for a reason found in the tree: the types it carries — `Scores`, `Evidence`, `Probes`, `Params` —
are `report.json`'s own and use `skip_serializing_if`, which `report.json`'s bytes depend on and
which a non-self-describing format (`postcard`, `bincode`) cannot round-trip. About 35 MB and well
under a second to read for `karas`'s 54 000 candidates; no new dependency.

### 3.3 Discovery with exclusions

`collection::discover` gains a variant taking a set of excluded fragment names. The CLI does not
use it.

### 3.4 Progress in the silent stages

`Watch::advance` is called today from preprocessing and matching only. It is added to `tiers` (per
candidate probed) and to `refine` (per fracture cloud built, which is where R §9 reads the source
scans). Matching is 94 % of `karas`'s wall time, so the bar was already honest; this removes the
two places it stood still in every app run. The output stage writes only tables in an app run; its
per-file progress arrives with Export (milestone 6), which is what writes meshes.

### 3.5 Display meshes and thumbnails — no core change

`Prepare` needs a thumbnail and a display mesh per fragment. It gets them the way the scene export
does today: a second read of each source through the public `export::scene::display_mesh` and
`render`, under the same memory semaphore. That is ≈ 10–15 s for 155 scans, once per workspace —
the files are kept in `fragments/` — and it leaves preprocessing untouched. (An observer inside
preprocessing would save that read, and was rejected: a cache hit never loads the source, so the
observer would need a second path anyway.)

### 3.6 Backend resolution

`--backend auto|cpu|gpu`, the adapter choice and the self-test gate live in
`crates/sherd-cli/src/gpu.rs`. `resolve`, `Resolved`, `info_lines` and `selftest_lines` move,
unchanged, to a new crate `sherd-backend` with the same optional `gpu` feature — it cannot be
`sherd-gpu` itself, because the CPU-only build has no `sherd-gpu` and still needs `resolve`. The
`gpu-check` adapter stays in the CLI. The app and the CLI then resolve `auto` identically.

## 4. Workspace on disk

```
karas/
  sherd-workspace.json      format version; input path (absolute, and relative to the workspace
                            so both can move together); excluded fragments; last RunSpec
  sherd-workspace.lock      pid of the app that has it open
  cache/<name>.sherd        R §3.7's fragment cache, shared by every run
  fragments/<name>.png      thumbnail
  fragments/<name>.glb      display mesh, source-file coordinates
  fragments/index.json      per fragment: file, size, mtime, faces, thickness, watertight,
                            fracture area fraction, warnings
  runs/2026-09-20_1412/
    run.json                status; RunSpec and the full resolved Params; input snapshot
                            (file, size, mtime per fragment, and the exclusions); engine block;
                            timings; tier counts; app and core versions
    engine.log              everything the worker said
    match.state             §3.2 — JSON, ≈ 35 MB on `karas`
    report.json report.md transforms.json transforms.csv joins.csv     the engine's, immutable
    candidates.json         the UI's index: every confirmed and probable join, and per fragment
                            its ten best rejected candidates, each with pose, scores and reason
    decisions.json          §8.1
    assembly.json           the current assembly: groups, poses, joins used, `refined` per group
  exports/<date>_<what>/    written only by Export
```

A run in the app writes no `placed/`, no `review/`, no `preview_*.png`, no `scene.glb` and no
`viewer.html`: a run is tens of megabytes instead of ~1 GB, which is what makes keeping a history
affordable.

The run id is local time `YYYY-MM-DD_HHMM`, with `-2`, `-3` on collision. Deleting a run moves its
folder to the OS trash.

**Stale.** A run is stale when its input snapshot differs from a fresh scan of the input folder
(names, size, mtime) or from the workspace's exclusions. No content hashing — the input is
gigabytes, and a file changed under the same size and mtime is still caught by R §3.7's cache
validation at the next `Prepare`. Checked when the workspace opens, when the window regains focus,
and on demand.

**Run lifecycle.** `running → done | cancelled | failed { kind, message }`. A `running` found at
start-up with no live worker becomes `interrupted`. A failed run's folder stays in the history
with its log.

Recent workspaces, GPU self-test results, ETA calibration and settings live in the OS's app-config
directory, not in a workspace.

## 5. States

The workspace status is **derived**, never stored: a pure function
`deriveStatus(workspace, fragmentsIndex, runs, activeJob)` on the frontend, unit-tested against
every row.

| status | when | primary action |
|---|---|---|
| empty | no input path | Выбрать папку со сканами (button and folder drop) |
| preparing | `Prepare` is running | progress `37 / 155`; pause |
| ready | input prepared, no run | Собрать… |
| running | `Run` is running | Отменить |
| current | selected run `done` and not stale | Перегенерировать… (secondary) |
| stale | selected run `done` and stale | Перегенерировать… (primary); banner lists `+3 files, −1, 2 changed` |
| draft | a group of the current assembly is not refined | Уточнить позы |
| failed / cancelled / interrupted | selected run ended so | Показать лог · Повторить |
| input missing | input path does not resolve | Указать папку заново; results stay viewable, runs and refinement are off |

`Prepare` starts by itself when an input folder is linked or found changed: it is the
preprocessing a run needs anyway, and the thumbnails, display meshes and warnings fall out of it.

While a `Run` is going the app stays usable — input and earlier runs can be browsed — but the
review draft is off: one worker at a time.

### 5.1 Exclusion

«Исключить из сборки» on a fragment writes its name into `sherd-workspace.json`. The file is
untouched. Excluded fragments stay visible in Вход, struck through, and make existing runs stale.

## 6. Progress and time left

- Stages shown as a strip: Подготовка · Сопоставление · Уверенность · Сборка · Уточнение · Файлы,
  from `Stage` events; the bar inside the current one from `Progress`.
- **Time left** during matching: remaining pairs at the run's own pair rate (exponential moving
  average over the last 60 s), plus the later stages at the ratio to matching time recorded by the
  last completed run on this machine; with no history, `karas`'s ratios (tiers 3.3 %, refine 1.5 %,
  output 1.6 % of matching).
- Before a run, the launch sheet shows `pairs × calibrated seconds per pair`; with no calibration
  it says an estimate will appear a minute into the run.
- Dock / taskbar progress (`set_progress_bar`), an OS notification on completion or failure, and
  the system is kept awake while a `Run` or `Prepare` is going.

## 7. Interface

### 7.1 Frame

One window. Top bar: workspace name · run selector · mode tabs with their own status
(`Вход · 155 · 11 предупр.`, `Сборка · 17 групп`, `Ревью · 36 из 39`) · panel toggles · Экспорт ·
primary action. Left pane, 3D viewport, right inspector, a status line, and a pull-up engine-log
panel with a level filter. Both side panes collapse (`⌘/Ctrl+B`, `⌘/Ctrl+⌥B`); with both collapsed
the viewport fills the window.

```
src/
  app/      frame: welcome ↔ workspace, panes, collapsing
  modes/    input · assembly · review — each supplies a left and a right pane
  viewer/   three.js, no React; typed API and events
  state/    workspace · run · review (decisions, undo) · jobs (progress, ETA)
  ipc/      typed wrappers over Tauri commands and events; `bindings/` from ts-rs
  i18n/ ui/
```

Keys: `1 2 3` modes, `F` fit, `L` labels, `C` colours, `⌘Z` / `⇧⌘Z`; in review `A` accept,
`X` reject, `Space` skip.

### 7.2 Viewer

Display meshes are loaded once per workspace from `fragments/*.glb` through Tauri's asset protocol,
scoped to the open workspace. An assembly is applied as one matrix per fragment, so a draft
reassembly moves meshes and regenerates nothing. Face budget: `export::scene::DEFAULT_FACES`
(600 000) over the collection, as today.

**Matrix convention.** Files carry `M` row-major with `p' = M·p` on column vectors (README,
«Одно соглашение о матрице»). `THREE.Matrix4.set()` takes row-major arguments and `fromArray()`
column-major; the viewer has one conversion function and a golden test over a known transform.

Carried over from `viewer.html`: name, group and join count under the cursor; click selects;
double click flies to; labels on every fragment; colour modes scan / fragment / group; hide and
frame a group; search; deep link to a fragment.

New: layout «все группы разнесены» ↔ «одна группа»; the unassembled tray, hidden by default;
«Разъединить», a slider pushing a group's fragments away from its centroid; **ghost** — a
candidate's partner drawn translucent in its proposed pose; **seam** — `PairDetail`'s fracture
points of B coloured tight / within the gap limit / beyond, and the seam voxels; «Снимок PNG».
Picking by `three-mesh-bvh`.

`viewer.html` and `scene.glb` remain export artefacts and their template is not touched here.

### 7.3 Modes

- **Вход.** Left: thumbnail grid or list, filters all / with warnings / excluded. Centre: one
  fragment, toggles «Скан» / «Излом красным» (R §3.4's labels) / wireframe. Right: file, faces,
  wall thickness, watertight, fracture fraction, which group it landed in, warnings in plain words
  (thickness off the collection median by more than 40 %; mesh not closed — *its joins can never be
  confirmed by the engine*, R §6.4), «Исключить».
- **Сборка.** Left: tree of groups → fragments, «Без пары», search. Centre: the assembly. Right:
  inspector of the selected fragment (its joins by band; its candidates from **all** bands out of
  `candidates.json`, each with ghost on hover and accept / reject), of a group (members, joins,
  «Открыть в Blender»), or of a join.
- **Ревью.** §8.

### 7.4 Launch sheet, history, settings

«Собрать…»: preset **Стандарт** (defaults) / **Тщательно** (`tier_agree_seeds = 2`, about twice
the time) / **Свои**; backend auto · CPU · GPU with the adapter list; «перенести решения»; the time
estimate. «Свои» exposes eleven parameters — seed, target faces, tiers on/off, tier-agree-seeds,
thick-ratio, min-tight, max-gap, min-seam, max-pen, workers, memory budget — each with its
description and a reset. The parameter table is one Rust table in `sherd-app-core` whose
defaults are read from `Params::default()` and `RunOptions::default()`.

Run history is the run selector's dropdown: status, date, duration, counts; open, delete.

Welcome screen: recent workspaces, create, open. Settings: language, theme (system / light / dark),
default backend, memory and thread limits, Blender path.

## 8. Join review

### 8.1 Decisions

```json
{ "version": 1,
  "decisions": [
    { "a": "FY234111_reduced", "b": "FY234058_reduced", "verdict": "accept",
      "pose": [[…4×4…]], "source": "probable", "bulk": false, "at": "2026-09-20T14:31:07Z" },
    { "a": "FY234092_reduced", "b": "FY234060_reduced", "verdict": "reject", "at": "…" } ] }
```

The key is the unordered pair. `accept` carries the candidate's pose, so it does not depend on
`match.state` surviving. `reject` is on the pair as a whole — the semantics of `must_not_join` —
and accepting another pose of the same pair replaces it. A join the engine **confirmed** can be
rejected: on a real mixed collection it is wrong about once in forty (README, «Настоящая
коллекция»). Undo and redo are an in-memory history in the review store; the file is the current
state, written on every change.

### 8.2 Decisions → constraints

`sherd_app_core::decisions::to_constraints`: `accept` → `must_join { a, b, pose }`, `reject` →
`must_not_join [a, b]`. A decision naming a fragment that is no longer in the collection is dropped
and reported — `constraints::resolve` fails on an unknown name, so the filter is the app's duty.

### 8.3 The screen

Left: the queue, best score first, band filter (вероятные / подтверждённые / отклонённые ядром),
each row showing its verdict. Centre: the pair in its pose — A grey, B orange, the two colours of
the engine's review images — with «Пара» / «В сборке» / «Разъединить» and the seam colouring.
Right: the scores against their thresholds, **why the engine did not confirm it** in one sentence
built from the tier evidence (margin to the rival pose, independent support, redraws, slide,
penetration not measurable), and the pair's other poses. Bottom: the draft line — decisions made,
groups and unassembled before → after, undo, reset, «Уточнить позы».

### 8.4 Draft, bulk accept, refinement

Every decision sends `Reassemble`; the answer replaces `assembly.json` and the viewer's matrices.

R §9 refines a group by a spanning walk over its used joins, so **refined is a property of a
group**. The session keeps the last refined assembly — read from `assembly.json` when it opens,
replaced after every `Refine`. A reassembled group whose members and used joins are exactly those
of a refined group takes that group's poses and stays `refined`; any other group carries R §8's
unrefined poses and `refined: false`, and R §8.2's recentring is applied to every group either way.
With no decisions every group is unchanged, so `Reassemble` returns the run's assembly as it was.

«Принять все оставшиеся вероятные» expands into one `accept` per undecided probable join, flagged
`bulk`; the greedy assembly resolves conflicts by score as it always does, and what it could not
place comes back in `unplaced` with the engine's own reason. One undo reverts the whole expansion.

«Уточнить позы» sends `Refine`: R §9 at full resolution over the unrefined groups only; it needs
the input folder. Export refines
first by itself when anything is unrefined.

### 8.5 Carry-over

A new full run with «перенести решения» on receives the decisions as constraints: accepted pairs
are not matched again, rejected pairs are skipped before matching, so the run is faster. The new
run's `decisions.json` starts from the carried ones.

## 9. Export

### 9.1 Экспорт

A review-session request; the UI opens a session by itself when none is open. `what`:

| option | writes | through |
|---|---|---|
| Папка результата | what `sherd-refit-rs run` writes by default, plus opt-ins for review images, `--placed-all`, `--merged-meshes` | `session::write_outputs` |
| Таблицы | `transforms.csv`, `transforms.json`, `joins.csv`, `matrices/` | same |
| Сцена | `scene.glb`, `viewer.html` | same |
| Меши | `placed/*.ply` | same |
| Отчёт | `report.md`, `report.json` | same |

All of it reflects the reviewed assembly. The human's part is documented by the engine itself:
decisions arrive as constraints, and `report.md`'s `## Constraints` section already lists each and
what it did. The destination defaults to `exports/<date>_<what>/` and can be any folder; when done,
«Показать в папке».

### 9.2 Открыть в Blender

On a group or on the whole assembly. `sherd_app_core::blender::script` writes a Python script and
the UI launches `blender --python <script>`:

- imports the **original scans** with Blender's native PLY / OBJ / STL importers, axes given
  explicitly (`forward_axis='Y', up_axis='Z'`) so no importer default rotates anything, then sets
  each object's `matrix_world` to its `M`. No scale is applied; units stay the scan's;
- one Blender collection per group, objects named after fragments; groups offset by an empty
  parent each (R §8.2 centres every group on the origin), no offset when a single group is opened;
- a choice of full resolution or the display meshes (`fragments/*.glb`). `.off` sources, which
  Blender cannot read, always use the display mesh;
- Blender is looked for in the standard install locations of macOS and Windows and can be set in
  settings; when it is not found the script is saved and shown in its folder. Minimum Blender 4.0;
  the operators' names are verified against an installed Blender during implementation.

## 10. Errors

| kind | cause | what the user sees |
|---|---|---|
| `input` | a scan cannot be read or parsed | the file by name, and «Исключить и повторить» |
| `too_few` | fewer than two fragments | — |
| `disk` | write failed or no space | path and free space; space is also checked before a run or an export |
| `gpu` | adapter or device error | «Повторить на CPU» |
| `protocol` | `Hello.protocol` mismatch | reinstall hint |
| `internal` | a panic, caught by a hook; backtrace to `engine.log` | «Показать лог» |
| `crashed` | the process ended without `Done` / `Failed` | exit code, last 50 log lines, a hint about memory |
| `cancelled` | the button | — |

The GPU self-test (D §6.8) runs once per machine and driver in a worker and is cached in the app
config; when it fails `auto` resolves to the CPU and the launch sheet says why. A `match.state` of
another version leaves the run viewable and the review draft off, with the reason and
«Перегенерировать». A second app instance on a locked workspace is refused.

## 11. Testing, and its budget

The user's instruction of 2026-09-20: a reliable result, without tests, rebuilds and engine runs
after every small change — those cost hours here (`lto = "thin"`, `codegen-units = 1`,
dependencies at `opt-level = 2` even in dev; a real collection is 17–80 minutes). So tests go
where a bug would be **silent or expensive**, and nowhere else.

**What is tested**

- **Core.** `MatchState` save → load exact; `session::assemble` with no constraints equals the run's
  pre-refinement assembly bit for bit, and `Reassemble` with no decisions returns the run's refined
  assembly unchanged; an accepted probable join places its fragment; a rejected confirmed join
  splits its group; bulk accept names every conflict it could not place. The existing
  `run_cli.rs` and `parity --stage all` are the byte-for-byte gate of §3 and gain nothing new.
- **`sherd-app-core`.** Protocol round-trip and version refusal; decisions → constraints with
  dropped names; stale detection; interrupted-run recovery; **one** headless end-to-end test that
  drives the worker through pipes on `fixtures/slab`: Prepare → Run → Review → Reassemble →
  Export, then cancel and stdin EOF.
- **Frontend.** Vitest on pure logic only: `deriveStatus` over every row of §5, the review store
  (undo, bulk), the ETA estimator, the matrix golden test. No component tests, no Playwright, no
  WebDriver. The interface is checked by hand in the running app at the end of each milestone,
  against that milestone's list of flows.

**How often**

- While iterating: `cargo check`, and the tests of the one crate touched (`-p <crate>`).
- The workspace-wide gates — `run_cli.rs`, `parity --stage all`, clippy — once at the end of
  milestone 1, and afterwards only in a milestone that touched `sherd-core` or `sherd-gpu`.
- Everything automated runs on `fixtures/slab`. `karas` and `mixed_all` are never part of a test
  loop; there are two manual acceptance runs on `karas` in the app — after milestone 4 (run,
  progress, result) and after milestone 5 (review, draft, refinement, export).
- Release builds (`tauri build`) at milestones 3 and 6, not in between.
- **CI** carries the rest, off the developer's clock: `sherd-app-core` tests on the three OS
  runners, frontend lint / typecheck / Vitest, an unsigned `tauri build` on macOS and Windows.

## 12. What this spec leaves out, and to whom

| sub-project | contents |
|---|---|
| 3 — museum data and formats | inventory numbers from a filename ↔ number table, shown in lists and under the cursor (the mock-ups reserve the column); `.tfm` for Geomagic Wrap; `placed/` sorted into a folder per group |
| 4 — release | signing, notarisation, Windows certificate, auto-update, the tagged release pipeline (D §11) |

## 13. Milestones

Each leaves something that works.

1. **Core.** `assemble_stage` and `write_outputs` extracted, `session` with `MatchState` and
   `reassemble`, exclusions, progress in tiers and refine, `sherd-backend`. CLI byte for byte.
2. **`sherd-app-core`.** Protocol, worker, workspace model, `Prepare` and `Run`; the headless test.
3. **Shell and frame.** Tauri app, welcome, workspace, mode Вход with preparation and the
   single-fragment viewer.
4. **Run.** Launch sheet, progress, time left, cancel, history, mode Сборка with the full viewer.
5. **Review.** Warm session, decisions, instant draft, seam view, bulk accept, refinement,
   carry-over.
6. **Export and finish.** Экспорт, Открыть в Blender, settings, log panel, i18n pass, CI builds for
   macOS and Windows.
