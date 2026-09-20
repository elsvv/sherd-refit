# Desktop App, Milestone 4 — a run from the window, and the «Сборка» mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Press «Собрать…», choose how, watch the stages go by with the pairs counted and an honest «осталось ≈ 11 мин», cancel if need be; afterwards see the assembled groups in 3D — hover for the fragment's name, click for its joins, spread the groups out or isolate one, explode a group to look at its seams — and come back to any earlier run.

**Architecture:** The shell's job thread, which milestone 3 wrote for `Prepare`, becomes one function for both job kinds; a `Run` additionally files `assembly.json` when the worker reports it, finishes `run.json`, and teaches a per-machine calibration (seconds per pair, later stages as a ratio of matching) that the launch sheet's estimate reads. The window loads every fragment's display GLB once and lays a run over them as matrices (A §7.2); groups are packed on a plane by a pure function; picking goes through `three-mesh-bvh`. Time left is a pure function of the progress events. Everything visual is looked at with `tools/look.mjs`.

**Tech Stack:** as milestone 3, plus Rust `keepawake =0.6.1`, `trash =5.2.9`, `tauri-plugin-notification =2.4.0`; npm `three-mesh-bvh 0.9.15`, `@tauri-apps/plugin-notification 2.4.0`. Spec: `docs/superpowers/specs/2026-09-20-desktop-app-design.md` (`A §n`). Screens: `docs/superpowers/specs/desktop-app-mockups/main-layout.html` (variant A — the «Сборка» mode) and part 2 of `states-and-input.html` (the running job).

## Global Constraints

- **Toolchain.** Every shell that runs cargo (or `pnpm tauri`) starts with `export PATH="$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin:$PATH"`. JavaScript: pnpm 9, from `apps/desktop/`; exact versions, no `^`.
- **Verification budget (A §11).** Run only what a task lists. No Playwright/WebDriver/component tests; Vitest on pure logic only. No `cargo test --workspace`, no clippy inside a task (Task 10 is the gate). Nothing automated runs on anything but `fixtures/slab`. `cargo test -p sherd-cli --test run_cli` is **not** part of this milestone: nothing here touches `sherd-core`.
- **Looking at a screen is part of building it.** Every task that changes what the window shows ends with `node tools/look.mjs '<steps>' look/<task>` from `apps/desktop/` (it starts Vite and a headless Chrome itself, drives the **mock** IPC layer, writes PNGs) and with the agent **reading those PNGs** and fixing what is wrong in them. Read `tools/look.mjs`'s header for the step syntax. The mock must therefore be able to play every state a task builds. `look/` is git-ignored; never commit it.
- **No panics on data from outside** in Rust; in TypeScript no `any`, no non-null assertions.
- **Every user-visible string** through i18next, `ru.json` and `en.json` with identical key sets; Russian is the primary wording.
- **Render on demand**, never a free-running loop. Dispose what you create.
- **The window never reads `report.json` or `match.state`.** A run is `assembly.json` + `candidates.json` + `run.json`, through shell commands.
- **Ownership (A §2.1):** the worker writes engine artefacts; `run.json`, `assembly.json`, `engine.log`, the calibration and `sherd-workspace.json` are written by host-side code only, atomically.
- **Commits:** branch `desktop-app`, subject `M4.<task>: <sentence>`, trailer as your own system instructions give it.
- **Do not touch** `crates/sherd-core`, `crates/sherd-parity`, `sherd_refit/`, `fixtures/`.

## Where things are (tree at `734018f`)

```
crates/sherd-app-core/src/   host.rs (start_run, finish_run, save_assembly, drive, Worker, Canceller, Outcome)
                             protocol.rs (RunSpec, Event, AssemblyDto, CandidateRow, RunCounts…)   run.rs   view.rs   worker/
apps/desktop/src-tauri/src/  main.rs  state.rs (AppState, JobSlot)  commands.rs  jobs.rs (start_prepare, cancel, finish, EngineEvent, EngineFinished, OutcomeDto)  recent.rs  error.rs
apps/desktop/src/            ipc/{api.ts, tauri.ts, mock.ts, bindings/}   state/{workspace.ts, jobs.ts, ui.ts, status.ts}
                             app/{Frame, TopBar, StatusLine, Banner, Welcome, shortcuts}   modes/input/   viewer/{FragmentViewer.ts, FragmentView.tsx, matrix.ts, principal.ts}   ui/
apps/desktop/tools/look.mjs
```

`host::start_run(ws, &WorkerCommand, &RunSpec, constraints: Option<Constraints>, carried_from: Option<String>, now) -> Result<(RunFile, Worker)>`; `host::finish_run(ws, &mut RunFile, &Outcome, now)`; `host::save_assembly(run_dir, &AssemblyDto)`; a worker's `Run` emits `Stage`/`Progress` for `preprocess`, `matching`, `tiers`, `refine`, then `Assembly`, then `Done { counts, engine, params }`. `RunCounts { fragments, pairs, skipped_pairs, candidates, confirmed, probable, groups, unassembled, timings: Vec<StageTime { stage, seconds }> }`.

---

### Task 1: what milestone 3's reviews left open

Small, each a line or a few; all from the two reviews' minor findings.

**Files:** `apps/desktop/src-tauri/src/{main.rs, jobs.rs, commands.rs}`, `apps/desktop/src/state/{jobs.ts, workspace.ts}`, `src/viewer/FragmentViewer.ts`, `src/ipc/mock.ts`, `src/modes/input/InputLeft.tsx`

- [ ] **Step 1: The shell**
  - `jobs.rs`: the job slot is freed by a **drop guard** created on the job thread before `host::drive`, so a panic inside the drive or the emit closure cannot leave the app `busy` for the rest of the session. The guard's `Drop` empties the slot; the normal path still empties it *before* emitting `engine:finished` (keep that order) — make the guard idempotent.
  - `main.rs`: `std::env::args_os()` and a comparison against `OsStr::new(ENGINE_WORKER)` — `args()` panics on an argument that is not Unicode, which a non-UTF-8 install path is.
  - `commands.rs`, `open_with`: open the **new** workspace first and replace the held one only when that succeeded; opening the path that is already open returns its view. A mistyped folder must not close the user's workspace.
  - `main.rs`, window role: a `tracing_subscriber::fmt()` to stderr (`EnvFilter`, default `sherd=info,sherd_desktop=info`) so `recent.rs`'s warnings go somewhere. Add `tracing-subscriber = { version = "0.3.23", features = ["env-filter"] }` to the shell's manifest. The worker role must still initialise its own and only its own (it branches first).
- [ ] **Step 2: The window**
  - `state/jobs.ts`, `applyFinished`: when the payload's `view` is `null`, call the workspace store's `refresh()` instead of keeping a view whose `job` is still set.
  - `lastFailure` is cleared when the workspace changes (open, create, close).
  - `FragmentViewer.show`: two calls for the same URL before the first resolved share **one** load (a map of in-flight promises by URL), so neither scene is leaked.
  - `ipc/mock.ts`: `workspaceOpen`/`Create`/`Close` reject with `{ kind: "busy", … }` while the mock job plays, as the shell does.
  - `InputLeft`: the scroll container itself is focusable (`tabIndex={0}`) and carries the key handler, so the list stays reachable by Tab when the selected cell is scrolled out of the virtual window.
- [ ] **Step 3: Verify** — `cd apps/desktop/src-tauri && cargo check 2>&1 | tail -2 && cargo test 2>&1 | tail -3`; `cd apps/desktop && pnpm typecheck && pnpm lint && pnpm test 2>&1 | tail -4`. Expected: clean; 22 tests.
- [ ] **Step 4: Commit** — `git commit -m "M4.1: what milestone 3's reviews left open -- a job thread that cannot leave the app busy, an open that cannot close the workspace it replaces, and five smaller things"`

---

### Task 2: calibration — what this machine takes per pair

**Files:**
- Create: `crates/sherd-app-core/src/eta.rs`
- Modify: `crates/sherd-app-core/src/lib.rs`; regenerate bindings

**Interfaces:**
- `eta::Calibration { version: u32, runs: u32, pair_seconds: f64, ratios: BTreeMap<String, f64> }` (`Clone, Debug, PartialEq, Serialize, Deserialize`, TS-exported; `Default` = `runs: 0`, `pair_seconds: 0.0`, `ratios` = A §6's `karas` ratios: `tiers 0.033`, `refine 0.015`, `output 0.016`)
- `Calibration::learn(&mut self, counts: &RunCounts)` — from a finished run with `pairs > 0` and a positive `matching` timing: `pair_seconds` = matching seconds ÷ pairs; each other stage in `timings` except `preprocess` and `matching` → its ratio to matching. Both are blended into what is known as an exponential moving average with weight ½ once `runs > 0`, taken whole on the first run. A run with no usable timing changes nothing.
- `Calibration::estimate(&self, pairs: usize) -> Option<f64>` — seconds for a run of `pairs` pairs: `pairs · pair_seconds · (1 + Σ ratios)`; `None` while `runs == 0`.
- `Calibration::{load(path: &Path) -> Self, save(&self, path: &Path) -> Result<()>}` — a missing or damaged file is `Default`.
- `eta::pairs_upper_bound(fragments: usize) -> usize` = `n·(n−1)/2` (R §4.1 skips some by wall ratio; the sheet says «до»).

- [ ] **Step 1: Write the failing tests** (in `eta.rs`): (a) `learn` on `RunCounts { pairs: 11_097, timings: [preprocess 0.1, matching 996.9, tiers 33.2, assembly 0.1, refine 15.3] }` gives `pair_seconds ≈ 0.08984` (1e-4) and `ratios["tiers"] ≈ 0.0333`; (b) a second, twice-as-slow run moves `pair_seconds` to the mean of the two; (c) `estimate` is `None` before any run and `pairs · pair_seconds · (1 + Σ ratios)` after; (d) a run with `pairs: 0` or no `matching` timing leaves the calibration equal to what it was; (e) `load` of a missing path and of a file holding `"garbage"` is `Default`; (f) `pairs_upper_bound(155) == 11_935`, `(0) == 0`, `(1) == 0`.
- [ ] **Step 2: Implement; regenerate the bindings** — `TS_RS_EXPORT_DIR="$PWD/apps/desktop/src/ipc/bindings" cargo test -p sherd-app-core --features ts --lib 2>&1 | tail -2` → `Calibration.ts` appears, no `bigint` in it.
- [ ] **Step 3: Verify** — `cargo test -p sherd-app-core --lib eta 2>&1 | tail -3` → 6 passed.
- [ ] **Step 4: Commit** — `git commit -m "M4.2: what this machine takes per pair -- learnt from every finished run, and what a run of N pairs will take"`

---

### Task 3: a run from the shell

**Files:**
- Modify: `apps/desktop/src-tauri/src/{jobs.rs, commands.rs, main.rs, state.rs}`, `apps/desktop/src-tauri/Cargo.toml`

**Interfaces:**
- Commands: `run_start(spec: RunSpec) -> Result<String, CommandError>` (the new run's id), `calibration() -> Result<CalibrationView, CommandError>` where `CalibrationView { calibration: Calibration, pairs_upper_bound: usize, estimate_seconds: Option<f64> }` for the **open** workspace's included scans.
- `jobs::start(app, state, kind: JobStart) -> Result<Option<String>, CommandError>` where `enum JobStart { Prepare, Run { spec: RunSpec } }` — one function for both; `start_prepare` becomes a call to it.
- Events unchanged in shape; for a run `run_id` is `Some`.

What a run adds on the job thread, in this order:
1. before spawning: `ws.set_last_spec(serde_json::to_value(&spec))`; `host::start_run(ws, &worker_command()?, &spec, None, None, Local::now())` — constraints and carry-over are milestone 5's;
2. in the drive closure: on `Event::Assembly(a)` → `host::save_assembly(&run_dir, a)` (an error is logged with `tracing::error!` and does not stop the run), then forward every event as before;
3. after the drive: `host::finish_run(ws, &mut run, &outcome, Local::now())`; on `Outcome::Done { counts: Some(c), .. }` → `Calibration::load(path)`, `learn(&c)`, `save(path)` with `path = app_config_dir/calibration.json`; then free the slot, build the view, emit `engine:finished`.

The workspace lock is **not** held across the drive: take what the thread needs (paths, the `RunFile`) before it starts, and re-lock briefly for `finish_run` and `view::build`. If the workspace was closed meanwhile (it cannot be — `busy` — but do not `unwrap` on it), finish with a logged error.

- [ ] **Step 1: Implement.** `cargo check` clean.
- [ ] **Step 2: The shell's binary runs the slab as a worker** (the run path end to end without a window; the host side is `sherd-app-core`'s tested code):

```bash
cd apps/desktop/src-tauri && cargo build 2>&1 | tail -1
WS="$(mktemp -d)/ws" && mkdir -p "$WS"
SPEC='{"preset":"standard","backend":"cpu","adapter":null,"gpu_memory_gb":null,"seed":0,"target_faces":200000,"tiers":true,"agree_seeds":1,"thick_ratio":2.5,"min_tight":0.25,"max_gap":0.03,"max_pen":0.005,"min_seam":3.0,"workers":0,"memory_gb":null}'
printf '%s\n' "{\"job\":\"run\",\"workspace\":\"$WS\",\"input\":\"$PWD/../../../fixtures/slab/input\",\"run_id\":\"t\",\"excluded\":[],\"spec\":$SPEC,\"constraints\":null}" \
  | ../../../target/debug/sherd-desktop --engine-worker | grep -o '"event":"[a-z_]*"' | sort | uniq -c
ls "$WS/runs/t"
```

Expected: `hello`, several `stage`/`progress`, one `assembly`, one `done`; the folder holds `match.state`, `candidates.json`, `report.json`. (Take `RunSpec`'s exact field names and defaults from `apps/desktop/src/ipc/bindings/RunSpec.ts` / `protocol.rs` if the JSON above is refused; `agree_seeds`' default is `Thresholds::default().agree_seeds`.)
- [ ] **Step 3: Commit** — `git commit -m "M4.3: a run from the shell -- its assembly filed when the worker reports it, its run.json finished, and the machine's calibration taught"`

---

### Task 4: a run's data, and the machine around a job

**Files:**
- Modify: `apps/desktop/src-tauri/src/{commands.rs, jobs.rs, main.rs}`, `Cargo.toml`, `capabilities/default.json`; `apps/desktop/package.json`

**Interfaces — commands:**

| command | arguments | returns |
|---|---|---|
| `run_assembly` | `run_id: String` | `AssemblyDto` (`assembly.json`) |
| `run_candidates` | `run_id: String` | `Vec<CandidateRow>` (`candidates.json`) |
| `run_log` | `run_id: Option<String>, max_lines: usize` | `String` — the tail of the run's `engine.log`, or of `prepare.log` for `None` |
| `run_delete` | `run_id: String` | `WorkspaceView` — the folder goes to the OS trash (`trash::delete`), never `remove_dir_all`; refused with `busy` for the run in the job slot |
| `engine_info` | — | `EngineInfoView { backends: Vec<String> }` — one `Job::Info { selftest: false }` worker, its `Event::Info`; cached in `AppState` after the first call |

A `run_id` is validated before it becomes a path: it must be the id of a run `run::list` returns (no `..`, no separators reach the file system). A missing `assembly.json` is a `CommandError { kind: "io" }`, not a panic.

**The machine around a job** (in `jobs::start`, both kinds): a `keepawake` guard (`Builder::default().idle(true).reason("sherd-refit: a job is running").app_name("Sherd Refit")`) held for the job's life — a run is 17–80 minutes and a sleeping laptop loses it; the main window's `set_progress_bar` fed from `Progress` events of the stage `matching` for a run and `preprocess`/`display` for a prepare (`ProgressBarStatus::Normal` with a percentage, `None` when the job ends); and when the job ends while the window is not focused, one notification through `tauri-plugin-notification` — title «Sherd Refit», body by outcome («Сборка завершена: 17 групп», «Сборка не удалась», «Подготовка завершена») — in the UI's language, which the window passes to `run_start`/`prepare_start` as `lang: String` (`"ru" | "en"`, anything else → `ru`). A failure of any of the three is logged and ignored: none of them is the job.

Add `tauri-plugin-notification = "=2.4.0"`, `keepawake = "=0.6.1"`, `trash = "=5.2.9"` to the shell's manifest, `"notification:default"` to the capability, `@tauri-apps/plugin-notification 2.4.0` to `package.json` (the Rust side sends the notification; the npm package is there for the permission request the window makes once, on the first run).

- [ ] **Step 1: Implement**, with one unit test in `commands.rs` for the pure helper `tail(text: &str, max_lines: usize) -> &str` (the last `max_lines` lines; all of it when shorter; `""` for `0`).
- [ ] **Step 2: Verify** — `cd apps/desktop/src-tauri && cargo test 2>&1 | tail -3 && cargo check 2>&1 | tail -2`; `cd apps/desktop && pnpm install && pnpm typecheck`.
- [ ] **Step 3: Commit** — `git commit -m "M4.4: a run's assembly, candidates and log for the window, its folder to the trash -- and a machine that stays awake, shows the progress in the dock and says when it is done"`

---

### Task 5: the window's side of a run — API, mock, stores, time left

**Files:**
- Modify: `apps/desktop/src/ipc/{api.ts, tauri.ts, mock.ts}`, `src/state/{jobs.ts, workspace.ts}`
- Create: `src/state/{assembly.ts, eta.ts, eta.test.ts}`

**Interfaces:**
- `Api` gains: `runStart(spec: RunSpec, lang: Language): Promise<string>`, `prepareStart(lang)`, `calibration(): Promise<CalibrationView>`, `runAssembly(id)`, `runCandidates(id)`, `runLog(id: string | null, maxLines: number): Promise<string>`, `runDelete(id): Promise<WorkspaceView>`, `engineInfo(): Promise<{ backends: string[] }>`.
- `state/assembly.ts` (Zustand): `{ runId: string | null, assembly: AssemblyDto | null, candidates: CandidateRow[], loading: boolean, error: CommandError | null, load(runId), clear() }` — `load` fetches both; selecting a run in the workspace store triggers it; a `finished` run job selects the new run and loads it.
- `state/eta.ts`:

```ts
export interface StageSample { stage: string; done: number; total: number; at: number }   // `at` in ms
/** Seconds left, or null when there is nothing to go on yet. Pure. */
export function remainingSeconds(samples: readonly StageSample[], now: number, ratios: Readonly<Record<string, number>>): number | null
```

  Rule (A §6): only `matching` is extrapolated — its rate is the pairs done in the last 60 s of samples (or since its first sample when younger than 60 s; `null` until 5 s and 1 % have passed); remaining = pairs left ÷ rate + (matching's projected total seconds × Σ ratios of the stages that have not reported a finished `done === total` yet). Before `matching` starts: `null`. After `matching` ended: projected matching seconds × Σ ratios of the unfinished later stages, minus the time since matching ended, floored at 0. `jobs.ts` keeps the samples (at most one a second per stage) and the calibration's ratios.
- The mock plays a **run**: `runStart` → over ~10 s: `stage`/`progress` for `preprocess` (12), `matching` (66 pairs), `tiers`, `refine`, then an `assembly` event and `finished` with a view that has one more `done` run. The mock's assembly: the twelve mock fragments in three groups (5, 4, 2 members) plus one unassembled; poses that fan the members of a group out around the origin (rotations about Z by multiples of 40°, translations of ~180 units along the rotated X, so that identical slabs do not coincide); `candidates`: for each group the chain of joins between consecutive members as `confirmed` and `used`, plus four `probable` and a handful of `rejected` rows with plausible `scores`. It starts with one finished run already in the history, one `failed` run (kind `gpu`) and one `cancelled`, so every state of the selector can be looked at.

- [ ] **Step 1: Write the failing test** `eta.test.ts`: (a) `null` before any `matching` sample, and before 5 s / 1 %; (b) 1 000 of 11 000 pairs in 100 s at a steady rate, ratios `{tiers: 0.03, refine: 0.02}` → `1000 + 1100·0.05 = 1055` s (±1); (c) the rate is the **recent** one: 100 s at 10 pairs/s then 60 s at 5 pairs/s → remaining uses 5 pairs/s; (d) after `matching` ended, with `tiers` finished and `refine` not: projected matching seconds × `refine`'s ratio minus the time since, never negative; (e) a stage absent from `ratios` costs nothing.
- [ ] **Step 2: Implement.** `pnpm test` → 27 tests; `pnpm typecheck && pnpm lint` clean.
- [ ] **Step 3: Commit** — `git commit -m "M4.5: the window's side of a run -- the API and a mock that plays one, the selected run's assembly, and time left as a pure function of the progress"`

---

### Task 6: «Собрать…» — the launch sheet, the running job, cancel

**Files:**
- Create: `apps/desktop/src/app/{LaunchSheet.tsx, RunOverlay.tsx, params.ts}`, `src/ui/{Dialog.tsx, NumberField.tsx, Radio.tsx}`
- Modify: `src/app/{TopBar.tsx, Frame.tsx, StatusLine.tsx}`, `src/i18n/*.json`

**The sheet** (a modal dialog, 520 px, `role="dialog"`, focus trapped, `Esc` closes): title «Сборка коллекции»; **Пресет** as three radio cards — «Стандарт» (the engine's defaults), «Тщательно» («сопоставить дважды с разными сидами и оставить только то, в чём оба прогона сошлись; примерно вдвое дольше» — `agree_seeds: 2`), «Свои параметры»; **Вычисления**: Авто · CPU · GPU, and under it `engineInfo().backends` as small `text-muted` lines; with «Свои параметры» an **eleven-row** table (A §7.4): seed, target faces, tiers on/off, agree-seeds, thick-ratio, min-tight, max-gap, min-seam, max-pen, workers (0 = авто), memory budget GB (empty = половина памяти) — each a `NumberField` (or a switch) with its default beside a «сбросить» button and a one-sentence description under it; the descriptions are the CLI's own doc comments of those flags (`crates/sherd-cli/src/main.rs`, struct `RunArgs`), translated — `params.ts` is the one table of `{ key, kind, min, max, step }` and the i18n keys carry the wording; a value outside `[min, max]` disables the button and says why. **Оценка**: from `calibration()` — «≈ 18 мин на этой машине (до 11 935 пар)» or, with no calibration, «Оценка появится через минуту после старта — это первая сборка на этой машине». The button «Собрать» calls `runStart(spec, language)`; the sheet opens pre-filled from `view`'s last spec when it parses.

**The overlay** (part 2 of `states-and-input.html`): while a `run` job is on, a card at the bottom centre of the viewport, 480 px: the current stage's name large; «прошло 6:41 · осталось ≈ 11 мин» (or «оценка времени появится скоро»); the stage strip Подготовка · Сопоставление · Уверенность · Сборка · Уточнение · Файлы — done ones `ok`, the current one `accent` and wider; a `Meter`; «4 210 из 11 097 пар» for `matching`, `done / total` otherwise; the backend from the run's spec. The strip lights a stage by the **name** of the last `stage` event (`preprocess`, `matching`, `tiers`, `refine`; `assembly` and `output` light when the next one starts or the job ends — they report no progress). «Отменить» in the top bar asks once («Остановить сборку? Уже сопоставленные пары не сохранятся.») and calls `jobCancel()`; while the worker winds down the button says «Останавливается…».

`deriveStatus` already answers `running`; the primary action for `ready`, `current`, `stale`, `failed`, `cancelled`, `interrupted` opens the sheet («Собрать…» / «Перегенерировать…» / «Повторить…»).

- [ ] **Step 1: Implement.** `pnpm typecheck && pnpm lint && pnpm test 2>&1 | tail -3`.
- [ ] **Step 2: Look at it.** `node tools/look.mjs` with steps that: create → link → wait for the preparation → open the sheet (`shot`) → choose «Свои параметры» (`shot`) → «Собрать» → `shot` at 3 s (matching under way, time left shown or promised) → `shot` after it ended. Read the four PNGs. Also once with `{"storage":{"sherd.theme":"dark","sherd.language":"en"}}`. Fix what is wrong.
- [ ] **Step 3: Commit** — `git commit -m "M4.6: Собрать… -- a sheet that says what the choice costs, and a running job that says where it is and how long it has left"`

---

### Task 7: the history of runs, and the states of one

**Files:**
- Create: `apps/desktop/src/app/{RunSelector.tsx, LogDrawer.tsx}`
- Modify: `src/app/{TopBar.tsx, Frame.tsx, Banner.tsx}`, `src/state/{workspace.ts, ui.ts}`, `src/i18n/*.json`

The run selector (top bar, left): a chip showing the selected run — «20.09 14:12 · 17 групп», with a coloured dot by status — opening a menu of all runs, newest first: status dot, date and time from `created` in the UI's language, duration (`finished − created`, `m:ss` or `ч:мм`), and by status «17 групп · 81 без пары» / «не удалась: GPU» / «отменена» / «прервана»; the stale ones carry a small `warn` «устарел». Selecting loads the run (Task 5's store). Each row has a «…» menu: «Удалить» (confirm: «Папка прогона будет перемещена в корзину.») → `runDelete`. With no run: the disabled «Прогонов ещё нет» of milestone 3. After a finished run job the new run is selected; on opening a workspace, the newest `done` run is.

Banners (under the top bar, one at a time, by `deriveStatus`): `stale` → warn, «Вход изменился с этой сборки: +3 файла, −1, 2 изменены, 1 исключён» (only the non-zero parts; plural forms) + «Перегенерировать…»; `failed` → danger, the failure kind's own sentence (`failure.<kind>`: `gpu` «Видеокарта не справилась…», `input` «Не удалось прочитать скан…», `disk`, `internal`, `crashed` «Процесс сборки завершился аварийно — возможно, не хватило памяти», …) then the engine's message in `text-muted`, + «Показать лог» and «Повторить…» (for `gpu`: «Повторить на CPU», which opens the sheet with `backend: "cpu"`); `cancelled` / `interrupted` → warn with «Повторить…».

The log drawer: a panel pulled up from the status line (`Mod+J`, and the «Показать лог» actions), 40 % of the height, monospace 11 px, the last 400 lines of `runLog(selectedRunId ?? null, 400)`, a level filter (all · warnings and errors — by the `WARN`/`ERROR` token of `tracing`'s format), «Обновить», auto-refresh every 2 s while a job runs, and it keeps its scroll position unless it was at the bottom.

- [ ] **Step 1: Implement.** Checks as before.
- [ ] **Step 2: Look at it** — the selector open with its three kinds of run (`shot`), the failed run selected with its banner (`shot`), the log drawer open (`shot`). Read them; fix.
- [ ] **Step 3: Commit** — `git commit -m "M4.7: the history of runs -- any of them one click away, a failed one saying why in a sentence, and the engine's log under the window"`

---

### Task 8: the assembly in 3D

**Files:**
- Create: `apps/desktop/src/viewer/{AssemblyViewer.ts, layout.ts, layout.test.ts, colours.ts, AssemblyView.tsx}`
- Modify: `apps/desktop/package.json` (`three-mesh-bvh 0.9.15`)

**Interfaces:**
- `layout.ts` (pure):

```ts
export interface Disc { id: number; radius: number }
export interface Placed { id: number; x: number; y: number }
/** Discs packed on shelves, largest first, into a block about `aspect` wide to 1 tall, `gap` apart; centred on the origin. */
export function packDiscs(discs: readonly Disc[], gap: number, aspect?: number): Placed[]
```

- `colours.ts`: `groupColour(index: number): string` — twelve distinguishable hues that read on `--viewport` in both themes, cycling; `fragmentColour(index)` likewise from a second palette; both pure, used by the tree's dots as well.
- `AssemblyViewer` (no React):

```ts
export interface AssemblyInput {
  fragments: { name: string; url: string }[];             // every fragment of the run's input snapshot that has a display GLB
  assembly: AssemblyDto;
}
export type ColourMode = "scan" | "fragment" | "group";
export type LayoutMode = "spread" | "single";
export interface ViewerEvents {
  onHover(name: string | null, at: { x: number; y: number } | null): void;
  onSelect(name: string | null): void;
  onProgress(loaded: number, total: number): void;
}
export class AssemblyViewer {
  constructor(canvas: HTMLCanvasElement, events: ViewerEvents);
  load(input: AssemblyInput): Promise<void>;       // GLBs fetched at most 4 at a time, each shown as it arrives
  setAssembly(assembly: AssemblyDto): void;        // matrices and groups only — no mesh is reloaded (A §7.2; milestone 5's draft reassembly)
  setColourMode(mode: ColourMode): void;
  setLayout(mode: LayoutMode, group?: number): void;
  setUnassembledVisible(on: boolean): void;        // off by default: 81 loose sherds are clutter
  setGroupVisible(group: number, on: boolean): void;
  setExplode(amount: number): void;                // 0…1: members pushed from their group's centroid by amount × the group's radius
  select(name: string | null): void;               // outline/emphasis on one fragment
  flyTo(name: string): void;
  fit(): void;
  screenshot(): string;                            // a PNG data URL of the current frame
  dispose(): void;
}
```

How it is built:
- One `THREE.Group` per assembly group holding one `Object3D` per member; a member's `matrix` is `rowsToMatrix4(assembly.poses[name])` with `matrixAutoUpdate = false` — **the only route from a file's matrix into the scene** (`viewer/matrix.ts`). Every group is centred on the origin by the engine (R §8.2), so in `spread` layout a group's node is translated to `packDiscs`' place for its bounding sphere (in the XY plane, `gap` = 15 % of the largest radius) and in `single` layout only the chosen group is visible, at the origin. Groups of one are the «unassembled tray»: a second block under the first, shown only when asked.
- `computeBoundsTree` of `three-mesh-bvh` on every geometry after load, `acceleratedRaycast` on `Mesh.prototype.raycast`, `firstHitOnly`; hover picks at most once per animation frame and only when the pointer moved; 600 000 faces must hover smoothly. Dispose the bounds trees with the geometries.
- Colour modes swap **materials**, not vertex data: `scan` = the loaded materials; `fragment`/`group` = one shared `MeshStandardMaterial` per colour (`vertexColors: false`, roughness 0.9, double-sided). The selected fragment is drawn with an emissive lift, the hovered one slightly less.
- The same lighting, on-demand rendering, resize handling, context-loss and disposal discipline as `FragmentViewer` — read it first and share what can be shared (a small `viewer/stage.ts` for renderer + camera + controls + lights + the dirty flag is welcome if both viewers then use it; do not leave two copies drifting).
- `fit()` frames what is visible; the default direction is the visible set's principal axes as in `FragmentViewer` (a block of groups lying in XY is looked at from above at a tilt).

- [ ] **Step 1: Write the failing test** `layout.test.ts`: (a) no two placed discs overlap, with the gap, for 17 discs of mixed radii; (b) the block is centred on the origin (bounding box centre within 1e-6) and roughly `aspect` wide to tall (within 40 %); (c) one disc sits at the origin; none → `[]`; (d) the order of the result follows the input ids, whatever order the packing took them in; (e) deterministic: the same input twice gives the same output.
- [ ] **Step 2: Implement** `layout.ts`, `colours.ts`, the viewer, and `AssemblyView.tsx` (a canvas + a thin loading bar from `onProgress` + the hover tooltip positioned at the pointer: name · «группа 3» / «без пары» · «4 стыка»).
- [ ] **Step 3: Verify** — `pnpm test` (32 tests), `pnpm typecheck`, `pnpm lint`, `pnpm build 2>&1 | tail -3`.
- [ ] **Step 4: Commit** — `git commit -m "M4.8: the assembly in 3D -- every fragment's mesh loaded once and a run laid over them as matrices, groups packed apart, a name under the cursor"`

(It is looked at in Task 9, where it is on a screen.)

---

### Task 9: the «Сборка» mode

**Files:**
- Create: `apps/desktop/src/modes/assembly/{AssemblyLeft.tsx, AssemblyCentre.tsx, AssemblyRight.tsx, tree.ts, tree.test.ts}`
- Modify: `src/app/{Frame.tsx, TopBar.tsx, shortcuts.ts}`, `src/state/ui.ts`, `src/i18n/*.json`

**Left** (300 px): a search box («поиск: FY2340…», filters by substring, case-insensitive, keeps the groups of the matches open); then a **virtualised tree** — «Группа 0 · 20 фр.» rows with a colour dot (`groupColour`), a chevron, an eye toggle (`setGroupVisible`) and, on hover, «показать одну» (`setLayout("single", k)`); under an open group its members, indented; last «Без пары · 81 фр.» collapsed, with an eye that is off by default. `tree.ts` is the pure part — `buildRows(assembly, open: Set<number>, query: string): Row[]` — with `tree.test.ts`: groups largest first as the engine gave them; a closed group hides its members; a query keeps only matching members and opens their groups; «без пары» is one section of all singleton groups; counts are right. Selecting a row selects in the viewer and the other way round; the selected row scrolls into view.

**Centre:** `AssemblyView` with a toolbar of chips over the viewport's top-left: «Вписать» (`F`), «Подписи» (`L` — every visible fragment's name as an HTML label at its centroid, re-placed on each rendered frame, hidden when more than 60 would show… then only the selected group's), «Цвет: скан / фрагменты / группы» (`C` cycles), «Все группы / Одна группа», «Без пары» (toggle), «Разъединить» (a slider 0…1 in a popover), «Снимок PNG» (downloads `screenshot()`; in Tauri through a save dialog is milestone 6 — a plain download link now). While a run job is on and there is no earlier run: the quiet line «Здесь появится сборка» behind the overlay; with an earlier run selected it stays on screen under the overlay, marked «предыдущая сборка».

**Right** (260 px): for a selected **fragment** — name; «группа 3 · 4 фр.» or «без пары»; its stats (as in «Вход», shorter); **Стыки** — its rows of `candidates` grouped by band: «Подтверждённые» (dot `ok`), «Вероятные» (dot `warn`), «Отклонённые» (collapsed, `text-muted`), each row the partner's name (click selects it and flies to it), the score to one decimal, and `used` as a small «в сборке»; nothing to accept or reject yet — that is milestone 5, and the rows must not pretend otherwise. For a selected **group** (a click on its tree row): members, joins used, «Показать одну». Nothing selected: the run's summary — counts, backend, duration, the stage timings as a small table.

The «Сборка» tab is enabled when a `done` run is selected and says «Сборка · 17 групп»; `2` switches to it; after a finished run job the window switches to it by itself. «Ревью» stays disabled.

- [ ] **Step 1: Write `tree.test.ts` (failing), implement `tree.ts`, then the three panes.** Checks: `pnpm test` (37 tests), `typecheck`, `lint`, `build`.
- [ ] **Step 2: Look at it.** Steps: create → link → prepare → «Собрать…» → «Собрать» → wait for the mock run to end → `shot` of «Сборка» (groups spread) → click a fragment in the tree (`shot`: selected, inspector with joins) → «Цвет» to groups (`shot`) → «Одна группа» (`shot`) → the explode slider at 0.6 via `eval` on the store (`shot`) → both panes collapsed (`shot`) → «Подписи» (`shot`). Read all seven. The mock's twelve fragments are the same slab twelve times — judge layout, colours, labels, panes and states, not the pottery. Fix what is wrong.
- [ ] **Step 3: Commit** — `git commit -m "M4.9: the Сборка mode -- the groups as a tree, the assembly in the middle, a fragment's joins by band on the right"`

---

### Task 10: the milestone gate, once

- [ ] **Step 1: Rust** — `cargo fmt --all --check && (cd apps/desktop/src-tauri && cargo fmt --check)`; `cargo clippy -p sherd-app-core --all-targets --locked -- -D warnings`; the same with `--features ts`; `(cd apps/desktop/src-tauri && cargo clippy --all-targets --locked -- -D warnings && cargo test 2>&1 | tail -3)`; `cargo test -p sherd-app-core 2>&1 | grep 'test result'`.
- [ ] **Step 2: Bindings in step** — regenerate (`TS_RS_EXPORT_DIR=… cargo test -p sherd-app-core --features ts --lib`), `git status --short apps/desktop/src/ipc/bindings` prints nothing.
- [ ] **Step 3: Frontend** — `cd apps/desktop && pnpm install --frozen-lockfile && pnpm lint && pnpm typecheck && pnpm test 2>&1 | tail -4 && pnpm build 2>&1 | tail -3`.
- [ ] **Step 4: One walk through everything, looked at.** One `tools/look.mjs` script from the welcome screen to the explode slider, in `ru`/light and again in `en`/dark, a PNG at each screen; read them all as a user would — clipped text, overlapping controls, an empty pane, English in the Russian UI, a missing translation key shown raw. Fix, re-run.
- [ ] **Step 5: README** — in «Настольное приложение (в разработке)»: what works now (подготовка входа, сборка с прогрессом и оценкой времени, история прогонов, режим «Сборка»), and `node tools/look.mjs` in one sentence.
- [ ] **Step 6: Commit** — `git commit -m "M4.10: the milestone's gate -- fmt and clippy, bindings in step, the frontend's checks, and the whole walk looked at in both languages and both themes"`

---

## What is left for a person

The spec's first manual acceptance (A §11) is due after this milestone, and only a person at the machine can do it: `cd apps/desktop && pnpm tauri dev`, a workspace on `input/karas_reduced`, a run on the GPU — the real window, the real asset protocol, a real 17-minute run with its time left, the dock's progress, the notification, a cancel half-way, a second run in the history. What is wrong there goes into milestone 5's first task.

## What the next plan can rely on

| need of milestone 5 (review) | provided by |
|---|---|
| candidates by band, with poses, for the queue | `run_candidates` → `CandidateRow[]` in `state/assembly.ts` |
| a draft assembly shown at once | `AssemblyViewer.setAssembly` — matrices and groups only |
| a ghost and a pair view | the per-fragment objects and materials of `AssemblyViewer`; `rowsToMatrix4` |
| the «draft» state | `deriveStatus`' `hasUnrefinedGroups`, fed from `assembly.groups[].refined` |
| the job thread for a long-lived review session | `jobs::start`'s slot, guard and event forwarding |
