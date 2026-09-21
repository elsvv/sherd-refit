# Desktop App, Milestone 5 — join review — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The screen the app exists for (D §12, A §8): go through the joins the engine was not sure about — each shown as a pair in its pose with the seam coloured by distance, the scores against their limits, and *why the engine did not confirm it* in a sentence — accept or reject with a key, see the assembly change at once, accept all the rest in one action, undo anything, refine the poses, and carry the decisions into the next run.

**Architecture:** A third job kind, `Review`, is a long-lived worker session: it loads the fragment cache and the run's `match.state` once and then answers `Reassemble { decisions }` in under a second through `session::reassemble`, `PairDetail` through a new public `review::seam_view` of the core, and `Refine` through `session::refine_poses` over the unrefined groups only. "Refined" is a property of a group (A §8.4): a reassembled group with the same members and joins as a refined one keeps its poses — a pure merge, tested. The window owns the list of decisions with undo and redo and sends it whole; the shell persists `decisions.json` atomically, forwards it, and files the `assembly.json` that comes back. A full run takes the previous run's decisions as constraints.

**Tech Stack:** as milestone 4. Spec: `docs/superpowers/specs/2026-09-20-desktop-app-design.md` (`A §n`). Screen: `docs/superpowers/specs/desktop-app-mockups/review-mode.html` — **open it; it is the «Ревью» mode.**

## Global Constraints

- **Toolchain.** Every shell that runs cargo starts with `export PATH="$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin:$PATH"`. JavaScript: pnpm 9 from `apps/desktop/`; exact versions.
- **Verification budget (A §11).** Run only what a task lists. No Playwright/WebDriver/component tests; Vitest on pure logic. No `cargo test --workspace`, no clippy inside a task (Task 10 is the gate). Only `fixtures/slab` is ever run. `cargo test -p sherd-cli --test run_cli` (4–7 min, in the background) runs **once**, in Task 10, because Task 2 adds a function to `sherd-core`.
- **The CLI's output stays byte for byte.** Task 2 is the only task that touches `crates/sherd-core`, and it only adds a public function.
- **Looking is part of building.** Every task that changes a screen ends with `node tools/look.mjs '<steps>' look/<task>` from `apps/desktop/` and with the agent **reading the PNGs**. Start every script with `{"goto":"/"},{"storage":{"sherd.language":"ru","sherd.theme":"light"}}` (headless Chrome's language is English). Click targets match by substring of the button's text or `aria-label`; when a click reports `NOT FOUND` the list it prints is what is on the page. Never commit `look/`.
- **The slab under the shipped rule** confirms nothing: with tiers on, its one join is *probable* and no group is assembled — which is exactly a review's starting point, and what the end-to-end test uses.
- **No panics on data from outside**; TypeScript strict, no `any`, no non-null assertions; every string through i18next, `ru.json` and `en.json` with identical key sets; render on demand; dispose what you create.
- **Ownership (A §2.1):** the worker writes engine artefacts only; `decisions.json`, `assembly.json`, `run.json` are the host's, written atomically.
- **Commits:** branch `desktop-app`, subject `M5.<task>: <sentence>`, trailer as your own system instructions give it.
- **Do not touch** `crates/sherd-parity`, `sherd_refit/`, `fixtures/`.

## Where things are (tree at `8045a1a`)

```
crates/sherd-core/src/        session.rs (MatchState, load_fragments, reassemble -> Reassembled, refine_poses, write_reviewed)
                              review.rs (private pair_evidence, contact_map, seam data)   assembly/ (recenter, Piece, constraints)
crates/sherd-app-core/src/    protocol.rs  worker/{mod.rs, prepare.rs, run.rs}  host.rs  decisions.rs  run.rs  view.rs  eta.rs
apps/desktop/src-tauri/src/   jobs.rs (JobStart, start, SlotGuard, finish)  commands.rs  state.rs
apps/desktop/src/             ipc/{api.ts, tauri.ts, mock.ts}  state/{workspace, jobs, assembly, ui, status, eta}.ts
                              viewer/{AssemblyViewer.ts, stage.ts?, matrix.ts, layout.ts, colours.ts}  modes/{input,assembly}/  app/
```

Real signatures: `session::reassemble(engine: Engine<'_>, fragments: &[Fragment], state: &MatchState, constraints: Option<&Constraints>) -> Result<Reassembled>` with `Reassembled { assembly: Assembly { poses, groups, used, rejected, .. }, candidates, tiers, constraints: Option<constraints::Report>, objects, poses /* recentred, unrefined */, used: Vec<(FragId, FragId)> }`; `session::refine_poses(engine, fragments, groups: &[Vec<FragId>], poses, used, params, budget, watch) -> Result<Vec<Matrix4<f64>>>` (not recentred); `sherd_core::assembly::{recenter(poses, pieces: &[Piece], groups), Piece { thick, res, watertight, mesh: Option<&RayScene>, s_pen: &Vec<[f64;3]> }}`; `decisions::{DecisionsFile, Decision { a, b, verdict, pose, source, bulk, at, carried_from }, to_constraints(&DecisionsFile, names) -> Result<(Option<Constraints>, Vec<Decision>)>}`; `CandidateRow { a, b, pose, score, scores, tier, used, evidence }`.

The engine's own words for why a join is not confirmed are `Evidence::failed: Vec<String>`, in these shapes (from `tiers.rs` and a real run): `"tight 0.2767 < 0.35"`, `"gap 0.0358 > 0.015"`, `"seam 4.0000 < 5"`, `"cont_n 0.8485 < 0.9"`, `"pen 0.0008 > 0"`, `"redraw 1: pen 0.0006 > 0"`, `"pen: penetration not measurable, a fragment is not watertight"`, `"slide: no shared seam to slide along"`, `"redraws accepted 1 < 2"`, `"determined 1e-3 deg > 1e-6 deg"`, `"no arm: support 0 < 1 and margin 1.00 < 2"`, `"no arm: support 0 < 1 and re-search agreed 1/2 < 2"`, `"no arm: support 0 < 1 and the second placement is 3.69 t away, under 5"`.

---

### Task 1: the seam between the window and the shell, and what milestone 4 left open

Two bugs so far lived exactly where the mock ends and the shell begins (a command invoked without an argument the shell had made required). That seam gets a test; the rest are the reviews' minor findings.

**Files:** `apps/desktop/src/ipc/contract.test.ts` (new), `apps/desktop/src/ipc/{tauri.ts, mock.ts}`, `src/app/LaunchSheet.tsx`, `src/viewer/AssemblyViewer.ts`, `apps/desktop/src-tauri/src/{commands.rs, jobs.rs}`, `crates/sherd-app-core/src/eta.rs`

- [ ] **Step 1: `contract.test.ts`** (Vitest, node environment, reads files with `node:fs`): from `src-tauri/src/*.rs` collect every `#[tauri::command]` function — its name and the names of its parameters other than `app`, `state`, `window` (whatever their types: `AppHandle`, `State<…>`, `Window`); from `src/ipc/tauri.ts` collect every `invoke("name", { … })` call — the command name and the object's keys. Assert: the two sets of command names are equal; for each command the window's keys, converted camelCase → snake_case, equal the Rust parameter names. The test must fail on today's tree if an argument is removed from either side — prove it once by hand and say so in the report. Parsing by regular expression over these two small, regular files is enough; say in a comment what shape of code it relies on.
- [ ] **Step 2: The shell** — `engine_info` becomes `#[tauri::command(async)]` (it spawns and drives a process; a sync command runs on the main thread and freezes the window); the notification is sent **after** `finish` (it must not sit between the job's end and the window being told); `run_delete` releases the job-slot lock before `trash::delete`; `engine_info` returns `EngineInfoView { adapters: Vec<String>, gpu: bool }` — the adapter names parsed from `info_lines()`' indented `[n] …` lines, without the engine's developer prose.
- [ ] **Step 3: `eta.rs`** — a stage that a run did **not** report decays (its ratio halves) instead of being kept for ever, so «Тщательно»'s `tier_agree` does not inflate every later standard estimate; the averaging test uses three runs so that an arithmetic mean and the EMA at ½ give different answers.
- [ ] **Step 4: The window** — the sheet's «Вычисления» hint shows «Видеокарта: Metal Apple M2 Pro» per adapter (or «Видеокарта не найдена — сборка пойдёт на CPU»), nothing else; the no-calibration estimate is one sentence; `AssemblyViewer` disposes its shared palette materials exactly once (in `dispose`, not per mesh in `release`); the mock's fresh workspace has **no** runs (the seeded history belongs to the recent workspace «karas» only), `STANDARD_SPEC.agree_seeds` is the engine's default, and it has one `interrupted` run and a `calibration()` with `runs: 0` reachable through `?nocal` in the URL.
- [ ] **Step 5: Verify** — `cargo test -p sherd-app-core --lib eta`, `cd apps/desktop/src-tauri && cargo check && cargo test`, `cd apps/desktop && pnpm test && pnpm typecheck && pnpm lint`; one `look.mjs` pass over the sheet (both calibration states).
- [ ] **Step 6: Commit** — `M5.1: a test on the seam where two bugs hid -- the window's invoke calls against the shell's commands -- and what milestone 4's reviews left open`

---

### Task 2: `review::seam_view` — a seam as data

**Files:** `crates/sherd-core/src/review.rs`, `crates/sherd-core/tests/session.rs`

**Interfaces:**
```rust
/// What a review image draws of one placement, as numbers (A §2.2's `PairDetail`).
#[derive(Clone, Debug, PartialEq)]
pub struct SeamView {
    /// B's fracture samples at the pose, in A's frame (R §3.5.2).
    pub contact: Vec<[f64; 3]>,
    /// One class per contact point: 0 under the tight limit, 1 under the gap limit, 2 beyond (R §6.1, R §6.5).
    pub contact_class: Vec<u8>,
    /// Centres of R §6.2's seam voxels, in A's frame.
    pub seam: Vec<[f64; 3]>,
    /// The pair's tight and gap limits, in scan units.
    pub tight: f64,
    pub gap: f64,
}
pub fn seam_view(engine: Engine<'_>, a: &Fragment, b: &Fragment, transform: &Matrix4<f64>, params: &Params) -> SeamView
```
Built from the private `contact_map` and the seam-cell code `pair_evidence` already uses — **share them, do not copy them**; a pair with no surfaces gives an empty view. `write_review_images` must produce the bytes it produced (it has a determinism test in `run_cli.rs`, run in Task 10).

- [ ] **Step 1: Failing test** in `tests/session.rs`: on the `accepted_run()` slab, `seam_view` at the best accepted candidate's pose has a non-empty `contact` of the same length as `contact_class`, more than half of the classes are `0`, `seam` is non-empty, `0 < tight < gap`; at the identity pose (the two pieces apart) no class is `0`.
- [ ] **Step 2: Implement; `cargo test -p sherd-core --test session 2>&1 | tail -4`** → 8 passed.
- [ ] **Step 3: Commit** — `M5.2: a seam as data -- what a review image draws of a placement, for a window that draws it live`

---

### Task 3: the review session in the worker

**Files:** `crates/sherd-app-core/src/protocol.rs`, `src/worker/{mod.rs, review.rs (new)}`, `src/review.rs (new: the pure merge)`, `src/lib.rs`

**Interfaces:**
- `Job::Review(ReviewJob { workspace, input, run_id, excluded, target_faces, seed, memory_gb, workers })`
- `Request::{Cancel, Reassemble { decisions: DecisionsFile }, PairDetail { a: String, b: String, pose: [[f64; 4]; 4] }, Refine { decisions: DecisionsFile }, Close }`
- `Event::{Ready { fragments: usize, candidates: usize }, PairDetail(PairDetailDto), Dropped { decisions: Vec<Decision> }}` added; `Assembly(AssemblyDto)` is the answer to `Reassemble` and `Refine`.
- `PairDetailDto { a, b, contact: Vec<[f32; 3]>, contact_class: Vec<u8>, seam: Vec<[f32; 3]>, tight: f64, gap: f64 }` (`f32`: it is for drawing)
- `review::merge_refined(baseline: &AssemblyDto, fresh: AssemblyDto) -> AssemblyDto` — pure: a group of `fresh` whose member set **and** set of joins among those members equal a `refined` group's of `baseline` takes that group's poses and `refined: true`; every other group keeps `fresh`'s poses and `refined: false` (singletons are `refined: false` and irrelevant). `unplaced` is `fresh`'s.
- The worker's input reader forwards every request that is not `Cancel` into an `mpsc` channel on the job's `Context` (`requests: Receiver<Request>`); `Prepare` and `Run` ignore it.

The session: `Hello`; resolve the engine as **CPU** (`Engine::REFERENCE` — a draft must not depend on a device, and `write_reviewed` records the match's own backend); `discover_excluding` → `load_fragments` from `cache/` (progress `preprocess`); `MatchState::load(runs/<id>/match.state)` — an `Error::State` is `Failed { kind: Protocol, … }` with the engine's sentence (A §10: the run stays viewable, review is off); read `runs/<id>/assembly.json` as the baseline; emit `Ready`. Then, per request:
- `Reassemble` → `decisions::to_constraints` (dropped ones → `Event::Dropped`) → `session::reassemble` → `AssemblyDto` from the result (names, recentred poses, `used` as joins, `unplaced` = for each **accepted** decision whose pair is not among `used`: the engine's own reason — the matching entry of `assembly.rejected` via its `reason.message(names)`, else the constraints report's outcome for that pair, else «не поставлен») → `merge_refined(&baseline, fresh)` → `Event::Assembly`.
- `PairDetail` → `review::seam_view` → `Event::PairDetail`.
- `Refine` → reassemble as above, then `session::refine_poses` over the groups that came out `refined: false` with two or more members, from `Reassembled.assembly.poses` (un-recentred), then `assembly::recenter` with pieces built from the fragments' surface samples (`mesh: None`), merged so that groups that were already refined keep their poses; every group of two or more `refined: true`; the result becomes the new baseline; `Event::Assembly`. Progress `refine`.
- `Close` or the end of input → `Done`.
A request that fails (`constraints::resolve` refusing a pose) is an `Event::Failed`-free, per-request error: add `Event::RequestFailed { message }` and keep serving.

- [ ] **Step 1: Failing tests** — `review.rs` (pure): same members and joins → baseline poses, `refined`; a group that gained a member → fresh poses, not refined; same members but a different join set → not refined; `unplaced` passes through. Protocol: the new requests and events are one tagged line each.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: End to end**, appended to `tests/worker_e2e.rs`: prepare and run the slab (tiers on: nothing placed); spawn `Job::Review`; wait for `Ready`; send `Reassemble` with one `accept` decision whose pose is the probable row of `candidates.json` → an `Assembly` with one group of two, `refined: false`, no `unplaced`; send `PairDetail` for that pair and pose → non-empty contact; send `Refine` → `Assembly` with the group `refined: true`; send `Reassemble` with the same decision again → still `refined: true` and the **refined** poses (the merge); send a `reject` of the same pair instead → no group; `Close` → `Done`, exit code 0. This needs `host::Worker` to send requests: add `Worker::requester() -> Requester` (`Clone + Send + Sync`, `send(&Request) -> Result<()>`), as `canceller()` is built.
- [ ] **Step 4: Verify** — `cargo test -p sherd-app-core 2>&1 | grep 'test result'`.
- [ ] **Step 5: Commit** — `M5.3: a review session -- the match loaded once, a decision answered with an assembly in under a second, and refined groups that stay refined`

---

### Task 4: decisions carried into the next run

**Files:** `crates/sherd-app-core/src/host.rs`, `src/decisions.rs`

**Interfaces:** `host::carry(ws: &Workspace, from_run: &str, names: &[String], now: &str) -> Result<Carried>` with `Carried { constraints: Option<Constraints>, decisions: DecisionsFile /* each marked carried_from */, dropped: Vec<Decision> }`; `host::start_run` writes `carried.decisions` as the new run's `decisions.json` when given one (add a parameter `carried: Option<Carried>` replacing the separate `constraints`/`carried_from` pair, and update its callers).

- [ ] **Step 1: Failing test** (unit, no worker): a run folder with two decisions, one naming a fragment that is gone → `constraints` has one entry, `dropped` one, the new file's decisions carry `carried_from`.
- [ ] **Step 2: Implement; `cargo test -p sherd-app-core 2>&1 | grep 'test result'`.**
- [ ] **Step 3: Commit** — `M5.4: a run starts from the last review -- accepted pairs pinned and not matched again, rejected pairs skipped, orphans dropped and said`

---

### Task 5: the review session in the shell

**Files:** `apps/desktop/src-tauri/src/{jobs.rs, commands.rs, state.rs}`; regenerate bindings

**Interfaces — commands:**

| command | arguments | what it does |
|---|---|---|
| `review_open` | `run_id` | spawns `Job::Review`; the slot holds `JobKind::Review` with a `Requester`; events forwarded as `engine:event`; `busy` if a prepare or a run is on; opening the run that is already open is a no-op |
| `review_apply` | `decisions: DecisionsFile` | validates, writes `runs/<id>/decisions.json` atomically, sends `Reassemble` |
| `review_pair` | `a, b, pose` | sends `PairDetail` |
| `review_refine` | — | sends `Refine` with the file's decisions |
| `review_close` | — | sends `Close` |
| `run_decisions` | `run_id` | `DecisionsFile` of a run (empty when none) |
| `run_start` | `spec, lang, carry_from: Option<String>` | as before, with `host::carry` when `carry_from` is `Some` |

On `Event::Assembly` during a review the job thread files `assembly.json` (as a run's does). `prepare_start` and `run_start` while a review session is open **close it first** and wait for its thread to free the slot (bounded: 10 s, then `kill`): a warm session is a cache, never a reason to refuse work. `JobKind` gains `Review`; `view::JobView` carries it; bindings regenerated.

- [ ] **Step 1: Implement; `cargo check`, `cargo test` in the shell.**
- [ ] **Step 2: Bindings** — regenerate; add TS mirrors `ScoresTs` and `EvidenceTs` in `protocol.rs` for `CandidateRow::{scores, evidence}` (as `FragmentStatsTs` was done, each with its key-sync test against the engine's type), so the window stops seeing `Record<string, unknown>`.
- [ ] **Step 3: `contract.test.ts` still passes** once Task 6 adds the window's side — note for Task 6 the exact command names and argument names.
- [ ] **Step 4: Commit** — `M5.5: the review session in the shell -- decisions persisted before they are sent, the assembly filed when it comes back, and a session that never stands in a run's way`

---

### Task 6: the window's side — API, mock, the review store, the reasons

**Files:** `apps/desktop/src/ipc/{api.ts, tauri.ts, mock.ts}`, `src/state/{review.ts, review.test.ts, reasons.ts, reasons.test.ts}`, `src/state/status.ts`

**Interfaces:**
- `Api`: `reviewOpen(runId)`, `reviewApply(decisions)`, `reviewPair(a, b, pose)`, `reviewRefine()`, `reviewClose()`, `runDecisions(runId)`, `runStart(spec, lang, carryFrom)`.
- `state/review.ts` (Zustand): `{ runId, ready, decisions: Decision[], past: Decision[][], future: Decision[][], pending: boolean, dropped: Decision[], detail: PairDetailDto | null, selected: { a, b, pose } | null }` with `accept(row)`, `reject(row)`, `clear(a, b)`, `acceptAllProbable(rows)` (one undo step; each decision `bulk: true`), `undo()`, `redo()`, `reset()`; every change calls `reviewApply` with the whole list and sets `pending` until the `assembly` event arrives; the pure parts — `applyDecision(list, decision)`, `bulkAccept(list, rows, at)`, the undo stack — are exported functions tested without the store.
- `state/reasons.ts` (pure): `explain(failed: readonly string[], t): string[]` — the engine's `Evidence::failed` strings (shapes listed in this plan's header) into sentences: `tight 0.2767 < 0.35` → «плотное прилегание 0.28 при нужных 0.35», `gap … > …` → «зазор 0.036 t при допустимых 0.015 t», `seam … < …` → «шов 4.0 t короче нужных 5 t», `pen … > 0` → «поверхности проникают друг в друга», `pen: penetration not measurable…` → «проникание нельзя проверить: меш одного из фрагментов не замкнут», `redraw N: …` → «на повторной выборке точек: …», `redraws accepted 1 < 2` → «устоял на 1 из 2 повторных выборок», `no arm: support 0 < 1 and margin 1.00 < 2` → «ни один другой стык не ставит фрагмент туда же, а вторая поза этой пары почти так же хороша (отрыв 1.0× при нужных 2×)», `…re-search agreed 1/2 < 2` → «…, а повторный поиск пришёл в ту же позу 1 раз из 2», `…the second placement is 3.69 t away, under 5` → «…, а вторая поза стоит слишком близко (3.7 t при нужных 5 t) — это может быть тот же шов со сдвигом»; anything unrecognised is shown as it is. `headline(row): string` — one sentence for the inspector's «Почему не подтверждён сам».
- `deriveStatus`'s `hasUnrefinedGroups` comes from the assembly store: any group of two or more with `refined: false`.
- The mock plays a session: `reviewOpen` → `ready`; `reviewApply` → after 150 ms an `assembly` event in which each accepted probable row's two fragments are joined into one group (`refined: false`) and each rejected used row splits its group; `reviewPair` → a synthetic `PairDetailDto` (a few hundred points along a wavy line between the two pieces, mostly class 0, some 1 and 2); `reviewRefine` → 1.5 s, then all `refined`.

- [ ] **Step 1: Failing tests** — `review.test.ts` (a second decision on a pair replaces the first whichever way round it is named; `bulkAccept` skips decided pairs and is one undo step; undo/redo round-trip; `reset`), `reasons.test.ts` (every shape of the header, in `ru`; an unknown string passes through).
- [ ] **Step 2: Implement; `pnpm test && pnpm typecheck && pnpm lint`** (the contract test passes with the new commands).
- [ ] **Step 3: Commit** — `M5.6: the window's side of a review -- the decisions with their undo, the engine's reasons in sentences, and a mock that plays a session`

---

### Task 7: a pair in 3D

**Files:** `apps/desktop/src/viewer/AssemblyViewer.ts` (+ a helper file if it grows), `src/viewer/pair.test.ts` for any pure part

**Interfaces** (added to `AssemblyViewer`):
```ts
showPair(pair: { a: string; b: string; pose: number[][] } | null): void   // A grey at the identity, B orange at `pose`; everything else hidden; framed face-on to the seam
setPairDetail(detail: PairDetailDto | null): void                          // contact points green / yellow / red, seam voxels white, as THREE.Points in A's frame
setPairSeparation(amount: number): void                                     // 0…1: B pushed away from A along the line between their centroids
setGhost(ghost: { name: string; anchor: string; pose: number[][]; flip: boolean } | null): void
```
`pose` maps **b into a's frame** (R §0); in the assembly view a ghost of `name` is drawn translucent at `world(anchor) · pose` (or `world(anchor) · pose⁻¹` when `flip` — the candidate names the fragments the other way round). The two pair colours are the review images' own: A `#b0b0b0`, B `#e8872b`. Points are `sizeAttenuation: false`, 3 px, depth-tested. Leaving pair mode restores layout, colours and visibility exactly.

- [ ] **Step 1: Implement** (pure helpers — the ghost's matrix, the separation vector — tested in `pair.test.ts` with a known pose).
- [ ] **Step 2: `pnpm test && pnpm typecheck && pnpm lint && pnpm build`.**
- [ ] **Step 3: Commit** — `M5.7: a pair in 3D -- grey and orange as the engine's review images have them, the seam coloured by distance, a ghost where a candidate would put its partner`

---

### Task 8: the «Ревью» mode

**Files:** `apps/desktop/src/modes/review/{ReviewLeft.tsx, ReviewCentre.tsx, ReviewRight.tsx, DraftLine.tsx, queue.ts, queue.test.ts}`, `src/app/{Frame.tsx, TopBar.tsx, shortcuts.ts}`, `src/i18n/*.json`

Follow `docs/superpowers/specs/desktop-app-mockups/review-mode.html`.

**Left** (300 px): band filter chips «Вероятные N», «Подтверждённые N», «Отклонённые ядром»; the queue, best score first, virtualised; each row: a dot (warn / ok / muted), «FY234111 – FY234058», the score, and its verdict («✓ принят», «✕ отклонён», nothing); under the list «Принять все оставшиеся вероятные» with its one sentence of explanation (A §8.4). `queue.ts` (pure, tested): rows for a band from `CandidateRow[]` — one row per **pair** (its best-scoring candidate of that band; the pair's other poses are the inspector's «Другие позы»), sorted, with the verdict joined in; `nextUndecided(rows, from)`.
**Centre:** the pair (`showPair` + `setPairDetail`), chips «Пара / В сборке», «Разъединить» (slider), «Шов: расстояния» (toggle); the legend (плотно · в пределах зазора · дальше); bottom right the three actions with their keys: «Пропустить ␣», «Отклонить X», «Подтвердить A» — after a verdict the selection moves to the next undecided row. «В сборке» shows the assembly with the pair emphasised and, when B is not placed, as a ghost.
**Right** (260 px): the pair's names and where each stands («группа 0», «без пары»); the scores against their limits as a three-column table (оценка, шов, прилегание, зазор, проникание — value, limit from the run's `params`, green or amber); **«Почему не подтверждён сам»** — `headline` and the list from `explain`; «Другие позы пары · N» (each: score, «показать» → the viewer shows that pose; accepting it replaces the pair's decision); for a confirmed join the block says what confirmed it (`evidence.arm`: «опора» / «отрыв»).
**Draft line** (above the status line, only with decisions or unrefined groups): «Черновик: 2 принято · 1 отклонено» · «стало 18 групп (было 17) · без пары 78 (было 81)» · «Отменить ⌘Z» · «Вернуть» · «Сбросить решения» (confirm) · «Уточнить позы · ≈30 с» (enabled when a group is unrefined; progress in the status line) · `unplaced` as a warn chip «2 не встали» opening a list with the engine's reason for each · dropped decisions once as a dismissible notice.
The «Ревью · 36 из 39» tab is enabled for a `done` run whose `match.state` could be opened; entering the mode calls `reviewOpen` (the status line says «Загрузка сопоставления…» until `ready`); a session that fails to open (`protocol`) shows a banner «Ревью для этого прогона недоступно: … Перегенерируйте сборку.» and leaves the mode. `3` switches to it; `A`/`X`/`Space`/`⌘Z`/`⇧⌘Z` work while no input has focus.

- [ ] **Step 1: `queue.test.ts` failing, `queue.ts`, then the panes.** Checks as before.
- [ ] **Step 2: Look at it** — open «karas» from the recent list (it has the seeded runs) → «Ревью» (`shot`) → select a row (`shot`: pair, seam, reasons) → `A` (`shot`: verdict, next row, draft line) → `X` on another → ⌘Z → «Принять все оставшиеся» (`shot`) → «Уточнить позы» (`shot` during, `shot` after) → «В сборке» (`shot`). Read them all; also `en`/dark once. Fix what is wrong.
- [ ] **Step 3: Commit** — `M5.8: the Ревью mode -- a queue, a pair with its seam, why the engine held back, a key to decide, and the assembly answering at once`

---

### Task 9: review from the «Сборка» mode, and carry-over in the sheet

**Files:** `apps/desktop/src/modes/assembly/AssemblyRight.tsx`, `src/app/LaunchSheet.tsx`, `src/i18n/*.json`

- The fragment inspector's candidate rows gain «Подтвердить» / «Отклонить» (and «Снять решение» when decided), wired to the review store (opening the session on first use); hovering a row whose partner is not placed with it shows the **ghost** (`setGhost`), leaving it clears it. Rejected-by-the-engine rows can be accepted too (A §7.3) — behind the collapsed section, as before.
- The launch sheet gains «Перенести решения ревью (N)» — on by default when the selected run has decisions — passing `carryFrom`; its one sentence: «Принятые стыки не сопоставляются заново, отклонённые пропускаются — сборка пойдёт быстрее.»
- [ ] **Step 1: Implement; checks; look** (a fragment without a pair selected, a probable row hovered — the ghost — `shot`; accepted — `shot`; the sheet with the checkbox — `shot`).
- [ ] **Step 2: Commit** — `M5.9: a decision made where the fragment is looked at -- with a ghost of where it would go -- and the last review carried into the next run`

---

### Task 10: the milestone gate, once

- [ ] **Step 1: Rust** — `cargo fmt --all --check` and the shell's; `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings`; `cargo clippy -p sherd-app-core --all-targets --features ts --locked -- -D warnings`; the shell's clippy and tests; `cargo test -p sherd-app-core`; `cargo test -p sherd-core --test session`.
- [ ] **Step 2: The CLI's byte-for-byte gate** (Task 2 touched `review.rs`), in the background: `cargo test -p sherd-cli --test run_cli 2>&1 | tail -5` → `12 passed; 2 ignored`.
- [ ] **Step 3: Bindings in step; frontend** — regenerate and `git status --short apps/desktop/src/ipc/bindings` is empty; `pnpm install --frozen-lockfile && pnpm lint && pnpm typecheck && pnpm test && pnpm build`.
- [ ] **Step 4: The whole walk, looked at** — from the welcome screen through a review to «Уточнить позы», `ru`/light and `en`/dark; read every PNG as a user would.
- [ ] **Step 5: README** — the app's section: what «Ревью» does, in a paragraph a museum colleague can read.
- [ ] **Step 6: Commit** — `M5.10: the milestone's gate -- the core's byte-for-byte tests, clippy in every shape, bindings in step, and a review walked through and looked at`

---

## What is left for a person

A §11's second manual acceptance: in `pnpm tauri dev`, on the `karas` run — open «Ревью», decide a dozen probable joins with the keys, watch the groups change, «Принять все оставшиеся», «Уточнить позы», then a new run with the decisions carried over. What is wrong there goes into milestone 6's first task.

## What the next plan can rely on

| need of milestone 6 (export, Blender, settings, release build) | provided by |
|---|---|
| the reviewed assembly written by the engine's writers | the review session (`session::write_reviewed` behind a new `Export` request), decisions as constraints |
| groups, names and matrices for the Blender script | `assembly.json` (`AssemblyDto`) and the run's input snapshot |
| a long-lived worker with requests | `Job::Review`, `Requester`, the shell's session slot |
