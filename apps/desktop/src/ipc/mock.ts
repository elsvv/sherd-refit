import type { AssemblyDto } from "./bindings/AssemblyDto";
import type { CandidateRow } from "./bindings/CandidateRow";
import type { FileStamp } from "./bindings/FileStamp";
import type { FragmentInfo } from "./bindings/FragmentInfo";
import type { JobKind } from "./bindings/JobKind";
import type { RunCounts } from "./bindings/RunCounts";
import type { RunFile } from "./bindings/RunFile";
import type { RunSpec } from "./bindings/RunSpec";
import type { RunStatus } from "./bindings/RunStatus";
import type { RunView } from "./bindings/RunView";
import type { StaleDiff } from "./bindings/StaleDiff";
import type { Warning } from "./bindings/Warning";
import type { WorkspaceView } from "./bindings/WorkspaceView";
import type { Api, CommandError, EngineEventPayload, EngineFinishedPayload, Unlisten } from "./api";

/**
 * The window without a shell under it: `pnpm dev` in a plain browser talks to this instead of
 * Tauri, so every screen of A §7 can be looked at — and a preparation, and a whole run, watched
 * arriving — without building the app or running the engine (A §11: the interface is checked by
 * hand, and this is what makes that cheap).
 *
 * It is a toy on purpose. It holds one workspace in memory, forgets it on reload, and its meshes
 * and thumbnails are three files of `fixtures/slab` served from `public/mock/`.
 */

/** Twelve fragments, the size of collection the mock-ups draw a grid of. */
const NAMES = Array.from({ length: 12 }, (_, i) => `FY2340${String(i + 1).padStart(2, "0")}`);

/** The two thick walls and the one open shell that give the warnings column something to show. */
const THICK = new Set(["FY234002", "FY234008"]);
const OPEN_SHELL = "FY234005";
/** The collection's median wall, as the mock-up's warning sentence quotes it. */
const MEDIAN_THICKNESS = 34.2;

/** Where `pickFolder` says the scans are. */
const INPUT_PATH = "/mock/scans/karas_reduced";
/** Where the mock's own workspace lives, for the recent list. */
const WORKSPACE_PATH = "/mock/workspaces/karas";

/** How long a mocked `Prepare` takes, in milliseconds — long enough to watch, short enough to sit through. */
const PREPARE_MS = 6000;

/** One fragment, from a single template varied just enough to be told apart. */
function fragment(name: string, index: number): FragmentInfo {
  const thickness = THICK.has(name) ? 78.2 - index : 30.2 + (index % 5) * 0.9;
  const faces = 45100 - index * 1200;
  const warnings: Warning[] = [];
  if (THICK.has(name)) {
    warnings.push({ warning: "thickness_outlier", thickness, median: MEDIAN_THICKNESS });
  }
  if (name === OPEN_SHELL) {
    warnings.push({ warning: "not_watertight" });
  }
  return {
    name,
    file: `${name}_reduced.ply`,
    size: 1127755 - index * 9001,
    mtime_ms: 1789129634585 + index * 1000,
    stats: {
      name,
      faces,
      orig_faces: faces,
      orig_vertices: Math.round(faces / 2),
      thickness,
      thickness_mode: thickness - 0.4,
      resolution: 2.12 + (index % 3) * 0.08,
      watertight: name !== OPEN_SHELL,
      extent: [149.3 + index, 257.5 - index, 221.4 + index * 0.5],
      area: 86535.7 - index * 900,
      fracture_area_fraction: 0.25 + (index % 4) * 0.02,
    },
    warnings,
    coloured: false,
    display_faces: faces,
  };
}

/** The file the fragment was made from, as the input snapshot sees it. */
function stamp(info: FragmentInfo): FileStamp {
  return { name: info.name, file: info.file, size: info.size, mtime_ms: info.mtime_ms };
}

const ALL: FragmentInfo[] = NAMES.map((name, index) => fragment(name, index));

/**
 * What the mock's runs assemble: three groups and one fragment that found no partner, which is
 * the shape A §8's screens are drawn for. Singletons are groups too — that is what the engine
 * writes, and what «Без пары» counts.
 */
const GROUPS: readonly (readonly string[])[] = [
  ["FY234001", "FY234003", "FY234004", "FY234007", "FY234009"],
  ["FY234002", "FY234005", "FY234006", "FY234011"],
  ["FY234008", "FY234010"],
  ["FY234012"],
];

/** A pose as the protocol carries it: row-major, the order `transforms.json` writes. */
type Pose = AssemblyDto["poses"][string];

/** A rotation about Z by `degrees` and a translation in the XY plane, as a row-major 4×4. */
function rigid(degrees: number, x: number, y: number): Pose {
  const angle = (degrees * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    [cos, -sin, 0, x],
    [sin, cos, 0, y],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

/** How far apart the members of a group are fanned, in the units the slab is modelled in. */
const REACH = 180;
/** And by how much each is turned, so that twelve copies of one slab do not coincide. */
const TURN = 40;

/** Where the `index`-th member of a group sits in its group's frame. */
function fan(index: number): Pose {
  const angle = (index * TURN * Math.PI) / 180;
  return rigid(index * TURN, REACH * index * Math.cos(angle), REACH * index * Math.sin(angle));
}

/**
 * The pose that takes member `b`'s own frame into member `a`'s — what a candidate row carries
 * (`p_a = T · p_b`). Derived from [`fan`] rather than made up, so the mock's joins agree with the
 * mock's assembly.
 */
function between(a: number, b: number): Pose {
  const to = (index: number) => (index * TURN * Math.PI) / 180;
  const dx = REACH * b * Math.cos(to(b)) - REACH * a * Math.cos(to(a));
  const dy = REACH * b * Math.sin(to(b)) - REACH * a * Math.sin(to(a));
  const cos = Math.cos(to(a));
  const sin = Math.sin(to(a));
  return rigid((b - a) * TURN, dx * cos + dy * sin, -dx * sin + dy * cos);
}

/** What the mock's runs wrote as `assembly.json` (A §8.1). */
function assemblyDto(): AssemblyDto {
  const poses: Record<string, Pose> = {};
  const joins: { a: string; b: string }[] = [];
  for (const members of GROUPS) {
    members.forEach((name, i) => {
      poses[name] = fan(i);
      const previous = members[i - 1];
      if (previous !== undefined) {
        joins.push({ a: previous, b: name });
      }
    });
  }
  return {
    // R §9 refines a run's joins in one pass, so the groups of a run are all refined or none
    // are; a group of one has no join to refine.
    groups: GROUPS.map((members) => ({ members: [...members], refined: members.length > 1 })),
    poses,
    joins,
    // A run places what it can and argues with nobody: `unplaced` is `reassemble`'s, and the
    // engine leaves it empty here (A §8.4 is milestone 5's).
    unplaced: [],
  };
}

/** Every number R §6 produces for one candidate, under the keys the report writes (R §6.5). */
function scores(tight: number, gap: number, seam: number, pen: number): CandidateRow["scores"] {
  return {
    tightA: Math.min(1, tight + 0.03),
    tightB: tight,
    tight,
    gapA: Math.max(0, gap - 0.002),
    gapB: gap,
    gap,
    contactA: 44.1 - seam,
    contactB: 41.7 - seam,
    contact: 41.7 - seam,
    seam,
    gap_limit: 0.03,
    tight_delta: 0.01,
    cont: 0.09 + gap,
    cont_n: 0.97 - gap,
    pen,
    pen_depth: pen * 2.4,
    brk: 0.62 + tight / 4,
    brk_best: 0.78,
  };
}

/**
 * The rows of `candidates.json` (A §4): the joins the assembly was built from, four pairs that
 * were good enough to look at and were not used, and a few the engine threw out — which is the
 * three bands A §8.3's inspector groups its rows into.
 */
function candidateRows(): CandidateRow[] {
  const rows: CandidateRow[] = [];
  GROUPS.forEach((members) => {
    members.forEach((name, i) => {
      const previous = members[i - 1];
      if (previous === undefined) {
        return;
      }
      const tight = 0.42 + (i % 3) * 0.06;
      const seam = 7.4 + i * 0.8;
      rows.push({
        a: previous,
        b: name,
        pose: between(i - 1, i),
        score: seam * tight,
        scores: scores(tight, 0.012 + (i % 2) * 0.003, seam, 0.0011),
        tier: "confirmed",
        used: true,
        evidence: { margin: 6.2 + i, agree_seeds: 2, rival: null },
      });
    });
  });

  const probable: [string, string][] = [
    ["FY234001", "FY234006"],
    ["FY234004", "FY234011"],
    ["FY234008", "FY234012"],
    ["FY234003", "FY234010"],
  ];
  probable.forEach(([a, b], i) => {
    const tight = 0.29 + i * 0.02;
    const seam = 4.1 + i * 0.3;
    rows.push({
      a,
      b,
      pose: between(i, i + 2),
      score: seam * tight,
      scores: scores(tight, 0.021 + i * 0.002, seam, 0.0019),
      tier: "probable",
      used: false,
      evidence: { refusals: ["cont_n"], margin: 1.1 + i * 0.2 },
    });
  });

  const rejected: [string, string][] = [
    ["FY234002", "FY234009"],
    ["FY234005", "FY234012"],
    ["FY234007", "FY234010"],
    ["FY234001", "FY234008"],
    ["FY234006", "FY234012"],
  ];
  rejected.forEach(([a, b], i) => {
    const tight = 0.08 + i * 0.01;
    const seam = 1.2 + i * 0.2;
    rows.push({
      a,
      b,
      pose: between(i, i + 3),
      score: seam * tight,
      scores: scores(tight, 0.061 + i * 0.004, seam, 0.0082),
      tier: "rejected",
      used: false,
      evidence: null,
    });
  });
  return rows;
}

const ASSEMBLY: AssemblyDto = assemblyDto();
const CANDIDATES: CandidateRow[] = candidateRows();

/** What a finished mock run found — the numbers the history and the mode tabs show. */
const COUNTS: RunCounts = {
  fragments: ALL.length,
  pairs: 66,
  skipped_pairs: 4,
  candidates: 24,
  confirmed: CANDIDATES.filter((row) => row.tier === "confirmed").length,
  probable: CANDIDATES.filter((row) => row.tier === "probable").length,
  groups: GROUPS.filter((members) => members.length > 1).length,
  unassembled: GROUPS.filter((members) => members.length === 1).length,
  timings: [
    { stage: "preprocess", seconds: 41.2 },
    { stage: "matching", seconds: 913.7 },
    { stage: "tiers", seconds: 30.4 },
    { stage: "assembly", seconds: 8.1 },
    { stage: "refine", seconds: 13.9 },
    { stage: "output", seconds: 14.6 },
  ],
};

/** The sheet a run was started from, as `run.json` keeps it (A §7.4's «Стандарт»). */
const STANDARD_SPEC: RunSpec = {
  preset: "standard",
  backend: "auto",
  adapter: null,
  gpu_memory_gb: null,
  seed: 0,
  target_faces: 200000,
  tiers: true,
  agree_seeds: 1,
  thick_ratio: 2.5,
  min_tight: 0.25,
  max_gap: 0.03,
  max_pen: 0.005,
  min_seam: 3.0,
  workers: 0,
  memory_gb: null,
};

/** Two digits, the way every timestamp below needs them. */
function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** `now` as `run.json` writes a time: RFC 3339 with the machine's own offset. */
function timestamp(at: Date): string {
  const offset = -at.getTimezoneOffset();
  const sign = offset < 0 ? "-" : "+";
  const minutes = Math.abs(offset);
  const date = `${String(at.getFullYear())}-${pad(at.getMonth() + 1)}-${pad(at.getDate())}`;
  const time = `${pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())}`;
  return `${date}T${time}${sign}${pad(Math.floor(minutes / 60))}:${pad(minutes % 60)}`;
}

/** A run's id: local time to the minute, `-2`, `-3`, … when the minute is taken (A §4). */
function newRunId(existing: readonly string[], at: Date): string {
  const base = `${String(at.getFullYear())}-${pad(at.getMonth() + 1)}-${pad(at.getDate())}_${pad(at.getHours())}${pad(at.getMinutes())}`;
  if (!existing.includes(base)) {
    return base;
  }
  for (let n = 2; n <= existing.length + 1; n += 1) {
    const id = `${base}-${String(n)}`;
    if (!existing.includes(id)) {
      return id;
    }
  }
  return base;
}

/** One `run.json`, with the input snapshot every run of the mock was made from. */
function runFile(
  id: string,
  created: string,
  finished: string | null,
  status: RunStatus,
  counts: RunCounts | null,
): RunFile {
  return {
    version: 1,
    id,
    created,
    finished,
    status,
    spec: { ...STANDARD_SPEC },
    params: null,
    input: { files: ALL.map(stamp), excluded: [] },
    engine:
      counts === null
        ? null
        : {
            core_version: "0.1.0",
            algo_ref: "2026-09-06/9d4b9d3",
            commit: "mock",
            backend: "gpu:Apple M2 Pro",
          },
    counts,
    carried_from: null,
  };
}

/**
 * The history the mock starts with: one run that worked, one the GPU gave up on and one the user
 * stopped — so that every row of A §5's table, and every colour of the run selector, can be
 * looked at without waiting ten seconds for a mocked run first.
 */
function seededRuns(): RunFile[] {
  const failed: RunStatus = {
    state: "failed",
    kind: "gpu",
    message: "wgpu: device lost while scoring pair 4 812 of 11 097 (Metal command buffer aborted)",
  };
  return [
    runFile("2026-09-20_1412", "2026-09-20T14:12:00+03:00", "2026-09-20T14:29:02+03:00", { state: "done" }, COUNTS),
    runFile("2026-09-19_1806", "2026-09-19T18:06:00+03:00", "2026-09-19T18:11:37+03:00", failed, null),
    runFile("2026-09-18_0930", "2026-09-18T09:30:00+03:00", "2026-09-18T09:44:10+03:00", { state: "cancelled" }, null),
  ];
}

/**
 * How the input has moved since a run saw it (A §5's «устарел»). Computed here, as the shell
 * computes it, rather than written down once: excluding a fragment must make the runs stale in
 * the mock too, or the banner and the «Перегенерировать…» it carries can only be seen on a real
 * collection.
 */
function staleOf(run: RunFile, view: WorkspaceView): StaleDiff {
  const was = new Map(run.input.files.map((file) => [file.name, file]));
  const now = new Map(view.files.map((file) => [file.name, file]));
  const changed = [...now.entries()]
    .filter(([name, file]) => {
      const before = was.get(name);
      return before !== undefined && (before.size !== file.size || before.mtime_ms !== file.mtime_ms);
    })
    .map(([name]) => name);
  return {
    added: [...now.keys()].filter((name) => !was.has(name)),
    removed: [...was.keys()].filter((name) => !now.has(name)),
    changed,
    excluded_added: view.excluded.filter((name) => !run.input.excluded.includes(name)),
    excluded_removed: run.input.excluded.filter((name) => !view.excluded.includes(name)),
  };
}

/** The workspace as it is right after it is created: no input, nothing prepared, a history. */
function fresh(path: string): WorkspaceView {
  const name = path.split("/").filter(Boolean).at(-1) ?? path;
  return {
    root: path,
    name,
    input: { linked: false, path: null, available: false },
    excluded: [],
    files: [],
    fragments: [],
    prepared: false,
    runs: [],
    fragments_dir: `${path}/fragments`,
    job: null,
  };
}

/** The mock's whole world. */
const world: {
  view: WorkspaceView | null;
  /** Every run of the workspace, newest first; the views carry them with their staleness. */
  runs: RunFile[];
  timers: number[];
  events: Set<(payload: EngineEventPayload) => void>;
  finished: Set<(payload: EngineFinishedPayload) => void>;
  drops: Set<(path: string) => void>;
  dropsWired: boolean;
} = {
  view: null,
  runs: seededRuns(),
  timers: [],
  events: new Set(),
  finished: new Set(),
  drops: new Set(),
  dropsWired: false,
};

/** Refuses the way the shell refuses: a plain `{ kind, message }`, never an `Error`. */
function refuse(kind: string, message: string): Promise<never> {
  const error: CommandError = { kind, message };
  // Tauri rejects a command with whatever its `Err` serialised to, and `isCommandError` is a
  // shape check on that object; a mock rejecting with an `Error` would exercise a path the real
  // app never takes.
  // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors -- see above
  return Promise.reject(error);
}

/**
 * Stores a view, with the history and its staleness recomputed from what the world holds now.
 * Every write of `world.view` goes through here, so no screen can be shown a run list that is
 * older than the exclusion the user has just made.
 */
function store(view: WorkspaceView): WorkspaceView {
  const runs: RunView[] = world.runs.map((run) => ({ run, stale: staleOf(run, view) }));
  const stored: WorkspaceView = { ...view, runs };
  world.view = stored;
  return stored;
}

/** The open workspace, or `null` when none is — every command that needs one starts here. */
function held(): WorkspaceView | null {
  return world.view;
}

/**
 * The refusal the shell gives for opening, creating or closing a workspace while a worker is
 * writing into one, or `null` when nothing is running.
 *
 * The mock refuses it too, and not out of pedantry: a `busy` that only the real shell gives is a
 * path nothing can be looked at, and the window's handling of it — the banner, the disabled
 * buttons of A §10 — would first be exercised on a user's machine.
 */
function busy(): Promise<never> | null {
  const view = held();
  return view !== null && view.job !== null ? refuse("busy", "ядро занято другой задачей") : null;
}

/** Puts a changed view in and hands it back, the way a command returns the view it just made. */
function put(view: WorkspaceView): Promise<WorkspaceView> {
  return Promise.resolve(store(view));
}

function emit(payload: EngineEventPayload): void {
  for (const handler of world.events) {
    handler(payload);
  }
}

function finish(payload: EngineFinishedPayload): void {
  for (const handler of world.finished) {
    handler(payload);
  }
}

function at(ms: number, run: () => void): void {
  world.timers.push(window.setTimeout(run, ms));
}

function clearTimers(): void {
  for (const id of world.timers) {
    window.clearTimeout(id);
  }
  world.timers = [];
}

/** One stage: its announcement at `from`, then `total` progress lines spread over `ms`. */
function playStage(job: JobKind, runId: string | null, from: number, ms: number, stage: string, total: number): void {
  at(from, () => {
    emit({ job, run_id: runId, event: { event: "stage", name: stage } });
  });
  for (let done = 1; done <= total; done += 1) {
    at(from + (ms * done) / total, () => {
      emit({ job, run_id: runId, event: { event: "progress", stage, done, total } });
    });
  }
}

/**
 * Plays a `Prepare`: the two stages a real one reports, a progress line per fragment, and a
 * `fragment_ready` for each as its thumbnail and display mesh would land.
 */
function playPrepare(): void {
  const total = ALL.length;
  const half = PREPARE_MS / 2;
  const step = half / total;
  const ready: FragmentInfo[] = [];

  // On a timer, not straight away: a real event never arrives before the command it belongs to
  // has returned.
  at(0, () => {
    emit({ job: "prepare", run_id: null, event: { event: "stage", name: "preprocess" } });
  });
  ALL.forEach((_info, i) => {
    at(step * (i + 1), () => {
      emit({ job: "prepare", run_id: null, event: { event: "progress", stage: "preprocess", done: i + 1, total } });
    });
  });

  at(half, () => {
    emit({ job: "prepare", run_id: null, event: { event: "stage", name: "display" } });
  });
  ALL.forEach((info, i) => {
    at(half + step * (i + 1), () => {
      emit({ job: "prepare", run_id: null, event: { event: "progress", stage: "display", done: i + 1, total } });
      emit({ job: "prepare", run_id: null, event: { event: "fragment_ready", ...info } });
      ready.push(info);
    });
  });

  at(PREPARE_MS, () => {
    world.timers = [];
    // From the view as it stands, not the one the job started on: an exclusion made while the
    // preparation ran must survive its ending.
    const now = held();
    if (now === null) {
      return;
    }
    const done = store({ ...now, fragments: ready, prepared: true, job: null });
    finish({
      job: "prepare",
      run_id: null,
      outcome: { Done: { counts: null, engine: null, params: null } },
      view: done,
    });
  });
}

/** How long a mocked run takes. Ten seconds: long enough for A §6's estimate to appear in it. */
const RUN_MS = 10_000;
/** When matching starts and how long it goes on for — the stage every screen of A §6 is about. */
const MATCHING_FROM = 1000;
const MATCHING_MS = 7000;

/**
 * Plays a `Run`: the four stages that report progress, the assembly the worker hands over when it
 * has built one, and the ending that puts the run in the history.
 *
 * Matching takes seven of the ten seconds because it is the only stage the window extrapolates
 * (A §6): less than that and «осталось ≈ …» would never have the five seconds of samples it
 * waits for, and the one line of the running overlay that is hard to get right could not be
 * looked at at all.
 */
function playRun(runId: string): void {
  playStage("run", runId, 0, MATCHING_FROM - 100, "preprocess", ALL.length);
  playStage("run", runId, MATCHING_FROM, MATCHING_MS, "matching", COUNTS.pairs);
  playStage("run", runId, 8000, 1000, "tiers", COUNTS.candidates);
  playStage("run", runId, 9000, 600, "refine", COUNTS.groups);

  at(9800, () => {
    emit({ job: "run", run_id: runId, event: { event: "assembly", ...ASSEMBLY } });
  });

  at(RUN_MS, () => {
    world.timers = [];
    const now = held();
    if (now === null) {
      return;
    }
    const finishedAt = timestamp(new Date());
    world.runs = world.runs.map((run) =>
      run.id === runId ? { ...run, status: { state: "done" }, finished: finishedAt, counts: COUNTS } : run,
    );
    const done = store({ ...now, job: null });
    finish({
      job: "run",
      run_id: runId,
      outcome: { Done: { counts: COUNTS, engine: done.runs[0]?.run.engine ?? null, params: null } },
      view: done,
    });
  });
}

/**
 * A plain browser knows no filesystem paths, so a dropped folder is reported under
 * `/mock/scans/<its name>` — enough for the empty state's drop target to be exercised.
 */
function wireDrops(): void {
  if (world.dropsWired) {
    return;
  }
  world.dropsWired = true;
  window.addEventListener("dragover", (e) => {
    e.preventDefault();
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    const item = e.dataTransfer?.items[0];
    const name = item?.webkitGetAsEntry()?.name;
    if (name === undefined) {
      return;
    }
    for (const handler of world.drops) {
      handler(`/mock/scans/${name}`);
    }
  });
}

/** A run of the history by id, or `undefined` when there is no such run. */
function runOf(runId: string): RunFile | undefined {
  return world.runs.find((run) => run.id === runId);
}

/**
 * The refusal the shell gives for a file a run never wrote (`{ kind: "io" }`), which the window
 * reads as «nothing to draw». Anything but a finished run has one.
 */
function unwritten(runId: string, file: string): Promise<never> {
  return refuse("io", `${WORKSPACE_PATH}/runs/${runId}/${file}: файл не найден`);
}

/**
 * The other refusal a run's file can give: it is there and it will not parse (`{ kind: "json" }`),
 * which A §10 makes a banner rather than «nothing to draw».
 *
 * Asked for with `?corrupt=assembly`, `?corrupt=candidates` or `?corrupt=both` on the dev URL,
 * because no amount of clicking reaches a half-written file — and the state has a screen of its
 * own, which `tools/look.mjs` has to be able to open.
 */
function corrupted(runId: string, file: string): Promise<never> {
  return refuse("json", `${WORKSPACE_PATH}/runs/${runId}/${file}: EOF while parsing an object at line 84 column 3`);
}

/** Whether `?corrupt=` names `file` (or `both`). */
function isCorrupt(file: "assembly" | "candidates"): boolean {
  const asked = new URLSearchParams(window.location.search).get("corrupt");
  return asked === file || asked === "both";
}

/** A few lines of `tracing`, with the `WARN` and `ERROR` tokens A §10's level filter looks for. */
function engineLog(run: RunFile | null): string {
  const id = run?.id ?? "prepare";
  const lines = [
    `2026-09-20T14:12:00.114+03:00  INFO sherd_desktop::jobs: ${id}: worker started, protocol 1`,
    "2026-09-20T14:12:00.902+03:00  INFO sherd::pipeline: preprocess: 12 fragments, median wall 34.2 mm",
    "2026-09-20T14:12:41.338+03:00  WARN sherd::prepare: FY234005 is not watertight; the penetration test will be skipped for its pairs",
    "2026-09-20T14:12:41.512+03:00  INFO sherd::matching: 66 pairs, 4 skipped by the wall-ratio filter",
    "2026-09-20T14:20:14.771+03:00  INFO sherd::matching: 4210 / 66 pairs, 21 candidates so far",
    "2026-09-20T14:27:55.006+03:00  WARN sherd::tiers: FY234002 has no rival pose; its joins cannot be confirmed by margin",
    "2026-09-20T14:28:35.410+03:00  INFO sherd::assembly: 3 groups, 1 fragment without a pair",
    "2026-09-20T14:29:02.118+03:00  INFO sherd::output: transforms.json, joins.csv, report.md written",
  ];
  const status = run?.status;
  if (status?.state === "failed") {
    lines.push(
      `2026-09-20T14:29:02.980+03:00 ERROR sherd::matching: ${status.message}`,
      "2026-09-20T14:29:03.001+03:00 ERROR sherd_desktop::jobs: the worker exited with code 1",
    );
  }
  return lines.join("\n");
}

export const mockApi: Api = {
  appInfo: () => Promise.resolve({ version: "0.1.0", core_version: "0.1.0", commit: "mock" }),

  recentList: () =>
    Promise.resolve([
      { path: WORKSPACE_PATH, name: "karas", opened_at: "2026-09-20T14:12:00+03:00", available: true },
      { path: "/mock/workspaces/slab", name: "slab", opened_at: "2026-09-18T09:30:00+03:00", available: false },
    ]),

  workspaceCreate: (path) => busy() ?? put(fresh(path)),
  workspaceOpen: (path) => busy() ?? put(fresh(path)),

  workspaceClose: () => {
    const refusal = busy();
    if (refusal !== null) {
      return refusal;
    }
    clearTimers();
    world.view = null;
    return Promise.resolve();
  },

  workspaceView: () => {
    const view = held();
    return view === null ? refuse("no_workspace", "нет открытого воркспейса") : Promise.resolve(view);
  },

  inputLink: (path) => {
    const view = held();
    if (view === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    return put({
      ...view,
      input: { linked: true, path, available: true },
      files: ALL.map(stamp),
      fragments: [],
      prepared: false,
    });
  },

  fragmentExclude: (name, excluded) => {
    const view = held();
    if (view === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    const kept = view.excluded.filter((other) => other !== name);
    return put({ ...view, excluded: excluded ? [...kept, name].sort() : kept });
  },

  prepareStart: () => {
    const view = held();
    if (view === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    const refusal = busy();
    if (refusal !== null) {
      return refusal;
    }
    store({ ...view, job: { kind: "prepare", run_id: null } });
    playPrepare();
    return Promise.resolve();
  },

  runStart: (spec) => {
    const view = held();
    if (view === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    const refusal = busy();
    if (refusal !== null) {
      return refusal;
    }
    const now = new Date();
    const id = newRunId(
      world.runs.map((run) => run.id),
      now,
    );
    // The run folder exists from the moment the run starts (A §4): the history shows it as
    // running, and it is closed where the play ends.
    const started = runFile(id, timestamp(now), null, { state: "running" }, null);
    world.runs = [{ ...started, spec: { ...spec } }, ...world.runs];
    store({ ...view, job: { kind: "run", run_id: id } });
    playRun(id);
    return Promise.resolve(id);
  },

  jobCancel: () => {
    const view = held();
    if (view === null || view.job === null) {
      return Promise.resolve();
    }
    clearTimers();
    const job = view.job;
    if (job.run_id !== null) {
      const runId = job.run_id;
      const finishedAt = timestamp(new Date());
      world.runs = world.runs.map((run) =>
        run.id === runId ? { ...run, status: { state: "cancelled" }, finished: finishedAt } : run,
      );
    }
    const stopped = store({ ...view, job: null });
    finish({
      job: job.kind,
      run_id: job.run_id,
      outcome: { Failed: { kind: "cancelled", message: "отменено" } },
      view: stopped,
    });
    return Promise.resolve();
  },

  calibration: () => {
    const view = held();
    const included = view === null ? 0 : view.files.filter((file) => !view.excluded.includes(file.name)).length;
    const pairs = included < 2 ? 0 : (included * (included - 1)) / 2;
    const ratios = { tiers: 0.033, refine: 0.015, output: 0.016 };
    const pairSeconds = 0.0898;
    const total = Object.values(ratios).reduce((sum, ratio) => sum + ratio, 0);
    return Promise.resolve({
      calibration: { version: 1, runs: 3, pair_seconds: pairSeconds, ratios },
      pairs_upper_bound: pairs,
      estimate_seconds: pairs * pairSeconds * (1 + total),
    });
  },

  runAssembly: (runId) => {
    const run = runOf(runId);
    if (run?.status.state !== "done") {
      return unwritten(runId, "assembly.json");
    }
    return isCorrupt("assembly") ? corrupted(runId, "assembly.json") : Promise.resolve(ASSEMBLY);
  },

  runCandidates: (runId) => {
    const run = runOf(runId);
    if (run?.status.state !== "done") {
      return unwritten(runId, "candidates.json");
    }
    return isCorrupt("candidates") ? corrupted(runId, "candidates.json") : Promise.resolve(CANDIDATES);
  },

  runLog: (runId, maxLines) => {
    const lines = engineLog(runId === null ? null : (runOf(runId) ?? null)).split("\n");
    return Promise.resolve(lines.slice(Math.max(0, lines.length - maxLines)).join("\n"));
  },

  runDelete: (runId) => {
    const view = held();
    if (view === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    if (view.job?.run_id === runId) {
      return refuse("busy", "ядро занято другой задачей");
    }
    world.runs = world.runs.filter((run) => run.id !== runId);
    return put(view);
  },

  engineInfo: () =>
    Promise.resolve({
      backends: [
        "cpu, gpu (wgpu 30.0.1, 1 adapter; R §5.2's coarse score and R §5.4's stage-1 ICP rungs on the device, R §5.6's stage 2 and R §6's two methods on the CPU (policy; D §12's 2c struck))",
        "  [0] Metal Apple M2 Pro (IntegratedGpu)",
      ],
    }),

  pickFolder: () => Promise.resolve(INPUT_PATH),

  assetUrl: (path) => {
    if (path.endsWith(".seg.glb")) {
      return "/mock/pieceA.seg.glb";
    }
    if (path.endsWith(".glb")) {
      return "/mock/pieceA.glb";
    }
    if (path.endsWith(".png")) {
      return "/mock/pieceA.png";
    }
    return path;
  },

  onEngineEvent: (handler) => {
    world.events.add(handler);
    const off: Unlisten = () => {
      world.events.delete(handler);
    };
    return Promise.resolve(off);
  },

  onEngineFinished: (handler) => {
    world.finished.add(handler);
    const off: Unlisten = () => {
      world.finished.delete(handler);
    };
    return Promise.resolve(off);
  },

  onFolderDropped: (handler) => {
    wireDrops();
    world.drops.add(handler);
    const off: Unlisten = () => {
      world.drops.delete(handler);
    };
    return Promise.resolve(off);
  },
};
