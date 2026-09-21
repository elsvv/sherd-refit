import type { AssemblyDto } from "./bindings/AssemblyDto";
import type { CandidateRow } from "./bindings/CandidateRow";
import type { Decision } from "./bindings/Decision";
import type { FileStamp } from "./bindings/FileStamp";
import type { FragmentInfo } from "./bindings/FragmentInfo";
import type { GroupDto } from "./bindings/GroupDto";
import type { JobKind } from "./bindings/JobKind";
import type { JoinDto } from "./bindings/JoinDto";
import type { PairDetailDto } from "./bindings/PairDetailDto";
import type { RunCounts } from "./bindings/RunCounts";
import type { RunFile } from "./bindings/RunFile";
import type { RunSpec } from "./bindings/RunSpec";
import type { RunStatus } from "./bindings/RunStatus";
import type { RunView } from "./bindings/RunView";
import type { StaleDiff } from "./bindings/StaleDiff";
import type { UnplacedDto } from "./bindings/UnplacedDto";
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
 * The tier pass's evidence for one candidate (A §8.3): what «Почему не подтверждён сам» is
 * written from, and what says why a confirmed join is confirmed.
 *
 * `failed` carries the engine's own refusal strings in the shapes `tiers.rs` writes them —
 * `tight 0.2767 < 0.35`, `no arm: support 0 < 1 and margin 1.00 < 2` — so that the sentences the
 * window makes of them can be read on a screen without a run behind it.
 */
function evidence(failed: readonly string[], arm: string | null, margin: number): CandidateRow["evidence"] {
  const held = failed.length === 0;
  return {
    margin,
    rival_moved_t: 8.4,
    placements: held ? 3 : 1,
    determined_deg: 1e-7,
    determined_t: 4e-8,
    slide_t: 1.42,
    resample_tight_min: held ? 0.39 : 0.24,
    resample_gap_max: held ? 0.014 : 0.038,
    resample_accept: held ? 3 : 1,
    support: arm === "support" ? 2 : 0,
    // Left out rather than emptied: the engine writes neither key when it has nothing to say,
    // and a window that read `[]` as «нет причин» would say it of a confirmed join too.
    ...(held ? {} : { failed: [...failed] }),
    ...(arm === null ? {} : { arm }),
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
        evidence: evidence([], "support", 6.2 + i),
      });
    });
  });

  const probable: [string, string][] = [
    ["FY234001", "FY234006"],
    ["FY234004", "FY234011"],
    ["FY234008", "FY234012"],
    ["FY234003", "FY234010"],
  ];
  // One band, four reasons: A §8.3's «Почему не подтверждён сам» has a sentence for each shape
  // the engine writes, and the mock is where they are read side by side.
  const held: string[][] = [
    ["tight 0.2767 < 0.35"],
    ["gap 0.0358 > 0.015", "cont_n 0.8485 < 0.9"],
    ["no arm: support 0 < 1 and margin 1.00 < 2"],
    ["seam 4.0000 < 5", "redraw 1: pen 0.0006 > 0"],
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
      evidence: evidence(held[i] ?? ["tight 0.2767 < 0.35"], null, 1.1 + i * 0.2),
    });
  });

  // A second pose for the queue's best pair (A §8.3's «Другие позы пары»). R §6's NMS keeps a
  // rival placement when it is far enough from the winner, and the whole point of the arms is
  // that a pair with two plausible poses is not confirmed by score alone — so one pair of the
  // mock has one, and the inspector has something to offer «показать».
  rows.push({
    a: "FY234003",
    b: "FY234010",
    pose: between(4, 9),
    score: 1.15,
    scores: scores(0.26, 0.028, 3.8, 0.0024),
    tier: "probable",
    used: false,
    evidence: evidence(["no arm: support 0 < 1 and margin 1.00 < 2"], null, 1.0),
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

/**
 * What the ten-second play of a run leaves behind: the same findings as [`COUNTS`], but with the
 * seconds the play really spent. A run whose summary says «длительность 0:10» must not go on to
 * report a matching stage of 15:13 — the two lines are read one under the other.
 */
const PLAYED_COUNTS: RunCounts = {
  ...COUNTS,
  timings: [
    { stage: "preprocess", seconds: 0.9 },
    { stage: "matching", seconds: 7.0 },
    { stage: "tiers", seconds: 1.0 },
    { stage: "assembly", seconds: 0.2 },
    { stage: "refine", seconds: 0.6 },
    { stage: "output", seconds: 0.3 },
  ],
};

/** What the engine says about itself in every mock run that got as far as finishing (A §8.2). */
const ENGINE: NonNullable<RunFile["engine"]> = {
  core_version: "0.1.0",
  algo_ref: "2026-09-06/9d4b9d3",
  commit: "mock",
  backend: "gpu:Apple M2 Pro",
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
  // The engine's own default, and the one thing «Стандарт» must not differ from it in: agreement
  // is a flag «Тщательно» raises (`tiers::Thresholds::agree_seeds` is `0`, never a default), and
  // a mock that says `1` here shows the sheet a «Стандарт» that no run of the engine would make.
  agree_seeds: 0,
  thick_ratio: 2.5,
  min_tight: 0.25,
  max_gap: 0.03,
  max_pen: 0.005,
  min_seam: 3.0,
  workers: 0,
  memory_gb: null,
};

/**
 * The thresholds a finished run resolved to, as `run.json`'s `params` holds them (R §1.1's names,
 * with `tiers` as the 47th key). `Params` has forty-six of them and the window reads six: the
 * four gates of R §6.5 that A §8.3's inspector puts beside a candidate's scores, and the tier
 * pass's two arms. The rest would be dead weight in a fixture — a real `run.json` has them all.
 */
const PARAMS: Record<string, unknown> = {
  min_tight: 0.25,
  max_gap: 0.03,
  max_pen: 0.005,
  min_seam: 3.0,
  tiers: { min_tight: 0.35, max_gap_t: 0.015, min_seam: 5.0, min_cont_n: 0.9, min_margin: 2.0, min_support: 1 },
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
    // Filed when the run ends (A §2.1), so a run that failed or was stopped has none — which is
    // also the shape A §8.3's inspector must survive: a limit column with nothing in it.
    params: counts === null ? null : { ...PARAMS },
    input: { files: ALL.map(stamp), excluded: [] },
    engine: counts === null ? null : ENGINE,
    counts,
    carried_from: null,
  };
}

/**
 * The history «karas» carries: one run that worked, one the GPU gave up on, one the user stopped
 * and one a crash left behind — so that every row of A §5's table, and every colour of the run
 * selector, can be looked at without waiting ten seconds for a mocked run first.
 *
 * It belongs to that one workspace and not to the mock at large: a folder the user has just made
 * has no runs in it, and a mock that hands four to every new workspace hides the one state A §5
 * opens on.
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
    // A §4: a run the app died under is marked `interrupted` the next time the folder is opened.
    // Nobody can reach that state by clicking, so the mock has to hold one.
    runFile("2026-09-17_1540", "2026-09-17T15:40:00+03:00", null, { state: "interrupted" }, null),
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

/** The workspace as it is right after it is created: no input, nothing prepared, no runs. */
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

/**
 * Opening a workspace: «karas» is the one the recent list points at and the one that has been
 * worked in — its scans linked, its cache built and its four runs behind it — and every other
 * path is a folder made a moment ago (A §5's empty state).
 *
 * The distinction is the mock's whole history: `world.runs` is set here and nowhere else, so a
 * «Создать воркспейс» cannot come up with somebody else's runs in it.
 */
function opened(path: string): WorkspaceView {
  // Another workspace's runs are another workspace's drafts: both go with the history.
  world.filed = {};
  world.decisions = {};
  if (path !== WORKSPACE_PATH) {
    world.runs = [];
    return fresh(path);
  }
  world.runs = seededRuns();
  return {
    ...fresh(path),
    input: { linked: true, path: INPUT_PATH, available: true },
    files: ALL.map(stamp),
    fragments: ALL,
    prepared: true,
  };
}

/**
 * The review session, as the mock keeps it (A §8). `baseline` is the last **refined** assembly,
 * which is what a reassembled group's poses are taken from when its members and joins have not
 * moved (A §8.4's `merge_refined`); `assembly` is the last one answered, which is what «Уточнить
 * позы» refines.
 */
interface Session {
  runId: string | null;
  baseline: AssemblyDto;
  assembly: AssemblyDto;
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
  review: Session;
  /**
   * `assembly.json` as the **host** has filed it, by run (A §2.1): a review session's every
   * answer is written over the run's own file, so a draft is what the «Сборка» mode draws after
   * the session is closed and what the next session starts from.
   */
  filed: Record<string, AssemblyDto>;
  /**
   * And `decisions.json` the same way (A §8.1): the shell files the whole list before it is sent,
   * so leaving the «Ревью» mode and coming back finds the verdicts where they were, the «Сборка»
   * inspector shows them, and A §8.5's «Перенести решения ревью (N)» has an N.
   */
  decisions: Record<string, Decision[]>;
} = {
  view: null,
  runs: [],
  timers: [],
  events: new Set(),
  finished: new Set(),
  drops: new Set(),
  dropsWired: false,
  review: { runId: null, baseline: ASSEMBLY, assembly: ASSEMBLY },
  filed: {},
  decisions: {},
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

/**
 * The same refusal for a command that first ends a review session, which is what the shell does
 * before it opens, creates or closes a workspace (`jobs::close_session`): a session is a cache
 * over a finished run and never a reason to refuse the user «Закрыть» or «Открыть другой…»
 * (A §8.4). What is left in the slot after that — a `Prepare`, a run — is refused as before.
 */
function freed(): Promise<never> | null {
  closeSession();
  return busy();
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
      run.id === runId
        ? {
            ...run,
            status: { state: "done" },
            finished: finishedAt,
            counts: PLAYED_COUNTS,
            engine: ENGINE,
            params: { ...PARAMS },
          }
        : run,
    );
    const done = store({ ...now, job: null });
    finish({
      job: "run",
      run_id: runId,
      outcome: { Done: { counts: PLAYED_COUNTS, engine: ENGINE, params: { ...PARAMS } } },
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

/**
 * Whether the dev URL carries `?nocal`: the machine has finished no run, so A §6's sheet says an
 * estimate will appear rather than giving one.
 *
 * A flag and not a click, for the reason `?corrupt` is one: it is the state of a file in the
 * app's config folder, nothing on any screen reaches it, and it has a sentence of its own that
 * `tools/look.mjs` has to be able to open.
 */
function noCalibration(): boolean {
  return new URLSearchParams(window.location.search).has("nocal");
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

/**
 * A §8's review session, played out of the same twelve fragments the rest of the mock is made
 * of: a match that takes a moment to load, a reassembly for every decision, a seam for every
 * placement and a refinement that takes a second and a half.
 *
 * It is what makes the «Ревью» screen reviewable at all (A §11): a real session is a worker with
 * three gigabytes of fragment cache in it over a run that took twenty minutes, and nobody is
 * going to produce one to look at a queue row's spacing.
 */

/** How long the mock takes to load a run's match before it says `ready`. */
const REVIEW_OPEN_MS = 900;
/** And to answer one decision — A §8.2 asks for «under a second» and this is what that feels like. */
const REVIEW_APPLY_MS = 150;
/** And to answer one seam. */
const REVIEW_PAIR_MS = 120;
/** And to run R §9 over the groups a draft left unrefined (A §8.4). */
const REVIEW_REFINE_MS = 1500;

/** How many of B's fracture samples the mock draws; a real pair has five thousand (R §3.5.2). */
const CONTACT_POINTS = 420;
/** And how many seam voxel centres; a real pair has a few dozen (R §6.2). */
const SEAM_POINTS = 48;
/** The pair's two limits, in the scans' own length unit (R §1.2); the mock's resolution is ~2. */
const TIGHT_LIMIT = 1.1;
const GAP_LIMIT = 3.3;
/** How long the mock's seam is, in the same unit — about the width of one of its slabs. */
const SEAM_LENGTH = 150;

/** Three numbers in the scans' own frame. */
type Vec3 = [number, number, number];

/** Whether a pair — a join, a candidate, a decision — is about `a` and `b` (A §8.1: unordered). */
function samePair(pair: { a: string; b: string }, a: string, b: string): boolean {
  return (pair.a === a && pair.b === b) || (pair.a === b && pair.b === a);
}

/** The pair's best-scoring candidate, whichever way round the row names it. */
function bestRow(a: string, b: string): CandidateRow | undefined {
  return CANDIDATES.filter((row) => samePair(row, a, b)).sort((x, y) => y.score - x.score)[0];
}

/**
 * Union–find over the collection's names, which is how R §8 turns a list of joins into groups.
 * `union` answers whether the two were in different groups — `false` is a join that closes a
 * loop, and the greedy assembly refuses those.
 */
function forest(): { find: (name: string) => string; union: (a: string, b: string) => boolean } {
  const parent = new Map<string, string>();
  const find = (name: string): string => {
    let root = name;
    for (;;) {
      const up = parent.get(root);
      if (up === undefined || up === root) {
        return root;
      }
      root = up;
    }
  };
  const union = (a: string, b: string): boolean => {
    const left = find(a);
    const right = find(b);
    if (left === right) {
      return false;
    }
    parent.set(right, left);
    return true;
  };
  return { find, union };
}

/**
 * The groups a set of joins makes, largest first, each member in the collection's own order —
 * the order the engine writes and the order [`assemblyDto`] built the run's own groups in.
 */
function componentsOf(joins: readonly JoinDto[]): string[][] {
  const tree = forest();
  for (const join of joins) {
    tree.union(join.a, join.b);
  }
  const by = new Map<string, string[]>();
  for (const name of NAMES) {
    const root = tree.find(name);
    const members = by.get(root);
    if (members === undefined) {
      by.set(root, [name]);
    } else {
      members.push(name);
    }
  }
  return [...by.values()].sort((x, y) => y.length - x.length || ((x[0] ?? "") < (y[0] ?? "") ? -1 : 1));
}

/**
 * What makes a group *the same group* for A §8.4's «refined stays refined»: its members and the
 * joins whose both ends are inside it, both in a fixed order. The same key as the shell's
 * `review::merge_refined` uses, so the mock and the real session agree about what refinement
 * survives a decision.
 */
function groupKey(members: readonly string[], joins: readonly JoinDto[]): string {
  const inside = new Set(members);
  const edges = joins
    .filter((join) => inside.has(join.a) && inside.has(join.b))
    .map((join) => [join.a, join.b].sort().join("-"))
    .sort();
  return `${[...members].sort().join(",")}|${edges.join(",")}`;
}

/**
 * R §8's own sentence for a join both of whose fragments are already placed relative to each
 * other (`Rejection::InconsistentWithAssembled`): the assembly takes joins best score first, and
 * one that closes a loop disagrees with what is already standing. The two numbers are derived
 * from the pair's score rather than measured — this is a mock — but the shape is the engine's,
 * which is what A §8.4's «2 не встали» list is read against.
 */
function inconsistent(a: string, b: string): string {
  const score = bestRow(a, b)?.score ?? 0;
  const angle = ((score * 4.7) % 30).toFixed(1);
  const distance = ((score * 0.37) % 1).toFixed(2);
  return `inconsistent with the assembled poses (${angle} deg, ${distance} t)`;
}

/**
 * What `Reassemble` answers (A §8.2): the run's own joins less the rejected pairs, plus the
 * accepted ones taken best score first, and the groups that fall out of them.
 *
 * A group whose members and joins are exactly those of a group of the last **refined** assembly
 * keeps its poses and stays refined (A §8.4); every other group is laid out afresh and is a
 * draft. A decision about a fragment the collection no longer has is dropped and reported, as
 * `to_constraints` drops it — `constraints::resolve` would fail the whole run on an unknown name.
 */
function reassembled(decisions: readonly Decision[]): { assembly: AssemblyDto; dropped: Decision[] } {
  const known = new Set<string>(NAMES);
  const live = decisions.filter((decision) => known.has(decision.a) && known.has(decision.b));
  const dropped = decisions.filter((decision) => !known.has(decision.a) || !known.has(decision.b));

  const vetoed = live.filter((decision) => decision.verdict === "reject");
  const joins: JoinDto[] = ASSEMBLY.joins.filter(
    (join) => !vetoed.some((decision) => samePair(decision, join.a, join.b)),
  );
  const tree = forest();
  for (const join of joins) {
    tree.union(join.a, join.b);
  }

  const unplaced: UnplacedDto[] = [];
  const wanted = live
    .filter((decision) => decision.verdict === "accept")
    .sort((x, y) => (bestRow(y.a, y.b)?.score ?? 0) - (bestRow(x.a, x.b)?.score ?? 0));
  for (const decision of wanted) {
    if (joins.some((join) => samePair(join, decision.a, decision.b))) {
      // A join the run already built with: confirming it changes nothing and is not a refusal.
      continue;
    }
    if (tree.union(decision.a, decision.b)) {
      joins.push({ a: decision.a, b: decision.b });
    } else {
      unplaced.push({ a: decision.a, b: decision.b, reason: inconsistent(decision.a, decision.b) });
    }
  }

  const baseline = world.review.baseline;
  const before = new Map(baseline.groups.map((group) => [groupKey(group.members, baseline.joins), group]));
  const groups: GroupDto[] = [];
  const poses: AssemblyDto["poses"] = {};
  for (const members of componentsOf(joins)) {
    const was = before.get(groupKey(members, joins));
    const keeps = members.length > 1 && was?.refined === true;
    members.forEach((name, i) => {
      poses[name] = (keeps ? baseline.poses[name] : undefined) ?? fan(i);
    });
    groups.push({ members: [...members], refined: keeps });
  }
  return { assembly: { groups, poses, joins, unplaced }, dropped };
}

/** The same assembly with R §9 behind it: every group of two or more is refined (A §8.4). */
function allRefined(assembly: AssemblyDto): AssemblyDto {
  return {
    ...assembly,
    groups: assembly.groups.map((group) => ({ ...group, refined: group.members.length > 1 })),
  };
}

/** `v` at unit length, or `fallback` when it has no length to speak of. */
function normalized(v: Vec3, fallback: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]);
  return length < 1e-9 ? fallback : [v[0] / length, v[1] / length, v[2] / length];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * The seam of one placement (A §2.2's `PairDetail`): a few hundred points along a wavy line
 * halfway between the two pieces, most of them within the tight limit, some in the gap and a few
 * beyond it — which is what a real contact map looks like once R §6.1's facing window has thrown
 * the back of the sherd out.
 *
 * Everything is in A's frame, as the real one is, and everything is derived from the pose, so two
 * different pairs do not draw the same seam in the same place.
 */
function pairDetail(a: string, b: string, pose: readonly (readonly number[])[]): PairDetailDto {
  const cell = (row: number, column: number): number => pose[row]?.[column] ?? 0;
  const offset: Vec3 = [cell(0, 3), cell(1, 3), cell(2, 3)];
  const centre: Vec3 = [offset[0] / 2, offset[1] / 2, offset[2] / 2];
  // A frame on the seam: `along` points from A to B, the other two span the plane between them.
  const along = normalized(offset, [1, 0, 0]);
  const across = normalized(cross(along, [0, 0, 1]), [0, 1, 0]);
  const up = cross(along, across);
  /** A point of the seam plane at `s` along it, pushed `off` towards B. */
  const point = (s: number, off: number): [number, number, number] => {
    const wave = 11 * Math.sin(s * 0.06) + 4 * Math.sin(s * 0.21);
    return [
      centre[0] + across[0] * s + up[0] * wave + along[0] * off,
      centre[1] + across[1] * s + up[1] * wave + along[1] * off,
      centre[2] + across[2] * s + up[2] * wave + along[2] * off,
    ];
  };

  const contact: [number, number, number][] = [];
  const contactClass: number[] = [];
  for (let i = 0; i < CONTACT_POINTS; i += 1) {
    const s = (i / (CONTACT_POINTS - 1) - 0.5) * SEAM_LENGTH;
    // How far this sample of B stands off A's surface: a gentle wobble, and every so often one
    // on a face that points away from A, which R §6.1 reports as «beyond» however close it is.
    const away = Math.abs(0.55 * Math.sin(i * 0.37) + 0.75 * Math.sin(i * 0.11 + 1.3));
    const distance = away + (i % 11 === 0 ? 3.2 : 0);
    contact.push(point(s, i % 2 === 0 ? distance : -distance));
    contactClass.push(distance < TIGHT_LIMIT ? 0 : distance < GAP_LIMIT ? 1 : 2);
  }

  const seam: [number, number, number][] = [];
  for (let i = 0; i < SEAM_POINTS; i += 1) {
    seam.push(point((i / (SEAM_POINTS - 1) - 0.5) * SEAM_LENGTH, 0));
  }

  return { a, b, contact, contact_class: contactClass, seam, tight: TIGHT_LIMIT, gap: GAP_LIMIT };
}

/**
 * What `review_apply` refuses before it files anything (kind `json`): a list that could never be
 * a `decisions.json`. The window's own store cannot make one, which is the point — a bug there
 * becomes a visible refusal rather than a corrupt file.
 */
function malformed(decisions: readonly Decision[]): string | null {
  const seen = new Set<string>();
  for (const decision of decisions) {
    if (decision.a === "" || decision.b === "") {
      return "решение без имени фрагмента";
    }
    if (decision.a === decision.b) {
      return `решение о паре ${decision.a} с самим собой`;
    }
    const key = [decision.a, decision.b].sort().join("-");
    if (seen.has(key)) {
      return `о паре ${decision.a} — ${decision.b} сказано дважды`;
    }
    seen.add(key);
  }
  return null;
}

/** The refusal every session request gives when none is open. */
function noSession(): Promise<never> {
  return refuse("worker", "нет открытой сессии ревью");
}

/**
 * Ends the session, as `jobs::close_session` does: the slot is freed and the window is told with
 * an `engine:finished`, because that — and not the command's return — is «the session is gone».
 *
 * Called by `reviewClose` and by every job that needs the worker: a warm session is a cache over
 * a run that has already finished, never a reason to refuse a preparation or a run (A §8.4).
 */
function closeSession(): void {
  const runId = world.review.runId;
  if (runId === null) {
    return;
  }
  clearTimers();
  world.review = { runId: null, baseline: ASSEMBLY, assembly: ASSEMBLY };
  const view = held();
  const freed = view === null ? null : store({ ...view, job: null });
  finish({
    job: "review",
    run_id: runId,
    outcome: { Done: { counts: null, engine: null, params: null } },
    view: freed,
  });
}

/** Plays the opening of a session: the collection loaded from the cache, then `ready` (A §8). */
function playReviewOpen(runId: string): void {
  playStage("review", runId, 0, REVIEW_OPEN_MS, "preprocess", ALL.length);
  at(REVIEW_OPEN_MS, () => {
    emit({
      job: "review",
      run_id: runId,
      event: { event: "ready", fragments: ALL.length, candidates: CANDIDATES.length },
    });
  });
}

export const mockApi: Api = {
  appInfo: () => Promise.resolve({ version: "0.1.0", core_version: "0.1.0", commit: "mock" }),

  recentList: () =>
    Promise.resolve([
      { path: WORKSPACE_PATH, name: "karas", opened_at: "2026-09-20T14:12:00+03:00", available: true },
      { path: "/mock/workspaces/slab", name: "slab", opened_at: "2026-09-18T09:30:00+03:00", available: false },
    ]),

  workspaceCreate: (path) => freed() ?? put(opened(path)),
  workspaceOpen: (path) => freed() ?? put(opened(path)),

  workspaceClose: () => {
    const refusal = freed();
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
    if (held() === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    // A §8.4: a review session is closed to make room, never a reason to refuse the work the
    // user has just asked for. It has to go before `busy`, which counts it as a job.
    closeSession();
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
    if (held() === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    closeSession();
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
    if (noCalibration()) {
      // Nothing measured yet: A §6 has the shell answer `null` rather than invent a figure, and
      // `pair_seconds` is `0` because no run has said what one costs here.
      return Promise.resolve({
        calibration: { version: 1, runs: 0, pair_seconds: 0, ratios },
        pairs_upper_bound: pairs,
        estimate_seconds: null,
      });
    }
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
    // What the host last filed for this run — a review's draft, or the run's own assembly.
    return isCorrupt("assembly") ? corrupted(runId, "assembly.json") : Promise.resolve(world.filed[runId] ?? ASSEMBLY);
  },

  runCandidates: (runId) => {
    const run = runOf(runId);
    if (run?.status.state !== "done") {
      return unwritten(runId, "candidates.json");
    }
    return isCorrupt("candidates") ? corrupted(runId, "candidates.json") : Promise.resolve(CANDIDATES);
  },

  // What was filed for this run, or the empty file the shell answers for a run nobody has
  // reviewed — never a refusal, as `DecisionsFile::load_or_default` never is.
  runDecisions: (runId) => Promise.resolve({ version: 1, decisions: world.decisions[runId] ?? [] }),

  reviewOpen: (runId) => {
    const view = held();
    if (view === null) {
      return refuse("no_workspace", "нет открытого воркспейса");
    }
    // The shell answers instantly for the run that is already open, so the mode may call this
    // on every entry without guarding.
    if (world.review.runId === runId) {
      return Promise.resolve();
    }
    if (view.job !== null && view.job.kind !== "review") {
      return refuse("busy", "ядро занято другой задачей");
    }
    if (runOf(runId)?.status.state !== "done") {
      // A §10: a run with no saved match stays viewable and review is off; the shell refuses
      // with the same `io` a missing file gives.
      return unwritten(runId, "match.state");
    }
    closeSession();
    // A §8.4: the session's baseline is `assembly.json` as it stands — the run's own assembly,
    // or the draft a reviewer left there last time.
    const standing = world.filed[runId] ?? ASSEMBLY;
    world.review = { runId, baseline: standing, assembly: standing };
    const held_ = held();
    if (held_ !== null) {
      store({ ...held_, job: { kind: "review", run_id: runId } });
    }
    playReviewOpen(runId);
    return Promise.resolve();
  },

  reviewApply: (decisions) => {
    const runId = world.review.runId;
    if (runId === null) {
      return noSession();
    }
    const wrong = malformed(decisions.decisions);
    if (wrong !== null) {
      return refuse("json", `решения не сохранены: ${wrong}`);
    }
    const { assembly, dropped } = reassembled(decisions.decisions);
    world.review = { ...world.review, assembly };
    // A §2.1: the host files every assembly a session answers over the run's own `assembly.json`,
    // and the list itself over its `decisions.json` — before it is sent, as `review_apply` does.
    world.filed[runId] = assembly;
    world.decisions[runId] = decisions.decisions;
    at(REVIEW_APPLY_MS, () => {
      // What could not come along is said first, so the notice is up before the assembly the
      // reviewer will be looking at (the order the worker sends them in).
      if (dropped.length > 0) {
        emit({ job: "review", run_id: runId, event: { event: "dropped", decisions: dropped } });
      }
      emit({ job: "review", run_id: runId, event: { event: "assembly", ...assembly } });
    });
    return Promise.resolve();
  },

  reviewPair: (a, b, pose) => {
    const runId = world.review.runId;
    if (runId === null) {
      return noSession();
    }
    const missing = [a, b].find((name) => !NAMES.includes(name));
    at(REVIEW_PAIR_MS, () => {
      // A request the session cannot answer does not end it: it says so and goes on serving.
      const event =
        missing === undefined
          ? ({ event: "pair_detail", ...pairDetail(a, b, pose) } as const)
          : ({ event: "request_failed", message: `the collection has no fragment named ${missing}` } as const);
      emit({ job: "review", run_id: runId, event });
    });
    return Promise.resolve();
  },

  reviewRefine: () => {
    const runId = world.review.runId;
    if (runId === null) {
      return noSession();
    }
    const drafts = world.review.assembly.groups.filter((group) => group.members.length > 1 && !group.refined);
    playStage("review", runId, 0, REVIEW_REFINE_MS, "refine", Math.max(1, drafts.length));
    at(REVIEW_REFINE_MS, () => {
      const refined = allRefined(world.review.assembly);
      // The session's baseline is what it last refined (A §8.4), so a group decided against and
      // decided for again comes back refined.
      world.review = { ...world.review, baseline: refined, assembly: refined };
      world.filed[runId] = refined;
      emit({ job: "review", run_id: runId, event: { event: "assembly", ...refined } });
    });
    return Promise.resolve();
  },

  reviewClose: () => {
    closeSession();
    return Promise.resolve();
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

  // What the shell hands over once it has parsed the engine's `info`: the card, not the prose.
  engineInfo: () => Promise.resolve({ adapters: ["Metal Apple M2 Pro"], gpu: true }),

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
