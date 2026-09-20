import type { FileStamp } from "./bindings/FileStamp";
import type { FragmentInfo } from "./bindings/FragmentInfo";
import type { Warning } from "./bindings/Warning";
import type { WorkspaceView } from "./bindings/WorkspaceView";
import type { Api, CommandError, EngineEventPayload, EngineFinishedPayload, Unlisten } from "./api";

/**
 * The window without a shell under it: `pnpm dev` in a plain browser talks to this instead of
 * Tauri, so every screen of A §7 can be looked at — and a preparation watched arriving — without
 * building the app or running the engine (A §11: the interface is checked by hand, and this is
 * what makes that cheap).
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

/** The workspace as it is right after it is created: no input, nothing prepared. */
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
  timers: number[];
  events: Set<(payload: EngineEventPayload) => void>;
  finished: Set<(payload: EngineFinishedPayload) => void>;
  drops: Set<(path: string) => void>;
  dropsWired: boolean;
} = { view: null, timers: [], events: new Set(), finished: new Set(), drops: new Set(), dropsWired: false };

/** Refuses the way the shell refuses: a plain `{ kind, message }`, never an `Error`. */
function refuse(kind: string, message: string): Promise<never> {
  const error: CommandError = { kind, message };
  // Tauri rejects a command with whatever its `Err` serialised to, and `isCommandError` is a
  // shape check on that object; a mock rejecting with an `Error` would exercise a path the real
  // app never takes.
  // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors -- see above
  return Promise.reject(error);
}

/** The open workspace, or `null` when none is — every command that needs one starts here. */
function held(): WorkspaceView | null {
  return world.view;
}

/** Puts a changed view in and hands it back, the way a command returns the view it just made. */
function put(view: WorkspaceView): Promise<WorkspaceView> {
  world.view = view;
  return Promise.resolve(view);
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
    const done: WorkspaceView = { ...now, fragments: ready, prepared: true, job: null };
    world.view = done;
    finish({
      job: "prepare",
      run_id: null,
      outcome: { Done: { counts: null, engine: null, params: null } },
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

export const mockApi: Api = {
  appInfo: () => Promise.resolve({ version: "0.1.0", core_version: "0.1.0", commit: "mock" }),

  recentList: () =>
    Promise.resolve([
      { path: WORKSPACE_PATH, name: "karas", opened_at: "2026-09-20T14:12:00+03:00", available: true },
      { path: "/mock/workspaces/slab", name: "slab", opened_at: "2026-09-18T09:30:00+03:00", available: false },
    ]),

  workspaceCreate: (path) => put(fresh(path)),
  workspaceOpen: (path) => put(fresh(path)),

  workspaceClose: () => {
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
    if (view.job !== null) {
      return refuse("busy", "ядро занято другой задачей");
    }
    world.view = { ...view, job: { kind: "prepare", run_id: null } };
    playPrepare();
    return Promise.resolve();
  },

  jobCancel: () => {
    const view = held();
    if (view === null || view.job === null) {
      return Promise.resolve();
    }
    clearTimers();
    const stopped: WorkspaceView = { ...view, job: null };
    world.view = stopped;
    finish({
      job: "prepare",
      run_id: null,
      outcome: { Failed: { kind: "cancelled", message: "отменено" } },
      view: stopped,
    });
    return Promise.resolve();
  },

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
