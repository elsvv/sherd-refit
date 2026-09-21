import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";

import type { AssemblyDto } from "./bindings/AssemblyDto";
import type { CandidateRow } from "./bindings/CandidateRow";
import type { DecisionsFile } from "./bindings/DecisionsFile";
import type { Settings } from "./bindings/Settings";
import type { WorkspaceView } from "./bindings/WorkspaceView";
import type {
  Api,
  AppInfo,
  BlenderOutcome,
  CalibrationView,
  EngineEventPayload,
  EngineFinishedPayload,
  EngineInfoView,
  RecentEntry,
} from "./api";

/** The shell's event names, as `src-tauri/src/jobs.rs` emits them. */
const ENGINE_EVENT = "engine:event";
const ENGINE_FINISHED = "engine:finished";

/**
 * The real thing: every method is one `#[tauri::command]` of `src-tauri/src/commands.rs`, by the
 * same snake_case name, and nothing else in the window calls `invoke` (A §2.1).
 *
 * The commands take the workspace's own paths as text and hand back one `WorkspaceView`; files
 * are reached only through the asset protocol, which the shell scoped to the open workspace when
 * it opened it, so `assetUrl` on anything outside it simply will not load.
 */
export const tauriApi: Api = {
  appInfo: () => invoke<AppInfo>("app_info"),
  recentList: () => invoke<RecentEntry[]>("recent_list"),
  workspaceCreate: (path) => invoke<WorkspaceView>("workspace_create", { path }),
  workspaceOpen: (path) => invoke<WorkspaceView>("workspace_open", { path }),
  // `Result<(), _>` comes back as `null`, which is not a `void`: await it and return nothing.
  workspaceClose: async () => {
    await invoke("workspace_close");
  },
  workspaceView: () => invoke<WorkspaceView>("workspace_view"),
  inputLink: (path) => invoke<WorkspaceView>("input_link", { path }),
  fragmentExclude: (name, excluded) => invoke<WorkspaceView>("fragment_exclude", { name, excluded }),
  // `lang` is a required argument of the command (`jobs::Lang`), so it goes in the payload under
  // the very name the Rust parameter has — Tauri refuses the call outright without it.
  prepareStart: async (lang) => {
    await invoke("prepare_start", { lang });
  },
  // `carryFrom` is A §8.5's carry-over; `null` is a run that starts from nobody's decisions.
  runStart: (spec, lang, carryFrom) => invoke<string>("run_start", { spec, lang, carryFrom }),
  jobCancel: async () => {
    await invoke("job_cancel");
  },

  calibration: () => invoke<CalibrationView>("calibration"),
  // Tauri 2 matches a camelCase argument to its snake_case parameter, which is what the commands
  // below take: `run_id`, `max_lines`.
  runAssembly: (runId) => invoke<AssemblyDto>("run_assembly", { runId }),
  runCandidates: (runId) => invoke<CandidateRow[]>("run_candidates", { runId }),
  runDecisions: (runId) => invoke<DecisionsFile>("run_decisions", { runId }),
  runLog: (runId, maxLines) => invoke<string>("run_log", { runId, maxLines }),
  runDelete: (runId) => invoke<WorkspaceView>("run_delete", { runId }),
  engineInfo: () => invoke<EngineInfoView>("engine_info"),

  // A §8's review session. Each of these answers `null`, because what the window is waiting for
  // is the event the session sends back — `ready`, `assembly`, `pair_detail` — and not the
  // return of the call that asked for it.
  reviewOpen: async (runId) => {
    await invoke("review_open", { runId });
  },
  reviewApply: async (decisions) => {
    await invoke("review_apply", { decisions });
  },
  reviewPair: async (a, b, pose) => {
    await invoke("review_pair", { a, b, pose });
  },
  reviewRefine: async () => {
    await invoke("review_refine");
  },
  reviewClose: async () => {
    await invoke("review_close");
  },

  // A §9's two exports and A §11's settings. `exportStart` and `blenderOpen` take the run the
  // window is showing: which run is selected is the window's own state, and the shell opens the
  // session for it.
  exportDefaultDest: (what) => invoke<string>("export_default_dest", { what }),
  exportStart: async (runId, what, dest) => {
    await invoke("export_start", { runId, what, dest });
  },
  blenderOpen: (runId, scope, resolution) =>
    invoke<BlenderOutcome>("blender_open", { runId, scope, resolution }),
  reveal: async (path) => {
    await invoke("reveal", { path });
  },
  settingsGet: () => invoke<Settings>("settings_get"),
  settingsSet: (settings) => invoke<Settings>("settings_set", { settings }),

  /**
   * «Снимок PNG» through the OS's own save dialog. The webview cannot write a file and a
   * `<a download>` inside a Tauri window downloads nowhere, so the path is asked for here and
   * the bytes are written by the shell — which is also the only side that may touch a folder
   * the user picked outside the workspace (A §2.1).
   *
   * The data URL is split rather than parsed: `HTMLCanvasElement.toDataURL` writes exactly
   * `data:image/png;base64,…`, and anything else is not a PNG this window drew.
   */
  savePng: async (name, dataUrl) => {
    const png = dataUrl.slice(dataUrl.indexOf(",") + 1);
    if (!dataUrl.startsWith("data:image/png;base64,") || png.length === 0) {
      return;
    }
    const path = await save({ defaultPath: name, filters: [{ name: "PNG", extensions: ["png"] }] });
    if (path === null) {
      return;
    }
    await invoke("snapshot_save", { path, png });
  },

  pickFolder: async (title) => {
    // `directory: true, multiple: false` has to survive inference as the literal `true`/`false`,
    // or the plugin's return type widens to `string[] | null`.
    const picked = await open({ directory: true, multiple: false, title } as const);
    return picked;
  },

  assetUrl: (path) => convertFileSrc(path),

  onEngineEvent: (handler) =>
    listen<EngineEventPayload>(ENGINE_EVENT, (e) => {
      handler(e.payload);
    }),

  onEngineFinished: (handler) =>
    listen<EngineFinishedPayload>(ENGINE_FINISHED, (e) => {
      handler(e.payload);
    }),

  onFolderDropped: (handler) =>
    getCurrentWebview().onDragDropEvent((e) => {
      // Only the drop itself, and only its first path: A §5's empty state takes one folder.
      if (e.payload.type !== "drop") {
        return;
      }
      const first = e.payload.paths[0];
      if (first !== undefined) {
        handler(first);
      }
    }),
};
