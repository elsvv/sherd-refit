import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";

import type { WorkspaceView } from "./bindings/WorkspaceView";
import type { Api, AppInfo, EngineEventPayload, EngineFinishedPayload, RecentEntry } from "./api";

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
  prepareStart: async () => {
    await invoke("prepare_start");
  },
  jobCancel: async () => {
    await invoke("job_cancel");
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
