import { useEffect, useRef } from "react";

import Frame from "./app/Frame";
import { useShortcuts } from "./app/shortcuts";
import Welcome from "./app/Welcome";
import type { Unlisten } from "./ipc/api";
import { toCommandError } from "./ipc/api";
import type { WorkspaceView } from "./ipc/bindings/WorkspaceView";
import { listenToEngine, useJobs } from "./state/jobs";
import { deriveStatus } from "./state/status";
import { useWorkspace } from "./state/workspace";
// For its side effect: i18next has to be configured before the first component asks for a word.
import "./i18n";

/**
 * The input as one string: every scan's name with its size and modification time, under the
 * workspace it belongs to. Two views with the same signature need the same preparation, which is
 * what makes it safe to remember that this one has already been tried.
 *
 * The workspace's own folder is in there because the files alone are not unique: two workspaces
 * can be linked to the very same folder of scans, and the second one would then never be
 * prepared at all.
 */
function inputSignature(view: WorkspaceView): string {
  const files = view.files.map((file) => `${file.name}:${String(file.size)}:${String(file.mtime_ms)}`);
  return [view.root, ...files].join("|");
}

/**
 * The window's root: the welcome screen with no workspace open, the frame with one (A §7.1).
 *
 * It also owns the two subscriptions that belong to the window rather than to any screen — the
 * engine's events, and the moment the window is looked at again. The second one is how «устарел»
 * is noticed (A §4: staleness is checked when the workspace opens, when the window regains focus
 * and on demand); the shell recomputes it, the window only asks.
 */
export default function App() {
  const view = useWorkspace((state) => state.view);
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  useShortcuts();

  // The signature of the folder the last automatic preparation was started for. A ref and not
  // state: it must not cause a render, and it must be read by the effect that writes it.
  const attempted = useRef<string | null>(null);

  useEffect(() => {
    let stop: Unlisten | null = null;
    let gone = false;
    void listenToEngine().then(
      (off) => {
        // StrictMode mounts twice: an effect already cleaned up must not leave a listener behind.
        if (gone) {
          off();
        } else {
          stop = off;
        }
      },
      (e: unknown) => {
        useWorkspace.getState().setError(toCommandError(e));
      },
    );
    return () => {
      gone = true;
      stop?.();
    };
  }, []);

  /**
   * A `Prepare` starts by itself the moment there is something to prepare (A §5: «`Prepare`
   * starts by itself when an input folder is linked or found changed»): the preprocessing is
   * what a run needs anyway, and the thumbnails, display meshes and warnings fall out of it.
   *
   * Once per state of the folder, and no more. A preparation that fails leaves the workspace
   * exactly as it was, so without this guard the failure would start it again, and again; the
   * banner's «Повторить подготовку» is how a user asks for the retry the app will not take by
   * itself. A folder that has actually changed has a new signature and is prepared again.
   */
  useEffect(() => {
    if (view === null) {
      return;
    }
    if (deriveStatus(view, selectedRunId, false).kind !== "unprepared") {
      return;
    }
    const signature = inputSignature(view);
    if (attempted.current === signature) {
      return;
    }
    attempted.current = signature;
    void useJobs.getState().start();
  }, [view, selectedRunId]);

  useEffect(() => {
    const onFocus = () => {
      // With no workspace open there is nothing to refresh, and asking would only earn a
      // `no_workspace` refusal on the welcome screen.
      if (useWorkspace.getState().view !== null) {
        void useWorkspace.getState().refresh();
      }
    };
    window.addEventListener("focus", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  return view === null ? <Welcome /> : <Frame view={view} />;
}
