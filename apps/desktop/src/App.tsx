import { useEffect } from "react";

import Frame from "./app/Frame";
import { useShortcuts } from "./app/shortcuts";
import Welcome from "./app/Welcome";
import type { Unlisten } from "./ipc/api";
import { toCommandError } from "./ipc/api";
import { listenToEngine } from "./state/jobs";
import { useWorkspace } from "./state/workspace";
// For its side effect: i18next has to be configured before the first component asks for a word.
import "./i18n";

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
  useShortcuts();

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
