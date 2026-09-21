import { create } from "zustand";

import { api } from "../ipc";
import type { CommandError, RecentEntry } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { FragmentInfo } from "../ipc/bindings/FragmentInfo";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { useAssembly } from "./assembly";

/**
 * The open workspace, as the shell last described it (A §5). The window keeps no second opinion:
 * every command hands back a whole `WorkspaceView`, and that value replaces this one — so a
 * screen that is out of date can only be a screen that has not re-rendered, never a screen that
 * disagrees with the disk. The one exception is a view that arrives in the middle of a `Prepare`;
 * see [`carryFragments`] for why, and for how little it carries over.
 *
 * Nothing here derives anything. What the workspace *means* — ready, stale, preparing — is
 * [`deriveStatus`]'s job, computed at render time from this view (A §5).
 */
export interface WorkspaceState {
  /** The shell's last word, or `null` when no workspace is open (the welcome screen). */
  view: WorkspaceView | null;
  /** The welcome screen's list, newest first. */
  recent: RecentEntry[];
  /** The last refusal, until something dismisses it or a command succeeds. */
  error: CommandError | null;
  /** Which run of `view.runs` the window is showing; `null` is «the input as it stands». */
  selectedRunId: string | null;

  /**
   * Puts a view the window was handed rather than asked for in place — how `engine:finished`
   * delivers the fresh one. `null` closes the workspace; anything else goes through
   * [`carryFragments`], which for the view of a *finished* job means «replace outright».
   */
  setView(view: WorkspaceView | null): void;
  /** Files a refusal that did not come from one of the actions below. */
  setError(error: CommandError | null): void;
  /**
   * Picks a run of the history, or none, and puts what it assembled in [`useAssembly`] — the
   * selection and the thing selected are one act, so no screen can be showing a run's 3D over
   * another run's name.
   */
  selectRun(runId: string | null): void;

  loadRecent(): Promise<void>;
  create(path: string): Promise<void>;
  open(path: string): Promise<void>;
  close(): Promise<void>;
  refresh(): Promise<void>;
  linkInput(path: string): Promise<void>;
  excludeFragment(name: string, excluded: boolean): Promise<void>;
  /**
   * Moves a run's folder to the OS trash (A §7.4's history menu). The shell hands back the
   * workspace without it; a run that was being shown is replaced by the newest one that is left,
   * so the window is never pointing at a folder that is no longer there.
   */
  deleteRun(runId: string): Promise<void>;

  /**
   * Puts one `fragment_ready` into the view (A §5's «preparing» row): the grid fills in as the
   * preparation goes, without asking the shell for a whole view twelve times a second.
   */
  applyFragmentReady(info: FragmentInfo): void;
}

/**
 * `incoming`, with the fragment rows a running `Prepare` has already produced kept alive.
 *
 * `WorkspaceView.fragments` is `fragments/index.json`, and the worker writes that file once, when
 * the whole pass is over: every view built while a `Prepare` runs carries the *previous* index —
 * usually none at all. Taking it as it comes would wipe every `fragment_ready` row
 * [`WorkspaceState.applyFragmentReady`] has collected, and the grid of thumbnails the user is
 * watching fill would empty itself at the next command — the refresh that follows
 * `prepare_start`, or an exclusion made while the preparation runs.
 *
 * So while a prepare is in flight the rows are carried over by name, `incoming` winning wherever
 * it has one; a different `root` is a different workspace and carries nothing. The authoritative
 * index arrives with `engine:finished`, whose view has no job and therefore replaces outright.
 */
function carryFragments(current: WorkspaceView | null, incoming: WorkspaceView): WorkspaceView {
  if (current === null || current.root !== incoming.root || incoming.job?.kind !== "prepare") {
    return incoming;
  }
  const byName = new Map(current.fragments.map((info) => [info.name, info]));
  for (const info of incoming.fragments) {
    byName.set(info.name, info);
  }
  return { ...incoming, fragments: [...byName.values()] };
}

/**
 * The run a workspace should open on: the newest one that finished (A §7.4 — the history is
 * newest first, and what someone coming back to a collection wants to see is its last *result*,
 * not its last attempt). `null` when nothing here has ever finished, which leaves the window on
 * the input as it stands and A §5's whole-workspace answers.
 */
function newestDone(view: WorkspaceView): string | null {
  return view.runs.find((row) => row.run.status.state === "done")?.run.id ?? null;
}

export const useWorkspace = create<WorkspaceState>()((set, get) => {
  /** Runs one command: its view on success, its refusal on failure, never both. */
  const command = async (call: () => Promise<WorkspaceView>): Promise<void> => {
    try {
      const view = await call();
      set((state) => ({ view: carryFragments(state.view, view), error: null }));
    } catch (e) {
      set({ error: toCommandError(e) });
    }
  };

  /**
   * Opens or creates a workspace and shows its newest finished run.
   *
   * A workspace opened is a different workspace: the run selected in the last one means nothing
   * here, so the selection goes first and the new one's own newest result takes its place. The
   * two views are compared by identity — [`command`] puts a fresh object in on success and
   * touches nothing on a refusal — so a workspace that would *not* open cannot move the selection
   * of the one that is still there.
   */
  const opened = async (call: () => Promise<WorkspaceView>): Promise<void> => {
    get().selectRun(null);
    const before = get().view;
    await command(call);
    const after = get().view;
    if (after !== null && after !== before) {
      get().selectRun(newestDone(after));
    }
    await get().loadRecent();
  };

  return {
    view: null,
    recent: [],
    error: null,
    selectedRunId: null,

    setView: (view) => {
      set((state) => ({ view: view === null ? null : carryFragments(state.view, view) }));
    },
    setError: (error) => {
      set({ error });
    },
    selectRun: (runId) => {
      if (get().selectedRunId === runId) {
        return;
      }
      set({ selectedRunId: runId });
      // Downwards only: the assembly store knows nothing about the workspace, so this direction
      // can never become a cycle — and a run selected is a run loaded, wherever it was selected
      // from (the history menu, or a run that has just ended).
      if (runId === null) {
        useAssembly.getState().clear();
      } else {
        void useAssembly.getState().load(runId);
      }
    },

    loadRecent: async () => {
      try {
        set({ recent: await api.recentList() });
      } catch (e) {
        set({ error: toCommandError(e) });
      }
    },

    create: (path) => opened(() => api.workspaceCreate(path)),
    open: (path) => opened(() => api.workspaceOpen(path)),

    close: async () => {
      try {
        await api.workspaceClose();
        get().selectRun(null);
        set({ view: null, error: null });
      } catch (e) {
        set({ error: toCommandError(e) });
      }
    },

    refresh: () => command(() => api.workspaceView()),
    linkInput: (path) => command(() => api.inputLink(path)),
    excludeFragment: (name, excluded) => command(() => api.fragmentExclude(name, excluded)),

    deleteRun: async (runId) => {
      try {
        const view = await api.runDelete(runId);
        set((state) => ({ view: carryFragments(state.view, view), error: null }));
        // Only when it was the one being shown: deleting an old run must not throw away the
        // assembly the user is looking at, and `selectRun` would reload it for nothing.
        if (get().selectedRunId === runId) {
          get().selectRun(newestDone(view));
        }
      } catch (e) {
        set({ error: toCommandError(e) });
      }
    },

    applyFragmentReady: (info) => {
      set((state) => {
        const view = state.view;
        if (view === null) {
          return state;
        }
        const at = view.fragments.findIndex((other) => other.name === info.name);
        const fragments =
          at === -1
            ? [...view.fragments, info]
            : view.fragments.map((other, i) => (i === at ? info : other));
        return { view: { ...view, fragments } };
      });
    },
  };
});
