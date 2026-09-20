import { create } from "zustand";

import { api } from "../ipc";
import type { CommandError, RecentEntry } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { FragmentInfo } from "../ipc/bindings/FragmentInfo";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";

/**
 * The open workspace, as the shell last described it (A §5). The window keeps no second opinion:
 * every command hands back a whole `WorkspaceView`, and that value replaces this one — so a
 * screen that is out of date can only be a screen that has not re-rendered, never a screen that
 * disagrees with the disk.
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

  /** Replaces the view outright — how `engine:finished` delivers the fresh one. */
  setView(view: WorkspaceView | null): void;
  /** Files a refusal that did not come from one of the actions below. */
  setError(error: CommandError | null): void;
  /** Picks a run of the history, or none. */
  selectRun(runId: string | null): void;

  loadRecent(): Promise<void>;
  create(path: string): Promise<void>;
  open(path: string): Promise<void>;
  close(): Promise<void>;
  refresh(): Promise<void>;
  linkInput(path: string): Promise<void>;
  excludeFragment(name: string, excluded: boolean): Promise<void>;

  /**
   * Puts one `fragment_ready` into the view (A §5's «preparing» row): the grid fills in as the
   * preparation goes, without asking the shell for a whole view twelve times a second.
   */
  applyFragmentReady(info: FragmentInfo): void;
}

export const useWorkspace = create<WorkspaceState>()((set, get) => {
  /** Runs one command: its view on success, its refusal on failure, never both. */
  const command = async (call: () => Promise<WorkspaceView>): Promise<void> => {
    try {
      set({ view: await call(), error: null });
    } catch (e) {
      set({ error: toCommandError(e) });
    }
  };

  return {
    view: null,
    recent: [],
    error: null,
    selectedRunId: null,

    setView: (view) => {
      set({ view });
    },
    setError: (error) => {
      set({ error });
    },
    selectRun: (runId) => {
      set({ selectedRunId: runId });
    },

    loadRecent: async () => {
      try {
        set({ recent: await api.recentList() });
      } catch (e) {
        set({ error: toCommandError(e) });
      }
    },

    // A workspace opened is a different workspace: the run selected in the last one means
    // nothing here, so the selection goes with it.
    create: async (path) => {
      set({ selectedRunId: null });
      await command(() => api.workspaceCreate(path));
      await get().loadRecent();
    },
    open: async (path) => {
      set({ selectedRunId: null });
      await command(() => api.workspaceOpen(path));
      await get().loadRecent();
    },

    close: async () => {
      try {
        await api.workspaceClose();
        set({ view: null, selectedRunId: null, error: null });
      } catch (e) {
        set({ error: toCommandError(e) });
      }
    },

    refresh: () => command(() => api.workspaceView()),
    linkInput: (path) => command(() => api.inputLink(path)),
    excludeFragment: (name, excluded) => command(() => api.fragmentExclude(name, excluded)),

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
