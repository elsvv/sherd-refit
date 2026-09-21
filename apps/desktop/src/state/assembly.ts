import { create } from "zustand";

import { api } from "../ipc";
import type { CommandError } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { AssemblyDto } from "../ipc/bindings/AssemblyDto";
import type { CandidateRow } from "../ipc/bindings/CandidateRow";

/**
 * The run the window is showing, as its two files describe it (A §8): `assembly.json`, which is
 * the groups and the poses, and `candidates.json`, which is every pair that was scored.
 *
 * A store of its own and not part of [`useWorkspace`], because this is the *selected* run's
 * content and not the workspace's state: it is fetched, it takes a moment, it can fail on its own,
 * and the workspace is perfectly usable while it is being read. Nothing else depends on it, which
 * is why the workspace store may call into it without a cycle.
 *
 * A run that assembled nothing — cancelled, or failed in matching — has neither file, and that is
 * not an error: the shell refuses with kind `io`, and here that reads as «nothing to draw» (A §4,
 * the run folder is the worker's and a run that died young leaves an empty one).
 */
export interface AssemblyState {
  /** Which run is loaded, or being loaded; `null` when none is. */
  runId: string | null;
  /** What it assembled, or `null` when it assembled nothing or is still being read. */
  assembly: AssemblyDto | null;
  /** Every candidate it scored; empty for a run that wrote none. */
  candidates: CandidateRow[];
  /** Whether the two files are on their way. */
  loading: boolean;
  /** A refusal that is not «this run has no such file». */
  error: CommandError | null;

  /** Reads a run's two files. The last call wins, however the earlier ones finish. */
  load(runId: string): Promise<void>;
  /** Back to no run shown — the workspace was closed, or the selection was cleared. */
  clear(): void;
}

/** What one of a run's files came to: its content, or the refusal that is worth showing. */
interface Fetched<T> {
  value: T | null;
  error: CommandError | null;
}

/**
 * One file of a run, with «it is not there» told apart from «it would not be read». A run need
 * not have written either file, and A §4 makes that a normal end of a run rather than damage, so
 * a kind `io` refusal comes back as no value and no error; anything else — a `json` that will not
 * parse, a `no_workspace` — is a refusal the window has to say out loud.
 */
async function fetched<T>(call: () => Promise<T>): Promise<Fetched<T>> {
  try {
    return { value: await call(), error: null };
  } catch (e) {
    const error = toCommandError(e);
    return { value: null, error: error.kind === "io" ? null : error };
  }
}

export const useAssembly = create<AssemblyState>()((set) => {
  /**
   * Which load is the current one. Selecting three runs in a row starts three pairs of reads, and
   * they can come back in any order; without this the first run clicked could be the one left on
   * the screen. A counter and not state: nothing renders it.
   */
  let ticket = 0;

  return {
    runId: null,
    assembly: null,
    candidates: [],
    loading: false,
    error: null,

    load: async (runId) => {
      ticket += 1;
      const mine = ticket;
      set({ runId, assembly: null, candidates: [], loading: true, error: null });
      // Both at once: they are two independent reads of two files, and the viewer has nothing to
      // draw until the first of them is in either way.
      const [assembly, candidates] = await Promise.all([
        fetched(() => api.runAssembly(runId)),
        fetched(() => api.runCandidates(runId)),
      ]);
      if (mine !== ticket) {
        return;
      }
      set({
        assembly: assembly.value,
        candidates: candidates.value ?? [],
        loading: false,
        error: assembly.error ?? candidates.error,
      });
    },

    clear: () => {
      // A load in flight must not land after this one: the run it belongs to is no longer shown.
      ticket += 1;
      set({ runId: null, assembly: null, candidates: [], loading: false, error: null });
    },
  };
});
