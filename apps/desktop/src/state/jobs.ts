import { create } from "zustand";

import { api } from "../ipc";
import type { EngineEventPayload, EngineFinishedPayload, Unlisten } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { Event as EngineEvent } from "../ipc/bindings/Event";
import type { FailKind } from "../ipc/bindings/FailKind";
import type { FragmentInfo } from "../ipc/bindings/FragmentInfo";
import type { RunSpec } from "../ipc/bindings/RunSpec";
import type { StageSample } from "./eta";
import { useUi } from "./ui";
import { useWorkspace } from "./workspace";

/** How far one stage has got. */
export interface Progress {
  done: number;
  total: number;
}

/** Why the last job ended badly (A §10), kept until the next one starts. */
export interface Failure {
  kind: FailKind;
  message: string;
}

/**
 * What the running job is doing (A §5's «preparing» and «running» rows, A §6's strip). Only what
 * the events carry: the stage, how far each stage got, when this job started, and how the last
 * one failed. Everything about the *workspace* stays in [`useWorkspace`] — these two stores never
 * hold the same fact twice.
 *
 * The direction is one-way: this store writes into the workspace store (a `fragment_ready` as it
 * lands, the fresh view when the job ends) and the workspace store knows nothing about jobs.
 */
export interface JobsState {
  /** The stage the worker last announced, or `null` between jobs. */
  stage: string | null;
  /** By stage name, so a finished stage keeps its counts while the next one runs. */
  progress: Record<string, Progress>;
  /** `Date.now()` when this job was asked for; `null` when none is running. */
  startedAt: number | null;
  /** The last `Failed`, for A §10's sentence and its «Показать лог». */
  lastFailure: Failure | null;
  /**
   * Every `Progress` this job has reported, thinned to one a second per stage — what A §6's
   * «осталось ≈ 11 мин» is computed from ([`remainingSeconds`]). Kept here and not in the
   * status line because it is the *job's* history and outlives any component that shows it.
   */
  samples: StageSample[];
  /**
   * What the stages after matching cost on this machine, as fractions of matching (A §6). Read
   * once when a run starts; empty while nothing has been read, which the estimate treats as «the
   * later stages cost nothing» rather than as a reason to say nothing at all.
   */
  ratios: Record<string, number>;

  /**
   * Asks the shell to prepare the input, and clears what the last job left behind. The UI's
   * language goes with it: the shell's end-of-job notification (A §6) is written where the job
   * ends, by which time the window may be behind another one.
   */
  start(): Promise<void>;
  /**
   * Asks the shell for a run of the launch sheet's `spec` (A §7.4), as [`JobsState.start`] does
   * for a preparation. The run's id is not returned: it arrives in the view's `job`, and again
   * with `engine:finished`, and a second copy here would be a second opinion to keep in step.
   */
  startRun(spec: RunSpec): Promise<void>;
  /** Stops the running job; a no-op when none is. */
  cancel(): Promise<void>;

  applyEvent(payload: EngineEventPayload): void;
  applyFinished(payload: EngineFinishedPayload): void;
}

/**
 * How often a stage's progress is worth remembering. A matching pass reports several times a
 * second for an hour; the estimate reads a minute of samples at a time, so one a second is all
 * the resolution it can use and all the memory it should cost.
 */
const SAMPLE_MS = 1000;

/**
 * `samples` with `next` in it — appended when the stage's last sample is a second old or older,
 * and otherwise dropped, *unless* it is the one that says the stage is over: that one replaces
 * the sample it is too close to, so that no two samples of a stage share a moment (a rate
 * measured between them would be a division by zero) and the estimate can still see that the
 * stage has finished.
 */
function sampled(samples: StageSample[], next: StageSample): StageSample[] {
  let at = -1;
  for (const [i, sample] of samples.entries()) {
    if (sample.stage === next.stage) {
      at = i;
    }
  }
  const last = at === -1 ? undefined : samples[at];
  if (last === undefined || next.at - last.at >= SAMPLE_MS) {
    return [...samples, next];
  }
  if (next.done < next.total) {
    // The very same array, not a copy of it: a run reports progress ten times a second, and a
    // new array each time would re-render everything that watches the samples for nothing.
    return samples;
  }
  return samples.map((sample, i) => (i === at ? next : sample));
}

/** What a job starts from: nothing said, nothing measured, and the clock running. */
function fresh(): Pick<JobsState, "stage" | "progress" | "samples" | "startedAt" | "lastFailure"> {
  return { stage: null, progress: {}, samples: [], startedAt: Date.now(), lastFailure: null };
}

/**
 * Reads this machine's calibration into the store, for A §6's «осталось ≈». It is asked for when
 * a run starts and not when the window opens: it changes only when a run ends, and a window that
 * is never used for a run should not spawn the command at all.
 *
 * A refusal is not worth a banner — the estimate degrades to the pairs alone, which is most of
 * it — so it is swallowed and the ratios are left as they were.
 */
async function learnRatios(): Promise<void> {
  try {
    useJobs.setState({ ratios: (await api.calibration()).calibration.ratios });
  } catch {
    // An estimate is not worth interrupting a run that has already started.
  }
}

/** The fields of a `fragment_ready`, named one by one so a new one cannot be dropped silently. */
function fragmentOf(event: Extract<EngineEvent, { event: "fragment_ready" }>): FragmentInfo {
  return {
    name: event.name,
    file: event.file,
    size: event.size,
    mtime_ms: event.mtime_ms,
    stats: event.stats,
    warnings: event.warnings,
    coloured: event.coloured,
    display_faces: event.display_faces,
  };
}

export const useJobs = create<JobsState>()((set) => ({
  stage: null,
  progress: {},
  startedAt: null,
  lastFailure: null,
  samples: [],
  ratios: {},

  start: async () => {
    set(fresh());
    try {
      await api.prepareStart(useUi.getState().language);
    } catch (e) {
      set({ startedAt: null });
      useWorkspace.getState().setError(toCommandError(e));
      return;
    }
    // The command returns as soon as the worker is on its way; the view that says so has to be
    // asked for, because only `engine:finished` brings one along by itself.
    await useWorkspace.getState().refresh();
  },

  startRun: async (spec) => {
    set(fresh());
    try {
      await api.runStart(spec, useUi.getState().language);
    } catch (e) {
      set({ startedAt: null });
      useWorkspace.getState().setError(toCommandError(e));
      return;
    }
    // Not awaited: the run is already going, and the window must show that before it knows what
    // the run will cost.
    void learnRatios();
    await useWorkspace.getState().refresh();
  },

  cancel: async () => {
    try {
      await api.jobCancel();
    } catch (e) {
      useWorkspace.getState().setError(toCommandError(e));
    }
  },

  applyEvent: (payload) => {
    const event = payload.event;
    switch (event.event) {
      case "hello":
      case "info":
        // The handshake and the backend list: nothing the strip shows.
        break;
      case "stage":
        set({ stage: event.name });
        break;
      case "progress":
        set((state) => ({
          progress: { ...state.progress, [event.stage]: { done: event.done, total: event.total } },
          samples: sampled(state.samples, {
            stage: event.stage,
            done: event.done,
            total: event.total,
            at: Date.now(),
          }),
        }));
        break;
      case "fragment_ready":
        useWorkspace.getState().applyFragmentReady(fragmentOf(event));
        break;
      case "assembly":
        // The assembled scene arrives with milestone 4's «Сборка»; nothing here reads it yet.
        break;
      case "done":
        // `engine:finished` carries the same counts and a fresh view with it — wait for that one
        // rather than acting twice on one ending.
        break;
      case "failed":
        set({ lastFailure: { kind: event.kind, message: event.message } });
        break;
    }
  },

  applyFinished: (payload) => {
    const outcome = payload.outcome;
    set({
      stage: null,
      progress: {},
      samples: [],
      startedAt: null,
      lastFailure: "Failed" in outcome ? { kind: outcome.Failed.kind, message: outcome.Failed.message } : null,
    });
    if (payload.view === null) {
      // The shell could not build one — the workspace was closed under the job, or its input
      // folder would not be read. Keeping what we have would leave a view whose `job` is still
      // set: a window stuck on «идёт подготовка» over a job that has just ended. Ask instead.
      void useWorkspace.getState().refresh();
    } else {
      useWorkspace.getState().setView(payload.view);
    }
    // A run that has ended is the run the user is now looking at, however it ended: what they
    // asked for is on the screen if it worked, and A §5's `failed`/`cancelled` rows — with their
    // «Показать лог» and «Повторить…» — need it selected to be shown at all. Selecting it loads
    // what it assembled, and a run that assembled nothing simply has nothing to draw.
    if (payload.run_id !== null) {
      useWorkspace.getState().selectRun(payload.run_id);
    }
    // And a run that assembled something is what the user pressed «Собрать» to see, so the
    // window goes there by itself (A §7.3). Only for a run, and only for one that finished: a
    // preparation has nothing to show, and a run that failed is reported where the user is.
    if (payload.job === "run" && "Done" in outcome) {
      useUi.getState().setMode("assembly");
    }
  },
}));

/**
 * Subscribes the store to the shell's two event names, and to the one thing about the workspace
 * this store has to know. Called once, when the window mounts; the returned function undoes all
 * three, which is what a React effect's cleanup needs in StrictMode.
 *
 * The workspace store is *watched* from here rather than told from there: the direction stays
 * one-way, and nothing in `workspace.ts` has to know that jobs exist.
 */
export async function listenToEngine(): Promise<Unlisten> {
  const offEvent = await api.onEngineEvent((payload) => {
    useJobs.getState().applyEvent(payload);
  });
  const offFinished = await api.onEngineFinished((payload) => {
    useJobs.getState().applyFinished(payload);
  });
  // A failure belongs to the workspace it happened in (A §10: the banner names a run of *this*
  // collection). Opening another workspace, creating one or closing this one must not leave it
  // standing over a workspace that never saw it — and it is by `root` and not by identity,
  // because every command hands back a fresh view of the same workspace many times a minute.
  const offWorkspace = useWorkspace.subscribe((state, previous) => {
    if (state.view?.root !== previous.view?.root) {
      useJobs.setState({ lastFailure: null });
    }
  });
  return () => {
    offEvent();
    offFinished();
    offWorkspace();
  };
}
