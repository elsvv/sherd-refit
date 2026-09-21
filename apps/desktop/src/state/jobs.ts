import { create } from "zustand";

import { api } from "../ipc";
import type { EngineEventPayload, EngineFinishedPayload, Unlisten } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { Event as EngineEvent } from "../ipc/bindings/Event";
import type { FailKind } from "../ipc/bindings/FailKind";
import type { FragmentInfo } from "../ipc/bindings/FragmentInfo";
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

  /** Asks the shell to prepare the input, and clears what the last job left behind. */
  start(): Promise<void>;
  /** Stops the running job; a no-op when none is. */
  cancel(): Promise<void>;

  applyEvent(payload: EngineEventPayload): void;
  applyFinished(payload: EngineFinishedPayload): void;
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

  start: async () => {
    set({ stage: null, progress: {}, startedAt: Date.now(), lastFailure: null });
    try {
      await api.prepareStart();
    } catch (e) {
      set({ startedAt: null });
      useWorkspace.getState().setError(toCommandError(e));
      return;
    }
    // The command returns as soon as the worker is on its way; the view that says so has to be
    // asked for, because only `engine:finished` brings one along by itself.
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
