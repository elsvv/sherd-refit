import { create } from "zustand";

import { api } from "../ipc";
import type { BlenderOutcome, CommandError, EngineEventPayload, EngineFinishedPayload } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { ExportWhat } from "../ipc/bindings/ExportWhat";
import type { ResolutionDto } from "../ipc/bindings/ResolutionDto";
import type { ScopeDto } from "../ipc/bindings/ScopeDto";
import { useAssembly } from "./assembly";

/**
 * A §9: getting the work out of the app — «Экспорт», which the review session answers, and
 * «Открыть в Blender», which is the shell's own.
 *
 * One store for both because they are one corner of the window (the top bar's «Экспорт» menu
 * and the group inspector's two buttons) and neither is worth a store of its own; they share
 * nothing else and never run at the same time by accident.
 *
 * **What an export reports is events and not a return value** (A §2.1). `export_start` answers
 * as soon as the request is written, and then the session says, in this order: the `refine`
 * stage while R §9 runs over whatever the decisions left rough, an `assembly` — which *is* the
 * run's new baseline, filed by the shell over its `assembly.json` — then, for a folder export
 * only, the `output` stage with one unit per mesh file, then `exported`. A refusal is a
 * `request_failed` and the session stays open, which is what lets the dialog offer another
 * folder without starting over.
 */

/** How far an export has got. */
export type ExportPhase = "idle" | "running" | "done" | "failed";

/** What «Готово: 273 файла, 212 МБ» is made of, as `Event::Exported` carries it. */
export interface ExportResult {
  dest: string;
  files: number;
  bytes: number;
}

export interface ExportState {
  /** The run being exported, or `null` when nothing is. */
  runId: string | null;
  phase: ExportPhase;
  /** The engine's own stage name while it runs: `preprocess`, then `refine`, then `output`. */
  stage: string | null;
  /** How far that stage has got; `total` of `0` is a stage that has said nothing yet. */
  done: number;
  total: number;
  result: ExportResult | null;
  /** Why it did not happen — a refused command, or the session's own `request_failed`. */
  error: CommandError | null;

  /** Whether a Blender launch is being prepared (the script is written before Blender starts). */
  blenderBusy: boolean;
  /** What the last launch came to, for A §9.2's toast; `null` when there is nothing to say. */
  blender: BlenderOutcome | null;
  /** And a launch that could not even get as far as a script. */
  blenderError: CommandError | null;

  /** Asks for A §9.1's export; everything after this arrives as events. */
  start(runId: string, what: ExportWhat, dest: string): Promise<void>;
  /** «Показать в папке» (A §9.1, A §9.2) — the shell shows only what it has agreed to. */
  reveal(path: string): Promise<void>;
  /** Back to «nothing is being exported»: the dialog was closed, or another export is starting. */
  reset(): void;
  /** Puts the refusal away without touching what was exported. */
  dismissError(): void;

  openInBlender(runId: string, scope: ScopeDto, resolution: ResolutionDto): Promise<void>;
  /** Puts A §9.2's toast away. */
  dismissBlender(): void;

  applyEvent(payload: EngineEventPayload): void;
  applyFinished(payload: EngineFinishedPayload): void;
}

/** What an export starts from, and what closing the dialog leaves behind. */
function blank(): Pick<ExportState, "runId" | "phase" | "stage" | "done" | "total" | "result" | "error"> {
  return { runId: null, phase: "idle", stage: null, done: 0, total: 0, result: null, error: null };
}

export const useExport = create<ExportState>()((set, get) => ({
  ...blank(),
  blenderBusy: false,
  blender: null,
  blenderError: null,

  start: async (runId, what, dest) => {
    set({ ...blank(), runId, phase: "running" });
    try {
      await api.exportStart(runId, what, dest);
    } catch (e) {
      // A command that was refused outright — no such run, a worker busy with a real job, a
      // relative path. Nothing was written and no session will say anything about it.
      set({ phase: "failed", error: toCommandError(e) });
    }
  },

  reveal: async (path) => {
    try {
      await api.reveal(path);
    } catch (e) {
      set({ error: toCommandError(e) });
    }
  },

  reset: () => {
    set(blank());
  },

  dismissError: () => {
    set({ error: null });
  },

  openInBlender: async (runId, scope, resolution) => {
    set({ blenderBusy: true, blender: null, blenderError: null });
    try {
      set({ blender: await api.blenderOpen(runId, scope, resolution), blenderBusy: false });
    } catch (e) {
      set({ blenderBusy: false, blenderError: toCommandError(e) });
    }
  },

  dismissBlender: () => {
    set({ blender: null, blenderError: null });
  },

  applyEvent: (payload) => {
    const runId = get().runId;
    // Only this export's own session, and only while it is going: the same two channels carry
    // every job of the app (A §2.1), and a reassembly asked for in the «Ревью» mode must not
    // move a dialog that is showing what an export came to.
    if (runId === null || get().phase !== "running" || payload.job !== "review" || payload.run_id !== runId) {
      return;
    }
    const event = payload.event;
    switch (event.event) {
      case "stage":
        set({ stage: event.name, done: 0, total: 0 });
        break;
      case "progress":
        set({ stage: event.stage, done: event.done, total: event.total });
        break;
      case "assembly":
        // A §9.1: the refinement runs first and its result is the run's new baseline — the
        // shell has already filed it over `assembly.json`, so what is on the screen has to be
        // it. The «Ревью» mode does the same with the same value when it is open, which is one
        // extra render and never a different assembly.
        if (useAssembly.getState().runId === payload.run_id) {
          useAssembly.getState().show({
            groups: event.groups,
            poses: event.poses,
            joins: event.joins,
            unplaced: event.unplaced,
          });
        }
        break;
      case "exported":
        set({
          phase: "done",
          stage: null,
          result: { dest: event.dest, files: event.files.length, bytes: event.bytes },
        });
        break;
      case "request_failed":
        // A §9.1's four sentences — the folder is not empty, it is a file, there is no room, a
        // scan has gone. The session goes on serving, so the dialog stays up and can be asked
        // again with another folder.
        set({ phase: "failed", stage: null, error: { kind: "worker", message: event.message } });
        break;
      default:
        // `ready` while the session loads, and a `pair_detail` or a `dropped` that belongs to
        // the reviewer and not to the export.
        break;
    }
  },

  applyFinished: (payload) => {
    const runId = get().runId;
    if (runId === null || get().phase !== "running" || payload.job !== "review" || payload.run_id !== runId) {
      return;
    }
    // The session ended under the export: a worker that died, or one a job took away. Either
    // way nothing more is coming, and a dialog left on «идёт запись» would wait for ever.
    const outcome = payload.outcome;
    const failure = "Failed" in outcome ? outcome.Failed : null;
    set({
      phase: "failed",
      stage: null,
      error:
        failure === null
          ? { kind: "worker", message: "the review session ended before the export was written" }
          : { kind: failure.kind, message: failure.message },
    });
  },
}));
