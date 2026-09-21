import type { Language } from "../state/ui";
import type { AssemblyDto } from "./bindings/AssemblyDto";
import type { Calibration } from "./bindings/Calibration";
import type { CandidateRow } from "./bindings/CandidateRow";
import type { Event as EngineEvent } from "./bindings/Event";
import type { JobKind } from "./bindings/JobKind";
import type { Outcome } from "./bindings/Outcome";
import type { RunSpec } from "./bindings/RunSpec";
import type { WorkspaceView } from "./bindings/WorkspaceView";

/**
 * What the app is, for the About line and for a bug report: the app's own version, the engine it
 * was built against and the commit both came from (A §10 — a reinstall hint needs something to
 * point at).
 */
export interface AppInfo {
  version: string;
  core_version: string;
  commit: string;
}

/**
 * One row of the welcome screen's «recent» list. `available` is computed at every read, because a
 * list on disk that claims a folder exists is stale the moment someone moves it.
 */
export interface RecentEntry {
  path: string;
  name: string;
  opened_at: string;
  available: boolean;
}

/**
 * How a command refuses. `kind` is one of the shell's closed set — `io`, `json`,
 * `not_a_workspace`, `locked`, `version`, `engine`, `worker`, `busy`, `no_workspace` — which the
 * window turns into a sentence of its own (`error.<kind>`, A §10); `message` is the detail
 * underneath it, in the engine's or the OS's words.
 */
export interface CommandError {
  kind: string;
  message: string;
}

/** Something the running job said, as the shell forwards it on `engine:event`. */
export interface EngineEventPayload {
  job: JobKind;
  run_id: string | null;
  event: EngineEvent;
}

/**
 * The end of a job, on `engine:finished`. `view` is built by the shell after the job slot is
 * emptied, so it already says `job: null` — the window can start the next job straight from here.
 */
export interface EngineFinishedPayload {
  job: JobKind;
  run_id: string | null;
  outcome: Outcome;
  view: WorkspaceView | null;
}

/**
 * What a run of this collection will take, for the launch sheet's «≈ 18 мин» (A §6). Three
 * things and not one: what this machine has measured, how many pairs the collection can make at
 * most — A §7.4's «до N пар», an upper bound because R §4.1's wall-ratio filter skips some — and
 * the seconds those pairs come to. `estimate_seconds` is `null` until a run has finished here,
 * and the sheet says so rather than inventing a figure.
 */
export interface CalibrationView {
  calibration: Calibration;
  pairs_upper_bound: number;
  estimate_seconds: number | null;
}

/** What this build can run on (A §7.4's «Вычисления»), one line per executor in the engine's words. */
export interface EngineInfoView {
  backends: string[];
}

/** Undoes one subscription. */
export type Unlisten = () => void;

/**
 * Everything the window is allowed to ask of the outside world (A §2.1). One interface with two
 * implementations — Tauri's, and an in-memory mock for a plain browser — so that every screen can
 * be looked at without a build of the shell, and so nothing in the app reaches for `invoke`
 * itself.
 */
export interface Api {
  appInfo(): Promise<AppInfo>;
  recentList(): Promise<RecentEntry[]>;
  workspaceCreate(path: string): Promise<WorkspaceView>;
  workspaceOpen(path: string): Promise<WorkspaceView>;
  workspaceClose(): Promise<void>;
  workspaceView(): Promise<WorkspaceView>;
  inputLink(path: string): Promise<WorkspaceView>;
  fragmentExclude(name: string, excluded: boolean): Promise<WorkspaceView>;
  /**
   * Starts preparing the input. `lang` is the language the window is being read in: the shell
   * carries it to the end of the job so that A §6's notification, which the OS shows when the
   * window is behind something else, is worded in it.
   */
  prepareStart(lang: Language): Promise<void>;
  /**
   * Starts a run and answers with its id — the folder under `runs/` everything below asks about.
   * Returns as soon as the worker is on its way; the run itself arrives as `engine:event` and
   * ends with `engine:finished`. `lang` is as [`Api.prepareStart`]'s.
   */
  runStart(spec: RunSpec, lang: Language): Promise<string>;
  jobCancel(): Promise<void>;
  /** What a run of the open workspace would take (A §6), for the launch sheet. */
  calibration(): Promise<CalibrationView>;
  /**
   * What a run assembled, from its own `assembly.json` (A §8). Refuses with kind `io` for a run
   * that never got as far as assembling anything — a cancelled one, or one that failed in
   * matching — which is «nothing to draw» and not a broken workspace.
   */
  runAssembly(runId: string): Promise<AssemblyDto>;
  /** Every candidate of a run, from its `candidates.json` (A §8.3); refuses as [`Api.runAssembly`]. */
  runCandidates(runId: string): Promise<CandidateRow[]>;
  /**
   * The last `maxLines` lines of a run's `engine.log`, or of the workspace's `prepare.log` for
   * `null` (A §10's «Показать лог»). A log that does not exist yet is an empty string.
   */
  runLog(runId: string | null, maxLines: number): Promise<string>;
  /** Moves a run's folder to the OS trash (A §4) and hands back the workspace without it. */
  runDelete(runId: string): Promise<WorkspaceView>;
  /** What the engine can run on; asked of a worker once and kept by the shell. */
  engineInfo(): Promise<EngineInfoView>;
  pickFolder(title: string): Promise<string | null>;
  /** A URL the window can fetch for a file inside the open workspace. */
  assetUrl(path: string): string;
  onEngineEvent(handler: (payload: EngineEventPayload) => void): Promise<Unlisten>;
  onEngineFinished(handler: (payload: EngineFinishedPayload) => void): Promise<Unlisten>;
  onFolderDropped(handler: (path: string) => void): Promise<Unlisten>;
}

/** An object, for a value that came over IPC and is not to be trusted to be one. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * Whether a rejected command rejected with the shell's own error. Tauri rejects with whatever the
 * command's `Err` serialised to, so this is a shape check and not an `instanceof`.
 */
export function isCommandError(e: unknown): e is CommandError {
  return isRecord(e) && typeof e.kind === "string" && typeof e.message === "string";
}

/** Whatever a thrown value is, in words, without `String(unknown)` on an object. */
function describe(value: unknown): string {
  if (value instanceof Error) {
    return value.message;
  }
  switch (typeof value) {
    case "string":
      return value;
    case "number":
    case "boolean":
    case "bigint":
      return value.toString();
    case "object":
      return value === null ? "null" : JSON.stringify(value);
    default:
      return typeof value;
  }
}

/**
 * Every failure as one shape, so a store never has to think about what it caught. Anything that
 * is not the shell's error — a bug in the window, a dropped connection — becomes kind `unknown`,
 * which has its own `error.unknown` sentence rather than being shown as a blank one.
 */
export function toCommandError(e: unknown): CommandError {
  return isCommandError(e) ? e : { kind: "unknown", message: describe(e) };
}
