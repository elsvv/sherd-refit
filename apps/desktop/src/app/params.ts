import type { TFunction } from "i18next";

import type { BackendChoice } from "../ipc/bindings/BackendChoice";
import type { Preset } from "../ipc/bindings/Preset";
import type { RunSpec } from "../ipc/bindings/RunSpec";

/**
 * A §7.4's launch sheet, as data: what the eleven parameters are, what a sheet nobody has touched
 * says, and how a sheet becomes a [`RunSpec`] the shell will take.
 *
 * One table and not eleven fields spread through the component, because every one of them needs
 * the same four things — a range, a step, a default and a way back to it — and a parameter whose
 * bounds live beside its input is a parameter whose bounds are checked in one place and enforced
 * in another.
 *
 * The defaults are `RunSpec::default()`'s (`crates/sherd-app-core/src/protocol.rs`), which are in
 * turn `Params::default()`'s and `RunOptions::default()`'s. They are repeated here rather than
 * asked for over IPC: the sheet has to draw a «сбросить» before any command has answered, and a
 * default that arrived late would move a field the user was already typing in.
 */

/** The `tiers` switch — the one of the eleven that is not a number. */
export type SwitchKey = "tiers";

/** The ten that are. */
export type NumberKey =
  | "seed"
  | "target_faces"
  | "agree_seeds"
  | "thick_ratio"
  | "min_tight"
  | "max_gap"
  | "min_seam"
  | "max_pen"
  | "workers"
  | "memory_gb";

/** One of A §7.4's eleven; also the i18n key of its name and of its sentence (`param.<key>.*`). */
export type ParamKey = SwitchKey | NumberKey;

/** What a row of the table is. */
export type ParamRow =
  | { key: SwitchKey; kind: "switch" }
  | {
      key: NumberKey;
      /** `int` refuses a fraction; `float` takes one. */
      kind: "int" | "float";
      min: number;
      max: number;
      step: number;
      /** Whether an empty field is a value — it is `None`, and only `memory_gb` has one. */
      optional: boolean;
    };

/** «Тщательно»: the agreement arm on, which is what makes a run cost about twice as much (A §7.4). */
export const THOROUGH_AGREE_SEEDS = 2;

/** The sheet as it opens on a workspace that has never been run (`RunSpec::default()`). */
export const DEFAULT_SPEC: RunSpec = {
  preset: "standard",
  backend: "auto",
  adapter: null,
  gpu_memory_gb: null,
  seed: 0,
  target_faces: 200_000,
  tiers: true,
  agree_seeds: 0,
  thick_ratio: 2.5,
  min_tight: 0.25,
  max_gap: 0.03,
  max_pen: 0.005,
  min_seam: 3.0,
  workers: 0,
  memory_gb: null,
};

/**
 * The eleven, in the order A §7.4 lists them. The bounds are what the engine can be asked for and
 * not what it was measured at: a seed is any seed, a threshold that is a fraction lives in
 * `[0, 1]`, and the two budgets are bounded by nothing but the absurd.
 */
export const PARAMS: readonly ParamRow[] = [
  { key: "seed", kind: "int", min: 0, max: 4_294_967_295, step: 1, optional: false },
  { key: "target_faces", kind: "int", min: 1_000, max: 5_000_000, step: 10_000, optional: false },
  { key: "tiers", kind: "switch" },
  { key: "agree_seeds", kind: "int", min: 0, max: 8, step: 1, optional: false },
  { key: "thick_ratio", kind: "float", min: 1, max: 20, step: 0.1, optional: false },
  { key: "min_tight", kind: "float", min: 0, max: 1, step: 0.01, optional: false },
  { key: "max_gap", kind: "float", min: 0, max: 1, step: 0.005, optional: false },
  { key: "min_seam", kind: "float", min: 0, max: 100, step: 0.5, optional: false },
  { key: "max_pen", kind: "float", min: 0, max: 1, step: 0.001, optional: false },
  { key: "workers", kind: "int", min: 0, max: 1_024, step: 1, optional: false },
  { key: "memory_gb", kind: "float", min: 0, max: 4_096, step: 1, optional: true },
];

/** Every preset, for the cards and for reading one back out of a `run.json`. */
export const PRESETS: readonly Preset[] = ["standard", "thorough", "custom"];

/** Every executor, likewise (A §7.4's «Вычисления»). */
export const BACKENDS: readonly BackendChoice[] = ["auto", "cpu", "gpu"];

/**
 * The sheet as it stands. The numbers are **text**, exactly as they were typed: a field that is
 * being emptied to be retyped is not a zero, and a value out of range has to survive long enough
 * for the sheet to say so (A §7.4: «a value outside `[min, max]` disables the button and says
 * why»).
 */
export interface Draft {
  preset: Preset;
  backend: BackendChoice;
  tiers: boolean;
  values: Record<NumberKey, string>;
}

/** What one number of a spec looks like in its field; an absent budget is an empty field. */
function textOf(key: NumberKey, spec: RunSpec): string {
  const value = spec[key];
  return value === null ? "" : String(value);
}

/** The spec as a sheet. */
export function draftOf(spec: RunSpec): Draft {
  return {
    preset: spec.preset,
    backend: spec.backend,
    tiers: spec.tiers,
    values: {
      seed: textOf("seed", spec),
      target_faces: textOf("target_faces", spec),
      agree_seeds: textOf("agree_seeds", spec),
      thick_ratio: textOf("thick_ratio", spec),
      min_tight: textOf("min_tight", spec),
      max_gap: textOf("max_gap", spec),
      min_seam: textOf("min_seam", spec),
      max_pen: textOf("max_pen", spec),
      workers: textOf("workers", spec),
      memory_gb: textOf("memory_gb", spec),
    },
  };
}

/** The spec each of A §7.4's three buttons stands for, before the executor is chosen. */
export function presetSpec(preset: Preset): RunSpec {
  switch (preset) {
    case "standard":
    case "custom":
      return { ...DEFAULT_SPEC, preset };
    case "thorough":
      return { ...DEFAULT_SPEC, preset, agree_seeds: THOROUGH_AGREE_SEEDS };
  }
}

/**
 * The sheet after one of the three cards was clicked. «Стандарт» and «Тщательно» *are* the
 * engine's defaults with one threshold moved, so choosing them puts every threshold back —
 * otherwise a value left over from a custom sheet would be carried into a run the history calls
 * «Стандарт». «Свои параметры» changes nothing but the label: it opens on whatever the last card
 * left, which is what makes it a starting point rather than a blank form.
 *
 * The three machine answers survive the card, as [`machineOf`] explains.
 */
export function withPreset(draft: Draft, preset: Preset): Draft {
  if (preset === "custom") {
    return { ...draft, preset };
  }
  const reset = draftOf(presetSpec(preset));
  return {
    ...reset,
    backend: draft.backend,
    values: { ...reset.values, workers: draft.values.workers, memory_gb: draft.values.memory_gb },
  };
}

/** Why a typed value cannot be used, or `null` when it can. */
export type FieldError = "number" | "range";

/**
 * One field as the arithmetic sees it. A comma is a decimal point here: the window is written in
 * Russian, «2,5» is how a comma-separator keyboard types it, and a platform whose number field
 * does not normalise it would otherwise refuse a value the user typed correctly.
 */
function numberOf(text: string): number {
  return Number(text.replace(",", "."));
}

/** What is wrong with one field (A §7.4: the sheet says why, and «Собрать» waits). */
export function fieldError(row: ParamRow, text: string): FieldError | null {
  if (row.kind === "switch") {
    return null;
  }
  const trimmed = text.trim();
  if (trimmed === "") {
    return row.optional ? null : "number";
  }
  const value = numberOf(trimmed);
  if (!Number.isFinite(value) || (row.kind === "int" && !Number.isInteger(value))) {
    return "number";
  }
  return value < row.min || value > row.max ? "range" : null;
}

/** Every field's complaint, by key, for the sheet to draw under each row. */
export function fieldErrors(draft: Draft): Partial<Record<NumberKey, FieldError>> {
  const found: Partial<Record<NumberKey, FieldError>> = {};
  for (const row of PARAMS) {
    if (row.kind === "switch") {
      continue;
    }
    const error = fieldError(row, draft.values[row.key]);
    if (error !== null) {
      found[row.key] = error;
    }
  }
  return found;
}

/** One field as a number, or `null` for an empty optional one. */
function valueOf(draft: Draft, key: NumberKey): number | null {
  const trimmed = draft.values[key].trim();
  return trimmed === "" ? null : numberOf(trimmed);
}

/**
 * `workers` and `memory_gb` as the sheet has them — the two numbers that travel with `backend`
 * rather than with the preset.
 *
 * All three are about *this computer* and not about this collection: which executor a run starts
 * on, how much memory it may take, how many threads it may take. A §11's settings screen is where
 * the machine answers them once, and the shell reads exactly these two out of it for a `Prepare`
 * it starts by itself (`jobs.rs`' `prepare`). A card that put them back to the engine's defaults
 * would mean «Стандарт» quietly gave a run the whole machine after the user had asked it not to.
 *
 * A field that does not read as a number falls back to the default instead of making the spec
 * `null`: outside «Свои параметры» these two are not on the screen, so there would be nothing for
 * the sheet to point at while «Собрать» stayed disabled.
 */
function machineOf(draft: Draft): Pick<RunSpec, "workers" | "memory_gb"> {
  const errors = fieldErrors(draft);
  const workers = errors.workers === undefined ? valueOf(draft, "workers") : null;
  const memory = errors.memory_gb === undefined ? valueOf(draft, "memory_gb") : DEFAULT_SPEC.memory_gb;
  return { workers: workers ?? DEFAULT_SPEC.workers, memory_gb: memory };
}

/**
 * What the sheet would start, or `null` while something in it cannot be read as a number in its
 * range — which is what disables «Собрать».
 *
 * Only «Свои параметры» reads the thresholds. The other two presets *are* the defaults (with
 * «Тщательно»'s one threshold), so they are built from [`presetSpec`] and cannot be made invalid
 * by a field left over from a sheet the user backed out of. The executor and [`machineOf`]'s two
 * budgets are the sheet's own either way.
 */
export function specOf(draft: Draft): RunSpec | null {
  if (draft.preset !== "custom") {
    return { ...presetSpec(draft.preset), backend: draft.backend, ...machineOf(draft) };
  }
  if (Object.keys(fieldErrors(draft)).length > 0) {
    return null;
  }
  return {
    ...DEFAULT_SPEC,
    preset: "custom",
    backend: draft.backend,
    tiers: draft.tiers,
    seed: valueOf(draft, "seed") ?? DEFAULT_SPEC.seed,
    target_faces: valueOf(draft, "target_faces") ?? DEFAULT_SPEC.target_faces,
    agree_seeds: valueOf(draft, "agree_seeds") ?? DEFAULT_SPEC.agree_seeds,
    thick_ratio: valueOf(draft, "thick_ratio") ?? DEFAULT_SPEC.thick_ratio,
    min_tight: valueOf(draft, "min_tight") ?? DEFAULT_SPEC.min_tight,
    max_gap: valueOf(draft, "max_gap") ?? DEFAULT_SPEC.max_gap,
    min_seam: valueOf(draft, "min_seam") ?? DEFAULT_SPEC.min_seam,
    max_pen: valueOf(draft, "max_pen") ?? DEFAULT_SPEC.max_pen,
    workers: valueOf(draft, "workers") ?? DEFAULT_SPEC.workers,
    memory_gb: valueOf(draft, "memory_gb"),
  };
}

/** An object, for a value that came out of a file and is not to be trusted to be one. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** One of a closed set, or `null` for anything else. */
function oneOf<T extends string>(value: unknown, all: readonly T[]): T | null {
  return typeof value === "string" ? (all.find((one) => one === value) ?? null) : null;
}

/** A number that is a number, or `null` — which for an optional field is also a value. */
function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * The sheet a run was started from, read back out of its `run.json` (A §7.4: «the sheet opens
 * pre-filled from the last spec when it parses»).
 *
 * `RunFile.spec` is whatever the window sent, kept verbatim and never validated on the way in, so
 * this is a parse and not a cast: a spec written by another version of the app, or by hand, is
 * refused whole rather than filling half the sheet with `NaN`.
 */
export function parseSpec(value: unknown): RunSpec | null {
  if (!isRecord(value)) {
    return null;
  }
  const preset = oneOf(value.preset, PRESETS);
  const backend = oneOf(value.backend, BACKENDS);
  const seed = finite(value.seed);
  const targetFaces = finite(value.target_faces);
  const agreeSeeds = finite(value.agree_seeds);
  const thickRatio = finite(value.thick_ratio);
  const minTight = finite(value.min_tight);
  const maxGap = finite(value.max_gap);
  const minSeam = finite(value.min_seam);
  const maxPen = finite(value.max_pen);
  const workers = finite(value.workers);
  if (
    preset === null ||
    backend === null ||
    seed === null ||
    targetFaces === null ||
    agreeSeeds === null ||
    thickRatio === null ||
    minTight === null ||
    maxGap === null ||
    minSeam === null ||
    maxPen === null ||
    workers === null ||
    typeof value.tiers !== "boolean"
  ) {
    return null;
  }
  return {
    preset,
    backend,
    adapter: typeof value.adapter === "string" ? value.adapter : null,
    gpu_memory_gb: finite(value.gpu_memory_gb),
    seed,
    target_faces: targetFaces,
    tiers: value.tiers,
    agree_seeds: agreeSeeds,
    thick_ratio: thickRatio,
    min_tight: minTight,
    max_gap: maxGap,
    min_seam: minSeam,
    max_pen: maxPen,
    workers,
    memory_gb: finite(value.memory_gb),
  };
}

/** A span of time nobody measured. Punctuation, the same in both languages (as `format.ts` has it). */
const UNKNOWN = "—";

/**
 * A number of seconds in words: «45 с», «18 мин», «1 ч 20 мин».
 *
 * Here rather than in either of the two components that need it — the sheet's «≈ 18 мин» before a
 * run and the overlay's «осталось ≈ 11 мин» during one — because the two must agree: a user who
 * is told «≈ 18 мин» and then watches «осталось ≈ 1080 с» has been told two different things.
 *
 * Minutes from a minute up and never seconds beside them: an estimate good to the second is a
 * claim this app cannot make (A §6 measures a rate over the last minute).
 */
export function formatDuration(seconds: number, t: TFunction): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return UNKNOWN;
  }
  const total = Math.round(seconds);
  if (total < 60) {
    return t("duration.seconds", { n: total });
  }
  const minutes = Math.round(total / 60);
  if (minutes < 60) {
    return t("duration.minutes", { n: minutes });
  }
  return t("duration.hours", { h: Math.floor(minutes / 60), m: String(minutes % 60).padStart(2, "0") });
}
