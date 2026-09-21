/**
 * How long the running job has left (A §6), as a pure function of the progress events and of what
 * the last runs on this machine measured.
 *
 * Pure and here rather than in a component, because it is the one number of the window a user
 * will hold against a clock: it has to be a thing that can be tested against arithmetic, not
 * something that only exists while a run is going.
 */

/** One `Progress` event, as [`useJobs`] keeps it: at most one a second per stage. */
export interface StageSample {
  /** R §11.2's stage name. */
  stage: string;
  /** Units finished. */
  done: number;
  /** Units in the stage. */
  total: number;
  /** `Date.now()` when the window heard about it. */
  at: number;
}

/**
 * The one stage that is extrapolated (A §6). Everything after it is a ratio of it, because
 * matching is the only stage whose work is known in advance — the pairs — and the only one that
 * takes long enough for its own rate to mean anything.
 */
const MATCHING = "matching";

/**
 * The stage before it, which is never a ratio either: it runs *before* matching, so its seconds
 * are already spent by the time there is a rate to multiply. [`Calibration::learn`] keeps neither
 * of these two in `ratios`; skipping them here is what makes that a fact and not a coincidence.
 */
const BEFORE = "preprocess";

/** The rate is the last minute's, so that a machine that has slowed down says so within a minute. */
const WINDOW_MS = 60_000;

/** Nothing is said about a rate measured over less than this: the first tile is not the run. */
const MIN_AGE_MS = 5_000;

/** Nor about one measured over less than this much of the work, however long it took. */
const MIN_FRACTION = 0.01;

/**
 * Seconds left, or `null` when there is nothing to go on yet.
 *
 * A §6: the pairs that are left at the rate matching is going now, plus the stages after it at
 * the ratios the calibration holds — of which the ones that have already reported themselves
 * finished cost nothing. Before matching starts, and for its first five seconds, the honest
 * answer is that there is no answer; the window says so in words instead of showing a figure that
 * halves every second.
 *
 * Once matching is over the estimate is what those later stages were projected to take, less the
 * time since — a countdown and not a measurement, because the stages after matching report almost
 * no progress of their own (a run of `fixtures/slab` announced three stages of the six it timed).
 * It reaches zero and stays there rather than growing, which is the failure mode a user forgives.
 */
export function remainingSeconds(
  samples: readonly StageSample[],
  now: number,
  ratios: Readonly<Record<string, number>>,
): number | null {
  const pairs = samples.filter((sample) => sample.stage === MATCHING);
  const first = pairs[0];
  const last = pairs.at(-1);
  if (first === undefined || last === undefined || last.total <= 0) {
    return null;
  }
  if (now - first.at < MIN_AGE_MS || last.done < last.total * MIN_FRACTION) {
    return null;
  }
  const rate = rateOf(pairs);
  if (rate === null) {
    return null;
  }
  // The whole of matching at the rate it is going: what the later stages' ratios are ratios *of*.
  const projected = last.total / rate;
  const later = laterSeconds(samples, ratios, projected);
  if (last.done >= last.total) {
    return Math.max(0, later - (now - last.at) / 1000);
  }
  return Math.max(0, (last.total - last.done) / rate + later);
}

/**
 * Pairs a second, over the last [`WINDOW_MS`] of `samples` — or over all of them while matching
 * is younger than that. `null` when the samples say nothing: one sample, or two of the same
 * moment, or a counter that has not moved.
 *
 * With samples further apart than the window — the engine reports progress when it has something
 * to report, not on a clock — the last interval is used instead of nothing at all.
 */
function rateOf(samples: readonly StageSample[]): number | null {
  const last = samples.at(-1);
  if (last === undefined) {
    return null;
  }
  const cutoff = last.at - WINDOW_MS;
  const inWindow = samples.findIndex((sample) => sample.at >= cutoff);
  const from = samples[inWindow === -1 || inWindow === samples.length - 1 ? samples.length - 2 : inWindow];
  if (from === undefined) {
    return null;
  }
  const seconds = (last.at - from.at) / 1000;
  const done = last.done - from.done;
  return seconds > 0 && done > 0 ? done / seconds : null;
}

/**
 * The seconds the stages after matching still owe: every ratio the calibration holds, less the
 * ones whose stage has already reported `done === total`.
 *
 * Over the calibration's keys and not over the samples', so a stage this machine has never timed
 * — `objects` appeared on a real run that M4.2's defaults had never heard of — costs nothing
 * rather than guessing at a ratio for it.
 */
function laterSeconds(
  samples: readonly StageSample[],
  ratios: Readonly<Record<string, number>>,
  projected: number,
): number {
  let sum = 0;
  for (const [stage, ratio] of Object.entries(ratios)) {
    if (stage === MATCHING || stage === BEFORE || !Number.isFinite(ratio) || ratio <= 0) {
      continue;
    }
    if (!finished(samples, stage)) {
      sum += ratio;
    }
  }
  return projected * sum;
}

/** Whether a stage has said it is done. A stage with no samples at all has not. */
function finished(samples: readonly StageSample[], stage: string): boolean {
  return samples.some((sample) => sample.stage === stage && sample.total > 0 && sample.done >= sample.total);
}
