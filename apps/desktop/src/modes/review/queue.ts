import type { CandidateRow } from "../../ipc/bindings/CandidateRow";
import type { Decision } from "../../ipc/bindings/Decision";
import type { Verdict } from "../../ipc/bindings/Verdict";
import { decisionOf } from "../../state/review";

/**
 * What the «Ревью» mode *is*, as opposed to what it looks like (A §8.3): the queue of pairs to go
 * through, which of them still wants a decision, and every number of a candidate put beside the
 * limit the run asked of it.
 *
 * Pure and tested on its own, for the same reason `tree.ts` is: A §11 rules out a component-test
 * harness, not a test, and these four questions — «is this one pair or two rows?», «where does
 * `A` go next?», «is 0.0047 t a pass?» — are exactly the ones a screenshot cannot answer.
 */

/** The three bands of `candidates.json`, in the order A §8.3's filter chips read them. */
export type Band = CandidateRow["tier"];
export const BANDS: readonly Band[] = ["probable", "confirmed", "rejected"];

/** One line of the queue: a **pair**, the pose it is shown in, and what was decided about it. */
export interface QueueRow {
  /** The fragment the pose maps into, as the shown candidate names it. */
  a: string;
  /** The one it moves. */
  b: string;
  /** The band's best-scoring candidate of this pair — what the centre and the inspector show. */
  best: CandidateRow;
  /** A §8.3's «Другие позы пары», best first: every other candidate of the pair, any band. */
  others: CandidateRow[];
  /** `null` while nobody has said anything about the pair. */
  verdict: Verdict | null;
}

/** A pair's identity, unordered — the key A §8.1 decides by, so `A–B` and `B–A` are one row. */
function pairKey(a: string, b: string): string {
  return a < b ? `${a}\u0000${b}` : `${b}\u0000${a}`;
}

/**
 * A pose as one string, for «is this the placement on screen?». Compared and never parsed: the
 * numbers come from one `candidates.json` read once, so two candidates of a pair either are the
 * same object or were written differently.
 */
export function poseKey(pose: CandidateRow["pose"]): string {
  return JSON.stringify(pose);
}

/**
 * The queue of one band (A §8.3), best score first.
 *
 * **One row per pair, not per candidate.** A pair routinely leaves three or four poses in
 * `candidates.json` that differ by a millimetre, and a queue that listed them separately would
 * ask the reviewer the same question four times — so the band's best-scoring pose stands for the
 * pair and the rest become the inspector's «Другие позы». `a` and `b` are that candidate's own,
 * because its pose is read against them.
 */
export function queueRows(
  candidates: readonly CandidateRow[],
  band: Band,
  decisions: readonly Decision[],
): QueueRow[] {
  const byPair = new Map<string, CandidateRow[]>();
  for (const candidate of candidates) {
    const key = pairKey(candidate.a, candidate.b);
    const seen = byPair.get(key);
    if (seen === undefined) {
      byPair.set(key, [candidate]);
    } else {
      seen.push(candidate);
    }
  }

  const best = new Map<string, CandidateRow>();
  for (const candidate of candidates) {
    if (candidate.tier !== band) {
      continue;
    }
    const key = pairKey(candidate.a, candidate.b);
    const seen = best.get(key);
    if (seen === undefined || candidate.score > seen.score) {
      best.set(key, candidate);
    }
  }

  const rows: QueueRow[] = [];
  for (const [key, candidate] of best) {
    rows.push({
      a: candidate.a,
      b: candidate.b,
      best: candidate,
      others: (byPair.get(key) ?? [])
        .filter((other) => other !== candidate)
        .sort((one, other) => other.score - one.score),
      verdict: decisionOf(decisions, candidate.a, candidate.b)?.verdict ?? null,
    });
  }
  return rows.sort((one, other) => other.best.score - one.best.score);
}

/**
 * Where the selection goes after a verdict (A §8.3: «the selection moves to the next undecided
 * row»), as an index into the same rows; −1 when the band is done.
 *
 * It wraps: a reviewer who starts in the middle of the queue, or who comes back to change their
 * mind about a row near the end, is still owed the rows above it. `from` of −1 is «nothing is
 * selected», which starts at the top.
 */
export function nextUndecided(rows: readonly QueueRow[], from: number): number {
  const count = rows.length;
  if (count === 0) {
    return -1;
  }
  for (let step = 1; step <= count; step += 1) {
    const at = (((from + step) % count) + count) % count;
    if (rows[at]?.verdict === null) {
      return at;
    }
  }
  return -1;
}

/** Where a pair stands in the queue, however it is named, or −1 when this band has not got it. */
export function rowAt(rows: readonly QueueRow[], a: string, b: string): number {
  const key = pairKey(a, b);
  return rows.findIndex((row) => pairKey(row.a, row.b) === key);
}

/**
 * The candidate the centre pane is showing: the pair's candidate whose pose is the selected one
 * («Другие позы · показать» picks one of several), falling back to the pair's best.
 *
 * The selection carries a pose and not an index because a decision is about a *pose* (A §8.1)
 * and the run's rows may be reloaded under it; matching on the pose is what keeps «Подтвердить»
 * pinning the placement the reviewer is looking at.
 */
export function candidateOf(
  candidates: readonly CandidateRow[],
  a: string,
  b: string,
  pose: CandidateRow["pose"],
): CandidateRow | null {
  const key = pairKey(a, b);
  const mine = candidates.filter((row) => pairKey(row.a, row.b) === key);
  const wanted = poseKey(pose);
  return mine.find((row) => poseKey(row.pose) === wanted) ?? mine.sort((one, other) => other.score - one.score)[0] ?? null;
}

/** What the run asked of a candidate, out of its own `params` (A §8.3's second column). */
export interface Limits {
  /** R §6.5's tight-contact fraction. */
  tight: number | null;
  /** R §6.5's median gap, in `t` — the floor under a pair's own `gap_limit`. */
  gap: number | null;
  /** R §6.5's shortest shared seam, in `t`. */
  seam: number | null;
  /** R §6.5's penetration fraction. */
  pen: number | null;
  /** The tier's margin over a second placement, when the run computed a tier at all. */
  margin: number | null;
  /** And the independent agreeing joins it wants. */
  support: number | null;
}

/** An object, for a value that came out of a file and is not to be trusted to be one. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** A number that is a number, or `null` — `run.json`'s `params` is `Value` and never validated. */
function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * The limits a run resolved to, read back out of its `run.json` (`Params`, and `Params::tiers`
 * for the two arms). A run of an older build, or one that died before it could file them, names
 * no limits — and the scores are then shown as measurements with nothing to fail.
 */
export function limitsOf(params: Record<string, unknown> | null): Limits {
  const tiers = params !== null && isRecord(params.tiers) ? params.tiers : null;
  return {
    tight: params === null ? null : finite(params.min_tight),
    gap: params === null ? null : finite(params.max_gap),
    seam: params === null ? null : finite(params.min_seam),
    pen: params === null ? null : finite(params.max_pen),
    margin: tiers === null ? null : finite(tiers.min_margin),
    support: tiers === null ? null : finite(tiers.min_support),
  };
}

/** The rows of A §8.3's scores table, in the order the mock-up lists them. */
export type ScoreKey = "score" | "seam" | "tight" | "gap" | "pen" | "margin" | "support";

/** One of them: what was measured, what was asked, and whether the one meets the other. */
export interface ScoreLine {
  key: ScoreKey;
  /** Already formatted, with its unit where it has one; `—` for a number nobody measured. */
  value: string;
  /** `≥ 3`, `≤ 0.031`; `null` when the run filed no limit for it, or when it has none. */
  limit: string | null;
  /** `null` where there is nothing to meet — the score itself, and an unfiled limit. */
  ok: boolean | null;
}

/** A number nobody measured. Punctuation, identical in both languages (as `format.ts` has it). */
const UNKNOWN = "—";

/** A measurement at the precision it is worth reading at, or `—`. */
function fixed(value: number, decimals: number): string {
  return Number.isFinite(value) ? value.toFixed(decimals) : UNKNOWN;
}

/**
 * A limit as the run holds it — `0.35`, `3`, `0.005` — and not at the measurement's precision: a
 * threshold printed `3.0000` reads as something that was measured.
 */
function held(value: number): string {
  return String(Number(value.toFixed(6)));
}

/** One row, with the comparison done once so that the sentence and the colour cannot disagree. */
function line(key: ScoreKey, value: string, limit: number | null, mark: string, ok: boolean): ScoreLine {
  return limit === null ? { key, value, limit: null, ok: null } : { key, value, limit: `${mark} ${held(limit)}`, ok };
}

/**
 * Every number A §8.3 puts against its limit, for the candidate on the screen.
 *
 * The **gap** is read against the pair's own `gap_limit` and not against the run's `max_gap`:
 * R §1.2 scales the limit to the two walls, so a pair of thin sherds is judged more finely than
 * the run's floor, and showing the floor would call a refusal a pass. The two arms are only
 * there when the tier pass probed this candidate — a run with `tiers` off never asked.
 */
export function scoreLines(row: CandidateRow, limits: Limits): ScoreLine[] {
  const scores = row.scores;
  const gapLimit = Number.isFinite(scores.gap_limit) ? scores.gap_limit : limits.gap;
  const lines: ScoreLine[] = [
    { key: "score", value: fixed(row.score, 1), limit: null, ok: null },
    line("seam", `${fixed(scores.seam, 1)} t`, limits.seam, "≥", scores.seam >= (limits.seam ?? 0)),
    line("tight", fixed(scores.tight, 2), limits.tight, "≥", scores.tight >= (limits.tight ?? 0)),
    line("gap", `${fixed(scores.gap, 4)} t`, gapLimit, "≤", scores.gap <= (gapLimit ?? 0)),
    // R §6.4 could not run on an open mesh, and a blank is the honest reading: the join is held
    // back for it all the same, so it is never a pass (`Thresholds::refusals`).
    scores.pen_unavailable === undefined
      ? line("pen", fixed(scores.pen, 4), limits.pen, "≤", scores.pen <= (limits.pen ?? 0))
      : line("pen", UNKNOWN, limits.pen, "≤", false),
  ];

  const evidence = row.evidence;
  if (evidence === null) {
    return lines;
  }
  if (evidence.margin !== undefined) {
    const margin = evidence.margin;
    lines.push(
      limits.margin === null
        ? { key: "margin", value: `${fixed(margin, 1)}×`, limit: null, ok: null }
        : { key: "margin", value: `${fixed(margin, 1)}×`, limit: `≥ ${held(limits.margin)}×`, ok: margin >= limits.margin },
    );
  }
  lines.push(line("support", String(evidence.support), limits.support, "≥", evidence.support >= (limits.support ?? 0)));
  return lines;
}
