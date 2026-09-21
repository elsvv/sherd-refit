import { describe, expect, it } from "vitest";

import type { CandidateRow } from "../../ipc/bindings/CandidateRow";
import type { Decision } from "../../ipc/bindings/Decision";
import type { Evidence } from "../../ipc/bindings/Evidence";
import type { Scores } from "../../ipc/bindings/Scores";
import { candidateOf, limitsOf, nextUndecided, queueRows, rowAt, scoreLines } from "./queue";

/** A pose that is only ever compared, never applied: the translation is what tells two apart. */
function pose(x: number): CandidateRow["pose"] {
  return [
    [1, 0, 0, x],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

/** Every number R §6 produces, with the four the scores table reads set and the rest plausible. */
function scores(partial: Partial<Scores>): Scores {
  return {
    tightA: 0.5,
    tightB: 0.5,
    tight: 0.5,
    gapA: 0.01,
    gapB: 0.01,
    gap: 0.01,
    contactA: 40,
    contactB: 40,
    contact: 40,
    seam: 8,
    gap_limit: 0.031,
    tight_delta: 0.01,
    cont: 0.1,
    cont_n: 0.95,
    pen: 0,
    pen_depth: 0,
    brk: 0.7,
    brk_best: 0.8,
    ...partial,
  };
}

/** The tier pass's evidence, with only the fields the scores table asks about worth naming. */
function evidence(partial: Partial<Evidence>): Evidence {
  return {
    placements: 1,
    resample_tight_min: 0.4,
    resample_gap_max: 0.01,
    resample_accept: 3,
    support: 0,
    ...partial,
  };
}

/** One row of `candidates.json`. */
function row(
  a: string,
  b: string,
  tier: CandidateRow["tier"],
  score: number,
  at = 0,
  extra: Partial<CandidateRow> = {},
): CandidateRow {
  return { a, b, pose: pose(at), score, scores: scores({}), tier, used: false, evidence: null, ...extra };
}

/** A decision about a pair, named the way the caller names it. */
function decided(a: string, b: string, verdict: Decision["verdict"]): Decision {
  return { a, b, verdict, pose: null, source: "probable", bulk: false, at: "2026-09-21T10:00:00Z", carried_from: null };
}

describe("queueRows", () => {
  it("gives one row per pair — the band's best-scoring pose — best first", () => {
    const rows = queueRows(
      [
        row("A", "B", "probable", 4.0, 1),
        row("A", "B", "probable", 9.5, 2),
        row("C", "D", "probable", 7.1, 3),
        row("E", "F", "confirmed", 99, 4),
      ],
      "probable",
      [],
    );
    expect(rows.map((one) => [one.a, one.b, one.best.score])).toEqual([
      ["A", "B", 9.5],
      ["C", "D", 7.1],
    ]);
  });

  it("puts the pair's other poses beside it, best first, whatever band they are in", () => {
    const rows = queueRows(
      [
        row("A", "B", "probable", 9.5, 2),
        row("A", "B", "probable", 4.0, 1),
        row("B", "A", "rejected", 6.2, 3),
      ],
      "probable",
      [],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]?.others.map((one) => one.score)).toEqual([6.2, 4.0]);
  });

  it("joins the verdict in however the decision names the pair", () => {
    const rows = queueRows(
      [row("A", "B", "probable", 9.5), row("C", "D", "probable", 7.1), row("E", "F", "probable", 5)],
      "probable",
      [decided("B", "A", "accept"), decided("C", "D", "reject")],
    );
    expect(rows.map((one) => one.verdict)).toEqual(["accept", "reject", null]);
  });

  it("answers an empty queue for a band nothing is in", () => {
    expect(queueRows([row("A", "B", "probable", 1)], "confirmed", [])).toEqual([]);
  });
});

describe("nextUndecided", () => {
  const rows = queueRows(
    [row("A", "B", "probable", 9), row("C", "D", "probable", 8), row("E", "F", "probable", 7)],
    "probable",
    [decided("C", "D", "accept")],
  );

  it("starts at the first row when nothing is selected yet", () => {
    expect(nextUndecided(rows, -1)).toBe(0);
  });

  it("skips what has been decided", () => {
    expect(nextUndecided(rows, 0)).toBe(2);
  });

  it("wraps round to the top", () => {
    expect(nextUndecided(rows, 2)).toBe(0);
  });

  it("answers −1 when every row has a verdict", () => {
    const all = queueRows([row("A", "B", "probable", 9)], "probable", [decided("A", "B", "reject")]);
    expect(nextUndecided(all, -1)).toBe(-1);
    expect(nextUndecided([], -1)).toBe(-1);
  });
});

describe("rowAt", () => {
  const rows = queueRows([row("A", "B", "probable", 9), row("C", "D", "probable", 8)], "probable", []);

  it("finds a pair whichever way round it is asked for", () => {
    expect(rowAt(rows, "B", "A")).toBe(0);
    expect(rowAt(rows, "C", "D")).toBe(1);
  });

  it("answers −1 for a pair the band does not hold", () => {
    expect(rowAt(rows, "A", "D")).toBe(-1);
  });
});

describe("candidateOf", () => {
  const candidates = [row("A", "B", "probable", 9.5, 2), row("A", "B", "probable", 4, 1)];

  it("finds the candidate whose pose is the one on the screen", () => {
    expect(candidateOf(candidates, "A", "B", pose(1))?.score).toBe(4);
  });

  it("falls back to the pair's best when no pose matches", () => {
    expect(candidateOf(candidates, "B", "A", pose(7))?.score).toBe(9.5);
  });

  it("answers null for a pair that was never scored", () => {
    expect(candidateOf(candidates, "A", "Z", pose(1))).toBeNull();
  });
});

describe("limitsOf", () => {
  it("reads the run's own thresholds and the tier's two arms", () => {
    expect(
      limitsOf({
        min_tight: 0.25,
        max_gap: 0.03,
        min_seam: 3,
        max_pen: 0.005,
        tiers: { min_tight: 0.35, min_margin: 2, min_support: 1 },
      }),
    ).toEqual({ tight: 0.25, gap: 0.03, seam: 3, pen: 0.005, margin: 2, support: 1 });
  });

  it("names no limit a run of an older build did not file", () => {
    expect(limitsOf(null)).toEqual({ tight: null, gap: null, seam: null, pen: null, margin: null, support: null });
    expect(limitsOf({ min_tight: "0.25", tiers: 7 }).tight).toBeNull();
  });
});

describe("scoreLines", () => {
  const limits = limitsOf({
    min_tight: 0.25,
    max_gap: 0.03,
    min_seam: 3,
    max_pen: 0.005,
    tiers: { min_margin: 2, min_support: 1 },
  });

  it("marks a score that meets its limit and one that does not", () => {
    const lines = scoreLines(
      row("A", "B", "probable", 13.9, 0, { scores: scores({ tight: 0.71, seam: 19.7, gap: 0.0047 }) }),
      limits,
    );
    const by = new Map(lines.map((line) => [line.key, line]));
    expect(by.get("score")).toEqual({ key: "score", value: "13.9", limit: null, ok: null });
    expect(by.get("seam")).toEqual({ key: "seam", value: "19.7 t", limit: "≥ 3", ok: true });
    expect(by.get("tight")).toEqual({ key: "tight", value: "0.71", limit: "≥ 0.25", ok: true });
    // The pair's own gap limit, not the run's: R §1.2 scales it to the two walls.
    expect(by.get("gap")).toEqual({ key: "gap", value: "0.0047 t", limit: "≤ 0.031", ok: true });
  });

  it("marks a contact under the limit amber", () => {
    const lines = scoreLines(row("A", "B", "probable", 1, 0, { scores: scores({ tight: 0.11 }) }), limits);
    expect(lines.find((line) => line.key === "tight")?.ok).toBe(false);
  });

  it("does not call a penetration that could not be measured a pass", () => {
    const lines = scoreLines(
      row("A", "B", "probable", 1, 0, { scores: scores({ pen: 0, pen_unavailable: 1 }) }),
      limits,
    );
    expect(lines.find((line) => line.key === "pen")).toEqual({ key: "pen", value: "—", limit: "≤ 0.005", ok: false });
  });

  it("shows the two arms when the tier probed them, and nothing when it did not", () => {
    const probed = scoreLines(
      row("A", "B", "probable", 1, 0, { evidence: evidence({ margin: 1.4, support: 0 }) }),
      limits,
    );
    const by = new Map(probed.map((line) => [line.key, line]));
    expect(by.get("margin")).toEqual({ key: "margin", value: "1.4×", limit: "≥ 2×", ok: false });
    expect(by.get("support")).toEqual({ key: "support", value: "0", limit: "≥ 1", ok: false });

    const unprobed = scoreLines(row("A", "B", "rejected", 1), limits);
    expect(unprobed.some((line) => line.key === "margin" || line.key === "support")).toBe(false);
  });

  it("says nothing about a limit the run did not file", () => {
    const lines = scoreLines(row("A", "B", "probable", 1), limitsOf(null));
    expect(lines.find((line) => line.key === "seam")).toEqual({ key: "seam", value: "8.0 t", limit: null, ok: null });
  });
});
