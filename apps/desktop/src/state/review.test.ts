import { describe, expect, it } from "vitest";

import type { CandidateRow } from "../ipc/bindings/CandidateRow";
import type { Decision } from "../ipc/bindings/Decision";
import { accepted, applyDecision, bulkAccept, clearDecision, decisionOf, recorded, redone, rejected, undone } from "./review";
import type { Undo } from "./review";

/** An identity pose, which is all these tests need of one. */
const POSE: CandidateRow["pose"] = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

/** A candidate row of `tier`, scoring `score`; only the pair, the band and the score are read. */
function row(a: string, b: string, tier: CandidateRow["tier"], score: number): CandidateRow {
  return {
    a,
    b,
    pose: POSE,
    score,
    scores: {
      tightA: 0.31,
      tightB: 0.28,
      tight: 0.28,
      gapA: 0.01,
      gapB: 0.012,
      gap: 0.012,
      contactA: 37,
      contactB: 34,
      contact: 34,
      seam: 19.7,
      gap_limit: 0.031,
      tight_delta: 0.01,
      cont: 0.1,
      cont_n: 0.96,
      pen: 0,
      pen_depth: 0,
      brk: 0.62,
      brk_best: 0.78,
    },
    tier,
    used: false,
    evidence: null,
  };
}

const AT = "2026-09-21T12:00:00Z";
const LATER = "2026-09-21T12:00:09Z";

/** The three fields undo and redo move between, starting from `decisions`. */
function history(decisions: Decision[] = []): Undo {
  return { decisions, past: [], future: [] };
}

describe("applyDecision (A §8.1: the key is the unordered pair)", () => {
  it("replaces an earlier decision on the same pair, whichever way round it is named", () => {
    const first = applyDecision([], rejected(row("FY234111", "FY234058", "probable", 8), AT));
    const second = applyDecision(first, accepted(row("FY234058", "FY234111", "probable", 8), LATER));
    expect(second).toHaveLength(1);
    expect(second[0]?.verdict).toBe("accept");
    // The names stay as the decision that won names them: it carries that candidate's pose, and
    // a pose read against the other order would be its inverse.
    expect(second[0]?.a).toBe("FY234058");
    expect(second[0]?.at).toBe(LATER);
  });

  it("keeps the other pairs, and keeps them in the order they were decided", () => {
    let list = applyDecision([], accepted(row("A", "B", "probable", 8), AT));
    list = applyDecision(list, rejected(row("C", "D", "confirmed", 7), AT));
    list = applyDecision(list, rejected(row("B", "A", "probable", 8), LATER));
    expect(list.map((d) => `${d.a}-${d.b}:${d.verdict}`)).toEqual(["B-A:reject", "C-D:reject"]);
  });

  it("carries the candidate's pose on an accept and none on a reject (A §8.1)", () => {
    expect(accepted(row("A", "B", "probable", 8), AT).pose).toEqual(POSE);
    expect(rejected(row("A", "B", "confirmed", 8), AT).pose).toBeNull();
    expect(accepted(row("A", "B", "probable", 8), AT).source).toBe("probable");
  });

  it("finds and clears a pair however it is named", () => {
    const list = applyDecision([], accepted(row("A", "B", "probable", 8), AT));
    expect(decisionOf(list, "B", "A")?.verdict).toBe("accept");
    expect(clearDecision(list, "B", "A")).toEqual([]);
    expect(clearDecision(list, "A", "C")).toEqual(list);
  });
});

describe("bulkAccept (A §8.4: «Принять все оставшиеся вероятные»)", () => {
  const rows = [
    row("A", "B", "probable", 9),
    row("C", "D", "probable", 8),
    row("E", "F", "confirmed", 12),
    row("G", "H", "rejected", 1),
  ];

  it("accepts every undecided probable pair, and only those, flagged bulk", () => {
    const list = bulkAccept([], rows, AT);
    expect(list.map((d) => `${d.a}-${d.b}`)).toEqual(["A-B", "C-D"]);
    expect(list.every((d) => d.bulk && d.verdict === "accept")).toBe(true);
  });

  it("skips a pair that is already decided, whichever way round that decision named it", () => {
    const decided = applyDecision([], rejected(row("B", "A", "probable", 9), AT));
    const list = bulkAccept(decided, rows, LATER);
    expect(list.map((d) => `${d.a}-${d.b}:${d.verdict}`)).toEqual(["B-A:reject", "C-D:accept"]);
  });

  it("takes one pose per pair — the first row given, which the queue sorts best first", () => {
    const twice = [row("A", "B", "probable", 9), row("A", "B", "probable", 4)];
    expect(bulkAccept([], twice, AT)).toHaveLength(1);
  });

  it("is one step, so one undo takes the whole expansion back", () => {
    const before = history();
    const after = recorded(before, bulkAccept(before.decisions, rows, AT));
    expect(after.decisions).toHaveLength(2);
    expect(undone(after).decisions).toEqual([]);
  });
});

describe("the undo stack (A §8.1: the history is the window's)", () => {
  const one = accepted(row("A", "B", "probable", 9), AT);
  const two = rejected(row("C", "D", "probable", 8), AT);

  it("goes back and forward over every change and lands on what it started from", () => {
    let state = history();
    state = recorded(state, applyDecision(state.decisions, one));
    state = recorded(state, applyDecision(state.decisions, two));
    expect(state.decisions).toHaveLength(2);

    state = undone(state);
    expect(state.decisions).toEqual([one]);
    state = undone(state);
    expect(state.decisions).toEqual([]);
    expect(state.past).toEqual([]);

    state = redone(state);
    expect(state.decisions).toEqual([one]);
    state = redone(state);
    expect(state.decisions).toEqual([one, two]);
    expect(state.future).toEqual([]);
  });

  it("does nothing at either end, rather than throwing on an empty stack", () => {
    const empty = history();
    expect(undone(empty)).toEqual(empty);
    expect(redone(empty)).toEqual(empty);
  });

  it("throws the redo away as soon as something else is decided", () => {
    let state = recorded(history(), [one]);
    state = undone(state);
    expect(state.future).toHaveLength(1);
    state = recorded(state, [two]);
    expect(state.future).toEqual([]);
    expect(state.decisions).toEqual([two]);
  });

  it("makes «Сбросить решения» one undoable step like any other", () => {
    const state = recorded(history([one, two]), []);
    expect(state.decisions).toEqual([]);
    expect(undone(state).decisions).toEqual([one, two]);
  });
});
