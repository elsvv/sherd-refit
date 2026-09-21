import { describe, expect, it } from "vitest";

import type { StageSample } from "./eta";
import { remainingSeconds } from "./eta";

/** The pairs of a collection the size of `karas`, which is what A §6's example is about. */
const TOTAL = 11_000;

/**
 * One `matching` sample a second: `seconds + 1` of them, from `done` at `at`, going at `rate`
 * pairs a second. That is what the shell's progress events look like once `jobs.ts` has thinned
 * them to one a second.
 */
function matching(at: number, done: number, seconds: number, rate: number): StageSample[] {
  return Array.from({ length: seconds + 1 }, (_, i) => ({
    stage: "matching",
    done: done + i * rate,
    total: TOTAL,
    at: at + i * 1000,
  }));
}

describe("remainingSeconds (A §6)", () => {
  it("says nothing before matching, before 5 s and before 1 %", () => {
    const preprocess: StageSample[] = [{ stage: "preprocess", done: 12, total: 12, at: 0 }];
    expect(remainingSeconds(preprocess, 10_000, {})).toBeNull();
    // Three seconds of matching: a rate measured over that says more about the first tile than
    // about the run.
    expect(remainingSeconds(matching(0, 0, 3, 10), 3_000, {})).toBeNull();
    // Ten seconds, but 100 pairs of 11 000 — under the one per cent A §6 waits for.
    expect(remainingSeconds(matching(0, 0, 10, 10), 10_000, {})).toBeNull();
  });

  it("extrapolates matching and adds the later stages at their ratios", () => {
    // 1 000 pairs in 100 s → 10 pairs/s; 10 000 left → 1 000 s, and the whole of matching
    // projected at 1 100 s, of which `tiers` and `refine` are 5 %.
    const left = remainingSeconds(matching(0, 0, 100, 10), 100_000, { tiers: 0.03, refine: 0.02 });
    expect(left).toBeCloseTo(1_000 + 1_100 * 0.05, 0);
  });

  it("goes by the recent rate and not by the average", () => {
    const samples = [...matching(0, 0, 100, 10), ...matching(101_000, 1_005, 59, 5)];
    // 1 300 done at 160 s: the average is 8.1 pairs/s (1 193 s left), the last minute is 5.
    const left = remainingSeconds(samples, 160_000, {});
    expect(left).toBeCloseTo((TOTAL - 1_300) / 5, 0);
  });

  it("counts down the later stages once matching is over, and never below zero", () => {
    const samples: StageSample[] = [
      ...matching(0, 0, 1_100, 10),
      { stage: "tiers", done: 40, total: 40, at: 1_101_000 },
      { stage: "refine", done: 3, total: 50, at: 1_102_000 },
    ];
    // Matching ended at 1 100 s having been projected at 1 100 s; `tiers` has reported itself
    // finished, so only `refine`'s 2 % — 22 s — is left, less the 5 s gone since.
    expect(remainingSeconds(samples, 1_105_000, { tiers: 0.03, refine: 0.02 })).toBeCloseTo(17, 0);
    expect(remainingSeconds(samples, 1_200_000, { tiers: 0.03, refine: 0.02 })).toBe(0);
  });

  it("charges nothing for a stage the calibration has never measured", () => {
    const samples = [...matching(0, 0, 100, 10), { stage: "objects", done: 1, total: 9, at: 100_000 }];
    const left = remainingSeconds(samples, 100_000, { tiers: 0.03, refine: 0.02 });
    expect(left).toBeCloseTo(1_000 + 1_100 * 0.05, 0);
  });
});
