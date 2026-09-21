import { describe, expect, it } from "vitest";

import type { Disc, Placed } from "./layout";
import { packDiscs } from "./layout";

/**
 * Seventeen groups the size a real collection gives them: one big vessel, a few middling ones and
 * a tail of pairs. The radii are the bounding spheres of the assembled groups in millimetres,
 * which is the unit everything in the viewer is in.
 */
const MIXED: Disc[] = [
  { id: 0, radius: 210 },
  { id: 1, radius: 160 },
  { id: 2, radius: 145 },
  { id: 3, radius: 120 },
  { id: 4, radius: 118 },
  { id: 5, radius: 96 },
  { id: 6, radius: 90 },
  { id: 7, radius: 74 },
  { id: 8, radius: 70 },
  { id: 9, radius: 66 },
  { id: 10, radius: 61 },
  { id: 11, radius: 58 },
  { id: 12, radius: 52 },
  { id: 13, radius: 47 },
  { id: 14, radius: 44 },
  { id: 15, radius: 41 },
  { id: 16, radius: 38 },
];

const GAP = 31.5; // 15 % of the largest radius, which is what the viewer asks for.

/** The box the *discs* take up — not the boxes they were packed in, which carry the gap. */
function extent(placed: readonly Placed[], discs: readonly Disc[]): { w: number; h: number; cx: number; cy: number } {
  const radius = new Map(discs.map((disc) => [disc.id, disc.radius]));
  const xs = placed.map((p) => [p.x - (radius.get(p.id) ?? 0), p.x + (radius.get(p.id) ?? 0)]).flat();
  const ys = placed.map((p) => [p.y - (radius.get(p.id) ?? 0), p.y + (radius.get(p.id) ?? 0)]).flat();
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  return { w: x1 - x0, h: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
}

describe("packDiscs", () => {
  it("leaves the gap between every two groups", () => {
    const placed = packDiscs(MIXED, GAP);
    const radius = new Map(MIXED.map((disc) => [disc.id, disc.radius]));
    for (const a of placed) {
      for (const b of placed) {
        if (a.id >= b.id) {
          continue;
        }
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        const want = (radius.get(a.id) ?? 0) + (radius.get(b.id) ?? 0) + GAP;
        // Touching boxes put two discs exactly `gap` apart, so the comparison is not strict.
        expect(distance).toBeGreaterThanOrEqual(want - 1e-9);
      }
    }
  });

  it("centres the block on the origin and keeps it about as wide as asked", () => {
    const aspect = 1.5;
    const { w, h, cx, cy } = extent(packDiscs(MIXED, GAP, aspect), MIXED);
    expect(Math.abs(cx)).toBeLessThan(1e-6);
    expect(Math.abs(cy)).toBeLessThan(1e-6);
    // A block of seventeen discs of very different sizes cannot hit a ratio exactly; A §7.2 only
    // needs a block that reads as a block rather than as a queue.
    expect(w / h).toBeGreaterThan(aspect * 0.6);
    expect(w / h).toBeLessThan(aspect * 1.4);
  });

  it("puts a single group at the origin, and answers nothing for nothing", () => {
    expect(packDiscs([{ id: 7, radius: 90 }], GAP)).toEqual([{ id: 7, x: 0, y: 0 }]);
    expect(packDiscs([], GAP)).toEqual([]);
  });

  it("answers in the order it was asked, whatever order it packed in", () => {
    // Smallest first on the way in: the packing takes the biggest first and must not say so.
    const ascending = [...MIXED].reverse();
    const placed = packDiscs(ascending, GAP);
    expect(placed.map((p) => p.id)).toEqual(ascending.map((disc) => disc.id));
  });

  it("gives the same answer twice", () => {
    expect(packDiscs(MIXED, GAP)).toEqual(packDiscs(MIXED, GAP));
  });
});
