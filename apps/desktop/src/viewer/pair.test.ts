import { describe, expect, it } from "vitest";
import { Vector3 } from "three";

import { rowsToMatrix4 } from "./matrix";
import { contactColour, ghostMatrix, separationOffset } from "./pair";

/**
 * The two places A §8.3's pair can be wrong without looking wrong (see `pair.ts`): a ghost at
 * `pose` where `pose⁻¹` was meant, and a separation along the wrong line. Both are checked
 * against a transform whose answer can be worked out by hand.
 */

/** A quarter turn about Z and a hundred along X — `p' = M·p` on column vectors, by rows. */
const ANCHOR = [
  [0, -1, 0, 100],
  [1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

/** And a pose that only shifts, so that the two can be told apart in the answer. */
const POSE = [
  [1, 0, 0, 10],
  [0, 1, 0, 20],
  [0, 0, 1, 30],
  [0, 0, 0, 1],
];

const at = (x: number, y: number, z: number): [number, number, number] => [x, y, z];
const moved = (point: Vector3): [number, number, number] => [point.x, point.y, point.z];

describe("ghostMatrix", () => {
  it("puts b's own frame where the anchor's frame puts it", () => {
    const where = ghostMatrix(rowsToMatrix4(ANCHOR), POSE, false);
    // pose·(1,2,3) = (11,22,33); anchor turns that by 90° about Z and slides it 100 along X.
    expect(moved(new Vector3(1, 2, 3).applyMatrix4(where))).toEqual(at(78, 11, 33));
  });

  it("goes the other way round when the candidate names the fragments the other way round", () => {
    const flipped = ghostMatrix(rowsToMatrix4(ANCHOR), POSE, true);
    // The anchor is now the fragment the pose maps *into*, so a point already in b's pose comes
    // back to where the anchor's own frame has it: anchor·(1,2,3), not anchor·pose·(1,2,3).
    const inPose = new Vector3(1, 2, 3).applyMatrix4(rowsToMatrix4(POSE));
    expect(moved(inPose.applyMatrix4(flipped))).toEqual(at(98, 1, 3));
  });

  it("leaves the anchor it was given alone", () => {
    const anchor = rowsToMatrix4(ANCHOR);
    ghostMatrix(anchor, POSE, true);
    expect(Array.from(anchor.elements)).toEqual(Array.from(rowsToMatrix4(ANCHOR).elements));
  });

  it("refuses a pose that is not sixteen finite numbers", () => {
    expect(() => ghostMatrix(rowsToMatrix4(ANCHOR), [[1, 0, 0, 0]], false)).toThrow(RangeError);
  });

  it("refuses to invert a pose that has no inverse, and draws it happily the other way", () => {
    // Every rotation column zero: three.js answers `invert()` with sixteen zeroes rather than an
    // error, which would put the ghost on the anchor's origin — a place, and a wrong one.
    const flat = [
      [0, 0, 0, 1],
      [0, 0, 0, 2],
      [0, 0, 0, 3],
      [0, 0, 0, 1],
    ];
    expect(() => ghostMatrix(rowsToMatrix4(ANCHOR), flat, true)).toThrow(RangeError);
    expect(() => ghostMatrix(rowsToMatrix4(ANCHOR), flat, false)).not.toThrow();
  });
});

describe("separationOffset", () => {
  const from = new Vector3(0, 0, 0);
  const to = new Vector3(30, 40, 0); // fifty apart
  const SPAN = 80; // how big a piece of this pair is — the radius of the larger of the two

  it("pushes along the line between the centroids, by that fraction of the pair's own size", () => {
    expect(moved(separationOffset(from, to, SPAN, 0.5))).toEqual(at(24, 32, 0));
    expect(separationOffset(from, to, SPAN, 0.5).length()).toBeCloseTo(40, 10);
  });

  it("measures the push by the pieces and not by how far the pose happens to put them", () => {
    // Twice as far apart, same two pieces: the same slider opens the same gap, in the same
    // direction — a candidate whose pose is nowhere near must not fling B off the screen.
    const far = new Vector3(60, 80, 0);
    expect(moved(separationOffset(from, far, SPAN, 0.5))).toEqual(moved(separationOffset(from, to, SPAN, 0.5)));
  });

  it("opens a gap the size of a piece at one, and none at zero", () => {
    expect(separationOffset(from, to, SPAN, 1).length()).toBeCloseTo(SPAN, 10);
    expect(moved(separationOffset(from, to, SPAN, 0))).toEqual(at(0, 0, 0));
  });

  it("clamps a slider that went past either end, and refuses what is not a number", () => {
    expect(separationOffset(from, to, SPAN, 4).length()).toBeCloseTo(SPAN, 10);
    expect(moved(separationOffset(from, to, SPAN, -2))).toEqual(at(0, 0, 0));
    expect(moved(separationOffset(from, to, SPAN, Number.NaN))).toEqual(at(0, 0, 0));
  });

  it("pushes nothing when the two centroids are in the same place, or the pair has no size", () => {
    expect(moved(separationOffset(from, new Vector3(0, 0, 0), SPAN, 1))).toEqual(at(0, 0, 0));
    expect(moved(separationOffset(from, to, 0, 1))).toEqual(at(0, 0, 0));
  });

  it("leaves both centroids alone", () => {
    separationOffset(from, to, SPAN, 1);
    expect([moved(from), moved(to)]).toEqual([at(0, 0, 0), at(30, 40, 0)]);
  });
});

describe("contactColour", () => {
  it("gives R §6.1's three classes the three tokens the legend's dots wear", () => {
    expect([0, 1, 2].map((cls) => contactColour(cls).token)).toEqual(["--ok", "--warn", "--danger"]);
  });

  it("counts a class it has no colour for as «beyond», never as «tight»", () => {
    for (const cls of [3, -1, 255, Number.NaN]) {
      expect(contactColour(cls).token).toBe("--danger");
    }
  });
});
