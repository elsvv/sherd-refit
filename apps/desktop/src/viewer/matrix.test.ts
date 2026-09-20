import { describe, expect, it } from "vitest";
import { Vector3 } from "three";
import { rowsToMatrix4 } from "./matrix";

// README, «Одно соглашение о матрице»: files carry M by rows, p' = M·p on column vectors.
// THREE.Matrix4.set() takes rows; .fromArray() and .elements are column-major. This is the one
// place the two meet, and the classic place to get a mirrored assembly.
describe("rowsToMatrix4", () => {
  const rows = [
    [0, -1, 0, 10],
    [1, 0, 0, 20],
    [0, 0, 1, 30],
    [0, 0, 0, 1],
  ];
  it("moves a point as M·p does", () => {
    const p = new Vector3(1, 2, 3).applyMatrix4(rowsToMatrix4(rows));
    expect([p.x, p.y, p.z]).toEqual([8, 21, 33]); // (0·1 − 1·2 + 10, 1·1 + 20, 3 + 30)
  });
  it("keeps the translation in the fourth column", () => {
    expect(Array.from(rowsToMatrix4(rows).elements.slice(12, 15))).toEqual([10, 20, 30]);
  });
  it("refuses what is not 4×4", () => {
    expect(() => rowsToMatrix4([[1, 0, 0, 0]])).toThrow(RangeError);
  });
});
