import { describe, expect, it } from "vitest";
import { principalAxes } from "./principal";

/** A thin plate: long along `u`, shorter along `v`, a hair thick along `u × v`. */
function plate(u: [number, number, number], v: [number, number, number]): number[] {
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]] as const;
  const out: number[] = [];
  for (let i = -20; i <= 20; i += 1) {
    for (let j = -10; j <= 10; j += 1) {
      for (const k of [-0.2, 0.2]) {
        out.push(
          500 + i * u[0] + j * v[0] + k * n[0],
          -300 + i * u[1] + j * v[1] + k * n[1],
          40 + i * u[2] + j * v[2] + k * n[2],
        );
      }
    }
  }
  return out;
}

const dot = (a: readonly number[], b: readonly number[]): number =>
  (a[0] ?? 0) * (b[0] ?? 0) + (a[1] ?? 0) * (b[1] ?? 0) + (a[2] ?? 0) * (b[2] ?? 0);

describe("principalAxes", () => {
  it("finds the wall's direction of a plate lying in XY, wherever the plate is", () => {
    const axes = principalAxes(plate([1, 0, 0], [0, 1, 0]));
    expect(axes).not.toBeNull();
    expect(Math.abs(dot(axes?.normal ?? [0, 0, 0], [0, 0, 1]))).toBeCloseTo(1, 6);
    expect(Math.abs(dot(axes?.major ?? [0, 0, 0], [1, 0, 0]))).toBeCloseTo(1, 6);
  });

  it("finds it for a plate that is not aligned with anything", () => {
    const s = Math.SQRT1_2;
    const axes = principalAxes(plate([s, s, 0], [0, 0, 1]));
    expect(Math.abs(dot(axes?.normal ?? [0, 0, 0], [s, -s, 0]))).toBeCloseTo(1, 6);
    expect(Math.abs(dot(axes?.major ?? [0, 0, 0], [s, s, 0]))).toBeCloseTo(1, 6);
  });

  it("has no opinion about fewer than three points, or about NaN", () => {
    expect(principalAxes([1, 2, 3, 4, 5, 6])).toBeNull();
    expect(principalAxes([NaN, 0, 0, NaN, 1, 0, NaN, 0, 1])).toBeNull();
  });
});
