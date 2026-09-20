import { Matrix4 } from "three";

/**
 * The one place a pose from a file becomes a three.js matrix (A §7.2, «Matrix convention»).
 *
 * The repository's README, «Одно соглашение о матрице», fixes it: a file carries `M` **by rows**
 * and a point moves as `p' = M·p` on column vectors. `Matrix4.set()` is the only entry point of
 * three.js that takes its sixteen numbers in that order — `fromArray()` and `.elements` are
 * column-major — so every conversion goes through this function and nowhere else. Read the rows
 * into `elements` directly and the assembly comes out transposed, which looks like a plausible
 * mirrored fit rather than like a bug.
 *
 * The rows come from a run's `assembly.json` over IPC, so they are data from outside: anything
 * that is not four rows of four finite numbers is refused with a `RangeError` instead of quietly
 * producing a matrix full of `NaN` that renders as nothing at all.
 */
export function rowsToMatrix4(rows: readonly (readonly number[])[]): Matrix4 {
  const cells: number[] = [];
  for (const row of rows) {
    if (row.length !== 4) {
      throw new RangeError(`a pose is 4×4; one of its rows has ${String(row.length)} numbers`);
    }
    cells.push(...row);
  }
  if (cells.length !== 16) {
    throw new RangeError(`a pose is 4×4; this one has ${String(rows.length)} rows`);
  }

  /** `cells` is known to hold sixteen numbers; the index type does not know it, and NaN is not one. */
  const n = (index: number): number => {
    const value = cells[index];
    if (value === undefined || !Number.isFinite(value)) {
      throw new RangeError(`a pose is sixteen finite numbers; cell ${String(index)} is ${String(value)}`);
    }
    return value;
  };

  // Row by row, which is exactly the order `set()` reads its arguments in.
  return new Matrix4().set(
    n(0),
    n(1),
    n(2),
    n(3),
    n(4),
    n(5),
    n(6),
    n(7),
    n(8),
    n(9),
    n(10),
    n(11),
    n(12),
    n(13),
    n(14),
    n(15),
  );
}
