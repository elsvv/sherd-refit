/**
 * Which way a fragment faces.
 *
 * A sherd is a thin shell: two of its principal axes span its face and the third — the direction
 * its points vary least along — goes through the wall. A camera placed by a fixed direction meets
 * most sherds edge-on, which is the one view that tells a conservator nothing; the engine's own
 * previews and thumbnails are framed by the principal axes for the same reason
 * (`sherd_core::render::principal_views`).
 */

/** A direction, or a point. */
export type Vec3 = readonly [number, number, number];

/** A fragment's principal axes, unit length: `major` along its longest side, `normal` through its wall. */
export interface PrincipalAxes {
  major: Vec3;
  normal: Vec3;
}

/** More vertices than this are strided over: the axes of 5 000 points are the axes of 60 000. */
const SAMPLE = 5000;

/** Cyclic Jacobi sweeps; a symmetric 3×3 is diagonal to machine precision after five or six. */
const SWEEPS = 12;

/**
 * The principal axes of a vertex buffer (`x, y, z, x, y, z, …`), or `null` for fewer than three
 * finite points — a caller then keeps whatever default it had.
 */
export function principalAxes(positions: ArrayLike<number>): PrincipalAxes | null {
  const count = Math.floor(positions.length / 3);
  const stride = Math.max(1, Math.floor(count / SAMPLE));
  const mean = [0, 0, 0];
  let used = 0;
  for (let i = 0; i < count; i += stride) {
    const [x, y, z] = [positions[3 * i] ?? NaN, positions[3 * i + 1] ?? NaN, positions[3 * i + 2] ?? NaN];
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      mean[0] = (mean[0] ?? 0) + x;
      mean[1] = (mean[1] ?? 0) + y;
      mean[2] = (mean[2] ?? 0) + z;
      used += 1;
    }
  }
  if (used < 3) {
    return null;
  }
  const [mx, my, mz] = [(mean[0] ?? 0) / used, (mean[1] ?? 0) / used, (mean[2] ?? 0) / used];

  // The covariance, upper triangle: xx xy xz yy yz zz.
  let [xx, xy, xz, yy, yz, zz] = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < count; i += stride) {
    const x = (positions[3 * i] ?? NaN) - mx;
    const y = (positions[3 * i + 1] ?? NaN) - my;
    const z = (positions[3 * i + 2] ?? NaN) - mz;
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      xx += x * x;
      xy += x * y;
      xz += x * z;
      yy += y * y;
      yz += y * z;
      zz += z * z;
    }
  }

  const a = [
    [xx, xy, xz],
    [xy, yy, yz],
    [xz, yz, zz],
  ];
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const at = (m: number[][], r: number, c: number): number => m[r]?.[c] ?? 0;
  const put = (m: number[][], r: number, c: number, value: number): void => {
    const row = m[r];
    if (row !== undefined) {
      row[c] = value;
    }
  };
  for (let sweep = 0; sweep < SWEEPS; sweep += 1) {
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as const) {
      const apq = at(a, p, q);
      if (Math.abs(apq) < 1e-300) {
        continue;
      }
      const theta = (at(a, q, q) - at(a, p, p)) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k += 1) {
        const [akp, akq] = [at(a, k, p), at(a, k, q)];
        put(a, k, p, c * akp - s * akq);
        put(a, k, q, s * akp + c * akq);
      }
      for (let k = 0; k < 3; k += 1) {
        const [apk, aqk] = [at(a, p, k), at(a, q, k)];
        put(a, p, k, c * apk - s * aqk);
        put(a, q, k, s * apk + c * aqk);
      }
      for (let k = 0; k < 3; k += 1) {
        const [vkp, vkq] = [at(v, k, p), at(v, k, q)];
        put(v, k, p, c * vkp - s * vkq);
        put(v, k, q, s * vkp + c * vkq);
      }
    }
  }

  const values = [at(a, 0, 0), at(a, 1, 1), at(a, 2, 2)];
  const column = (k: number): Vec3 => [at(v, 0, k), at(v, 1, k), at(v, 2, k)];
  let [least, most] = [0, 0];
  for (let k = 1; k < 3; k += 1) {
    if ((values[k] ?? 0) < (values[least] ?? 0)) {
      least = k;
    }
    if ((values[k] ?? 0) > (values[most] ?? 0)) {
      most = k;
    }
  }
  return { major: column(most), normal: column(least) };
}
