import { Matrix4, Vector3 } from "three";

import { rowsToMatrix4 } from "./matrix";

/**
 * What the «Ревью» screen's pair in 3D is made of that is not the scene (A §8.3): the two
 * colours, and the two pieces of arithmetic over a pose.
 *
 * Both live here rather than inside [`AssemblyViewer`](./AssemblyViewer.ts) because both are
 * exactly where this screen can be wrong without *looking* wrong. A ghost drawn at `pose` where
 * `pose⁻¹` was meant stands in a perfectly plausible place beside the fragment it belongs to, and
 * a separation taken along the wrong line pushes B through A instead of off it — neither is
 * visible in a screenshot, and both are a line of arithmetic a test with a known transform can
 * pin down (A §11).
 */

/**
 * A's grey and B's orange, the two colours `sherd_core::review`'s own images use for the same two
 * fragments: every review PNG's caption reads «A=… GREY | B=… ORANGE».
 *
 * These are the exception the plan names to «colours only from the tokens of `styles.css`», and
 * for the same reason `colours.ts`'s two palettes are: a token is a colour with a job in the
 * interface, and these two are *data* — they say «this is the piece the pose moves» on a screen a
 * person reads beside `review/<a>__<b>.png`. A different orange in the two places would read as
 * two different things rather than as one join looked at twice.
 */
export const PAIR_A_COLOUR = "#b0b0b0";
export const PAIR_B_COLOUR = "#e8872b";

/**
 * And the seam's own voxels, white — again the review images': *«the seam's `t/3` voxels … drawn
 * white»* (`sherd_core::review`). They are the only thing on the screen that is neither of the
 * two fragments nor one of the three contact classes, and white is what sets them apart from all
 * five of those at once.
 */
export const SEAM_COLOUR = "#ffffff";

/**
 * What each of R §6.1's contact classes is drawn in — 0 under the tight limit, 1 under the gap
 * limit, 2 beyond — and what to use before the stylesheet has loaded.
 *
 * These *are* tokens of `styles.css`, and the same three the legend's dots wear («плотно · в
 * пределах зазора · дальше», A §8.3): the point in the viewport and the dot under it must be the
 * one colour, or the legend explains a picture that is not on the screen. The fallbacks are the
 * light theme's values.
 */
const BEYOND: Token = { token: "--danger", fallback: 0xd6372f };
export const CONTACT_COLOURS: readonly Token[] = [
  { token: "--ok", fallback: 0x2f9e55 },
  { token: "--warn", fallback: 0xd98a0b },
  BEYOND,
];

/** One colour of the palette, and what to draw with before `styles.css` has been parsed. */
export interface Token {
  token: string;
  fallback: number;
}

/**
 * Which of [`CONTACT_COLOURS`] one class from the wire is drawn in. A class the engine has not
 * got a colour for counts as «beyond»: the reviewer may be told a point is further away than it
 * is, never that it is closer.
 */
export function contactColour(cls: number): Token {
  return CONTACT_COLOURS[cls] ?? BEYOND;
}

/**
 * Where a ghost of a candidate's partner stands (A §7.2's «ghost»): `world(anchor) · pose`, or
 * `world(anchor) · pose⁻¹` when the candidate names the two fragments the other way round.
 *
 * A candidate carries `p_a = T · p_b` (R §0) — the matrix that takes **b's own frame into a's**.
 * So a ghost of `b` hung on an `a` that is already placed is that matrix under a's world matrix;
 * a ghost of `a` hung on `b` is its inverse under b's. Getting the two the wrong way round is not
 * a crash and not a blank screen: it is a fragment in a wrong place that looks like a right one.
 *
 * `pose` comes out of `candidates.json` over IPC, so it is data from outside: four rows of four
 * finite numbers or a `RangeError`, never a matrix of `NaN` in the scene graph.
 */
export function ghostMatrix(anchor: Matrix4, pose: readonly (readonly number[])[], flip: boolean): Matrix4 {
  const matrix = rowsToMatrix4(pose);
  if (flip) {
    // three.js answers a matrix it cannot invert with sixteen zeroes rather than with an error,
    // which would collapse the ghost onto the anchor's origin — a place, and a wrong one.
    const determinant = matrix.determinant();
    if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) {
      throw new RangeError("a pose that cannot be inverted names no place for a ghost");
    }
    matrix.invert();
  }
  return new Matrix4().multiplyMatrices(anchor, matrix);
}

/**
 * «Разъединить» for a pair (A §8.3): how far and which way B is pushed off A.
 *
 * The **direction** is the line between the two centroids — the only direction the two pieces
 * themselves name, and the one that opens the seam rather than sliding along it. The **distance**
 * is `amount` of `span`, how big a piece of this pair is, and deliberately *not* of how far apart
 * they currently stand: a pose under review may be nowhere near a fit, and a slider scaled by
 * that would fling B off the screen for a bad candidate and barely move it for a good one.
 * Scaled by the piece, one full slider always opens about the same gap, whatever the pose — which
 * is what «Разъединить» does over a group's own radius on the «Сборка» screen.
 *
 * The contact points and the seam voxels do not move with B: they are in A's frame, which is
 * exactly what makes pushing B away *reveal* them (A §8.3's «Шов: расстояния»).
 *
 * Two centroids in the same place name no line and nothing is pushed — a pair of identical
 * boxes, not an error.
 */
export function separationOffset(from: Vector3, to: Vector3, span: number, amount: number): Vector3 {
  const along = new Vector3().subVectors(to, from);
  const length = along.length();
  if (!Number.isFinite(length) || length < 1e-9 || !Number.isFinite(span) || span <= 0) {
    return new Vector3();
  }
  const push = Number.isFinite(amount) ? Math.min(Math.max(amount, 0), 1) : 0;
  return along.multiplyScalar((push * span) / length);
}
