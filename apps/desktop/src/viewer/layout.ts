/**
 * Where the assembled groups stand when they are spread out (A §7.2, «все группы разнесены»).
 *
 * The engine centres every group on the origin (R §8.2), so a run drawn as it comes out of the
 * file is seventeen vessels inside one another. This is the pure part of pulling them apart: each
 * group as a disc the size of its bounding sphere, laid out on shelves in the XY plane, biggest
 * first, into a block that suits a viewport rather than a single long row.
 *
 * Pure and testable on purpose — it is the one part of the viewer that can be wrong in a way a
 * screenshot does not show (two groups a millimetre into each other read as one vessel).
 */

/** One group, as the layout sees it: something round of a known size, known by its number. */
export interface Disc {
  id: number;
  radius: number;
}

/** Where that group's centre goes, in the XY plane, with the whole block centred on the origin. */
export interface Placed {
  id: number;
  x: number;
  y: number;
}

/**
 * A block a little wider than tall. The viewport is between 800 × 740 (both panes open) and
 * 1360 × 740 (both collapsed) on the smallest window the app is designed for, and `fit()` frames
 * the block by its bounding sphere anyway; a ratio in the middle of that range wastes the least
 * screen in either state.
 */
const DEFAULT_ASPECT = 1.5;

/**
 * How many shelf widths are tried before the best is kept. Shelf packing is a step function of
 * the row width — a width one millimetre wider can pull a whole group up into the row above — so
 * the shape of the block is searched for rather than computed. Sixty-four tries is a fraction of
 * a millisecond for the 155 fragments A §7.2 budgets for, and it happens on a load and on an
 * explode, never on a frame.
 */
const TRIES = 64;

/** A number from outside (a bounding sphere of a mesh with a broken vertex) that is not usable. */
function size(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 0;
}

/** One shelf-packing attempt: what each disc's centre came to, and the box the discs take up. */
interface Block {
  centres: { index: number; x: number; y: number }[];
  width: number;
  height: number;
  cx: number;
  cy: number;
}

/**
 * Discs into rows no wider than `maxWidth`, each disc in a square box of `2r + gap` so that
 * touching boxes leave exactly the gap between the discs inside them — in a row, and between
 * rows, where a disc sits at the middle of a band as tall as the row's tallest box.
 */
function shelves(order: readonly { index: number; r: number }[], maxWidth: number, gap: number): Block {
  const rows: { items: { index: number; r: number; x: number }[]; width: number; height: number }[] = [];
  let row: { items: { index: number; r: number; x: number }[]; width: number; height: number } = {
    items: [],
    width: 0,
    height: 0,
  };
  for (const disc of order) {
    const box = 2 * disc.r + gap;
    if (row.items.length > 0 && row.width + box > maxWidth + 1e-9) {
      rows.push(row);
      row = { items: [], width: 0, height: 0 };
    }
    row.items.push({ index: disc.index, r: disc.r, x: row.width + box / 2 });
    row.width += box;
    row.height = Math.max(row.height, box);
  }
  rows.push(row);

  const centres: { index: number; x: number; y: number }[] = [];
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  let top = 0;
  for (const shelf of rows) {
    const y = top - shelf.height / 2;
    for (const item of shelf.items) {
      centres.push({ index: item.index, x: item.x, y });
      x0 = Math.min(x0, item.x - item.r);
      x1 = Math.max(x1, item.x + item.r);
      y0 = Math.min(y0, y - item.r);
      y1 = Math.max(y1, y + item.r);
    }
    top -= shelf.height;
  }
  return { centres, width: x1 - x0, height: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
}

/**
 * Discs packed on shelves, largest first, into a block about `aspect` wide to 1 tall, `gap` apart;
 * centred on the origin. The answer is in the order the discs were given, whatever order they
 * were packed in, so a caller can zip it with its own list.
 */
export function packDiscs(discs: readonly Disc[], gap: number, aspect: number = DEFAULT_ASPECT): Placed[] {
  if (discs.length === 0) {
    return [];
  }
  const space = size(gap);
  const want = size(aspect) > 0 ? aspect : DEFAULT_ASPECT;
  const order = discs
    .map((disc, index) => ({ index, r: size(disc.radius) }))
    // Largest first, and ties by the order they came in: the same input must pack the same way.
    .sort((a, b) => b.r - a.r || a.index - b.index);

  const widest = order.reduce((most, disc) => Math.max(most, 2 * disc.r + space), 0);
  const everything = order.reduce((sum, disc) => sum + 2 * disc.r + space, 0);
  let best: Block | null = null;
  let closest = Infinity;
  for (let i = 0; i < TRIES; i += 1) {
    const maxWidth = widest + ((everything - widest) * i) / (TRIES - 1);
    const block = shelves(order, maxWidth, space);
    // The ratio of ratios, so that half as wide as asked and twice as wide count the same.
    const error = block.height > 0 ? Math.abs(Math.log(block.width / block.height / want)) : 0;
    if (error < closest - 1e-12) {
      closest = error;
      best = block;
    }
  }
  if (best === null) {
    return discs.map((disc) => ({ id: disc.id, x: 0, y: 0 }));
  }

  const placed: Placed[] = discs.map((disc) => ({ id: disc.id, x: 0, y: 0 }));
  for (const centre of best.centres) {
    const row = placed[centre.index];
    if (row !== undefined) {
      row.x = centre.x - best.cx;
      row.y = centre.y - best.cy;
    }
  }
  return placed;
}
