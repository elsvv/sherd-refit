/**
 * The two palettes the assembly is painted with (A §7.2, «colour modes scan / fragment / group»).
 *
 * These are the one exception to «colours only from the tokens of `styles.css`»: a token is a
 * colour with a job in the interface, and these are *data* — twelve hues whose only duty is to be
 * told apart from one another. They are read in two very different places: on the 3D viewport,
 * which is dark in both themes (`--viewport` is #26262a and #161618), and on the tree's dots in
 * the left pane, which is white in the light theme. So every entry is a mid-tone: light enough to
 * read against #161618, dark enough to read against #ffffff, and none of them near either.
 *
 * Both functions are pure and both cycle, because a collection has more groups than a person can
 * tell colours apart anyway — the colour says «not the one next to it», not «number seven».
 */

/**
 * The groups, in the order they are handed out. Neighbouring entries are far apart in hue, since
 * the engine gives the groups largest first and the eye compares group 0 with group 1.
 */
const GROUPS = [
  "#e0763c", // orange — `main-layout.html`'s «Группа 0»
  "#4f9d69", // green — its «Группа 1»
  "#5b8fd8", // blue — its «Группа 2»
  "#c264c0", // magenta — its «Группа 3»
  "#d9a93a", // gold
  "#3fb0a3", // teal
  "#d15a5f", // red
  "#8b7fe0", // violet
  "#93ab3e", // olive
  "#de85b4", // pink
  "#58b4e0", // sky
  "#a97a4e", // brown
];

/**
 * The fragments. A second palette and not the first one shifted: «Цвет: фрагменты» and «Цвет:
 * группы» are two states of one toggle, and if they shared their hues a user could not tell from
 * the screen which of the two they are looking at. These are the lighter, softer half of the same
 * hues, which also reads as «this is a piece, not a vessel».
 */
const FRAGMENTS = [
  "#d98c6a",
  "#79c08a",
  "#7aa8e6",
  "#cf8fd6",
  "#e3c05f",
  "#6fc7bd",
  "#e08487",
  "#a89ceb",
  "#b3c66a",
  "#eaa7ca",
  "#86cbe9",
  "#c49a76",
];

/**
 * What «без пары» is drawn in — the grey of `main-layout.html`'s last tree row. A singleton group
 * is not a vessel and must not look like one, in the tray or on the dot beside it.
 */
export const UNPAIRED_COLOUR = "#8b8b93";

/** An index from a file or a list length: anything that is not a whole number counts as the first. */
function wrap(index: number, of: number): number {
  return Number.isFinite(index) ? Math.abs(Math.trunc(index)) % of : 0;
}

/** The colour of assembly group number `index`, cycling; also the colour of its dot in the tree. */
export function groupColour(index: number): string {
  return GROUPS[wrap(index, GROUPS.length)] ?? UNPAIRED_COLOUR;
}

/** The colour of fragment number `index` in «Цвет: фрагменты», cycling. */
export function fragmentColour(index: number): string {
  return FRAGMENTS[wrap(index, FRAGMENTS.length)] ?? UNPAIRED_COLOUR;
}
