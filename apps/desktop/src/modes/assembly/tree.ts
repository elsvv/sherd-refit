import type { AssemblyDto } from "../../ipc/bindings/AssemblyDto";

/**
 * The index the «Без пары» section is opened and closed under.
 *
 * The tray is one row of the same list as the groups (A §7.3: «Left: tree of groups → fragments,
 * «Без пары», search»), so it needs a key in the same set of open rows — and −1 is the index no
 * group can have. It is also what a loose fragment's [`Row`] carries as its group, which is how
 * the pane knows to paint its dot grey rather than in a vessel's colour.
 */
export const UNPAIRED = -1;

/**
 * One line of the left pane's tree, already flattened: the rows are what a virtualiser scrolls
 * through, so «is this group open» is decided here and not by the component drawing the line.
 */
export type Row =
  | {
      kind: "group";
      /** Its index in `assembly.groups` — what `setGroupVisible` and `groupColour` are given. */
      index: number;
      /** How many fragments the engine put in it, whatever the search is showing of them. */
      count: number;
      open: boolean;
    }
  | {
      kind: "unpaired";
      /** How many fragments found no partner at all — every singleton group's one member. */
      count: number;
      open: boolean;
    }
  | {
      kind: "member";
      name: string;
      /** The group it belongs to, or [`UNPAIRED`]. */
      group: number;
    };

/**
 * The tree of A §7.3's «Сборка» left pane as a flat list of rows.
 *
 * Pure, and the whole of the pane's logic: what the tree *is* — which groups there are, in which
 * order, which fragments are under them and which of them a search leaves — is a function of the
 * run's `assembly.json` and of two small pieces of interface state, so it can be tested without
 * a screen (A §11 rules out a component-test harness, not a test).
 *
 * The groups keep the order the engine gave them, which is largest first (`AssemblyDto.groups`),
 * because that order is a fact about the result and re-sorting it here would make the tree
 * disagree with the viewport's packing and with `counts.groups`.
 *
 * A group of one is not a vessel: every singleton group's member is gathered into one «Без пары»
 * section at the end, collapsed unless [`UNPAIRED`] is in `open` — the same rule the viewer uses
 * for its tray, so the eye beside that row and the tray in the viewport mean the same thing.
 *
 * A search (case-insensitive, by substring of the fragment's name) keeps only the members that
 * match and opens every group that still has one, whatever `open` says: someone typing a name is
 * asking to be shown it, not to be told which closed group it is in. A group all of whose
 * members the search dropped disappears with them; the counts on the rows that remain stay the
 * counts of the *groups*, since «Группа 0 · 20 фр.» is what that group is and not what the
 * search left of it.
 */
export function buildRows(assembly: AssemblyDto, open: ReadonlySet<number>, query: string): Row[] {
  const needle = query.trim().toLowerCase();
  const searching = needle !== "";
  const matches = (name: string): boolean => !searching || name.toLowerCase().includes(needle);

  const rows: Row[] = [];
  const loose: string[] = [];

  assembly.groups.forEach((group, index) => {
    if (group.members.length < 2) {
      loose.push(...group.members);
      return;
    }
    const shown = group.members.filter(matches);
    if (shown.length === 0 && searching) {
      return;
    }
    const opened = searching || open.has(index);
    rows.push({ kind: "group", index, count: group.members.length, open: opened });
    if (opened) {
      for (const name of shown) {
        rows.push({ kind: "member", name, group: index });
      }
    }
  });

  const shownLoose = loose.filter(matches);
  if (loose.length > 0 && (shownLoose.length > 0 || !searching)) {
    const opened = searching || open.has(UNPAIRED);
    rows.push({ kind: "unpaired", count: loose.length, open: opened });
    if (opened) {
      for (const name of shownLoose) {
        rows.push({ kind: "member", name, group: UNPAIRED });
      }
    }
  }

  return rows;
}
