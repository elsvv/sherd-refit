import { describe, expect, it } from "vitest";

import type { AssemblyDto } from "../../ipc/bindings/AssemblyDto";
import type { Row } from "./tree";
import { buildRows, UNPAIRED } from "./tree";

/**
 * An assembly the shape a run gives one: the groups largest first, and three fragments that found
 * no partner, each in a group of its own — which is what `assembly_dto` writes and what «Без
 * пары» counts (A §7.2).
 */
function assembly(): AssemblyDto {
  const groups = [
    ["FY234001", "FY234003", "FY234004", "FY234007"],
    ["FY234002", "FY234005", "FY234006"],
    ["FY234008", "FY234010"],
    ["FY234012"],
    ["FZ234020"],
    ["FY234030"],
  ];
  return {
    groups: groups.map((members) => ({ members, refined: members.length > 1 })),
    poses: {},
    joins: [],
    unplaced: [],
  };
}

/** The names of the member rows, in the order the tree puts them in. */
function members(rows: readonly Row[]): string[] {
  return rows.filter((row) => row.kind === "member").map((row) => row.name);
}

describe("buildRows", () => {
  it("lists the groups in the engine's order, with their own sizes, and nothing under a closed one", () => {
    const rows = buildRows(assembly(), new Set(), "");
    expect(rows).toEqual([
      { kind: "group", index: 0, count: 4, open: false },
      { kind: "group", index: 1, count: 3, open: false },
      { kind: "group", index: 2, count: 2, open: false },
      { kind: "unpaired", count: 3, open: false },
    ]);
  });

  it("puts a group's members under it, indented and in order, once it is open", () => {
    const rows = buildRows(assembly(), new Set([1]), "");
    expect(rows.map((row) => row.kind)).toEqual(["group", "group", "member", "member", "member", "group", "unpaired"]);
    expect(members(rows)).toEqual(["FY234002", "FY234005", "FY234006"]);
    expect(rows.filter((row) => row.kind === "member").every((row) => row.group === 1)).toBe(true);
  });

  it("gathers every singleton group into the one «Без пары» section at the end", () => {
    const rows = buildRows(assembly(), new Set([UNPAIRED]), "");
    const last = rows.at(-4);
    expect(last).toEqual({ kind: "unpaired", count: 3, open: true });
    expect(members(rows)).toEqual(["FY234012", "FZ234020", "FY234030"]);
    expect(rows.filter((row) => row.kind === "member").every((row) => row.group === UNPAIRED)).toBe(true);
  });

  it("keeps only the matching members for a search, opens their groups and drops the rest", () => {
    const rows = buildRows(assembly(), new Set(), "fy23400");
    expect(rows).toEqual([
      { kind: "group", index: 0, count: 4, open: true },
      { kind: "member", name: "FY234001", group: 0 },
      { kind: "member", name: "FY234003", group: 0 },
      { kind: "member", name: "FY234004", group: 0 },
      { kind: "member", name: "FY234007", group: 0 },
      { kind: "group", index: 1, count: 3, open: true },
      { kind: "member", name: "FY234002", group: 1 },
      { kind: "member", name: "FY234005", group: 1 },
      { kind: "member", name: "FY234006", group: 1 },
      { kind: "group", index: 2, count: 2, open: true },
      { kind: "member", name: "FY234008", group: 2 },
    ]);
    // The tray is gone with its three: FY234012, FZ234020 and FY234030 all fail the substring.
    expect(buildRows(assembly(), new Set(), "fz").map((row) => row.kind)).toEqual(["unpaired", "member"]);
  });

  it("answers with nothing for an empty assembly and for a search nothing matches", () => {
    const empty: AssemblyDto = { groups: [], poses: {}, joins: [], unplaced: [] };
    expect(buildRows(empty, new Set([0, UNPAIRED]), "")).toEqual([]);
    expect(buildRows(assembly(), new Set([0, 1, 2, UNPAIRED]), "нет такого")).toEqual([]);
  });
});
