import { describe, expect, it } from "vitest";

import { DEFAULT_SPEC, draftOf, specOf, withPreset } from "./params";

/**
 * A §7.4's sheet has three answers that are about this computer and not about this collection —
 * the executor, the memory limit and the thread count — and A §11's settings screen is where the
 * machine gives them once. The sheet is seeded from there when the workspace has no run of its
 * own to repeat, so nothing between that seeding and «Собрать» may quietly put them back: a
 * «Стандарт» that reset the two budgets would hand a run the whole machine after the user had
 * asked it not to.
 */
describe("the machine's three answers", () => {
  const machine = draftOf({ ...DEFAULT_SPEC, backend: "cpu", workers: 6, memory_gb: 8 });

  it("reach the spec under every preset, not only «Свои параметры»", () => {
    for (const preset of ["standard", "thorough", "custom"] as const) {
      const spec = specOf(withPreset(machine, preset));
      expect(spec).not.toBeNull();
      expect({ preset, ...spec }).toMatchObject({ backend: "cpu", workers: 6, memory_gb: 8 });
    }
  });

  it("survive a preset card that puts every threshold back", () => {
    const custom = withPreset({ ...machine, values: { ...machine.values, max_gap: "0.5" } }, "custom");
    const back = withPreset(custom, "standard");
    expect(back.values.max_gap).toBe(String(DEFAULT_SPEC.max_gap));
    expect(back.backend).toBe("cpu");
    expect(back.values.workers).toBe("6");
    expect(back.values.memory_gb).toBe("8");
  });

  it("fall back to the engine's own when a field outside «Свои параметры» cannot be read", () => {
    // Nothing on the screen can produce this — the two fields are only drawn for «Свои
    // параметры» — but a spec that became `null` would disable «Собрать» with nothing to point
    // at, so the unreadable value is dropped rather than carried.
    const broken = { ...machine, preset: "standard" as const, values: { ...machine.values, workers: "—" } };
    expect(specOf(broken)).toMatchObject({ workers: DEFAULT_SPEC.workers, memory_gb: 8 });
  });
});
