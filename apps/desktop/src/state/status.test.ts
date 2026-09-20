import { describe, expect, it } from "vitest";

import type { RunStatus } from "../ipc/bindings/RunStatus";
import type { RunView } from "../ipc/bindings/RunView";
import type { StaleDiff } from "../ipc/bindings/StaleDiff";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { deriveStatus } from "./status";

/** A workspace that is linked, available and prepared — the uninteresting half of every row. */
function view(overrides: Partial<WorkspaceView> = {}): WorkspaceView {
  return {
    root: "/ws/karas",
    name: "karas",
    input: { linked: true, path: "/scans/karas_reduced", available: true },
    excluded: [],
    files: [{ name: "FY234001", file: "FY234001_reduced.ply", size: 1127755, mtime_ms: 1789129634585 }],
    fragments: [],
    prepared: true,
    runs: [],
    fragments_dir: "/ws/karas/fragments",
    job: null,
    ...overrides,
  };
}

/** One row of the history; `stale` defaults to «nothing has changed since». */
function run(id: string, status: RunStatus, stale: Partial<StaleDiff> = {}): RunView {
  return {
    run: {
      version: 1,
      id,
      created: "2026-09-20T14:12:00+03:00",
      finished: "2026-09-20T14:30:00+03:00",
      status,
      spec: {},
      params: null,
      input: { files: [], excluded: [] },
      engine: null,
      counts: null,
      carried_from: null,
    },
    stale: { added: [], removed: [], changed: [], excluded_added: [], excluded_removed: [], ...stale },
  };
}

const RUN_ID = "20260920-141200";

describe("deriveStatus (A §5)", () => {
  it("is empty with no input linked", () => {
    expect(deriveStatus(view({ input: { linked: false, path: null, available: false } }), null, false).kind).toBe(
      "empty",
    );
  });

  it("is preparing while a prepare job runs, whatever else is true", () => {
    const withEverything = view({
      job: { kind: "prepare", run_id: null },
      prepared: false,
      runs: [run(RUN_ID, { state: "done" }, { added: ["FY234013"] })],
    });
    expect(deriveStatus(withEverything, RUN_ID, true)).toEqual({ kind: "preparing" });
  });

  it("is running while a run job runs, and names the run", () => {
    const going = view({ job: { kind: "run", run_id: RUN_ID }, runs: [run(RUN_ID, { state: "running" })] });
    expect(deriveStatus(going, null, false)).toEqual({ kind: "running", runId: RUN_ID });
  });

  it("is input_missing when the folder is gone and no run is selected", () => {
    const gone = view({ input: { linked: true, path: null, available: false } });
    expect(deriveStatus(gone, null, false)).toEqual({ kind: "input_missing" });
  });

  it("is unprepared when a scan has no current row", () => {
    expect(deriveStatus(view({ prepared: false }), null, false)).toEqual({ kind: "unprepared" });
  });

  it("is ready when prepared and nothing is selected", () => {
    expect(deriveStatus(view(), null, false)).toEqual({ kind: "ready" });
  });

  it("is current for a finished run whose input has not changed", () => {
    const done = view({ runs: [run(RUN_ID, { state: "done" })] });
    expect(deriveStatus(done, RUN_ID, false)).toEqual({ kind: "current", runId: RUN_ID });
  });

  it("is stale, with the diff, when the input changed since", () => {
    const diff: Partial<StaleDiff> = { added: ["FY234013", "FY234014", "FY234015"], removed: ["FY234002"], changed: ["FY234003", "FY234004"] };
    const drifted = view({ runs: [run(RUN_ID, { state: "done" }, diff)] });
    const status = deriveStatus(drifted, RUN_ID, false);
    expect(status.kind).toBe("stale");
    if (status.kind !== "stale") return;
    expect(status.runId).toBe(RUN_ID);
    expect(status.diff.added).toEqual(["FY234013", "FY234014", "FY234015"]);
    expect(status.diff.removed).toEqual(["FY234002"]);
    expect(status.diff.changed).toEqual(["FY234003", "FY234004"]);
  });

  it("is current, not stale, for a finished run while the input is missing", () => {
    const gone = view({
      input: { linked: true, path: null, available: false },
      runs: [run(RUN_ID, { state: "done" }, { removed: ["FY234002"] })],
    });
    expect(deriveStatus(gone, RUN_ID, false)).toEqual({ kind: "current", runId: RUN_ID });
  });

  it("is draft when a group of the current assembly is not refined", () => {
    const done = view({ runs: [run(RUN_ID, { state: "done" })] });
    expect(deriveStatus(done, RUN_ID, true)).toEqual({ kind: "draft", runId: RUN_ID });
  });

  it("is failed with its kind and message", () => {
    const blown = view({ runs: [run(RUN_ID, { state: "failed", kind: "gpu", message: "device lost" })] });
    expect(deriveStatus(blown, RUN_ID, false)).toEqual({
      kind: "failed",
      runId: RUN_ID,
      failKind: "gpu",
      message: "device lost",
    });
  });

  it("is cancelled", () => {
    const stopped = view({ runs: [run(RUN_ID, { state: "cancelled" })] });
    expect(deriveStatus(stopped, RUN_ID, false)).toEqual({ kind: "cancelled", runId: RUN_ID });
  });

  it("is interrupted, also for a run left `running` with no job behind it", () => {
    const marked = view({ runs: [run(RUN_ID, { state: "interrupted" })] });
    expect(deriveStatus(marked, RUN_ID, false)).toEqual({ kind: "interrupted", runId: RUN_ID });
    const orphaned = view({ runs: [run(RUN_ID, { state: "running" })] });
    expect(deriveStatus(orphaned, RUN_ID, false)).toEqual({ kind: "interrupted", runId: RUN_ID });
  });
});
