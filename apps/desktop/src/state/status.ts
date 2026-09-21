import type { AssemblyDto } from "../ipc/bindings/AssemblyDto";
import type { FailKind } from "../ipc/bindings/FailKind";
import type { StaleDiff } from "../ipc/bindings/StaleDiff";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";

/**
 * Where the workspace stands (A §5). One value, twelve shapes, each carrying exactly what the
 * screen that shows it needs — the run it is about, the diff to list, the reason to print.
 */
export type Status =
  | { kind: "empty" }
  | { kind: "input_missing" }
  | { kind: "preparing" }
  | { kind: "unprepared" }
  | { kind: "ready" }
  | { kind: "running"; runId: string }
  | { kind: "current"; runId: string }
  | { kind: "stale"; runId: string; diff: StaleDiff }
  | { kind: "draft"; runId: string }
  | { kind: "failed"; runId: string; failKind: FailKind; message: string }
  | { kind: "cancelled"; runId: string }
  | { kind: "interrupted"; runId: string };

/**
 * Whether the input is exactly what a run saw. A run's `stale` is empty when nothing about the
 * input has moved since — five empty lists, and no sixth way to be stale.
 */
function unchanged(diff: StaleDiff): boolean {
  return (
    diff.added.length === 0 &&
    diff.removed.length === 0 &&
    diff.changed.length === 0 &&
    diff.excluded_added.length === 0 &&
    diff.excluded_removed.length === 0
  );
}

/**
 * Whether the assembly on the screen is a **draft** — A §5's `draft` row, and what the «Ревью»
 * mode's draft line is about (A §8.4).
 *
 * «Refined» is a property of a group: a reassembly after a decision hands back R §8's unrefined
 * poses for every group whose members and joins moved, and only «Уточнить позы» puts R §9 back
 * over them. A group of one has no join to refine and is never counted — the engine writes it
 * `refined: false` for want of anything truer, and a workspace of singletons would otherwise
 * read as a permanent draft.
 */
export function hasUnrefinedGroups(assembly: AssemblyDto | null): boolean {
  return assembly !== null && assembly.groups.some((group) => group.members.length > 1 && !group.refined);
}

/**
 * A §5's table as a pure function: the status is **derived**, never stored, so there is no second
 * opinion about the workspace to keep in step with the first — and every row of the table is a
 * unit test rather than a screen someone has to reach by hand (A §11).
 *
 * First match wins, in this order: a workspace with no input is `empty`; a running job speaks over
 * everything, because what the user is waiting on is the only thing they can act on; then the run
 * they have selected, if it is still in the history; and only with nothing selected do the
 * whole-workspace answers — the folder gone, the input not prepared, or ready — apply.
 *
 * `selectedRunId` names a row of `view.runs`; one that is no longer there (the run was deleted
 * under the window) reads as no selection rather than as an error. `hasUnrefinedGroups` is a fact
 * about the assembly the window has loaded, which this function cannot see, so the caller passes
 * it (A §5's «draft» row, A §8.4).
 */
export function deriveStatus(
  view: WorkspaceView,
  selectedRunId: string | null,
  hasUnrefinedGroups: boolean,
): Status {
  if (!view.input.linked) {
    return { kind: "empty" };
  }
  if (view.job?.kind === "prepare") {
    return { kind: "preparing" };
  }
  if (view.job?.kind === "run") {
    // The host fills `run_id` for every `Run`; a missing one would be its bug, and the window
    // shows the strip without a name rather than throwing on data it was given (A §2.1).
    return { kind: "running", runId: view.job.run_id ?? "" };
  }

  const selected = selectedRunId === null ? undefined : view.runs.find((row) => row.run.id === selectedRunId);
  if (selected) {
    const runId = selected.run.id;
    const status = selected.run.status;
    switch (status.state) {
      case "failed":
        return { kind: "failed", runId, failKind: status.kind, message: status.message };
      case "cancelled":
        return { kind: "cancelled", runId };
      case "interrupted":
        return { kind: "interrupted", runId };
      case "running":
        // No job is behind it — this process did not start it, or it did and died. The host marks
        // such runs on open; a row still `running` here is one nobody is writing any more.
        return { kind: "interrupted", runId };
      case "done":
        // Staleness is only meaningful while the input can be compared: with the folder gone the
        // run stays viewable and current, which is what A §5's «вход недоступен» row asks for.
        if (view.input.available && !unchanged(selected.stale)) {
          return { kind: "stale", runId, diff: selected.stale };
        }
        return hasUnrefinedGroups ? { kind: "draft", runId } : { kind: "current", runId };
    }
  }

  if (!view.input.available) {
    return { kind: "input_missing" };
  }
  if (!view.prepared) {
    return { kind: "unprepared" };
  }
  return { kind: "ready" };
}
