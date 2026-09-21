import clsx from "clsx";
import type { TFunction } from "i18next";
import { ChevronDown, MoreHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import type { RunFile } from "../ipc/bindings/RunFile";
import type { RunStatus } from "../ipc/bindings/RunStatus";
import type { RunView } from "../ipc/bindings/RunView";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { useWorkspace } from "../state/workspace";
import Button, { FOCUS_RING } from "../ui/Button";
import Chip from "../ui/Chip";
import Dialog from "../ui/Dialog";
import IconButton from "../ui/IconButton";
import { formatDuration } from "./params";
import { elapsedText } from "./StatusLine";

/** An hour, from which a duration is better read as «1 ч 20 мин» than as «80:14». */
const HOUR_MS = 3_600_000;

/**
 * The colour of a run's state, as A §5's table and the mock-ups' dots use it: green for a result,
 * red for a failure, amber for a run that did not get to finish, and the accent for the one going
 * on now.
 */
const DOT: Record<RunStatus["state"], string> = {
  running: "bg-accent",
  done: "bg-ok",
  failed: "bg-danger",
  cancelled: "bg-warn",
  interrupted: "bg-warn",
};

/**
 * When a run was started, as the mock-ups write it: `20.09 14:12`.
 *
 * Built by hand rather than with `Intl`, for the reason `modes/input/format.ts` gives at length —
 * a webview's ICU data is not the same on every machine, and the selector's rows have to line up
 * in a column. A `created` that will not parse is shown as it stands rather than as «Invalid
 * Date»: it came off the disk and the window does not get to refuse it (A §2.1).
 *
 * Exported because the «Сборка» mode's inspector heads its run summary with the same date, and
 * two ways of writing one moment in one window is one too many.
 */
export function when(created: string): string {
  const at = new Date(created);
  if (Number.isNaN(at.getTime())) {
    return created;
  }
  const pad = (n: number): string => String(n).padStart(2, "0");
  return `${pad(at.getDate())}.${pad(at.getMonth() + 1)} ${pad(at.getHours())}:${pad(at.getMinutes())}`;
}

/**
 * How long a run took — `finished − created`, as `17:46` under the hour and «1 ч 20 мин» over it.
 * `null` while it is still going (the overlay counts that one) and for a run whose two timestamps
 * do not make a span.
 *
 * Exported for the same reason as [`when`]: the «Сборка» inspector's summary says «длительность»
 * and the history says it in its column, and they must agree to the second.
 */
export function took(run: RunFile, t: TFunction): string | null {
  if (run.finished === null) {
    return null;
  }
  const from = new Date(run.created).getTime();
  const to = new Date(run.finished).getTime();
  if (Number.isNaN(from) || Number.isNaN(to) || to < from) {
    return null;
  }
  const ms = to - from;
  return ms < HOUR_MS ? elapsedText(ms) : formatDuration(ms / 1000, t);
}

/**
 * What a run came to, in the fewest words that distinguish it: «17 групп · 81 без пары» for one
 * that finished, and its ending for one that did not (A §5's `failed` / `cancelled` /
 * `interrupted` rows).
 *
 * `long` is the dropdown's; the chip in the top bar has one line to say it in and drops the
 * fragments that found no partner.
 */
function summary(run: RunFile, t: TFunction, long: boolean): string {
  const status = run.status;
  switch (status.state) {
    case "running":
      return t("run.state_running");
    case "cancelled":
      return t("run.state_cancelled");
    case "interrupted":
      return t("run.state_interrupted");
    case "failed":
      return t("run.state_failed", { kind: t(`fail.${status.kind}`) });
    case "done": {
      const counts = run.counts;
      if (counts === null) {
        // A run filed as done whose `counts` never arrived: the host writes both together, so
        // this is a run.json from an older build rather than a number to invent.
        return t("run.state_done");
      }
      const groups = t("counts.groups", { count: counts.groups });
      return long ? `${groups} · ${t("counts.unpaired", { n: counts.unassembled })}` : groups;
    }
  }
}

/** Whether the input has moved since a run saw it (A §5's «устарел»); five empty lists is «no». */
function isStale(row: RunView): boolean {
  const diff = row.stale;
  return (
    diff.added.length > 0 ||
    diff.removed.length > 0 ||
    diff.changed.length > 0 ||
    diff.excluded_added.length > 0 ||
    diff.excluded_removed.length > 0
  );
}

/** The coloured dot that says what a run came to, beside its date. */
function Dot({ state }: { state: RunStatus["state"] }) {
  return <span aria-hidden="true" className={clsx("h-2 w-2 shrink-0 rounded-full", DOT[state])} />;
}

/**
 * The run selector of the top bar (A §7.4: «Run history is the run selector's dropdown»): a chip
 * naming the run the window is showing, and behind it every run of the workspace, newest first,
 * with what it came to and a way to throw it away.
 *
 * Selecting a run here is the one way into an earlier result: the workspace store's `selectRun`
 * both records the choice and loads what that run assembled, so the 3D and the name above it can
 * never belong to two different runs.
 */
export default function RunSelector({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  const [open, setOpen] = useState(false);
  /** Which row's «…» menu is up; at most one, and never while the list itself is closed. */
  const [rowMenu, setRowMenu] = useState<string | null>(null);
  /** The run the confirmation is asking about, which outlives the menu it was opened from. */
  const [confirming, setConfirming] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  const close = (): void => {
    setOpen(false);
    setRowMenu(null);
  };

  // A menu that stays open after the pointer has gone elsewhere is a menu in the way; Escape is
  // the keyboard's way out of the same corner. Both are installed only while it is open.
  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (e: PointerEvent) => {
      if (e.target instanceof Node && box.current?.contains(e.target) !== true) {
        setOpen(false);
        setRowMenu(null);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        setRowMenu(null);
      }
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const runs = view.runs;
  const selected = runs.find((row) => row.run.id === selectedRunId);
  const asked = confirming === null ? undefined : runs.find((row) => row.run.id === confirming);

  return (
    <div className="relative" ref={box}>
      <Chip
        active={open}
        disabled={runs.length === 0}
        aria-haspopup="menu"
        aria-expanded={open}
        title={runs.length === 0 ? t("topbar.no_runs") : t("topbar.run_history")}
        onClick={() => {
          if (open) {
            close();
          } else {
            setOpen(true);
          }
        }}
        className="max-w-72"
      >
        {selected === undefined ? null : <Dot state={selected.run.status.state} />}
        <span className="min-w-0 truncate">
          {runs.length === 0
            ? t("topbar.no_runs")
            : selected === undefined
              ? t("topbar.runs", { n: runs.length })
              : `${when(selected.run.created)} · ${summary(selected.run, t, false)}`}
        </span>
        {runs.length === 0 ? null : <ChevronDown size={14} aria-hidden="true" className="shrink-0" />}
      </Chip>

      {open ? (
        <div
          role="menu"
          aria-label={t("topbar.run_history")}
          className="absolute top-full left-0 z-20 mt-1 max-h-[60vh] w-[440px] overflow-y-auto rounded-md border border-border bg-panel p-1 shadow-lg"
        >
          {runs.map((row) => {
            const run = row.run;
            const duration = took(run, t);
            return (
              <div key={run.id} role="none">
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    role="menuitemradio"
                    aria-checked={run.id === selectedRunId}
                    onClick={() => {
                      close();
                      useWorkspace.getState().selectRun(run.id);
                    }}
                    className={clsx(
                      "flex min-w-0 flex-1 items-center gap-2 rounded px-2 py-1 text-left text-xs hover:bg-panel-2",
                      run.id === selectedRunId && "bg-select",
                      FOCUS_RING,
                    )}
                  >
                    <Dot state={run.status.state} />
                    <span className="shrink-0 tabular-nums">{when(run.created)}</span>
                    <span className="w-12 shrink-0 text-right text-muted tabular-nums">{duration ?? ""}</span>
                    <span className="min-w-0 flex-1 truncate text-muted">{summary(run, t, true)}</span>
                    {/* Only a result can be out of date: A §5 compares the input with what a run
                        *assembled*, and a run that failed has nothing to be stale about. */}
                    {run.status.state === "done" && isStale(row) ? (
                      <span className="shrink-0 rounded border border-warn px-1 text-[10px] text-warn">
                        {t("run.stale")}
                      </span>
                    ) : null}
                  </button>
                  <IconButton
                    role="menuitem"
                    label={t("run.menu")}
                    aria-haspopup="menu"
                    aria-expanded={rowMenu === run.id}
                    className="shrink-0 border-transparent bg-transparent"
                    onClick={() => {
                      setRowMenu(rowMenu === run.id ? null : run.id);
                    }}
                  >
                    <MoreHorizontal size={14} aria-hidden="true" />
                  </IconButton>
                </div>
                {/* In the flow under its row rather than floating over it: the list scrolls when
                    the history is long, and a scroll box clips anything absolutely positioned
                    inside it — the menu would simply not be there. */}
                {rowMenu === run.id ? (
                  <div
                    role="menu"
                    className="mt-0.5 mb-1 flex justify-end rounded border border-border bg-panel-2 p-1"
                  >
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        close();
                        setConfirming(run.id);
                      }}
                      className={clsx(
                        "rounded px-2 py-0.5 text-xs text-danger hover:bg-panel",
                        FOCUS_RING,
                      )}
                    >
                      {t("run.delete")}
                    </button>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}

      {confirming === null ? null : (
        <Dialog
          title={t("run.delete_title", { run: asked === undefined ? confirming : when(asked.run.created) })}
          width={380}
          onClose={() => {
            setConfirming(null);
          }}
          footer={
            <>
              <span className="flex-1" />
              <Button
                onClick={() => {
                  setConfirming(null);
                }}
              >
                {t("run.delete_keep")}
              </Button>
              <Button
                variant="danger"
                onClick={() => {
                  setConfirming(null);
                  void useWorkspace.getState().deleteRun(confirming);
                }}
              >
                {t("run.delete_confirm")}
              </Button>
            </>
          }
        >
          <p className="text-xs leading-snug">{t("run.delete_body")}</p>
        </Dialog>
      )}
    </div>
  );
}
