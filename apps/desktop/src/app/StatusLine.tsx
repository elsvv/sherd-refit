import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { useJobs } from "../state/jobs";
import type { Status } from "../state/status";
import Meter from "../ui/Meter";

/**
 * `m:ss`, the way the mock-up writes «прошло 6:41». Hours would need an `h:mm:ss` of their own;
 * a `Prepare` that ran for an hour is a bug report, not a layout problem, so the minutes simply
 * keep counting.
 */
function elapsedText(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const seconds = total % 60;
  return `${String(Math.floor(total / 60))}:${String(seconds).padStart(2, "0")}`;
}

/**
 * Ticks once a second while a job runs, and not at all between jobs — the window must not wake up
 * every second to redraw a line that cannot change (the engine may be using the same machine).
 */
function useElapsed(startedAt: number | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (startedAt === null) {
      return;
    }
    setNow(Date.now());
    const id = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => {
      window.clearInterval(id);
    };
  }, [startedAt]);
  return startedAt === null ? 0 : Math.max(0, now - startedAt);
}

/**
 * The 26 px line at the foot of the frame (A §6): where the workspace stands on the left, what
 * the running job is doing in the middle, and what the collection adds up to on the right. Every
 * number here is read off the view and the job's events — nothing is counted twice or stored.
 */
export default function StatusLine({ view, status }: { view: WorkspaceView; status: Status }) {
  const { t } = useTranslation();
  const stage = useJobs((state) => state.stage);
  const progress = useJobs((state) => state.progress);
  const startedAt = useJobs((state) => state.startedAt);
  const elapsed = useElapsed(startedAt);

  const running = view.job !== null;
  const stageProgress = stage === null ? undefined : progress[stage];
  const stageName = stage === null ? null : t(`stage.${stage}`, { defaultValue: stage });

  const warnings = view.fragments.filter((fragment) => fragment.warnings.length > 0).length;

  return (
    <footer className="flex h-[26px] shrink-0 items-center gap-3 border-t border-border bg-panel-2 px-3 text-xs text-muted">
      <span className="shrink-0 text-text">{t(`status.${status.kind}`)}</span>

      {running && stageName !== null ? (
        <>
          <span className="truncate">{stageName}</span>
          {stageProgress === undefined ? null : (
            <>
              <span className="shrink-0 tabular-nums">
                {stageProgress.done} / {stageProgress.total}
              </span>
              <Meter value={stageProgress.done} max={stageProgress.total} label={stageName} />
            </>
          )}
        </>
      ) : null}
      {running ? <span className="shrink-0 tabular-nums">{t("job.elapsed", { time: elapsedText(elapsed) })}</span> : null}

      <span className="flex-1" />

      <span className="shrink-0 truncate">
        {[
          t("counts.fragments", { count: view.files.length }),
          warnings > 0 ? t("counts.warnings", { n: warnings }) : null,
          view.excluded.length > 0 ? t("counts.excluded", { count: view.excluded.length }) : null,
        ]
          .filter((part) => part !== null)
          .join(" · ")}
      </span>
    </footer>
  );
}
