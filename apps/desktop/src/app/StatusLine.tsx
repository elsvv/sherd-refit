import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { useJobs } from "../state/jobs";
import { useReview } from "../state/review";
import type { Status } from "../state/status";
import Meter from "../ui/Meter";

/**
 * `m:ss`, the way the mock-up writes «прошло 6:41». Hours would need an `h:mm:ss` of their own;
 * a `Prepare` that ran for an hour is a bug report, not a layout problem, so the minutes simply
 * keep counting.
 *
 * Exported for [`RunOverlay`], which writes the very same «прошло 6:41» over the viewport: two
 * clocks on one screen that disagree by a second would be the app's own bug report.
 */
export function elapsedText(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const seconds = total % 60;
  return `${String(Math.floor(total / 60))}:${String(seconds).padStart(2, "0")}`;
}

/**
 * `Date.now()`, re-read once a second while `running` and not at all otherwise — the window must
 * not wake up every second to redraw a line that cannot change (the engine may be using the same
 * machine).
 *
 * A moment and not an elapsed span, because A §6's «осталось ≈» is a function of *now* as much as
 * the elapsed time is, and the overlay must not run a second clock of its own.
 */
export function useNow(running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) {
      return;
    }
    setNow(Date.now());
    const id = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => {
      window.clearInterval(id);
    };
  }, [running]);
  return now;
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
  const now = useNow(startedAt !== null);
  const elapsed = startedAt === null ? 0 : Math.max(0, now - startedAt);

  const running = view.job !== null;
  // A run has A §6's card over the viewport, which says the stage, the counts and the bar in full;
  // repeating all three down here would be the same fact twice on one screen, a second apart.
  const overlaid = status.kind === "running";

  // A §8: a review session is a job in the shell's eyes and nothing the user waits on — no strip,
  // no «Отменить», no clock (it may be open all afternoon between two clicks). The one thing it
  // owes them is that the engine is not answering *yet*: loading a collection's `match.state`
  // takes a few seconds, and until `ready` every key on the screen is dead.
  //
  // And nothing else: the `stage` a session leaves in the jobs store is the one it announced
  // while it was loading, and it stands there for the rest of the session — a status line
  // reading «Предобработка сканов» over a screen that has been answering for ten minutes. What
  // «Уточнить позы» should say while it runs is the draft line's question, not this one's.
  // Asked of the review store and not of `view.job`: `review_open` answers with nothing and the
  // window does not ask for a fresh view afterwards, so the session shows up in the view only
  // when something else has refreshed it. The store knows the moment the mode was entered.
  const session = useReview((state) => state.runId !== null) || view.job?.kind === "review";
  const waiting = useReview((state) => state.runId !== null && !state.ready);
  const stageProgress = stage === null ? undefined : progress[stage];
  const stageName = stage === null ? null : t(`stage.${stage}`, { defaultValue: stage });

  const warnings = view.fragments.filter((fragment) => fragment.warnings.length > 0).length;

  return (
    <footer className="flex h-[26px] shrink-0 items-center gap-3 border-t border-border bg-panel-2 px-3 text-xs text-muted">
      <span className="shrink-0 text-text">{t(`status.${status.kind}`)}</span>

      {waiting ? <span className="truncate">{t("review.loading")}</span> : null}

      {running && !session && !overlaid && stageName !== null ? (
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
      {running && !session ? (
        <span className="shrink-0 tabular-nums">{t("job.elapsed", { time: elapsedText(elapsed) })}</span>
      ) : null}

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
