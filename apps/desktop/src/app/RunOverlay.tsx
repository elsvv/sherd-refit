import clsx from "clsx";
import { useTranslation } from "react-i18next";

import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { formatCount } from "../modes/input/format";
import { remainingSeconds } from "../state/eta";
import type { Progress } from "../state/jobs";
import { useJobs } from "../state/jobs";
import { useUi } from "../state/ui";
import Meter from "../ui/Meter";
import { formatDuration, parseSpec } from "./params";
import { elapsedText, useNow } from "./StatusLine";

/**
 * A §6's strip, left to right: Подготовка · Сопоставление · Уверенность · Сборка · Уточнение ·
 * Файлы. It is the *window's* list and not the engine's — the set of stages a run reports varies
 * with its options, and a real run of `fixtures/slab` announced three of the six stages it later
 * reported timings for — so the strip lights a cell by name and shows the rest as still to come.
 */
const STRIP = ["preprocess", "matching", "tiers", "assembly", "refine", "output"] as const;

/**
 * Which cell of the strip is the current one. The last `stage` event when it names one of the
 * six; otherwise — `objects`, or a stage a later engine adds — the furthest cell that has
 * reported any progress, so an unknown stage moves the strip forward instead of resetting it.
 */
function currentCell(stage: string | null, progress: Record<string, Progress>): number {
  const named = STRIP.findIndex((name) => name === stage);
  if (named !== -1) {
    return named;
  }
  let furthest = 0;
  STRIP.forEach((name, i) => {
    if (progress[name] !== undefined) {
      furthest = i;
    }
  });
  return furthest;
}

/**
 * The card over the viewport while a run is going (part 2 of `states-and-input.html`): what the
 * engine is doing, how long it has been doing it, how long it has left, and where that is in the
 * run as a whole.
 *
 * It sits over the viewport rather than in the status line because it is the one thing the user
 * is waiting for — and because the app stays usable underneath it (A §5: the input and the
 * earlier runs can be browsed while a run goes).
 */
export default function RunOverlay({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const stage = useJobs((state) => state.stage);
  const progress = useJobs((state) => state.progress);
  const samples = useJobs((state) => state.samples);
  const ratios = useJobs((state) => state.ratios);
  const startedAt = useJobs((state) => state.startedAt);

  const now = useNow(true);
  const elapsed = startedAt === null ? 0 : Math.max(0, now - startedAt);
  const left = remainingSeconds(samples, now, ratios);

  const cell = currentCell(stage, progress);
  const done = stage === null ? undefined : progress[stage];
  const title = stage === null ? t("run.starting") : t(`stage.${stage}`, { defaultValue: stage });

  // The executor is the sheet's, read back from the run's own `run.json`: the window is not told
  // what the engine resolved «Авто» to until the run ends (`EngineInfo.backend`).
  const runId = view.job?.run_id ?? null;
  const spec = runId === null ? null : parseSpec(view.runs.find((row) => row.run.id === runId)?.run.spec);

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-5 flex justify-center px-4">
      <div className="pointer-events-auto w-[480px] max-w-full rounded-lg border border-border bg-panel px-3.5 py-3 text-text shadow-lg">
        <div className="flex items-baseline gap-2">
          <b className="min-w-0 flex-1 truncate text-[13px]">{title}</b>
          <span className="shrink-0 tabular-nums">{t("job.elapsed", { time: elapsedText(elapsed) })}</span>
          <span className="shrink-0 text-muted">·</span>
          <span className="shrink-0 text-muted tabular-nums">
            {left === null ? t("run.remaining_soon") : t("run.remaining", { time: formatDuration(left, t) })}
          </span>
        </div>

        <ol className="mt-2 flex gap-1" aria-label={t("run.strip")}>
          {STRIP.map((name, i) => (
            <li
              key={name}
              aria-current={i === cell ? "step" : undefined}
              className={clsx(
                // The cells take the width of their own names and the current one takes what is
                // left over, which is what makes it «wider» (A §6) without cutting «Уверенность»
                // down to «Уверенн…»: six equal sixths of 480 px fit none of the long ones.
                "min-w-0 truncate rounded px-1.5 py-0.5 text-center text-[10px]",
                i === cell && "flex-1",
                i < cell && "bg-ok/20 text-ok",
                i === cell && "bg-accent text-accent-text",
                i > cell && "bg-panel-2 text-muted",
              )}
            >
              {t(`strip.${name}`)}
              {i < cell ? " ✓" : ""}
            </li>
          ))}
        </ol>

        <Meter value={done?.done ?? 0} max={done?.total ?? 0} label={title} className="mt-2 w-full" />

        <div className="mt-1.5 flex items-baseline gap-2 text-muted">
          <span className="min-w-0 flex-1 truncate tabular-nums">
            {done === undefined
              ? ""
              : t(stage === "matching" ? "run.pairs_done" : "run.units_done", {
                  done: formatCount(done.done, language),
                  total: formatCount(done.total, language),
                })}
          </span>
          {spec === null ? null : (
            <span className="shrink-0">{t("run.backend", { backend: t(`sheet.backend_${spec.backend}`) })}</span>
          )}
        </div>
      </div>
    </div>
  );
}
