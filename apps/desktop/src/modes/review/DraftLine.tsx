import type { TFunction } from "i18next";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { formatDuration } from "../../app/params";
import { MOD_LABEL } from "../../app/shortcuts";
import { assembledRun } from "../../app/TopBar";
import type { RunFile } from "../../ipc/bindings/RunFile";
import type { UnplacedDto } from "../../ipc/bindings/UnplacedDto";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useAssembly } from "../../state/assembly";
import { explain } from "../../state/reasons";
import { useReview } from "../../state/review";
import { hasUnrefinedGroups } from "../../state/status";
import { useWorkspace } from "../../state/workspace";
import Button from "../../ui/Button";
import Chip from "../../ui/Chip";
import Dialog from "../../ui/Dialog";
import type { Tally } from "./queue";
import { notPlaced, tally } from "./queue";

/**
 * How long «Уточнить позы» is likely to take, out of the run's own R §9 stage (A §8.4's «≈30 с»).
 *
 * The run's number is an over-estimate — it refined every group and this refines only the ones a
 * decision left unrefined — and that is the right way round for a wait: being finished sooner
 * than promised is not a broken promise. A run that filed no timings promises nothing.
 */
function refineEstimate(run: RunFile | null, t: TFunction): string | null {
  const seconds = run?.counts?.timings.find((stage) => stage.stage === "refine")?.seconds;
  return seconds === undefined || seconds <= 0 ? null : formatDuration(seconds, t);
}

/** «стало 18 групп (было 17)» — one count with what it was before the draft, when that is known. */
function changed(now: number, before: number | null, word: string, t: TFunction): string {
  return before === null || before === now ? word : `${word} ${t("review.was", { n: before })}`;
}

/**
 * A §8.4's «N не встали»: the accepted joins the reassembly could not use, each with the engine's
 * own reason. A list and not a sentence, because the reason names the pair's own numbers and the
 * reviewer will want to go back to those pairs.
 */
function NotPlaced({ rows, onClose }: { rows: readonly UnplacedDto[]; onClose: () => void }) {
  const { t } = useTranslation();
  return (
    <Dialog title={t("review.not_placed_title")} width={480} onClose={onClose}>
      <p className="text-[11px] leading-snug text-muted">{t("review.not_placed_body")}</p>
      <ul className="mt-2 flex flex-col gap-2">
        {rows.map((row) => (
          <li key={`${row.a}\u0000${row.b}`}>
            <p className="text-xs font-semibold">{`${row.a} – ${row.b}`}</p>
            <p className="text-[11px] leading-snug text-muted">{explain([row.reason], t)[0]}</p>
          </li>
        ))}
      </ul>
    </Dialog>
  );
}

/**
 * The draft line (A §8.4, the mock-up's `wf-foot`): what the reviewer has decided, what it did to
 * the assembly, and everything that can be done with the draft as a whole — undo, redo, reset,
 * and the refinement of the poses the fast reassembly left rough.
 *
 * It is above the status line and only while there is a draft to talk about: with no decisions
 * and every group refined the run on the screen is the engine's own, and a row of disabled
 * buttons under it would be furniture.
 *
 * Every action here needs the warm session, so the line lives with the mode that owns one
 * (`Frame`). The status line says «Черновик» from anywhere; what to *do* about it is asked for
 * in «Ревью», where the engine is loaded and can answer in a second.
 */
export default function DraftLine({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  const assembly = useAssembly((state) => state.assembly);
  const decisions = useReview((state) => state.decisions);
  const past = useReview((state) => state.past);
  const future = useReview((state) => state.future);
  const pending = useReview((state) => state.pending);
  const refining = useReview((state) => state.refining);
  const dropped = useReview((state) => state.dropped);

  /** The two questions the line asks before it does something irreversible or long. */
  const [asking, setAsking] = useState(false);
  const [listing, setListing] = useState(false);

  const run = assembledRun(view, selectedRunId);
  const accepted = decisions.filter((one) => one.verdict === "accept").length;
  const rejected = decisions.length - accepted;
  // «2 принято · 1 отклонено», and neither half when nobody has done that — «0 отклонено» is a
  // number the reviewer has to read and subtract to learn nothing.
  const decided = [
    accepted > 0 ? t("review.draft_accepted", { n: accepted }) : null,
    rejected > 0 ? t("review.draft_rejected", { n: rejected }) : null,
  ].filter((part) => part !== null);
  const unrefined = hasUnrefinedGroups(assembly);
  const now: Tally | null = tally(assembly);
  const missed = notPlaced(assembly, decisions);
  const estimate = refineEstimate(run, t);

  // Nothing decided, nothing rough, nothing dropped: the run is as the engine left it.
  if (decisions.length === 0 && !unrefined && dropped.length === 0) {
    return null;
  }

  return (
    <>
      <div className="flex shrink-0 flex-col border-t border-border bg-panel-2">
        {dropped.length === 0 ? null : (
          // A §8.5: decisions carried from an earlier run that name a fragment this collection no
          // longer has. Said once, then put away — nothing can be done about them here.
          <div className="flex items-center gap-2 border-b border-border px-3 py-1 text-xs text-warn">
            <span className="min-w-0 flex-1 truncate" title={dropped.map((one) => `${one.a} – ${one.b}`).join(", ")}>
              {t("review.dropped", { count: dropped.length })}
            </span>
            <Button
              onClick={() => {
                useReview.getState().dismissDropped();
              }}
            >
              {t("review.dropped_ok")}
            </Button>
          </div>
        )}

        <div className="flex h-9 items-center gap-2 px-3 text-xs">
          {decided.length === 0 ? null : (
            <span className="shrink-0 rounded bg-warn/20 px-2 py-0.5 text-warn">
              {`${t("review.draft")}: ${decided.join(" · ")}`}
            </span>
          )}
          {now === null ? null : (
            <span className="min-w-0 truncate text-muted">
              {[
                changed(now.groups, run?.counts?.groups ?? null, t("review.now_groups", { count: now.groups }), t),
                changed(
                  now.unpaired,
                  run?.counts?.unassembled ?? null,
                  t("review.now_unpaired", { n: now.unpaired }),
                  t,
                ),
              ].join(" · ")}
            </span>
          )}

          {missed.length === 0 ? null : (
            <Chip
              tone="warn"
              onClick={() => {
                setListing(true);
              }}
            >
              {t("review.not_placed", { count: missed.length })}
            </Chip>
          )}

          <span className="flex-1" />

          <Button
            disabled={past.length === 0 || pending}
            onClick={() => {
              useReview.getState().undo();
            }}
          >
            {t("review.undo")}
            <kbd className="ml-0.5 rounded border border-current px-1 text-[9px] opacity-70">{`${MOD_LABEL}Z`}</kbd>
          </Button>
          <Button
            disabled={future.length === 0 || pending}
            onClick={() => {
              useReview.getState().redo();
            }}
          >
            {t("review.redo")}
          </Button>
          <Button
            disabled={decisions.length === 0 || pending}
            onClick={() => {
              setAsking(true);
            }}
          >
            {t("review.reset")}
          </Button>
          <Button
            variant="primary"
            // Only the rough groups are worth R §9's half a minute: with every group refined
            // the poses on the screen are already the exact ones (A §8.4).
            disabled={!unrefined || pending}
            title={unrefined ? undefined : t("review.refine_done")}
            onClick={() => {
              useReview.getState().refine();
            }}
          >
            {refining
              ? t("review.refining")
              : estimate === null
                ? t("review.refine")
                : `${t("review.refine")} · ${t("review.about", { time: estimate })}`}
          </Button>
        </div>
      </div>

      {listing ? (
        <NotPlaced
          rows={missed}
          onClose={() => {
            setListing(false);
          }}
        />
      ) : null}

      {asking ? (
        <Dialog
          title={t("review.reset_title")}
          width={380}
          onClose={() => {
            setAsking(false);
          }}
          footer={
            <>
              <span className="flex-1" />
              <Button
                onClick={() => {
                  setAsking(false);
                }}
              >
                {t("review.reset_keep")}
              </Button>
              <Button
                variant="danger"
                onClick={() => {
                  setAsking(false);
                  useReview.getState().reset();
                }}
              >
                {t("review.reset_confirm")}
              </Button>
            </>
          }
        >
          <p className="text-xs leading-snug">{t("review.reset_body", { count: decisions.length })}</p>
        </Dialog>
      ) : null}
    </>
  );
}
