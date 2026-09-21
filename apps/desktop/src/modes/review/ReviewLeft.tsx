import { useVirtualizer } from "@tanstack/react-virtual";
import clsx from "clsx";
import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";

import { useAssembly } from "../../state/assembly";
import { useReview } from "../../state/review";
import { useUi } from "../../state/ui";
import Button, { FOCUS_RING } from "../../ui/Button";
import Chip from "../../ui/Chip";
import { formatDecimal } from "../input/format";
import type { Band, QueueRow } from "./queue";
import { BANDS, nextUndecided, queueRows, rowAt } from "./queue";

/** One line of the queue, in pixels — the height the virtualiser lays the list out with. */
const ROW = 24;
/** Rows drawn above and below the window, so a fast scroll shows pairs and not holes. */
const OVERSCAN = 8;

/**
 * Which token a row's dot wears. A decided pair says what was decided; an undecided one says
 * which band it came from — warn for the probable ones the screen exists for, ok for what the
 * engine confirmed, muted for what it threw out (A §8.3, and the same three the «Сборка»
 * inspector paints its candidate rows with).
 */
function dotOf(row: QueueRow, band: Band): string {
  if (row.verdict === "accept") {
    return "bg-ok";
  }
  if (row.verdict === "reject") {
    return "bg-danger";
  }
  return band === "confirmed" ? "bg-ok" : band === "rejected" ? "bg-muted" : "bg-warn";
}

/**
 * The left pane of the «Ревью» mode (A §8.3, `review-mode.html`): the band filter, the queue of
 * pairs best score first, and «Принять все оставшиеся вероятные» underneath it.
 *
 * What the queue *is* lives in [`queueRows`], which is pure and tested; this draws it, virtualised
 * for the same reason the assembly's tree is — a real collection leaves hundreds of probable pairs
 * and the window must stay responsive while a warm engine process sits beside it.
 *
 * The selection is the review store's, not this pane's: the centre and the inspector show the same
 * pair, and the keyboard (`A`, `X`, `␣`) moves it without this component being on the screen at
 * all — the user may have collapsed the pane.
 */
export default function ReviewLeft() {
  const { t } = useTranslation();
  const candidates = useAssembly((state) => state.candidates);
  const decisions = useReview((state) => state.decisions);
  const selected = useReview((state) => state.selected);
  const band = useUi((state) => state.reviewBand);

  const scroller = useRef<HTMLDivElement>(null);

  /** The probable queue, in the order «Принять все оставшиеся» will take its poses from. */
  const probable = useMemo(() => queueRows(candidates, "probable", decisions), [candidates, decisions]);
  const confirmed = useMemo(() => queueRows(candidates, "confirmed", decisions).length, [candidates, decisions]);
  const rows = useMemo(
    // The band on the screen is one of the two already counted for the chips more often than not,
    // and a collection leaves thousands of candidates: counting them three times over would be
    // three passes on every keystroke.
    () => (band === "probable" ? probable : queueRows(candidates, band, decisions)),
    [candidates, band, decisions, probable],
  );
  const counts = { probable: probable.length, confirmed };
  const left = probable.filter((row) => row.verdict === null).length;

  const at = selected === null ? -1 : rowAt(rows, selected.a, selected.b);

  // Something has to be on the screen when the mode opens, and the first pair that still wants a
  // decision is what the reviewer came for. Only when nothing is selected: a pair chosen in
  // another band, or left over from the last visit, is the user's own choice and stays.
  useEffect(() => {
    if (selected !== null || rows.length === 0) {
      return;
    }
    const first = nextUndecided(rows, -1);
    const row = rows[first === -1 ? 0 : first];
    if (row !== undefined) {
      useReview.getState().select(row.best);
    }
  }, [rows, selected]);

  const virtualizer = useVirtualizer<HTMLDivElement, HTMLDivElement>({
    count: rows.length,
    getScrollElement: () => scroller.current,
    estimateSize: () => ROW,
    overscan: OVERSCAN,
  });

  // The queue is walked with the keyboard as much as with the mouse (A §8.3's `A` / `X` / `␣`),
  // and a selection that moved out of sight would make the list useless to the hand.
  const shown = useRef(-1);
  useEffect(() => {
    if (at >= 0 && shown.current !== at) {
      shown.current = at;
      virtualizer.scrollToIndex(at);
    }
  }, [at, virtualizer]);

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border px-2 py-1.5">
        <h2 className="text-[10px] font-semibold tracking-wide text-muted uppercase">{t("review.queue_title")}</h2>
        <div className="mt-1.5 flex flex-wrap gap-1" role="group" aria-label={t("review.bands")}>
          {BANDS.map((one) => (
            <Chip
              key={one}
              active={band === one}
              onClick={() => {
                useUi.getState().setReviewBand(one);
              }}
            >
              {one === "rejected" ? t("review.band_rejected") : t(`review.band_${one}`, { n: counts[one] })}
            </Chip>
          ))}
        </div>
      </div>

      <div
        ref={scroller}
        role="group"
        aria-label={t("review.queue_title")}
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-1 py-1"
      >
        {rows.length === 0 ? (
          <p className="px-1 py-2 text-[11px] text-muted">{t("review.band_empty")}</p>
        ) : (
          <div className="relative w-full" style={{ height: `${String(virtualizer.getTotalSize())}px` }}>
            {virtualizer.getVirtualItems().map((item) => {
              const row = rows[item.index];
              if (row === undefined) {
                return null;
              }
              const label = `${row.a} – ${row.b}`;
              return (
                <div
                  key={item.key}
                  className="absolute top-0 left-0 w-full"
                  style={{ height: `${String(item.size)}px`, transform: `translateY(${String(item.start)}px)` }}
                >
                  <button
                    type="button"
                    aria-pressed={item.index === at}
                    title={label}
                    onClick={() => {
                      useReview.getState().select(row.best);
                    }}
                    className={clsx(
                      "flex h-full w-full items-center gap-1.5 rounded px-1 text-left hover:bg-panel-2",
                      item.index === at && "bg-select",
                      FOCUS_RING,
                    )}
                  >
                    <span aria-hidden="true" className={clsx("h-2 w-2 shrink-0 rounded-full", dotOf(row, band))} />
                    <span className="min-w-0 flex-1 truncate text-[11px]">{label}</span>
                    <span
                      className={clsx(
                        "shrink-0 text-[10px] tabular-nums",
                        row.verdict === "accept" ? "text-ok" : row.verdict === "reject" ? "text-danger" : "text-muted",
                      )}
                    >
                      {row.verdict === "accept"
                        ? t("review.row_accepted")
                        : row.verdict === "reject"
                          ? t("review.row_rejected")
                          : formatDecimal(row.best.score)}
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {left === 0 ? null : (
        <div className="shrink-0 border-t border-border px-2 py-2">
          <Button
            className="w-full"
            onClick={() => {
              useReview.getState().acceptAllProbable(probable.map((row) => row.best));
            }}
          >
            {t("review.accept_all")}
          </Button>
          <p className="mt-1.5 text-[10px] leading-snug text-muted">{t("review.accept_all_hint")}</p>
        </div>
      )}
    </div>
  );
}
