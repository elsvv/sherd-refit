import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { useFitSignal } from "../../app/shortcuts";
import { api } from "../../ipc";
import type { CandidateRow } from "../../ipc/bindings/CandidateRow";
import type { Verdict } from "../../ipc/bindings/Verdict";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useAssembly } from "../../state/assembly";
import { decisionOf, useReview } from "../../state/review";
import { useUi } from "../../state/ui";
import Button from "../../ui/Button";
import Chip from "../../ui/Chip";
import type { GhostView } from "../../viewer/AssemblyView";
import AssemblyView from "../../viewer/AssemblyView";
import { candidateOf, nextUndecided, queueRows, rowAt } from "./queue";

/** How the «Разъединить» slider is stepped, as the assembly's own is. */
const SEPARATION_STEP = 0.01;

/**
 * One decision from the keyboard or from the bar under the viewport, and the move to the next
 * pair that wants one (A §8.3). `null` is «Пропустить»: nothing is decided and the queue moves on.
 *
 * Exported because the same act has two homes — the three buttons here and the `A` / `X` / `␣`
 * keys of [`useShortcuts`] — and a second copy of «which row is next» would be a second opinion
 * about what the reviewer just did. It reads the stores rather than taking arguments for exactly
 * that reason: the keyboard works while this pane is collapsed.
 *
 * Nothing happens before the session says `ready`: a decision is filed *and* sent (A §8.2), and
 * the shell has nobody to send it to yet.
 */
export function decide(verdict: Verdict | null): void {
  const review = useReview.getState();
  if (!review.ready || review.selected === null) {
    return;
  }
  const candidates = useAssembly.getState().candidates;
  const band = useUi.getState().reviewBand;
  const selection = review.selected;
  const before = queueRows(candidates, band, review.decisions);
  const at = rowAt(before, selection.a, selection.b);

  if (verdict !== null) {
    // The candidate the centre is *showing*, which is not always the pair's best: «Другие позы ·
    // показать» puts another pose on the screen, and A §8.1 says the pose travels with the
    // decision.
    const row = candidateOf(candidates, selection.a, selection.b, selection.pose);
    if (row === null) {
      return;
    }
    if (verdict === "accept") {
      review.accept(row);
    } else {
      review.reject(row);
    }
  }

  // Against the decisions as they now stand: the row just decided must not be offered again.
  // The order does not move — the queue is sorted by score — so `at` still names the same pair.
  const after = queueRows(candidates, band, useReview.getState().decisions);
  const next = nextUndecided(after, at);
  const row = after[next];
  if (row !== undefined) {
    useReview.getState().select(row.best);
  }
}

/** Whether the two fragments of a pair stand in one group of the assembly on the screen. */
function placedTogether(groups: readonly { members: string[] }[], a: string, b: string): boolean {
  return groups.some((group) => group.members.includes(a) && group.members.includes(b));
}

/** One of the three legend dots under the viewport (A §8.3's «плотно · в пределах зазора · дальше»). */
function Legend({ colour, children }: { colour: string; children: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span aria-hidden="true" className={clsx("h-2 w-2 rounded-full", colour)} />
      {children}
    </span>
  );
}

/** One of the three actions, with the key that does the same thing (A §8.3's bar). */
function Act({
  variant,
  shortcut,
  label,
  disabled,
  onClick,
}: {
  variant: "ghost" | "danger" | "primary";
  shortcut: string;
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <Button variant={variant} disabled={disabled} onClick={onClick}>
      {label}
      <kbd className="ml-0.5 rounded border border-current px-1 text-[9px] opacity-70">{shortcut}</kbd>
    </Button>
  );
}

/**
 * The centre of the «Ревью» mode (A §8.3, `review-mode.html`): one candidate in 3D — A grey, B
 * orange, the seam coloured by distance — the chips that decide what is drawn, the legend that
 * says what the colours mean, and the three actions with their keys.
 *
 * «Пара» and «В сборке» are two views of one question. The pair alone is how a seam is judged;
 * the assembly with a ghost of B where this candidate would put it is how the *consequence* is
 * judged, which is the half a review image cannot show. They are mutually exclusive because the
 * viewer's pair mode hides everything else by construction.
 */
export default function ReviewCentre({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const assembly = useAssembly((state) => state.assembly);
  const candidates = useAssembly((state) => state.candidates);
  const ready = useReview((state) => state.ready);
  const selected = useReview((state) => state.selected);
  const detail = useReview((state) => state.detail);
  const decisions = useReview((state) => state.decisions);
  const inAssembly = useUi((state) => state.reviewInAssembly);
  const separation = useUi((state) => state.reviewSeparation);
  const seam = useUi((state) => state.reviewSeam);
  const highlighted = useUi((state) => state.selectedFragment);
  const fitSignal = useFitSignal((state) => state.signal);

  /** Whether the «Разъединить» slider is out; one popover, and only while it is asked for. */
  const [slider, setSlider] = useState(false);

  // Every fragment the run placed that also has a display mesh — the same list the «Сборка»
  // centre builds, and for the same reason: the viewer loads meshes, and `fragments/` is where
  // the window is allowed to look (A §2.1).
  const fragments = useMemo(() => {
    if (assembly === null) {
      return [];
    }
    const wanted = new Set(assembly.groups.flatMap((group) => group.members));
    return view.fragments
      .filter((info) => wanted.has(info.name))
      .map((info) => ({ name: info.name, url: api.assetUrl(`${view.fragments_dir}/${info.name}.glb`) }));
  }, [assembly, view.fragments, view.fragments_dir]);

  // «В сборке» is asked exactly to see where *this* pair stands, so entering it lights A up — the
  // one thing that emphasises a fragment inside an assembly — and B is the ghost beside it. A
  // click in the viewport moves the light from there, as it does in the «Сборка» mode.
  useEffect(() => {
    if (inAssembly && selected !== null) {
      useUi.getState().selectFragment(selected.a);
    }
  }, [inAssembly, selected]);

  // And the assembly is framed on the way in. Leaving «Пара» puts the camera back where the pair
  // found it, which in this mode is nowhere in particular: the screen opens on a pair, so the
  // camera was parked before the first mesh had arrived — the first look showed one fragment
  // clipped against the top edge of an otherwise empty viewport. Only on the way *in*: «Пара»
  // frames its own pair, and every later trip is one press of «Вписать» away either way.
  const shown = useRef(inAssembly);
  useEffect(() => {
    if (inAssembly && !shown.current) {
      useFitSignal.getState().requestFit();
    }
    shown.current = inAssembly;
  }, [inAssembly]);

  if (assembly === null) {
    return (
      <div className="flex h-full w-full items-center justify-center px-6 text-center text-xs text-viewport-text">
        {t("assembly.will_appear")}
      </div>
    );
  }

  const row: CandidateRow | null =
    selected === null ? null : candidateOf(candidates, selected.a, selected.b, selected.pose);
  const pair = selected === null || inAssembly ? null : { a: selected.a, b: selected.b, pose: selected.pose };
  // A §7.2's ghost, and only where it says something: with both fragments already in one group
  // the assembly is *already* showing where B stands, and a translucent copy on top of it would
  // read as a second piece.
  const ghost: GhostView | null =
    selected === null || !inAssembly || placedTogether(assembly.groups, selected.a, selected.b)
      ? null
      : { name: selected.b, anchor: selected.a, pose: selected.pose, flip: false };

  // What has been decided about the pair on the screen, straight off the list rather than out of
  // the queue: a band the pair is not in would not hold its row, and this is the same answer
  // [`queueRows`] joins in — the decision's own, whichever way round it names the two.
  const verdict = selected === null ? null : (decisionOf(decisions, selected.a, selected.b)?.verdict ?? null);
  const idle = !ready || row === null;

  return (
    <div className="relative h-full w-full">
      <AssemblyView
        fragments={fragments}
        assembly={assembly}
        colourMode="scan"
        layout="spread"
        unassembled
        explode={0}
        labels={false}
        selected={inAssembly ? highlighted : null}
        onSelect={(name) => {
          useUi.getState().selectFragment(name);
        }}
        fitSignal={fitSignal}
        pair={pair}
        pairDetail={seam && !inAssembly ? detail : null}
        separation={separation}
        ghost={ghost}
      />

      <div
        className="absolute top-2 right-2 left-2 flex flex-wrap items-start gap-1"
        role="group"
        aria-label={t("review.toolbar")}
      >
        <Chip
          active={!inAssembly}
          onClick={() => {
            useUi.getState().setReviewInAssembly(false);
          }}
        >
          {t("review.view_pair")}
        </Chip>
        <Chip
          active={inAssembly}
          onClick={() => {
            useUi.getState().setReviewInAssembly(true);
          }}
        >
          {t("review.view_assembly")}
        </Chip>
        <div className="relative">
          <Chip
            active={slider || separation > 0}
            disabled={inAssembly}
            aria-expanded={slider}
            onClick={() => {
              setSlider(!slider);
            }}
          >
            {t("assembly.explode")}
          </Chip>
          {slider && !inAssembly ? (
            <div className="absolute top-full left-0 z-10 mt-1 flex w-52 items-center gap-2 rounded-md border border-border bg-panel p-2 shadow-lg">
              <input
                type="range"
                min={0}
                max={1}
                step={SEPARATION_STEP}
                value={separation}
                aria-label={t("assembly.explode")}
                onChange={(e) => {
                  useUi.getState().setReviewSeparation(Number(e.target.value));
                }}
                className="min-w-0 flex-1 accent-accent"
              />
              <span className="w-9 shrink-0 text-right text-[11px] text-muted tabular-nums">
                {t("assembly.explode_value", { percent: `${String(Math.round(separation * 100))}%` })}
              </span>
            </div>
          ) : null}
        </div>
        <Chip
          active={seam}
          disabled={inAssembly}
          onClick={() => {
            useUi.getState().setReviewSeam(!seam);
          }}
        >
          {t("review.seam")}
        </Chip>
        <Chip
          onClick={() => {
            useFitSignal.getState().requestFit();
          }}
        >
          {t("assembly.fit")}
        </Chip>
      </div>

      {seam && !inAssembly ? (
        <div className="pointer-events-none absolute bottom-2 left-2 flex flex-wrap gap-3 rounded-md bg-panel/80 px-2 py-1 text-[10px] text-text">
          <Legend colour="bg-ok">{t("review.legend_tight")}</Legend>
          <Legend colour="bg-warn">{t("review.legend_gap")}</Legend>
          <Legend colour="bg-danger">{t("review.legend_far")}</Legend>
        </div>
      ) : null}

      <div
        className="absolute right-3 bottom-3 flex items-center gap-1.5 rounded-lg border border-border bg-panel p-1.5 shadow-lg"
        role="group"
        aria-label={t("review.actions")}
      >
        {verdict === null ? null : (
          <span className={clsx("px-1 text-[10px]", verdict === "accept" ? "text-ok" : "text-danger")}>
            {verdict === "accept" ? t("review.row_accepted") : t("review.row_rejected")}
          </span>
        )}
        <Act
          variant="ghost"
          shortcut="␣"
          label={t("review.skip")}
          disabled={idle}
          onClick={() => {
            decide(null);
          }}
        />
        <Act
          variant="danger"
          shortcut="X"
          label={t("review.reject")}
          disabled={idle}
          onClick={() => {
            decide("reject");
          }}
        />
        <Act
          variant="primary"
          shortcut="A"
          label={t("review.accept")}
          disabled={idle}
          onClick={() => {
            decide("accept");
          }}
        />
      </div>
    </div>
  );
}
