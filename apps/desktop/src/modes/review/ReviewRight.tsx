import clsx from "clsx";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import { assembledRun } from "../../app/TopBar";
import type { AssemblyDto } from "../../ipc/bindings/AssemblyDto";
import type { CandidateRow } from "../../ipc/bindings/CandidateRow";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useAssembly } from "../../state/assembly";
import { explain, headline } from "../../state/reasons";
import { useReview } from "../../state/review";
import { useWorkspace } from "../../state/workspace";
import { FOCUS_RING } from "../../ui/Button";
import { formatDecimal } from "../input/format";
import type { ScoreLine } from "./queue";
import { candidateOf, limitsOf, poseKey, posesOf, scoreLines } from "./queue";

/** Where a fragment stands in the assembly on the screen: «группа 0», or «без пары» (A §8.3). */
function placeOf(assembly: AssemblyDto | null, name: string, t: TFunction): string {
  if (assembly === null) {
    return t("assembly.unpaired");
  }
  const at = assembly.groups.findIndex((group) => group.members.includes(name));
  const group = assembly.groups[at];
  return group === undefined || group.members.length < 2 ? t("assembly.unpaired") : t("assembly.group", { n: at });
}

/** A small heading over a section of the pane, as the «Сборка» inspector writes it. */
function Heading({ children }: { children: string }) {
  return <h3 className="mt-3 text-[10px] font-semibold tracking-wide text-muted uppercase">{children}</h3>;
}

/**
 * The scores against their limits (A §8.3's three columns): what was measured, what the run
 * asked for, and whether the one meets the other — green for a pass, amber for what held the
 * join back. Amber and not red: none of these is damage, and the reviewer is here precisely
 * because a failed test may still be a join.
 */
function Scores({ lines, t }: { lines: readonly ScoreLine[]; t: TFunction }) {
  return (
    <dl className="mt-2 grid grid-cols-[auto_1fr_auto] items-baseline gap-x-2 gap-y-0.5 text-[11px]">
      {lines.map((line) => (
        <div key={line.key} className="contents">
          <dt className="text-muted">{t(`review.score_${line.key}`)}</dt>
          <dd className="min-w-0 truncate text-right tabular-nums">{line.value}</dd>
          <dd className={clsx("tabular-nums", line.ok === null ? "text-muted" : line.ok ? "text-ok" : "text-warn")}>
            {line.limit ?? ""}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * The right pane of the «Ревью» mode (A §8.3, `review-mode.html`): everything about the selected
 * join that is not a picture — where its two fragments stand, every number the engine measured
 * beside the limit it was measured against, **why it was not confirmed on its own**, and the
 * other poses the same pair was found in.
 *
 * The «почему» block is the reason this screen exists: a reviewer who can read «вторая поза этой
 * пары слишком близка по оценке» knows what to look for in the viewport, and one who is shown
 * `no arm: support 0 < 1 and margin 1.00 < 2` does not. The sentences are [`explain`]'s, which
 * is pure and tested against the engine's own wording.
 */
export default function ReviewRight({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  const assembly = useAssembly((state) => state.assembly);
  const candidates = useAssembly((state) => state.candidates);
  const selected = useReview((state) => state.selected);

  const row: CandidateRow | null =
    selected === null ? null : candidateOf(candidates, selected.a, selected.b, selected.pose);
  if (selected === null || row === null) {
    return (
      <div className="px-2 py-2">
        <p className="text-[11px] text-muted">{t("review.no_pair")}</p>
      </div>
    );
  }

  // The limits are the *run's own*, out of its `run.json`: two runs of one workspace may have
  // been asked for different thresholds, and a score judged against the other one's would be a
  // pass or a failure the engine never claimed.
  const limits = limitsOf(assembledRun(view, selectedRunId)?.params ?? null);
  const lines = scoreLines(row, limits);
  // The headline is one of these sentences promoted to the front (the arms line, or the first
  // refusal), so the list under it is the *rest* — printed twice, one under the other, it reads
  // as two separate findings that happen to be word for word the same.
  const head = headline(row, t);
  const all = explain(row.evidence?.failed ?? [], t);
  const at = all.indexOf(head);
  const reasons = at === -1 ? all : [...all.slice(0, at), ...all.slice(at + 1)];
  // Numbered over *every* pose of the pair, best first, and only then narrowed to the ones not on
  // the screen: a pose has to keep its name while the reviewer walks between them. Numbering the
  // narrowed list would call whatever is left «поза 2» each time, so two different placements
  // would answer to one name a click apart.
  const shown = poseKey(selected.pose);
  const others = posesOf(candidates, selected.a, selected.b)
    .map((one, i) => ({ row: one, n: i + 1 }))
    .filter((one) => poseKey(one.row.pose) !== shown);

  return (
    <div className="h-full overflow-x-hidden overflow-y-auto px-2 py-2">
      <h2 className="truncate text-xs font-semibold" title={`${selected.a} – ${selected.b}`}>
        {`${selected.a} – ${selected.b}`}
      </h2>
      <p className="mt-0.5 truncate text-[11px] text-muted">
        {`${placeOf(assembly, selected.a, t)} + ${placeOf(assembly, selected.b, t)}`}
      </p>

      <Scores lines={lines} t={t} />

      {/* «Почему не подтверждён сам» is the wrong question over a join the engine *did* confirm,
          and the reviewer is here for those too (A §7.3: the engine is wrong about one in forty).
          Same block, same sentence from [`headline`], the heading turned round. */}
      <Heading>{t(all.length === 0 && row.tier === "confirmed" ? "review.why_confirmed" : "review.why")}</Heading>
      <p className="mt-1 text-[11px] leading-snug">{head}</p>
      {reasons.length === 0 ? null : (
        <ul className="mt-1 flex flex-col gap-0.5">
          {reasons.map((reason) => (
            <li key={reason} className="flex gap-1 text-[11px] leading-snug text-muted">
              <span aria-hidden="true">·</span>
              <span className="min-w-0">{reason}</span>
            </li>
          ))}
        </ul>
      )}

      {others.length === 0 ? null : (
        <>
          <Heading>{`${t("review.other_poses")} · ${String(others.length)}`}</Heading>
          {others.map((one) => (
            <button
              // Two poses of one pair differ by their matrix and by nothing else the eye can
              // key on, so the key is the matrix.
              key={poseKey(one.row.pose)}
              type="button"
              onClick={() => {
                useReview.getState().select(one.row);
              }}
              className={clsx(
                "mt-0.5 flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left hover:bg-panel-2",
                FOCUS_RING,
              )}
            >
              <span className="min-w-0 flex-1 truncate text-[11px]">{t("review.pose", { n: one.n })}</span>
              <span className="shrink-0 text-[11px] text-muted tabular-nums">{formatDecimal(one.row.score)}</span>
              <span className="shrink-0 text-[10px] text-accent">{t("review.show_pose")}</span>
            </button>
          ))}
          <p className="mt-1 text-[10px] leading-snug text-muted">{t("review.other_poses_hint")}</p>
        </>
      )}
    </div>
  );
}
