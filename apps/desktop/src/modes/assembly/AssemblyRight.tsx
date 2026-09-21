import clsx from "clsx";
import type { TFunction } from "i18next";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { took, when } from "../../app/RunSelector";
import { elapsedText } from "../../app/StatusLine";
import { api } from "../../ipc";
import type { AssemblyDto } from "../../ipc/bindings/AssemblyDto";
import type { CandidateRow } from "../../ipc/bindings/CandidateRow";
import type { Decision } from "../../ipc/bindings/Decision";
import type { RunFile } from "../../ipc/bindings/RunFile";
import type { Verdict } from "../../ipc/bindings/Verdict";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useAssembly } from "../../state/assembly";
import { useExport } from "../../state/export";
import { decisionOf, useReview } from "../../state/review";
import { useUi } from "../../state/ui";
import { useWorkspace } from "../../state/workspace";
import Button, { FOCUS_RING } from "../../ui/Button";
import type { GhostView } from "../../viewer/AssemblyView";
import { groupColour, UNPAIRED_COLOUR } from "../../viewer/colours";
import { formatCount, formatDecimal } from "../input/format";

/** The three bands of `candidates.json`, in the order A §8.3 reads them. */
const BANDS: readonly CandidateRow["tier"][] = ["confirmed", "probable", "rejected"];

/** Which token a band's dot wears: the engine's own confidence, in the colours A §5 uses for it. */
const BAND_DOT: Record<CandidateRow["tier"], string> = {
  confirmed: "bg-ok",
  probable: "bg-warn",
  rejected: "bg-muted",
};

/** One line of a key–value list — the shape both inspectors and the summary are written in. */
interface Field {
  key: string;
  label: string;
  value: string;
}

/** The inspector's key–value list, as «Вход»'s right pane draws it. */
function Fields({ fields }: { fields: readonly Field[] }) {
  return (
    <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-[11px]">
      {fields.map((field) => (
        <div key={field.key} className="contents">
          <dt className="text-muted">{field.label}</dt>
          <dd className="min-w-0 truncate text-right tabular-nums" title={field.value}>
            {field.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** A small heading over a section of the pane. */
function Heading({ children }: { children: string }) {
  return <h3 className="mt-3 text-[10px] font-semibold tracking-wide text-muted uppercase">{children}</h3>;
}

/**
 * The `open` that is on its way, so that two clicks a second apart do not send the second
 * decision before `review_open` has been invoked at all — the shell would refuse it with «нет
 * открытой сессии ревью» and the row would silently stay undecided. Module-level, because it
 * outlives the pane: the inspector is remounted whenever the mode is left and come back to.
 */
let opening: { runId: string; done: Promise<void> } | null = null;

/**
 * A decision made where the fragment is being looked at (A §7.3's «тот же выбор доступен и из
 * режима Сборка»), and the session it needs.
 *
 * The session is opened on the **first** decision and not on entering the mode: it costs the
 * engine's three gigabytes and its whole life is A §8's, so looking at an assembly must not pay
 * for it. Afterwards everything is the review store's — the same list, the same undo, the same
 * `decisions.json` — and the assembly that comes back lands in [`useAssembly`], which is what
 * the viewport beside this pane is already drawing.
 *
 * `null` is «Снять решение»: back to «нет решения», not to «отклонено».
 */
async function decide(runId: string, row: CandidateRow, verdict: Verdict | null): Promise<void> {
  const review = useReview.getState();
  if (review.runId !== runId) {
    // `open` marks the store with the run synchronously, so a second click while this one is in
    // flight lands in the branch below and waits for the same promise.
    opening = { runId, done: review.open(runId) };
  }
  await (opening?.runId === runId ? opening.done : Promise.resolve());
  const session = useReview.getState();
  if (session.runId !== runId) {
    // Another run — or another workspace — was chosen while the match was loading.
    return;
  }
  if (verdict === null) {
    session.clear(row.a, row.b);
  } else if (verdict === "accept") {
    session.accept(row);
  } else {
    session.reject(row);
  }
}

/** Which group of the run holds a fragment, and whether that «group» is a group of one. */
function placeOf(assembly: AssemblyDto, name: string): { group: number; members: number } | null {
  const at = assembly.groups.findIndex((row) => row.members.includes(name));
  const found = assembly.groups[at];
  return found === undefined ? null : { group: at, members: found.members.length };
}

/**
 * One candidate of the selected fragment (A §7.3): who the partner is, what the pair scored, and
 * the two buttons that settle it — or, once it is settled, what was decided and the way back.
 *
 * The cursor on the row hangs the **ghost**: the partner, translucent, where this candidate would
 * put it. That is the one question a list of names and numbers cannot answer — «и куда он тогда
 * встанет?» — and it costs nothing to ask, so it is asked by hovering and not by clicking. The
 * keyboard gets the same on focus, and moving the focus from the row to its own buttons keeps it.
 */
function JoinRow({
  name,
  partner,
  row,
  together,
  used,
  verdict,
  onPick,
  onGhost,
  onDecide,
}: {
  /** The fragment the inspector is about; the ghost hangs on it and the pose is read against it. */
  name: string;
  partner: string;
  row: CandidateRow;
  /** Whether the assembly as it now stands already puts the two in one group. */
  together: boolean;
  /** And whether it is *this* join that holds them there (A §7.3's «в сборке»). */
  used: boolean;
  verdict: Verdict | null;
  onPick: (name: string) => void;
  onGhost: (ghost: GhostView | null) => void;
  onDecide: (row: CandidateRow, verdict: Verdict | null) => void;
}) {
  const { t } = useTranslation();
  // A ghost only says something where the assembly does not: with both fragments already in one
  // group the partner *is* standing there, and a translucent copy on top of it would read as a
  // second piece (A §7.2). `pose` maps b into a's frame (R §0), so a ghost of the candidate's `a`
  // hung on its `b` is the inverse — which [`ghostMatrix`] takes as `flip`.
  const ghost: GhostView | null = together
    ? null
    : { name: partner, anchor: name, pose: row.pose, flip: row.b === name };
  const dot = verdict === "accept" ? "bg-ok" : verdict === "reject" ? "bg-danger" : BAND_DOT[row.tier];

  return (
    <div
      className="rounded px-1 py-0.5 hover:bg-panel-2"
      onMouseEnter={() => {
        onGhost(ghost);
      }}
      onMouseLeave={() => {
        onGhost(null);
      }}
      onFocus={() => {
        onGhost(ghost);
      }}
      onBlur={(e) => {
        // Tabbing from the row onto its own «Подтвердить» must not put the ghost out: the button
        // is about the very placement the ghost is showing.
        if (!e.currentTarget.contains(e.relatedTarget)) {
          onGhost(null);
        }
      }}
    >
      <button
        type="button"
        onClick={() => {
          onPick(partner);
        }}
        title={partner}
        className={clsx("flex w-full items-center gap-1.5 rounded text-left", FOCUS_RING)}
      >
        <span aria-hidden="true" className={clsx("h-2 w-2 shrink-0 rounded-full", dot)} />
        <span className="min-w-0 flex-1 truncate text-[11px]">{partner}</span>
        {used ? <span className="shrink-0 text-[10px] text-muted">{t("assembly.used")}</span> : null}
        <span className="w-9 shrink-0 text-right text-[11px] tabular-nums">{formatDecimal(row.score)}</span>
      </button>

      <div className="mt-0.5 flex items-center gap-1">
        {verdict === null ? (
          <>
            <Button
              variant="danger"
              className="min-w-0 flex-1"
              aria-label={`${t("review.reject")}: ${name} – ${partner}`}
              onClick={() => {
                onDecide(row, "reject");
              }}
            >
              <span className="truncate">{t("review.reject")}</span>
            </Button>
            <Button
              variant="primary"
              className="min-w-0 flex-1"
              aria-label={`${t("review.accept")}: ${name} – ${partner}`}
              onClick={() => {
                onDecide(row, "accept");
              }}
            >
              <span className="truncate">{t("review.accept")}</span>
            </Button>
          </>
        ) : (
          <>
            <span className={clsx("min-w-0 flex-1 truncate text-[10px]", verdict === "accept" ? "text-ok" : "text-danger")}>
              {verdict === "accept" ? t("review.row_accepted") : t("review.row_rejected")}
            </span>
            <Button
              className="min-w-0 shrink-0"
              aria-label={`${t("review.undecide")}: ${name} – ${partner}`}
              onClick={() => {
                onDecide(row, null);
              }}
            >
              <span className="truncate">{t("review.undecide")}</span>
            </Button>
          </>
        )}
      </div>
    </div>
  );
}

/** A candidate as the inspector lists it: the partner's name, and the row it came from. */
interface Join {
  partner: string;
  row: CandidateRow;
}

/**
 * Every **pair** `name` took part in, best first inside each band (R §5.7's ranking key is
 * `score`).
 *
 * One row per pair and not one per candidate, although `candidates.json` holds a row per
 * placement: R §6's NMS keeps a rival pose when it stands far enough from the winner, and a
 * decision is about the pair as a whole (A §8.1 keys it by the unordered pair). Two rows named
 * «FY234010» with a «Подтвердить» each would offer the same decision twice and show the same
 * verdict on both. The pair's other poses are the «Ревью» inspector's «Другие позы пары», where
 * they can be looked at one at a time; here the best is the one that speaks for the pair.
 */
function joinsOf(candidates: readonly CandidateRow[], name: string): Record<CandidateRow["tier"], Join[]> {
  const bands: Record<CandidateRow["tier"], Join[]> = { confirmed: [], probable: [], rejected: [] };
  const best = new Map<string, CandidateRow>();
  for (const row of candidates) {
    if (row.a !== name && row.b !== name) {
      continue;
    }
    const partner = row.a === name ? row.b : row.a;
    const standing = best.get(partner);
    if (standing === undefined || row.score > standing.score) {
      best.set(partner, row);
    }
  }
  for (const [partner, row] of best) {
    bands[row.tier].push({ partner, row });
  }
  for (const band of BANDS) {
    bands[band].sort((one, other) => other.row.score - one.row.score);
  }
  return bands;
}

/**
 * The selected fragment: where the run put it, what it is, and every pair it took part in by
 * band (A §7.3, A §8.3). The rejected ones are folded away, because they are the long tail and
 * the reason to open them is curiosity rather than work — but they can be *accepted*, which is
 * A §7.3's whole point about them: on a real collection the engine throws out about one join in
 * forty that a person can see is right.
 */
function FragmentInspector({
  view,
  assembly,
  candidates,
  name,
  runId,
  decisions,
}: {
  view: WorkspaceView;
  assembly: AssemblyDto;
  candidates: readonly CandidateRow[];
  name: string;
  /** The run being looked at; `null` while none is selected, which is when nothing can be decided. */
  runId: string | null;
  /** What has been decided about this run's pairs, from the session or from its own file. */
  decisions: readonly Decision[];
}) {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const [showRejected, setShowRejected] = useState(false);

  const place = placeOf(assembly, name);
  const info = view.fragments.find((row) => row.name === name);
  const bands = joinsOf(candidates, name);
  const total = BANDS.reduce((sum, band) => sum + bands[band].length, 0);
  // Which of this fragment's partners the assembly already stands beside it, so that a ghost is
  // hung only where there is nothing to see yet — and which of them it is held to by this very
  // join. Both come from the assembly **as it now stands** and not from `candidates.json`'s own
  // `used`: a decision made here is answered with a reassembly within the second (A §8.2), and a
  // row reading «в сборке» under «✕ отклонён» would be describing a run that no longer exists.
  const beside = new Set(assembly.groups.find((group) => group.members.includes(name))?.members ?? []);
  const holding = new Set(
    assembly.joins.filter((join) => join.a === name || join.b === name).map((join) => (join.a === name ? join.b : join.a)),
  );

  // The ghost belongs to the cursor, and the cursor leaves a pane by being somewhere else — the
  // tree, the viewport, another fragment — without any row ever reporting that it left.
  useEffect(() => {
    return () => {
      useUi.getState().setAssemblyGhost(null);
    };
  }, [name]);

  // And it is taken away by the assembly that granted it: «Подтвердить» on a row is answered
  // within the second with a reassembly that puts the partner where the ghost was standing
  // (A §8.2), and a translucent copy over the real piece reads as a second piece (A §7.2). The
  // cursor has not moved, so no row will report it — the row's own `together` is already true
  // and its `onMouseEnter` will not fire again.
  useEffect(() => {
    const ghost = useUi.getState().assemblyGhost;
    if (ghost === null || ghost.anchor !== name) {
      return;
    }
    const group = assembly.groups.find((row) => row.members.includes(name));
    if (group !== undefined && group.members.includes(ghost.name)) {
      useUi.getState().setAssemblyGhost(null);
    }
  }, [assembly, name]);

  const fields: Field[] = [];
  if (info !== undefined) {
    fields.push(
      { key: "thickness", label: t("input.field_thickness"), value: formatDecimal(info.stats.thickness) },
      {
        key: "watertight",
        label: t("input.field_watertight"),
        value: info.stats.watertight ? t("input.yes") : t("input.no"),
      },
      { key: "faces", label: t("input.field_faces"), value: formatCount(info.stats.faces, language) },
    );
  }

  const pick = (partner: string): void => {
    // Following a partner takes the cursor away from the row it was on, and the ghost with it.
    useUi.getState().setAssemblyGhost(null);
    useUi.getState().selectFragment(partner);
    useUi.getState().flyToFragment(partner);
  };

  const setGhost = (ghost: GhostView | null): void => {
    useUi.getState().setAssemblyGhost(ghost);
  };

  /** One row of the list, however deep in the pane it sits. */
  const rowOf = (join: Join) => (
    <JoinRow
      // The partner names the row: there is exactly one of them per pair now, whatever the
      // fragment is called and however many poses the pair was found in.
      key={join.partner}
      name={name}
      partner={join.partner}
      row={join.row}
      together={beside.has(join.partner)}
      used={holding.has(join.partner)}
      verdict={decisionOf(decisions, join.row.a, join.row.b)?.verdict ?? null}
      onPick={pick}
      onGhost={setGhost}
      onDecide={(row, verdict) => {
        if (runId !== null) {
          void decide(runId, row, verdict);
        }
      }}
    />
  );

  return (
    <>
      <h2 className="truncate text-xs font-semibold" title={name}>
        {name}
      </h2>
      <p className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted">
        <span
          aria-hidden="true"
          className="h-2 w-2 shrink-0 rounded-full"
          style={{ background: place === null || place.members < 2 ? UNPAIRED_COLOUR : groupColour(place.group) }}
        />
        {place === null || place.members < 2
          ? t("assembly.unpaired")
          : `${t("assembly.group", { n: place.group })} · ${t("assembly.pieces", { count: place.members })}`}
      </p>

      <Fields fields={fields} />

      <Heading>{t("assembly.joins_title")}</Heading>
      {total === 0 ? (
        <p className="mt-1 text-[11px] text-muted">{t("assembly.no_joins")}</p>
      ) : (
        <>
          {(["confirmed", "probable"] as const).map((band) =>
            bands[band].length === 0 ? null : (
              <div key={band} className="mt-1">
                <p className="px-1 text-[10px] text-muted">{t(`assembly.band_${band}`)}</p>
                {bands[band].map(rowOf)}
              </div>
            ),
          )}
          {bands.rejected.length === 0 ? null : (
            <div className="mt-1">
              <button
                type="button"
                aria-expanded={showRejected}
                onClick={() => {
                  setShowRejected(!showRejected);
                }}
                className={clsx("rounded px-1 text-[10px] text-muted hover:text-text", FOCUS_RING)}
              >
                {`${showRejected ? "▾" : "▸"} ${t("assembly.band_rejected")} · ${String(bands.rejected.length)}`}
              </button>
              {showRejected ? bands.rejected.map(rowOf) : null}
            </div>
          )}
        </>
      )}
      <p className="mt-2 text-[10px] leading-snug text-muted">{t("assembly.decide_hint")}</p>
    </>
  );
}

/** The selected group: what is in it, how many joins hold it together, and «Показать одну». */
function GroupInspector({
  assembly,
  index,
  runId,
}: {
  assembly: AssemblyDto;
  index: number;
  /** The run this group belongs to; `null` while none is selected, which takes Blender away. */
  runId: string | null;
}) {
  const { t } = useTranslation();
  const blenderBusy = useExport((state) => state.blenderBusy);
  const group = assembly.groups[index];
  if (group === undefined) {
    // The run changed under the selection — the inspector says nothing rather than about nothing.
    return <p className="px-2 py-3 text-[11px] text-muted">{t("input.nothing_selected")}</p>;
  }
  const inside = new Set(group.members);
  const joins = assembly.joins.filter((join) => inside.has(join.a) && inside.has(join.b)).length;

  return (
    <>
      <h2 className="flex items-center gap-1.5 text-xs font-semibold">
        <span
          aria-hidden="true"
          className="h-2 w-2 shrink-0 rounded-full"
          style={{ background: groupColour(index) }}
        />
        {t("assembly.group_title", { n: index })}
      </h2>
      <p className="mt-0.5 text-[11px] text-muted">
        {`${t("assembly.pieces", { count: group.members.length })} · ${t("assembly.joins", { count: joins })}`}
      </p>

      <div className="mt-2">
        <Button
          onClick={() => {
            useUi.getState().setAssemblyLayout("single", index);
          }}
        >
          {t("assembly.show_one")}
        </Button>
      </div>

      {/* A §9.2's «эта группа»: the whole point of opening one group in Blender is that it is
          the group being looked at, so the two resolutions are offered here and not in a menu
          at the other end of the window. */}
      {runId === null ? null : (
        <>
          <Heading>{t("blender.menu")}</Heading>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {(["full", "display"] as const).map((resolution) => (
              <Button
                key={resolution}
                disabled={blenderBusy}
                onClick={() => {
                  void useExport.getState().openInBlender(runId, { kind: "group", index }, resolution);
                }}
              >
                {t(`blender.${resolution}`)}
              </Button>
            ))}
          </div>
        </>
      )}

      <Heading>{t("assembly.members")}</Heading>
      {group.members.map((name) => (
        <button
          key={name}
          type="button"
          title={name}
          onClick={() => {
            useUi.getState().selectFragment(name);
            useUi.getState().flyToFragment(name);
          }}
          className={clsx("flex w-full items-center rounded px-1 py-0.5 text-left hover:bg-panel-2", FOCUS_RING)}
        >
          <span className="min-w-0 truncate text-[11px]">{name}</span>
        </button>
      ))}
    </>
  );
}

/** The run itself, when neither a fragment nor a group is chosen: what it found and what it cost. */
function RunSummary({ run, t }: { run: RunFile; t: TFunction }) {
  const counts = run.counts;
  const duration = took(run, t);
  const fields: Field[] = [];
  if (counts !== null) {
    fields.push(
      { key: "fragments", label: t("assembly.field_fragments"), value: String(counts.fragments) },
      { key: "pairs", label: t("assembly.field_pairs"), value: String(counts.pairs) },
      { key: "candidates", label: t("assembly.field_candidates"), value: String(counts.candidates) },
      { key: "confirmed", label: t("assembly.field_confirmed"), value: String(counts.confirmed) },
      { key: "probable", label: t("assembly.field_probable"), value: String(counts.probable) },
      { key: "groups", label: t("assembly.field_groups"), value: String(counts.groups) },
      { key: "unassembled", label: t("assembly.field_unassembled"), value: String(counts.unassembled) },
    );
  }
  if (run.engine !== null) {
    fields.push({ key: "backend", label: t("assembly.field_backend"), value: run.engine.backend });
  }
  if (duration !== null) {
    fields.push({ key: "duration", label: t("assembly.field_duration"), value: duration });
  }

  return (
    <>
      <h2 className="text-xs font-semibold">{t("assembly.summary")}</h2>
      <p className="mt-0.5 text-[11px] text-muted tabular-nums">{when(run.created)}</p>
      {fields.length === 0 ? <p className="mt-2 text-[11px] text-muted">{t("assembly.no_counts")}</p> : null}
      <Fields fields={fields} />

      {counts === null || counts.timings.length === 0 ? null : (
        <>
          <Heading>{t("assembly.stages")}</Heading>
          <Fields
            fields={counts.timings.map((row) => ({
              key: row.stage,
              // A stage the window has no word for is shown under the engine's own name rather
              // than as a raw key: the set grows with the engine (M4.3 found `objects`).
              label: t(`stage.${row.stage}`, { defaultValue: row.stage }),
              value: elapsedText(row.seconds * 1000),
            }))}
          />
        </>
      )}
    </>
  );
}

/**
 * The right pane of the «Сборка» mode (A §7.3): the inspector of whatever is chosen — a
 * fragment, a group, or, with nothing chosen, the run itself.
 *
 * One pane and three contents rather than three panes, because they answer one question at three
 * scales and the user swaps between them by clicking in the tree or in the viewport.
 */
export default function AssemblyRight({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const assembly = useAssembly((state) => state.assembly);
  const candidates = useAssembly((state) => state.candidates);
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  const selectedFragment = useUi((state) => state.selectedFragment);
  const selectedGroup = useUi((state) => state.assemblyGroup);
  const sessionRunId = useReview((state) => state.runId);
  const live = useReview((state) => state.decisions);

  // What has already been decided about this run. While a session is up it is the session's own
  // list — the one the undo stack moves and the one being filed — and while there is none it is
  // the run's `decisions.json`: a review done in the «Ревью» mode closes its session on the way
  // out, and an inspector that showed those pairs as undecided would be lying about work the
  // user has done. A run nobody has reviewed answers an empty list and not an error.
  //
  // Read again whenever a session comes or goes, and not once per run: a session that *ends* —
  // left behind, or the worker taken by a run (A §10) — takes its list with it, and the file it
  // filed before every apply is then the only place those verdicts still are.
  const [filed, setFiled] = useState<readonly Decision[]>([]);
  useEffect(() => {
    if (selectedRunId === null) {
      setFiled([]);
      return undefined;
    }
    let gone = false;
    void api.runDecisions(selectedRunId).then(
      (file) => {
        if (!gone) {
          setFiled(file.decisions);
        }
      },
      () => {
        // A file that will not be read is the «Ревью» mode's banner to put up, not this pane's:
        // here it only means there is nothing to show as decided.
        if (!gone) {
          setFiled([]);
        }
      },
    );
    return () => {
      gone = true;
    };
  }, [selectedRunId, sessionRunId]);

  const run = view.runs.find((row) => row.run.id === selectedRunId)?.run;
  const decisions = sessionRunId !== null && sessionRunId === selectedRunId ? live : filed;

  return (
    <div className="flex h-full flex-col overflow-y-auto p-2">
      {assembly !== null && selectedFragment !== null ? (
        <FragmentInspector
          view={view}
          assembly={assembly}
          candidates={candidates}
          name={selectedFragment}
          runId={selectedRunId}
          decisions={decisions}
        />
      ) : assembly !== null && selectedGroup !== null ? (
        <GroupInspector assembly={assembly} index={selectedGroup} runId={selectedRunId} />
      ) : run !== undefined ? (
        <RunSummary run={run} t={t} />
      ) : (
        <p className="px-2 py-3 text-[11px] text-muted">{t("input.nothing_selected")}</p>
      )}
    </div>
  );
}
