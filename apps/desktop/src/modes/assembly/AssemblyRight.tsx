import clsx from "clsx";
import type { TFunction } from "i18next";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { took, when } from "../../app/RunSelector";
import { elapsedText } from "../../app/StatusLine";
import type { AssemblyDto } from "../../ipc/bindings/AssemblyDto";
import type { CandidateRow } from "../../ipc/bindings/CandidateRow";
import type { RunFile } from "../../ipc/bindings/RunFile";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useAssembly } from "../../state/assembly";
import { useUi } from "../../state/ui";
import { useWorkspace } from "../../state/workspace";
import Button, { FOCUS_RING } from "../../ui/Button";
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

/** Which group of the run holds a fragment, and whether that «group» is a group of one. */
function placeOf(assembly: AssemblyDto, name: string): { group: number; members: number } | null {
  const at = assembly.groups.findIndex((row) => row.members.includes(name));
  const found = assembly.groups[at];
  return found === undefined ? null : { group: at, members: found.members.length };
}

/** One candidate of the selected fragment: who the partner is, and what the pair scored. */
function JoinRow({
  partner,
  row,
  onPick,
}: {
  partner: string;
  row: CandidateRow;
  onPick: (name: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <button
      type="button"
      onClick={() => {
        onPick(partner);
      }}
      title={partner}
      className={clsx("flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left hover:bg-panel-2", FOCUS_RING)}
    >
      <span aria-hidden="true" className={clsx("h-2 w-2 shrink-0 rounded-full", BAND_DOT[row.tier])} />
      <span className="min-w-0 flex-1 truncate text-[11px]">{partner}</span>
      {row.used ? <span className="shrink-0 text-[10px] text-muted">{t("assembly.used")}</span> : null}
      <span className="w-9 shrink-0 text-right text-[11px] tabular-nums">{formatDecimal(row.score)}</span>
    </button>
  );
}

/** A candidate as the inspector lists it: the partner's name, and the row it came from. */
interface Join {
  partner: string;
  row: CandidateRow;
}

/** Every candidate of `name`, best first inside each band (R §5.7's ranking key is `score`). */
function joinsOf(candidates: readonly CandidateRow[], name: string): Record<CandidateRow["tier"], Join[]> {
  const bands: Record<CandidateRow["tier"], Join[]> = { confirmed: [], probable: [], rejected: [] };
  for (const row of candidates) {
    if (row.a === name || row.b === name) {
      bands[row.tier].push({ partner: row.a === name ? row.b : row.a, row });
    }
  }
  for (const band of BANDS) {
    bands[band].sort((one, other) => other.row.score - one.row.score);
  }
  return bands;
}

/**
 * The selected fragment: where the run put it, what it is, and every pair it took part in by
 * band (A §7.3, A §8.3). The rejected ones are folded away, because they are the long tail and
 * the reason to open them is curiosity rather than work.
 *
 * Nothing here accepts or rejects anything: that is milestone 5's «Ревью», and a row with a
 * disabled «Подтвердить» on it would promise something this build cannot do.
 */
function FragmentInspector({
  view,
  assembly,
  candidates,
  name,
}: {
  view: WorkspaceView;
  assembly: AssemblyDto;
  candidates: readonly CandidateRow[];
  name: string;
}) {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const [showRejected, setShowRejected] = useState(false);

  const place = placeOf(assembly, name);
  const info = view.fragments.find((row) => row.name === name);
  const bands = joinsOf(candidates, name);
  const total = BANDS.reduce((sum, band) => sum + bands[band].length, 0);

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
    useUi.getState().selectFragment(partner);
    useUi.getState().flyToFragment(partner);
  };

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
                {bands[band].map((join) => (
                  <JoinRow key={`${join.row.a}-${join.row.b}`} partner={join.partner} row={join.row} onPick={pick} />
                ))}
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
              {showRejected
                ? bands.rejected.map((join) => (
                    <JoinRow key={`${join.row.a}-${join.row.b}`} partner={join.partner} row={join.row} onPick={pick} />
                  ))
                : null}
            </div>
          )}
        </>
      )}
      <p className="mt-2 text-[10px] leading-snug text-muted">{t("assembly.review_later")}</p>
    </>
  );
}

/** The selected group: what is in it, how many joins hold it together, and «Показать одну». */
function GroupInspector({ assembly, index }: { assembly: AssemblyDto; index: number }) {
  const { t } = useTranslation();
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

  const run = view.runs.find((row) => row.run.id === selectedRunId)?.run;

  return (
    <div className="flex h-full flex-col overflow-y-auto p-2">
      {assembly !== null && selectedFragment !== null ? (
        <FragmentInspector view={view} assembly={assembly} candidates={candidates} name={selectedFragment} />
      ) : assembly !== null && selectedGroup !== null ? (
        <GroupInspector assembly={assembly} index={selectedGroup} />
      ) : run !== undefined ? (
        <RunSummary run={run} t={t} />
      ) : (
        <p className="px-2 py-3 text-[11px] text-muted">{t("input.nothing_selected")}</p>
      )}
    </div>
  );
}
