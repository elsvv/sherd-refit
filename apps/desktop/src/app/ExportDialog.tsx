import clsx from "clsx";
import type { TFunction } from "i18next";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { api } from "../ipc";
import { toCommandError } from "../ipc/api";
import type { ExportWhat } from "../ipc/bindings/ExportWhat";
import type { RunFile } from "../ipc/bindings/RunFile";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { formatBytes, formatCount, middleEllipsis } from "../modes/input/format";
import { useAssembly } from "../state/assembly";
import { useExport } from "../state/export";
import { hasUnrefinedGroups } from "../state/status";
import { useUi } from "../state/ui";
import Button from "../ui/Button";
import Dialog from "../ui/Dialog";
import Meter from "../ui/Meter";
import type { RadioOption } from "../ui/Radio";
import Radio from "../ui/Radio";
import { formatDuration } from "./params";
import { assembledRun } from "./TopBar";

/** A §9.1's two kinds, as the dialog's radio group names them. */
type Kind = ExportWhat["kind"];

/** The three opt-ins of «Папка результата», which are the run's own `--placed-all` and friends. */
interface Opts {
  placed_all: boolean;
  merged_meshes: boolean;
  previews: boolean;
}

/** What the dialog is asking the session for, out of the two answers above it. */
function whatOf(kind: Kind, opts: Opts): ExportWhat {
  return kind === "tables" ? { kind: "tables" } : { kind: "folder", ...opts };
}

/**
 * One opt-in. A real `<input type="checkbox">` under the paint, for [`Radio`]'s reason: three
 * independent switches are what a checkbox *is*, and the browser already says «checked» to a
 * screen reader and toggles it with `Space`.
 */
function Check({
  label,
  hint,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  disabled: boolean;
  onChange: (on: boolean) => void;
}) {
  return (
    <label
      className={clsx(
        "flex cursor-default gap-2 border-b border-border px-2 py-1.5 last:border-b-0",
        "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-[-2px] has-[:focus-visible]:outline-accent",
        disabled && "opacity-45",
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => {
          onChange(e.target.checked);
        }}
        className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-[var(--color-accent)]"
      />
      <span className="min-w-0">
        <span className="block text-[11px]">{label}</span>
        <span className="block text-[10px] leading-snug text-muted">{hint}</span>
      </span>
    </label>
  );
}

/**
 * How long R §9 is likely to take over this run, out of its own `refine` stage (A §8.4's «≈30
 * с»), or `null` for a run that filed no timings. The run's figure is an over-estimate — it
 * refined every group and an export refines only what a decision left rough — which is the right
 * way round for a promise about a wait.
 */
function refineEstimate(run: RunFile | null, t: TFunction): string | null {
  const seconds = run?.counts?.timings.find((stage) => stage.stage === "refine")?.seconds;
  return seconds === undefined || seconds <= 0 ? null : formatDuration(seconds, t);
}

/**
 * A §9.1's «Экспорт»: what to write, where to write it, what the review has to do with it, and
 * then the writing itself.
 *
 * The dialog stays up for the whole export — its progress is the session's own events, read off
 * [`useExport`] — and for a refusal as well: «эта папка не пуста» is answered by choosing
 * another one, not by starting over. It is the one dialog of the app that goes on meaning
 * something after it is closed, so closing it while an export runs leaves the export running
 * and re-opening it finds it where it was.
 */
export default function ExportDialog({
  view,
  runId,
  onClose,
}: {
  view: WorkspaceView;
  /** The run being exported: the one the window is showing, as the top bar's action knows it. */
  runId: string;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const phase = useExport((state) => state.phase);
  const stage = useExport((state) => state.stage);
  const done = useExport((state) => state.done);
  const total = useExport((state) => state.total);
  const result = useExport((state) => state.result);
  const error = useExport((state) => state.error);
  const assembly = useAssembly((state) => state.assembly);

  const [kind, setKind] = useState<Kind>("folder");
  const [opts, setOpts] = useState<Opts>({ placed_all: false, merged_meshes: false, previews: false });
  /** Where it will be written; empty until the shell has suggested somewhere (A §9.1). */
  const [dest, setDest] = useState("");
  /** Whether that is the user's own answer, which no change of the kind may overwrite. */
  const [picked, setPicked] = useState(false);
  /** How many decisions the run carries — from its file, which is what the export will read. */
  const [decided, setDecided] = useState<{ accepted: number; rejected: number } | null>(null);

  const running = phase === "running";
  // What was exported is a receipt and not a form: once «Готово» is up, changing a checkbox
  // would be changing the description of a folder that is already on disk. A second export is
  // a second opening of the dialog.
  const frozen = running || phase === "done";
  const what = whatOf(kind, opts);

  // The suggestion, asked for when the dialog opens and again when the kind changes — the two
  // kinds are offered different folders so that one does not land inside the other's.
  useEffect(() => {
    if (picked) {
      return undefined;
    }
    let gone = false;
    // Only the kind is in the folder's name (`exports/<date>_folder`), so the three opt-ins are
    // not asked about and changing one does not move the destination under the user.
    const asked: ExportWhat =
      kind === "tables"
        ? { kind: "tables" }
        : { kind: "folder", placed_all: false, merged_meshes: false, previews: false };
    void api.exportDefaultDest(asked).then(
      (suggested) => {
        if (!gone) {
          setDest(suggested);
        }
      },
      () => {
        // No suggestion is «Выбрать…» and nothing else; the dialog still exports where told.
      },
    );
    return () => {
      gone = true;
    };
  }, [kind, picked]);

  // A §9.1's «экспорт отражает ревью»: read off the run's own `decisions.json`, because that is
  // the file the session will read — not the window's draft, which may not have been filed yet.
  useEffect(() => {
    let gone = false;
    void api.runDecisions(runId).then(
      (file) => {
        if (!gone) {
          const accepted = file.decisions.filter((one) => one.verdict === "accept").length;
          setDecided({ accepted, rejected: file.decisions.length - accepted });
        }
      },
      () => {
        // A run whose decisions will not be read has nothing to promise about them here.
      },
    );
    return () => {
      gone = true;
    };
  }, [runId]);

  const kinds: RadioOption<Kind>[] = [
    { value: "folder", label: t("export.what_folder"), hint: t("export.what_folder_hint") },
    { value: "tables", label: t("export.what_tables"), hint: t("export.what_tables_hint") },
  ];

  const pick = (): void => {
    void (async () => {
      try {
        const path = await api.pickFolder(t("export.pick_title"));
        if (path !== null) {
          setDest(path);
          setPicked(true);
        }
      } catch (e) {
        useExport.setState({ error: toCommandError(e) });
      }
    })();
  };

  const close = (): void => {
    // A finished or a refused export has been read by now; one that is still going is left
    // alone, and re-opening the dialog finds it where it was.
    if (phase === "done" || phase === "failed") {
      useExport.getState().reset();
    }
    onClose();
  };

  const run = assembledRun(view, runId);
  const estimate = refineEstimate(run, t);
  const unrefined = hasUnrefinedGroups(assembly);
  const stageText = stage === null ? t("export.starting") : t(`stage.${stage}`, { defaultValue: stage });

  return (
    <Dialog
      title={t("export.title")}
      width={520}
      onClose={close}
      footer={
        <>
          <span className="min-w-0 flex-1 truncate text-[11px] text-muted">
            {running ? t("export.working") : ""}
          </span>
          {phase === "done" && result !== null ? (
            <Button
              onClick={() => {
                void useExport.getState().reveal(result.dest);
              }}
            >
              {t("export.show")}
            </Button>
          ) : null}
          <Button variant={phase === "done" ? "primary" : "ghost"} onClick={close}>
            {phase === "done" ? t("export.close") : t("export.cancel")}
          </Button>
          {phase === "done" ? null : (
            <Button
              variant="primary"
              disabled={running || dest.length === 0}
              onClick={() => {
                void useExport.getState().start(runId, what, dest);
              }}
            >
              {t("export.submit")}
            </Button>
          )}
        </>
      }
    >
      <Radio
        name="sherd-export-what"
        legend={t("export.what")}
        value={kind}
        options={kinds}
        disabled={frozen}
        onChange={setKind}
      />

      {kind === "folder" ? (
        <div className="mt-2 rounded-md border border-border">
          <Check
            label={t("export.opt_placed_all")}
            hint={t("export.opt_placed_all_hint")}
            checked={opts.placed_all}
            disabled={frozen}
            onChange={(on) => {
              setOpts({ ...opts, placed_all: on });
            }}
          />
          <Check
            label={t("export.opt_merged")}
            hint={t("export.opt_merged_hint")}
            checked={opts.merged_meshes}
            disabled={frozen}
            onChange={(on) => {
              setOpts({ ...opts, merged_meshes: on });
            }}
          />
          <Check
            label={t("export.opt_previews")}
            hint={t("export.opt_previews_hint")}
            checked={opts.previews}
            disabled={frozen}
            onChange={(on) => {
              setOpts({ ...opts, previews: on });
            }}
          />
        </div>
      ) : null}

      <section className="mt-3">
        <h3 className="mb-1 text-[10px] tracking-wide text-muted uppercase">{t("export.where")}</h3>
        <div className="flex items-center gap-2">
          {/* Cut in the middle and not at the end: every export of one workspace shares the
              first half of its path, and it is the last folder that says which one this is. */}
          <span
            title={dest}
            className="min-w-0 flex-1 rounded-md border border-border bg-panel-2 px-1.5 py-1 font-mono text-[11px]"
          >
            {dest === "" ? "…" : middleEllipsis(dest, 58)}
          </span>
          <Button className="shrink-0" disabled={frozen} onClick={pick}>
            {t("export.pick")}
          </Button>
        </div>
        <p className="mt-1 text-[10px] leading-snug text-muted">{t("export.dest_hint")}</p>
      </section>

      <section className="mt-3">
        <h3 className="mb-1 text-[10px] tracking-wide text-muted uppercase">{t("export.reflects")}</h3>
        <p className="text-[11px] leading-snug">
          {decided === null || decided.accepted + decided.rejected === 0
            ? t("export.reflects_none")
            : t("export.reflects_some", {
                decided: [
                  decided.accepted > 0 ? t("review.draft_accepted", { n: decided.accepted }) : null,
                  decided.rejected > 0 ? t("review.draft_rejected", { n: decided.rejected }) : null,
                ]
                  .filter((part) => part !== null)
                  .join(" · "),
              })}
        </p>
        {unrefined ? (
          <p className="mt-0.5 text-[11px] leading-snug text-muted">
            {estimate === null ? t("export.refine_first") : t("export.refine_first_about", { time: estimate })}
          </p>
        ) : null}
      </section>

      {running ? (
        <section className="mt-3">
          {/* The engine's own stage names, which the status line already has words for: the
              collection loading, R §9 over the rough groups, then the writers. A stage this
              build has no word for is said as the engine named it rather than as a raw key. */}
          <p className="text-[11px]">{stageText}</p>
          <Meter className="mt-1 w-full" value={done} max={total} label={stageText} />
          {total > 0 ? (
            <p className="mt-0.5 text-[10px] text-muted tabular-nums">
              {t("export.of", { done: formatCount(done, language), total: formatCount(total, language) })}
            </p>
          ) : null}
        </section>
      ) : null}

      {phase === "done" && result !== null ? (
        <section className="mt-3 rounded-md border border-accent bg-select px-2 py-1.5">
          <p className="text-[11px]">
            {t("export.done", {
              files: t("export.files", { count: result.files }),
              size: formatBytes(result.bytes, language),
            })}
          </p>
          <p className="mt-0.5 truncate font-mono text-[10px] text-muted" title={result.dest}>
            {result.dest}
          </p>
        </section>
      ) : null}

      {phase === "failed" && error !== null ? (
        <section className="mt-3 rounded-md border border-danger px-2 py-1.5">
          <p className="text-[11px] text-danger">{t("export.failed")}</p>
          <p className="mt-0.5 text-[10px] leading-snug break-words text-muted">{error.message}</p>
        </section>
      ) : null}
    </Dialog>
  );
}
