import clsx from "clsx";
import type { TFunction } from "i18next";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { api } from "../ipc";
import type { Unlisten } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { RunSpec } from "../ipc/bindings/RunSpec";
import type { StaleDiff } from "../ipc/bindings/StaleDiff";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import AssemblyCentre from "../modes/assembly/AssemblyCentre";
import AssemblyLeft from "../modes/assembly/AssemblyLeft";
import AssemblyRight from "../modes/assembly/AssemblyRight";
import InputCentre from "../modes/input/InputCentre";
import InputLeft from "../modes/input/InputLeft";
import InputRight from "../modes/input/InputRight";
import { useAssembly } from "../state/assembly";
import { useJobs } from "../state/jobs";
import type { Status } from "../state/status";
import { deriveStatus } from "../state/status";
import type { Mode } from "../state/ui";
import { useUi } from "../state/ui";
import { useWorkspace } from "../state/workspace";
import Button from "../ui/Button";
import type { BannerAction, BannerProps } from "./Banner";
import Banner from "./Banner";
import LaunchSheet from "./LaunchSheet";
import LogDrawer from "./LogDrawer";
import RunOverlay from "./RunOverlay";
import StatusLine from "./StatusLine";
import TopBar, { assembledRun, pickAndLinkInput } from "./TopBar";

/** What the active mode puts in the three places of the frame (A §7.3). */
interface Panes {
  left: ReactNode;
  centre: ReactNode;
  right: ReactNode;
}

/**
 * The «Вход» pane is wider than the others because it holds a grid of thumbnails; the tree of
 * groups that «Сборка» will put on the left needs less (A §7.1's 300 px).
 */
function leftWidth(mode: Mode): string {
  return mode === "input" ? "w-[360px]" : "w-[300px]";
}

/**
 * What a mode supplies until it supplies something. «Ревью» is milestone 5's and is drawn as a
 * disabled tab that cannot be entered, so its panes stay empty rather than pretending to be a
 * screen (A §7.3).
 */
const NO_PANES: Panes = { left: null, centre: null, right: null };

/** What the chosen mode puts in the three places. */
function panesOf(mode: Mode, view: WorkspaceView): Panes {
  switch (mode) {
    case "input":
      return {
        left: <InputLeft view={view} />,
        centre: <InputCentre view={view} />,
        right: <InputRight view={view} />,
      };
    case "assembly":
      return {
        left: <AssemblyLeft />,
        centre: <AssemblyCentre view={view} />,
        right: <AssemblyRight view={view} />,
      };
    case "review":
      return NO_PANES;
  }
}

/** «+3 файла, −1, 2 изменены» — A §5's «устарел» row, in the words the mock-up writes it in. */
function staleText(diff: StaleDiff, t: TFunction): string {
  return [
    diff.added.length > 0 ? t("banner.stale_added", { count: diff.added.length }) : null,
    diff.removed.length > 0 ? t("banner.stale_removed", { n: diff.removed.length }) : null,
    diff.changed.length > 0 ? t("banner.stale_changed", { count: diff.changed.length }) : null,
    diff.excluded_added.length > 0 ? t("banner.stale_excluded", { count: diff.excluded_added.length }) : null,
    diff.excluded_removed.length > 0 ? t("banner.stale_included", { count: diff.excluded_removed.length }) : null,
  ]
    .filter((part) => part !== null)
    .join(", ");
}

/**
 * The centre of an empty workspace (A §5's first row): a folder of scans is what the app needs
 * before it can do anything at all, so the invitation takes the whole viewport and accepts both
 * a click and a folder dropped on the window.
 */
function DropZone() {
  const { t } = useTranslation();

  useEffect(() => {
    let stop: Unlisten | null = null;
    let gone = false;
    void api
      .onFolderDropped((path) => {
        void useWorkspace.getState().linkInput(path);
      })
      .then(
        (off) => {
          if (gone) {
            off();
          } else {
            stop = off;
          }
        },
        (e: unknown) => {
          useWorkspace.getState().setError(toCommandError(e));
        },
      );
    return () => {
      gone = true;
      stop?.();
    };
  }, []);

  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="flex w-[420px] flex-col items-center gap-3 rounded-lg border-2 border-dashed border-border px-6 py-10 text-center">
        <p className="text-sm">{t("drop.title")}</p>
        <p className="font-mono text-xs text-muted">{t("drop.formats")}</p>
        <Button
          variant="primary"
          size="md"
          onClick={() => {
            pickAndLinkInput(t("action.pick_input_title"));
          }}
        >
          {t("action.pick_input")}
        </Button>
      </div>
    </div>
  );
}

/**
 * The window with a workspace in it (`main-layout.html`, variant A): the top bar, whatever the
 * state of the workspace has to say, three panes of which the two sides collapse, and the status
 * line. With both sides collapsed the centre fills the window, which is what the user asked for.
 */
export default function Frame({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const mode = useUi((state) => state.mode);
  const leftOpen = useUi((state) => state.leftOpen);
  const rightOpen = useUi((state) => state.rightOpen);
  const logOpen = useUi((state) => state.logOpen);
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  const error = useWorkspace((state) => state.error);
  const runError = useAssembly((state) => state.error);
  const lastFailure = useJobs((state) => state.lastFailure);

  // What the launch sheet should open with, over the last run's own sheet; `null` is «closed».
  // The frame owns it because more than one place opens the sheet — the top bar's action now,
  // A §10's «Повторить на CPU» next — and only one of them may be up at a time.
  const [sheet, setSheet] = useState<Partial<RunSpec> | null>(null);

  // Milestone 3 loads no assembly, so no group of one can be waiting for refinement (A §8.4).
  const status: Status = deriveStatus(view, selectedRunId, false);

  // The «Сборка» mode is only a mode while there is a finished run to show. A run deleted, or a
  // selection cleared, takes its tab away — and leaving the window standing on a mode whose tab
  // is disabled would be three empty panes with no way back but the keyboard.
  const assembled = assembledRun(view, selectedRunId) !== null;
  useEffect(() => {
    if (!assembled && useUi.getState().mode === "assembly") {
      useUi.getState().setMode("input");
    }
  }, [assembled]);

  const panes: Panes = panesOf(assembled || mode !== "assembly" ? mode : "input", view);

  /** A §5's «Показать лог»: the pull-up panel, never closed by an action that says «show». */
  const showLog = (): void => {
    useUi.getState().setLogOpen(true);
  };
  /** «Перегенерировать…» / «Повторить…» / «Повторить на CPU» — all one sheet, differently filled. */
  const again = (patch: Partial<RunSpec>): BannerAction["onClick"] => {
    return () => {
      setSheet(patch);
    };
  };

  // One banner from the status, because a status is one value: A §5's table gives each of these
  // rows its own sentence, and two of them can never be true at once.
  //
  // The banner never repeats the action the top bar is already showing for the same status. A §5
  // gives every row one primary action and the top bar is where it lives; a second button with
  // the same word on it, twenty pixels below the first, reads as a fault in the window rather
  // than as a second offer. So «Перегенерировать…» and «Указать папку заново» stay up there, and
  // the banner carries only what the top bar has no room for — the log, and the one repeat that
  // differs from it («Повторить на CPU»).
  const banners: BannerProps[] = [];
  switch (status.kind) {
    case "input_missing":
      banners.push({ tone: "danger", text: t("banner.input_missing") });
      break;
    case "stale":
      banners.push({ tone: "warn", text: t("banner.stale", { diff: staleText(status.diff, t) }) });
      break;
    case "failed":
      banners.push({
        tone: "danger",
        // The kind is the part the user can act on; the engine's own words go underneath it.
        text: t(`failure.${status.failKind}`),
        detail: status.message,
        actions: [
          { label: t("action.show_log"), onClick: showLog },
          // A §10: an adapter that gave up is the one failure with a different run to offer, and
          // the sheet opens on it rather than making the user find «Вычисления» themselves.
          ...(status.failKind === "gpu"
            ? [{ label: t("action.repeat_cpu"), onClick: again({ backend: "cpu" }) }]
            : []),
        ],
      });
      break;
    case "cancelled":
    case "interrupted":
      banners.push({
        tone: "warn",
        text: t(`banner.${status.kind}`),
        actions: [{ label: t("action.show_log"), onClick: showLog }],
      });
      break;
    default:
      break;
  }

  // A cancelled job is not a failure: the user stopped it, and the button they pressed is all the
  // report they need (A §10's table gives `cancelled` no message of its own). Nor is a run that
  // the status has already reported above — the same ending twice would read as two of them.
  const reported = status.kind === "failed" || status.kind === "cancelled" || status.kind === "interrupted";
  if (lastFailure !== null && lastFailure.kind !== "cancelled" && !reported) {
    // A preparation that failed leaves the input unprepared, and the automatic one will not try
    // again on its own (App.tsx retries only once per state of the folder) — so the offer to
    // repeat it by hand is the banner's, and only while there is something left to prepare.
    const retry: BannerAction[] =
      status.kind === "unprepared"
        ? [
            {
              label: t("action.retry_prepare"),
              onClick: () => {
                void useJobs.getState().start();
              },
            },
          ]
        : [];
    banners.push({
      tone: "danger",
      text: t("banner.job_failed"),
      detail: lastFailure.message,
      actions: [...retry, { label: t("action.show_log"), onClick: showLog }],
    });
  }
  if (error !== null) {
    banners.push({
      tone: "danger",
      text: t(`error.${error.kind}`),
      detail: error.message,
      onDismiss: () => {
        useWorkspace.getState().setError(null);
      },
    });
  }
  // A §10: a run whose `assembly.json` or `candidates.json` is there but will not be read — a
  // half-written file, a workspace closed under the load — is a refusal, and without saying it
  // the window would draw a run that assembled twelve fragments as one that assembled nothing,
  // or a fragment with eleven scored joins as one with none. A file a run never wrote is not
  // this: [`useAssembly`] turns that (kind `io`) into no error at all.
  //
  // Here and not only in the «Сборка» centre, because either of the two files can be the one
  // that failed and the other can have arrived: a run drawn in 3D with its candidates missing
  // has nothing empty on the screen to hang the sentence on. The wording is the run's own and
  // not `error.<kind>`'s, whose `json` says «the workspace file», which this is not.
  if (runError !== null) {
    banners.push({
      tone: "danger",
      text: t("banner.run_unreadable"),
      detail: runError.message,
      onDismiss: () => {
        useAssembly.getState().dismissError();
      },
    });
  }

  return (
    <div className="flex h-full flex-col bg-bg">
      <TopBar
        view={view}
        status={status}
        onAssemble={() => {
          setSheet({});
        }}
      />
      {banners.map((banner, i) => (
        // Two banners of the same tone can only differ by their text, which is what keys them.
        <Banner key={`${banner.tone}-${String(i)}-${banner.text}`} {...banner} />
      ))}

      <div className="flex min-h-0 flex-1">
        {leftOpen ? (
          <aside className={clsx("shrink-0 overflow-hidden border-r border-border bg-panel", leftWidth(mode))}>
            {panes.left}
          </aside>
        ) : null}

        <main
          className={clsx(
            "relative min-w-0 flex-1 overflow-hidden",
            status.kind === "empty" ? "bg-bg" : "bg-viewport text-viewport-text",
          )}
        >
          {status.kind === "empty" ? <DropZone /> : panes.centre}
          {status.kind === "running" ? <RunOverlay view={view} /> : null}
        </main>

        {rightOpen ? (
          <aside className="w-[260px] shrink-0 overflow-hidden border-l border-border bg-panel">{panes.right}</aside>
        ) : null}
      </div>

      {logOpen ? <LogDrawer view={view} /> : null}

      <StatusLine view={view} status={status} />

      {sheet === null ? null : (
        <LaunchSheet
          view={view}
          patch={sheet}
          onClose={() => {
            setSheet(null);
          }}
        />
      )}
    </div>
  );
}
