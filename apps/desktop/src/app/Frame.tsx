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
import InputCentre from "../modes/input/InputCentre";
import InputLeft from "../modes/input/InputLeft";
import InputRight from "../modes/input/InputRight";
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
import RunOverlay from "./RunOverlay";
import StatusLine from "./StatusLine";
import TopBar, { pickAndLinkInput } from "./TopBar";

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
 * What a mode supplies until it supplies something. Milestone 3 has only «Вход»; «Сборка» and
 * «Ревью» are drawn as disabled tabs and cannot be entered, so their panes stay empty rather
 * than pretending to be screens (A §7.3).
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
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  const error = useWorkspace((state) => state.error);
  const lastFailure = useJobs((state) => state.lastFailure);

  // What the launch sheet should open with, over the last run's own sheet; `null` is «closed».
  // The frame owns it because more than one place opens the sheet — the top bar's action now,
  // A §10's «Повторить на CPU» next — and only one of them may be up at a time.
  const [sheet, setSheet] = useState<Partial<RunSpec> | null>(null);

  // Milestone 3 loads no assembly, so no group of one can be waiting for refinement (A §8.4).
  const status: Status = deriveStatus(view, selectedRunId, false);
  const panes: Panes = panesOf(mode, view);

  const banners: BannerProps[] = [];
  if (status.kind === "input_missing") {
    banners.push({
      tone: "danger",
      text: t("banner.input_missing"),
      action: {
        label: t("action.relink"),
        onClick: () => {
          pickAndLinkInput(t("action.pick_input_title"));
        },
      },
    });
  }
  if (status.kind === "stale") {
    banners.push({ tone: "warn", text: t("banner.stale", { diff: staleText(status.diff, t) }) });
  }
  // A cancelled job is not a failure: the user stopped it, and the button they pressed is all the
  // report they need (A §10's table gives `cancelled` no message of its own).
  if (lastFailure !== null && lastFailure.kind !== "cancelled") {
    // A preparation that failed leaves the input unprepared, and the automatic one will not try
    // again on its own (App.tsx retries only once per state of the folder) — so the offer to
    // repeat it by hand is the banner's, and only while there is something left to prepare.
    const retry: BannerAction | undefined =
      status.kind === "unprepared"
        ? {
            label: t("action.retry_prepare"),
            onClick: () => {
              void useJobs.getState().start();
            },
          }
        : undefined;
    banners.push({ tone: "danger", text: t("banner.job_failed", { message: lastFailure.message }), action: retry });
  }
  if (error !== null) {
    banners.push({
      tone: "danger",
      text: `${t(`error.${error.kind}`)} · ${error.message}`,
      onDismiss: () => {
        useWorkspace.getState().setError(null);
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
