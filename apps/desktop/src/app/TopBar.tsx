import clsx from "clsx";
import { ChevronDown, PanelLeft, PanelRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { api } from "../ipc";
import { toCommandError } from "../ipc/api";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { useJobs } from "../state/jobs";
import type { Status } from "../state/status";
import type { Language, Mode, Theme } from "../state/ui";
import { useUi } from "../state/ui";
import { useWorkspace } from "../state/workspace";
import type { ButtonVariant } from "../ui/Button";
import Button, { FOCUS_RING } from "../ui/Button";
import Chip from "../ui/Chip";
import Dialog from "../ui/Dialog";
import IconButton from "../ui/IconButton";

/** The three modes of A §7.3, left to right, which is also the order of the `1` `2` `3` keys. */
export const MODES: readonly Mode[] = ["input", "assembly", "review"];

/**
 * Which modes can be entered. Milestone 3 has no assembly and no review, so their tabs are drawn
 * disabled with a tooltip rather than hidden: the frame is the same frame throughout (A §7.1),
 * and hiding them would make the window look like a different app once they arrive.
 */
export function modeEnabled(mode: Mode): boolean {
  return mode === "input";
}

/**
 * Asks for a folder of scans and links it. A §5 gives the same action three homes — the top bar
 * of an empty workspace, the drop zone in its centre and the «Указать папку заново» of a folder
 * that has moved — so they share one function, and a cancelled dialog or a refusal is handled
 * once rather than three times.
 */
export function pickAndLinkInput(title: string): void {
  void (async () => {
    try {
      const path = await api.pickFolder(title);
      if (path !== null) {
        await useWorkspace.getState().linkInput(path);
      }
    } catch (e) {
      useWorkspace.getState().setError(toCommandError(e));
    }
  })();
}

/** The primary action of the top bar's right end, as A §5's table assigns it to the status. */
interface PrimaryAction {
  label: string;
  variant: ButtonVariant;
  /** Shown as a `title` on the button and on the span around it, so a disabled one still tells why. */
  tooltip?: string;
  disabled?: boolean;
  run: () => void;
}

/** The menu behind the workspace name: where to go, and the two settings A §7.4 keeps. */
function WorkspaceMenu({ name }: { name: string }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const theme = useUi((state) => state.theme);
  const language = useUi((state) => state.language);

  // A menu that stays open after the pointer has gone elsewhere is a menu in the way; Escape is
  // the keyboard's way out of the same corner.
  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (e: PointerEvent) => {
      if (e.target instanceof Node && box.current?.contains(e.target) !== true) {
        setOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const openAnother = () => {
    setOpen(false);
    void (async () => {
      try {
        const path = await api.pickFolder(t("welcome.open_title"));
        if (path !== null) {
          await useWorkspace.getState().open(path);
        }
      } catch (e) {
        useWorkspace.getState().setError(toCommandError(e));
      }
    })();
  };

  const themes: Theme[] = ["system", "light", "dark"];
  const languages: Language[] = ["ru", "en"];

  return (
    <div className="relative" ref={box}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        // The accessible name is the workspace's own, which is what the button says; what the
        // button *is* comes from `aria-haspopup`, and the tooltip spells it out for a mouse.
        title={t("topbar.menu")}
        onClick={() => {
          setOpen(!open);
        }}
        className={clsx(
          "inline-flex h-6 max-w-56 items-center gap-1 rounded-md px-1.5 font-semibold hover:bg-panel",
          FOCUS_RING,
        )}
      >
        <span className="truncate">{name}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {open ? (
        <div
          role="menu"
          className="absolute top-full left-0 z-20 mt-1 w-60 rounded-md border border-border bg-panel p-1 shadow-lg"
        >
          <MenuItem onClick={openAnother}>{t("menu.open_other")}</MenuItem>
          <MenuItem
            onClick={() => {
              setOpen(false);
              void useWorkspace.getState().close();
            }}
          >
            {t("menu.close")}
          </MenuItem>
          <MenuLabel>{t("menu.language")}</MenuLabel>
          {languages.map((code) => (
            <MenuItem
              key={code}
              checked={language === code}
              onClick={() => {
                useUi.getState().setLanguage(code);
              }}
            >
              {t(`menu.language_${code}`)}
            </MenuItem>
          ))}
          <MenuLabel>{t("menu.theme")}</MenuLabel>
          {themes.map((choice) => (
            <MenuItem
              key={choice}
              checked={theme === choice}
              onClick={() => {
                useUi.getState().setTheme(choice);
              }}
            >
              {t(`menu.theme_${choice}`)}
            </MenuItem>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** A heading inside the menu; not focusable, because there is nothing to do with it. */
function MenuLabel({ children }: { children: string }) {
  return <div className="mt-1 px-2 pt-1 pb-0.5 text-[10px] tracking-wide text-muted uppercase">{children}</div>;
}

/** One row of the menu. With `checked` given it is one of a set, and says which one is on. */
function MenuItem({
  children,
  checked,
  onClick,
}: {
  children: string;
  checked?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role={checked === undefined ? "menuitem" : "menuitemradio"}
      aria-checked={checked}
      onClick={onClick}
      className={clsx("flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs hover:bg-panel-2", FOCUS_RING)}
    >
      <span className="w-3 text-accent">{checked === true ? "•" : ""}</span>
      <span className="truncate">{children}</span>
    </button>
  );
}

/**
 * The 40 px bar of `main-layout.html`, variant A: the workspace and its menu, the run selector,
 * the three mode tabs with their own counts, the two pane toggles, and the one action A §5's
 * table gives the current status.
 */
export default function TopBar({
  view,
  status,
  onAssemble,
}: {
  view: WorkspaceView;
  status: Status;
  /** Opens A §7.4's launch sheet, with whatever the action wants changed in it. */
  onAssemble: () => void;
}) {
  const { t } = useTranslation();
  const mode = useUi((state) => state.mode);
  const leftOpen = useUi((state) => state.leftOpen);
  const rightOpen = useUi((state) => state.rightOpen);

  // «Отменить» asks before it stops a run, and says «Останавливается…» until the worker is gone:
  // D §5's cancel is a flag the job reads at its next unit of work, so a click is a request and
  // not an ending, and a button that stayed «Отменить» would invite a second one.
  const [asking, setAsking] = useState(false);
  const [stopping, setStopping] = useState(false);
  const job = view.job;
  useEffect(() => {
    if (job === null) {
      setAsking(false);
      setStopping(false);
    }
  }, [job]);

  const warnings = view.fragments.filter((fragment) => fragment.warnings.length > 0).length;

  const pickInput = () => {
    pickAndLinkInput(t("action.pick_input_title"));
  };

  const stop = () => {
    setAsking(false);
    setStopping(true);
    void useJobs.getState().cancel();
  };

  const primary = ((): PrimaryAction | null => {
    switch (status.kind) {
      case "empty":
        return { label: t("action.pick_input"), variant: "primary", run: pickInput };
      case "input_missing":
        return { label: t("action.relink"), variant: "primary", run: pickInput };
      case "preparing":
        // A preparation loses a few minutes of thumbnails and can be asked for again from the
        // banner; only a run is worth a question of its own (A §6).
        return { label: stopping ? t("run.stopping") : t("action.cancel"), variant: "danger", disabled: stopping, run: stop };
      case "running":
        return {
          label: stopping ? t("run.stopping") : t("action.cancel"),
          variant: "danger",
          disabled: stopping,
          run: () => {
            setAsking(true);
          },
        };
      case "unprepared":
        return {
          label: t("action.prepare"),
          variant: "primary",
          run: () => {
            void useJobs.getState().start();
          },
        };
      case "ready":
        return { label: t("action.assemble"), variant: "primary", run: onAssemble };
      case "current":
        // A §5: with the result current the action is still there, in the second voice — nothing
        // needs doing, and re-running the same input is a deliberate act.
        return { label: t("action.regenerate"), variant: "ghost", run: onAssemble };
      case "stale":
        return { label: t("action.regenerate"), variant: "primary", run: onAssemble };
      case "failed":
      case "cancelled":
      case "interrupted":
        return { label: t("action.repeat"), variant: "primary", run: onAssemble };
      case "draft":
        // «Уточнить позы» is milestone 5's: it refines the poses of the assembly the review has
        // changed, and there is no review yet.
        return null;
    }
  })();

  return (
    <header className="flex h-10 shrink-0 items-center gap-2 border-b border-border bg-panel-2 px-2">
      <WorkspaceMenu name={view.name} />

      {/* The run selector. Milestone 3 writes no runs, so it is a label until milestone 4 gives
          it the history dropdown of A §7.4. */}
      <Chip disabled title={t("topbar.no_runs")}>
        {view.runs[0]?.run.id ?? t("topbar.no_runs")}
      </Chip>

      <span className="flex-1" />

      <nav className="flex items-center gap-1" aria-label={t("topbar.modes")}>
        {MODES.map((candidate) => {
          const enabled = modeEnabled(candidate);
          const label =
            candidate === "input" && view.input.linked
              ? `${t("mode.input")} · ${String(view.files.length)}${warnings > 0 ? ` · ${t("counts.warnings", { n: warnings })}` : ""}`
              : t(`mode.${candidate}`);
          return (
            // The tooltip is on the wrapper as well as on the chip: «Появится после первой
            // сборки» is the whole content of a disabled tab, and a disabled button gets no
            // mouse events of its own in every engine.
            <span key={candidate} title={enabled ? undefined : t("mode.locked")}>
              <Chip
                active={mode === candidate}
                disabled={!enabled}
                title={enabled ? undefined : t("mode.locked")}
                onClick={() => {
                  useUi.getState().setMode(candidate);
                }}
              >
                {label}
              </Chip>
            </span>
          );
        })}
      </nav>

      <span className="flex-1" />

      <IconButton
        label={t("pane.left")}
        pressed={leftOpen}
        onClick={() => {
          useUi.getState().toggleLeft();
        }}
      >
        <PanelLeft size={14} aria-hidden="true" />
      </IconButton>
      <IconButton
        label={t("pane.right")}
        pressed={rightOpen}
        onClick={() => {
          useUi.getState().toggleRight();
        }}
      >
        <PanelRight size={14} aria-hidden="true" />
      </IconButton>

      {primary === null ? null : (
        // The tooltip goes on a wrapper as well: a disabled button gets no mouse events of its
        // own in every engine, and a disabled action that cannot say why is just a dead button.
        <span title={primary.tooltip}>
          <Button
            variant={primary.variant}
            disabled={primary.disabled ?? false}
            title={primary.tooltip}
            onClick={primary.run}
          >
            {primary.label}
          </Button>
        </span>
      )}

      {asking ? (
        <Dialog
          title={t("run.cancel_title")}
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
                {t("run.cancel_keep")}
              </Button>
              <Button variant="danger" onClick={stop}>
                {t("run.cancel_confirm")}
              </Button>
            </>
          }
        >
          <p className="text-xs leading-snug">{t("run.cancel_body")}</p>
        </Dialog>
      ) : null}
    </header>
  );
}
