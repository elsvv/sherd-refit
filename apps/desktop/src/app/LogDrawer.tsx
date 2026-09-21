import clsx from "clsx";
import { X } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { api } from "../ipc";
import type { CommandError } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import type { LogLevel } from "../state/ui";
import { useUi } from "../state/ui";
import { useWorkspace } from "../state/workspace";
import Button from "../ui/Button";
import Chip from "../ui/Chip";
import IconButton from "../ui/IconButton";

/** How much of the log the panel asks for. A §7.4's drawer is for the end of a run, not for all of it. */
const MAX_LINES = 400;

/**
 * How often the panel re-reads the log while a job is writing one. Two seconds is slow enough
 * that the engine keeps the machine and fast enough that a line the user is waiting for is not
 * behind a button press; the shell reads only the tail of the file, never the whole of it.
 */
const REFRESH_MS = 2000;

/**
 * A line `tracing` marked as a problem. The level sits after the timestamp and is padded to five
 * characters (`  INFO`, ` ERROR`), so it is matched where it is and not anywhere in the line: an
 * engine message that happens to contain the word ERROR is not a line of that level.
 */
const PROBLEM = /^\S+\s+(?:WARN|ERROR)\s/u;

/** Which of the two colours a line wears — the tokens of `styles.css`, as everywhere else. */
function tone(line: string): string {
  if (/^\S+\s+ERROR\s/u.test(line)) {
    return "text-danger";
  }
  if (/^\S+\s+WARN\s/u.test(line)) {
    return "text-warn";
  }
  return "text-muted";
}

/** The two filters, in the order the panel draws them. */
const LEVELS: readonly LogLevel[] = ["all", "problems"];

/**
 * A §7.1's pull-up engine log: the tail of the selected run's `engine.log`, or of the workspace's
 * `prepare.log` when no run is selected, under a level filter.
 *
 * It is the one place the window shows the engine's own words in full (A §10: every failure that
 * has a `internal` or a `crashed` behind it is answered with «Показать лог»), and the one screen
 * that polls: while a job is writing the file there is no event that says «a line was added», so
 * the panel asks again every two seconds and otherwise not at all.
 *
 * Mounted only while it is open, so the poll and the reads cannot outlive the panel.
 */
export default function LogDrawer({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const level = useUi((state) => state.logLevel);
  const selectedRunId = useWorkspace((state) => state.selectedRunId);
  const running = view.job !== null;

  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<CommandError | null>(null);
  /** Bumped by «Обновить»: a read the user asked for, outside the poll's rhythm. */
  const [asked, setAsked] = useState(0);

  const box = useRef<HTMLDivElement>(null);
  /**
   * Whether the user is reading the end of the log. A panel that jumped to the bottom every two
   * seconds could not be read at all; one that never did would stop following a running job. So
   * it follows only while the view is already at the end, which is where it starts.
   */
  const atBottom = useRef(true);

  useEffect(() => {
    let gone = false;
    const read = (): void => {
      void api.runLog(selectedRunId, MAX_LINES).then(
        (answer) => {
          if (!gone) {
            setText(answer);
            setError(null);
          }
        },
        (e: unknown) => {
          if (!gone) {
            setError(toCommandError(e));
          }
        },
      );
    };
    read();
    const id = running ? window.setInterval(read, REFRESH_MS) : null;
    return () => {
      gone = true;
      if (id !== null) {
        window.clearInterval(id);
      }
    };
  }, [selectedRunId, running, asked]);

  const all = text === null || text === "" ? [] : text.replace(/\n+$/u, "").split("\n");
  const lines = level === "problems" ? all.filter((line) => PROBLEM.test(line)) : all;

  // After the paint that added the lines, not during it: `scrollHeight` is only the new one once
  // they are laid out.
  useLayoutEffect(() => {
    const el = box.current;
    if (el !== null && atBottom.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [text, level]);

  const empty =
    error !== null
      ? `${t(`error.${error.kind}`)} · ${error.message}`
      : text === null
        ? t("log.loading")
        : all.length === 0
          ? t("log.empty")
          : lines.length === 0
            ? t("log.nothing")
            : null;

  return (
    <section
      aria-label={t("log.title")}
      // A §8.3's `A` / `X` / `␣` decide a join on one key press. The log is read and filtered
      // with the keyboard over a screen that still has a pair selected, so the keys stop here
      // ([`elsewhere`] in `shortcuts.ts`).
      data-shortcuts="off"
      className="flex h-[40%] min-h-[120px] shrink-0 flex-col border-t border-border bg-panel"
    >
      <header className="flex h-8 shrink-0 items-center gap-2 border-b border-border px-3 text-xs">
        <span className="shrink-0 font-semibold">{t("log.title")}</span>
        <span className="min-w-0 truncate text-muted">{selectedRunId ?? t("log.prepare")}</span>
        <span className="flex-1" />
        {LEVELS.map((choice) => (
          <Chip
            key={choice}
            active={level === choice}
            onClick={() => {
              useUi.getState().setLogLevel(choice);
            }}
          >
            {t(`log.level_${choice}`)}
          </Chip>
        ))}
        <Button
          onClick={() => {
            setAsked((n) => n + 1);
          }}
        >
          {t("log.refresh")}
        </Button>
        <IconButton
          label={t("log.close")}
          onClick={() => {
            useUi.getState().setLogOpen(false);
          }}
        >
          <X size={14} aria-hidden="true" />
        </IconButton>
      </header>

      <div
        ref={box}
        onScroll={() => {
          const el = box.current;
          if (el !== null) {
            atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 4;
          }
        }}
        // Selectable, unlike the rest of the window: a log is quoted into a bug report.
        className="min-h-0 flex-1 overflow-auto px-3 py-2 font-mono text-[11px] leading-[1.45] [user-select:text]"
      >
        {empty === null ? (
          lines.map((line, i) => (
            // The index is the key on purpose: the list is the whole file every time, two lines
            // of a log can be identical, and nothing here has state to keep across a read.
            <div key={i} className={clsx("break-words whitespace-pre-wrap", tone(line))}>
              {line}
            </div>
          ))
        ) : (
          <p className="text-muted">{empty}</p>
        )}
      </div>
    </section>
  );
}
