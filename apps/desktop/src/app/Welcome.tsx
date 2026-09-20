import clsx from "clsx";
import { FolderOpen, FolderPlus } from "lucide-react";
import type { TFunction } from "i18next";
import { useEffect } from "react";
import { useTranslation } from "react-i18next";

import { api } from "../ipc";
import { toCommandError } from "../ipc/api";
import type { Language } from "../state/ui";
import { useUi } from "../state/ui";
import { useWorkspace } from "../state/workspace";
import Button, { FOCUS_RING } from "../ui/Button";

/** Local midnight, so «вчера» means the day before and not «26 hours ago». */
function midnight(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/**
 * When the workspace was last open, in as few words as carry the meaning. `opened_at` is written
 * by the shell as RFC 3339, but it comes off a file someone may have edited, so an unparseable
 * one is shown as it stands rather than as «Invalid Date».
 */
function openedText(openedAt: string, language: Language, t: TFunction): string {
  const then = new Date(openedAt);
  if (Number.isNaN(then.getTime())) {
    return openedAt;
  }
  const days = Math.round((midnight(new Date()) - midnight(then)) / 86_400_000);
  if (days <= 0) {
    return t("date.today");
  }
  if (days === 1) {
    return t("date.yesterday");
  }
  if (days < 30) {
    return t("date.days_ago", { n: days });
  }
  return new Intl.DateTimeFormat(language, { day: "2-digit", month: "2-digit", year: "numeric" }).format(then);
}

/**
 * The screen with no workspace behind it (A §7.4): what the app is, the two ways in, and the
 * workspaces this machine has opened before. Both buttons pick a folder — «Создать» makes a
 * workspace in it, «Открыть» expects one to be there — and the shell decides which of the two
 * the folder can be, so the window never has to guess.
 */
export default function Welcome() {
  const { t } = useTranslation();
  const recent = useWorkspace((state) => state.recent);
  const error = useWorkspace((state) => state.error);
  const language = useUi((state) => state.language);

  useEffect(() => {
    void useWorkspace.getState().loadRecent();
  }, []);

  /** Picks a folder and hands it to `create` or `open`; a cancelled dialog does nothing. */
  const pick = (title: string, then: (path: string) => Promise<void>) => {
    void (async () => {
      try {
        const path = await api.pickFolder(title);
        if (path !== null) {
          await then(path);
        }
      } catch (e) {
        useWorkspace.getState().setError(toCommandError(e));
      }
    })();
  };

  return (
    <div className="flex h-full items-center justify-center overflow-auto bg-bg px-6 py-10">
      <div className="w-[520px]">
        <h1 className="text-2xl font-semibold tracking-tight">{t("app.name")}</h1>
        <p className="mt-1 text-sm text-muted">{t("welcome.tagline")}</p>

        <div className="mt-6 flex gap-2">
          <Button
            variant="primary"
            size="md"
            onClick={() => {
              pick(t("welcome.create_title"), (path) => useWorkspace.getState().create(path));
            }}
          >
            <FolderPlus size={15} aria-hidden="true" />
            {t("welcome.create")}
          </Button>
          <Button
            size="md"
            onClick={() => {
              pick(t("welcome.open_title"), (path) => useWorkspace.getState().open(path));
            }}
          >
            <FolderOpen size={15} aria-hidden="true" />
            {t("welcome.open")}
          </Button>
        </div>

        {error === null ? null : (
          <p className="mt-3 text-xs text-danger" role="alert">
            {t(`error.${error.kind}`)}
            <span className="text-muted"> · {error.message}</span>
          </p>
        )}

        <h2 className="mt-8 text-[10px] font-semibold tracking-wide text-muted uppercase">{t("welcome.recent")}</h2>
        {recent.length === 0 ? (
          <p className="mt-2 text-xs text-muted">{t("welcome.recent_empty")}</p>
        ) : (
          <ul className="mt-2 rounded-md border border-border bg-panel">
            {recent.map((entry) => (
              <li key={entry.path} className="border-b border-border last:border-b-0">
                <button
                  type="button"
                  disabled={!entry.available}
                  title={entry.path}
                  onClick={() => {
                    void useWorkspace.getState().open(entry.path);
                  }}
                  className={clsx(
                    "flex w-full items-baseline gap-3 px-3 py-2 text-left hover:bg-panel-2",
                    "disabled:cursor-default disabled:opacity-45 disabled:hover:bg-panel",
                    FOCUS_RING,
                  )}
                >
                  <span className="shrink-0 font-medium">{entry.name}</span>
                  <span className="min-w-0 flex-1 truncate text-xs text-muted">{entry.path}</span>
                  <span className="shrink-0 text-xs text-muted">
                    {entry.available ? openedText(entry.opened_at, language, t) : t("welcome.unavailable")}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
