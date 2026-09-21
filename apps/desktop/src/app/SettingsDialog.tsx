import clsx from "clsx";
import type { ReactNode } from "react";
import { useEffect, useId, useState } from "react";
import { useTranslation } from "react-i18next";

import type { BackendChoice } from "../ipc/bindings/BackendChoice";
import type { Settings } from "../ipc/bindings/Settings";
import { useSettings } from "../state/settings";
import type { Language, Theme } from "../state/ui";
import { useUi } from "../state/ui";
import Button, { FOCUS_RING } from "../ui/Button";
import Dialog from "../ui/Dialog";
import type { RadioOption } from "../ui/Radio";
import Radio from "../ui/Radio";
import { BACKENDS } from "./params";

/** The four answers the settings file holds, as the fields hold them: text, until they are read. */
interface Draft {
  backend: BackendChoice;
  /** Gigabytes, or empty for D §5's own budget — half of what the machine has. */
  memory: string;
  /** Threads; `0` is rayon's default, which is every core. */
  workers: string;
  /** Where Blender is, or empty for «wherever it installed itself» (A §9.2). */
  blender: string;
}

/** The engine's own ceiling on threads, as `sherd_app_core::settings` reads a file back. */
const MAX_WORKERS = 1024;

/** A memory limit this build would read back: nothing at all, or a positive number of gigabytes. */
function memoryOf(text: string): number | null | undefined {
  const written = text.trim().replace(",", ".");
  if (written === "") {
    return null;
  }
  const gb = Number(written);
  return Number.isFinite(gb) && gb > 0 ? gb : undefined;
}

/** And a thread count: a whole number from `0` — rayon's own default — to the file's ceiling. */
function workersOf(text: string): number | undefined {
  const threads = Number(text.trim());
  return Number.isInteger(threads) && threads >= 0 && threads <= MAX_WORKERS ? threads : undefined;
}

/** What the file holds, as a draft of text. */
function draftOf(settings: Settings): Draft {
  return {
    backend: settings.backend,
    memory: settings.memory_gb === null ? "" : String(settings.memory_gb),
    workers: String(settings.workers),
    blender: settings.blender_path ?? "",
  };
}

/**
 * The draft as settings, or `null` when a field holds something the shell would refuse.
 *
 * The check is the file's own (`Settings::is_usable`) and not a policy about hardware: a memory
 * limit is a positive number of gigabytes or nothing at all, and a thread count is a whole
 * number this build would read back. Refusing here rather than at the write is what lets
 * «Сохранить» be disabled instead of answering with a banner.
 */
function settingsOf(draft: Draft, version: number): Settings | null {
  const memory_gb = memoryOf(draft.memory);
  const workers = workersOf(draft.workers);
  if (memory_gb === undefined || workers === undefined) {
    return null;
  }
  const blender = draft.blender.trim();
  return { version, backend: draft.backend, memory_gb, workers, blender_path: blender === "" ? null : blender };
}

/**
 * One row: what it is called, the control, and the sentence that says what it does.
 *
 * `stack` puts the control on a line of its own. A number fits beside its label; a *path* does
 * not — «Путь к Blender» with a text field and «Найти автоматически» on one line leaves the
 * label two characters wide, which the first look at this screen showed as «Путь к …».
 */
function Row({
  label,
  hint,
  stack = false,
  control,
}: {
  label: string;
  hint: string;
  stack?: boolean;
  control: (id: string) => ReactNode;
}) {
  const id = useId();
  return (
    <div className="border-b border-border px-2 py-1.5 last:border-b-0">
      <div className={clsx("gap-2", stack ? "flex flex-col" : "flex items-center")}>
        <label htmlFor={id} className={clsx("truncate text-[11px]", stack ? "w-full" : "min-w-0 flex-1")}>
          {label}
        </label>
        {stack ? <div className="flex items-center gap-2">{control(id)}</div> : control(id)}
      </div>
      <p className="mt-0.5 text-[10px] leading-snug text-muted">{hint}</p>
    </div>
  );
}

/** The one text field of this screen, in the shape [`NumberField`] gives the launch sheet's. */
function Text({
  id,
  value,
  onChange,
  invalid,
  grow,
  placeholder,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  invalid: boolean;
  /** Takes the width that is left instead of the number fields' fixed column. */
  grow?: boolean;
  placeholder?: string;
}) {
  return (
    <input
      id={id}
      type="text"
      value={value}
      placeholder={placeholder}
      aria-invalid={invalid}
      onChange={(e) => {
        onChange(e.target.value);
      }}
      className={clsx(
        "h-6 rounded-md border bg-panel px-1.5 text-xs",
        grow === true ? "min-w-0 flex-1 text-left" : "w-28 shrink-0 text-right tabular-nums",
        invalid ? "border-danger" : "border-border",
        FOCUS_RING,
      )}
    />
  );
}

/**
 * A §7.4's «Настройки»: the two things about the window that outlive it — язык and тема — and
 * the four that belong to this computer, which the shell keeps in its config folder (A §11).
 *
 * The two halves behave differently on purpose, because they *are* different. Language and
 * theme are the window's own arrangement, kept in local storage and applied the moment they are
 * chosen — they are the very controls the workspace menu has, moved here, and a theme nobody can
 * see until they press «Сохранить» would be a preview that is not one. The other four are a file
 * the shell writes: they are a draft until «Сохранить», because a memory limit being typed
 * passes through «1» on its way to «16» and no run should ever be started under that.
 */
export default function SettingsDialog({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation();
  const theme = useUi((state) => state.theme);
  const language = useUi((state) => state.language);
  const settings = useSettings((state) => state.settings);
  const saving = useSettings((state) => state.saving);
  const error = useSettings((state) => state.error);

  const [draft, setDraft] = useState<Draft | null>(null);
  /** Whether what is on the screen is what is on disk — the footer's «Сохранено». */
  const [saved, setSaved] = useState(false);

  // Read when the screen opens and not when the window does: it is a command, and a session
  // that never opens this screen should not pay for one.
  useEffect(() => {
    void useSettings.getState().load();
  }, []);

  // The draft follows the file until the user touches something: `load` answers a moment after
  // the dialog is up, and `save` answers with what was written — including a `version` this
  // window did not choose.
  useEffect(() => {
    if (settings !== null) {
      setDraft(draftOf(settings));
    }
  }, [settings]);

  const languages: RadioOption<Language>[] = [
    { value: "ru", label: t("menu.language_ru") },
    { value: "en", label: t("menu.language_en") },
  ];
  const themes: RadioOption<Theme>[] = (["system", "light", "dark"] as const).map((choice) => ({
    value: choice,
    label: t(`menu.theme_${choice}`),
  }));
  const executors: RadioOption<BackendChoice>[] = BACKENDS.map((backend) => ({
    value: backend,
    label: t(`sheet.backend_${backend}`),
  }));

  const next = draft === null || settings === null ? null : settingsOf(draft, settings.version);
  const changed = next !== null && settings !== null && JSON.stringify(next) !== JSON.stringify(settings);

  const edit = (patch: Partial<Draft>): void => {
    if (draft !== null) {
      setDraft({ ...draft, ...patch });
      setSaved(false);
    }
  };

  return (
    <Dialog
      title={t("settings.title")}
      width={520}
      onClose={onClose}
      footer={
        <>
          <span className="min-w-0 flex-1 truncate text-[11px] text-muted">
            {next === null && draft !== null ? (
              <span className="text-danger">{t("settings.invalid")}</span>
            ) : saved ? (
              t("settings.saved")
            ) : (
              ""
            )}
          </span>
          <Button onClick={onClose}>{t("settings.close")}</Button>
          <Button
            variant="primary"
            disabled={next === null || !changed || saving}
            onClick={() => {
              if (next === null) {
                return;
              }
              void useSettings
                .getState()
                .save(next)
                .then((ok) => {
                  setSaved(ok);
                });
            }}
          >
            {t("settings.save")}
          </Button>
        </>
      }
    >
      <section>
        <h3 className="mb-2 border-b border-border pb-1 text-[11px] font-semibold">{t("settings.window")}</h3>
        <Radio
          name="sherd-settings-language"
          legend={t("settings.language")}
          variant="inline"
          value={language}
          options={languages}
          onChange={(code) => {
            useUi.getState().setLanguage(code);
          }}
        />
        <div className="mt-2">
          <Radio
            name="sherd-settings-theme"
            legend={t("settings.theme")}
            variant="inline"
            value={theme}
            options={themes}
            onChange={(choice) => {
              useUi.getState().setTheme(choice);
            }}
          />
        </div>
      </section>

      <section className="mt-4">
        <h3 className="mb-2 border-b border-border pb-1 text-[11px] font-semibold">{t("settings.machine")}</h3>
        {draft === null ? (
          <p className="text-[11px] text-muted">{t("settings.loading")}</p>
        ) : (
          <>
            <Radio
              name="sherd-settings-backend"
              legend={t("settings.backend")}
              variant="inline"
              value={draft.backend}
              options={executors}
              onChange={(backend) => {
                edit({ backend });
              }}
            />
            <p className="mt-1 text-[10px] leading-snug text-muted">{t("settings.backend_hint")}</p>

            <div className="mt-2 rounded-md border border-border">
              <Row
                label={t("settings.memory")}
                hint={t("settings.memory_hint")}
                control={(id) => (
                  <Text
                    id={id}
                    value={draft.memory}
                    placeholder={t("settings.memory_auto")}
                    invalid={memoryOf(draft.memory) === undefined}
                    onChange={(memory) => {
                      edit({ memory });
                    }}
                  />
                )}
              />
              <Row
                label={t("settings.workers")}
                hint={t("settings.workers_hint")}
                control={(id) => (
                  <Text
                    id={id}
                    value={draft.workers}
                    invalid={workersOf(draft.workers) === undefined}
                    onChange={(workers) => {
                      edit({ workers });
                    }}
                  />
                )}
              />
              <Row
                label={t("settings.blender")}
                hint={t("settings.blender_hint")}
                stack
                control={(id) => (
                  <>
                    <Text
                      id={id}
                      grow
                      value={draft.blender}
                      placeholder={t("settings.blender_auto_value")}
                      invalid={false}
                      onChange={(blender) => {
                        edit({ blender });
                      }}
                    />
                    {/* «Найти автоматически» is this field emptied: with no path of its own the
                        shell looks in the standard install locations of macOS and Windows and on
                        the `PATH` (A §9.2), which is what «автоматически» means here. */}
                    <Button
                      className="shrink-0"
                      disabled={draft.blender.trim() === ""}
                      onClick={() => {
                        edit({ blender: "" });
                      }}
                    >
                      {t("settings.blender_auto")}
                    </Button>
                  </>
                )}
              />
            </div>
          </>
        )}
        {error === null ? null : (
          <p className="mt-2 text-[10px] leading-snug break-words text-danger">{error.message}</p>
        )}
      </section>
    </Dialog>
  );
}
