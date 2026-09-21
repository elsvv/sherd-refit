import clsx from "clsx";
import { useTranslation } from "react-i18next";

import { useExport } from "../state/export";
import { useSettings } from "../state/settings";
import Button from "../ui/Button";

/**
 * What «Открыть в Blender» came to (A §9.2), said over the bottom-right corner of the window
 * and not in a dialog: the whole point of the action is that Blender opens *beside* the app, and
 * a modal asking to be dismissed would be in the way of looking at it.
 *
 * Three outcomes, and only one of them is a failure of anything. Blender started; Blender was
 * not found on this machine — the ordinary answer in a museum, and the script is on disk either
 * way, so the toast offers the folder and the settings screen; or the shell could not even write
 * the script, which is a refusal with the OS's own words under it.
 *
 * It stays until it is closed. A toast that fades takes «Показать в папке» with it, and the
 * script's folder is the whole answer for a machine with no Blender.
 */
export default function BlenderToast() {
  const { t } = useTranslation();
  const busy = useExport((state) => state.blenderBusy);
  const outcome = useExport((state) => state.blender);
  const error = useExport((state) => state.blenderError);

  if (!busy && outcome === null && error === null) {
    return null;
  }

  const tone = error !== null ? "border-danger" : outcome?.launched === true ? "border-accent" : "border-warn";
  const text = busy
    ? t("blender.working")
    : error !== null
      ? t("blender.refused")
      : outcome === null
        ? ""
        : outcome.launched
          ? t("blender.launched")
          : outcome.blender === null
            ? t("blender.not_found")
            : t("blender.not_started");

  return (
    <div
      role="status"
      className={clsx(
        "absolute right-3 bottom-3 z-30 w-[400px] rounded-md border bg-panel p-2 shadow-lg",
        tone,
      )}
    >
      <p className="text-[11px] leading-snug">{text}</p>
      {error !== null ? (
        <p className="mt-0.5 text-[10px] leading-snug break-words text-muted">{error.message}</p>
      ) : outcome === null ? null : (
        <p className="mt-0.5 truncate font-mono text-[10px] text-muted" title={outcome.script}>
          {outcome.script}
        </p>
      )}
      {busy ? null : (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          {outcome === null ? null : (
            <Button
              onClick={() => {
                void useExport.getState().reveal(outcome.script);
              }}
            >
              {t("blender.show")}
            </Button>
          )}
          {outcome !== null && !outcome.launched ? (
            <Button
              onClick={() => {
                useExport.getState().dismissBlender();
                useSettings.getState().setOpen(true);
              }}
            >
              {t("blender.set_path")}
            </Button>
          ) : null}
          <span className="flex-1" />
          <Button
            onClick={() => {
              useExport.getState().dismissBlender();
            }}
          >
            {t("blender.close")}
          </Button>
        </div>
      )}
    </div>
  );
}
