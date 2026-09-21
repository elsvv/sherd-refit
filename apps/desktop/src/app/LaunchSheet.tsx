import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { api } from "../ipc";
import type { CalibrationView } from "../ipc/api";
import type { BackendChoice } from "../ipc/bindings/BackendChoice";
import type { Preset } from "../ipc/bindings/Preset";
import type { RunSpec } from "../ipc/bindings/RunSpec";
import type { WorkspaceView } from "../ipc/bindings/WorkspaceView";
import { formatCount } from "../modes/input/format";
import { useJobs } from "../state/jobs";
import { useUi } from "../state/ui";
import Button from "../ui/Button";
import Dialog from "../ui/Dialog";
import NumberField, { SwitchField } from "../ui/NumberField";
import type { RadioOption } from "../ui/Radio";
import Radio from "../ui/Radio";
import type { Draft, NumberKey, ParamRow } from "./params";
import {
  BACKENDS,
  DEFAULT_SPEC,
  draftOf,
  fieldErrors,
  formatDuration,
  PARAMS,
  parseSpec,
  PRESETS,
  specOf,
  withPreset,
} from "./params";

export interface LaunchSheetProps {
  view: WorkspaceView;
  /**
   * What the sheet should open with, over the last run's. A §10's «Повторить на CPU» is the whole
   * reason it exists: a run the GPU gave up on is repeated with one thing changed, and nothing
   * else about it re-chosen.
   */
  patch: Partial<RunSpec>;
  onClose: () => void;
}

/** The sheet as it opens: the newest run's own sheet when it can be read, the engine's otherwise. */
function initial(view: WorkspaceView, patch: Partial<RunSpec>): Draft {
  const last = parseSpec(view.runs[0]?.run.spec);
  return draftOf({ ...(last ?? DEFAULT_SPEC), ...patch });
}

/**
 * «Собрать…» (A §7.4): the three presets, what to compute on, the eleven thresholds when they are
 * asked for, and what all of that will cost on this machine.
 *
 * The estimate and the executor list are asked for when the sheet opens and not when the window
 * does: both mean a command, one of which spawns a worker, and a session that never assembles
 * anything should not pay for either. Neither is worth a banner if it fails — the sheet still
 * starts a run without knowing how long it will take.
 */
export default function LaunchSheet({ view, patch, onClose }: LaunchSheetProps) {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const [draft, setDraft] = useState<Draft>(() => initial(view, patch));
  const [calibration, setCalibration] = useState<CalibrationView | null>(null);
  const [backends, setBackends] = useState<readonly string[]>([]);

  useEffect(() => {
    let gone = false;
    void api.calibration().then(
      (answer) => {
        if (!gone) {
          setCalibration(answer);
        }
      },
      () => {
        // No estimate is a sentence the sheet already has; a refusal here is not the user's problem.
      },
    );
    void api.engineInfo().then(
      (answer) => {
        if (!gone) {
          setBackends(answer.backends);
        }
      },
      () => {
        // The three buttons are what the user chooses between; the lines under them are a note.
      },
    );
    return () => {
      gone = true;
    };
  }, []);

  const errors = fieldErrors(draft);
  const spec = specOf(draft);

  const presets: RadioOption<Preset>[] = PRESETS.map((preset) => ({
    value: preset,
    label: t(`sheet.preset_${preset}`),
    hint: t(`sheet.preset_${preset}_hint`),
  }));
  const executors: RadioOption<BackendChoice>[] = BACKENDS.map((backend) => ({
    value: backend,
    label: t(`sheet.backend_${backend}`),
  }));

  /** «≈ 18 мин на этой машине (до 11 935 пар)», or the sentence that says why there is no figure. */
  const estimate = ((): string => {
    if (calibration === null) {
      return t("sheet.estimate_loading");
    }
    const pairs = t("sheet.pairs", {
      count: calibration.pairs_upper_bound,
      n: formatCount(calibration.pairs_upper_bound, language),
    });
    if (calibration.estimate_seconds === null) {
      return `${t("sheet.estimate_none")} ${pairs}.`;
    }
    return t("sheet.estimate", { time: formatDuration(calibration.estimate_seconds, t), pairs });
  })();

  const setValue = (key: NumberKey, text: string): void => {
    setDraft({ ...draft, values: { ...draft.values, [key]: text } });
  };

  const row = (param: ParamRow) => {
    const label = t(`param.${param.key}.label`);
    const hint = t(`param.${param.key}.hint`);
    if (param.kind === "switch") {
      return (
        <SwitchField
          key={param.key}
          label={label}
          hint={hint}
          value={draft.tiers}
          defaultValue={DEFAULT_SPEC.tiers}
          onChange={(tiers) => {
            setDraft({ ...draft, tiers });
          }}
        />
      );
    }
    const error = errors[param.key];
    const defaultText = draftOf(DEFAULT_SPEC).values[param.key];
    return (
      <NumberField
        key={param.key}
        label={label}
        hint={hint}
        value={draft.values[param.key]}
        onChange={(text) => {
          setValue(param.key, text);
        }}
        min={param.min}
        max={param.max}
        step={param.step}
        defaultText={defaultText}
        onReset={() => {
          setValue(param.key, defaultText);
        }}
        error={
          error === undefined
            ? undefined
            : error === "range"
              ? t("sheet.out_of_range", { min: param.min, max: param.max })
              : t("sheet.need_number")
        }
      />
    );
  };

  return (
    <Dialog
      title={t("sheet.title")}
      onClose={onClose}
      width={520}
      footer={
        <>
          <span className="min-w-0 flex-1 truncate text-[11px] text-danger">
            {spec === null ? t("sheet.invalid") : ""}
          </span>
          <Button onClick={onClose}>{t("sheet.cancel")}</Button>
          <Button
            variant="primary"
            disabled={spec === null}
            onClick={() => {
              if (spec === null) {
                return;
              }
              onClose();
              void useJobs.getState().startRun(spec);
            }}
          >
            {t("sheet.submit")}
          </Button>
        </>
      }
    >
      <Radio
        name="sherd-preset"
        legend={t("sheet.preset")}
        value={draft.preset}
        options={presets}
        onChange={(preset) => {
          setDraft(withPreset(draft, preset));
        }}
      />

      <div className="mt-3">
        <Radio
          name="sherd-backend"
          legend={t("sheet.backend")}
          variant="inline"
          value={draft.backend}
          options={executors}
          onChange={(backend) => {
            setDraft({ ...draft, backend });
          }}
        />
        {backends.length === 0 ? null : (
          <ul className="mt-1.5">
            {backends.map((line) => (
              // The engine's own words, indentation included: the adapter lines under a backend
              // are indented by two spaces, and reflowing them loses which adapter belongs where.
              <li key={line} className="font-mono text-[10px] leading-snug break-words whitespace-pre-wrap text-muted">
                {line}
              </li>
            ))}
          </ul>
        )}
      </div>

      {draft.preset === "custom" ? (
        <section className="mt-3">
          <h3 className="mb-1.5 text-[10px] tracking-wide text-muted uppercase">{t("sheet.params")}</h3>
          <div className="rounded-md border border-border">{PARAMS.map(row)}</div>
        </section>
      ) : null}

      <section className="mt-3">
        <h3 className="mb-1 text-[10px] tracking-wide text-muted uppercase">{t("sheet.estimate_title")}</h3>
        <p className="text-[11px] leading-snug">{estimate}</p>
      </section>
    </Dialog>
  );
}
