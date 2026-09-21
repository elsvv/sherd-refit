import clsx from "clsx";
import type { ReactNode } from "react";
import { useId } from "react";
import { useTranslation } from "react-i18next";

import Button, { FOCUS_RING } from "./Button";

/** What every row of A §7.4's parameter table has, whatever its control is. */
interface RowProps {
  /** The `id` of the control, so the label points at it. */
  id: string;
  label: string;
  /** One sentence saying what the parameter does — the CLI's own doc comment, translated. */
  hint: string;
  /** The engine's own value, in words, beside the «сбросить» button. */
  defaultText: string;
  /** Whether the value is already the engine's, which is when there is nothing to reset. */
  isDefault: boolean;
  onReset: () => void;
  /** Why the value cannot be used, when it cannot; the row is drawn in `danger` while it is set. */
  error?: string | undefined;
  control: ReactNode;
}

/**
 * One row of the eleven (A §7.4): what it is called, the control, its default with a way back to
 * it, and the sentence that says what it does. The description is not a tooltip — these are
 * thresholds nobody remembers, and a value someone typed because they misread the name is a run
 * of forty minutes thrown away.
 */
function Row({ id, label, hint, defaultText, isDefault, onReset, error, control }: RowProps) {
  const { t } = useTranslation();
  return (
    <div className="border-b border-border px-2 py-1.5 last:border-b-0">
      <div className="flex items-center gap-2">
        <label htmlFor={id} className="min-w-0 flex-1 truncate text-[11px]">
          {label}
        </label>
        {control}
        <Button onClick={onReset} disabled={isDefault} className="shrink-0">
          {t("sheet.reset")}
        </Button>
      </div>
      <div className="mt-0.5 flex items-start gap-2">
        <p className="min-w-0 flex-1 text-[10px] leading-snug text-muted">{hint}</p>
        <span className="shrink-0 text-[10px] text-muted tabular-nums">
          {t("sheet.default", { value: defaultText })}
        </span>
      </div>
      {error === undefined ? null : <p className="mt-0.5 text-[10px] text-danger">{error}</p>}
    </div>
  );
}

export interface NumberFieldProps {
  label: string;
  hint: string;
  /** What the user has typed, verbatim: the sheet validates text, not a number it guessed at. */
  value: string;
  onChange: (value: string) => void;
  min: number;
  max: number;
  step: number;
  defaultText: string;
  onReset: () => void;
  error?: string | undefined;
}

/** How many decimals a step implies, so that stepping 0.25 by 0.01 gives 0.26 and not 0.26000000004. */
function decimalsOf(step: number): number {
  const dot = String(step).indexOf(".");
  return dot === -1 ? 0 : String(step).length - dot - 1;
}

/**
 * A number of A §7.4's table. The value is carried as **text** and not as a number: a field being
 * emptied to be retyped is not a zero, and an out-of-range value has to stay on the screen long
 * enough to be told off for — the sheet's «Собрать» is what refuses it (A §7.4).
 *
 * `type="text"` and not `type="number"`, for the reason `format.ts` gives for doing its own
 * formatting: a number field is drawn in the *browser's* locale, so on a machine set to Russian
 * the very same 2.5 reads «2,5» in the field and «по умолчанию 2.5» beside it. The two arrow keys
 * do what the spinner would, and a comma is read as a decimal point on the way in.
 */
export default function NumberField({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
  defaultText,
  onReset,
  error,
}: NumberFieldProps) {
  const id = useId();
  const nudge = (by: number): void => {
    const now = Number(value.trim().replace(",", "."));
    const from = Number.isFinite(now) ? now : min;
    const next = Math.min(max, Math.max(min, from + by));
    // Through `toFixed` and back, so that 0.25 + 0.01 is 0.26 and not 0.26000000000000001.
    onChange(String(Number(next.toFixed(decimalsOf(step)))));
  };
  return (
    <Row
      id={id}
      label={label}
      hint={hint}
      defaultText={defaultText}
      isDefault={value === defaultText}
      onReset={onReset}
      error={error}
      control={
        <input
          id={id}
          type="text"
          inputMode="decimal"
          value={value}
          aria-invalid={error !== undefined}
          onKeyDown={(e) => {
            if (e.key === "ArrowUp" || e.key === "ArrowDown") {
              e.preventDefault();
              nudge(e.key === "ArrowUp" ? step : -step);
            }
          }}
          onChange={(e) => {
            onChange(e.target.value);
          }}
          className={clsx(
            "h-6 w-28 shrink-0 rounded-md border bg-panel px-1.5 text-right text-xs tabular-nums",
            error === undefined ? "border-border" : "border-danger",
            FOCUS_RING,
          )}
        />
      }
    />
  );
}

export interface SwitchFieldProps {
  label: string;
  hint: string;
  value: boolean;
  onChange: (value: boolean) => void;
  /** What the engine does when nobody says otherwise. */
  defaultValue: boolean;
}

/**
 * The one row of the eleven that is not a number (`tiers`). A real `role="switch"` button rather
 * than a checkbox, because it is on or off and never indeterminate, and it reads the same way
 * both to a screen reader and to the eye.
 */
export function SwitchField({ label, hint, value, onChange, defaultValue }: SwitchFieldProps) {
  const { t } = useTranslation();
  const id = useId();
  return (
    <Row
      id={id}
      label={label}
      hint={hint}
      defaultText={t(defaultValue ? "sheet.on" : "sheet.off")}
      isDefault={value === defaultValue}
      onReset={() => {
        onChange(defaultValue);
      }}
      control={
        <button
          id={id}
          type="button"
          role="switch"
          aria-checked={value}
          onClick={() => {
            onChange(!value);
          }}
          className={clsx(
            "inline-flex h-6 w-28 shrink-0 items-center justify-center rounded-md border text-xs",
            value ? "border-accent bg-accent text-accent-text" : "border-border bg-panel text-muted",
            FOCUS_RING,
          )}
        >
          {t(value ? "sheet.on" : "sheet.off")}
        </button>
      }
    />
  );
}
