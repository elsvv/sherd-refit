import clsx from "clsx";

/** How far something has got, and how far it can get. */
export interface MeterProps {
  value: number;
  max: number;
  /** What the bar is measuring, for a screen reader — a progress bar with no name says nothing. */
  label?: string | undefined;
  className?: string | undefined;
}

/**
 * The status line's 160 px progress bar (A §6). The numbers come from the engine's `Progress`
 * events, so they are data from outside: a `max` of zero, a `NaN` or a `done` past `total` must
 * draw *something* rather than a bar of `Infinity` percent.
 */
export default function Meter({ value, max, label, className }: MeterProps) {
  const usable = Number.isFinite(value) && Number.isFinite(max) && max > 0;
  const fraction = usable ? Math.min(Math.max(value / max, 0), 1) : 0;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={usable ? max : 1}
      aria-valuenow={usable ? Math.min(Math.max(value, 0), max) : 0}
      className={clsx("h-1.5 w-40 overflow-hidden rounded-full bg-border", className)}
    >
      <div className="h-full bg-accent" style={{ width: `${String(fraction * 100)}%` }} />
    </div>
  );
}
