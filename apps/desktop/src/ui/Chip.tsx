import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

import { FOCUS_RING } from "./Button";

/** A chip that is a warning or a refusal says so in its border, the way the mock-ups draw it. */
export type ChipTone = "default" | "warn" | "danger";

/** Everything a `<button>` takes, plus whether this chip is the chosen one and in what colour. */
export interface ChipProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean | undefined;
  tone?: ChipTone | undefined;
}

const TONES: Record<ChipTone, string> = {
  default: "border-border bg-panel text-text hover:bg-panel-2",
  warn: "border-warn bg-panel text-warn hover:bg-panel-2",
  danger: "border-danger bg-panel text-danger hover:bg-panel-2",
};

/**
 * The pill of the mock-ups' top bar and filter rows (`main-layout.html`, `.chip`): the mode tabs,
 * the run selector, the «Все / С предупреждениями / Исключённые» filters. It is a real `<button>`
 * and reports its state as `aria-pressed`, because a chip is a toggle and not a link.
 */
export default function Chip({ active = false, tone = "default", className, type = "button", ...rest }: ChipProps) {
  return (
    <button
      type={type}
      aria-pressed={active}
      className={clsx(
        "inline-flex h-6 items-center gap-1 rounded-full border px-2.5 text-xs whitespace-nowrap",
        "disabled:cursor-default disabled:opacity-45",
        active ? "border-accent bg-accent text-accent-text" : TONES[tone],
        FOCUS_RING,
        className,
      )}
      {...rest}
    />
  );
}
