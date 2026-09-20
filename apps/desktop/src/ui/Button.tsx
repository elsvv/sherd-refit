import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

/**
 * The one focus ring of the app, worn by every interactive element (A §7.1: the frame is worked
 * with the keyboard as much as with the mouse). The outline sits *outside* the element, so it
 * stays visible on the accent-filled primary button as well as on the outlined ones.
 */
export const FOCUS_RING = "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

/** How loud a button is: one primary per screen, ghosts beside it, danger for «Отменить». */
export type ButtonVariant = "primary" | "ghost" | "danger";
/** `sm` fits the 40 px top bar; `md` is the welcome screen's. */
export type ButtonSize = "sm" | "md";

/** Everything a `<button>` takes, plus how it should look. */
export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant | undefined;
  size?: ButtonSize | undefined;
}

/** Colours come only from the tokens of `styles.css`; a literal colour here would be a bug. */
const VARIANTS: Record<ButtonVariant, string> = {
  primary: "border-accent bg-accent text-accent-text hover:opacity-90",
  ghost: "border-border bg-panel text-text hover:bg-panel-2",
  danger: "border-danger bg-transparent text-danger hover:bg-panel-2",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "h-6 px-2.5 text-xs",
  md: "h-8 px-3 text-sm",
};

/**
 * A real `<button>` and nothing more: `type="button"` by default so that a stray form never
 * submits, and no transition anywhere — nothing in this window animates by default (A §7.1).
 */
export default function Button({ variant = "ghost", size = "sm", className, type = "button", ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={clsx(
        "inline-flex items-center justify-center gap-1.5 rounded-md border whitespace-nowrap",
        "disabled:cursor-default disabled:opacity-45",
        VARIANTS[variant],
        SIZES[size],
        FOCUS_RING,
        className,
      )}
      {...rest}
    />
  );
}
