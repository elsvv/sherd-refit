import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

import { FOCUS_RING } from "./Button";

/**
 * A button whose whole content is an icon, so its name has to be said twice: once as the
 * `aria-label` a screen reader announces and once as the `title` a hovering mouse shows. Passing
 * the two separately is how they drift apart, so there is one `label` and it fills both.
 */
export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label" | "title"> {
  label: string;
  /** For a toggle (the two pane switches of A §7.1): its state, as `aria-pressed`. */
  pressed?: boolean | undefined;
}

/** The pane toggles and the banner's «×» — 24 px square, tokens only, no animation. */
export default function IconButton({ label, pressed, className, type = "button", ...rest }: IconButtonProps) {
  return (
    <button
      type={type}
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      className={clsx(
        "inline-flex h-6 w-6 items-center justify-center rounded-md border",
        "disabled:cursor-default disabled:opacity-45",
        pressed === true ? "border-accent bg-accent text-accent-text" : "border-border bg-panel text-text hover:bg-panel-2",
        FOCUS_RING,
        className,
      )}
      {...rest}
    />
  );
}
