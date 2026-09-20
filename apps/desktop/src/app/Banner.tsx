import clsx from "clsx";
import { TriangleAlert, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import Button from "../ui/Button";
import IconButton from "../ui/IconButton";

/** Warning or refusal — the two the frame ever shows (A §5, A §10). */
export type BannerTone = "warn" | "danger";

/** The one thing the banner offers to do about what it says. */
export interface BannerAction {
  label: string;
  onClick: () => void;
}

export interface BannerProps {
  tone: BannerTone;
  /** Already in the user's language: the banner translates nothing itself. */
  text: string;
  action?: BannerAction | undefined;
  /** Given for a banner the user may put away — a `CommandError`, and only that (A §10). */
  onDismiss?: (() => void) | undefined;
}

/**
 * The strip under the top bar (A §5's «устарел» and «вход недоступен» rows, A §10's errors). It
 * is a strip and not a dialog on purpose: none of these stop the user from looking at what is
 * already open, which is what the spec asks of a failed run and of a folder that has moved.
 */
export default function Banner({ tone, text, action, onDismiss }: BannerProps) {
  const { t } = useTranslation();
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className={clsx(
        "flex shrink-0 items-center gap-2 border-b border-l-4 border-border bg-panel px-3 py-1.5 text-xs",
        tone === "danger" ? "border-l-danger" : "border-l-warn",
      )}
    >
      <TriangleAlert size={14} aria-hidden="true" className={tone === "danger" ? "text-danger" : "text-warn"} />
      <span className="min-w-0 flex-1 truncate" title={text}>
        {text}
      </span>
      {action === undefined ? null : (
        <Button variant="ghost" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
      {onDismiss === undefined ? null : (
        <IconButton label={t("banner.dismiss")} onClick={onDismiss}>
          <X size={14} aria-hidden="true" />
        </IconButton>
      )}
    </div>
  );
}
