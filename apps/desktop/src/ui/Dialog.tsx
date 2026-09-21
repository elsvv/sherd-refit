import { X } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useId, useRef } from "react";
import { useTranslation } from "react-i18next";

import IconButton from "./IconButton";

/**
 * Everything inside the panel that a Tab can reach. Disabled controls are left out on purpose:
 * the browser skips them, so a trap that counted them would send the focus to a dead end at the
 * edges of the cycle.
 */
const FOCUSABLE =
  'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])';

export interface DialogProps {
  title: string;
  /** Called by «×», by `Escape` and by a click outside — never by the dialog's own content. */
  onClose: () => void;
  /** The panel's width in pixels; A §7.4's launch sheet is 520. */
  width?: number | undefined;
  children: ReactNode;
  /** The row of buttons at the foot, right-aligned. */
  footer?: ReactNode | undefined;
}

/**
 * The one modal of the app (A §7.4's launch sheet, and the one question «Отменить» asks). It is a
 * real dialog and not a floating panel: while it is up nothing behind it may be reached, because
 * both things it is used for start or stop a job, and a click that lands somewhere else while the
 * user thinks they are answering a question is exactly the mistake a modal exists to prevent.
 *
 * The keyboard is kept inside it — Tab cycles, `Escape` closes, the focus goes back where it came
 * from — and the frame's own bare-key shortcuts (`1` `2` `3`, `F`) are stopped at the panel, so a
 * dialog cannot change the mode of the window underneath it.
 */
export default function Dialog({ title, onClose, width = 520, children, footer }: DialogProps) {
  const { t } = useTranslation();
  const box = useRef<HTMLDivElement>(null);
  const titleId = useId();

  // The callback as the last render gave it, so the trap below can be installed once. Depending on
  // `onClose` itself would re-run the effect on every render — and every render would then move
  // the focus back to the first control, which is unusable.
  const close = useRef(onClose);
  useEffect(() => {
    close.current = onClose;
  });

  useEffect(() => {
    const panel = box.current;
    const previous = document.activeElement;
    const inside = (): HTMLElement[] => (panel === null ? [] : [...panel.querySelectorAll<HTMLElement>(FOCUSABLE)]);
    (inside()[0] ?? panel)?.focus();

    // Capture, so that the trap sees the key before the control under the cursor does and before
    // anything the panel's own `onKeyDown` stops.
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close.current();
        return;
      }
      if (e.key !== "Tab" || panel === null) {
        return;
      }
      const items = inside();
      const first = items[0];
      const last = items.at(-1);
      if (first === undefined || last === undefined) {
        // Nothing to move to: the focus stays on the panel rather than leaving for the window.
        e.preventDefault();
        return;
      }
      const active = document.activeElement;
      if (!panel.contains(active)) {
        e.preventDefault();
        first.focus();
      } else if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      if (previous instanceof HTMLElement) {
        previous.focus();
      }
    };
  }, []);

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-viewport/60 p-4"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        ref={box}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        style={{ width: `${String(width)}px`, maxWidth: "100%" }}
        className="flex max-h-[88vh] flex-col rounded-lg border border-border bg-panel shadow-lg"
        // The frame's bare-key shortcuts listen on the window; a key pressed inside a modal is
        // meant for the modal, and `1` must not change the mode behind it.
        onKeyDown={(e) => {
          e.stopPropagation();
        }}
      >
        <header className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
          <h2 id={titleId} className="min-w-0 flex-1 truncate text-sm font-semibold">
            {title}
          </h2>
          <IconButton label={t("dialog.close")} onClick={onClose}>
            <X size={14} aria-hidden="true" />
          </IconButton>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">{children}</div>

        {footer === undefined ? null : (
          <footer className="flex shrink-0 items-center gap-2 border-t border-border px-3 py-2">{footer}</footer>
        )}
      </div>
    </div>
  );
}
