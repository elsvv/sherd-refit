import { useEffect } from "react";
import { create } from "zustand";

import { decide } from "../modes/review/ReviewCentre";
import { useReview } from "../state/review";
import { useSettings } from "../state/settings";
import { useUi } from "../state/ui";
import { useWorkspace } from "../state/workspace";
import { assembledRun, MODES, modeEnabled } from "./TopBar";

/**
 * The «вписать в экран» request, as a number that only ever goes up (A §7.1's `F`). The viewer is
 * not React and has no props to re-read, so the key and the «Вписать» chip both bump this and the
 * component that owns the canvas refits whenever it changes.
 */
export interface FitState {
  signal: number;
  requestFit(): void;
}

export const useFitSignal = create<FitState>()((set) => ({
  signal: 0,
  requestFit: () => {
    set((state) => ({ signal: state.signal + 1 }));
  },
}));

/**
 * `Mod` is ⌘ on macOS and Ctrl everywhere else. `navigator.platform` is deprecated and still the
 * only thing a webview answers truthfully about the *keyboard*: `userAgentData.platform` is not
 * in WebKit, and the user agent string of a Tauri window is not one to parse.
 */
const MOD_IS_META = typeof navigator !== "undefined" && /mac/i.test(navigator.platform);

/**
 * How `Mod` is written on a button that says which key it is (A §8.4's «Отменить ⌘Z»). The
 * labels have to agree with what the window actually listens for, so they come from the same
 * constant rather than from a translation — «⌘» is not Russian or English, and «Ctrl» is neither
 * either.
 */
export const MOD_LABEL = MOD_IS_META ? "⌘" : "Ctrl+";

/** A key pressed into a text field is text, never a shortcut. */
function typing(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

/**
 * Where the window's own keys are not the window's: inside a modal, which is modal (nothing
 * behind it may be reached while it is up, wherever the focus sits), and inside anything that
 * has marked itself off — A §7.1's pull-up engine log, which is read with the keyboard while the
 * screen behind it still has a pair selected.
 */
function elsewhere(target: EventTarget | null): boolean {
  if (document.querySelector('[role="dialog"]') !== null) {
    return true;
  }
  return target instanceof HTMLElement && target.closest("[data-shortcuts='off']") !== null;
}

/**
 * Whether the focus is on something `Space` already means something to. A button is activated by
 * `Space`, so the key must not *also* decide the pair behind it: pressing «Подтвердить» with the
 * keyboard would accept the join and then skip the next one.
 */
function activates(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && target.closest("button, summary, a[href], [role='button']") !== null;
}

/** Whether the «Сборка» mode can be entered right now — the tab's own rule, asked of the stores. */
function assemblyReady(): boolean {
  const state = useWorkspace.getState();
  return state.view !== null && assembledRun(state.view, state.selectedRunId) !== null;
}

/**
 * The frame's keys (A §7.1): `Mod+B` and `Mod+Alt+B` collapse the two side panes, `Mod+J` pulls
 * the engine log up and back down, `1` `2` `3` choose a mode, `F` fits the viewer, in the
 * «Сборка» mode `L` turns the names over the fragments on and off and `C` walks the three colour
 * modes, and in «Ревью» (A §8.3) `A` confirms the pair, `X` rejects it, `␣` skips it and
 * `Mod+Z` / `Shift+Mod+Z` walk the draft's history. They are matched on `event.code`, the
 * physical key, and not on `event.key`: the window's own language is Russian, and on a Russian
 * layout `F` types «а».
 *
 * None of them fires while a text field, a modal or the engine log has the focus ([`typing`],
 * [`elsewhere`]): a review is one key per pair, and a reviewer who has just typed a name into
 * the log's filter must not find that they have accepted a join.
 */
export function useShortcuts(): void {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.repeat || typing(e.target) || elsewhere(e.target)) {
        return;
      }
      const mod = MOD_IS_META ? e.metaKey : e.ctrlKey;
      const otherMod = MOD_IS_META ? e.ctrlKey : e.metaKey;

      // A §8.1's undo and redo, over the review's own history and only where it is being made:
      // the draft belongs to the session, and the session belongs to the «Ревью» mode.
      if (e.code === "KeyZ" && mod && !otherMod && !e.altKey) {
        if (useUi.getState().mode !== "review") {
          return;
        }
        e.preventDefault();
        if (e.shiftKey) {
          useReview.getState().redo();
        } else {
          useReview.getState().undo();
        }
        return;
      }

      // A §7.4's settings screen, on the key every desktop app opens its preferences with. It
      // is matched on `event.code` like the rest: the comma is the same physical key on a
      // Russian layout, where it types «б».
      if (e.code === "Comma" && mod && !otherMod && !e.shiftKey && !e.altKey) {
        e.preventDefault();
        useSettings.getState().setOpen(true);
        return;
      }

      if (e.code === "KeyB" && mod && !otherMod && !e.shiftKey) {
        e.preventDefault();
        if (e.altKey) {
          useUi.getState().toggleRight();
        } else {
          useUi.getState().toggleLeft();
        }
        return;
      }

      // A §7.1's pull-up engine log. With a modifier, because the panel is also reached from the
      // «Показать лог» of a failed run and a bare key would fire while the user is reading it.
      if (e.code === "KeyJ" && mod && !otherMod && !e.shiftKey && !e.altKey) {
        e.preventDefault();
        useUi.getState().toggleLog();
        return;
      }

      // Everything below is a bare key: a modifier held means the browser or the OS wants it.
      if (mod || otherMod || e.altKey || e.shiftKey) {
        return;
      }
      switch (e.code) {
        case "Digit1":
        case "Digit2":
        case "Digit3": {
          const mode = MODES[Number(e.code.slice(-1)) - 1];
          if (mode !== undefined && modeEnabled(mode, assemblyReady())) {
            e.preventDefault();
            useUi.getState().setMode(mode);
          }
          break;
        }
        case "KeyF":
          e.preventDefault();
          useFitSignal.getState().requestFit();
          break;
        // Both belong to the assembly's viewport (A §7.2) and mean nothing anywhere else, so
        // they are not taken from the rest of the window while it is on another mode.
        case "KeyL":
          if (useUi.getState().mode === "assembly") {
            e.preventDefault();
            useUi.getState().toggleAssemblyLabels();
          }
          break;
        case "KeyC":
          if (useUi.getState().mode === "assembly") {
            e.preventDefault();
            useUi.getState().cycleAssemblyColour();
          }
          break;
        // A §8.3's three verdicts, the whole point of having a keyboard on this screen: a
        // reviewer goes through hundreds of pairs and every one of them is one key. They belong
        // to the «Ревью» mode and are not taken from the window anywhere else.
        case "KeyA":
        case "KeyX":
        case "Space": {
          if (useUi.getState().mode !== "review") {
            break;
          }
          // `Space` is how the focused button is pressed; deciding the pair as well would be two
          // things from one key.
          if (e.code === "Space" && activates(e.target)) {
            break;
          }
          e.preventDefault();
          decide(e.code === "KeyA" ? "accept" : e.code === "KeyX" ? "reject" : null);
          break;
        }
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);
}
