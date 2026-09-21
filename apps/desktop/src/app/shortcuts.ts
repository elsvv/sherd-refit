import { useEffect } from "react";
import { create } from "zustand";

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

/** A key pressed into a text field is text, never a shortcut. */
function typing(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

/** Whether the «Сборка» mode can be entered right now — the tab's own rule, asked of the stores. */
function assemblyReady(): boolean {
  const state = useWorkspace.getState();
  return state.view !== null && assembledRun(state.view, state.selectedRunId) !== null;
}

/**
 * The frame's keys (A §7.1): `Mod+B` and `Mod+Alt+B` collapse the two side panes, `Mod+J` pulls
 * the engine log up and back down, `1` `2` `3` choose a mode, `F` fits the viewer, and in the
 * «Сборка» mode `L` turns the names over the fragments on and off and `C` walks the three colour
 * modes. They are matched on `event.code`, the physical key, and not
 * on `event.key`: the window's own language is Russian, and on a Russian layout `F` types «а».
 */
export function useShortcuts(): void {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.repeat || typing(e.target)) {
        return;
      }
      const mod = MOD_IS_META ? e.metaKey : e.ctrlKey;
      const otherMod = MOD_IS_META ? e.ctrlKey : e.metaKey;

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
