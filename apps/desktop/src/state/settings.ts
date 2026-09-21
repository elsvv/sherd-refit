import { create } from "zustand";

import { api } from "../ipc";
import type { CommandError } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { Settings } from "../ipc/bindings/Settings";

/**
 * A §11's «Настройки», the half that belongs to the machine: which executor a launch sheet
 * starts on, how much memory and how many threads a job may take here, and where Blender is.
 *
 * The other half — язык and тема — is not here. Those are the *window's* arrangement and live in
 * [`useUi`] with the rest of it, remembered in local storage; the settings screen only shows
 * their controls beside these four (A §7.4 lists all six in one place, which is a fact about the
 * screen and not about where the answers are kept).
 *
 * The shell owns the file, so this store keeps no second opinion about it (A §5's rule): every
 * write answers with what was written, and that value replaces this one — including the
 * `version`, which the shell stamps with its own. `null` is «not read yet», and the screen says
 * so rather than showing defaults it would then save over.
 */
export interface SettingsState {
  /**
   * Whether A §7.4's screen is up. It lives here and not in the frame's own state — as
   * [`useFitSignal`] does, and for the same reason: three places open it (the workspace menu,
   * `Mod+,`, and A §9.2's «Указать путь к Blender…» from a toast that is not the frame's child),
   * and only one of them has a component to hold a flag in.
   */
  open: boolean;
  /** What the shell last said is on disk, or `null` before it has been asked. */
  settings: Settings | null;
  /** Whether a write is on its way; the screen's «Сохранить» waits for it. */
  saving: boolean;
  /** A refusal worth showing — a limit this build would not read back, a config folder gone. */
  error: CommandError | null;

  /**
   * Reads them. Called when the settings screen opens, and when «Собрать…» opens over a workspace
   * that has no run of its own to repeat (A §7.4: the sheet starts from the machine's three
   * answers then) — not when the window does: it is a command, and a session that opens neither
   * should not pay for one.
   *
   * A refusal is swallowed: `settings_get` cannot fail, and a failure here can only be the IPC
   * itself — in which case the screen has a bigger problem than its own banner.
   */
  load(): Promise<void>;
  /** Writes them and keeps what came back; `false` when the shell refused. */
  save(settings: Settings): Promise<boolean>;
  setOpen(open: boolean): void;
  dismissError(): void;
}

export const useSettings = create<SettingsState>()((set) => ({
  open: false,
  settings: null,
  saving: false,
  error: null,

  load: async () => {
    try {
      set({ settings: await api.settingsGet() });
    } catch (e) {
      set({ error: toCommandError(e) });
    }
  },

  save: async (settings) => {
    set({ saving: true, error: null });
    try {
      set({ settings: await api.settingsSet(settings), saving: false });
      return true;
    } catch (e) {
      set({ saving: false, error: toCommandError(e) });
      return false;
    }
  },

  setOpen: (open) => {
    // The refusal goes with the screen it was about: one shown again over a limit the user has
    // meanwhile stopped trying to save would be a sentence about nothing.
    set(open ? { open } : { open, error: null });
  },

  dismissError: () => {
    set({ error: null });
  },
}));
