import { create } from "zustand";

/** The three modes of A §7.3, as the top bar's tabs name them. */
export type Mode = "input" | "assembly" | "review";
/** Which fragments the «Вход» list shows (the mock-up's «Все · С предупреждениями · Исключённые»). */
export type InputFilter = "all" | "warnings" | "excluded";
/** The mock-up's `▦ ☰` toggle. */
export type InputLayout = "grid" | "list";
/** The viewer's two meshes (A §7.2): the scan, or the fracture faces in red. */
export type FragmentView = "scan" | "seg";
/**
 * What the pull-up engine log shows (A §7.1's «level filter»): every line the worker wrote, or
 * only the ones `tracing` marked `WARN` or `ERROR` — which is what someone who opened the log
 * after a failure came for.
 */
export type LogLevel = "all" | "problems";
/** `system` follows the OS; the other two override it. */
export type Theme = "system" | "light" | "dark";
/** Russian is the primary wording; English is the translation. */
export type Language = "ru" | "en";

/**
 * What the window looks like, as opposed to what it is showing (A §7.1). None of this is on
 * disk and none of it comes from the shell — it is the user's arrangement of the frame, and only
 * the two settings they would be annoyed to set twice, the theme and the language, outlive the
 * window.
 */
export interface UiState {
  mode: Mode;
  leftOpen: boolean;
  rightOpen: boolean;
  selectedFragment: string | null;
  inputFilter: InputFilter;
  inputLayout: InputLayout;
  fragmentView: FragmentView;
  wireframe: boolean;
  /** Whether A §7.1's engine-log panel is pulled up over the status line (`Mod+J`). */
  logOpen: boolean;
  /** Which of its lines are shown. */
  logLevel: LogLevel;
  theme: Theme;
  language: Language;

  setMode(mode: Mode): void;
  toggleLeft(): void;
  toggleRight(): void;
  selectFragment(name: string | null): void;
  setInputFilter(filter: InputFilter): void;
  setInputLayout(layout: InputLayout): void;
  setFragmentView(view: FragmentView): void;
  setWireframe(on: boolean): void;
  /** Used by A §10's «Показать лог», which must open the panel and never close it. */
  setLogOpen(open: boolean): void;
  toggleLog(): void;
  setLogLevel(level: LogLevel): void;
  setTheme(theme: Theme): void;
  setLanguage(language: Language): void;
}

const THEME_KEY = "sherd.theme";
const LANGUAGE_KEY = "sherd.language";

/** Local storage is missing in a test runner and can throw in a locked-down webview. */
function remembered(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function remember(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // A setting that cannot be saved is still a setting that works this session.
  }
}

function initialTheme(): Theme {
  const saved = remembered(THEME_KEY);
  return saved === "light" || saved === "dark" || saved === "system" ? saved : "system";
}

/**
 * Russian unless the machine says otherwise (A §7: the wording is Russian first, and the people
 * this is for work in Russian).
 */
function initialLanguage(): Language {
  const saved = remembered(LANGUAGE_KEY);
  if (saved === "ru" || saved === "en") {
    return saved;
  }
  return typeof navigator !== "undefined" && navigator.language.startsWith("ru") ? "ru" : "en";
}

/** `styles.css` has one dark palette, under `data-theme="dark"`, so `system` is resolved here. */
function resolveTheme(theme: Theme): "light" | "dark" {
  if (theme !== "system") {
    return theme;
  }
  if (typeof window === "undefined") {
    return "light";
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") {
    return;
  }
  document.documentElement.dataset.theme = resolveTheme(theme);
}

export const useUi = create<UiState>()((set) => ({
  mode: "input",
  leftOpen: true,
  rightOpen: true,
  selectedFragment: null,
  inputFilter: "all",
  inputLayout: "grid",
  fragmentView: "scan",
  wireframe: false,
  logOpen: false,
  logLevel: "all",
  theme: initialTheme(),
  language: initialLanguage(),

  setMode: (mode) => {
    set({ mode });
  },
  toggleLeft: () => {
    set((state) => ({ leftOpen: !state.leftOpen }));
  },
  toggleRight: () => {
    set((state) => ({ rightOpen: !state.rightOpen }));
  },
  selectFragment: (name) => {
    set({ selectedFragment: name });
  },
  setInputFilter: (filter) => {
    set({ inputFilter: filter });
  },
  setInputLayout: (layout) => {
    set({ inputLayout: layout });
  },
  setFragmentView: (view) => {
    set({ fragmentView: view });
  },
  setWireframe: (on) => {
    set({ wireframe: on });
  },
  setLogOpen: (open) => {
    set({ logOpen: open });
  },
  toggleLog: () => {
    set((state) => ({ logOpen: !state.logOpen }));
  },
  setLogLevel: (level) => {
    set({ logLevel: level });
  },
  setTheme: (theme) => {
    set({ theme });
    remember(THEME_KEY, theme);
    applyTheme(theme);
  },
  setLanguage: (language) => {
    set({ language });
    remember(LANGUAGE_KEY, language);
  },
}));

// The theme is a fact about the document, not about React, so it is written to <html> the moment
// the store exists and again whenever the OS changes its mind while the setting is `system`.
applyTheme(useUi.getState().theme);
if (typeof window !== "undefined") {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (useUi.getState().theme === "system") {
      applyTheme("system");
    }
  });
}
