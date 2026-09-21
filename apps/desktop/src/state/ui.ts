import { create } from "zustand";

import type { ColourMode, LayoutMode } from "../viewer/AssemblyViewer";

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
/** «Цвет: скан / фрагменты / группы» and «Все группы / Одна группа», in the words of A §7.2. */
export type { ColourMode, LayoutMode };
/** The order the `C` key walks the three colour modes in. */
const COLOURS: readonly ColourMode[] = ["scan", "fragment", "group"];
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

  /**
   * Which run the «Сборка» arrangement below belongs to. Every one of those fields names groups
   * by their index in *that* run's `assembly.json`, and two runs of one collection need not have
   * assembled the same groups in the same order — so the arrangement is thrown away when the run
   * changes, and only then ([`UiState.showRun`]).
   */
  assemblyRunId: string | null;
  /** «Цвет: скан / фрагменты / группы» (`C` cycles). */
  assemblyColour: ColourMode;
  /** «Все группы» ↔ «Одна группа». */
  assemblyLayout: LayoutMode;
  /** Which group «Одна группа» shows; `null` before one has been asked for. */
  assemblySingle: number | null;
  /** The groups whose members the tree is showing; [`UNPAIRED`] is the «Без пары» section. */
  assemblyOpen: ReadonlySet<number>;
  /** The groups whose eye is off — the inverse of what the viewer's `setGroupVisible` takes. */
  assemblyHidden: ReadonlySet<number>;
  /** Whether the tray of fragments that found no partner is in the viewport (off by default). */
  assemblyUnpaired: boolean;
  /** «Разъединить», 0…1. */
  assemblyExplode: number;
  /** «Подписи» (`L`). */
  assemblyLabels: boolean;
  /** What the left pane's search box holds; empty is «everything». */
  assemblyQuery: string;
  /** The group the inspector is about, when a group's row rather than a fragment is chosen. */
  assemblyGroup: number | null;
  /**
   * The last «покажи мне этот фрагмент» (A §7.2's «double click flies to»), which the inspector's
   * partner rows also ask for. A counter beside the name, because asking twice for the same
   * fragment is two requests and the camera has to move both times; the viewport clears it.
   */
  assemblyFly: { name: string; n: number } | null;

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

  /**
   * Says which run the «Сборка» mode is now showing. A no-op while it is the same one, so that
   * leaving the mode and coming back keeps the arrangement the user made; a different run — or
   * none — starts from the defaults, since its group numbers mean something else.
   */
  showRun(runId: string | null): void;
  setAssemblyColour(mode: ColourMode): void;
  /** The `C` key and the «Цвет: …» chip: scan → фрагменты → группы → scan. */
  cycleAssemblyColour(): void;
  /** «Одна группа» carries which one; «Все группы» leaves the last choice where it was. */
  setAssemblyLayout(mode: LayoutMode, group?: number): void;
  /** The chevron of a tree row. */
  toggleAssemblyOpen(index: number): void;
  /** Opening one that may already be open — how a fragment chosen in the viewport is revealed. */
  openAssemblyGroup(index: number): void;
  /** The eye of a tree row. */
  toggleAssemblyHidden(index: number): void;
  setAssemblyUnpaired(on: boolean): void;
  setAssemblyExplode(amount: number): void;
  setAssemblyLabels(on: boolean): void;
  toggleAssemblyLabels(): void;
  setAssemblyQuery(query: string): void;
  /** Chooses a group for the inspector; a group and a fragment are never both chosen. */
  selectGroup(index: number | null): void;
  /** Asks the viewport to put one fragment in the middle of itself. */
  flyToFragment(name: string): void;
  /** Called by the viewport once it has flown, so the next request is seen as a new one. */
  clearFly(): void;
}

/** What the «Сборка» mode looks like before anyone has arranged it (A §7.2's defaults). */
function freshAssembly(): Pick<
  UiState,
  | "assemblyColour"
  | "assemblyLayout"
  | "assemblySingle"
  | "assemblyOpen"
  | "assemblyHidden"
  | "assemblyUnpaired"
  | "assemblyExplode"
  | "assemblyLabels"
  | "assemblyQuery"
  | "assemblyGroup"
  | "assemblyFly"
> {
  return {
    assemblyColour: "scan",
    assemblyLayout: "spread",
    assemblySingle: null,
    assemblyOpen: new Set<number>(),
    assemblyHidden: new Set<number>(),
    assemblyUnpaired: false,
    assemblyExplode: 0,
    assemblyLabels: false,
    assemblyQuery: "",
    assemblyGroup: null,
    assemblyFly: null,
  };
}

/** A set with one member added or taken out — zustand compares by identity, so never in place. */
function toggled(set: ReadonlySet<number>, index: number): ReadonlySet<number> {
  const next = new Set(set);
  if (!next.delete(index)) {
    next.add(index);
  }
  return next;
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
  assemblyRunId: null,
  ...freshAssembly(),

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
    // A fragment and a group are one choice with two shapes: the inspector shows whichever was
    // made last, and leaving the other one set would make «nothing is selected» unreachable.
    set({ selectedFragment: name, assemblyGroup: null });
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

  showRun: (runId) => {
    set((state) => (state.assemblyRunId === runId ? state : { assemblyRunId: runId, ...freshAssembly() }));
  },
  setAssemblyColour: (mode) => {
    set({ assemblyColour: mode });
  },
  cycleAssemblyColour: () => {
    set((state) => {
      const at = COLOURS.indexOf(state.assemblyColour);
      return { assemblyColour: COLOURS[(at + 1) % COLOURS.length] ?? "scan" };
    });
  },
  setAssemblyLayout: (mode, group) => {
    set((state) => ({
      assemblyLayout: mode,
      assemblySingle: mode === "single" ? (group ?? state.assemblySingle ?? 0) : state.assemblySingle,
    }));
  },
  toggleAssemblyOpen: (index) => {
    set((state) => ({ assemblyOpen: toggled(state.assemblyOpen, index) }));
  },
  openAssemblyGroup: (index) => {
    set((state) => (state.assemblyOpen.has(index) ? state : { assemblyOpen: new Set(state.assemblyOpen).add(index) }));
  },
  toggleAssemblyHidden: (index) => {
    set((state) => ({ assemblyHidden: toggled(state.assemblyHidden, index) }));
  },
  setAssemblyUnpaired: (on) => {
    set({ assemblyUnpaired: on });
  },
  setAssemblyExplode: (amount) => {
    // The slider is the user's, but this is also what a look script and milestone 5 will call.
    const want = Number.isFinite(amount) ? Math.min(Math.max(amount, 0), 1) : 0;
    set({ assemblyExplode: want });
  },
  setAssemblyLabels: (on) => {
    set({ assemblyLabels: on });
  },
  toggleAssemblyLabels: () => {
    set((state) => ({ assemblyLabels: !state.assemblyLabels }));
  },
  setAssemblyQuery: (query) => {
    set({ assemblyQuery: query });
  },
  selectGroup: (index) => {
    set({ assemblyGroup: index, selectedFragment: null });
  },
  flyToFragment: (name) => {
    set((state) => ({ assemblyFly: { name, n: (state.assemblyFly?.n ?? 0) + 1 } }));
  },
  clearFly: () => {
    set({ assemblyFly: null });
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
