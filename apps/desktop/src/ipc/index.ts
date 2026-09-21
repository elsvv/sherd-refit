import type { Api } from "./api";
import { mockApi } from "./mock";
import { tauriApi } from "./tauri";

/**
 * The one way out of the window (A §2.1). Tauri injects `__TAURI_INTERNALS__` into the webview
 * before any of our code runs, so its presence is how the app tells a real shell from `pnpm dev`
 * in a browser — and the browser gets the mock, which is what makes the screens reviewable
 * without a build of the shell (A §11).
 *
 * There is a third place this module is loaded: Vitest's `node` environment, where a store that
 * calls the API is imported for the sake of the pure functions beside it (A §11 runs no DOM). No
 * window there at all, so the check is for one first — the mock is the right answer for anything
 * that is not the shell, and a test that actually called it would be a test of the mock.
 */
export const api: Api = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window ? tauriApi : mockApi;
