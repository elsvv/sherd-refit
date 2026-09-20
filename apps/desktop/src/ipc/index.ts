import type { Api } from "./api";
import { mockApi } from "./mock";
import { tauriApi } from "./tauri";

/**
 * The one way out of the window (A §2.1). Tauri injects `__TAURI_INTERNALS__` into the webview
 * before any of our code runs, so its presence is how the app tells a real shell from `pnpm dev`
 * in a browser — and the browser gets the mock, which is what makes the screens reviewable
 * without a build of the shell (A §11).
 */
export const api: Api = "__TAURI_INTERNALS__" in window ? tauriApi : mockApi;
