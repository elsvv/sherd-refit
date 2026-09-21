import { describe, expect, it } from "vitest";

import tauriSource from "./tauri.ts?raw";

/**
 * The seam between the window and the shell (A §2.1), as a test.
 *
 * Two of milestone 4's bugs lived exactly here and nowhere else: a command the shell had given a
 * new required parameter, invoked by the window without it — which no type checks, because
 * `invoke` takes a bag of unknowns, and which the mock cannot catch, because the mock *is* the
 * other side of that bag. Tauri then refuses the call at runtime, on a user's machine, with a
 * message about a missing argument.
 *
 * So the two sides are read as text and compared: every `#[tauri::command]` of `src-tauri/src/`
 * against every `invoke("…", { … })` of `src/ipc/tauri.ts`, by name and by argument. Nothing else
 * in the window is allowed to call `invoke` (A §2.1), which is what makes one file enough.
 *
 * **What the parsing relies on**, in both files, both of them small and written by hand:
 *
 * - a command is `#[tauri::command]` or `#[tauri::command(async)]`, then an optional `pub…`, then
 *   `fn name(…)` — the attribute and the signature with nothing but whitespace between them;
 * - its parameter list holds no comment and no nested `(` that is not closed inside it (the
 *   scanner below balances `(`, `<` and `[`, so `State<'_, AppState>` is one parameter);
 * - a call is `invoke("name")` or `invoke<T>("name", { … })` with a double-quoted literal name and
 *   an object literal — no computed key, no nested object, no variable payload;
 * - `main.rs` registers them in one `generate_handler![…]` of `commands::name` entries.
 *
 * A shape outside that is a parse that finds nothing, which fails as loudly as a mismatch: the
 * sets stop being equal. Regular expressions are enough here and a Rust parser would not be; if
 * either file ever grows past this, the fix is to make the shell print its own contract.
 */

/** Every file of the shell, inlined as text by Vite at transform time — no `node:fs` needed. */
const SHELL: Record<string, string> = import.meta.glob("../../src-tauri/src/*.rs", {
  query: "?raw",
  import: "default",
  eager: true,
});

/**
 * What the shell injects itself and the window never sends (Tauri fills them from the command's
 * types): the app handle, the managed state, the window the call came from.
 */
const INJECTED = new Set(["app", "state", "window"]);

/** One command, as either side of the seam describes it: its name and the arguments it takes. */
type Contract = Record<string, string[]>;

/** The text between the `(` at `open` and its matching `)`. */
function balanced(text: string, open: number): string {
  let depth = 0;
  for (let i = open; i < text.length; i += 1) {
    const char = text[i];
    if (char === "(") {
      depth += 1;
    } else if (char === ")") {
      depth -= 1;
      if (depth === 0) {
        return text.slice(open + 1, i);
      }
    }
  }
  return "";
}

/**
 * A comma-separated list split at its own top level: a Rust `State<'_, AppState>` is one
 * parameter and not two, and an object literal's `{ a, b }` is two keys.
 */
function splitTop(list: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let from = 0;
  for (let i = 0; i < list.length; i += 1) {
    const char = list[i];
    if (char === "<" || char === "(" || char === "[") {
      depth += 1;
    } else if (char === ">" || char === ")" || char === "]") {
      depth -= 1;
    } else if (char === "," && depth === 0) {
      parts.push(list.slice(from, i));
      from = i + 1;
    }
  }
  parts.push(list.slice(from));
  return parts.map((part) => part.trim()).filter((part) => part.length > 0);
}

/** `runId` → `run_id`: the one rewriting Tauri 2 does to an argument on its way to Rust. */
function snakeCase(key: string): string {
  return key.replace(/[A-Z]/g, (char) => `_${char.toLowerCase()}`);
}

/** Every `#[tauri::command]` of one Rust file, with the parameters the window has to send. */
function commandsOf(source: string, into: Contract): void {
  const signature = /#\[tauri::command(?:\([^)]*\))?]\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*\(/g;
  let match = signature.exec(source);
  while (match !== null) {
    const name = match[1];
    if (name !== undefined) {
      const open = match.index + match[0].length - 1;
      into[name] = splitTop(balanced(source, open))
        .map((parameter) => (parameter.split(":")[0] ?? "").trim())
        .filter((parameter) => parameter.length > 0 && !INJECTED.has(parameter));
    }
    match = signature.exec(source);
  }
}

/** Every `invoke` of the window, with the keys it puts in the payload, in the shell's spelling. */
function invocations(source: string): Contract {
  const found: Contract = {};
  const call = /invoke\s*(?:<[^<>()]*>)?\s*\(\s*"(\w+)"\s*(?:,\s*\{([^{}]*)\})?\s*\)/g;
  let match = call.exec(source);
  while (match !== null) {
    const name = match[1];
    if (name !== undefined) {
      found[name] = splitTop(match[2] ?? "")
        .map((entry) => (entry.split(":")[0] ?? "").trim())
        .filter((key) => key.length > 0)
        .map(snakeCase);
    }
    match = call.exec(source);
  }
  return found;
}

/** The commands `main.rs` hands to Tauri: one `generate_handler![commands::…]`. */
function registered(source: string): string[] {
  const list = /generate_handler!\[([^\]]*)]/.exec(source);
  if (list === null) {
    return [];
  }
  return [...(list[1] ?? "").matchAll(/commands::(\w+)/g)].map((entry) => entry[1] ?? "");
}

const shell: Contract = {};
for (const source of Object.values(SHELL)) {
  commandsOf(source, shell);
}
const window = invocations(tauriSource);
const handler = registered(Object.entries(SHELL).find(([path]) => path.endsWith("main.rs"))?.[1] ?? "");

/** Sorted, so a failure reads as a difference of sets and not of the order two files happen to be in. */
function sorted(names: readonly string[]): string[] {
  return [...names].sort((a, b) => a.localeCompare(b));
}

describe("the window and the shell agree on every command", () => {
  /** The parsing is load-bearing: a regex that matched nothing would make every set equal. */
  it("reads both sides", () => {
    expect(Object.keys(shell).length).toBeGreaterThan(10);
    expect(Object.keys(window).length).toBeGreaterThan(10);
    expect(handler.length).toBeGreaterThan(10);
  });

  it("calls every command the shell has, and no command it has not", () => {
    expect(sorted(Object.keys(window))).toEqual(sorted(Object.keys(shell)));
  });

  /** A command Tauri was never handed is refused at runtime exactly as a misspelt one is. */
  it("has every command registered with Tauri", () => {
    expect(sorted(handler)).toEqual(sorted(Object.keys(shell)));
  });

  it("sends every argument each command requires, and only those", () => {
    // One `expect` over the whole table rather than one per command: a failure then names every
    // command that disagrees, which is the report someone changing a signature wants.
    const asked: Contract = {};
    const sent: Contract = {};
    for (const name of sorted(Object.keys(shell))) {
      asked[name] = sorted(shell[name] ?? []);
      sent[name] = sorted(window[name] ?? []);
    }
    expect(sent).toEqual(asked);
  });
});
