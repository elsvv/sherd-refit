#!/usr/bin/env node
// A look at the window without a window: the frontend on its mock IPC layer, in a headless Chrome
// driven over the DevTools protocol, screenshots to a folder. No test runner and no dependencies —
// Node's own fetch and WebSocket. It asserts nothing: it exists so that whoever changed a screen
// can see it, which the agents that build this app otherwise cannot (A §11 rules out a UI test
// harness, not eyes).
//
//   node tools/look.mjs <steps.json | inline JSON> [out-dir]
//
// Steps, in order:
//   { "goto": "/" , "wait": 1500 }          navigate (relative to the dev server)
//   { "click": "Создать воркспейс" }        click the first button/cell whose text or aria-label contains it
//   { "key": { "key": "b", "code": "KeyB", "modifiers": 4 } }   a key press (modifiers: 1 alt, 2 ctrl, 4 meta, 8 shift)
//   { "eval": "document.title" }            evaluate and print
//   { "storage": { "sherd.language": "ru", "sherd.theme": "dark" } }   set localStorage, then reload
//   { "sleep": 6000 }
// Any step may carry "shot": "name" (a PNG after it) and "wait": ms.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ORIGIN = "http://localhost:1420";
const PORT = 9333;
const [stepsArg, outArg] = process.argv.slice(2);
if (!stepsArg) {
  console.error("usage: node tools/look.mjs <steps.json | inline JSON> [out-dir]");
  process.exit(2);
}
const steps = JSON.parse(existsSync(stepsArg) ? readFileSync(stepsArg, "utf8") : stepsArg);
const out = resolve(outArg ?? "look");
mkdirSync(out, { recursive: true });

const chromePath = [
  process.env.CHROME,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].find((p) => p && existsSync(p));
if (!chromePath) {
  console.error("no Chrome found; set CHROME=/path/to/chrome");
  process.exit(2);
}

const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const answers = async (url) => {
  try {
    return (await fetch(url)).ok;
  } catch {
    return false;
  }
};
const children = [];
const stop = () => children.forEach((c) => c.kill());
process.on("exit", stop);

if (!(await answers(ORIGIN))) {
  children.push(spawn("pnpm", ["exec", "vite", "--port", "1420", "--strictPort"], { stdio: "ignore" }));
  for (let i = 0; i < 60 && !(await answers(ORIGIN)); i += 1) await sleep(250);
}
children.push(
  spawn(
    chromePath,
    [
      "--headless=new",
      `--remote-debugging-port=${PORT}`,
      `--user-data-dir=${mkdtempSync(join(tmpdir(), "sherd-look-"))}`,
      "--window-size=1360,860",
      "--hide-scrollbars",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
      "about:blank",
    ],
    { stdio: "ignore" },
  ),
);
for (let i = 0; i < 60 && !(await answers(`http://127.0.0.1:${PORT}/json/version`)); i += 1) await sleep(250);

const targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
const socket = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((ok) => socket.addEventListener("open", ok, { once: true }));

let id = 0;
const pending = new Map();
const pageLog = [];
socket.addEventListener("message", (message) => {
  const msg = JSON.parse(message.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  } else if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(msg.params.type)) {
    pageLog.push(`[console.${msg.params.type}] ${msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ")}`);
  } else if (msg.method === "Runtime.exceptionThrown") {
    const details = msg.params.exceptionDetails;
    pageLog.push(`[exception] ${details.exception?.description ?? details.text}`);
  }
});
const send = (method, params = {}) =>
  new Promise((ok) => {
    pending.set(++id, ok);
    socket.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) => {
  const answer = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  return answer.result?.exceptionDetails
    ? `EVAL ERROR: ${answer.result.exceptionDetails.exception?.description}`
    : answer.result?.result?.value;
};

await send("Runtime.enable");
await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1360, height: 860, deviceScaleFactor: 1, mobile: false });

for (const step of steps) {
  if (step.goto !== undefined) {
    await send("Page.navigate", { url: new URL(step.goto, ORIGIN).href });
    await sleep(step.wait ?? 1500);
  } else if (step.storage !== undefined) {
    await evaluate(`Object.entries(${JSON.stringify(step.storage)}).forEach(([k, v]) => localStorage.setItem(k, v))`);
    await send("Page.reload");
    await sleep(step.wait ?? 1500);
  } else if (step.click !== undefined) {
    console.log(
      await evaluate(`(() => {
        const want = ${JSON.stringify(step.click)};
        const all = [...document.querySelectorAll('button, [role=button], [role=option], [role=tab], [role=menuitem]')];
        const text = (e) => (e.innerText || e.getAttribute('aria-label') || '');
        const hit = all.find((e) => text(e).includes(want));
        if (!hit) return 'NOT FOUND: ' + want + ' — on the page: ' + all.slice(0, 50).map((e) => JSON.stringify(text(e).slice(0, 24))).join(', ');
        hit.click();
        return 'clicked: ' + want;
      })()`),
    );
    await sleep(step.wait ?? 600);
  } else if (step.key !== undefined) {
    await send("Input.dispatchKeyEvent", { type: "keyDown", ...step.key });
    await send("Input.dispatchKeyEvent", { type: "keyUp", ...step.key });
    await sleep(step.wait ?? 400);
  } else if (step.eval !== undefined) {
    console.log("eval:", JSON.stringify(await evaluate(step.eval)));
    await sleep(step.wait ?? 200);
  } else if (step.sleep !== undefined) {
    await sleep(step.sleep);
  }
  if (step.shot !== undefined) {
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(join(out, `${step.shot}.png`), Buffer.from(shot.result.data, "base64"));
    console.log(`shot: ${join(out, `${step.shot}.png`)}`);
  }
}
console.log(pageLog.length ? `--- the page complained ---\n${pageLog.join("\n")}` : "--- the page logged no error ---");
socket.close();
stop();
process.exit(0);
