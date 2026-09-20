/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The window is the only consumer of this bundle, and Tauri's dev URL is fixed in
// `src-tauri/tauri.conf.json`: the port must not wander, and Vite must not wipe the terminal
// `tauri dev` is printing the engine's log into.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  build: { target: "es2022", chunkSizeWarningLimit: 1200 },
  // Vitest runs on pure logic only (A §11) — no DOM, no component tests.
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
