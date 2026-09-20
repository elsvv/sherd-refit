// @ts-check
import tseslint from "typescript-eslint";

// `strictTypeChecked` over our own sources only: `src/ipc/bindings/**` is written by ts-rs and is
// not ours to style, and `dist/`, `src-tauri/` hold no TypeScript of ours at all. The two rules
// spelled out below are the milestone's constraints as errors rather than warnings — nothing that
// came over IPC may be typed `any` or forced with `!`.
export default tseslint.config(
  { ignores: ["dist/**", "src-tauri/**", "src/ipc/bindings/**"] },
  ...tseslint.configs.strictTypeChecked,
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-non-null-assertion": "error",
    },
  },
);
