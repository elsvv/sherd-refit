import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles.css";

// `index.html` carries the node; if it ever does not, say so rather than force it with `!` — the
// constraint against non-null assertions holds for the entry point too.
const root = document.getElementById("root");
if (!root) {
  throw new Error("index.html has no #root to mount the window in");
}
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
