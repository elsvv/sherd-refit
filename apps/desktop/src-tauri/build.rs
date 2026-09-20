//! Tauri's build step: it reads `tauri.conf.json` and `capabilities/`, and writes `gen/` — the
//! context `tauri::generate_context!()` expands to, and the permission schemas.

fn main() {
    tauri_build::build();
}
