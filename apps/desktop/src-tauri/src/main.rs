//! sherd-refit's desktop app (A §2): one executable, two roles. Started with `--engine-worker`
//! it is the engine — one job on stdin, events on stdout — and never opens a window; started any
//! other way it is the window, and spawns itself for every job.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// The argument that makes this process the engine.
pub const ENGINE_WORKER: &str = "--engine-worker";

fn main() {
    // Before anything of Tauri's: the worker must not initialise a GUI toolkit it never shows.
    // `serve_stdio` also installs the process's only `tracing` subscriber, so nothing may set one
    // ahead of this branch.
    if std::env::args().nth(1).as_deref() == Some(ENGINE_WORKER) {
        std::process::exit(sherd_app_core::worker::serve_stdio());
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("the window could not be opened");
}
