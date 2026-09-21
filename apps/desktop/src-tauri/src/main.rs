//! sherd-refit's desktop app (A §2): one executable, two roles. Started with `--engine-worker`
//! it is the engine — one job on stdin, events on stdout — and never opens a window; started any
//! other way it is the window, and spawns itself for every job.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::ffi::OsStr;

mod commands;
mod error;
mod jobs;
mod recent;
mod settings;
mod state;

/// The argument that makes this process the engine.
pub const ENGINE_WORKER: &str = "--engine-worker";

/// What the window role logs when `RUST_LOG` says nothing: the engine crates and the shell at
/// `info`, and none of Tauri's or wry's own chatter.
const DEFAULT_LOG: &str = "sherd=info,sherd_desktop=info";

fn main() {
    // Before anything of Tauri's: the worker must not initialise a GUI toolkit it never shows.
    // `serve_stdio` installs the worker role's own `tracing` subscriber — the only one that
    // process gets — so nothing may set one ahead of this branch, and `init_log` below belongs
    // to the window role alone.
    //
    // `args_os`, not `args`: the latter panics on an argument that is not Unicode, and on a
    // machine where the app is installed under a path the OS keeps in some other encoding every
    // argument can be one. A process that panics before it has a window has nowhere to say so.
    if std::env::args_os().nth(1).as_deref() == Some(OsStr::new(ENGINE_WORKER)) {
        std::process::exit(sherd_app_core::worker::serve_stdio());
    }
    init_log();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        // A §6: the shell says when a job is over, in the language the window told it. The
        // window's own side of the plugin is there only to ask the OS for the permission once.
        .plugin(tauri_plugin_notification::init())
        .manage(state::AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::app_info,
            commands::recent_list,
            commands::workspace_create,
            commands::workspace_open,
            commands::workspace_close,
            commands::workspace_view,
            commands::input_link,
            commands::fragment_exclude,
            commands::prepare_start,
            commands::run_start,
            commands::calibration,
            commands::job_cancel,
            commands::run_assembly,
            commands::run_candidates,
            commands::run_decisions,
            commands::run_log,
            commands::run_delete,
            commands::engine_info,
            commands::review_open,
            commands::review_apply,
            commands::review_pair,
            commands::review_refine,
            commands::review_close,
            commands::export_default_dest,
            commands::export_start,
            commands::blender_open,
            commands::reveal,
            commands::settings_get,
            commands::settings_set,
        ])
        .run(tauri::generate_context!())
        .expect("the window could not be opened");
}

/// Gives the window role's `tracing` somewhere to go: stderr, filtered by `RUST_LOG` or by
/// [`DEFAULT_LOG`].
///
/// Without this the shell's own warnings — `recent.rs` failing to write the list, `jobs.rs`
/// failing to file an assembly — are formatted and then dropped on the floor, because `tracing`
/// with no subscriber records nothing at all. A §10 asks the app to be able to say why something
/// did not work; a log that exists only in the source is not an answer.
///
/// A failure to install one is reported and shrugged off: it can only mean a subscriber is
/// already there, and the window is worth more than its log.
fn init_log() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_LOG));
    if let Err(error) =
        tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init()
    {
        eprintln!("sherd-refit: the log could not be set up: {error}");
    }
}
