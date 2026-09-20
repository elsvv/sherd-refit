//! Everything the window may ask of the shell.
//!
//! One shape throughout (A §5): every command that changes the workspace answers with the whole
//! [`WorkspaceView`], so the window never has to guess what its own change did and never keeps a
//! second opinion about the workspace to hold in step with the first. The window asks; the shell
//! reads and writes the folder (A §2.1).

use std::path::{Path, PathBuf};

use serde::Serialize;
use sherd_app_core::run;
use sherd_app_core::view::{self, WorkspaceView};
use sherd_app_core::workspace::Workspace;
use tauri::{AppHandle, Manager, State};

use crate::error::CommandError;
use crate::jobs;
use crate::recent::{self, RecentEntry};
use crate::state::AppState;

/// Which build this is (A §7.4's settings screen, and every bug report).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct AppInfo {
    /// The app's own version.
    pub(crate) version: String,
    /// The engine's, as `run.json` records it for every run.
    pub(crate) core_version: String,
    /// The commit both were built from, or `unknown` outside a git checkout.
    pub(crate) commit: String,
}

/// Which build this is. Cannot fail: the three strings are compiled in.
#[tauri::command]
pub(crate) fn app_info() -> AppInfo {
    AppInfo {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        core_version: sherd_core::CORE_VERSION.to_owned(),
        commit: sherd_core::GIT_COMMIT.to_owned(),
    }
}

/// The welcome screen's list of workspaces opened before (A §4).
///
/// # Errors
///
/// [`CommandError`] of kind `io` when this machine has no config directory.
#[tauri::command]
pub(crate) fn recent_list(app: AppHandle) -> Result<Vec<RecentEntry>, CommandError> {
    Ok(recent::load(&config_dir(&app)?))
}

/// Makes `path` a workspace and opens it. A folder that already is one is simply opened, so that
/// «Создать» on a folder the user made last week is not an error they have to understand.
///
/// # Errors
///
/// As [`workspace_open`], and `io` when the folder cannot be made.
#[tauri::command]
pub(crate) fn workspace_create(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
) -> Result<WorkspaceView, CommandError> {
    open_with(&app, state.inner(), &PathBuf::from(path), Workspace::create)
}

/// Opens the workspace at `path` (A §4).
///
/// # Errors
///
/// [`CommandError`] of kind `busy` while a job is running, `not_a_workspace`, `locked` when
/// another window of the app has it (A §10), `version` for a folder a newer app wrote, `json`,
/// `io`, `engine`.
#[tauri::command]
pub(crate) fn workspace_open(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
) -> Result<WorkspaceView, CommandError> {
    open_with(&app, state.inner(), &PathBuf::from(path), Workspace::open)
}

/// Closes the open workspace, releasing its lock; closing none is no error.
///
/// # Errors
///
/// [`CommandError`] of kind `busy` while a job is running — the worker is writing into the
/// folder, and letting go of the lock under it is the one thing A §10's lock exists to prevent.
#[tauri::command]
pub(crate) fn workspace_close(state: State<'_, AppState>) -> Result<(), CommandError> {
    if state.job()?.is_some() {
        return Err(CommandError::busy());
    }
    *state.workspace()? = None;
    Ok(())
}

/// The workspace as it stands (A §5). The window asks again whenever it may have changed
/// underneath — the window regaining focus, a job ending.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, or `io`/`engine` when the input folder is there and
/// cannot be listed.
#[tauri::command]
pub(crate) fn workspace_view(state: State<'_, AppState>) -> Result<WorkspaceView, CommandError> {
    let job = state.job_view()?;
    let held = state.workspace()?;
    let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
    Ok(view::build(ws, job)?)
}

/// Links the folder of scans (A §5's «empty» row: «Выбрать папку со сканами»). Linking again is
/// how «Указать папку заново» repairs a workspace whose input moved.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, or `io` when `path` is not a folder that resolves.
#[tauri::command]
pub(crate) fn input_link(
    state: State<'_, AppState>,
    path: String,
) -> Result<WorkspaceView, CommandError> {
    let job = state.job_view()?;
    let mut held = state.workspace()?;
    let ws = held.as_mut().ok_or_else(CommandError::no_workspace)?;
    ws.set_input(&PathBuf::from(path))?;
    Ok(view::build(ws, job)?)
}

/// Leaves a fragment out of every run, or takes it back in (A §5.1). The scan file is untouched.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, or `io` when `sherd-workspace.json` cannot be written.
#[tauri::command]
pub(crate) fn fragment_exclude(
    state: State<'_, AppState>,
    name: String,
    excluded: bool,
) -> Result<WorkspaceView, CommandError> {
    let job = state.job_view()?;
    let mut held = state.workspace()?;
    let ws = held.as_mut().ok_or_else(CommandError::no_workspace)?;
    ws.set_excluded(&name, excluded)?;
    Ok(view::build(ws, job)?)
}

/// Starts preparing the input (A §5's «preparing» row). Returns as soon as the worker is on its
/// way: everything the job says arrives at the window as `engine:event`, and its end, with a
/// fresh view, as `engine:finished`.
///
/// # Errors
///
/// [`CommandError`] of kind `busy`, `no_workspace`, `worker`, `io` or `engine`; see
/// [`jobs::start_prepare`].
#[tauri::command]
pub(crate) fn prepare_start(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), CommandError> {
    jobs::start_prepare(&app, state.inner())
}

/// «Отменить» (A §2.2). Asks the job to stop; it ends at its next unit of work and reports
/// itself `Failed` with `cancelled`, which is what the window shows. Cancelling nothing is no
/// error — the button and the job ending on its own race.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when the job slot is poisoned.
#[tauri::command]
pub(crate) fn job_cancel(state: State<'_, AppState>) -> Result<(), CommandError> {
    jobs::cancel(state.inner())
}

/// Opening a workspace, whichever door it came through.
///
/// The order matters and is A §4's: the previous workspace is let go of first, because dropping
/// it is what releases its lock and the folder being opened may be that same one; the runs left
/// `running` by a crash are marked before any worker of ours exists, so a `running` on disk can
/// only be nobody's; and the asset protocol is widened to the new root before the window is given
/// a view naming files under it (A §2.1 — the window is given nothing outside the workspace).
fn open_with(
    app: &AppHandle,
    state: &AppState,
    path: &Path,
    open: fn(&Path) -> sherd_app_core::Result<Workspace>,
) -> Result<WorkspaceView, CommandError> {
    if state.job()?.is_some() {
        return Err(CommandError::busy());
    }
    let mut held = state.workspace()?;
    *held = None;
    let ws = open(path)?;
    let now = run::timestamp(chrono::Local::now());
    run::mark_interrupted(&ws.runs_dir(), &now)?;
    app.asset_protocol_scope().allow_directory(ws.root(), true)?;
    recent::touch(&config_dir(app)?, ws.root(), &now);
    // No job: nothing may be running, or the `busy` above would have answered.
    let view = view::build(&ws, None)?;
    *held = Some(ws);
    Ok(view)
}

/// Where the app keeps what is not a workspace's (A §4): the recent list, and later the GPU
/// self-test and the settings.
fn config_dir(app: &AppHandle) -> Result<PathBuf, CommandError> {
    Ok(app.path().app_config_dir()?)
}
