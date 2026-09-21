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
use crate::state::{AppState, JobSlot};

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
    // Held, not sampled — and to the end of the function, for the reason `open_with` gives: a
    // slot let go of after the check leaves a gap in which `prepare_start` starts a worker, and
    // the `None` below would then release the worker's own workspace lock under it.
    let slot = state.job()?;
    if slot.is_some() {
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
    // The slot stays locked while the view is built, so the view cannot say `job: null` about a
    // job started in between (A §5: the window keeps no second opinion, so this one must be true).
    let slot = state.job()?;
    let job = slot.as_ref().map(JobSlot::view);
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
    // Locked across the write and the view, as in `workspace_view`.
    let slot = state.job()?;
    let job = slot.as_ref().map(JobSlot::view);
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
    // Locked across the write and the view, as in `workspace_view`.
    let slot = state.job()?;
    let job = slot.as_ref().map(JobSlot::view);
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
/// **The workspace the user has is not let go of until the new one is in hand.** A path typed
/// wrong, a folder on a disk that has been unplugged, a workspace another window of the app is
/// holding (A §10's lock) — each of those is a refusal the user should be able to read with their
/// own workspace still on screen behind it. Letting go first and opening second turns every one
/// of them into «you now have nothing open», which is a worse answer than the error itself.
///
/// The folder that *is* open is therefore answered before anything is opened at all: its own lock
/// would refuse a second [`Workspace::open`] on it, and «open what I already have» means «show me
/// what I have» — not `locked`.
///
/// The rest of the order is A §4's: the runs left `running` by a crash are marked before any
/// worker of ours exists, so a `running` on disk can only be nobody's; and the asset protocol is
/// widened to the new root before the window is given a view naming files under it (A §2.1 — the
/// window is given nothing outside the workspace). The old workspace is dropped, and its lock
/// released, by the assignment at the end, once nothing can fail any more.
///
/// The job slot is *held* for all of it, and not merely read: let go of after the check, it
/// leaves a gap in which `prepare_start` can fill the slot and spawn a worker on the workspace
/// that the assignment below is about to drop — releasing A §10's `sherd-workspace.lock` under a
/// live worker of ours, and letting `run::mark_interrupted` write `interrupted` over a run that
/// is still being written. Job first, then the workspace, as [`crate::state`] requires.
fn open_with(
    app: &AppHandle,
    state: &AppState,
    path: &Path,
    open: fn(&Path) -> sherd_app_core::Result<Workspace>,
) -> Result<WorkspaceView, CommandError> {
    let slot = state.job()?;
    if slot.is_some() {
        return Err(CommandError::busy());
    }
    let mut held = state.workspace()?;
    if let Some(open_already) = held.as_ref()
        && same_folder(open_already.root(), path)
    {
        // No job: nothing was running, and the slot is still held here.
        return Ok(view::build(open_already, None)?);
    }
    let ws = open(path)?;
    let now = run::timestamp(chrono::Local::now());
    run::mark_interrupted(&ws.runs_dir(), &now)?;
    app.asset_protocol_scope().allow_directory(ws.root(), true)?;
    recent::touch(&config_dir(app)?, ws.root(), &now);
    // No job, as above.
    let view = view::build(&ws, None)?;
    // And here the previous workspace is dropped — the one point in the function after which
    // nothing can go wrong, so the one point at which it is safe to lose it.
    *held = Some(ws);
    Ok(view)
}

/// Whether two paths name the same folder.
///
/// Resolved first, so that a symlink and its target, `/var` and `/private/var` on macOS, and a
/// path with a `.` or a trailing separator in it are one folder and not two. A path that does not
/// resolve — the folder `workspace_create` is about to make — is compared as it was written,
/// which cannot match an open workspace's root, which does.
fn same_folder(a: &Path, b: &Path) -> bool {
    let resolve = |path: &Path| path.canonicalize().unwrap_or_else(|_| path.to_owned());
    resolve(a) == resolve(b)
}

/// Where the app keeps what is not a workspace's (A §4): the recent list, and later the GPU
/// self-test and the settings.
fn config_dir(app: &AppHandle) -> Result<PathBuf, CommandError> {
    Ok(app.path().app_config_dir()?)
}
