//! Everything the window may ask of the shell.
//!
//! One shape throughout (A §5): every command that changes the workspace answers with the whole
//! [`WorkspaceView`], so the window never has to guess what its own change did and never keeps a
//! second opinion about the workspace to hold in step with the first. The window asks; the shell
//! reads and writes the folder (A §2.1).

use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sherd_app_core::blender::{self, Resolution, ResolutionDto, ScopeDto};
use sherd_app_core::decisions::{DECISIONS_FILE, DECISIONS_VERSION, DecisionsFile};
use sherd_app_core::eta::{self, Calibration};
use sherd_app_core::host::{self, Outcome, Requester, Worker};
use sherd_app_core::protocol::{
    AssemblyDto, CandidateRow, Event, ExportWhat, Job, Request, RunSpec,
};
use sherd_app_core::settings::Settings;
use sherd_app_core::view::{self, JobKind, WorkspaceView};
use sherd_app_core::workspace::Workspace;
use sherd_app_core::{AppError, atomic, run, snapshot};
use tauri::{AppHandle, Manager, State};

use crate::error::CommandError;
use crate::jobs::{self, Lang};
use crate::recent::{self, RecentEntry};
use crate::settings;
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
///
/// **`async`** for [`workspace_open`]'s reason: [`open_with`] may have a review session to close
/// and wait for.
#[tauri::command(async)]
pub(crate) fn workspace_create(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
) -> Result<WorkspaceView, CommandError> {
    open_with(&app, state.inner(), &PathBuf::from(path), Workspace::create)
}

/// Opens the workspace at `path` (A §4).
///
/// **`async`**: a review session over the workspace being left is closed and waited for
/// ([`open_with`]), and a wait on a command that runs on the main thread is a frozen window.
///
/// # Errors
///
/// [`CommandError`] of kind `busy` while a job is running, `not_a_workspace`, `locked` when
/// another window of the app has it (A §10), `version` for a folder a newer app wrote, `json`,
/// `io`, `engine`.
#[tauri::command(async)]
pub(crate) fn workspace_open(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
) -> Result<WorkspaceView, CommandError> {
    open_with(&app, state.inner(), &PathBuf::from(path), Workspace::open)
}

/// Closes the open workspace, releasing its lock; closing none is no error.
///
/// A review session in the job slot is closed first and waited for (A §8.4): it is a cache over
/// a finished run and never a reason to refuse the user «Закрыть». A `Prepare` or a run is —
/// the worker is writing into the folder this is about to let go of.
///
/// **`async`** for that wait, which on the main thread would be a frozen window.
///
/// # Errors
///
/// [`CommandError`] of kind `busy` while a job is running — the worker is writing into the
/// folder, and letting go of the lock under it is the one thing A §10's lock exists to prevent.
#[tauri::command(async)]
pub(crate) fn workspace_close(state: State<'_, AppState>) -> Result<(), CommandError> {
    // Before the slot is taken, and with nothing held: the session's own thread takes that very
    // lock to give the slot back (`jobs::finish`).
    jobs::close_session(state.inner())?;
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
/// `lang` is the window's own language (`ru`, `en`; anything else reads as `ru`). The job carries
/// it so that A §6's end-of-job notification — written by the shell and shown by the OS, by which
/// time the window may be behind something else — is in the language the user is reading.
///
/// **`async`, and not for the sake of an `await`.** A review session open over some run holds the
/// worker, and this command closes it and waits for its thread — bounded, but a wait all the
/// same (A §8.4). On the main thread that wait would be a window that does not repaint.
///
/// # Errors
///
/// [`CommandError`] of kind `busy`, `no_workspace`, `worker`, `io` or `engine`; see
/// [`jobs::start_prepare`].
#[tauri::command(async)]
pub(crate) fn prepare_start(
    app: AppHandle,
    state: State<'_, AppState>,
    lang: String,
) -> Result<(), CommandError> {
    jobs::start_prepare(&app, state.inner(), Lang::of(&lang))
}

/// Starts a run on the open workspace (A §7) and answers with its id — the folder under `runs/`
/// the window asks about from here on. Returns as soon as the worker is on its way: everything
/// the run says arrives as `engine:event`, and its end, with `run.json` already closed and a
/// fresh view, as `engine:finished`.
///
/// The sheet is remembered before the run starts, so the next «Подготовить» prepares the cache
/// *this* run would have wanted (A §7.4). `lang` is as [`prepare_start`]'s, and `async` for the
/// same reason.
///
/// `carry_from` is A §8.5's «Перенести решения ревью»: the run whose `decisions.json` this one
/// starts from — accepted pairs pinned at their pose and not matched again, rejected pairs
/// skipped. The decisions that could not come along (a fragment removed or excluded since) are
/// filed nowhere and said once, as an `engine:event` carrying `dropped`.
///
/// # Errors
///
/// [`CommandError`] of kind `busy`, `no_workspace`, `worker`, `io`, `json` or `engine`; see
/// [`jobs::start`].
#[tauri::command(async)]
pub(crate) fn run_start(
    app: AppHandle,
    state: State<'_, AppState>,
    spec: RunSpec,
    lang: String,
    carry_from: Option<String>,
) -> Result<String, CommandError> {
    jobs::start_run(&app, state.inner(), spec, Lang::of(&lang), carry_from)
}

/// What a run of this collection will take, for the launch sheet's «≈ 11 мин» (A §6).
///
/// Three things and not one, because the sheet shows all three: what this machine has measured,
/// how many pairs the collection can make at most — A §7.4's «до N пар» — and the seconds those
/// pairs come to. The estimate is `None` until a run has finished on this machine; A §6 asks the
/// sheet to say so rather than to invent a figure.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct CalibrationView {
    /// What the last runs on this machine measured (A §6).
    pub(crate) calibration: Calibration,
    /// `n·(n−1)/2` over the scans a run would take in. An upper bound: R §4.1's wall-ratio filter
    /// skips pairs whose thicknesses cannot belong together, and how many is not known until
    /// matching has started.
    pub(crate) pairs_upper_bound: usize,
    /// Seconds those pairs will take, or `None` before the first finished run.
    pub(crate) estimate_seconds: Option<f64>,
}

/// [`CalibrationView`] for the open workspace.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, `io` when this machine has no config directory, or
/// `io`/`engine` when the input folder is there and cannot be listed.
#[tauri::command]
pub(crate) fn calibration(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<CalibrationView, CommandError> {
    let calibration = Calibration::load(&config_dir(&app)?.join(eta::CALIBRATION_FILE));
    let pairs = eta::pairs_upper_bound(included_scans(state.inner())?);
    Ok(CalibrationView {
        estimate_seconds: calibration.estimate(pairs),
        calibration,
        pairs_upper_bound: pairs,
    })
}

/// How many scans a run would take in: the input folder as it stands, less A §5.1's exclusions.
///
/// An input folder that cannot be reached is no scans rather than an error — A §5 gives «вход
/// недоступен» its own row and the sheet is not reachable from it, but an estimate asked for
/// anyway should say «0 пар» and not fail. The job slot is not read: the answer does not depend
/// on what is running, and the sheet asks for this while a run is on as well as before one.
fn included_scans(state: &AppState) -> Result<usize, CommandError> {
    let held = state.workspace()?;
    let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
    let Some(input) = ws.input() else { return Ok(0) };
    let excluded = &ws.file().excluded;
    let scans = snapshot::scan(&input, excluded)?;
    Ok(scans.files.iter().filter(|file| !excluded.contains(&file.name)).count())
}

/// What a run assembled (A §8), from its own `assembly.json`.
///
/// The window has every fragment's display mesh already and needs only the poses (A §7.2), which
/// is why this is the whole of what a run's 3D costs: one file of a few hundred kilobytes, and
/// no worker.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, `io` when there is no such run or the run has no
/// `assembly.json` — a run that failed before it assembled anything has none — or `json` when
/// the file does not parse.
#[tauri::command]
pub(crate) fn run_assembly(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<AssemblyDto, CommandError> {
    let dir = run_dir(state.inner(), &run_id)?;
    Ok(atomic::read_json(&dir.join(host::ASSEMBLY_FILE))?)
}

/// Every candidate of a run (A §8.3), from its `candidates.json` — the index the review screen
/// works from, and what the window shows about a join it is asked to explain.
///
/// # Errors
///
/// As [`run_assembly`], about `candidates.json`.
#[tauri::command]
pub(crate) fn run_candidates(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<Vec<CandidateRow>, CommandError> {
    let dir = run_dir(state.inner(), &run_id)?;
    Ok(atomic::read_json(&dir.join(sherd_app_core::worker::CANDIDATES_FILE))?)
}

/// What the reviewer decided about a run (A §8.1), from its own `decisions.json`.
///
/// A run nobody has reviewed has no file and is not an error: it answers an empty list, which is
/// what the «Ревью» screen starts from and what the launch sheet counts for A §8.5's «Перенести
/// решения ревью (N)».
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, `io` when there is no such run, `json` when the file
/// does not parse, or `version` when a newer app wrote it.
#[tauri::command]
pub(crate) fn run_decisions(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<DecisionsFile, CommandError> {
    let dir = run_dir(state.inner(), &run_id)?;
    Ok(DecisionsFile::load_or_default(&dir)?)
}

/// Opens a review session over a finished run (A §8): the worker loads the collection and the
/// run's saved match once, says `ready`, and then answers [`review_apply`], [`review_pair`] and
/// [`review_refine`] until [`review_close`].
///
/// Everything the session says reaches the window as `engine:event`, exactly as a run's does, and
/// the assembly of every answer is filed by the shell (A §2.1). Opening the run that is already
/// open does nothing — entering the mode twice must not throw a loaded match away — and a
/// session over another run replaces it. A `Prepare` or a run that is actually running is `busy`:
/// the worker is theirs, and A §5 has one.
///
/// **`async`** for [`prepare_start`]'s reason, and one more: a session over another run is closed
/// and waited for first.
///
/// # Errors
///
/// [`CommandError`] of kind `busy`, `no_workspace`, `worker`, `io` when there is no such run, or
/// `json` when its `run.json` holds a launch sheet this build cannot read.
#[tauri::command(async)]
pub(crate) fn review_open(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: String,
) -> Result<(), CommandError> {
    jobs::start_review(&app, state.inner(), run_id)
}

/// Files the reviewer's decisions and asks the session to assemble again (A §8.2).
///
/// **Written before it is sent**, and atomically: the answer that comes back is what the window
/// will draw, and a crash between the two must not leave a screen showing an assembly the folder
/// on disk cannot account for. The whole list travels every time because undo and redo are the
/// window's (A §8.1) — the shell keeps no second copy to hold in step with it.
///
/// **`async`**, for what that writing costs: `DecisionsFile::save` is an atomic write and ends
/// with an `fsync`, which on a slow or a network disk is tens of milliseconds the main thread
/// would spend not drawing. A §8.2's «under a second» is about the whole round trip, and a
/// reviewer holding `A` down through a queue makes one of these per keypress.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when no session is open or it is no longer listening,
/// `json` when the list is not one this build can file, `no_workspace`, or `io`.
#[tauri::command(async)]
pub(crate) fn review_apply(
    state: State<'_, AppState>,
    decisions: DecisionsFile,
) -> Result<(), CommandError> {
    let (run_id, requester) = session(state.inner())?;
    checked(&decisions)?;
    let dir = run_dir(state.inner(), &run_id)?;
    decisions.save(&dir)?;
    requester.send(&Request::Reassemble { decisions })?;
    Ok(())
}

/// Asks the session for the seam of one placement (A §8.3), which arrives as a `pair_detail`
/// event. `pose` maps `b` into `a`'s frame, row-major — the matrix a `CandidateRow` carries.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when no session is open or it is no longer listening. A
/// pair the collection does not have is the *session's* answer — a `request_failed` event — and
/// not a refusal here: the window asked something answerable and the answer is «no such pair».
#[tauri::command]
pub(crate) fn review_pair(
    state: State<'_, AppState>,
    a: String,
    b: String,
    pose: [[f64; 4]; 4],
) -> Result<(), CommandError> {
    let (_, requester) = session(state.inner())?;
    requester.send(&Request::PairDetail { a, b, pose })?;
    Ok(())
}

/// Asks the session to run R §9 over the groups this run's decisions left unrefined (A §8.4).
///
/// From the file and not from an argument: what is refined must be the assembly the decisions on
/// disk describe, and a list sent here that [`review_apply`] had not filed would refine a draft
/// nobody could get back to.
///
/// # Errors
///
/// As [`review_apply`], without the validation.
#[tauri::command]
pub(crate) fn review_refine(state: State<'_, AppState>) -> Result<(), CommandError> {
    let (run_id, requester) = session(state.inner())?;
    let dir = run_dir(state.inner(), &run_id)?;
    let decisions = DecisionsFile::load_or_default(&dir)?;
    requester.send(&Request::Refine { decisions })?;
    Ok(())
}

/// Closes the review session (A §8): the worker answers `Done` and its ~3 GB go back to the OS.
///
/// Returns as soon as the line is out, as «Отменить» does: the session's own thread frees the job
/// slot and sends `engine:finished`, which is how the window learns it is over. Closing nothing
/// is no error — the window leaves the mode, and a run started a moment earlier may have closed
/// the session already.
///
/// The requester is **taken out of the slot** rather than copied from it, which is how the rest
/// of the shell tells a session that is going from one that is there: it answers no more
/// questions ([`crate::state::JobSlot::requester`]), and [`jobs::start_review`] over that same
/// run waits for the slot instead of answering «already open» to a session nobody will ever get
/// a `ready` from.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when the job slot is poisoned.
#[tauri::command]
pub(crate) fn review_close(state: State<'_, AppState>) -> Result<(), CommandError> {
    let open = {
        let mut slot = state.job()?;
        slot.as_mut()
            .filter(|job| job.kind == JobKind::Review)
            .and_then(|job| job.requester.take())
    };
    if let Some(requester) = open {
        // A session already gone is what was asked for; the error would say nothing useful.
        let _ = requester.send(&Request::Close);
    }
    Ok(())
}

/// What the app remembers about this computer (A §11): the executor a sheet starts on, the
/// memory limit, the threads, and where Blender is. Read by the settings screen, and by the
/// launch sheet for the defaults it offers a workspace that has no sheet of its own to repeat.
///
/// Cannot fail: a file that is missing, damaged or written by a newer build is the defaults and
/// a line in the log ([`settings::current`]) — a preferences file is not worth a screen that
/// will not open.
#[tauri::command]
pub(crate) fn settings_get(app: AppHandle) -> Settings {
    settings::current(&app)
}

/// Writes them, and answers with what is now on disk.
///
/// The answer is the file's content and not the argument, for A §5's reason: the window keeps no
/// second opinion about state the shell owns, and what it should now show is what was written —
/// this build's `version`, whatever the window sent.
///
/// **`async`**: `Settings::save` is an atomic write that ends with an `fsync`, which on a slow or
/// a network home directory is tens of milliseconds the main thread would spend not drawing.
///
/// # Errors
///
/// As [`settings::write`]: `json` for a limit this build would not read back, `io` when this
/// machine has no config folder or the file cannot be written.
#[tauri::command(async)]
pub(crate) fn settings_set(app: AppHandle, settings: Settings) -> Result<Settings, CommandError> {
    crate::settings::write(&app, settings)
}

/// The stamp an export's own folder is named after: local time to the minute, exactly as a run's
/// id is (A §4) — two things the user sees side by side in `runs/` and `exports/` should not be
/// dated in two different ways.
const STAMP: &str = "%Y-%m-%d_%H%M";

/// Where «Экспорт» offers to write, before the user has picked anywhere (A §9.1):
/// `<workspace>/exports/<YYYY-MM-DD_HHMM>_<folder|tables>`.
///
/// Inside the workspace, because that is the folder the user already keeps this collection's work
/// in and the one place the app can be sure it may write; the dialog's «Выбрать…» is there for
/// everywhere else. A suggestion and not a decision: the window sends back whatever the user
/// settled on, and [`export_start`] writes there.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, or `io` when the workspace's path is not text this
/// process can hand to the window.
#[tauri::command]
pub(crate) fn export_default_dest(
    state: State<'_, AppState>,
    what: ExportWhat,
) -> Result<String, CommandError> {
    let kind = match what {
        ExportWhat::Folder { .. } => "folder",
        ExportWhat::Tables => "tables",
    };
    let stamp = chrono::Local::now().format(STAMP);
    let held = state.workspace()?;
    let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
    as_text(&ws.exports_dir().join(format!("{stamp}_{kind}")))
}

/// Writes the reviewed assembly of `run_id` into `dest` with the engine's own writers (A §9.1).
///
/// The session is what does it — it is holding the fragments and the match already (A §8) — so
/// this opens one over `run_id` if there is none, and sends A §9.1's `Export` down the same line
/// [`review_apply`] sends a decision. A session over *another* run is closed first and waited
/// for, exactly as a run would close it ([`jobs::start_review`]): an export of run A must not be
/// answered by a session that is holding run B's match.
///
/// **The decisions come from the file and not from the window**, as [`review_refine`]'s do and
/// for the same reason: what is exported must be the assembly `decisions.json` describes, so that
/// the folder written and the folder the app would build again from disk are the same one.
///
/// Everything after this arrives as the session's own events (A §2.1): an `assembly` — the
/// refinement A §9.1 runs first becomes the session's baseline, so the window's state is the
/// state that was exported — then the `output` stage's progress, then `exported`. A refusal (the
/// folder is not empty, there is no room, the scans are gone) is a `request_failed` and the
/// session stays open, which is what lets the dialog offer another folder.
///
/// **`async`** for [`review_apply`]'s reason and one more: opening a session spawns a worker and
/// may wait for another to let the slot go.
///
/// # Errors
///
/// [`CommandError`] of kind `busy` while a `Prepare` or a run has the worker, `no_workspace`,
/// `io` when there is no such run or `dest` is not an absolute path, `json` when the run's
/// `decisions.json` cannot be read, or `worker`.
#[tauri::command(async)]
pub(crate) fn export_start(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: String,
    what: ExportWhat,
    dest: String,
) -> Result<(), CommandError> {
    // Absolute, and checked here rather than trusted: `dest` becomes a folder the worker creates
    // and writes gigabytes into, and a relative one would be resolved against whatever directory
    // this app happens to have been started from — which on a desktop is nobody's choice at all.
    let dest = PathBuf::from(dest);
    if !dest.is_absolute() {
        let why = std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "an export needs the whole path of a folder, not a relative one",
        );
        return Err(AppError::io(&dest, why).into());
    }
    // Nothing when that very session is already open; otherwise the worker is started and the
    // slot is filled before this returns, so the `Export` below reaches it. A request written
    // before the session says `ready` waits in its queue and is answered as soon as it has
    // loaded — the session reads its input from the first moment (A §8).
    jobs::start_review(&app, state.inner(), run_id.clone())?;
    let (open, requester) = session(state.inner())?;
    if open != run_id {
        // The gap between the two lines above is wide enough for another command to have taken
        // the worker for another run. Refusing is right: the export the user asked for is of this
        // run, and the session in the slot cannot answer it.
        return Err(CommandError::busy());
    }
    let dir = run_dir(state.inner(), &run_id)?;
    let decisions = DecisionsFile::load_or_default(&dir)?;
    // Before the request and not after the answer: the `exported` event the window will reveal
    // carries this same folder, and A §2.1 lets the window ask to reveal only what the shell has
    // agreed to. A send that then fails leaves a remembered folder and nothing else.
    *state.last_export()? = Some(dest.clone());
    requester.send(&Request::Export { decisions, what, dest })?;
    Ok(())
}

/// The open review session: the run it is over, and the handle its questions go down.
///
/// The lock is taken and let go of here, not held: what follows writes a file and writes to a
/// pipe, and no lock of this app is held across either (see [`crate::state`]). What the gap can
/// hold is the session ending underneath, and that is the `worker` error [`Requester::send`]
/// gives — the same answer, arrived at a moment later.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when no session is open or the job slot is poisoned.
fn session(state: &AppState) -> Result<(String, Requester), CommandError> {
    let slot = state.job()?;
    let open = slot.as_ref().filter(|job| job.kind == JobKind::Review);
    match open.and_then(|job| job.run_id.clone().zip(job.requester.clone())) {
        Some(session) => Ok(session),
        None => Err(CommandError::no_session()),
    }
}

/// What the shell checks before a decision list becomes `decisions.json` (A §8.1).
///
/// Two things, and both are about the *file*, not about the review: a version this build could
/// not read back, and a pair decided twice. The second is the file's own invariant — one decision
/// per unordered pair — and a list that broke it would put two lines about one pair into the
/// engine's `constraints.json` and leave the window's own undo stack disagreeing with the disk.
///
/// Everything else is the session's to answer, not the shell's: a decision naming a fragment the
/// collection no longer has is reported as `dropped` (A §8.5), and a pose that is not rigid is
/// `constraints::resolve`'s refusal.
///
/// # Errors
///
/// [`CommandError`] of kind `json`.
fn checked(decisions: &DecisionsFile) -> Result<(), CommandError> {
    if decisions.version > DECISIONS_VERSION {
        return Err(CommandError::malformed(format!(
            "{DECISIONS_FILE}: the window sent format {}, this build writes {DECISIONS_VERSION}",
            decisions.version
        )));
    }
    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    for decision in &decisions.decisions {
        let (a, b) = (decision.a.as_str(), decision.b.as_str());
        if a.is_empty() || b.is_empty() || a == b {
            return Err(CommandError::malformed(format!(
                "{DECISIONS_FILE}: {a:?} and {b:?} are not two fragments"
            )));
        }
        if !seen.insert(if a <= b { (a, b) } else { (b, a) }) {
            return Err(CommandError::malformed(format!(
                "{DECISIONS_FILE}: {a} – {b} is decided twice"
            )));
        }
    }
    Ok(())
}

/// The end of a job's log, for A §10's «Показать лог»: a run's `engine.log` by id, or the
/// `prepare.log` beside `sherd-workspace.json` for `None` (A §4 — a `Prepare` has no run folder).
///
/// The tail and not the file: a long run's log is megabytes of `tracing`, and what the drawer
/// shows is its last few hundred lines. A log that does not exist yet is an empty string rather
/// than a refusal — the window offers «Показать лог» before anything has been written into it.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, or `io` when there is no such run or the log is
/// there and cannot be read.
#[tauri::command]
pub(crate) fn run_log(
    state: State<'_, AppState>,
    run_id: Option<String>,
    max_lines: usize,
) -> Result<String, CommandError> {
    let path = {
        let held = state.workspace()?;
        let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
        match &run_id {
            Some(id) => run_dir_of(ws, id)?.join(host::ENGINE_LOG),
            None => ws.root().join(jobs::PREPARE_LOG),
        }
    };
    let bytes = read_log_tail(&path)?;
    // Lossy, deliberately: a log is what the user is shown when something has already gone wrong,
    // and a byte the engine's own panic message mangled must not be what stops them reading it.
    let text = String::from_utf8_lossy(&bytes);
    Ok(tail(&text, max_lines).to_owned())
}

/// How much of a log is read to find its last lines. Four mebibytes is thousands of lines of
/// `tracing` — far more than any `max_lines` A §10's drawer asks for — and a log that has grown
/// past it must not be loaded whole into the window's process every two seconds, which is how
/// often the drawer refreshes itself while a job is running.
const LOG_TAIL_BYTES: u64 = 4 * 1024 * 1024;

/// The last [`LOG_TAIL_BYTES`] of `path`, or all of it when it is shorter.
///
/// A log that is not there yet is no bytes and not a refusal: A §10 offers «Показать лог» from
/// the moment a job starts, and the worker may not have written its first line.
fn read_log_tail(path: &Path) -> Result<Vec<u8>, CommandError> {
    let failed = |source| AppError::io(path, source);
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => return Err(failed(source).into()),
    };
    let len = file.seek(SeekFrom::End(0)).map_err(failed)?;
    let from = len.saturating_sub(LOG_TAIL_BYTES);
    file.seek(SeekFrom::Start(from)).map_err(failed)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(failed)?;
    // A read that began in the middle of the file began in the middle of a line, and possibly in
    // the middle of a character: both go with everything up to the first newline.
    if from > 0
        && let Some(at) = bytes.iter().position(|byte| *byte == b'\n')
    {
        bytes.drain(..=at);
    }
    Ok(bytes)
}

/// Deletes a run: its folder goes to the OS trash (A §4), never to `remove_dir_all`.
///
/// The trash and not a delete, because a run is hours of someone's machine and the confirmation
/// dialog is one keystroke away from an accident; the OS's own undo is the only one the app has.
///
/// The run a worker is writing is refused with `busy`: its folder is open, and moving it under a
/// live worker would leave the run's own files landing in a folder that is no longer there.
///
/// # Errors
///
/// [`CommandError`] of kind `busy`, `no_workspace`, `io` when there is no such run or the trash
/// will not take the folder, or `engine` when the view cannot be rebuilt afterwards.
#[tauri::command]
pub(crate) fn run_delete(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<WorkspaceView, CommandError> {
    // The folder is resolved under both locks — job slot first, then the workspace, as
    // `crate::state` requires — and **both are let go of before the trash is asked to take it**.
    // `trash::delete` is a call into the desktop's own trash service and takes as long as the
    // disk does: a folder on a network share that has gone to sleep can hold it for a minute, and
    // with the job slot held that minute is one in which every command the window makes waits —
    // «Отменить» among them. No lock of this app is held across a file operation of unbounded
    // time.
    //
    // What the gap can hold is a job started between the check and the delete, and it is not the
    // job this check is about: `host::start_run` names a new run after the current minute and
    // never after a folder that already exists, so the run being moved away cannot become the one
    // being written. The run that *is* being written is refused above, under the lock.
    let dir = {
        let slot = state.job()?;
        if slot.as_ref().is_some_and(|job| job.run_id.as_deref() == Some(run_id.as_str())) {
            return Err(CommandError::busy());
        }
        let held = state.workspace()?;
        let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
        run_dir_of(ws, &run_id)?
    };
    trash::delete(&dir).map_err(|source| {
        let why = format!("the run could not be moved to the trash: {source}");
        AppError::io(&dir, std::io::Error::other(why))
    })?;
    // And the view is read again afterwards, in the same order, so that it describes the
    // workspace as the delete left it and not as it was before.
    let slot = state.job()?;
    let job = slot.as_ref().map(JobSlot::view);
    let held = state.workspace()?;
    let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
    Ok(view::build(ws, job)?)
}

/// What «Открыть в Blender» came to (A §9.2).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct BlenderOutcome {
    /// Whether Blender was found and started. `false` is the ordinary answer on a machine that
    /// has none, not a failure: the script is written either way.
    pub(crate) launched: bool,
    /// The script that was written, whole path — what «Показать в папке» reveals.
    pub(crate) script: String,
    /// The Blender that was started, or `None` when none was found; the toast names it.
    pub(crate) blender: Option<String>,
}

/// What the file a Blender launch writes is called, inside its own dated folder.
const BLENDER_SCRIPT: &str = "open_in_blender.py";

/// «Открыть в Blender» (A §9.2): the assembly, or one group of it, as a Python script — written
/// into `exports/<stamp>_blender/`, and run by Blender when this machine has one.
///
/// **The script first, the launch second, and neither depends on the other.** Blender is not
/// installed on most museum machines and is installed in five different places on the rest, so
/// «не найден» is an ordinary outcome and the file must already be on disk when it happens: the
/// window then offers «Показать в папке» and «Указать путь к Blender…», and the colleague who
/// has Blender on another computer has something to carry there.
///
/// Started **detached** and never waited for: Blender is a window of its own that the person will
/// keep open for an hour, and the app must not hold a thread, a pipe or its own exit on it. Its
/// three streams go to nowhere for the same reason — a full pipe nobody reads would stop it.
///
/// **`async`**: it reads two of the run's files, writes a third and spawns a process, none of
/// which belongs on the thread that draws the window.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, `io` when there is no such run, when the original
/// scans are asked for and the input folder is not available, or when the script cannot be
/// written, and `json` when the run's `assembly.json` or `run.json` does not parse.
#[tauri::command(async)]
pub(crate) fn blender_open(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: String,
    scope: ScopeDto,
    resolution: ResolutionDto,
) -> Result<BlenderOutcome, CommandError> {
    let resolution = Resolution::from(resolution);
    // Everything the workspace has to say, under its lock and no longer: what follows reads
    // files, writes one and starts a process, and no lock of this app is held across any of those
    // ([`crate::state`]).
    let (dir, input, fragments, exports, title) = {
        let held = state.workspace()?;
        let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
        let dir = run_dir_of(ws, &run_id)?;
        // The original scans need the input folder; the display meshes are the workspace's own
        // and do not. A collection whose input has moved can still be opened in Blender as the
        // window draws it, which is the lighter of A §9.2's two choices anyway.
        let input = match (ws.input(), resolution) {
            (Some(input), _) => input,
            (None, Resolution::Display) => PathBuf::new(),
            (None, Resolution::Full) => {
                let why = std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "the original scans are not available; open the lighter models instead, or \
                     point the workspace at the folder again",
                );
                return Err(AppError::io(ws.root(), why).into());
            }
        };
        let name = ws.root().file_name().unwrap_or_default().to_string_lossy().into_owned();
        let title = if name.is_empty() { run_id.clone() } else { format!("{name} · {run_id}") };
        (dir, input, ws.fragments_dir(), ws.exports_dir(), title)
    };
    let assembly: AssemblyDto = atomic::read_json(&dir.join(host::ASSEMBLY_FILE))?;
    // The run's own snapshot and not the folder as it is now: it names the file each fragment was
    // read from, which is what the script has to import (A §9.2).
    let snapshot = run::RunFile::load(&dir)?.input;
    let groups = blender::groups_of(&assembly, &snapshot, &input, &fragments, scope.into());
    let script = blender::script(&title, &groups, resolution);
    let path = write_script(&exports, &script)?;
    // Read back as text before anything is started: a path this process cannot put into JSON is
    // a toast with nowhere to send the user, and finding that out after Blender has opened would
    // leave the window reporting a failure over a launch that worked.
    let written = as_text(&path)?;
    let blender = blender::find_blender(settings::current(&app).blender_path.as_deref());
    let launched = blender.as_deref().is_some_and(|exe| launch(exe, &path));
    Ok(BlenderOutcome {
        launched,
        script: written,
        blender: blender.as_deref().map(|exe| exe.to_string_lossy().into_owned()),
    })
}

/// Writes one generated script into a folder of its own under `exports/`, and answers where.
///
/// A folder per launch, dated to the minute like a run (A §4) and given `-2`, `-3` when that
/// minute is taken — which is not a corner case here: opening group 1 and then group 2 is two
/// launches within a few seconds, and the second overwriting the first would pull the file out
/// from under a Blender that is still starting up.
fn write_script(exports: &Path, script: &str) -> Result<PathBuf, CommandError> {
    let stamp = chrono::Local::now().format(STAMP);
    let mut dir = exports.join(format!("{stamp}_blender"));
    // Bounded rather than open, as `run::new_id` is: a hundred launches in one minute is already
    // impossible by hand, and a loop with no end has no business in a command.
    for n in 2..=100 {
        if !dir.exists() {
            break;
        }
        dir = exports.join(format!("{stamp}_blender-{n}"));
    }
    std::fs::create_dir_all(&dir).map_err(|source| AppError::io(&dir, source))?;
    let path = dir.join(BLENDER_SCRIPT);
    std::fs::write(&path, script).map_err(|source| AppError::io(&path, source))?;
    Ok(path)
}

/// Starts `blender --python <script>` and leaves it to itself; answers whether it started.
///
/// The child is reaped on a thread of its own rather than waited for or forgotten: forgotten, it
/// would sit in the process table as a zombie for as long as this app is open; waited for, it
/// would hold a command thread for the whole hour the person spends in Blender. The thread does
/// nothing but block in `wait`, and it ends when Blender does.
///
/// Nothing here fails the command. A Blender that will not start is «не запустился» in the
/// toast, with the script already written beside it — see [`blender_open`].
fn launch(blender: &Path, script: &Path) -> bool {
    let started = std::process::Command::new(blender)
        .arg("--python")
        .arg(script)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    match started {
        Ok(mut child) => {
            let reaped =
                std::thread::Builder::new().name("sherd-blender".to_owned()).spawn(move || {
                    let _ = child.wait();
                });
            if let Err(error) = reaped {
                tracing::warn!(%error, "Blender was started but will not be reaped until exit");
            }
            true
        }
        Err(error) => {
            tracing::warn!(
                blender = %blender.display(),
                %error,
                "Blender was found but would not start; the script is on disk"
            );
            false
        }
    }
}

/// «Показать в папке» (A §9.1, A §9.2): asks the desktop to open the folder holding `path` with
/// that file selected.
///
/// **Two folders and no others**: the open workspace, and the folder the last export of this
/// session was written into. A §2.1 gives the window nothing outside the workspace and lets it
/// ask for nothing outside it, and an export's destination is the single thing the user
/// deliberately puts elsewhere — so it is remembered ([`export_start`]) rather than trusted from
/// the argument. A path that does not resolve is refused as well: there is nothing to show, and
/// a comparison of prefixes is only meaningful between paths that exist.
///
/// **`async`**: `reveal_item_in_dir` is a call into the desktop's own file manager, which on a
/// cold Finder or Explorer is a second of somebody else's work.
///
/// # Errors
///
/// [`CommandError`] of kind `io` when the path does not exist, is not one of the two the window
/// may ask for, or the desktop will not show it.
#[tauri::command(async)]
pub(crate) fn reveal(state: State<'_, AppState>, path: String) -> Result<(), CommandError> {
    let path = PathBuf::from(path);
    let refused = |why: &str| -> CommandError {
        AppError::io(&path, std::io::Error::new(std::io::ErrorKind::PermissionDenied, why)).into()
    };
    let Ok(resolved) = path.canonicalize() else {
        return Err(refused("there is nothing at this path to show"));
    };
    // One lock at a time and neither held with the other: these two are about different things
    // — the folder that is open and what this session has written — and a path that needs both
    // held at once would put them into a locking order they do not have ([`crate::state`]).
    let workspace = state.workspace()?.as_ref().map(|ws| ws.root().to_owned());
    let export = state.last_export()?.clone();
    let inside = [workspace, export]
        .into_iter()
        .flatten()
        .filter_map(|root| root.canonicalize().ok())
        .any(|root| resolved.starts_with(&root));
    if !inside {
        return Err(refused("this is neither the open workspace nor the last export"));
    }
    tauri_plugin_opener::reveal_item_in_dir(&resolved)
        .map_err(|source| AppError::io(&resolved, std::io::Error::other(source)).into())
}

/// «Снимок PNG» (A §7.2): writes the frame the window drew to the file the user named in the
/// OS's own save dialog.
///
/// **The path is the dialog's answer and is therefore not checked against the workspace**, which
/// is the one place in this file where that is so. A §2.1 keeps the *window* from reaching
/// outside the workspace; a save dialog is the user reaching outside it themselves, with the
/// desktop's own file picker, and a snapshot nobody may put on their desktop is not a snapshot.
/// Nothing is read, nothing is overwritten that the picker did not already warn about, and the
/// bytes are the window's own canvas.
///
/// Base64 and not an array of bytes: `toDataURL` hands the window base64 already, and a
/// megapixel PNG as a JSON array of numbers is twenty times its own size on the way through the
/// IPC. A string that is not base64 at all is a refusal and never a panic (A §10 — everything
/// the window sends is data from outside).
///
/// **`async`**: it writes a file of a few megabytes, which is not the drawing thread's work.
///
/// # Errors
///
/// [`CommandError`] of kind `json` when `png` is not base64, or `io` when the file cannot be
/// written.
#[tauri::command(async)]
pub(crate) fn snapshot_save(path: String, png: String) -> Result<(), CommandError> {
    let path = PathBuf::from(path);
    let bytes = from_base64(&png).ok_or_else(|| {
        CommandError::malformed(format!(
            "{}: the snapshot did not arrive as base64",
            path.display()
        ))
    })?;
    std::fs::write(&path, bytes).map_err(|source| AppError::io(&path, source).into())
}

/// Standard base64 (RFC 4648, no URL alphabet) as bytes, or `None` for anything that is not.
///
/// Written here rather than taken from a crate: it is fifteen lines, this is its only caller,
/// and a dependency added to a desktop shell is a dependency in every release build of it.
/// Padding is accepted and not required; whitespace is not — `toDataURL` writes none, and being
/// lenient about what a malformed payload may contain is how a decoder grows holes.
fn from_base64(text: &str) -> Option<Vec<u8>> {
    /// One character's six bits, or `None` for anything outside the alphabet.
    fn sextet(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some(u32::from(c - b'A')),
            b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let body = text.trim_end_matches('=').as_bytes();
    // Four characters carry three bytes; a group of one is a length no encoder can produce.
    if body.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(body.len() / 4 * 3);
    for group in body.chunks(4) {
        let mut bits = 0_u32;
        for (i, &c) in group.iter().enumerate() {
            bits |= sextet(c)? << (18 - 6 * i);
        }
        // One byte per whole eight bits the group carries: 4 → 3, 3 → 2, 2 → 1. Through
        // `to_le_bytes` and not an `as u8`, so nothing here is a truncating cast.
        for i in 0..group.len() - 1 {
            out.push((bits >> (16 - 8 * i)).to_le_bytes()[0]);
        }
    }
    Some(out)
}

/// A path as the window receives it.
///
/// Lossless or nothing: a `to_string_lossy` here would hand the window a path with a replacement
/// character in it, which it would send back as the folder to export into — and that folder is
/// not the one the user picked. Such a path cannot be put into JSON at all, so it is said once,
/// here, with the path named as far as it can be printed.
///
/// # Errors
///
/// [`CommandError`] of kind `io`.
fn as_text(path: &Path) -> Result<String, CommandError> {
    match path.to_str() {
        Some(text) => Ok(text.to_owned()),
        None => {
            let why = std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{} cannot be written as text", path.display()),
            );
            Err(AppError::io(path, why).into())
        }
    }
}

/// What this build can run on (A §7.4's «Вычисления»), as the launch sheet says it under the
/// backend choice.
///
/// The names and not the engine's lines. `sherd-refit-rs info` writes for an operator reading a
/// terminal — «cpu, gpu (wgpu 30.0.1, 1 adapter; R §5.2's coarse score and R §5.4's stage-1 ICP
/// rungs on the device, …)» — and A §7.4 asks the sheet for one thing: which card a run would go
/// to. Everything else in those lines is the app talking to its own developers.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct EngineInfoView {
    /// Every adapter the engine offered, by name: «Metal Apple M2 Pro». Empty when it found none.
    pub(crate) adapters: Vec<String>,
    /// Whether A §7.4's «GPU» is a real choice on this machine — a fact the window branches on,
    /// rather than the length of a list it would have to know the meaning of.
    pub(crate) gpu: bool,
}

/// What the engine can run on. Asked of a worker once and kept (A §2.1: only the engine's own
/// process knows what adapters this build sees, and the window may not load a GPU driver).
///
/// The job slot is not taken: this is a process that lives for a few milliseconds and writes
/// nothing, not one of A §5's two job kinds, and the sheet asks for it while a run is going as
/// well as before one. Two sheets opened at the same moment would ask twice and store the same
/// answer twice, which is a wasted process and not a wrong one — the alternative is a lock held
/// across a spawn, which [`crate::state`] does not allow.
///
/// **`async`, and not for the sake of an `await`.** A plain `#[tauri::command]` runs on the main
/// thread, and this one spawns a process and drives it to its end: opening a GPU instance and
/// enumerating its adapters is tens of milliseconds on a warm machine and can be a second on a
/// cold driver, and all of it would be a window that does not repaint. `(async)` puts the call on
/// Tauri's pool instead; nothing inside it is held across a suspension point, because there is
/// none.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when the engine process cannot be started or says it
/// failed.
#[tauri::command(async)]
pub(crate) fn engine_info(state: State<'_, AppState>) -> Result<EngineInfoView, CommandError> {
    {
        let cached = state.engine_info()?;
        if let Some(info) = cached.as_ref() {
            return Ok(info.clone());
        }
    }
    let info = ask_engine_info()?;
    *state.engine_info()? = Some(info.clone());
    Ok(info)
}

/// One `Job::Info` worker, driven to its end. `selftest: false`: opening the device costs a
/// second, and the sheet wants the lines, not the test (A §7.4 — the self-test is its own button).
fn ask_engine_info() -> Result<EngineInfoView, CommandError> {
    let command = jobs::worker_command()?;
    let job = Job::Info { adapter: None, selftest: false };
    // No log: an `Info` has no run folder to keep one in, and its stderr is read and dropped by
    // the worker's own reader thread so that a full pipe cannot stop it.
    let mut worker = Worker::spawn(&command, &job, None)?;
    let mut backends = Vec::new();
    let outcome = host::drive(&mut worker, |event| {
        if let Event::Info { backends: lines, .. } = event {
            backends.clone_from(lines);
        }
    });
    match outcome {
        Outcome::Done { .. } => {
            let adapters = adapter_names(&backends);
            Ok(EngineInfoView { gpu: !adapters.is_empty(), adapters })
        }
        Outcome::Failed { message, .. } => Err(AppError::Worker(message).into()),
    }
}

/// The adapters named in `sherd-refit-rs info`'s backend lines, in enumeration order.
///
/// The engine writes one summary line and then one **indented** line per adapter, which is
/// `sherd_gpu::AdapterEntry`'s own `Display`: `  [0] Metal Apple M2 Pro (IntegratedGpu)`, with a
/// ` driver=… …` tail where the platform has one. A build with no adapter, and a build without
/// the GPU feature at all, write the summary line and nothing else — so no adapter lines is the
/// honest «none», and not a parse that failed.
fn adapter_names(lines: &[String]) -> Vec<String> {
    lines.iter().filter_map(|line| adapter_name(line)).collect()
}

/// One adapter line as a name, or `None` for anything that is not one.
///
/// The indentation is the test, then a bracketed number: a summary line is not indented, and a
/// line some later build adds under one is not `[0] …`. What comes off the end is the developer's
/// half — the device type and the driver — because the sheet's sentence is «Видеокарта: Metal
/// Apple M2 Pro» and A §7.4 asks for which card and nothing more.
fn adapter_name(line: &str) -> Option<String> {
    if !line.starts_with([' ', '\t']) {
        return None;
    }
    let (index, tail) = line.trim().strip_prefix('[')?.split_once("] ")?;
    index.parse::<usize>().ok()?;
    let named = match tail.split_once(" driver=") {
        Some((head, _)) => head,
        None => tail,
    };
    // `(IntegratedGpu)` last, after the driver tail is off: an adapter's own name may hold
    // brackets («AMD Radeon (TM) Graphics»), and it is the final group that is the device type.
    let named = match named.trim_end().strip_suffix(')').and_then(|body| body.rsplit_once(" (")) {
        Some((name, _type)) => name,
        None => named,
    };
    let named = named.trim();
    (!named.is_empty()).then(|| named.to_owned())
}

/// The folder of the run `run_id` names, on the open workspace.
///
/// # Errors
///
/// [`CommandError`] of kind `no_workspace`, or `io` as [`run_dir_of`].
fn run_dir(state: &AppState, run_id: &str) -> Result<PathBuf, CommandError> {
    let held = state.workspace()?;
    run_dir_of(held.as_ref().ok_or_else(CommandError::no_workspace)?, run_id)
}

/// The folder of the run `run_id` names, on a workspace already in hand.
///
/// **A run id becomes a path, so it is checked against the runs that exist and not against a list
/// of forbidden characters.** `run::list` answers the names of folders it read out of `runs/`, so
/// a `..`, a separator or the name of something elsewhere on the disk is simply not among them,
/// and no spelling of any of those can reach the file system through here (A §2.1: the window is
/// given nothing outside the workspace and may ask for nothing outside it).
///
/// # Errors
///
/// [`CommandError`] of kind `io` when `runs/` cannot be listed, or when it holds no such run.
fn run_dir_of(ws: &Workspace, run_id: &str) -> Result<PathBuf, CommandError> {
    if run::list(&ws.runs_dir())?.iter().any(|run| run.id == run_id) {
        return Ok(ws.run_dir(run_id));
    }
    // The folder that was searched, and the name as it was asked for — not the two joined, which
    // would be the path this function exists to refuse to build.
    let why = std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("the workspace has no run named {run_id:?}"),
    );
    Err(AppError::io(ws.runs_dir(), why).into())
}

/// The last `max_lines` lines of `text`, as a slice of it: all of it when it has fewer, and
/// nothing at all for `0`.
///
/// A slice and not a `String`, so that showing the end of a twenty-megabyte log copies the few
/// kilobytes the window asked for and not the log. A final newline ends the last line rather than
/// beginning an empty one, which is what makes «the last 400 lines» of a log a worker is still
/// writing the same 400 lines a moment later.
fn tail(text: &str, max_lines: usize) -> &str {
    if max_lines == 0 {
        return "";
    }
    let body = text.strip_suffix('\n').unwrap_or(text);
    match body.match_indices('\n').nth_back(max_lines - 1) {
        Some((at, _)) => &text[at + 1..],
        None => text,
    }
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
/// The folder that *is* open is therefore answered before anything is opened *or closed* at all:
/// its own lock would refuse a second [`Workspace::open`] on it, and «open what I already have»
/// means «show me what I have» — not `locked`, and not a warm review session ended for a
/// workspace that is not changing. Whatever is in the job slot goes into that view, exactly as
/// [`workspace_view`] builds it.
///
/// A review session over the workspace being *left* is then closed and waited for (A §8.4): a
/// session is a cache over a finished run and never a reason to refuse the user the workspace
/// they have just asked for. A `Prepare` or a run still is: its worker is writing into the folder
/// this is about to drop.
///
/// The rest of the order is A §4's: the runs left `running` by a crash are marked before any
/// worker of ours exists, so a `running` on disk can only be nobody's; and the asset protocol is
/// widened to the new root before the window is given a view naming files under it (A §2.1 — the
/// window is given nothing outside the workspace). The old workspace is dropped, and its lock
/// released, by the assignment at the end, once nothing can fail any more.
///
/// From the moment the slot is found empty the job slot is *held*, and not merely read: let go
/// of after the check, it leaves a gap in which `prepare_start` can fill the slot and spawn a
/// worker on the workspace
/// that the assignment below is about to drop — releasing A §10's `sherd-workspace.lock` under a
/// live worker of ours, and letting `run::mark_interrupted` write `interrupted` over a run that
/// is still being written. Job first, then the workspace, as [`crate::state`] requires.
fn open_with(
    app: &AppHandle,
    state: &AppState,
    path: &Path,
    open: fn(&Path) -> sherd_app_core::Result<Workspace>,
) -> Result<WorkspaceView, CommandError> {
    {
        // Both locks in [`crate::state`]'s order, and let go of at the end of this block: the
        // wait below must hold neither.
        let slot = state.job()?;
        let held = state.workspace()?;
        if let Some(open_already) = held.as_ref()
            && same_folder(open_already.root(), path)
        {
            return Ok(view::build(open_already, slot.as_ref().map(JobSlot::view))?);
        }
    }
    // A §8.4, and with nothing held: the session's own thread takes the job lock to give the
    // slot back (`jobs::finish`), so holding it here would be a deadlock and not a wait.
    jobs::close_session(state)?;
    let slot = state.job()?;
    if slot.is_some() {
        return Err(CommandError::busy());
    }
    let mut held = state.workspace()?;
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

/// Where the app keeps what is not a workspace's (A §4): the recent list, A §6's calibration and
/// A §11's settings.
pub(crate) fn config_dir(app: &AppHandle) -> Result<PathBuf, CommandError> {
    Ok(app.path().app_config_dir()?)
}

#[cfg(test)]
mod tests {
    use sherd_app_core::decisions::{Decision, DecisionsFile, Verdict};

    use super::{adapter_names, checked, from_base64, tail};

    fn decision(a: &str, b: &str) -> Decision {
        Decision {
            a: a.to_owned(),
            b: b.to_owned(),
            verdict: Verdict::Accept,
            pose: None,
            source: Some("probable".to_owned()),
            bulk: false,
            at: "2026-09-21T12:00:00+03:00".to_owned(),
            carried_from: None,
        }
    }

    /// A §8.1: `decisions.json` holds one decision per unordered pair, and a version this build
    /// can read back. Both are refused **before** anything is written, because what the shell
    /// will not file is exactly what the next `review_open` could not load.
    #[test]
    fn a_decision_list_that_could_not_be_filed_again_is_refused_before_it_is_sent() {
        let mut file = DecisionsFile::default();
        file.decisions.push(decision("pieceA", "pieceB"));
        file.decisions.push(decision("pieceB", "pieceC"));
        assert!(checked(&file).is_ok());

        // The same pair, named the other way round: `DecisionsFile::set` keeps one of these and
        // the window's own list must too, or the engine is told twice about one join.
        let mut twice = file.clone();
        twice.decisions.push(decision("pieceB", "pieceA"));
        let refused = checked(&twice).expect_err("one pair, one decision");
        assert_eq!(refused.kind, "json");
        assert!(refused.message.contains("pieceB – pieceA"), "{}", refused.message);

        // A fragment with itself, and an empty name: neither is a pair.
        let mut itself = DecisionsFile::default();
        itself.decisions.push(decision("pieceA", "pieceA"));
        assert!(checked(&itself).is_err());
        let mut nameless = DecisionsFile::default();
        nameless.decisions.push(decision("pieceA", ""));
        assert!(checked(&nameless).is_err());

        // And a list a newer app wrote, which this build would not read back.
        let newer = DecisionsFile { version: file.version + 1, ..file };
        assert!(checked(&newer).is_err());
    }

    /// A §7.4: the sheet says which card a run would go to, and the engine's `info` says a great
    /// deal more than that. What it says is parsed, not shown.
    #[test]
    fn the_sheet_is_given_the_adapters_and_none_of_the_prose() {
        let lines = [
            "cpu, gpu (wgpu 30.0.1, 2 adapters; R §5.2's coarse score and R §5.4's stage-1 ICP \
             rungs on the device, R §5.6's stage 2 and R §6's two methods on the CPU (policy; D \
             §12's 2c struck))",
            "  [0] Metal Apple M2 Pro (IntegratedGpu)",
            "  [1] Vulkan AMD Radeon (TM) Graphics (DiscreteGpu) driver=amdvlk 2024.Q3.1",
        ]
        .map(str::to_owned);
        assert_eq!(
            adapter_names(&lines),
            ["Metal Apple M2 Pro", "Vulkan AMD Radeon (TM) Graphics"]
        );

        // A build that found no adapter, and a build without the GPU feature, each say so in one
        // unindented line: no adapters, and nothing mistaken for one.
        let none = ["cpu; gpu built in (wgpu 30.0.1) but no Metal, Vulkan or DX12 adapter found"]
            .map(str::to_owned);
        assert!(adapter_names(&none).is_empty());
        let no_feature = ["cpu (built without the `gpu` feature)".to_owned()];
        assert!(adapter_names(&no_feature).is_empty());

        // And a line the engine may add under a backend that is not an adapter is not one.
        let other = ["  limits: maxBufferSize 2 GiB".to_owned(), "  [x] nonsense".to_owned()];
        assert!(adapter_names(&other).is_empty());
    }

    /// A §10's «Показать лог» shows the end of a log and not the whole of one, and the end of a
    /// log is the part a reviewer needs: the lines around what went wrong, last.
    #[test]
    fn a_log_is_shown_from_its_end() {
        assert_eq!(tail("a\nb\nc", 2), "b\nc");
        // A trailing newline ends the last line; it does not begin an empty one.
        assert_eq!(tail("a\nb\nc\n", 2), "b\nc\n");
        // Fewer lines than asked for is all of them, and asking for none is none.
        assert_eq!(tail("a\nb", 5), "a\nb");
        assert_eq!(tail("one line", 1), "one line");
        assert_eq!(tail("a\nb\nc", 0), "");
        assert_eq!(tail("", 400), "");
        // Blank lines are lines: three newlines are three empty ones, and the last is the last.
        assert_eq!(tail("\n\n\n", 1), "\n");
        // The whole of a text whose line count is exactly what was asked for.
        assert_eq!(tail("a\nb\nc", 3), "a\nb\nc");
    }

    /// A §7.2's «Снимок PNG» arrives as base64 out of the webview, which is data from outside:
    /// every shape of it has to answer, and none of them may panic (A §10).
    #[test]
    fn a_snapshot_is_decoded_or_refused_and_never_panicked_over() {
        // RFC 4648's own vectors, with and without the padding the decoder does not require.
        assert_eq!(from_base64("Zm9vYmFy").as_deref(), Some(&b"foobar"[..]));
        assert_eq!(from_base64("Zm9vYmE=").as_deref(), Some(&b"fooba"[..]));
        assert_eq!(from_base64("Zm9vYmE").as_deref(), Some(&b"fooba"[..]));
        assert_eq!(from_base64("Zm9vYg==").as_deref(), Some(&b"foob"[..]));
        assert_eq!(from_base64("").as_deref(), Some(&b""[..]));
        // A PNG's own first eight bytes, which is what a real snapshot starts with.
        assert_eq!(
            from_base64("iVBORw0KGgo=").as_deref(),
            Some(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A][..])
        );
        // Both alphabet's last two characters, which a URL-safe encoder would write as `-_`.
        assert_eq!(from_base64("+/8=").as_deref(), Some(&[0xFB, 0xFF][..]));
        assert_eq!(from_base64("-_8="), None);
        // A group of one carries no whole byte: no encoder writes one.
        assert_eq!(from_base64("Zm9vYmFyZ"), None);
        // And anything that is not the alphabet — a newline, a space, a data URL left whole.
        assert_eq!(from_base64("Zm9v YmFy"), None);
        assert_eq!(from_base64("Zm9v\nYmFy"), None);
        assert_eq!(from_base64("data:image/png;base64,Zm9v"), None);
    }
}
