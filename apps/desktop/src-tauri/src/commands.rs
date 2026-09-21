//! Everything the window may ask of the shell.
//!
//! One shape throughout (A §5): every command that changes the workspace answers with the whole
//! [`WorkspaceView`], so the window never has to guess what its own change did and never keeps a
//! second opinion about the workspace to hold in step with the first. The window asks; the shell
//! reads and writes the folder (A §2.1).

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sherd_app_core::eta::{self, Calibration};
use sherd_app_core::host::{self, Outcome, Worker};
use sherd_app_core::protocol::{AssemblyDto, CandidateRow, Event, Job, RunSpec};
use sherd_app_core::view::{self, WorkspaceView};
use sherd_app_core::workspace::Workspace;
use sherd_app_core::{AppError, atomic, run, snapshot};
use tauri::{AppHandle, Manager, State};

use crate::error::CommandError;
use crate::jobs::{self, Lang};
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
/// `lang` is the window's own language (`ru`, `en`; anything else reads as `ru`). The job carries
/// it so that A §6's end-of-job notification — written by the shell and shown by the OS, by which
/// time the window may be behind something else — is in the language the user is reading.
///
/// # Errors
///
/// [`CommandError`] of kind `busy`, `no_workspace`, `worker`, `io` or `engine`; see
/// [`jobs::start_prepare`].
#[tauri::command]
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
/// *this* run would have wanted (A §7.4). `lang` is as [`prepare_start`]'s.
///
/// # Errors
///
/// [`CommandError`] of kind `busy`, `no_workspace`, `worker`, `io`, `json` or `engine`; see
/// [`jobs::start`].
#[tauri::command]
pub(crate) fn run_start(
    app: AppHandle,
    state: State<'_, AppState>,
    spec: RunSpec,
    lang: String,
) -> Result<String, CommandError> {
    jobs::start_run(&app, state.inner(), spec, Lang::of(&lang))
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
    // Job slot first, then the workspace, as `crate::state` requires — and held to the end, not
    // sampled: let go of after the check, it leaves a gap in which a run starts on the very
    // folder about to be moved away.
    let slot = state.job()?;
    if slot.as_ref().is_some_and(|job| job.run_id.as_deref() == Some(run_id.as_str())) {
        return Err(CommandError::busy());
    }
    let job = slot.as_ref().map(JobSlot::view);
    let held = state.workspace()?;
    let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
    let dir = run_dir_of(ws, &run_id)?;
    trash::delete(&dir).map_err(|source| {
        let why = format!("the run could not be moved to the trash: {source}");
        AppError::io(&dir, std::io::Error::other(why))
    })?;
    Ok(view::build(ws, job)?)
}

/// What this build can run on (A §7.4's «Вычисления»), as the launch sheet lists it under the
/// backend choice.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct EngineInfoView {
    /// One line per executor, in the engine's own words.
    pub(crate) backends: Vec<String>,
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
/// # Errors
///
/// [`CommandError`] of kind `worker` when the engine process cannot be started or says it
/// failed.
#[tauri::command]
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
        Outcome::Done { .. } => Ok(EngineInfoView { backends }),
        Outcome::Failed { message, .. } => Err(AppError::Worker(message).into()),
    }
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

#[cfg(test)]
mod tests {
    use super::tail;

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
}
