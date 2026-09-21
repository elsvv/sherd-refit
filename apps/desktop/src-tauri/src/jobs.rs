//! A job in the shell (A §2.1): the app starts *itself* as the engine, drives that process on a
//! thread of its own, and forwards everything it says to the window.
//!
//! Nothing of the engine runs in the window's process — that is the whole point of the two roles
//! (A §2.1): a 3 GB match and a GPU device live in a child that can be killed, and the window
//! stays answerable while it works. So [`start`] returns as soon as the worker is on its way, and
//! everything after that reaches the frontend as two events: `engine:event` for each line the
//! worker wrote, and `engine:finished` once, carrying the whole [`WorkspaceView`] the window
//! should now be showing (A §5: one shape, so the window never keeps a second opinion).
//!
//! All three kinds of job go through one function, because everything around them is the same:
//! the one slot, the one thread, the two events. What a run adds is what the host owns and the
//! worker may not touch (A §2.1) — `assembly.json` the moment the groups are reported, `run.json`
//! closed when it is over, and what it measured taught to A §6's calibration. A review session
//! (A §8) adds the other half of the slot: a handle the window's questions go down, and the
//! rule that a `Prepare` or a run closes the session rather than being refused by it.
//!
//! Around all three sits the machine A §6 asks for, and none of it is the job: the computer is
//! kept awake while a worker runs, the dock shows how far along it is, and a job that ends behind
//! another window says so. Each of the three is logged and shrugged off when the OS will not play
//! along — a run that matched for forty minutes must not be lost to a notification.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use sherd_app_core::AppError;
use sherd_app_core::eta::{self, Calibration};
use sherd_app_core::host::{self, Carried, Outcome, Worker, WorkerCommand};
use sherd_app_core::protocol::{Decision, Event, Job, Request, ReviewJob, RunSpec};
use sherd_app_core::run::{self, EngineInfo, FailKind, RunCounts, RunFile};
use sherd_app_core::view::{self, JobKind, WorkspaceView};
use sherd_app_core::workspace::{WORKSPACE_FILE, Workspace};
use sherd_core::Params;
use tauri::window::{ProgressBarState, ProgressBarStatus};
use tauri::{AppHandle, Emitter, Manager, WebviewWindow};
use tauri_plugin_notification::NotificationExt;

use crate::ENGINE_WORKER;
use crate::error::CommandError;
use crate::state::{AppState, JobSlot};

/// One line a worker said, on its way to the window.
const EVENT: &str = "engine:event";
/// A job that is over, with the workspace as it now stands.
const FINISHED: &str = "engine:finished";
/// Where a `Prepare`'s worker keeps what it writes on stderr. A `Prepare` has no run folder, so
/// its log sits beside `sherd-workspace.json` rather than in `runs/<id>/engine.log` (A §4); A
/// §10's «Показать лог» has somewhere to send the user when a preparation fails, which is why
/// [`crate::commands::run_log`] reads this same name for a `None` run.
pub(crate) const PREPARE_LOG: &str = "prepare.log";
/// The job thread's name. Named, so that a backtrace or a profiler says which thread this is.
const THREAD: &str = "sherd-job";
/// The one window (`tauri.conf.json`): whose dock progress this is, and whose focus decides
/// whether the end of a job is worth a notification.
const MAIN_WINDOW: &str = "main";
/// What every notification of ours is titled — the app, as the OS already knows it.
const NOTIFICATION_TITLE: &str = "Sherd Refit";
/// Why the machine is being kept awake, for the OS's own list of who is holding it (`pmset -g
/// assertions` on macOS).
const AWAKE_REASON: &str = "sherd-refit: a job is running";
/// Who is holding it.
const AWAKE_APP: &str = "Sherd Refit";
/// How long a `Prepare` or a run waits for an open review session to let go of the job slot
/// (A §8.4: a warm session is a cache, never a reason to refuse work).
///
/// Ten seconds is far more than a session needs: idle, it reads `Close` at once and exits; busy,
/// D §5's flag ends its request at the next unit of work. What the bound is really for is the
/// case where neither happens — a worker wedged in a driver or a disk — and there «ядро занято»
/// is the truth, not a refusal to be worked around.
const SESSION_CLOSE_WAIT: Duration = Duration::from_secs(10);
/// How often the slot is looked at while waiting for that. Short enough that the usual close is
/// not noticeable, long enough that ten seconds is five hundred cheap locks and not a spin.
const SESSION_CLOSE_POLL: Duration = Duration::from_millis(20);

/// The payload of `engine:event`.
///
/// The job is named beside the event because the window shows more than one thing at a time: a
/// `Prepare`'s thumbnails arrive into the «Вход» list while a run's history is on screen, and an
/// event with no sender is an event the frontend would have to guess about.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct EngineEvent {
    /// Which kind of job said it.
    pub(crate) job: JobKind,
    /// The run it is writing, for a [`JobKind::Run`], or the run being reviewed, for a
    /// [`JobKind::Review`]; `None` for a `Prepare`.
    pub(crate) run_id: Option<String>,
    /// The line itself, exactly as the protocol carries it (A §2.2).
    pub(crate) event: Event,
}

/// The payload of `engine:finished`.
///
/// The view travels with the outcome so that the end of a job is one message and not two: the
/// window does not have to ask for a fresh [`WorkspaceView`] and briefly show a state that is
/// neither the old one nor the new (A §5).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct EngineFinished {
    /// Which kind of job ended.
    pub(crate) job: JobKind,
    /// The run it was writing, or `None` for a `Prepare`.
    pub(crate) run_id: Option<String>,
    /// How it ended (A §10).
    pub(crate) outcome: OutcomeDto,
    /// The workspace as it stands now, or `None` when it was closed under the job or could not be
    /// read — the window then asks again rather than being handed a half-truth.
    pub(crate) view: Option<WorkspaceView>,
}

/// [`Outcome`] as the window receives it.
///
/// A mirror, and not the engine's own type, because `host::Outcome` derives no `Serialize` —
/// nothing in milestone 2 ever sent one anywhere — and a `Serialize` for a type this crate does
/// not own cannot be written here. The shape is serde's default external tagging, which is
/// exactly what `ts-rs` generated into `src/ipc/bindings/Outcome.ts`
/// (`{ "Done": { … } } | { "Failed": { … } }`), so the frontend imports that binding and this is
/// the one thing that has to keep matching it. [`From<&Outcome>`] below names every variant and
/// every field, so a change in `sherd-app-core` is a compile error here and not a wrong payload.
#[allow(
    clippy::large_enum_variant,
    reason = "the mirrored `host::Outcome` allows it for the same reason — `Done` carries the run's \
              resolved `Params`, and exactly one of these is built per job, on its way out"
)]
#[derive(Clone, Debug, Serialize)]
pub(crate) enum OutcomeDto {
    /// It finished. A `Prepare` fills none of the three blocks; a run fills all of them.
    Done {
        /// What the run found.
        counts: Option<RunCounts>,
        /// What ran it.
        engine: Option<EngineInfo>,
        /// Every threshold it resolved to.
        params: Option<Params>,
    },
    /// It did not (A §10: the window turns `kind` into a sentence and an action).
    Failed {
        /// Why, by class.
        kind: FailKind,
        /// Why, in the engine's words.
        message: String,
    },
}

impl From<&Outcome> for OutcomeDto {
    /// Field for field. Exhaustive on purpose; see the type's own note.
    fn from(outcome: &Outcome) -> Self {
        match outcome {
            Outcome::Done { counts, engine, params } => {
                Self::Done { counts: counts.clone(), engine: engine.clone(), params: *params }
            }
            Outcome::Failed { kind, message } => {
                Self::Failed { kind: *kind, message: message.clone() }
            }
        }
    }
}

/// Empties the job slot when the job thread ends, however it ends.
///
/// The normal path frees the slot itself, in [`finish`], *before* `engine:finished` goes out — the
/// order the window depends on (A §5: a `prepare_start` issued from its own `finished` handler
/// must not meet a `busy`). This guard is for the other path: a panic inside [`host::drive`] or
/// inside the closure it calls unwinds the job thread, and a slot left full there would say
/// «ядро занято» for the rest of the session, with no job to cancel and nothing but a restart to
/// clear it. Unwinding runs `Drop`, so the slot is freed either way.
///
/// It is therefore idempotent on purpose: [`disarm`](Self::disarm) marks the work already done,
/// and a second free is a no-op rather than a second `None` written over a job that has since
/// started.
#[derive(Debug)]
struct SlotGuard {
    /// Where the slot is; the guard lives on the job thread and the state does not.
    app: AppHandle,
    /// Whether the slot still has to be emptied by this guard.
    armed: bool,
}

impl SlotGuard {
    /// Arms a guard for the job about to be driven. Takes no lock: the slot it will free is
    /// filled by `start`'s caller *after* this thread is spawned, under a lock it still holds.
    fn new(app: AppHandle) -> Self {
        Self { app, armed: true }
    }

    /// Says the slot is already empty, so `Drop` leaves it alone.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SlotGuard {
    /// Nothing here can fail loudly — it may be running inside an unwind, where a panic would
    /// abort the process. A poisoned slot stays as it is; the window is already being told that
    /// something went wrong.
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        if let Some(state) = self.app.try_state::<AppState>()
            && let Ok(mut slot) = state.job()
        {
            *slot = None;
        }
    }
}

/// How to start the engine: this very executable, with the one argument that makes it the worker
/// (A §2.1). One binary and two roles means the window and the engine can never be two different
/// builds of sherd-refit, which is what the `Hello` check exists to catch when they are.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when the OS will not say what this process is running.
pub(crate) fn worker_command() -> Result<WorkerCommand, CommandError> {
    let program = std::env::current_exe().map_err(|e| {
        AppError::Worker(format!("the app cannot find its own executable to start the engine: {e}"))
    })?;
    Ok(WorkerCommand { program, args: vec![ENGINE_WORKER.to_owned()] })
}

/// The language the window is showing, for the one sentence the shell writes itself.
///
/// Everything else the user reads is a translation in the frontend's `ru.json`/`en.json`, and the
/// end-of-job notification cannot be: it is shown by the OS, after the job, possibly while the
/// window is behind something else — so the window says which language it is in when it starts
/// the job, and the shell keeps that word until the job ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lang {
    /// Russian, the app's primary wording.
    Ru,
    /// English.
    En,
}

impl Lang {
    /// The language `tag` names. Anything that is not `en` is Russian — a tag the shell does not
    /// know is a new translation of the window's, and the app's own wording is the better guess
    /// for it than English.
    pub(crate) fn of(tag: &str) -> Self {
        if tag.eq_ignore_ascii_case("en") { Self::En } else { Self::Ru }
    }
}

/// Which job to put on the worker. The three kinds differ only in what the host files around them
/// (A §2.1), which is why one function starts all of them.
#[derive(Clone, Debug)]
pub(crate) enum JobStart {
    /// Preprocess the collection into the workspace's cache (A §5).
    Prepare,
    /// One run of the pipeline, on this launch sheet (A §7.4).
    Run {
        /// What the user launched.
        spec: RunSpec,
        /// The run whose review this one starts from (A §8.5), or `None` for a run that carries
        /// nobody's decisions.
        carry_from: Option<String>,
    },
    /// A review session over a finished run (A §8): the worker loads the match once and then
    /// answers the window's questions until it is closed.
    Review {
        /// The run being reviewed.
        run_id: String,
    },
}

/// What [`start`] has in hand once the worker is on its way and before the job thread is spawned.
#[derive(Debug)]
struct Started {
    /// Which kind of job it is, for the slot and for every event the window is sent.
    job: JobKind,
    /// The worker, to be driven on the job thread.
    worker: Worker,
    /// The run this job is about: the one it writes, for a [`JobKind::Run`], or the one it
    /// reviews; `None` for a `Prepare`.
    run_id: Option<String>,
    /// That run's folder, where the host files `assembly.json` the moment the groups are
    /// reported (A §2.1) — a run's own and a review session's drafts alike.
    ///
    /// Taken here, under the workspace lock, because the job thread may not hold that lock while
    /// it drives a worker for an hour.
    dir: Option<PathBuf>,
    /// A run's `run.json` as it was opened, for the thread to close when the run is over; `None`
    /// for a `Prepare` and for a review session, which reviews a run that has already ended and
    /// must not rewrite how it ended.
    run: Option<RunFile>,
    /// Decisions A §8.5's carry-over could not bring along, to be said once (A §8.5); empty for
    /// every job that carries nothing.
    dropped: Vec<Decision>,
}

/// Starts a job on the open workspace and returns as soon as its worker is on its way: the new
/// run's id for a [`JobStart::Run`], `None` for a [`JobStart::Prepare`].
///
/// The job slot is taken first and held until the worker is in it. That is deliberate: two
/// «Подготовить» a millisecond apart would otherwise both pass the `busy` check and put two
/// workers on the same `cache/`. The workspace lock is taken and released inside that, in the
/// order [`crate::state`] documents, and **neither is held while the worker runs** — an hour of
/// matching with the workspace locked would block every command the window makes in the meantime.
/// What the thread needs from the workspace is therefore taken before it starts ([`Started::run`])
/// and the lock is taken again, briefly, in [`close_run`] and [`finish`].
///
/// The job thread is also where A §6's machine lives — [`keep_awake`], [`DockProgress`] and the
/// [`notify`] at the end — because all three begin and end with the job and none of them is the
/// job: each is logged and shrugged off when the OS will not play along. `lang` is what the
/// notification is written in; see [`Lang`].
///
/// A review session in the slot is **closed first** and waited for (A §8.4): a warm session is a
/// cache over a finished run, never a reason to refuse the work the user has just asked for. The
/// wait is bounded by [`SESSION_CLOSE_WAIT`]; a session that will not go by then is a worker that
/// is genuinely stuck, and that is `busy`.
///
/// # Errors
///
/// [`CommandError`] of kind `busy` when a job is already on the worker, `no_workspace` when none
/// is open, `worker` when the engine process or its thread cannot be started, `io`/`json` when
/// `sherd-workspace.json` or `run.json` cannot be written, and `io`/`engine` when the input folder
/// cannot be read.
pub(crate) fn start(
    app: &AppHandle,
    state: &AppState,
    kind: JobStart,
    lang: Lang,
) -> Result<Option<String>, CommandError> {
    close_session(state)?;
    let mut slot = state.job()?;
    if slot.is_some() {
        return Err(CommandError::busy());
    }
    let command = worker_command()?;
    let Started { job, mut worker, run_id, dir: run_dir, run, dropped } = {
        let mut held = state.workspace()?;
        let ws = held.as_mut().ok_or_else(CommandError::no_workspace)?;
        match kind {
            JobStart::Prepare => prepare(ws, &command)?,
            JobStart::Run { spec, carry_from } => {
                open_run(ws, &command, &spec, carry_from.as_deref())?
            }
            JobStart::Review { run_id } => open_review(ws, &command, &run_id)?,
        }
    };
    let canceller = worker.canceller();
    // Only a session is ever asked anything; see [`crate::state::JobSlot::requester`].
    let requester = (job == JobKind::Review).then(|| worker.requester());
    let mut run_file = run;

    let held_by_thread = app.clone();
    let id = run_id.clone();
    // The thread is started before the slot is filled, and cannot get ahead of it: its first act
    // after the run is to take this same lock, which is still held here.
    std::thread::Builder::new()
        .name(THREAD.to_owned())
        .spawn(move || {
            let app = held_by_thread;
            // Armed before the drive and disarmed by `finish`: whatever happens between those
            // two lines, this thread does not leave the app «busy» behind it.
            let mut guard = SlotGuard::new(app.clone());
            // A §6: a run is 17–80 minutes, and a laptop that goes to sleep in the middle of one
            // loses it. Bound to this thread, so the machine is let go of however the thread ends
            // — including an unwind, where `finish` below is never reached.
            //
            // Not for a review session: it is idle between two clicks and may be open all
            // afternoon, and holding a laptop awake through someone's lunch is not what A §6
            // asks for — the work being protected there is the forty minutes of matching that
            // would otherwise be lost.
            let _awake = (job != JobKind::Review).then(keep_awake).flatten();
            let mut dock = DockProgress::of(&app);
            let outcome = host::drive(&mut worker, |event| {
                // A §2.1: `assembly.json` is the host's file, not the worker's, and it is written
                // the moment the groups are reported rather than at the end — a run that dies
                // after assembling still leaves the window something to draw. Failing to write it
                // does not stop the run: the log says so, and the run's own files are unharmed.
                //
                // A review session's answers are filed the same way and for the same reason
                // (A §8.4): the draft the reviewer is looking at is what the window draws when
                // the run is opened again, and the session's own baseline is the file it left.
                if let Event::Assembly(assembly) = event
                    && let Some(dir) = run_dir.as_deref()
                    && let Err(error) = host::save_assembly(dir, assembly)
                {
                    tracing::error!(
                        path = %dir.display(),
                        %error,
                        "the run's assembly could not be filed; the run goes on without it"
                    );
                }
                if let Event::Progress { stage, done, total } = event
                    && dock_stage(job, stage)
                {
                    dock.show(*done, *total);
                }
                let payload = EngineEvent { job, run_id: id.clone(), event: event.clone() };
                // A window that has gone away is not a reason to stop the job; the worker is
                // ended by dropping it, not by an event that could not be delivered.
                let _ = app.emit(EVENT, payload);
            });
            dock.clear();
            // Before the slot is freed: `finish` is the line after which another job may start,
            // and `run.json` must be closed and the calibration taught before that happens.
            if let Some(file) = run_file.as_mut() {
                close_run(&app, file, &outcome);
            }
            finish(&app, &mut guard, job, id, &outcome);
            // **After** `finish`, never between the job's end and the window being told about
            // it: showing a notification takes a round trip to the OS's own service, and one
            // that is slow — or a desktop that puts a dialog up for the permission — would hold
            // the window in «Идёт сборка…» for as long as it lasts, with the run already over and
            // the slot already free. The user reading the window must see it end first; the one
            // who is not looking is being written to precisely because a moment does not matter.
            notify(&app, job, lang, &outcome);
        })
        // Nothing to undo: the worker was moved into the closure, and dropping a `Worker` kills
        // and reaps the process it holds.
        .map_err(|e| AppError::Worker(format!("the job's thread could not be started: {e}")))?;
    *slot = Some(JobSlot { kind: job, run_id: run_id.clone(), canceller, requester });
    // The slot is let go of here and not at the end of the function: what follows is an emit,
    // and no lock of this app is held across one ([`crate::state`]).
    drop(slot);
    // A §8.5: the decisions the carry-over left behind are said once, and this is the only place
    // that can say them — the worker knows nothing of a carry, and `run_start` answers an id.
    // The same line a review session sends for the same thing, so the window reads one shape.
    if !dropped.is_empty() {
        let payload = EngineEvent {
            job,
            run_id: run_id.clone(),
            event: Event::Dropped { decisions: dropped },
        };
        let _ = app.emit(EVENT, payload);
    }
    Ok(run_id)
}

/// Closes an open review session and waits for its thread to give the job slot back (A §8.4).
///
/// No session, or a job that is not one, is nothing to do — and the `busy` check the caller makes
/// next is what refuses a `Prepare` or a run that is actually running.
///
/// **No lock is held while waiting.** The thread that has to empty the slot takes that very lock
/// to do it ([`finish`]), so holding it here would be a deadlock and not a wait; the slot is
/// looked at, let go of, and looked at again.
///
/// Two ways of asking, in this order: `Close`, which an idle session reads at once and answers by
/// ending well; and then, half way through the budget, D §5's flag, because a session in the
/// middle of R §9 will not read `Close` until that is over. Whichever ends it, the session's own
/// thread files nothing and frees the slot exactly as a run's does.
///
/// # Errors
///
/// [`CommandError`] of kind `busy` when the session is still there after [`SESSION_CLOSE_WAIT`],
/// or `worker` when the job slot is poisoned.
fn close_session(state: &AppState) -> Result<(), CommandError> {
    let session = {
        let slot = state.job()?;
        slot.as_ref()
            .filter(|job| job.kind == JobKind::Review)
            .map(|job| (job.requester.clone(), job.canceller.clone()))
    };
    let Some((requester, canceller)) = session else { return Ok(()) };
    if let Some(requester) = requester.as_ref() {
        // A session already over is not an error: its thread is on its way to the slot anyway.
        let _ = requester.send(&Request::Close);
    }
    let deadline = Instant::now() + SESSION_CLOSE_WAIT;
    let mut flagged = false;
    loop {
        let freed = { state.job()?.is_none() };
        if freed {
            return Ok(());
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        if !flagged && left <= SESSION_CLOSE_WAIT / 2 {
            flagged = true;
            canceller.cancel();
        }
        std::thread::sleep(SESSION_CLOSE_POLL.min(left));
    }
    tracing::warn!("the review session did not close; the worker is not answering");
    Err(CommandError::busy())
}

/// Starts a `Prepare` (A §5: the preprocessing every run needs anyway, and the thumbnails, display
/// meshes and warnings fall out of it).
///
/// # Errors
///
/// As [`start`].
pub(crate) fn start_prepare(
    app: &AppHandle,
    state: &AppState,
    lang: Lang,
) -> Result<(), CommandError> {
    start(app, state, JobStart::Prepare, lang)?;
    Ok(())
}

/// Starts a run and answers with its id (A §7), which is the folder under `runs/` the window will
/// be asking about from here on. `carry_from` names the run whose review it continues (A §8.5).
///
/// # Errors
///
/// As [`start`].
pub(crate) fn start_run(
    app: &AppHandle,
    state: &AppState,
    spec: RunSpec,
    lang: Lang,
    carry_from: Option<String>,
) -> Result<String, CommandError> {
    // `start` answers `Some` for every `JobStart::Run` — the id comes from the `run.json` it has
    // just written — so this is unreachable rather than a case the window has to handle. It is an
    // error and not an `expect` because a panic on the command thread poisons the job slot for
    // the rest of the session, and there is nothing here worth that.
    start(app, state, JobStart::Run { spec, carry_from }, lang)?.ok_or_else(|| {
        AppError::Worker("the run was started without an id of its own".to_owned()).into()
    })
}

/// Opens a review session over a finished run (A §8), or does nothing when that very run is
/// already open: entering the mode twice must not throw away a loaded match and load it again.
///
/// A session over *another* run, or a `Prepare` or a run that has ended, is closed first by
/// [`start`]; a `Prepare` or a run that is still going is `busy`, because the worker is theirs.
///
/// # Errors
///
/// As [`start`], plus `io` when the workspace has no such run and `json` when its `run.json`
/// holds a launch sheet this build cannot read.
pub(crate) fn start_review(
    app: &AppHandle,
    state: &AppState,
    run_id: String,
) -> Result<(), CommandError> {
    let open_already = {
        let slot = state.job()?;
        slot.as_ref().is_some_and(|job| {
            job.kind == JobKind::Review && job.run_id.as_deref() == Some(run_id.as_str())
        })
    };
    if open_already {
        return Ok(());
    }
    // A §6's notification is a job's ending said out loud, and a session's ending is the user
    // leaving a screen: [`notify`] returns before it ever reads this word.
    start(app, state, JobStart::Review { run_id }, Lang::Ru)?;
    Ok(())
}

/// The `Prepare` job and a worker on it.
fn prepare(ws: &Workspace, command: &WorkerCommand) -> Result<Started, CommandError> {
    // A §7.4: the sheet the user last launched decides `target_faces`, the seed and the budgets,
    // so the cache a `Prepare` leaves is the cache that run would have wanted. Anything else on
    // disk — an older app's sheet, a hand edit — falls back to the defaults rather than refusing
    // to prepare: the sheet is a convenience, not the user's work.
    let spec = ws
        .file()
        .last_spec
        .clone()
        .and_then(|sheet| serde_json::from_value::<RunSpec>(sheet).ok())
        .unwrap_or_default();
    let job = host::prepare_job(ws, &spec)?;
    // A `Prepare` has no run folder, so its log sits beside `sherd-workspace.json` (A §4).
    let worker = Worker::spawn(command, &job, Some(&ws.root().join(PREPARE_LOG)))?;
    Ok(Started {
        job: JobKind::Prepare,
        worker,
        run_id: None,
        dir: None,
        run: None,
        dropped: Vec::new(),
    })
}

/// Opens a run: the sheet remembered for the next `Prepare` (A §7.4), the last review carried
/// over when the user asked for it (A §8.5), `run.json` written as `running` before anything can
/// go wrong (A §4), and a worker on the job.
fn open_run(
    ws: &mut Workspace,
    command: &WorkerCommand,
    spec: &RunSpec,
    carry_from: Option<&str>,
) -> Result<Started, CommandError> {
    // Serialising a `RunSpec` cannot fail in practice — it is fifteen numbers and three enums —
    // but the sheet is on its way into a file, so its failure is that file's, not a panic.
    let sheet = serde_json::to_value(spec)
        .map_err(|source| AppError::Json { path: ws.root().join(WORKSPACE_FILE), source })?;
    ws.set_last_spec(sheet)?;
    let carried = match carry_from {
        Some(from) => Some(carry(ws, from)?),
        None => None,
    };
    let dropped = carried.as_ref().map_or_else(Vec::new, |c| c.dropped.clone());
    let (file, worker) = host::start_run(ws, command, spec, carried, chrono::Local::now())?;
    let dir = ws.run_dir(&file.id);
    Ok(Started {
        job: JobKind::Run,
        worker,
        run_id: Some(file.id.clone()),
        dir: Some(dir),
        run: Some(file),
        dropped,
    })
}

/// A §8.5's «перенести решения»: the review of `from_run` as the run about to start.
///
/// The names handed over are the collection **after** A §5.1's exclusions, because those are the
/// fragments the new run will actually match and `constraints::resolve` fails a whole run on a
/// name it does not know. A decision about an excluded fragment is therefore left behind — and
/// said, through [`Carried::dropped`].
fn carry(ws: &Workspace, from_run: &str) -> Result<Carried, CommandError> {
    if !run::list(&ws.runs_dir())?.iter().any(|run| run.id == from_run) {
        // As `commands::run_dir_of`: a run id becomes a path, so it is checked against the runs
        // that exist and never against a list of forbidden characters.
        let why = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("the workspace has no run named {from_run:?} to carry decisions from"),
        );
        return Err(AppError::io(ws.runs_dir(), why).into());
    }
    let input = ws
        .input()
        .ok_or_else(|| AppError::Worker("the input folder is not available".to_owned()))?;
    let names: Vec<String> =
        sherd_core::collection::discover_excluding(&input, &ws.file().excluded)
            .map_err(AppError::from)?
            .into_iter()
            .map(|entry| entry.name)
            .collect();
    Ok(host::carry(ws, from_run, &names, &run::timestamp(chrono::Local::now()))?)
}

/// Opens a review session over a finished run (A §8): the fragments and `match.state` loaded
/// once, and then the window's questions answered until it closes.
///
/// The run's **own** four preprocessing numbers go into the job, out of its `run.json`. A session
/// preprocesses through `cache/` exactly as a run does, and a working mesh built at another face
/// budget or another seed is not the one the match was made on — the session would refuse at
/// `Ready` with A §10's `protocol`, which is a confusing way to say «the shell sent the wrong
/// sheet».
fn open_review(
    ws: &Workspace,
    command: &WorkerCommand,
    run_id: &str,
) -> Result<Started, CommandError> {
    let Some(reviewed) = run::list(&ws.runs_dir())?.into_iter().find(|run| run.id == run_id) else {
        let why = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("the workspace has no run named {run_id:?}"),
        );
        return Err(AppError::io(ws.runs_dir(), why).into());
    };
    let dir = ws.run_dir(run_id);
    let spec: RunSpec = serde_json::from_value(reviewed.spec)
        .map_err(|source| AppError::Json { path: dir.join(run::RUN_FILE), source })?;
    let input = ws
        .input()
        .ok_or_else(|| AppError::Worker("the input folder is not available".to_owned()))?;
    let job = Job::Review(ReviewJob {
        workspace: ws.root().to_owned(),
        input,
        run_id: run_id.to_owned(),
        excluded: ws.file().excluded.clone(),
        target_faces: spec.target_faces,
        seed: spec.seed,
        memory_gb: spec.memory_gb,
        workers: spec.workers,
    });
    // Into the reviewed run's own `engine.log`: a session writes nothing else into that folder
    // (A §2.1), and A §10's «Показать лог» for this run is where the reviewer will look when a
    // session fails to open.
    let worker = Worker::spawn(command, &job, Some(&dir.join(host::ENGINE_LOG)))?;
    Ok(Started {
        job: JobKind::Review,
        worker,
        run_id: Some(run_id.to_owned()),
        dir: Some(dir),
        run: None,
        dropped: Vec::new(),
    })
}

/// Asks the running job to stop (A §2.2's «Отменить»); no job is no error, because the button and
/// the job ending on its own race and the user meant the same thing either way.
///
/// The canceller is cloned under the lock and used outside it: [`host::Canceller::cancel`] writes
/// to the worker's pipe, and the slot must not be held while a write blocks.
///
/// # Errors
///
/// [`CommandError`] of kind `worker` when the job slot is poisoned.
pub(crate) fn cancel(state: &AppState) -> Result<(), CommandError> {
    let canceller = state.job()?.as_ref().map(|slot| slot.canceller.clone());
    if let Some(canceller) = canceller {
        canceller.cancel();
    }
    Ok(())
}

/// Closes a run once its worker is gone: `run.json` says how it ended and what it found (A §4),
/// and what it measured is taught to this machine's calibration (A §6).
///
/// Runs on the job thread, where there is nobody to return an error to, so everything here is
/// logged and shrugged off: a `run.json` that could not be rewritten still describes a run whose
/// files are on disk, and the next open marks it `interrupted` rather than losing it.
///
/// The workspace is locked again here and let go of before [`finish`] takes the job slot — the
/// order [`crate::state`] requires is job first then workspace, and this holds only the second.
/// It cannot have been closed under us (`workspace_close` and `open_with` both refuse while the
/// slot is full), which is a reason to expect a workspace and not a reason to `unwrap` one.
fn close_run(app: &AppHandle, run: &mut RunFile, outcome: &Outcome) {
    if let Some(state) = app.try_state::<AppState>()
        && let Ok(held) = state.workspace()
        && let Some(ws) = held.as_ref()
    {
        if let Err(error) = host::finish_run(ws, run, outcome, chrono::Local::now()) {
            tracing::error!(run = %run.id, %error, "the run's run.json could not be finished");
        }
    } else {
        tracing::error!(
            run = %run.id,
            "the workspace is no longer held by this process; run.json stays «running» until the \
             folder is opened again"
        );
    }
    // Only a run that finished has anything to teach: a cancelled or failed one measured a part
    // of a matching, and A §6's average is of whole runs.
    if let Outcome::Done { counts: Some(counts), .. } = outcome {
        learn(app, counts);
    }
}

/// Takes what a finished run measured into this machine's calibration (A §6).
///
/// The launch sheet's «≈ 11 мин» is this file and nothing else — the same collection on a laptop
/// and on a workstation are an hour apart, and only what *this* machine did last time can say
/// which one it is. It lives in the app's config folder and not in the workspace, because it is a
/// property of the computer and follows no folder anywhere.
///
/// Derived data on the side (A §2.1: host-side, written atomically): a config folder that cannot
/// be found or written costs the next estimate its accuracy and nothing else, so it is logged
/// rather than turned into a run that failed after it had succeeded.
fn learn(app: &AppHandle, counts: &RunCounts) {
    let path = match app.path().app_config_dir() {
        Ok(dir) => dir.join(eta::CALIBRATION_FILE),
        Err(error) => {
            tracing::error!(%error, "this machine has no config folder; nothing is calibrated");
            return;
        }
    };
    let mut calibration = Calibration::load(&path);
    calibration.learn(counts);
    if let Err(error) = calibration.save(&path) {
        tracing::error!(
            path = %path.display(),
            %error,
            "the calibration could not be written; the next estimate is the one before it"
        );
    }
}

/// Holds the machine awake for as long as the returned value lives (A §6).
///
/// `idle(true)` and nothing else: a run must survive the idle sleep of a laptop left alone for
/// forty minutes, but the display may go dark — keeping a screen lit through an hour of matching
/// is a battery the user did not offer. A machine whose OS refuses the assertion is warned about
/// and goes on: the job is the job, and most of them finish before any sleep timer.
fn keep_awake() -> Option<keepawake::KeepAwake> {
    match keepawake::Builder::default().idle(true).reason(AWAKE_REASON).app_name(AWAKE_APP).create()
    {
        Ok(awake) => Some(awake),
        Err(error) => {
            tracing::warn!(
                %error,
                "the machine could not be kept awake; a long job may be lost to a sleep"
            );
            None
        }
    }
}

/// The dock's (or taskbar's) progress bar, for the stages of a job worth showing there (A §6).
///
/// Not every stage feeds it, because the bar has no second row and no name: see [`dock_stage`]
/// for which ones do and why.
#[derive(Debug)]
struct DockProgress {
    /// The window the bar belongs to, or `None` when there is no window to put one on — the app
    /// is closing, or this build was started without one.
    window: Option<WebviewWindow>,
    /// Whether it is worth trying again. A platform with no progress bar to show (a Linux desktop
    /// without `libunity`) refuses every call, and a job reports progress a hundred times: warn
    /// once, then leave the OS alone.
    working: bool,
}

impl DockProgress {
    /// The bar of the app's one window.
    fn of(app: &AppHandle) -> Self {
        let window = app.get_webview_window(MAIN_WINDOW);
        Self { working: window.is_some(), window }
    }

    /// Shows `done` of `total` as a percentage. A stage with no length yet says nothing rather
    /// than dividing by zero.
    fn show(&mut self, done: usize, total: usize) {
        if total == 0 {
            return;
        }
        // Rounded down, and clamped at both ends: `done` past `total` — a stage that recounted
        // its work — is a full bar and not an overflow, and `saturating_mul` keeps a debug build
        // from panicking on a count no collection of scans can reach.
        let percent = u64::try_from(done.min(total).saturating_mul(100) / total).unwrap_or(100);
        self.set(ProgressBarState {
            status: Some(ProgressBarStatus::Normal),
            progress: Some(percent),
        });
    }

    /// Takes the bar away: the job is over, however it ended.
    fn clear(&mut self) {
        self.set(ProgressBarState { status: Some(ProgressBarStatus::None), progress: None });
    }

    /// One call to the OS, at most one complaint about it.
    fn set(&mut self, state: ProgressBarState) {
        if !self.working {
            return;
        }
        let Some(window) = self.window.as_ref() else { return };
        if let Err(error) = window.set_progress_bar(state) {
            self.working = false;
            tracing::warn!(%error, "this desktop shows no progress for a running job");
        }
    }
}

/// Whether a `Progress` event of `stage` is one the dock's bar follows (A §6).
///
/// A run's is `matching` alone, which is all but the whole of its wall clock: `tiers`, `refine`
/// and the rest report progress too, and letting them through would run the bar to the end four
/// times over and read as four jobs. A preparation's two are `preprocess` and `display`, which
/// are the two halves of what it does and between them all of it.
///
/// A session's two are the only waits it has: `preprocess` while it loads the collection, and
/// `refine` while R §9 walks the groups «Уточнить позы» asked about. Its `Reassemble` answers
/// report nothing and are over in a tenth of a second (A §3), which is why there is no third.
fn dock_stage(job: JobKind, stage: &str) -> bool {
    match job {
        JobKind::Run => stage == "matching",
        JobKind::Prepare => stage == "preprocess" || stage == "display",
        JobKind::Review => stage == "preprocess" || stage == "refine",
    }
}

/// Tells the user their job is over, when they are not already looking at it (A §6).
///
/// Only when the window is definitely not in front: someone watching the stage strip go by does
/// not need the OS to tell them what they are reading, and a window this process can no longer
/// ask about is a window that is going away. Nothing here can fail a job that has already run:
/// an OS that refuses to show notifications is logged and that is the end of it.
fn notify(app: &AppHandle, job: JobKind, lang: Lang, outcome: &Outcome) {
    // Never for a review session (A §8): nobody is waiting for it to be over — it ends because
    // the reviewer left the screen, or because the run they started needed the worker — and an
    // OS notification saying so would be the app announcing its own housekeeping.
    if job == JobKind::Review {
        return;
    }
    let Some(window) = app.get_webview_window(MAIN_WINDOW) else { return };
    // `unwrap_or(true)`: when it cannot be told, say nothing. A stray notification is worse than
    // a missing one — the user is in another app, and the window is one click away regardless.
    if window.is_focused().unwrap_or(true) {
        return;
    }
    let body = ended(job, lang, outcome);
    if let Err(error) = app.notification().builder().title(NOTIFICATION_TITLE).body(body).show() {
        tracing::warn!(%error, "the end of the job could not be announced");
    }
}

/// The one sentence the notification carries: which job, and how it went (A §6).
///
/// A cancel is its own wording and not a failure. A §10 keeps the two apart everywhere else — the
/// history shows a cancelled run in grey and a failed one in red — and telling someone who has
/// just pressed «Отменить» that their run «не удалась» would be the app disagreeing with them.
///
/// The rest is two halves chosen apart from each other, which Russian allows here because both
/// «Сборка» and «Подготовка» are feminine and take the same «завершена / отменена / не удалась».
fn ended(job: JobKind, lang: Lang, outcome: &Outcome) -> String {
    // The one sentence that carries a number: a finished run's groups are what the reviewer
    // wants from the other side of the room, and the rest they will read on the screen.
    if let (JobKind::Run, Outcome::Done { counts: Some(counts), .. }) = (job, outcome) {
        return match lang {
            Lang::Ru => format!("Сборка завершена: {}", groups_ru(counts.groups)),
            Lang::En => format!("Assembly finished: {}", groups_en(counts.groups)),
        };
    }
    let what = match (lang, job) {
        (Lang::Ru, JobKind::Run) => "Сборка",
        (Lang::Ru, JobKind::Prepare) => "Подготовка",
        (Lang::En, JobKind::Run) => "Assembly",
        (Lang::En, JobKind::Prepare) => "Preparation",
        // Never reached: [`notify`] returns before this for a session. Named all the same, so
        // that a fourth kind of job has to be thought about here rather than defaulted.
        (Lang::Ru, JobKind::Review) => "Ревью",
        (Lang::En, JobKind::Review) => "Review",
    };
    let how = match (lang, outcome) {
        (Lang::Ru, Outcome::Done { .. }) => "завершена",
        (Lang::Ru, Outcome::Failed { kind: FailKind::Cancelled, .. }) => "отменена",
        (Lang::Ru, Outcome::Failed { .. }) => "не удалась",
        (Lang::En, Outcome::Done { .. }) => "finished",
        (Lang::En, Outcome::Failed { kind: FailKind::Cancelled, .. }) => "cancelled",
        (Lang::En, Outcome::Failed { .. }) => "failed",
    };
    format!("{what} {how}")
}

/// «1 группа», «2 группы», «17 групп»: Russian's three forms, which a sentence that carries a
/// number has to get right — this one is read by someone who was not watching.
fn groups_ru(n: usize) -> String {
    let form = match (n % 10, n % 100) {
        (_, 11..=14) => "групп",
        (1, _) => "группа",
        (2..=4, _) => "группы",
        _ => "групп",
    };
    format!("{n} {form}")
}

/// «1 group», «17 groups».
fn groups_en(n: usize) -> String {
    if n == 1 { "1 group".to_owned() } else { format!("{n} groups") }
}

/// Empties the job slot and tells the window how the job ended.
///
/// The slot is emptied *before* the window is told, so that a `prepare_start` issued from the
/// window's own `finished` handler cannot find the job it has just watched end still «busy».
/// Both locks are held while the view is built — job first, then workspace, as [`crate::state`]
/// requires — so the view handed over cannot be one that misses a job begun in that same handler.
///
/// Nothing here can fail loudly: this runs on the job thread, where there is nobody to return an
/// error to. A poisoned lock or an unreadable input folder simply means no view travels with the
/// outcome, and the window asks again — and the `guard` then has the slot to empty, because the
/// one line that would have emptied it was never reached.
fn finish(
    app: &AppHandle,
    guard: &mut SlotGuard,
    job: JobKind,
    run_id: Option<String>,
    outcome: &Outcome,
) {
    let mut view = None;
    if let Some(state) = app.try_state::<AppState>()
        && let Ok(mut slot) = state.job()
    {
        *slot = None;
        guard.disarm();
        view = state.workspace().ok().and_then(|held| view::build(held.as_ref()?, None).ok());
    }
    let payload = EngineFinished { job, run_id, outcome: outcome.into(), view };
    let _ = app.emit(FINISHED, payload);
}
