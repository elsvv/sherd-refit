//! A job in the shell (A §2.1): the app starts *itself* as the engine, drives that process on a
//! thread of its own, and forwards everything it says to the window.
//!
//! Nothing of the engine runs in the window's process — that is the whole point of the two roles
//! (A §2.1): a 3 GB match and a GPU device live in a child that can be killed, and the window
//! stays answerable while it works. So [`start_prepare`] returns as soon as the worker is on its
//! way, and everything after that reaches the frontend as two events: `engine:event` for each
//! line the worker wrote, and `engine:finished` once, carrying the whole [`WorkspaceView`] the
//! window should now be showing (A §5: one shape, so the window never keeps a second opinion).

use serde::Serialize;
use sherd_app_core::AppError;
use sherd_app_core::host::{self, Outcome, Worker, WorkerCommand};
use sherd_app_core::protocol::{Event, RunSpec};
use sherd_app_core::run::{EngineInfo, FailKind, RunCounts};
use sherd_app_core::view::{self, JobKind, WorkspaceView};
use sherd_core::Params;
use tauri::{AppHandle, Emitter, Manager};

use crate::ENGINE_WORKER;
use crate::error::CommandError;
use crate::state::{AppState, JobSlot};

/// One line a worker said, on its way to the window.
const EVENT: &str = "engine:event";
/// A job that is over, with the workspace as it now stands.
const FINISHED: &str = "engine:finished";
/// Where a `Prepare`'s worker keeps what it writes on stderr. A `Prepare` has no run folder, so
/// its log sits beside `sherd-workspace.json` rather than in `runs/<id>/engine.log` (A §4); A
/// §10's «Показать лог» has somewhere to send the user when a preparation fails.
const PREPARE_LOG: &str = "prepare.log";
/// The job thread's name. Named, so that a backtrace or a profiler says which thread this is.
const THREAD: &str = "sherd-job";

/// The payload of `engine:event`.
///
/// The job is named beside the event because the window shows more than one thing at a time: a
/// `Prepare`'s thumbnails arrive into the «Вход» list while a run's history is on screen, and an
/// event with no sender is an event the frontend would have to guess about.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct EngineEvent {
    /// Which kind of job said it.
    pub(crate) job: JobKind,
    /// The run it is writing, for a [`JobKind::Run`]; `None` for a `Prepare`.
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

/// Starts a `Prepare` on the open workspace (A §5: the preprocessing every run needs anyway, and
/// the thumbnails, display meshes and warnings fall out of it).
///
/// The job slot is taken first and held until the worker is in it. That is deliberate: two
/// «Подготовить» a millisecond apart would otherwise both pass the `busy` check and put two
/// workers on the same `cache/`. The workspace lock is taken and released inside that, in the
/// order [`crate::state`] documents, and neither is held while the worker runs.
///
/// # Errors
///
/// [`CommandError`] of kind `busy` when a job is already on the worker, `no_workspace` when none
/// is open, `worker` when the engine process or its thread cannot be started, and `io`/`engine`
/// when the input folder cannot be read.
pub(crate) fn start_prepare(app: &AppHandle, state: &AppState) -> Result<(), CommandError> {
    let mut slot = state.job()?;
    if slot.is_some() {
        return Err(CommandError::busy());
    }
    let (job, log) = {
        let held = state.workspace()?;
        let ws = held.as_ref().ok_or_else(CommandError::no_workspace)?;
        // A §7.4: the sheet the user last launched decides `target_faces`, the seed and the
        // budgets, so the cache a `Prepare` leaves is the cache that run would have wanted.
        // Anything else on disk — an older app's sheet, a hand edit — falls back to the defaults
        // rather than refusing to prepare: the sheet is a convenience, not the user's work.
        let spec = ws
            .file()
            .last_spec
            .clone()
            .and_then(|sheet| serde_json::from_value::<RunSpec>(sheet).ok())
            .unwrap_or_default();
        (host::prepare_job(ws, &spec)?, ws.root().join(PREPARE_LOG))
    };
    let mut worker = Worker::spawn(&worker_command()?, &job, Some(&log))?;
    let canceller = worker.canceller();

    let app = app.clone();
    // The thread is started before the slot is filled, and cannot get ahead of it: its first act
    // after the run is to take this same lock, which is still held here.
    std::thread::Builder::new()
        .name(THREAD.to_owned())
        .spawn(move || {
            let outcome = host::drive(&mut worker, |event| {
                let payload =
                    EngineEvent { job: JobKind::Prepare, run_id: None, event: event.clone() };
                // A window that has gone away is not a reason to stop preparing; the worker is
                // ended by dropping it, not by an event that could not be delivered.
                let _ = app.emit(EVENT, payload);
            });
            finish(&app, JobKind::Prepare, None, &outcome);
        })
        // Nothing to undo: the worker was moved into the closure, and dropping a `Worker` kills
        // and reaps the process it holds.
        .map_err(|e| AppError::Worker(format!("the job's thread could not be started: {e}")))?;
    *slot = Some(JobSlot { kind: JobKind::Prepare, run_id: None, canceller });
    Ok(())
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

/// Empties the job slot and tells the window how the job ended.
///
/// The slot is emptied *before* the window is told, so that a `prepare_start` issued from the
/// window's own `finished` handler cannot find the job it has just watched end still «busy».
/// Both locks are held while the view is built — job first, then workspace, as [`crate::state`]
/// requires — so the view handed over cannot be one that misses a job begun in that same handler.
///
/// Nothing here can fail loudly: this runs on the job thread, where there is nobody to return an
/// error to. A poisoned lock or an unreadable input folder simply means no view travels with the
/// outcome, and the window asks again.
fn finish(app: &AppHandle, job: JobKind, run_id: Option<String>, outcome: &Outcome) {
    let view = app.try_state::<AppState>().and_then(|state| {
        let mut slot = state.job().ok()?;
        *slot = None;
        let held = state.workspace().ok()?;
        view::build(held.as_ref()?, None).ok()
    });
    let payload = EngineFinished { job, run_id, outcome: outcome.into(), view };
    let _ = app.emit(FINISHED, payload);
}
