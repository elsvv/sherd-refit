//! What the window's process holds while it is open: the one workspace, the one job, and the one
//! answer about this build that is worth asking a process for only once.
//!
//! Each is behind a `Mutex` and none is ever `unwrap`ped. A command runs on Tauri's thread pool
//! and the job thread (`jobs.rs`) runs beside it, so these are the only things in the shell that
//! are shared — and A §5 has exactly one of the first two: one workspace open, one worker at a
//! time.
//!
//! **Locking order: the job slot first, then the workspace.** Every path that needs both takes
//! them in that order, which is what keeps a command and the job thread from waiting on each
//! other. Neither is ever held while a worker is being *driven*; the job slot alone is held
//! across the spawn of one, which is what keeps two clicks from starting two workers on the same
//! folder (see [`crate::jobs::start_prepare`]).
//!
//! A command that only *reads* the slot holds it just the same, to the end of what it is doing.
//! Sampling it and letting go leaves a gap in which a job can start, and everything such a
//! command goes on to do is then wrong about that job: a view would tell the window `job: null`
//! about a worker that is running, and closing or reopening the workspace would drop the
//! `Workspace` — and with it A §10's lock — out from under one.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use sherd_app_core::host::{Canceller, Killer, Requester};
use sherd_app_core::view::{JobKind, JobView};
use sherd_app_core::workspace::Workspace;

use crate::commands::EngineInfoView;
use crate::error::CommandError;

/// The job a worker of ours is on (A §5's «preparing» and «running» rows).
///
/// The canceller and not the [`Worker`](sherd_app_core::host::Worker) itself: `host::drive`
/// borrows the worker for the whole run on the job thread, so «Отменить» — which arrives on a
/// command thread — can only reach it through the handle taken before the run began (A §2.2).
#[derive(Debug)]
pub(crate) struct JobSlot {
    /// Which kind of job, for the view and for the events the window is sent.
    pub(crate) kind: JobKind,
    /// The run it is writing, for a [`JobKind::Run`], or the run it is reviewing, for a
    /// [`JobKind::Review`]; `None` for a `Prepare`.
    pub(crate) run_id: Option<String>,
    /// How to stop it (A §2.2).
    pub(crate) canceller: Canceller,
    /// How to ask it something, for a [`JobKind::Review`] (A §8): a session sits in the slot
    /// answering `Reassemble`, `PairDetail` and `Refine` until it is closed.
    ///
    /// `None` for the two jobs that answer no questions, so that a `review_apply` arriving while
    /// a run is on is refused here rather than writing a line into a worker that ignores it —
    /// and `None` again once a session has been asked to close
    /// ([`crate::commands::review_close`] takes it out): a session on its way out answers no more
    /// questions either, and `Some` is therefore exactly «there is somebody here to ask».
    pub(crate) requester: Option<Requester>,
    /// How to end it outright (A §10's hard kill), for a [`JobKind::Review`].
    ///
    /// A session is a cache over a finished run and never a reason to refuse work (A §8.4), so
    /// the «Подготовить» or the run that finds one in the slot must be able to have the slot
    /// whatever state the session's worker is in — asking it to close is the first move, and this
    /// is what happens when asking gets no answer ([`crate::jobs`]'s `close_session`).
    ///
    /// `None` for a `Prepare` and a run: those are the user's own work, and what stops them is
    /// «Отменить» (A §2.2), never another command taking the worker out from under them.
    pub(crate) killer: Option<Killer>,
}

impl JobSlot {
    /// What the window is told about this job (A §5's «preparing» and «running» rows).
    ///
    /// A method on the slot rather than on [`AppState`], so that the caller keeps the guard it
    /// read this from: a view is only true of the moment the job slot says it is, and the
    /// commands that build one hold that lock while they do.
    pub(crate) fn view(&self) -> JobView {
        JobView { kind: self.kind, run_id: self.run_id.clone() }
    }
}

/// Everything the shell keeps between commands.
#[derive(Debug, Default)]
pub(crate) struct AppState {
    /// The open workspace. Holding it holds its lock on disk (A §10: a second app instance on the
    /// same folder is refused); dropping it releases the lock.
    workspace: Mutex<Option<Workspace>>,
    /// The running job, if one is. A fact about this process, not about the folder, which is why
    /// it is here and not in the view built from disk.
    job: Mutex<Option<JobSlot>>,
    /// What the engine says it can run on (A §7.4's «Вычисления»), once it has been asked.
    ///
    /// Kept because the answer costs a process to obtain and cannot change while the app is
    /// open: it is a property of this build and of the machine's adapters, and the launch sheet
    /// asks for it every time it opens.
    engine_info: Mutex<Option<EngineInfoView>>,
    /// Where the last export of this session was written (A §9.1), or `None` before the first.
    ///
    /// The one folder outside the open workspace that [`crate::commands::reveal`] will show.
    /// A §2.1 gives the window nothing outside the workspace and lets it ask for nothing outside
    /// it — and an export is the one thing the user deliberately puts elsewhere, so the shell
    /// remembers that one place rather than opening «Показать в папке» to any path the window
    /// names.
    last_export: Mutex<Option<PathBuf>>,
}

impl AppState {
    /// The open workspace, locked.
    ///
    /// # Errors
    ///
    /// [`CommandError::poisoned`] when a thread panicked holding it. The window is told rather
    /// than the process brought down: the user's work is on disk, and a message they can copy is
    /// worth more than a crash report they cannot.
    pub(crate) fn workspace(&self) -> Result<MutexGuard<'_, Option<Workspace>>, CommandError> {
        self.workspace.lock().map_err(|_| CommandError::poisoned("workspace"))
    }

    /// The job slot, locked. Take this before [`workspace`](Self::workspace); see the module's
    /// note on locking order.
    ///
    /// # Errors
    ///
    /// [`CommandError::poisoned`].
    pub(crate) fn job(&self) -> Result<MutexGuard<'_, Option<JobSlot>>, CommandError> {
        self.job.lock().map_err(|_| CommandError::poisoned("job"))
    }

    /// What the engine can run on, locked. Unrelated to the two above — it is about the build and
    /// not about the folder — so it takes part in no locking order and is never held with them.
    ///
    /// # Errors
    ///
    /// [`CommandError::poisoned`].
    pub(crate) fn engine_info(
        &self,
    ) -> Result<MutexGuard<'_, Option<EngineInfoView>>, CommandError> {
        self.engine_info.lock().map_err(|_| CommandError::poisoned("engine info"))
    }

    /// The last export's folder, locked. Unrelated to the two above, as
    /// [`engine_info`](Self::engine_info) is: it is about what this session has written, not
    /// about the folder that is open, and it takes part in no locking order.
    ///
    /// # Errors
    ///
    /// [`CommandError::poisoned`].
    pub(crate) fn last_export(&self) -> Result<MutexGuard<'_, Option<PathBuf>>, CommandError> {
        self.last_export.lock().map_err(|_| CommandError::poisoned("last export"))
    }
}
