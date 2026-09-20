//! What the window's process holds while it is open: the one workspace, and the one job.
//!
//! Both are behind a `Mutex` and neither is ever `unwrap`ped. A command runs on Tauri's thread
//! pool and the job thread (Task 4's `jobs.rs`) runs beside it, so these are the only two things
//! in the shell that are shared — and A §5 has exactly one of each: one workspace open, one
//! worker at a time.
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

use std::sync::{Mutex, MutexGuard};

use sherd_app_core::host::Canceller;
use sherd_app_core::view::{JobKind, JobView};
use sherd_app_core::workspace::Workspace;

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
    /// The run it is writing, for a [`JobKind::Run`]; `None` for a `Prepare`.
    pub(crate) run_id: Option<String>,
    /// How to stop it (A §2.2).
    pub(crate) canceller: Canceller,
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
}
