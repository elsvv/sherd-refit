//! The window's side of A §2.1: a worker process, its events, its log, and the run's bookkeeping.
//!
//! The host never loads a mesh. It starts a process, reads one JSON line at a time from its
//! stdout, keeps its stderr in `engine.log`, and writes the two files the worker is not allowed to
//! touch — `run.json` and `assembly.json` — so that a worker that dies mid-run still leaves a run
//! the history can explain.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use sherd_core::Params;
use sherd_core::assembly::constraints::Constraints;

use crate::decisions::{Decision, DecisionsFile, to_constraints};
use crate::protocol::{
    AssemblyDto, EngineInfo, Event, FailKind, Job, PROTOCOL, PrepareJob, Request, RunCounts,
    RunJob, RunSpec,
};
use crate::run::{RunFile, RunStatus};
use crate::workspace::Workspace;
use crate::{AppError, Result, atomic, run, snapshot};

/// The run's assembly, in its folder (A §4). The host's, not the worker's: it is replaced on
/// every decision of a review and must never be half-written.
pub const ASSEMBLY_FILE: &str = "assembly.json";
/// Everything the worker said on stderr, in the run's folder (A §4) — where A §10's «Показать
/// лог» sends the reviewer when a run ends badly.
pub const ENGINE_LOG: &str = "engine.log";

/// `engine.log` while a worker is running: one handle behind a lock, because both reader threads
/// write to it and two appenders can interleave half a line. `None` when there is no log to keep.
type Log = Option<Arc<Mutex<std::fs::File>>>;

/// How to start a worker (A §2.1). In the app it is `std::env::current_exe()` with
/// `--engine-worker`; in the headless tests it is the `sherd-engine-worker` binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerCommand {
    /// The program to run.
    pub program: PathBuf,
    /// Its arguments.
    pub args: Vec<String>,
}

/// What a worker's stream yields: its events, and then the one fact the protocol cannot carry —
/// that the process itself is over.
#[allow(
    clippy::large_enum_variant,
    reason = "it is `protocol::Event` plus one word; the size is that enum's, and its own reason"
)]
#[derive(Clone, Debug, PartialEq)]
pub enum HostEvent {
    /// One line the worker wrote.
    Event(Event),
    /// The process has ended. Yielded once, after the last event.
    Exited {
        /// Its exit code — 0 done, 1 failed, 2 cancelled — or `None` where a signal ended it.
        code: Option<i32>,
    },
}

/// The pipe a job is cancelled through, shared between the worker and every [`Canceller`] taken
/// from it. `None` once the process is over and the pipe has been let go of.
type Pipe = Arc<Mutex<Option<ChildStdin>>>;

/// A running worker: the process, the pipe the window talks back through, and the threads that
/// turn its two output streams into events and into `engine.log`.
///
/// Dropping a `Worker` drops its stdin, which is the end of the worker's input and therefore
/// D §5's cancel (A §2.1): a window that goes away leaves no 3 GB orphan behind it. An outstanding
/// [`Canceller`] does not keep that pipe open — see [`Worker::canceller`].
#[derive(Debug)]
pub struct Worker {
    child: Child,
    // Kept for the worker's whole life: closing it cancels the job, so it is let go of only once
    // the process is over. Behind a lock because a `Canceller` writes to it from the window's
    // thread while `drive` is reading events on another.
    stdin: Pipe,
    events: Receiver<Event>,
    readers: Vec<JoinHandle<()>>,
    exited: bool,
}

/// The one thing a window may do to a job it has already handed to [`drive`]: stop it (A §2.2).
///
/// [`drive`] borrows the [`Worker`] for the whole run, so «Отменить» cannot be a method on the
/// worker — by the time there is a run to cancel, the button's thread cannot reach it. A
/// `Canceller` is taken before the run starts, is cheap to clone and `Send`, and writes the same
/// line [`Worker::cancel`] does.
#[derive(Clone, Debug)]
pub struct Canceller {
    stdin: Pipe,
}

impl Canceller {
    /// Asks the job to stop (A §2.2). The worker raises D §5's flag, ends at its next unit of
    /// work and says `Failed` with [`FailKind::Cancelled`]; a worker already gone is not an error
    /// — its exit is what [`Worker::next`] will report anyway.
    pub fn cancel(&self) {
        let _ = write_request(&self.stdin, &Request::Cancel);
    }
}

/// Everything else a window may say to a job that is already running (A §2.2): a review session's
/// `Reassemble`, `PairDetail`, `Refine` and `Close`.
///
/// The same pipe and the same reason as [`Canceller`] — [`drive`] owns the worker for the whole
/// session, so the thread that has a question cannot reach it — with one difference: a cancel
/// that arrives too late has simply happened anyway, while a question the worker never heard has
/// an answer the window is still waiting for. This one therefore says whether the line went out.
#[derive(Clone, Debug)]
pub struct Requester {
    stdin: Pipe,
}

impl Requester {
    /// Sends one request.
    ///
    /// # Errors
    ///
    /// [`AppError::Worker`] when the worker is over, its pipe is gone, or the line cannot be
    /// written — the three ways a session stops being able to answer.
    pub fn send(&self, request: &Request) -> Result<()> {
        write_request(&self.stdin, request)
    }
}

/// One request down the worker's stdin, through serde, so that the line the host writes and the
/// one the worker parses cannot drift apart.
fn write_request(pipe: &Pipe, request: &Request) -> Result<()> {
    let gone = || AppError::Worker("the engine process is not listening".to_owned());
    let mut pipe = pipe.lock().map_err(|_| gone())?;
    let stdin = pipe.as_mut().ok_or_else(gone)?;
    let line = serde_json::to_string(request)
        .map_err(|e| AppError::Worker(format!("the request does not serialise: {e}")))?;
    writeln!(stdin, "{line}")
        .and_then(|()| stdin.flush())
        .map_err(|e| AppError::Worker(format!("the request could not be sent: {e}")))
}

impl Worker {
    /// Starts a worker on `job`, appending its stderr to `log` when a run wants one kept.
    ///
    /// The job is written before this returns, so a caller may send [`Worker::cancel`] straight
    /// afterwards and know the worker will find it — the two lines are in the pipe before it has
    /// read either.
    ///
    /// # Errors
    ///
    /// [`AppError::Worker`] when the process cannot be started or the job cannot be sent,
    /// [`AppError::Io`] when `log` cannot be opened.
    pub fn spawn(command: &WorkerCommand, job: &Job, log: Option<&Path>) -> Result<Self> {
        let log = open_log(log)?;
        let mut builder = Command::new(&command.program);
        builder
            .args(&command.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // A §10 promises an `internal` failure's backtrace in `engine.log`. The default panic
            // hook writes one to stderr — which is the log — only when asked to.
            .env("RUST_BACKTRACE", "1");
        // CREATE_NO_WINDOW: a worker started from a windowed app must not flash a console.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            builder.creation_flags(0x0800_0000);
        }
        let mut child = builder
            .spawn()
            .map_err(|e| AppError::Worker(format!("{}: {e}", command.program.display())))?;
        // A child that cannot be handed its job is ended and reaped here: returning with it
        // alive would leave a process nobody holds and, once it died, a zombie nobody waits for.
        let abandon = |mut child: std::process::Child, why: String| {
            let _ = child.kill();
            let _ = child.wait();
            AppError::Worker(why)
        };
        let (Some(mut stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            return Err(abandon(child, "the engine process has no pipes".to_owned()));
        };
        let line = match serde_json::to_string(job) {
            Ok(line) => line,
            Err(e) => return Err(abandon(child, format!("the job does not serialise: {e}"))),
        };
        if let Err(e) = writeln!(stdin, "{line}").and_then(|()| stdin.flush()) {
            return Err(abandon(child, format!("the job could not be sent: {e}")));
        }

        let (sender, events) = mpsc::channel();
        let stray = log.clone();
        let reading = std::thread::spawn(move || {
            for line in BufReader::new(stdout).split(b'\n') {
                let Ok(line) = line else { break };
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_slice::<Event>(&line) {
                    Ok(event) => {
                        if sender.send(event).is_err() {
                            break;
                        }
                    }
                    // A line that is not an event belongs to whoever printed it, not to the
                    // protocol: keep it in the log rather than fail a run over a stray word.
                    Err(_) => append(stray.as_deref(), &line),
                }
            }
        });
        let copying = std::thread::spawn(move || {
            // Read stderr even with no log to write it to: a full pipe stops the engine.
            for line in BufReader::new(stderr).split(b'\n') {
                let Ok(line) = line else { break };
                append(log.as_deref(), &line);
            }
        });
        Ok(Self {
            child,
            stdin: Arc::new(Mutex::new(Some(stdin))),
            events,
            readers: vec![reading, copying],
            exited: false,
        })
    }

    /// A handle on this worker's cancel, for the thread that will not be holding the worker.
    ///
    /// Take it before handing the worker to [`drive`]: that call owns the worker until the run is
    /// over, so this is the only way «Отменить» reaches a job that is actually running (A §2.2).
    /// Cancelling through a handle after the worker is gone does nothing, and a handle kept past
    /// the worker's life does not hold the pipe open — dropping the worker closes it either way.
    pub fn canceller(&self) -> Canceller {
        Canceller { stdin: Arc::clone(&self.stdin) }
    }

    /// A handle that asks this worker questions, for the thread that will not be holding it.
    ///
    /// Take it before handing the worker to [`drive`], for the same reason [`Worker::canceller`]
    /// is taken then: a review session is driven on one thread and clicked on another (A §8).
    pub fn requester(&self) -> Requester {
        Requester { stdin: Arc::clone(&self.stdin) }
    }

    /// Asks the job to stop (A §2.2), for a caller that still holds the worker — before
    /// [`drive`], or in a test. Same line as [`Canceller::cancel`].
    pub fn cancel(&mut self) {
        self.canceller().cancel();
    }

    /// Ends the process now — A §10's hard kill, behind the cooperative cancel. Waits for it, so
    /// that nothing is still writing into the run's folder when this returns.
    pub fn kill(&mut self) {
        self.close_stdin();
        let _ = self.child.kill();
        let _ = self.child.wait();
        // The pipes are closed now, so both readers reach their end; joining them is what makes
        // "nothing is still writing into the run's folder" true of `engine.log` as well.
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
        self.exited = true;
    }

    /// Lets go of the worker's input, which is D §5's cancel by EOF (A §2.1). Says nothing when
    /// the lock is poisoned: the pipe then goes with the last handle to it instead.
    fn close_stdin(&self) {
        if let Ok(mut pipe) = self.stdin.lock() {
            *pipe = None;
        }
    }

    /// The next thing that happened: an event, then [`HostEvent::Exited`] once, then `None`.
    /// Blocks until there is one.
    #[allow(
        clippy::should_implement_trait,
        reason = "a worker is a process to drive, not a sequence to map and collect"
    )]
    pub fn next(&mut self) -> Option<HostEvent> {
        if let Ok(event) = self.events.recv() {
            return Some(HostEvent::Event(event));
        }
        if self.exited {
            return None;
        }
        self.exited = true;
        // Its stdout is closed, so the process is over or a breath away from it: let go of stdin,
        // reap it, and collect the readers, so that `engine.log` is whole before anyone reads it.
        self.close_stdin();
        let code = self.child.wait().ok().and_then(|status| status.code());
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
        Some(HostEvent::Exited { code })
    }
}

/// A window that goes away leaves no 3 GB orphan (A §2.1): the pipe is closed here rather than
/// left to the last [`Canceller`] a window may still be holding, so that the end of the worker's
/// input stays the end of the worker.
impl Drop for Worker {
    /// A worker let go of before it said how it ended is stopped and reaped: the end of its input
    /// is only a *request* to stop, and a process nobody waits for stays a zombie for as long as
    /// the app runs.
    fn drop(&mut self) {
        if !self.exited {
            self.kill();
        }
    }
}

/// How a job ended, as the host files it (A §10).
#[allow(
    clippy::large_enum_variant,
    reason = "`Done` carries the run's resolved `Params`, and one `Outcome` is returned per run"
)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// It finished. The three blocks are a run's; a `Prepare` and an `Info` fill none of them.
    Done {
        /// What the run found.
        counts: Option<RunCounts>,
        /// What ran it.
        engine: Option<EngineInfo>,
        /// Every threshold it resolved to.
        #[cfg_attr(feature = "ts", ts(type = "Record<string, unknown> | null"))]
        params: Option<Params>,
    },
    /// It did not.
    Failed {
        /// Why, by class.
        kind: FailKind,
        /// Why, in the engine's words.
        message: String,
    },
}

/// Runs a worker to its end, handing every event to `on_event` on the way.
///
/// Events are consumed until the **process** has exited and not merely until it has said `Done`:
/// a run is over when its ~3 GB are back with the OS (A §2.1), and a worker that dies after its
/// last word is reaped here rather than left for the window's next question.
pub fn drive(worker: &mut Worker, mut on_event: impl FnMut(&Event)) -> Outcome {
    let (mut outcome, mut code) = (None, None);
    while let Some(next) = worker.next() {
        let event = match next {
            HostEvent::Exited { code: status } => {
                code = status;
                continue;
            }
            HostEvent::Event(event) => event,
        };
        match &event {
            // A worker left over from a half-finished update (A §2.2). Nothing it says after
            // this can be trusted to mean what this build reads, so it is not read.
            Event::Hello { protocol, .. } if *protocol != PROTOCOL => {
                worker.kill();
                let message =
                    format!("the engine speaks protocol {protocol}, this one speaks {PROTOCOL}");
                return Outcome::Failed { kind: FailKind::Protocol, message };
            }
            Event::Done { counts, engine, params } => {
                outcome = Some(Outcome::Done {
                    counts: counts.clone(),
                    engine: engine.clone(),
                    params: *params,
                });
            }
            Event::Failed { kind, message } => {
                outcome = Some(Outcome::Failed { kind: *kind, message: message.clone() });
            }
            _ => {}
        }
        on_event(&event);
    }
    // A process that ended without saying how: an OOM kill, a stack overflow, a driver taking the
    // process with it. A §10 has a class for exactly that, and the log has the rest.
    outcome.unwrap_or_else(|| Outcome::Failed {
        kind: FailKind::Crashed,
        message: match code {
            Some(code) => format!("exit code {code}"),
            None => "the engine process was killed".to_owned(),
        },
    })
}

/// The `Prepare` job for this workspace and this launch sheet (A §5): the folder, what is left
/// out of it, and the four numbers preprocessing shares with a run.
///
/// # Errors
///
/// [`AppError::Worker`] when the input folder is linked but cannot be found (A §5, «input
/// missing»).
pub fn prepare_job(ws: &Workspace, spec: &RunSpec) -> Result<Job> {
    Ok(Job::Prepare(PrepareJob {
        workspace: ws.root().to_owned(),
        input: input_of(ws)?,
        excluded: ws.file().excluded.clone(),
        target_faces: spec.target_faces,
        seed: spec.seed,
        memory_gb: spec.memory_gb,
        workers: spec.workers,
    }))
}

/// A §8.5's «перенести решения»: one run's review, ready to be the next run's starting point.
///
/// The three things a carry-over is, kept together so that the shell cannot send the engine one
/// set of decisions and file another: what the engine is told, what the new run's
/// `decisions.json` holds, and what could not come along and has to be said.
#[derive(Clone, Debug, PartialEq)]
pub struct Carried {
    /// The run the decisions come from — `run.json`'s `carried_from` of the run about to start.
    pub from: String,
    /// The decisions as `constraints.json` (A §8.2): accepted pairs pinned at their pose and not
    /// matched again, rejected pairs skipped before matching. `None` when nothing survived.
    pub constraints: Option<Constraints>,
    /// The new run's `decisions.json`, each decision marked [`Decision::carried_from`] — the
    /// review of the new run starts where the old one left off rather than from nothing.
    pub decisions: DecisionsFile,
    /// The decisions that name a fragment the collection no longer holds. Not carried — they
    /// would fail `constraints::resolve` and with it the whole run — and reported so that the
    /// window can tell the reviewer once (A §8.5).
    pub dropped: Vec<Decision>,
}

/// The review of `from_run` as the next run's starting point (A §8.5), against the collection
/// `names` the new run will actually have.
///
/// A run with no `decisions.json` carries nothing and is not an error: «перенести решения» is a
/// checkbox, and a run nobody reviewed is simply a run with nothing to carry.
///
/// # Errors
///
/// [`AppError::Io`], [`AppError::Json`] or [`AppError::Version`] when the old run's
/// `decisions.json` cannot be read as one this build wrote.
pub fn carry(ws: &Workspace, from_run: &str, names: &[String], now: &str) -> Result<Carried> {
    let file = DecisionsFile::load_or_default(&ws.run_dir(from_run))?;
    let (constraints, dropped) = to_constraints(&file, names)?;
    Ok(Carried {
        from: from_run.to_owned(),
        constraints,
        decisions: file.carried(names, from_run, now),
        dropped,
    })
}

/// Opens a run (A §4): the input as it stands now, the next free id, `run.json` in `running`, the
/// review it continues if it continues one (A §8.5), and a worker on the job.
///
/// `run.json` is written *before* the worker is spawned, and that order is the point: whichever
/// of the two processes dies, the folder on disk is already a run, and `run::mark_interrupted`
/// finds it at the next start instead of leaving a folder nobody can account for. `decisions.json`
/// is written in the same breath, for the same reason: a carried review the window never filed
/// would be a run whose constraints nobody could account for either.
///
/// # Errors
///
/// [`AppError::Worker`] when the input is missing or the worker cannot be started,
/// [`AppError::Core`] when the input folder cannot be listed, [`AppError::Io`].
pub fn start_run(
    ws: &Workspace,
    command: &WorkerCommand,
    spec: &RunSpec,
    carried: Option<Carried>,
    now: chrono::DateTime<chrono::Local>,
) -> Result<(RunFile, Worker)> {
    let input = input_of(ws)?;
    let snapshot = snapshot::scan(&input, &ws.file().excluded)?;
    let existing: Vec<String> = run::list(&ws.runs_dir())?.into_iter().map(|r| r.id).collect();
    let id = run::new_id(&existing, now);
    let dir = ws.run_dir(&id);
    let sheet = serde_json::to_value(spec)
        .map_err(|source| AppError::Json { path: dir.join(run::RUN_FILE), source })?;
    let mut file = RunFile::new(&id, sheet, snapshot, run::timestamp(now));
    let (constraints, decisions) = match carried {
        Some(carried) => {
            file.carried_from = Some(carried.from);
            (carried.constraints, Some(carried.decisions))
        }
        None => (None, None),
    };
    file.save(&dir)?;
    if let Some(decisions) = decisions {
        decisions.save(&dir)?;
    }
    let job = Job::Run(RunJob {
        workspace: ws.root().to_owned(),
        input,
        run_id: id,
        excluded: ws.file().excluded.clone(),
        spec: spec.clone(),
        constraints,
    });
    let worker = Worker::spawn(command, &job, Some(&dir.join(ENGINE_LOG)))?;
    Ok((file, worker))
}

/// Closes a run: how it ended, what it found, and when (A §4).
///
/// # Errors
///
/// [`AppError::Io`].
pub fn finish_run(
    ws: &Workspace,
    run: &mut RunFile,
    outcome: &Outcome,
    now: chrono::DateTime<chrono::Local>,
) -> Result<()> {
    match outcome {
        Outcome::Done { counts, engine, params } => {
            run.status = RunStatus::Done;
            run.counts.clone_from(counts);
            run.engine.clone_from(engine);
            run.params = *params;
        }
        // The reviewer's own button is not a failure the history shows in red (A §10).
        Outcome::Failed { kind: FailKind::Cancelled, .. } => run.status = RunStatus::Cancelled,
        Outcome::Failed { kind, message } => {
            run.status = RunStatus::Failed { kind: *kind, message: message.clone() };
        }
    }
    run.finished = Some(run::timestamp(now));
    run.save(&ws.run_dir(&run.id))
}

/// Writes [`ASSEMBLY_FILE`], which is the host's (A §2.1): the window reads it when a run is
/// reopened, instead of asking a worker to lay the groups out again.
///
/// # Errors
///
/// [`AppError::Io`].
pub fn save_assembly(run_dir: &Path, assembly: &AssemblyDto) -> Result<()> {
    atomic::write_json(&run_dir.join(ASSEMBLY_FILE), assembly)
}

/// The workspace's input folder as it can be reached now (A §4).
fn input_of(ws: &Workspace) -> Result<PathBuf> {
    ws.input().ok_or_else(|| AppError::Worker("the input folder is not available".to_owned()))
}

/// Opens `engine.log` for appending, making the run's folder if the host has not yet.
fn open_log(path: Option<&Path>) -> Result<Log> {
    let Some(path) = path else { return Ok(None) };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| AppError::io(parent, e))?;
    }
    let file = std::fs::File::options()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| AppError::io(path, e))?;
    Ok(Some(Arc::new(Mutex::new(file))))
}

/// Appends one line, and says nothing when it cannot: a log that has run out of disk must not be
/// what ends a run that is otherwise going well.
fn append(log: Option<&Mutex<std::fs::File>>, line: &[u8]) {
    let Some(file) = log else { return };
    let Ok(mut file) = file.lock() else { return };
    let _ = file.write_all(line).and_then(|()| file.write_all(b"\n"));
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Canceller, Carried, Requester, WorkerCommand, carry, start_run};
    use crate::decisions::{Decision, DecisionsFile, Verdict};
    use crate::protocol::RunSpec;
    use crate::workspace::Workspace;
    use crate::{AppError, run};

    /// The run the decisions were made in.
    const FROM: &str = "2026-09-20_1412";
    /// When they were carried into the next one.
    const CARRIED_AT: &str = "2026-09-21T09:15:00+03:00";

    const POSE: [[f64; 4]; 4] =
        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, -2.25], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]];

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-host-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn decided(a: &str, b: &str, verdict: Verdict) -> Decision {
        Decision {
            a: a.to_owned(),
            b: b.to_owned(),
            verdict,
            pose: (verdict == Verdict::Accept).then_some(POSE),
            source: Some("probable".to_owned()),
            bulk: false,
            at: "2026-09-20T14:31:07+03:00".to_owned(),
            carried_from: None,
        }
    }

    /// A §2.2: «Отменить» is pressed while [`super::drive`] holds the worker, so the handle has to
    /// outlive that borrow and cross to the window's own thread. Nothing else about the cancel is
    /// testable without a process — `tests/worker_e2e.rs` does that part.
    ///
    /// A review session's questions come from the same place and go down the same pipe (A §8),
    /// so its handle has to travel exactly as far.
    #[test]
    fn a_cancel_handle_leaves_the_thread_that_drives_the_run() {
        const fn goes_anywhere<T: Clone + Send + Sync + 'static>() {}
        goes_anywhere::<Canceller>();
        goes_anywhere::<Requester>();
    }

    /// A §8.5: the next run starts from the last review — the accepted pair pinned at its pose,
    /// the rejected one skipped — and a decision about a fragment the collection no longer holds
    /// is left behind **and said**, because `constraints::resolve` fails a whole run on a name it
    /// does not know.
    #[test]
    fn the_next_run_carries_the_last_review_and_says_what_could_not_come_along() {
        let ws = Workspace::create(&scratch("carry").join("karas")).unwrap();
        let mut made = DecisionsFile::default();
        made.set(decided("pieceA", "pieceB", Verdict::Accept));
        made.set(decided("pieceA", "gone", Verdict::Reject));
        std::fs::create_dir_all(ws.run_dir(FROM)).unwrap();
        made.save(&ws.run_dir(FROM)).unwrap();

        let names = ["pieceA", "pieceB"].map(str::to_owned);
        let carried = carry(&ws, FROM, &names, CARRIED_AT).unwrap();

        let json =
            serde_json::to_value(carried.constraints.as_ref().expect("one decision survives"))
                .unwrap();
        assert_eq!(json["must_join"].as_array().unwrap().len(), 1);
        assert_eq!(json["must_join"][0]["a"], "pieceA");
        assert_eq!(json["must_join"][0]["pose"][1][3], -2.25);
        assert_eq!(json["must_not_join"], serde_json::json!([]));
        assert_eq!(carried.dropped.len(), 1);
        assert_eq!(carried.dropped[0].b, "gone");
        // and the new run's own file starts from what came along, each entry saying where from
        assert_eq!(carried.decisions.decisions.len(), 1);
        let one = &carried.decisions.decisions[0];
        assert_eq!((one.a.as_str(), one.b.as_str()), ("pieceA", "pieceB"));
        assert_eq!(one.carried_from.as_deref(), Some(FROM));
        assert_eq!(one.at, CARRIED_AT);
        assert_eq!(carried.from, FROM);
    }

    /// A run that continues a review says so on disk before its worker exists (A §4, A §8.5):
    /// here the worker cannot even be started, and `runs/<id>/` is still a run that names the
    /// review it came from and holds it.
    #[test]
    fn a_run_started_from_a_review_files_it_before_the_worker() {
        let root = scratch("carried-run");
        let mut ws = Workspace::create(&root.join("karas")).unwrap();
        std::fs::create_dir_all(root.join("scans")).unwrap();
        ws.set_input(&root.join("scans")).unwrap();
        let mut made = DecisionsFile::default();
        made.set(decided("pieceA", "pieceB", Verdict::Accept));
        let names = ["pieceA", "pieceB"].map(str::to_owned);
        let carried = Carried {
            from: FROM.to_owned(),
            constraints: None,
            decisions: made.carried(&names, FROM, CARRIED_AT),
            dropped: Vec::new(),
        };

        let command = WorkerCommand { program: root.join("no-such-engine"), args: Vec::new() };
        let started =
            start_run(&ws, &command, &RunSpec::default(), Some(carried), chrono::Local::now());
        assert!(matches!(started, Err(AppError::Worker(_))), "{:?}", started.map(|(f, _)| f));

        let runs = run::list(&ws.runs_dir()).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].carried_from.as_deref(), Some(FROM));
        let filed = DecisionsFile::load_or_default(&ws.run_dir(&runs[0].id)).unwrap();
        assert_eq!(filed.decisions.len(), 1);
        assert_eq!(filed.decisions[0].carried_from.as_deref(), Some(FROM));
    }
}
