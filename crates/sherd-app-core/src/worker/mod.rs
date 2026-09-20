//! The engine role of the app's binary (A §2.1): one job in, events out, one line of JSON each.
//!
//! Why a process: a run is 17–80 minutes, holds ~3 GB and drives `wgpu` on drivers nobody has
//! tested. An abort must end a run and not the window, the memory must go back to the OS, and a
//! hard kill must exist behind the cooperative cancel.

mod prepare;

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sherd_core::progress::{Cancel, Progress, Watch};

pub use prepare::{INDEX_FILE, THICKNESS_OUTLIER};

use crate::protocol::{Event, FailKind, Job, PROTOCOL, Request};

/// Where events go. One line each, flushed, under a lock: reports come from rayon's workers.
#[derive(Clone)]
pub(crate) struct Emitter(Arc<Mutex<Box<dyn Write + Send>>>);

impl std::fmt::Debug for Emitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Emitter")
    }
}

impl Emitter {
    /// Wraps the stream every event is written to — the process's stdout, in the real worker.
    pub(crate) fn new(output: impl Write + Send + 'static) -> Self {
        Self(Arc::new(Mutex::new(Box::new(output))))
    }

    /// Writes one event. A host that has gone away is not an error worth stopping for here: its
    /// going away closes stdin, which is what stops the job.
    pub(crate) fn emit(&self, event: &Event) {
        let Ok(mut line) = serde_json::to_vec(event) else { return };
        line.push(b'\n');
        if let Ok(mut out) = self.0.lock() {
            let _ = out.write_all(&line).and_then(|()| out.flush());
        }
    }
}

/// D §5's `Progress`, onto the wire. A stage's first report also announces the stage; after
/// that at most one line per [`WorkerProgress::EVERY`] per stage, and always the last one —
/// 11 000 pairs are 11 000 reports, and the window needs ten a second.
#[derive(Debug)]
pub(crate) struct WorkerProgress {
    emitter: Emitter,
    last: Mutex<HashMap<String, Instant>>,
}

impl WorkerProgress {
    /// The shortest gap between two [`Event::Progress`] lines of one stage.
    const EVERY: Duration = Duration::from_millis(100);

    /// Reports to `emitter`.
    pub(crate) fn new(emitter: Emitter) -> Self {
        Self { emitter, last: Mutex::new(HashMap::new()) }
    }
}

impl Progress for WorkerProgress {
    fn advance(&self, stage: &str, done: usize, total: usize) {
        let now = Instant::now();
        let Ok(mut last) = self.last.lock() else { return };
        let due = match last.get(stage) {
            None => {
                self.emitter.emit(&Event::Stage { name: stage.to_owned() });
                true
            }
            Some(at) => done >= total || now.duration_since(*at) >= Self::EVERY,
        };
        if due {
            last.insert(stage.to_owned(), now);
            self.emitter.emit(&Event::Progress { stage: stage.to_owned(), done, total });
        }
    }
}

/// What a job is handed.
#[derive(Debug)]
pub(crate) struct Context {
    /// Where the job's own events go.
    pub(crate) emitter: Emitter,
    /// D §5's flag, for the places that read it without a [`Watch`].
    pub(crate) cancel: Cancel,
    /// The flag and the progress sink together, as the engine's signatures take them.
    pub(crate) watch: Watch,
}

/// How a job ended, other than well.
#[derive(Debug)]
pub(crate) struct Failure {
    /// A §10's class.
    pub(crate) kind: FailKind,
    /// What to show, in the engine's words.
    pub(crate) message: String,
}

impl Failure {
    /// A failure of `kind`, saying `message`.
    pub(crate) fn new(kind: FailKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl From<sherd_core::Error> for Failure {
    fn from(error: sherd_core::Error) -> Self {
        Self { kind: fail_kind(&error), message: error.to_string() }
    }
}

/// A §10's class for an engine error.
pub(crate) fn fail_kind(error: &sherd_core::Error) -> FailKind {
    use sherd_core::Error as E;
    match error {
        E::Cancelled => FailKind::Cancelled,
        E::Write { .. } => FailKind::Disk,
        E::Read { .. } | E::UnsupportedFormat { .. } | E::EmptyMesh { .. } => FailKind::Input,
        _ => FailKind::Internal,
    }
}

/// Serves one job: `Hello`, the job's events, then `Done` or `Failed`. Returns the process's exit
/// code — 0 done, 1 failed, 2 cancelled.
///
/// After the job line the input is read on a thread of its own: a `cancel` request raises D §5's
/// flag, and so does the end of the input — a window that died leaves no 3 GB orphan (A §2.1).
pub fn serve(mut input: impl BufRead + Send + 'static, output: impl Write + Send + 'static) -> i32 {
    let emitter = Emitter::new(output);
    emitter.emit(&Event::Hello {
        protocol: PROTOCOL,
        core_version: sherd_core::CORE_VERSION.to_owned(),
        algo_ref: sherd_core::ALGO_REF.to_owned(),
        commit: sherd_core::GIT_COMMIT.to_owned(),
    });
    let mut line = String::new();
    let job: Job = match input.read_line(&mut line).map(|_| serde_json::from_str(line.trim())) {
        Ok(Ok(job)) => job,
        Ok(Err(e)) => {
            return fail(&emitter, &Failure::new(FailKind::Protocol, format!("not a job: {e}")));
        }
        Err(e) => return fail(&emitter, &Failure::new(FailKind::Protocol, e.to_string())),
    };

    let cancel = Cancel::new();
    let flag = cancel.clone();
    std::thread::spawn(move || {
        for line in input.lines() {
            match line.as_deref().map(str::trim).map(serde_json::from_str::<Request>) {
                Ok(Ok(Request::Cancel)) => flag.cancel(),
                Ok(Err(_)) => {} // an unknown request is ignored, not fatal: the window may be newer
                Err(_) => break,
            }
        }
        flag.cancel(); // end of input: the host is gone
    });

    let watch = Watch {
        cancel: Some(cancel.clone()),
        progress: Some(Arc::new(WorkerProgress::new(emitter.clone()))),
    };
    let context = Context { emitter: emitter.clone(), cancel, watch };
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dispatch(job, &context)));
    match outcome {
        Ok(Ok(done)) => {
            emitter.emit(&done);
            0
        }
        Ok(Err(failure)) => fail(&emitter, &failure),
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "the engine panicked".to_owned());
            fail(&emitter, &Failure::new(FailKind::Internal, message))
        }
    }
}

/// Says why, and gives the exit code that goes with it.
fn fail(emitter: &Emitter, failure: &Failure) -> i32 {
    emitter.emit(&Event::Failed { kind: failure.kind, message: failure.message.clone() });
    if failure.kind == FailKind::Cancelled { 2 } else { 1 }
}

/// Runs the job; the `Event` it returns is the job's `Done`.
fn dispatch(job: Job, context: &Context) -> Result<Event, Failure> {
    match job {
        Job::Info { adapter, selftest } => {
            context.emitter.emit(&Event::Info {
                backends: sherd_backend::info_lines(),
                selftest: if selftest {
                    sherd_backend::selftest_lines(adapter.as_deref())
                } else {
                    Vec::new()
                },
            });
            Ok(Event::Done { counts: None, engine: None, params: None })
        }
        Job::Prepare(job) => prepare::prepare(&job, context),
        Job::Run(_) => Err(Failure::new(
            FailKind::Protocol,
            "this build of the worker does not run this job yet",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn events(sink: &Sink) -> Vec<Event> {
        String::from_utf8(sink.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).expect("every line is an event"))
            .collect()
    }

    #[test]
    fn a_worker_says_hello_first_answers_info_and_ends_with_done() {
        let sink = Sink::default();
        let job = r#"{"job":"info","adapter":null,"selftest":false}"#;
        let code = serve(std::io::Cursor::new(format!("{job}\n")), sink.clone());
        let seen = events(&sink);
        assert!(matches!(&seen[0], Event::Hello { protocol, .. } if *protocol == PROTOCOL));
        assert!(matches!(&seen[1], Event::Info { backends, .. } if !backends.is_empty()));
        assert!(matches!(seen.last(), Some(Event::Done { .. })));
        assert_eq!(code, 0);
    }

    #[test]
    fn a_line_that_is_not_a_job_is_a_protocol_failure_and_not_a_panic() {
        let sink = Sink::default();
        let code = serve(std::io::Cursor::new("make me a sandwich\n".to_owned()), sink.clone());
        let seen = events(&sink);
        assert!(matches!(seen.last(), Some(Event::Failed { kind: FailKind::Protocol, .. })));
        assert_eq!(code, 1);
    }

    #[test]
    fn progress_is_throttled_but_a_stage_always_starts_and_ends_on_the_wire() {
        let sink = Sink::default();
        let progress = WorkerProgress::new(Emitter::new(sink.clone()));
        for done in 1..=1000 {
            sherd_core::progress::Progress::advance(&progress, "matching", done, 1000);
        }
        let seen = events(&sink);
        assert_eq!(seen[0], Event::Stage { name: "matching".into() });
        assert_eq!(
            seen.last(),
            Some(&Event::Progress { stage: "matching".into(), done: 1000, total: 1000 })
        );
        assert!(
            seen.len() < 50,
            "1000 reports in a few milliseconds are a handful of lines, not 1000"
        );
    }
}
