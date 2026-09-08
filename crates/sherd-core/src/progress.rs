//! D §5's last two sentences: cancellation and progress.
//!
//! > *"Cancellation: every stage checks an `AtomicBool` between units of work (a pair, a
//! > fragment); the CLI wires Ctrl-C, the desktop app wires a button. Progress: a `Progress` trait
//! > with `(stage, done, total)` callbacks; the CLI prints, the app emits events."*
//!
//! Both are optional and both default to nothing, so a run that asks for neither pays neither: a
//! [`Cancel`] that was never handed one is a `None` and its check is a branch on a null pointer,
//! and a run with no [`Progress`] never formats a line.
//!
//! # Where the checks are, and why there
//!
//! Between units of work, never inside one. A fragment's preprocessing and a pair's match are the
//! two units, and a check between them costs one relaxed atomic load per unit — 20 fragments and
//! 190 pairs on `synthetic_20` — against the several hundred milliseconds a unit takes. Checking
//! *inside* a unit would buy a faster stop and cost the property that makes cancellation safe: a
//! stage either produced a whole fragment or produced none of it, so nothing half-built is ever
//! written or cached.
//!
//! # What cancellation does to the device
//!
//! Nothing it has to undo. A cancelled run stops handing batches to the executor; the batches
//! already on the device finish, because a dispatch is bounded by construction (D §6.4: ≤ 512
//! candidates, ≤ 20 M point-queries) and `sherd_gpu::pipeline`'s submitting thread retires every
//! command buffer it submitted before its channel closes. There is no partial submission to
//! reclaim and no fence to break: the run returns [`Error::Cancelled`](crate::Error::Cancelled)
//! and the device is left exactly as a finished run leaves it.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A flag every stage checks between units of work (D §5).
///
/// Cloning shares the flag, which is the point: the CLI keeps one for its signal handler and the
/// pipeline keeps another.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A flag that has not been raised.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Raises it. Every stage stops at its next unit of work.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether it has been raised.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Where a run reports what it has finished (D §5).
///
/// `Debug` because [`RunOptions`](crate::pipeline::RunOptions) is, `Send + Sync` because the
/// callbacks come from rayon's workers, and `&self` because there is one of these for the run.
pub trait Progress: Send + Sync + fmt::Debug {
    /// `done` of `total` units of `stage` are finished.
    ///
    /// Called from whichever worker finished the unit, so an implementation that prints has to
    /// expect them out of order and interleaved with the run's own log lines.
    fn advance(&self, stage: &str, done: usize, total: usize);
}

/// What a run was given to report to and to stop on — neither, either or both.
///
/// One value rather than two fields on every signature: the two are always passed together, they
/// are both optional, and a stage that has one almost always wants the other.
#[derive(Clone, Debug, Default)]
pub struct Watch {
    /// The flag to stop on, if any.
    pub cancel: Option<Cancel>,
    /// Where to report progress, if anywhere.
    pub progress: Option<Arc<dyn Progress>>,
}

impl Watch {
    /// A watch that stops on `cancel` and reports nowhere.
    #[must_use]
    pub fn cancelled_by(cancel: Cancel) -> Self {
        Self { cancel: Some(cancel), progress: None }
    }

    /// Whether the run has been asked to stop.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(Cancel::is_cancelled)
    }

    /// `Err(Cancelled)` when the run has been asked to stop, `Ok(())` otherwise.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`](crate::Error::Cancelled), and only that.
    pub fn check(&self) -> crate::Result<()> {
        if self.is_cancelled() { Err(crate::Error::Cancelled) } else { Ok(()) }
    }

    /// Reports one unit of `stage` finished.
    pub fn advance(&self, stage: &str, done: usize, total: usize) {
        if let Some(progress) = self.progress.as_ref() {
            progress.advance(stage, done, total);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cancel, Progress, Watch};
    use std::sync::Arc;
    use std::sync::Mutex;

    /// A `Progress` that keeps what it was told, for the tests and for nothing else.
    #[derive(Debug, Default)]
    struct Recorder(Mutex<Vec<(String, usize, usize)>>);

    impl Progress for Recorder {
        fn advance(&self, stage: &str, done: usize, total: usize) {
            if let Ok(mut seen) = self.0.lock() {
                seen.push((stage.to_owned(), done, total));
            }
        }
    }

    /// The empty watch is free and says no to everything: no flag, no callback, no error.
    #[test]
    fn a_watch_with_nothing_in_it_never_cancels_and_never_reports() {
        let watch = Watch::default();
        assert!(!watch.is_cancelled());
        assert!(watch.check().is_ok());
        watch.advance("matching", 1, 10);
    }

    /// The flag is shared by every clone, which is what lets a signal handler raise the one the
    /// pipeline is reading.
    #[test]
    fn the_flag_is_shared_and_the_check_is_the_only_error_it_makes() {
        let cancel = Cancel::new();
        let watch = Watch::cancelled_by(cancel.clone());
        assert!(watch.check().is_ok());
        cancel.cancel();
        assert!(watch.is_cancelled());
        assert!(matches!(watch.check(), Err(crate::Error::Cancelled)));
        assert_eq!(watch.check().unwrap_err().to_string(), "cancelled");
    }

    /// Progress reaches the implementation with the stage and the two counts, in the order the
    /// units finished.
    #[test]
    fn progress_carries_the_stage_and_the_two_counts() {
        let recorder = Arc::new(Recorder::default());
        let watch = Watch { cancel: None, progress: Some(recorder.clone()) };
        watch.advance("preprocess", 1, 4);
        watch.advance("matching", 7, 190);
        let seen = recorder.0.lock().expect("no panic in the recorder");
        assert_eq!(seen[0], ("preprocess".to_owned(), 1, 4));
        assert_eq!(seen[1], ("matching".to_owned(), 7, 190));
    }
}
