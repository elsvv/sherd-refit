//! D §5's cancellation and progress, on the committed slab pair (task G3, item 6).
//!
//! Two properties, and the second is the one that needs a real run to state:
//!
//! * a flag raised **before** the run stops it at the first unit of work and writes nothing;
//! * a flag raised **during** it — from the progress callback, so the moment is deterministic
//!   rather than a sleep — stops it at the next unit and still writes nothing, because every unit
//!   is whole and the outputs are the last stage.
//!
//! The slab pair is the one collection every checkout has (`fixtures/slab/input`), and it is small
//! enough that a cancelled run is a fraction of a second.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use sherd_core::error::Error;
use sherd_core::pipeline::{self, RunOptions};
use sherd_core::progress::{Cancel, Progress, Watch};

fn slab() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sherd-cancel-{}-{tag}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// Everything below `dir`, so that "wrote nothing" can be asserted rather than assumed.
fn files(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(next) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path.strip_prefix(dir).unwrap_or(&path).to_string_lossy().into_owned());
            }
        }
    }
    out.sort();
    out
}

/// A run with no previews, no meshes and no cache: nothing but `report.*` and `transforms.json`
/// at the end, so an empty directory means the run really stopped.
fn options(watch: Watch) -> RunOptions {
    RunOptions { preview: false, write_meshes: false, cache: None, watch, ..RunOptions::default() }
}

/// A `Progress` that raises the flag the moment the run reports its first unit — D §5's callback
/// used as the clock, so the cancellation lands inside the run and lands at the same place every
/// time.
#[derive(Debug)]
struct CancelOnFirst {
    cancel: Cancel,
    seen: AtomicUsize,
    stages: Mutex<Vec<String>>,
}

impl Progress for CancelOnFirst {
    fn advance(&self, stage: &str, done: usize, total: usize) {
        assert!(done <= total, "{stage}: {done} of {total}");
        assert!(total > 0, "{stage}: a stage with no units should not report");
        if let Ok(mut stages) = self.stages.lock()
            && !stages.iter().any(|s| s == stage)
        {
            stages.push(stage.to_owned());
        }
        if self.seen.fetch_add(1, Ordering::Relaxed) == 0 {
            self.cancel.cancel();
        }
    }
}

/// A flag that is already up stops the run at its first fragment, and the output directory stays
/// empty.
#[test]
fn a_run_cancelled_before_it_starts_writes_nothing() {
    let out = scratch("before");
    let cancel = Cancel::new();
    cancel.cancel();
    let result = pipeline::run(&slab(), &out, &options(Watch::cancelled_by(cancel)));
    assert!(matches!(result, Err(Error::Cancelled)), "expected Cancelled, got {result:?}");
    assert_eq!(files(&out), Vec::<String>::new(), "a cancelled run writes no outputs");
}

/// A flag raised from the first progress callback stops the run at the next unit of work, and
/// still writes nothing — because the outputs are the last stage and every unit is whole.
#[test]
fn a_run_cancelled_from_its_own_progress_callback_stops_and_writes_nothing() {
    let out = scratch("during");
    let cancel = Cancel::new();
    let progress = std::sync::Arc::new(CancelOnFirst {
        cancel: cancel.clone(),
        seen: AtomicUsize::new(0),
        stages: Mutex::new(Vec::new()),
    });
    let watch = Watch { cancel: Some(cancel), progress: Some(progress.clone()) };
    let result = pipeline::run(&slab(), &out, &options(watch));
    assert!(matches!(result, Err(Error::Cancelled)), "expected Cancelled, got {result:?}");
    assert!(progress.seen.load(Ordering::Relaxed) >= 1, "the run has to have reported something");
    let stages = progress.stages.lock().expect("no panic in the callback");
    assert_eq!(stages.first().map(String::as_str), Some("preprocess"), "{stages:?}");
    assert_eq!(files(&out), Vec::<String>::new(), "a cancelled run writes no outputs");
}

/// The same collection with no watch at all runs to the end — the control that says the two tests
/// above measured cancellation and not a broken pipeline.
#[test]
fn the_same_run_without_a_flag_finishes() {
    let out = scratch("control");
    let summary = pipeline::run(&slab(), &out, &options(Watch::default())).expect("the slab pair");
    assert_eq!(summary.names.len(), 2);
    assert!(files(&out).iter().any(|f| f == "transforms.json"), "{:?}", files(&out));
}
