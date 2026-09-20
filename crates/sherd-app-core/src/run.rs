//! `run.json` (A §4): what a run was asked, on what input, how it ended. Written by the host —
//! the worker never touches it (A §2.1) — so that a worker that dies still leaves a run the
//! history can explain.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sherd_core::Params;

use crate::snapshot::InputSnapshot;
use crate::{AppError, Result, atomic};

/// The file's name inside a run folder.
pub const RUN_FILE: &str = "run.json";
/// The format this build writes.
pub const RUN_VERSION: u32 = 1;

/// Why a job failed (A §10). The window turns a kind into a sentence and an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailKind {
    /// A scan cannot be read or parsed.
    Input,
    /// Fewer than two fragments.
    TooFew,
    /// A write failed, or there is no space.
    Disk,
    /// The adapter or the device.
    Gpu,
    /// The window and the worker do not speak the same protocol.
    Protocol,
    /// A panic in the engine; the backtrace is in `engine.log`.
    Internal,
    /// The process ended without saying how.
    Crashed,
    /// The user's button, or the host going away.
    Cancelled,
}

/// Where a run is in its life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RunStatus {
    /// A worker is on it — or was, when the app last wrote.
    Running,
    /// Finished.
    Done,
    /// Stopped by the user.
    Cancelled,
    /// Ended badly.
    Failed {
        /// Why, by class.
        kind: FailKind,
        /// Why, in the engine's words.
        message: String,
    },
    /// Found `running` at start-up with no worker behind it.
    Interrupted,
}

/// What matched: D §4.3's engine block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    /// `sherd-core`'s version.
    pub core_version: String,
    /// The algorithm reference it implements.
    pub algo_ref: String,
    /// The commit it was built from.
    pub commit: String,
    /// `cpu`, or `gpu:<adapter>`.
    pub backend: String,
}

/// One stage's wall clock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageTime {
    /// R §11.2's stage name.
    pub stage: String,
    /// Seconds.
    pub seconds: f64,
}

/// What a finished run found — the numbers the history and the mode tabs show.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunCounts {
    /// Fragments in the run.
    pub fragments: usize,
    /// Pairs matched.
    pub pairs: usize,
    /// Pairs R §4.1's wall-ratio filter skipped.
    pub skipped_pairs: usize,
    /// Candidates scored.
    pub candidates: usize,
    /// Confirmed joins (with the tier off: accepted ones).
    pub confirmed: usize,
    /// Probable joins.
    pub probable: usize,
    /// Groups of two or more.
    pub groups: usize,
    /// Fragments in no group.
    pub unassembled: usize,
    /// R §11.2's timings, in the order the stages finished.
    pub timings: Vec<StageTime>,
}

/// `run.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunFile {
    /// [`RUN_VERSION`].
    pub version: u32,
    /// The folder's name.
    pub id: String,
    /// RFC 3339, local offset.
    pub created: String,
    /// When it ended, however it ended.
    #[serde(default)]
    pub finished: Option<String>,
    /// Where it is in its life.
    pub status: RunStatus,
    /// The launch sheet, as the window sent it.
    pub spec: serde_json::Value,
    /// Every threshold it resolved to; filled when the run ends.
    #[serde(default)]
    pub params: Option<Params>,
    /// The input it ran on (A §4, «Stale»).
    pub input: InputSnapshot,
    /// What matched.
    #[serde(default)]
    pub engine: Option<EngineInfo>,
    /// What it found.
    #[serde(default)]
    pub counts: Option<RunCounts>,
    /// The run whose decisions it started from (A §8.5).
    #[serde(default)]
    pub carried_from: Option<String>,
}

impl RunFile {
    /// A run that is about to start.
    pub fn new(id: &str, spec: serde_json::Value, input: InputSnapshot, created: String) -> Self {
        Self {
            version: RUN_VERSION,
            id: id.to_owned(),
            created,
            finished: None,
            status: RunStatus::Running,
            spec,
            params: None,
            input,
            engine: None,
            counts: None,
            carried_from: None,
        }
    }

    /// Reads `<dir>/run.json`.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`], [`AppError::Json`], [`AppError::Version`].
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(RUN_FILE);
        let run: Self = atomic::read_json(&path)?;
        if run.version > RUN_VERSION {
            return Err(AppError::Version { path, found: run.version, expected: RUN_VERSION });
        }
        Ok(run)
    }

    /// Writes `<dir>/run.json`, creating the folder.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| AppError::io(dir, e))?;
        atomic::write_json(&dir.join(RUN_FILE), self)
    }
}

/// Every run of the workspace, newest first. A folder without a readable `run.json` is not a run
/// and is skipped: the history must open even when one folder is damaged.
///
/// # Errors
///
/// [`AppError::Io`] when `runs/` exists and cannot be listed. A missing `runs/` is no runs.
pub fn list(runs_dir: &Path) -> Result<Vec<RunFile>> {
    if !runs_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut runs: Vec<RunFile> = std::fs::read_dir(runs_dir)
        .map_err(|e| AppError::io(runs_dir, e))?
        .filter_map(std::io::Result::ok)
        .filter_map(|entry| RunFile::load(&entry.path()).ok())
        .collect();
    runs.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(runs)
}

/// Marks every run still `running` as interrupted (A §4). Called once when a workspace opens,
/// before any worker is spawned, so a `running` on disk can only be one nobody is running.
///
/// # Errors
///
/// [`AppError::Io`].
pub fn mark_interrupted(runs_dir: &Path, finished: &str) -> Result<Vec<String>> {
    let mut marked = Vec::new();
    for mut run in list(runs_dir)? {
        if run.status == RunStatus::Running {
            run.status = RunStatus::Interrupted;
            run.finished = Some(finished.to_owned());
            run.save(&runs_dir.join(&run.id))?;
            marked.push(run.id);
        }
    }
    Ok(marked)
}

/// A run's id: local time to the minute, `-2`, `-3`, … when the minute is taken (A §4).
pub fn new_id(existing: &[String], now: chrono::DateTime<chrono::Local>) -> String {
    let base = now.format("%Y-%m-%d_%H%M").to_string();
    if !existing.contains(&base) {
        return base;
    }
    // `base` plus the suffixes up to `existing.len() + 1` are one more id than `existing` can
    // hold, so one of them is always free and the fallback never fires. The range is bounded
    // rather than open so that it is the type, and not an argument about the data, that says so.
    (2..=existing.len() + 1)
        .map(|n| format!("{base}-{n}"))
        .find(|id| !existing.contains(id))
        .unwrap_or(base)
}

/// `now` as `run.json` writes a time.
pub fn timestamp(now: chrono::DateTime<chrono::Local>) -> String {
    now.to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn runs(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-runs-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn a_run(id: &str) -> RunFile {
        RunFile::new(
            id,
            serde_json::json!({}),
            InputSnapshot::default(),
            "2026-09-20T14:12:00+03:00".to_owned(),
        )
    }

    #[test]
    fn run_ids_are_local_minutes_and_a_second_run_in_the_same_minute_gets_a_suffix() {
        let now = chrono::Local.with_ymd_and_hms(2026, 9, 20, 14, 12, 40).unwrap();
        assert_eq!(new_id(&[], now), "2026-09-20_1412");
        let taken = ["2026-09-20_1412".to_owned(), "2026-09-20_1412-2".to_owned()];
        assert_eq!(new_id(&taken, now), "2026-09-20_1412-3");
    }

    /// A §4: a `running` with no live worker behind it is a run the app did not see the end of.
    #[test]
    fn a_run_left_running_is_marked_interrupted_and_the_list_is_newest_first() {
        let dir = runs("lifecycle");
        let mut done = a_run("2026-09-19_0900");
        done.status = RunStatus::Done;
        done.save(&dir.join(&done.id)).unwrap();
        a_run("2026-09-20_1412").save(&dir.join("2026-09-20_1412")).unwrap();
        std::fs::create_dir_all(dir.join("not-a-run")).unwrap();

        assert_eq!(
            mark_interrupted(&dir, "2026-09-20T15:00:00+03:00").unwrap(),
            ["2026-09-20_1412"]
        );
        let listed = list(&dir).unwrap();
        assert_eq!(
            listed.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["2026-09-20_1412", "2026-09-19_0900"]
        );
        assert_eq!(listed[0].status, RunStatus::Interrupted);
        assert_eq!(listed[0].finished.as_deref(), Some("2026-09-20T15:00:00+03:00"));
        assert_eq!(listed[1].status, RunStatus::Done);
    }

    #[test]
    fn a_failure_is_stored_with_its_kind() {
        let failed = RunStatus::Failed { kind: FailKind::Gpu, message: "device lost".into() };
        let json = serde_json::to_value(failed).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "state": "failed", "kind": "gpu", "message": "device lost" })
        );
    }
}
