//! What the window is told about a workspace, in one value (A §5): the frontend derives every
//! state it shows from this and from the job's events, so it never has a second opinion to keep
//! in step with the first.
//!
//! Nothing here is stored. A view is built on demand from `sherd-workspace.json`, the input folder
//! as it stands, `fragments/index.json` and `runs/*/run.json`, which is why A §5's status table can
//! be a pure function of one value: there is no cached second copy of the workspace to go stale.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::protocol::FragmentInfo;
use crate::run::{self, RunFile};
use crate::snapshot::{self, FileStamp, InputSnapshot, StaleDiff};
use crate::worker::INDEX_FILE;
use crate::workspace::Workspace;
use crate::{Result, atomic};

/// Where the scans are (A §5). Three states, not an `Option`: «no folder linked» and «the linked
/// folder is gone» are different rows of A §5's table — the first asks for a folder, the second
/// keeps the results open and turns runs off — so the window must be able to tell them apart.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputView {
    /// Whether the workspace names an input folder at all.
    pub linked: bool,
    /// The folder as it resolves now, for the window to show; `None` when it does not resolve.
    pub path: Option<String>,
    /// Whether the folder can be reached now (A §5, «вход недоступен»).
    pub available: bool,
}

/// One run of the history, with what has changed under it since (A §5, «stale»). The staleness is
/// computed here and not on the frontend because it is a fact about the disk, and the window is
/// not allowed to walk the input folder itself (A §2.1).
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunView {
    /// The run's own `run.json`.
    pub run: RunFile,
    /// What the input has done since the run was made; empty when nothing has.
    pub stale: StaleDiff,
}

/// Which kind of job a worker is on (A §5). The window shows a different progress strip for each.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// Preprocessing the collection into the workspace's cache (A §5).
    Prepare,
    /// One run of the pipeline (A §7).
    Run,
}

/// The job that is running, if one is (A §5's «preparing» and «running» rows). The host owns this
/// — the view is built with it rather than reading it off the disk, because a running job is a
/// fact about this process and not about the workspace folder.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobView {
    /// What it is doing.
    pub kind: JobKind,
    /// The run it is writing, for a [`JobKind::Run`]; `None` for a `Prepare`.
    pub run_id: Option<String>,
}

/// Everything the window knows about the open workspace (A §5).
///
/// One value, and one command that returns it: every screen of the app derives what it shows from
/// this and from the events of the running job, so there is never a second opinion about the
/// workspace to keep in step with the first.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceView {
    /// The workspace folder, as text — the window shows it and never walks it (A §2.1).
    pub root: String,
    /// The folder's own name, which is what the title bar calls the workspace.
    pub name: String,
    /// Where the scans are.
    pub input: InputView,
    /// Fragments left out of every run (A §5.1), by name.
    pub excluded: Vec<String>,
    /// Every scan of the input folder as it stands now, the excluded ones included; empty when the
    /// input is not available.
    pub files: Vec<FileStamp>,
    /// The rows `fragments/index.json` holds — what the last `Prepare` produced.
    pub fragments: Vec<FragmentInfo>,
    /// Whether the input needs no `Prepare`; see [`prepared`].
    pub prepared: bool,
    /// The history, newest first.
    pub runs: Vec<RunView>,
    /// `fragments/`, as text: the window asks Tauri's asset protocol for meshes and thumbnails
    /// under it, and is given nothing outside the workspace (A §2.1).
    pub fragments_dir: String,
    /// The job a worker is on, if any.
    pub job: Option<JobView>,
}

/// Whether the input needs no `Prepare`: every scan that takes part has a row for the file as it
/// is now. No scans at all is `false` — there is nothing to show as ready.
pub fn prepared(
    files: &[FileStamp],
    excluded: &BTreeSet<String>,
    fragments: &[FragmentInfo],
) -> bool {
    let mut included = files.iter().filter(|f| !excluded.contains(&f.name)).peekable();
    included.peek().is_some()
        && included.all(|f| {
            fragments
                .iter()
                .any(|i| i.name == f.name && i.size == f.size && i.mtime_ms == f.mtime_ms)
        })
}

/// The whole view of `ws`, with `job` as the host knows it (A §5).
///
/// A missing input folder is a state and not an error (A §5, «вход недоступен»): the view is
/// built all the same, with no files and no staleness, so that the results of earlier runs stay
/// open. A damaged `index.json` is likewise no error — the next `Prepare` writes it again — so it
/// reads as no fragments rather than failing the window's only command.
///
/// # Errors
///
/// [`crate::AppError::Core`] or [`crate::AppError::Io`] when the input folder is there and cannot
/// be listed, and [`crate::AppError::Io`] when `runs/` cannot be listed.
pub fn build(ws: &Workspace, job: Option<JobView>) -> Result<WorkspaceView> {
    let root = ws.root();
    let excluded = &ws.file().excluded;

    let resolved = ws.input();
    let input = InputView {
        linked: ws.file().input.is_some(),
        path: resolved.as_ref().map(|p| p.to_string_lossy().into_owned()),
        available: resolved.is_some(),
    };
    let now = match &resolved {
        Some(dir) => snapshot::scan(dir, excluded)?,
        None => InputSnapshot::default(),
    };

    // A half-written or hand-edited index is not worth refusing to open the workspace over: the
    // next `Prepare` replaces it whole, and until then the collection simply reads as unprepared.
    let index = ws.fragments_dir().join(INDEX_FILE);
    let fragments: Vec<FragmentInfo> =
        if index.is_file() { atomic::read_json(&index).unwrap_or_default() } else { Vec::new() };

    let runs = run::list(&ws.runs_dir())?
        .into_iter()
        .map(|run| {
            // A run is not «stale» while the folder it ran on cannot be looked at: A §5 gives the
            // missing input its own row, and a diff against nothing would report every scan removed.
            let stale = if input.available {
                snapshot::diff(&run.input, &now)
            } else {
                StaleDiff::default()
            };
            RunView { run, stale }
        })
        .collect();

    Ok(WorkspaceView {
        root: root.to_string_lossy().into_owned(),
        name: root
            .file_name()
            .map_or_else(|| root.to_string_lossy(), std::ffi::OsStr::to_string_lossy)
            .into_owned(),
        prepared: prepared(&now.files, excluded, &fragments),
        input,
        excluded: excluded.iter().cloned().collect(),
        files: now.files,
        fragments,
        runs,
        fragments_dir: ws.fragments_dir().to_string_lossy().into_owned(),
        job,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FragmentInfo;

    fn stamp(name: &str, size: u64) -> FileStamp {
        FileStamp { name: name.into(), file: format!("{name}.ply"), size, mtime_ms: 1 }
    }

    fn info(name: &str, size: u64) -> FragmentInfo {
        let stats: sherd_core::report::FragmentStats = serde_json::from_value(serde_json::json!({
            "name": name, "faces": 1, "orig_faces": 1, "orig_vertices": 1, "thickness": 1.0,
            "thickness_mode": 1.0, "resolution": 1.0, "watertight": true, "extent": [1.0, 1.0, 1.0],
            "area": 1.0, "fracture_area_fraction": 0.1
        }))
        .unwrap();
        FragmentInfo {
            name: name.into(),
            file: format!("{name}.ply"),
            size,
            mtime_ms: 1,
            stats,
            warnings: Vec::new(),
            coloured: false,
            display_faces: 1,
        }
    }

    /// A §5: «preparing» ends when every scan that takes part has an up-to-date row — an excluded
    /// scan needs none, a changed scan's old row does not count, and no scans is not prepared.
    #[test]
    fn prepared_means_every_included_scan_has_a_current_row() {
        let files = [stamp("a", 10), stamp("b", 20)];
        let none = BTreeSet::new();
        assert!(prepared(&files, &none, &[info("a", 10), info("b", 20)]));
        assert!(!prepared(&files, &none, &[info("a", 10)]), "b has no row");
        assert!(!prepared(&files, &none, &[info("a", 10), info("b", 21)]), "b changed since");
        assert!(prepared(&files, &BTreeSet::from(["b".to_owned()]), &[info("a", 10)]));
        assert!(!prepared(&[], &none, &[]), "nothing to prepare is not «prepared»");
    }

    #[test]
    fn a_view_of_a_fresh_workspace_says_no_input_and_survives_a_missing_one() {
        let root = std::env::temp_dir().join(format!("sherd-view-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let scans = root.join("scans");
        std::fs::create_dir_all(&scans).unwrap();
        std::fs::write(scans.join("a.ply"), b"x").unwrap();
        let mut ws = crate::workspace::Workspace::create(&root.join("karas")).unwrap();

        let empty = build(&ws, None).unwrap();
        assert_eq!(empty.name, "karas");
        assert_eq!(empty.input, InputView { linked: false, path: None, available: false });

        ws.set_input(&scans).unwrap();
        let linked = build(&ws, None).unwrap();
        assert!(linked.input.linked && linked.input.available);
        assert_eq!(linked.files.len(), 1);
        assert!(!linked.prepared && linked.fragments.is_empty() && linked.runs.is_empty());

        std::fs::remove_dir_all(&scans).unwrap();
        let missing = build(&ws, None).unwrap();
        assert!(
            missing.input.linked && !missing.input.available,
            "A §5: input missing, not an error"
        );
        assert!(missing.files.is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    /// The mirror exists because `FragmentStats` is the engine's type and cannot derive `TS` here;
    /// this is what keeps the mirror honest.
    #[cfg(feature = "ts")]
    #[test]
    fn the_typescript_mirror_of_fragment_stats_has_the_engines_fields() {
        let real = serde_json::to_value(info("a", 1).stats).unwrap();
        let mirror = serde_json::to_value(crate::protocol::FragmentStatsTs::default()).unwrap();
        let keys =
            |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&real), keys(&mirror));
    }
}
