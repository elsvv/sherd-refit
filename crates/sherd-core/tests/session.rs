//! A §3: a match saved by a run, read back, and reassembled. Every run here is `fixtures/slab`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sherd_core::Params;
use sherd_core::assembly::constraints::Constraints;
use sherd_core::executor::Engine;
use sherd_core::memory::Budget;
use sherd_core::pipeline::{self, RunOptions, RunSummary};
use sherd_core::progress::Watch;
use sherd_core::session::{self, MatchState, Reassembled};
use sherd_core::tiers::{Thresholds, Tier};

fn slab() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("session-{tag}"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// One slab run without refinement, meshes or previews, leaving `match.state` in its folder.
fn slab_run(tag: &str, params: &Params) -> (RunSummary, PathBuf) {
    let out = scratch(tag);
    let state = out.join("match.state");
    let options = RunOptions {
        params: *params,
        refine: false,
        preview: false,
        write_meshes: false,
        match_state: Some(state.clone()),
        ..RunOptions::default()
    };
    (pipeline::run(&slab(), &out, &options).expect("the slab runs"), state)
}

/// The run most tests read: `Params::default()`, so R §6.5's own gate, and the slab assembled.
fn accepted_run() -> &'static (RunSummary, PathBuf) {
    static RUN: OnceLock<(RunSummary, PathBuf)> = OnceLock::new();
    RUN.get_or_init(|| slab_run("accepted", &Params::default()))
}

#[test]
fn a_saved_match_reads_back_exactly() {
    let (summary, path) = accepted_run();
    let state = MatchState::load(path).expect("the state loads");
    assert_eq!(state.names, summary.names);
    assert_eq!(state.candidates.len(), summary.candidates.len());
    assert_eq!(state.candidates[0].transform, summary.candidates[0].transform);
    assert_eq!(state.thickness.to_bits(), summary.thickness.to_bits());

    let again = path.with_extension("again");
    state.save(&again).expect("the state saves");
    assert_eq!(MatchState::load(&again).expect("and loads"), state);
    assert_eq!(std::fs::read(&again).unwrap(), std::fs::read(path).unwrap());
}

#[test]
fn a_state_of_another_version_or_format_is_refused() {
    let dir = scratch("refused");
    for (name, body) in [
        ("version", r#"{"format":"sherd-match-state","version":999}"#),
        ("format", r#"{"format":"something-else","version":1}"#),
        ("garbage", "not json at all"),
    ] {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        let err = MatchState::load(&path).expect_err("refused");
        assert!(matches!(err, sherd_core::Error::State { .. }), "{name}: {err}");
    }
}

fn fragments(params: &Params) -> Vec<sherd_core::fragment::Fragment> {
    let entries = sherd_core::collection::discover(slab()).expect("the slab is found");
    session::load_fragments(
        &entries,
        RunOptions::default().target_faces,
        None,
        Budget::default_for_machine(),
        params.seed,
        &Watch::default(),
    )
    .expect("the slab preprocesses")
}

fn reassembled(state: &MatchState, constraints: Option<&str>) -> Reassembled {
    let parsed: Option<Constraints> =
        constraints.map(|json| serde_json::from_str(json).expect("valid constraints"));
    session::reassemble(Engine::REFERENCE, &fragments(&state.params), state, parsed.as_ref())
        .expect("reassembles")
}

#[test]
fn reassembling_with_no_constraints_is_the_run_before_refinement() {
    let (summary, path) = accepted_run();
    let state = MatchState::load(path).unwrap();
    let out = reassembled(&state, None);
    assert_eq!(out.assembly.groups, summary.groups);
    assert_eq!(out.used, summary.used);
    assert_eq!(out.poses, summary.poses, "bit for bit: the run had `refine: false`");
}

#[test]
fn a_rejected_join_splits_its_group() {
    let (summary, path) = accepted_run();
    assert!(summary.groups.iter().any(|g| g.len() == 2), "the slab assembles under R §6.5's gate");
    let state = MatchState::load(path).unwrap();
    let out = reassembled(&state, Some(r#"{"version":1,"must_not_join":[["pieceA","pieceB"]]}"#));
    assert!(out.assembly.groups.iter().all(|g| g.len() == 1));
    assert!(out.used.is_empty());
}

#[test]
fn an_accepted_probable_join_is_placed_at_the_pose_that_was_accepted() {
    // The shipped tier rule confirms nothing on this slab (`run_cli.rs`, `AMBIGUOUS`): its one
    // join is probable, and a run places nothing. That is the app's review case exactly.
    let params = Params { tiers: Some(Thresholds::default()), ..Params::default() };
    let (summary, path) = slab_run("probable", &params);
    assert!(summary.groups.iter().all(|g| g.len() == 1), "nothing is confirmed, nothing placed");
    let state = MatchState::load(&path).unwrap();

    // The reviewer accepts the pair's *second* accepted pose where there is one, to show that the
    // pose they chose is the pose that is placed, not the pair's best.
    let accepted: Vec<_> = state.candidates.iter().filter(|c| c.accepted).collect();
    let chosen = accepted.get(1).or(accepted.first()).expect("the slab has an accepted candidate");
    let m = chosen.transform;
    let rows: Vec<String> = (0..4)
        .map(|r| format!("[{:?},{:?},{:?},{:?}]", m[(r, 0)], m[(r, 1)], m[(r, 2)], m[(r, 3)]))
        .collect();
    let json = format!(
        r#"{{"version":1,"must_join":[{{"a":"{}","b":"{}","pose":[{}]}}]}}"#,
        state.names[chosen.a as usize],
        state.names[chosen.b as usize],
        rows.join(",")
    );

    let out = reassembled(&state, Some(&json));
    assert!(out.assembly.groups.iter().any(|g| g.len() == 2), "the accepted join is placed");
    let placed = out.candidates.last().expect("the pinned candidate is appended last");
    assert_eq!(placed.transform, chosen.transform, "the pose is the one that was accepted");
    assert_eq!(placed.scores, chosen.scores, "R §6 was not run again: the match had scored it");
    assert_eq!(placed.tier, Tier::Confirmed);
    assert_eq!(
        out.candidates.iter().filter(|c| (c.a, c.b) == (chosen.a, chosen.b)).count(),
        1,
        "the pair's matched candidates are gone, as in a full run that pins the pair"
    );
}

#[test]
fn a_reviewed_assembly_is_written_by_the_runs_own_writers() {
    let params = Params { tiers: Some(Thresholds::default()), ..Params::default() };
    let (_, path) = slab_run("reviewed", &params);
    let state = MatchState::load(&path).unwrap();
    let chosen = *state.candidates.iter().find(|c| c.accepted).unwrap();
    let m = chosen.transform;
    let rows: Vec<String> = (0..4)
        .map(|r| format!("[{:?},{:?},{:?},{:?}]", m[(r, 0)], m[(r, 1)], m[(r, 2)], m[(r, 3)]))
        .collect();
    let json = format!(
        r#"{{"version":1,"must_join":[{{"a":"pieceA","b":"pieceB","pose":[{}]}}]}}"#,
        rows.join(",")
    );
    let frags = fragments(&state.params);
    let parsed: Constraints = serde_json::from_str(&json).unwrap();
    let done = session::reassemble(Engine::REFERENCE, &frags, &state, Some(&parsed)).unwrap();

    let refined = session::refine_poses(
        Engine::REFERENCE,
        &frags,
        &done.assembly.groups,
        &done.assembly.poses,
        &done.used,
        &state.params,
        Budget::default_for_machine(),
        &Watch::default(),
    )
    .expect("R §9 runs");
    assert_eq!(refined.len(), frags.len());

    let out = scratch("reviewed-out");
    let options = RunOptions {
        params: state.params,
        preview: false,
        write_meshes: false,
        ..RunOptions::default()
    };
    let written =
        session::write_reviewed(&out, &slab(), &frags, &state, &done, &done.poses, &options)
            .expect("the writers run");
    for file in ["transforms.json", "report.json", "report.md", "transforms.csv", "joins.csv"] {
        assert!(written.iter().any(|p| p.ends_with(file)), "{file} is written");
    }
    let report = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(report.contains("## Constraints"), "the reviewer's decision is in the report");
    assert!(report.contains("pieceA"));
}

/// A `Progress` that keeps what it was told, so the test can read the last report of a stage.
#[derive(Debug, Default)]
struct Recorder(std::sync::Mutex<Vec<(String, usize, usize)>>);

impl sherd_core::progress::Progress for Recorder {
    fn advance(&self, stage: &str, done: usize, total: usize) {
        self.0.lock().unwrap().push((stage.to_owned(), done, total));
    }
}

#[test]
fn the_tier_pass_and_the_refinement_report_their_progress() {
    let recorder = std::sync::Arc::new(Recorder::default());
    let out = scratch("progress");
    let options = RunOptions {
        // `rival_refused: false` is the one conjunct the slab needs off to confirm its join
        // (`run_cli.rs`, `AMBIGUOUS`) — and R §9 only runs when something was assembled.
        params: Params {
            tiers: Some(Thresholds { rival_refused: false, ..Thresholds::default() }),
            ..Params::default()
        },
        preview: false,
        write_meshes: false,
        watch: Watch { cancel: None, progress: Some(recorder.clone()) },
        ..RunOptions::default()
    };
    pipeline::run(&slab(), &out, &options).expect("the slab runs");
    let seen = recorder.0.lock().unwrap();
    for stage in ["preprocess", "matching", "tiers", "refine"] {
        let last = seen.iter().filter(|(s, ..)| s == stage).max_by_key(|(_, done, _)| *done);
        let (_, done, total) = last.unwrap_or_else(|| panic!("`{stage}` reported nothing"));
        assert!(*total > 0 && done == total, "`{stage}` ended at {done} of {total}");
    }
}
