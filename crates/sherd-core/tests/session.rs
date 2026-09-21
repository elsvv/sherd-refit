//! A §3: a match saved by a run, read back, and reassembled. Every run here is `fixtures/slab`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use nalgebra::Matrix4;
use sherd_core::Params;
use sherd_core::assembly::constraints::Constraints;
use sherd_core::executor::Engine;
use sherd_core::matching::scales::Scales;
use sherd_core::memory::Budget;
use sherd_core::pipeline::{self, RunOptions, RunSummary};
use sherd_core::progress::Watch;
use sherd_core::review;
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

/// A §2.2's `PairDetail`: the window draws the seam live, and it must draw R §6's own numbers.
///
/// The review image (`review.rs`) is the same measurement rendered server-side, so this is the
/// guard that the two say the same thing about a pose: a placement the run accepted is mostly
/// green, and the same pair left where the two files put it is nowhere tight.
#[test]
fn a_seam_view_is_the_numbers_the_verification_judged_the_pose_by() {
    let (_, path) = accepted_run();
    let state = MatchState::load(path).unwrap();
    let frags = fragments(&state.params);
    let best = state
        .candidates
        .iter()
        .filter(|c| c.accepted)
        .max_by(|x, y| x.score().total_cmp(&y.score()))
        .expect("the slab has an accepted candidate");
    let (a, b) = (&frags[best.a as usize], &frags[best.b as usize]);

    let view = review::seam_view(Engine::REFERENCE, a, b, &best.transform, &state.params);
    assert_eq!(view.contact.len(), view.contact_class.len(), "one class per contact point");
    assert!(!view.contact.is_empty(), "the pair has fracture samples to colour");
    // R §6.1's `tightB` is a fraction of B's *facing* samples; the view colours every fracture
    // sample B has, and on this slab two thirds of them lie on faces pointing away from A and
    // come back from beyond the facing window. Of the points it shows as contact at all — green
    // and yellow — a pose R §6.5 accepted is mostly green.
    // Counted in one fold rather than two `filter().count()` passes: over a `Vec<u8>` clippy reads
    // the latter as a byte count and asks for the `bytecount` crate, which this test does not need.
    let (tight, near) = view.contact_class.iter().fold((0usize, 0usize), |(t, n), &c| match c {
        0 => (t + 1, n + 1),
        1 => (t, n + 1),
        _ => (t, n),
    });
    assert!(tight * 2 > near, "a join R §6.5 accepted is mostly tight: {tight} of {near}");
    assert!(!view.seam.is_empty(), "R §6.2 counted a shared seam");
    assert!(0.0 < view.tight && view.tight < view.gap, "{} < {}", view.tight, view.gap);
    // The two limits are R §1.2's own for this pair, not a distance the viewer invented.
    let scales = Scales::for_fragments(&state.params, a, b);
    assert_eq!((view.tight, view.gap), (scales.tight, scales.gap));

    // The identity pose leaves the two pieces where their own files put them, which is apart.
    let apart = review::seam_view(Engine::REFERENCE, a, b, &Matrix4::identity(), &state.params);
    assert_eq!(apart.contact.len(), view.contact.len(), "the same samples, drawn elsewhere");
    assert!(apart.contact_class.iter().all(|&c| c != 0), "nothing touches at the identity pose");
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
    let report = out.tiers.as_ref().expect("the tier pass ran");
    assert_eq!(
        [report.tiers.len(), report.evidence.len(), report.probes.len()],
        [out.candidates.len(); 3],
        "the tier report lost and gained the rows the candidate list did"
    );
    assert!(out.assembly.used.iter().all(|&i| i < out.candidates.len()));
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

/// The slab under a reviewer's accepted `pieceA`–`pieceB` join: the match, its fragments and the
/// reassembly, built once for the three tests that write it out with different output switches.
///
/// Once, because each of them would otherwise run the whole match and preprocess the collection a
/// second time to learn nothing new: the three differ only in what [`session::write_reviewed`] is
/// asked to write, and none of them touches what is here.
fn reviewed_slab() -> &'static (MatchState, Vec<sherd_core::fragment::Fragment>, Reassembled) {
    static REVIEWED: OnceLock<(MatchState, Vec<sherd_core::fragment::Fragment>, Reassembled)> =
        OnceLock::new();
    REVIEWED.get_or_init(|| {
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
        (state, frags, done)
    })
}

#[test]
fn a_reviewed_assembly_is_written_by_the_runs_own_writers() {
    let (state, frags, done) = reviewed_slab();
    let frags = frags.as_slice();

    let refined = session::refine_poses(
        Engine::REFERENCE,
        frags,
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
    let written = session::write_reviewed(&out, &slab(), frags, state, done, &done.poses, &options)
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

/// What the recorder was told about one stage, in the order it was told.
fn reports_of<'a>(
    seen: &'a [(String, usize, usize)],
    stage: &str,
) -> Vec<&'a (String, usize, usize)> {
    seen.iter().filter(|(s, ..)| s == stage).collect()
}

/// A §3.4's last stage, which milestone 6 needs: an export's minutes go into the mesh files, and
/// the window cannot draw a bar for a stage nobody reports.
///
/// The slab's accepted join places both its fragments, so `write_meshes` writes
/// `placed/pieceA.ply` and `placed/pieceB.ply`, and `viewer` adds `scene.glb`: three files, one
/// report each, ending at three of three.
#[test]
fn the_writers_report_one_report_per_mesh_file() {
    let (state, frags, done) = reviewed_slab();
    let recorder = std::sync::Arc::new(Recorder::default());
    let out = scratch("output-progress");
    let options = RunOptions {
        params: state.params,
        preview: false,
        write_meshes: true,
        viewer: true,
        watch: Watch { cancel: None, progress: Some(recorder.clone()) },
        ..RunOptions::default()
    };
    let written = session::write_reviewed(&out, &slab(), frags, state, done, &done.poses, &options)
        .expect("the writers run");
    for file in ["pieceA.ply", "pieceB.ply", "scene.glb"] {
        assert!(written.iter().any(|p| p.ends_with(file)), "{file} is written");
    }

    let seen = recorder.0.lock().unwrap();
    let output = reports_of(&seen, "output");
    assert_eq!(output.len(), 3, "one report per mesh file, no more: {output:?}");
    assert!(output.iter().all(|(_, _, total)| *total == 3), "the total is settled: {output:?}");
    // The placed meshes are written in parallel, so the reports arrive in whatever order the
    // threads finished — but each count is reported exactly once, and the last is the last file.
    let mut counts: Vec<usize> = output.iter().map(|(_, done, _)| *done).collect();
    let last = *counts.last().expect("three reports");
    counts.sort_unstable();
    assert_eq!(counts, [1, 2, 3], "every file moved the count by one: {output:?}");
    assert_eq!(last, 3, "the stage ends at the file it ends at: `scene.glb`");
}

/// With no meshes to write there is no stage: a bar over `0` of `0` is worse than no bar, and the
/// tables and the report are written from what the session already holds.
#[test]
fn with_no_meshes_the_output_stage_is_never_reported() {
    let (state, frags, done) = reviewed_slab();
    let recorder = std::sync::Arc::new(Recorder::default());
    let out = scratch("output-progress-tables");
    let options = RunOptions {
        params: state.params,
        preview: false,
        write_meshes: false,
        viewer: false,
        watch: Watch { cancel: None, progress: Some(recorder.clone()) },
        ..RunOptions::default()
    };
    let written = session::write_reviewed(&out, &slab(), frags, state, done, &done.poses, &options)
        .expect("the writers run");
    assert!(written.iter().any(|p| p.ends_with("report.md")), "the report is still written");
    assert!(!out.join("placed").exists(), "and no mesh is");
    let seen = recorder.0.lock().unwrap();
    assert!(reports_of(&seen, "output").is_empty(), "nothing to count: {seen:?}");
}
