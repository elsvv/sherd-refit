//! A §3: a match saved by a run, read back, and reassembled. Every run here is `fixtures/slab`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sherd_core::Params;
use sherd_core::pipeline::{self, RunOptions, RunSummary};
use sherd_core::session::MatchState;

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
fn slab_run(tag: &str, params: Params) -> (RunSummary, PathBuf) {
    let out = scratch(tag);
    let state = out.join("match.state");
    let options = RunOptions {
        params,
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
    RUN.get_or_init(|| slab_run("accepted", Params::default()))
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
