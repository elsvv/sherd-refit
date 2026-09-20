//! A §11: the worker driven headless, through a real process and real pipes, on `fixtures/slab`.

use std::path::{Path, PathBuf};

use sherd_app_core::host::{self, Outcome, Worker, WorkerCommand};
use sherd_app_core::protocol::{BackendChoice, Event, FailKind, FragmentInfo, RunSpec};
use sherd_app_core::run::{RunFile, RunStatus};
use sherd_app_core::workspace::Workspace;

fn slab() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
}

fn command() -> WorkerCommand {
    WorkerCommand { program: env!("CARGO_BIN_EXE_sherd-engine-worker").into(), args: Vec::new() }
}

fn workspace(tag: &str) -> Workspace {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("e2e-{tag}"));
    std::fs::remove_dir_all(&root).ok();
    let mut ws = Workspace::create(&root).expect("a workspace");
    ws.set_input(&slab()).expect("the slab is linked");
    ws
}

fn cpu() -> RunSpec {
    RunSpec { backend: BackendChoice::Cpu, ..RunSpec::default() }
}

fn now() -> chrono::DateTime<chrono::Local> {
    chrono::Local::now()
}

#[test]
fn a_collection_is_prepared_and_run_and_the_workspace_holds_what_a_4_says() {
    let ws = workspace("prepare-run");

    // Prepare
    let job = host::prepare_job(&ws, &cpu()).unwrap();
    let mut worker = Worker::spawn(&command(), &job, None).unwrap();
    let mut ready: Vec<FragmentInfo> = Vec::new();
    let outcome = host::drive(&mut worker, |event| {
        if let Event::FragmentReady(info) = event {
            ready.push(info.clone());
        }
    });
    assert!(matches!(outcome, Outcome::Done { .. }), "{outcome:?}");
    ready.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(ready.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["pieceA", "pieceB"]);
    for name in ["pieceA", "pieceB"] {
        for file in [format!("{name}.glb"), format!("{name}.seg.glb"), format!("{name}.png")] {
            assert!(ws.fragments_dir().join(&file).is_file(), "{file}");
        }
        assert!(ws.cache_dir().join(format!("{name}.sherd")).is_file());
    }
    let index: Vec<FragmentInfo> =
        sherd_app_core::atomic::read_json(&ws.fragments_dir().join("index.json")).unwrap();
    assert_eq!(index.len(), 2);
    assert!(index[0].stats.faces > 0 && index[0].display_faces > 0);

    // Run
    let (mut run, mut worker) =
        host::start_run(&ws, &command(), &cpu(), None, None, now()).unwrap();
    let dir = ws.run_dir(&run.id);
    assert_eq!(RunFile::load(&dir).unwrap().status, RunStatus::Running);
    let mut stages: Vec<String> = Vec::new();
    let mut matching_ended = false;
    let outcome = host::drive(&mut worker, |event| match event {
        Event::Stage { name } => stages.push(name.clone()),
        Event::Progress { stage, done, total } if stage == "matching" => {
            matching_ended = done == total;
        }
        Event::Assembly(assembly) => host::save_assembly(&dir, assembly).unwrap(),
        _ => {}
    });
    host::finish_run(&ws, &mut run, &outcome, now()).unwrap();

    let run = RunFile::load(&dir).unwrap();
    assert_eq!(run.status, RunStatus::Done, "{outcome:?}");
    assert!(matching_ended, "matching reported its last pair");
    assert!(stages.iter().any(|s| s == "matching") && stages.iter().any(|s| s == "tiers"));
    assert_eq!(run.input.files.len(), 2);
    let counts = run.counts.expect("a finished run has counts");
    assert_eq!((counts.fragments, counts.pairs), (2, 1));
    // the shipped rule confirms nothing on this slab: its join is probable and nothing is placed
    assert_eq!((counts.confirmed, counts.groups, counts.unassembled), (0, 0, 2));
    assert!(counts.probable >= 1);
    assert_eq!(run.engine.unwrap().backend, "cpu");
    for file in [
        "match.state",
        "candidates.json",
        "assembly.json",
        "report.json",
        "transforms.json",
        "engine.log",
    ] {
        assert!(dir.join(file).is_file(), "{file}");
    }
    for heavy in ["placed", "scene.glb", "viewer.html", "review"] {
        assert!(!dir.join(heavy).exists(), "a run in the app writes no {heavy}");
    }
}

#[test]
fn a_cancelled_run_ends_as_cancelled() {
    let ws = workspace("cancel");
    let (mut run, mut worker) =
        host::start_run(&ws, &command(), &cpu(), None, None, now()).unwrap();
    worker.cancel();
    let outcome = host::drive(&mut worker, |_| {});
    assert!(matches!(outcome, Outcome::Failed { kind: FailKind::Cancelled, .. }), "{outcome:?}");
    host::finish_run(&ws, &mut run, &outcome, now()).unwrap();
    assert_eq!(RunFile::load(&ws.run_dir(&run.id)).unwrap().status, RunStatus::Cancelled);
}

/// A §2.1: a window that dies leaves no orphan — the end of stdin is a cancel.
#[test]
fn a_worker_whose_host_goes_away_stops_by_itself() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let ws = workspace("orphan");
    let job = host::prepare_job(&ws, &cpu()).unwrap();
    let mut child = Command::new(command().program)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{}", serde_json::to_string(&job).unwrap()).unwrap();
    drop(stdin);
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(2), "cancelled, by the end of its input");
}
