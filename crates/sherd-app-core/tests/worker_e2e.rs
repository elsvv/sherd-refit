//! A §11: the worker driven headless, through a real process and real pipes, on `fixtures/slab`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sherd_app_core::decisions::{Decision, DecisionsFile, Verdict};
use sherd_app_core::host::{self, HostEvent, Outcome, Worker, WorkerCommand};
use sherd_app_core::protocol::{
    BackendChoice, CandidateRow, Event, FailKind, FragmentInfo, Job, Request, ReviewJob, RunSpec,
};
use sherd_app_core::run::{RunFile, RunStatus};
use sherd_app_core::workspace::Workspace;
use sherd_core::tiers::Tier;

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
    let (mut run, mut worker) = host::start_run(&ws, &command(), &cpu(), None, now()).unwrap();
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
    let (mut run, mut worker) = host::start_run(&ws, &command(), &cpu(), None, now()).unwrap();
    worker.cancel();
    let outcome = host::drive(&mut worker, |_| {});
    assert!(matches!(outcome, Outcome::Failed { kind: FailKind::Cancelled, .. }), "{outcome:?}");
    host::finish_run(&ws, &mut run, &outcome, now()).unwrap();
    assert_eq!(RunFile::load(&ws.run_dir(&run.id)).unwrap().status, RunStatus::Cancelled);
}

/// A §10's hard kill, through the handle the shell keeps beside the canceller: a worker that has
/// stopped reading its input is ended anyway, from a thread that is not the one driving it.
///
/// This is what A §8.4 stands on — a review session may be a cache and may never be a reason to
/// refuse a run — and what a cancel cannot do on its own: `Cancel` is a line down the same pipe
/// the wedged worker is not reading.
#[test]
fn a_worker_is_killed_through_its_handle_while_another_thread_drives_it() {
    let ws = workspace("kill");
    let job = host::prepare_job(&ws, &cpu()).unwrap();
    let mut worker = Worker::spawn(&command(), &job, None).unwrap();
    let killer = worker.killer();
    // At `Hello` the worker has a preprocessing of the slab ahead of it — seconds of work this
    // kill lands in the middle of, whatever the machine. The thread is the point: `drive` owns
    // the worker on this one, exactly as the job thread owns it while the window asks for a run.
    let outcome = host::drive(&mut worker, |event| {
        if matches!(event, Event::Hello { .. }) {
            let killer = killer.clone();
            std::thread::spawn(move || killer.kill()).join().unwrap();
        }
    });
    assert!(
        matches!(&outcome, Outcome::Failed { kind: FailKind::Crashed, message }
            if message.contains("killed")),
        "{outcome:?}"
    );
    // And a handle that outlives its worker signals nothing: the child has been reaped, so there
    // is no pid left here to send a second kill to.
    killer.kill();
}

/// The next event the session says that `want` is interested in.
///
/// A `failed` or a `request_failed` that nobody was waiting for is the end of the test and not
/// something to wait past: the session is meant to answer every one of these, and its own
/// sentence is the best report of why it did not.
fn wait_for(worker: &mut Worker, want: impl Fn(&Event) -> bool) -> Event {
    loop {
        match worker.next() {
            Some(HostEvent::Event(event)) => {
                if want(&event) {
                    return event;
                }
                assert!(
                    !matches!(event, Event::Failed { .. } | Event::RequestFailed { .. }),
                    "{event:?}"
                );
            }
            other => panic!("the session ended before it answered: {other:?}"),
        }
    }
}

/// The session's answer to whatever was just asked.
fn assembly(worker: &mut Worker) -> sherd_app_core::protocol::AssemblyDto {
    match wait_for(worker, |e| matches!(e, Event::Assembly(_))) {
        Event::Assembly(assembly) => assembly,
        other => unreachable!("{other:?}"),
    }
}

/// One decision about the pair, as the window would file it (A §8.1).
fn decided(a: &str, b: &str, verdict: Verdict, pose: Option<[[f64; 4]; 4]>) -> DecisionsFile {
    let mut file = DecisionsFile::default();
    file.set(Decision {
        a: a.to_owned(),
        b: b.to_owned(),
        verdict,
        pose,
        source: Some("probable".to_owned()),
        bulk: false,
        at: "2026-09-21T12:00:00+03:00".to_owned(),
        carried_from: None,
    });
    file
}

/// A §8: the review session end to end — the match loaded once, a decision answered with an
/// assembly, the seam of that placement measured, R §9 over what the decision left unrefined, and
/// a refined group that survives the next reassembly.
#[test]
fn a_review_session_answers_a_decision_with_an_assembly_and_keeps_what_it_refined() {
    let ws = workspace("review");
    let job = host::prepare_job(&ws, &cpu()).unwrap();
    let mut worker = Worker::spawn(&command(), &job, None).unwrap();
    assert!(matches!(host::drive(&mut worker, |_| {}), Outcome::Done { .. }));

    // The shipped rule places nothing on this slab — its one join is probable — which is exactly
    // the starting point a review exists for.
    let (mut run, mut worker) = host::start_run(&ws, &command(), &cpu(), None, now()).unwrap();
    let dir = ws.run_dir(&run.id);
    let outcome = host::drive(&mut worker, |event| {
        if let Event::Assembly(assembly) = event {
            host::save_assembly(&dir, assembly).unwrap();
        }
    });
    host::finish_run(&ws, &mut run, &outcome, now()).unwrap();
    assert!(matches!(outcome, Outcome::Done { .. }), "{outcome:?}");
    let rows: Vec<CandidateRow> =
        sherd_app_core::atomic::read_json(&dir.join("candidates.json")).unwrap();
    let join = rows.iter().find(|r| r.tier == Tier::Probable).expect("the slab's join is probable");

    let session = Job::Review(ReviewJob {
        workspace: ws.root().to_owned(),
        input: slab(),
        run_id: run.id.clone(),
        excluded: BTreeSet::new(),
        target_faces: cpu().target_faces,
        seed: cpu().seed,
        memory_gb: None,
        workers: 0,
    });
    let mut worker = Worker::spawn(&command(), &session, None).unwrap();
    let ask = worker.requester();
    let ready = wait_for(&mut worker, |e| matches!(e, Event::Ready { .. }));
    assert!(matches!(ready, Event::Ready { fragments: 2, candidates } if candidates >= 1));

    // Accepted: R §8 builds with the pinned pose and the two pieces are one group.
    let accepted = decided(&join.a, &join.b, Verdict::Accept, Some(join.pose));
    ask.send(&Request::Reassemble { decisions: accepted.clone() }).unwrap();
    let one = assembly(&mut worker);
    assert_eq!(one.groups.len(), 1, "{:?}", one.groups);
    assert_eq!(one.groups[0].members.len(), 2);
    assert!(!one.groups[0].refined, "a reassembly refines nothing (A §8.4)");
    assert!(one.unplaced.is_empty(), "{:?}", one.unplaced);
    assert_eq!(one.joins.len(), 1);

    // The seam of that placement, as the screen draws it.
    ask.send(&Request::PairDetail { a: join.a.clone(), b: join.b.clone(), pose: join.pose })
        .unwrap();
    let detail = match wait_for(&mut worker, |e| matches!(e, Event::PairDetail(_))) {
        Event::PairDetail(detail) => detail,
        other => unreachable!("{other:?}"),
    };
    assert_eq!(detail.contact.len(), detail.contact_class.len());
    assert!(!detail.contact.is_empty() && !detail.seam.is_empty());
    assert!(0.0 < detail.tight && detail.tight < detail.gap);
    assert!(detail.contact_class.contains(&0), "the accepted pose has tight contact");

    // R §9 over the one unrefined group.
    ask.send(&Request::Refine { decisions: accepted.clone() }).unwrap();
    let refined = assembly(&mut worker);
    assert!(refined.groups[0].refined, "the group R §9 has just walked is refined");
    assert_ne!(refined.poses, one.poses, "and full resolution moved it");

    // The same decision again: the group is the same group, so it keeps R §9's poses (A §8.4).
    ask.send(&Request::Reassemble { decisions: accepted }).unwrap();
    let again = assembly(&mut worker);
    assert!(again.groups[0].refined);
    assert_eq!(again.poses, refined.poses, "the merge gives back what refinement found");

    // Rejected instead: nothing is built, and nothing is refined either.
    ask.send(&Request::Reassemble { decisions: decided(&join.a, &join.b, Verdict::Reject, None) })
        .unwrap();
    let none = assembly(&mut worker);
    assert!(none.joins.is_empty(), "{:?}", none.joins);
    assert!(none.groups.iter().all(|g| g.members.len() == 1 && !g.refined), "{:?}", none.groups);

    // A request the session cannot answer is one request's problem: it says so and stays open.
    let unknown = Request::PairDetail { a: join.a.clone(), b: "notHere".into(), pose: join.pose };
    ask.send(&unknown).unwrap();
    let refused = wait_for(&mut worker, |e| matches!(e, Event::RequestFailed { .. }));
    assert!(matches!(&refused, Event::RequestFailed { message } if message.contains("notHere")));

    ask.send(&Request::Close).unwrap();
    assert!(matches!(
        wait_for(&mut worker, |e| matches!(e, Event::Done { .. })),
        Event::Done { .. }
    ));
    let mut code = None;
    while let Some(next) = worker.next() {
        if let HostEvent::Exited { code: status } = next {
            code = status;
        }
    }
    assert_eq!(code, Some(0), "a session that was closed ended well");
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
