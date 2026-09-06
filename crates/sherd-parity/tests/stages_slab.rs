//! The stage runners against the committed slab dump, and against a deliberately broken copy of
//! it (plan step S4; D §10.2).
//!
//! Two things have to be true of a parity harness, and only the first is usually tested: it must
//! pass on a correct port, **and it must fail on a wrong one**. `tools/compare_fixtures.py` proves
//! the second on the Python side by perturbing one file at a time and checking that the right
//! stage complains; these tests do the same for the Rust side. A harness that cannot be made to
//! fail is not evidence of anything.

use std::path::{Path, PathBuf};

use sherd_parity::FixtureDir;
use sherd_parity::report::Mode;
use sherd_parity::stages::{Collection, Stage};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn slab_dump() -> PathBuf {
    repo_root().join("fixtures/slab/dump")
}

fn slab_input() -> PathBuf {
    repo_root().join("fixtures/slab/input")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sherd-parity-it-{}-{tag}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// Copies the part of the dump the three stages of this build read: the manifest and the
/// per-fragment boundaries.
fn copy_dump(to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    std::fs::copy(slab_dump().join("manifest.json"), to.join("manifest.json")).unwrap();
    let from = slab_dump().join("fragments");
    for fragment in std::fs::read_dir(&from).unwrap() {
        let fragment = fragment.unwrap().path();
        let target = to.join("fragments").join(fragment.file_name().unwrap());
        std::fs::create_dir_all(&target).unwrap();
        for file in std::fs::read_dir(&fragment).unwrap() {
            let file = file.unwrap().path();
            if file.is_file() {
                std::fs::copy(&file, target.join(file.file_name().unwrap())).unwrap();
            }
        }
    }
}

/// Rewrites one JSON field of one fragment's dump.
fn perturb(dump: &Path, fragment: &str, file: &str, key: &str, value: serde_json::Value) {
    let path = dump.join("fragments").join(fragment).join(file);
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(json.get(key).is_some(), "{file} has no key {key}");
    json[key] = value;
    std::fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
}

#[test]
fn every_stage_passes_on_the_slab_in_both_modes() {
    let collection =
        Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).expect("the dump");
    for mode in [Mode::Injected, Mode::Native] {
        let reports = collection.run_all(&Stage::ALL, mode).expect("the stages run");
        assert_eq!(reports.len(), Stage::ALL.len());
        for report in &reports {
            assert_eq!(
                report.status(),
                "PASS",
                "{} {mode}: {:?}",
                report.stage,
                report.failures().map(sherd_parity::Check::line).collect::<Vec<_>>()
            );
            assert!(report.skips.is_empty(), "{} {mode} skipped something", report.stage);
        }
    }
}

#[test]
fn a_perturbed_working_mesh_fails_the_stage_that_measures_it() {
    let dump = scratch("perturbed");
    copy_dump(&dump);

    // `res` moved by 20 %: outside D §10.2's native ±10 %, and outside the injected column's
    // "exact" by a mile.
    let stats = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(dump.join("fragments/pieceA/mesh.stats.json")).unwrap(),
    )
    .unwrap();
    let res = stats["res"].as_f64().unwrap();
    let area = stats["area"].as_f64().unwrap();
    perturb(&dump, "pieceA", "mesh.stats.json", "res", (1.20 * res).into());
    // and the area by 2 %, which the native column allows only 0.5 % of.
    perturb(&dump, "pieceA", "mesh.stats.json", "area", (1.02 * area).into());
    // A fragment the reference called closed, called open.
    perturb(&dump, "pieceB", "mesh.stats.json", "watertight", false.into());

    let collection =
        Collection::open(FixtureDir::new(&dump), Some(&slab_input())).expect("the copy opens");
    for mode in [Mode::Injected, Mode::Native] {
        let report = collection.run(Stage::WorkingMesh, mode).expect("the stage runs");
        assert_eq!(report.status(), "FAIL", "{mode}");
        let failed: Vec<&str> = report.failures().map(|c| c.quantity).collect();
        assert!(failed.contains(&"res"), "{mode}: {failed:?}");
        assert!(failed.contains(&"area"), "{mode}: {failed:?}");
        assert!(failed.contains(&"watertight"), "{mode}: {failed:?}");
    }
    std::fs::remove_dir_all(&dump).ok();
}

#[test]
fn a_perturbed_thickness_fails_by_the_bins_it_moved() {
    let dump = scratch("thickness");
    copy_dump(&dump);
    let t: f64 = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(dump.join("fragments/pieceA/thick.t.json")).unwrap(),
    )
    .unwrap()
    .as_f64()
    .unwrap();
    // Ten per cent: more than three bins of the slab's histogram, so it fails the widened native
    // gate as well as D §10.2's own ±2 %.
    std::fs::write(
        dump.join("fragments/pieceA/thick.t.json"),
        serde_json::to_vec(&(1.10 * t)).unwrap(),
    )
    .unwrap();

    let collection =
        Collection::open(FixtureDir::new(&dump), Some(&slab_input())).expect("the copy opens");
    for mode in [Mode::Injected, Mode::Native] {
        let report = collection.run(Stage::Thickness, mode).expect("the stage runs");
        assert_eq!(report.status(), "FAIL", "{mode}");
        assert!(report.failures().any(|c| c.scope == "pieceA"), "{mode}");
        assert!(
            report.checks.iter().filter(|c| c.scope == "pieceB").all(sherd_parity::Check::passed)
        );
    }
    std::fs::remove_dir_all(&dump).ok();
}

#[test]
fn a_perturbed_load_count_fails_the_load_stage() {
    let dump = scratch("load");
    copy_dump(&dump);
    let counts = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(dump.join("fragments/pieceB/load.n_orig.json")).unwrap(),
    )
    .unwrap();
    let faces = counts["n_orig_faces"].as_u64().unwrap();
    perturb(&dump, "pieceB", "load.n_orig.json", "n_orig_faces", (faces + 7).into());

    let collection =
        Collection::open(FixtureDir::new(&dump), Some(&slab_input())).expect("the copy opens");
    let report = collection.run(Stage::Load, Mode::Native).expect("the stage runs");
    assert_eq!(report.status(), "FAIL");
    let failed: Vec<&str> = report.failures().map(|c| c.quantity).collect();
    assert_eq!(failed, ["n_orig_faces"], "seven triangles is a failure, and only that one");
    std::fs::remove_dir_all(&dump).ok();
}

#[test]
fn without_the_input_directory_native_mode_skips_and_injected_mode_does_not() {
    let collection = Collection::open(FixtureDir::new(slab_dump()), None).expect("the dump");
    let native = collection.run_all(&Stage::ALL, Mode::Native).unwrap();
    assert_eq!(
        native.iter().map(sherd_parity::StageReport::status).collect::<Vec<_>>(),
        ["SKIP"; Stage::ALL.len()]
    );
    let injected = collection.run_all(&Stage::ALL, Mode::Injected).unwrap();
    // `load` needs the file in both modes — its input *is* the file. The others run off the dump
    // alone.
    assert_eq!(injected[0].status(), "SKIP");
    assert!(
        injected[1..].iter().all(|r| r.status() == "PASS"),
        "{:?}",
        injected.iter().map(|r| (r.stage, r.status())).collect::<Vec<_>>()
    );
}
