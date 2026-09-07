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

/// Copies the part of the dump the stages of this build read: the manifest, the per-fragment
/// boundaries (including the `md_t` rebuild a pair asks for) and the pair directories.
fn copy_dump(to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    std::fs::copy(slab_dump().join("manifest.json"), to.join("manifest.json")).unwrap();
    for group in ["fragments", "pairs", "assembly", "outputs", "refine"] {
        let from = slab_dump().join(group);
        if from.is_dir() {
            copy_tree(&from, &to.join(group));
        }
    }
}

/// Copies a directory tree, files and subdirectories alike.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        let target = to.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_tree(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

/// Writes a one-dimensional `.npy` file — the perturbation tests below need to put a *changed*
/// array back into a copied dump, and the reader under test is `npyz`, so writing the bytes by
/// hand keeps the two sides independent.
fn write_npy(path: &Path, descr: &str, count: usize, data: &[u8]) {
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({count},), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    std::fs::write(path, out).unwrap();
}

/// Replaces a `float64` array of a pair directory.
fn write_f64(dump: &Path, file: &str, values: &[f64]) {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy(&dump.join("pairs/pieceA__pieceB").join(file), "<f8", values.len(), &data);
}

/// Replaces an `int64` array of a pair directory.
fn write_i64(dump: &Path, file: &str, values: &[i64]) {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy(&dump.join("pairs/pieceA__pieceB").join(file), "<i8", values.len(), &data);
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
            // D §10.2 gives `coarse`, `nms` and the two refinement stages no native column: all
            // four are functions of a draw the port makes with its own generator (PMC-9), so
            // natively there is nothing to compare and the stage says so instead of inventing a
            // tolerance.
            let no_native_column = mode == Mode::Native
                && matches!(report.stage, "coarse" | "nms" | "stage1" | "stage2" | "verify");
            assert_eq!(
                report.status(),
                if no_native_column { "SKIP" } else { "PASS" },
                "{} {mode}: {:?}",
                report.stage,
                report.failures().map(sherd_parity::Check::line).collect::<Vec<_>>()
            );
            assert_eq!(
                report.skips.is_empty(),
                !no_native_column,
                "{} {mode} skipped something",
                report.stage
            );
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

/// R §8's row, made to fail by exactly the thing it measures — the standard every other stage of
/// this file is held to.
///
/// Three perturbations of the *dump's own answer*, one per column: a group that is not the group
/// the port grows, a used join the reference did not use, and a rejection where there was none.
#[test]
fn a_perturbed_assembly_fails_the_stage_that_measures_it() {
    let dump = scratch("assembly");
    copy_dump(&dump);
    let assembly = dump.join("assembly");

    // The slab is one join and one group of two. Split the group.
    std::fs::write(assembly.join("groups.json"), r#"[["pieceA"], ["pieceB"]]"#).unwrap();
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let report = collection.run(Stage::Assembly, Mode::Injected).unwrap();
    let failed: Vec<&'static str> = report.failures().map(|c| c.quantity).collect();
    assert_eq!(
        failed,
        ["groups", "recentre"],
        "a different grouping fails the groups row, and R §8.2's row with it — `recentre` \
         translates the dump's own groups, so a wrong grouping moves the wrong fragments"
    );

    // Put the group back and take the join away.
    std::fs::write(assembly.join("groups.json"), r#"[["pieceA", "pieceB"]]"#).unwrap();
    std::fs::write(assembly.join("used.json"), "[]").unwrap();
    let report = collection.run(Stage::Assembly, Mode::Injected).unwrap();
    let failed: Vec<&'static str> = report.failures().map(|c| c.quantity).collect();
    assert_eq!(failed, ["used"]);

    // Put the join back and invent a rejection.
    let used = std::fs::read_to_string(slab_dump().join("assembly/used.json")).unwrap();
    std::fs::write(assembly.join("used.json"), &used).unwrap();
    let invented = used.replacen('{', r#"{"reason": "penetrates pieceA (0.500)","#, 1);
    std::fs::write(assembly.join("rejected.json"), invented).unwrap();
    let report = collection.run(Stage::Assembly, Mode::Injected).unwrap();
    let failed: Vec<&'static str> = report.failures().map(|c| c.quantity).collect();
    assert_eq!(failed, ["rejected"]);
}

/// A dump whose assembly boundary is not there is skipped, not compared against nothing.
#[test]
fn an_assembly_without_its_own_samples_skips() {
    let dump = scratch("assembly-min");
    copy_dump(&dump);
    std::fs::remove_file(dump.join("assembly/md_t_median.json")).unwrap();
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let report = collection.run(Stage::Assembly, Mode::Injected).unwrap();
    assert_eq!(report.status(), "SKIP");
    assert!(report.checks.is_empty());
    assert!(report.skips[0].reason.contains("md_t_median"), "{}", report.skips[0].reason);
}

#[test]
fn without_the_input_directory_native_mode_skips_and_injected_mode_does_not() {
    let collection = Collection::open(FixtureDir::new(slab_dump()), None).expect("the dump");
    // Every native column starts from the file, with one exception: `outputs` natively compares
    // the port's own renderer and JSON writer against themselves and against the dump's own
    // views, and needs no mesh at all.
    let native = collection.run_all(&Stage::ALL, Mode::Native).unwrap();
    for report in &native {
        let expected = if report.stage == "outputs" { "PASS" } else { "SKIP" };
        assert_eq!(report.status(), expected, "{}", report.stage);
    }
    let injected = collection.run_all(&Stage::ALL, Mode::Injected).unwrap();
    // Two stages need the file in *both* modes, because the file is their input: `load`, whose
    // subject it is, and `refine`, whose fracture cloud is built on the original mesh's own
    // vertices and which no dump carries (R §9). The rest run off the dump alone.
    let needs_the_file = ["load", "refine"];
    for report in &injected {
        let expected = if needs_the_file.contains(&report.stage) { "SKIP" } else { "PASS" };
        assert_eq!(report.status(), expected, "{}", report.stage);
    }
}

/// R §9's row, made to fail by exactly the thing it measures: a refined pose the reference did not
/// produce, and a vertex selection that is not the reference's.
#[test]
fn a_perturbed_refine_fails_the_stage_that_measures_it() {
    let dump = scratch("refine-perturbed");
    copy_dump(&dump);
    let open = |dir: &Path, name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };

    // A pose 0.4 t out of place: the row's tolerance is 0.02 t.
    let mut poses = open(&dump, "refine/poses_final.json");
    let thickness = 30.154_368_401_f64;
    poses["pieceB"][0][3] =
        serde_json::json!(poses["pieceB"][0][3].as_f64().unwrap() + 0.4 * thickness);
    std::fs::write(dump.join("refine/poses_final.json"), serde_json::to_vec(&poses).unwrap())
        .unwrap();

    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let report = collection.run(Stage::Refine, Mode::Injected).unwrap();
    let failed: Vec<&str> = report.failures().map(|c| c.quantity).collect();
    assert!(failed.contains(&"pose trans"), "{failed:?}");
    assert!(!failed.contains(&"idx"), "the selection is untouched: {failed:?}");

    // And a selection that is one vertex short of the reference's.
    let dump = scratch("refine-idx");
    copy_dump(&dump);
    let idx = sherd_parity::npy::read_indices(dump.join("refine/pieceA.idx.npy")).unwrap();
    let shortened: Vec<u8> = idx[1..].iter().flat_map(|&i| i64::from(i).to_le_bytes()).collect();
    write_npy(&dump.join("refine/pieceA.idx.npy"), "<i8", idx.len() - 1, &shortened);
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let report = collection.run(Stage::Refine, Mode::Injected).unwrap();
    let failed: Vec<&str> = report.failures().map(|c| c.quantity).collect();
    assert!(failed.contains(&"idx"), "{failed:?}");
}

/// R §11's row, made to fail by each of the three things it measures: a pose, a mesh and a pixel.
#[test]
fn a_perturbed_output_fails_the_row_that_measures_it() {
    // A `transforms.json` pose the recentring does not produce.
    let dump = scratch("outputs-pose");
    copy_dump(&dump);
    let mut transforms: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dump.join("outputs/transforms.json")).unwrap())
            .unwrap();
    let before = transforms["fragments"]["pieceB"]["matrix"][2][3].as_f64().unwrap();
    transforms["fragments"]["pieceB"]["matrix"][2][3] = serde_json::json!(before + 1e-3);
    std::fs::write(dump.join("outputs/transforms.json"), serde_json::to_vec(&transforms).unwrap())
        .unwrap();
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let failed: Vec<&str> = collection
        .run(Stage::Outputs, Mode::Injected)
        .unwrap()
        .failures()
        .map(|c| c.quantity)
        .collect();
    assert!(failed.contains(&"transforms pose"), "{failed:?}");

    // A placed mesh whose hash is not the one the port produces.
    let dump = scratch("outputs-ply");
    copy_dump(&dump);
    let mut index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dump.join("outputs/placed.sha256.json")).unwrap())
            .unwrap();
    index["files"]["placed/pieceA.ply"]["sha256"] = serde_json::json!("0".repeat(64));
    std::fs::write(dump.join("outputs/placed.sha256.json"), serde_json::to_vec(&index).unwrap())
        .unwrap();
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let failed: Vec<&str> = collection
        .run(Stage::Outputs, Mode::Injected)
        .unwrap()
        .failures()
        .map(|c| c.quantity)
        .collect();
    assert!(failed.contains(&"placed ply"), "{failed:?}");
    assert!(!failed.contains(&"placed shape"), "the size and counts are untouched: {failed:?}");

    // A preview the port did not render: the *labelled* image put where the unlabelled one goes,
    // which differs from the port's render by exactly the caption.
    let dump = scratch("outputs-preview");
    copy_dump(&dump);
    std::fs::copy(dump.join("outputs/preview_0.png"), dump.join("outputs/preview_0.nolabel.png"))
        .unwrap();
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let failed: Vec<&str> = collection
        .run(Stage::Outputs, Mode::Injected)
        .unwrap()
        .failures()
        .map(|c| c.quantity)
        .collect();
    assert!(failed.contains(&"preview px"), "{failed:?}");
}

/// A dump with no R §9 boundary and no R §11.4/11.5 files is skipped, not compared against
/// nothing.
#[test]
fn outputs_and_refine_skip_what_the_dump_does_not_carry() {
    let dump = scratch("refine-min");
    copy_dump(&dump);
    std::fs::remove_dir_all(dump.join("refine")).unwrap();
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let report = collection.run(Stage::Refine, Mode::Injected).unwrap();
    assert_eq!(report.status(), "SKIP");
    assert!(report.checks.is_empty());
    assert!(report.skips[0].reason.contains("no refine/"), "{}", report.skips[0].reason);

    // The outputs stage still compares the two JSON files, and says what it could not compare.
    let dump = scratch("outputs-min");
    copy_dump(&dump);
    std::fs::remove_file(dump.join("outputs/preview_index.json")).unwrap();
    std::fs::remove_file(dump.join("outputs/placed.sha256.json")).unwrap();
    let collection = Collection::open(FixtureDir::new(&dump), Some(&slab_input())).unwrap();
    let report = collection.run(Stage::Outputs, Mode::Injected).unwrap();
    assert_eq!(report.status(), "PASS");
    let reasons: Vec<&str> = report.skips.iter().map(|s| s.reason.as_str()).collect();
    assert_eq!(reasons.len(), 2, "{reasons:?}");
    assert!(reasons.iter().any(|r| r.contains("placed.sha256.json")), "{reasons:?}");
    assert!(reasons.iter().any(|r| r.contains("preview_index.json")), "{reasons:?}");
    assert!(report.checks.iter().any(|c| c.quantity == "report candidates"));
}

/// The three pair stages, each made to fail by exactly the thing it measures — the same standard
/// `tools/compare_fixtures.py` is held to on the Python side.
#[test]
fn a_perturbed_pair_fails_the_stage_that_measures_it() {
    let dump = scratch("pairs");
    copy_dump(&dump);
    let pair = dump.join("pairs/pieceA__pieceB");

    // R §1.2 resolved differently: the `scales` row of the hypotheses stage is exact.
    let mut scales: serde_json::Value =
        serde_json::from_slice(&std::fs::read(pair.join("scales.json")).unwrap()).unwrap();
    scales["coarse"] = (scales["coarse"].as_f64().unwrap() * 1.05).into();
    std::fs::write(pair.join("scales.json"), serde_json::to_vec_pretty(&scales).unwrap()).unwrap();

    // Five frame pairs dropped off the end of the hypothesis set.
    let pa = sherd_parity::npy::read_indices(pair.join("hyp.pa.npy")).unwrap();
    let pb = sherd_parity::npy::read_indices(pair.join("hyp.pb.npy")).unwrap();
    let keep = pa.len() - 5;
    let cut = |v: &[u32]| v[..keep].iter().map(|&x| i64::from(x)).collect::<Vec<i64>>();
    write_i64(&dump, "hyp.pa.npy", &cut(&pa));
    write_i64(&dump, "hyp.pb.npy", &cut(&pb));

    // One coarse score raised by 0.05 — three probe points, past D §10.2's one.
    let mut cs = sherd_parity::npy::read_f64(pair.join("coarse.cs.npy")).unwrap();
    cs.truncate(keep);
    cs[7] += 0.05;
    write_f64(&dump, "coarse.cs.npy", &cs);

    // And one kept hypothesis replaced by another.
    let mut kept = sherd_parity::npy::read_indices(pair.join("nms1.kept.npy"))
        .unwrap()
        .iter()
        .map(|&x| i64::from(x))
        .collect::<Vec<i64>>();
    kept[3] += 1;
    write_i64(&dump, "nms1.kept.npy", &kept);

    let collection =
        Collection::open(FixtureDir::new(&dump), Some(&slab_input())).expect("the copy opens");

    let report = collection.run(Stage::Hypotheses, Mode::Injected).unwrap();
    let failed: Vec<&str> = report.failures().map(|c| c.quantity).collect();
    assert_eq!(report.status(), "FAIL");
    assert!(failed.contains(&"scales"), "{failed:?}");
    assert!(failed.contains(&"n_hyp") && failed.contains(&"pairs"), "{failed:?}");

    let report = collection.run(Stage::Coarse, Mode::Injected).unwrap();
    let failed: Vec<&str> = report.failures().map(|c| c.quantity).collect();
    assert_eq!(report.status(), "FAIL");
    assert!(failed.contains(&"cs") && failed.contains(&"cs exact"), "{failed:?}");

    let report = collection.run(Stage::Nms, Mode::Injected).unwrap();
    let failed: Vec<&str> = report.failures().map(|c| c.quantity).collect();
    assert_eq!(report.status(), "FAIL");
    assert!(failed.contains(&"kept"), "{failed:?}");
    std::fs::remove_dir_all(&dump).ok();
}

/// A dump written before task C1 has no `nms1.order`, and the NMS stage refuses to compare rather
/// than walking its own order and calling the difference a failure (PMC-6).
#[test]
fn without_the_walk_order_the_nms_stage_skips() {
    let dump = scratch("no-order");
    copy_dump(&dump);
    std::fs::remove_file(dump.join("pairs/pieceA__pieceB/nms1.order.npy")).unwrap();

    let collection = Collection::open(FixtureDir::new(&dump), None).expect("the copy opens");
    let report = collection.run(Stage::Nms, Mode::Injected).unwrap();
    assert_eq!(report.status(), "SKIP");
    assert!(report.checks.is_empty());
    assert_eq!(report.skips.len(), 1);
    assert!(report.skips[0].reason.contains("nms1.order"), "{}", report.skips[0].reason);
    std::fs::remove_dir_all(&dump).ok();
}
