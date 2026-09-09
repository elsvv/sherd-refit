//! The fragment cache on a real mesh (R §3.7, D §4.2; plan step S4).
//!
//! The unit tests in `fragment::cache` build a tetrahedron by hand. This one runs the whole of
//! R §3.1–3.3 on `fixtures/slab/input/pieceA.ply` — the one mesh every checkout has — and asks the
//! question the cache exists to answer: **is a warm run the same run as a cold one?** Not "close
//! to": the same, in every field the later stages read, bit for bit. If it is not, then a rerun of
//! a collection would match differently from its first run, and no parity gate downstream would
//! ever be able to say why.

use std::path::{Path, PathBuf};

use sherd_core::fragment::Fragment;
use sherd_core::fragment::breakline::BrkParams;
use sherd_core::fragment::cache;
use sherd_core::fragment::samples::SampleParams;

fn slab_piece(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/slab/input")
        .join(format!("{name}.ply"))
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sherd-cache-it-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

const TARGET_FACES: usize = 200_000;

#[test]
fn a_warm_run_is_the_same_run_as_a_cold_one() {
    let out = scratch("warm");
    let source = slab_piece("pieceA");

    let (cold, from_cache) = Fragment::load_or_build(
        &source,
        TARGET_FACES,
        "pieceA",
        Some(&cache::cache_path(&out, "pieceA")),
        0,
    )
    .expect("the slab preprocesses");
    assert!(!from_cache, "the first run has nothing to read");

    let (warm, from_cache) = Fragment::load_or_build(
        &source,
        TARGET_FACES,
        "pieceA",
        Some(&cache::cache_path(&out, "pieceA")),
        0,
    )
    .expect("the cache is readable");
    assert!(from_cache, "the second run must hit the cache it just wrote");

    assert_eq!(warm.name, cold.name);
    assert_eq!(warm.mesh.v, cold.mesh.v, "vertices");
    assert_eq!(warm.mesh.f, cold.mesh.f, "triangles");
    // The three per-face arrays are not stored; both paths derive them from the stored `V`, so
    // they have to come out identical (`WorkingMesh::from_parts`).
    assert_eq!(warm.mesh.face_normals, cold.mesh.face_normals, "face normals");
    assert_eq!(warm.mesh.face_areas, cold.mesh.face_areas, "face areas");
    assert_eq!(warm.mesh.face_centroids, cold.mesh.face_centroids, "face centroids");
    assert_eq!(warm.mesh.res.to_bits(), cold.mesh.res.to_bits(), "res");
    assert_eq!(warm.thick.to_bits(), cold.thick.to_bits(), "t");
    assert_eq!(warm.thick_mode.to_bits(), cold.thick_mode.to_bits(), "thick_mode");
    assert_eq!(warm.area0.to_bits(), cold.area0.to_bits(), "area0");
    assert_eq!(warm.face_budget, cold.face_budget);
    assert_eq!(warm.target_faces, cold.target_faces);
    assert_eq!((warm.watertight, warm.n_boundary), (cold.watertight, cold.n_boundary));
    assert_eq!(
        (warm.n_orig_vertices, warm.n_orig_faces),
        (cold.n_orig_vertices, cold.n_orig_faces)
    );
    assert_eq!(warm.labels, cold.labels, "labels");
    // The breaklines are stored, not derived, so this is the file's own round trip (R §3.5.3–3.5.5).
    assert!(!cold.brk.is_empty(), "the slab has a break to find");
    assert_eq!(warm.brk, cold.brk, "breakline points, frames, subset and parameters");
    assert_eq!(warm.brk.params, BrkParams::at(cold.thick));
    // And the sampled arrays, which are stored for the same reason (R §3.5.1–3.5.2, §3.5.6).
    assert!(cold.samples.has_fracture() && cold.samples.n_margin() > 0);
    assert_eq!(warm.samples, cold.samples, "S, sp, Pf, fp, margin_idx and their parameters");
    assert_eq!(warm.samples.params, SampleParams::at(cold.thick, 0));
    assert_eq!(warm.area.to_bits(), cold.area.to_bits(), "area");
    assert_eq!(warm.frac_area.to_bits(), cold.frac_area.to_bits(), "frac_area");

    std::fs::remove_dir_all(&out).ok();
}

/// R §3.7's rule again for the *sampled* half: a cache whose `md_params` are not this run's has
/// `S`, `sp`, `Pf`, `fp` and `margin_idx` recomputed, and the mesh, the labels and the breaklines
/// still come off the disk.
#[test]
fn a_cache_built_with_other_sampling_parameters_has_the_arrays_recomputed() {
    let out = scratch("mdparams");
    let source = slab_piece("pieceA");
    let path = cache::cache_path(&out, "pieceA");

    let (mut fragment, _) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 0).expect("cold");
    let wanted = fragment.samples.clone();
    assert_eq!(wanted.params, SampleParams::at(fragment.thick, 0));

    // A cache whose samples were drawn at another count, with the arrays to match.
    fragment.samples.params = SampleParams { surface_points: 500, ..wanted.params };
    fragment.samples.s.truncate(500);
    fragment.samples.sp.truncate(500);
    fragment.samples.margin_idx.retain(|&i| i < 500);
    cache::write(&fragment, &path).expect("the doctored cache is written");

    let (back, from_cache) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 0).expect("warm");
    assert!(from_cache, "the mesh, the labels and the breaklines still came from the cache");
    assert_eq!(back.samples, wanted, "the arrays were redrawn at this run's parameters");
    assert_eq!(cache::read(&path).expect("the cache reads").samples, wanted);

    std::fs::remove_dir_all(&out).ok();
}

/// R §3.7's other half: a cache that is valid but was built with other match-array parameters has
/// those arrays recomputed rather than the whole fragment thrown away.
///
/// In this build the knobs are constants, so the only way to reach the branch is to write a cache
/// that claims other ones — which is exactly what a cache from another build would be.
#[test]
fn a_cache_built_with_other_breakline_parameters_has_them_recomputed() {
    let out = scratch("brkparams");
    let source = slab_piece("pieceA");
    let path = cache::cache_path(&out, "pieceA");

    let (mut fragment, _) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 0).expect("cold");
    let wanted = fragment.brk.clone();
    assert_eq!(wanted.params, BrkParams::at(fragment.thick));

    // A cache whose breaklines were built at another outer radius: the arrays are wrong for this
    // run, and their `brk_params` say so.
    fragment.brk.params = BrkParams { macro_outer: 0.35, ..wanted.params };
    fragment.brk.sub.clear();
    cache::write(&fragment, &path).expect("the doctored cache is written");

    let (back, from_cache) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 0).expect("warm");
    assert!(from_cache, "the mesh and the labels still came from the cache");
    assert_eq!(back.brk, wanted, "the breaklines were rebuilt at this run's parameters");
    // And the corrected arrays were written back, so the next run does no work at all.
    assert_eq!(cache::read(&path).expect("the cache reads").brk, wanted);

    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn the_cache_of_a_real_fragment_is_reproducible_and_self_describing() {
    let out = scratch("bytes");
    let source = slab_piece("pieceB");
    let path = cache::cache_path(&out, "pieceB");

    let (fragment, _) = Fragment::load_or_build(&source, TARGET_FACES, "pieceB", Some(&path), 0)
        .expect("the slab preprocesses");
    let first = std::fs::read(&path).expect("the cache was written");

    // Writing it again from the same fragment must produce the same bytes — there is no
    // timestamp in the file and no hash-map order in its header.
    cache::write(&fragment, &path).expect("rewritten");
    assert_eq!(std::fs::read(&path).unwrap(), first, "two writes must agree byte for byte");

    let meta = cache::read_meta(&path).expect("the metadata parses");
    assert_eq!(meta.format, cache::FORMAT);
    assert_eq!(meta.cache_version, sherd_core::CACHE_VERSION);
    assert_eq!(meta.algo_ref, sherd_core::ALGO_REF);
    assert_eq!(meta.name, "pieceB");
    assert_eq!(meta.target_faces, 200_000);
    assert!(meta.source_path.ends_with("pieceB.ply") && meta.source_path.starts_with('/'));
    assert_eq!(meta.res.to_bits(), f64::from(fragment.mesh.res).to_bits());
    assert!(meta.valid_for(&source, 200_000, "pieceB"));

    // A cache built at another face cap does not describe this run.
    assert!(cache::load_valid(&path, &source, 50_000, "pieceB").is_none());

    std::fs::remove_dir_all(&out).ok();
}

/// Audit §B.3: the two BVHs are released once their readers are done, and the release is a reset
/// rather than a poisoning.
///
/// What it has to guarantee is that dropping the trees cannot change an answer. It cannot, because
/// a fragment asked for a tree after the release simply builds it again — same mesh, same faces,
/// same tree — and that is what this asserts, on the same closest-point query on both sides.
#[test]
fn releasing_the_scenes_frees_them_and_they_come_back() {
    let fragment = {
        let mut fragment =
            Fragment::from_mesh_file(slab_piece("pieceA"), TARGET_FACES, 0).expect("pieceA");
        let query = [1.0_f32, 2.0, 3.0];
        let before = (
            fragment.surface_scene().expect("a whole-mesh tree").closest_face(query),
            fragment.fracture_scene().expect("a fracture tree").closest_face(query),
        );
        fragment.release_scenes();
        fragment.release_scenes(); // twice: releasing what is not there is not an error
        let after = (
            fragment
                .surface_scene()
                .expect("the whole-mesh tree is rebuilt on demand")
                .closest_face(query),
            fragment
                .fracture_scene()
                .expect("the fracture tree is rebuilt on demand")
                .closest_face(query),
        );
        assert_eq!(before.0, after.0, "the whole-mesh tree answers the same query");
        assert_eq!(before.1, after.1, "the fracture tree answers the same query");
        fragment
    };
    assert!(fragment.n_faces() > 0);
}

/// R §10's seed reaches R §3.5's three samplers, and the cache knows which seed drew its arrays.
///
/// `--seed` (task H3) exists because R §13's per-set rows are a *spread* the reference produces
/// under `Params(seed = 0..4)` and no CLI on either side could ask for it. What the flag has to
/// do is exactly this: move the sampled arrays and nothing else — the working mesh, the labels
/// and the breaklines are functions of the file — and make a cache drawn at another seed rebuild
/// those arrays alone, which is R §3.7's rule.
#[test]
fn the_seed_moves_the_sampled_arrays_and_nothing_else() {
    let source = slab_piece("pieceA");
    let zero = Fragment::from_mesh_file(&source, TARGET_FACES, 0).expect("seed 0");
    let one = Fragment::from_mesh_file(&source, TARGET_FACES, 1).expect("seed 1");

    assert_eq!(one.mesh.v, zero.mesh.v, "R §3.3's working mesh draws nothing");
    assert_eq!(one.mesh.f, zero.mesh.f);
    assert_eq!(one.labels, zero.labels, "R §3.4's segmentation draws nothing");
    assert_eq!(one.brk.p, zero.brk.p, "R §3.5.3-3.5.5's breaklines draw nothing");
    assert_eq!(one.thick.to_bits(), zero.thick.to_bits(), "R §3.2's estimator draws nothing (T1)");

    assert_eq!(zero.samples.params.seed, 0);
    assert_eq!(one.samples.params.seed, 1);
    assert_ne!(one.samples.s, zero.samples.s, "R §3.5.1's surface samples");
    assert_ne!(one.samples.pf, zero.samples.pf, "R §3.5.2's fracture samples");
    assert_eq!(one.samples.s.len(), zero.samples.s.len(), "the same counts, other points");

    // And a cache written at one seed has those arrays — and only those — redrawn for the other.
    let out = scratch("seed");
    let path = cache::cache_path(&out, "pieceA");
    std::fs::remove_file(&path).ok();
    let (cold, from_cache) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 0).expect("cold");
    assert!(!from_cache);
    assert_eq!(cold.samples.params.seed, 0);
    let (warm, from_cache) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 1).expect("warm");
    assert!(from_cache, "the mesh, the labels and the breaklines still came from the cache");
    assert_eq!(warm.mesh.v, cold.mesh.v);
    assert_eq!(warm.brk.p, cold.brk.p);
    assert_eq!(warm.samples.params.seed, 1, "the arrays were redrawn at this run's seed");
    assert_eq!(warm.samples.s, one.samples.s, "and they are the seed's own arrays");
}

/// Audit §D.2's object features are part of R §3, so a warm run reads them back and a run at
/// another seed redraws them (task O1, `cache_version` 6).
///
/// The two halves are different quantities and the test says so. Every *geometric* feature is
/// measured on R §3.5's draw, so it moves with the seed exactly as the samples do; the *colour* is
/// a property of the file's vertices and moves with nothing — which is why the cache carries it
/// through a redraw instead of re-reading a full-resolution scan to learn it again.
#[test]
fn the_object_features_are_cached_with_the_fragment_and_redrawn_with_its_samples() {
    let out = scratch("features");
    let source = slab_piece("pieceA");
    let path = cache::cache_path(&out, "pieceA");
    std::fs::remove_file(&path).ok();

    let (cold, from_cache) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 0).expect("cold");
    assert!(!from_cache);
    let cold_f = cold.features.clone().expect("a fragment built from its file has a table");
    assert_eq!(cold_f.name, "pieceA");
    assert_eq!(cold_f.thick.to_bits(), cold.thick.to_bits(), "the wall is R §3.2's own");
    assert!(cold_f.shell_radius.is_some(), "the slab's shell fits");
    assert!(cold_f.frac_rough.is_some(), "and its fracture face has a roughness");

    let (warm, from_cache) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 0).expect("warm");
    assert!(from_cache, "the second run reads the cache it just wrote");
    assert_eq!(warm.features, cold.features, "a warm run is the same run, features and all");

    // Another seed redraws R §3.5, so the fits move with it; the colour fields do not.
    let (other, _) =
        Fragment::load_or_build(&source, TARGET_FACES, "pieceA", Some(&path), 1).expect("seed 1");
    let other_f = other.features.expect("still a table");
    assert_eq!(other_f.thick.to_bits(), cold_f.thick.to_bits(), "R §3.2 draws nothing");
    assert_ne!(other_f.shell_radius, cold_f.shell_radius, "the sphere fit is over R §3.5's draw");
    assert_eq!(other_f.colour_points, cold_f.colour_points, "the fabric is a fact about the file");
    assert_eq!(other_f.lab_mean, cold_f.lab_mean);

    std::fs::remove_dir_all(&out).ok();
}
