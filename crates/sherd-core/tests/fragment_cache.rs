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
use sherd_core::fragment::cache;

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
    )
    .expect("the slab preprocesses");
    assert!(!from_cache, "the first run has nothing to read");

    let (warm, from_cache) = Fragment::load_or_build(
        &source,
        TARGET_FACES,
        "pieceA",
        Some(&cache::cache_path(&out, "pieceA")),
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

    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn the_cache_of_a_real_fragment_is_reproducible_and_self_describing() {
    let out = scratch("bytes");
    let source = slab_piece("pieceB");
    let path = cache::cache_path(&out, "pieceB");

    let (fragment, _) = Fragment::load_or_build(&source, TARGET_FACES, "pieceB", Some(&path))
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
