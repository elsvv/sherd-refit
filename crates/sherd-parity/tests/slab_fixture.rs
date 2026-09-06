//! The committed slab fixture (`fixtures/slab/`, D §10.4 layer 2) read from Rust.
//!
//! It is the only fixture in the repository — the rest are too large and, for the SfS++ sets,
//! not redistributable — so it is the one place where the manifest reader is exercised against a
//! real dump on every machine and in CI. That it parses, that its checksums hold and that its
//! parameters are the defaults of R §1.1 is what makes `Params` a checked port rather than a
//! transcription.

use sherd_core::Params;
use sherd_parity::FixtureDir;

fn slab() -> FixtureDir {
    FixtureDir::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/dump"),
    )
}

#[test]
fn the_manifest_describes_the_slab_pair() {
    let manifest = slab().load_manifest().expect("the committed slab manifest parses");

    assert_eq!(manifest.fragment_names(), ["pieceA", "pieceB"]);
    assert_eq!(manifest.collection.files, ["pieceA.ply", "pieceB.ply"]);
    assert_eq!(manifest.pairs.pairs, [["pieceA".to_owned(), "pieceB".to_owned()]]);
    assert!(manifest.pairs.skipped.is_empty());
    assert!(manifest.pairs.thickness_median > 0.0);
    assert_eq!(manifest.collection.keep_per_pair, 5);
    assert_eq!(manifest.collection.target_faces, 200_000);
    assert!(manifest.collection.refine);
    assert!(!manifest.commit.is_empty());
    assert!(!manifest.dirty, "a fixture is dumped from a clean tree");
    assert_eq!(manifest.open3d, "0.19.0");

    assert!(manifest.files.contains_key("fragments/pieceA/mesh.V.npy"));
    assert!(manifest.files.contains_key("outputs/transforms.json"));
    assert!(manifest.total_size() > 0);
}

#[test]
fn the_fixture_was_dumped_with_the_defaults_of_the_algorithm_reference() {
    let manifest = slab().load_manifest().expect("the committed slab manifest parses");
    assert!(
        manifest.uses_default_params(),
        "Params::default() must be the Python's defaults; the dump says {:?}",
        manifest.collection.params
    );
    assert_eq!(manifest.collection.params, Params::default());
}

#[test]
fn every_committed_file_still_hashes_to_what_the_manifest_says() {
    let dir = slab();
    let manifest = dir.load_manifest().expect("the committed slab manifest parses");
    let bad = dir.verify_checksums().expect("the fixture can be hashed");
    assert!(bad.is_empty(), "{} of {} files differ: {bad:?}", bad.len(), manifest.files.len());
}
