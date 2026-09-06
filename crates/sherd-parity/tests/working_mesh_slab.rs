//! The working-mesh stage (R §3.1–3.3) against the committed slab fixture (plan step S3).
//!
//! `fixtures/slab/dump/` is one run of the Python reference at commit `9d4b9d3`, dumped by
//! `tools/dump_fixtures.py` (plan step P0). This test uses it in both of D §10.2's modes:
//!
//! * **injected** — each Rust function is fed the Python's own arrays, so the comparison is exact
//!   where the reference is deterministic: `ΣA0`, the face budget, `t`, `thick_mode`, `res`,
//!   `area`, `watertight` and `n_boundary` must come out *bit-identical*, and Taubin, which is
//!   only summation-order-different from Open3D's, to within a hair of it;
//! * **native** — `Fragment::from_mesh_file` runs on `fixtures/slab/input/*.ply` and the stage
//!   outputs are compared with D §10.2's native tolerances. The decimator and the sampler are
//!   deliberately not the reference's (PMC-2, PMC-9), so this half is statistical.
//!
//! The wider run over all seven benchmark collections lives in
//! `docs/superpowers/notes/2026-09-06-s3-working-mesh.md`; this test is the part of it that can be
//! committed and run anywhere.

#![allow(
    clippy::float_cmp,
    reason = "the injected comparisons are exact on purpose: the reference's own float, bit for bit"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture counts are small integers carried through JSON as u64"
)]

use std::path::{Path, PathBuf};

use sherd_core::fragment::Fragment;
use sherd_core::fragment::thickness::{self, RayHits};
use sherd_core::mesh::Mesh;
use sherd_core::mesh::adjacency::closed_enough;
use sherd_core::mesh::decimate::face_budget;
use sherd_core::mesh::geometry::{face_geometry, median_edge};
use sherd_core::mesh::taubin::taubin;

/// The `--target-faces` the fixture was dumped with (`fixtures/slab/dump/collection.json`).
const TARGET_FACES: usize = 200_000;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

// `CARGO_MANIFEST_DIR` is `crates/sherd-parity`, so the repository root is two levels up — the
// same rule `tests/slab_fixture.rs` uses.

fn fragment_dir(name: &str) -> PathBuf {
    repo_root().join("fixtures/slab/dump/fragments").join(name)
}

/// One `.npy` array as a flat vector.
fn npy<T: npyz::Deserialize>(path: &Path) -> Vec<T> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    npyz::NpyFile::new(std::io::BufReader::new(file))
        .expect("a .npy header")
        .into_vec::<T>()
        .expect("the array's dtype")
}

fn vertices(path: &Path) -> Vec<[f64; 3]> {
    npy::<f64>(path).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn triangles(path: &Path) -> Vec<[u32; 3]> {
    npy::<i64>(path)
        .chunks_exact(3)
        .map(|c| {
            [
                u32::try_from(c[0]).unwrap(),
                u32::try_from(c[1]).unwrap(),
                u32::try_from(c[2]).unwrap(),
            ]
        })
        .collect()
}

fn json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("valid JSON")
}

/// The two fragments of the slab pair.
const PIECES: [&str; 2] = ["pieceA", "pieceB"];

#[test]
fn injected_geometry_reproduces_the_reference_exactly() {
    for name in PIECES {
        let d = fragment_dir(name);
        let v = vertices(&d.join("mesh.V.npy"));
        let f = triangles(&d.join("mesh.F.npy"));
        let stats = json(&d.join("mesh.stats.json"));

        let geom = face_geometry(&v, &f);
        let res = median_edge(&v, &f);
        let (watertight, n_boundary) = closed_enough(&f);

        assert_eq!(f.len(), stats["faces"].as_u64().unwrap() as usize, "{name}: faces");
        assert_eq!(v.len(), stats["vertices"].as_u64().unwrap() as usize, "{name}: vertices");
        // Bit-identical, not merely close: the reference's `res` is a median over the same unique
        // edges and its `area` a numpy pairwise sum over the same per-face areas.
        assert_eq!(res, stats["res"].as_f64().unwrap(), "{name}: res");
        assert_eq!(res, json(&d.join("mesh.res.json")).as_f64().unwrap(), "{name}: mesh.res");
        assert_eq!(geom.total_area(), stats["area"].as_f64().unwrap(), "{name}: area");
        assert_eq!(watertight, stats["watertight"].as_bool().unwrap(), "{name}: watertight");
        assert_eq!(
            n_boundary,
            stats["n_boundary"].as_u64().unwrap() as usize,
            "{name}: n_boundary"
        );
        let wt = json(&d.join("mesh.watertight.json"));
        assert_eq!(watertight, wt["watertight"].as_bool().unwrap());
        assert_eq!(n_boundary, wt["n_boundary"].as_u64().unwrap() as usize);
    }
}

#[test]
fn injected_thickness_and_face_budget_reproduce_the_reference_exactly() {
    for name in PIECES {
        let d = fragment_dir(name);
        let v0 = vertices(&d.join("load.V0.npy"));
        let f0 = triangles(&d.join("load.F0.npy"));
        let geom0 = face_geometry(&v0, &f0);

        let target = json(&d.join("thick.target.json"));
        assert_eq!(geom0.total_area(), target["area0"].as_f64().unwrap(), "{name}: ΣA0");
        assert_eq!(f0.len(), target["faces0"].as_u64().unwrap() as usize, "{name}: faces0");

        // The reference's own rays, replayed through the reference's own filter and histogram.
        let idx: Vec<u32> = npy::<i64>(&d.join("thick.idx.npy"))
            .into_iter()
            .map(|i| u32::try_from(i).unwrap())
            .collect();
        let hits = RayHits {
            t_hit: npy::<f32>(&d.join("thick.t_hit.npy")),
            prim: npy::<u32>(&d.join("thick.prim.npy")),
        };
        let (t, mode) = thickness::thickness_from_hits(&geom0.normals, &idx, &hits)
            .unwrap_or_else(|| panic!("{name}: the reference's rays produced an estimate"));
        assert_eq!(
            f64::from(t),
            json(&d.join("thick.t.json")).as_f64().unwrap(),
            "{name}: t is the same float32 the reference computed"
        );
        assert_eq!(
            f64::from(mode),
            json(&d.join("thick.thick_mode.json")).as_f64().unwrap(),
            "{name}: thick_mode"
        );

        assert_eq!(
            face_budget(geom0.total_area(), f64::from(t), TARGET_FACES),
            target["target"].as_u64().unwrap() as usize,
            "{name}: the adaptive face budget"
        );
    }
}

#[test]
fn injected_taubin_lands_on_open3ds_working_mesh() {
    for name in PIECES {
        let d = fragment_dir(name);
        let target = json(&d.join("thick.target.json"));
        assert!(
            target["faces0"].as_u64().unwrap() <= target["target"].as_u64().unwrap(),
            "{name}: the slab is under its budget, so the reference's working mesh is exactly \
             Taubin(V0, F0) and this comparison is meaningful"
        );

        let mut m = Mesh::new(vertices(&d.join("load.V0.npy")), triangles(&d.join("load.F0.npy")));
        taubin(&mut m);

        let expected = vertices(&d.join("mesh.V.npy"));
        let res = json(&d.join("mesh.res.json")).as_f64().unwrap();
        assert_eq!(m.v.len(), expected.len(), "{name}: vertex count");
        let worst =
            m.v.iter()
                .zip(&expected)
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                .fold(0.0_f64, f64::max);
        // Open3D sums each vertex's neighbours in `std::unordered_set` order and this module in
        // ascending index order; nothing else differs, so what is left is round-off. Measured
        // 2.3e-13 (pieceA) and 2.0e-13 (pieceB), i.e. 1e-13 of one edge length.
        assert!(
            worst < 1e-9 * res,
            "{name}: Taubin differs from Open3D by {worst:.3e} ({:.2e} of res)",
            worst / res
        );
    }
}

#[test]
fn the_native_stage_meets_the_design_tolerances() {
    for name in PIECES {
        let d = fragment_dir(name);
        let stats = json(&d.join("mesh.stats.json"));
        let path = repo_root().join("fixtures/slab/input").join(format!("{name}.ply"));

        let fr = Fragment::from_mesh_file(&path, TARGET_FACES).expect("the slab loads");
        assert_eq!(fr.name, name);

        let counts = json(&d.join("load.n_orig.json"));
        assert_eq!(
            u64::from(fr.n_orig_vertices),
            counts["n_orig_vertices"].as_u64().unwrap(),
            "{name}: vertices after cleaning"
        );
        assert_eq!(
            u64::from(fr.n_orig_faces),
            counts["n_orig_faces"].as_u64().unwrap(),
            "{name}: faces after cleaning"
        );

        // D §10.2, native column.
        let ref_faces = stats["faces"].as_u64().unwrap() as f64;
        let ref_res = stats["res"].as_f64().unwrap();
        let ref_area = stats["area"].as_f64().unwrap();
        let area: f64 = fr.mesh.face_areas.iter().map(|&a| f64::from(a)).sum();
        let rel = |a: f64, b: f64| (a - b) / b;

        assert!(
            rel(fr.n_faces() as f64, ref_faces).abs() <= 0.05,
            "{name}: faces {} vs {ref_faces}",
            fr.n_faces()
        );
        assert!(rel(fr.res(), ref_res).abs() <= 0.10, "{name}: res {} vs {ref_res}", fr.res());
        assert!(rel(area, ref_area).abs() <= 0.005, "{name}: area {area} vs {ref_area}");
        assert_eq!(
            fr.watertight,
            stats["watertight"].as_bool().unwrap(),
            "{name}: the watertight verdict must agree"
        );

        // Thickness: the sampler is `ChaCha8Rng`, not numpy's PCG64 (PMC-9), so `t` is a
        // different draw of the same estimator. On the slab the two agree to 0.13 %; the note
        // records 6.6 % as the worst over all seven collections, inside the estimator's own
        // seed-to-seed spread.
        let ref_t = stats["thick"].as_f64().unwrap();
        assert!(rel(fr.thick, ref_t).abs() <= 0.02, "{name}: t {} vs {ref_t}", fr.thick);
        let ref_mode = stats["thick_mode"].as_f64().unwrap();
        assert!(
            rel(fr.thick_mode, ref_mode).abs() <= 0.02,
            "{name}: thick_mode {} vs {ref_mode}",
            fr.thick_mode
        );
    }
}

#[test]
fn the_native_stage_is_bit_reproducible() {
    let path = repo_root().join("fixtures/slab/input/pieceA.ply");
    let a = Fragment::from_mesh_file(&path, TARGET_FACES).expect("the slab loads");
    let b = Fragment::from_mesh_file(&path, TARGET_FACES).expect("the slab loads");
    assert_eq!(a.mesh.f, b.mesh.f);
    assert_eq!(a.mesh.v, b.mesh.v);
    assert_eq!(a.mesh.face_normals, b.mesh.face_normals);
    assert_eq!(a.mesh.face_areas, b.mesh.face_areas);
    assert_eq!(a.mesh.res.to_bits(), b.mesh.res.to_bits());
    assert_eq!(a.thick.to_bits(), b.thick.to_bits());
    assert_eq!(a.thick_mode.to_bits(), b.thick_mode.to_bits());
    assert_eq!((a.watertight, a.n_boundary), (b.watertight, b.n_boundary));
    assert_eq!(a.face_budget, b.face_budget);
}
