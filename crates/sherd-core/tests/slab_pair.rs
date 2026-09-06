//! The first half of R §5 on a pair whose answer is known: the committed synthetic slab.
//!
//! `fixtures/slab/input` is a curved 300 × 200 slab of wall thickness 30, cut in two along a bumpy
//! fracture surface, with each half given its own random rigid pose — so the pose that reassembles
//! it is known to machine precision (`ground_truth.json`), and R §5.1–5.3 can be asked the
//! question the whole stage exists to answer: **is the true pose among the ones we keep?**
//!
//! That is a different claim from the parity harness's. Parity says the port computes what the
//! reference computes; this says the computation finds the seam. A port could reproduce the
//! reference's arrays exactly with the frame mapping transposed in both implementations, and only
//! a ground truth would notice.
//!
//! The tolerances are the stage's own, not the pipeline's. These are *coarse* poses: one frame of
//! B laid onto one frame of A, before any ICP, so their accuracy is the spacing of the breakline
//! subset (R §3.5.5 thins it to `0.5 t` voxels) and the width of a macro-normal, not the 2° / 0.1 t
//! that `tests/test_synthetic.py` demands of the *refined* candidate.

use std::path::{Path, PathBuf};

use sherd_core::Params;
use sherd_core::fragment::Fragment;
use sherd_core::matching::pair::Pair;

fn slab_input() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
}

/// `ground_truth.json`'s `matrix` for one piece: the rigid transform that maps that fragment's
/// **file** coordinates into the assembled frame (`tools/make_slab.py`, and the format the
/// synthetic generator shares with the SfS++ staging tool).
fn truth(name: &str) -> [[f64; 4]; 4] {
    let text = std::fs::read(slab_input().join("ground_truth.json")).expect("the ground truth");
    let json: serde_json::Value = serde_json::from_slice(&text).expect("it is JSON");
    let rows = json["fragments"][name]["matrix"].as_array().expect("a 4x4").clone();
    let mut m = [[0.0; 4]; 4];
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.as_array().expect("a row").iter().enumerate() {
            m[i][j] = v.as_f64().expect("a number");
        }
    }
    m
}

/// `T_rel = M_A⁻¹ · M_B`, the pose that maps B's file coordinates into A's — R §0's convention for
/// a candidate, and `tests/test_synthetic.py`'s `pair["T_rel"]` (which writes it as
/// `T_A · T_B⁻¹`, the same thing: the dump's `matrix` is that test's `T` inverted).
fn relative_truth() -> ([[f64; 3]; 3], [f64; 3]) {
    let (a, b) = (truth("pieceA"), truth("pieceB"));
    // The inverse of a rigid `(R, τ)` is `(Rᵀ, −Rᵀτ)`, so the composition is
    // `(R_Aᵀ R_B, R_Aᵀ (τ_B − τ_A))`.
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[k][i] * b[k][j]).sum();
        }
    }
    let mut tau = [0.0; 3];
    for (i, out) in tau.iter_mut().enumerate() {
        *out = (0..3).map(|k| a[k][i] * (b[k][3] - a[k][3])).sum();
    }
    (r, tau)
}

/// `tests/test_synthetic.py::pose_error`: the rotation of `R_estᵀ R_true` in degrees, and the
/// largest displacement the two poses give any of `points`.
fn pose_error(
    r: &nalgebra::Matrix3<f64>,
    tau: &nalgebra::Vector3<f64>,
    truth: &([[f64; 3]; 3], [f64; 3]),
    points: &[[f64; 3]],
) -> (f64, f64) {
    let mut trace = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            trace += r[(j, i)] * truth.0[j][i];
        }
    }
    let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
    let mut worst = 0.0_f64;
    for p in points {
        let mut d = 0.0;
        for i in 0..3 {
            let est = r[(i, 0)] * p[0] + r[(i, 1)] * p[1] + r[(i, 2)] * p[2] + tau[i];
            let want =
                truth.0[i][0] * p[0] + truth.0[i][1] * p[1] + truth.0[i][2] * p[2] + truth.1[i];
            d += (est - want).powi(2);
        }
        worst = worst.max(d.sqrt());
    }
    (angle, worst)
}

/// R §5.1–5.3 on the slab: the true pose is among the 250 the NMS keeps, and it is not there by
/// accident — it is one of the highest-scoring poses of the whole set.
#[test]
fn the_slab_pairs_true_pose_survives_the_coarse_stage() {
    let params = Params::default();
    let a = Fragment::from_mesh_file(slab_input().join("pieceA.ply"), 200_000).expect("pieceA");
    let b = Fragment::from_mesh_file(slab_input().join("pieceB.ply"), 200_000).expect("pieceB");
    assert!(!Pair::skipped(&a, &b, &params), "two halves of one slab have the same wall");

    let pair = Pair::build(&a, &b, &params);
    assert!(pair.matchable());
    let hyp = pair.hypotheses(&params);
    assert!(hyp.len() > 1000, "{} hypotheses", hyp.len());
    let cs = pair.coarse(&hyp, &pair.probe(&params));
    let kept = pair.suppress(&hyp, &cs, &params);
    assert_eq!(kept.len(), params.stage1 as usize, "the walk fills its budget on a true pair");

    // Every twentieth vertex of B, which is what the Python test measures the displacement over.
    let points: Vec<[f64; 3]> = b.mesh.v.iter().step_by(20).map(|v| v.to_f64()).collect();
    let want = relative_truth();

    let mut best = (f64::INFINITY, f64::INFINITY, usize::MAX);
    for (rank, &h) in kept.iter().enumerate() {
        let (angle, distance) =
            pose_error(&hyp.r[h as usize], &hyp.tau[h as usize], &want, &points);
        if distance < best.1 {
            best = (angle, distance, rank);
        }
    }
    let (angle, distance, rank) = best;

    // A hypothesis is one frame of B laid onto one frame of A, before any ICP: its accuracy is
    // the `0.5 t` spacing of R §3.5.5's subset and the width of a macro normal, and the
    // displacement is a maximum over a 300 × 200 slab, so a few degrees of it reach ten units.
    // Measured here: **3.20°, 9.21 units = 0.305 t, at rank 0** — the highest-scoring pose the
    // suppression keeps is the true one.
    assert!(angle <= 10.0, "the best kept pose is {angle:.2}° from the truth");
    assert!(
        distance <= 0.5 * a.thick,
        "the best kept pose moves a point of B by {distance:.2} ({:.3} t)",
        distance / a.thick
    );
    assert!(rank < 25, "the true pose is kept at rank {rank} of {}", kept.len());

    // And it is a high-scoring pose, not one the suppression happened to let through: the coarse
    // score of the kept pose nearest the truth is within a probe point of the best score there is.
    let top = cs.iter().copied().fold(0.0_f64, f64::max);
    let found = cs[kept[rank] as usize];
    assert!(
        found >= top - 2.0 / f64::from(params.coarse_points),
        "the true pose scores {found:.4} against a best of {top:.4}"
    );
}
