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
use std::sync::OnceLock;

use sherd_core::Params;
use sherd_core::fragment::Fragment;
use sherd_core::matching::icp::Numerics;
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

/// The two fragments, preprocessed once: R §3 is the expensive half of this file and both tests
/// need the same two answers from it.
fn slab_fragments() -> &'static (Fragment, Fragment) {
    static FRAGMENTS: OnceLock<(Fragment, Fragment)> = OnceLock::new();
    FRAGMENTS.get_or_init(|| {
        (
            Fragment::from_mesh_file(slab_input().join("pieceA.ply"), 200_000).expect("pieceA"),
            Fragment::from_mesh_file(slab_input().join("pieceB.ply"), 200_000).expect("pieceB"),
        )
    })
}

/// Every twentieth vertex of B, which is what `tests/test_synthetic.py` measures a pose over.
fn probe_points(b: &Fragment) -> Vec<[f64; 3]> {
    b.mesh.v.iter().step_by(20).map(|v| v.to_f64()).collect()
}

/// R §5.1–5.3 on the slab: the true pose is among the 250 the NMS keeps, and it is not there by
/// accident — it is one of the highest-scoring poses of the whole set.
#[test]
fn the_slab_pairs_true_pose_survives_the_coarse_stage() {
    let params = Params::default();
    let (a, b) = slab_fragments();
    assert!(!Pair::skipped(a, b, &params), "two halves of one slab have the same wall");

    let pair = Pair::build(a, b, &params);
    assert!(pair.matchable());
    let hyp = pair.hypotheses(&params);
    assert!(hyp.len() > 1000, "{} hypotheses", hyp.len());
    let cs = pair.coarse(&hyp, &pair.probe(&params));
    let kept = pair.suppress(&hyp, &cs, &params);
    assert_eq!(kept.len(), params.stage1 as usize, "the walk fills its budget on a true pair");

    let points = probe_points(b);
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

/// R §5.4–5.6 on the slab: the two ladders turn one of those coarse poses into the true one.
///
/// The coarse stage above leaves the truth 3.20° and 0.305 t away — the accuracy of a pose built
/// from one pair of breakline frames. This is the claim the refinement exists to make: after the
/// two point-to-point breakline rungs and the four point-to-plane surface rungs, one of the ten
/// candidates R §5.5 keeps is the pose that reassembles the slab, to the 2° / 0.1 t that
/// `tests/test_synthetic.py` demands of a *refined* candidate. Measured here: **0.019° and
/// 0.0026 t**, a hundred times inside it on the rotation and forty on the displacement.
///
/// It is the ground-truth counterpart of the parity rows. Parity says the port's ladder computes
/// what Open3D's computes; this says the ladder converges on the seam — a port could reproduce
/// the reference bit for bit with both implementations refining towards the wrong surface, and
/// only a known answer would notice.
#[test]
fn the_slab_pairs_two_ladders_reach_the_ground_truth() {
    let params = Params::default();
    let (a, b) = slab_fragments();
    let pair = Pair::build(a, b, &params);
    let hyp = pair.hypotheses(&params);
    let cs = pair.coarse(&hyp, &pair.probe(&params));
    let kept = pair.suppress(&hyp, &cs, &params);

    let points = probe_points(b);
    let want = relative_truth();
    let error = |t: &nalgebra::Matrix4<f64>| {
        let r = t.fixed_view::<3, 3>(0, 0).into_owned();
        let tau = t.fixed_view::<3, 1>(0, 3).into_owned();
        pose_error(&r, &tau, &want, &points)
    };

    // R §5.4: the breakline ladder, from every pose the coarse suppression kept.
    let stage1 = pair.stage1(&hyp, &kept, Numerics::REFERENCE);
    assert_eq!(stage1.len(), kept.len());
    let best1 = stage1
        .iter()
        .map(|c| error(&c.transform))
        .fold((f64::INFINITY, f64::INFINITY), |acc, e| if e.1 < acc.1 { e } else { acc });
    assert!(
        best1.0 <= 2.0 && best1.1 <= 0.2 * a.thick,
        "the best breakline pose is {:.3}° and {:.3} t from the truth",
        best1.0,
        best1.1 / a.thick
    );
    // And the ladder is a refinement, not a random walk: it *improves* on the hypothesis it
    // started from — 9.21 units (0.305 t) to 1.75 (0.058 t) on this pair, a factor of five.
    let best0 = kept
        .iter()
        .map(|&h| pose_error(&hyp.r[h as usize], &hyp.tau[h as usize], &want, &points))
        .fold((f64::INFINITY, f64::INFINITY), |acc, e| if e.1 < acc.1 { e } else { acc });
    assert!(best1.1 < best0.1 / 4.0, "{best0:?} -> {best1:?}");

    // R §5.5–5.6: the ten candidates, each through the four surface rungs.
    let kept2 = pair.suppress_stage1(&stage1, &params);
    assert!(!kept2.is_empty() && kept2.len() <= params.stage2 as usize);
    let mut best2 = (f64::INFINITY, f64::INFINITY);
    for &k in &kept2 {
        let out = pair.stage2(&stage1[k as usize].transform, Numerics::REFERENCE);
        assert_eq!(out.len(), 4, "R §5.6 is four rungs");
        let e = error(&out[3].transform);
        if e.1 < best2.1 {
            best2 = e;
        }
    }
    assert!(
        best2.0 <= 2.0 && best2.1 <= 0.1 * a.thick,
        "the best refined candidate is {:.3}° and {:.4} t from the truth",
        best2.0,
        best2.1 / a.thick
    );
    assert!(best2.1 < best1.1, "the surface rungs improve on the breakline ones");
}

/// R §4–§6 end to end on the slab: `match_pair` accepts the join, at the true pose.
///
/// The two tests above take the pipeline apart stage by stage; this one asks it the question the
/// whole pair stage exists to answer — *are these two fragments adjacent, and where?* — through
/// the one entry point the pipeline uses, and checks the verdict as well as the pose.
///
/// The reference's own answer for this pair is in the committed dump
/// (`fixtures/slab/dump/pairs/pieceA__pieceB/result.candidates.json`): one accepted candidate,
/// `seam` 21.0, `tight` 0.840, `gap` 0.00103, `pen` 0, `cont_n` 0.992. The port draws its own
/// samples (PMC-9), so the numbers here are its own; what has to agree is the decision, the pose
/// and the order of magnitude of every score.
#[test]
fn the_slab_pair_is_accepted_at_the_true_pose() {
    let params = Params::default();
    let (a, b) = slab_fragments();
    let candidates = sherd_core::matching::pair::match_pair(a, b, &params, 5);
    assert!(!candidates.is_empty(), "the pair produces candidates");
    assert!(candidates.len() <= 5, "R §5.7 returns at most `keep`");

    // R §5.7's ranking: `seam · tight`, descending.
    for pair in candidates.windows(2) {
        assert!(pair[0].score() >= pair[1].score(), "the candidates are not ranked");
    }

    // More than one candidate can pass R §6.5, and the reference's own answer for this pair says
    // so: it accepts **two** of its ten, the join (`seam` 21.0, `tight` 0.840) and a second
    // placement of the curved slab against itself 88° away (`seam` 15.0, `tight` 0.275, right on
    // `min_tight`). The port accepts three, for the same reason and with the same shape — R §13's
    // note on `pot_C` is the same observation on real sherds. What has to be true is that the
    // *ranking* separates them: R §5.7's `seam · tight` puts the join first, by a factor of four
    // here, and the assembly of R §8 sees the pair through that ranking.
    let accepted: Vec<_> = candidates.iter().filter(|c| c.accepted).collect();
    assert!(!accepted.is_empty(), "the join is not accepted at all");
    let best = &candidates[0];
    assert!(best.accepted, "the best-ranked candidate is not the accepted one");
    assert!(
        candidates.len() == 1 || best.score() > 2.0 * candidates[1].score(),
        "the join does not stand out: {:.2} against {:.2}",
        best.score(),
        candidates[1].score()
    );

    let s = &best.scores;
    assert!(s.tight >= params.min_tight, "tight {}", s.tight);
    assert!(s.seam >= params.min_seam, "seam {}", s.seam);
    assert!(s.cont_n >= params.min_cont_n, "cont_n {}", s.cont_n);
    assert!(s.pen <= params.max_pen, "pen {}", s.pen);
    assert!(s.gap * a.thick <= 0.03 * a.thick + f64::EPSILON, "gap {}", s.gap);
    assert!(!s.partial && !s.pen_unavailable, "the full verification ran");
    assert!(s.brk_best >= s.brk, "`brk_best` is the pair's best stage-1 score");
    // The reference's are 21.0 and 0.840 on its own samples; a factor of two either way is a
    // regression alarm on the port's, not a parity claim.
    assert!(s.seam > 10.0 && s.tight > 0.4, "seam {} tight {}", s.seam, s.tight);

    // And the accepted pose is the one that reassembles the slab, to `test_synthetic.py`'s bounds.
    let r = best.transform.fixed_view::<3, 3>(0, 0).into_owned();
    let tau = best.transform.fixed_view::<3, 1>(0, 3).into_owned();
    let (angle, distance) = pose_error(&r, &tau, &relative_truth(), &probe_points(b));
    assert!(
        angle <= 2.0 && distance <= 0.1 * a.thick,
        "the accepted candidate is {angle:.3}° and {:.4} t from the truth",
        distance / a.thick
    );
}
