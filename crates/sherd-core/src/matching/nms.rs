//! Non-maximum suppression over poses (R §5.3, R §5.5).
//!
//! The hypothesis set is dense: a true seam is found by thousands of frame pairs at once, all
//! describing the same pose to within a fraction of a wall thickness, and scoring them is cheap
//! while refining them is not. NMS walks the hypotheses best score first and keeps one from each
//! cluster — a pose is dropped when an already kept one is nearer than `trans_tol` in translation
//! **and** turned by less than about 18.2°.
//!
//! Two things about the rule are worth stating, because both look like details and neither is.
//!
//! * **The two tests are an `and`.** A pose at the same place turned by 90° is a different
//!   hypothesis about how the sherds meet, and a pose with the same orientation a wall thickness
//!   away is too. Only the conjunction means "the same candidate".
//! * **`trace(Rᵀ R') > 2.9` is the rotation test**, and it is written that way rather than as an
//!   angle because the trace *is* `1 + 2cos θ` for a rotation: 2.9 is `θ < 18.19°`, and no
//!   `arccos` is evaluated in the inner loop.
//!
//! The walk stops at `topk` kept poses, and it *breaks* — not skips — at the first score below
//! `floor`, because the order is descending and nothing after it can pass.
//!
//! # PMC-6: the order is an input, and the reference's is arbitrary
//!
//! The reference walks `np.argsort(score)[::-1]`, an **unstable** quicksort. Coarse scores are
//! multiples of `1/60`, so thousands of hypotheses tie at every level and which of them the sort
//! puts first is an artefact of numpy's partitioning. This port sorts stably, with the hypothesis
//! index as the tie-break ([`order_by_score`]), which is reproducible but is *not* the reference's
//! permutation — so the kept sets differ wherever a tie decides. The greedy loop and the duplicate
//! test are compared against the reference on the reference's own order, which the fixture dump
//! carries as `nms1.order` for exactly this reason; what the port's own order does instead is
//! measured in `notes/2026-09-07-c1-hypotheses.md`.

use nalgebra::{Matrix3, Vector3};

/// R §5.3's rotation test: `trace(R_hᵀ R_k) > 2.9`, which is a turn of less than 18.19°.
pub const ROT_TOL: f64 = 2.9;

/// R §5.3's walk order: descending by score, ties by ascending index, truncated to `limit`.
///
/// `limit` is `usize::MAX` for the un-truncated form R §5.5 uses. The tie-break is what makes this
/// reproducible; it is not the reference's (PMC-6, see the module documentation).
pub fn order_by_score(score: &[f64], limit: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..u32::try_from(score.len()).unwrap_or(u32::MAX)).collect();
    order.sort_unstable_by(|&a, &b| {
        // `total_cmp` rather than `partial_cmp`: a total order keeps the sort a sort even if a
        // score were ever NaN, and the tie-break below then still decides.
        score[b as usize].total_cmp(&score[a as usize]).then(a.cmp(&b))
    });
    order.truncate(limit);
    order
}

/// R §5.3's greedy suppression over `order`.
///
/// `r`, `tau` and `score` are indexed by hypothesis; `order` names the hypotheses to walk, in the
/// order to walk them. Returns the kept hypotheses, in the order they were kept.
pub fn nms(
    order: &[u32],
    r: &[Matrix3<f64>],
    tau: &[Vector3<f64>],
    score: &[f64],
    trans_tol: f64,
    topk: usize,
    floor: f64,
) -> Vec<u32> {
    let mut kept: Vec<u32> = Vec::new();
    if topk == 0 {
        return kept;
    }
    for &h in order {
        let k = h as usize;
        if score[k] < floor {
            break;
        }
        let duplicate = kept.iter().any(|&i| {
            (tau[k] - tau[i as usize]).norm() < trans_tol
                && trace_of_transpose_product(&r[k], &r[i as usize]) > ROT_TOL
        });
        if !duplicate {
            kept.push(h);
        }
        if kept.len() >= topk {
            break;
        }
    }
    kept
}

/// `trace(Aᵀ B)`, summed the way `np.trace(A.T @ B)` sums it: each diagonal entry over `j`, then
/// the three of them.
///
/// It is the Frobenius inner product, so a different association would differ in the last bits
/// only — which matters here only because a hypothesis sitting exactly on [`ROT_TOL`] would then
/// be kept by one implementation and dropped by the other.
#[inline]
fn trace_of_transpose_product(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let mut trace = 0.0;
    for i in 0..3 {
        trace += a[(0, i)] * b[(0, i)] + a[(1, i)] * b[(1, i)] + a[(2, i)] * b[(2, i)];
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::{ROT_TOL, nms, order_by_score, trace_of_transpose_product};
    use nalgebra::{Matrix3, Vector3};

    fn rot_z(deg: f64) -> Matrix3<f64> {
        let (s, c) = deg.to_radians().sin_cos();
        Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    }

    fn at(x: f64) -> Vector3<f64> {
        Vector3::new(x, 0.0, 0.0)
    }

    /// The order is descending by score and ties go to the lower hypothesis index — the whole of
    /// PMC-6's substitution, and the reason two runs of the port keep the same poses.
    #[test]
    fn the_order_is_descending_with_the_index_breaking_ties() {
        let score = [0.1, 0.5, 0.5, 0.2, 0.5];
        assert_eq!(order_by_score(&score, usize::MAX), vec![1, 2, 4, 3, 0]);
        assert_eq!(order_by_score(&score, 3), vec![1, 2, 4]);
        assert_eq!(order_by_score(&[], 10), Vec::<u32>::new());
    }

    /// A cluster of poses at the same place and orientation collapses to its best member, and a
    /// pose far enough away, or turned far enough, survives beside it.
    #[test]
    fn one_pose_survives_a_cluster_and_a_distinct_pose_survives_beside_it() {
        let r = vec![rot_z(0.0), rot_z(2.0), rot_z(40.0), rot_z(0.0)];
        let tau = vec![at(0.0), at(0.1), at(0.0), at(5.0)];
        let score = [0.9, 0.8, 0.7, 0.6];
        let order = order_by_score(&score, usize::MAX);

        // 1 is a duplicate of 0 (near and barely turned); 2 is turned by 40°, 3 is 5 away.
        assert_eq!(nms(&order, &r, &tau, &score, 1.0, 10, 0.0), vec![0, 2, 3]);
        // With a translation radius wide enough to cover pose 3 it is suppressed as well.
        assert_eq!(nms(&order, &r, &tau, &score, 9.0, 10, 0.0), vec![0, 2]);
    }

    /// `topk` stops the walk and `floor` breaks it, and the floor is a `<` on the score.
    #[test]
    fn the_walk_stops_at_topk_and_breaks_at_the_floor() {
        let r = vec![rot_z(0.0); 4];
        let tau = vec![at(0.0), at(10.0), at(20.0), at(30.0)];
        let score = [0.9, 0.5, 0.5, 0.1];
        let order = order_by_score(&score, usize::MAX);

        assert_eq!(nms(&order, &r, &tau, &score, 1.0, 2, 0.0), vec![0, 1]);
        assert_eq!(nms(&order, &r, &tau, &score, 1.0, 10, 0.5), vec![0, 1, 2], "0.5 is not < 0.5");
        assert_eq!(nms(&order, &r, &tau, &score, 1.0, 10, 0.6), vec![0]);
        assert_eq!(nms(&order, &r, &tau, &score, 1.0, 0, 0.0), Vec::<u32>::new());
        assert!(nms(&[], &r, &tau, &score, 1.0, 10, 0.0).is_empty());
    }

    /// Both tests have to fire before a pose is a duplicate: same place, other orientation stays;
    /// same orientation, other place stays.
    #[test]
    fn the_duplicate_test_is_a_conjunction() {
        let r = vec![rot_z(0.0), rot_z(90.0), rot_z(0.0)];
        let tau = vec![at(0.0), at(0.0), at(100.0)];
        let score = [0.9, 0.8, 0.7];
        let order = order_by_score(&score, usize::MAX);
        assert_eq!(nms(&order, &r, &tau, &score, 1.0, 10, 0.0), vec![0, 1, 2]);
    }

    /// The rotation threshold is a trace, and 2.9 is 18.19°: the test is exercised on both sides
    /// of that angle rather than on the trace's own units.
    #[test]
    fn the_rotation_threshold_is_eighteen_degrees() {
        let identity = rot_z(0.0);
        assert!((trace_of_transpose_product(&identity, &identity) - 3.0).abs() < 1e-15);
        let angle = ((ROT_TOL - 1.0) / 2.0).acos().to_degrees();
        assert!((angle - 18.194_872).abs() < 1e-5, "{angle}");
        assert!(trace_of_transpose_product(&identity, &rot_z(18.1)) > ROT_TOL);
        assert!(trace_of_transpose_product(&identity, &rot_z(18.3)) < ROT_TOL);
        // And it does not care which way the turn went.
        assert!(trace_of_transpose_product(&identity, &rot_z(-18.1)) > ROT_TOL);
    }
}
