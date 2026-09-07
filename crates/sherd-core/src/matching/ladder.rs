//! The two refinement ladders (R §5.4, R §5.5, R §5.6's ICP half).
//!
//! A hypothesis is a pose built from one pair of breakline frames, so it is right to about the
//! spacing of the thinned breakline and the width of a macro normal — near enough to refine, far
//! too coarse to score. Both stages therefore run the same ICP ([`icp`](super::icp)) over a
//! **ladder of shrinking correspondence radii**, each rung starting where the last one stopped:
//!
//! ```text
//! stage 1   B.pc_brk  → A.pc_brk_full   point-to-point   0.2 t, 0.08 t   20 iterations each
//! stage 2   B.pc_reg  → A.pc_reg        point-to-plane   0.2 t, 0.08 t   30 iterations each
//!           B.pc_frac → A.pc_frac       point-to-plane   0.08 t, 0.04 t  30 iterations each
//! ```
//!
//! # Why a ladder, and why these two clouds
//!
//! A wide radius has a large basin of attraction and a coarse optimum: it pulls a pose that is a
//! fifth of a wall thickness out into contact, and it is dominated by correspondences that are
//! wrong. A narrow one cannot move far but is right. Running the wide rung first and the narrow
//! rung from its answer buys both, and the rungs are *stretched together* by
//! [`Scales::icp`](super::scales::Scales::icp) rather than floored one at a time, so that a coarse
//! mesh cannot make the fine rung overtake the coarse one.
//!
//! The clouds differ for the same reason. Stage 1 registers **breakline to breakline** — two
//! curves, point-to-point, because a curve has no surface to be normal to — and it is the cheapest
//! way to tell a pose that nearly lays the seams together from one that does not. Stage 2 then
//! registers **surface to surface**: first `pc_reg` (fracture samples plus the shell margin, 6 000
//! points, R §3.6), which is what brings the pose into the basin of the fine rungs, and then
//! `pc_frac` (the fracture samples alone), which is what a fit between two fracture surfaces
//! actually is. Point-to-plane throughout, because two samples of the same surface never land on
//! each other and the point-to-point distance between them has a floor at the sample spacing that
//! the point-to-plane distance does not.
//!
//! # The re-score is not the ICP's own answer
//!
//! R §5.4 scores a refined pose with [`brk_score`], the coarse estimator of R §5.2 at the tighter
//! `sc.stage1` radius over the whole of B's `brk_sub`. The ICP's `fitness` is *not* used: it counts
//! source points with a target inside the radius, whatever the shells are doing, and a pose that
//! lays B's breakline on A's back to front would score well on it. `brk_score` carries the normal
//! test, which is what makes it a seam test rather than a proximity test.

use nalgebra::Matrix4;

use super::coarse::{self, Target};
use super::icp::{self, Estimation, IcpTarget, Numerics, Options, Registration};
use super::nms;
use super::scales::Scales;

/// R §5.4's rungs, as multiples of `t_pair` before [`Scales::icp_dist`] stretches them.
pub const STAGE1_RUNGS: [f64; 2] = [0.2, 0.08];
/// R §5.4's iteration cap per rung.
pub const STAGE1_ITERATIONS: usize = 20;
/// R §5.6's two `pc_reg` rungs.
pub const STAGE2_REG_RUNGS: [f64; 2] = [0.2, 0.08];
/// R §5.6's two `pc_frac` rungs.
pub const STAGE2_FRAC_RUNGS: [f64; 2] = [0.08, 0.04];
/// R §5.6's iteration cap per rung.
pub const STAGE2_ITERATIONS: usize = 30;
/// R §5.5's floor under the stage-1 score.
pub const STAGE1_FLOOR: f64 = 0.05;

/// One rung: which clouds, which estimator, which radius and how many iterations.
///
/// The radius is given as `k` in `sc.icp_dist(k)` rather than as a distance, because that is the
/// form R §5.4 and R §5.6 write and because the stretch belongs to the pair rather than to the
/// rung.
#[derive(Clone, Copy, Debug)]
pub struct Rung<'a> {
    /// The moving cloud, in B's own frame.
    pub source: &'a [[f64; 3]],
    /// The fixed cloud, with its tree (and, for point-to-plane, its normals).
    pub target: &'a IcpTarget,
    /// The rung, in wall thicknesses before the stretch.
    pub k: f64,
    /// Which update this rung computes.
    pub estimation: Estimation,
    /// The rung's iteration cap.
    pub iterations: usize,
}

/// Runs a ladder from `init`, returning the result of **every** rung in ladder order.
///
/// Every rung is returned rather than only the last because the fixture dump carries a pose per
/// rung (`s2.T_reg1`, `s2.T_reg2`, `s2.T_frac1`, `s2.T_frac2`, D §10.1) and the parity harness
/// compares them one at a time: a ladder that ends in the right place by a different route is a
/// different ladder, and only the intermediate poses say so.
pub fn climb(
    rungs: &[Rung<'_>],
    init: &Matrix4<f64>,
    scales: &Scales,
    numerics: Numerics,
) -> Vec<Registration> {
    let mut pose = *init;
    let mut out = Vec::with_capacity(rungs.len());
    for rung in rungs {
        let options = Options {
            estimation: rung.estimation,
            max_correspondence_distance: scales.icp_dist(rung.k),
            max_iteration: rung.iterations,
            numerics,
        };
        let result = icp::register(rung.source, rung.target, &pose, &options);
        pose = result.transform;
        out.push(result);
    }
    out
}

/// R §5.4's ladder: two point-to-point rungs of B's breakline subset against A's whole breakline.
pub fn stage1_rungs<'a>(source: &'a [[f64; 3]], target: &'a IcpTarget) -> [Rung<'a>; 2] {
    STAGE1_RUNGS.map(|k| Rung {
        source,
        target,
        k,
        estimation: Estimation::PointToPoint,
        iterations: STAGE1_ITERATIONS,
    })
}

/// R §5.6's ladder: two point-to-plane rungs on `pc_reg`, then two on `pc_frac`.
pub fn stage2_rungs<'a>(
    reg_source: &'a [[f64; 3]],
    reg_target: &'a IcpTarget,
    frac_source: &'a [[f64; 3]],
    frac_target: &'a IcpTarget,
) -> [Rung<'a>; 4] {
    let rung = |source, target, k| Rung {
        source,
        target,
        k,
        estimation: Estimation::PointToPlane,
        iterations: STAGE2_ITERATIONS,
    };
    [
        rung(reg_source, reg_target, STAGE2_REG_RUNGS[0]),
        rung(reg_source, reg_target, STAGE2_REG_RUNGS[1]),
        rung(frac_source, frac_target, STAGE2_FRAC_RUNGS[0]),
        rung(frac_source, frac_target, STAGE2_FRAC_RUNGS[1]),
    ]
}

/// R §5.4's `brk_score`: the fraction of B's breakline subset that lands on A's breakline within
/// `delta` with an agreeing shell normal, under the pose `transform`.
pub fn brk_score(
    target: &Target<'_>,
    points: &[[f64; 3]],
    normals: &[[f64; 3]],
    transform: &Matrix4<f64>,
    delta: f64,
) -> f64 {
    coarse::score_pose(target, points, normals, transform, delta)
}

/// R §5.5's walk order: the stage-1 scores, descending, ties by ascending candidate (PMC-6).
///
/// Un-truncated, unlike R §5.3's: there are at most `stage1` poses to walk.
pub fn stage1_order(score: &[f64]) -> Vec<u32> {
    nms::order_by_score(score, usize::MAX)
}

/// R §5.5's suppression over the stage-1 poses, walking `order`.
///
/// The order is a parameter for the same reason it is one in R §5.3 (PMC-6): the reference's is
/// `np.argsort(s1)[::-1]`, an unstable sort over scores that are multiples of `1/|brk_sub|`, and
/// the fixture dump carries it as `nms2.order` so that an injected comparison measures the greedy
/// loop rather than numpy's partitioning.
pub fn suppress_stage1(
    poses: &[Matrix4<f64>],
    score: &[f64],
    order: &[u32],
    trans_tol: f64,
    topk: usize,
) -> Vec<u32> {
    let (rotations, translations) = split(poses);
    nms::nms(order, &rotations, &translations, score, trans_tol, topk, STAGE1_FLOOR)
}

/// The rotations and translations of a pose list, as R §5.5's suppression reads them.
fn split(poses: &[Matrix4<f64>]) -> (Vec<nalgebra::Matrix3<f64>>, Vec<nalgebra::Vector3<f64>>) {
    let mut rotations = Vec::with_capacity(poses.len());
    let mut translations = Vec::with_capacity(poses.len());
    for pose in poses {
        rotations.push(pose.fixed_view::<3, 3>(0, 0).into_owned());
        translations.push(pose.fixed_view::<3, 1>(0, 3).into_owned());
    }
    (rotations, translations)
}

#[cfg(test)]
mod tests {
    use super::{
        STAGE1_FLOOR, brk_score, climb, stage1_order, stage1_rungs, stage2_rungs, suppress_stage1,
    };
    use crate::matching::coarse::Target;
    use crate::matching::icp::{IcpTarget, Numerics, homogeneous};
    use crate::matching::scales::Scales;
    use crate::params::Params;
    use crate::spatial::kdtree::PointTree;
    use nalgebra::{Matrix3, Matrix4, Vector3};

    /// A curve of `n` points along x with shell normals up, and the same curve moved.
    fn curve(n: u32, dx: f64) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let p =
            (0..n).map(|k| [dx + f64::from(k) * 0.5, (0.3 * f64::from(k)).sin(), 0.0]).collect();
        (p, vec![[0.0, 0.0, 1.0]; n as usize])
    }

    /// The stage-1 ladder pulls a displaced breakline back onto the one it came from, and the
    /// re-score rises from nothing to one as it does.
    #[test]
    fn the_stage_one_ladder_lays_one_breakline_on_the_other() {
        let (points, normals) = curve(60, 0.0);
        let target = IcpTarget::new(points.clone(), normals.clone());
        let tree = PointTree::build(&points).expect("60 points");
        let scoring = Target { points: &points, normals: &normals, tree: &tree };
        let scales = Scales::for_pair(&Params::default(), 1.0, 0.0);

        // A pose 0.25 t out of place: inside the wide rung's radius, outside the re-score's.
        let init = homogeneous(&Matrix3::identity(), &Vector3::new(0.12, 0.10, 0.05));
        assert!(brk_score(&scoring, &points, &normals, &init, scales.stage1) < 1.0);

        let rungs = stage1_rungs(&points, &target);
        let out = climb(&rungs, &init, &scales, Numerics::REFERENCE);
        assert_eq!(out.len(), 2);
        let refined = out[1].transform;
        assert!(
            (brk_score(&scoring, &points, &normals, &refined, scales.stage1) - 1.0).abs() < 1e-12
        );
        // And the pose it found is the identity it started from.
        let (angle, distance) = {
            let r = refined.fixed_view::<3, 3>(0, 0).into_owned();
            let trace = r.trace();
            (
                ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees(),
                refined.fixed_view::<3, 1>(0, 3).norm(),
            )
        };
        assert!(angle < 1e-6 && distance < 1e-6, "{angle:e} {distance:e}");
    }

    /// The stage-2 ladder is four rungs on two clouds, in R §5.6's order and at R §5.6's radii.
    #[test]
    fn the_stage_two_ladder_is_two_clouds_and_four_rungs() {
        let (points, normals) = curve(40, 0.0);
        let reg = IcpTarget::new(points.clone(), normals.clone());
        let frac = IcpTarget::new(points.clone(), normals);
        let rungs = stage2_rungs(&points, &reg, &points, &frac);
        assert_eq!(rungs.map(|r| r.k.to_bits()), [0.2, 0.08, 0.08, 0.04].map(f64::to_bits));
        assert!(rungs.iter().all(|r| r.iterations == 30));
        let scales = Scales::for_pair(&Params::default(), 2.0, 0.0);
        assert!((scales.icp_dist(rungs[0].k) - 0.4).abs() < 1e-15);
        let out = climb(&rungs, &Matrix4::identity(), &scales, Numerics::REFERENCE);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|r| (r.fitness - 1.0).abs() < 1e-12));
    }

    /// R §5.5's suppression keeps the best of a cluster and stops at the floor.
    #[test]
    fn the_stage_one_suppression_keeps_one_pose_per_cluster() {
        let at = |x: f64| homogeneous(&Matrix3::identity(), &Vector3::new(x, 0.0, 0.0));
        let poses = vec![at(0.0), at(0.1), at(5.0), at(5.05)];
        let score = vec![0.9, 0.8, 0.7, 0.6];
        let order = stage1_order(&score);
        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(suppress_stage1(&poses, &score, &order, 1.0, 10), vec![0, 2]);
        // A cluster radius below the spacing keeps all four.
        assert_eq!(suppress_stage1(&poses, &score, &order, 0.01, 10), vec![0, 1, 2, 3]);
        // The floor ends the walk — which is score-ordered, so it is the last pose that falls
        // below it — and `topk` caps it.
        let score = vec![0.9, 0.8, 0.7, STAGE1_FLOOR - 1e-9];
        assert_eq!(suppress_stage1(&poses, &score, &stage1_order(&score), 0.01, 10), vec![0, 1, 2]);
        assert_eq!(suppress_stage1(&poses, &score, &stage1_order(&score), 0.01, 1), vec![0]);
    }
}
