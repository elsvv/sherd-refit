//! The coarse breakline score (R §5.2).
//!
//! Every hypothesis of R §5.1 is a pose, and there are tens of thousands of them; the coarse score
//! is the cheapest question that separates them. Sixty points of B's breakline are moved by the
//! pose, and each one asks: is there a point of **A's whole breakline** within `sc.coarse`
//! (0.15 t) of me, and does its shell normal point the same way as mine? The score is the fraction
//! that can answer yes.
//!
//! Three details carry the stage.
//!
//! * **Sixty points, not the whole subset.** The score is a fraction, so its precision is `1/60`
//!   and its cost is 60 nearest-neighbour queries per hypothesis instead of a thousand. The
//!   subset is drawn once per pair, from `B.brk_sub`, and every hypothesis is scored on the *same*
//!   sixty points — otherwise the ranking would compare poses through different lenses.
//! * **The target is A's full breakline**, not its thinned subset: the hypotheses are anchored at
//!   subset points, but a pose is good if B's curve lands on A's curve anywhere along it.
//! * **The normal test is what makes it a seam test.** Two breaklines can cross at a point with
//!   their shells facing opposite ways — the pot turned inside out — and `ns_A · R·ns_B > 0.7`
//!   (about 45°) rejects that without any distance being wrong.
//!
//! # PMC-9, and what "injected" means here
//!
//! The sixty points come out of a generator, and the port's is `ChaCha8Rng` rather than numpy's
//! PCG64, so the *drawn subset* differs from the reference's and with it every score. Nothing else
//! in this module has a random number in it: fed the reference's own `coarse.idx` the port
//! reproduces the reference's scores, which is what the injected parity row measures. Natively the
//! two implementations evaluate the same estimator on a different sixty points.

use nalgebra::Matrix4;

use crate::executor::Executor;
use crate::executor::batch::{CoarseBatch, Poses};
use crate::matching::hypotheses::{Frames, Hypotheses};
use crate::rng::{self, Draw};
use crate::spatial::grid::NearMask;
use crate::spatial::kdtree::PointTree;

/// The agreement threshold between the two shell normals, `cos θ > 0.7` (R §5.2, R §5.4).
///
/// The same number gates the stage-1 re-score and R §6.2's seam, and it is a cosine of about 45°:
/// two shells that agree to within a right angle are the same side of the same wall.
pub const NORMAL_AGREE: f64 = 0.7;

/// A's breakline as the score queries it: the points, their shell normals, and a tree over them.
///
/// The tree must be the one built over `points` — [`MatchData::kd_brk`] is, by construction
/// (R §3.6) — because the score reads `normals[j]` at the index the tree returns.
///
/// [`MatchData::kd_brk`]: crate::fragment::samples::MatchData::kd_brk
#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    /// `P_A`: every breakline point of A, in adjacency order.
    pub points: &'a [[f64; 3]],
    /// `ns_A`: the shell macro-normal at each of them.
    pub normals: &'a [[f64; 3]],
    /// A KD-tree over `points`.
    pub tree: &'a PointTree,
}

/// The sixty points of B the score is measured on, with their shell normals (R §5.2's `Q`, `QN`).
#[derive(Clone, Debug, Default)]
pub struct Probe {
    /// The indices drawn from `B.brk_sub`, in draw order.
    pub idx: Vec<u32>,
    /// `P_B[idx]`.
    pub q: Vec<[f64; 3]>,
    /// `ns_B[idx]`.
    pub qn: Vec<[f64; 3]>,
}

impl Probe {
    /// The probe points named by `idx`, which index B's breakline arrays.
    pub fn at(b: &Frames, idx: &[u32]) -> Self {
        Self {
            idx: idx.to_vec(),
            q: idx.iter().map(|&i| b.p[i as usize]).collect(),
            qn: idx.iter().map(|&i| b.ns[i as usize]).collect(),
        }
    }

    /// R §5.2's own draw: `min(points, |pool|)` distinct entries of `pool`, then [`Probe::at`].
    ///
    /// This is the first draw of the reference's `rng_pair = rng(p.seed)`; the port gives it its
    /// own stream instead (D §7, PMC-9's addendum), so it does not move when another draw's count
    /// changes. The order is the draw order and is not sorted — the score is a mean over the set,
    /// so the order cannot reach the answer, but it is the reference's shape and the fixture dump
    /// carries it that way.
    pub fn draw(b: &Frames, pool: &[u32], points: usize, seed: u64) -> Self {
        let mut rng = rng::seeded_for(seed, Draw::CoarsePoints);
        let picked = rng::without_replacement(pool.len(), points, &mut rng);
        let idx: Vec<u32> = picked.into_iter().map(|k| pool[k as usize]).collect();
        Self::at(b, &idx)
    }

    /// How many points the score is a fraction of.
    #[inline]
    pub fn len(&self) -> usize {
        self.q.len()
    }

    /// True when there is nothing to score with.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }
}

/// R §5.2 for every hypothesis: the fraction of the probe that lands on A's breakline.
///
/// The hypotheses are independent, so this is one [`Executor`] batch and the executor's to spread;
/// each score is computed from its own pose alone and the result does not depend on the thread
/// count (D §7).
pub fn scores(
    exec: &dyn Executor,
    target: &Target<'_>,
    probe: &Probe,
    hyp: &Hypotheses,
    delta: f64,
) -> Vec<f64> {
    // One mask over A's breakline for the whole pair (D §6.2, `spatial::grid`): tens of thousands
    // of hypotheses throw the same sixty points at the same curve, and the mask answers "nothing
    // within `delta`" for most of them without a tree descent. It is a filter — a `true` still
    // goes to `nearest_below` — so the scores are the scores.
    let mask = NearMask::of(target.points, delta);
    exec.coarse_scores(&CoarseBatch {
        target: *target,
        mask: mask.as_ref(),
        points: &probe.q,
        normals: &probe.qn,
        poses: Poses::Split { r: &hyp.r, tau: &hyp.tau },
        radius: delta,
        normal_agree: NORMAL_AGREE,
    })
}

/// R §5.4's `brk_score`: the same estimator as [`scores`], on one pose and a point set of the
/// caller's choosing.
///
/// Stage 1 re-scores each refined pose against A's breakline at `sc.stage1` (0.06 t) rather than
/// `sc.coarse` (0.15 t), and over the whole of B's `brk_sub` rather than sixty of it — a different
/// radius and a different point set, the same kernel (D §6.5). `points` and `normals` are B's
/// breakline points and shell normals at `brk_sub`, in that order.
///
/// No near mask: R §5.2 builds one because tens of thousands of poses share it, and here one pose
/// does.
pub fn score_pose(
    exec: &dyn Executor,
    target: &Target<'_>,
    points: &[[f64; 3]],
    normals: &[[f64; 3]],
    transform: &Matrix4<f64>,
    delta: f64,
) -> f64 {
    score_poses(exec, target, points, normals, std::slice::from_ref(transform), delta)[0]
}

/// [`score_pose`] for a batch of poses over the same points — the shape stage 1's re-score has
/// once R §5.5's candidates are refined together (D §6.4 step 4).
pub fn score_poses(
    exec: &dyn Executor,
    target: &Target<'_>,
    points: &[[f64; 3]],
    normals: &[[f64; 3]],
    transforms: &[Matrix4<f64>],
    delta: f64,
) -> Vec<f64> {
    exec.coarse_scores(&CoarseBatch {
        target: *target,
        mask: None,
        points,
        normals,
        poses: Poses::Homogeneous(transforms),
        radius: delta,
        normal_agree: NORMAL_AGREE,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "a score is k/n and the tests assert which k")]

    use super::{Probe, Target, scores};
    use crate::executor::CPU;
    use crate::matching::hypotheses::{Frames, build_with};
    use crate::spatial::kdtree::PointTree;

    /// A breakline of `n` points along the x axis at `x0 + k·spacing`, shells up.
    fn line(n: u32, x0: f64, spacing: f64, dih: f64) -> Frames {
        let p: Vec<[f64; 3]> = (0..n).map(|k| [x0 + spacing * f64::from(k), 0.0, 0.0]).collect();
        let count = p.len();
        Frames {
            ns: vec![[0.0, 0.0, 1.0]; count],
            f: vec![[-1.0, 0.0, 0.0]; count],
            tangent: vec![[0.0, -1.0, 0.0]; count],
            dih: vec![dih; count],
            sub: (0..n).collect(),
            p,
        }
    }

    /// The identity pose over two coincident breaklines scores 1, and the same pose after a shift
    /// larger than `delta` scores 0 — the score is the fraction that lands.
    #[test]
    fn a_pose_that_lays_the_curves_on_each_other_scores_one() {
        let a = line(10, 0.0, 1.0, 90.0);
        let b = line(10, 0.0, 1.0, 90.0);
        let tree = PointTree::build(&a.p).unwrap();
        let target = Target { points: &a.p, normals: &a.ns, tree: &tree };
        let probe = Probe::at(&b, &[0, 3, 7, 9]);

        // One hypothesis, the identity (the frames of a's and b's point 0 already coincide).
        let mut b_flip = b.clone();
        b_flip.f = vec![[1.0, 0.0, 0.0]; 10];
        b_flip.tangent = vec![[0.0, 1.0, 0.0]; 10];
        let hyp = build_with(&a, &b_flip, &[0], &[0], 25.0);
        assert_eq!(hyp.len(), 1);

        assert_eq!(scores(&CPU, &target, &probe, &hyp, 0.5), vec![1.0]);

        // The same probe moved off the curve by more than delta lands nowhere.
        let far = Probe {
            idx: probe.idx.clone(),
            q: probe.q.iter().map(|p| [p[0], p[1] + 0.6, p[2]]).collect(),
            qn: probe.qn.clone(),
        };
        assert_eq!(scores(&CPU, &target, &far, &hyp, 0.5), vec![0.0]);
    }

    /// The radius is scipy's `distance_upper_bound`, which is **exclusive**: a probe point exactly
    /// `delta` away is a miss.
    #[test]
    fn a_point_exactly_at_the_radius_is_a_miss() {
        let a = line(1, 0.0, 1.0, 90.0);
        let mut b = line(1, 0.0, 1.0, 90.0);
        b.f = vec![[1.0, 0.0, 0.0]];
        b.tangent = vec![[0.0, 1.0, 0.0]];
        let tree = PointTree::build(&a.p).unwrap();
        let target = Target { points: &a.p, normals: &a.ns, tree: &tree };
        let hyp = build_with(&a, &b, &[0], &[0], 25.0);

        let at =
            |dy: f64| Probe { idx: vec![0], q: vec![[0.0, dy, 0.0]], qn: vec![[0.0, 0.0, 1.0]] };
        assert_eq!(scores(&CPU, &target, &at(0.25), &hyp, 0.5), vec![1.0]);
        assert_eq!(
            scores(&CPU, &target, &at(0.5), &hyp, 0.5),
            vec![0.0],
            "exclusive at the radius"
        );
    }

    /// A pose that lands B's curve on A's with the shells facing opposite ways scores 0, however
    /// close the points are: the normal test is what makes this a seam test.
    #[test]
    fn opposed_shell_normals_do_not_agree() {
        let a = line(4, 0.0, 1.0, 90.0);
        let mut b = line(4, 0.0, 1.0, 90.0);
        b.f = vec![[1.0, 0.0, 0.0]; 4];
        b.tangent = vec![[0.0, 1.0, 0.0]; 4];
        let tree = PointTree::build(&a.p).unwrap();
        let target = Target { points: &a.p, normals: &a.ns, tree: &tree };
        let hyp = build_with(&a, &b, &[0], &[0], 25.0);

        let flipped = Probe {
            idx: vec![0, 1],
            q: vec![[0.0; 3], [1.0, 0.0, 0.0]],
            qn: vec![[0.0, 0.0, -1.0]; 2],
        };
        assert_eq!(scores(&CPU, &target, &flipped, &hyp, 0.5), vec![0.0]);

        // Just inside 45° is still agreement; just outside is not.
        let tilt = |c: f64| Probe {
            idx: vec![0],
            q: vec![[0.0; 3]],
            qn: vec![[(1.0 - c * c).sqrt(), 0.0, c]],
        };
        assert_eq!(scores(&CPU, &target, &tilt(0.71), &hyp, 0.5), vec![1.0]);
        assert_eq!(scores(&CPU, &target, &tilt(0.7), &hyp, 0.5), vec![0.0], "the test is `> 0.7`");
    }

    /// The score is a fraction of the probe, in steps of `1/n`.
    #[test]
    fn the_score_is_the_fraction_that_lands() {
        let a = line(4, 0.0, 1.0, 90.0);
        let mut b = line(4, 0.0, 1.0, 90.0);
        b.f = vec![[1.0, 0.0, 0.0]; 4];
        b.tangent = vec![[0.0, 1.0, 0.0]; 4];
        let tree = PointTree::build(&a.p).unwrap();
        let target = Target { points: &a.p, normals: &a.ns, tree: &tree };
        let hyp = build_with(&a, &b, &[0], &[0], 25.0);

        // Two of four probe points sit on the curve, two are far away.
        let probe = Probe {
            idx: vec![0, 1, 2, 3],
            q: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, -9.0, 0.0]],
            qn: vec![[0.0, 0.0, 1.0]; 4],
        };
        assert_eq!(scores(&CPU, &target, &probe, &hyp, 0.5), vec![0.5]);
        assert_eq!(
            scores(&CPU, &target, &Probe::default(), &hyp, 0.5),
            vec![0.0],
            "no probe, no score"
        );
    }

    /// The draw is R §5.2's: `min(points, |pool|)` distinct members of the pool, seeded, and the
    /// same seed gives the same set. It is the port's stream, not numpy's (PMC-9).
    #[test]
    fn the_probe_is_drawn_from_the_pool_without_replacement() {
        let b = line(200, 0.0, 1.0, 90.0);
        let pool: Vec<u32> = (0..200).step_by(2).collect();
        let probe = Probe::draw(&b, &pool, 60, 0);
        assert_eq!(probe.len(), 60);
        let mut sorted = probe.idx.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 60, "distinct");
        assert!(probe.idx.iter().all(|i| pool.contains(i)), "drawn from the pool");
        assert_eq!(probe.q[0], b.p[probe.idx[0] as usize]);
        assert_eq!(Probe::draw(&b, &pool, 60, 0).idx, probe.idx, "seeded");
        assert_ne!(Probe::draw(&b, &pool, 60, 1).idx, probe.idx, "and it uses the seed");

        // A pool smaller than the budget is taken whole.
        let small: Vec<u32> = (0..7).collect();
        let probe = Probe::draw(&b, &small, 60, 0);
        assert_eq!(probe.len(), 7);
    }
}
