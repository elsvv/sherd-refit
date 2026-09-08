//! [`CpuExecutor`] — the reference implementation of every [`Executor`] method (D §6.1).
//!
//! The four inner loops live here and nowhere else: R §5.2's coarse score, one rung of R §7's ICP,
//! R §6.1's bounded point-to-surface distance and R §6.4's inside test. Everything in
//! `matching::coarse`, `matching::ladder` and `matching::verify` that used to hold one of them now
//! forms a batch and calls through the trait, so that the GPU executor of phase 2b receives
//! exactly what the CPU received and the cross-check of D §10.4 layer 3 compares two answers to
//! one question.
//!
//! **This code is the phase-1 code, moved.** The arithmetic is not restated in a batched form,
//! reassociated or vectorised: R §5.2's probe loop still multiplies the same nine `f64` products
//! in the same order, R §7's rung is still one call of `icp::register` per candidate, and R §6.4's
//! two passes are still the two passes. That is the contract of task G1 — the CPU executor's
//! outputs stay byte-identical to `9bf35d6`'s — and a batch boundary that could not be drawn
//! without changing them would be the wrong boundary.
//!
//! # Where the threads are
//!
//! `coarse_scores` spreads over poses, `icp_rung` over candidates and `inside` over points, which
//! is where `matching::coarse`, `matching::pair` and `matching::verify` respectively put them
//! before the boundary existed. `bounded_distance` is sequential in both of its modes: R §6.1's
//! array is a `map` its caller already collected in order, and R §6.4's minimum is a scan whose
//! window shrinks to the best found so far, which is an ordering and not a schedule.

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use super::Executor;
use super::batch::{CoarseBatch, DistBatch, DistReduce, IcpBatch, InsideBatch, InsideOutcome};
use crate::matching::icp::{self, Registration};
use crate::spatial::bvh::RayScene;
use crate::spatial::grid::NearMask;

/// The CPU implementation of D §6.1's five inner loops.
///
/// Stateless: it holds no device, no cache and no configuration, so one value serves the whole
/// process and [`super::CPU`] is that value.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuExecutor;

impl Executor for CpuExecutor {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn coarse_scores(&self, batch: &CoarseBatch<'_>) -> Vec<f64> {
        if batch.points.is_empty() {
            // The reference would divide by zero here; it cannot reach this, because R §5 returns
            // before the coarse stage when B has no breakline and `brk_sub` is what the probe is
            // drawn from. A score of zero is the answer that ranks such a pair last rather than
            // NaN.
            return vec![0.0; batch.poses.len()];
        }
        // The reference's `mean` is a *division*, and `k · (1/60)` is not `k / 60`: 1/60 has no
        // exact double, so multiplying by the reciprocal moves the last bit of some scores.
        // Measured on pot_G, 52 of one pair's 40 029 hypotheses came out one ulp away before this
        // was a division.
        #[allow(clippy::cast_precision_loss, reason = "the probe is 60 points")]
        let points = batch.points.len() as f64;
        (0..batch.poses.len())
            .into_par_iter()
            .map(|h| {
                let (rot, tau) = batch.poses.get(h);
                let agree = agreeing(batch, &rot, &tau);
                f64::from(agree) / points
            })
            .collect()
    }

    fn icp_rung(&self, batch: &IcpBatch<'_>) -> Vec<Registration> {
        match batch.inits.len() {
            0 => Vec::new(),
            // One candidate is the shape R §9's refinement and every single-pose caller has; the
            // rayon bridge would cost more than the rung.
            1 => vec![icp::register(batch.source, batch.target, &batch.inits[0], &batch.options)],
            _ => batch
                .inits
                .par_iter()
                .map(|init| icp::register(batch.source, batch.target, init, &batch.options))
                .collect(),
        }
    }

    fn bounded_distance(&self, batch: &DistBatch<'_>) -> Vec<f64> {
        #[allow(clippy::cast_possible_truncation, reason = "the scene is f32, as Open3D's is")]
        let window = batch.max_dist as f32;
        match batch.reduce {
            DistReduce::All => batch
                .points
                .iter()
                .map(|p| {
                    batch
                        .scene
                        .bounded_distance(narrow(apply(batch.transform, *p)), window)
                        .map_or(f64::INFINITY, f64::from)
                })
                .collect(),
            DistReduce::Min => {
                // R §6.4's fallback: each query only has to beat the best distance so far, so the
                // window shrinks as the scan goes and the answer is the same `min` a full array
                // would give.
                let mut best = window;
                for p in batch.points {
                    if let Some(distance) =
                        batch.scene.bounded_distance(narrow(apply(batch.transform, *p)), best)
                    {
                        best = best.min(distance);
                    }
                }
                vec![f64::from(best)]
            }
        }
    }

    fn inside(&self, batch: &InsideBatch<'_>) -> Vec<InsideOutcome> {
        if batch.points.is_empty() {
            return Vec::new();
        }
        let (lo, hi) = batch.scene.aabb();
        batch
            .points
            .par_iter()
            .map(|p| {
                depth_inside(narrow(apply(batch.transform, *p)), batch.scene, lo, hi)
                    .map_or(InsideOutcome::OUTSIDE, InsideOutcome::inside_at)
            })
            .collect()
    }
}

/// R §5.2's inner loop for one pose: how many of the batch's points land on the target breakline
/// with an agreeing shell normal.
///
/// The mask can only turn a query that would have missed into a query that is not made; a `true`
/// falls through to the same `nearest_below` call, so the count is the same count.
fn agreeing(batch: &CoarseBatch<'_>, rot: &Matrix3<f64>, tau: &Vector3<f64>) -> u32 {
    let (target_normals, tree) = (batch.target.normals, batch.target.tree);
    let mut agree = 0_u32;
    for (point, normal) in batch.points.iter().zip(batch.normals) {
        let moved = [
            rot[(0, 0)] * point[0] + rot[(0, 1)] * point[1] + rot[(0, 2)] * point[2] + tau[0],
            rot[(1, 0)] * point[0] + rot[(1, 1)] * point[1] + rot[(1, 2)] * point[2] + tau[1],
            rot[(2, 0)] * point[0] + rot[(2, 1)] * point[1] + rot[(2, 2)] * point[2] + tau[2],
        ];
        // scipy's `distance_upper_bound` is *exclusive* — a neighbour exactly at `radius` comes
        // back as `inf`, which is what `np.isfinite(d)` then reads — and the bound is also what
        // makes this stage affordable: most probe points of most poses land nowhere near A's
        // breakline, and a bounded search abandons those in a few comparisons. `nearest_below` is
        // both halves, and its radius is widened by the rounding of `radius · radius` so that the
        // traversal can never drop a neighbour the strict test would have kept.
        if skipped(batch.mask, &moved) {
            continue;
        }
        let Some((near, _)) = tree.nearest_below(&moved, batch.radius) else {
            continue;
        };
        let turned = [
            rot[(0, 0)] * normal[0] + rot[(0, 1)] * normal[1] + rot[(0, 2)] * normal[2],
            rot[(1, 0)] * normal[0] + rot[(1, 1)] * normal[1] + rot[(1, 2)] * normal[2],
            rot[(2, 0)] * normal[0] + rot[(2, 1)] * normal[1] + rot[(2, 2)] * normal[2],
        ];
        let theirs = target_normals[near as usize];
        let dot = theirs[0] * turned[0] + theirs[1] * turned[1] + theirs[2] * turned[2];
        if dot > batch.normal_agree {
            agree += 1;
        }
    }
    agree
}

/// Whether the near mask rules this query out before the tree is asked.
#[inline]
fn skipped(mask: Option<&NearMask>, moved: &[f64; 3]) -> bool {
    mask.is_some_and(|mask| !mask.may_be_near(moved))
}

/// How deep one already-moved point sits inside a mesh, or `None` when it is outside it.
///
/// `−sd` of Open3D's signed distance for a point the parity test calls inside, with the AABB
/// reject in front of it: a point outside the box is outside the mesh, and no ray is cast for it.
#[inline]
fn depth_inside(point: [f32; 3], scene: &RayScene, lo: [f32; 3], hi: [f32; 3]) -> Option<f64> {
    let outside_box = (0..3).any(|k| point[k] < lo[k] || point[k] > hi[k]);
    if outside_box || !scene.inside(point) {
        return None;
    }
    Some(f64::from(scene.distance(point)))
}

/// `T · p`, in `f64`.
#[inline]
fn apply(t: &nalgebra::Matrix4<f64>, p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| t[(i, 0)] * p[0] + t[(i, 1)] * p[1] + t[(i, 2)] * p[2] + t[(i, 3)])
}

/// The `f32` the BVH is built in (PMC-15, R §0).
#[inline]
#[allow(clippy::cast_possible_truncation, reason = "the scene is f32, as Open3D's is")]
fn narrow(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}
