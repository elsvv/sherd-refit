//! Every distance one pair of fragments is measured with (R §1.2, R §4.2).
//!
//! The matcher never sees a bare `t`. Each threshold of R §1.1 arrives as a pair `(k, m)` and is
//! resolved once, here, into `max(k·t_pair, m·res_pair)`: `k·t` is the scale-free half — a wall
//! thickness means the same thing on a 39-unit terracotta relief and on a 3.5 mm pot wall — and
//! `m·res` is the floor that stops the matcher demanding a precision the triangles cannot carry.
//! On a mesh with more than `k/m` edges across the wall the first term wins and the floor is
//! inert, which is how the thick terracotta set behaves throughout.
//!
//! The pair supplies both numbers itself (R §4.2): `t_pair = min(t_A, t_B)`, because a fragment
//! carrying the pot's rim measures a thicker wall than the body it broke off and the wall is the
//! thinner of the two; `res_pair = max(res_A, res_B)`, because the coarser of the two meshes is
//! what limits how precisely the pair can be told to fit. No collection-wide median is involved —
//! one collection can hold walls of 2.4 mm and 13.5 mm at once and its median describes neither.

use serde::{Deserialize, Serialize};

use crate::fragment::Fragment;
use crate::params::Params;

/// The resolved distances of one pair, all in the meshes' own length unit.
///
/// Built once per pair by [`Scales::for_pair`] and passed down instead of a `t`, so that R §1.1's
/// two-term rule lives in exactly one place.
///
/// The field names are the reference's, because the fixture dump writes this struct out per pair
/// (`scales.json`, D §10.1) and the parity harness reads it back into this type.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scales {
    /// `t_pair = min(t_A, t_B)` — the unit every score is reported in.
    pub t: f64,
    /// `res_pair = max(res_A, res_B)`.
    pub res: f64,
    /// R §5.2's breakline proximity for the coarse score.
    pub coarse: f64,
    /// R §5.4's breakline proximity when a stage-1 pose is re-scored.
    pub stage1: f64,
    /// R §6.1's tight-contact distance.
    pub tight: f64,
    /// R §6.1's window that selects the fracture points facing the other fragment.
    pub facing: f64,
    /// R §6.5's gap limit.
    pub gap: f64,
    /// R §6.2's breakline proximity counted as a shared seam.
    pub seam: f64,
    /// R §6.3's shell-margin radius for the continuity test.
    pub near: f64,
    /// R §6.4's penetration depth.
    pub pen: f64,
    /// R §5.3's translation radius of the non-maximum suppression — `0.5·t`, with **no**
    /// resolution floor, which is the one threshold of R §1.2 that has none.
    pub nms: f64,
    /// The dimensionless factor (`≥ 1`) by which the whole ICP ladder of R §5.4–5.6 is stretched.
    pub icp: f64,
}

/// The two acceptance limits that depend on the pair, in units of `t` (R §1.2, R §11.2).
///
/// They travel with every candidate because a report read months later has no other way of
/// knowing how tight "tight" was on that pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// `gap / t_pair` — what R §6.5 compares the reported `gap` against.
    pub gap_limit: f64,
    /// `tight / t_pair` — the distance R §6.1's tight fraction was counted within.
    pub tight_delta: f64,
}

impl Scales {
    /// R §1.2 for a pair whose wall is `t` and whose resolution is `res`.
    pub fn for_pair(p: &Params, t: f64, res: f64) -> Self {
        let f = |k: f64, m: f64| (k * t).max(m * res);
        Self {
            t,
            res,
            coarse: f(p.coarse_delta, p.coarse_res),
            stage1: f(p.stage1_delta, p.stage1_res),
            tight: f(p.tight_delta, p.tight_res),
            facing: f(p.facing_delta, p.facing_res),
            gap: f(p.max_gap, p.gap_res),
            seam: f(p.seam_delta, p.seam_res),
            near: f(p.near_delta, p.near_res),
            pen: f(p.pen_delta, p.pen_res),
            nms: p.nms_delta * t,
            icp: f(p.icp_delta, p.icp_res) / (p.icp_delta * t),
        }
    }

    /// R §4.2's `t_pair` and `res_pair` for two fragments, then [`Scales::for_pair`].
    pub fn for_fragments(p: &Params, a: &Fragment, b: &Fragment) -> Self {
        Self::for_pair(p, a.thick.min(b.thick), a.res().max(b.res()))
    }

    /// One rung of R §5.4–5.6's ICP ladder: `k·t`, stretched by [`icp`](Scales::icp).
    ///
    /// The ladder is stretched as a whole rather than floored rung by rung, so that its steps keep
    /// their ratios and a coarse mesh cannot make the fine rung overtake the coarse one.
    #[inline]
    pub fn icp_dist(&self, k: f64) -> f64 {
        k * self.t * self.icp
    }

    /// The two per-pair limits every candidate carries (R §11.2).
    #[inline]
    pub fn limits(&self) -> Limits {
        Limits { gap_limit: self.gap / self.t, tight_delta: self.tight / self.t }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "R §1.2 is arithmetic on two doubles, asserted exactly")]

    use super::Scales;
    use crate::params::Params;

    /// R §1.1's claim about the terracotta set, as arithmetic: every floor is set just below
    /// `k/0.058`, and `0.058 t` is the coarsest working mesh that set produces, so at 17.2 edges
    /// per `t` no floor binds and `Scales` is `k·t` throughout.
    #[test]
    fn thick_walls_leave_every_resolution_floor_inert() {
        let p = Params::default();
        let t = 39.0;
        let res = 0.058 * t;
        let sc = Scales::for_pair(&p, t, res);

        assert!((sc.coarse - 0.15 * t).abs() < 1e-12);
        assert!((sc.stage1 - 0.06 * t).abs() < 1e-12);
        assert!((sc.tight - 0.01 * t).abs() < 1e-12);
        assert!((sc.facing - 0.3 * t).abs() < 1e-12);
        assert!((sc.gap - 0.03 * t).abs() < 1e-12);
        assert!((sc.seam - 0.12 * t).abs() < 1e-12);
        assert!((sc.near - 0.5 * t).abs() < 1e-12);
        assert!((sc.pen - 0.06 * t).abs() < 1e-12);
        assert!((sc.nms - 0.5 * t).abs() < 1e-12);
        // With no floor binding the ladder is not stretched at all.
        assert!((sc.icp - 1.0).abs() < 1e-12, "{}", sc.icp);
        assert!((sc.icp_dist(0.2) - 0.2 * t).abs() < 1e-12);
    }

    /// These are the numbers the slab pair's `scales.json` carries, and they are reproduced to the
    /// last bit rather than to a tolerance: `Scales` is arithmetic on two doubles.
    ///
    /// The slab is 13.2 edges per `t`, below the 17.2 the floors cross over at, so on this pair
    /// every floored threshold *is* `m·res` — which is why the numbers below are not multiples of
    /// `t` and why the floors are exercised here rather than only in the synthetic test above.
    #[test]
    fn the_slab_pairs_scales_are_the_references_own() {
        let sc =
            Scales::for_pair(&Params::default(), 30.127_979_278_564_453, 2.278_450_132_720_530_4);
        assert_eq!(sc.coarse, 5.240_435_305_257_219);
        assert_eq!(sc.stage1, 2.050_605_119_448_477_3);
        assert_eq!(sc.tight, 0.341_767_519_908_079_6);
        assert_eq!(sc.facing, 9.038_393_783_569_335);
        assert_eq!(sc.gap, 1.025_302_559_724_238_7);
        assert_eq!(sc.seam, 4.101_210_238_896_955);
        assert_eq!(sc.near, 15.063_989_639_282_227);
        assert_eq!(sc.pen, 2.050_605_119_448_477_3);
        assert_eq!(sc.nms, 15.063_989_639_282_227);
        assert_eq!(sc.icp, 1.134_385_803_800_792_6);
    }

    /// A wall thin enough for the floors to bite: every threshold becomes `m·res`, the ladder is
    /// stretched by the ratio, and `nms` — the one with no floor — does not move.
    #[test]
    fn a_thin_wall_on_a_coarse_mesh_lifts_every_floored_threshold() {
        let p = Params::default();
        let (t, res) = (1.0, 1.0);
        let sc = Scales::for_pair(&p, t, res);

        assert_eq!(sc.coarse, p.coarse_res);
        assert_eq!(sc.stage1, p.stage1_res);
        assert_eq!(sc.tight, p.tight_res);
        assert_eq!(sc.facing, p.facing_res);
        assert_eq!(sc.gap, p.gap_res);
        assert_eq!(sc.seam, p.seam_res);
        assert_eq!(sc.near, p.near_res);
        assert_eq!(sc.pen, p.pen_res);
        assert_eq!(sc.nms, 0.5, "R §1.2: `nms` has no resolution floor");
        assert_eq!(sc.icp, p.icp_res / p.icp_delta);
        // The rungs keep their ratios: the ladder is stretched, not floored rung by rung.
        assert!((sc.icp_dist(0.2) / sc.icp_dist(0.04) - 5.0).abs() < 1e-12);
    }

    /// R §4.2: the thinner wall and the coarser mesh, and both from the pair alone.
    #[test]
    fn a_pair_takes_the_thinner_wall_and_the_coarser_mesh() {
        let p = Params::default();
        let sc = Scales::for_pair(&p, 3.0_f64.min(5.0), 0.1_f64.max(0.4));
        assert_eq!(sc.t, 3.0);
        assert_eq!(sc.res, 0.4);
        let l = sc.limits();
        assert!((l.gap_limit - sc.gap / 3.0).abs() < 1e-15);
        assert!((l.tight_delta - sc.tight / 3.0).abs() < 1e-15);
    }
}
