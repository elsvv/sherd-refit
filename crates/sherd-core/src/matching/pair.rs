//! One pair of fragments, up to the candidate set (R §4.1–4.2, R §5.1–5.3).
//!
//! A pair is the unit of work of the matcher, and R §4.2 says what a pair *is*: the two fragments'
//! match arrays rebuilt at `t_pair = min(t_A, t_B)`, plus the [`Scales`] both are measured with.
//! The fragment whose own `t` is `t_pair` uses its cached arrays; the other rebuilds R §3.5 at
//! `t_pair`, which is why this type exists at all — nothing below it may assume a fragment is at
//! its own wall thickness.
//!
//! Step C1 fills in the first half of `match_pair`: [`Pair::hypotheses`] (R §5.1),
//! [`Pair::coarse`] (R §5.2) and [`Pair::suppress`] (R §5.3). The ICP ladder, the verification and
//! the ranking follow in step C2, on top of exactly this state.

use crate::fragment::Fragment;
use crate::fragment::samples::MatchData;
use crate::matching::coarse::{self, Probe, Target};
use crate::matching::hypotheses::{self, Frames, Hypotheses};
use crate::matching::nms;
use crate::matching::scales::Scales;
use crate::params::Params;

/// R §5.3's cap on the hypotheses the coarse NMS walks.
///
/// The score is a fraction of sixty points, so below the five thousandth-best hypothesis the
/// ranking carries no information the ICP could use; the truncation is what keeps NMS linear in
/// the kept count rather than in the hypothesis count.
pub const COARSE_ORDER_LIMIT: usize = 5000;
/// R §5.3's floor under the coarse score.
pub const COARSE_FLOOR: f64 = 0.1;

/// Two fragments and everything R §5 measures them with.
#[derive(Debug)]
pub struct Pair<'a> {
    /// A's match arrays at `t_pair` (R §4.2).
    pub a: MatchData<'a>,
    /// B's match arrays at `t_pair`.
    pub b: MatchData<'a>,
    /// The pair's resolved distances (R §1.2).
    pub scales: Scales,
    /// A's breakline frames in `f64` (R §5.1).
    pub frames_a: Frames,
    /// B's breakline frames in `f64`.
    pub frames_b: Frames,
}

impl<'a> Pair<'a> {
    /// R §4.1's wall-ratio test: a pair whose walls differ by more than `thick_ratio` is not
    /// matched at all.
    ///
    /// Two sherds of one vessel have the same wall to within the accuracy of R §3.2; a factor of
    /// two and a half apart means one of them is a rim, a base or another pot, and the thresholds
    /// of R §1.2 — all of them in units of the *thinner* wall — would describe neither.
    pub fn skipped(a: &Fragment, b: &Fragment, p: &Params) -> bool {
        if !(a.thick > 0.0 && b.thick > 0.0) {
            return true;
        }
        let ratio = a.thick / b.thick;
        ratio > p.thick_ratio || ratio < 1.0 / p.thick_ratio
    }

    /// R §4.2: both fragments' arrays at `t_pair`, and the pair's scales.
    ///
    /// Building this is the expensive part of a pair — one of the two fragments usually has to
    /// redraw R §3.5 — and it is why the pipeline groups a fragment's pairs together (D §5).
    pub fn build(a: &'a Fragment, b: &'a Fragment, p: &Params) -> Self {
        let scales = Scales::for_fragments(p, a, b);
        let reg_points = p.reg_points as usize;
        let a = MatchData::at(a, scales.t, reg_points);
        let b = MatchData::at(b, scales.t, reg_points);
        Self { frames_a: Frames::of(&a), frames_b: Frames::of(&b), a, b, scales }
    }

    /// R §5's first exit: both fragments need a fracture sample and a breakline.
    #[inline]
    pub fn matchable(&self) -> bool {
        self.a.matchable() && self.b.matchable()
    }

    /// R §5.1's hypotheses over the two fragments' own subsets.
    pub fn hypotheses(&self, p: &Params) -> Hypotheses {
        hypotheses::build(&self.frames_a, &self.frames_b, p)
    }

    /// R §5.2's probe: the sixty points of B every hypothesis is scored on.
    pub fn probe(&self, p: &Params) -> Probe {
        Probe::draw(&self.frames_b, &self.frames_b.sub, p.coarse_points as usize, p.seed)
    }

    /// R §5.2's coarse score of every hypothesis.
    pub fn coarse(&self, hyp: &Hypotheses, probe: &Probe) -> Vec<f64> {
        let Some(tree) = self.a.kd_brk.as_ref() else {
            return vec![0.0; hyp.len()];
        };
        let target = Target { points: &self.frames_a.p, normals: &self.frames_a.ns, tree };
        coarse::scores(&target, probe, hyp, self.scales.coarse)
    }

    /// R §5.3's non-maximum suppression over the coarse scores: the poses stage 1 refines.
    pub fn suppress(&self, hyp: &Hypotheses, cs: &[f64], p: &Params) -> Vec<u32> {
        let order = nms::order_by_score(cs, COARSE_ORDER_LIMIT);
        nms::nms(&order, &hyp.r, &hyp.tau, cs, self.scales.nms, p.stage1 as usize, COARSE_FLOOR)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "R §4.2 selects one of two doubles; there is no rounding")]

    use super::Pair;
    use crate::fragment::Fragment;
    use crate::params::Params;
    use std::path::{Path, PathBuf};

    fn slab(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input").join(name)
    }

    /// R §4.1's ratio is on the wall thicknesses and it is symmetric in effect: the same pair is
    /// skipped whichever way round it is written.
    #[test]
    fn a_pair_of_very_different_walls_is_skipped() {
        let mut a = Fragment::from_mesh_file(slab("pieceA.ply"), 200_000).unwrap();
        let mut b = a.clone();
        let p = Params::default();

        a.thick = 3.0;
        b.thick = 7.0;
        assert!(!Pair::skipped(&a, &b, &p), "7/3 = 2.33 is inside the ratio of 2.5");
        assert!(!Pair::skipped(&b, &a, &p));
        b.thick = 7.6;
        assert!(Pair::skipped(&a, &b, &p), "7.6/3 = 2.53 is outside it");
        assert!(Pair::skipped(&b, &a, &p), "and the test does not depend on the order");
        b.thick = 0.0;
        assert!(Pair::skipped(&a, &b, &p), "a fragment with no wall has no scale");
    }

    /// R §4.2: the pair is built at the thinner wall, and the fragment that is not at it has its
    /// arrays rebuilt while the other keeps its own.
    #[test]
    fn a_pair_is_built_at_the_thinner_wall() {
        let a = Fragment::from_mesh_file(slab("pieceA.ply"), 200_000).unwrap();
        let b = Fragment::from_mesh_file(slab("pieceB.ply"), 200_000).unwrap();
        let p = Params::default();
        let pair = Pair::build(&a, &b, &p);

        assert_eq!(pair.scales.t, a.thick.min(b.thick));
        assert_eq!(pair.scales.res, a.res().max(b.res()));
        assert_eq!(pair.a.t, pair.scales.t);
        assert_eq!(pair.b.t, pair.scales.t);
        assert!(pair.matchable());
        assert_eq!(pair.frames_a.len(), pair.a.brk.len());
        assert_eq!(pair.frames_b.sub, pair.b.brk.sub);
    }
}
