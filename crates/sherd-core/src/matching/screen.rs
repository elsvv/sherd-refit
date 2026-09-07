//! The optional partner screening of R §4.3 — off by default (`screen_top_k = 0`).
//!
//! Matching is quadratic in the collection and a pair costs seconds; a 164-fragment collection is
//! 13 366 pairs and 40 CPU-hours. The screening pass runs the matcher's own coarse stage on a
//! capped subsample of both breaklines — `screen_points` points a side instead of a thousand,
//! which makes the hypothesis count `≈ 0.3·screen_points²` instead of tens of thousands — and
//! keeps each fragment's `screen_top_k` best-scoring partners. Measured on `mixed_all`: 0.17
//! CPU-seconds a pair, five minutes for the whole collection.
//!
//! # It ranks a fragment's partners; it does not decide whether they touched
//!
//! [`screen_pair`] returns the best coarse score of the capped hypothesis set, and that number
//! separates almost nothing on its own: a true adjacent pair overlaps on 5–22 % of its breakline
//! even at the ground-truth pose, because one seam is a fraction of a fragment's whole crack line,
//! and the best of several thousand poses reaches the same 10–28 % on pairs that never touched.
//! Keeping the best fifteen partners per fragment keeps 47 % of `mixed_all`'s ground-truth
//! adjacent pairs. The whole measurement, and three other partner searches that do no better, are
//! in `docs/superpowers/notes/2026-09-06-scale-pairs.md`.
//!
//! That is why [`top_partners`] ranks **within a fragment's own row** rather than against one
//! global threshold: a large sherd with a long breakline scores lower against everything than a
//! small one whose whole crack line can lie on its partner, so the scores are not comparable
//! across fragments, and a fragment with few neighbours costs only the `k` pairs it keeps.

use std::collections::{BTreeMap, BTreeSet};

use crate::matching::coarse::{self, Probe, Target};
use crate::matching::hypotheses;
use crate::matching::scales::Scales;
use crate::params::Params;
use crate::rng::{self, Draw};
use crate::spatial::kdtree::PointTree;

/// One fragment as the screening pass reads it: its frames at its **own** `t`, and a tree over
/// its whole breakline.
///
/// R §4.3 is explicit that the screening runs on each fragment's own arrays rather than on a
/// pair's rebuild at `t_pair` — the pass exists to avoid building pairs, so it may not build one.
#[derive(Debug)]
pub struct Screened<'a> {
    /// The fragment's breakline frames (R §3.6).
    pub frames: &'a hypotheses::Frames,
    /// A KD-tree over `frames.p`.
    pub tree: &'a PointTree,
    /// The fragment's own wall thickness (R §3.2).
    pub thick: f64,
    /// The fragment's own working-mesh resolution (R §3.3).
    pub res: f64,
}

/// R §4.3's `_cap`: at most `n` of `idx`, drawn without replacement and returned **sorted**.
///
/// Sorted because the cap is a subsample of a curve and the frames are paired in the order the
/// subsets are given (PMC-4); the reference sorts for the same reason. The generator is the port's
/// own stream (PMC-9), so *which* points are kept differs from the reference's — a screening pass
/// is a ranking, and this is one of the two places R §10 says the two implementations diverge by
/// design.
pub fn cap(idx: &[u32], n: usize, seed: u64) -> Vec<u32> {
    if n == 0 || idx.len() <= n {
        return idx.to_vec();
    }
    let mut rng = rng::seeded_for(seed, Draw::ScreenSubset);
    let mut picked: Vec<u32> = rng::without_replacement(idx.len(), n, &mut rng)
        .into_iter()
        .map(|k| idx[k as usize])
        .collect();
    picked.sort_unstable();
    picked
}

/// R §4.3: how well the two breaklines can be laid on top of each other, cheaply.
///
/// The matcher's own R §5.1 and R §5.2 over the two capped subsets, with the probe drawn from the
/// capped `ib` rather than from the whole of B's subset. Zero when either fragment has no usable
/// breakline or the dihedral filter keeps nothing.
pub fn screen_pair(a: &Screened<'_>, b: &Screened<'_>, p: &Params) -> f64 {
    if a.frames.sub.is_empty() || b.frames.sub.is_empty() {
        return 0.0;
    }
    let points = p.screen_points as usize;
    let (ia, ib) = (cap(&a.frames.sub, points, p.seed), cap(&b.frames.sub, points, p.seed));
    let hyp = hypotheses::build_with(a.frames, b.frames, &ia, &ib, p.dihedral_tol);
    if hyp.is_empty() {
        return 0.0;
    }
    let sc = Scales::for_pair(p, a.thick.min(b.thick), a.res.max(b.res));
    let probe = Probe::draw(b.frames, &ib, p.coarse_points as usize, p.seed);
    let target = Target { points: &a.frames.p, normals: &a.frames.ns, tree: a.tree };
    coarse::scores(&target, &probe, &hyp, sc.coarse).into_iter().fold(0.0_f64, f64::max)
}

/// R §4.3's `top_partners`: the union over fragments of each fragment's `k` best-scoring partners.
///
/// `score` is keyed by the pair as R §4.1 writes it — `(a, b)` with `a` before `b` in collection
/// order — and the result is keyed the same way, so a caller can test its own pair list against
/// it. A pair is kept when **either** endpoint keeps it. Ties inside one fragment's row go to the
/// partner that comes first by name, which is the reference's `sorted(..., key=(-v, name))`.
pub fn top_partners(
    score: &BTreeMap<(String, String), f64>,
    names: &[String],
    k: usize,
) -> BTreeSet<(String, String)> {
    let order: BTreeMap<&str, usize> =
        names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let mut per: BTreeMap<&str, Vec<(f64, &str)>> =
        names.iter().map(|n| (n.as_str(), Vec::new())).collect();
    for ((a, b), &v) in score {
        per.entry(a.as_str()).or_default().push((v, b.as_str()));
        per.entry(b.as_str()).or_default().push((v, a.as_str()));
    }
    let mut keep = BTreeSet::new();
    for (name, mut partners) in per {
        partners.sort_by(|x, y| {
            y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal).then(x.1.cmp(y.1))
        });
        for (_, partner) in partners.into_iter().take(k) {
            let (first, second) = if order.get(name) < order.get(partner) {
                (name, partner)
            } else {
                (partner, name)
            };
            keep.insert((first.to_owned(), second.to_owned()));
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::{cap, top_partners};
    use std::collections::BTreeMap;

    /// R §4.3's cap is a subset, is sorted, and leaves a subset that already fits untouched — the
    /// last of which is what keeps a small collection's screening identical to no screening.
    #[test]
    fn the_cap_is_a_sorted_subset_and_a_no_op_when_it_fits() {
        let idx: Vec<u32> = (0..500).map(|i| i * 3).collect();
        assert_eq!(cap(&idx, 500, 0), idx, "exactly the size it is");
        assert_eq!(cap(&idx, 900, 0), idx);
        assert_eq!(cap(&idx, 0, 0), idx, "0 means no cap, as `screen_points = 0` would");

        let capped = cap(&idx, 150, 0);
        assert_eq!(capped.len(), 150);
        assert!(capped.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
        assert!(capped.iter().all(|i| idx.contains(i)));
        assert_eq!(capped, cap(&idx, 150, 0), "the draw is a function of the seed");
        assert_ne!(capped, cap(&idx, 150, 1), "and of nothing else");
    }

    /// Each fragment keeps its own best `k`, a pair kept by either endpoint survives, and the pair
    /// is written the way R §4.1 writes it.
    #[test]
    fn the_partners_are_ranked_inside_each_fragments_own_row() {
        let names: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| (*s).to_owned()).collect();
        let pair = |x: &str, y: &str| (x.to_owned(), y.to_owned());
        let score: BTreeMap<(String, String), f64> = [
            (pair("a", "b"), 0.9),
            (pair("a", "c"), 0.5),
            (pair("a", "d"), 0.1),
            (pair("b", "c"), 0.4),
            (pair("b", "d"), 0.3),
            (pair("c", "d"), 0.8),
        ]
        .into_iter()
        .collect();

        let keep = top_partners(&score, &names, 1);
        // a keeps b (0.9); b keeps a; c keeps d (0.8); d keeps c. Two pairs, in R §4.1's order.
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&pair("a", "b")) && keep.contains(&pair("c", "d")));

        // `d`'s best partner is `c`, and `c`'s row does not need `d` for the pair to survive:
        // with k = 1 the pair a-d is dropped even though it is a's third choice.
        assert!(!keep.contains(&pair("a", "d")));

        let keep = top_partners(&score, &names, 3);
        assert_eq!(keep.len(), 6, "every pair is somebody's top three");
    }

    /// A tie inside one row goes to the partner whose name comes first.
    #[test]
    fn a_tie_inside_a_row_is_broken_by_name() {
        let names: Vec<String> = ["a", "b", "c"].iter().map(|s| (*s).to_owned()).collect();
        let pair = |x: &str, y: &str| (x.to_owned(), y.to_owned());
        let score: BTreeMap<(String, String), f64> =
            [(pair("a", "b"), 0.5), (pair("a", "c"), 0.5)].into_iter().collect();
        let keep = top_partners(&score, &names, 1);
        assert!(keep.contains(&pair("a", "b")), "b before c at equal score");
        // `c`'s own row keeps `a`, so the pair survives from the other side.
        assert!(keep.contains(&pair("a", "c")));
    }
}
