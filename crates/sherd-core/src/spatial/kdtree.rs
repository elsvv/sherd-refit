//! Radius-bounded and unbounded nearest neighbours on the CPU (D §6.2, experiment E3).
//!
//! `kiddo` 6.2's `ImmutableKdTree` is what E3 chose: 79–251 ns per bounded query on the benchmark
//! clouds, zero errors against brute force over ~0.9 M bounded queries, and one build per cloud
//! instead of one grid per ICP ladder rung
//! (`docs/superpowers/notes/2026-09-06-e3e4-spatial.md` §5).
//!
//! [`PointTree`] is the one wrapper the algorithm needs, and it is deliberately `f64`. Its callers
//! in R §3.4 and R §3.5 — `coarse_grid`'s nearest representative, `ball_matrix`'s radius query,
//! and the breakline distance `0.15·t` selects the macro-normal annulus by —
//! run on the `f64` face centroids the reference computes, and scipy's `cKDTree` compares squared
//! `f64` distances against `r²`. Narrowing the coordinates to `f32` would move a point across the
//! ball boundary whenever `|C − q|` sits within an `f32` ulp of the radius, which on a 200 000-face
//! mesh is a handful of faces per fragment; `f64` removes the question. `kiddo` is generic over its
//! axis type, so this costs nothing but the memory.
//!
//! Two rules the rest of the port relies on:
//!
//! * [`PointTree::within`] returns its neighbours **sorted by index**, not by distance, so every
//!   sum over a neighbourhood is accumulated in the same order on every machine (D §7). scipy's
//!   `query_ball_point(..., return_sorted=False)` returns them in an unspecified order, so the
//!   reference's own sums are *not* reproducible bit for bit; ours are.
//! * the radius test is inclusive (`d ≤ r`), which is `query_ball_point`'s.

use std::num::NonZeroUsize;

use kiddo::{ImmutableKdTree, SquaredEuclidean};

/// A KD-tree over `f64` points, built once and queried many times.
#[derive(Debug)]
pub struct PointTree {
    tree: ImmutableKdTree<f64, 3>,
    len: usize,
}

impl PointTree {
    /// Builds the tree. Returns `None` for an empty point set, which `kiddo` refuses.
    pub fn build(points: &[[f64; 3]]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        ImmutableKdTree::new_from_slice(points).ok().map(|tree| Self { tree, len: points.len() })
    }

    /// Index of the point nearest to `query`, ties going to the lowest index.
    ///
    /// This is `cKDTree.query(x)[1]`: an unbounded search, so it always answers on a non-empty
    /// tree. `kiddo` resolves exact ties to the lowest index on every case E3 could construct,
    /// though it does not document that as a guarantee; nothing in R depends on which of two
    /// coincident points is returned.
    pub fn nearest(&self, query: &[f64; 3]) -> u32 {
        self.nearest_distance(query).0
    }

    /// The nearest point and how far away it is — `cKDTree.query(x)` in full.
    ///
    /// The distance is the Euclidean one, `sqrt` of the squared distance the search minimises, so
    /// it is scipy's number to the last bit on any pair of coordinates where the two libraries
    /// accumulate `Σ(a−b)²` the same way. R §3.5.4 and R §3.5.6 compare it against `0.15·t`,
    /// `0.12·t` and `1.5·t`, thresholds no real distance sits on.
    pub fn nearest_distance(&self, query: &[f64; 3]) -> (u32, f64) {
        let hit = self.tree.query(query).nearest_one::<SquaredEuclidean<f64>>().execute();
        (hit.item, hit.distance.sqrt())
    }

    /// scipy's `query(x, distance_upper_bound=r)`: the nearest point within `radius`, or `None`
    /// when there is none — which is the `inf` the reference then reads as a miss.
    ///
    /// The bound is what makes R §5.2 affordable. An unbounded nearest-neighbour search has to
    /// find the true nearest however far away it is, and most of the millions of probe points a
    /// pair's hypotheses throw at a breakline are far away; with the radius the traversal prunes
    /// at the first node whose box is further than `radius` and a miss costs a handful of
    /// comparisons. Measured on the coarse stage over synthetic_20's 190 pairs and their 1.67 G
    /// probe queries: **1.08 µs per query unbounded against 0.11 µs bounded**, the whole stage
    /// 1 800 core-seconds against 178, with bit-identical scores.
    ///
    /// The radius test is `d² ≤ radius·radius`, evaluated in squared units — which is *not* the
    /// same as `d ≤ radius`. `radius · radius` rounds, so a point whose distance is exactly
    /// `radius` can have a squared distance a bit above the rounded square and come back as a
    /// miss. Callers that need the reference's own `d < r` want [`PointTree::nearest_below`],
    /// which widens the square by the rounding before it applies the test.
    pub fn nearest_within(&self, query: &[f64; 3], radius: f64) -> Option<(u32, f64)> {
        if radius < 0.0 {
            return None;
        }
        self.nearest_within_squared(query, radius * radius).map(|(i, d2)| (i, d2.sqrt()))
    }

    /// `cKDTree.query(x)` followed by the reference's own `d < bound`: the nearest point when it is
    /// strictly nearer than `bound`, and `None` otherwise.
    ///
    /// **This is the unbounded query's answer, and the bound is only a traversal hint.** R §5.2,
    /// R §6.2 and R §6.3 all read `d, j = tree.query(...)` — an unbounded search — and then test
    /// `d < r`; a port that searches within `r` instead has to show that the two cannot differ.
    ///
    /// *Pruning* cannot change the winner: a node is dropped only when its box is further than the
    /// search radius, and no such node can hold a point at the minimum distance when that minimum
    /// is itself under the radius, so every candidate at the minimum — ties included — is visited
    /// either way.
    ///
    /// *Rounding* cannot change it either, and here the square is widened so that it cannot for a
    /// reason that fits on one line: `sqrt(x) < bound` implies `x < bound²` exactly, because `sqrt`
    /// is correctly rounded and monotone, and `(bound·bound)·(1 + 4ε)` is above `bound²` for every
    /// finite `bound` — so a point the strict test would accept is never outside the searched ball,
    /// and whatever the widening lets in beyond `bound` that same strict test drops.
    ///
    /// Calling [`PointTree::nearest_within`] and applying `d < bound` afterwards gives the same
    /// answers, and the phase-1c work checked that rather than assuming it: `d² > fl(bound·bound)`
    /// forces `fl(sqrt(d²)) ≥ bound`, because the window between `fl(bound·bound)` and `bound²` is
    /// under half an ULP of `bound` once the square root has been taken, so nothing can land in it
    /// and still test below `bound` (searched over 2.4 M `(bound, d²)` pairs; no counterexample).
    /// That argument is true and it is *fragile* — it holds only because every one of the three
    /// call sites happens to use a **strict** `<`, and it is invisible at the call site. This
    /// function needs no such argument, which is why the call sites use it.
    ///
    /// The bound is what makes R §5.2 affordable. An unbounded nearest-neighbour search has to find
    /// the true nearest however far away it is, and most of the millions of probe points a pair's
    /// hypotheses throw at a breakline are far away; with the radius the traversal prunes at the
    /// first node whose box is further than it and a miss costs a handful of comparisons. Measured
    /// on the coarse stage over synthetic_20's 190 pairs and their 1.67 G probe queries:
    /// **1.08 µs per query unbounded against 0.11 µs bounded**, the whole stage 1 800
    /// core-seconds against 178, with bit-identical scores.
    pub fn nearest_below(&self, query: &[f64; 3], bound: f64) -> Option<(u32, f64)> {
        if bound <= 0.0 || bound.is_nan() {
            return None;
        }
        let widened = (bound * bound) * (1.0 + 4.0 * f64::EPSILON);
        self.nearest_within_squared(query, widened)
            .map(|(i, d2)| (i, d2.sqrt()))
            .filter(|&(_, d)| d < bound)
    }

    /// [`PointTree::nearest_within`] with the radius given **squared**, and the squared distance
    /// returned — the form R §7's correspondence search needs.
    ///
    /// Open3D's `KDTreeFlann::SearchHybrid` hands FLANN `max_correspondence_distance²` and gets
    /// squared distances back, and both the radius test and the `error²` it accumulates are then
    /// in squared units. Going through [`PointTree::nearest_within`] would take a square root here
    /// and square it again in the caller, which is two roundings the reference does not have.
    ///
    /// The bound is inclusive (`d² ≤ r²`); R §7's caller wants Open3D's strict `<` and applies it
    /// to the returned distance itself, so that the rule is stated where it is used.
    pub fn nearest_within_squared(
        &self,
        query: &[f64; 3],
        radius_squared: f64,
    ) -> Option<(u32, f64)> {
        if radius_squared < 0.0 {
            return None;
        }
        let hit = self
            .tree
            .query(query)
            .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(1).expect("1 is not zero"))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .execute();
        hit.first().map(|hit| (hit.item, hit.distance))
    }

    /// Every point within `radius` of `query` (inclusive), ascending by index.
    pub fn within(&self, query: &[f64; 3], radius: f64) -> Vec<u32> {
        let mut out = Vec::new();
        self.within_into(query, radius, &mut out);
        out
    }

    /// [`PointTree::within`] into a caller-owned buffer, which is cleared first.
    ///
    /// The radius query is the inner loop of R §3.4's three `ball_matrix` calls — one per
    /// representative, three times per fragment — so the buffer is worth reusing.
    pub fn within_into(&self, query: &[f64; 3], radius: f64, out: &mut Vec<u32>) {
        out.clear();
        if radius < 0.0 {
            return;
        }
        let found =
            self.tree.query(query).within::<SquaredEuclidean<f64>>(radius * radius).execute();
        out.extend(found.iter().map(|hit| hit.item));
        out.sort_unstable();
    }

    /// Number of points in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: an empty point set has no tree.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        reason = "a tie is an exact equality of distances; that is what these tests count"
    )]

    use super::PointTree;

    fn grid() -> Vec<[f64; 3]> {
        let mut v = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                v.push([f64::from(i), f64::from(j), 0.0]);
            }
        }
        v
    }

    #[test]
    fn the_nearest_point_is_the_nearest_point() {
        let p = grid();
        let tree = PointTree::build(&p).expect("25 points");
        assert_eq!(tree.len(), 25);
        assert!(!tree.is_empty());
        assert_eq!(tree.nearest(&[0.1, 0.1, 0.0]), 0);
        let (i, d) = tree.nearest_distance(&[0.1, 0.2, 0.0]);
        assert_eq!(i, 0);
        assert!((d - 0.2_f64.hypot(0.1)).abs() < 1e-15, "{d}");
        assert_eq!(tree.nearest_distance(&[3.0, 4.0, 0.0]), (19, 0.0), "a point of the tree");
        assert_eq!(tree.nearest(&[3.9, 4.1, 0.0]), 24);
        // A query far away still answers: the search is unbounded, as `cKDTree.query` is.
        assert_eq!(tree.nearest(&[100.0, 100.0, 0.0]), 24);
        assert!(PointTree::build(&[]).is_none());
    }

    /// The bounded search is `cKDTree.query(x, distance_upper_bound=r)`: the nearest point, or
    /// nothing at all when the nearest is further than the radius.
    #[test]
    fn a_bounded_search_answers_only_inside_its_radius() {
        let p = grid();
        let tree = PointTree::build(&p).expect("25 points");

        assert_eq!(tree.nearest_within(&[0.1, 0.2, 0.0], 1.0), Some((0, 0.2_f64.hypot(0.1))));
        // The nearest of several inside the radius, not the first found.
        assert_eq!(tree.nearest_within(&[2.9, 3.1, 0.0], 2.0).map(|(i, _)| i), Some(18));
        // Far away: the unbounded search still answers, the bounded one does not.
        assert_eq!(tree.nearest(&[100.0, 100.0, 0.0]), 24);
        assert_eq!(tree.nearest_within(&[100.0, 100.0, 0.0], 1.0), None);
        assert_eq!(tree.nearest_within(&[0.0, 0.0, 0.0], -1.0), None);
        assert_eq!(tree.nearest_within_squared(&[0.0, 0.0, 0.0], -1.0), None);

        // The squared form is the same search without the two roundings: `d²`, not `sqrt(d²)²`.
        let (i, d2) = tree.nearest_within_squared(&[0.1, 0.2, 0.0], 1.0).expect("inside");
        assert_eq!(i, 0);
        assert!((d2 - 0.05).abs() < 1e-16, "{d2}");

        // Inside the radius it is the unbounded answer, on every point of a fine sweep.
        for k in 0..200 {
            let q = [0.037 * f64::from(k), 0.021 * f64::from(k), 0.0];
            let (i, d) = tree.nearest_distance(&q);
            let bounded = tree.nearest_within(&q, 0.6);
            assert_eq!(bounded, if d <= 0.6 { Some((i, d)) } else { None }, "{q:?}");
        }
    }

    /// The bounded search returns the **unbounded** answer whenever there is one inside the radius
    /// — the same index, the same distance, ties included.
    ///
    /// This is what licenses R §6.2's and R §6.3's call sites, whose reference is an unbounded
    /// `cKDTree.query` followed by a threshold. The argument is that `nearest_n(1).within(r²)`
    /// prunes only nodes whose box is further than `r`, and no such node can hold a point at the
    /// minimum distance when that minimum is itself under `r`, so the candidate at the minimum is
    /// visited either way and the same one wins. The test is the argument checked rather than
    /// asserted, on a cloud built to have **exact ties** — every point is mirrored through the
    /// origin, so a query on the plane `x = 0` is equidistant from two of them — and at radii that
    /// straddle each query's own answer.
    #[test]
    fn the_bounded_search_is_the_unbounded_one_filtered_by_its_radius() {
        let mut points = Vec::new();
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss, reason = "a 53-bit mantissa from a 64-bit word")]
            let unit = (state >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
            unit * 8.0 - 4.0
        };
        for _ in 0..300 {
            let p = [next(), next(), next()];
            points.push(p);
            // The mirror image, so that the plane x = 0 is a tie for every pair.
            points.push([-p[0], p[1], p[2]]);
        }
        let tree = PointTree::build(&points).expect("600 points");

        let mut ties = 0_usize;
        let mut answered = 0_usize;
        for k in 0..400 {
            let q = [0.0, next(), next()];
            let (i, d) = tree.nearest_distance(&q);
            let equal = points.iter().filter(|p| dist(p, &q) == d).count();
            ties += usize::from(equal > 1);
            // `d` itself is the interesting bound: the reference's test is strict, so the answer
            // there is a miss, and the two neighbouring doubles are the boundary either side.
            for bound in [d * 0.5, d, f64::from_bits(d.to_bits() + 1), d + 1.0, 100.0] {
                let want = if d < bound { Some((i, d)) } else { None };
                assert_eq!(tree.nearest_below(&q, bound), want, "query {k} at bound {bound}");
                answered += usize::from(want.is_some());
            }
        }
        assert!(ties > 100, "the mirrored cloud should tie on most queries, not {ties}");
        assert!(answered > 1000, "and the bounds should mostly answer, not {answered}");
        assert_eq!(tree.nearest_below(&[0.0, 0.0, 0.0], 0.0), None, "a non-positive bound answers");
        assert_eq!(tree.nearest_below(&[0.0, 0.0, 0.0], -1.0), None);
    }

    /// [`PointTree::nearest_within`]'s radius test is on the *square*, and it is not `d ≤ r`.
    ///
    /// `radius · radius` rounds, so a point at exactly `radius` can have a squared distance above
    /// the rounded square and come back as a miss — which its documentation used to deny. No caller
    /// is wrong because of it: all three want the reference's strict `d < r`, and a distance that
    /// clears `fl(r·r)` can no longer test below `r` after the square root. The case is written out
    /// here so that the next caller reads the contract rather than the name.
    #[test]
    fn the_squared_radius_is_not_the_distance_radius_at_the_boundary() {
        let origin = [0.0, 0.0, 0.0];
        let tree = PointTree::build(&[[7.0, 11.0, 13.0]]).expect("one point");
        let (_, d) = tree.nearest_distance(&origin);
        // 7² + 11² + 13² = 339 exactly, and `√339` squared rounds back *below* 339 — so a query
        // whose radius is the distance itself has `d² > radius · radius` and reports a miss.
        assert!(d * d < 339.0, "{}", d * d);
        assert_eq!(tree.nearest_within(&origin, d), None, "the square rounds down");
        // One double up on the radius and the square clears 339, so the same query answers.
        let above = f64::from_bits(d.to_bits() + 1);
        assert_eq!(tree.nearest_within(&origin, above), Some((0, d)));
        // The reference asks `d < bound`: false at the distance itself, true one double above it,
        // and `nearest_below` gives both answers without the square's rounding in the way.
        assert_eq!(tree.nearest_below(&origin, d), None);
        assert_eq!(tree.nearest_below(&origin, above), Some((0, d)));
    }

    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn a_ball_is_inclusive_and_sorted_by_index() {
        let p = grid();
        let tree = PointTree::build(&p).expect("25 points");
        // Radius exactly 1 around the origin: itself and its two axis neighbours, and the
        // diagonal at √2 excluded.
        let ball = tree.within(&[0.0, 0.0, 0.0], 1.0);
        assert_eq!(ball, vec![0, 1, 5], "d = r is inside the ball");
        let ball = tree.within(&[0.0, 0.0, 0.0], 1.5);
        assert_eq!(ball, vec![0, 1, 5, 6], "and √2 joins at 1.5");
        assert_eq!(tree.within(&[0.0, 0.0, 0.0], 0.5), vec![0]);
        assert!(tree.within(&[100.0, 0.0, 0.0], 1.0).is_empty());
        assert!(tree.within(&[0.0, 0.0, 0.0], -1.0).is_empty());

        // Every ball comes back ascending, whatever order the tree found them in.
        let ball = tree.within(&[2.0, 2.0, 0.0], 2.0);
        assert!(ball.windows(2).all(|w| w[0] < w[1]), "{ball:?}");
        let brute: Vec<u32> = (0..25)
            .filter(|&i| {
                let q = p[i as usize];
                ((q[0] - 2.0).powi(2) + (q[1] - 2.0).powi(2)).sqrt() <= 2.0
            })
            .collect();
        assert_eq!(ball, brute);
    }
}
