//! Radius-bounded and unbounded nearest neighbours on the CPU (D §6.2, experiment E3).
//!
//! `kiddo` 6.2's `ImmutableKdTree` is what E3 chose: 79–251 ns per bounded query on the benchmark
//! clouds, zero errors against brute force over ~0.9 M bounded queries, and one build per cloud
//! instead of one grid per ICP ladder rung
//! (`docs/superpowers/notes/2026-09-06-e3e4-spatial.md` §5).
//!
//! [`PointTree`] is the one wrapper the algorithm needs, and it is deliberately `f64`. The two
//! callers in R §3.4 — `coarse_grid`'s nearest representative and `ball_matrix`'s radius query —
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
        self.tree.query(query).nearest_one::<SquaredEuclidean<f64>>().execute().item
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
        assert_eq!(tree.nearest(&[3.9, 4.1, 0.0]), 24);
        // A query far away still answers: the search is unbounded, as `cKDTree.query` is.
        assert_eq!(tree.nearest(&[100.0, 100.0, 0.0]), 24);
        assert!(PointTree::build(&[]).is_none());
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
