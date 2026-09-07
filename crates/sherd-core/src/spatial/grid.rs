//! Uniform grids over a point cloud (D §6.2).
//!
//! D §6.2 specifies a hash grid for radius-bounded nearest neighbours, and E3 measured that on the
//! CPU at 0.4–2.0× `kiddo` — never the ≥ 3× D §3 expected — so as a *search* it stays a GPU
//! structure, written with the WGSL kernels in phase 2b: cell size `r`, keys hashed into
//! `next_pow2(2n)` slots, the 27 neighbouring cells visited in a fixed order, ties resolved by the
//! lowest index.
//!
//! What the CPU does use, since task E2, is the same cell layout as a conservative **near mask**
//! rather than as a second search — and the difference is the whole point: a mask that answers
//! "certainly nothing within `r`" or "maybe" cannot resolve a tie differently from the KD-tree,
//! because it never resolves anything.
//!
//! [`PointTree`](super::kdtree::PointTree) already refuses a bounded query whose point is outside
//! the cloud's bounding box, and on R §5.2's probe that is three quarters of the work
//! (`notes/2026-09-07-e2-tuning.md` §4). What is left is the quarter that lands *inside* the box
//! and still finds nothing: a breakline is a curve, so its box is the fragment's box and most of
//! that volume is empty. [`NearMask`] is the cheapest structure that can say so — one bit per
//! cell, one lookup per query.
//!
//! # What it promises
//!
//! [`NearMask::may_be_near`] returns `false` **only** when no point of the cloud is within the
//! mask's radius of the query. It never says `false` about a query that has a neighbour, so a
//! caller that falls through to the real query on `true` computes exactly what it computed before:
//! the mask is a filter, not an implementation of the query, and it cannot move a result. What it
//! *may* do is answer `true` where there is no neighbour, and how often it does is a matter of
//! speed alone.
//!
//! # Why the cells are what they are
//!
//! The cell side is `max(longest extent / 63, radius)`, so the grid is at most `64³` cells and a
//! cell is never smaller than the radius. Both bounds matter:
//!
//! * **at most 64 cells per axis** puts every `(x, y)` column of 64 `z`-cells in one `u64`, so the
//!   whole mask is at most 32 KiB — L1-resident while a pair's hypotheses are scored — and the
//!   dilation below is three passes of word-wide ORs;
//! * **a cell at least as wide as the radius** means a point within the radius of a query is in
//!   the query's own cell or one of the 26 around it. The mask is therefore *dilated* once at
//!   build time — each cell records whether that 3×3×3 block holds a point — and a query is one
//!   lookup instead of twenty-seven.
//!
//! The cell side is nudged up by `1 + 1e-9` so that `radius / cell` is strictly below one with
//! room to spare: the index of a point and the index of a query a radius apart must never differ
//! by more than one cell, and that has to survive the rounding of `(x − lo) · (1/cell)`.

/// Cells per axis at most; `z` is packed into one `u64` per `(x, y)` column, so 64 is the ceiling.
const CELLS: usize = 64;

/// The slack on the cell side, so that `radius / cell < 1` by more than any rounding of the index
/// arithmetic can be.
const SLACK: f64 = 1.0 + 1e-9;

/// A conservative "is anything within `radius` of this point?" over a fixed cloud.
#[derive(Clone, Debug)]
pub struct NearMask {
    lo: [f64; 3],
    hi: [f64; 3],
    radius_squared: f64,
    inv_cell: f64,
    dims: [usize; 3],
    /// One word per `(x, y)` column; bit `z` is set when the 3×3×3 block around `(x, y, z)` holds
    /// a point of the cloud.
    columns: Vec<u64>,
}

impl NearMask {
    /// A mask over `points` for queries at `radius`.
    ///
    /// `None` when there is nothing to build one over — an empty cloud, a non-positive radius, or
    /// coordinates that are not finite — and the caller then simply does not filter.
    #[must_use]
    pub fn of(points: &[[f64; 3]], radius: f64) -> Option<Self> {
        if points.is_empty() || radius <= 0.0 || !radius.is_finite() {
            return None;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for point in points {
            for ((low, high), &c) in lo.iter_mut().zip(&mut hi).zip(point) {
                *low = low.min(c);
                *high = high.max(c);
            }
        }
        if !lo.iter().all(|c| c.is_finite()) || !hi.iter().all(|c| c.is_finite()) {
            return None;
        }
        let ext: [f64; 3] = std::array::from_fn(|k| hi[k] - lo[k]);
        #[allow(clippy::cast_precision_loss, reason = "CELLS is 64")]
        let cell =
            (ext.iter().copied().fold(0.0_f64, f64::max) / (CELLS - 1) as f64).max(radius) * SLACK;
        if cell <= 0.0 || !cell.is_finite() {
            return None;
        }
        let inv_cell = 1.0 / cell;
        let dims: [usize; 3] = std::array::from_fn(|k| cell_index(ext[k], inv_cell, CELLS) + 1);
        let mut columns = vec![0_u64; dims[0] * dims[1]];
        for point in points {
            let cell = index_of(point, &lo, inv_cell, &dims);
            columns[cell[0] * dims[1] + cell[1]] |= 1 << cell[2];
        }
        // Dilate by one cell in each direction: z inside the word, then y, then x.
        for word in &mut columns {
            *word |= (*word << 1) | (*word >> 1);
        }
        let along_z = columns.clone();
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                let mut v = along_z[x * dims[1] + y];
                if y > 0 {
                    v |= along_z[x * dims[1] + y - 1];
                }
                if y + 1 < dims[1] {
                    v |= along_z[x * dims[1] + y + 1];
                }
                columns[x * dims[1] + y] = v;
            }
        }
        let along_y = columns.clone();
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                let mut v = along_y[x * dims[1] + y];
                if x > 0 {
                    v |= along_y[(x - 1) * dims[1] + y];
                }
                if x + 1 < dims[0] {
                    v |= along_y[(x + 1) * dims[1] + y];
                }
                columns[x * dims[1] + y] = v;
            }
        }
        // The `z` dilation sets bits above the last cell; a query never reads them, but masking
        // them off keeps `occupied` a count of real cells.
        let live = if dims[2] >= 64 { u64::MAX } else { (1_u64 << dims[2]) - 1 };
        for word in &mut columns {
            *word &= live;
        }
        Some(Self { lo, hi, radius_squared: radius * radius, inv_cell, dims, columns })
    }

    /// False only when **no** point of the cloud is within the mask's radius of `query`.
    ///
    /// Two tests, cheapest first: the cloud's bounding box, which is the same one
    /// [`PointTree`](super::kdtree::PointTree) applies and is stated the same conservative way,
    /// and then the dilated cell the query falls in. A query outside the grid is clamped to the
    /// boundary cell, whose dilated block is a superset of the block the true index would have
    /// named, so clamping can only make the answer more permissive.
    #[inline]
    #[must_use]
    pub fn may_be_near(&self, query: &[f64; 3]) -> bool {
        let mut d2 = 0.0;
        for ((&q, &low), &high) in query.iter().zip(&self.lo).zip(&self.hi) {
            let gap = if q < low {
                low - q
            } else if q > high {
                q - high
            } else {
                0.0
            };
            d2 += gap * gap;
        }
        if d2 > self.radius_squared * (1.0 + 16.0 * f64::EPSILON) {
            return false;
        }
        let cell = index_of(query, &self.lo, self.inv_cell, &self.dims);
        (self.columns[cell[0] * self.dims[1] + cell[1]] >> cell[2]) & 1 != 0
    }

    /// Cells per axis, for the tests and for a log line.
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// How many of the dilated cells are set — the fraction of the box a query can land in and
    /// still reach the tree.
    #[must_use]
    pub fn occupied(&self) -> usize {
        self.columns.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// `⌊offset · inv_cell⌋`, clamped into `0..limit`.
///
/// The float-to-integer cast saturates in Rust, so a negative offset lands on 0 and an enormous
/// one on `limit − 1`; both are the clamp the caller wants.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the cast saturates and the result is clamped into the grid"
)]
fn cell_index(offset: f64, inv_cell: f64, limit: usize) -> usize {
    ((offset * inv_cell) as usize).min(limit - 1)
}

/// The cell a point falls in, clamped into the grid on every axis.
#[inline]
fn index_of(point: &[f64; 3], lo: &[f64; 3], inv_cell: f64, dims: &[usize; 3]) -> [usize; 3] {
    std::array::from_fn(|k| cell_index(point[k] - lo[k], inv_cell, dims[k]))
}

#[cfg(test)]
mod tests {
    use super::{CELLS, NearMask};

    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The one property the filter has to have: it never says `false` about a query that has a
    /// neighbour. Checked against brute force on a random cloud, a curve (the shape a breakline
    /// actually is) and a plane, at radii from well under a cell to well over the cloud.
    #[test]
    fn the_mask_never_hides_a_neighbour() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss, reason = "a 53-bit mantissa from a 64-bit word")]
            let unit = (state >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
            unit
        };

        let cloud: Vec<[f64; 3]> =
            (0..500).map(|_| [next() * 40.0, next() * 10.0, next() * 3.0]).collect();
        // A curve through the same box, which is what a breakline looks like to the mask.
        let curve: Vec<[f64; 3]> = (0..2000)
            .map(|k| {
                let t = f64::from(k) * 0.01;
                [t * 2.0, 5.0 + 4.0 * t.sin(), 1.5 + 1.4 * (t * 0.7).cos()]
            })
            .collect();
        let plane: Vec<[f64; 3]> =
            (0..400).map(|k| [f64::from(k % 20), f64::from(k / 20), 0.0]).collect();

        let mut rejected = 0_usize;
        let mut asked = 0_usize;
        for points in [&cloud, &curve, &plane] {
            for radius in [0.05, 0.2, 1.0, 3.0, 25.0] {
                let mask = NearMask::of(points, radius).expect("a non-empty cloud");
                assert!(mask.dims().iter().all(|&d| (1..=CELLS).contains(&d)), "{:?}", mask.dims());
                for _ in 0..4000 {
                    let q = [next() * 50.0 - 5.0, next() * 14.0 - 2.0, next() * 6.0 - 1.5];
                    let near = points.iter().any(|p| dist(p, &q) <= radius);
                    let says = mask.may_be_near(&q);
                    assert!(says || !near, "the mask hid a neighbour at {q:?}, radius {radius}");
                    asked += 1;
                    rejected += usize::from(!says);
                }
            }
        }
        assert!(rejected * 4 > asked, "a filter that rejects nothing is not a filter: {rejected}");
    }

    /// A point exactly at the radius, on every axis and both signs, is still "maybe".
    #[test]
    fn a_neighbour_exactly_at_the_radius_is_not_rejected() {
        let points = vec![[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]];
        for radius in [0.25, 1.0, 4.0] {
            let mask = NearMask::of(&points, radius).expect("two points");
            for axis in 0..3 {
                for sign in [-1.0, 1.0] {
                    let mut q = [0.0; 3];
                    q[axis] = sign * radius;
                    assert!(mask.may_be_near(&q), "at the radius on axis {axis}");
                    // And one ULP inside it, where the exact answer is unambiguous.
                    q[axis] = sign * f64::from_bits(radius.to_bits() - 1);
                    assert!(mask.may_be_near(&q), "inside the radius on axis {axis}");
                }
            }
            assert!(!mask.may_be_near(&[0.0, 1e6, 0.0]), "and far away is a rejection");
        }
    }

    /// The degenerate inputs answer rather than panic.
    #[test]
    fn nothing_to_mask_is_no_mask() {
        assert!(NearMask::of(&[], 1.0).is_none());
        assert!(NearMask::of(&[[0.0; 3]], 0.0).is_none());
        assert!(NearMask::of(&[[0.0; 3]], -1.0).is_none());
        assert!(NearMask::of(&[[0.0; 3]], f64::NAN).is_none());
        assert!(NearMask::of(&[[f64::NAN, 0.0, 0.0]], 1.0).is_none());

        // A cloud of one point, and a cloud where every point coincides: one cell, always near
        // inside the radius and never outside it.
        let one = NearMask::of(&[[1.0, 2.0, 3.0]], 0.5).expect("one point");
        assert_eq!(one.dims(), [1, 1, 1]);
        assert_eq!(one.occupied(), 1);
        assert!(one.may_be_near(&[1.2, 2.0, 3.0]));
        assert!(!one.may_be_near(&[9.0, 2.0, 3.0]));
        let same = NearMask::of(&[[1.0, 2.0, 3.0]; 40], 0.5).expect("forty of the same point");
        assert!(same.may_be_near(&[1.0, 2.0, 3.4]));
        assert!(!same.may_be_near(&[1.0, 2.0, 4.0]));
    }
}
