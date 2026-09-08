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
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "brute force mirrors the grid's own arithmetic, exactly"
    )]

    use super::{CELLS, HashGrid, NearMask};

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

    /// Brute force over the same cloud, with the same `f32` arithmetic the grid uses.
    fn brute(points: &[[f32; 3]], q: &[f32; 3], radius: f32, strict: bool) -> Option<(u32, f32)> {
        let r2 = radius * radius;
        let mut best = u32::MAX;
        let mut best_d2 = f32::INFINITY;
        for (i, p) in points.iter().enumerate() {
            let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
            let d2 = dx * dx + dy * dy + dz * dz;
            let inside = if strict { d2 < r2 } else { d2 <= r2 };
            #[allow(clippy::cast_possible_truncation, reason = "clouds of a few thousand points")]
            let i = i as u32;
            if inside && (d2 < best_d2 || (d2 == best_d2 && i < best)) {
                best = i;
                best_d2 = d2;
            }
        }
        (best != u32::MAX).then(|| (best, best_d2.sqrt()))
    }

    /// A deterministic stream, so the sweep below is the same sweep on every machine.
    fn stream() -> impl FnMut() -> f32 {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss, reason = "24 bits into an f32 mantissa")]
            let unit = (state >> 40) as f32 * (1.0 / 16_777_216.0);
            unit
        }
    }

    /// D §6.2's grid answers exactly what brute force does — the index, not only the distance —
    /// on a random cloud, on a curve and on a plane, at radii from well under a cell to well over
    /// the cloud, and with both radius conventions.
    #[test]
    fn the_hash_grid_finds_what_brute_force_finds() {
        let mut next = stream();
        let cloud: Vec<[f32; 3]> =
            (0..800).map(|_| [next() * 40.0, next() * 10.0, next() * 3.0]).collect();
        let curve: Vec<[f32; 3]> = (0..1500)
            .map(|k| {
                let t = k as f32 * 0.01;
                [t * 2.0, 5.0 + 4.0 * t.sin(), 1.5 + 1.4 * (t * 0.7).cos()]
            })
            .collect();
        #[allow(clippy::cast_precision_loss, reason = "a 20x20 lattice")]
        let plane: Vec<[f32; 3]> =
            (0..400).map(|k| [(k % 20) as f32, (k / 20) as f32, 0.0]).collect();

        let mut hits = 0_usize;
        for points in [&cloud, &curve, &plane] {
            for radius in [0.05_f32, 0.2, 1.0, 3.0, 25.0] {
                let grid = HashGrid::of_f32(points, radius).expect("a non-empty cloud");
                assert_eq!(grid.points().len(), points.len());
                assert_eq!(grid.sorted_idx().len(), points.len());
                assert!(grid.occupied() > 0 && grid.occupied() <= points.len());
                for _ in 0..3000 {
                    let q = [next() * 50.0 - 5.0, next() * 14.0 - 2.0, next() * 6.0 - 1.5];
                    for strict in [false, true] {
                        let want = brute(points, &q, radius, strict);
                        let got =
                            if strict { grid.nearest_below(&q) } else { grid.nearest_within(&q) };
                        assert_eq!(got, want, "q {q:?} radius {radius} strict {strict}");
                        hits += usize::from(got.is_some());
                    }
                }
            }
        }
        assert!(hits > 1000, "a sweep that never finds a neighbour proves nothing: {hits}");
    }

    /// The tie rule is the lowest index, and it is the *original* index, not the sorted one.
    #[test]
    fn a_tie_goes_to_the_lowest_index() {
        // Twelve copies of the same point, plus one slightly nearer to a query on the other side.
        let mut points = vec![[1.0_f32, 2.0, 3.0]; 12];
        points.push([1.0, 2.0, 3.5]);
        let grid = HashGrid::of_f32(&points, 1.0).expect("thirteen points");
        assert_eq!(grid.nearest_within(&[1.0, 2.0, 3.0]), Some((0, 0.0)));
        assert_eq!(grid.nearest_within(&[1.0, 2.0, 3.6]), Some((12, 0.099_999_905)));
        // Two cells, equidistant: the lower index wins wherever the traversal meets it.
        let pair = vec![[1.0_f32, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let grid = HashGrid::of_f32(&pair, 2.0).expect("two points");
        assert_eq!(grid.nearest_within(&[0.0, 0.0, 0.0]).map(|(i, _)| i), Some(0));
        let flipped = vec![[-1.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let grid = HashGrid::of_f32(&flipped, 2.0).expect("two points");
        assert_eq!(grid.nearest_within(&[0.0, 0.0, 0.0]).map(|(i, _)| i), Some(0));
    }

    /// The radius bound: inclusive for `nearest_within`, strict for `nearest_below`.
    #[test]
    fn the_radius_bound_is_the_one_the_caller_asked_for() {
        let points = vec![[0.0_f32, 0.0, 0.0]];
        let grid = HashGrid::of_f32(&points, 1.0).expect("one point");
        assert_eq!(grid.nearest_within(&[1.0, 0.0, 0.0]), Some((0, 1.0)));
        assert_eq!(grid.nearest_below(&[1.0, 0.0, 0.0]), None, "exclusive at the radius");
        let inside = f32::from_bits(1.0_f32.to_bits() - 1);
        assert!(grid.nearest_below(&[inside, 0.0, 0.0]).is_some(), "one ulp inside it is a hit");
        assert_eq!(grid.nearest_within(&[1.001, 0.0, 0.0]), None);
    }

    /// The device buffers are the shapes D §6.2 names, and the table is a power of two at least
    /// twice the cloud.
    #[test]
    fn the_device_layout_is_the_one_the_kernel_binds() {
        assert_eq!(std::mem::size_of::<super::GridHeader>(), 32);
        assert_eq!(std::mem::size_of::<super::GridSlot>(), 16);
        let mut next = stream();
        let points: Vec<[f32; 3]> = (0..600).map(|_| [next(), next(), next()]).collect();
        let grid = HashGrid::of_f32(&points, 0.1).expect("600 points");
        let header = grid.header();
        assert_eq!(header.n, 600);
        assert_eq!(header.cap, 2048, "next_pow2(2n)");
        assert_eq!(grid.slots().len(), 2048);
        assert_eq!(grid.counts().len(), 2048);
        assert!((header.inv_cell - 10.0).abs() < 1e-6);
        // Every point is filed under exactly one slot, and each slot's run is ascending.
        let total: u32 = grid.counts().iter().sum();
        assert_eq!(total, 600);
        for (slot, &count) in grid.counts().iter().enumerate() {
            if count == 0 {
                assert_eq!(grid.slots()[slot].start, -1);
                continue;
            }
            let start = grid.slots()[slot].start as usize;
            let run = &grid.sorted_idx()[start..start + count as usize];
            assert!(run.windows(2).all(|w| w[0] < w[1]), "ascending inside a cell");
        }
        assert!(grid.device_bytes() > 0);
        assert!(grid.max_per_cell() >= 1);
    }

    /// The degenerate inputs answer rather than panic.
    #[test]
    fn nothing_to_grid_is_no_grid() {
        assert!(HashGrid::of_f32(&[], 1.0).is_none());
        assert!(HashGrid::of_f32(&[[0.0; 3]], 0.0).is_none());
        assert!(HashGrid::of_f32(&[[0.0; 3]], -1.0).is_none());
        assert!(HashGrid::of_f32(&[[0.0; 3]], f32::NAN).is_none());
        assert!(HashGrid::of_f32(&[[f32::NAN, 0.0, 0.0]], 1.0).is_none());
        assert!(HashGrid::build(&[], 1.0).is_none());
        let one = HashGrid::build(&[[1.0, 2.0, 3.0]], 0.5).expect("one point");
        assert_eq!(one.nearest_within(&[1.2, 2.0, 3.0]).map(|(i, _)| i), Some(0));
        assert!(one.nearest_within(&[9.0, 2.0, 3.0]).is_none());
    }
}

// ---------------------------------------------------------------------------------------------
// D §6.2's hash grid: the radius-bounded nearest-neighbour structure both executors share.
// ---------------------------------------------------------------------------------------------

/// One cell of [`HashGrid`], as the WGSL kernels read it (`slots: array<vec4<i32>>`, D §6.2).
///
/// `start` is `-1` for an empty slot; a used slot's `start` indexes [`HashGrid::sorted_idx`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridSlot {
    /// Cell coordinate along x.
    pub ix: i32,
    /// Cell coordinate along y.
    pub iy: i32,
    /// Cell coordinate along z.
    pub iz: i32,
    /// First entry of this cell in `sorted_idx`, or `-1` when the slot is empty.
    pub start: i32,
}

/// The uniform half of a [`HashGrid`] (`GridHeader { origin: vec4<f32> (w = 1/r), cap, n, pad }`,
/// D §6.2).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridHeader {
    /// The grid's origin: the cloud's minimum corner.
    pub origin: [f32; 3],
    /// `1/r`, the reciprocal of the cell side — the kernel multiplies rather than divides,
    /// because Metal's division is 2 ULP (E7 §4.2) and a reciprocal computed once on the host is
    /// the same number on both sides.
    pub inv_cell: f32,
    /// Slots in the table, a power of two.
    pub cap: u32,
    /// Points in the cloud.
    pub n: u32,
    /// Padding to sixteen bytes.
    pub pad: [u32; 2],
}

/// The multipliers of D §6.2's cell hash.
const HASH: [i64; 3] = [73_856_093, 19_349_663, 83_492_791];

/// D §6.2's hash grid over a point cloud, built on the CPU and uploaded with a batch.
///
/// This is the structure the WGSL kernels of phase 2b query, and it is built here so that the two
/// executors traverse **the same cells in the same order**: cell side `r`, integer cell
/// coordinates `⌊(p − origin)·(1/r)⌋`, keys hashed into `cap = next_pow2(2n)` slots by
/// `(ix·73856093) ^ (iy·19349663) ^ (iz·83492791)`, open addressing with linear probing, and one
/// `sorted_idx` array holding the points of each cell in ascending original index. A query visits
/// the 27 neighbouring cells in a fixed `(dx, dy, dz)` order and the points of each cell in
/// ascending index, keeping the nearest inside the radius; ties go to the lowest index.
///
/// # Why the CPU path does not search with it
///
/// E3 measured this grid at 0.4–2.0× `kiddo` on the CPU — never the ≥ 3× D §3 hoped for — so
/// [`PointTree`](super::kdtree::PointTree) stayed the CPU's search structure and every parity row
/// of D §10.2 is measured through it. Nothing in the pipeline queries a `HashGrid`: it is built
/// from an [`Executor`](crate::executor::Executor) batch by the executor that needs it, which is
/// the GPU one. [`HashGrid::nearest_below`] and [`HashGrid::nearest_within`] exist so that the
/// kernel has a CPU mirror to be cross-checked against (D §10.4 layer 3) and so that the
/// traversal order can be tested against brute force without an adapter.
///
/// The coordinates are `f32` because the kernels are (D §6.8: f32 only); the caller narrows once,
/// here, rather than once per query.
#[derive(Clone, Debug)]
pub struct HashGrid {
    header: GridHeader,
    slots: Vec<GridSlot>,
    counts: Vec<u32>,
    sorted_idx: Vec<u32>,
    points: Vec<[f32; 3]>,
    radius: f32,
}

impl HashGrid {
    /// Builds the grid of D §6.2 over `points` for queries at `radius`.
    ///
    /// `None` when there is nothing to build over: an empty cloud, a non-positive or non-finite
    /// radius, or a coordinate that is not finite once narrowed to `f32`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "the grid is f32, as the kernels are")]
    pub fn build(points: &[[f64; 3]], radius: f64) -> Option<Self> {
        let narrowed: Vec<[f32; 3]> =
            points.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        Self::of_f32(&narrowed, radius as f32)
    }

    /// [`HashGrid::build`] over coordinates already narrowed.
    #[must_use]
    #[allow(
        clippy::cast_sign_loss,
        reason = "a claimed slot's `start` is non-negative by the time it is read"
    )]
    pub fn of_f32(points: &[[f32; 3]], radius: f32) -> Option<Self> {
        if points.is_empty() || !(radius.is_finite() && radius > 0.0) {
            return None;
        }
        if points.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
            return None;
        }
        let mut origin = [f32::INFINITY; 3];
        for p in points {
            for (o, &c) in origin.iter_mut().zip(p) {
                *o = o.min(c);
            }
        }
        let n = points.len();
        let cap = (2 * n).next_power_of_two();
        let inv_cell = 1.0 / radius;
        let header = GridHeader {
            origin,
            inv_cell,
            cap: u32::try_from(cap).ok()?,
            n: u32::try_from(n).ok()?,
            pad: [0; 2],
        };

        // Pass one: every point's cell, and the slot it hashes to. Open addressing, linear
        // probing; `cap = 2n` guarantees a free slot exists.
        let mut slots = vec![GridSlot { ix: 0, iy: 0, iz: 0, start: -1 }; cap];
        let mut counts = vec![0_u32; cap];
        let mut slot_of = vec![0_u32; n];
        for (i, p) in points.iter().enumerate() {
            let cell = cell_of(p, &origin, inv_cell);
            let slot = probe(&mut slots, cap, cell);
            counts[slot] += 1;
            slot_of[i] = u32::try_from(slot).unwrap_or(u32::MAX);
        }
        // Pass two: a prefix sum over the slots in slot order gives each cell its run of
        // `sorted_idx`; walking the points in ascending index fills each run in ascending index,
        // which is the tie rule.
        let mut cursor = 0_u32;
        for (slot, &count) in counts.iter().enumerate() {
            if count > 0 {
                slots[slot].start = i32::try_from(cursor).ok()?;
                cursor += count;
            }
        }
        let mut fill = vec![0_u32; cap];
        let mut sorted_idx = vec![0_u32; n];
        for (i, &slot) in slot_of.iter().enumerate() {
            let slot = slot as usize;
            let at = slots[slot].start as usize + fill[slot] as usize;
            sorted_idx[at] = u32::try_from(i).unwrap_or(u32::MAX);
            fill[slot] += 1;
        }
        Some(Self { header, slots, counts, sorted_idx, points: points.to_vec(), radius })
    }

    /// The nearest point within `radius` **inclusive** (`d ≤ r`), as D §6.2 states the rule.
    #[must_use]
    pub fn nearest_within(&self, query: &[f32; 3]) -> Option<(u32, f32)> {
        self.search(query, false)
    }

    /// The nearest point **strictly** inside `radius` (`d < r`).
    ///
    /// R §5.2's bound is scipy's `distance_upper_bound`, which is exclusive, and R §7's is FLANN's
    /// strict `<` on the squared distance; a kernel that mirrors either of those calls this one.
    #[must_use]
    pub fn nearest_below(&self, query: &[f32; 3]) -> Option<(u32, f32)> {
        self.search(query, true)
    }

    /// D §6.2's traversal, with the radius test the caller picked.
    ///
    /// Written out the way the WGSL must be: the squared distance as `dx*dx + dy*dy + dz*dz`
    /// rather than a `dot`, because `metal::dot` stays a fused chain even when contraction is
    /// disabled (E7 §4.2).
    #[allow(clippy::cast_sign_loss, reason = "a slot the probe returns has a non-negative `start`")]
    #[allow(clippy::float_cmp, reason = "D §6.2's tie rule is an exact equality on `d2`")]
    fn search(&self, query: &[f32; 3], strict: bool) -> Option<(u32, f32)> {
        let r2 = self.radius * self.radius;
        let base = cell_of(query, &self.header.origin, self.header.inv_cell);
        let mut best = u32::MAX;
        let mut best_d2 = f32::INFINITY;
        for dx in -1..=1_i32 {
            for dy in -1..=1_i32 {
                for dz in -1..=1_i32 {
                    let cell = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let Some(slot) = self.find(cell) else { continue };
                    let start = self.slots[slot].start as usize;
                    let count = self.counts[slot] as usize;
                    for &i in &self.sorted_idx[start..start + count] {
                        let p = self.points[i as usize];
                        let dxq = p[0] - query[0];
                        let dyq = p[1] - query[1];
                        let dzq = p[2] - query[2];
                        let d2 = dxq * dxq + dyq * dyq + dzq * dzq;
                        let inside = if strict { d2 < r2 } else { d2 <= r2 };
                        // `<` on the distance and ascending index inside a cell together give
                        // D §6.2's rule: ties go to the lowest index.
                        if inside && (d2 < best_d2 || (d2 == best_d2 && i < best)) {
                            best = i;
                            best_d2 = d2;
                        }
                    }
                }
            }
        }
        (best != u32::MAX).then(|| (best, best_d2.sqrt()))
    }

    /// The slot holding `cell`, or `None` when the cell is empty.
    fn find(&self, cell: [i32; 3]) -> Option<usize> {
        let cap = self.header.cap as usize;
        let mut slot = hash_cell(cell) & (cap - 1);
        for _ in 0..cap {
            let s = self.slots[slot];
            if s.start < 0 {
                return None;
            }
            if [s.ix, s.iy, s.iz] == cell {
                return Some(slot);
            }
            slot = (slot + 1) & (cap - 1);
        }
        None
    }

    /// The uniform block a kernel binds.
    #[must_use]
    pub fn header(&self) -> GridHeader {
        self.header
    }

    /// `slots: array<vec4<i32>>`.
    #[must_use]
    pub fn slots(&self) -> &[GridSlot] {
        &self.slots
    }

    /// `counts: array<u32>`.
    #[must_use]
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// `sorted_idx: array<u32>`.
    #[must_use]
    pub fn sorted_idx(&self) -> &[u32] {
        &self.sorted_idx
    }

    /// The cloud, narrowed once at build time.
    #[must_use]
    pub fn points(&self) -> &[[f32; 3]] {
        &self.points
    }

    /// The radius the grid was built for; the cell side is the same number.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// How many slots are occupied — the number E7 §5 reports as "occupied cells".
    #[must_use]
    pub fn occupied(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    /// The largest number of points in one cell.
    #[must_use]
    pub fn max_per_cell(&self) -> u32 {
        self.counts.iter().copied().max().unwrap_or(0)
    }

    /// Bytes a batch uploads for this grid: header, slots, counts, indices and the cloud.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        std::mem::size_of::<GridHeader>()
            + std::mem::size_of_val(self.slots.as_slice())
            + std::mem::size_of_val(self.counts.as_slice())
            + std::mem::size_of_val(self.sorted_idx.as_slice())
            + self.points.len() * 16
    }
}

/// `⌊(p − origin)·inv_cell⌋` on each axis, as the kernel computes it.
#[inline]
#[allow(clippy::cast_possible_truncation, reason = "cell coordinates are small integers")]
fn cell_of(p: &[f32; 3], origin: &[f32; 3], inv_cell: f32) -> [i32; 3] {
    std::array::from_fn(|k| ((p[k] - origin[k]) * inv_cell).floor() as i32)
}

/// D §6.2's cell hash, in `i64` so that the multiplication cannot overflow before the xor.
#[inline]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the xor is a bit pattern, and the mask that follows fits every pointer width"
)]
fn hash_cell(cell: [i32; 3]) -> usize {
    let h = (i64::from(cell[0]).wrapping_mul(HASH[0]))
        ^ (i64::from(cell[1]).wrapping_mul(HASH[1]))
        ^ (i64::from(cell[2]).wrapping_mul(HASH[2]));
    (h as u64 as usize) & (usize::MAX >> 1)
}

/// The slot `cell` belongs in, claiming a free one on the way (linear probing).
fn probe(slots: &mut [GridSlot], cap: usize, cell: [i32; 3]) -> usize {
    let mut slot = hash_cell(cell) & (cap - 1);
    loop {
        let s = slots[slot];
        if [s.ix, s.iy, s.iz] == cell && s.start == i32::MIN {
            return slot;
        }
        if s.start == -1 {
            slots[slot] = GridSlot { ix: cell[0], iy: cell[1], iz: cell[2], start: i32::MIN };
            return slot;
        }
        slot = (slot + 1) & (cap - 1);
    }
}
