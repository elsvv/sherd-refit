//! Per-face geometry of the working mesh (R §0, R §3.3): unit face normals `FN`, face areas `A`,
//! face centroids `C`, and `res`, the median length of the unique edges.
//!
//! `res` is the unit every resolution floor of R §1.2 is counted in, and `ΣA` of the *original*
//! component is what sets the face budget (R §3.3), so both are computed in `f64` from the `f64`
//! vertices the readers produce, exactly as `sherd_refit.geometry.face_geometry` and
//! `sherd_refit.geometry.median_edge` do. Narrowing to the `f32` of [`WorkingMesh`] happens once,
//! at the end (D §4.1, D §7).
//!
//! [`WorkingMesh`]: crate::WorkingMesh

use super::adjacency::unique_edges;

/// The three per-face arrays of R §0, in face order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FaceGeometry {
    /// Unit face normals `FN`.
    pub normals: Vec<[f64; 3]>,
    /// Face areas `A`.
    pub areas: Vec<f64>,
    /// Face centroids `C`.
    pub centroids: Vec<[f64; 3]>,
}

impl FaceGeometry {
    /// Number of faces.
    #[inline]
    pub fn len(&self) -> usize {
        self.areas.len()
    }

    /// True when the mesh has no face.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.areas.is_empty()
    }

    /// Total surface area, summed the way numpy sums (see [`pairwise_sum`]).
    ///
    /// This is `A0.sum()` of R §3.3, the numerator of the face budget, and the `area` the parity
    /// harness compares (D §10.2, ±0.5 % native).
    pub fn total_area(&self) -> f64 {
        pairwise_sum(&self.areas)
    }
}

/// The floor the reference puts under a face's normal length before dividing by it.
///
/// `FN = n / max(|n|, 1e-12)`: a triangle of exactly zero area keeps a zero normal instead of a
/// NaN one. R §3.1 leaves such triangles in the mesh when their three indices differ.
const NORMAL_FLOOR: f64 = 1e-12;

/// `sherd_refit.geometry.face_geometry`: unit normals, areas and centroids of every triangle.
///
/// ```text
/// n  = (V[F1] − V[F0]) × (V[F2] − V[F0])
/// FN = n / max(|n|, 1e-12)
/// A  = |n| / 2
/// C  = (V[F0] + V[F1] + V[F2]) / 3
/// ```
///
/// Every operation is in the reference's order — the cross product component by component, the
/// norm as `sqrt((x² + y²) + z²)`, the centroid as a left-to-right sum divided by three — so the
/// arrays are bit-identical to numpy's on the same input. (Measured on the parity fixtures: see
/// `docs/superpowers/notes/2026-09-06-s3-working-mesh.md`.)
#[allow(
    clippy::many_single_char_names,
    reason = "V, F, A, C and the triangle's a, b, c are the algorithm reference's own names"
)]
pub fn face_geometry(v: &[[f64; 3]], f: &[[u32; 3]]) -> FaceGeometry {
    let mut normals = Vec::with_capacity(f.len());
    let mut areas = Vec::with_capacity(f.len());
    let mut centroids = Vec::with_capacity(f.len());
    for t in f {
        let a = v[t[0] as usize];
        let b = v[t[1] as usize];
        let c = v[t[2] as usize];
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = ((n[0] * n[0] + n[1] * n[1]) + n[2] * n[2]).sqrt();
        let den = len.max(NORMAL_FLOOR);
        normals.push([n[0] / den, n[1] / den, n[2] / den]);
        areas.push(0.5 * len);
        centroids.push([
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ]);
    }
    FaceGeometry { normals, areas, centroids }
}

/// numpy's pairwise summation, reproduced exactly.
///
/// `np.sum` does not add left to right: it accumulates in eight registers over blocks of at most
/// 128 elements and splits anything longer in half (`numpy/_core/src/umath/loops_utils.h`). The
/// difference from a naive sum is 6e-14 relative on the benchmark meshes — never enough to move
/// the face budget by one triangle, but reproducing it makes `ΣA0` *bit-identical* to the
/// reference's, and with it the `target` of R §3.3 and the `area` of the parity table.
///
/// Verified against numpy 2.5.2 on the area arrays of all 48 fragments of the parity fixtures.
pub fn pairwise_sum(x: &[f64]) -> f64 {
    /// numpy's `PW_BLOCKSIZE`.
    const BLOCK: usize = 128;
    let n = x.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in x {
            res += v;
        }
        return res;
    }
    if n <= BLOCK {
        let mut r = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]];
        let end = n - n % 8;
        let mut i = 8;
        while i < end {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += x[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += x[i];
            i += 1;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    pairwise_sum(&x[..half]) + pairwise_sum(&x[half..])
}

/// numpy's `np.median` of a slice: the middle value, or the mean of the two middle values.
///
/// The slice is copied and sorted; NaNs are not expected here (edge lengths are finite) and would
/// sort last, as they do in numpy's `partition`.
#[allow(
    clippy::manual_midpoint,
    reason = "numpy averages the two middle values as (a + b) / 2; `f64::midpoint` agrees for \
              every finite value here, but the reference's own expression is what must be read"
)]
pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// `res`: the median length of the mesh's *unique* undirected edges
/// (`sherd_refit.geometry.median_edge`).
///
/// Unique, so the value does not depend on how many faces happen to share an edge. An empty mesh
/// has `res = 0`, which is what the reference returns.
#[allow(clippy::many_single_char_names, reason = "V and F are the reference's own names")]
pub fn median_edge(v: &[[f64; 3]], f: &[[u32; 3]]) -> f64 {
    let (edges, _) = unique_edges(f);
    if edges.is_empty() {
        return 0.0;
    }
    let lengths: Vec<f64> = edges
        .iter()
        .map(|e| {
            let a = v[e[0] as usize];
            let b = v[e[1] as usize];
            let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt()
        })
        .collect();
    median(&lengths)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these tests assert exact arithmetic on purpose")]

    use super::{face_geometry, median, median_edge, pairwise_sum};
    use approx::assert_relative_eq;

    #[test]
    fn a_right_triangle_has_the_expected_normal_area_and_centroid() {
        let v = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let g = face_geometry(&v, &[[0, 1, 2]]);
        assert_eq!(g.len(), 1);
        assert_eq!(g.normals[0], [0.0, 0.0, 1.0]);
        assert_eq!(g.areas[0], 4.0);
        assert_relative_eq!(g.centroids[0][0], 2.0 / 3.0, epsilon = 1e-15);
        assert_relative_eq!(g.centroids[0][1], 4.0 / 3.0, epsilon = 1e-15);
        assert_eq!(g.centroids[0][2], 0.0);
        assert_eq!(g.total_area(), 4.0);
    }

    #[test]
    fn winding_flips_the_normal_and_a_zero_area_face_keeps_a_zero_normal() {
        let v = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let g = face_geometry(&v, &[[0, 2, 1]]);
        assert_eq!(g.normals[0], [0.0, 0.0, -1.0]);

        // Three distinct indices, three identical points: R §3.1 keeps such a triangle.
        let v = vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let g = face_geometry(&v, &[[0, 1, 2]]);
        assert_eq!(g.areas[0], 0.0);
        assert_eq!(g.normals[0], [0.0, 0.0, 0.0], "the 1e-12 floor, not a NaN");
        assert_eq!(g.centroids[0], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_pairwise_sum_is_numpys_and_not_a_left_to_right_one() {
        // Every value below is `float(np.array(x).sum())` from numpy 2.5.2, and every one of the
        // long cases differs from a left-to-right sum in the last bits.
        let x = vec![0.1_f64; 1000];
        let naive = x.iter().fold(0.0, |a, b| a + b);
        assert_eq!(pairwise_sum(&x), 100.000_000_000_000_01);
        assert_eq!(naive, 99.999_999_999_998_6, "a naive sum drifts, which is the point");

        // A large term first: the naive sum swallows the rest, the tree keeps most of it.
        let mut x = vec![1.0_f64; 101];
        x[0] = 1e16;
        assert_eq!(pairwise_sum(&x), 1.000_000_000_000_008_4e16);
        assert_eq!(x.iter().fold(0.0, |a, b| a + b), 1e16);

        // Short slices fall through to the plain loop.
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[0.1; 20]), 2.000_000_000_000_000_4);

        // Longer than one block: the recursion splits on a multiple of eight.
        let long: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
        assert_relative_eq!(pairwise_sum(&long), 499.5, epsilon = 1e-12);
    }

    #[test]
    fn the_median_is_numpys_for_both_parities() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5, "mean of the two middle values");
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn res_is_the_median_over_unique_edges_not_over_face_corners() {
        // Two triangles sharing the edge (1, 2). Unique edges and their lengths:
        //   (0,1) 1, (0,2) 1, (1,2) √2, (1,3) 1, (2,3) 1  ->  median 1
        // Counting the shared edge twice would give the same answer here, so make it long:
        let v = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]];
        let f = vec![[0, 1, 2], [2, 1, 3]];
        // lengths: (0,1) 3, (0,2) 3, (1,2) 4.2426, (1,3) 3, (2,3) 3 -> median 3
        assert_eq!(median_edge(&v, &f), 3.0);

        assert_eq!(median_edge(&v, &[]), 0.0);
    }
}
