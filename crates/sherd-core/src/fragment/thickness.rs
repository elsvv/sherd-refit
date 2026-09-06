//! Wall thickness `t` (R §3.2): the histogram mode of inward ray hits.
//!
//! Rays are cast inwards from the centroids of 20 000 randomly chosen faces of the *original*
//! largest component — before decimation, because `t` is what sets the face budget. The hit
//! distances are histogrammed and the mode of the histogram is the wall thickness:
//!
//! ```text
//! idx    = rng_pre.choice(len(C0), min(20000, len(C0)), replace=False)
//! dvec   = −FN0[idx]
//! origin = C0[idx] + dvec · 1e-3                       # PMC-1: step off the surface
//! (d, prim) = first hit along (origin, dvec)
//! ok     = isfinite(d) & (prim < n_faces)              # fewer than 100 -> the OBB fallback
//! raw    = hist_mode(d[ok])
//! far    = d[ok][ FN0[prim[ok]] · dvec[ok] > 0.7 ]     # the hit face looks back along the ray
//! t      = hist_mode(far) if len(far) ≥ 100 else raw
//! ```
//!
//! The `> 0.7` filter is the whole trick: a ray that leaves the outer shell and lands on the
//! *inner* surface of the same wall hits a face whose normal points the way the ray travels, so
//! it measures the wall. A ray that runs along a rim, down a lip or into the fracture surface hits
//! something side-on and is dropped. About two thirds of the rays survive on this data.
//!
//! Every number here is `f32`, because Open3D's `RaycastingScene` is: it casts the vertices and
//! the queries to `float32`, `t_hit` comes back `float32`, and `np.percentile` and `np.histogram`
//! on a `float32` array stay in `float32` (numpy 2's weak scalars never promote them). Reproducing
//! that exactly is what makes `t` and `thick_mode` bit-identical to the reference's on the parity
//! fixtures. The one place `f64` is used is the reference's own: the `> 0.7` test compares the
//! `f64` face normals, not the `f32` ray directions.
//!
//! The ray casting itself is `parry3d` (experiment E3/E4), through the shared
//! [`RayScene`] of D §6.2 — the same structure the seven-ray cone
//! of R §3.4.3 casts through, re-exported here because this module was its first caller.

use nalgebra::Matrix3;
use parry3d::math::Vector;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::Rng;
use rayon::prelude::*;

use crate::mesh::geometry::FaceGeometry;
pub use crate::spatial::bvh::RayScene;

/// The reference hard-codes this seed for the thickness rays (R §3, `rng_pre = rng(0)`).
pub const SEED: u64 = 0;
/// Rays cast, or all faces when the mesh has fewer.
pub const RAYS: usize = 20_000;
/// How far along the ray the origin is pushed off the surface (PMC-1).
pub const RAY_OFFSET: f64 = 1e-3;
/// A hit face "looks back along the ray" when its normal agrees with the ray direction this well.
pub const LOOKS_BACK_COS: f64 = 0.7;
/// Fewer valid hits than this and the estimate is refused (both for `ok` and for `far`).
pub const MIN_HITS: usize = 100;
/// Bins in the mode histogram.
pub const BINS: usize = 60;
/// Percentile of the hit distances the histogram spans.
pub const HIST_PERCENTILE: f64 = 90.0;
/// What Open3D reports as the primitive id of a ray that hit nothing.
pub const MISS: u32 = u32::MAX;

/// The result of one batch of ray casts, in Open3D's own encoding.
///
/// A miss carries `t_hit = ∞` and `prim = 0xFFFFFFFF`, so a fixture's `thick.t_hit` /`thick.prim`
/// arrays can be fed straight into [`thickness_from_hits`] for the injected parity mode.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RayHits {
    /// Distance to the first hit along the ray, `f32::INFINITY` on a miss.
    pub t_hit: Vec<f32>,
    /// Index of the face hit, [`MISS`] on a miss.
    pub prim: Vec<u32>,
}

impl RayHits {
    /// Number of rays.
    #[inline]
    pub fn len(&self) -> usize {
        self.t_hit.len()
    }

    /// True when no ray was cast.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.t_hit.is_empty()
    }
}

/// R §3.2 end to end: sample faces, cast the rays, take the two modes.
///
/// Returns `(t, thick_mode)` as the reference's pair — the filtered estimate and the plain mode
/// over every valid hit — or `None` when fewer than [`MIN_HITS`] rays hit anything, which is the
/// reference's signal to fall back to [`obb_min_extent`].
pub fn estimate_thickness(
    v: &[[f64; 3]],
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    rng: &mut ChaCha8Rng,
) -> Option<(f32, f32)> {
    let scene = RayScene::new(v, f)?;
    let idx = sample_face_indices(f.len(), RAYS, rng);
    let hits = cast_inward_rays(&scene, geom, &idx);
    thickness_from_hits(&geom.normals, &idx, &hits)
}

/// Casts one ray inwards from each sampled face.
///
/// Origin and direction are computed in `f64` and narrowed to `f32` once, at the end, which is
/// what `np.concatenate([...]).astype(np.float32)` does in the reference.
#[allow(clippy::cast_possible_truncation, reason = "the rays are f32, as Open3D's are")]
pub fn cast_inward_rays(scene: &RayScene, geom: &FaceGeometry, idx: &[u32]) -> RayHits {
    let hits: Vec<(f32, u32)> = idx
        .par_iter()
        .map(|&i| {
            let c = geom.centroids[i as usize];
            let n = geom.normals[i as usize];
            let d = [-n[0], -n[1], -n[2]];
            let origin = [
                (c[0] + d[0] * RAY_OFFSET) as f32,
                (c[1] + d[1] * RAY_OFFSET) as f32,
                (c[2] + d[2] * RAY_OFFSET) as f32,
            ];
            let dir = [d[0] as f32, d[1] as f32, d[2] as f32];
            scene.first_hit(origin, dir).map_or((f32::INFINITY, MISS), |(prim, t)| (t, prim))
        })
        .collect();
    RayHits { t_hit: hits.iter().map(|h| h.0).collect(), prim: hits.iter().map(|h| h.1).collect() }
}

/// The two modes of R §3.2 from a batch of hits.
///
/// `normals` are the `f64` face normals of the mesh the rays were cast on, `idx` the face each ray
/// started from. Fed a Python fixture's `thick.t_hit` / `thick.prim` / `thick.idx`, this
/// reproduces the reference's `t` and `thick_mode` bit for bit.
pub fn thickness_from_hits(
    normals: &[[f64; 3]],
    idx: &[u32],
    hits: &RayHits,
) -> Option<(f32, f32)> {
    let n_faces = normals.len();
    let mut ok_d: Vec<f32> = Vec::with_capacity(hits.len());
    let mut ok_at: Vec<usize> = Vec::with_capacity(hits.len());
    for k in 0..hits.len() {
        let d = hits.t_hit[k];
        let p = hits.prim[k];
        if d.is_finite() && (p as usize) < n_faces {
            ok_d.push(d);
            ok_at.push(k);
        }
    }
    if ok_d.len() < MIN_HITS {
        return None;
    }
    let raw = hist_mode(&ok_d);
    let far = looking_back(normals, idx, hits, &ok_d, &ok_at);
    let t = if far.len() >= MIN_HITS { hist_mode(&far) } else { raw };
    Some((t, raw))
}

/// The distances R §3.2 takes its *filtered* mode over: the hits whose face looks back along the
/// ray it was hit by (`FN[prim] · dvec > 0.7`).
///
/// Exposed because the width of one bin of the histogram over exactly these distances —
/// `percentile(far, 90) / 60` — is the resolution of the whole thickness estimate, and the parity
/// harness reports `t` in those bins as well as in per cent (D §10.2, and §5 of the S3 note).
pub fn filtered_distances(normals: &[[f64; 3]], idx: &[u32], hits: &RayHits) -> Vec<f32> {
    let n_faces = normals.len();
    let mut ok_d = Vec::with_capacity(hits.len());
    let mut ok_at = Vec::with_capacity(hits.len());
    for k in 0..hits.len() {
        if hits.t_hit[k].is_finite() && (hits.prim[k] as usize) < n_faces {
            ok_d.push(hits.t_hit[k]);
            ok_at.push(k);
        }
    }
    looking_back(normals, idx, hits, &ok_d, &ok_at)
}

/// The filter of R §3.2, over hits already known to be valid: `ok_d[j]` is the distance of ray
/// `ok_at[j]`.
fn looking_back(
    normals: &[[f64; 3]],
    idx: &[u32],
    hits: &RayHits,
    ok_d: &[f32],
    ok_at: &[usize],
) -> Vec<f32> {
    ok_at
        .iter()
        .enumerate()
        .filter_map(|(j, &k)| {
            let hit = normals[hits.prim[k] as usize];
            let from = normals[idx[k] as usize];
            // dvec = -FN[idx], so the dot product is the negated one; the reference computes it
            // on the f64 normals, not on the f32 ray directions.
            let dot = -((hit[0] * from[0] + hit[1] * from[1]) + hit[2] * from[2]);
            (dot > LOOKS_BACK_COS).then_some(ok_d[j])
        })
        .collect()
}

/// `_hist_mode`: the centre of the fullest of 60 equal bins spanning `[0, p90]`.
///
/// This is `np.histogram(d, bins=60, range=(0, np.percentile(d, 90)))` followed by
/// `0.5·(edges[k] + edges[k+1])` at the *first* maximal bin, reproduced in `f32` down to numpy's
/// rounding corrections. Verified bit-exact against numpy 2.5.2 on all 48 fragments of the parity
/// fixtures, for both the filtered and the unfiltered call.
pub fn hist_mode(x: &[f32]) -> f32 {
    let p90 = percentile90(x);
    // numpy's `_get_outer_edges` widens an empty range rather than dividing by zero.
    let (first, last) = if p90 == 0.0 { (-0.5_f32, 0.5_f32) } else { (0.0_f32, p90) };
    let edges = bin_edges(first, last);

    let denom = last - first;
    let mut counts = [0_u64; BINS];
    for &v in x {
        if !(v >= first && v <= last) {
            continue;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the value is inside [first, last], so the index is inside 0..=BINS"
        )]
        let mut i = (((v - first) / denom) * f32::from(u16::try_from(BINS).unwrap())) as usize;
        if i == BINS {
            i -= 1;
        }
        // numpy's two corrections for values within a ULP of a bin edge.
        if v < edges[i] {
            i -= 1;
        }
        if i != BINS - 1 && v >= edges[i + 1] {
            i += 1;
        }
        counts[i] += 1;
    }
    let k = counts
        .iter()
        .enumerate()
        .max_by_key(|(i, c)| (**c, std::cmp::Reverse(*i)))
        .map_or(0, |(i, _)| i);
    0.5 * (edges[k] + edges[k + 1])
}

/// `np.linspace(first, last, 61)` in `f32`, including numpy's exact final-value assignment.
fn bin_edges(first: f32, last: f32) -> [f32; BINS + 1] {
    #[allow(clippy::cast_precision_loss, reason = "BINS is 60")]
    let step = (last - first) / BINS as f32;
    let mut edges = [0.0_f32; BINS + 1];
    for (k, e) in edges.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss, reason = "k <= 60")]
        {
            *e = k as f32 * step + first;
        }
    }
    edges[BINS] = last;
    edges
}

/// `np.percentile(x, 90)` on an `f32` array, with numpy's linear interpolation and dtypes.
///
/// numpy computes the virtual index `0.9·(n−1)` in `f64`, then interpolates *in the array's own
/// dtype* because a Python-scalar quantile is weak under NEP 50 — and it interpolates from the
/// upper end when `γ ≥ 0.5`. Both details are worth a ULP each, and a ULP moves the bin edges,
/// which is why they are reproduced here.
#[allow(clippy::many_single_char_names, reason = "the symbols of numpy's own quantile code")]
pub fn percentile90(x: &[f32]) -> f32 {
    assert!(!x.is_empty(), "percentile of an empty slice");
    let mut v = x.to_vec();
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    #[allow(clippy::cast_precision_loss, reason = "n is at most 20 000")]
    let virtual_index = (n - 1) as f64 * (HIST_PERCENTILE / 100.0);
    let previous = virtual_index.floor();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0 <= i < n")]
    let i = previous as usize;
    let gamma = virtual_index - previous;
    let (a, b) = (v[i], v[(i + 1).min(n - 1)]);
    let diff = b - a;
    #[allow(clippy::cast_possible_truncation, reason = "numpy interpolates in the array's dtype")]
    if gamma >= 0.5 { b - diff * (1.0 - gamma) as f32 } else { a + diff * gamma as f32 }
}

/// `min(extent of the PCA oriented bounding box) / 10` — the reference's fallback when the rays
/// fail (R §3.2).
///
/// Open3D's `get_oriented_bounding_box()` is `OrientedBoundingBox::CreateFromPoints`: it takes the
/// **convex hull** of the points, runs the PCA over the *hull's* vertices, and measures the
/// extents along those axes. This does the same. The extents themselves may be taken over the hull
/// or over every point — the hull contains every extreme point, so the two agree — but the axes
/// may not: a PCA is a weighted average of positions, and the interior of a scan is not
/// distributed like its hull.
///
/// **Finding F6 of the phase-1a verification, and the size of it was measured rather than
/// guessed.** Against Open3D 0.19: on `fixtures/slab/input/pieceA.ply` the hull PCA gives 42.0619
/// where a PCA over all 22 552 vertices gives 40.8516 (3.0 % low); on `pieceB.ply` 41.7494 against
/// 38.3579 (8.1 % low); on a terracotta scan 93.803 against 102.075 (8.8 % high). The earlier
/// "about 1 %" recorded here was optimistic, and since `t` is the unit of every threshold in
/// R §1.2, an 8 % error would move all of them by 8 %.
///
/// The hull comes from `parry3d`, which computes it in `f32`; the PCA and the projections are
/// `f64`, over the `f32`-rounded hull vertices. Measured cost of that narrowing against Open3D's
/// `f64` Qhull: 1.6e-9 relative on `pieceA`, 1.0e-8 on `pieceB` — five orders of magnitude below
/// what it buys. When no hull can be built — fewer than three points, or a degenerate set that
/// leaves `try_convex_hull` with nothing — the PCA falls back to every vertex, which is the old
/// behaviour and the best available.
///
/// The path is unreachable on the benchmark: the fragment with the fewest valid hits of the 68 in
/// the parity fixtures has 7 154 of 20 000, against a threshold of 100.
pub fn obb_min_extent(v: &[[f64; 3]]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let axes = principal_axes(&hull_or_all(v));
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in v {
        for k in 0..3 {
            let proj = p[0] * axes[(0, k)] + p[1] * axes[(1, k)] + p[2] * axes[(2, k)];
            lo[k] = lo[k].min(proj);
            hi[k] = hi[k].max(proj);
        }
    }
    (0..3).map(|k| hi[k] - lo[k]).fold(f64::INFINITY, f64::min)
}

/// The vertices of the convex hull of `v`, or `v` itself when no hull can be built.
#[allow(clippy::cast_possible_truncation, reason = "parry3d's hull is f32 by construction")]
fn hull_or_all(v: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let points: Vec<Vector> =
        v.iter().map(|p| Vector::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect();
    match parry3d::transformation::try_convex_hull(&points) {
        Ok((hull, _)) if hull.len() >= 3 => {
            hull.iter().map(|p| [f64::from(p.x), f64::from(p.y), f64::from(p.z)]).collect()
        }
        _ => v.to_vec(),
    }
}

/// The eigenvectors of the covariance of `p`, as the columns of a 3×3 matrix — Open3D's
/// `ComputeMeanAndCovariance` followed by Eigen's `SelfAdjointEigenSolver`.
///
/// The extents an OBB reports do not depend on the mean (a shift cancels in `max − min`), only on
/// these axes, so the caller projects its own points and never sees the mean.
fn principal_axes(p: &[[f64; 3]]) -> Matrix3<f64> {
    #[allow(clippy::cast_precision_loss, reason = "vertex counts are far below 2^53")]
    let n = p.len() as f64;
    let mut mean = [0.0_f64; 3];
    for q in p {
        for c in 0..3 {
            mean[c] += q[c];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut cov = Matrix3::<f64>::zeros();
    for q in p {
        let d = [q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]];
        for r in 0..3 {
            for c in 0..3 {
                cov[(r, c)] += d[r] * d[c];
            }
        }
    }
    cov /= n;
    cov.symmetric_eigen().eigenvectors
}

/// `rng.choice(n_faces, min(n, n_faces), replace=False)`: distinct face indices, uniformly.
///
/// A partial Fisher–Yates shuffle, so the draw is O(n) and needs no hash set (D §7 keeps unordered
/// containers off every result path). The *sequence* is not numpy's — this is `ChaCha8Rng`, PMC-9
/// — so in native mode the sampled faces differ from the reference's and the thickness comparison
/// is statistical (±2 %, D §10.2). In injected mode the fixture's `thick.idx` is used instead.
pub fn sample_face_indices(n_faces: usize, n: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let take = n.min(n_faces);
    let mut pool: Vec<u32> = (0..u32::try_from(n_faces).expect("face count fits in u32")).collect();
    for i in 0..take {
        let span = u32::try_from(n_faces - i).expect("face count fits in u32");
        let j = i + uniform_below(rng, span) as usize;
        pool.swap(i, j);
    }
    pool.truncate(take);
    pool
}

/// A uniform integer in `0..n`, by Lemire's multiply-shift with rejection (unbiased).
fn uniform_below(rng: &mut ChaCha8Rng, n: u32) -> u32 {
    debug_assert!(n > 0);
    let mut m = u64::from(rng.next_u32()) * u64::from(n);
    #[allow(clippy::cast_possible_truncation, reason = "the low word is the fractional part")]
    let mut low = m as u32;
    if low < n {
        let threshold = n.wrapping_neg() % n;
        while low < threshold {
            m = u64::from(rng.next_u32()) * u64::from(n);
            #[allow(clippy::cast_possible_truncation, reason = "the low word is the fraction")]
            {
                low = m as u32;
            }
        }
    }
    #[allow(clippy::cast_possible_truncation, reason = "the high word is below n")]
    {
        (m >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the histogram code is asserted exactly on purpose")]

    use super::{
        MIN_HITS, MISS, RayHits, estimate_thickness, hist_mode, obb_min_extent, percentile90,
        sample_face_indices, thickness_from_hits,
    };
    use crate::mesh::Mesh;
    use crate::mesh::geometry::face_geometry;
    use crate::rng::seeded;
    use std::collections::HashSet;

    /// An axis-aligned box from `lo` to `hi`, outward normals, twelve triangles.
    fn box_mesh(lo: [f64; 3], hi: [f64; 3]) -> Mesh {
        let v = vec![
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2], // z = lo
            [4, 5, 6],
            [4, 6, 7], // z = hi
            [0, 1, 5],
            [0, 5, 4], // y = lo
            [3, 7, 6],
            [3, 6, 2], // y = hi
            [0, 4, 7],
            [0, 7, 3], // x = lo
            [1, 2, 6],
            [1, 6, 5], // x = hi
        ];
        Mesh::new(v, f)
    }

    /// Splits every triangle into `n²` coplanar ones, keeping the winding. Corner vertices end up
    /// duplicated per face, which nothing here minds: only the ray casts read this mesh.
    #[allow(clippy::many_single_char_names, reason = "a test helper over triangle corners")]
    fn subdivide(m: &Mesh, n: usize) -> Mesh {
        let mut v = Vec::new();
        let mut f = Vec::new();
        for t in &m.f {
            let (a, b, c) = (m.v[t[0] as usize], m.v[t[1] as usize], m.v[t[2] as usize]);
            let base = v.len();
            let at = |i: usize, j: usize| base + i * (i + 1) / 2 + j;
            for i in 0..=n {
                for j in 0..=i {
                    #[allow(clippy::cast_precision_loss, reason = "n is small")]
                    let (u, w) = (i as f64 / n as f64, j as f64 / n as f64);
                    v.push([
                        a[0] + (b[0] - a[0]) * (u - w) + (c[0] - a[0]) * w,
                        a[1] + (b[1] - a[1]) * (u - w) + (c[1] - a[1]) * w,
                        a[2] + (b[2] - a[2]) * (u - w) + (c[2] - a[2]) * w,
                    ]);
                }
            }
            for i in 0..n {
                for j in 0..=i {
                    let idx = |k: usize| u32::try_from(k).unwrap();
                    f.push([idx(at(i, j)), idx(at(i + 1, j)), idx(at(i + 1, j + 1))]);
                    if j < i {
                        f.push([idx(at(i, j)), idx(at(i + 1, j + 1)), idx(at(i, j + 1))]);
                    }
                }
            }
        }
        Mesh::new(v, f)
    }

    #[test]
    fn the_percentile_is_numpys_linear_interpolation() {
        // Ten values 1..10: virtual index 0.9·9 = 8.1, so 9 + 0.1·(10 − 9) = 9.1.
        let x: Vec<f32> = (1..=10_u16).map(f32::from).collect();
        assert!((percentile90(&x) - 9.1).abs() < 1e-6);
        // A single value is its own percentile.
        assert_eq!(percentile90(&[7.5]), 7.5);
        // Order does not matter.
        let mut y = x.clone();
        y.reverse();
        assert_eq!(percentile90(&y), percentile90(&x));
    }

    #[test]
    fn the_mode_is_the_centre_of_the_fullest_bin() {
        // 1000 values at 5.0 and a long thin tail: p90 = 5, the mode bin is the last one.
        let mut x = vec![5.0_f32; 1000];
        x.extend((0..100_u16).map(|i| 5.0 + f32::from(i) * 0.5));
        let m = hist_mode(&x);
        assert!((m - 4.958_333).abs() < 1e-4, "mode {m}");

        // A flat histogram takes the first maximal bin, as np.argmax does.
        let x: Vec<f32> = (0..600_u16).map(|i| f32::from(i % 60) + 0.5).collect();
        assert!(hist_mode(&x) < 1.0);
    }

    #[test]
    fn a_hollow_box_measures_its_own_wall() {
        // Two nested boxes: an 80×80×80 shell of wall 10 around a 60³ cavity.
        let outer = subdivide(&box_mesh([-40.0; 3], [40.0; 3]), 4);
        let mut inner = subdivide(&box_mesh([-30.0; 3], [30.0; 3]), 4);
        // Flip the inner box so its normals point into the cavity's wall.
        for t in &mut inner.f {
            t.swap(1, 2);
        }
        let offset = u32::try_from(outer.v.len()).unwrap();
        let mut m = outer;
        m.v.extend(inner.v);
        m.f.extend(inner.f.iter().map(|t| [t[0] + offset, t[1] + offset, t[2] + offset]));

        let geom = face_geometry(&m.v, &m.f);
        let mut rng = seeded(super::SEED);
        let (t, raw) = estimate_thickness(&m.v, &m.f, &geom, &mut rng).expect("the rays hit");
        assert!((t - 10.0).abs() < 0.5, "wall estimate {t}, expected 10");
        assert!((raw - 10.0).abs() < 0.5, "plain mode {raw}");
    }

    #[test]
    fn too_few_hits_refuse_an_estimate() {
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let idx: Vec<u32> = (0..4).collect();
        let hits = RayHits { t_hit: vec![f32::INFINITY; 4], prim: vec![MISS; 4] };
        assert!(thickness_from_hits(&normals, &idx, &hits).is_none());

        // Exactly MIN_HITS − 1 valid hits is still a refusal.
        let n = MIN_HITS - 1;
        let normals = vec![[0.0, 0.0, 1.0]; n];
        let idx: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
        let hits = RayHits { t_hit: vec![3.0; n], prim: (0..u32::try_from(n).unwrap()).collect() };
        assert!(thickness_from_hits(&normals, &idx, &hits).is_none());
    }

    #[test]
    fn a_hit_that_does_not_look_back_is_dropped_from_the_filtered_mode() {
        // 200 rays: 150 hit a face pointing the way they travel (a wall), 50 hit one side-on.
        let n = 200;
        let mut normals = vec![[0.0, 0.0, 1.0]; n + 2];
        normals[n] = [0.0, 0.0, -1.0]; // looks back along a ray fired from a +z face
        normals[n + 1] = [1.0, 0.0, 0.0]; // side-on
        let idx: Vec<u32> = vec![0; n];
        let mut t_hit = Vec::new();
        let mut prim = Vec::new();
        for k in 0..n {
            if k < 150 {
                t_hit.push(4.0);
                prim.push(u32::try_from(n).unwrap());
            } else {
                t_hit.push(20.0);
                prim.push(u32::try_from(n + 1).unwrap());
            }
        }
        let hits = RayHits { t_hit, prim };
        let (t, raw) = thickness_from_hits(&normals, &idx, &hits).expect("enough hits");
        assert!((t - 4.0).abs() < 0.2, "the filtered mode must find the wall, got {t}");
        assert!(raw < t + 20.0);
        assert_ne!(t, raw, "the unfiltered mode also sees the side-on hits");
    }

    #[test]
    fn sampling_is_without_replacement_seeded_and_capped() {
        let mut rng = seeded(0);
        let a = sample_face_indices(1000, 20_000, &mut rng);
        assert_eq!(a.len(), 1000, "asking for more faces than exist takes all of them");
        assert_eq!(a.iter().copied().collect::<HashSet<_>>().len(), 1000);

        let mut rng = seeded(0);
        let b = sample_face_indices(1000, 100, &mut rng);
        assert_eq!(b.len(), 100);
        assert_eq!(b.iter().copied().collect::<HashSet<_>>().len(), 100);
        assert_eq!(b, a[..100], "the same stream draws the same prefix");

        let mut rng = seeded(1);
        assert_ne!(sample_face_indices(1000, 100, &mut rng), b);
    }

    #[test]
    fn the_obb_fallback_is_the_smallest_pca_extent_over_ten() {
        // A 2 × 6 × 30 box, rotated: the smallest extent is 2, so the fallback is 0.2.
        let m = box_mesh([-1.0, -3.0, -15.0], [1.0, 3.0, 15.0]);
        let c = std::f64::consts::FRAC_PI_4.cos();
        let s = std::f64::consts::FRAC_PI_4.sin();
        let rotated: Vec<[f64; 3]> =
            m.v.iter().map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]).collect();
        assert!((obb_min_extent(&rotated) / 10.0 - 0.2).abs() < 1e-9);
        assert_eq!(obb_min_extent(&[]), 0.0);
        // Three points cannot bound a volume; the hull is refused and the PCA runs over them.
        assert_eq!(obb_min_extent(&[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.0);
    }

    /// The OBB fallback against Open3D's own `get_oriented_bounding_box()` (finding F6).
    ///
    /// The two expectations are `min(extent)` as Open3D 0.19 reports it for the two committed slab
    /// pieces, the meshes every checkout has. They are not round numbers and they are not this
    /// port's arithmetic: a PCA over *all* the vertices gives 40.8516 and 38.3579 instead, 3.0 %
    /// and 8.1 % out, which is what this test exists to catch. What is left — 1.6e-9 and 1.0e-8
    /// relative — is `parry3d`'s `f32` hull against Open3D's `f64` Qhull.
    #[test]
    fn the_obb_fallback_is_open3ds_oriented_bounding_box() {
        for (name, open3d) in [("pieceA", 42.061_872_931_582_74), ("pieceB", 41.749_393_252_514_84)]
        {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/slab/input")
                .join(format!("{name}.ply"));
            let mesh = crate::io::read_mesh(&path).expect("the slab piece reads");
            let got = obb_min_extent(&mesh.v);
            let rel = (got - open3d).abs() / open3d;
            assert!(rel < 1e-7, "{name}: {got} against Open3D's {open3d} ({rel:.3e} relative)");
        }
    }
}
