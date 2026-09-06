//! Taubin smoothing with Open3D's weights and iteration count (R §3.3.1).
//!
//! One iteration is one Laplacian step with `λ = 0.5` followed by one with `μ = −0.53`; the
//! reference runs three. A Laplacian step with factor `s` moves *every* vertex, boundary vertices
//! included, to
//!
//! ```text
//! v' = v + s · ( Σ_j w_j v_j / Σ_j w_j − v ),   w_j = 1 / (|v − v_j| + 1e-12)
//! ```
//!
//! over the vertex's edge neighbours, from the positions before the step (Jacobi, not
//! Gauss–Seidel). The `1e-12` is Open3D's own guard against a zero-length edge and is *additive*,
//! not a floor: measured against `filter_smooth_taubin` on a jittered sphere, `1/(d + 1e-12)`
//! agrees to 4e-16 while `1/d` is off by 4e-14 — a hundredfold difference that only that constant
//! explains.
//!
//! **Why the surface has to be smoothed at all.** The segmentation of R §3.4 votes on the
//! direction of a cone of rays around the face normal; on a raw scan the normals of neighbouring
//! triangles disagree by tens of degrees and the vote is noise. Three Taubin iterations bring that
//! down to 1–10° (the reference's own note in `SegParams`) without the shrinkage a pure Laplacian
//! would cause — that is what the negative `μ` pass undoes.
//!
//! **The one thing that is not bit-identical.** Open3D accumulates the sums in the iteration order
//! of a `std::unordered_set<int>`; this module accumulates them in ascending neighbour order,
//! because D §7 forbids unordered containers on a result path. The two orders differ only in
//! round-off — measured over the parity fixtures in
//! `docs/superpowers/notes/2026-09-06-s3-working-mesh.md`.

use super::Mesh;
use super::adjacency::{VertexAdjacency, vertex_adjacency};

/// Iterations the reference runs (`filter_smooth_taubin(number_of_iterations=3)`).
pub const ITERATIONS: usize = 3;
/// The shrinking half-step, Open3D's `lambda_filter` default.
pub const LAMBDA: f64 = 0.5;
/// The un-shrinking half-step, Open3D's `mu` default.
pub const MU: f64 = -0.53;
/// Open3D's additive guard in `weight = 1 / (dist + 1e-12)`.
const WEIGHT_EPS: f64 = 1e-12;

/// R §3.3.1 on a whole mesh: three iterations, `λ = 0.5`, `μ = −0.53`.
///
/// Colours are left alone. Open3D smooths them (and the vertex normals) alongside the positions,
/// but the working mesh carries neither into any later stage — R §11.4 writes the *original*
/// cleaned mesh, not this one — so smoothing them would cost a pass and change nothing.
pub fn taubin(m: &mut Mesh) {
    taubin_with(m, ITERATIONS, LAMBDA, MU);
}

/// [`taubin`] with the three constants exposed, for tests and experiments.
pub fn taubin_with(m: &mut Mesh, iterations: usize, lambda: f64, mu: f64) {
    let adj = vertex_adjacency(m.v.len(), &m.f);
    let mut scratch = m.v.clone();
    for _ in 0..iterations {
        laplacian_step(&mut m.v, &mut scratch, &adj, lambda);
        laplacian_step(&mut m.v, &mut scratch, &adj, mu);
    }
}

/// One Jacobi Laplacian step, in place.
///
/// `scratch` holds the positions from before the step; it is swapped rather than reallocated, so a
/// three-iteration run allocates one extra vertex array in total.
#[allow(
    clippy::many_single_char_names,
    reason = "v, s, p, q, d and w are the symbols R §3.3.1 writes the step in"
)]
fn laplacian_step(v: &mut [[f64; 3]], scratch: &mut Vec<[f64; 3]>, adj: &VertexAdjacency, s: f64) {
    scratch.clear();
    scratch.extend_from_slice(v);
    let prev: &[[f64; 3]] = scratch;
    for (i, out) in v.iter_mut().enumerate() {
        let p = prev[i];
        let neighbours = adj.neighbours(i);
        if neighbours.is_empty() {
            // Open3D divides by a zero total weight here and produces NaN. R §3.1 removes
            // unreferenced vertices before this runs and R §3.3 removes them again after
            // decimation, so the mesh reaching this function has none; leaving the vertex where it
            // is keeps a hand-built mesh usable instead of poisoning it.
            continue;
        }
        let mut sum = [0.0_f64; 3];
        let mut total = 0.0_f64;
        for &j in neighbours {
            let q = prev[j as usize];
            let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            let dist = ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt();
            let w = 1.0 / (dist + WEIGHT_EPS);
            total += w;
            sum[0] += w * q[0];
            sum[1] += w * q[1];
            sum[2] += w * q[2];
        }
        for c in 0..3 {
            out[c] = p[c] + s * (sum[c] / total - p[c]);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "exact fixed points are the point of these tests")]

    use super::{ITERATIONS, LAMBDA, MU, taubin, taubin_with};
    use crate::mesh::Mesh;
    use approx::assert_relative_eq;

    use crate::mesh::geometry::pairwise_sum;

    /// A unit icosphere by two-fold subdivision of a regular icosahedron, and its signed volume.
    #[allow(clippy::many_single_char_names, reason = "a test helper over triangle corners")]
    fn icosphere(subdivisions: usize) -> Mesh {
        let t = f64::midpoint(1.0, 5.0_f64.sqrt());
        let mut v: Vec<[f64; 3]> = vec![
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ];
        let mut f: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for p in &mut v {
            let n = ((p[0] * p[0] + p[1] * p[1]) + p[2] * p[2]).sqrt();
            *p = [p[0] / n, p[1] / n, p[2] / n];
        }
        for _ in 0..subdivisions {
            let mut mid = std::collections::HashMap::new();
            let mut nf = Vec::with_capacity(f.len() * 4);
            for t in &f {
                let mut m = [0_u32; 3];
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    m[k] = *mid.entry(key).or_insert_with(|| {
                        let (pa, pb) = (v[a as usize], v[b as usize]);
                        let mut p = [
                            f64::midpoint(pa[0], pb[0]),
                            f64::midpoint(pa[1], pb[1]),
                            f64::midpoint(pa[2], pb[2]),
                        ];
                        let n = ((p[0] * p[0] + p[1] * p[1]) + p[2] * p[2]).sqrt();
                        p = [p[0] / n, p[1] / n, p[2] / n];
                        v.push(p);
                        u32::try_from(v.len() - 1).unwrap()
                    });
                }
                nf.push([t[0], m[0], m[2]]);
                nf.push([m[0], t[1], m[1]]);
                nf.push([m[2], m[1], t[2]]);
                nf.push([m[0], m[1], m[2]]);
            }
            f = nf;
        }
        Mesh::new(v, f)
    }

    /// Signed volume of a closed mesh, by the divergence theorem.
    fn volume(m: &Mesh) -> f64 {
        let terms: Vec<f64> =
            m.f.iter()
                .map(|t| {
                    let a = m.v[t[0] as usize];
                    let b = m.v[t[1] as usize];
                    let c = m.v[t[2] as usize];
                    (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                        + a[2] * (b[0] * c[1] - b[1] * c[0]))
                        / 6.0
                })
                .collect();
        pairwise_sum(&terms)
    }

    #[test]
    fn the_constants_are_open3ds() {
        assert_eq!((ITERATIONS, LAMBDA, MU), (3, 0.5, -0.53));
    }

    /// Open3D 0.19 on exactly the mesh `icosphere(2)` builds, measured once and pinned here:
    /// `filter_smooth_taubin(number_of_iterations=3)` takes its volume from 4.047044679978849 to
    /// 4.076543327369995, i.e. it *grows* by 0.7289 %. (Three iterations of λ = 0.5 / μ = −0.53
    /// over-correct on a mesh this coarse; the point of the pair is that the volume barely moves
    /// at all.) Six λ-only steps shrink the same sphere by 33.46 %, which is what the μ pass
    /// exists to undo.
    const OPEN3D_VOLUME_BEFORE: f64 = 4.047_044_679_978_849;
    const OPEN3D_VOLUME_AFTER: f64 = 4.076_543_327_369_995;
    const OPEN3D_LAPLACIAN_SHRINK: f64 = 0.334_623_461_724_434_4;

    #[test]
    fn a_sphere_changes_volume_by_exactly_what_open3d_changes_it_by() {
        let mut m = icosphere(2);
        assert_eq!((m.v.len(), m.f.len()), (162, 320), "the same mesh Open3D was measured on");
        let v0 = volume(&m);
        assert_relative_eq!(v0, OPEN3D_VOLUME_BEFORE, max_relative = 1e-12);

        let mut lap_only = m.clone();
        taubin(&mut m);
        let v1 = volume(&m);
        assert_relative_eq!(v1, OPEN3D_VOLUME_AFTER, max_relative = 1e-9);

        // Three iterations of λ = 0.5 alone, i.e. Open3D's
        // `filter_smooth_laplacian(number_of_iterations=6, lambda_filter=0.5)`: Taubin without its
        // un-shrinking half.
        taubin_with(&mut lap_only, 3, LAMBDA, LAMBDA);
        let shrink_lap = (v0 - volume(&lap_only)) / v0;
        assert_relative_eq!(shrink_lap, OPEN3D_LAPLACIAN_SHRINK, max_relative = 1e-9);

        let taubin_change = ((v1 - v0) / v0).abs();
        assert!(
            shrink_lap > 40.0 * taubin_change,
            "the λ-only filter must move the volume far more: {shrink_lap:.4} vs {taubin_change:.4}"
        );
    }

    #[test]
    fn smoothing_pulls_a_noisy_sphere_back_towards_its_surface() {
        let mut m = icosphere(3);
        let mut rng = crate::rng::seeded(7);
        for p in &mut m.v {
            for c in p.iter_mut() {
                let u = f64::from(rand_chacha::rand_core::Rng::next_u32(&mut rng))
                    / f64::from(u32::MAX);
                *c += (u - 0.5) * 0.06;
            }
        }
        // Root-mean-square distance from the unit sphere the vertices started on.
        let roughness = |m: &Mesh| {
            let sq: Vec<f64> =
                m.v.iter()
                    .map(|p| {
                        let r = ((p[0] * p[0] + p[1] * p[1]) + p[2] * p[2]).sqrt();
                        (r - 1.0) * (r - 1.0)
                    })
                    .collect();
            #[allow(clippy::cast_precision_loss, reason = "a few thousand vertices")]
            (pairwise_sum(&sq) / sq.len() as f64).sqrt()
        };
        let before = roughness(&m);
        taubin(&mut m);
        let after = roughness(&m);
        assert!(after < 0.6 * before, "roughness {before:.5} -> {after:.5}");

        // And the surface stays where it was: no systematic inflation or collapse.
        let mean_radius = |m: &Mesh| {
            let r: Vec<f64> =
                m.v.iter().map(|p| ((p[0] * p[0] + p[1] * p[1]) + p[2] * p[2]).sqrt()).collect();
            #[allow(clippy::cast_precision_loss, reason = "a few thousand vertices")]
            (pairwise_sum(&r) / r.len() as f64)
        };
        assert!((mean_radius(&m) - 1.0).abs() < 0.01, "radius {}", mean_radius(&m));
    }

    #[test]
    fn a_flat_grid_is_a_fixed_point_up_to_round_off() {
        // Two triangles in the z = 0 plane: every neighbour average is in the plane, so no vertex
        // may leave it, and the interior of a regular figure may not move at all.
        let mut m = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            vec![[0, 1, 2], [2, 1, 3]],
        );
        let before = m.v.clone();
        taubin(&mut m);
        for (a, b) in before.iter().zip(&m.v) {
            assert_eq!(a[2], b[2], "the filter must not lift a planar mesh out of its plane");
        }
    }

    #[test]
    fn an_isolated_vertex_survives_instead_of_becoming_nan() {
        let mut m = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]],
            vec![[0, 1, 2]],
        );
        taubin(&mut m);
        assert_eq!(m.v[3], [9.0, 9.0, 9.0]);
        assert!(m.v.iter().flatten().all(|x| x.is_finite()));
    }
}
