//! Breaklines and their frames (R §3.5.3–3.5.5).
//!
//! The breakline is where a fragment's fracture meets its shell: on the working mesh, the set of
//! edges whose two faces carry different labels, taken at their midpoints. On an unbroken sherd it
//! is one closed loop; on a sherd broken twice it is two. Everything the matcher does at the
//! coarse scale is anchored on it — the hypotheses of R §5.1 pair one point of A with one point of
//! B and read the pose off their frames, the seam test of R §6.2 measures how much of it a
//! candidate pose brings into contact — so what matters is not only where the points are but which
//! way the surface leaves them.
//!
//! # The frame
//!
//! Each point carries two macro normals and the frame they span:
//!
//! * `ns` — the shell's normal near the point, area-weighted over the shell faces in a
//!   `0.15 t – 0.60 t` annulus around it;
//! * `nf` — the same over the fracture faces;
//! * `f = nf − (nf·ns) ns`, normalised: the fracture's direction with the shell's component taken
//!   out, so that `(ns, f)` is an orthonormal pair whatever the dihedral;
//! * `tangent = ns × f` (R §3.6), which runs *along* the break, and the dihedral
//!   `∠(ns, nf)`, which says how the fracture leaves the wall.
//!
//! **The inner radius is the whole point.** Both surfaces round over into each other at the arris
//! — worn on a real sherd, and Taubin-smoothed on top of that — so faces next to the breakline
//! lean towards the other surface and pull the two macro normals together. Measured at the
//! ground-truth poses over ~1 800 breakline points that really do meet, the two dihedrals of a
//! mating pair sum to 141° with the old neighbourhood (everything within `0.35 t`) where the
//! geometry says 180, and to 179° once the innermost `0.15 t` is left out
//! (`docs/superpowers/notes/2026-09-06-scale-pairs.md`). A point that finds nothing in the
//! annulus — a narrow strip, the end of a chain — falls back to the whole `0.60 t` neighbourhood
//! rather than losing its frame, and one that finds nothing even then is marked invalid and takes
//! no part in the hypotheses.
//!
//! # What this port does differently
//!
//! * **PMC-4.** [`Breaklines::sub`] is the reference's `brk_sub`, and the reference reads it out of
//!   an Open3D hash map, so its *order* is unspecified; this port sorts it ascending. The set is
//!   the same one either way (see [`voxel_representatives`]), but the order fixes the tie-breaking
//!   of R §5.2–5.3 and must be re-verified there.
//! * **Summation order,** exactly as in R §3.4: `query_ball_point(..., return_sorted=False)` hands
//!   the reference each annulus in an unspecified order and this port in ascending index order, so
//!   the two area-weighted sums are free to differ by round-off (D §7). On the fixtures they do
//!   not: the `f64` macro normals are the reference's bit for bit on every fragment measured
//!   (`docs/superpowers/notes/2026-09-06-b2-breaklines.md` §3).
//! * **`f32` storage** (D §4.1), which is therefore the *only* difference this module produces.
//!   The arrays are computed in `f64` and narrowed once, at the end, because that is what the
//!   cache holds and what the GPU will read; the derived tangent and dihedral are then computed
//!   back in `f64` from the narrowed values, so a fragment read from the cache and the same
//!   fragment computed from its file are the same fragment bit for bit. The narrowing is
//!   ≤ 3.7e-5 of a coordinate on the parity fixtures against an injected gate of `1e-4 t`
//!   ≥ 2.3e-4, and ≤ 2.8e-6° of a frame direction. It is loudest in the *dihedral*, where
//!   `arccos` has an infinite derivative at 0° and 180°: a point whose surfaces meet at 0.06°
//!   moves by 0.0026°, which the reference's own arrays reproduce exactly when rounded the same
//!   way (D §10.2).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::segment::{ball_normals, voxel_representatives};
use crate::mesh::adjacency::{FaceAdjacency, face_adjacency};
use crate::mesh::geometry::FaceGeometry;
use crate::spatial::kdtree::PointTree;
use crate::types::FaceLabel;
use crate::vec3::Vec3f;

/// Inner radius of the macro-normal annulus, in wall thicknesses (R §1.1 `macro_inner`).
pub const MACRO_INNER: f64 = 0.15;
/// Outer radius of the macro-normal annulus, in wall thicknesses (R §1.1 `macro_outer`).
pub const MACRO_OUTER: f64 = 0.60;
/// Voxel side of the hypothesis subsample, in wall thicknesses (R §1.1 `brk_voxel`).
pub const BRK_VOXEL: f64 = 0.5;
/// The floor the reference puts under `|f|` before dividing by it (R §3.5.4).
pub const FRAME_FLOOR: f64 = 1e-9;
/// A macro normal, or an in-plane axis, shorter than this is not a direction (R §3.5.4).
pub const VALID_MIN: f64 = 0.5;

/// The knobs R §3.5.3–3.5.5 depend on: the wall thickness they are measured in, and the three
/// radii of R §1.1.
///
/// This is the breakline half of the reference's `md_params` (R §3.7's `mdp_*`): the fields the
/// *sampling* half adds — `seed`, `surface_points`, `frac_per_t2`, `margin_points` and the two
/// fracture-count clamps — draw from the RNG and belong to the arrays of R §3.5.1–3.5.2, which
/// are the next step. A cache whose `brk_params` differ from the run's has its breaklines
/// recomputed and nothing else, which is R §3.7's rule for the match arrays.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrkParams {
    /// Wall thickness `t` the radii below are in units of.
    pub t: f64,
    /// Inner radius of the annulus, in `t`.
    pub macro_inner: f64,
    /// Outer radius of the annulus, in `t`.
    pub macro_outer: f64,
    /// Voxel side of the hypothesis subsample, in `t`.
    pub brk_voxel: f64,
}

impl BrkParams {
    /// The shipped knobs at a wall thickness.
    pub fn at(t: f64) -> Self {
        Self { t, macro_inner: MACRO_INNER, macro_outer: MACRO_OUTER, brk_voxel: BRK_VOXEL }
    }
}

impl Default for BrkParams {
    /// The shipped knobs at `t = 0`, which describes no fragment — [`BrkParams::at`] is the
    /// constructor with a meaning.
    fn default() -> Self {
        Self::at(0.0)
    }
}

/// The breakline of one fragment: the points, their frames, and the hypothesis subset.
///
/// The four point arrays are parallel and in *adjacency order* — the order
/// [`face_adjacency`](crate::mesh::adjacency::face_adjacency) produces the crossing edges in,
/// which is the reference's. [`sub`](Breaklines::sub) indexes into them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Breaklines {
    /// What the arrays were built with (R §3.7).
    pub params: BrkParams,
    /// `brk_P`: the midpoint of every edge whose two faces disagree (R §3.5.3).
    pub p: Vec<Vec3f>,
    /// `brk_ns`: the shell's macro normal at each point (R §3.5.4).
    pub ns: Vec<Vec3f>,
    /// `brk_nf`: the fracture's macro normal at each point.
    pub nf: Vec<Vec3f>,
    /// `brk_f`: `nf` orthogonalised against `ns`, unit.
    pub f: Vec<Vec3f>,
    /// `brk_sub`: the points of the `0.5 t` voxel subsample whose frame is valid (R §3.5.5),
    /// ascending (PMC-4).
    pub sub: Vec<u32>,
}

impl Breaklines {
    /// Number of breakline points.
    #[inline]
    pub fn len(&self) -> usize {
        self.p.len()
    }

    /// True when the fragment has no breakline at all — every face shell, or every face fracture.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }

    /// The points as `f64`, which is what a [`PointTree`] over them wants.
    pub fn points_f64(&self) -> Vec<[f64; 3]> {
        self.p.iter().map(|p| p.to_f64()).collect()
    }

    /// R §3.6's `brk_t = ns × f`: the tangent, running along the break.
    ///
    /// Computed in `f64` from the stored frames and narrowed once, like every other product of
    /// this module.
    pub fn tangents(&self) -> Vec<Vec3f> {
        self.ns
            .iter()
            .zip(&self.f)
            .map(|(ns, f)| Vec3f::from_f64(cross(ns.to_f64(), f.to_f64())))
            .collect()
    }

    /// R §3.6's `brk_dih`: the angle between the two macro normals, in degrees.
    ///
    /// A point whose frame is invalid has a zero macro normal and so a dihedral of 90°, which is
    /// what the reference's `arccos(clip(0))` gives it; nothing reads the dihedral of an invalid
    /// point.
    pub fn dihedrals(&self) -> Vec<f64> {
        self.ns
            .iter()
            .zip(&self.nf)
            .map(|(ns, nf)| dot(ns.to_f64(), nf.to_f64()).clamp(-1.0, 1.0).acos().to_degrees())
            .collect()
    }

    /// R §3.5.4's frame-validity mask: both macro normals are directions, and so is `ns × f`.
    pub fn valid(&self) -> Vec<bool> {
        (0..self.len())
            .map(|i| {
                let (ns, f) = (self.ns[i].to_f64(), self.f[i].to_f64());
                norm(ns) > VALID_MIN
                    && norm(self.nf[i].to_f64()) > VALID_MIN
                    && norm(cross(ns, f)) > VALID_MIN
            })
            .collect()
    }
}

/// R §3.5.3–3.5.5 for one labelled working mesh, computing the face adjacency it needs.
///
/// `v` and `geom` are the mesh's `f64` vertices and per-face arrays — the same ones R §3.4 was run
/// on, which on this port means the ones derived from the *narrowed* working mesh (D §4.1).
pub fn build(
    v: &[[f64; 3]],
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    labels: &[FaceLabel],
    params: BrkParams,
) -> Breaklines {
    build_with(v, &face_adjacency(f), geom, labels, params)
}

/// [`build`] with the face adjacency already in hand.
pub fn build_with(
    v: &[[f64; 3]],
    adjacency: &FaceAdjacency,
    geom: &FaceGeometry,
    labels: &[FaceLabel],
    params: BrkParams,
) -> Breaklines {
    // --- R §3.5.3: the points ------------------------------------------------------------------
    let mut points: Vec<[f64; 3]> = Vec::new();
    for k in 0..adjacency.len() {
        let (left, right) = (adjacency.fa[k] as usize, adjacency.fb[k] as usize);
        if labels[left].is_fracture() != labels[right].is_fracture() {
            let edge = adjacency.edge[k];
            let (a, b) = (v[edge[0] as usize], v[edge[1] as usize]);
            points.push([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]);
        }
    }
    let Some(tree) = PointTree::build(&points) else {
        return Breaklines { params, ..Breaklines::default() };
    };

    // --- R §3.5.4: the macro normals -----------------------------------------------------------
    // The distance from every face centroid to the nearest breakline point, which is what selects
    // the annulus. The reference queries it once per macro normal; it does not depend on the mask.
    let distance: Vec<f64> =
        geom.centroids.par_iter().map(|c| tree.nearest_distance(c).1).collect();
    let ns = macro_normals(geom, labels, false, &points, &distance, params);
    let nf = macro_normals(geom, labels, true, &points, &distance, params);

    let f: Vec<[f64; 3]> = ns
        .iter()
        .zip(&nf)
        .map(|(&ns, &nf)| {
            let d = dot(nf, ns);
            let mut f = [nf[0] - d * ns[0], nf[1] - d * ns[1], nf[2] - d * ns[2]];
            let den = norm(f).max(FRAME_FLOOR);
            for c in &mut f {
                *c /= den;
            }
            f
        })
        .collect();
    let valid: Vec<bool> = (0..points.len())
        .map(|i| {
            norm(ns[i]) > VALID_MIN
                && norm(nf[i]) > VALID_MIN
                && norm(cross(ns[i], f[i])) > VALID_MIN
        })
        .collect();

    // --- R §3.5.5: the hypothesis subset -------------------------------------------------------
    let sub: Vec<u32> = voxel_representatives(&points, params.brk_voxel * params.t)
        .into_iter()
        .filter(|&i| valid[i as usize])
        .collect();

    Breaklines {
        params,
        p: points.iter().map(|&p| Vec3f::from_f64(p)).collect(),
        ns: ns.iter().map(|&n| Vec3f::from_f64(n)).collect(),
        nf: nf.iter().map(|&n| Vec3f::from_f64(n)).collect(),
        f: f.iter().map(|&n| Vec3f::from_f64(n)).collect(),
        sub,
    }
}

/// R §3.5.4's `macro_normals` for one of the two masks: the area-weighted mean normal of the faces
/// of that mask lying between `macro_inner` and `macro_outer` of each breakline point.
///
/// `fracture` picks the mask — `true` for the fracture faces, `false` for the shell. A point whose
/// annulus is empty falls back to the whole `macro_outer` ball over the same mask, and a point
/// that finds nothing there either keeps the zero vector, which is what R §3.5.4's validity test
/// looks for.
fn macro_normals(
    geom: &FaceGeometry,
    labels: &[FaceLabel],
    fracture: bool,
    points: &[[f64; 3]],
    distance: &[f64],
    params: BrkParams,
) -> Vec<[f64; 3]> {
    let (inner, outer) = (params.macro_inner * params.t, params.macro_outer * params.t);
    let all: Vec<u32> = (0..geom.len())
        .filter(|&i| labels[i].is_fracture() == fracture)
        .map(|i| u32::try_from(i).expect("the face count fits in u32"))
        .collect();
    if points.is_empty() || all.is_empty() {
        return vec![[0.0; 3]; points.len()];
    }

    let far: Vec<u32> = all.iter().copied().filter(|&i| distance[i as usize] >= inner).collect();
    let mut n = vec![[0.0; 3]; points.len()];
    if let Some((tree, normals, areas)) = subset(geom, &far) {
        n = ball_normals(&tree, points, &normals, &areas, outer).0;
    }

    let bad: Vec<usize> = (0..points.len()).filter(|&q| norm(n[q]) < VALID_MIN).collect();
    if !bad.is_empty()
        && let Some((tree, normals, areas)) = subset(geom, &all)
    {
        let queries: Vec<[f64; 3]> = bad.iter().map(|&q| points[q]).collect();
        let fallback = ball_normals(&tree, &queries, &normals, &areas, outer).0;
        for (k, &q) in bad.iter().enumerate() {
            n[q] = fallback[k];
        }
    }
    n
}

/// A KD-tree over a subset of the face centroids, with that subset's normals and areas beside it
/// — which is how [`ball_normals`] wants them indexed.
fn subset(geom: &FaceGeometry, faces: &[u32]) -> Option<(PointTree, Vec<[f64; 3]>, Vec<f64>)> {
    let points: Vec<[f64; 3]> = faces.iter().map(|&i| geom.centroids[i as usize]).collect();
    let tree = PointTree::build(&points)?;
    Some((
        tree,
        faces.iter().map(|&i| geom.normals[i as usize]).collect(),
        faces.iter().map(|&i| geom.areas[i as usize]).collect(),
    ))
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{Breaklines, BrkParams, build};
    use crate::mesh::geometry::face_geometry;
    use crate::types::FaceLabel;

    /// A closed, **vertex-welded** box of `n` cells of side `edge`, spanning `[0, n·edge]` per
    /// axis, with outward normals.
    ///
    /// Welded is the point: `segment`'s own test slab builds each side independently, which is
    /// enough for a ray cast but leaves no edge shared between a large face and a side — and a
    /// breakline is made of exactly those edges. Here every side's grid points come out of one
    /// lattice map, so the six sides meet.
    fn welded_box(n: [usize; 3], edge: f64) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        #[allow(clippy::cast_precision_loss, reason = "a lattice of at most a few hundred cells")]
        fn index(
            map: &mut HashMap<[usize; 3], u32>,
            v: &mut Vec<[f64; 3]>,
            p: [usize; 3],
            edge: f64,
        ) -> u32 {
            *map.entry(p).or_insert_with(|| {
                v.push([p[0] as f64 * edge, p[1] as f64 * edge, p[2] as f64 * edge]);
                u32::try_from(v.len() - 1).expect("a small lattice")
            })
        }
        let mut map: HashMap<[usize; 3], u32> = HashMap::new();
        let mut v: Vec<[f64; 3]> = Vec::new();
        let mut f: Vec<[u32; 3]> = Vec::new();
        // (origin, du axis, dv axis) per side, chosen so that `du × dv` points out of the box.
        let sides = [
            ([0, 0, 0], 1, 0),
            ([0, 0, n[2]], 0, 1),
            ([0, 0, 0], 0, 2),
            ([0, n[1], 0], 2, 0),
            ([0, 0, 0], 2, 1),
            ([n[0], 0, 0], 1, 2),
        ];
        for (origin, du, dv) in sides {
            for i in 0..n[du] {
                for j in 0..n[dv] {
                    let at = |a: usize, b: usize| {
                        let mut p = origin;
                        p[du] += a;
                        p[dv] += b;
                        p
                    };
                    let p00 = index(&mut map, &mut v, at(i, j), edge);
                    let p10 = index(&mut map, &mut v, at(i + 1, j), edge);
                    let p11 = index(&mut map, &mut v, at(i + 1, j + 1), edge);
                    let p01 = index(&mut map, &mut v, at(i, j + 1), edge);
                    f.push([p00, p10, p11]);
                    f.push([p00, p11, p01]);
                }
            }
        }
        (v, f)
    }

    /// The slab of R §3.4's own test, welded: 36 × 36 × 6 at 0.5 per edge, so `t = 6` and the wall
    /// is twelve edges across. Its two large faces are the shell and its four sides the fracture
    /// band — which is what `segment_faces` labels it, and here that labelling is given rather
    /// than measured, so the two tests fail independently.
    const T: f64 = 6.0;
    const W: f64 = 36.0;

    fn slab() -> (Vec<[f64; 3]>, Vec<[u32; 3]>, Vec<FaceLabel>) {
        let (v, f) = welded_box([72, 72, 12], 0.5);
        let geom = face_geometry(&v, &f);
        let labels = geom
            .normals
            .iter()
            .map(|n| if n[2].abs() > 0.5 { FaceLabel::Shell } else { FaceLabel::Fracture })
            .collect();
        (v, f, labels)
    }

    fn slab_breaklines(params: BrkParams) -> Breaklines {
        let (v, f, labels) = slab();
        let geom = face_geometry(&v, &f);
        build(&v, &f, &geom, &labels, params)
    }

    /// The slab breaks along its two rims: two closed loops of edge midpoints, one per large face,
    /// every point on the boundary rectangle of its own plane.
    #[test]
    fn the_slab_breaks_along_its_two_rims() {
        let brk = slab_breaklines(BrkParams::at(T));
        assert_eq!(brk.len(), brk.ns.len());
        assert_eq!(brk.len(), brk.nf.len());
        assert_eq!(brk.len(), brk.f.len());

        let (mut low, mut high) = (0, 0);
        for p in &brk.p {
            let q = p.to_f64();
            assert!(q[2].abs() < 1e-4 || (q[2] - T).abs() < 1e-4, "{q:?} is not on a rim");
            if q[2].abs() < 1e-4 {
                low += 1;
            } else {
                high += 1;
            }
            let on_edge = q[0].abs() < 1e-4
                || (q[0] - W).abs() < 1e-4
                || q[1].abs() < 1e-4
                || (q[1] - W).abs() < 1e-4;
            assert!(on_edge, "{q:?} is off the boundary rectangle");
        }
        // 72 edges per side, four sides, one loop per large face and nothing else.
        assert_eq!((low, high), (288, 288));

        // The subsample thins them onto a `0.5·t = 3` grid: 144 of perimeter per rim over 3 is 48
        // voxels, less the corners two runs share, and the two rims are three voxels apart in z.
        assert!(brk.sub.windows(2).all(|w| w[0] < w[1]), "ascending (PMC-4): {:?}", brk.sub);
        assert!(brk.sub.iter().all(|&i| (i as usize) < brk.len()));
        assert!(
            (80..=96).contains(&brk.sub.len()),
            "{} subsampled points for two 48-voxel rims",
            brk.sub.len()
        );
    }

    /// The frame of R §3.5.4–3.6 on a right-angled break: `ns` is the large face's outward normal,
    /// `nf` lies in the wall, the dihedral is 90°, and `ns × f` runs along the rim, right-handed.
    #[test]
    fn the_frames_are_right_angled_and_the_tangent_runs_along_the_rim() {
        let brk = slab_breaklines(BrkParams::at(T));
        let valid = brk.valid();
        let dih = brk.dihedrals();
        let tangent = brk.tangents();
        assert!(valid.iter().all(|&v| v), "every frame of a clean slab is valid");

        for i in 0..brk.len() {
            let p = brk.p[i].to_f64();
            let (ns, nf, f) = (brk.ns[i].to_f64(), brk.nf[i].to_f64(), brk.f[i].to_f64());

            // `ns` is the outward normal of the large face the point sits on: ∓z.
            let want = if p[2].abs() < 1e-4 { -1.0 } else { 1.0 };
            assert!((ns[2] - want).abs() < 1e-5, "point {i}: ns = {ns:?}");
            // `nf` is a side's normal, so it lies in the wall and has nothing along z.
            assert!(nf[2].abs() < 1e-5, "point {i}: nf = {nf:?}");
            assert!(super::dot(f, ns).abs() < 1e-6, "point {i}: f is orthogonal to ns");
            assert!((super::norm(f) - 1.0).abs() < 1e-6);
            assert!((dih[i] - 90.0).abs() < 1e-3, "point {i}: dihedral {}", dih[i]);

            // The tangent is a unit vector along the rim: in the large face's plane, across the
            // direction the fracture faces.
            let tv = tangent[i].to_f64();
            assert!((super::norm(tv) - 1.0).abs() < 1e-5, "point {i}: |t| = {}", super::norm(tv));
            assert!(tv[2].abs() < 1e-5, "the tangent stays in the shell's plane");
            assert!(super::dot(tv, nf).abs() < 1e-5, "point {i}: t·nf = {}", super::dot(tv, nf));

            // R's convention, and the reason one point of A and one of B fix a pose (R §5.1): the
            // frame is right-handed, `ns × f = tangent`, so `f × tangent = ns`.
            let back = super::cross(f, tv);
            for k in 0..3 {
                assert!((back[k] - ns[k]).abs() < 1e-5, "point {i}: f × t = {back:?} ≠ {ns:?}");
            }
        }
    }

    /// A mesh with no fracture face has no breakline, and says so without dividing by anything.
    #[test]
    fn a_fragment_that_never_broke_has_no_breakline() {
        let (v, f) = welded_box([10, 10, 4], 1.0);
        let geom = face_geometry(&v, &f);
        for label in [FaceLabel::Shell, FaceLabel::Fracture] {
            let labels = vec![label; f.len()];
            let brk = build(&v, &f, &geom, &labels, BrkParams::at(4.0));
            assert!(brk.is_empty(), "{label:?}");
            assert_eq!(brk.len(), 0);
            assert!(brk.sub.is_empty());
            assert!(brk.valid().is_empty());
            assert!(brk.dihedrals().is_empty());
            assert!(brk.tangents().is_empty());
            assert_eq!(brk.params, BrkParams::at(4.0));
        }
    }

    /// R §3.5.4's fallback: when the annulus holds nothing, the point takes the whole
    /// neighbourhood rather than losing its frame.
    ///
    /// An inner radius of `10·t = 60` is larger than the slab, so *no* face is far enough from the
    /// breakline to be in any annulus and the first pass leaves every macro normal at zero. The
    /// frames still come out right, which can only be the fallback.
    #[test]
    fn an_empty_annulus_falls_back_to_the_whole_neighbourhood() {
        let brk = slab_breaklines(BrkParams { macro_inner: 10.0, ..BrkParams::at(T) });
        assert_eq!(brk.len(), 576);
        assert!(brk.valid().iter().all(|&v| v), "the fallback keeps every frame");
        assert!(brk.dihedrals().iter().all(|&d| (d - 90.0).abs() < 1e-3));
        assert_eq!(brk.sub.len(), slab_breaklines(BrkParams::at(T)).sub.len());
    }

    /// And a point with no neighbourhood at all — an outer radius below the mesh's own resolution
    /// — keeps a zero frame, is invalid, and drops out of the hypothesis subset.
    #[test]
    fn a_point_with_no_neighbourhood_loses_its_frame_and_its_vote() {
        let brk = slab_breaklines(BrkParams { macro_outer: 1e-6, ..BrkParams::at(T) });
        assert_eq!(brk.len(), 576);
        assert!(brk.valid().iter().all(|&v| !v), "no ball, no frame");
        assert!(brk.ns.iter().all(|n| n.norm() == 0.0));
        assert!(brk.nf.iter().all(|n| n.norm() == 0.0));
        assert!(brk.sub.is_empty(), "an invalid frame is not a hypothesis");
    }
}
