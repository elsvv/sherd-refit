//! Full-resolution refinement (R §9): after assembly, each join used is re-registered on the
//! **original** meshes' fracture vertices and the correction is propagated along each group's
//! spanning tree.
//!
//! Everything R §8 worked on is decimated — a 1.2 M-face scan is a 150 000-face working mesh by
//! the time the matcher sees it — so a pose that is right to a tenth of a wall on the working
//! mesh can still be a fraction of a millimetre out on the scan. R §9 spends one more pair of ICP
//! rungs per join on the full-resolution geometry, and it spends them only where the assembly
//! already committed: the joins in `used`, in `used` order, walked outward from each group's first
//! fragment so that a fragment is corrected exactly once and always against something already
//! placed.
//!
//! # What a fracture cloud is
//!
//! [`fracture_cloud`] takes the **original** mesh — R §3.1's cleaning steps 1–2, and *not* reduced
//! to the largest component, which is the one place in the pipeline where the discarded shells
//! come back — and keeps the vertices whose nearest working-mesh face centroid is a fracture face
//! and is close enough to be that face's own vertex. "Close enough" is
//! `max(0.15·t, 1.5·res)`: a vertex sits up to about half an edge from the nearest centroid, so
//! `0.15·t` alone is shorter than one triangle on a coarse mesh and would throw the whole fracture
//! away. Normals are Open3D's `ComputeVertexNormals` — the area-weighted sum of the incident
//! faces' *unnormalised* normals, then one normalisation ([`vertex_normals`]).
//!
//! # The ladder
//!
//! Two point-to-plane rungs at `sc.icp_dist(0.05)` and `sc.icp_dist(0.02)`, 40 iterations each,
//! through the same [`icp::register`] every other stage uses —
//! R §7 has one implementation in this port and R §9 is one more caller of it. The pair's scales
//! are R §1.2's at `min(t_fixed, t_moving)` and `max(res_fixed, res_moving)`, the pair's own
//! numbers and not the collection's.
//!
//! # The one draw (R §10, PMC-9)
//!
//! A fragment whose fracture keeps more than [`MAX_POINTS`] vertices has the excess dropped by
//! `rng(0).choice(idx, 150000, replace=False)`, whose *order* is the ICP's summation order. The
//! port cannot reproduce numpy's PCG64 (PMC-9), so [`select_vertices`] draws from this port's own
//! seeded stream; the parity harness injects the reference's own `refine/<name>.idx` instead, and
//! measures the port's own selection against it on every fragment that stays under the cap — which
//! is 47 of the 50 fragments the eight dumps refine.
//!
//! Filled in by phase-1d step D2, with the `refine` row of D §10.2 behind it.

use nalgebra::Matrix4;

use crate::matching::icp::{self, IcpTarget, Options};
use crate::matching::scales::Scales;
use crate::mesh::Mesh;
use crate::params::Params;
use crate::rng;
use crate::spatial::kdtree::PointTree;
use crate::types::{FragId, apply_transform_fused, rotate_fused};

/// R §9's cap on one fragment's fracture cloud.
pub const MAX_POINTS: usize = 150_000;

/// The seed R §9's cap draws with: the literal **0**, not [`Params::seed`](crate::params::Params).
///
/// R §10's inventory gives this stream its own line — "refinement (§9) | 0 |
/// `choice(idx, 150000, replace=False)`" — and `refine.py:35` is `np.random.default_rng(0)`, a
/// literal, where every other stream of the reference takes `p.seed`. The two are the same number
/// today because no CLI exposes `--seed`; they would part on the day one does, and the port would
/// be the side that moved (V4-D9).
pub const CAP_SEED: u64 = 0;
/// R §9's acceptance radius in wall thicknesses.
pub const SELECT_T: f64 = 0.15;
/// R §9's acceptance radius in working-mesh edges — the floor that keeps a coarse mesh's fracture.
pub const SELECT_RES: f64 = 1.5;
/// R §9's two rungs, in the `k` of `sc.icp_dist(k)`.
pub const RUNGS: [f64; 2] = [0.05, 0.02];
/// R §9's iteration cap per rung.
pub const ITERATIONS: usize = 40;

/// One fragment's full-resolution fracture cloud: which vertices, and where they are.
///
/// `idx` is kept beside the points because it is what the fixture dump carries
/// (`refine/<name>.idx`) and because its **order** is the ICP's summation order (D §7): a capped
/// selection is a permutation of a subset, and two implementations that keep the same vertices in
/// a different order do not compute the same pose.
#[derive(Clone, Debug, Default)]
pub struct FractureCloud {
    /// Indices into the original mesh's vertices, in the order the ICP will read them.
    pub idx: Vec<u32>,
    /// The vertices themselves.
    pub points: Vec<[f64; 3]>,
    /// Their vertex normals.
    pub normals: Vec<[f64; 3]>,
}

impl FractureCloud {
    /// How many vertices the cloud holds.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True when nothing was selected.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The cloud moved through a pose, as Open3D's `PointCloud::Transform` moves it: points
    /// through the full 4×4, normals through the rotation block alone.
    pub fn placed(&self, pose: &Matrix4<f64>) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        (
            self.points.iter().map(|&p| apply_transform_fused(pose, p)).collect(),
            self.normals.iter().map(|&n| rotate_fused(pose, n)).collect(),
        )
    }
}

/// Open3D's `ComputeVertexNormals`: the sum of the incident faces' **unnormalised** normals,
/// normalised once at the end.
///
/// The unnormalised face normal is `(v₁ − v₀) × (v₂ − v₀)`, whose length is twice the face's area,
/// so the sum is area-weighted without ever computing an area. The accumulation is in face order,
/// which is Open3D's loop and therefore the reference's rounding. A vertex no face references —
/// or three collinear ones — leaves a zero vector, which Open3D replaces by `(0, 0, 1)` after the
/// division produces `NaN`.
pub fn vertex_normals(mesh: &Mesh) -> Vec<[f64; 3]> {
    let mut normals = vec![[0.0_f64; 3]; mesh.v.len()];
    for tri in &mesh.f {
        let origin = mesh.v[tri[0] as usize];
        let (second, third) = (mesh.v[tri[1] as usize], mesh.v[tri[2] as usize]);
        let along = |p: [f64; 3]| [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
        let (edge1, edge2) = (along(second), along(third));
        let face = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        for &vertex in tri {
            let slot = &mut normals[vertex as usize];
            for (axis, term) in slot.iter_mut().zip(face) {
                *axis += term;
            }
        }
    }
    for n in &mut normals {
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        // Eigen's `normalize()` divides by the norm and leaves a NaN behind on a zero vector,
        // which Open3D's `NormalizeNormals` then replaces by `(0, 0, 1)`.
        if length > 0.0 {
            for axis in n.iter_mut() {
                *axis /= length;
            }
        } else {
            *n = [0.0, 0.0, 1.0];
        }
    }
    normals
}

/// R §9's predicate, uncapped: every vertex with `frac[j] ∧ d < max(0.15·t, 1.5·res)`, ascending.
///
/// `centroids` and `fracture` describe the **working** mesh — its face centroids and R §3.4's
/// label per face — and `v` the original mesh's vertices. `j` is the nearest centroid of each
/// vertex and `d` its distance, both from one KD-tree over the centroids (the reference's
/// `cKDTree(fr.C).query(V)`). This is `np.where(sel)[0]` and it is what both implementations
/// compute exactly; the only place they part is [`cap_selection`] above 150 000.
pub fn select_candidates(
    v: &[[f64; 3]],
    centroids: &[[f64; 3]],
    fracture: &[bool],
    thick: f64,
    res: f64,
) -> Vec<u32> {
    let Some(tree) = PointTree::build(centroids) else { return Vec::new() };
    let radius = (SELECT_T * thick).max(SELECT_RES * res);
    let mut idx: Vec<u32> = Vec::new();
    for (i, point) in v.iter().enumerate() {
        let (face, distance) = tree.nearest_distance(point);
        if fracture.get(face as usize).copied().unwrap_or(false) && distance < radius {
            idx.push(u32::try_from(i).expect("fewer than 2^32 vertices"));
        }
    }
    idx
}

/// R §9's cap: `rng(seed).choice(idx, max_points, replace=False)` when there are too many.
///
/// **PMC-9.** This is not numpy's draw and the result is not sorted — it is a permutation of a
/// subset, and its order is the ICP's summation order. Under the cap the selection comes back
/// untouched, which is where the two implementations agree exactly.
pub fn cap_selection(mut idx: Vec<u32>, max_points: usize, seed: u64) -> Vec<u32> {
    if idx.len() > max_points {
        let mut generator = rng::seeded(seed);
        let take = rng::without_replacement(idx.len(), max_points, &mut generator);
        idx = take.into_iter().map(|k| idx[k as usize]).collect();
    }
    idx
}

/// [`select_candidates`] then [`cap_selection`] — R §9's selection end to end.
pub fn select_vertices(
    v: &[[f64; 3]],
    centroids: &[[f64; 3]],
    fracture: &[bool],
    thick: f64,
    res: f64,
    max_points: usize,
    seed: u64,
) -> Vec<u32> {
    cap_selection(select_candidates(v, centroids, fracture, thick, res), max_points, seed)
}

/// The points and normals of a selection, in the selection's own order.
pub fn cloud_from_indices(v: &[[f64; 3]], normals: &[[f64; 3]], idx: &[u32]) -> FractureCloud {
    FractureCloud {
        idx: idx.to_vec(),
        points: idx.iter().map(|&i| v[i as usize]).collect(),
        normals: idx.iter().map(|&i| normals[i as usize]).collect(),
    }
}

/// R §9's `fracture_cloud` end to end: the original mesh in, one fragment's cloud out.
pub fn fracture_cloud(
    original: &Mesh,
    centroids: &[[f64; 3]],
    fracture: &[bool],
    thick: f64,
    res: f64,
    seed: u64,
) -> FractureCloud {
    let idx = select_vertices(&original.v, centroids, fracture, thick, res, MAX_POINTS, seed);
    cloud_from_indices(&original.v, &vertex_normals(original), &idx)
}

/// One fragment as R §9 reads it: the two numbers R §1.2 resolves the rungs from, and its cloud.
///
/// A fragment with no cloud — one outside every group of two or more, or one whose original file
/// holds no vertex — takes no part in the walk. Taking the cloud as a borrow rather than a
/// `Fragment` is what lets the parity harness build a piece out of the reference's own
/// `refine/<name>.idx` (D §10.2's injected column for this stage).
#[derive(Debug)]
pub struct RefinePiece<'a> {
    /// R §3.2's wall thickness — the fragment's own.
    pub thick: f64,
    /// R §3.3's `res`.
    pub res: f64,
    /// The full-resolution fracture cloud, or `None` when the fragment has none.
    pub cloud: Option<&'a FractureCloud>,
}

/// What one join's refinement produced — the fields the fixture's `refine/joins.json` carries.
#[derive(Clone, Copy, Debug)]
pub struct RefinedJoin {
    /// The endpoint already placed; its pose does not move.
    pub fixed: FragId,
    /// The endpoint being corrected.
    pub moving: FragId,
    /// The two rungs' correspondence distances.
    pub dist: [f64; 2],
    /// The pose after each rung, in ladder order.
    pub rungs: [Matrix4<f64>; 2],
    /// The last rung's `fitness`.
    pub fitness: f64,
    /// The last rung's `inlier_rmse`, in wall thicknesses.
    pub rmse_t: f64,
}

/// R §9's result: one pose per fragment and one entry per join walked.
#[derive(Clone, Debug)]
pub struct Refinement {
    /// The poses, with every refined fragment's corrected.
    pub poses: Vec<Matrix4<f64>>,
    /// The joins the walk took, in the order it took them.
    pub joins: Vec<RefinedJoin>,
}

/// R §9's spanning walk over every group, in group order.
///
/// `used` is R §8's list of joins **in the order the assembly took them**, as `(a, b)` pairs; the
/// walk filters it per group and keeps that order, because "the first edge with exactly one
/// endpoint in `done`" is a statement about that order and about nothing else. A group of one is
/// skipped, and so is a join whose moving side has no cloud — the reference would raise there, and
/// there is no fragment in the eight dumps that does.
///
/// The correction is applied as `poses[moving] ← T · poses[moving]` and nothing else moves: the
/// walk reaches a fragment only once, from a neighbour that is already final, so there is never a
/// subtree hanging off `moving` to carry along.
pub fn refine_joins(
    pieces: &[RefinePiece<'_>],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
    used: &[(FragId, FragId)],
    params: &Params,
    numerics: icp::Numerics,
) -> Refinement {
    let mut out = poses.to_vec();
    let mut joins = Vec::new();
    for group in groups {
        if group.len() < 2 {
            continue;
        }
        let mut done = vec![group[0]];
        let mut edges: Vec<(FragId, FragId)> =
            used.iter().copied().filter(|(a, b)| group.contains(a) && group.contains(b)).collect();
        loop {
            let placed = |n: &FragId| done.contains(n);
            let Some(step) = edges.iter().position(|(a, b)| placed(a) != placed(b)) else {
                break;
            };
            let (a, b) = edges.remove(step);
            let (fixed, moving) = if done.contains(&a) { (a, b) } else { (b, a) };
            if let Some(join) = refine_one(pieces, &out, fixed, moving, params, numerics) {
                out[moving as usize] = join.rungs[1] * out[moving as usize];
                joins.push(join);
            } else {
                tracing::warn!(
                    fixed,
                    moving,
                    "no full-resolution fracture cloud for one side of a join; pose left as it is"
                );
            }
            done.push(moving);
        }
    }
    Refinement { poses: out, joins }
}

/// One join's two rungs, at the poses the walk has reached.
fn refine_one(
    pieces: &[RefinePiece<'_>],
    poses: &[Matrix4<f64>],
    fixed: FragId,
    moving: FragId,
    params: &Params,
    numerics: icp::Numerics,
) -> Option<RefinedJoin> {
    let (target_piece, source_piece) = (&pieces[fixed as usize], &pieces[moving as usize]);
    let (source_cloud, target_cloud) = (source_piece.cloud?, target_piece.cloud?);
    let scales = Scales::for_pair(
        params,
        target_piece.thick.min(source_piece.thick),
        target_piece.res.max(source_piece.res),
    );

    let (source, _) = source_cloud.placed(&poses[moving as usize]);
    let (target_points, target_normals) = target_cloud.placed(&poses[fixed as usize]);
    let target = IcpTarget::new(target_points, target_normals);

    let mut pose = Matrix4::identity();
    let mut rungs = [Matrix4::identity(); 2];
    let mut dist = [0.0; 2];
    let mut last = None;
    for (i, k) in RUNGS.into_iter().enumerate() {
        let options = Options {
            estimation: icp::Estimation::PointToPlane,
            max_correspondence_distance: scales.icp_dist(k),
            max_iteration: ITERATIONS,
            numerics,
        };
        let result = icp::register(&source, &target, &pose, &options);
        pose = result.transform;
        rungs[i] = pose;
        dist[i] = options.max_correspondence_distance;
        last = Some(result);
    }
    let last = last.expect("R §9's ladder has two rungs");
    Some(RefinedJoin {
        fixed,
        moving,
        dist,
        rungs,
        fitness: last.fitness,
        rmse_t: last.inlier_rmse / scales.t,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        FractureCloud, MAX_POINTS, RefinePiece, refine_joins, select_vertices, vertex_normals,
    };
    use crate::matching::icp::Numerics;
    use crate::mesh::Mesh;
    use crate::params::Params;
    use nalgebra::Matrix4;

    /// Open3D's rule: the sum of the incident faces' *unnormalised* normals — each of length twice
    /// its face's area — then one normalisation.
    ///
    /// Vertex 0 below sits on a small triangle whose unit normal is `(0, 0, 1)` and a twenty times
    /// larger one whose unit normal is `(0, 1, 0)`. Open3D's answer is `(0, 10, 1)` normalised;
    /// the mean of the two *unit* normals — the rule this is not — would be `(0, 1, 1)/√2`, which
    /// is 39° away.
    #[test]
    fn vertex_normals_are_open3ds_area_weighted_sum() {
        let mesh = Mesh {
            v: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [10.0, 0.0, 0.0],
            ],
            f: vec![[0, 1, 2], [0, 3, 4]],
            colors: None,
        };
        let n = vertex_normals(&mesh);
        let scale = 101.0_f64.sqrt();
        for (axis, expected) in [0.0, 10.0 / scale, 1.0 / scale].into_iter().enumerate() {
            assert!((n[0][axis] - expected).abs() < 1e-15, "{:?}", n[0]);
        }
        // A vertex only the first face touches keeps that face's own normal.
        assert_eq!(n[1].map(f64::to_bits), [0.0_f64, 0.0, 1.0].map(f64::to_bits));

        // A vertex no face references keeps Open3D's `(0, 0, 1)` rather than a NaN.
        let mut lonely = mesh;
        lonely.v.push([9.0, 9.0, 9.0]);
        assert_eq!(
            vertex_normals(&lonely)[5].map(f64::to_bits),
            [0.0_f64, 0.0, 1.0].map(f64::to_bits)
        );
    }

    /// The selection is the reference's predicate and nothing else: a fracture centroid *and* a
    /// distance strictly under `max(0.15 t, 1.5 res)`.
    #[test]
    fn the_selection_is_the_fracture_faces_own_vertices() {
        let v = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let centroids = vec![[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [9.5, 0.0, 0.0]];
        let fracture = vec![true, false, true];
        // radius = max(0.15 * 2, 1.5 * 0.1) = 0.3: vertex 0 is 0.1 from a fracture centroid,
        // vertex 1's nearest centroid is a shell one, vertex 2 is 0.5 from its fracture centroid.
        let idx = select_vertices(&v, &centroids, &fracture, 2.0, 0.1, MAX_POINTS, 0);
        assert_eq!(idx, vec![0]);

        // The floor is what keeps a coarse mesh's fracture: at res = 1.0 the radius is 1.5 and
        // vertex 2 comes in as well, still in ascending order.
        let idx = select_vertices(&v, &centroids, &fracture, 2.0, 1.0, MAX_POINTS, 0);
        assert_eq!(idx, vec![0, 2]);

        // The cap keeps `max_points` of them and is a permutation, not a prefix.
        let idx = select_vertices(&v, &centroids, &fracture, 2.0, 1.0, 1, 0);
        assert_eq!(idx.len(), 1);
        assert!(idx[0] == 0 || idx[0] == 2);
    }

    /// The walk is R §9's: `used` order inside a group, outward from `g[0]`, one correction per
    /// fragment, and a singleton group left alone.
    #[test]
    fn the_walk_starts_at_the_groups_first_fragment_and_reaches_each_once() {
        // Three copies of the same saddle patch — a surface whose normals span enough directions
        // for a point-to-plane step to be well posed. Fragments 1 and 2 start displaced from where
        // the assembly put them, so a corrected pose is one that puts them back at the origin.
        let saddle = || -> FractureCloud {
            let (mut points, mut normals) = (Vec::new(), Vec::new());
            for i in 0..20 {
                for j in 0..20 {
                    let (x, y) = (f64::from(i) * 0.1, f64::from(j) * 0.1);
                    let z = 0.4 * x * x - 0.25 * y * y + 0.1 * x * y;
                    let g = [-(0.8 * x + 0.1 * y), -(0.1 * x - 0.5 * y), 1.0];
                    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                    points.push([x, y, z]);
                    normals.push([g[0] / len, g[1] / len, g[2] / len]);
                }
            }
            FractureCloud { idx: (0..400).collect(), points, normals }
        };
        let clouds = [saddle(), saddle(), saddle()];
        let pieces: Vec<RefinePiece<'_>> =
            clouds.iter().map(|c| RefinePiece { thick: 1.0, res: 0.05, cloud: Some(c) }).collect();
        let mut poses = vec![Matrix4::identity(); 3];
        poses[1][(2, 3)] = 0.004;
        poses[2][(2, 3)] = 0.006;

        let out = refine_joins(
            &pieces,
            &poses,
            &[vec![0, 1, 2], vec![]],
            &[(1, 2), (0, 1)],
            &Params::default(),
            Numerics::REFERENCE,
        );
        // `used` order is (1,2) then (0,1); the walk starts at 0, so (1,2) has no endpoint in
        // `done` on the first pass and (0,1) is taken first. Then (1,2).
        assert_eq!(
            out.joins.iter().map(|j| (j.fixed, j.moving)).collect::<Vec<_>>(),
            vec![(0, 1), (1, 2)]
        );
        // Both moved fragments are pulled back onto fragment 0's plane.
        for n in 1..3 {
            assert!(
                out.poses[n][(2, 3)].abs() < 1e-9,
                "fragment {n} left at {}",
                out.poses[n][(2, 3)]
            );
        }
        // The empty group contributed nothing.
        assert_eq!(out.joins.len(), 2);
    }

    /// A group whose join has no cloud on one side leaves the pose alone instead of panicking.
    #[test]
    fn a_join_without_a_cloud_is_skipped() {
        let cloud = FractureCloud {
            idx: vec![0],
            points: vec![[0.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]],
        };
        let pieces = vec![
            RefinePiece { thick: 1.0, res: 0.1, cloud: Some(&cloud) },
            RefinePiece { thick: 1.0, res: 0.1, cloud: None },
        ];
        let poses = vec![Matrix4::identity(); 2];
        let out = refine_joins(
            &pieces,
            &poses,
            &[vec![0, 1]],
            &[(0, 1)],
            &Params::default(),
            Numerics::REFERENCE,
        );
        assert!(out.joins.is_empty());
        assert_eq!(out.poses, poses);
    }
}
