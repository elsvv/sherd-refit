//! BVH over triangles (R §3.2, R §3.4.3, R §6.1, R §6.4): first-hit ray casts and the closest
//! face.
//!
//! Built on `parry3d` 0.30 (`enhanced-determinism`), **without** `TriMeshFlags::ORIENTED`: the
//! pseudo-normals that flag computes are wrong on decimated fracture surfaces (29 of 30 000 points
//! on one closed manifold fragment, 202 of 30 000 on a non-watertight one), and nothing here needs
//! them — the inside test of R §6.4 is ray parity over the same BVH and lands with the penetration
//! score in phase 1c.
//!
//! Experiment E4 measured this structure against Open3D's `RaycastingScene`, which is what the
//! reference casts through (`docs/superpowers/notes/2026-09-06-e3e4-spatial.md` §2):
//!
//! * **7.87 M cone rays** of R §3.4.3 over ten meshes — 2 hit/miss disagreements, no `t_hit`
//!   difference over tolerance, and 5 primitive ids out of 7.87 M different, every one of them on
//!   a ray grazing a shared edge where either adjacent triangle is a correct answer;
//! * 300 000 closest-point queries — |Δd| ≤ 3.1e-5 against a tolerance of 1e-4·t, ~100× inside the
//!   gate. The projected *primitive id* differs on 12–19 % of them, always between equidistant
//!   faces, so [`RayScene::closest_face`] is used where a nearby face is wanted and never where an
//!   identity is.
//!
//! Vertices are `f32`, which is what Open3D's scene uses too, so both implementations see the same
//! geometry.

use parry3d::math::{Real, Vector};
use parry3d::partitioning::BvhNode;
use parry3d::query::{PointQuery, Ray};
use parry3d::shape::{CompositeShapeRef, TriMesh};

use crate::vec3::Vec3f;

/// A BVH over a triangle mesh: first-hit rays and the closest face.
///
/// One is built per fragment in R §3.2 (on the original component, for the thickness rays) and
/// once more in R §3.4 (on the working mesh, for the cone of seven).
#[derive(Clone, Debug)]
pub struct RayScene {
    mesh: TriMesh,
}

impl RayScene {
    /// Builds the BVH. Returns `None` for a mesh with no triangle, which `parry` refuses.
    #[allow(clippy::cast_possible_truncation, reason = "the scene is f32, as Open3D's is")]
    pub fn new(v: &[[f64; 3]], f: &[[u32; 3]]) -> Option<Self> {
        if f.is_empty() {
            return None;
        }
        let vertices: Vec<Vector> =
            v.iter().map(|p| Vector::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect();
        TriMesh::new(vertices, f.to_vec()).ok().map(|mesh| Self { mesh })
    }

    /// [`RayScene::new`] over vertices that are already `f32` — the working mesh's own (D §4.1).
    ///
    /// No narrowing happens here, so a scene built this way sees exactly the coordinates the rest
    /// of the port measures against.
    pub fn of_mesh(v: &[Vec3f], f: &[[u32; 3]]) -> Option<Self> {
        if f.is_empty() {
            return None;
        }
        let vertices: Vec<Vector> = v.iter().map(|p| Vector::new(p.x, p.y, p.z)).collect();
        TriMesh::new(vertices, f.to_vec()).ok().map(|mesh| Self { mesh })
    }

    /// A scene over the faces `keep` selects, on the same vertices — R §6.1's fracture-only scene.
    ///
    /// `tight` and `gap` are distances to the other fragment's *fracture* surface: against the
    /// whole mesh a fragment laid flat on its neighbour's outer shell would score perfect contact.
    /// A mask that selects nothing falls back to the mesh's **first face**, which is the
    /// reference's own `self.F[:1]`: a scene must have a triangle, and a fragment with no fracture
    /// face has no candidate to score anyway.
    pub fn of_subset(v: &[Vec3f], f: &[[u32; 3]], keep: impl Fn(usize) -> bool) -> Option<Self> {
        let mut faces: Vec<[u32; 3]> =
            f.iter().enumerate().filter(|(i, _)| keep(*i)).map(|(_, t)| *t).collect();
        if faces.is_empty() {
            faces.extend(f.first().copied());
        }
        Self::of_mesh(v, &faces)
    }

    /// First hit along a ray, as `(face index, distance)`.
    ///
    /// The direction is not normalised by this call, so the distance is in units of `|dir|` —
    /// exactly Embree's contract, and the callers here always pass a unit direction.
    pub fn first_hit(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<(u32, f32)> {
        let ray = Ray::new(
            Vector::new(origin[0], origin[1], origin[2]),
            Vector::new(dir[0], dir[1], dir[2]),
        );
        CompositeShapeRef(&self.mesh).cast_local_ray(&ray, f32::MAX, true)
    }

    /// The face nearest to a point, and the distance to it.
    ///
    /// `solid = false`: the projection goes to the surface even for a point inside the mesh, which
    /// is what a label transfer wants. Faces equidistant from the query are resolved by the BVH's
    /// traversal, not by index — E4 measured 12–19 % of queries landing on a different but
    /// equidistant face than Open3D's, so this answers "a nearest face", never "the nearest face".
    pub fn closest_face(&self, point: [f32; 3]) -> Option<(u32, f32)> {
        let p = Vector::new(point[0], point[1], point[2]);
        CompositeShapeRef(&self.mesh)
            .project_local_point(p, f32::MAX, false)
            .map(|(face, proj)| (face, (proj.point - p).length()))
    }

    /// The unsigned distance from a point to the surface — Open3D's `compute_distance`
    /// (R §6.1, R §6.4).
    ///
    /// E4 measured it against Embree on 300 000 queries over ten meshes: the worst difference is
    /// 3.1e-5 against a tolerance of `1e-4·t`, and the residual is float32 summation order.
    pub fn distance(&self, point: [f32; 3]) -> f32 {
        let p = Vector::new(point[0], point[1], point[2]);
        (self.mesh.project_local_point(p, false).point - p).length()
    }

    /// [`RayScene::distance`] with a window: the distance when it is at most `max_dist`, and
    /// `None` when the surface is further away than that.
    ///
    /// This is D §6.5's `bounded_distance`, and R §6.1 is what it exists for: every use of the
    /// fracture distance is a comparison against a threshold at or below `sc.facing`, so a point
    /// whose surface is further than that needs no number at all. E4 measured the call exact
    /// inside the window (0 of 300 000 wrong) and 2.0–3.3× faster than the unbounded one, which
    /// Open3D has no equivalent of.
    pub fn bounded_distance(&self, point: [f32; 3], max_dist: f32) -> Option<f32> {
        if max_dist <= 0.0 || max_dist.is_nan() {
            return None;
        }
        let p = Vector::new(point[0], point[1], point[2]);
        self.mesh
            .project_local_point_with_max_dist(p, false, max_dist)
            .map(|proj| (proj.point - p).length())
    }

    /// Whether a point is inside the mesh, by **ray parity**: three axis-aligned rays, majority
    /// vote (D §6.2, PMC-7).
    ///
    /// Open3D signs its distance from the parity of one ray. E4 measured both rules against
    /// `compute_occupancy` on 270 000 points over the nine watertight benchmark meshes and both
    /// agreed on every one of them, while `parry`'s pseudo-normal test — the reason
    /// [`RayScene::new`] does not ask for `TriMeshFlags::ORIENTED` — called 29 demonstrably
    /// outside points inside on a closed 200 000-face fragment. The majority of three is what is
    /// kept: it costs 1.7–2.5 µs against one ray's 0.55–0.95, and it cannot be fooled by a single
    /// ray grazing an edge.
    ///
    /// The answer is only meaningful on a closed mesh; R §6.4 asks the question only of a
    /// fragment [`Fragment::watertight`](crate::fragment::Fragment::watertight) accepts.
    pub fn inside(&self, point: [f32; 3]) -> bool {
        let origin = Vector::new(point[0], point[1], point[2]);
        let mut votes = 0_u32;
        for axis in 0..3 {
            let mut dir = [0.0_f32; 3];
            dir[axis] = 1.0;
            if self.crossings(origin, Vector::new(dir[0], dir[1], dir[2])) % 2 == 1 {
                votes += 1;
            }
        }
        votes >= 2
    }

    /// How many triangles one ray from `origin` crosses — the inner half of [`RayScene::inside`].
    fn crossings(&self, origin: Vector, dir: Vector) -> u32 {
        let ray = Ray::new(origin, dir);
        let vertices = self.mesh.vertices();
        let mut hits = 0_u32;
        for leaf in
            self.mesh.bvh().leaves(|node: &BvhNode| node.cast_ray(&ray, Real::MAX) < Real::MAX)
        {
            let t = self.mesh.indices()[leaf as usize];
            let (a, b, c) =
                (vertices[t[0] as usize], vertices[t[1] as usize], vertices[t[2] as usize]);
            if moller_trumbore(origin, dir, a, b, c) {
                hits += 1;
            }
        }
        hits
    }

    /// The mesh's axis-aligned bounding box, as `(min, max)`.
    ///
    /// R §6.4's penetration test asks the inside question of 20 000 points per fragment per
    /// candidate, and a point outside this box is outside the mesh with no ray cast at all
    /// (D §6.5).
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let aabb = self.mesh.local_aabb();
        ([aabb.mins.x, aabb.mins.y, aabb.mins.z], [aabb.maxs.x, aabb.maxs.y, aabb.maxs.z])
    }

    /// Number of triangles in the scene.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.mesh.indices().len()
    }
}

/// Möller–Trumbore: does the ray `origin + s·dir`, `s > 0`, cross the triangle `v0 v1 v2`?
///
/// The `1e-9` on the determinant drops a ray parallel to the triangle's plane, and the same
/// constant on `t` drops a hit behind the origin; both are the E4 harness's, which is what
/// measured the parity rule against Open3D.
fn moller_trumbore(origin: Vector, dir: Vector, v0: Vector, v1: Vector, v2: Vector) -> bool {
    const EPS: f32 = 1e-9;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < EPS {
        return false;
    }
    let inv = 1.0 / det;
    let tvec = origin - v0;
    let u = inv * tvec.dot(pvec);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let qvec = tvec.cross(e1);
    let v = inv * dir.dot(qvec);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    inv * e2.dot(qvec) > EPS
}

#[cfg(test)]
mod tests {
    use super::RayScene;

    /// An axis-aligned box from `lo` to `hi`, outward normals, twelve triangles.
    fn box_mesh(lo: [f64; 3], hi: [f64; 3]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
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
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    #[test]
    fn the_scene_reports_first_hits_and_misses() {
        let (v, f) = box_mesh([-1.0; 3], [1.0; 3]);
        let scene = RayScene::new(&v, &f).expect("twelve triangles");
        assert_eq!(scene.n_faces(), 12);
        let (_, t) = scene.first_hit([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]).expect("hits the lid");
        assert!((t - 4.0).abs() < 1e-5, "distance {t}");
        let (_, t) = scene.first_hit([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("hits from inside");
        assert!((t - 1.0).abs() < 1e-5, "distance {t}");
        assert!(scene.first_hit([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]).is_none(), "away from the box");
        assert!(RayScene::new(&v, &[]).is_none());
    }

    #[test]
    fn the_closest_face_is_on_the_surface_from_either_side() {
        let (v, f) = box_mesh([-1.0; 3], [1.0; 3]);
        let scene = RayScene::new(&v, &f).expect("twelve triangles");
        // Outside, straight above the lid.
        let (face, d) = scene.closest_face([0.0, 0.0, 3.0]).expect("a nearest face");
        assert!((d - 2.0).abs() < 1e-5, "distance {d}");
        assert!(matches!(face, 2 | 3), "one of the two lid triangles, got {face}");
        // Inside: `solid = false` projects to the boundary rather than answering zero.
        let (_, d) = scene.closest_face([0.0, 0.0, 0.5]).expect("a nearest face");
        assert!((d - 0.5).abs() < 1e-5, "distance {d}");
    }
}
