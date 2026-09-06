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

use parry3d::math::Vector;
use parry3d::query::Ray;
use parry3d::shape::{CompositeShapeRef, TriMesh};

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

    /// Number of triangles in the scene.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.mesh.indices().len()
    }
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
