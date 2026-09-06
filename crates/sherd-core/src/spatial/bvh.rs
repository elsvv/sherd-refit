//! BVH over triangles (R §3.2, R §3.4.3, R §6.1, R §6.4): closest point, bounded closest point,
//! ray casts, and the inside test.
//!
//! Built on `parry3d` 0.30 (`enhanced-determinism`), **without** `TriMeshFlags::ORIENTED`:
//! `project_local_point_with_max_dist` for the bounded closest point (2.0–3.3× faster than the
//! unbounded one at `r_max = 0.35·t`), `CompositeShapeRef::cast_local_ray` for rays, and own
//! Möller–Trumbore ray parity over `TriMesh::bvh()` for the inside test. E4 measured |Δd| against
//! Open3D's `RaycastingScene` at most 3.1e-5 over half a million queries, ~100× inside the gate.
//! Filled in in phase 1b.
