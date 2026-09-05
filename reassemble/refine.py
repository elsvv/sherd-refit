"""Full-resolution refinement of accepted joins along the assembly's spanning tree."""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .fragment import Fragment, load_mesh
from .geometry import threads, apply_transform
from .matching import Candidate

log = logging.getLogger("reassemble")


def fracture_cloud(fr: Fragment, mesh: o3d.geometry.TriangleMesh, t: float, max_points: int = 150000):
    """Original-resolution vertices lying on the working mesh's fracture faces (nearest centroid)."""
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices); N = np.asarray(mesh.vertex_normals)
    d, j = cKDTree(fr.C).query(V, workers=threads())
    sel = fr.frac[j] & (d < 0.15 * t)
    idx = np.where(sel)[0]
    if len(idx) > max_points:
        idx = np.random.default_rng(0).choice(idx, max_points, replace=False)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[idx]))
    pc.normals = o3d.utility.Vector3dVector(N[idx])
    return pc


def refine_joins(frags: dict[str, Fragment], meshes: dict[str, o3d.geometry.TriangleMesh], poses: dict[str, np.ndarray],
                 groups: list[list[str]], used: list[Candidate], t: float) -> dict[str, np.ndarray]:
    """Re-run point-to-plane ICP on full-resolution fracture vertices for every join used by the
    assembly, propagating corrections outward from each group's first fragment."""
    clouds = {n: fracture_cloud(frags[n], meshes[n], t) for n in frags if len(np.asarray(meshes[n].vertices))}
    poses = dict(poses)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    for g in groups:
        if len(g) < 2:
            continue
        done = {g[0]}
        edges = [c for c in used if c.a in g and c.b in g]
        while True:
            step = next((c for c in edges if (c.a in done) != (c.b in done)), None)
            if step is None:
                break
            fixed, moving = (step.a, step.b) if step.a in done else (step.b, step.a)
            src = o3d.geometry.PointCloud(clouds[moving]); src.transform(poses[moving])
            tgt = o3d.geometry.PointCloud(clouds[fixed]); tgt.transform(poses[fixed])
            T = np.eye(4)
            for dist in (0.05 * t, 0.02 * t):
                r = o3d.pipelines.registration.registration_icp(src, tgt, dist, T, est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                T = r.transformation
            # apply the correction to `moving` and everything already hanging off it (none yet: tree order)
            poses[moving] = T @ poses[moving]
            log.info("refine %s -> %s: fitness %.3f rmse %.3f t", moving, fixed, r.fitness, r.inlier_rmse / t)
            done.add(moving)
            edges.remove(step)
    return poses
