"""Full-resolution refinement of accepted joins along the assembly's spanning tree."""
from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from . import fixture
from .fragment import Fragment, load_mesh
from .geometry import threads, apply_transform
from .matching import Candidate, Params, Scales

log = logging.getLogger("sherd_refit")


def fracture_cloud(fr: Fragment, mesh: o3d.geometry.TriangleMesh, max_points: int = 150000):
    """Original-resolution vertices lying on the working mesh's fracture faces (nearest centroid).

    A vertex can sit up to about half an edge from the nearest face centroid, so the acceptance
    radius is floored at the working mesh's own resolution; on a coarse mesh `0.15 t` alone is
    shorter than one triangle and would throw the fracture away.
    """
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices); N = np.asarray(mesh.vertex_normals)
    d, j = cKDTree(fr.C).query(V, workers=threads())
    sel = fr.frac[j] & (d < max(0.15 * fr.thick, 1.5 * fr.res))
    idx = np.where(sel)[0]
    if len(idx) > max_points:
        idx = np.random.default_rng(0).choice(idx, max_points, replace=False)
    fixture.put(f"{fr.name}.idx", idx, "refine")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[idx]))
    pc.normals = o3d.utility.Vector3dVector(N[idx])
    return pc


def refine_joins(frags: dict[str, Fragment], paths: dict[str, str], poses: dict[str, np.ndarray],
                 groups: list[list[str]], used: list[Candidate], p: Params | None = None) -> dict[str, np.ndarray]:
    """Re-run point-to-plane ICP on full-resolution fracture vertices for every join used by the
    assembly, propagating corrections outward from each group's first fragment.

    The full-resolution meshes are read one at a time and only for fragments that ended up in a
    group: on a collection of 164 they are hundreds of megabytes together, and the ones nothing
    was joined to are never looked at.
    """
    p = p or Params()
    clouds = {}
    joins = []
    with fixture.auto_scope("refine"):
        for n in sorted({n for g in groups if len(g) > 1 for n in g}):
            m = load_mesh(paths[n])
            if len(np.asarray(m.vertices)):
                clouds[n] = fracture_cloud(frags[n], m)
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
            sc = Scales.for_fragments(p, frags[fixed], frags[moving])
            rungs = []
            for dist in (sc.icp_dist(0.05), sc.icp_dist(0.02)):
                r = o3d.pipelines.registration.registration_icp(src, tgt, dist, T, est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                T = r.transformation
                rungs.append(np.asarray(T).tolist())
            joins.append(dict(fixed=fixed, moving=moving, dist=[sc.icp_dist(0.05), sc.icp_dist(0.02)],
                              T_rung=rungs, fitness=float(r.fitness), rmse_t=float(r.inlier_rmse / sc.t)))
            # apply the correction to `moving` and everything already hanging off it (none yet: tree order)
            poses[moving] = T @ poses[moving]
            log.info("refine %s -> %s: fitness %.3f rmse %.3f t", moving, fixed, r.fitness, r.inlier_rmse / sc.t)
            done.add(moving)
            edges.remove(step)
    with fixture.auto_scope("refine"):
        fixture.put("joins", joins, "refine")
        fixture.put("poses_final", {n: np.asarray(T).tolist() for n, T in poses.items()}, "refine")
    return poses
