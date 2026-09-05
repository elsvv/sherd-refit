"""Per-fragment preprocessing: load, decimate, estimate wall thickness, segment shell vs
fracture, extract breaklines with local frames. Results are cached as .npz so that the
matching stage can run in worker processes without recomputation."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .geometry import (threads, apply_transform, ball_matrix, drop_small_components, face_adjacency,
                       face_geometry, sample_on_faces, smoothed_normals)

log = logging.getLogger("reassemble")

MESH_EXT = (".ply", ".obj", ".stl", ".off")
FACES_PER_T2 = 600      # working-mesh faces per t^2 of surface (~12 edges across the wall)
MIN_FACES = 50000
CACHE_VERSION = 2       # bump when preprocessing/segmentation changes so stale caches are recomputed


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) == 0:
        raise ValueError(f"{path}: no triangles")
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    return m


def largest_component(m: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    labels, counts, _ = m.cluster_connected_triangles()
    labels = np.asarray(labels)
    if len(counts) > 1:
        keep = labels == int(np.argmax(counts))
        m = o3d.geometry.TriangleMesh(m)
        m.remove_triangles_by_mask(~keep)
        m.remove_unreferenced_vertices()
    return m


def estimate_thickness(scene, C, FN, rng, n=20000):
    idx = rng.choice(len(C), min(n, len(C)), replace=False)
    rays = np.concatenate([C[idx] - FN[idx] * 1e-3, -FN[idx]], 1).astype(np.float32)
    d = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    d = d[np.isfinite(d)]
    if len(d) < 100:
        return None
    hist, edges = np.histogram(d, bins=60, range=(0, np.percentile(d, 90)))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def classify_faces(scene, C, FN, NS, thick, n_faces):
    """Shell test: cone of 7 rays around -NS; shell if >= 5 rays hit the far wall at 0.5..1.8 t from behind."""
    n = NS
    a = np.where((np.abs(n[:, 0]) < 0.9)[:, None], np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(n, e1)
    ang = np.deg2rad(15.0)
    K = 7
    good = np.zeros(len(C), int)
    for k in range(K):
        if k == 0:
            dvec = -n
        else:
            phi = 2 * np.pi * (k - 1) / (K - 1)
            dvec = -(np.cos(ang) * n) + np.sin(ang) * (np.cos(phi) * e1 + np.sin(phi) * e2)
        rays = np.concatenate([C - FN * 1e-3, dvec], 1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        dh = ans["t_hit"].numpy(); prim = ans["primitive_ids"].numpy()
        ok = np.isfinite(dh) & (prim < n_faces) & (dh > 0.1 * thick)
        al = np.full(len(C), -1.0)
        al[ok] = np.einsum("ij,ij->i", FN[prim[ok]], dvec[ok])
        good += (ok & (dh / thick > 0.5) & (dh / thick < 1.8) & (al > 0.7)).astype(int)
    return good >= 5


def closed_enough(F, max_boundary_fraction=0.002):
    """True if the mesh has (almost) no boundary edges: every edge shared by exactly two faces."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    key = E[:, 0].astype(np.int64) * (F.max() + 1) + E[:, 1]
    _, counts = np.unique(key, return_counts=True)
    n_boundary = int((counts != 2).sum())
    return n_boundary <= max_boundary_fraction * len(counts), n_boundary


def coarse_grid(C, spacing):
    """Representative face indices on a voxel grid of the given spacing, and the nearest-grid-point index per face."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(C))
    _, _, lists = pc.voxel_down_sample_and_trace(spacing, C.min(0) - 1, C.max(0) + 1)
    rep = np.array([l[0] for l in lists], dtype=int)
    _, near = cKDTree(C[rep]).query(C, workers=threads())
    return rep, near


def refine_boundary(frac, fa, fb, C, FN, A, thick, grid, max_passes=60, angle_deg=25.0):
    """Grow the shell into the fracture band while face normals stay within angle of the
    original shell's area-weighted normal within t/2 (fixed reference, no drift)."""
    cos_grow = np.cos(np.deg2rad(angle_deg))
    rep, near = grid
    shell0 = ~frac
    sidx = np.where(shell0)[0]
    if len(sidx) == 0:
        return frac
    W = ball_matrix(C[sidx], C[rep], thick / 2.0)
    ref_g, _ = smoothed_normals(W, FN[sidx], A[sidx])
    ref = ref_g[near]
    has_ref = (np.asarray(W.sum(1)).ravel() > 0)[near]
    frac = frac.copy()
    for _ in range(max_passes):
        cand = np.unique(np.concatenate([fa[frac[fa] & ~frac[fb]], fb[frac[fb] & ~frac[fa]]]))
        if len(cand) == 0:
            break
        flip = has_ref[cand] & (np.einsum("ij,ij->i", FN[cand], ref[cand]) > cos_grow)
        if not flip.any():
            break
        frac[cand[flip]] = False
    return frac


@dataclass
class Fragment:
    """Working-resolution fragment with segmentation and breakline data (cache-able)."""
    name: str
    path: str
    V: np.ndarray
    F: np.ndarray
    frac: np.ndarray
    thick: float
    watertight: bool
    n_orig_vertices: int
    n_orig_faces: int
    target_faces: int = 0
    cache_version: int = CACHE_VERSION
    # derived (filled by `finalize`)
    FN: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)
    C: np.ndarray = field(default=None, repr=False)
    scene: object = field(default=None, repr=False)

    # ---------- construction ----------
    @classmethod
    def from_mesh_file(cls, path: str, target_faces: int = 200000, seed: int = 0, name: str | None = None) -> "Fragment":
        t0 = time.time()
        name = name or os.path.splitext(os.path.basename(path))[0]
        m = load_mesh(path)
        n_orig_v, n_orig_f = len(m.vertices), len(m.triangles)
        m = largest_component(m)
        rng = np.random.default_rng(seed)
        # wall thickness from the original mesh, then a face budget that keeps ~12 edges across the wall
        V0 = np.asarray(m.vertices, dtype=np.float64); F0 = np.asarray(m.triangles, dtype=np.int64)
        FN0, A0, C0 = face_geometry(V0, F0)
        scene0 = o3d.t.geometry.RaycastingScene()
        scene0.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V0.astype(np.float32)), o3d.core.Tensor(F0.astype(np.uint32))))
        thick = estimate_thickness(scene0, C0, FN0, rng)
        if thick is None or thick <= 0:
            thick = float(np.min(m.get_oriented_bounding_box().extent) / 10.0)
            log.warning("%s: thickness estimate failed, using OBB fallback %.2f", name, thick)
        target = int(np.clip(FACES_PER_T2 * A0.sum() / thick ** 2, MIN_FACES, target_faces))
        del scene0, V0, F0, FN0, A0, C0
        if len(m.triangles) > target:
            m = m.simplify_quadric_decimation(target_number_of_triangles=target)
            m.remove_degenerate_triangles(); m.remove_duplicated_vertices(); m.remove_unreferenced_vertices()
        m = m.filter_smooth_taubin(number_of_iterations=3)
        m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
        V = np.asarray(m.vertices, dtype=np.float64); F = np.asarray(m.triangles, dtype=np.int64)
        # "watertight" here means: closed enough for a reliable signed distance (ray parity).
        # Decimation may open a few tiny holes; tolerate up to 0.2 % boundary edges.
        watertight, n_boundary = closed_enough(F)
        if not watertight:
            log.warning("%s: working mesh has %d boundary edges; penetration tests will be skipped for it", name, n_boundary)
        FN, A, C = face_geometry(V, F)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32))))
        log.info("%s: %d faces (from %d, budget %d), thickness %.2f, watertight=%s (%.1fs)", name, len(F), n_orig_f, target, thick, watertight, time.time() - t0)

        # segmentation (smooth fields are evaluated on a t/8 grid and looked up per face)
        ctree = cKDTree(C)
        grid = coarse_grid(C, thick / 8.0)
        rep, near = grid
        W = ball_matrix(C, C[rep], thick / 3.0, tree=ctree)
        NS_g, _ = smoothed_normals(W, FN, A)
        NS = NS_g[near]
        shell = classify_faces(scene, C, FN, NS, thick, len(F))
        frac = ~shell
        raw_frac = float(A[frac].sum() / A.sum())
        Wm = ball_matrix(C, C[rep], thick / 4.0, tree=ctree)
        frac = (np.asarray(Wm @ (A * frac)).ravel() > 0.5 * np.asarray(Wm @ A).ravel())[near]
        fa, fb, _ = face_adjacency(F)
        frac = drop_small_components(frac, True, 0.5 * thick ** 2, fa, fb, A)
        frac = drop_small_components(frac, False, 2.0 * thick ** 2, fa, fb, A)
        frac = refine_boundary(frac, fa, fb, C, FN, A, thick, grid)
        frac = drop_small_components(frac, True, 0.5 * thick ** 2, fa, fb, A)
        log.info("%s: fracture area fraction raw %.3f -> final %.3f (%.1fs)", name, raw_frac, A[frac].sum() / A.sum(), time.time() - t0)
        fr = cls(name=name, path=os.path.abspath(path), V=V, F=F, frac=frac, thick=float(thick), watertight=watertight,
                 n_orig_vertices=n_orig_v, n_orig_faces=n_orig_f, target_faces=int(target_faces))
        fr.FN, fr.A, fr.C, fr.scene = FN, A, C, scene
        return fr

    # ---------- cache ----------
    def save(self, path: str):
        np.savez_compressed(path, name=self.name, path=self.path, V=self.V, F=self.F, frac=self.frac, thick=self.thick,
                            watertight=self.watertight, n_orig_vertices=self.n_orig_vertices, n_orig_faces=self.n_orig_faces,
                            target_faces=self.target_faces, cache_version=self.cache_version,
                            mtime=os.path.getmtime(self.path) if os.path.exists(self.path) else 0.0)

    @classmethod
    def load(cls, path: str) -> "Fragment":
        d = np.load(path, allow_pickle=False)
        fr = cls(name=str(d["name"]), path=str(d["path"]), V=d["V"], F=d["F"], frac=d["frac"], thick=float(d["thick"]),
                 watertight=bool(d["watertight"]), n_orig_vertices=int(d["n_orig_vertices"]), n_orig_faces=int(d["n_orig_faces"]),
                 target_faces=int(d["target_faces"]) if "target_faces" in d else 0,
                 cache_version=int(d["cache_version"]) if "cache_version" in d else 0)
        fr.cache_mtime = float(d["mtime"]) if "mtime" in d else 0.0
        fr.FN, fr.A, fr.C = face_geometry(fr.V, fr.F)
        fr.scene = o3d.t.geometry.RaycastingScene()
        fr.scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(fr.V.astype(np.float32)), o3d.core.Tensor(fr.F.astype(np.uint32))))
        return fr

    # ---------- queries ----------
    def cache_valid_for(self, path: str, target_faces: int) -> bool:
        """True if this cached fragment was computed from `path` with the same settings and file version."""
        same_file = self.path == os.path.abspath(path) and os.path.exists(path) and abs(getattr(self, "cache_mtime", 0.0) - os.path.getmtime(path)) < 1.0
        return same_file and self.target_faces == int(target_faces) and self.cache_version == CACHE_VERSION

    def signed_distance(self, Q: np.ndarray) -> np.ndarray:
        return self.scene.compute_signed_distance(o3d.core.Tensor(Q.astype(np.float32))).numpy()

    @property
    def fracture_area(self) -> float:
        return float(self.A[self.frac].sum())

    @property
    def area(self) -> float:
        return float(self.A.sum())

    def stats(self) -> dict:
        ext = self.V.max(0) - self.V.min(0)
        return dict(name=self.name, faces=int(len(self.F)), orig_faces=self.n_orig_faces, orig_vertices=self.n_orig_vertices,
                    thickness=self.thick, watertight=self.watertight, extent=[float(x) for x in ext],
                    area=self.area, fracture_area_fraction=self.fracture_area / self.area)


class MatchData:
    """Runtime structures for matching one fragment: breakline with frames, samples, KD-trees, point clouds."""

    def __init__(self, fr: Fragment, t: float, seed: int = 0, n_samples: int = 30000):
        self.fr = fr
        self.name = fr.name
        self.t = t
        rng = np.random.default_rng(seed)
        V, F, frac, FN, A, C = fr.V, fr.F, fr.frac, fr.FN, fr.A, fr.C
        # surface samples
        self.S, sp = sample_on_faces(V, F, A, np.ones(len(F), bool), n_samples, rng)
        self.SN = FN[sp]
        self.S_frac = frac[sp]
        # breakline points: midpoints of edges between shell and fracture faces
        fa, fb, ke = face_adjacency(F)
        cross = frac[fa] != frac[fb]
        ke = ke[cross]
        P = 0.5 * (V[ke[:, 0]] + V[ke[:, 1]])
        self.brk_P = P
        self.brk_tree = cKDTree(P) if len(P) else None
        sh_idx, fr_idx = np.where(~frac)[0], np.where(frac)[0]
        if len(P) and len(sh_idx) and len(fr_idx):
            Ws = ball_matrix(C[sh_idx], P, 0.35 * t); Wf = ball_matrix(C[fr_idx], P, 0.35 * t)
            ns, _ = smoothed_normals(Ws, FN[sh_idx], A[sh_idx]); nf, _ = smoothed_normals(Wf, FN[fr_idx], A[fr_idx])
        else:
            ns = np.zeros((len(P), 3)); nf = np.zeros((len(P), 3))
        f = nf - np.einsum("ij,ij->i", nf, ns)[:, None] * ns
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
        self.brk_ns, self.brk_nf, self.brk_f, self.brk_t = ns, nf, f, np.cross(ns, f)
        self.brk_dih = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", ns, nf), -1, 1)))
        valid = (np.linalg.norm(ns, axis=1) > 0.5) & (np.linalg.norm(nf, axis=1) > 0.5) & (np.linalg.norm(self.brk_t, axis=1) > 0.5)
        # subsample for hypotheses (voxel t/3), valid frames only
        if len(P):
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
            _, _, lists = pc.voxel_down_sample_and_trace(t / 3.0, P.min(0) - 1, P.max(0) + 1)
            sub = np.array([l[0] for l in lists], dtype=int)
            self.brk_sub = sub[valid[sub]]
        else:
            self.brk_sub = np.zeros(0, int)
        # fracture points and shell margin for ICP
        d_brk = self.brk_tree.query(self.S, workers=threads())[0] if self.brk_tree is not None else np.full(len(self.S), np.inf)
        # shell margin for ICP / continuity: shell points near the seam, excluding a thin band next to the
        # breakline where crease faces misclassified as shell would otherwise dominate the nearest-neighbour test
        self.margin = (~self.S_frac) & (d_brk > 0.12 * t) & (d_brk < 1.5 * t)
        self.pc_reg = _pc(np.concatenate([self.S[self.S_frac], self.S[self.margin]]), np.concatenate([self.SN[self.S_frac], self.SN[self.margin]]))
        self.pc_frac = _pc(self.S[self.S_frac], self.SN[self.S_frac])
        self.pc_brk = _pc(P[self.brk_sub], ns[self.brk_sub])
        self.pc_brk_full = _pc(P, ns)
        self.tree_frac = cKDTree(self.S[self.S_frac]) if self.S_frac.any() else None
        self.tree_margin = cKDTree(self.S[self.margin]) if self.margin.any() else None
        self.frac_area = fr.fracture_area

    def signed_distance(self, Q):
        return self.fr.signed_distance(Q)


def _pc(P, N):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(P)))
    pc.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(N))
    return pc
