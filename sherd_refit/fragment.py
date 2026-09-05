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
                       face_geometry, median_edge, sample_on_faces, smoothed_normals)

log = logging.getLogger("sherd_refit")

MESH_EXT = (".ply", ".obj", ".stl", ".off")
FACES_PER_T2 = 600      # working-mesh faces per t^2 of surface (~12 edges across the wall)
MIN_FACES = 50000
CACHE_VERSION = 4       # bump when preprocessing/segmentation changes so stale caches are recomputed


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


def _hist_mode(d):
    hist, edges = np.histogram(d, bins=60, range=(0, np.percentile(d, 90)))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def estimate_thickness(scene, C, FN, rng, n=20000):
    """Wall thickness from rays cast inward from random faces, as (robust estimate, plain mode).

    A hit counts only when the face it lands on looks back along the ray (its normal points the
    way the ray travels, cos > 0.7): that is the opposite wall seen from behind, and it drops the
    rays that run along a rim, down a lip or into the fracture surface.  About a third of the rays
    go that way on this data.

    Taking the mode of the *lower* part of what survives was tried against a fragment carrying a
    pot's mouth and does not work, in either direction.  On the terracotta the wall sits at the
    top of the aligned distances -- FY234007's aligned hits run 21.6 at the 5th percentile to
    41.7 at the 90th with the wall at 39 -- because the low tail is oblique rays near the crease,
    not a thinner wall; truncating at the 60th percentile cuts the peak off and the estimate falls
    to 29.3, a 25 % error on the one number every threshold is expressed in.  And on the fragment
    it was meant to rescue it changes almost nothing: pot A piece 01 is genuinely mostly collar,
    its aligned hits start at 4.6 against the pot's 3.5 mm wall, and the estimate moves only from
    6.01 to 5.84.  What actually stops a rim from distorting a comparison is using `min(t_A, t_B)`
    for the pair (see `Scales`) and letting the wall-ratio filter be loose enough to keep the pair
    at all.

    Both numbers are returned; the report prints the unfiltered mode beside the estimate so that a
    fragment whose two values disagree is visible.
    """
    idx = rng.choice(len(C), min(n, len(C)), replace=False)
    dvec = -FN[idx]
    rays = np.concatenate([C[idx] + dvec * 1e-3, dvec], 1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    d = ans["t_hit"].numpy(); prim = ans["primitive_ids"].numpy()
    ok = np.isfinite(d) & (prim < len(FN))
    if ok.sum() < 100:
        return None, None
    raw = _hist_mode(d[ok])
    far = d[ok][np.einsum("ij,ij->i", FN[prim[ok]], dvec[ok]) > 0.7]
    return (_hist_mode(far) if len(far) >= 100 else raw), raw


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
    res: float                  # median edge length of this working mesh (the resolution unit)
    watertight: bool
    n_orig_vertices: int
    n_orig_faces: int
    target_faces: int = 0
    thick_mode: float = 0.0     # plain mode over every inward ray; `thick` is the robust version
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
        thick, thick_mode = estimate_thickness(scene0, C0, FN0, rng)
        if thick is None or thick <= 0:
            thick = float(np.min(m.get_oriented_bounding_box().extent) / 10.0)
            log.warning("%s: thickness estimate failed, using OBB fallback %.2f", name, thick)
        thick_mode = thick if not thick_mode else thick_mode
        if thick_mode > 1.15 * thick:
            log.info("%s: wall %.2f, but the plain ray mode says %.2f -- a rim or a collar", name, thick, thick_mode)
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
        res = median_edge(V, F)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32))))
        log.info("%s: %d faces (from %d, budget %d), thickness %.2f, edge %.3f (%.1f per t), watertight=%s (%.1fs)",
                 name, len(F), n_orig_f, target, thick, res, thick / max(res, 1e-9), watertight, time.time() - t0)

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
        fr = cls(name=name, path=os.path.abspath(path), V=V, F=F, frac=frac, thick=float(thick), res=float(res),
                 watertight=watertight, n_orig_vertices=n_orig_v, n_orig_faces=n_orig_f, target_faces=int(target_faces),
                 thick_mode=float(thick_mode))
        fr.FN, fr.A, fr.C, fr.scene = FN, A, C, scene
        return fr

    # ---------- cache ----------
    def save(self, path: str):
        np.savez_compressed(path, name=self.name, path=self.path, V=self.V, F=self.F, frac=self.frac, thick=self.thick,
                            res=self.res, watertight=self.watertight, n_orig_vertices=self.n_orig_vertices, n_orig_faces=self.n_orig_faces,
                            target_faces=self.target_faces, thick_mode=self.thick_mode, cache_version=self.cache_version,
                            mtime=os.path.getmtime(self.path) if os.path.exists(self.path) else 0.0)

    @classmethod
    def load(cls, path: str) -> "Fragment":
        d = np.load(path, allow_pickle=False)
        fr = cls(name=str(d["name"]), path=str(d["path"]), V=d["V"], F=d["F"], frac=d["frac"], thick=float(d["thick"]),
                 res=float(d["res"]), watertight=bool(d["watertight"]), n_orig_vertices=int(d["n_orig_vertices"]), n_orig_faces=int(d["n_orig_faces"]),
                 target_faces=int(d["target_faces"]) if "target_faces" in d else 0,
                 thick_mode=float(d["thick_mode"]) if "thick_mode" in d else 0.0,
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
    def frac_scene(self):
        """Raycasting scene over the fracture faces alone, built on first use.

        `tight` and `gap` are distances to the other fragment's fracture *surface*.  Measuring them
        against a point sample of that surface put a floor under them equal to the sample spacing;
        measuring them against the triangles themselves removes the floor and leaves only the mesh
        resolution, which the `res` terms of `Scales` already account for.  The scene covers the
        fracture faces only: against the whole mesh a fragment laid flat on its neighbour's outer
        shell would score perfect contact.
        """
        if getattr(self, "_frac_scene", None) is None:
            idx = np.where(self.frac)[0]
            Ff = self.F[idx] if len(idx) else self.F[:1]
            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(self.V.astype(np.float32)),
                                                         o3d.core.Tensor(Ff.astype(np.uint32))))
            self._frac_scene = sc
        return self._frac_scene

    def fracture_distance(self, Q: np.ndarray) -> np.ndarray:
        """Unsigned distance from each query point to this fragment's fracture surface."""
        return self.frac_scene.compute_distance(
            o3d.core.Tensor(np.ascontiguousarray(Q, dtype=np.float32))).numpy().astype(np.float64)

    @property
    def fracture_area(self) -> float:
        return float(self.A[self.frac].sum())

    @property
    def area(self) -> float:
        return float(self.A.sum())

    def stats(self) -> dict:
        ext = self.V.max(0) - self.V.min(0)
        return dict(name=self.name, faces=int(len(self.F)), orig_faces=self.n_orig_faces, orig_vertices=self.n_orig_vertices,
                    thickness=self.thick, thickness_mode=self.thick_mode, resolution=self.res,
                    watertight=self.watertight, extent=[float(x) for x in ext],
                    area=self.area, fracture_area_fraction=self.fracture_area / self.area)


class MatchData:
    """Runtime structures for matching one fragment: breakline with frames, samples, KD-trees, point clouds."""

    def __init__(self, fr: Fragment, t: float, seed: int = 0, surface_points: int = 20000,
                 frac_per_t2: float = 150.0, min_frac_points: int = 5000, max_frac_points: int = 12000,
                 margin_points: int = 6000):
        self.fr = fr
        self.name = fr.name
        self.t = t
        rng = np.random.default_rng(seed)
        V, F, frac, FN, A, C = fr.V, fr.F, fr.frac, fr.FN, fr.A, fr.C
        # Whole-surface samples: the penetration test and the shell margin live on these.
        self.S, sp = sample_on_faces(V, F, A, np.ones(len(F), bool), surface_points, rng)
        self.SN = FN[sp]
        self.S_frac = frac[sp]
        self.S_pen = self.S
        # Fracture samples, drawn on the fracture faces alone at a density fixed in units of `t`,
        # so that a big sherd and a small one are described equally finely.  Since `tight` and
        # `gap` are measured against the other fragment's triangles rather than against its
        # samples, the count no longer sets a floor under them: at a fixed pose the scores are flat
        # in it (pot A at its ground-truth poses moves from tight 0.19 to 0.20 between 50 and 150
        # per t^2).  What the count still buys is the ICP, which averages over that many
        # correspondences -- at 50 per t^2 pot A ends at 5 of 8 fragments placed and at 150 it
        # ends at 7 of 8, on identical thresholds.  Since the cost of the ICP is what grows, the
        # upper bound is what keeps it affordable: uncapped, the largest sherds take 23k-49k
        # points and matching runs 45 % to 226 % longer per pot; at 12k it is within 30 % of what
        # the flat 30 000-sample scheme cost, with the same result.
        n_frac = int(np.clip(frac_per_t2 * float(A[frac].sum()) / t ** 2, min_frac_points, max_frac_points))
        self.Pf, fp = sample_on_faces(V, F, A, frac, n_frac, rng)
        self.Nf = FN[fp]
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
        # Point sets used by ICP and verification, materialised once.  The margin is the larger
        # half of the whole-surface sample and dominates the cost of the pc_reg ICP, so it is
        # thinned to `margin_points`: a uniform random subset of an area-weighted sample is still
        # area-weighted, and the true joins' scores move by less than the sampling noise of the
        # metric itself.  Subsets come from the seeded rng, so two runs see the same points.
        self.margin_idx = _subsample(np.where(self.margin)[0], margin_points, rng)
        self.Pm, self.Nm = self.S[self.margin_idx], self.SN[self.margin_idx]
        self.pc_reg = _pc(np.concatenate([self.Pf, self.Pm]), np.concatenate([self.Nf, self.Nm]))
        self.pc_frac = _pc(self.Pf, self.Nf)
        self.has_frac = len(self.Pf) > 0
        self.pc_brk = _pc(P[self.brk_sub], ns[self.brk_sub])
        self.pc_brk_full = _pc(P, ns)
        self.tree_frac = cKDTree(self.Pf) if len(self.Pf) else None
        self.tree_margin = cKDTree(self.Pm) if len(self.Pm) else None
        self.frac_area = fr.fracture_area

    def signed_distance(self, Q):
        return self.fr.signed_distance(Q)

    def fracture_distance(self, Q):
        return self.fr.fracture_distance(Q)


def _subsample(idx: np.ndarray, n: int, rng) -> np.ndarray:
    """A deterministic random subset of `idx` with at most n entries (sorted, so the order of the
    underlying samples is preserved)."""
    if n <= 0 or len(idx) <= n:
        return idx
    return np.sort(rng.choice(idx, n, replace=False))


def _pc(P, N):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(P)))
    pc.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(N))
    return pc
