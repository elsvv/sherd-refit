"""Small geometry helpers shared by the pipeline stages."""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

_tls = threading.local()


def threads() -> int:
    """Threads for KD-tree queries; the pipeline sets SHERD_REFIT_THREADS per worker to avoid
    oversubscription, and `single_threaded` pins them to one inside a thread pool."""
    if getattr(_tls, "single", False):
        return 1
    return int(os.environ.get("SHERD_REFIT_THREADS", "-1"))


def worker_threads() -> int:
    """Thread budget of this process for parallelism inside one pair.

    SHERD_REFIT_THREADS is set by the pipeline for its worker processes; when it is unset (a
    library call outside the pipeline) the answer is 1, because nothing has capped OpenMP and
    Open3D's ICP is already using every core on its own.
    """
    n = int(os.environ.get("SHERD_REFIT_THREADS", "0"))
    return max(1, n)


@contextmanager
def single_threaded():
    """Force KD-tree queries made in this thread to use a single worker."""
    prev = getattr(_tls, "single", False)
    _tls.single = True
    try:
        yield
    finally:
        _tls.single = prev


def face_geometry(V: np.ndarray, F: np.ndarray):
    """Unit face normals, face areas and centroids."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a2 = np.linalg.norm(n, axis=1)
    FN = n / np.maximum(a2[:, None], 1e-12)
    return FN, 0.5 * a2, V[F].mean(1)


def median_edge(V: np.ndarray, F: np.ndarray) -> float:
    """Median length of the mesh's unique edges: the pipeline's resolution unit `res`.

    Every distance threshold is floored at a multiple of it, so that the pipeline never asks for a
    precision the triangles cannot carry.  Unique edges, so the value does not depend on how many
    faces happen to share one.
    """
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    if len(E) == 0:
        return 0.0
    return float(np.median(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))


def face_adjacency(F: np.ndarray):
    """Pairs (fa, fb) of faces sharing an edge, plus the shared edge (v0, v1) per pair."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fo = np.tile(np.arange(len(F)), 3)
    key = np.sort(E, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, fo = key[order], fo[order]
    same = np.all(ks[1:] == ks[:-1], 1)
    return fo[:-1][same], fo[1:][same], ks[:-1][same]


def ball_matrix(points: np.ndarray, queries: np.ndarray, radius: float, tree: cKDTree | None = None):
    """Sparse (n_queries x n_points) 0/1 matrix of points within radius of each query."""
    tree = tree or cKDTree(points)
    lists = tree.query_ball_point(queries, radius, workers=threads(), return_sorted=False)
    counts = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
    cols = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) if counts.sum() else np.zeros(0, np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return sparse.csr_matrix((np.ones(len(cols)), cols, indptr), shape=(len(queries), len(points)))


def smoothed_normals(W: sparse.csr_matrix, FN: np.ndarray, A: np.ndarray):
    """Area-weighted mean normal over the neighbourhoods encoded in W; returns unit vectors and 1-|mean|."""
    m = W @ (FN * A[:, None])
    wsum = W @ A
    m = m / np.maximum(wsum[:, None], 1e-12)
    norm = np.linalg.norm(m, axis=1)
    return m / np.maximum(norm[:, None], 1e-12), 1.0 - norm


def components_of(mask: np.ndarray, fa: np.ndarray, fb: np.ndarray, n: int):
    """Connected-component label per face restricted to mask (-1 outside)."""
    from scipy.sparse.csgraph import connected_components
    keep = mask[fa] & mask[fb]
    g = sparse.coo_matrix((np.ones(keep.sum()), (fa[keep], fb[keep])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    lab = lab.copy()
    lab[~mask] = -1
    return lab


def drop_small_components(mask: np.ndarray, target: bool, min_area: float, fa, fb, A):
    """Flip components of `mask == target` whose area is below min_area."""
    lab = components_of(mask == target, fa, fb, len(mask))
    if lab.max() < 0:
        return mask
    sizes = np.bincount(lab[lab >= 0], weights=A[lab >= 0], minlength=lab.max() + 1)
    flip = np.isin(lab, np.where(sizes < min_area)[0])
    out = mask.copy()
    out[flip] = not target
    return out


def sample_on_faces(V, F, A, mask, n, rng):
    """Area-weighted random surface samples on the faces selected by mask; returns points and face ids."""
    idx = np.where(mask)[0]
    if len(idx) == 0 or n == 0:
        return np.zeros((0, 3)), np.zeros(0, int)
    p = A[idx] / A[idx].sum()
    pick = idx[rng.choice(len(idx), n, p=p)]
    u, v = rng.random(n), rng.random(n)
    sw = u + v > 1
    u[sw], v[sw] = 1 - u[sw], 1 - v[sw]
    P = V[F[pick, 0]] + u[:, None] * (V[F[pick, 1]] - V[F[pick, 0]]) + v[:, None] * (V[F[pick, 2]] - V[F[pick, 0]])
    return P, pick


def apply_transform(T: np.ndarray, P: np.ndarray):
    return P @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R: np.ndarray):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def make_frames(e1: np.ndarray, e2: np.ndarray, e3: np.ndarray):
    """Stack three (n,3) column vectors into (n,3,3) rotation matrices."""
    return np.stack([e1, e2, e3], 2)
