"""Unit tests for the geometry helpers (reassemble.geometry)."""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
from scipy import sparse

from reassemble.geometry import (apply_transform, ball_matrix, components_of, drop_small_components,
                                 face_adjacency, face_geometry, make_frames, rotation_angle_deg,
                                 sample_on_faces, smoothed_normals)


# ---------------------------------------------------------------- helpers

def closed_meshes():
    box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=3.0, depth=5.0)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=12)
    out = []
    for m in (box, sphere):
        m.remove_duplicated_vertices()
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
        assert m.is_watertight()
        out.append((np.asarray(m.vertices, float), np.asarray(m.triangles, np.int64)))
    return out


def rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = np.deg2rad(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


# ---------------------------------------------------------------- face_adjacency

@pytest.mark.parametrize("V,F", closed_meshes())
def test_face_adjacency_three_neighbours_per_face(V, F):
    fa, fb, ke = face_adjacency(F)
    assert len(fa) == len(fb) == len(ke)
    # a closed triangle mesh has 3F/2 interior edges, each shared by exactly two faces
    assert len(fa) == 3 * len(F) // 2
    deg = np.bincount(np.concatenate([fa, fb]), minlength=len(F))
    assert (deg == 3).all()


@pytest.mark.parametrize("V,F", closed_meshes())
def test_face_adjacency_pairs_are_symmetric_and_share_the_returned_edge(V, F):
    fa, fb, ke = face_adjacency(F)
    # every pair appears once; the relation built from it is symmetric and self-free
    assert (fa != fb).all()
    pairs = {(min(a, b), max(a, b)) for a, b in zip(fa, fb)}
    assert len(pairs) == len(fa)
    nbr = {}
    for a, b in zip(fa, fb):
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    for f, ns in nbr.items():
        for n in ns:
            assert f in nbr[n]
    # the returned edge is exactly the two vertices the two faces have in common
    for a, b, e in zip(fa, fb, ke):
        shared = set(F[a]).intersection(F[b])
        assert shared == {int(e[0]), int(e[1])}
        assert e[0] < e[1]


def test_face_adjacency_ignores_boundary_edges():
    # two triangles sharing one edge: a single adjacency, the other four edges are boundary
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    F = np.array([[0, 1, 2], [1, 3, 2]])
    fa, fb, ke = face_adjacency(F)
    assert len(fa) == 1
    assert {int(fa[0]), int(fb[0])} == {0, 1}
    assert list(ke[0]) == [1, 2]


# ---------------------------------------------------------------- components

def hand_made_graph():
    """8 faces: {0,1,2} chain, {3,4} pair, {5} alone, {6,7} pair; areas 1..8."""
    edges = [(0, 1), (1, 2), (3, 4), (6, 7)]
    fa = np.array([e[0] for e in edges])
    fb = np.array([e[1] for e in edges])
    A = np.arange(1, 9, dtype=float)
    return fa, fb, A


def test_components_of_labels_masked_components():
    fa, fb, _ = hand_made_graph()
    mask = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    lab = components_of(mask, fa, fb, 8)
    assert (lab[~mask] == -1).all()
    assert lab[0] == lab[1] == lab[2]
    assert lab[3] == lab[4]
    assert len({lab[0], lab[3], lab[5]}) == 3       # three separate components
    assert (lab[mask] >= 0).all()


def test_components_of_splits_when_the_mask_cuts_a_chain():
    fa, fb, _ = hand_made_graph()
    mask = np.array([1, 0, 1, 0, 0, 0, 0, 0], bool)   # 0 and 2 are only connected through 1
    lab = components_of(mask, fa, fb, 8)
    assert lab[0] != lab[2]
    assert lab[1] == -1


def test_drop_small_components_flips_only_the_small_ones():
    fa, fb, A = hand_made_graph()
    mask = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    # component areas: {0,1,2} = 1+2+3 = 6, {3,4} = 4+5 = 9, {5} = 6
    out = drop_small_components(mask, True, min_area=7.0, fa=fa, fb=fb, A=A)
    assert list(out) == [False, False, False, True, True, False, False, False]
    # nothing is small enough
    out = drop_small_components(mask, True, min_area=1.0, fa=fa, fb=fb, A=A)
    assert list(out) == list(mask)
    # the False side: {6,7} is one component of area 7 + 8 = 15
    out = drop_small_components(mask, False, min_area=7.5, fa=fa, fb=fb, A=A)
    assert list(out) == list(mask)
    out = drop_small_components(mask, False, min_area=16.0, fa=fa, fb=fb, A=A)
    assert list(out) == [True] * 8


def test_drop_small_components_with_empty_target_returns_input():
    fa, fb, A = hand_made_graph()
    mask = np.ones(8, bool)
    out = drop_small_components(mask, False, min_area=10.0, fa=fa, fb=fb, A=A)
    assert list(out) == list(mask)


# ---------------------------------------------------------------- ball_matrix

def test_ball_matrix_matches_brute_force():
    rng = np.random.default_rng(0)
    P = rng.random((300, 3))
    Q = rng.random((40, 3))
    r = 0.23
    W = ball_matrix(P, Q, r)
    assert W.shape == (len(Q), len(P))
    ref = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2) <= r
    assert np.array_equal(np.asarray(W.todense()) > 0, ref)
    assert set(np.unique(W.data)) <= {1.0}


def test_ball_matrix_empty_and_reused_tree():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1)
    P = rng.random((50, 3))
    Q = P[:5] + 10.0
    W = ball_matrix(P, Q, 0.1)
    assert W.nnz == 0 and W.shape == (5, 50)
    tree = cKDTree(P)
    W1 = ball_matrix(P, P[:5], 0.4, tree=tree)
    W2 = ball_matrix(P, P[:5], 0.4)
    assert np.array_equal(np.asarray(W1.todense()), np.asarray(W2.todense()))
    assert np.diag(np.asarray(W1.todense())[:, :5]).all()      # every point is inside its own ball


# ---------------------------------------------------------------- smoothed_normals

def test_smoothed_normals_on_a_plane_equals_the_plane_normal():
    rng = np.random.default_rng(2)
    n = 200
    FN = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    A = rng.random(n) + 0.1
    W = ball_matrix(rng.random((n, 3)), rng.random((25, 3)), 2.0)   # every query sees every face
    NS, deficit = smoothed_normals(W, FN, A)
    assert np.allclose(NS, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(deficit, 0.0, atol=1e-12)


def test_smoothed_normals_are_unit_vectors_and_area_weighted():
    rng = np.random.default_rng(3)
    FN = rng.normal(size=(60, 3))
    FN /= np.linalg.norm(FN, axis=1, keepdims=True)
    A = rng.random(60) + 0.05
    dense = (rng.random((12, 60)) < 0.5).astype(float)
    dense[0] = 0.0                                             # a query with no neighbours
    W = sparse.csr_matrix(dense)
    NS, deficit = smoothed_normals(W, FN, A)
    busy = np.asarray(W.sum(1)).ravel() > 0
    assert np.allclose(np.linalg.norm(NS[busy], axis=1), 1.0)
    assert np.allclose(NS[~busy], 0.0)
    assert (deficit[busy] >= -1e-12).all() and (deficit[busy] <= 1.0 + 1e-12).all()
    # explicit area weighting for one row
    row = np.asarray(W[1].todense()).ravel() > 0
    m = (FN[row] * A[row, None]).sum(0) / A[row].sum()
    assert np.allclose(NS[1], m / np.linalg.norm(m))
    assert np.isclose(deficit[1], 1.0 - np.linalg.norm(m))


def test_smoothed_normals_cancel_for_opposite_normals():
    FN = np.array([[0.0, 0, 1], [0, 0, -1]])
    A = np.array([1.0, 1.0])
    W = sparse.csr_matrix(np.ones((1, 2)))
    NS, deficit = smoothed_normals(W, FN, A)
    assert np.allclose(NS, 0.0)
    assert np.isclose(deficit[0], 1.0)


# ---------------------------------------------------------------- sample_on_faces

def two_triangles():
    """Face 0 (area 1) in z = 0, face 1 (area 9) in a tilted plane."""
    h = 3.0 / np.sqrt(2.0)
    V = np.array([[0.0, 0, 0], [2, 0, 0], [0, 1, 0],          # face 0: area 1, in z = 0
                  [0.0, 0, 5], [6, 0, 5], [0, h, 5 + h]])      # face 1: area 9, tilted
    F = np.array([[0, 1, 2], [3, 4, 5]])
    FN, A, _ = face_geometry(V, F)
    return V, F, FN, A


def barycentric(V, F, P, pick):
    v0 = V[F[pick, 0]]
    e1 = V[F[pick, 1]] - v0
    e2 = V[F[pick, 2]] - v0
    d = P - v0
    a, b, c = np.einsum("ij,ij->i", e1, e1), np.einsum("ij,ij->i", e1, e2), np.einsum("ij,ij->i", e2, e2)
    d1, d2 = np.einsum("ij,ij->i", d, e1), np.einsum("ij,ij->i", d, e2)
    det = a * c - b * b
    return (c * d1 - b * d2) / det, (a * d2 - b * d1) / det


def test_sample_on_faces_points_lie_on_the_selected_faces():
    V, F, FN, A = two_triangles()
    rng = np.random.default_rng(4)
    P, pick = sample_on_faces(V, F, A, np.ones(2, bool), 5000, rng)
    assert P.shape == (5000, 3) and pick.shape == (5000,)
    # on the plane of the face it came from
    off = np.einsum("ij,ij->i", P - V[F[pick, 0]], FN[pick])
    assert np.abs(off).max() < 1e-9
    # and inside the triangle
    u, v = barycentric(V, F, P, pick)
    assert u.min() > -1e-9 and v.min() > -1e-9 and (u + v).max() < 1 + 1e-9


def test_sample_on_faces_is_area_weighted_and_respects_the_mask():
    V, F, FN, A = two_triangles()
    assert np.allclose(A, [1.0, 9.0])
    rng = np.random.default_rng(5)
    _, pick = sample_on_faces(V, F, A, np.ones(2, bool), 40000, rng)
    assert abs((pick == 1).mean() - 0.9) < 0.02
    P, pick = sample_on_faces(V, F, A, np.array([True, False]), 500, rng)
    assert (pick == 0).all()
    assert np.abs(P[:, 2]).max() < 1e-12          # face 0 lies in z = 0


def test_sample_on_faces_empty_selection():
    V, F, FN, A = two_triangles()
    rng = np.random.default_rng(6)
    P, pick = sample_on_faces(V, F, A, np.zeros(2, bool), 100, rng)
    assert P.shape == (0, 3) and pick.shape == (0,)
    P, pick = sample_on_faces(V, F, A, np.ones(2, bool), 0, rng)
    assert P.shape == (0, 3) and pick.shape == (0,)


# ---------------------------------------------------------------- transforms

def test_rotation_angle_deg_known_rotations():
    assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)
    assert rotation_angle_deg(rot([0, 0, 1], 90.0)) == pytest.approx(90.0)
    # arccos loses precision at the ends of its range, hence the looser tolerance at 180 deg
    assert rotation_angle_deg(rot([1, 1, 0], 180.0)) == pytest.approx(180.0, abs=1e-4)
    rng = np.random.default_rng(7)
    for _ in range(20):
        ang = rng.uniform(1.0, 179.0)
        R = rot(rng.normal(size=3), ang)
        assert rotation_angle_deg(R) == pytest.approx(ang, abs=1e-6)
        assert rotation_angle_deg(R.T) == pytest.approx(ang, abs=1e-6)


def test_rotation_angle_deg_is_clipped_for_slightly_non_orthogonal_input():
    R = np.eye(3) * (1 + 1e-9)
    assert rotation_angle_deg(R) == pytest.approx(0.0)


def test_apply_transform_matches_explicit_matrix_product():
    rng = np.random.default_rng(8)
    P = rng.normal(size=(50, 3))
    T = np.eye(4)
    T[:3, :3] = rot([0.3, -1.0, 0.7], 37.0)
    T[:3, 3] = [1.5, -2.0, 9.0]
    got = apply_transform(T, P)
    want = (T @ np.concatenate([P, np.ones((len(P), 1))], 1).T).T[:, :3]
    assert np.allclose(got, want)
    # composition and inverse
    T2 = np.eye(4)
    T2[:3, :3] = rot([1.0, 0.2, -0.4], 121.0)
    T2[:3, 3] = [-3.0, 4.0, 0.5]
    assert np.allclose(apply_transform(T2, apply_transform(T, P)), apply_transform(T2 @ T, P))
    assert np.allclose(apply_transform(np.linalg.inv(T), apply_transform(T, P)), P)
    assert np.allclose(apply_transform(np.eye(4), P), P)


def test_make_frames_builds_rotation_matrices_from_columns():
    rng = np.random.default_rng(9)
    e1 = rng.normal(size=(7, 3))
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    tmp = rng.normal(size=(7, 3))
    e2 = np.cross(e1, tmp)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    e3 = np.cross(e1, e2)
    R = make_frames(e1, e2, e3)
    assert R.shape == (7, 3, 3)
    assert np.allclose(R[:, :, 0], e1) and np.allclose(R[:, :, 1], e2) and np.allclose(R[:, :, 2], e3)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.broadcast_to(np.eye(3), (7, 3, 3)), atol=1e-12)


def test_face_geometry_normals_areas_centroids():
    V, F, FN, A = two_triangles()
    FN2, A2, C = face_geometry(V, F)
    assert np.allclose(FN, FN2) and np.allclose(A, A2)
    assert np.allclose(np.linalg.norm(FN2, axis=1), 1.0)
    assert np.allclose(C, V[F].mean(1))
    assert np.allclose(FN2[0], [0.0, 0.0, 1.0])
