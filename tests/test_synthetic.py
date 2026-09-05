"""End-to-end checks on a synthetic sherd pair with a known ground-truth transform.

A curved slab (300 x 200 units, wall thickness 30) is cut in two along a bumpy fracture
surface x = f(y, z).  Both halves are built from the same fracture-surface vertex grid, so
they are exactly complementary, and each is a closed triangle mesh.  Every piece then gets its
own random rigid transform, so the relative pose the pipeline has to recover is known exactly.
"""
from __future__ import annotations

import json
import os

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import cKDTree

from sherd_refit import assembly, matching, pipeline
from sherd_refit.fragment import Fragment, MatchData
from sherd_refit.geometry import apply_transform, face_geometry, rotation_angle_deg
from sherd_refit.matching import Params

# ---------------------------------------------------------------- slab geometry

X_LEFT, X_RIGHT, X_CUT0 = -150.0, 150.0, 20.0      # extent along x and where the cut sits
HY, HZ = 100.0, 15.0                                # half width, half wall thickness
R1, R2 = 600.0, 1400.0                              # curvature radii of the shell (elliptic, not a surface of revolution)
THICK = 2 * HZ                                      # ground-truth wall thickness
EDGE = 2.0                                          # target triangle edge length
SHELL_NOISE = 0.2                                   # per-vertex noise on the shell faces


def _cut_x(yv, zeta):
    """Fracture surface: bumps of ~5 units at wavelengths 25-64 units, leaning through the wall
    by 3 units so that crease faces are far from perpendicular to the shells (this used to make
    `classify_faces` label a band of fracture as shell; the margin now excludes a thin band next to
    the breakline, and this test guards that fix).
    """
    z = THICK * zeta
    return (X_CUT0 + 5.0 * np.sin(2 * np.pi * yv / 60.0)
            + 3.0 * np.sin(2 * np.pi * z / 40.0 + 0.9)
            + 1.0 * np.sin(2 * np.pi * yv / 25.0 + 2.0))


def _end_x(yv, zeta, side):
    """The far end of each piece: also a rough (fracture-like) surface, uncorrelated with the cut."""
    z = THICK * zeta
    if side < 0:
        return X_LEFT + 3.0 * np.sin(2 * np.pi * yv / 45.0 + 0.4) + 2.0 * np.sin(2 * np.pi * z / 35.0 + 1.7)
    return X_RIGHT + 3.0 * np.sin(2 * np.pi * yv / 38.0 + 2.4) + 2.0 * np.sin(2 * np.pi * z / 33.0 + 0.2)


def _y_of(yv, x, zeta, v):
    """Tapered, gently bent plan shape with wavy side rims (functions of x only, so the cut
    grid is identical for both pieces)."""
    z = THICK * zeta
    b0 = 3.0 * np.sin(2 * np.pi * x / 47.0 + 0.6) + 1.5 * np.sin(2 * np.pi * z / 29.0 + 1.1)
    b1 = 3.0 * np.sin(2 * np.pi * x / 41.0 + 2.9) + 1.5 * np.sin(2 * np.pi * z / 26.0 + 0.4)
    return yv * (1.0 + 0.18 * x / 150.0) + 9.0 * np.sin(2 * np.pi * x / 520.0) + (1 - v) * b0 + v * b1


def _z_of(x, y, zeta):
    return -(x * x / (2 * R1) + y * y / (2 * R2)) + THICK * zeta


def _quads(G):
    """Triangles of an index grid G of shape (m+1, n+1), wound so the normal is d/da x d/db."""
    a, b, c, d = G[:-1, :-1], G[1:, :-1], G[1:, 1:], G[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                           np.stack([a, c, d], -1).reshape(-1, 3)])


def build_piece(side: int, seed: int = 0, edge: float = EDGE):
    """One half of the slab as a closed triangle mesh.

    Returns (V, F, is_cut_face, is_shell_face); `side` < 0 keeps x <= cut, > 0 keeps x >= cut.
    The parameter cube (u, v, w) maps to the piece with u = 0 on the fracture surface, so both
    pieces share the fracture grid exactly.
    """
    x_end = X_LEFT if side < 0 else X_RIGHT
    nu = max(4, int(round(abs(x_end - X_CUT0) / edge)))
    nv = max(4, int(round(2 * HY / edge)))
    nw = max(3, int(round(2 * HZ / edge)))
    U, V, W = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1),
                          np.linspace(0, 1, nw + 1), indexing="ij")
    zeta = W - 0.5
    yv = -HY + 2 * HY * V
    xc = _cut_x(yv, zeta)
    X = xc + U * (_end_x(yv, zeta, side) - xc)
    Y = _y_of(yv, X, zeta, V)
    P = np.stack([X, Y, _z_of(X, Y, zeta)], -1)
    if SHELL_NOISE > 0:            # roughen the shell, but never the shared fracture ring (u = 0)
        rng = np.random.default_rng(seed)
        m = np.zeros(P.shape[:3], bool)
        m[1:, :, 0] = True
        m[1:, :, -1] = True
        P[m, 2] += SHELL_NOISE * rng.standard_normal(int(m.sum()))
    idx = np.arange(P[..., 0].size).reshape(P.shape[:3])
    parts = [_quads(idx[0].T),          # u = 0  fracture
             _quads(idx[-1]),           # u = 1  far end
             _quads(idx[:, 0, :]),      # v = 0  rim
             _quads(idx[:, -1, :].T),   # v = 1  rim
             _quads(idx[:, :, 0].T),    # w = 0  shell
             _quads(idx[:, :, -1])]     # w = 1  shell
    F = np.concatenate(parts)
    tag = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
    used, F = np.unique(F, return_inverse=True)
    F = F.reshape(-1, 3)
    Vv = P.reshape(-1, 3)[used]
    if np.einsum("ij,ij->i", Vv[F[:, 0]], np.cross(Vv[F[:, 1]], Vv[F[:, 2]])).sum() < 0:
        F = F[:, ::-1].copy()          # orient outward
    return Vv, F, tag == 0, tag >= 4


def rigid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    ang = rng.uniform(0.4, 2.5)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    T[:3, 3] = rng.uniform(-200.0, 200.0, 3)
    return T


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Two complementary PLY pieces in their own directory, with ground truth."""
    d = tmp_path_factory.mktemp("synthetic")
    inp = d / "input"
    inp.mkdir()
    truth = {}
    for side, name, seed in ((-1, "pieceA", 1), (1, "pieceB", 2)):
        V, F, is_cut, is_shell = build_piece(side, seed=seed)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        assert m.is_watertight() and m.is_edge_manifold()
        T = rigid(seed)
        Vt = apply_transform(T, V)
        mt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vt), o3d.utility.Vector3iVector(F))
        path = str(inp / f"{name}.ply")
        assert o3d.io.write_triangle_mesh(path, mt, write_ascii=False)
        _, A0, C0 = face_geometry(Vt, F)
        truth[name] = dict(path=path, T=T, C=C0, area=A0, cut=is_cut, shell=is_shell, n_faces=len(F))
    # transform that maps pieceB's file coordinates into pieceA's
    truth["T_rel"] = truth["pieceA"]["T"] @ np.linalg.inv(truth["pieceB"]["T"])
    truth["dir"] = str(inp)
    truth["out"] = str(d / "out")
    return truth


@pytest.fixture(scope="module")
def frags(pair):
    return {n: Fragment.from_mesh_file(pair[n]["path"]) for n in ("pieceA", "pieceB")}


@pytest.fixture(scope="module")
def thickness(frags):
    return float(np.mean([fr.thick for fr in frags.values()]))


@pytest.fixture(scope="module")
def md(frags, thickness):
    return {n: MatchData(fr, thickness) for n, fr in frags.items()}


@pytest.fixture(scope="module")
def cands(md, thickness):
    return matching.match_pair(md["pieceA"], md["pieceB"], Params(), keep=5)


def pose_error(T_est, T_true, points):
    """(rotation error in degrees, largest displacement of `points`)."""
    ang = rotation_angle_deg(T_est[:3, :3].T @ T_true[:3, :3])
    d = np.linalg.norm(apply_transform(T_est, points) - apply_transform(T_true, points), axis=1).max()
    return ang, float(d)


# ---------------------------------------------------------------- A: segmentation

def test_pieces_are_complementary_closed_meshes(pair):
    """Sanity check on the test data itself: the two cut surfaces are the same vertex set."""
    cuts = []
    for side, seed in ((-1, 1), (1, 2)):
        V, F, is_cut, _ = build_piece(side, seed=seed)
        cuts.append(V[np.unique(F[is_cut])])
    assert len(cuts[0]) == len(cuts[1])
    d, _ = cKDTree(cuts[1]).query(cuts[0])
    assert d.max() == 0.0
    assert 20000 <= pair["pieceA"]["n_faces"] <= 80000
    assert 20000 <= pair["pieceB"]["n_faces"] <= 80000


@pytest.mark.parametrize("name", ["pieceA", "pieceB"])
def test_segmentation_finds_the_fracture_surface(pair, frags, name):
    fr = frags[name]
    g = pair[name]
    assert fr.watertight
    assert fr.thick == pytest.approx(THICK, rel=0.20)
    frac_fraction = fr.fracture_area / fr.area
    assert 0.05 <= frac_fraction <= 0.35
    # ground-truth label per working-mesh face: the label of the nearest generated face
    _, j = cKDTree(g["C"]).query(fr.C, workers=-1)
    true_cut, true_shell = g["cut"][j], g["shell"][j]
    cut_flagged = fr.A[true_cut & fr.frac].sum() / fr.A[true_cut].sum()
    shell_flagged = fr.A[true_shell & fr.frac].sum() / fr.A[true_shell].sum()
    assert cut_flagged >= 0.70, f"{name}: only {cut_flagged:.2f} of the fracture surface flagged"
    assert shell_flagged <= 0.15, f"{name}: {shell_flagged:.2f} of the shell flagged as fracture"


# ---------------------------------------------------------------- B: matching

def test_match_pair_recovers_the_known_transform(pair, frags, thickness, cands):
    assert cands, "no candidate at all for the true pair"
    best = cands[0]
    assert best.a == "pieceA" and best.b == "pieceB"
    assert best.accepted, f"best candidate not accepted: {best.scores}"
    ang, dist = pose_error(best.T, pair["T_rel"], frags["pieceB"].V[::200])
    assert ang <= 2.0, f"rotation off by {ang:.2f} deg"
    assert dist <= 0.1 * thickness, f"points off by {dist:.2f} units (> {0.1 * thickness:.2f})"
    s = best.scores
    assert s["tight"] >= 0.4
    assert s["pen"] <= 0.005
    assert s["seam"] >= 3.0
    assert s["gap"] <= Params().max_gap
    assert s["cont_n"] >= Params().min_cont_n


def test_candidates_are_ranked_by_score(cands):
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)
    assert len(cands) <= 5


# ---------------------------------------------------------------- C: assembly

def test_assemble_places_both_fragments(pair, md, thickness, cands):
    poses, groups, used, rejected = assembly.assemble(md, cands, Params())
    assert len(groups) == 1
    assert set(groups[0]) == {"pieceA", "pieceB"}
    assert len(used) == 1
    assert used[0].accepted
    rel = np.linalg.inv(poses["pieceA"]) @ poses["pieceB"]
    ang, dist = pose_error(rel, pair["T_rel"], md["pieceB"].fr.V[::200])
    assert ang <= 2.0 and dist <= 0.1 * thickness


# ---------------------------------------------------------------- D: pipeline

@pytest.mark.slow
def test_pipeline_end_to_end(pair, thickness):
    out = pair["out"]
    poses, groups, used, cands = pipeline.run(pair["dir"], out, workers=1, preview=True, refine=True)
    for rel in ("transforms.json", "report.md", "report.json", "assembly_0.ply",
                "preview_0.png", "preview_segmentation.png",
                os.path.join("placed", "pieceA.ply"), os.path.join("placed", "pieceB.ply")):
        p = os.path.join(out, rel)
        assert os.path.isfile(p) and os.path.getsize(p) > 0, f"missing output {rel}"

    assert len(groups) == 1 and set(groups[0]) == {"pieceA", "pieceB"}
    assert len(used) == 1

    report = json.load(open(os.path.join(out, "report.json")))
    assert report["groups"] == [list(groups[0])]
    assert len(report["joins_used"]) == 1
    assert {report["joins_used"][0]["a"], report["joins_used"][0]["b"]} == {"pieceA", "pieceB"}
    assert report["thickness"] == pytest.approx(THICK, rel=0.20)
    md_text = open(os.path.join(out, "report.md")).read()
    assert "pieceA" in md_text and "pieceB" in md_text and "not assembled" not in md_text

    tf = json.load(open(os.path.join(out, "transforms.json")))
    assert set(tf["fragments"]) == {"pieceA", "pieceB"}
    assert all(tf["fragments"][n]["placed"] for n in ("pieceA", "pieceB"))
    TA = np.array(tf["fragments"]["pieceA"]["matrix"])
    TB = np.array(tf["fragments"]["pieceB"]["matrix"])
    V, F, _, _ = build_piece(1, seed=2)
    ang, dist = pose_error(np.linalg.inv(TA) @ TB, pair["T_rel"], apply_transform(rigid(2), V[::200]))
    assert ang <= 2.0, f"rotation off by {ang:.2f} deg"
    assert dist <= 0.1 * thickness, f"points off by {dist:.2f} units"
