"""Pairwise fragment matching: breakline-frame hypotheses, staged ICP refinement, verification."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .fragment import MatchData
from .geometry import threads, apply_transform, make_frames

log = logging.getLogger("reassemble")


@dataclass
class Params:
    dihedral_tol: float = 40.0      # degrees, |dih_A + dih_B - 180| tolerance for a hypothesis
    coarse_delta: float = 0.15      # t, breakline proximity for the coarse score
    coarse_points: int = 60         # B breakline points used by the coarse score
    stage1: int = 400               # hypotheses refined with breakline ICP
    stage2: int = 40                # candidates refined with full ICP
    tight_delta: float = 0.04       # t, distance for the tight-contact fraction
    facing_delta: float = 0.3       # t, fracture points considered "facing" the other fragment
    pen_delta: float = 0.06         # t, penetration depth counted
    min_tight: float = 0.25
    max_gap: float = 0.065
    max_pen: float = 0.005
    min_seam: float = 3.0
    min_cont_n: float = 0.8
    seed: int = 0


@dataclass
class Candidate:
    a: str
    b: str
    T: np.ndarray               # maps b into a's frame
    scores: dict = field(default_factory=dict)
    accepted: bool = False

    @property
    def score(self) -> float:
        return float(self.scores.get("seam", 0.0) * self.scores.get("tight", 0.0))

    def to_json(self) -> dict:
        return dict(a=self.a, b=self.b, T=self.T.tolist(), accepted=self.accepted, score=self.score,
                    **{k: float(v) for k, v in self.scores.items()})

    @classmethod
    def from_json(cls, d: dict) -> "Candidate":
        skip = {"a", "b", "T", "accepted", "score"}
        return cls(a=d["a"], b=d["b"], T=np.array(d["T"]), accepted=bool(d["accepted"]), scores={k: v for k, v in d.items() if k not in skip})


# ---------------------------------------------------------------- hypotheses

def hypotheses(A: MatchData, B: MatchData, p: Params):
    ia, ib = A.brk_sub, B.brk_sub
    if len(ia) == 0 or len(ib) == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    da, db = A.brk_dih[ia], B.brk_dih[ib]
    ok = np.abs(da[:, None] + db[None, :] - 180.0) < p.dihedral_tol
    pa, pb = np.where(ok)
    RA = make_frames(A.brk_t[ia][pa], A.brk_ns[ia][pa], A.brk_f[ia][pa])
    RB = make_frames(-B.brk_t[ib][pb], B.brk_ns[ib][pb], -B.brk_f[ib][pb])
    R = np.einsum("nij,nkj->nik", RA, RB)
    tr = A.brk_P[ia][pa] - np.einsum("nij,nj->ni", R, B.brk_P[ib][pb])
    return R, tr


def coarse_score(A: MatchData, B: MatchData, R, tr, t, p: Params, rng):
    idx = rng.choice(B.brk_sub, min(p.coarse_points, len(B.brk_sub)), replace=False)
    Q, QN = B.brk_P[idx], B.brk_ns[idx]
    delta = p.coarse_delta * t
    scores = np.zeros(len(R))
    chunk = 4000
    for s in range(0, len(R), chunk):
        Rc, tc = R[s:s + chunk], tr[s:s + chunk]
        Pq = (np.einsum("nij,mj->nmi", Rc, Q) + tc[:, None, :]).reshape(-1, 3)
        Nq = np.einsum("nij,mj->nmi", Rc, QN).reshape(-1, 3)
        d, j = A.brk_tree.query(Pq, distance_upper_bound=delta, workers=threads())
        hit = np.isfinite(d)
        nn = np.zeros((len(d), 3)); nn[hit] = A.brk_ns[j[hit]]
        agree = hit & (np.einsum("ij,ij->i", nn, Nq) > 0.7)
        scores[s:s + chunk] = agree.reshape(len(Rc), -1).mean(1)
    return scores


def nms(order, R, tr, sc, t, topk, floor, rot_tol=2.9):
    kept = []
    for k in order:
        if sc[k] < floor:
            break
        dup = False
        for kk in kept:
            if np.linalg.norm(tr[k] - tr[kk]) < 0.5 * t and np.trace(R[k].T @ R[kk]) > rot_tol:
                dup = True
                break
        if not dup:
            kept.append(k)
        if len(kept) >= topk:
            break
    return kept


# ---------------------------------------------------------------- ICP wrappers

def _icp(src, tgt, T0, dist, iters, plane=True):
    est = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if plane
           else o3d.pipelines.registration.TransformationEstimationPointToPoint())
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
    return o3d.pipelines.registration.registration_icp(src, tgt, dist, T0, est, crit).transformation


def brk_score(A: MatchData, B: MatchData, T, delta):
    P = apply_transform(T, B.brk_P[B.brk_sub]); N = B.brk_ns[B.brk_sub] @ T[:3, :3].T
    d, j = A.brk_tree.query(P, distance_upper_bound=delta, workers=threads())
    hit = np.isfinite(d)
    nn = np.zeros((len(d), 3)); nn[hit] = A.brk_ns[j[hit]]
    return float((hit & (np.einsum("ij,ij->i", nn, N) > 0.7)).mean())


# ---------------------------------------------------------------- verification

def verify(A: MatchData, B: MatchData, T: np.ndarray, t: float, p: Params) -> dict:
    """All verification scores for the transform T (b -> a)."""
    s = {}
    Tinv = np.linalg.inv(T)
    # fracture contact, tight fraction and gap
    PBf = apply_transform(T, B.S[B.S_frac]); PAf = A.S[A.S_frac]
    d1, _ = A.tree_frac.query(PBf, workers=threads())
    d2, _ = cKDTree(PBf).query(PAf, workers=threads())
    for tag, d, area in (("A", d2, A.frac_area), ("B", d1, B.frac_area)):
        face = d < p.facing_delta * t
        if face.sum() < 20:
            s["tight" + tag], s["gap" + tag], s["contact" + tag] = 0.0, 1.0, 0.0
        else:
            s["tight" + tag] = float((d[face] < p.tight_delta * t).mean())
            s["gap" + tag] = float(np.median(d[face]) / t)
            s["contact" + tag] = float((d < 2 * p.tight_delta * t).mean() * area / t ** 2)
    s["tight"] = min(s["tightA"], s["tightB"]); s["gap"] = max(s["gapA"], s["gapB"])
    s["contact"] = min(s["contactA"], s["contactB"])
    # seam length along A's breakline
    PBb = apply_transform(T, B.brk_P); NBb = B.brk_ns @ T[:3, :3].T
    dA, jA = cKDTree(PBb).query(A.brk_P, workers=threads())
    seamA = (dA < 0.12 * t) & (np.einsum("ij,ij->i", A.brk_ns, NBb[jA]) > 0.7)
    if seamA.any():
        vox = np.unique(np.floor(A.brk_P[seamA] / (t / 3.0)).astype(int), axis=0)
        s["seam"] = float(len(vox) / 3.0)
    else:
        s["seam"] = 0.0
    # shell continuity across the seam
    if A.tree_margin is not None and B.margin.any():
        PBm = apply_transform(T, B.S[B.margin]); NBm = B.SN[B.margin] @ T[:3, :3].T
        dm, jm = A.tree_margin.query(PBm, workers=threads())
        near = dm < 0.5 * t
        if near.sum() > 20:
            Am = A.S[A.margin][jm[near]]; An = A.SN[A.margin][jm[near]]
            s["cont"] = float(np.median(np.abs(np.einsum("ij,ij->i", PBm[near] - Am, An))) / t)
            s["cont_n"] = float(np.median(np.einsum("ij,ij->i", NBm[near], An)))
        else:
            s["cont"], s["cont_n"] = 1.0, -1.0
    else:
        s["cont"], s["cont_n"] = 1.0, -1.0
    # penetration (both directions)
    if A.fr.watertight and B.fr.watertight:
        sdA = A.signed_distance(apply_transform(T, B.S)); sdB = B.signed_distance(apply_transform(Tinv, A.S))
        s["pen"] = float(max((sdA < -p.pen_delta * t).mean(), (sdB < -p.pen_delta * t).mean()))
        s["pen_depth"] = float(max(-sdA.min(), -sdB.min()) / t)
    else:
        s["pen"], s["pen_depth"] = 0.0, 0.0
        s["pen_unavailable"] = 1.0
    return s


def accept(s: dict, p: Params) -> bool:
    return (s["tight"] >= p.min_tight and s["gap"] <= p.max_gap and s["pen"] <= p.max_pen
            and s["seam"] >= p.min_seam and s["cont_n"] >= p.min_cont_n)


# ---------------------------------------------------------------- driver

def match_pair(A: MatchData, B: MatchData, t: float, p: Params, keep: int = 5) -> list[Candidate]:
    """Return the best `keep` candidates (b -> a) for the pair, best first."""
    t0 = time.time()
    rng = np.random.default_rng(p.seed)
    if A.tree_frac is None or B.tree_frac is None or A.brk_tree is None or B.brk_tree is None:
        return []
    R, tr = hypotheses(A, B, p)
    if len(R) == 0:
        log.info("%s-%s: no hypotheses", A.name, B.name)
        return []
    sc = coarse_score(A, B, R, tr, t, p, rng)
    kept = nms(np.argsort(sc)[::-1][:5000], R, tr, sc, t, p.stage1, 0.1)
    Ts, s1 = [], []
    for k in kept:
        T = np.eye(4); T[:3, :3] = R[k]; T[:3, 3] = tr[k]
        T = _icp(B.pc_brk, A.pc_brk_full, T, 0.2 * t, 20, plane=False)
        T = _icp(B.pc_brk, A.pc_brk_full, T, 0.08 * t, 20, plane=False)
        Ts.append(T); s1.append(brk_score(A, B, T, 0.06 * t))
    if not Ts:
        log.info("%s-%s: nothing passed the coarse stage", A.name, B.name)
        return []
    s1 = np.array(s1)
    Rs = np.array([T[:3, :3] for T in Ts]); trs = np.array([T[:3, 3] for T in Ts])
    kept2 = nms(np.argsort(s1)[::-1], Rs, trs, s1, t, p.stage2, 0.05)
    cands = []
    for k in kept2:
        T = _icp(B.pc_reg, A.pc_reg, Ts[k], 0.2 * t, 30)
        T = _icp(B.pc_reg, A.pc_reg, T, 0.08 * t, 30)
        T = _icp(B.pc_frac, A.pc_frac, T, 0.08 * t, 30)
        T = _icp(B.pc_frac, A.pc_frac, T, 0.04 * t, 30)
        s = verify(A, B, T, t, p); s["brk"] = float(s1[k])
        c = Candidate(A.name, B.name, T, s); c.accepted = accept(s, p)
        cands.append(c)
    cands.sort(key=lambda c: -c.score)
    best = cands[0].scores if cands else {}
    log.info("%s-%s: %d hyp, %d/%d refined, best seam %.1f tight %.2f gap %.3f pen %.3f accepted=%s (%.1fs)",
             A.name, B.name, len(R), len(kept), len(kept2), best.get("seam", 0), best.get("tight", 0), best.get("gap", 0),
             best.get("pen", 0), cands[0].accepted if cands else False, time.time() - t0)
    return cands[:keep]
