"""Pairwise fragment matching: breakline-frame hypotheses, staged ICP refinement, verification."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .fragment import MatchData
from .geometry import threads, apply_transform, make_frames, single_threaded, worker_threads

log = logging.getLogger("sherd_refit")


@dataclass
class Params:
    """Thresholds of the matcher.

    Every distance threshold comes as a pair `(k, m)`: the distance used is `max(k * t, m * res)`,
    with `t` the pair's wall thickness and `res` the pair's working-mesh resolution (see `Scales`).
    `k * t` is the scale-free part -- a wall thickness means the same thing on a 39-unit terracotta
    relief and on a 3.5 mm pot wall.  `m * res` is the resolution floor: it counts triangle edges,
    and it stops the pipeline from demanding a precision the mesh cannot carry.  On a mesh with
    more than `k / m` edges across the wall the first term wins and nothing changes.
    """
    dihedral_tol: float = 25.0      # degrees, |dih_A + dih_B - 180| tolerance for a hypothesis
    coarse_delta: float = 0.15      # t, breakline proximity for the coarse score
    coarse_points: int = 60         # B breakline points used by the coarse score
    stage1: int = 250               # hypotheses refined with breakline ICP
    stage2: int = 10                # candidates refined with full ICP
    stage1_delta: float = 0.06      # t, breakline proximity when stage-1 poses are re-scored
    tight_delta: float = 0.01       # t, distance for the tight-contact fraction
    facing_delta: float = 0.3       # t, fracture points considered "facing" the other fragment
    seam_delta: float = 0.12        # t, breakline proximity counted as a shared seam
    near_delta: float = 0.5         # t, shell-margin radius for the continuity test
    pen_delta: float = 0.06         # t, penetration depth counted
    nms_delta: float = 0.5          # t, translation radius of the non-maximum suppression
    icp_delta: float = 0.04         # t, finest rung of the ICP ladder (the whole ladder scales with it)
    # `tight_delta` and `max_gap` are an order of magnitude below the rest because they are the
    # only two distances measured point-to-*surface*; the others are point-to-point between
    # samples, or breakline to breakline, and still carry a sample spacing inside them.
    # Resolution floors, counted in working-mesh edges; `Scales` below turns them into distances.
    # Each one is set just below `k / 0.058`, because 0.058 t is the coarsest working mesh the
    # thick terracotta reference set produces: on that set every floor stays inactive and the
    # pipeline behaves exactly as it did before they existed.  Raising them further does change
    # it, and for the worse -- at `tight_res = 1.5` the tight-contact distance on the terracotta
    # grows from 0.040 t to 0.087 t, the false pair 007-094 scores tight 0.56 instead of 0.14 and
    # is accepted.  `facing_res` is the exception: it selects *which* points are compared rather
    # than how precisely, and widening it drags far-away points into the median, so it is left at
    # a value that never binds (on pot G a floor of 1.5 edges lifts the median gap of the true
    # joins from 0.173 t to 0.220 t).  See docs/superpowers/notes/2026-09-05-thin-walls.md.
    coarse_res: float = 2.3          # every floor below crosses over at ~15 edges per t
    stage1_res: float = 0.9
    tight_res: float = 0.15
    facing_res: float = 1.0
    gap_res: float = 0.45
    seam_res: float = 1.8
    near_res: float = 4.0
    pen_res: float = 0.9
    icp_res: float = 0.6
    min_tight: float = 0.25
    max_gap: float = 0.03
    max_pen: float = 0.005
    min_seam: float = 3.0
    min_cont_n: float = 0.8
    early_reject_tight: float = 0.0     # >0: skip the fracture-only ICPs and the costly verification below this
    stage1_floor: float = 0.0           # >0: skip stage 2 when the pair's best stage-1 breakline score is below this
    thick_ratio: float = 2.5            # a pair whose walls differ by more than this is not matched at all
    screen_top_k: int = 0               # partners kept per fragment by the screening pass (0: no screening)
    screen_points: int = 150            # breakline points per fragment the screening pass uses
    screen_min_pairs: int = 200         # screening is skipped below this many pairs
    margin_points: int = 6000           # shell-margin points kept for the pc_reg ICP and the continuity test
    surface_points: int = 20000         # whole-surface samples per fragment (penetration test and shell margin)
    macro_inner: float = 0.15           # t, faces this close to the breakline are left out of the macro normals
    macro_outer: float = 0.60           # t, outer radius of the macro-normal neighbourhood
    brk_voxel: float = 0.5              # t, voxel the breakline is thinned to before frames are paired
    frac_per_t2: float = 150.0          # fracture samples per t^2 of fracture area
    min_frac_points: int = 5000
    max_frac_points: int = 12000
    seed: int = 0


@dataclass(frozen=True)
class Scales:
    """Every distance the matcher uses, resolved once for one pair of fragments.

    Built from the pair's wall thickness and mesh resolution, and passed down instead of a bare
    `t`, so that the two-term rule of `Params` lives in exactly one place.

    `t` is `min(t_A, t_B)`: a fragment carrying the pot's rim measures a thicker wall than the
    body it broke off, and the wall is the thinner of the two.  `res` is `max(res_A, res_B)`: the
    coarser of the two meshes is what limits how precisely the pair can be told to fit.
    """
    t: float
    res: float
    coarse: float
    stage1: float
    tight: float
    facing: float
    gap: float
    seam: float
    near: float
    pen: float
    nms: float
    icp: float          # factor by which the whole ICP ladder is stretched (>= 1)

    @classmethod
    def for_pair(cls, p: Params, t: float, res: float) -> "Scales":
        def f(k, m):
            return max(k * t, m * res)
        return cls(t=float(t), res=float(res),
                   coarse=f(p.coarse_delta, p.coarse_res), stage1=f(p.stage1_delta, p.stage1_res),
                   tight=f(p.tight_delta, p.tight_res), facing=f(p.facing_delta, p.facing_res),
                   gap=f(p.max_gap, p.gap_res), seam=f(p.seam_delta, p.seam_res),
                   near=f(p.near_delta, p.near_res), pen=f(p.pen_delta, p.pen_res),
                   nms=p.nms_delta * t,
                   icp=f(p.icp_delta, p.icp_res) / (p.icp_delta * t))

    @classmethod
    def for_fragments(cls, p: Params, a, b) -> "Scales":
        """Scales for the pair (a, b), given as `MatchData` or `Fragment`.

        The pair supplies both numbers itself; no collection-wide thickness is involved, because a
        collection can hold walls of 2.4 mm and 13.5 mm at once and its median describes neither.
        """
        fa, fb = getattr(a, "fr", a), getattr(b, "fr", b)
        return cls.for_pair(p, min(fa.thick, fb.thick), max(fa.res, fb.res))

    def icp_dist(self, k: float) -> float:
        """One rung of the ICP ladder, `k * t`, stretched by the resolution floor.

        The ladder is stretched as a whole rather than floored rung by rung, so that its steps
        keep their ratios and a coarse mesh cannot make the fine rung overtake the coarse one.
        """
        return k * self.t * self.icp

    def limits(self) -> dict:
        """The two acceptance limits that depend on the pair, in units of `t`, for the report."""
        return dict(gap_limit=self.gap / self.t, tight_delta=self.tight / self.t)


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

def hypotheses(A: MatchData, B: MatchData, p: Params, ia=None, ib=None):
    """Frame pairs whose dihedrals are complementary, as (rotations, translations) mapping b to a.

    `ia` / `ib` override the breakline subsets used, which is how the screening pass
    (`screen_pair`) buys a cheaper version of the same test.
    """
    ia = A.brk_sub if ia is None else ia
    ib = B.brk_sub if ib is None else ib
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


def coarse_score(A: MatchData, B: MatchData, R, tr, sc: Scales, p: Params, rng, pool=None):
    pool = B.brk_sub if pool is None else pool
    idx = rng.choice(pool, min(p.coarse_points, len(pool)), replace=False)
    Q, QN = B.brk_P[idx], B.brk_ns[idx]
    delta = sc.coarse
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


def _cap(idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """A deterministic subset of at most `n` breakline indices, order preserved."""
    if n <= 0 or len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def screen_pair(A: MatchData, B: MatchData, p: Params, points: int = 150) -> float:
    """How well the two breaklines can be laid on top of each other, cheaply.

    The matcher's own coarse stage, run on a capped subsample of both breaklines: every
    dihedral-compatible pair of frames is turned into a pose and scored by how much of B's
    breakline lands on A's.  Capping the two subsets at `points` makes the hypothesis count
    `0.3 * points^2` instead of tens of thousands, which turns a one-second stage into 0.17
    CPU-seconds -- cheap enough to run on all 12 589 pairs of a 164-fragment collection in five
    minutes.

    **It ranks a fragment's partners; it does not decide whether they touched.**  A true adjacent
    pair overlaps on 5-22 % of its breakline even at the exact ground-truth pose, because one seam
    is a fraction of a fragment's whole crack line, and the best of several thousand poses reaches
    the same 10-28 % on pairs that never touched.  Keeping the best 15 partners per fragment keeps
    47 % of the ground-truth adjacent pairs of `mixed_all`; the whole measurement, and three other
    partner searches that do no better, are in
    docs/superpowers/notes/2026-09-06-scale-pairs.md.

    Returns the best coarse score; 0 when the pair has no usable breakline.
    """
    if A.brk_tree is None or B.brk_tree is None or len(A.brk_sub) == 0 or len(B.brk_sub) == 0:
        return 0.0
    sc = Scales.for_fragments(p, A, B)
    ia, ib = _cap(A.brk_sub, points, p.seed), _cap(B.brk_sub, points, p.seed)
    R, tr = hypotheses(A, B, p, ia=ia, ib=ib)
    if len(R) == 0:
        return 0.0
    cs = coarse_score(A, B, R, tr, sc, p, np.random.default_rng(p.seed), pool=ib)
    return float(cs.max())


def top_partners(score: dict[tuple, float], names: list[str], k: int) -> set[tuple]:
    """The union over fragments of each fragment's `k` best-scoring partners.

    Per fragment rather than by one global threshold, because the coarse score is not comparable
    across fragments: a large sherd with a long breakline scores lower everywhere than a small one
    whose whole crack line can lie on its partner.  What matters is the ranking within a
    fragment's own row, and a fragment with few neighbours costs only the `k` pairs it keeps.
    """
    order = {n: i for i, n in enumerate(names)}
    per: dict[str, list] = {n: [] for n in names}
    for (a, b), v in score.items():
        per.setdefault(a, []).append((v, b))
        per.setdefault(b, []).append((v, a))
    keep = set()
    for n, lst in per.items():
        for _, m in sorted(lst, key=lambda x: (-x[0], x[1]))[:k]:
            # the pair is written the way `itertools.combinations(names, 2)` writes it, so the
            # caller can test membership against its own pair list
            keep.add((n, m) if order[n] < order[m] else (m, n))
    return keep


def nms(order, R, tr, sc, trans_tol, topk, floor, rot_tol=2.9):
    """Greedy non-maximum suppression over poses: walk `order`, keeping a pose unless an already
    kept one is within `trans_tol` of it and points nearly the same way.

    The translation test is evaluated against all kept poses at once, which is what makes this
    loop cheap; the rotation test then runs only for the few poses that are close in translation,
    with the same arithmetic as before.
    """
    kept: list[int] = []
    kept_tr = np.empty((max(topk, 1), 3))
    for k in order:
        if sc[k] < floor:
            break
        n = len(kept)
        dup = n > 0 and any(np.trace(R[k].T @ R[kept[i]]) > rot_tol
                            for i in np.flatnonzero(np.linalg.norm(tr[k] - kept_tr[:n], axis=1) < trans_tol))
        if not dup:
            kept_tr[n] = tr[k]
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

def fracture_scores(A: MatchData, B: MatchData, T: np.ndarray, sc: Scales, p: Params) -> dict:
    """Tight-contact fraction, median gap and contact area from the facing fracture points.

    The distances are point-to-*surface*: each fragment's fracture samples are measured against the
    other's fracture triangles, not against the other's sample cloud.  Two independent samples of
    the same surface never land on each other, so the point-to-point form had a floor equal to the
    sample spacing -- about `0.5 / sqrt(density)`, which grew with the fragment's area-to-wall
    ratio and reached 0.075 t on a large pot A sherd against 0.043 t on the terracotta.  That floor
    alone was enough to keep every true join of the thin pots away from the gap threshold, and the
    synthetic benchmark shows the same thing on fragments that fit to 0.001 mm.  Against the
    triangles the only floor left is the mesh resolution, which `Scales` already carries.
    """
    s = {}
    t = sc.t
    d1 = A.fracture_distance(apply_transform(T, B.Pf))
    d2 = B.fracture_distance(apply_transform(np.linalg.inv(T), A.Pf))
    for tag, d, area in (("A", d2, A.frac_area), ("B", d1, B.frac_area)):
        face = d < sc.facing
        if face.sum() < 20:
            s["tight" + tag], s["gap" + tag], s["contact" + tag] = 0.0, 1.0, 0.0
        else:
            s["tight" + tag] = float((d[face] < sc.tight).mean())
            s["gap" + tag] = float(np.median(d[face]) / t)
            s["contact" + tag] = float((d < 2 * sc.tight).mean() * area / t ** 2)
    s["tight"] = min(s["tightA"], s["tightB"]); s["gap"] = max(s["gapA"], s["gapB"])
    s["contact"] = min(s["contactA"], s["contactB"])
    return s


def _seam_score(A: MatchData, B: MatchData, T: np.ndarray, sc: Scales) -> dict:
    """Length of A's breakline (in t) covered by B's breakline with agreeing shell normals."""
    t = sc.t
    PBb = apply_transform(T, B.brk_P); NBb = B.brk_ns @ T[:3, :3].T
    dA, jA = cKDTree(PBb).query(A.brk_P, workers=threads())
    seamA = (dA < sc.seam) & (np.einsum("ij,ij->i", A.brk_ns, NBb[jA]) > 0.7)
    if not seamA.any():
        return dict(seam=0.0)
    vox = np.unique(np.floor(A.brk_P[seamA] / (t / 3.0)).astype(int), axis=0)
    return dict(seam=float(len(vox) / 3.0))


def _continuity_scores(A: MatchData, B: MatchData, T: np.ndarray, sc: Scales) -> dict:
    """Step height and normal agreement of the outer shell across the seam."""
    t = sc.t
    if A.tree_margin is not None and len(B.Pm):
        PBm = apply_transform(T, B.Pm); NBm = B.Nm @ T[:3, :3].T
        dm, jm = A.tree_margin.query(PBm, workers=threads())
        near = dm < sc.near
        if near.sum() > 20:
            Am = A.Pm[jm[near]]; An = A.Nm[jm[near]]
            return dict(cont=float(np.median(np.abs(np.einsum("ij,ij->i", PBm[near] - Am, An))) / t),
                        cont_n=float(np.median(np.einsum("ij,ij->i", NBm[near], An))))
    return dict(cont=1.0, cont_n=-1.0)


def _penetration_scores(A: MatchData, B: MatchData, T: np.ndarray, sc: Scales, p: Params) -> dict:
    """Fraction of surface samples of either fragment inside the other, and the deepest excursion."""
    if not (A.fr.watertight and B.fr.watertight):
        return dict(pen=0.0, pen_depth=0.0, pen_unavailable=1.0)
    sdA = A.signed_distance(apply_transform(T, B.S_pen))
    sdB = B.signed_distance(apply_transform(np.linalg.inv(T), A.S_pen))
    return dict(pen=float(max((sdA < -sc.pen).mean(), (sdB < -sc.pen).mean())),
                pen_depth=float(max(-sdA.min(), -sdB.min()) / sc.t))


def verify(A: MatchData, B: MatchData, T: np.ndarray, sc: Scales, p: Params, full: bool = True,
           frac: dict | None = None) -> dict:
    """All verification scores for the transform T (b -> a).

    With `full=False` only the cheap half is computed (fracture contact and seam length); shell
    continuity and penetration are marked as not computed (`cont_n = -1`, `pen = 0`, `partial = 1`),
    which also makes the candidate fail `accept`.  `frac` passes in fracture scores already
    computed for this very transform.
    """
    s = dict(frac) if frac is not None else fracture_scores(A, B, T, sc, p)
    s.update(_seam_score(A, B, T, sc))
    s.update(sc.limits())
    if not full:
        return dict(s, cont=1.0, cont_n=-1.0, pen=0.0, pen_depth=0.0, partial=1.0)
    s.update(_continuity_scores(A, B, T, sc))
    s.update(_penetration_scores(A, B, T, sc, p))
    return s


def _partial_scores(sc: Scales, brk: float) -> dict:
    """Scores for a candidate cut before stage 2: every field the report prints, with the ones
    that were never computed set so that `accept` refuses them."""
    s = dict(tightA=0.0, tightB=0.0, tight=0.0, gapA=1.0, gapB=1.0, gap=1.0,
             contactA=0.0, contactB=0.0, contact=0.0, seam=0.0, cont=1.0, cont_n=-1.0,
             pen=0.0, pen_depth=0.0, partial=1.0, brk=brk, brk_best=brk)
    s.update(sc.limits())
    return s


def accept(s: dict, p: Params, sc: Scales) -> bool:
    """`gap` is reported in units of t, so it is compared against the pair's own gap limit."""
    return (s["tight"] >= p.min_tight and s["gap"] * sc.t <= sc.gap and s["pen"] <= p.max_pen
            and s["seam"] >= p.min_seam and s["cont_n"] >= p.min_cont_n)


# ---------------------------------------------------------------- driver

def _map(fn, jobs: list, n_threads: int) -> list:
    """`[fn(j) for j in jobs]`, spread over `n_threads` threads of this process.

    Both stages of the refinement are independent per hypothesis and spend nearly all their time
    in Open3D's ICP and in scipy KD-tree queries, which release the GIL, so threads scale well
    (measured 3.1x on 4 threads, 5.9x on 10 for stage 2 of one real pair).  Results keep the job
    order, so the ranking does not depend on the thread count.  KD-tree queries inside the pool
    are pinned to one worker each so that scipy's own threads do not multiply with the pool.
    """
    if n_threads <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]

    def wrapped(j):
        with single_threaded():
            return fn(j)

    with ThreadPoolExecutor(max_workers=min(n_threads, len(jobs))) as ex:
        return list(ex.map(wrapped, jobs))


def _stage2(A: MatchData, B: MatchData, T0: np.ndarray, sc: Scales, p: Params, brk: float) -> Candidate:
    """Refine one stage-1 pose with the full ICP chain and verify it.

    With `Params.early_reject_tight > 0` a cheap tight-contact estimate is taken after the two
    ICPs on fracture + shell margin, and a candidate below the threshold skips the two
    fracture-only ICPs and the expensive half of the verification; it keeps its cheap scores (so
    it still shows up in the report), is marked `partial`, and can never be accepted.  This is
    off by default: measured on the test set, the estimate can still rise by 0.09 during the two
    remaining ICPs, so a threshold safe against the 0.25 acceptance limit saves almost nothing.
    See docs/superpowers/notes/2026-09-05-performance.md.
    """
    T = _icp(B.pc_reg, A.pc_reg, T0, sc.icp_dist(0.2), 30)
    T = _icp(B.pc_reg, A.pc_reg, T, sc.icp_dist(0.08), 30)
    if p.early_reject_tight > 0.0:
        frac = fracture_scores(A, B, T, sc, p)
        if frac["tight"] < p.early_reject_tight:
            s = verify(A, B, T, sc, p, full=False, frac=frac)
            s["brk"] = brk
            return Candidate(A.name, B.name, T, s)          # accepted stays False
    T = _icp(B.pc_frac, A.pc_frac, T, sc.icp_dist(0.08), 30)
    T = _icp(B.pc_frac, A.pc_frac, T, sc.icp_dist(0.04), 30)
    s = verify(A, B, T, sc, p)
    s["brk"] = brk
    c = Candidate(A.name, B.name, T, s)
    c.accepted = accept(s, p, sc)
    return c


def match_pair(A: MatchData, B: MatchData, p: Params, keep: int = 5, n_threads: int | None = None) -> list[Candidate]:
    """Return the best `keep` candidates (b -> a) for the pair, best first.

    `n_threads` threads are used inside the pair (default: this process's SHERD_REFIT_THREADS
    budget, 1 outside the pipeline).  The result does not depend on it.
    """
    t0 = time.time()
    n_threads = worker_threads() if n_threads is None else max(1, n_threads)
    rng = np.random.default_rng(p.seed)
    if not (A.has_frac and B.has_frac) or A.brk_tree is None or B.brk_tree is None:
        return []
    sc = Scales.for_fragments(p, A, B)
    R, tr = hypotheses(A, B, p)
    if len(R) == 0:
        log.info("%s-%s: no hypotheses", A.name, B.name)
        return []
    cs = coarse_score(A, B, R, tr, sc, p, rng)
    kept = nms(np.argsort(cs)[::-1][:5000], R, tr, cs, sc.nms, p.stage1, 0.1)

    def stage1(k):
        T = np.eye(4); T[:3, :3] = R[k]; T[:3, 3] = tr[k]
        T = _icp(B.pc_brk, A.pc_brk_full, T, sc.icp_dist(0.2), 20, plane=False)
        T = _icp(B.pc_brk, A.pc_brk_full, T, sc.icp_dist(0.08), 20, plane=False)
        return T, brk_score(A, B, T, sc.stage1)

    out = _map(stage1, kept, n_threads)
    Ts = [o[0] for o in out]; s1 = [o[1] for o in out]
    if not Ts:
        log.info("%s-%s: nothing passed the coarse stage", A.name, B.name)
        return []
    s1 = np.array(s1)
    best1 = float(s1.max())
    if p.stage1_floor > 0.0 and best1 < p.stage1_floor:
        # Nothing the breakline stage found comes near a seam, and stage 2 is where all the time
        # goes.  The pair keeps its best stage-1 pose and is marked `partial`, so it still appears
        # in the report and can never be accepted.
        k = int(np.argmax(s1))
        log.info("%s-%s: best stage-1 breakline score %.3f below the floor %.3f, stage 2 skipped (%.1fs)",
                 A.name, B.name, best1, p.stage1_floor, time.time() - t0)
        return [Candidate(A.name, B.name, Ts[k], _partial_scores(sc, best1))]
    Rs = np.array([T[:3, :3] for T in Ts]); trs = np.array([T[:3, 3] for T in Ts])
    kept2 = nms(np.argsort(s1)[::-1], Rs, trs, s1, sc.nms, p.stage2, 0.05)
    cands = _map(lambda k: _stage2(A, B, Ts[k], sc, p, float(s1[k])), kept2, n_threads)
    cands.sort(key=lambda c: -c.score)
    for c in cands:
        c.scores["brk_best"] = best1
    best = cands[0].scores if cands else {}
    log.info("%s-%s: t %.2f res %.2f (%.1f edges per t); %d hyp, %d/%d refined (%d threads), "
             "best seam %.1f tight %.2f (<%.3f t) gap %.3f (<%.3f t) pen %.3f accepted=%s (%.1fs)",
             A.name, B.name, sc.t, sc.res, sc.t / max(sc.res, 1e-9), len(R), len(kept), len(kept2), n_threads,
             best.get("seam", 0), best.get("tight", 0), sc.tight / sc.t, best.get("gap", 0), sc.gap / sc.t,
             best.get("pen", 0), cands[0].accepted if cands else False, time.time() - t0)
    return cands[:keep]
