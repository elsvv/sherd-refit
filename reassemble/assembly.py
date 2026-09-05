"""Global assembly: greedy growth over accepted pairwise joins with penetration and
consistency checks, followed by optional pose-graph loop closure."""
from __future__ import annotations

import logging

import numpy as np

from .fragment import MatchData
from .geometry import apply_transform, rotation_angle_deg
from .matching import Candidate, Params

log = logging.getLogger("reassemble")


def _penetration(A: MatchData, B: MatchData, T_ab: np.ndarray, t: float, p: Params) -> float:
    """Fraction of surface samples of either fragment inside the other, given b->a transform."""
    if not (A.fr.watertight and B.fr.watertight):
        return 0.0
    sdA = A.signed_distance(apply_transform(T_ab, B.S))
    sdB = B.signed_distance(apply_transform(np.linalg.inv(T_ab), A.S))
    return float(max((sdA < -p.pen_delta * t).mean(), (sdB < -p.pen_delta * t).mean()))


def assemble(md: dict[str, MatchData], cands: list[Candidate], t: float, p: Params,
             rot_tol_deg: float = 10.0, trans_tol: float = 0.5):
    """Greedy assembly.

    Returns (poses, groups, used, rejected) where poses maps name -> 4x4 world transform,
    groups is a list of lists of names (size >= 2 first, then singletons), used is the list of
    joins that built the assembly and rejected the accepted joins that were skipped and why.
    """
    best_per_pair: dict[tuple, Candidate] = {}
    for c in cands:
        if c.accepted and ((c.a, c.b) not in best_per_pair or c.score > best_per_pair[(c.a, c.b)].score):
            best_per_pair[(c.a, c.b)] = c
    accepted = sorted(best_per_pair.values(), key=lambda c: -c.score)
    names = list(md.keys())
    poses: dict[str, np.ndarray] = {}
    group_of: dict[str, int] = {}
    groups: list[list[str]] = []
    used, rejected = [], []

    def rel(c: Candidate, x: str):
        """Transform of fragment x's partner into x's frame, from candidate c (T maps b -> a)."""
        return c.T if x == c.a else np.linalg.inv(c.T)

    def try_place(c: Candidate, placed: str, new: str):
        # world pose of `new` implied by c
        T_new = poses[placed] @ rel(c, placed)
        g = group_of[placed]
        # penetration against every fragment already in the group
        for other in groups[g]:
            if other == placed:
                continue
            T_rel = np.linalg.inv(poses[other]) @ T_new       # new -> other
            pen = _penetration(md[other], md[new], T_rel, t, p)
            if pen > p.max_pen:
                return None, f"penetrates {other} ({pen:.3f})"
        # consistency with other accepted joins between `new` and the group
        for c2 in accepted:
            if c2 is c or new not in (c2.a, c2.b):
                continue
            other = c2.b if c2.a == new else c2.a
            if other not in poses or group_of[other] != g:
                continue
            T_alt = poses[other] @ rel(c2, other)
            D = np.linalg.inv(T_alt) @ T_new
            ang, dist = rotation_angle_deg(D[:3, :3]), np.linalg.norm(D[:3, 3]) / t
            if ang > rot_tol_deg or dist > trans_tol:
                # the two joins disagree; keep the stronger one, reject this placement if c is weaker
                if c2.score > c.score:
                    return None, f"inconsistent with stronger join {c2.a}-{c2.b} ({ang:.1f} deg, {dist:.2f} t)"
        return T_new, None

    remaining = list(accepted)
    while remaining:
        progressed = False
        for c in list(remaining):
            a_in, b_in = c.a in poses, c.b in poses
            if a_in and b_in:
                remaining.remove(c)
                if group_of[c.a] == group_of[c.b]:
                    used.append(c)      # loop-closing edge inside a group; consistency was checked at placement
                else:
                    rejected.append((c, "would merge two groups (not supported)"))
                continue
            if not a_in and not b_in:
                continue
            placed, new = (c.a, c.b) if a_in else (c.b, c.a)
            T_new, why = try_place(c, placed, new)
            remaining.remove(c)
            if T_new is None:
                rejected.append((c, why))
                continue
            poses[new] = T_new
            group_of[new] = group_of[placed]
            groups[group_of[placed]].append(new)
            used.append(c)
            progressed = True
            break
        if not progressed and remaining:
            # seed a new group with the best remaining join whose fragments are both unplaced
            seed = next((c for c in remaining if c.a not in poses and c.b not in poses), None)
            if seed is None:
                break
            remaining.remove(seed)
            groups.append([seed.a, seed.b])
            g = len(groups) - 1
            poses[seed.a] = np.eye(4); poses[seed.b] = seed.T
            group_of[seed.a] = group_of[seed.b] = g
            used.append(seed)
    for n in names:
        if n not in poses:
            poses[n] = np.eye(4)
            groups.append([n])
            group_of[n] = len(groups) - 1
    groups.sort(key=lambda g: -len(g))
    log.info("assembly: %d groups (%s), %d joins used, %d accepted joins rejected",
             len(groups), ", ".join(str(len(g)) for g in groups), len(used), len(rejected))
    return poses, groups, used, rejected


def recenter(poses: dict[str, np.ndarray], md: dict[str, MatchData], groups: list[list[str]]):
    """Translate each group so that its centroid sits at the origin (cosmetic, keeps numbers small)."""
    out = dict(poses)
    for g in groups:
        pts = np.concatenate([apply_transform(poses[n], md[n].S[::10]) for n in g])
        c = pts.mean(0)
        for n in g:
            T = out[n].copy(); T[:3, 3] -= c; out[n] = T
    return out
