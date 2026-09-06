#!/usr/bin/env python3
"""Breakline signatures: a partner search that was built, measured and NOT kept.

    python tools/breakline_signature.py CACHE_DIR GROUND_TRUTH.json [--window 3 --nn 30 ...]

The idea: two fragments that broke apart share the *same physical curve* where their fracture
surfaces meet the outer shell, so a window of that curve, described in a way that does not depend
on where the fragment lies in space, is the same on both sides.  Put every window of every
fragment in one KD-tree, and a fragment's partners fall out for the price of one query.

It does not work on this data, and the measurements are in
docs/superpowers/notes/2026-09-06-scale-pairs.md.  In one line: the breakline is the boundary of
an estimated fracture mask, not the crack itself, and on the Structure-from-Sherds++ pots that
mask is 50-63 % precise, so the two sides do not describe the same curve closely enough.  A window
that really does mate scores 0.86 against 1.16 for a random window pair, and the correct partner is
the nearest neighbour of a true correspondence in about 1 % of cases.  The pipeline ships the
coarse-stage screen (`matching.screen_pair`) instead.  This file is kept so the numbers can be
reproduced.

What is described
-----------------

The breakline points (`MatchData.brk_P`, the midpoints of the mesh edges between a shell face and
a fracture face) are chained into ordered curves and every window of a curve gets a descriptor
made of

* the window's own points, written in the local frame `(t, ns, f)` that the matcher already builds
  at the window's centre -- `ns` the smoothed shell normal, `f` the fracture normal projected into
  the shell, `t = ns x f` the tangent.  A rotation and a translation of the fragment move the frame
  with the curve, so these coordinates do not change;
* the dihedral angle between shell and fracture along the window, the same quantity the hypothesis
  generator pairs up;
* the local wall thickness along the window, measured as the distance from the outer breakline to
  the inner one (the two rims of the fracture ribbon).

Lengths are divided by that local thickness rather than by the fragment's own wall estimate: a
fragment carrying a rim measures a thicker wall than the body it broke from, but the crack line
itself has one thickness and both sides of it agree on the number.

The two sides of a crack
------------------------

A mating pair traverses the shared curve in opposite directions and sees complementary dihedrals
(`d_A + d_B ~ 180 degrees`), because the hypothesis frame of the second fragment is
`(-t, ns, -f)`.  Each fragment therefore gets two descriptor sets: `desc_F`, built in its own
frame, and `desc_R`, built in the flipped frame, with the window walked backwards and the
dihedral replaced by `180 - d`.  Fragments A and B touched if some window of `A.desc_F` is close
to some window of `B.desc_R`; `partner_votes` counts those hits per fragment pair.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from sherd_refit.geometry import face_adjacency, threads

log = logging.getLogger("sherd_refit")

SIGNATURE_VERSION = 1       # bump when the descriptor changes so stale caches are recomputed


@dataclass(frozen=True)
class SigParams:
    """Shape of the descriptor.  Distances are in units of the *local* wall thickness."""
    window: float = 3.0         # window length along the curve, in local wall thicknesses
    step: float = 0.5           # distance between neighbouring window centres
    samples: int = 3            # window points on each side of the centre (2*samples+1 in all)
    dihedral_weight: float = 1.0    # degrees are divided by 45 before this weight is applied
    thickness_weight: float = 1.0   # applied to (local thickness / centre thickness - 1)
    min_length: float = 3.0     # a chain shorter than this many local thicknesses is dropped

    def dim(self) -> int:
        return 2 * self.samples * 5

    def key(self) -> tuple:
        return (SIGNATURE_VERSION, self.window, self.step, self.samples,
                self.dihedral_weight, self.thickness_weight, self.min_length)


def breakline_edges(F: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """The mesh edges between a shell face and a fracture face, in `MatchData.brk_P` order."""
    fa, fb, ke = face_adjacency(F)
    return ke[frac[fa] != frac[fb]]


def chains(E: np.ndarray) -> list[np.ndarray]:
    """Order the breakline edges into curves: lists of edge indices, consecutive edges sharing a
    vertex.  A vertex where more than two breakline edges meet ends the chain rather than being
    guessed through, so a pinch point costs a join, not a wrong ordering."""
    inc: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(E.tolist()):
        inc.setdefault(a, []).append(k)
        inc.setdefault(b, []).append(k)
    used = np.zeros(len(E), bool)
    out = []

    def nxt(edge: int, vert: int):
        nb = inc[vert]
        if len(nb) != 2:
            return None
        e = nb[0] if nb[1] == edge else nb[1]
        if used[e]:
            return None
        a, b = E[e]
        return int(e), int(b if a == vert else a)

    for k in range(len(E)):
        if used[k]:
            continue
        used[k] = True
        a, b = (int(x) for x in E[k])
        seq = [k]
        cur, vert, closed = k, b, False
        while True:
            s = nxt(cur, vert)
            if s is None:
                break
            cur, vert = s
            used[cur] = True
            seq.append(cur)
            if vert == a:
                closed = True
                break
        if not closed:
            cur, vert = k, a
            while True:
                s = nxt(cur, vert)
                if s is None:
                    break
                cur, vert = s
                used[cur] = True
                seq.insert(0, cur)
        out.append(np.array(seq, dtype=np.int64))
    return out


def local_thickness(P: np.ndarray, ns: np.ndarray, fallback: float, k: int = 256) -> np.ndarray:
    """Wall thickness at each breakline point: the distance to the nearest breakline point whose
    shell normal points the other way, smoothed over half a wall along the curve.

    The fracture surface of a sherd is a ribbon between two breaklines, one on the outer shell and
    one on the inner; the wall is the distance across it.  Mesh normals point out of the solid, so
    the two rims see opposite shell normals, which is what identifies them.  Points that find no
    counterpart (a rim, a lip, a chain end) keep the fragment's own estimate.  This number is what
    the descriptor divides its lengths by, and its virtue is that both sides of a crack measure
    the same one -- unlike the per-fragment wall estimate, which a rim inflates.
    """
    th = np.full(len(P), float(fallback))
    if len(P) < 2:
        return th
    tree = cKDTree(P)
    d, j = tree.query(P, k=min(k, len(P)), workers=threads(), distance_upper_bound=3.0 * fallback)
    j = np.minimum(j, len(P) - 1)
    opp = np.isfinite(d) & (np.einsum("nj,nkj->nk", ns, ns[j]) < -0.5)
    hit = opp.any(1)
    th[hit] = d[hit, np.argmax(opp, axis=1)[hit]]
    # smooth along the curve: the raw distance jitters with the triangulation, and a window whose
    # length jitters cannot line up with the same window seen from the other side
    lists = tree.query_ball_point(P, 0.5 * fallback, workers=threads(), return_sorted=False)
    return np.array([float(np.median(th[l])) if l else t for l, t in zip(lists, th)])


def _resample(s: np.ndarray, A: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Linear interpolation of the columns of `A` (sampled at arc lengths `s`) at positions `q`."""
    return np.stack([np.interp(q, s, A[:, c]) for c in range(A.shape[1])], 1)


def _empty(sp: SigParams, thick: float = 1.0) -> dict:
    return dict(desc_F=np.zeros((0, sp.dim()), np.float32), desc_R=np.zeros((0, sp.dim()), np.float32),
                centre=np.zeros(0, np.int32), chain=np.zeros(0, np.int32), arc=np.zeros(0, np.float32),
                pos=np.zeros((0, 3), np.float32), frame=np.zeros((0, 3, 3), np.float32), t=float(thick))


def signature(brk_P: np.ndarray, brk_ns: np.ndarray, brk_nf: np.ndarray, F: np.ndarray,
              frac: np.ndarray, thick: float, sp: SigParams | None = None) -> dict:
    """Both descriptor sets of one fragment, plus the breakline index of every window centre.

    Returns `desc_F`, `desc_R` (n x `SigParams.dim()`, float32) and `centre`, all in the same
    order, so descriptor `i` of one set and descriptor `i` of the other describe the same window
    seen from the two sides of the crack.
    """
    sp = sp or SigParams()
    if len(brk_P) < 4:
        return _empty(sp, thick)
    f = brk_nf - np.einsum("ij,ij->i", brk_nf, brk_ns)[:, None] * brk_ns
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    tang = np.cross(brk_ns, f)
    dih = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", brk_ns, brk_nf), -1, 1)))
    th = local_thickness(brk_P, brk_ns, thick)
    cols = np.concatenate([brk_P, brk_ns, f, dih[:, None], th[:, None]], 1)
    k = np.arange(-sp.samples, sp.samples + 1) / sp.samples
    out_F, out_R, centres, chain_id, arcs, poss, frames = [], [], [], [], [], [], []
    for ci, ch in enumerate(chains(breakline_edges(F, frac))):
        if len(ch) < 4:
            continue
        # orient the chain along the frame's own tangent, so that the two sides of a crack walk it
        # in opposite directions and `desc_R` can simply reverse the window
        d = np.diff(brk_P[ch], axis=0)
        if float(np.einsum("ij,ij->i", d, tang[ch[:-1]]).sum()) < 0:
            ch = ch[::-1]
            d = np.diff(brk_P[ch], axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])
        L = float(s[-1])
        tl = th[ch]
        if L < sp.min_length * float(np.median(tl)):
            continue
        # window centres, walked in steps of the *local* wall thickness
        u, pos = 0.0, []
        while u <= L:
            t_loc = float(np.interp(u, s, tl))
            if u - 0.5 * sp.window * t_loc >= 0.0 and u + 0.5 * sp.window * t_loc <= L:
                pos.append((u, t_loc))
            u += max(1e-9, sp.step * t_loc)
        if not pos:
            continue
        u = np.array([p[0] for p in pos]); t_loc = np.array([p[1] for p in pos])
        q = u[:, None] + (0.5 * sp.window * t_loc)[:, None] * k[None, :]
        W = _resample(s, cols[ch], q.ravel()).reshape(len(u), len(k), cols.shape[1])
        c = W[:, sp.samples, :]
        ns_c = c[:, 3:6] / np.maximum(np.linalg.norm(c[:, 3:6], axis=1, keepdims=True), 1e-9)
        f_c = c[:, 6:9] - np.einsum("ij,ij->i", c[:, 6:9], ns_c)[:, None] * ns_c
        nf = np.linalg.norm(f_c, axis=1)
        keep = nf > 1e-3
        if not keep.any():
            continue
        f_c = f_c[keep] / nf[keep, None]
        ns_c, t_loc = ns_c[keep], t_loc[keep]
        t_c = np.cross(ns_c, f_c)
        W = W[keep]
        rel = (W[:, :, :3] - W[:, sp.samples, None, :3]) / t_loc[:, None, None]
        # The dihedral is taken *relative to the window's centre*.  Both fragments measure it
        # between normals smoothed over a third of a wall, and that smoothing pulls both of them
        # towards the crease: on the Structure-from-Sherds++ pots the sum over a mating pair is
        # 112-163 degrees where the geometry says 180, so `d_A` and `180 - d_B` are 37 degrees
        # apart on average and an absolute profile is pure noise.  The bias varies slowly along
        # the curve, so it cancels in the difference, and what is left -- how the fracture leans
        # over as the crack runs -- is the same on both sides with the sign flipped.
        dh = (W[:, :, 9] - W[:, sp.samples, None, 9]) / 45.0 * sp.dihedral_weight
        tt = (W[:, :, 10] / t_loc[:, None] - 1.0) * sp.thickness_weight
        for sign, out in ((1.0, out_F), (-1.0, out_R)):
            B = np.stack([sign * t_c, ns_c, sign * f_c], 2)         # rotation, det +1 either way
            loc = np.einsum("mkj,mjc->mkc", rel, B)
            prof_d, prof_t = sign * dh, tt
            if sign < 0:
                loc, prof_d, prof_t = loc[:, ::-1], prof_d[:, ::-1], prof_t[:, ::-1]
            cut = lambda X: np.delete(X, sp.samples, axis=1)         # the centre row is zero by construction
            out.append(np.concatenate([cut(loc).reshape(len(loc), -1), cut(prof_d), cut(prof_t)], 1))
        centres.append(ch[np.minimum(np.searchsorted(s, u[keep]), len(ch) - 1)])
        chain_id.append(np.full(int(keep.sum()), ci, np.int32))
        arcs.append(u[keep])        # arc length along the chain, in mesh units
        poss.append(W[:, sp.samples, :3])
        frames.append(np.stack([t_c, ns_c, f_c], 2))
    if not centres:
        return _empty(sp, thick)
    return dict(desc_F=np.concatenate(out_F).astype(np.float32),
                desc_R=np.concatenate(out_R).astype(np.float32),
                centre=np.concatenate(centres).astype(np.int32),
                chain=np.concatenate(chain_id).astype(np.int32),
                arc=np.concatenate(arcs).astype(np.float32),
                pos=np.concatenate(poss).astype(np.float32),
                frame=np.concatenate(frames).astype(np.float32), t=float(thick))


def _stack(sigs: dict[str, dict], names: list[str], key: str) -> np.ndarray:
    return np.concatenate([sigs[n][key] for n in names])


def window_matches(sigs: dict[str, dict], names: list[str], nn: int = 3, radius: float = np.inf):
    """For every window of every fragment, its `nn` nearest windows on the other side of a crack.

    Returns (i, j, d) into the concatenated `desc_F` / `desc_R` arrays, self-matches removed.
    """
    Fd, Rd = _stack(sigs, names, "desc_F"), _stack(sigs, names, "desc_R")
    owner = np.concatenate([np.full(len(sigs[n]["desc_F"]), i, np.int32) for i, n in enumerate(names)])
    k = min(len(Rd), nn + 8)
    d, j = cKDTree(Rd).query(Fd, k=k, workers=threads(), distance_upper_bound=radius)
    j = np.minimum(j, len(Rd) - 1)
    ok = np.isfinite(d) & (owner[:, None] != owner[j])
    # keep the first `nn` surviving neighbours of every window
    rank = np.cumsum(ok, axis=1)
    ok &= rank <= nn
    i = np.repeat(np.arange(len(Fd))[:, None], k, 1)
    return i[ok], j[ok], d[ok]


def partner_votes(sigs: dict[str, dict], nn: int = 30, radius: float = np.inf,
                  seam_tol: float = 2.0, min_run: float = 2.0) -> dict[tuple, float]:
    """Score every fragment pair by the length of seam their windows agree on, in wall thicknesses.

    A nearest-neighbour count alone is weak: a three-thickness window of a crack line is a gentle
    curve, and gentle curves resemble each other -- on the mixed collection the descriptor distance
    of a true correspondence is 0.86 against 1.16 for a random window pair.  What a real seam adds
    is *consistency*.  The two fragments walk the shared curve in opposite directions, so along one
    seam the sum of the two arc positions is one and the same constant; a false pair scatters.

    Matches are binned by (chain of A, chain of B, arc_A + arc_B) and the pair's score is the arc
    length that the matched windows of its best bin span, divided by the wall thickness.  Spanned
    length rather than a count of matches, because one window matches several neighbouring windows
    of the other fragment and they all land in the same bin.  `seam_tol` is the bin width in wall
    thicknesses, and every bin is also scored shifted by half a width so that a seam falling on a
    bin edge is not split in two.
    """
    names = [n for n in sigs if len(sigs[n]["desc_F"])]
    if len(names) < 2:
        return {}
    owner = np.concatenate([np.full(len(sigs[n]["desc_F"]), i, np.int32) for i, n in enumerate(names)])
    chain = _stack(sigs, names, "chain").astype(np.int64)
    arc = _stack(sigs, names, "arc").astype(np.float64)
    scale = np.concatenate([np.full(len(sigs[n]["desc_F"]), sigs[n].get("t", 1.0)) for n in names])
    i, j, _ = window_matches(sigs, names, nn=nn, radius=radius)
    if not len(i):
        return {}
    a, b = owner[i], owner[j]
    swap = a >= b
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    ca, cb = np.where(swap, chain[j], chain[i]), np.where(swap, chain[i], chain[j])
    sa = np.where(swap, arc[j], arc[i])
    t_pair = np.minimum(scale[i], scale[j])
    off = (arc[i] + arc[j]) / t_pair
    best: dict[tuple, float] = {}
    for shift in (0.0, 0.5):
        keys = np.stack([lo, hi, ca, cb, np.floor(off / seam_tol + shift).astype(np.int64)], 1)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        n = int(inv.max()) + 1
        # covered length, not spanned length: two matches with the same arc sum can sit at
        # opposite ends of the curve and span it without describing a seam at all
        cover = np.unique(np.stack([inv, np.floor(sa / t_pair).astype(np.int64)], 1), axis=0)
        span = np.bincount(cover[:, 0], minlength=n).astype(float)
        pair_of = np.zeros((n, 2), np.int64)
        pair_of[inv] = np.stack([lo, hi], 1)
        for b_id in np.flatnonzero(span >= min_run):
            key = (names[int(pair_of[b_id, 0])], names[int(pair_of[b_id, 1])])
            if span[b_id] > best.get(key, 0.0):
                best[key] = float(span[b_id])
    return best


def top_partners(votes: dict[tuple, float], names: list[str], k: int) -> set[tuple]:
    """The union over fragments of each fragment's `k` best-scoring partners."""
    per: dict[str, list] = {n: [] for n in names}
    for (a, b), v in votes.items():
        per.setdefault(a, []).append((v, b))
        per.setdefault(b, []).append((v, a))
    keep = set()
    for n, lst in per.items():
        for v, m in sorted(lst, key=lambda x: (-x[0], x[1]))[:k]:
            keep.add((n, m) if n < m else (m, n))
    return keep


FLIP = np.diag([-1.0, 1.0, -1.0])       # (t, ns, f) -> (-t, ns, -f), the mating frame


def match_poses(sigs: dict[str, dict], names: list[str], nn: int = 30, radius: float = np.inf,
                per_pair: int = 12):
    """The rigid transforms that the best matching windows of each fragment pair imply.

    A matched window pair fixes the pose completely: the two local frames must coincide, so
    `T = R_A R_B^T`, `c_A - T c_B`.  Yields `(a, b, T)` with `a < b` as indices into `names` and
    `T` mapping b's coordinates into a's, at most `per_pair` per pair, best descriptor distance
    first.
    """
    pos = _stack(sigs, names, "pos").astype(np.float64)
    frm = _stack(sigs, names, "frame").astype(np.float64)
    owner = np.concatenate([np.full(len(sigs[n]["desc_F"]), i, np.int32) for i, n in enumerate(names)])
    i, j, d = window_matches(sigs, names, nn=nn, radius=radius)
    if not len(i):
        return
    a, b = owner[i], owner[j]
    swap = a >= b
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    key = lo.astype(np.int64) * len(names) + hi
    order = np.lexsort((d, key))
    key, i, j, d = key[order], i[order], j[order], d[order]
    bounds = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1], [True]]))
    for s, e in zip(bounds[:-1], bounds[1:]):
        e = min(e, s + per_pair)
        RA, RB = frm[i[s:e]], frm[j[s:e]] @ FLIP
        R = np.einsum("nij,nkj->nik", RA, RB)
        tr = pos[i[s:e]] - np.einsum("nij,nj->ni", R, pos[j[s:e]])
        yield int(key[s] // len(names)), int(key[s] % len(names)), R, tr


def breakline_overlap(A: dict, B: dict, R: np.ndarray, tr: np.ndarray, delta: float) -> np.ndarray:
    """Fraction of B's sampled breakline that lands on A's breakline, per pose.

    The same quantity the matcher's coarse stage scores, but evaluated only on the handful of
    poses the descriptor matches propose instead of on every dihedral-compatible frame pair.
    """
    Q, QN = B["Q"], B["QN"]
    out = np.zeros(len(R))
    for k in range(len(R)):
        P = Q @ R[k].T + tr[k]
        d, jj = A["tree"].query(P, distance_upper_bound=delta, workers=1)
        hit = np.isfinite(d)
        if not hit.any():
            continue
        agree = np.einsum("ij,ij->i", A["ns"][jj[hit]], QN[hit] @ R[k].T) > 0.7
        out[k] = agree.sum() / len(Q)
    return out


def pair_scores(sigs: dict[str, dict], brk: dict[str, dict], nn: int = 30, radius: float = np.inf,
                per_pair: int = 12, delta: float = 0.15) -> dict[tuple, float]:
    """Best breakline overlap per fragment pair, over the poses its matched windows propose.

    This is the partner search the pipeline uses: a pair that never touched has no pose at which
    the two crack lines lie on top of each other, and the score collapses; a pair that did has
    one.  `delta` is in wall thicknesses.
    """
    names = [n for n in sigs if len(sigs[n]["desc_F"])]
    out: dict[tuple, float] = {}
    for ia, ib, R, tr in match_poses(sigs, names, nn=nn, radius=radius, per_pair=per_pair):
        a, b = names[ia], names[ib]
        t = min(brk[a]["t"], brk[b]["t"])
        s = float(breakline_overlap(brk[a], brk[b], R, tr, delta * t).max())
        if s > out.get((a, b), 0.0):
            out[(a, b)] = s
    return out


def brk_side(P: np.ndarray, ns: np.ndarray, sub: np.ndarray, thick: float, points: int = 250,
             seed: int = 0) -> dict:
    """The breakline of one fragment, ready for `breakline_overlap`: a KD-tree over all of it and
    a fixed random sample of the voxel-thinned subset to query with."""
    idx = sub if len(sub) else np.arange(len(P))
    if len(idx) > points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, points, replace=False))
    return dict(tree=cKDTree(P) if len(P) else None, ns=ns, Q=P[idx], QN=ns[idx], t=float(thick))


# ---------------------------------------------------------------- measurement harness

def _sig_job(args):
    from sherd_refit.fragment import Fragment
    path, sp = args
    fr = Fragment.load(path)
    s = signature(fr.md["brk_P"], fr.md["brk_ns"], fr.md["brk_nf"], fr.F, fr.frac, fr.thick, sp)
    return fr.name, fr.thick, s, fr.md["brk_P"], fr.md["brk_ns"], fr.md["brk_sub"]


def main(argv=None):
    import argparse, glob, json, os, time
    from concurrent.futures import ProcessPoolExecutor
    from sherd_refit.matching import top_partners
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("ground_truth")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--nn", type=int, default=30)
    ap.add_argument("--score", choices=("seam", "pose"), default="pose")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--ks", type=int, nargs="*", default=[5, 10, 15, 20, 30, 50])
    a = ap.parse_args(argv)
    sp = SigParams(window=a.window, step=a.step, samples=a.samples)
    caches = sorted(glob.glob(os.path.join(a.cache_dir, "*.npz")))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_sig_job, [(c, sp) for c in caches]))
    t_sig = time.time() - t0
    sigs = {n: s for n, _, s, _, _, _ in res}
    names = [n for n, _, _, _, _, _ in res]
    t0 = time.time()
    if a.score == "pose":
        brk = {n: brk_side(P, N, S, t) for n, t, _, P, N, S in res}
        score = pair_scores(sigs, brk, nn=a.nn)
    else:
        score = partner_votes(sigs, nn=a.nn)
    t_score = time.time() - t0
    gt = json.load(open(a.ground_truth))
    adj = {tuple(sorted(p)) for p in gt.get("adjacency", []) if p[0] in sigs and p[1] in sigs}
    n_all = len(names) * (len(names) - 1) // 2
    print("%d fragments, %d descriptors per side, signatures %.1f s, scoring %.1f s"
          % (len(names), sum(len(s["desc_F"]) for s in sigs.values()), t_sig, t_score))
    print("%4s %9s %9s %9s" % ("K", "pairs", "of all", "recall"))
    for K in a.ks:
        keep = top_partners(score, names, K)
        print("%4d %9d %9.3f %9.3f" % (K, len(keep), len(keep) / n_all, len(adj & keep) / max(1, len(adj))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
