#!/usr/bin/env python3
"""Compare two stage-boundary fixture dumps against the tolerances of the design document §10.2.

    tools/compare_fixtures.py REF_DIR CAND_DIR [--mode injected|native] [--stage NAME ...] [-v]

`REF_DIR` is the reference dump (the Python pipeline, `tools/dump_fixtures.py`), `CAND_DIR` the
one to judge — another Python dump, or the Rust core's `--dump-fixtures` output.

Two modes, as in §10.2: **injected**, where the candidate stage ran on the reference stage's own
inputs and must reproduce it closely, and **native**, where the candidate ran its whole pipeline
itself and only the statistical tolerances apply.

Exit status is 0 when every stage passes and 1 otherwise; a per-stage table is always printed.
Each row shows how many quantities were compared, how many failed, and the worst quantity as a
fraction of its tolerance (`1.00` is exactly at the limit).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

SEG_SAMPLES = 200_000

TOL = {
    # stage: {quantity: (injected, native)}; None means "not compared in that mode"
    "thickness":     dict(t=(None, 0.02), thick_mode=(None, 0.02)),
    "working_mesh":  dict(faces=(0.0, 0.05), res=(1e-9, 0.10), area=(1e-9, 0.005)),
    "segmentation":  dict(agreement=(0.995, 0.97), frac_fraction=(0.005, 0.02)),
    "breakline":     dict(count=(0.0, 0.10), hausdorff_t=(1e-4, 0.5), dih_deg=(0.1, None), ks=(None, 0.05)),
    "hypotheses":    dict(count=(0.0, 0.30)),
    "coarse":        dict(cs=(1.0 / 60 + 1e-6, None)),
    "stage1":        dict(rot_deg=(0.05, None), trans_t=(0.01, None), s1=(0.02, None)),
    "stage2":        dict(rot_deg=(0.05, None), trans_t=(0.01, None), tight=(0.01, None),
                          gap=(0.002, None), seam=(0.34, None), cont=(0.005, None),
                          cont_n=(0.01, None), pen=(0.0005, None)),
    "pair_result":   dict(rot_deg=(1.0, 1.0), trans_t=(0.05, 0.05)),
    "refine":        dict(rot_deg=(0.2, 0.2), trans_t=(0.02, 0.02)),
    "outputs":       dict(rot_deg=(0.2, 0.2), trans_t=(0.02, 0.02)),
}


# ---------------------------------------------------------------- dump access

class Dump:
    """Read-only view of one fixture directory."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise SystemExit(f"no such fixture directory: {root}")

    def path(self, rel: str) -> str:
        return os.path.join(self.root, *rel.split("/"))

    def has(self, rel: str) -> bool:
        return os.path.exists(self.path(rel))

    def npy(self, rel: str):
        p = self.path(rel + ".npy")
        return np.load(p, allow_pickle=False) if os.path.exists(p) else None

    def js(self, rel: str):
        p = self.path(rel + ".json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def subdirs(self, rel: str) -> list[str]:
        p = self.path(rel)
        if not os.path.isdir(p):
            return []
        return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))

    def fragments(self) -> list[str]:
        return self.subdirs("fragments")

    def pairs(self, group: str = "pairs") -> list[str]:
        return self.subdirs(group)


# ---------------------------------------------------------------- geometry helpers

def face_areas(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return 0.5 * np.linalg.norm(n, axis=1)


def centroids(V, F):
    return V[F].mean(1)


def rot_deg(R) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def pose_delta(T1, T2, t: float):
    """(rotation in degrees, translation in wall thicknesses) of the relative pose T1^-1 T2."""
    D = np.linalg.inv(np.asarray(T1, float)) @ np.asarray(T2, float)
    return rot_deg(D[:3, :3]), float(np.linalg.norm(D[:3, 3]) / max(t, 1e-12))


def rel_poses(poses: dict, groups: list[list[str]]) -> dict:
    """Pose of every group member relative to the group's first member."""
    out = {}
    for g in groups:
        anchor = np.asarray(poses[g[0]], float)
        inv = np.linalg.inv(anchor)
        for n in g:
            out[n] = inv @ np.asarray(poses[n], float)
    return out


# ---------------------------------------------------------------- result accumulation

class Stage:
    """One row of the report: how many quantities were compared and how badly the worst missed."""

    def __init__(self, name: str, mode: str):
        self.name = name
        self.mode = mode
        self.checks = 0
        self.fails: list[str] = []
        self.worst = 0.0
        self.worst_what = "-"
        self.skipped = False
        self.notes: list[str] = []

    # -- primitives ------------------------------------------------
    def _record(self, what: str, ratio: float, detail: str):
        self.checks += 1
        if ratio > self.worst or math.isinf(ratio):
            self.worst, self.worst_what = ratio, f"{what} {detail}"
        if ratio > 1.0:
            self.fails.append(f"{what}: {detail}")

    def near(self, what, ref, cand, tol):
        """|ref - cand| <= tol (tol == 0 means exact equality)."""
        if tol is None:
            return
        ref, cand = float(ref), float(cand)
        d = abs(ref - cand)
        ratio = (0.0 if d == 0 else float("inf")) if tol == 0 else d / tol
        self._record(what, ratio, f"|{ref:.6g} - {cand:.6g}| = {d:.3g} (tol {tol:g})")

    def rel(self, what, ref, cand, tol):
        """|ref - cand| <= tol * |ref| (a relative tolerance)."""
        if tol is None:
            return
        ref, cand = float(ref), float(cand)
        d = abs(ref - cand)
        scale = abs(ref) if ref else 1.0
        ratio = (0.0 if d == 0 else float("inf")) if tol == 0 else d / (tol * scale)
        self._record(what, ratio, f"{ref:.6g} vs {cand:.6g} ({d / scale:.3%}, tol {tol:.3%})")

    def at_least(self, what, value, lo, best: float = 1.0):
        """`value >= lo`, scored as the shortfall against the best attainable value.

        `best` (1.0 for a fraction) is what a perfect candidate reaches, so a perfect result
        scores 0 and one exactly at the limit scores 1.
        """
        if lo is None:
            return
        value = float(value)
        span = best - lo
        ratio = (best - value) / span if span > 0 else (0.0 if value >= lo else float("inf"))
        self._record(what, max(ratio, 0.0), f"{value:.6g} (>= {lo:g})")

    def at_most(self, what, value, hi):
        if hi is None:
            return
        value = float(value)
        ratio = value / hi if hi else (0.0 if value == 0 else float("inf"))
        self._record(what, ratio, f"{value:.6g} (<= {hi:g})")

    def same(self, what, ref, cand):
        equal = bool(ref == cand)
        self._record(what, 0.0 if equal else float("inf"),
                     "identical" if equal else f"{ref!r} != {cand!r}")

    def same_set(self, what, ref, cand):
        ref, cand = set(ref), set(cand)
        missing, extra = sorted(ref - cand), sorted(cand - ref)
        equal = not missing and not extra
        self._record(what, 0.0 if equal else float("inf"),
                     f"{len(ref)} entries, identical" if equal
                     else f"{len(missing)} missing e.g. {missing[:3]}, {len(extra)} extra e.g. {extra[:3]}")

    def skip(self, why: str):
        self.skipped = True
        self.notes.append(why)

    # -- reporting -------------------------------------------------
    @property
    def ok(self) -> bool:
        return not self.fails

    def status(self) -> str:
        if self.skipped and not self.checks:
            return "SKIP"
        return "PASS" if self.ok else "FAIL"


def tol_of(stage: str, key: str, mode: str):
    inj, nat = TOL[stage][key]
    return inj if mode == "injected" else nat


# ---------------------------------------------------------------- stages

def check_load(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("load", mode)
    names = sorted(set(ref.fragments()) | set(cand.fragments()))
    if not names:
        st.skip("no fragments dumped")
        return st
    st.same_set("fragments", ref.fragments(), cand.fragments())
    for n in names:
        r, c = ref.js(f"fragments/{n}/load.n_orig"), cand.js(f"fragments/{n}/load.n_orig")
        if r is None or c is None:
            st.skip(f"{n}: no load.n_orig")
            continue
        for k in ("n_orig_vertices", "n_orig_faces", "n_vertices", "n_faces"):
            st.same(f"{n}.{k}", r.get(k), c.get(k))
        Vr, Vc = ref.npy(f"fragments/{n}/load.V0"), cand.npy(f"fragments/{n}/load.V0")
        if Vr is not None and Vc is not None:
            st.same(f"{n}.V0.shape", Vr.shape, Vc.shape)
            if mode == "injected" and Vr.shape == Vc.shape:
                st.at_most(f"{n}.V0.max_diff", np.abs(Vr - Vc).max() if len(Vr) else 0.0, 1e-12)
    return st


def check_thickness(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("thickness", mode)
    for n in sorted(set(ref.fragments()) & set(cand.fragments())):
        tr, tc = ref.js(f"fragments/{n}/thick.t"), cand.js(f"fragments/{n}/thick.t")
        if tr is None or tc is None:
            continue
        if mode == "injected":
            # "same histogram bin, or one bin out on a count tie": the bin width is the reference
            # histogram's own, recovered from the dumped ray distances
            bins = _thick_bin(ref, n)
            st.near(f"{n}.t", tr, tc, bins if bins else 1e-9)
        else:
            st.rel(f"{n}.t", tr, tc, tol_of("thickness", "t", mode))
        mr, mc = ref.js(f"fragments/{n}/thick.thick_mode"), cand.js(f"fragments/{n}/thick.thick_mode")
        if mr is not None and mc is not None:
            if mode == "injected":
                bins = _thick_bin(ref, n)
                st.near(f"{n}.thick_mode", mr, mc, bins if bins else 1e-9)
            else:
                st.rel(f"{n}.thick_mode", mr, mc, tol_of("thickness", "thick_mode", mode))
    if not st.checks:
        st.skip("no thickness dumped")
    return st


def _thick_bin(d: Dump, name: str) -> float:
    """Width of one bin of the thickness histogram (R §3.2), or 0 when the rays are not dumped."""
    t_hit = d.npy(f"fragments/{name}/thick.t_hit")
    prim = d.npy(f"fragments/{name}/thick.prim")
    stats = d.js(f"fragments/{name}/mesh.stats")
    if t_hit is None or prim is None or stats is None:
        return 0.0
    n_faces0 = d.js(f"fragments/{name}/load.n_orig")
    limit = n_faces0["n_faces"] if n_faces0 else np.iinfo(np.int64).max
    ok = np.isfinite(t_hit) & (prim < limit)
    if ok.sum() < 100:
        return 0.0
    return float(np.percentile(t_hit[ok], 90) / 60.0)


def check_working_mesh(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("working_mesh", mode)
    for n in sorted(set(ref.fragments()) & set(cand.fragments())):
        r, c = ref.js(f"fragments/{n}/mesh.stats"), cand.js(f"fragments/{n}/mesh.stats")
        if r is None or c is None:
            continue
        st.rel(f"{n}.faces", r["faces"], c["faces"], tol_of("working_mesh", "faces", mode))
        st.rel(f"{n}.res", r["res"], c["res"], tol_of("working_mesh", "res", mode))
        st.rel(f"{n}.area", r["area"], c["area"], tol_of("working_mesh", "area", mode))
        st.same(f"{n}.watertight", r["watertight"], c["watertight"])
    if not st.checks:
        st.skip("no working mesh dumped")
    return st


def _mesh_and_frac(d: Dump, name: str):
    V, F = d.npy(f"fragments/{name}/mesh.V"), d.npy(f"fragments/{name}/mesh.F")
    frac = d.npy(f"fragments/{name}/seg.frac_final")
    return V, F, frac


def check_segmentation(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("segmentation", mode)
    for n in sorted(set(ref.fragments()) & set(cand.fragments())):
        Vr, Fr, fr = _mesh_and_frac(ref, n)
        Vc, Fc, fc = _mesh_and_frac(cand, n)
        if fr is None or fc is None or Fr is None or Fc is None:
            continue
        Ar, Ac = face_areas(Vr, Fr), face_areas(Vc, Fc)
        st.near(f"{n}.frac_fraction", Ar[fr].sum() / Ar.sum(), Ac[fc].sum() / Ac.sum(),
                tol_of("segmentation", "frac_fraction", mode))
        if Fr.shape == Fc.shape and np.array_equal(Fr, Fc) and np.array_equal(Vr, Vc):
            agree = float(Ar[fr == fc].sum() / Ar.sum())        # same mesh: compare labels directly
        else:
            agree = _sampled_agreement(Vr, Fr, Ar, fr, Vc, Fc, fc)
        st.at_least(f"{n}.agreement", agree, tol_of("segmentation", "agreement", mode))
    if not st.checks:
        st.skip("no segmentation dumped")
    return st


def _sampled_agreement(Vr, Fr, Ar, fr, Vc, Fc, fc) -> float:
    """Area-weighted label agreement over 200 000 points sampled on the reference mesh.

    Each point takes the label of the nearest face centroid on each mesh, which is what makes the
    number comparable when the two working meshes are different tessellations of the same surface.
    """
    rng = np.random.default_rng(0)
    p = Ar / Ar.sum()
    pick = rng.choice(len(Ar), SEG_SAMPLES, p=p)
    u, v = rng.random(SEG_SAMPLES), rng.random(SEG_SAMPLES)
    sw = u + v > 1
    u[sw], v[sw] = 1 - u[sw], 1 - v[sw]
    P = (Vr[Fr[pick, 0]] + u[:, None] * (Vr[Fr[pick, 1]] - Vr[Fr[pick, 0]])
         + v[:, None] * (Vr[Fr[pick, 2]] - Vr[Fr[pick, 0]]))
    lab_r = fr[cKDTree(centroids(Vr, Fr)).query(P, workers=-1)[1]]
    lab_c = fc[cKDTree(centroids(Vc, Fc)).query(P, workers=-1)[1]]
    return float((lab_r == lab_c).mean())


def check_breakline(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("breakline", mode)
    for n in sorted(set(ref.fragments()) & set(cand.fragments())):
        Pr, Pc = ref.npy(f"fragments/{n}/md.brk_P"), cand.npy(f"fragments/{n}/md.brk_P")
        if Pr is None or Pc is None:
            continue
        t = ref.js(f"fragments/{n}/thick.t") or 1.0
        st.rel(f"{n}.count", len(Pr), len(Pc), tol_of("breakline", "count", mode))
        if not len(Pr) or not len(Pc):
            continue
        d_rc, j_rc = cKDTree(Pc).query(Pr, workers=-1)
        d_cr, _ = cKDTree(Pr).query(Pc, workers=-1)
        if mode == "injected":
            st.at_most(f"{n}.hausdorff_t", max(d_rc.max(), d_cr.max()) / t,
                       tol_of("breakline", "hausdorff_t", mode))
        else:
            st.at_most(f"{n}.p99_t", max(np.percentile(d_rc, 99), np.percentile(d_cr, 99)) / t,
                       tol_of("breakline", "hausdorff_t", mode))
        dr, dc = ref.npy(f"fragments/{n}/md.brk_dih"), cand.npy(f"fragments/{n}/md.brk_dih")
        if dr is None or dc is None:
            continue
        if mode == "injected":
            st.at_most(f"{n}.dih_deg", float(np.abs(dr - dc[j_rc]).max()),
                       tol_of("breakline", "dih_deg", mode))
        else:
            from scipy.stats import ks_2samp
            st.at_most(f"{n}.dih_ks", float(ks_2samp(dr, dc).statistic), tol_of("breakline", "ks", mode))
    if not st.checks:
        st.skip("no breakline dumped")
    return st


def _pair_t(d: Dump, group: str, pair: str) -> float:
    sc = d.js(f"{group}/{pair}/scales")
    return float(sc["t"]) if sc else 1.0


def check_hypotheses(ref: Dump, cand: Dump, mode: str, group="pairs") -> Stage:
    st = Stage("hypotheses", mode)
    for pr in sorted(set(ref.pairs(group)) & set(cand.pairs(group))):
        pa_r, pb_r = ref.npy(f"{group}/{pr}/hyp.pa"), ref.npy(f"{group}/{pr}/hyp.pb")
        pa_c, pb_c = cand.npy(f"{group}/{pr}/hyp.pa"), cand.npy(f"{group}/{pr}/hyp.pb")
        if pa_r is None or pa_c is None:
            continue
        if mode == "injected":
            for what, a, b in (("ia", ref.npy(f"{group}/{pr}/hyp.ia"), cand.npy(f"{group}/{pr}/hyp.ia")),
                               ("ib", ref.npy(f"{group}/{pr}/hyp.ib"), cand.npy(f"{group}/{pr}/hyp.ib"))):
                if a is not None and b is not None:
                    st.same(f"{pr}.{what}", a.tolist(), b.tolist())
            st.same(f"{pr}.pairs", (pa_r.tolist(), pb_r.tolist()), (pa_c.tolist(), pb_c.tolist()))
        else:
            st.rel(f"{pr}.count", len(pa_r), len(pa_c), tol_of("hypotheses", "count", mode))
    if not st.checks:
        st.skip("no hypotheses dumped")
    return st


def check_coarse(ref: Dump, cand: Dump, mode: str, group="pairs") -> Stage:
    st = Stage("coarse", mode)
    if mode != "injected":
        st.skip("coarse scores are compared in injected mode only")
        return st
    for pr in sorted(set(ref.pairs(group)) & set(cand.pairs(group))):
        cr, cc = ref.npy(f"{group}/{pr}/coarse.cs"), cand.npy(f"{group}/{pr}/coarse.cs")
        if cr is None or cc is None:
            continue
        if cr.shape != cc.shape:
            st.same(f"{pr}.cs.shape", cr.shape, cc.shape)
            continue
        st.at_most(f"{pr}.cs", float(np.abs(cr - cc).max()) if len(cr) else 0.0,
                   tol_of("coarse", "cs", mode))
    if not st.checks:
        st.skip("no coarse scores dumped")
    return st


def _stage1_by_id(d: Dump, group: str, pr: str):
    kept = d.npy(f"{group}/{pr}/nms1.kept")
    T = d.npy(f"{group}/{pr}/s1.T")
    s1 = d.npy(f"{group}/{pr}/s1.score")
    if kept is None or T is None or s1 is None:
        return None
    return {int(h): (T[i], float(s1[i]), i) for i, h in enumerate(kept[:len(T)])}


def check_stage1(ref: Dump, cand: Dump, mode: str, group="pairs") -> Stage:
    st = Stage("stage1", mode)
    if mode != "injected":
        st.skip("stage-1 poses are compared in injected mode only")
        return st
    for pr in sorted(set(ref.pairs(group)) & set(cand.pairs(group))):
        R, C = _stage1_by_id(ref, group, pr), _stage1_by_id(cand, group, pr)
        if R is None or C is None:
            continue
        t = _pair_t(ref, group, pr)
        st.same_set(f"{pr}.kept", R, C)
        for h in sorted(set(R) & set(C)):
            ang, dist = pose_delta(R[h][0], C[h][0], t)
            st.at_most(f"{pr}.h{h}.rot_deg", ang, tol_of("stage1", "rot_deg", mode))
            st.at_most(f"{pr}.h{h}.trans_t", dist, tol_of("stage1", "trans_t", mode))
            st.near(f"{pr}.h{h}.s1", R[h][1], C[h][1], tol_of("stage1", "s1", mode))
    if not st.checks:
        st.skip("no stage-1 poses dumped")
    return st


def _stage2_by_id(d: Dump, group: str, pr: str):
    """Candidates keyed by the hypothesis id their stage-1 pose came from."""
    kept1 = d.npy(f"{group}/{pr}/nms1.kept")
    kept2 = d.npy(f"{group}/{pr}/nms2.kept")
    T = d.npy(f"{group}/{pr}/s2.T")
    sc = d.js(f"{group}/{pr}/s2.scores")
    acc = d.npy(f"{group}/{pr}/s2.accepted")
    if kept1 is None or kept2 is None or T is None or sc is None or acc is None:
        return None
    out = {}
    for i, pos in enumerate(kept2):
        if i >= len(T) or pos >= len(kept1):
            continue
        out[int(kept1[int(pos)])] = (T[i], sc[i], bool(acc[i]))
    return out


def check_stage2(ref: Dump, cand: Dump, mode: str, group="pairs") -> Stage:
    st = Stage("stage2", mode)
    if mode != "injected":
        st.skip("stage-2 poses are compared in injected mode only")
        return st
    fields = ("tight", "gap", "seam", "cont", "cont_n", "pen")
    for pr in sorted(set(ref.pairs(group)) & set(cand.pairs(group))):
        R, C = _stage2_by_id(ref, group, pr), _stage2_by_id(cand, group, pr)
        if R is None or C is None:
            continue
        t = _pair_t(ref, group, pr)
        st.same_set(f"{pr}.candidates", R, C)
        for h in sorted(set(R) & set(C)):
            ang, dist = pose_delta(R[h][0], C[h][0], t)
            st.at_most(f"{pr}.h{h}.rot_deg", ang, tol_of("stage2", "rot_deg", mode))
            st.at_most(f"{pr}.h{h}.trans_t", dist, tol_of("stage2", "trans_t", mode))
            for f in fields:
                if f in R[h][1] and f in C[h][1]:
                    st.near(f"{pr}.h{h}.{f}", R[h][1][f], C[h][1][f], tol_of("stage2", f, mode))
            st.same(f"{pr}.h{h}.accepted", R[h][2], C[h][2])
    if not st.checks:
        st.skip("no stage-2 candidates dumped")
    return st


def check_pair_result(ref: Dump, cand: Dump, mode: str, group="pairs") -> Stage:
    st = Stage("pair_result", mode)
    prs = sorted(set(ref.pairs(group)) | set(cand.pairs(group)))
    if not prs:
        st.skip("no pair results dumped")
        return st
    st.same_set("pairs", ref.pairs(group), cand.pairs(group))
    acc_r, acc_c = set(), set()
    for pr in prs:
        cr, cc = ref.js(f"{group}/{pr}/result.candidates"), cand.js(f"{group}/{pr}/result.candidates")
        if cr and any(c["accepted"] for c in cr):
            acc_r.add(pr)
        if cc and any(c["accepted"] for c in cc):
            acc_c.add(pr)
    st.same_set("accepted pairs", acc_r, acc_c)
    for pr in sorted(acc_r & acc_c):
        cr, cc = ref.js(f"{group}/{pr}/result.candidates"), cand.js(f"{group}/{pr}/result.candidates")
        t = _pair_t(ref, group, pr)
        ang, dist = pose_delta(np.array(cr[0]["T"]), np.array(cc[0]["T"]), t)
        st.at_most(f"{pr}.best.rot_deg", ang, tol_of("pair_result", "rot_deg", mode))
        st.at_most(f"{pr}.best.trans_t", dist, tol_of("pair_result", "trans_t", mode))
    return st


def check_assembly(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("assembly", mode)
    gr, gc = ref.js("assembly/groups"), cand.js("assembly/groups")
    if gr is None or gc is None:
        st.skip("no assembly dumped")
        return st
    st.same("groups", sorted(sorted(g) for g in gr), sorted(sorted(g) for g in gc))
    ur, uc = ref.js("assembly/used") or [], cand.js("assembly/used") or []
    st.same_set("joins used", [(c["a"], c["b"]) for c in ur], [(c["a"], c["b"]) for c in uc])
    rr, rc = ref.js("assembly/rejected") or [], cand.js("assembly/rejected") or []
    st.same_set("joins rejected", [(c["a"], c["b"]) for c in rr], [(c["a"], c["b"]) for c in rc])
    return st


def _rel_pose_check(st: Stage, stage_key: str, mode: str, poses_r, poses_c, groups, t: float, tag: str):
    common = [[n for n in g if n in poses_r and n in poses_c] for g in groups]
    common = [g for g in common if len(g) > 1]
    if not common:
        st.skip(f"{tag}: no group with two placed fragments")
        return
    rr, rc = rel_poses(poses_r, common), rel_poses(poses_c, common)
    for g in common:
        for n in g[1:]:
            ang, dist = pose_delta(rr[n], rc[n], t)
            st.at_most(f"{tag}.{n}.rot_deg", ang, tol_of(stage_key, "rot_deg", mode))
            st.at_most(f"{tag}.{n}.trans_t", dist, tol_of(stage_key, "trans_t", mode))


def check_refine(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("refine", mode)
    pr, pc = ref.js("refine/poses_final"), cand.js("refine/poses_final")
    groups = ref.js("assembly/groups")
    if pr is None or pc is None or groups is None:
        st.skip("no refinement dumped")
        return st
    med = ref.js("assembly/md_t_median") or {}
    _rel_pose_check(st, "refine", mode, pr, pc, groups, float(med.get("t", 1.0)), "rel")
    return st


def check_outputs(ref: Dump, cand: Dump, mode: str) -> Stage:
    st = Stage("outputs", mode)
    tr, tc = ref.js("outputs/transforms"), cand.js("outputs/transforms")
    if tr is None or tc is None:
        st.skip("no outputs dumped")
        return st
    st.same_set("transforms.fragments", tr["fragments"], tc["fragments"])
    st.same("transforms.groups", sorted(sorted(g) for g in tr["groups"]),
            sorted(sorted(g) for g in tc["groups"]))
    for n in sorted(set(tr["fragments"]) & set(tc["fragments"])):
        st.same(f"placed.{n}", tr["fragments"][n]["placed"], tc["fragments"][n]["placed"])
    poses_r = {n: d["matrix"] for n, d in tr["fragments"].items()}
    poses_c = {n: d["matrix"] for n, d in tc["fragments"].items()}
    _rel_pose_check(st, "outputs", mode, poses_r, poses_c, tr["groups"], float(tr["thickness"]), "rel")
    rr, rc = ref.js("outputs/report"), cand.js("outputs/report")
    if rr is not None and rc is not None:
        st.same_set("report.keys", rr, rc)
        if rr.get("fragments") and rc.get("fragments"):
            st.same_set("report.fragment keys", rr["fragments"][0], rc["fragments"][0])
        if rr.get("candidates") and rc.get("candidates"):
            st.same_set("report.candidate keys", rr["candidates"][0], rc["candidates"][0])
        st.same("report.groups", sorted(sorted(g) for g in rr["groups"]),
                sorted(sorted(g) for g in rc["groups"]))
    return st


STAGES = [check_load, check_thickness, check_working_mesh, check_segmentation, check_breakline,
          check_hypotheses, check_coarse, check_stage1, check_stage2, check_pair_result,
          check_assembly, check_refine, check_outputs]


# ---------------------------------------------------------------- driver

def compare(ref_dir: str, cand_dir: str, mode: str = "injected", only=None) -> list[Stage]:
    ref, cand = Dump(ref_dir), Dump(cand_dir)
    out = []
    for fn in STAGES:
        name = fn.__name__[len("check_"):]
        if only and name not in only:
            continue
        out.append(fn(ref, cand, mode))
    return out


def print_table(stages: list[Stage], verbose: bool = False, limit: int = 8, stream=sys.stdout):
    w = max([len(s.name) for s in stages] + [5])
    print(f"{'stage'.ljust(w)}  {'checks':>6}  {'failed':>6}  {'worst/tol':>9}  status", file=stream)
    print("-" * (w + 34), file=stream)
    for s in stages:
        worst = "-" if not s.checks else ("inf" if math.isinf(s.worst) else f"{s.worst:.2f}")
        print(f"{s.name.ljust(w)}  {s.checks:6d}  {len(s.fails):6d}  {worst:>9}  {s.status()}", file=stream)
    for s in stages:
        if s.fails:
            print(f"\n{s.name}: {len(s.fails)} failure(s)", file=stream)
            for f in s.fails[:limit]:
                print(f"  - {f}", file=stream)
            if len(s.fails) > limit:
                print(f"  ... {len(s.fails) - limit} more", file=stream)
        elif verbose:
            if s.notes:
                print(f"\n{s.name}: {'; '.join(s.notes)}", file=stream)
            if s.checks:
                print(f"\n{s.name}: worst {s.worst_what}", file=stream)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--mode", default="injected", choices=("injected", "native"))
    ap.add_argument("--stage", action="append", default=None,
                    help="only this stage (repeatable): " + ", ".join(fn.__name__[6:] for fn in STAGES))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    stages = compare(a.ref_dir, a.cand_dir, a.mode, a.stage)
    print_table(stages, a.verbose)
    failed = [s.name for s in stages if not s.ok]
    print(("\nFAIL: " + ", ".join(failed)) if failed else "\nPASS: all stages within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
