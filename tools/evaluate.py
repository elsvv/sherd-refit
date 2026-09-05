#!/usr/bin/env python3
"""Score a sherd-refit output folder against a staged ground truth.

    python tools/evaluate.py OUT_DIR INPUT_DIR

``OUT_DIR`` holds ``transforms.json`` and ``report.json`` from ``sherd-refit
run``; ``INPUT_DIR`` holds the ``ground_truth.json`` written by
``tools/stage_sfspp.py`` (or by the synthetic generator, same format).  The
summary is printed and written to ``OUT_DIR/evaluation.json``.

What is measured
----------------

**Joins.**  Every join the assembly used (``report.json`` ``joins_used``) is
put in exactly one bucket:

``correct``
    the pair is adjacent in the ground truth *and* the relative pose is right —
    ``inv(T_a) @ T_b`` from ``transforms.json`` agrees with the same product of
    the ground-truth poses to within ``--rot-tol`` degrees and ``--trans-tol``
    wall thicknesses (thickness comes from ``transforms.json``);
``wrong_pose``
    adjacent pair, pose outside those tolerances;
``non_adjacent``
    both fragments have ground-truth poses and belong to the same object, but
    the ground truth does not call them neighbours;
``cross_object``
    the two fragments come from different objects (only possible in a mixed
    collection); always wrong;
``unscorable``
    at least one endpoint has no ground-truth pose, so nothing can be said.

Precision is ``correct / joins_used``; recall is ``correct`` over the number of
ground-truth adjacent pairs.

**Fragment accuracy** counts a fragment as correctly placed when it takes part
in at least one correct join, over the fragments that have a ground-truth pose.
This mirrors the "Sherd Accuracy" of Structure-from-Sherds++, so the number is
comparable with their published per-pot figures.

**Group purity** is, for each output group, the fraction of its fragments that
belong to the group's majority object; the overall figure is the
fragment-weighted mean.  Only meaningful for mixed collections.

A note on the translation test.  The obvious reading — the difference between
the two matrices' translation columns — measures the displacement *at the file
origin*, and in this dataset the meshes sit 300-500 mm away from theirs, so a
0.5 deg rotation error alone shows up as several millimetres of "translation".
The number reported and thresholded here is therefore the displacement of the
fragment's own centroid: ``|M_est c - M_gt c|`` with ``c`` the centroid of the
fragment's vertices, read from the meshes in ``INPUT_DIR``.  The origin
displacement is still recorded (``trans_origin_t``) and ``--translation
origin`` restores it as the thresholded quantity.  If the meshes cannot be
read, the origin form is used and the report says so.

**Missed true joins** lists the highest-scoring candidates that the pipeline did
*not* use although the pair really is adjacent, with the scores from
``report.json``, so the thresholds that rejected them can be read off directly.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

MESH_EXT = (".ply", ".obj", ".stl", ".off")


def centroids(input_dir, names):
    """Centroid of each fragment's vertices, in its own file coordinates."""
    try:
        import open3d as o3d
    except ImportError:
        return {}
    by_stem = {}
    # the meshes sit next to ground_truth.json, or one level down in fragments/ (synthetic sets)
    for d in (input_dir, os.path.join(input_dir, "fragments")):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(f)
            if ext.lower() in MESH_EXT:
                by_stem.setdefault(stem, os.path.join(d, f))
    out = {}
    for n in names:
        p = by_stem.get(n)
        if p is None:
            continue
        v = np.asarray(o3d.io.read_triangle_mesh(p).vertices)
        if len(v):
            out[n] = v.mean(0)
    return out


def load(out_dir, input_dir):
    with open(os.path.join(out_dir, "transforms.json")) as f:
        tr = json.load(f)
    with open(os.path.join(out_dir, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(input_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    return tr, rep, gt


def rel(Ta, Tb):
    """Pose of b expressed in a's frame."""
    return np.linalg.inv(np.asarray(Ta)) @ np.asarray(Tb)


def pose_error(M_est, M_gt, centroid=None):
    """(rotation error in deg, centroid displacement, file-origin displacement)."""
    M_est, M_gt = np.asarray(M_est), np.asarray(M_gt)
    R = M_est[:3, :3].T @ M_gt[:3, :3]
    c = (np.trace(R) - 1.0) / 2.0
    deg = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    d_org = float(np.linalg.norm(M_est[:3, 3] - M_gt[:3, 3]))
    if centroid is None:
        return deg, d_org, d_org
    p = np.asarray(centroid)
    d_cen = float(np.linalg.norm((M_est[:3, :3] @ p + M_est[:3, 3]) - (M_gt[:3, :3] @ p + M_gt[:3, 3])))
    return deg, d_cen, d_org


def evaluate(tr, rep, gt, rot_tol=5.0, trans_tol=0.5, top_missed=10, cen=None, translation="centroid"):
    thickness = float(tr["thickness"])
    trans_limit = trans_tol * thickness
    poses = {n: np.asarray(v["matrix"]) for n, v in tr["fragments"].items()}
    groups = tr["groups"]
    gt_poses = {n: np.asarray(v["matrix"]) for n, v in gt["fragments"].items()}
    object_of = gt.get("object_of", {})
    adjacency = {tuple(sorted(p)) for p in gt.get("adjacency", [])}
    unknown = set(gt.get("unknown", []))
    cen = cen or {}
    use_centroid = translation == "centroid" and bool(cen)

    # ground-truth adjacent pairs restricted to fragments actually staged here
    present = set(poses)
    gt_pairs = {p for p in adjacency if p[0] in present and p[1] in present}

    def classify(a, b, M_est):
        key = tuple(sorted((a, b)))
        # a join between two objects is wrong whether or not both poses are known
        if object_of.get(a, "?") != object_of.get(b, "?"):
            return "cross_object", None, None, None
        if a in unknown or b in unknown or a not in gt_poses or b not in gt_poses:
            return "unscorable", None, None, None
        M_gt = rel(gt_poses[key[0]], gt_poses[key[1]])
        M = M_est if (a, b) == key else np.linalg.inv(M_est)
        deg, d_cen, d_org = pose_error(M, M_gt, cen.get(key[1]))
        d = d_cen if use_centroid else d_org
        if key not in adjacency:
            return "non_adjacent", deg, d, d_org
        return ("correct" if deg <= rot_tol and d <= trans_limit else "wrong_pose"), deg, d, d_org

    joins = []
    buckets = collections.Counter()
    correct_pairs = set()
    for j in rep.get("joins_used", []):
        a, b = j["a"], j["b"]
        if a not in poses or b not in poses:
            continue
        M_est = rel(poses[a], poses[b])
        verdict, deg, d, d_org = classify(a, b, M_est)
        buckets[verdict] += 1
        if verdict == "correct":
            correct_pairs.add(tuple(sorted((a, b))))
        joins.append(dict(a=a, b=b, verdict=verdict, rot_deg=deg, trans=d,
                          trans_t=None if d is None else d / thickness,
                          trans_origin_t=None if d_org is None else d_org / thickness,
                          score=j.get("score")))

    used = len(joins)
    precision = buckets["correct"] / used if used else 0.0
    recall = buckets["correct"] / len(gt_pairs) if gt_pairs else 0.0

    scorable_frags = sorted(n for n in present if n in gt_poses and n not in unknown)
    placed_ok = {n for p in correct_pairs for n in p}
    frag_acc = len(placed_ok & set(scorable_frags)) / len(scorable_frags) if scorable_frags else 0.0

    # per object (a mixed collection holds several)
    per_object = {}
    for obj in sorted(set(object_of.get(n, "?") for n in scorable_frags)):
        frs = [n for n in scorable_frags if object_of.get(n) == obj]
        pairs = {p for p in gt_pairs if object_of.get(p[0]) == obj}
        ok = {n for n in frs if n in placed_ok}
        per_object[obj] = dict(fragments=len(frs), correct_fragments=len(ok),
                               fragment_accuracy=len(ok) / len(frs) if frs else 0.0,
                               gt_pairs=len(pairs),
                               correct_joins=len([p for p in correct_pairs if object_of.get(p[0]) == obj]),
                               recall=len([p for p in correct_pairs if object_of.get(p[0]) == obj]) / len(pairs) if pairs else 0.0)

    # group purity
    group_rows = []
    purity_num = purity_den = 0
    for k, g in enumerate(groups):
        objs = collections.Counter(object_of.get(n, "?") for n in g)
        obj, cnt = objs.most_common(1)[0]
        group_rows.append(dict(group=k, size=len(g), majority_object=obj,
                               purity=cnt / len(g), members=list(g)))
        if len(g) > 1:
            purity_num += cnt
            purity_den += len(g)
    overall_purity = purity_num / purity_den if purity_den else None

    # true adjacent pairs the pipeline had a candidate for but did not use
    used_keys = {tuple(sorted((j["a"], j["b"]))) for j in rep.get("joins_used", [])}
    missed = []
    best_per_pair = {}
    for c in rep.get("candidates", []):
        key = tuple(sorted((c["a"], c["b"])))
        if key not in gt_pairs or key in used_keys:
            continue
        if key not in best_per_pair or c["score"] > best_per_pair[key]["score"]:
            best_per_pair[key] = c
    for key, c in best_per_pair.items():
        M = np.asarray(c["T"])                       # maps b into a's frame
        M_gt = rel(gt_poses[c["a"]], gt_poses[c["b"]])
        deg, d_cen, d_org = pose_error(M, M_gt, cen.get(c["b"]))
        d = d_cen if use_centroid else d_org
        missed.append(dict(a=c["a"], b=c["b"], score=c["score"], rot_deg=deg, trans_t=d / thickness,
                           trans_origin_t=d_org / thickness,
                           pose_ok=bool(deg <= rot_tol and d <= trans_limit),
                           seam=c.get("seam"), tightA=c.get("tightA"), tightB=c.get("tightB"),
                           gap=c.get("gap"), pen=c.get("pen"), cont_n=c.get("cont_n"),
                           partial=c.get("partial", 0.0), accepted=c.get("accepted")))
    missed.sort(key=lambda r: -r["score"])

    return dict(
        thickness=thickness,
        tolerances=dict(rot_deg=rot_tol, trans_t=trans_tol, trans_units=trans_limit,
                        translation="centroid" if use_centroid else "file origin"),
        fragments=dict(total=len(present), with_ground_truth=len(scorable_frags), unknown=len(unknown & present)),
        joins=dict(used=used, **{k: buckets[k] for k in
                                 ("correct", "wrong_pose", "non_adjacent", "cross_object", "unscorable")},
                   gt_adjacent_pairs=len(gt_pairs), precision=precision, recall=recall),
        fragment_accuracy=dict(correct=len(placed_ok & set(scorable_frags)),
                               total=len(scorable_frags), fraction=frag_acc),
        per_object=per_object,
        groups=dict(count=len(groups), assembled=len([g for g in groups if len(g) > 1]),
                    largest=max((len(g) for g in groups), default=0),
                    overall_purity=overall_purity, detail=group_rows),
        join_detail=joins,
        missed_true_pairs=missed[:top_missed],
        timings=rep.get("timings", {}),
    )


def render(ev, out_dir, input_dir):
    L = []
    j, f = ev["joins"], ev["fragment_accuracy"]
    L.append(f"{out_dir}  vs  {input_dir}")
    L.append(f"wall thickness {ev['thickness']:.2f} units; a join is correct within "
             f"{ev['tolerances']['rot_deg']:.0f} deg and {ev['tolerances']['trans_t']:.2f} t "
             f"({ev['tolerances']['trans_units']:.2f} units), translation measured at the "
             f"{ev['tolerances']['translation']}")
    L.append(f"fragments: {ev['fragments']['total']} staged, {ev['fragments']['with_ground_truth']} with ground truth, "
             f"{ev['fragments']['unknown']} without")
    L.append("")
    L.append("joins used         %d" % j["used"])
    L.append("  correct          %d" % j["correct"])
    L.append("  wrong pose       %d   (adjacent pair, pose outside tolerance)" % j["wrong_pose"])
    L.append("  non-adjacent     %d   (same object, not neighbours in the ground truth)" % j["non_adjacent"])
    L.append("  cross-object     %d" % j["cross_object"])
    L.append("  unscorable       %d   (a fragment without ground truth)" % j["unscorable"])
    L.append("ground-truth adjacent pairs %d" % j["gt_adjacent_pairs"])
    L.append("precision %.3f   recall %.3f" % (j["precision"], j["recall"]))
    L.append("")
    L.append("fragment accuracy  %d / %d = %.1f %%" % (f["correct"], f["total"], 100 * f["fraction"]))
    if len(ev["per_object"]) > 1:
        L.append("")
        L.append("%-10s %8s %10s %8s %8s" % ("object", "frags", "frag acc", "gt pairs", "recall"))
        for obj, r in ev["per_object"].items():
            L.append("%-10s %8d %9.1f%% %8d %8.3f" % (obj, r["fragments"], 100 * r["fragment_accuracy"],
                                                      r["gt_pairs"], r["recall"]))
    g = ev["groups"]
    L.append("")
    L.append("groups: %d (%d with 2+ fragments), largest %d%s" % (
        g["count"], g["assembled"], g["largest"],
        "" if g["overall_purity"] is None else ", purity %.3f" % g["overall_purity"]))
    for r in g["detail"]:
        if r["size"] > 1:
            L.append("  group %-3d size %-3d majority %-8s purity %.2f" % (r["group"], r["size"], r["majority_object"], r["purity"]))
    bad = [r for r in ev["join_detail"] if r["verdict"] != "correct"]
    if bad:
        L.append("")
        L.append("wrong joins:")
        for r in bad:
            e = "" if r["rot_deg"] is None else "  %.1f deg, %.2f t" % (r["rot_deg"], r["trans_t"])
            L.append("  %-14s %s -- %s%s" % (r["verdict"], r["a"], r["b"], e))
    if ev["missed_true_pairs"]:
        L.append("")
        L.append("true adjacent pairs whose best candidate was not used (top %d by score):" % len(ev["missed_true_pairs"]))
        L.append("  %-28s %-28s %6s %6s %6s %11s %6s %6s %7s %7s %5s" %
                 ("A", "B", "score", "seam", "gap", "tight A/B", "pen", "cont", "rot deg", "trans t", "pose"))
        for r in ev["missed_true_pairs"]:
            L.append("  %-28s %-28s %6.2f %6.1f %6.3f %5.2f/%-5.2f %6.4f %6.2f %7.1f %7.2f %5s" % (
                r["a"], r["b"], r["score"], r["seam"] or 0, r["gap"] or 0, r["tightA"] or 0, r["tightB"] or 0,
                r["pen"] or 0, r["cont_n"] if r["cont_n"] is not None else -1, r["rot_deg"], r["trans_t"],
                "ok" if r["pose_ok"] else "no"))
    if ev["timings"]:
        L.append("")
        L.append("timing: " + ", ".join("%s %.1f s" % (k, v) for k, v in ev["timings"].items()))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("input_dir")
    ap.add_argument("--rot-tol", type=float, default=5.0, help="rotation tolerance in degrees")
    ap.add_argument("--trans-tol", type=float, default=0.5, help="translation tolerance in wall thicknesses")
    ap.add_argument("--top-missed", type=int, default=10)
    ap.add_argument("--translation", choices=("centroid", "origin"), default="centroid",
                    help="where the translation error is measured (default: the fragment centroid)")
    a = ap.parse_args(argv)
    tr, rep, gt = load(a.out_dir, a.input_dir)
    cen = {} if a.translation == "origin" else centroids(a.input_dir, sorted(tr["fragments"]))
    ev = evaluate(tr, rep, gt, a.rot_tol, a.trans_tol, a.top_missed, cen, a.translation)
    text = render(ev, a.out_dir, a.input_dir)
    print(text)
    with open(os.path.join(a.out_dir, "evaluation.json"), "w") as f:
        json.dump(ev, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
