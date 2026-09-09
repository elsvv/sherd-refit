#!/usr/bin/env python3
"""Run the Rust core on the development sets at five seeds and score every run.

    python tools/quality_gate.py [--bin target/release/sherd-refit-rs]
                                 [--seeds 0 1 2 3 4] [--backend cpu]
                                 [--out output/quality] [--sets NAME ...]

This is the quality gate of everything built after the reference hand-over
(D §13 question 7, task H4): from that commit the algorithm lives in Rust, the
parity harness is frozen at the sixteen stages R already describes, and a change
to the algorithm is judged by what it does to the ground truth rather than by
what it does to a Python dump.  The gate is therefore ``tools/evaluate.py`` over
the sets that have ground truth, at seeds 0-4, because R §13's own rows are
bands over a seed sweep and a single draw cannot be compared with them.

For each of the eight development sets and each seed it runs

    sherd-refit-rs run <input> --out <work> --backend cpu \
        --no-preview --no-meshes --seed <s>

and scores the result with ``tools/evaluate.py``, then writes one table --
fragment accuracy, precision, the four join buckets, group purity and wall
clock, one row per (set, seed) -- as ``<out>/quality.md`` and ``<out>/quality.json``.
Under the table each set gets two verdicts, and they are different things.  The
**gate** is what R §13 states as a prohibition -- its last row (cross-object
joins 0, group purity 1.000) on the seven single-object collections, the
terracotta's decision row, and pot_G's "at most two joins used, every one a
wrong-pose join on an adjacent pair" with its 0 % accuracy -- and a failure of
one of those exits non-zero.  The **band** is R §13's quoted spread of fragment
accuracy and precision, and it is reported rather than gated: those bands are the
*reference's* own five draws, the port's five are a second sample of the same
chaotic quantity, and task H4 measured the port's to be wider on three sets.
Gating one five-draw sample with another fails a run for a draw instead of for a
change; the band line exists so that a step which moves a set out of the
reference's spread, or back into it, has to say so.

Why this is Python and not a subcommand of the binary
-----------------------------------------------------
The score is ``tools/evaluate.py`` and nothing else.  A Rust twin would have to
reimplement the buckets, the centroid-referenced pose error and the purity
weighting, and two scorers that can disagree are worse than one scorer in a
language the museum already installs: ``evaluate.py`` needs numpy and (for the
centroid form of the translation error) Open3D, so the environment this script
runs in is the environment the metric already required.  What the museum runs
for the *algorithm* is the binary; what it runs for the *number* is this file,
which calls the binary as a subprocess and imports ``evaluate`` as a module.
Importing rather than shelling out matters: Open3D loads once for all forty runs
and each set's vertex centroids are read once instead of five times.

Sets and their ground truth
---------------------------
Seven of the eight sets carry a ``ground_truth.json`` beside their meshes and are
scored by ``evaluate.py``.  The terracotta (``input/test_fragments_1``) has none
-- the museum's assembly was never staged as poses -- so its R §13 row is a set
of *decisions*, and this script checks those decisions directly from
``report.json``: the two joins 021-094 and 094-104 and no others, 007 unplaced,
both penetrations 0, both tight contacts >= 0.27, and the two seams within 20 %
of 20.3 t and 10.7 t.  Its evaluate.py columns read ``-`` in the table, which is
a missing ground truth and not a missing measurement.

Cost
----
The work directory of a set is reused across its five seeds, so only the first
seed of each set pays for the fragment cache; another seed recomputes R §3.5's
three sampled arrays and nothing else (R §3.7), which is what makes five seeds
affordable.  Measured on an M2 Pro 10-core: see the header the script writes.
``--keep-work`` leaves the work trees behind; by default they are removed once
the last seed of a set has been scored, because forty of them do not fit beside
the caches they came from.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import evaluate as ev_mod  # noqa: E402  (after sys.path)

# name -> (directory the run is pointed at, directory holding ground_truth.json or None)
SETS = [
    ("terracotta", "input/test_fragments_1/fragments", None),
    ("pot_A", "input/sfspp/pot_A", "input/sfspp/pot_A"),
    ("pot_B", "input/sfspp/pot_B", "input/sfspp/pot_B"),
    ("pot_C", "input/sfspp/pot_C", "input/sfspp/pot_C"),
    ("pot_G", "input/sfspp/pot_G", "input/sfspp/pot_G"),
    ("pot_H", "input/sfspp/pot_H", "input/sfspp/pot_H"),
    ("synthetic_20", "input/synthetic_pingsdorf_20/fragments", "input/synthetic_pingsdorf_20"),
    ("mixed_ABG", "input/sfspp/mixed_ABG", "input/sfspp/mixed_ABG"),
]

# R §13's rows, split into the two things they are.
#
#   gate  -- a *prohibition*: R §13's last row ("cross-object joins 0; group purity 1.000") on the
#            seven single-object collections, R §13's terracotta decision row, and pot_G's "at most
#            two joins used, every one a wrong-pose join on a ground-truth-adjacent pair" with its
#            0 % fragment accuracy.  A failure here exits non-zero.
#   band  -- the *spread* R §13 quotes for fragment accuracy and precision.  Reported, never a gate.
#            Those bands are the **reference's** own five draws (tasks Y and Z).  The port's five
#            draws are a second sample of the same chaotic quantity, not a subset of the first, and
#            task H4 measured that they are wider on three sets: pot_A at seed 2, pot_B at seeds 2
#            and 4, pot_H at seed 2.  Gating one five-draw sample with another would fail a run for
#            a draw rather than for a change.  What the band line is for is the comparison itself:
#            a step that moves a set out of the reference's spread, or back into it, has to say so.
BANDS = {
    "terracotta": dict(acc=None, prec=None, pure=True,
                       note="R §13's decision row, checked from report.json"),
    "pot_A": dict(acc=(0.875, 1.0), prec=(1.0, 1.0), pure=True),
    "pot_B": dict(acc=(0.889, 1.0), prec=(1.0, 1.0), pure=True),
    "pot_C": dict(acc=(0.50, 0.75), prec=(0.500, 0.667), pure=True),
    "pot_G": dict(acc=(0.0, 0.0), prec=None, pure=True, zero_acc=True, wrong_pose_only=True,
                  note="at most two joins used, every one a wrong-pose join on an adjacent pair"),
    "pot_H": dict(acc=(0.273, 0.364), prec=(0.333, 0.500), pure=True),
    "synthetic_20": dict(acc=(0.85, 0.95), prec=(1.0, 1.0), pure=True),
    # D §10.3, decision 2026-09-08: the one mixed development set is roadmap item 4's baseline,
    # not a phase gate.  Its numbers are reported and never fail the run.
    "mixed_ABG": dict(acc=None, prec=None, pure=False,
                      note="roadmap item 4's baseline (D §10.3), not a gate"),
}

EPS = 5e-3  # the bands above are quoted to three digits


def sh(cmd, cwd=ROOT):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return time.perf_counter() - t0, p.returncode, p.stdout


def terracotta_gate(rep, tr):
    """R §13's terracotta row, read off report.json/transforms.json.

    joins used exactly {021-094, 094-104}; 007 unplaced; both pen 0; tight of both >= 0.27;
    the two seams within 20 % of 20.3 t and 10.7 t.
    """
    def num(n):  # FY234021_reduced -> "021"
        d = "".join(c for c in n if c.isdigit())
        return d[-3:] if len(d) >= 3 else d

    used = [(num(j["a"]), num(j["b"]), j) for j in rep.get("joins_used", [])]
    pairs = {tuple(sorted((a, b))) for a, b, _ in used}
    fail = []
    if pairs != {("021", "094"), ("094", "104")}:
        fail.append("joins used %s, expected {021-094, 094-104}" % sorted(pairs))
    placed = {num(n) for g in tr["groups"] if len(g) > 1 for n in g}
    if "007" in placed:
        fail.append("007 is placed")
    seams = {}
    for a, b, j in used:
        key = tuple(sorted((a, b)))
        seams[key] = j.get("seam")
        if (j.get("pen") or 0.0) != 0.0:
            fail.append("pen %.4g on %s-%s" % (j["pen"], a, b))
        for side in ("tightA", "tightB"):
            if (j.get(side) or 0.0) < 0.27:
                fail.append("%s %.3f < 0.27 on %s-%s" % (side, j[side] or 0.0, a, b))
    for key, want in ((("021", "094"), 20.3), (("094", "104"), 10.7)):
        got = seams.get(key)
        if got is None:
            continue
        if abs(got - want) > 0.20 * want:
            fail.append("seam %s = %.2f t, outside 20 %% of %.1f t" % ("-".join(key), got, want))
    detail = "; ".join("seam %s %.2f t, tight %.3f/%.3f" %
                       ("-".join(k), seams[k],
                        min(j.get("tightA", 0.0) for a, b, j in used if tuple(sorted((a, b))) == k),
                        max(j.get("tightB", 0.0) for a, b, j in used if tuple(sorted((a, b))) == k))
                       for k in sorted(seams))
    return (not fail), fail, detail


def score(name, gt_dir, work, cen_cache):
    """evaluate.py over one finished run, or the terracotta's decision check."""
    with open(os.path.join(work, "transforms.json")) as f:
        tr = json.load(f)
    with open(os.path.join(work, "report.json")) as f:
        rep = json.load(f)
    row = dict(joins_used=len(rep.get("joins_used", [])),
               groups=len(tr["groups"]),
               largest=max((len(g) for g in tr["groups"]), default=0))
    if gt_dir is None:
        ok, fail, detail = terracotta_gate(rep, tr)
        row.update(scored=False, decision_ok=ok, decision_fail=fail, decision=detail)
        return row, None
    with open(os.path.join(ROOT, gt_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    if gt_dir not in cen_cache:
        cen_cache[gt_dir] = ev_mod.centroids(os.path.join(ROOT, gt_dir), sorted(tr["fragments"]))
    ev = ev_mod.evaluate(tr, rep, gt, 5.0, 0.5, 10, cen_cache[gt_dir], "centroid")
    j = ev["joins"]
    row.update(scored=True,
               frag_acc=ev["fragment_accuracy"]["fraction"],
               precision=j["precision"], recall=j["recall"],
               correct=j["correct"], wrong_pose=j["wrong_pose"],
               non_adjacent=j["non_adjacent"], cross_object=j["cross_object"],
               unscorable=j["unscorable"], gt_pairs=j["gt_adjacent_pairs"],
               purity=ev["groups"]["overall_purity"])
    return row, ev


def verdict(name, rows):
    """R §13 over one set's seeds: (gate, [gate reasons], band, [band notes])."""
    b = BANDS[name]
    gate_bad, band_bad = [], []

    if not rows[0]["scored"]:                       # the terracotta's decision row
        for r in rows:
            if not r["decision_ok"]:
                gate_bad.append("seed %d: %s" % (r["seed"], "; ".join(r["decision_fail"])))
        detail = "; ".join("seed %d: %s" % (r["seed"], r["decision"]) for r in rows)
        return (("FAIL" if gate_bad else "pass"),
                gate_bad or ["R §13's two joins at every seed."], "n/a", [detail])

    accs = [r["frag_acc"] for r in rows]
    precs = [r["precision"] for r in rows]

    if b["pure"]:                                   # R §13's last row, the prohibition
        for r in rows:
            if r["cross_object"]:
                gate_bad.append("seed %d: %d cross-object joins" % (r["seed"], r["cross_object"]))
            if r["purity"] is not None and r["purity"] < 1.0 - EPS:
                gate_bad.append("seed %d: group purity %.3f" % (r["seed"], r["purity"]))
    if b.get("zero_acc"):
        for r in rows:
            if r["frag_acc"] > EPS:
                gate_bad.append("seed %d: fragment accuracy %.3f, R §13 says 0 %%"
                                % (r["seed"], r["frag_acc"]))
    if b.get("wrong_pose_only"):
        for r in rows:
            if r["joins_used"] > 2:
                gate_bad.append("seed %d: %d joins used, at most two allowed"
                                % (r["seed"], r["joins_used"]))
            if r["joins_used"] != r["wrong_pose"]:
                gate_bad.append("seed %d: %d of %d used joins are not wrong-pose joins on "
                                "ground-truth-adjacent pairs"
                                % (r["seed"], r["joins_used"] - r["wrong_pose"], r["joins_used"]))

    for key, label, vals in (("acc", "fragment accuracy", accs), ("prec", "precision", precs)):
        if b[key] is None:
            continue
        lo, hi = b[key]
        out = [(r["seed"], v) for r, v in zip(rows, vals) if not (lo - EPS <= v <= hi + EPS)]
        if out:
            band_bad.append("%s %s, outside R §13's %.3f-%.3f." % (
                label, " and ".join("%.3f at seed %d" % (v, k) for k, v in out), lo, hi))

    checked = []
    if b["pure"]:
        checked.append("cross-object 0 and group purity 1.000")
    if b.get("zero_acc"):
        checked.append("fragment accuracy 0 %")
    if b.get("wrong_pose_only"):
        checked.append("at most two joins, every one a wrong-pose join on an adjacent pair")
    if gate_bad:
        gate = "FAIL"
    elif checked:
        gate = "pass"
    else:
        gate = "baseline"
    ok = ("%s at every seed." % "; ".join(checked).capitalize() if checked else
          "Not gated: %s. Cross-object %d-%d, purity %.3f-%.3f over the five seeds."
          % (b.get("note", "no prohibition stated"),
             min(r["cross_object"] for r in rows), max(r["cross_object"] for r in rows),
             min(r["purity"] for r in rows), max(r["purity"] for r in rows)))
    band = "n/a" if (b["acc"] is None and b["prec"] is None) else ("outside" if band_bad else "inside")
    span = "Accuracy %.3f-%.3f, precision %.3f-%.3f." % (min(accs), max(accs), min(precs), max(precs))
    return (gate, gate_bad or [ok], band, ([span] + band_bad) if band_bad else [span])


def fmt(x, spec="%.3f"):
    return "-" if x is None else spec % x


def markdown(meta, rows, verdicts):
    L = []
    L.append("# Quality gate — the Rust core on the development sets at seeds %s"
             % "-".join(str(s) for s in meta["seeds"]))
    L.append("")
    L.append("`tools/quality_gate.py`, %s, commit `%s`, backend `%s`, `--no-preview --no-meshes`, "
             "warm cache from the second seed of each set. Total wall **%.1f s** (%.1f min) for "
             "%d runs. Scores are `tools/evaluate.py` (5 deg / 0.5 t, translation at the fragment "
             "centroid); the terracotta has no staged ground truth and carries R §13's decision "
             "row instead."
             % (meta["generated"], meta["commit"], meta["backend"], meta["wall_total_s"],
                meta["wall_total_s"] / 60.0, len(rows)))
    L.append("")
    head = ("| set | seed | frag acc | precision | correct | wrong pose | non-adj | cross-obj | "
            "purity | joins | wall s |")
    L.append(head)
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["scored"]:
            L.append("| `%s` | %d | %.1f %% | %.3f | %d | %d | %d | %d | %s | %d | %.1f |" % (
                r["set"], r["seed"], 100 * r["frag_acc"], r["precision"], r["correct"],
                r["wrong_pose"], r["non_adjacent"], r["cross_object"],
                fmt(r["purity"]), r["joins_used"], r["wall_s"]))
        else:
            L.append("| `%s` | %d | — | — | — | — | — | — | — | %d | %.1f |" % (
                r["set"], r["seed"], r["joins_used"], r["wall_s"]))
    L.append("")
    L.append("R §13 over the five seeds of each set. **Gate** is what R §13 states as a "
             "prohibition -- its last row (cross-object joins 0, group purity 1.000) on the seven "
             "single-object collections, the terracotta's decision row, and pot_G's rule -- and a "
             "failure of one exits non-zero. **Band** compares the port's five draws of fragment "
             "accuracy and precision with the *reference's* five draws, which is a comparison and "
             "not a bound (task H4's note).")
    L.append("")
    for name, (g, why, band, span) in verdicts.items():
        L.append("* **`%s` \u2014 gate %s, band %s.** %s %s" % (name, g, band,
                                                                " ".join(why), " ".join(span)))
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join("target", "release", "sherd-refit-rs"))
    ap.add_argument("--out", default=os.path.join("output", "quality"))
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sets", nargs="+", default=None, help="a subset of the eight names")
    ap.add_argument("--keep-work", action="store_true", help="do not delete each set's run tree")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render <out>/quality.{md,json} from the rows of a finished run, "
                         "without running anything")
    a = ap.parse_args(argv)

    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    sets = [s for s in SETS if a.sets is None or s[0] in a.sets]
    missing = set(a.sets or []) - {s[0] for s in SETS}
    if missing:
        ap.error("unknown set(s): %s" % ", ".join(sorted(missing)))

    rows, cen_cache, commit = [], {}, "unknown"
    if a.render_only:
        with open(os.path.join(out, "quality.json")) as f:
            prev = json.load(f)
        return finish(out, prev["meta"], prev["rows"],
                      [s[0] for s in sets if any(r["set"] == s[0] for r in prev["rows"])])

    t_all = time.perf_counter()
    for name, indir, gt_dir in sets:
        if not os.path.isdir(os.path.join(ROOT, indir)):
            print("skip %-13s (no %s)" % (name, indir))
            continue
        work = os.path.join(out, "work", name)
        for seed in a.seeds:
            cmd = [os.path.join(ROOT, a.bin), "run", indir, "--out", work,
                   "--backend", a.backend, "--no-preview", "--no-meshes", "--seed", str(seed)]
            wall, rc, log = sh(cmd)
            if rc != 0:
                print(log)
                raise SystemExit("run failed: %s seed %d (exit %d)" % (name, seed, rc))
            row, ev = score(name, gt_dir, work, cen_cache)
            row.update(set=name, seed=seed, wall_s=wall)
            rows.append(row)
            with open(os.path.join(work, "report.json")) as f:
                commit = json.load(f).get("engine", {}).get("commit", commit)
            if ev is not None:
                with open(os.path.join(out, "runs", "%s_seed%d.json" % (name, seed)), "w") as f:
                    json.dump(ev, f, indent=1)
            print("%-13s seed %d  %6.1f s  %s" % (
                name, seed, wall,
                "joins %d" % row["joins_used"] if not row["scored"] else
                "acc %5.1f %%  prec %.3f  correct %d  wrong %d  cross %d"
                % (100 * row["frag_acc"], row["precision"], row["correct"],
                   row["wrong_pose"], row["cross_object"])))
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    wall_total = time.perf_counter() - t_all
    meta = dict(generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                commit=commit, backend=a.backend, seeds=a.seeds, binary=a.bin,
                wall_total_s=wall_total, runs=len(rows))
    return finish(out, meta, rows, [s[0] for s in sets])


def finish(out, meta, rows, names):
    verdicts = {}
    for name in names:
        rs = [r for r in rows if r["set"] == name]
        if rs:
            verdicts[name] = verdict(name, rs)
    md = markdown(meta, rows, verdicts)
    with open(os.path.join(out, "quality.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out, "quality.json"), "w") as f:
        json.dump(dict(meta=meta, rows=rows,
                       verdicts={k: dict(gate=g, gate_detail=why, band=b, band_detail=span)
                                 for k, (g, why, b, span) in verdicts.items()}),
                  f, indent=1)
    print()
    print(md)
    failed = [k for k, (g, _, _, _) in verdicts.items() if g == "FAIL"]
    outside = [k for k, (_, _, b, _) in verdicts.items() if b == "outside"]
    print("total wall %.1f s (%.1f min); %s" % (
        meta["wall_total_s"], meta["wall_total_s"] / 60.0,
        "FAILED: " + ", ".join(failed) if failed else
        "every set holds R §13's prohibitions" +
        ("" if not outside else "; outside the reference's own band: " + ", ".join(outside))))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
