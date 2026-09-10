#!/usr/bin/env python3
"""Measure everything a confidence tier could be built on, before any threshold is chosen.

    python tools/measure_tiers.py [--stage runs|tiers|features|colour|rule-runs|rules|all]
                                  [--bin target/release/sherd-refit-rs]
                                  [--seeds 0 1 2 3 4] [--out output/measure]

This is roadmap step 7 (audit §D.1 and §D.2, D §12 row 7).  It has four halves
-- two of them from step 7 itself, one added by task S2 and one by task S3 --
and they answer four different questions.

**The tier table** (``--stage runs`` then ``tiers``).  The eight development
sets are run at seeds 0-4 -- the same forty runs ``tools/quality_gate.py``
makes, from the same binary with the same flags -- with ``--measure``, which
adds one pass after R §8's assembly and one file per run.  Every *accepted*
candidate of every run comes back with R §6's five scores, the margin to its
pair's second placement, the twelve one-ULP neighbours of its pose, two
independent redraws of R §3.5's samples, the slide along the seam and its
support count.  This script gives each of them ``tools/evaluate.py``'s class --
computed from the candidate's **own** pose, so a candidate R §8 never used still
has one -- and then asks, per quantity, how far apart the correct joins and the
false ones are (AUC), and which combination of thresholds reaches **zero false
joins over all forty runs** with the most correct joins left.

**The feature table** (``--stage features``).  ``segment --features`` computes
audit §D.2's per-fragment object features on pots A-H, ``mixed_ABG``,
``synthetic_20``, the two coloured ``synthetic_mix3`` collections and -- the
audit's one stated exception to the 27-fragment rule, because preprocessing is
linear in the fragment count and matches nothing -- ``synthetic_60``,
``synthetic_170`` and ``mixed_all``.  Per feature it reports the AUC of
same-object against different-object pairs, the share of pairs a ``k*MAD`` veto
would remove at k = 2 and 3, and how many true adjacent pairs that costs.  Task
S2 added the six split-colour channels to that list and, beside it, the
pair-level colour distances with an absolute sweep, because a Lab distance has a
scale of its own and ``k*MAD`` is not it.

**The colour table** (``--stage colour``, task S2).  The feature table above is
the *object* question -- over every fragment pair, whether two sherds are one
vessel.  This is the *join* question: it runs the development sets that carry
both colour and object ids at seeds 0-4 and reads the candidates R §6.5
accepted, counting how many of them cross an object and asking whether the
colour recorded on each -- ``evidence.colour``, the clay bodies' CIE76 distance
and the skins' histogram distance -- separates those from the rest.  That is the
population a confirmation rule would be chosen on.  ``--object-demote`` is
passed through, so what a shortlist costs is measured here rather than argued.

**The rule table** (``--stage rule-runs`` then ``rules``, task S3).  The three
halves above measure *quantities*; this one measures *rules*.  Task M1 fixed the
strict half of the confirmed tier and left the second arm open, shipping the
disjunction ``support >= 1 OR margin >= 2`` whose margin half almost never fires
because R §5.7 returns one placement for most pairs.  Task S3 added two
witnesses -- the **wide rival**, the pair's second placement read off R §5.6's
full list before ``keep`` truncated it, and the **re-search**, the pair's whole
search run again on another draw -- and this stage runs the nine development
sets at seeds 0-4 with both recorded and then evaluates every candidate rule
**offline** from the one dump per run.  Forty-five runs answer for the whole
table, and the run's own verdict does not constrain a single row of it.

Nothing here decides anything.  The thresholds this script *proposes* are the
argmax of a search it prints in full; the note is where they are chosen.

Why the terracotta is classified differently
--------------------------------------------
``input/test_fragments_1`` has no staged ground truth -- the museum's assembly
was never written down as poses -- so ``evaluate.py`` cannot see it and every
candidate of it would be ``unscorable``.  R §13 states the answer as a set of
*decisions* instead: the joins 021-094 and 094-104 and no others.  This script
reads that row literally: an accepted candidate on one of those two pairs is
``correct``, an accepted candidate on any other pair of that collection is a
false join.  It is the only set scored that way and the tables say so.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import evaluate as ev  # noqa: E402
import quality_gate as qg  # noqa: E402  (the frozen gate: its set list, unmodified)

# R §13's terracotta decision row, as the gate itself checks it.
TERRACOTTA_JOINS = {("021", "094"), ("094", "104")}

# Collections the feature pass runs on.  The first eight are one object each and
# are pooled into `pots_A_H` for the cross-object half of the AUC; the last four
# carry `object_of` of their own.
FEATURE_SETS = [
    ("pot_A", "input/sfspp/pot_A"), ("pot_B", "input/sfspp/pot_B"),
    ("pot_C", "input/sfspp/pot_C"), ("pot_D", "input/sfspp/pot_D"),
    ("pot_E", "input/sfspp/pot_E"), ("pot_F", "input/sfspp/pot_F"),
    ("pot_G", "input/sfspp/pot_G"), ("pot_H", "input/sfspp/pot_H"),
    ("terracotta", "input/test_fragments_1/fragments"),
    ("mixed_ABG", "input/sfspp/mixed_ABG"),
    ("synthetic_20", "input/synthetic_pingsdorf_20/fragments"),
    ("synthetic_60", "input/synthetic_pingsdorf_60/fragments"),
    ("synthetic_170", "input/synthetic_pingsdorf_170/fragments"),
    ("mixed_all", "input/sfspp/mixed_all"),
    # Task S2: the two coloured collections task S1 built, which are the only sets in the
    # benchmark that carry real photographic colour **and** object ids at the same time.
    ("synthetic_mix3_24", "input/synthetic_mix3_24/fragments"),
    ("synthetic_mix3_60", "input/synthetic_mix3_60/fragments"),
]
# Where each of those keeps its ground_truth.json (object ids and adjacency).
FEATURE_GT = {
    "synthetic_20": "input/synthetic_pingsdorf_20",
    "synthetic_60": "input/synthetic_pingsdorf_60",
    "synthetic_170": "input/synthetic_pingsdorf_170",
    "synthetic_mix3_24": "input/synthetic_mix3_24",
    "synthetic_mix3_60": "input/synthetic_mix3_60",
}

# The per-fragment features whose AUC is measured, and whether the fit that made
# them can refuse (a `None` row is dropped from that feature's table, not zeroed).
#
# The `shell_*` and `frac_*` rows are task S2's: the file's own vertex colours split by R §3.4's
# labels, so that the fabric a break shows is measured apart from the photograph a skin carries.
FEATURES = ["thick", "thick_mode", "shell_radius", "frac_rough", "axis_diameter",
            "axis_residual", "shell_rms", "axis_rms", "lab_L", "lab_a", "lab_b", "lab_spread_L",
            "shell_lab_L", "shell_lab_a", "shell_lab_b", "shell_lab_mad_L",
            "frac_lab_L", "frac_lab_a", "frac_lab_b", "frac_lab_mad_L"]

# Task S2's PAIR-level colour distances: one number per fragment pair rather than the difference
# of two per-fragment scalars, which is what a join-level rule would actually read.
PAIR_DISTANCES = [
    ("frac_delta_e", "CIE76 between the two fracture faces' mean Lab"),
    ("shell_delta_e", "CIE76 between the two shell faces' mean Lab"),
    ("frac_hist", "total variation between the two fracture Lab histograms"),
    ("shell_hist", "total variation between the two shell Lab histograms"),
]

# Absolute thresholds the pair distances are swept at, in their own units.
PAIR_LIMITS = {
    "frac_delta_e": [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0],
    "shell_delta_e": [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0],
    "frac_hist": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "shell_hist": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

# scale-pairs §4.3's own gate, to reproduce its numbers with these fits: keep a pair whose
# thicknesses are within 2x and whose shell radii are within 3x.
RATIO_GATE = [("thick", 2.0), ("shell_radius", 3.0)]

# The candidate quantities of audit §D.1, and the direction in which "more" means
# "more likely a true join".  The AUC below is stated in that direction, so 0.5 is
# no separation and 1.0 is perfect separation the way the tier would read it.
QUANTITIES = [
    ("tight", +1, "R §6.1 tight-contact fraction"),
    ("gap", -1, "R §6.1 median fracture gap, in t"),
    ("gap_over_limit", -1, "gap as a fraction of the pair's own limit"),
    ("contact", +1, "R §6.1 contact area, in t^2"),
    ("seam", +1, "R §6.2 shared seam, in t"),
    ("cont", -1, "R §6.3 shell step across the seam, in t"),
    ("cont_n", +1, "R §6.3 shell-normal agreement"),
    ("pen", -1, "R §6.4 penetrating surface fraction"),
    ("pen_depth", -1, "R §6.4 deepest excursion, in t"),
    ("score", +1, "R §5.7 ranking key, seam * tight"),
    ("brk", +1, "R §5.4 stage-1 re-score of the pose"),
    ("margin", +1, "score over the second placement's score"),
    ("margin_strict", +1, "the same, with 'no second placement' read as a failure and not as "
                          "an infinite margin"),
    ("determined_deg", -1, "worst rotation over twelve one-ULP neighbours, deg"),
    ("determined_t", -1, "the same in translation, in t"),
    ("slide_t", -1, "how far the pose stays away after a 0.5 t push, in t"),
    ("support", +1, "independent accepted joins agreeing with the placement"),
    ("placements", -1, "distinct placements the pair's kept list makes"),
    ("resample_tight_min", +1, "smallest tight over the three draws"),
    ("resample_gap_max", -1, "largest gap over the three draws, in t"),
    ("resample_accept", +1, "how many of the three draws R §6.5 accepts"),
]

FALSE = ("wrong_pose", "non_adjacent", "cross_object")


def sh(cmd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return time.perf_counter() - t0, p.returncode, p.stdout


# ----------------------------------------------------------------------------- runs
def stage_runs(a, out):
    """The forty runs of the quality gate again, with --measure."""
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    sets = [s for s in qg.SETS if a.sets is None or s[0] in a.sets]
    total = 0.0
    for name, indir, _gt in sets:
        if not os.path.isdir(os.path.join(ROOT, indir)):
            print("skip %-13s (no %s)" % (name, indir))
            continue
        work = os.path.join(out, "work", name)
        for seed in a.seeds:
            dump = os.path.join(out, "runs", "%s_seed%d.json" % (name, seed))
            # `--tiers off`: this is step 7's measurement of every candidate R §6.5 *accepts*,
            # and the tier of step 8 is what it exists to choose. Leaving the tier on would make
            # R §8 assemble from the confirmed joins, which changes `used` and the support graph
            # this table is measured over.
            cmd = [os.path.join(ROOT, a.bin), "run", indir, "--out", work,
                   "--backend", a.backend, "--no-preview", "--no-meshes", "--tiers", "off",
                   "--seed", str(seed), "--measure", dump]
            wall, rc, log = sh(cmd)
            total += wall
            if rc != 0:
                print(log)
                raise SystemExit("run failed: %s seed %d (exit %d)" % (name, seed, rc))
            shutil.copyfile(os.path.join(work, "transforms.json"),
                            os.path.join(out, "runs", "%s_seed%d.transforms.json" % (name, seed)))
            shutil.copyfile(os.path.join(work, "report.json"),
                            os.path.join(out, "runs", "%s_seed%d.report.json" % (name, seed)))
            n = len(json.load(open(dump))["rows"])
            print("%-13s seed %d  %6.1f s  %3d accepted-candidate rows" % (name, seed, wall, n))
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    print("forty runs in %.1f s (%.1f min)" % (total, total / 60.0))


# ------------------------------------------------------------------- classification
def digits(name):
    d = "".join(c for c in name if c.isdigit())
    return d[-3:] if len(d) >= 3 else d


def classify_rows(name, gt_dir, dump, cen_cache):
    """evaluate.py's class for every accepted candidate, from its own pose."""
    rows = dump["rows"]
    thickness = dump["thickness"]
    if gt_dir is None:                       # the terracotta: R §13's decision row
        for r in rows:
            key = tuple(sorted((digits(r["a"]), digits(r["b"]))))
            r["verdict"] = "correct" if key in TERRACOTTA_JOINS else "non_adjacent"
            r["rot_deg"] = None
            r["trans_t"] = None
        return rows, len(TERRACOTTA_JOINS)

    with open(os.path.join(ROOT, gt_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    if gt_dir not in cen_cache:
        cen_cache[gt_dir] = ev.centroids(os.path.join(ROOT, gt_dir), sorted(dump["names"]))
    cen = cen_cache[gt_dir]
    object_of = gt.get("object_of", {})
    unknown = set(gt.get("unknown", []))
    gt_poses = {n: np.asarray(v["matrix"]) for n, v in gt["fragments"].items()}
    adjacency = {tuple(sorted(p)) for p in gt.get("adjacency", [])}
    present = set(dump["names"])
    gt_pairs = {p for p in adjacency if p[0] in present and p[1] in present}

    for r in rows:
        a, b = r["a"], r["b"]
        M_est = np.asarray(r["transform"])
        key = tuple(sorted((a, b)))
        if object_of.get(a, "?") != object_of.get(b, "?"):
            r.update(verdict="cross_object", rot_deg=None, trans_t=None)
            continue
        if a in unknown or b in unknown or a not in gt_poses or b not in gt_poses:
            r.update(verdict="unscorable", rot_deg=None, trans_t=None)
            continue
        M_gt = ev.rel(gt_poses[key[0]], gt_poses[key[1]])
        M = M_est if (a, b) == key else np.linalg.inv(M_est)
        deg, d_cen, d_org = ev.pose_error(M, M_gt, cen.get(key[1]))
        d = d_cen if cen else d_org
        if key not in adjacency:
            r.update(verdict="non_adjacent", rot_deg=deg, trans_t=d / thickness)
            continue
        ok = deg <= 5.0 and d <= 0.5 * thickness
        r.update(verdict="correct" if ok else "wrong_pose",
                 rot_deg=deg, trans_t=d / thickness)
    return rows, len(gt_pairs)


def derive(r):
    """The quantities that are functions of what the dump carries."""
    s = r["scores"]
    r.update(s)
    r["gap_over_limit"] = s["gap"] / s["gap_limit"] if s["gap_limit"] else float("inf")
    draws = [s] + r["resamples"]
    r["resample_tight_min"] = min(d["tight"] for d in draws)
    r["resample_gap_max"] = max(d["gap"] for d in draws)
    r["resample_accept"] = sum(1 for d in draws if d["accepted"])
    r["has_rival"] = 0 if r["margin"] is None else 1
    # No second placement in the pair's kept list is an *unbounded* margin, not a
    # missing one: there was nothing for this candidate to be ahead of.
    if r["margin"] is None:
        r["margin"] = float("inf")
    # The other reading of the same number, and the one the measurement prefers: a pair whose kept
    # list holds one placement has produced **no evidence** that its placement is the distinguished
    # one, so it fails the margin test rather than passing it perfectly. `rival_score` of zero is
    # the same case — a second placement exists but scores nothing, so there is nothing to beat.
    r["margin_strict"] = (r["score"] / r["rival_score"]
                          if r.get("rival_score") else float("-inf"))
    return r


# --------------------------------------------------------------------------- stats
def auc(pos, neg):
    """P(a random positive ranks above a random negative), ties at one half."""
    if not pos or not neg:
        return None
    n = 0.0
    for p in pos:
        for q in neg:
            n += 1.0 if p > q else (0.5 if p == q else 0.0)
    return n / (len(pos) * len(neg))


def finite(values):
    """Values a fit or a median may use: no `None`, no NaN, no infinity."""
    return [v for v in values if v is not None and math.isfinite(v)]


def defined(values):
    """Values an *order* may use: infinity is kept, because an unbounded margin is the best
    margin there is and dropping those rows would bias the AUC towards the pairs that had a
    second placement."""
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def quantiles(values, qs=(0.0, 0.05, 0.5, 0.95, 1.0)):
    v = sorted(defined(values))
    if not v:
        return [None] * len(qs)
    return [v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))] for q in qs]


def fmt(x, spec="%.3f"):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return spec % x


# ------------------------------------------------------------------- tier search
# The two tests a tier may take as a **disjunction** rather than as conjuncts, when its
# definition carries `or_support`. They are the two independent ways a placement can be
# distinguished — a second path through the collection agrees with it, or the pair's own search
# found another placement and this one beat it — and audit §D.1 offers no reason they should both
# have to hold.
DISJUNCTS = ("support", "margin_strict")


def passes(r, th):
    """One candidate against one threshold set (the readable form; the search uses bitsets)."""
    either = th.get("or_support")
    for key, field, sign in TESTS:
        if either and key in DISJUNCTS:
            continue
        limit = th.get(key)
        if limit is None:
            continue
        v = r.get(field)
        if v is None:
            return False
        if sign > 0 and v < limit:
            return False
        if sign < 0 and v > limit:
            return False
    if either and not any(th.get(key) is not None and r.get(field) is not None
                          and r[field] >= th[key] for key, field, _s in TESTS
                          if key in DISJUNCTS):
        return False
    return True


# (threshold key, the row's field, +1 when the threshold is a floor and -1 when it is a ceiling)
TESTS = [("min_tight", "tight", +1), ("max_gap_t", "gap", -1), ("min_seam", "seam", +1),
         ("min_cont_n", "cont_n", +1), ("max_pen", "pen", -1), ("margin", "margin", +1),
         ("margin_strict", "margin_strict", +1), ("slide", "slide_t", -1),
         ("determined", "determined_deg", -1), ("resample", "resample_accept", +1),
         ("support", "support", +1)]


def tally(rows, th):
    """(confirmed correct, confirmed false, the false ones themselves)."""
    ok = bad = 0
    which = []
    for r in rows:
        if not passes(r, th):
            continue
        if r["verdict"] == "correct":
            ok += 1
        elif r["verdict"] in FALSE:
            bad += 1
            which.append(r)
    return ok, bad, which


def base_thresholds():
    return dict({key: None for key, _f, _s in TESTS}, or_support=False)


def LADDER(audit):
    """The tiers the note compares, in the order it argues them.

    Every one of them is either something a document already proposes (R §6.5 as it ships, the
    audit's own candidate numbers) or one step of the argument the measurement forces: what the
    five scores can do alone, what the probes add, and what the support count of audit §D.2 adds
    on top. Nothing here was searched for; the search runs separately and its answer is quoted
    beside these.
    """
    strict = dict(audit)
    return [
        ("R §6.5 as it ships", dict(base_thresholds())),
        ("audit §D.1's candidates", strict),
        ("… + slide ≤ 0.1 t", dict(strict, slide=0.1)),
        ("… + all three draws accepted", dict(strict, slide=0.1, resample=3)),
        ("gap ≤ 0.0064 t alone (the search's answer)", dict(base_thresholds(), max_gap_t=0.0064)),
        ("gap ≤ 0.006 t alone", dict(base_thresholds(), max_gap_t=0.006)),
        ("gap ≤ 0.005 t alone", dict(base_thresholds(), max_gap_t=0.005)),
        ("support ≥ 1 alone", dict(base_thresholds(), support=1)),
        ("gap ≤ 0.015 t ∧ support ≥ 1", dict(base_thresholds(), max_gap_t=0.015, support=1)),
        ("audit's ∧ gap ≤ 0.015 t ∧ slide ≤ 0.1 t ∧ support ≥ 1",
         dict(strict, max_gap_t=0.015, slide=0.1, support=1)),
        ("audit's ∧ slide ∧ m ≥ 2 over a scoring rival (no support)",
         dict(strict, max_gap_t=0.015, slide=0.1, margin=None, margin_strict=2.0)),
        ("**chosen**: audit's ∧ gap ≤ 0.015 t ∧ slide ≤ 0.1 t ∧ (support ≥ 1 ∨ m ≥ 2)",
         dict(strict, max_gap_t=0.015, slide=0.1, margin=None, margin_strict=2.0, support=1,
              or_support=True)),
        ("… with m ≥ 3", dict(strict, max_gap_t=0.015, slide=0.1, margin=None,
                              margin_strict=3.0, support=1, or_support=True)),
        ("… ∧ all three draws accepted",
         dict(strict, max_gap_t=0.015, slide=0.1, margin=None, margin_strict=2.0, support=1,
              or_support=True, resample=3)),
    ]


def search(rows, grid):
    """Every combination of the grid, best confirmed-correct count with zero false.

    The rows are turned into **bitsets** first -- one integer per (test, value) whose bit `i` is
    set when row `i` passes that one test -- so a combination is nine `&` of Python integers and
    two `bit_count()`s rather than a loop over the rows. The full grid is a quarter of a million
    combinations and this is what makes it a few seconds instead of several minutes; the answer is
    `tally`'s, test for test.
    """
    keys = [k for k, _f, _s in TESTS if k in grid]
    all_bits = (1 << len(rows)) - 1
    pos = sum(1 << i for i, r in enumerate(rows) if r["verdict"] == "correct")
    neg = sum(1 << i for i, r in enumerate(rows) if r["verdict"] in FALSE)
    field = {k: (f, s) for k, f, s in TESTS}
    masks = {}
    for key in keys:
        f, sign = field[key]
        for v in grid[key]:
            if v is None:
                masks[(key, None)] = all_bits
                continue
            bits = 0
            for i, r in enumerate(rows):
                x = r.get(f)
                if x is None:
                    continue
                if (sign > 0 and x >= v) or (sign < 0 and x <= v):
                    bits |= 1 << i
            masks[(key, v)] = bits

    best, trace = None, []
    combo = [None] * len(keys)

    def walk(i, bits):
        nonlocal best
        if bits == 0:
            return                                   # nothing survives; no child can add a row
        if i == len(keys):
            ok, bad = (bits & pos).bit_count(), (bits & neg).bit_count()
            th = dict(zip(keys, combo))
            trace.append((th, ok, bad))
            if bad == 0 and (best is None or ok > best[1]):
                best = (th, ok)
            return
        for v in grid[keys[i]]:
            combo[i] = v
            walk(i + 1, bits & masks[(keys[i], v)])
        combo[i] = None

    walk(0, all_bits)
    return best, trace


def worst_rejection(rows, th):
    """How comfortably the *worst* false join is rejected by the tier as a whole.

    A tier is a conjunction, so what matters is not each test's own distance to the nearest false
    join but, for every false join, the **largest** relative distance by which some active test
    throws it out — and then the smallest of those over the false joins. A tier whose worst false
    join sits 0.5 % outside one boundary and inside every other is calibrated onto these forty
    runs; one whose worst false join is 20 % outside something has a threshold rather than a fit.
    Returned as `(the relative margin, the test that does the rejecting, the row)`.
    """
    either = th.get("or_support")
    worst = None
    for r in rows:
        if r["verdict"] not in FALSE:
            continue
        best = None
        if either:
            # Both arms have to fail, and the tier would let the row in if *either* flipped, so
            # the comfort of the disjunction is the smaller of the two shortfalls.
            arms = []
            for key, field, _s in TESTS:
                if key not in DISJUNCTS or th.get(key) is None:
                    continue
                limit, v = th[key], r.get(field)
                if v is None or v == float("-inf"):
                    arms.append(float("inf"))
                else:
                    arms.append((limit - v) / (abs(limit) or 1.0))
            if arms:
                best = (min(arms), "support-or-margin")
        for key, field, sign in TESTS:
            limit = th.get(key)
            if limit is None or (either and key in DISJUNCTS):
                continue
            v = r.get(field)
            if v is None:
                best = (float("inf"), key)      # the probe could not answer: rejected outright
                break
            gap = (limit - v) if sign > 0 else (v - limit)
            if gap <= 0:
                continue
            scale = abs(limit) if limit else 1.0
            rel = gap / scale
            if best is None or rel > best[0]:
                best = (rel, key)
        if best is None:
            return (0.0, "nothing", r)          # this false join survives the tier
        if worst is None or best[0] < worst[0]:
            worst = (best[0], best[1], r)
    return worst


def margin_to_boundary(rows, th):
    """How far the nearest false join is from the boundary, per gated quantity."""
    out = {}
    false_rows = [r for r in rows if r["verdict"] in FALSE]
    for key, field, sign in TESTS:
        limit = th.get(key)
        if limit is None or (isinstance(limit, float) and math.isinf(limit)):
            continue
        near = None
        for r in false_rows:
            v = r.get(field)
            if v is None or not math.isfinite(float(v)):
                continue
            gap = (limit - v) if sign > 0 else (v - limit)   # > 0 means it fails this test
            if gap <= 0:
                continue                                     # this test does not reject it
            near = gap if near is None else min(near, gap)
        out[key] = near
    return out


# ---------------------------------------------------------------------- tier stage
def stage_tiers(a, out):
    sets = [s for s in qg.SETS if a.sets is None or s[0] in a.sets]
    cen_cache = {}
    rows, per_run = [], []
    for name, _indir, gt_dir in sets:
        for seed in a.seeds:
            path = os.path.join(out, "runs", "%s_seed%d.json" % (name, seed))
            if not os.path.exists(path):
                print("skip %s seed %d (no dump)" % (name, seed))
                continue
            with open(path) as f:
                dump = json.load(f)
            got, gt_pairs = classify_rows(name, gt_dir, dump, cen_cache)
            for r in got:
                derive(r)
                r["set"], r["seed"] = name, seed
            best = [r for r in got if r["best_of_pair"]]
            rows.extend(best)
            buckets = collections.Counter(r["verdict"] for r in best)
            per_run.append(dict(set=name, seed=seed, gt_pairs=gt_pairs,
                                accepted_rows=len(got), pairs=len(best),
                                used=len(dump["used"]), **{k: buckets[k] for k in
                                ("correct", "wrong_pose", "non_adjacent",
                                 "cross_object", "unscorable")}))

    scorable = [r for r in rows if r["verdict"] != "unscorable"]
    pos = [r for r in scorable if r["verdict"] == "correct"]
    neg = [r for r in scorable if r["verdict"] in FALSE]
    print("%d accepted candidates, %d of them the best of their pair, %d correct, %d false, "
          "%d unscorable" % (sum(p["accepted_rows"] for p in per_run), len(rows),
                             len(pos), len(neg), len(rows) - len(scorable)))

    separation = []
    for key, sign, what in QUANTITIES:
        p = [sign * v for v in defined(r.get(key) for r in pos)]
        q = [sign * v for v in defined(r.get(key) for r in neg)]
        with_rival = lambda rows: [sign * r[key] for r in rows
                                   if r.get("rival_score") and r.get(key) is not None
                                   and not (isinstance(r[key], float) and math.isnan(r[key]))]
        separation.append(dict(key=key, what=what, sign=sign, auc=auc(p, q),
                               auc_with_rival=auc(with_rival(pos), with_rival(neg)),
                               inf_correct=sum(1 for v in p if math.isinf(v)),
                               inf_false=sum(1 for v in q if math.isinf(v)),
                               correct=quantiles([r.get(key) for r in pos]),
                               false=quantiles([r.get(key) for r in neg]),
                               n_correct=len(p), n_false=len(q)))

    grid = dict(
        min_tight=[None, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        max_gap_t=[None, 0.03, 0.02, 0.015, 0.01, 0.008, 0.0064, 0.006, 0.005],
        min_seam=[None, 3.0, 5.0, 8.0, 12.0, 20.0],
        min_cont_n=[None, 0.85, 0.90, 0.95],
        max_pen=[None, 0.0],
        margin=[None, 1.5, 2.0, 3.0],
        margin_strict=[None, 1.5, 2.0, 3.0],
        slide=[None, 0.5, 0.2, 0.1, 0.05, 0.01],
        determined=[None, 1e-9, 1e-11],
        resample=[None, 3],
        support=[None, 1],
    )
    best, trace = search(scorable, grid)
    audit = dict(base_thresholds(), min_tight=0.35, max_gap_t=0.02, min_seam=5.0,
                 min_cont_n=0.9, max_pen=0.0, margin=2.0)
    audit_ok, audit_bad, audit_which = tally(scorable, audit)
    shipped = dict(base_thresholds())      # R §6.5 as it ships: everything accepted
    ship_ok, ship_bad, _ = tally(scorable, shipped)

    ladder = []
    for label, th in LADDER(audit):
        ok, bad, _ = tally(scorable, th)
        per = collections.Counter(r["set"] for r in scorable
                                  if passes(r, th) and r["verdict"] == "correct")
        rejection = worst_rejection(scorable, th)
        ladder.append(dict(
            label=label,
            thresholds={k: v for k, v in th.items() if v is not None and v is not False},
            correct=ok, false=bad, per_set=dict(per),
            worst_rejection=None if rejection is None else rejection[0],
            worst_rejection_test=None if rejection is None else rejection[1],
            worst_rejection_row=None if rejection is None else
            dict(set=rejection[2]["set"], seed=rejection[2]["seed"],
                 a=rejection[2]["a"], b=rejection[2]["b"], verdict=rejection[2]["verdict"])))

    chosen = next(t for label, t in LADDER(audit) if label.startswith("**chosen**"))
    chosen_rows = []
    for p in per_run:
        rows_here = [r for r in scorable if r["set"] == p["set"] and r["seed"] == p["seed"]]
        kept = [r for r in rows_here if passes(r, chosen)]
        chosen_rows.append(dict(
            set=p["set"], seed=p["seed"], gt_pairs=p["gt_pairs"], accepted=len(rows_here),
            accepted_correct=sum(1 for r in rows_here if r["verdict"] == "correct"),
            confirmed=len(kept),
            confirmed_correct=sum(1 for r in kept if r["verdict"] == "correct"),
            confirmed_false=sum(1 for r in kept if r["verdict"] in FALSE)))

    result = dict(
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        seeds=a.seeds, per_run=per_run, separation=separation, ladder=ladder,
        chosen_thresholds={k: v for k, v in chosen.items() if v is not None and v is not False},
        chosen_per_run=chosen_rows,
        gt_pairs=sum(p["gt_pairs"] for p in per_run),
        shipped=dict(correct=ship_ok, false=ship_bad),
        audit=dict(thresholds=audit, correct=audit_ok, false=audit_bad,
                   which=[dict(set=r["set"], seed=r["seed"], a=r["a"], b=r["b"],
                               verdict=r["verdict"]) for r in audit_which],
                   boundary=margin_to_boundary(scorable, audit)),
        chosen=None,
    )
    if best is not None:
        th, ok = best
        _, bad, _ = tally(scorable, th)
        result["chosen"] = dict(thresholds=th, correct=ok, false=bad,
                                boundary=margin_to_boundary(scorable, th))
    # Every zero-false combination, best first: the note quotes the top of this list.
    zero = sorted([t for t in trace if t[2] == 0], key=lambda t: -t[1])
    result["zero_false_top"] = [dict(thresholds=t[0], correct=t[1]) for t in zero[:20]]
    # `len(trace)` and not the product of the grid: the bitset walk prunes a subtree the moment
    # nothing survives it, and a combination that confirms nothing cannot be the best one.
    result["grid_size"] = len(trace)

    with open(os.path.join(out, "tiers.json"), "w") as f:
        json.dump(result, f, indent=1)
    md = render_tiers(result, rows)
    with open(os.path.join(out, "tiers.md"), "w") as f:
        f.write(md)
    print(md)


def render_tiers(res, rows):
    L = ["# Step 7, part 1 — every accepted candidate of the forty runs", "",
         "`tools/measure_tiers.py`, %s, seeds %s. One row per **pair** per run: the best accepted "
         "candidate of the pair, which is the one R §8's `best_per_pair` would consider. The class "
         "is `tools/evaluate.py`'s, computed from the candidate's **own** pose, so a candidate the "
         "assembly never used still has one; the terracotta has no staged ground truth and is "
         "classified by R §13's decision row instead (021-094 and 094-104 correct, any other "
         "accepted pair false)."
         % (res["generated"], "-".join(str(s) for s in res["seeds"])), ""]
    L.append("| set | seed | pairs accepted | correct | wrong pose | non-adj | cross-obj | "
             "unscorable | used joins | GT pairs |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in res["per_run"]:
        L.append("| `%s` | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (
            p["set"], p["seed"], p["pairs"], p["correct"], p["wrong_pose"], p["non_adjacent"],
            p["cross_object"], p["unscorable"], p["used"], p["gt_pairs"]))
    L += ["", "## Separation, quantity by quantity", "",
          "AUC is stated in the direction the tier would read the quantity (the sign column), so "
          "0.500 is no separation and 1.000 is a threshold that splits the two classes outright. "
          "The five numbers are min / p5 / median / p95 / max.", "",
          "| quantity | dir | AUC | AUC on the pairs with a second placement | "
          "correct: min/p5/p50/p95/max | false: min/p5/p50/p95/max |",
          "|---|:--:|---:|---:|---|---|"]
    for s in sorted(res["separation"], key=lambda s: -(s["auc"] or 0)):
        L.append("| `%s` — %s | %s | %s | %s | %s | %s |" % (
            s["key"], s["what"], "+" if s["sign"] > 0 else "-", fmt(s["auc"]),
            fmt(s["auc_with_rival"]),
            " / ".join(fmt(v, "%.4g") for v in s["correct"]),
            " / ".join(fmt(v, "%.4g") for v in s["false"])))
    L += ["", "## The candidate tiers, and what each one costs", "",
          "`correct` and `false` are over all forty runs, on the same rows as the table above. "
          "**Worst rejection** is the tier read as the conjunction it is: for every false join, "
          "the largest *relative* distance by which some active test throws it out, and then the "
          "smallest of those over the false joins. A tier that reaches zero false joins by sitting "
          "0.5 % outside one boundary is fitted to these forty runs; one whose worst false join is "
          "a fifth outside something has a threshold. `inf` means the worst false join is thrown "
          "out by a probe that could not answer for it at all.", "",
          "| tier | correct | false | worst rejection | by | per set |",
          "|---|---:|---:|---:|---|---|"]
    for row in res["ladder"]:
        margin = row["worst_rejection"]
        L.append("| %s | %d | %d | %s | %s | %s |" % (
            row["label"], row["correct"], row["false"],
            "n/a" if row["false"] else fmt(margin, "%.3g"),
            row["worst_rejection_test"] or "-",
            ", ".join("%s %d" % kv for kv in sorted(row["per_set"].items())) or "-"))
    L += ["", "### The chosen tier, run by run", "",
          "`confirmed` counts the pairs the tier keeps; `recall` is the confirmed **correct** ones "
          "over the collection's ground-truth adjacent pairs, which is the number a conservator "
          "would be handed. The terracotta's two joins are R §13's decision row rather than a "
          "staged adjacency.", "",
          "| set | seed | GT pairs | accepted | of them correct | confirmed | correct | false | "
          "recall |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["chosen_per_run"]:
        L.append("| `%s` | %d | %d | %d | %d | %d | %d | %d | %s |" % (
            r["set"], r["seed"], r["gt_pairs"], r["accepted"], r["accepted_correct"],
            r["confirmed"], r["confirmed_correct"], r["confirmed_false"],
            fmt(r["confirmed_correct"] / r["gt_pairs"] if r["gt_pairs"] else None)))
    L += ["", "## What the thresholds buy", "",
          "R §6.5 as it ships accepts **%d** correct and **%d** false joins over the forty runs."
          % (res["shipped"]["correct"], res["shipped"]["false"]), ""]
    a = res["audit"]
    L.append("The audit's own candidates (`min_tight` 0.35, `gap` 0.02 t, `seam` 5 t, `cont_n` 0.9,"
             " `pen` 0, `m` = 2) keep **%d** correct and **%d** false." % (a["correct"], a["false"]))
    if a["which"]:
        L.append("")
        L.append("The false joins that survive them: " + "; ".join(
            "`%s` seed %d %s-%s (%s)" % (w["set"], w["seed"], w["a"], w["b"], w["verdict"])
            for w in a["which"]))
    marg = next(s for s in res["separation"] if s["key"] == "margin")
    L += ["", "`margin` is unbounded — the pair's kept list holds one placement and there is "
          "nothing for the candidate to be ahead of — on %d of the %d correct joins and %d of the "
          "%d false ones." % (marg["inf_correct"], marg["n_correct"],
                              marg["inf_false"], marg["n_false"])]
    c = res["chosen"]
    if c:
        L += ["", "The best of the %d threshold combinations the search reached that leave **zero** "
              "false joins:" % res["grid_size"], "", "```", json.dumps(c["thresholds"], indent=1), "```", "",
              "It confirms **%d** of the %d correct joins." % (c["correct"], res["shipped"]["correct"]),
              "", "How far the nearest surviving false join sits from each boundary "
              "(the false-join margin; a blank means that test is not what rejects any of them):", "",
              "| test | limit | nearest false join is this far outside |", "|---|---:|---:|"]
        for key, gap in sorted(c["boundary"].items()):
            L.append("| `%s` | %s | %s |" % (key, fmt(c["thresholds"][key], "%.4g"),
                                             fmt(gap, "%.4g")))
        L += ["", "The twenty best zero-false combinations, so that the choice can be read as a "
              "plateau rather than a point:", "",
              "| correct | min_tight | gap t | seam t | cont_n | pen | margin | slide t | draws |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for z in res["zero_false_top"]:
            t = z["thresholds"]
            L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                z["correct"], fmt(t["min_tight"], "%.2f"), fmt(t["max_gap_t"], "%.4g"),
                fmt(t["min_seam"], "%.4g"), fmt(t["min_cont_n"], "%.2f"), fmt(t["max_pen"], "%.2g"),
                fmt(t["margin"], "%.2g") if t["margin"] else "-",
                fmt(t["slide"], "%.2g") if t["slide"] else "-",
                str(t["resample"]) if t["resample"] else "-"))
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ feature stage
def stage_features(a, out):
    os.makedirs(os.path.join(out, "features"), exist_ok=True)
    tables = {}
    for name, indir in FEATURE_SETS:
        if a.sets is not None and name not in a.sets:
            continue
        if not os.path.isdir(os.path.join(ROOT, indir)):
            print("skip %-13s (no %s)" % (name, indir))
            continue
        path = os.path.join(out, "features", "%s.json" % name)
        if not (a.reuse and os.path.exists(path)):
            work = os.path.join(out, "segment", name)
            cmd = [os.path.join(ROOT, a.bin), "segment", indir, "--out", work,
                   "--features", path] + (["--features-colour"] if a.colour else [])
            wall, rc, log = sh(cmd)
            if rc != 0:
                print(log)
                raise SystemExit("segment failed: %s (exit %d)" % (name, rc))
            print("%-14s %6.1f s  %s" % (name, wall, path))
            if not a.keep_work:
                shutil.rmtree(work, ignore_errors=True)
        with open(path) as f:
            tables[name] = json.load(f)
    result = analyse_features(tables)
    with open(os.path.join(out, "features.json"), "w") as f:
        json.dump(result, f, indent=1)
    md = render_features(result)
    with open(os.path.join(out, "features.md"), "w") as f:
        f.write(md)
    print(md)


def object_ids(name, table):
    """(object of each fragment, adjacency) for one feature collection."""
    if name.startswith("pot_"):
        return {f["name"]: name for f in table}, set()
    gt_dir = FEATURE_GT.get(name, dict(FEATURE_SETS)[name])
    path = os.path.join(ROOT, gt_dir, "ground_truth.json")
    if not os.path.exists(path):
        # The terracotta: four museum scans with no staged truth. It is here for its **colours**,
        # which are the only real vertex colours in the whole benchmark (audit §C.4), and it has
        # no object ids, so it contributes medians and a colour row and no AUC.
        return {}, set()
    with open(path) as f:
        gt = json.load(f)
    adjacency = {tuple(sorted(p)) for p in gt.get("adjacency", [])}
    return gt.get("object_of", {}), adjacency


def mad(values):
    v = sorted(values)
    if not v:
        return 0.0
    m = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    d = sorted(abs(x - m) for x in v)
    return d[len(d) // 2] if len(d) % 2 else 0.5 * (d[len(d) // 2 - 1] + d[len(d) // 2])


def flatten_colour(rows):
    """`lab_mean`, `lab_spread` and task S2's two `ColourStats` are arrays; the AUC table wants
    channels."""
    for r in rows:
        mean, spread = r.get("lab_mean"), r.get("lab_spread")
        for i, c in enumerate("Lab"):
            r["lab_%s" % c] = mean[i] if mean else None
            r["lab_spread_%s" % c] = spread[i] if spread else None
        for side, key in (("shell", "shell_colour"), ("frac", "frac_colour")):
            block = r.get(key)
            for i, c in enumerate("Lab"):
                r["%s_lab_%s" % (side, c)] = block["lab_mean"][i] if block else None
                r["%s_lab_mad_%s" % (side, c)] = block["lab_mad"][i] if block else None
    return rows


def delta_e76(a, b):
    """CIE76 between two Lab triples."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def hist_tv(a, b):
    """Total variation between two histograms, normalised by their own counts.

    The same arithmetic `ColourStats::hist_distance` ships, so the table and the binary cannot
    come to mean two different things by one column name."""
    na, nb = sum(a), sum(b)
    if not na or not nb or len(a) != len(b):
        return None
    return 0.5 * sum(abs(x / na - y / nb) for x, y in zip(a, b))


def pair_distance(key, a, b):
    """One of PAIR_DISTANCES for one pair of feature rows, or None where it cannot be answered."""
    side = "frac_colour" if key.startswith("frac") else "shell_colour"
    x, y = a.get(side), b.get(side)
    if not x or not y:
        return None
    if key.endswith("delta_e"):
        return delta_e76(x["lab_mean"], y["lab_mean"])
    return hist_tv(x["hist"], y["hist"])


def analyse_pairs(name, rows, object_of, adjacency):
    """Task S2's pair-level colour distances: AUC, a k*MAD veto and an absolute sweep.

    A collection with no staged object ids (the terracotta, and each single-object pot) is read as
    **one** object, so it contributes the within-object half of every distribution and no AUC --
    which is exactly what a control is for."""
    objects = object_of or {r["name"]: name for r in rows}
    out = {}
    for key, _ in PAIR_DISTANCES:
        have = [r for r in rows if r["name"] in objects]
        same, diff, all_pairs = [], [], []
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                a, b = have[i], have[j]
                d = pair_distance(key, a, b)
                if d is None:
                    continue
                (same if objects[a["name"]] == objects[b["name"]] else diff).append(d)
                all_pairs.append((tuple(sorted((a["name"], b["name"]))), d))
        if not all_pairs:
            out[key] = None
            continue
        adj = [p for p in all_pairs if p[0] in adjacency]
        spread = mad([d for _, d in all_pairs])
        row = dict(pairs=len(all_pairs), n_same=len(same), n_diff=len(diff),
                   adjacent=len(adj), mad=spread, auc=auc(diff, same),
                   same_q=quantiles(same), diff_q=quantiles(diff))
        for k in (2, 3):
            limit = k * spread
            row["k%d" % k] = dict(
                limit=limit,
                pairs_removed=sum(1 for _, d in all_pairs if d > limit) / len(all_pairs),
                adjacent_removed=(sum(1 for _, d in adj if d > limit) / len(adj)) if adj else None,
                adjacent_total=len(adj))
        sweep = []
        for limit in PAIR_LIMITS[key]:
            cut_diff = sum(1 for d in diff if d > limit)
            cut_same = sum(1 for d in same if d > limit)
            cut_adj = sum(1 for _, d in adj if d > limit)
            sweep.append(dict(limit=limit,
                              different_removed=(cut_diff / len(diff)) if diff else None,
                              same_removed=(cut_same / len(same)) if same else None,
                              adjacent_removed=(cut_adj / len(adj)) if adj else None))
        row["sweep"] = sweep
        out[key] = row
    return out


def ratio_gate(rows, object_of, adjacency):
    """scale-pairs §4.3's gate: what a 2x-thickness / 3x-radius rule keeps and what it costs."""
    have = [r for r in rows if r["name"] in object_of
            and all(r.get(k) is not None and r[k] > 0 for k, _ in RATIO_GATE)]
    kept = total = adj_kept = adj_total = same_cut = 0
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            a, b = have[i], have[j]
            ok = all(max(a[k], b[k]) / min(a[k], b[k]) <= limit for k, limit in RATIO_GATE)
            total += 1
            kept += ok
            if tuple(sorted((a["name"], b["name"]))) in adjacency:
                adj_total += 1
                adj_kept += ok
            if object_of[a["name"]] == object_of[b["name"]] and not ok:
                same_cut += 1
    return dict(pairs=total, kept=kept / total if total else None,
                adjacent=adj_total, adjacent_kept=adj_kept / adj_total if adj_total else None,
                same_object_cut=same_cut)


def analyse_one(rows, object_of, adjacency):
    """Per feature: AUC of |delta| between different-object and same-object pairs, and what a
    k*MAD veto on it would remove."""
    out = {}
    for key in FEATURES:
        have = [(r["name"], r[key]) for r in rows
                if r.get(key) is not None and math.isfinite(r[key])
                and r["name"] in object_of]
        if len(have) < 4:
            out[key] = None
            continue
        spread = mad([v for _, v in have])
        same, diff, all_pairs = [], [], []
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                (na, va), (nb, vb) = have[i], have[j]
                d = abs(va - vb)
                (same if object_of[na] == object_of[nb] else diff).append(d)
                all_pairs.append((tuple(sorted((na, nb))), d))
        row = dict(n=len(have), mad=spread, auc=auc(diff, same),
                   n_same=len(same), n_diff=len(diff))
        adj = [p for p in all_pairs if p[0] in adjacency]
        for k in (2, 3):
            limit = k * spread
            cut = sum(1 for _, d in all_pairs if d > limit)
            cut_adj = sum(1 for _, d in adj if d > limit)
            row["k%d" % k] = dict(limit=limit,
                                  pairs_removed=cut / len(all_pairs) if all_pairs else None,
                                  adjacent_removed=cut_adj / len(adj) if adj else None,
                                  adjacent_total=len(adj))
        out[key] = row
    return out


def analyse_features(tables):
    result = dict(collections={}, medians={}, colour={}, pairs={})
    pooled, pooled_objects = [], {}
    for name, rows in tables.items():
        object_of, adjacency = object_ids(name, flatten_colour(rows))
        result["collections"][name] = dict(
            ratio_gate=ratio_gate(rows, object_of, adjacency),
            fragments=len(rows), objects=len(set(object_of.get(r["name"], "?") for r in rows)),
            features=analyse_one(rows, object_of, adjacency))
        result["pairs"][name] = analyse_pairs(name, rows, object_of, adjacency)
        result["medians"][name] = {
            key: (sorted(v)[len(v) // 2] if (v := [r[key] for r in rows
                  if r.get(key) is not None and math.isfinite(r[key])]) else None)
            for key in FEATURES}
        result["colour"][name] = dict(
            with_colour=sum(1 for r in rows if r.get("lab_mean")),
            with_shell=sum(1 for r in rows if r.get("shell_colour")),
            with_fracture=sum(1 for r in rows if r.get("frac_colour")),
            distinct=sorted({r.get("colour_distinct", 0) for r in rows}),
            lab_mean=[round(x, 2) for x in rows[0]["lab_mean"]] if rows and rows[0].get("lab_mean") else None)
        result["collections"][name]["rims"] = sum(1 for r in rows if r.get("rim"))
        if name.startswith("pot_") and name != "pot_terracotta":
            pooled.extend(rows)
            for r in rows:
                pooled_objects[r["name"]] = name
    if pooled:
        result["collections"]["pots_A_H"] = dict(
            ratio_gate=ratio_gate(pooled, pooled_objects, set()),
            fragments=len(pooled), objects=len(set(pooled_objects.values())),
            rims=sum(1 for r in pooled if r.get("rim")),
            features=analyse_one(pooled, pooled_objects, set()))
    return result


def render_features(res):
    L = ["# Step 7, part 2 — audit §D.2's per-fragment object features", "",
         "`sherd-refit-rs segment --features`, preprocessing only. `pots_A_H` pools the eight "
         "single-object collections and treats each pot as an object, which is the only way those "
         "sets produce a different-object pair at all. AUC is `P(|Δf| of a different-object pair > "
         "|Δf| of a same-object pair)`: 0.500 is no separation, and audit §D.2's rule is that a "
         "feature may **veto** only above 0.800.", ""]
    L.append("| collection | fragments | objects | rim-flagged |")
    L.append("|---|---:|---:|---:|")
    for name, c in res["collections"].items():
        L.append("| `%s` | %d | %d | %d |" % (name, c["fragments"], c["objects"], c.get("rims", 0)))
    L += ["", "## Medians, feature by feature", "",
          "| collection | " + " | ".join("`%s`" % k for k in FEATURES) + " |",
          "|---|" + "---:|" * len(FEATURES)]
    for name, m in res["medians"].items():
        L.append("| `%s` | %s |" % (name, " | ".join(fmt(m[k], "%.4g") for k in FEATURES)))
    L += ["", "## Separation and what a `k·MAD` veto would cost", "",
          "| collection | feature | AUC | MAD | 2·MAD: pairs cut / adjacent cut | "
          "3·MAD: pairs cut / adjacent cut |", "|---|---|---:|---:|---|---|"]
    for name, c in res["collections"].items():
        for key in FEATURES:
            row = c["features"].get(key)
            if row is None:
                continue
            cells = []
            for k in (2, 3):
                v = row["k%d" % k]
                cells.append("%s / %s" % (fmt(v["pairs_removed"]), fmt(v["adjacent_removed"])))
            L.append("| `%s` | `%s` | %s | %s | %s | %s |" % (
                name, key, fmt(row["auc"]), fmt(row["mad"], "%.4g"), cells[0], cells[1]))
    L += ["", "## scale-pairs §4.3's own gate, with these fits", "",
          "Keep a pair whose wall thicknesses are within 2x **and** whose shell radii are within "
          "3x. §4.3 measured 66 % of `mixed_all`'s pairs kept for 6 % of its adjacent ones lost, "
          "on a *curvature* radius rather than this sphere fit.", "",
          "| collection | pairs | kept | adjacent pairs | adjacent kept | same-object pairs cut |",
          "|---|---:|---:|---:|---:|---:|"]
    for name, c in res["collections"].items():
        g = c.get("ratio_gate")
        if not g or not g["pairs"]:
            continue
        L.append("| `%s` | %d | %s | %d | %s | %d |" % (
            name, g["pairs"], fmt(g["kept"]), g["adjacent"], fmt(g["adjacent_kept"]),
            g["same_object_cut"]))
    L += ["", "## Colour", "",
          "A collection whose files carry no vertex colours reports **unavailable** and not a "
          "zero: every SfS++ set but the seven coloured fragments of `mixed_all` is such a set, "
          "and every colour row of theirs below is absent rather than neutral.", "",
          "| collection | fragments with vertex colours | with a shell colour | with a fracture "
          "colour | distinct RGB triples per fragment |",
          "|---|---:|---:|---:|---|"]
    for name, c in res["colour"].items():
        L.append("| `%s` | %s | %s | %s | %s |" % (
            name,
            c["with_colour"] or "unavailable",
            c.get("with_shell") or "unavailable",
            c.get("with_fracture") or "unavailable",
            ", ".join(str(d) for d in c["distinct"][:5])))
    L += ["", "## Task S2: the pair-level colour distances", "",
          "One number per fragment **pair** rather than the difference of two per-fragment "
          "scalars: `frac_delta_e` and `shell_delta_e` are CIE76 between the two sides' mean Lab, "
          "`frac_hist` and `shell_hist` the total variation between their 4x4x4 Lab histograms. "
          "AUC is `P(a different-object pair's distance > a same-object pair's)`. A collection "
          "with no staged object ids is read as one object and reports the same-object half "
          "alone.", "",
          "| collection | distance | pairs | same / diff | AUC | median same | median diff | "
          "2·MAD: pairs cut / adjacent cut | 3·MAD: pairs cut / adjacent cut |",
          "|---|---|---:|---|---:|---:|---:|---|---|"]
    for name, rows in res.get("pairs", {}).items():
        for key, _ in PAIR_DISTANCES:
            row = rows.get(key)
            if not row:
                continue
            cells = []
            for k in (2, 3):
                v = row["k%d" % k]
                cells.append("%s / %s" % (fmt(v["pairs_removed"]), fmt(v["adjacent_removed"])))
            L.append("| `%s` | `%s` | %d | %d / %d | %s | %s | %s | %s | %s |" % (
                name, key, row["pairs"], row["n_same"], row["n_diff"], fmt(row["auc"]),
                fmt(row["same_q"][2] if row["same_q"] else None, "%.4g"),
                fmt(row["diff_q"][2] if row["diff_q"] else None, "%.4g"),
                cells[0], cells[1]))
    L += ["", "### What an absolute threshold on those distances would remove", "",
          "Per collection and distance: the share of **different-object** pairs a veto at that "
          "limit removes (its gain), the share of **same-object** pairs it removes and the share "
          "of the ground truth's **adjacent** pairs it costs.", "",
          "A collection with no different-object pair (the terracotta, each single pot, the "
          "pingsdorf sets) is the **control**: its `same cut` is what the same limit would cost "
          "on a collection that is one vessel and nothing else.", "",
          "| collection | distance | limit | different cut | same cut | adjacent cut |",
          "|---|---|---:|---:|---:|---:|"]
    for name, rows in res.get("pairs", {}).items():
        for key, _ in PAIR_DISTANCES:
            row = rows.get(key)
            if not row:
                continue
            for step in row["sweep"]:
                L.append("| `%s` | `%s` | %.4g | %s | %s | %s |" % (
                    name, key, step["limit"], fmt(step["different_removed"]),
                    fmt(step["same_removed"]), fmt(step["adjacent_removed"])))
    L.append("")
    return "\n".join(L)


# Task S2's colour stage: the development sets whose files carry colour AND object ids.
COLOUR_SETS = [
    ("synthetic_mix3_24", "input/synthetic_mix3_24/fragments", "input/synthetic_mix3_24"),
    ("synthetic_20", "input/synthetic_pingsdorf_20/fragments", "input/synthetic_pingsdorf_20"),
]


def stage_colour(a, out):
    """What the colour evidence looks like on the candidates R §6.5 actually accepted.

    §2's tables are over *every* fragment pair, which is the object question.  This is the join
    question: of the candidates the matcher accepted, how many cross an object, and does the colour
    recorded on each of them separate those from the rest?  It is the population a confirmation
    rule would be chosen on, and it is measured here so that the rule can be chosen on it."""
    rows = []
    for name, indir, gt_dir in COLOUR_SETS:
        if a.sets is not None and name not in a.sets:
            continue
        if not os.path.isdir(os.path.join(ROOT, indir)):
            print("skip %-18s (no %s)" % (name, indir))
            continue
        with open(os.path.join(ROOT, gt_dir, "ground_truth.json")) as f:
            gt = json.load(f)
        object_of = gt.get("object_of", {})
        adjacency = {tuple(sorted(p)) for p in gt.get("adjacency", [])}
        work = os.path.join(out, "colour", name)
        for seed in a.seeds:
            cmd = [os.path.join(ROOT, a.bin), "run", indir, "--out", work,
                   "--backend", a.backend, "--no-preview", "--no-meshes", "--seed", str(seed)]
            if a.object_demote:
                cmd += ["--object-demote", a.object_demote]
            wall, rc, log = sh(cmd)
            if rc != 0:
                print(log)
                raise SystemExit("run failed: %s seed %d (exit %d)" % (name, seed, rc))
            with open(os.path.join(work, "report.json")) as f:
                report = json.load(f)
            rows.append(colour_row(name, seed, wall, report, object_of, adjacency))
            print("%-18s seed %d  %6.1f s  accepted %d (%d cross-object), confirmed %d (%d "
                  "cross-object), demoted %d" % (
                      name, seed, wall, rows[-1]["accepted"], rows[-1]["accepted_cross"],
                      rows[-1]["confirmed"], rows[-1]["confirmed_cross"], rows[-1]["demotions"]))
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    with open(os.path.join(out, "colour.json"), "w") as f:
        json.dump(rows, f, indent=1)
    md = render_colour(rows)
    with open(os.path.join(out, "colour.md"), "w") as f:
        f.write(md)
    print(md)


def colour_row(name, seed, wall, report, object_of, adjacency):
    """One run, read as the join question."""
    accepted = [c for c in report["candidates"] if c.get("accepted")]
    row = dict(set=name, seed=seed, wall_s=wall,
               candidates=len(report["candidates"]), accepted=len(accepted),
               demotions=len(report.get("objects", {}).get("demotions", [])),
               merges=report.get("objects", {}).get("merges", 0))
    pairs = {}
    for c in accepted:
        key = tuple(sorted((c["a"], c["b"])))
        best = pairs.get(key)
        if best is None or c["score"] > best["score"]:
            pairs[key] = c
    cross = [k for k in pairs if object_of.get(k[0]) != object_of.get(k[1])]
    confirmed = [k for k, c in pairs.items() if c.get("tier") == "confirmed"]
    row.update(accepted_pairs=len(pairs), accepted_cross=len(cross),
               confirmed=len(confirmed),
               confirmed_cross=sum(1 for k in confirmed
                                   if object_of.get(k[0]) != object_of.get(k[1])))
    for key in ("frac_delta_e", "shell_hist"):
        same, diff, adj = [], [], []
        for k, c in pairs.items():
            v = (c.get("evidence") or {}).get("colour", {}).get(key)
            if v is None:
                continue
            if object_of.get(k[0]) != object_of.get(k[1]):
                diff.append(v)
            else:
                same.append(v)
                if k in adjacency:
                    adj.append(v)
        row[key] = dict(n_same=len(same), n_cross=len(diff), n_adjacent=len(adj),
                        auc=auc(diff, same), same_q=quantiles(same), cross_q=quantiles(diff),
                        adjacent_q=quantiles(adj))
    return row


def render_colour(rows):
    L = ["# Task S2, part 3 — the colour evidence on the candidates R §6.5 accepted", "",
         "One row per run. `accepted` counts pairs (the best candidate of each), `cross-object` "
         "the ones whose two sherds belong to two vessels of the ground truth. AUC is "
         "`P(a cross-object accepted pair's distance > a same-object one's)` over the accepted "
         "pairs alone -- the population a confirmation rule would read.", "",
         "| set | seed | accepted pairs | cross-object | confirmed | cross-object confirmed | "
         "demoted | merges | frac dE AUC | shell hist AUC |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append("| `%s` | %d | %d | %d | %d | %d | %d | %d | %s | %s |" % (
            r["set"], r["seed"], r["accepted_pairs"], r["accepted_cross"], r["confirmed"],
            r["confirmed_cross"], r["demotions"], r["merges"],
            fmt(r["frac_delta_e"]["auc"]), fmt(r["shell_hist"]["auc"])))
    L += ["", "## The two distances over the accepted pairs, by class", "",
          "Quantiles are 0, 5, 50, 95, 100 %.", "",
          "| set | seed | distance | same-object (n) | quantiles | cross-object (n) | quantiles |",
          "|---|---:|---|---:|---|---:|---|"]

    def q(v):
        """The five quantiles, or `-` when the class this row describes is empty.

        ``quantiles`` answers an empty class with a list of ``None``, and on every collection this
        project can run the stage on the *cross-object* class is exactly that: R §6.5 accepts no
        pair that crosses an object (task S2 §5).  A dash is the honest cell; formatting it as a
        number is what raised ``TypeError`` before this was fixed."""
        return "-" if not v or v[0] is None else " / ".join(fmt(x, "%.3g") for x in v)

    for r in rows:
        for key in ("frac_delta_e", "shell_hist"):
            c = r[key]
            L.append("| `%s` | %d | `%s` | %d | %s | %d | %s |" % (
                r["set"], r["seed"], key, c["n_same"], q(c["same_q"]),
                c["n_cross"], q(c["cross_q"])))
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ task S3: the rule stage
# The population is the same one step 7 measured -- every candidate R §6.5 accepted, on all nine
# development sets at seeds 0-4 -- and the question is the one task M1 left open: **which second
# arm**.  M1 searched conjunctions of the five scores and the probes and found none that reaches
# zero false joins; what it shipped is a disjunction, `support >= 1 OR margin >= 2`, and the
# margin half of it almost never fires because R §5.7 returns one placement for most pairs.
#
# Task S3 adds two witnesses and this stage tabulates every rule they make possible:
#
#   * the **wide rival** -- the pair's second placement read off R §5.6's full list before
#     R §5.7's `keep` truncated it, and failing that the best stage-1 pose that is a second
#     placement, refined and scored by R §6.  It is what makes `margin` a test that can fire.
#   * the **re-search** -- the pair's whole R §5-§6 search run again on another draw, and whether
#     its best accepted candidate lands on this placement.
#
# The runs this stage makes are the quality gate's own forty-five, with `--tiers on` (so the
# rival is computed at all) and `--resample-seeds 2`.  Every rule below is then evaluated
# **offline** from the one dump per run, which is why forty-five runs answer for a whole table
# instead of forty-five runs per row.
S3_STRICT = dict(min_tight=0.35, max_gap_t=0.015, min_seam=5.0, min_cont_n=0.90,
                 max_pen=0.0, max_slide_t=0.1)


def stage_rule_runs(a, out):
    """The forty-five runs the rule table is chosen on: the tier on, two re-searches, --measure."""
    os.makedirs(os.path.join(out, "rules"), exist_ok=True)
    sets = [s for s in qg.SETS if a.sets is None or s[0] in a.sets]
    total = 0.0
    for name, indir, _gt in sets:
        if not os.path.isdir(os.path.join(ROOT, indir)):
            print("skip %-18s (no %s)" % (name, indir))
            continue
        work = os.path.join(out, "work", name)
        for seed in a.seeds:
            dump = os.path.join(out, "rules", "%s_seed%d.json" % (name, seed))
            if a.reuse and os.path.exists(dump):
                print("%-18s seed %d  reused" % (name, seed))
                continue
            # `--tiers on`, unlike `--stage runs`: the wide rival is evidence the tier reads, so a
            # run with the pass off neither computes nor carries it.  The tier set is the shipped
            # one -- the run behaves exactly as the gate's does -- and every rule below is decided
            # offline from this dump, so the run's own verdict does not constrain the table.
            cmd = [os.path.join(ROOT, a.bin), "run", indir, "--out", work,
                   "--backend", a.backend, "--no-preview", "--no-meshes", "--tiers", "on",
                   "--resample-seeds", str(a.resample_seeds),
                   "--seed", str(seed), "--measure", dump]
            wall, rc, log = sh(cmd)
            total += wall
            if rc != 0:
                print(log)
                raise SystemExit("run failed: %s seed %d (exit %d)" % (name, seed, rc))
            rep = json.load(open(os.path.join(work, "report.json")))
            shutil.copyfile(os.path.join(work, "report.json"),
                            os.path.join(out, "rules", "%s_seed%d.report.json" % (name, seed)))
            d = json.load(open(dump))
            d["timings"] = rep.get("timings", {})
            json.dump(d, open(dump, "w"))
            print("%-18s seed %d  %6.1f s  %3d rows  (match %.1f s, tiers %.1f s)"
                  % (name, seed, wall, len(d["rows"]),
                     rep.get("timings", {}).get("matching", 0.0),
                     rep.get("timings", {}).get("tiers", 0.0)))
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    print("forty-five runs in %.1f s (%.1f min)" % (total, total / 60.0))


def s3_derive(r):
    """Everything a rule below reads, as one flat row."""
    s = r["scores"]
    r["tight"], r["gap"], r["seam"] = s["tight"], s["gap"], s["seam"]
    r["cont_n"], r["pen"], r["score"] = s["cont_n"], s["pen"], s["score"]
    r["pen_unavailable"] = s.get("pen_unavailable", False)
    # `Thresholds::refusals`, line for line: a test whose evidence is missing **fails**.
    r["strict"] = (s["tight"] >= S3_STRICT["min_tight"]
                   and s["gap"] <= S3_STRICT["max_gap_t"]
                   and s["seam"] >= S3_STRICT["min_seam"]
                   and s["cont_n"] >= S3_STRICT["min_cont_n"]
                   and not s.get("pen_unavailable", False)
                   and s["pen"] <= S3_STRICT["max_pen"]
                   and r.get("slide_t") is not None
                   and r["slide_t"] <= S3_STRICT["max_slide_t"])
    # The two margins, both read strictly: no second placement is a failed test and not an
    # unbounded one (M1 §5.3, and the reading `Probes::margin` itself has).
    r["m_kept"] = r["margin"] if r.get("margin") is not None else 0.0
    r["m_wide"] = r["wide_margin"] if r.get("wide_margin") is not None else 0.0
    r["has_kept"] = r.get("margin") is not None
    r["has_wide"] = r.get("wide_margin") is not None
    r["wide_src"] = (r.get("wide_rival") or {}).get("source")
    r["wide_moved"] = (r.get("wide_rival") or {}).get("moved_t")
    r["wide_acc"] = (r.get("wide_rival") or {}).get("accepted")
    res = r.get("research") or []
    r["res_n"] = len(res)
    r["res_agree"] = sum(1 for x in res if x.get("agrees"))
    r["res_any"] = sum(1 for x in res if x.get("accepted_any"))
    return r


def s3_join_colour(rows, report_path):
    """Task S2's `evidence.colour` for each row, from the run's own `report.json`.

    The `--measure` dump carries what a *tier* reads and colour is not in it: `ColourAgreement`
    is computed from the two fragments' feature tables and travels with the report.  The join is
    by `index`, which is the row's own index into the run's candidate list, so it is exact.
    """
    if not os.path.exists(report_path):
        for r in rows:
            r["colour"] = None
        return
    cands = json.load(open(report_path))["candidates"]
    for r in rows:
        c = cands[r["index"]] if r["index"] < len(cands) else {}
        r["colour"] = (c.get("evidence") or {}).get("colour")


def s3_colour_ok(r, limit):
    """The **veto** reading of colour, with the third state S2 §10 says it needs.

    Three answers and not two.  *Has nothing to say*: the pair carries no colour at all (every
    SfS++ collection), or both sherds are one flat tone so that their agreement is an artefact of
    having no information -- `pot_H`'s eleven fragments report `frac_delta_e` 5e-12 and `shell_hist`
    exactly 0 for that reason.  Either way the rule abstains and the row passes.  *Agrees*: the
    clay bodies are within `limit` of each other in CIE76.  *Disagrees*: they are not, and the row
    fails.
    """
    c = r.get("colour")
    if not c:
        return True
    d = c.get("frac_delta_e")
    if d is None:
        return True
    if not s3_colour_speaks(r):
        return True
    return d <= limit


def s3_colour_speaks(r):
    """Whether this pair's colour is information rather than the absence of it."""
    c = r.get("colour") or {}
    d, h = c.get("frac_delta_e"), c.get("shell_hist")
    return (d is not None and d > 1e-6) or (h is not None and h > 1e-9)


def s3_rules(margins, researches):
    """Every rule the table compares, as (name, note, predicate on a derived row).

    The strict half is M1's and never moves; what varies is the **second arm**, which is the only
    thing task M1 left undecided and the only thing task S3 measured new evidence for.
    """
    rules = [
        ("shipped", "support >= 1 OR margin(kept) >= 2",
         lambda r: r["support"] >= 1 or r["m_kept"] >= 2.0),
        ("support-only", "support >= 1",
         lambda r: r["support"] >= 1),
        ("margin-kept-only", "margin(kept) >= 2",
         lambda r: r["m_kept"] >= 2.0),
    ]
    for m in margins:
        rules.append(("margin-wide>=%g" % m, "margin(wide) >= %g" % m,
                      lambda r, m=m: r["m_wide"] >= m))
    for k in researches:
        rules.append(("research>=%d" % k, "re-search agreed on %d of 2 draws" % k,
                      lambda r, k=k: r["res_agree"] >= k))
    for k in researches:
        for m in margins:
            rules.append(("research>=%d & margin-wide>=%g" % (k, m),
                          "re-search %d AND margin(wide) >= %g" % (k, m),
                          lambda r, k=k, m=m: r["res_agree"] >= k and r["m_wide"] >= m))
    for k in researches:
        for m in margins:
            rules.append(("support | (research>=%d & margin-wide>=%g)" % (k, m),
                          "support >= 1 OR (re-search %d AND margin(wide) >= %g)" % (k, m),
                          lambda r, k=k, m=m: r["support"] >= 1
                          or (r["res_agree"] >= k and r["m_wide"] >= m)))
    for k in researches:
        rules.append(("support | research>=%d" % k, "support >= 1 OR re-search %d" % k,
                      lambda r, k=k: r["support"] >= 1 or r["res_agree"] >= k))
    for m in margins:
        rules.append(("support | margin-wide>=%g" % m, "support >= 1 OR margin(wide) >= %g" % m,
                      lambda r, m=m: r["support"] >= 1 or r["m_wide"] >= m))
    # Task S3 item 3's last row: the same rules with colour as an **extra requirement**, read as a
    # veto with the third state S2 §10 asks for (`s3_colour_ok`).  Colour can only ever remove
    # confirmed joins here -- the nine development sets produce no false ones for it to remove --
    # so what these rows measure is a price, and the table is where that price is stated.
    base = list(rules)
    for dE in (5.0, 7.5):
        for rname, note, arm in base:
            if rname in ("shipped", "support-only") or rname.startswith("support | (research"):
                rules.append(("%s & colour<=%g" % (rname, dE),
                              "%s AND the clay bodies agree within %g dE" % (note, dE),
                              lambda r, arm=arm, dE=dE: arm(r) and s3_colour_ok(r, dE)))
    return rules


def s3_pairs(rows, arm):
    """`sherd_core::tiers::representatives` under one rule: which pairs it confirms.

    A pair is confirmed when **any** of its candidates clears the rule, and the pair's verdict is
    that candidate's -- the best-scoring one among those that clear it.  That is exactly what the
    Rust does: `representatives` ranks a pair's candidates by (band, -score), so the confirmed one
    with the highest score represents the pair.
    """
    best = {}
    for r in rows:
        if not (r["strict"] and arm(r)):
            continue
        key = (r["a"], r["b"])
        if key not in best or r["score"] > best[key]["score"]:
            best[key] = r
    return best


FALSE_S3 = ("wrong_pose", "non_adjacent", "cross_object")


def stage_rules(a, out):
    """The evidence table: every rule, on all forty-five runs, correct and false per set."""
    sets = [s for s in qg.SETS if a.sets is None or s[0] in a.sets]
    cen_cache = {}
    runs = []                                  # (set, seed, rows, ground-truth pair count)
    for name, _indir, gt in sets:
        for seed in a.seeds:
            path = os.path.join(out, "rules", "%s_seed%d.json" % (name, seed))
            if not os.path.exists(path):
                print("skip %-18s seed %d (no dump)" % (name, seed))
                continue
            dump = json.load(open(path))
            rows, gt_pairs = classify_rows(name, gt, dump, cen_cache)
            s3_join_colour(rows, os.path.join(out, "rules",
                                              "%s_seed%d.report.json" % (name, seed)))
            runs.append(dict(set=name, seed=seed, rows=[s3_derive(r) for r in rows],
                             gt_pairs=gt_pairs, timings=dump.get("timings", {})))
    if not runs:
        raise SystemExit("no dumps under %s/rules -- run --stage rule-runs first" % out)

    margins = [1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
    researches = [1, 2]
    rules = s3_rules(margins, researches)
    table = []
    for rname, note, arm in rules:
        per_set, correct, false, bad = {}, 0, 0, []
        for run in runs:
            conf = s3_pairs(run["rows"], arm)
            ok = sum(1 for r in conf.values() if r["verdict"] == "correct")
            no = sum(1 for r in conf.values() if r["verdict"] in FALSE_S3)
            row = per_set.setdefault(run["set"], dict(correct=0, false=0, gt=0, runs=0))
            row["correct"] += ok
            row["false"] += no
            row["gt"] += run["gt_pairs"]
            row["runs"] += 1
            correct += ok
            false += no
            bad += [(run["set"], run["seed"], r["a"], r["b"], r["verdict"])
                    for r in conf.values() if r["verdict"] in FALSE_S3]
        table.append(dict(rule=rname, note=note, correct=correct, false=false,
                          per_set=per_set, bad=bad))
    result = dict(runs=[dict(set=r["set"], seed=r["seed"], rows=len(r["rows"]),
                             gt_pairs=r["gt_pairs"], timings=r["timings"]) for r in runs],
                  table=table,
                  evidence=s3_evidence(runs))
    with open(os.path.join(out, "rules.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(out, "rules.md"), "w") as f:
        f.write(render_rules(result, runs, sets))
    print(open(os.path.join(out, "rules.md")).read())


def s3_evidence(runs):
    """What the two new witnesses say, before any rule reads them."""
    acc = [r for run in runs for r in run["rows"]]
    strict = [r for r in acc if r["strict"]]
    out = dict(accepted=len(acc), strict=len(strict))
    for label, pool in (("accepted", acc), ("clears the strict set", strict)):
        pos = [r for r in pool if r["verdict"] == "correct"]
        neg = [r for r in pool if r["verdict"] in FALSE_S3]
        out[label] = dict(
            n=len(pool), correct=len(pos), false=len(neg),
            kept_rival=sum(1 for r in pool if r["has_kept"]),
            wide_rival=sum(1 for r in pool if r["has_wide"]),
            wide_from_stage2=sum(1 for r in pool if r["wide_src"] == "stage2"),
            wide_from_stage1=sum(1 for r in pool if r["wide_src"] == "stage1"),
            gained=sum(1 for r in pool if r["has_wide"] and not r["has_kept"]),
            auc_kept=auc([r["m_kept"] for r in pos], [r["m_kept"] for r in neg]),
            auc_wide=auc([r["m_wide"] for r in pos], [r["m_wide"] for r in neg]),
            auc_research=auc([r["res_agree"] for r in pos], [r["res_agree"] for r in neg]),
            auc_support=auc([r["support"] for r in pos], [r["support"] for r in neg]),
            q_wide_correct=quantiles([r["m_wide"] for r in pos]),
            q_wide_false=quantiles([r["m_wide"] for r in neg]),
            q_kept_correct=quantiles([r["m_kept"] for r in pos]),
            q_kept_false=quantiles([r["m_kept"] for r in neg]),
            research_agree_correct=sum(1 for r in pos if r["res_agree"] >= 1),
            research_agree_false=sum(1 for r in neg if r["res_agree"] >= 1),
            research_both_correct=sum(1 for r in pos if r["res_agree"] >= 2),
            research_both_false=sum(1 for r in neg if r["res_agree"] >= 2),
        )
    return out


def render_rules(res, runs, sets):
    names = [s[0] for s in sets]
    L = ["# Task S3 -- the evidence table: which second arm confirms a join", "",
         "Every accepted candidate of the nine development sets at seeds 0-4, with the two "
         "witnesses task S3 added beside the ones task M1 measured.  The strict half of the tier "
         "is M1's and is held fixed at `tight >= 0.35, gap <= 0.015 t, seam >= 5 t, cont_n >= "
         "0.90, pen <= 0 (and measurable), slide <= 0.1 t`; what every row below varies is the "
         "**second arm**.  A pair counts once, under its best-scoring candidate that clears the "
         "rule -- `sherd_core::tiers::representatives`, in Python.", "",
         "## 1. What the new evidence is, before a rule reads it", ""]
    e = res["evidence"]
    L += ["| population | rows | correct | false | kept rival | wide rival | of those stage-2 | "
          "stage-1 | **rivals gained** |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("accepted", "clears the strict set"):
        d = e[label]
        L.append("| %s | %d | %d | %d | %d | %d | %d | %d | **%d** |"
                 % (label, d["n"], d["correct"], d["false"], d["kept_rival"], d["wide_rival"],
                    d["wide_from_stage2"], d["wide_from_stage1"], d["gained"]))
    L += ["", "*kept rival* is how many candidates have a second placement in R §5.7's returned "
          "list, which is the only rival that existed before this step; *wide rival* is how many "
          "have one once the full R §5.6 list and the stage-1 fallback are read; *rivals gained* "
          "is the difference -- the candidates whose margin arm could not fire at all and now "
          "can.", ""]
    L += ["| population | AUC margin(kept) | AUC margin(wide) | AUC re-search | AUC support |",
          "|---|---:|---:|---:|---:|"]
    for label in ("accepted", "clears the strict set"):
        d = e[label]
        L.append("| %s | %s | %s | %s | %s |"
                 % (label, fmt(d["auc_kept"]), fmt(d["auc_wide"]),
                    fmt(d["auc_research"]), fmt(d["auc_support"])))
    L += ["", "| population | margin(wide) correct min/5%/med/95%/max | margin(wide) false |",
          "|---|---|---|"]
    for label in ("accepted", "clears the strict set"):
        d = e[label]
        L.append("| %s | %s | %s |"
                 % (label, "/".join(fmt(x, "%.2f") for x in d["q_wide_correct"]),
                    "/".join(fmt(x, "%.2f") for x in d["q_wide_false"])))
    L += ["", "| population | re-search agreed >= 1 (correct/false) | agreed on both (correct/false) |",
          "|---|---|---|"]
    for label in ("accepted", "clears the strict set"):
        d = e[label]
        L.append("| %s | %d / %d | %d / %d |"
                 % (label, d["research_agree_correct"], d["research_agree_false"],
                    d["research_both_correct"], d["research_both_false"]))

    L += ["", "## 2. The rule table", "",
          "`correct` and `false` are **confirmed pairs** summed over the forty-five runs; a rule "
          "with a non-zero `false` is out, whatever its recall.  The per-set columns are correct "
          "confirmed joins on that set over its five seeds.", ""]
    head = "| rule | correct | **false** | " + " | ".join(names) + " |"
    L += [head, "|---|---:|---:|" + "---:|" * len(names)]
    ship = next(t for t in res["table"] if t["rule"] == "shipped")
    for t in res["table"]:
        cells = []
        for n in names:
            d = t["per_set"].get(n)
            if not d:
                cells.append("-")
                continue
            s = str(d["correct"])
            if d["false"]:
                s += " (**%d false**)" % d["false"]
            cells.append(s)
        L.append("| `%s` | %d | %s | %s |"
                 % (t["rule"], t["correct"],
                    "**%d**" % t["false"] if t["false"] else "0", " | ".join(cells)))
    gt_total = sum(d["gt"] for d in ship["per_set"].values())
    L += ["", "## 3. Every rule with zero false confirmed joins, by recall", "",
          "`recall` is confirmed **correct** pairs over ground-truth adjacent pairs, summed over "
          "the forty-five runs (%d). Each per-set cell is that rule's correct confirmed joins on "
          "the set over its five seeds, and in brackets what it gains or loses against the "
          "shipped rule -- the cost of the rule in correct joins, per set, which is the number "
          "the brief asks for." % gt_total, ""]
    clean = sorted([t for t in res["table"] if t["false"] == 0],
                   key=lambda t: -t["correct"])
    L += ["| rank | rule | correct | recall | vs shipped | " + " | ".join(names) + " |",
          "|---:|---|---:|---:|---:|" + "---:|" * len(names)]
    for i, t in enumerate(clean, 1):
        cells = []
        for n in names:
            d, s0 = t["per_set"].get(n), ship["per_set"].get(n)
            if not d:
                cells.append("-")
                continue
            delta = d["correct"] - (s0["correct"] if s0 else 0)
            cells.append("%d (%+d)" % (d["correct"], delta) if delta else str(d["correct"]))
        recall = "%.1f%%" % (100.0 * t["correct"] / gt_total) if gt_total else "-"
        L.append("| %d | `%s` | %d | %s | %+d | %s |"
                 % (i, t["rule"], t["correct"], recall,
                    t["correct"] - ship["correct"], " | ".join(cells)))
    L += ["", "## 4. Where every rule that fails does so", ""]
    L += ["| rule | false confirmed joins |", "|---|---|"]
    for t in res["table"]:
        if not t["false"]:
            continue
        L.append("| `%s` | %s |" % (t["rule"], "; ".join(
            "%s s%d %s-%s %s" % b for b in t["bad"][:12])
            + (" ..." if len(t["bad"]) > 12 else "")))
    L += ["", "## 5. What the two new witnesses cost in time", "",
          "| set | seeds | matching (s) | tiers (s) | tiers / matching |",
          "|---|---:|---:|---:|---:|"]
    per = {}
    for r in runs:
        p = per.setdefault(r["set"], [0.0, 0.0, 0])
        p[0] += r["timings"].get("matching", 0.0)
        p[1] += r["timings"].get("tiers", 0.0)
        p[2] += 1
    for n in names:
        if n not in per:
            continue
        m, t, k = per[n]
        L.append("| %s | %d | %.1f | %.1f | %s |"
                 % (n, k, m, t, fmt(t / m if m else None, "%.2f")))
    L += ["", "Written by `tools/measure_tiers.py --stage rules` on %s."
          % datetime.date.today().isoformat(), ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["runs", "tiers", "features", "colour", "rule-runs", "rules", "all"])
    ap.add_argument("--bin", default=os.path.join("target", "release", "sherd-refit-rs"))
    ap.add_argument("--out", default=os.path.join("output", "measure"))
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("--colour", action="store_true",
                    help="read every source scan again for its vertex colours")
    ap.add_argument("--reuse", action="store_true",
                    help="keep feature tables that are already on disk")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--resample-seeds", type=int, default=2,
                    help="task S3: independent re-searches per accepted pair in --stage rule-runs")
    ap.add_argument("--object-demote", default=None,
                    help="comma-separated feature list passed to `run --object-demote` in the "
                         "colour stage, to measure what a shortlist costs")
    a = ap.parse_args(argv)
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    if a.stage in ("runs", "all"):
        stage_runs(a, out)
    if a.stage in ("tiers", "all"):
        stage_tiers(a, out)
    if a.stage in ("features", "all"):
        stage_features(a, out)
    if a.stage in ("colour", "all"):
        stage_colour(a, out)
    if a.stage == "rule-runs":
        stage_rule_runs(a, out)
    if a.stage == "rules":
        stage_rules(a, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
