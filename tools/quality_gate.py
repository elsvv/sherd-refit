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

The confirmed tier
-----------------
Since roadmap item 3 (audit §D.1, task T1) a run also puts every candidate in
one of three bands, and R §8 assembles from the **confirmed** band alone.  The
gate therefore scores two different things from one run.  The columns on the
left are what they always were -- ``evaluate.py`` over the joins R §8 used,
which under ``--tiers on`` are confirmed joins.  The columns on the right are
the *tier itself*: every confirmed pair of the run, classified by
``evaluate.py``'s own rule applied to that candidate's **own** pose rather than
to the assembly's, so a confirmed join R §8 refused for penetration or for
merging two groups is still scored.  ``zero false joins in the confirmed tier``
is a prohibition on that set and it is gated on all eight collections,
``mixed_ABG`` included.

Audit §E's two roadmap rows, stated on the tier they are about
---------------------------------------------------------------
Task G (V8-D2, V8-D3) restated the two rows the audit wrote before the tier
existed, so that each is a check a run can fail rather than a sentence in a note:

* **step 10, ``mixed_ABG``** -- the *prohibition* is cross-object joins 0 and
  group purity 1.000 **in the confirmed tier**, gated at every seed.  The
  *recall* half, "correct joins >= 12", is gated over the **confirmed and
  probable bands together**: the audit's 12 counted the joins R §6.5's assembly
  used before there was a tier, the tier finds no join R §6.5 did not accept, and
  the two upper bands together are both that same population and the list a
  conservator actually works from.  Correct **confirmed** joins are reported per
  seed and are deliberately not gated -- a number fixed at what the tier happens
  to reach today makes the tier unimprovable in either direction.
* **step 8, the terracotta** -- "the two joins confirmed" at seeds 0-4 is ten
  (seed, join) slots, and the gate asks for ten.  On this tree it reaches nine;
  the run says which slot is short and why it is a recall row rather than a
  prohibition.  See ``TERRACOTTA_CONFIRMED_SLOTS``.

``--tiers off`` runs the binary the way it ran before the tier existed and
prints the table this script has always printed; that is the mode in which
R §13's bands are a comparison, because R §13's own draws are the reference's
accepted-tier draws.  Under ``--tiers on`` the band line reads ``n/a``: an
assembly of confirmed joins is not the same quantity R §13 measured, and
comparing the two would report a difference the tier was built to make.

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

import numpy as np  # noqa: E402  (after sys.path)
import evaluate as ev_mod  # noqa: E402

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
    # D §10.3, decision 2026-09-08: under ``--tiers off`` the one mixed development set is roadmap
    # item 4's baseline and nothing it does fails the run.  Under ``--tiers on`` audit §E's step-10
    # row applies and is stated here on the tier it is about (task G, V8-D2):
    #
    #   pure_tiers   -- cross-object joins 0 and group purity 1.000, *in the confirmed tier*.  This
    #                   is the prohibition: separating three pots is what item 4 was built for, and
    #                   it is the half of the row that must never regress.
    #   recall_floor -- correct joins >= 12 over **confirmed and probable together**.  The audit's
    #                   12 is the count of joins R §6.5's own assembly *used* before the tier
    #                   existed (D §10.3's baseline row); the tier does not add joins, it sorts the
    #                   ones R §6.5 accepted into bands, so the quantity that can be held against
    #                   that baseline is the two upper bands together -- which is also the list a
    #                   conservator is handed.  Confirmed-correct alone is *reported* per seed and
    #                   never gated: fixing a number the tier's own strictness moves would make the
    #                   tier unimprovable.
    "mixed_ABG": dict(acc=None, prec=None, pure=False, pure_tiers=True, recall_floor=12,
                      note="roadmap item 4's baseline (D §10.3), not a gate"),
}

EPS = 5e-3  # the bands above are quoted to three digits

# A gate reason that is a *recall* row rather than a prohibition, marked so the run's last line can
# say which kind of failure it is.  A museum cares about the difference: a breached prohibition is a
# join the tool asserts and the pot denies; an unmet recall row is a true join the tool declined to
# assert, which is still on the conservator's probable list with the reason printed beside it.
RECALL = "recall row -- "

# R §13's terracotta decision row, as the pairs it names.  The museum's four scans have no
# staged ground truth, so a confirmed join on any other pair of that collection is a false one.
TERRACOTTA_JOINS = {("021", "094"), ("094", "104")}

# Audit §E's step-8 gate, "terracotta's two joins confirmed", over the five seeds: both joins at
# every seed is **ten** (seed, join) slots.  It is stated at ten and not at the number this tree
# reaches (task G, V8-D3).  The gate used to read nine, which is what M1 §5.6 measured -- at seed 4
# the pair 021-094 comes back from R §5.7 as a single placement, so the margin arm has nothing to
# beat, and a three-fragment chain gives the support arm no second path either.  A constant fitted
# to the observed answer cannot fail, and a gate that cannot fail is not a gate: it would record
# the brief's row as met while the row is unmet, and it could not tell nine-at-seed-4 from
# nine-at-some-other-seed.  So the number is the brief's, the run fails on it, and the failure is
# printed with the slots that are missing.  Task G measured what it would take to close it and
# found no rule that does not cost a false join -- see notes/2026-09-10-g-tiers-findings.md.
TERRACOTTA_CONFIRMED_SLOTS = 2 * 5


def sh(cmd, cwd=ROOT):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return time.perf_counter() - t0, p.returncode, p.stdout


def digits(name):
    """FY234021_reduced -> "021"."""
    d = "".join(c for c in name if c.isdigit())
    return d[-3:] if len(d) >= 3 else d


def representatives(cands):
    """The candidate that represents each pair: the best band, then the best score inside it.

    `sherd_core::tiers::representatives`, in Python and over `report.json`'s own list.  A pair's
    band is the strongest thing the tool will say about those two sherds.
    """
    rank = {"confirmed": 0, "probable": 1, "rejected": 2}
    best = {}
    for c in cands:
        key = (c["a"], c["b"])
        k = (rank.get(c.get("tier", "rejected"), 2), -c.get("score", 0.0))
        if key not in best or k < best[key][0]:
            best[key] = (k, c)
    return [v[1] for v in best.values()]


def tier_score(name, gt_dir, rep, tr, cen_cache):
    """Every confirmed and every probable pair of one run, classified by evaluate.py's rule.

    A candidate is scored on its **own** pose rather than on the assembly, so a confirmed join
    R §8 refused is scored here all the same -- which is the whole point of gating the tier rather
    than the placement.  Returns (confirmed counts, probable counts, the collection's adjacent
    pairs, the false confirmed ones themselves).

    The probable band is counted for one reason: audit §E's step-10 recall row is a count of
    *joins found*, and the tier does not find joins -- it sorts the ones R §6.5 accepted into
    bands.  Confirmed and probable together are what R §6.5 accepted, and they are what a
    conservator is handed (V8-D2).
    """
    cands = representatives(rep.get("candidates", []))
    bands = dict(confirmed=[c for c in cands if c.get("tier") == "confirmed"],
                 probable=[c for c in cands if c.get("tier") == "probable"])

    def empty(band):
        return dict(band=len(bands[band]), correct=0, wrong_pose=0, non_adjacent=0,
                    cross_object=0, unscorable=0)

    conf, prob = empty("confirmed"), empty("probable")
    bad = []

    if gt_dir is None:                                  # the terracotta: R §13's decision row
        for counts, band in ((conf, "confirmed"), (prob, "probable")):
            for c in bands[band]:
                key = tuple(sorted((digits(c["a"]), digits(c["b"]))))
                if key in TERRACOTTA_JOINS:
                    counts["correct"] += 1
                else:
                    counts["non_adjacent"] += 1
                    if band == "confirmed":
                        bad.append((c["a"], c["b"], "non_adjacent"))
        return conf, prob, len(TERRACOTTA_JOINS), bad

    with open(os.path.join(ROOT, gt_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    names = sorted(tr["fragments"])
    if gt_dir not in cen_cache:
        cen_cache[gt_dir] = ev_mod.centroids(os.path.join(ROOT, gt_dir), names)
    cen = cen_cache[gt_dir]
    thickness = float(tr["thickness"])
    object_of = gt.get("object_of", {})
    unknown = set(gt.get("unknown", []))
    gt_poses = {n: np.asarray(v["matrix"]) for n, v in gt["fragments"].items()}
    adjacency = {tuple(sorted(p)) for p in gt.get("adjacency", [])}
    present = set(names)
    gt_pairs = {p for p in adjacency if p[0] in present and p[1] in present}

    for counts, band in ((conf, "confirmed"), (prob, "probable")):
        for c in bands[band]:
            a, b = c["a"], c["b"]
            key = tuple(sorted((a, b)))
            if object_of.get(a, "?") != object_of.get(b, "?"):
                verdict = "cross_object"
            elif a in unknown or b in unknown or a not in gt_poses or b not in gt_poses:
                verdict = "unscorable"
            else:
                M_est = np.asarray(c["T"])
                M_gt = ev_mod.rel(gt_poses[key[0]], gt_poses[key[1]])
                M = M_est if (a, b) == key else np.linalg.inv(M_est)
                deg, d_cen, d_org = ev_mod.pose_error(M, M_gt, cen.get(key[1]))
                d = d_cen if cen else d_org
                if key not in adjacency:
                    verdict = "non_adjacent"
                else:
                    verdict = ("correct" if deg <= 5.0 and d <= 0.5 * thickness else "wrong_pose")
            counts[verdict] += 1
            if band == "confirmed" and verdict in ("wrong_pose", "non_adjacent", "cross_object"):
                bad.append((a, b, verdict))
    return conf, prob, len(gt_pairs), bad


def terracotta_gate(rep, tr, tiers=False):
    """R §13's terracotta row, read off report.json/transforms.json.

    joins used exactly {021-094, 094-104}; 007 unplaced; both pen 0; tight of both >= 0.27;
    the two seams within 20 % of 20.3 t and 10.7 t.

    Under ``--tiers on`` R §8 places the *confirmed* joins, and M1 §5.6 measured that one of the
    two is only probable at seed 4 -- so the per-seed rule becomes "no join outside the two", and
    the count over the five seeds is checked once, in :func:`verdict`.
    """
    num = digits
    used = [(num(j["a"]), num(j["b"]), j) for j in rep.get("joins_used", [])]
    pairs = {tuple(sorted((a, b))) for a, b, _ in used}
    fail = []
    if tiers:
        outside = sorted(pairs - TERRACOTTA_JOINS)
        if outside:
            fail.append("joins used outside R §13's two: %s" % outside)
    elif pairs != TERRACOTTA_JOINS:
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


def score(name, gt_dir, work, cen_cache, tiers=False):
    """evaluate.py over one finished run, or the terracotta's decision check."""
    with open(os.path.join(work, "transforms.json")) as f:
        tr = json.load(f)
    with open(os.path.join(work, "report.json")) as f:
        rep = json.load(f)
    row = dict(joins_used=len(rep.get("joins_used", [])),
               groups=len(tr["groups"]),
               largest=max((len(g) for g in tr["groups"]), default=0))
    obj = rep.get("objects")
    if obj is not None:
        row.update(objects=len(obj["objects"]), merges=obj["merges"],
                   demoted=len(obj.get("demotions", [])),
                   consensus_rejects=sum(len(o.get("rejects", [])) for o in obj["objects"]))
    if tiers:
        conf, prob, gt_pairs, bad = tier_score(name, gt_dir, rep, tr, cen_cache)
        row.update(tiers=True, confirmed=conf["band"], probable=prob["band"],
                   conf_correct=conf["correct"], conf_wrong_pose=conf["wrong_pose"],
                   conf_non_adjacent=conf["non_adjacent"],
                   conf_cross_object=conf["cross_object"],
                   conf_unscorable=conf["unscorable"],
                   prob_correct=prob["correct"], prob_wrong_pose=prob["wrong_pose"],
                   prob_non_adjacent=prob["non_adjacent"],
                   prob_cross_object=prob["cross_object"],
                   prob_unscorable=prob["unscorable"],
                   found_correct=conf["correct"] + prob["correct"],
                   conf_gt_pairs=gt_pairs,
                   conf_recall=(conf["correct"] / gt_pairs) if gt_pairs else None,
                   found_recall=((conf["correct"] + prob["correct"]) / gt_pairs)
                   if gt_pairs else None,
                   conf_false=[list(x) for x in bad],
                   confirmed_pairs=sorted(
                       tuple(sorted((digits(c["a"]), digits(c["b"]))))
                       for c in representatives(rep.get("candidates", []))
                       if c.get("tier") == "confirmed"),
                   probable_pairs=sorted(
                       tuple(sorted((digits(c["a"]), digits(c["b"]))))
                       for c in representatives(rep.get("candidates", []))
                       if c.get("tier") == "probable"))
    else:
        row.update(tiers=False)
    if gt_dir is None:
        ok, fail, detail = terracotta_gate(rep, tr, tiers)
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


def tier_verdict(name, rows):
    """The confirmed tier's own prohibition, on all eight sets: zero false joins in it.

    Plus, for the terracotta, M1 §5.6's restatement of audit §E's step-8 gate: the two joins are
    at least probable at every seed, nothing else is ever confirmed, and nine of the ten
    (seed, join) slots are confirmed.
    """
    bad, notes = [], []
    for r in rows:
        false = r["conf_wrong_pose"] + r["conf_non_adjacent"] + r["conf_cross_object"]
        if false:
            bad.append("seed %d: %d false join%s in the confirmed tier (%s)"
                       % (r["seed"], false, "" if false == 1 else "s",
                          "; ".join("%s-%s %s" % tuple(x) for x in r["conf_false"])))
    if name == "terracotta":
        slots, short = 0, []
        for r in rows:
            confirmed = {tuple(x) for x in r["confirmed_pairs"]}
            known = confirmed | {tuple(x) for x in r["probable_pairs"]}
            slots += len(TERRACOTTA_JOINS & confirmed)
            short += ["%s at seed %d" % ("-".join(p), r["seed"])
                      for p in sorted(TERRACOTTA_JOINS - confirmed)]
            missing = sorted(TERRACOTTA_JOINS - known)
            if missing:
                bad.append("seed %d: %s not even probable" % (r["seed"], missing))
        if slots < TERRACOTTA_CONFIRMED_SLOTS:
            bad.append("%saudit §E step 8 (\"terracotta's two joins confirmed\"): %d of the %d "
                       "(seed, join) slots confirmed -- %s only probable. Nothing outside "
                       "R §13's two is confirmed at any seed and both are at least probable at "
                       "every seed, so no prohibition is breached: what is unmet is the audit's "
                       "recall row."
                       % (RECALL, slots, TERRACOTTA_CONFIRMED_SLOTS, ", ".join(short)))
        notes.append("R §13's two joins confirmed in %d of the %d (seed, join) slots and probable "
                     "in the rest." % (slots, TERRACOTTA_CONFIRMED_SLOTS))
    # Audit §E's step-10 recall row, on the sets that have one (V8-D2).  Correct confirmed joins
    # are *reported* per seed on every set, because that is the number a reader asks for and the
    # number a fitted gate would freeze.
    notes.append("Correct confirmed joins per seed: %s."
                 % "/".join(str(r["conf_correct"]) for r in rows))
    floor = BANDS[name].get("recall_floor")
    if floor is not None:
        notes.append("Correct confirmed + probable joins per seed: %s, against the floor of %d."
                     % ("/".join(str(r["found_correct"]) for r in rows), floor))
        for r in rows:
            if r["found_correct"] < floor:
                bad.append("%sseed %d: %d correct joins in the confirmed and probable bands "
                           "together, below audit §E step 10's floor of %d (D §10.3's baseline)"
                           % (RECALL, r["seed"], r["found_correct"], floor))
    recalls = [r["conf_recall"] for r in rows if r["conf_recall"] is not None]
    if recalls:
        found = [r["found_recall"] for r in rows]
        notes.append("Confirmed recall %.3f-%.3f of the ground truth's adjacent pairs "
                     "(%.3f-%.3f with the probable band); "
                     "%d-%d confirmed and %d-%d probable joins per seed."
                     % (min(recalls), max(recalls), min(found), max(found),
                        min(r["confirmed"] for r in rows), max(r["confirmed"] for r in rows),
                        min(r["probable"] for r in rows), max(r["probable"] for r in rows)))
    else:
        notes.append("%d-%d confirmed and %d-%d probable joins per seed."
                     % (min(r["confirmed"] for r in rows), max(r["confirmed"] for r in rows),
                        min(r["probable"] for r in rows), max(r["probable"] for r in rows)))
    return bad, notes


def verdict(name, rows):
    """R §13 over one set's seeds.

    Returns ``(gate, [what to print], band, [band notes], [the failures themselves])``.  The last
    element is kept apart from the fourth because the printed lines carry the tier's notes as well
    as its failures, and the run's last line has to tell a breached prohibition from an unmet
    recall row (:data:`RECALL`).
    """
    b = BANDS[name]
    gate_bad, band_bad = [], []
    tiers = rows[0].get("tiers", False)
    tier_bad, tier_notes = tier_verdict(name, rows) if tiers else ([], [])
    gate_bad += tier_bad

    if not rows[0]["scored"]:                       # the terracotta's decision row
        for r in rows:
            if not r["decision_ok"]:
                gate_bad.append("seed %d: %s" % (r["seed"], "; ".join(r["decision_fail"])))
        detail = "; ".join("seed %d: %s" % (r["seed"], r["decision"]) for r in rows)
        ok = (["Zero false joins in the confirmed tier."] if tiers else
              ["R §13's two joins at every seed."])
        # The notes are printed whether the row passes or fails: what the tier reached is the
        # number a reader of a failing row needs most (V8-D3).
        return (("FAIL" if gate_bad else "pass"), (gate_bad or ok) + tier_notes, "n/a", [detail],
                gate_bad)

    accs = [r["frag_acc"] for r in rows]
    precs = [r["precision"] for r in rows]

    # R §13's last row on the seven single-object sets, and -- under `--tiers on` -- audit §E's
    # step-10 prohibition on `mixed_ABG`, which is the same two numbers read off the confirmed
    # tier's own assembly (V8-D2).
    pure = b["pure"] or (tiers and b.get("pure_tiers", False))
    if pure:                                        # R §13's last row, the prohibition
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
    if tiers:
        checked.append("zero false joins in the confirmed tier")
    if pure:
        checked.append("cross-object 0 and group purity 1.000")
    if tiers and b.get("recall_floor") is not None:
        checked.append("%d correct joins in the confirmed and probable bands together"
                       % b["recall_floor"])
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
    purities = [r["purity"] for r in rows if r["purity"] is not None]
    ok = ("%s at every seed." % "; ".join(checked).capitalize() if checked else
          "Not gated: %s. Cross-object %d-%d, purity %s over the five seeds."
          % (b.get("note", "no prohibition stated"),
             min(r["cross_object"] for r in rows), max(r["cross_object"] for r in rows),
             "n/a (no group of two)" if not purities
             else "%.3f-%.3f" % (min(purities), max(purities))))
    # R §13's bands are the *reference's* five accepted-tier draws; an assembly built from the
    # confirmed tier is a different quantity, so the comparison is not made rather than made and
    # reported as a difference the tier was built to produce.
    band = ("n/a" if (tiers or (b["acc"] is None and b["prec"] is None))
            else ("outside" if band_bad else "inside"))
    span = "Accuracy %.3f-%.3f, precision %.3f-%.3f." % (min(accs), max(accs), min(precs), max(precs))
    detail = ([span] + band_bad) if (band_bad and not tiers) else [span]
    if tiers:
        detail = [span, "R §13's band is the reference's accepted-tier draws and is not compared "
                        "against a confirmed-tier assembly."]
    return (gate, (gate_bad or [ok]) + tier_notes, band, detail, gate_bad)


def fmt(x, spec="%.3f"):
    return "-" if x is None else spec % x


def markdown(meta, rows, verdicts):
    tiers = bool(rows and rows[0].get("tiers"))
    objects = bool(rows and rows[0].get("objects") is not None)
    L = []
    L.append("# Quality gate — the Rust core on the development sets at seeds %s"
             % "-".join(str(s) for s in meta["seeds"]))
    L.append("")
    L.append("`tools/quality_gate.py`, %s, commit `%s`, backend `%s`, `--tiers %s --objects %s "
             "--object-disagreement %s`, `--no-preview --no-meshes`, "
             "warm cache from the second seed of each set. Total wall **%.1f s** (%.1f min) for "
             "%d runs. Scores are `tools/evaluate.py` (5 deg / 0.5 t, translation at the fragment "
             "centroid); the terracotta has no staged ground truth and carries R §13's decision "
             "row instead."
             % (meta["generated"], meta["commit"], meta["backend"],
                "on" if tiers else "off", meta.get("objects", "off"),
                meta.get("object_disagreement", "off"), meta["wall_total_s"],
                meta["wall_total_s"] / 60.0, len(rows)))
    if tiers:
        L.append("")
        L.append("The left half is `evaluate.py` over the joins R §8 **used**, which under "
                 "`--tiers on` are confirmed joins. The right half is the **confirmed tier "
                 "itself**: every confirmed pair of the run, classified by `evaluate.py`'s rule "
                 "applied to that candidate's *own* pose, so a confirmed join R §8 refused is "
                 "scored too. `conf recall` is confirmed-correct over the ground truth's adjacent "
                 "pairs present in the collection, and `correct found` is the same classification "
                 "over the confirmed **and** probable bands together -- what R §6.5 accepted, and "
                 "the list a conservator works from (audit §E step 10, V8-D2).")
    L.append("")
    head = ("| set | seed | frag acc | precision | correct | wrong pose | non-adj | cross-obj | "
            "purity | joins | wall s |")
    rule = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    if tiers:
        head += " confirmed | conf correct | conf false | conf recall | probable | correct found |"
        rule += "---:|---:|---:|---:|---:|---:|"
    if objects:
        head += " merges | demoted | outside |"
        rule += "---:|---:|---:|"
    L.append(head)
    L.append(rule)
    for r in rows:
        if r["scored"]:
            line = "| `%s` | %d | %.1f %% | %.3f | %d | %d | %d | %d | %s | %d | %.1f |" % (
                r["set"], r["seed"], 100 * r["frag_acc"], r["precision"], r["correct"],
                r["wrong_pose"], r["non_adjacent"], r["cross_object"],
                fmt(r["purity"]), r["joins_used"], r["wall_s"])
        else:
            line = "| `%s` | %d | — | — | — | — | — | — | — | %d | %.1f |" % (
                r["set"], r["seed"], r["joins_used"], r["wall_s"])
        if tiers:
            false = r["conf_wrong_pose"] + r["conf_non_adjacent"] + r["conf_cross_object"]
            line += " %d | %d | %d | %s | %d | %d |" % (
                r["confirmed"], r["conf_correct"], false, fmt(r["conf_recall"]), r["probable"],
                r["found_correct"])
        if objects:
            line += " %d | %d | %d |" % (
                r["merges"], r["demoted"], r["consensus_rejects"])
        L.append(line)
    if objects:
        L.append("")
        L.append("`merges` is audit §D.2 (c) -- two groups joined through a confirmed join, under "
                 "the penetration test across both and consistency with every cross-group join "
                 "the gate admits. `demoted` is a confirmed join the object pass moved to "
                 "probable; `outside` is a member its own object's consensus does not fit, "
                 "**reported and never acted on** -- task M1 §4 measured no feature reaching "
                 "audit §D.2's own AUC of 0.800 on any collection with real object ids.")
    L.append("")
    L.append("R §13 over the five seeds of each set. **Gate** is what R §13 states as a "
             "prohibition -- its last row (cross-object joins 0, group purity 1.000) on the seven "
             "single-object collections, the terracotta's decision row, and pot_G's rule -- and a "
             "failure of one exits non-zero.%s **Band** compares the port's five draws of fragment "
             "accuracy and precision with the *reference's* five draws, which is a comparison and "
             "not a bound (task H4's note)%s."
             % (" Under `--tiers on` audit §D.1's own prohibition joins them on all eight sets "
                "(**zero wrong-pose, non-adjacent and cross-object joins in the confirmed tier**), "
                "and audit §E's two roadmap rows join them where they apply: `mixed_ABG`'s "
                "cross-object 0 and purity 1.000 in the confirmed tier with a floor of 12 correct "
                "joins over the confirmed **and** probable bands, and the terracotta's ten "
                "(seed, join) slots."
                if tiers else "",
                "; under `--tiers on` it is not made at all, because the reference's draws are "
                "accepted-tier assemblies and these are confirmed-tier ones" if tiers else ""))
    L.append("")
    for name, (g, why, band, span, _) in verdicts.items():
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
    ap.add_argument("--tiers", choices=("on", "off"), default="on",
                    help="roadmap item 3's confidence tier (default on): `on` also scores the "
                         "confirmed tier and gates it at zero false joins; `off` runs the binary "
                         "the way it ran before the tier existed")
    ap.add_argument("--objects", choices=("on", "off"), default="on",
                    help="roadmap item 4's object separation (default on): the per-group "
                         "consensus, the group merge through a confirmed join, and -- only with "
                         "--object-disagreement on -- audit §D.2 (b)'s mutual-disagreement "
                         "demotion; `off` runs the binary the way task T2 ran it")
    ap.add_argument("--object-disagreement", choices=("on", "off"), default="off",
                    help="audit §D.2 (b), off by default on task O1's own measurement: it removes "
                         "no false join -- the confirmed tier has none -- and costs nine of "
                         "mixed_ABG seed 0's eleven")
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
                   "--backend", a.backend, "--no-preview", "--no-meshes", "--seed", str(seed),
                   "--tiers", a.tiers,
                   "--objects", a.objects,
                   "--object-disagreement", a.object_disagreement]
            wall, rc, log = sh(cmd)
            if rc != 0:
                print(log)
                raise SystemExit("run failed: %s seed %d (exit %d)" % (name, seed, rc))
            row, ev = score(name, gt_dir, work, cen_cache, a.tiers == "on")
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
                commit=commit, backend=a.backend, seeds=a.seeds, binary=a.bin, tiers=a.tiers,
                objects=a.objects, object_disagreement=a.object_disagreement,
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
                       verdicts={k: dict(gate=g, gate_detail=why, band=b, band_detail=span,
                                         failures=fails)
                                 for k, (g, why, b, span, fails) in verdicts.items()}),
                  f, indent=1)
    print()
    print(md)
    failed = [k for k, v in verdicts.items() if v[0] == "FAIL"]
    outside = [k for k, v in verdicts.items() if v[2] == "outside"]
    print("total wall %.1f s (%.1f min); %s" % (
        meta["wall_total_s"], meta["wall_total_s"] / 60.0,
        "FAILED: " + ", ".join(failed) if failed else
        "every set holds R §13's prohibitions" +
        ("" if not outside else "; outside the reference's own band: " + ", ".join(outside))))
    if failed:
        fails = [(k, line) for k in failed for line in verdicts[k][4]]
        recall_only = all(line.startswith(RECALL) for _, line in fails)
        for k, line in fails:
            print("  %-13s %s" % (k, line))
        print("  %s" % ("no prohibition was breached: every failure above is a recall row -- a "
                        "true join the tier declined to assert, on the probable list with its "
                        "reason" if recall_only else
                        "at least one failure above is a prohibition, not a recall row"))
    if rows and rows[0].get("tiers"):
        false = sum(r["conf_wrong_pose"] + r["conf_non_adjacent"] + r["conf_cross_object"]
                    for r in rows)
        good = sum(r["conf_correct"] for r in rows)
        found = sum(r["found_correct"] for r in rows)
        pairs = sum(r["conf_gt_pairs"] for r in rows)
        print("confirmed tier over %d runs: %d correct, %d false, %.1f %% of the %d "
              "ground-truth adjacent pairs; with the probable band %d correct, %.1f %%"
              % (len(rows), good, false, 100.0 * good / pairs if pairs else 0.0, pairs,
                 found, 100.0 * found / pairs if pairs else 0.0))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
