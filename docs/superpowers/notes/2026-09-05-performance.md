# Making the pipeline faster without changing what it decides

**Date:** 2026-09-05. Branch `perf-python`. Apple M2 Pro (10 cores), 16 GB, Python 3.12,
open3d 0.19. Test set `input/test_fragments_1` (4 fragments, 6 pairs).

Quality comes first and speed second: no change was kept unless it could be shown, candidate by
candidate, not to move what the pipeline decides. Two of the four optimisations that were built
did not survive that test and are off in the shipped defaults; they are written up in section 5
with the measurements that killed them.

## 1. Where the time went

`cProfile` on one pair (FY234021 – FY234094), single-threaded
(`SHERD_REFIT_THREADS=1 OMP_NUM_THREADS=1`), 74 648 hypotheses, 366 poses through stage 1,
32 candidates through stage 2:

| part | s | share |
|---|---|---|
| hypotheses | 0.02 | 0.0 % |
| coarse score | 1.01 | 1.9 % |
| non-max suppression (both stages) | 0.15 | 0.3 % |
| **stage 1** (breakline ICP + re-score) | **2.26** | 4.2 % |
| **stage 2** | **50.00** | **93.6 %** |
| – ICP on `pc_reg` (fracture + shell margin), 2 per candidate | 39.38 | 73.7 % |
| – ICP on `pc_frac` (fracture only), 2 per candidate | 6.19 | 11.6 % |
| – `verify` | 4.44 | 8.3 % |
| ·· fracture KD-tree queries | 0.86 | 1.6 % |
| ·· seam | 0.10 | 0.2 % |
| ·· margin / continuity | 1.16 | 2.2 % |
| ·· penetration (signed distance) | 2.31 | 4.3 % |
| total | 53.44 | |

By function, `registration_icp` was 47.96 s of 52.5 s over 860 calls. Everything else — numpy,
KD-trees, the Python loops — was under 5 s together. So the target was the number and the size of
the ICP calls, not the Python around them.

The point clouds explain the split: `pc_reg` held 18 122 points for FY234021 and 17 975 for
FY234094, of which **13 021 and 13 381 were shell margin** and only ~5 000 were fracture points.
The ICP that matters ran mostly on the margin.

## 2. The quality gate

`dump_candidates.py` runs all six real pairs and writes, for every stage-2 candidate: the pose,
every verification score, and the cheap tight-contact estimate taken after the two `pc_reg` ICPs.
It always runs the whole ICP chain and the full verification, so the two states can be compared
candidate by candidate even where the shipped code would stop early. `compare_dumps.py` then
checks the pose of the best candidate per pair (1° / 0.05 t), the two true joins (rank 1, tight /
gap / seam within ±0.03 / ±0.01 t / ±1 t), acceptance flips, penetration crossings, and the
early-rejection margin. `dump_synthetic.py` does the same for the synthetic slab pair, reusing the
fixture code of `tests/test_synthetic.py`. All of it lives next to the dumps in the scratchpad
directory named at the end of this note.

**The harness was validated first.** Running the new code with `margin_points = 0` and
`pen_samples = 0` — the two subsampling knobs turned off — reproduces the baseline **bit for bit**:

```
197 candidates compared; max |dT| 0.000e+00, max |d score| 0.000e+00
```

That single line proves that the threading, the split of `verify` into four parts and the
vectorised non-max suppression change nothing at all, and that every difference seen later comes
from subsampling alone.

**What the criterion can and cannot measure.** For a pair with no real join, "the best candidate"
is not a stable quantity. Changing only the seed of the surface sampling — the same code, nothing
subsampled, a perturbation with no quality meaning — moves it wildly:

| pair | best candidate moves | tight | accepted |
|---|---|---|---|
| 007 – 021 | 164.25°, 16.36 t | 0.132 → 0.172 | no → no |
| 007 – 094 | 174.12°, 13.71 t | 0.144 → 0.092 | no → no |
| 007 – 104 | 111.09°, 3.49 t | 0.134 → 0.132 | no → no |
| 021 – 104 | 5.90°, 0.52 t | 0.175 → 0.181 | no → no |
| **021 – 094** | **0.02°, 0.007 t** | 0.296 → 0.284 | yes → yes |
| **094 – 104** | **0.20°, 0.017 t** | 0.277 → 0.301 | yes → yes |

Where there is something to find, the answer is stable to hundredths of a degree; where there is
not, several unrelated poses sit within a percent of each other on `seam × tight` and the top one
is a coin toss. The pose check is therefore meaningful for the two true joins and for nothing
else, and the numbers below are read that way.

## 3. What was kept

### a. Subsample the shell margin (commit `74704fa`, adjusted in `a10e9e4`)

`MatchData` keeps a seeded random subset of the margin (`Params.margin_points`, default 6000) for
the `pc_reg` ICP and for the continuity test. A uniform subset of an area-weighted sample is still
area-weighted. The arrays `verify` needs are materialised once instead of being re-gathered from
boolean masks on every call.

Single-threaded pair: **51.6 s → 33.9 s**; the `pc_reg` ICP alone 39.4 s → 20.9 s.

Quality, against the baseline dump:

| check | result |
|---|---|
| true joins, rank 1 and accepted | yes, both |
| 021 – 094 | Δtight **+0.0000**, Δgap **−0.00000 t**, Δseam **+0.00 t**, pen 0.00000 |
| 094 – 104 | Δtight **+0.0007**, Δgap **−0.00000 t**, Δseam **+0.00 t**, pen 0.00000 |
| acceptance flips over all returned candidates | **0**; accepted per pair 0, 0, 0, 4, 0, 2 before and after |
| penetration, largest change on any paired candidate | 0.00040, no candidate crosses 0.005 |
| best candidate per pair within 1° / 0.05 t | 5 of 6; the exception is 007 – 104 (1.61°, 0.053 t), a rejected candidate of a pair with no join |
| synthetic slab pair | pose error 0.104° / 0.339 units (0.0113 t), tight 0.6661, gap 0.0310, seam 20.33, pen 0.00000 — identical to the baseline, `cont_n` 0.991 → 0.992 |
| final assembled geometry | relative pose of all three placed fragments identical to 1e-4 units / 1e-4 ° |

The true joins move by less than a thirtieth of what a change of sampling seed does to them, and
the one pair that misses the pose check is the same kind of unstable false pair as in the table in
section 2. Raising `margin_points` to 10 000 does not help: it makes that check *worse* (two pairs
out of tolerance instead of one, 007 – 094 flipping by 64.7°) while leaving the true joins
identical, which is what one expects if the quantity being chased is noise. 6000 stays.

### Regression check against the baseline candidates

Verdict: **pass**. Every stage-2 candidate of the six real pairs was dumped before and after and
compared pose by pose.

- Both true joins keep rank 1 and stay accepted, with identical scores: Δtight +0.0000 and
  +0.0007, Δgap 0.00000 t, Δseam 0.00 t, penetration 0.0000.
- Zero acceptance flips over all returned candidates; the accepted count per pair is unchanged
  (0, 0, 0, 4, 0, 2).
- Penetration moves by at most 0.00040 on any paired candidate, and no candidate crosses the
  0.005 threshold in either direction.
- Early rejection, kept as an opt-in knob: at a threshold of 0.12 the highest final tight among
  the candidates it would drop is 0.166 on the new dump and 0.172 on the baseline one, against an
  acceptance threshold of 0.25 and the 0.20 safety bar, a margin of +0.034 and +0.028.
- One pose deviation: the best candidate of 007 – 104 moves 1.61° / 0.053 t, just outside the
  1° / 0.05 t tolerance. That pair has no true join, its best candidate reaches tight 0.153
  against an acceptance threshold of 0.25, and it is rejected before and after. The tolerance is
  meant for accepted joins, so this is not a regression; see the seed experiment above for why
  the top candidate of a non-matching pair is not a stable quantity in the first place.

`compare_dumps.py baseline_candidates.json final_candidates.json`, verbatim:

```
baseline: baseline   current: final   thickness 39.40

1. best candidate per pair (shipped match_pair output)
  FY234007 - FY234021: d_pose  0.00 deg  0.000 t   tight 0.132->0.132  gap 0.0924->0.0924  seam  11.0-> 11.0  accepted False->False  ok
  FY234007 - FY234094: d_pose  0.29 deg  0.026 t   tight 0.144->0.143  gap 0.0791->0.0784  seam   9.7-> 10.0  accepted False->False  ok
  FY234007 - FY234104: d_pose  1.61 deg  0.053 t   tight 0.134->0.153  gap 0.0850->0.0842  seam  10.0-> 11.3  accepted False->False  OUT OF TOLERANCE
  FAIL ('FY234007_reduced', 'FY234104_reduced'): best candidate moved by 1.61 deg / 0.053 t
  FY234021 - FY234094: d_pose  0.00 deg  0.000 t   tight 0.296->0.296  gap 0.0576->0.0576  seam  21.3-> 21.3  accepted True->True  ok
  FY234021 - FY234104: d_pose  0.13 deg  0.011 t   tight 0.175->0.174  gap 0.0739->0.0741  seam  12.3-> 12.3  accepted False->False  ok
  FY234094 - FY234104: d_pose  0.01 deg  0.001 t   tight 0.277->0.278  gap 0.0578->0.0578  seam  12.3-> 12.3  accepted True->True  ok

2. the two true joins
  FY234021 - FY234094: rank 1 accepted, d_tight +0.0000 (<= 0.03), d_gap -0.00000 t (<= 0.01), d_seam +0.00 t (<= 1.0), pen 0.00000
  FY234094 - FY234104: rank 1 accepted, d_tight +0.0007 (<= 0.03), d_gap -0.00000 t (<= 0.01), d_seam +0.00 t (<= 1.0), pen 0.00000

3. acceptance flips (all returned candidates, paired by pose)
  FY234007 - FY234021: 1 paired, 9 baseline-only, 9 current-only, accepted 0 -> 0
  FY234007 - FY234094: 1 paired, 5 baseline-only, 5 current-only, accepted 0 -> 0
  FY234007 - FY234104: 1 paired, 9 baseline-only, 9 current-only, accepted 0 -> 0
  FY234021 - FY234094: 4 paired, 6 baseline-only, 6 current-only, accepted 4 -> 4
  FY234021 - FY234104: 3 paired, 7 baseline-only, 7 current-only, accepted 0 -> 0
  FY234094 - FY234104: 3 paired, 7 baseline-only, 7 current-only, accepted 2 -> 2
  flips: 0

4. penetration under subsampling (full verification on both sides)
  FY234007 - FY234021: 3 paired, 1 with pen > 0.005 before, 0 of them now below, max |d_pen| 0.00000
  FY234007 - FY234094: 1 paired, 0 with pen > 0.005 before, 0 of them now below, max |d_pen| 0.00023
  FY234007 - FY234104: 5 paired, 4 with pen > 0.005 before, 0 of them now below, max |d_pen| 0.00010
  FY234021 - FY234094: 7 paired, 3 with pen > 0.005 before, 0 of them now below, max |d_pen| 0.00000
  FY234021 - FY234104: 7 paired, 7 with pen > 0.005 before, 0 of them now below, max |d_pen| 0.00040
  FY234094 - FY234104: 8 paired, 4 with pen > 0.005 before, 0 of them now below, max |d_pen| 0.00017
  largest penetration change on any candidate: 0.00040
```

### b. Threads over candidates inside a pair (commit `5a56d7f`)

Both refinement stages are independent per hypothesis and spend nearly all their time inside
Open3D's ICP and scipy KD-tree queries, both of which release the GIL. Stage 2 of one pair
(32 candidates, 26.9 s serial), varying Python threads against Open3D's own OpenMP:

| python threads × OpenMP | 1×1 | 2×1 | 4×1 | 10×1 | 1×2 | 1×4 | 1×10 | 2×2 | 2×5 | 5×2 | 4×4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage 2, s | 26.9 | 14.5 | 8.8 | 4.6 | 15.3 | 10.6 | 9.1 | 8.3 | 5.8 | 4.8 | 4.8 |
| speed-up | 1.0 | 1.85 | 3.05 | **5.86** | 1.76 | 2.53 | 2.94 | 3.25 | 4.65 | 5.64 | 5.57 |

Threading over candidates wins clearly (3.05x on four threads, over the 2.5x that was the
condition for taking this route; 5.86x on ten against 2.94x for OpenMP alone), so the worker
processes run ICP with a single OpenMP thread and spend their budget on candidates. Per primitive:
`registration_icp` 3.86x on 4 threads, `cKDTree` build+query 2.69x, `compute_signed_distance`
1.05x (it keeps the GIL, but it is a small part of a pair).

The pipeline splits the machine as processes × threads ≈ cores (`_match_workers`), and a run with
a single pair no longer bypasses the process pool. Matching stage of the real set, warm cache:

| processes × threads | 6×1 | 6×2 (default) | 6×3 | 5×2 |
|---|---|---|---|---|
| matching, s | 64.1 | **44.5** | 45.8 | 50.2 |

All four produced the same 30 candidates and the same 6 accepted joins. Two fragments only (one
pair, one process, ten threads): that pair went from **51.6 s to 7.4 s**. This was the worst case
before — a two- or three-fragment set used one core out of ten.

Determinism: results keep the job order (`ThreadPoolExecutor.map`), so ranking, `keep` and the
`Candidate` objects do not depend on the thread count. Two full runs of the real set produce
byte-identical `groups`, `candidates`, `joins_used` and `joins_rejected` in `report.json`,
transform matrices included. `match_pair(A, B, t, p, keep=5)` still works as before and stays
single-threaded outside the pipeline, because nothing has capped OpenMP there. KD-tree queries
inside a pool are pinned to one worker each (`geometry.single_threaded`) so that scipy's threads
do not multiply with the pool's.

### c. Vectorised non-max suppression (commit `a779195`)

`nms` compared each pose against every kept pose in a Python loop with a numpy call per
comparison. The translation test now runs against all kept poses at once and the rotation test
only for the few that are close, with the same arithmetic — checked identical to the old function
on 600 randomised cases, and part of the bit-for-bit reproduction above. Worth ~0.3 s per pair.

## 4. Result

Two full runs of each state, alternating, same machine, same background load
(`sherd-refit run input/test_fragments_1/fragments --out output/bench_{before,after}`, cold cache,
default flags):

| stage | before | before | after | after |
|---|---|---|---|---|
| preprocess, s | 16.2 | 17.1 | 16.3 | 16.9 |
| matching, s | 84.3 | 84.5 | **49.8** | **45.6** |
| assembly, s | 0.9 | 0.9 | 0.9 | 0.9 |
| refine, s | 5.6 | 5.8 | 9.4 | 9.3 |
| wall clock | 1:58.2 | 2:00.1 | **1:28.0** | **1:24.2** |
| CPU (user), s | 552.2 | 553.7 | 512.0 | 511.6 |

Matching **84.4 s → 47.7 s (1.77x)**, whole run **119.1 s → 86.1 s (1.38x)**. One pair
single-threaded: **51.6 s → 33.9 s**. A two-fragment set: **51.6 s → 7.4 s**.

**Refinement costs 3.7 s more, and that is real.** Timed from each state's own pre-refine poses,
interleaved: 6.06 / 6.00 s with the old poses against 9.80 / 9.76 s with the new ones. `refine.py`
is untouched; the full-resolution ICP simply needs more iterations to converge from a starting
pose that the margin subsample moved by a hair. It converges to the same place: same fitness
(0.198, 0.150), same inlier rmse (0.010 t), and a final assembled geometry identical to the
baseline's to 1e-4 units. The stage was left alone rather than tuned, because changing its
iteration counts would change the output.

Preprocessing (17 s) and writing the outputs (10 s) were not touched; nothing in them showed up as
an avoidable cost. Memory did not grow: peak RSS of a matching worker holding the two largest
fragments is 384 MB, of which 200 MB is the interpreter with open3d loaded. The number of worker
processes is capped by `--workers`, not by the number of fragments, so 30 fragments still means at
most 9 × 384 MB ≈ 3.5 GB on this machine.

### The result itself is unchanged

`output/bench_after/report.md`:

```
## Joins used

| A | B | score | seam (t) | tight A/B | gap (t) | contact (t²) | shell cont. | normal agr. | penetration |
|---|---|---|---|---|---|---|---|---|---|
| FY234021_reduced | FY234094_reduced | 6.31 | 21.3 | 0.30 / 0.41 | 0.058 | 8.4 | 0.014 | 1.00 | 0.0000 |
| FY234094_reduced | FY234104_reduced | 3.43 | 12.3 | 0.43 / 0.28 | 0.058 | 4.1 | 0.024 | 1.00 | 0.0000 |
```

Against the same table from the before-run, only `shell cont.` (0.013 → 0.014), `contact`
(4.2 → 4.1) and the ranking score (3.42 → 3.43) move at all; seam, tight, gap and penetration are
identical, the groups are the same (104 – 094 – 021, FY234007 unplaced), and so are the six
accepted candidates and the 30 reported ones. `pytest -q`: 30 passed.

## 5. Built, measured, and switched off

Two optimisations worked and were still rejected, because the quality gate showed what they cost.

### Penetration on a subsample of the surface points — reverted

Running the signed distance over 10 000 of the 30 000 samples in each direction cut that part of
`verify` from 2.31 s to 0.71 s per pair. It also moved `pen` by up to **0.0012**, a fifth of the
0.005 acceptance threshold, and one candidate of the 021 – 094 pair crossed it: `pen` 0.00503 →
0.00480. That candidate was nowhere near acceptable on its other scores (tight 0.139 against a
0.25 threshold), so nothing was actually accepted that should not have been — but the margin
between a false candidate and the penetration threshold is exactly what this test exists to
protect. With every sample kept, the largest change on any candidate is 0.00040 and nothing
crosses. `Params.pen_samples` survives as a knob and defaults to 0, meaning "use all of them".

### Early rejection in stage 2 — off by default

The idea: after the two `pc_reg` ICPs the tight-contact fraction is close to its final value, so a
candidate far below `min_tight` can skip the two fracture-only ICPs, the penetration test and the
continuity test. It worked — 57 % of candidates took the short path at a threshold of 0.12, and a
pair went from 31.8 s to 28.6 s — but the safety analysis does not support the threshold:

| threshold | dropped (baseline) | highest final tight among the dropped | margin to the 0.20 bar |
|---|---|---|---|
| 0.06 | 23/197 | 0.125 | +0.075 |
| 0.08 | 40/197 | 0.147 | +0.053 |
| 0.10 | 74/197 | 0.172 | +0.028 |
| 0.12 | 113/197 | 0.172 | +0.028 |
| 0.14 | 146/197 | 0.186 | +0.014 |

Over the same 197 candidates, the cheap estimate can still rise by as much as **0.0935** during
the two ICPs it would skip. A threshold guaranteed to respect the 0.20 bar in the worst case must
therefore sit at or below 0.107, and even at 0.10 the candidates it drops reach 0.172, only 0.028
short of the bar. At a genuinely comfortable 0.06 the optimisation pays for nothing: 23 dropped
candidates × ~0.22 s saved is about what the cheap check itself costs on all 197 (~0.025 s each).
At 0.10 it is worth roughly 7 % of the matching stage, for a margin nobody should have to think
about. `Params.early_reject_tight` defaults to 0 and the check is skipped entirely when it is, so
the reports are again exactly the baseline's, with no `partial` rows.

For the record, the two together were worth 6 s of the matching stage: 41.3 s with both on against
47.7 s as shipped.

## 6. Tried and not kept for other reasons

- **Open3D's OpenMP instead of Python threads.** 2.94x on ten cores against 5.86x for threading
  over candidates. ICP now runs with one OpenMP thread per worker.
- **Process-level parallelism over candidates** (candidate chunks as pool jobs with an LRU cache of
  `MatchData` per worker). Not implemented: threads passed the 2.5x bar with room to spare, and
  this would have added a cache, job ordering constraints and 60 MB per cached fragment for no
  measured gain.
- **Fewer ICP iterations, or a looser convergence criterion.** The cheapest possible win, and off
  limits: it changes the poses and therefore the scores.
- **Skipping `assembly_<k>.ply` when `write_meshes=False`.** Already the case: `write_placed_meshes`
  is only called when meshes are written.
- **Hoisting `np.linalg.inv(T)`.** About 2 µs per candidate, invisible in the profile. It now sits
  in the penetration part.
- **Building `cKDTree(PBf)` once per candidate.** With early rejection off there is only one build
  per candidate anyway; with it on, the cheap fracture scores are handed to `verify` instead of
  being recomputed.
- **Subsampling the fracture points** the way the margin was subsampled. Not tried on purpose: they
  are the points the tight-contact and gap thresholds are measured on, and there are only ~5 000.

## 7. New knobs

`--threads`, `--margin-points`, `--early-reject-tight` (off), `--pen-samples` (off), documented in
the README's CLI table. Every existing flag and `Params` field kept its name and default.

Dumps and scripts:
`/private/tmp/claude-501/-Users-vaceslaveliseev--dev-ceramic-reassembling/0ffcf053-0fbe-4b23-9b9d-538d546e185e/scratchpad/perf/`
(`dump_candidates.py`, `dump_synthetic.py`, `compare_dumps.py`, `profile_pair.py`,
`thread_scaling.py`, `stage2_config.py`, and the `*_candidates.json` / `*_synthetic.json` dumps).
