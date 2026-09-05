# Making the pipeline faster without changing what it decides

**Date:** 2026-09-05. Branch `perf-python`. Apple M2 Pro (10 cores), 16 GB, Python 3.12,
open3d 0.19. Test set `input/test_fragments_1` (4 fragments, 6 pairs).

The rule for everything below: the accepted joins, the groups and the numbers in the report had
to stay what they were. Where a score moves, it moves in the fourth digit.

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

## 2. What was changed

### a. Subsample the shell margin and the penetration samples (commit `74704fa`)

`MatchData` now keeps a seeded random subset of the margin (`Params.margin_points`, default 6000)
for the `pc_reg` ICP and for the continuity test, and a separate seeded subset of the surface
samples (`Params.pen_samples`, default 10000) for the penetration test, which used to run a
signed-distance query over all 30 000 samples in both directions per candidate. A uniform subset
of an area-weighted sample is still area-weighted, so the scores keep their meaning. `assembly`
still uses the full sample set. The arrays that `verify` needs are materialised once instead of
being re-gathered from boolean masks on every call.

Single-threaded pair: **51.6 s → 31.8 s**; the `pc_reg` ICP alone 39.4 s → 20.9 s.

### b. Early rejection in stage 2 (commit `30842ca`)

After the two `pc_reg` ICPs the tight-contact fraction is already close to its final value. Over
all 197 stage-2 candidates of the six real pairs:

| pair | candidates | below 0.12 at that point | their best final tight | cheap value of the accepted candidate |
|---|---|---|---|---|
| 007 – 021 | 39 | 29 | 0.137 | – |
| 007 – 094 | 6 | 4 | 0.103 | – |
| 007 – 104 | 40 | 30 | 0.147 | – |
| 021 – 094 | 32 | 23 | 0.172 | **0.261** |
| 021 – 104 | 40 | 8 | 0.172 | – |
| 094 – 104 | 40 | 19 | 0.149 | **0.274** |

No candidate that starts below 0.12 ever climbs past 0.172, and acceptance needs 0.25, while the
two true joins are already at 0.26 and 0.27 before the last two ICPs. `Params.early_reject_tight`
is therefore 0.12, about half of `min_tight`, and 113 of 197 candidates (57 %) skip the two
fracture-only ICPs, the penetration test and the continuity test. They keep their cheap scores
(tight, gap, contact, seam), are marked `partial = 1` with `pen = 0` and `cont_n = -1`, and can
never be accepted. `verify()` was split into its four parts so the cheap half can be handed over
instead of being computed twice.

Single-threaded pair: **31.8 s → 28.6 s**.

### c. Threads over candidates inside a pair (commit `5a56d7f`)

Both refinement stages are independent per hypothesis and spend nearly all their time inside
Open3D's ICP and scipy KD-tree queries, both of which release the GIL. Stage 2 of one pair
(32 candidates, 26.9 s serial), varying Python threads against Open3D's own OpenMP:

| python threads × OpenMP | 1×1 | 2×1 | 4×1 | 10×1 | 1×2 | 1×4 | 1×10 | 2×2 | 2×5 | 5×2 | 4×4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage 2, s | 26.9 | 14.5 | 8.8 | 4.6 | 15.3 | 10.6 | 9.1 | 8.3 | 5.8 | 4.8 | 4.8 |
| speed-up | 1.0 | 1.85 | 3.05 | **5.86** | 1.76 | 2.53 | 2.94 | 3.25 | 4.65 | 5.64 | 5.57 |

Threading over candidates wins clearly (3.05x on four threads, well over the 2.5x that was the
condition for taking this route; 5.86x on ten against 2.94x for OpenMP alone), so the worker
processes now run ICP with a single OpenMP thread and spend their budget on candidates. The
individual primitives scale as: `registration_icp` 3.86x on 4 threads, `cKDTree` build+query
2.69x, `compute_signed_distance` 1.05x (it keeps the GIL, but it is only ~0.7 s per pair now).

The pipeline splits the machine as processes × threads ≈ cores (`_match_workers`), and a run with
a single pair no longer bypasses the process pool. Whole matching stage of the real set, warm
cache:

| processes × threads | 6×1 | 6×2 (default) | 6×3 | 5×2 | 3×3 |
|---|---|---|---|---|---|
| matching, s | 51.0 | 39.1 | 36.9 | 37.3 | 37.1 |

All five produced the same 30 candidates and the same 6 accepted joins. Above two threads the
machine is saturated and the layout stops mattering; the default is what the formula gives.

Two fragments only (one pair, so one process with ten threads): that pair went from **51.6 s to
5.1 s**. This was the worst case before — a two- or three-fragment set used one core out of ten.

Determinism: results keep the job order (`ThreadPoolExecutor.map`), so ranking, `keep` and the
`Candidate` objects do not depend on the thread count. `match_pair(A, B, t, p, keep=5)` still
works exactly as before and stays single-threaded outside the pipeline, because nothing has
capped OpenMP there. KD-tree queries inside a pool are pinned to one worker each
(`geometry.single_threaded`) so that scipy's threads do not multiply with the pool's.

### d. Vectorised non-max suppression (commit `a779195`)

`nms` compared each pose against every kept pose in a Python loop with a numpy call per
comparison. The translation test now runs against all kept poses at once and the rotation test
only for the few that are close, with the same arithmetic — checked identical to the old function
on 600 randomised cases. Worth ~0.3 s per pair, which is inside the noise of a run; it was done
because it was the only Python-level entry left in the profile.

## 3. Result

Two full runs of each state, alternating, on the same machine under the same background load
(`sherd-refit run input/test_fragments_1/fragments --out output/bench_{before,after}`, cold cache,
default flags):

| stage | before | before | after | after |
|---|---|---|---|---|
| preprocess, s | 19.4 | 16.6 | 19.5 | 17.6 |
| matching, s | 94.9 | 93.4 | **39.5** | **43.0** |
| assembly, s | 0.9 | 0.9 | 0.9 | 0.9 |
| refine, s | 6.0 | 6.0 | 11.4 | 9.6 |
| wall clock | 2:14.1 | 2:08.9 | **1:23.5** | **1:22.9** |
| CPU (user), s | 578.7 | 579.7 | 417.6 | 426.5 |

Matching **94.2 s → 41.3 s (2.3x)**, whole run **131.5 s → 83.2 s (1.6x)**, and 27 % less CPU
burned. Of the matching gain, the work reductions (a, b, d) give 94.2 s → 51.0 s and the threads
take it to 41.3 s.

The refine numbers above are noise, not a regression: the machine was running someone else's
compiler during these measurements. Timed in isolation on the real data, with the poses each
state produces, `refine_joins` takes 5.92 / 6.07 s with the new code and 6.24 / 6.09 s with the
old one — `refine.py` is untouched.

Preprocessing (17 s) and writing the outputs (10 s) were not touched; nothing in them showed up
as an avoidable cost.

Memory did not grow: a worker still holds two fragments, and the subsampled point sets are
smaller than the full ones. Measured peak RSS of a matching worker with the two largest fragments
of the set: 384 MB (200 MB of it is the interpreter with open3d loaded). The number of worker
processes is capped by `--workers`, not by the number of fragments, so 30 fragments still means at
most 9 x 384 MB ~ 3.5 GB on this machine.

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
identical. Groups are the same (104 – 094 – 021, FY234007 unplaced), and so are the six accepted
candidates and the 30 reported ones.

One reporting change is deliberate: in **Best candidate per pair**, a row whose best candidate was
rejected early now shows `penetration 0.0000` and `normal agr. -1.00`, meaning "not computed"
(`partial = 1` in `report.json`). On this set that happens for one pair, 007 – 021.

## 4. Tried and not kept

- **Open3D's OpenMP instead of Python threads.** 2.94x on ten cores against 5.86x for threading
  over candidates (table above). ICP is now run with one OpenMP thread per worker.
- **Process-level parallelism over candidates** (chunks of candidates as separate pool jobs with
  an LRU cache of `MatchData` per worker). Not implemented: threads passed the 2.5x bar with room
  to spare, and this would have added a cache, job ordering constraints and 60 MB per cached
  fragment for no measured gain.
- **Fewer ICP iterations, or a looser convergence criterion.** The cheapest possible win, and off
  limits: it changes the poses and therefore the scores.
- **Skipping `assembly_<k>.ply` when `write_meshes=False`.** Already the case: `write_placed_meshes`
  is only called when meshes are written.
- **Hoisting `np.linalg.inv(T)`.** About 2 µs per candidate, invisible in the profile. It now sits
  in the penetration part, so the early-rejected candidates do not compute it at all.
- **Building `cKDTree(PBf)` once per candidate.** For a candidate that passes the early check the
  two builds happen at two different poses (before and after the fracture-only ICPs), so they
  cannot be shared without changing the result. For an early-rejected candidate the cheap
  fracture scores are handed to `verify` instead of being recomputed, which removes the only real
  duplicate.
- **Subsampling the fracture points** the way the margin was subsampled. Not tried on purpose:
  they are the points the tight-contact and gap thresholds are measured on, and there are only
  ~5 000 of them.

## 5. New knobs

`--threads`, `--early-reject-tight`, `--margin-points`, `--pen-samples`, documented in the README's
CLI table. Every existing flag and `Params` field kept its name and default.
