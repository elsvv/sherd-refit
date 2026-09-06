# Scaling to thousands of pairs

**Date:** 2026-09-06. Branch `scale-pairs`. Apple M2 Pro (10 cores), 16 GB, Python 3.12,
open3d 0.19, 9 worker processes unless stated otherwise.

Roadmap item 2. The matcher is all-pairs: `input/sfspp/mixed_all` (164 real sherds of 10 pots) has
12 589 pairs after the wall-thickness prune and `input/synthetic_pingsdorf_170` (164 fragments, 6
of them from another vessel) has 12 839, at tens of CPU-seconds each. The target was both sets end
to end in about an hour with no loss of true joins.

Three things came out of it. The order below is the order of their value.

**The dihedral angle that pairs frames into hypotheses was systematically wrong by 40 degrees**
(§2). Both macro normals were averaged over a neighbourhood that includes the worn crease, which
pulls them together, so the two dihedrals of a mating pair summed to 141 degrees where the
geometry says 180 — and the hypothesis filter's window is 40 degrees wide. Roughly half the
correct frame pairs were being discarded before anything was scored.

**A pair's time was going somewhere nobody had looked** (§3). Measuring where the correct pose
actually sits let `stage2` fall from 40 candidates to 10 and the hypothesis count to a quarter;
profiling one pair then showed three quarters of what was left in twenty ICP registrations against
a 16 000-point cloud that only needs 6 000. Together with the dihedral fix: **four to six times
faster on every benchmark set, and better on three of them** — pot C 50 % to 75 %, pot H 27.3 % to
36.4 %, the synthetic 20 80 % to 95 %.

**No cheap partner search works on this data** (§4). Not the rigid-invariant breakline signatures
the roadmap asked for, not the matcher's own coarse stage, not object separation by wall thickness
and shell curvature. Section 4 measures the ceiling all three hit: at the *exact ground-truth
pose* a true adjacent pair overlaps on only 5–22 % of its breakline, because one seam is a
fraction of a fragment's whole crack line, and the best of several thousand poses reaches the same
10–28 % on pairs that never touched. Both mechanisms ship as flags that are off.

All-pairs on 164 fragments still projects to about 4.6 hours rather than the hour the brief asked
for (§7), and that run is in progress rather than finished.

## 1. Matching data in the cache

`MatchData` used to cost 0.21 s per fragment per pair job, spent on `face_adjacency` over the
working mesh, two ball matrices over the face centroids that build the breakline frames, and the
sampling. All of it was recomputed for every pair the fragment took part in.

That array half is now its own function, `fragment.match_arrays`, which preprocessing runs once
and stores in the fragment cache (ten arrays plus the settings they were built with).
`MatchData` is left with the KD-trees and the Open3D point clouds over those arrays. A cache whose
geometry is still valid but whose arrays were built with other sampling settings has only the
arrays recomputed, not the segmentation.

Measured on the four terracotta fragments, single-threaded, warm file cache:

| | `Fragment.load` | `MatchData` from the cache | `MatchData` recomputed | `match_arrays` |
|---|---|---|---|---|
| mean per fragment, s | 0.029 | **0.003** | 0.206 | 0.212 |

Everything here depends on the wall thickness, and a pair is matched at `min(t_A, t_B)`, so the
cached copy (built at the fragment's own thickness) serves the thinner fragment of a pair and the
thicker one is rebuilt. That keeps the result identical to a from-scratch build — the single
seeded `rng` is consumed in the same order as before the split — and it is why the setup cost does
not fall to zero: 0.468 s per pair before, 0.267 s with one fragment cached, 0.064 s with both,
nothing at all on a worker LRU hit.

**On a small set this saves nothing worth measuring, and that is the point.** A pot's pair costs
tens of seconds, so 0.2 s of setup is under 2 %: pot A ran 341 s against the 345 s recorded on
`main`, pot B 419 against 429, pot C 103 against 105, pot G 90 against 90 — all inside the noise.
The work is what makes a *screening* pass possible at all, where the job itself is a tenth of a
second and the setup would otherwise be twice the job.

At this point the terracotta was still bit-identical to `main`: same two joins, same groups, 30
candidates each, `max |dT|` 0 and `max |d score|` 0 over every candidate and every score.

## 2. The dihedral angle was wrong by 40 degrees, and the crease is why

`hypotheses` pairs a frame of A with a frame of B when their dihedral angles between shell and
fracture are complementary, `|d_A + d_B - 180| < dihedral_tol`, with the tolerance at 40 degrees.
Both angles are measured between macro normals averaged over every face within 0.35 t of the
breakline point.

Take the fragments of a ground-truth-adjacent pair, put them at their ground-truth poses, pair up
the breakline points that actually meet, and add their two dihedrals. Over 1 800 such points on
`mixed_all` and 2 100 on the synthetic 170:

| neighbourhood (in `t`) | sum p25 | sum p50 | sum p75 | median \|180 − sum\| | frames kept |
|---|---|---|---|---|---|
| everything within 0.20 (mixed) | 104.3 | 113.6 | 123.2 | 66.4 | 1.000 |
| **everything within 0.35 — what shipped** | **135.9** | **141.8** | **147.6** | **38.2** | 1.000 |
| everything within 0.50 | 147.5 | 152.2 | 156.7 | 27.8 | 1.000 |
| everything within 0.80 | 156.3 | 161.3 | 165.0 | 18.7 | 1.000 |
| 0.10 to 0.50 | 172.0 | 175.7 | 179.4 | 5.2 | 0.997 |
| **0.15 to 0.60 — what ships now** | **175.8** | **179.0** | **182.9** | **3.6** | 0.985 |
| 0.20 to 0.60 | 176.5 | 180.0 | 183.7 | 3.6 | 0.973 |
| 0.25 to 0.60 | 176.1 | 180.4 | 184.0 | 3.9 | 0.951 |

The geometry says 180. The shipped neighbourhood said 142. **The cause is the crease itself**: the
arris where shell meets fracture is worn on real sherds and rounded further by the three Taubin
iterations the working mesh gets, so the faces closest to the breakline lean towards the other
surface and pull both macro normals together. It is not the outer radius — widening it from 0.35
to 0.80 only halves the error — it is the inner one. Excluding the innermost 0.15 t fixes it
outright, and where the annulus finds nothing (1.5 % of points, on narrow strips and chain ends)
the full neighbourhood is used as a fallback.

The consequence was not subtle. With true pairs sitting 38 degrees off the centre of a 40-degree
window, **roughly half of the correct frame pairs were being discarded before anything was
scored**. After the fix, 99.3 % (mixed) and 97.2 % (synthetic) of meeting points fall inside the
40-degree window, and the window itself can be tightened:

| `dihedral_tol` | 40 | 30 | 25 | 20 | 15 | 10 |
|---|---|---|---|---|---|---|
| meeting points kept, mixed_all | 0.993 | 0.987 | **0.976** | 0.956 | 0.926 | 0.847 |
| meeting points kept, synthetic 170 | 0.972 | 0.936 | **0.905** | 0.863 | 0.801 | 0.676 |
| hypotheses per pair, relative | 1.00 | 0.89/0.81 | **0.81/0.69** | 0.71/0.57 | 0.58/0.43 | 0.42/0.30 |

25 degrees ships. Note what the last row says about the *old* state: the filter was nearly
vacuous, because most breakline points measure a dihedral near 90 degrees and almost any two of
them sum to 180 within 40. Tightening it is worth a fifth of the hypotheses, no more — the
tolerance was never the expensive part. What it was, was wrong.

## 3. What a pair's time buys

The measurement that sets the budget: run the coarse stage and stage 1 for every pair of a set,
and ask where the pose within 5 degrees and 0.5 `t` of the truth actually sits. On pots A, B, C
and H and on `synthetic_pingsdorf_20`, with `stage1` at its old 400:

| set | pairs | correct pose found | rank in the coarse order (p50 / p95 / max) | rank after stage 1 + NMS (p50 / p95 / max) | CPU-s per pair |
|---|---|---|---|---|---|
| pot A | 28 | 11 of 15 | 1 / 136 / 174 | 0 / 0 / 1 | 10.1 |
| pot B | 36 | 13 of 15 | 0 / 193 / 260 | 0 / 24 / 60 | 7.8 |
| pot C | 21 | 4 of 5 | 0 / 129 / 152 | 0 / 0 / 0 | 3.7 |
| pot H | 55 | 8 of 17 | 28 / 228 / 281 | 0 / 5 / 8 | 2.6 |
| synthetic 20 | 190 | 29 of 50 | 0 / 284 / 379 | 0 / 0 / 2 | 15.4 |

**After stage 1, the truth is at rank 0 in the median and never below 6 except on pot B.** Stage 2
was verifying 40 candidates. It now verifies 10, and stage 2 is 93 % of a pair's cost.

In the coarse order the truth is deeper — below 150 in 78–92 % of the pairs that have it at all —
so `stage1` only came down from 400 to 250.

The third knob is the one nobody had looked at: **how finely the breakline is thinned before
frames are paired**. It was a voxel of `t/3`, and the hypothesis count grows as the inverse square
of it. Sweeping it on the two most expensive pots, with the correct-pose count as the guard:

| voxel, `t` | dihedral tol | coarse points | pot A: hyp/pair, CPU-s, found | pot H: hyp/pair, CPU-s, found |
|---|---|---|---|---|
| 1/3 | 40 | 60 | 390 k, 10.15, 10/15 | 73 k, 2.60, 9/17 |
| 0.5 | 40 | 60 | 177 k, 5.55, 12/15 | 36 k, 1.50, 8/17 |
| **0.5** | **25** | **60** | **151 k, 5.14, 12/15** | **28 k, 1.33, 9/17** |
| 0.5 | 25 | 30 | 151 k, 4.02, 8/15 | 28 k, 1.06, 7/17 |
| 0.7 | 25 | 30 | 68 k, 2.50, 10/15 | 15 k, 0.70, 7/17 |

Halving `coarse_points` is the one change that clearly costs recall (12 of 15 down to 8), so it
was rejected; the 0.5 voxel costs nothing and halves the stage.

### 3.1 Then profile one pair and look at what is left

With that budget in place, one typical non-adjacent pair of `mixed_all`
(`Pot_D_Piece_04` against `Pot_E_Piece_20`, 42 k and 26 k faces), single-threaded:

| stage | before the cap | after the cap |
|---|---|---|
| **stage 2, the two coarse ICPs on `pc_reg`** | **7.95 s (73 %)** | **2.84 s (41 %)** |
| stage 2, the two fine ICPs on `pc_frac` | 0.67 | 1.82 |
| coarse score (59 000 hypotheses) | 0.99 | 0.99 |
| stage 1 (250 poses) | 0.56 | 0.55 |
| verify: penetration | 0.29 | 0.30 |
| `MatchData` (both fragments) | 0.20 | 0.20 |
| verify: continuity / fracture / seam | 0.17 | 0.20 |
| `Fragment.load`, hypotheses, both NMS | 0.03 | 0.03 |
| **total** | **10.88 s** | **6.94 s** |

Three quarters of a pair was twenty ICP registrations against a 16 000-point cloud. Those two
ICPs only have to bring the pose from the breakline stage into the basin of the two fine ICPs that
follow on `pc_frac`; a few thousand points does that as well as sixteen. `reg_points` caps the
cloud at 6 000, split between fracture samples and shell margin in their existing proportion —
both are i.i.d. area-weighted draws, so a prefix of each is still area-weighted.

A true pair costs more and spends it differently: `Pot_B_Piece_02` against `Pot_B_Piece_04` is
25.0 s, of which 14.6 in the fine `pc_frac` ICPs and 6.4 in the coarse ones. The fine ICPs
converge slowly precisely because there is something to converge to.

**Stage-2 early rejection does not help here.** `early_reject_tight` at the 0.06 that the
performance note measured as safe dropped **zero** of the ten candidates of that non-adjacent
pair: on this data a hopeless candidate's cheap tight estimate is well above 0.06. It stays off.

**The budget that ships:** `brk_voxel` 0.5, `dihedral_tol` 25, `stage1` 250, `stage2` 10,
`reg_points` 6 000.

| set | before (thin-walls note) | budgeted | wall clock |
|---|---|---|---|
| terracotta | 021–094, 094–104; 007 unplaced | same | 69 s → **40 s** |
| pot A | 87.5 %, precision 1.000 | 87.5 %, 1.000 | 341 s → **83 s** |
| pot B | 100 %, 1.000 | 100 %, 1.000 | 419 s → **89 s** |
| pot C | 50 %, 0.500 | **75 %, 0.667** | 103 s → **35 s** |
| pot G | 0 %, – | 0 %, – | 90 s → **32 s** |
| pot H | 27.3 %, 0.286 | **36.4 %, 0.429** | 507 s → **81 s** |
| synthetic 20 | 80 %, 1.000 | **95 %, 1.000** | 458 s → **295 s** |

Four to six times faster, and better on three of the seven sets. Cross-object joins 0 and group
purity 1.000 everywhere. Most of the quality gain is the dihedral fix rather than the budget — the
frames the hypotheses are built from are right now — but pot C gains its whole extra fragment from
the `reg_points` cap alone, which is not what capping a point cloud is supposed to do: the dense
cloud was letting the coarse ICPs settle into a local minimum the fine ones could not leave.

## 4. Partner search: three of them, none kept

All the numbers in this section were measured before §2, with the biased dihedral. The ceiling in
§4.4 is geometry and does not move with it; the retention figures would improve somewhat and are
not worth re-measuring, because the ceiling is what decides the question.

### 4.1 Breakline signatures — the roadmap's plan

Two fragments that broke apart share the same physical curve where their fracture surfaces meet
the outer shell, so a window of that curve described in a rigid-invariant way should be the same
on both sides. Chain the breakline edges into curves, orient each chain by the frame's own tangent
`t = ns x f` so that the two sides walk it in opposite directions, and describe every window by
its points written in the local frame at the window centre (a stronger invariant than curvature
and torsion, and it needs no second derivatives), by the dihedral profile, and by the local wall
thickness — which is also the length unit, because a rim inflates a fragment's own wall estimate
but both sides of a crack measure the same local thickness.

Three things were learned. **The dihedral had to be taken relative to the window centre**, which
is the observation §2 grew out of: with an absolute profile a true correspondence sat at
descriptor distance 2.23 against 2.33 for a random window pair, pure noise. Centred, it is 0.86
against 1.16. **0.86 against 1.16 is still not enough to search with**: of 90 window pairs that
meet at the ground-truth pose, the correct partner was the nearest of all 31 429 descriptors in
**one**. **Longer windows are worse, not better**, which says the two sides do not describe the
same curve over any distance:

| window, wall thicknesses | 3 | 6 | 10 | 16 |
|---|---|---|---|---|
| adjacent pairs kept at K = 30 | 0.628 | 0.611 | 0.462 | 0.427 |

Demanding that the matched windows agree with each other does not rescue it. Two ways were built —
binning matches by (chain of A, chain of B, `arc_A + arc_B`), which is constant along one seam, and
turning every matched window pair into the rigid transform it implies and scoring the best by
breakline overlap. Best of either, on `mixed_all`: 0.233 of the adjacent pairs at K = 5, 0.583 at
K = 30. The code is `tools/breakline_signature.py`; it costs 3.1 s for 164 fragments, so it is
very cheap and simply not informative.

### 4.2 The matcher's own coarse stage

`matching.screen_pair` runs the coarse stage on a capped subsample of both breaklines: 0.17
CPU-seconds a pair, all 12 589 pairs of `mixed_all` in 278 s on 9 workers. Seven statistics were
computed from one pass. The best is the one that adds the stage-1 breakline ICP on the top 20
poses, which costs almost nothing extra:

| statistic | K = 10 | K = 15 | K = 20 | K = 25 | K = 30 |
|---|---|---|---|---|---|
| best coarse score | 0.243 | 0.333 | 0.424 | 0.493 | 0.538 |
| largest cluster of agreeing poses | 0.243 | 0.333 | 0.396 | 0.438 | 0.479 |
| **after the stage-1 breakline ICP** | **0.361** | **0.465** | **0.528** | **0.587** | **0.642** |
| ... normalised per fragment row | 0.396 | 0.521 | 0.569 | 0.601 | 0.656 |
| pairs kept | ~1 350 | ~1 950 | ~2 500 | ~3 050 | ~3 570 |

Tripling the screen's resolution (`screen_points` 150 to 350, three times the cost) moves the
stage-1 statistic from 0.646 to 0.642 at K = 30. On the synthetic 170 there is no signal at all:
the median statistic of adjacent pairs equals the median of non-adjacent pairs to three decimals
for every one of the seven, and the best retention at K = 30 is 0.312.

### 4.3 Object features

Cross-object pairs are 91 % of `mixed_all`, so splitting the collection into vessels first would be
worth more than any seam search. Two cheap rigid invariants per fragment, the wall thickness and
the radius of curvature of the outer shell:

| pot | A | B | C | D | E | F | G | H | I | J |
|---|---|---|---|---|---|---|---|---|---|---|
| wall, median | 3.56 | 3.51 | 5.97 | 8.13 | 6.44 | 4.45 | 2.83 | 5.65 | 7.44 | 5.52 |
| shell radius, median | 23 | 24 | 35 | 60 | 46 | 24 | 22 | 48 | 49 | 17 |

Pots A, B, F and G are geometric twins, and within one pot the radius spreads by a factor of five
between body, base and rim. Keeping only pairs whose thicknesses differ by at most 2x and whose
radii by at most 3x leaves 66 % of the pairs and loses 6 % of the adjacent ones — a 1.5x cut.

### 4.4 The ceiling all three hit

Place two ground-truth-adjacent fragments at their exact ground-truth relative pose and measure the
fraction of one breakline that lies within the coarse distance of the other, with agreeing shell
normals. That is the highest score any breakline screen could ever give the pair:

| set | true pair, coverage at the ground-truth pose | non-adjacent pair, best over ~7 000 poses |
|---|---|---|
| `mixed_all` | 0.034 – 0.282 | 0.100 – 0.283 |
| `synthetic_pingsdorf_170` | 0.000 – 0.221 | 0.067 – 0.283 |

The two ranges are the same range. A fragment's breakline borders every one of its neighbours, so
one seam is a tenth to a fifth of it, and the best of several thousand poses reaches that much by
accident. Requiring the matched breakline to be *contiguous* does not separate them either: the
largest connected run of matched points is 3.7–16.1 `t` for adjacent pairs and 4.3–14.4 `t` for
non-adjacent ones, and at the ground-truth pose it is often *shorter* than at the best false pose —
the same inversion the thin-walls note reports for the gap score.

The same ceiling explains why `--stage1-floor` cannot be set to anything useful either. From the
§3 measurement, the best stage-1 score of adjacent pairs against everything else: pot A 0.206
against 0.185 in the median, pot C 0.251 against 0.218, pot H 0.228 against 0.207, synthetic 20
0.069 against 0.042. A floor that cuts half the pairs costs 10–40 % of the adjacent ones. The
mechanism ships (it is exact and free when off) and the default stays 0.

### 4.5 What ships

`--screen-top-k` (default 0, off), with `--screen-points` and `--screen-min-pairs`, and
`--stage1-floor` (default 0, off). Below `screen_min_pairs` (200 pairs) the search never runs, so
every benchmark set below 21 fragments behaves exactly as it did. Above it the user is trading
recall for time at the rate in §4.2, and the README says so.

## 5. Scheduling and memory

`_match_workers` now answers three questions instead of two: how many processes, how many threads
inside a pair, and how many fragments per job block.

| regime | processes | threads per pair | fragments per block |
|---|---|---|---|
| fewer pairs than 4 x workers (every pot, the terracotta) | one per pair, capped at `--workers` | cores / processes | 1 |
| more pairs than that | `--workers` | 1 | 1 |
| more pairs than 16 x workers | `--workers` | 1 | 3 |
| the screening pass | `--workers` | 1 | 3 |

The middle row exists because of a measured regression. Blocking was switched on for every set with
more pairs than 4 x workers at first, and pot H — 55 pairs, 11 fragments, ten blocks over nine
workers — went from **505 s to 689 s**: the block count was so close to the worker count that one
worker had two blocks to chew through while the rest sat idle. Blocks only pay once there are many
more of them than workers, hence the 16x rule. Pot H's result is unchanged either way, which is
also the check that job grouping does not touch what the pipeline decides.

**Full-resolution meshes are streamed.** The pipeline used to read every original scan into a
dictionary before refinement and hold it until the outputs were written; on
`synthetic_pingsdorf_170` that is 358 MB of PLY. `refine_joins` now reads them one at a time and
only for fragments that ended up in a group, and `write_placed_meshes` takes paths.

## 6. The second pass

The budget of §3 is chosen so that the correct pose is almost always inside it, and "almost" is
what a second pass is for. `--second-pass-top K` takes every fragment the assembly left on its own,
picks its `K` best partners by the pair's best stage-1 breakline score — the only number available
for a pair that produced nothing — rematches those pairs with `second_pass_stage1` (400) and
`second_pass_stage2` (40), and reassembles.

Measured on the two sets where the pipeline places least, it recovers nothing. Pot C: 15 pairs
rematched, 3 of 4 fragments placed and precision 0.667 before and after. Pot G: 21 pairs
rematched, 0 of 7 before and after — expected, since pot G's own ground truth interpenetrates
(thin-walls note, §5) and no budget can make a contact-free assembly out of it. It is off by
default. The sets where it could plausibly pay are the ones with fragments left over *and* a
working assembly, which on these benchmarks means pot A (one fragment short) and the synthetic 20
(one short); neither was measured before this session ran out.

## 7. End to end

**Not finished in this session.** The all-pairs run of `mixed_all` with the budgeted defaults was
launched in the background and is projected at about 9 hours on this machine; the run of
`synthetic_pingsdorf_170` follows it. The numbers below are the projection, not the result.

The projection is measured, and it is biased **high**. Pairs are handed out in blocks of three
fragments in collection order, so the run starts with pots A and B — the only full-resolution
meshes in the set, 200 000 faces each — and the first hundred pairs average 23.5 CPU-seconds
against 6.9 for the profiled pair of §3.1. The median pair of the whole set carries 51 000 faces
against 146 000 for the pairs measured so far.

| | measured | projected all-pairs |
|---|---|---|
| the first pairs (pots A and B), flat | 20.7 CPU-s per pair | 72 CPU-h → 8.0 h on 9 workers |
| the same, modelled per pair | – | **41 CPU-h → 4.6 h on 9 workers** |
| the profiled mid-size pair of §3.1 | 6.9 CPU-s per pair | 24 CPU-h → 2.7 h on 9 workers |

The model fits the finished pairs to their two real cost drivers — the hypothesis count and the
size of the fracture clouds — and evaluates them for every pair of the set:
`cost = 8.6e-05 * hypotheses + 3.5e-04 * fracture points + 8.1`, with a 12.6 s rms residual on a
23.9 s mean. The pairs measured so far carry a median of 96 000 hypotheses against 24 500 for the
whole set, which is the whole of the difference between the flat 8.0 h and the modelled 4.6 h.

Against the brief's target of about 2.5 core-seconds per pair, a mid-size pair is at 6.9 and a
full-resolution one at 24. The gap is not scheduling any more; it is the four ICP registrations
per candidate, which is where §8 points next.

### 7.1 The quality gates

| gate | result |
|---|---|
| terracotta: joins 021–094 and 094–104, 007 unplaced | **pass**, on every state of the branch |
| pots A, B, C, G, H and synthetic 20 not worse than the thin-walls table | **pass**, three of them better (§3) |
| cross-object joins 0, group purity 1.000 | **pass** on all seven sets |
| `pytest -q`, 30 tests | **pass** after every source commit |
| determinism | **pass**: two runs of the terracotta give `max |dT|` 0 and `max |d score|` 0 over all 30 candidates, no acceptance flips |
| every existing `Params` field and CLI flag kept, new ones documented | **pass**, README CLI table |

## 8. What is still open

1. **A partner search that is safe does not exist yet, and it will not be built on the breakline.**
   §4.4 measures the ceiling: one seam is 5–22 % of a fragment's crack line and the best of
   several thousand poses reaches that much on pairs that never touched. Anything cheap enough to
   run 12 589 times has to look at the same breakline the matcher looks at, and that breakline is
   the boundary of a fracture mask whose precision is 0.50–0.63 on these pots (thin-walls note,
   §8.1). **The order of the roadmap may be wrong: segmentation precision is the blocker for item
   2 exactly as it is for item 1.** Item 4 (object separation) will not rescue it either — pots A,
   B, F and G are geometric twins in both wall thickness and shell curvature.
2. **All-pairs on 164 fragments is about six hours, not one**, and everything cheap has been
   spent. What is left is the four ICP registrations every stage-2 candidate pays, and two
   directions were opened but not finished here:
   *Progressive stage 2.* The correct pose is at stage-1 rank 0 in the median (§3), so the top
   candidates could be probed first and the remaining seven dropped when none of them is close
   after the coarse ICPs. That would cut stage 2 several-fold on the twelve thousand pairs that
   have nothing to find. It needs the distribution of the post-`pc_reg` tight estimate for true
   against false candidates, which is not measured here.
   *Capping `pc_frac` the way `pc_reg` was capped.* The two fine ICPs are now the largest item
   (26 % of a false pair, 58 % of a true one). The cap is `max_frac_points` at 12 000, and it was
   never swept for its own sake -- the thin-walls note measured the *density*, not the cap.
   `early_reject_tight` is not the answer: at the 0.06 that the performance note measured as safe
   it dropped zero of the ten candidates of the profiled pair.
3. **The macro-normal annulus was measured on two collections, not swept for its own sake.**
   `macro_inner` 0.15 t and `macro_outer` 0.60 t are the flattest point of the table in §2, but
   the inner radius is a length in wall thicknesses applied to a crease whose roundness comes from
   wear and from three Taubin iterations — it should probably be a function of the working mesh's
   own resolution, the way every distance threshold already is. Untested.
4. **`brk_voxel` at 0.5 t was the largest step of the sweep that lost nothing**; 0.7 t was still
   finding 10 of 15 on pot A for a further halving, and the sweep stopped there because the
   correct-pose counts on pot H were falling. It deserves a proper sweep on more sets.
5. **The assembly stage builds a `MatchData` for every fragment in the parent process**, at the
   collection median thickness, so none of them hits the cached arrays: 164 x 0.2 s of
   single-threaded work on a large collection. Building them at each fragment's own thickness
   would make every one a cache hit, but it changes which surface samples the penetration test in
   `assemble` sees, so it was left alone rather than slipped in.
