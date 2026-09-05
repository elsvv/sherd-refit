# Making the pipeline resolution-aware: thin-walled, coarsely meshed pot sherds

**Date:** 2026-09-05. Branch `thin-walls`.

The benchmark note `2026-09-05-sfspp-benchmark.md` showed that the matcher finds
the right relative pose on most true joins of the Structure-from-Sherds++ pots
and that the verifier then throws every one away: at the exact ground-truth
poses, 0 of 62 true joins passed either `gap ≤ 0.065 t` or `tight ≥ 0.25`.  The
thresholds were written in wall thicknesses so that the pipeline would be
scale-free, and they are — but only down to a resolution.  A 3.5 mm pot wall
carries four to six triangles across it, so `0.04 t` of tight contact is 0.15 mm,
well under one triangle edge; the thick terracotta the pipeline was tuned on
carries seventeen.

This note records what was changed, what each change bought, and what still does
not work.

## 0. The resolution unit

Every fragment now measures `res`, the median edge length of its **working**
mesh, and keeps it in the cache.  Every distance threshold became
`max(k · t, m · res)`, resolved once per pair by a `Scales` object built from
`t = min(t_A, t_B)` (a rim inflates the measured thickness; the wall is the
thinner of the two) and `res = max(res_A, res_B)` (the coarser mesh sets the
floor).  The ICP ladder is stretched as a whole rather than floored rung by
rung, so its steps keep their ratios.

The number that matters is `res / t`, the inverse of "edges across the wall":

| set | `t` | working-mesh edge | `res / t` | edges across the wall |
|---|---|---|---|---|
| terracotta `test_fragments_1` | 38.99 – 40.95 | 2.245 – 2.260 | 0.055 – 0.058 | 17 – 18 |
| pot A (full resolution) | 3.50 – 6.01 | 0.296 – 0.417 | 0.065 – 0.119 | 8 – 15 |
| pot C (decimated) | 5.07 – 9.98 | 0.478 – 1.925 | 0.087 – 0.326 | 3 – 12 |
| pot G (decimated) | 2.23 – 4.13 | 0.469 – 0.774 | 0.183 – 0.257 | 4 – 5 |
| pot H (decimated) | 4.41 – 6.69 | 0.562 – 1.556 | 0.092 – 0.253 | 4 – 11 |

**The brief's multipliers had to be cut.**  They were chosen for a terracotta
`res / t` of 0.02 – 0.05, which is the edge of the *original* scan (0.65 units
against `t` = 39, i.e. 60 across the wall).  The working mesh is decimated to
about 600 faces per `t²`, so its edge is 2.25 units and `res / t` is 0.058 —
three and a half times larger.  At the published multipliers every floor except
three would bind on the terracotta, and the run changes for the worse.  Measured
by re-verifying the terracotta's own stored candidate poses:

| multiplier set | tight distance | gap limit | true joins accepted | false joins accepted |
|---|---|---|---|---|
| `k · t` only (shipped before) | 0.040 t | 0.065 t | 2 of 2 | 0 of 4 |
| brief (`tight_res` 1.5, `gap_res` 2.0, …) | 0.087 t | 0.116 t | 2 of 2 | **1 of 4** (007–094, tight 0.14 → 0.56) |
| shipped here | 0.040 t | 0.065 t | 2 of 2 | 0 of 4 |

So the multipliers are set just under `k / 0.058`, which is the largest value
that leaves the terracotta untouched:

| threshold | `k` (in `t`) | `m` (in edges) | floor binds below |
|---|---|---|---|
| coarse breakline score | 0.15 | 2.3 | 15 edges per `t` |
| stage-1 breakline re-score | 0.06 | 0.9 | 15 |
| tight contact | 0.04 | 0.6 | 15 |
| gap limit | 0.065 | 1.0 | 15 |
| seam proximity | 0.12 | 1.8 | 15 |
| penetration depth | 0.06 | 0.9 | 15 |
| ICP ladder (finest rung) | 0.04 | 0.6 | 15 |
| shell-margin radius (continuity) | 0.5 | 4.0 | 8 |
| facing window | 0.3 | 1.0 | 3.3 |

**`facing_res` is deliberately left where it cannot bind.**  The facing window
picks *which* fracture points are compared, not how precisely, and widening it
drags points that face nothing into the median.  On pot G, at the ground-truth
poses, a floor of 4 edges (the brief's value) lifts the median gap of the true
joins from 0.173 t to 0.628 t and drops their median tight contact from 0.42 to
0.08 — it makes the metric worse, not more forgiving.  1.5 edges already costs
0.047 t of gap; 1.0 edge costs 0.001 t and is kept only as a guard.

## 1. What was changed, in the order it was done

| commit | change |
|---|---|
| resolution floors | `res` per fragment in the cache; a `Scales` object turns `(k, m)` into one distance per pair |
| contact metric | `tight`/`gap` measured against the other fragment's fracture **triangles**; fracture samples at a fixed density in `t²` |
| CLI defaults | `--max-gap` had its own copy of the old constant and was overriding the new one |
| sampling cost | the fracture sample capped at 12 000 rather than thinned |
| per-pair thickness | aligned-ray thickness estimate; `min(t_A, t_B)` everywhere; wall-ratio filter 1.5 → 2.5 |
| segmentation | 4 of 7 cone rays instead of 5 once the mesh is coarser than 0.1 `t` |

## 2. The contact metric, before and after

`tight` and `gap` were nearest-neighbour distances between two independent point samples of the
two fracture surfaces. Two samples of one surface never land on each other, so both scores had a
floor equal to the sample spacing. Measured at the **exact ground-truth poses**, on the synthetic
set whose fragments fit each other to 0.001 mm by construction:

| | point-to-point | point-to-surface (fracture triangles) | point-to-surface (whole mesh) |
|---|---|---|---|
| 50 true pairs, median `tight` | 0.03 | **0.66** | 0.73 |
| 50 true pairs, median `gap` (`t`) | 0.079 | **0.003** | 0.002 |
| best candidate per pair, median `tight` | 0.01 | 0.12 | 0.13 |
| best candidate per pair, median `gap` (`t`) | 0.112 | 0.094 | 0.092 |

The old metric reported a 0.079 `t` gap between surfaces that touch. Nothing about the fit
changed; the whole number was sampling noise, and it swamped the difference between a true join
and a false one.

On the terracotta, at the poses the pipeline itself finds:

| pair | | old metric | new metric |
|---|---|---|---|
| 021–094 | true | tight 0.30, gap 0.058 | tight 0.55, gap 0.0083 |
| 094–104 | true | tight 0.28, gap 0.058 | tight 0.58, gap 0.0066 |
| best false, by `tight` | 021–104 | tight 0.17, gap 0.074 | tight 0.13, gap 0.059 |
| best false, by `gap` | 007–094 | tight 0.14, gap 0.078 | tight 0.15, gap 0.040 |
| separation, true / false | | 1.7× on `tight`, 1.3× on `gap` | **3.7× on `tight`, 4.8× on `gap`** |

**Whole mesh against fracture-only.** The team lead's note pointed at the fragment's existing
scene, which covers the whole working mesh. The two agree within 0.005 `t` on every pose measured
here, so the fracture-only scene costs nothing — and it cannot reward a fragment laid flat against
its neighbour's outer shell, which the whole-mesh form would score as perfect contact. The
fracture-only scene is what ships.

**Recalibration.** The new quantity is five to twenty times smaller, so the two constants that
measure it had to move: tight contact 0.04 → 0.01 `t`, gap limit 0.065 → 0.03 `t`, with floors of
0.15 and 0.45 edges — the same crossover at 15 edges per wall as every other threshold. Nothing
else in `accept` changed. The thresholds were read off the distributions, not guessed: on the
terracotta the true joins put 52 % of their facing points within 0.01 `t` and the false pairs put
9–14 %.

## 3. Per-pot results

Fragment accuracy, the same quantity as the SfS++ "Sherd Accuracy": a fragment counts when it
takes part in at least one join that is both a true neighbour and within 5° and 0.5 `t`.

| set | before (main) | resolution floors | + contact metric | + per-pair `t` | + segmentation | SfS++ published |
|---|---|---|---|---|---|---|
| terracotta (4 pieces) | 3 of 4 | 3 of 4 | 3 of 4 | 3 of 4 | **3 of 4** | – |
| pot A (8) | 0 % | 62.5 % | 87.5 % | 75.0 % | **87.5 %** | 100 % |
| pot B (9) | 0 % | 55.6 % | 88.9 % | 88.9 % | **100 %** | 100 % |
| pot C (4 scorable) | 0 % | 0 % | 50 % | 50 % | **50 %** | 100 % |
| pot G (7) | 0 % | 0 % | 0 % | 0 % | **0 %** | 100 % |
| pot H (11) | 54.5 % | 0 % | 36.4 % | 18.2 % | **27.3 %** | 90.9 % |
| synthetic 20 | – | – | 85 % | 95 % | **80 %** | – |

Precision of the joins the assembly used:

| set | before | floors | + metric | + per-pair `t` | + segmentation |
|---|---|---|---|---|---|
| pot A | – (no joins) | 0.600 | 1.000 | 0.833 | **1.000** |
| pot B | – | 0.500 | 0.857 | 1.000 | **1.000** |
| pot C | – | 0.000 | 0.333 | 0.500 | **0.500** |
| pot H | 0.571 | 0.000 | 0.375 | 0.143 | **0.286** |
| synthetic 20 | – | – | 1.000 | 1.000 | **1.000** |

**Cross-object joins: 0 in every run.** Group purity is 1.000 everywhere. The failures are wrong
poses on genuinely neighbouring pairs, never a pairing of pieces that do not belong together.

## 4. Segmentation, measured against the SfS++ surface ground truth

`tools/eval_segmentation.py` calls a working-mesh face **shell** when its centroid lies within
0.3 `t` of either of the two surface point sets the dataset ships, and **fracture** otherwise, then
scores our mask area-weighted. `over` is the share of the fragment's area that is intact shell and
we call fracture — the quantity that actually hurts, because the matcher aligns whatever is
labelled fracture.

Base (5 of 7 cone rays, what shipped) and the four candidates, on the two full-resolution pots and
the three decimated ones:

| variant | pot A p / r / over | pot B | pot C | pot G | pot H |
|---|---|---|---|---|---|
| base, 5 of 7 | 0.524 / 0.884 / 7.3 % | 0.483 / 0.996 / 6.5 % | 0.610 / 0.845 / 8.1 % | 0.474 / 0.868 / 11.5 % | 0.520 / 0.786 / 13.2 % |
| (i) smoothed hit normal | 0.525 / 0.884 / 7.2 % | 0.479 / 0.996 / 6.6 % | 0.611 / 0.845 / 8.0 % | 0.472 / 0.873 / 11.6 % | 0.522 / 0.784 / 13.1 % |
| (ii) smoothing radius max(t/3, 3 res) | 0.524 / 0.884 / 7.3 % | 0.480 / 0.996 / 6.5 % | 0.599 / 0.863 / 8.5 % | 0.455 / 0.888 / 12.3 % | 0.513 / 0.784 / 13.6 % |
| **(iii) 4 of 7 when res > 0.1 t** | **0.546 / 0.877 / 6.9 %** | **0.502 / 0.994 / 6.3 %** | **0.631 / 0.825 / 7.3 %** | **0.497 / 0.844 / 10.3 %** | **0.544 / 0.760 / 11.6 %** |
| (iv) boundary angle from normal noise | 0.524 / 0.884 / 7.3 % | 0.483 / 0.996 / 6.5 % | 0.610 / 0.845 / 8.1 % | 0.474 / 0.868 / 11.5 % | 0.520 / 0.786 / 13.2 % |

Only (iii) is kept. It raises precision on all five pots and cuts the wrongly-labelled shell area
by about a fifth, for one to three points of recall. It is conditional on the resolution, so the
terracotta (0.058 `t` per edge) is untouched — its masks come out byte-identical.

**What did not help, and why.**

- **(i) the smoothed normal of the hit face** moves nothing anywhere (under 0.005). The test asks
  whether the far face points back along the ray; that answer is already robust, because the
  working mesh has had three Taubin iterations before segmentation runs.
- **(ii) a smoothing radius of max(t/3, 3 res)** is worse on all three decimated pots (C 0.610 →
  0.599, G 0.474 → 0.455, H 0.520 → 0.513). On a coarse mesh 3 res is 0.55–0.78 `t`, well past
  `t/3`, and the widened neighbourhood flattens the very crease the shell test needs.
- **(iv) the boundary angle from the shell's own normal noise** is a no-op *by construction* on
  this data. The rule is `max(25°, median angle(raw, smoothed) + 15°)`, and the median angle after
  Taubin smoothing is 1.0–4.4° on almost every piece (worst single fragment 9.6°), so the rule
  never clears the 25° already in use. Measured per fragment on pots A, G and H; every one gave
  exactly 25°.

Vote counts below 4 were also measured. 3 of 7 gives higher precision still (A 0.563, C 0.651,
G 0.514, H 0.567) but pays four points of recall on pot G, and three rays out of seven is a thin
majority to trust; 4 is what ships.

## 5. Why pot G cannot pass, and it is not a threshold

At its own ground-truth poses, pot G's pieces **interpenetrate**. Fraction of surface samples
sitting inside the neighbour, over the 10 true pairs, median:

| deeper than | 0.06 `t` | 0.1 `t` | 0.2 `t` | 0.3 `t` | 0.5 `t` |
|---|---|---|---|---|---|
| pot G | 0.053 | 0.047 | 0.036 | 0.026 | 0.013 |
| pot A | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 |

The deepest excursion is 0.88 `t` in the median pair and 1.41 `t` in the worst. The acceptance
limit is 0.005, and no scaling of the counted depth reaches it: even at half a wall thickness,
1.3 % of the surface is inside the neighbour. Pot G's ground truth is not a contact-free
assembly, so a pipeline that refuses to place fragments that interpenetrate cannot reproduce it.
Everything else about pot G is now in range — at the ground-truth poses its true joins score
`tight` 0.42 and `gap` 0.173 `t` against a limit of 0.26, and the seam gate that used to reject
all ten of them now passes.

## 6. Pot H, and where the remaining errors are

Pot H is the one set that is worse than before this branch (54.5 % → 27.3 %). Three of its five
wrong joins are near-misses: 2.4°, 2.7° and 1.7° of rotation with the neighbour slid 0.53, 0.57
and 0.62 `t` along the seam, against a tolerance of 0.5 `t`. Two are genuinely wrong, at 19°.
Pot H also has the worst segmentation of the five (26 % of area labelled fracture against a ground
truth of 21 %, `over` 11.6 %), and it is the pot where the matcher's own poses fit the labelled
fracture *better than the truth does*: median `gap` 0.016 `t` at the candidate pose against
0.168 `t` at the ground-truth pose.

That inversion is the single fact that explains every remaining failure. Measured at the
ground-truth poses against the best candidate per pair:

| pot | `gap` at ground truth | best candidate, true pair | best candidate, non-adjacent pair |
|---|---|---|---|
| A | 0.042 | 0.010 | 0.037 |
| C | 0.044 | 0.129 | 0.100 |
| G | 0.159 | 0.117 | 0.121 |
| H | 0.168 | 0.016 | 0.063 |

Only on pot C does the truth score best. On A, G and H a wrong pose fits the labelled fracture
surface better than the right one, so **no threshold can separate them** — the ranking itself is
inverted, and the cause is the shell arcs still inside the fracture mask. Segmentation precision
is now 0.50–0.63 on the decimated pots; it would have to be much closer to 1 before the ordering
flips back.

## 7. Runtime

Matching time per pot, seconds, against the totals recorded for the main branch in the benchmark
note:

| set | main (total) | this branch (total) | matching alone |
|---|---|---|---|
| terracotta | 82 s | 68 s | 42 s |
| pot A | 217 s | 345 s | 335 s |
| pot B | 197 s | 429 s | 421 s |
| pot C | 57 s | 105 s | 102 s |
| pot G | 11 s | 90 s | 88 s |
| pot H | 694 s | 505 s | 501 s |

**The ~30 % budget is met on the terracotta and pot H and missed on the rest**, and the reasons are
the changes themselves, not waste. Two of them:

- The resolution floors let candidates survive to full verification that used to be discarded at
  the coarse breakline gate. Pot G is the extreme: it finished in 11 s on main because *nothing*
  survived to be scored, so its old figure is not a runtime to compare against.
- The wall-ratio filter at 2.5 instead of 1.5 admits every pair on pots A and B, 28 and 36 instead
  of 21 and 28. That is a third more work on A and a quarter more on B, and it is what makes their
  rim pieces placeable at all.

The fracture sampling was tuned against this budget rather than for its own sake. At 150 points
per `t²` uncapped the largest sherds took 23k–49k points and matching ran 45 % to 226 % longer;
capping the sample at 12 000 gives the same placement everywhere and brings pot A from 307 s back
to 265 s. Cutting the density to 50 per `t²` is cheaper still (pot A 149 s) but costs placement:
pot A falls to 5 of 8, pot C to 0 of 4, the synthetic set to 16 of 20. The scores at a fixed pose
do not move with the count — it is the ICP that averages over the correspondences.

## 8. What is still open

1. **Segmentation precision on decimated meshes, 0.50–0.63.** This is the blocker for pots A, G
   and H, in the precise sense of §6: until the fracture mask stops including arcs of intact
   shell, the matcher will keep finding poses that beat the truth on our own score. The vote
   change bought 0.02–0.03; the remaining gap is much larger than anything the four candidates
   tried here can close.
2. **Pot G's ground truth interpenetrates** (§5). Unreachable without dropping the penetration
   test, which is the main defence against false joins.
3. **The synthetic set regressed with the vote change**, 19 of 20 to 16 of 20, while its precision
   stayed at 1.000 — it lost joins rather than gaining wrong ones. Eight of its twenty fragments
   sit at 0.103–0.129 `t` per edge, just over the 0.1 `t` crossover, so they take the 4-vote rule
   while being nearly full-resolution. Either the crossover is slightly too low, or the rule
   should fade in rather than switch; both are guesses until measured on a set that spans the
   crossover densely.
4. **Runtime on pots B and G** (§7).
5. **`mixed_all` is still not run.** At the per-pair cost measured here it remains a multi-hour
   job.
