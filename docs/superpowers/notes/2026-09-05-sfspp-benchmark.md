# Structure-from-Sherds++ as a real benchmark — the pipeline scores 0 on four pots of five

**Date:** 2026-09-05.

**Verdict in one sentence: the matcher finds the right relative pose on most true
joins and the verifier then throws every one of them away, because on thin-walled
pots the tight-contact and gap thresholds are tighter than the point sampling can
resolve — at the *exact ground-truth pose* 0 of 62 true joins across the five pots
run here pass either threshold.**

With the two thresholds loosened once for the whole set (`--min-tight 0.10
--max-gap 0.11`) and nothing else changed, pot C goes from 0 % to the published
100 %, pot A from 0 % to 87.5 % and pot B from 0 % to 66.7 %, with no false pairing
in any of them (§4.1). The matcher and the assembly are sound on this data; the
acceptance test is what is not scale-free.

Nothing under `sherd_refit/` was changed. Every number below comes from the
shipped CLI and the shipped scoring functions.

Tools added by this work: `tools/stage_sfspp.py` (staging + a ground-truth
sanity check) and `tools/evaluate.py` (scoring against `ground_truth.json`).

## 1. The dataset

Structure-from-Sherds++ (`SfS_pp`), ten wheel-thrown pots, 164 fragment meshes,
staged into `input/sfspp/pot_<X>/` plus `input/sfspp/mixed_all/`. Units are
millimetres. Licence CC BY-NC-SA 4.0; cite Yoo, Liu, Arshad, Kim, Kim,
Aloimonos, Fermüller, Joo, Kim and Hong, *Structure-From-Sherds++*,
arXiv:2502.13986, 2025, <https://sj-yoo.info/sfs/>. The data stays local:
`input/` is gitignored and the staging tool symlinks the meshes.

| pot | meshes | with GT pose | GT adjacent pairs | faces per mesh | median edge (mm) | wall `t` (mm) | per-fragment `t` | `t` / edge |
|---|---|---|---|---|---|---|---|---|
| A | 8 | 8 | 15 | 20 728 – 200 124 | 0.39 | 3.63 | 3.50 – 6.01 | 9.2 |
| B | 9 | 9 | 15 | 25 590 – 157 281 | 0.37 | 3.50 | 3.11 – 5.59 | 9.4 |
| C | 7 | 4 | 5 | 7 158 – 50 756 | 0.88 | 5.90 | 5.07 – 9.98 | 6.7 |
| D | 32 | 28 | 69 | 4 558 – 48 556 | 0.80 | 8.17 | 4.95 – 11.93 | 10.2 |
| E | 34 | 31 | 62 | 6 048 – 49 216 | 0.76 | 6.46 | 4.30 – 12.13 | 8.4 |
| F | 7 | 6 | 9 | 6 266 – 29 660 | 0.67 | 4.51 | 4.26 – 5.38 | 6.7 |
| G | 7 | 7 | 10 | 8 396 – 37 378 | 0.58 | 2.79 | 2.23 – 4.13 | 4.8 |
| H | 11 | 11 | 17 | 7 614 – 41 958 | 0.81 | 5.69 | 4.41 – 6.69 | 7.0 |
| I | 30 | 27 | 65 | 5 470 – 45 708 | 0.88 | 7.48 | 4.01 – 8.82 | 8.5 |
| J | 19 | 11 | 21 | 4 914 – 44 120 | 0.76 | 5.43 | 4.73 – 9.37 | 7.1 |

`t` is the collection median our own preprocessing measures; the per-fragment
column is its spread within the pot. For comparison, `input/test_fragments_1` —
the thick terracotta the pipeline was tuned on — has `t` = 38.6 units against a
0.65-unit median edge, so **`t` / edge = 59 there against 4.8 – 10.2 here.** That
ratio, not the absolute wall thickness, is what the thresholds depend on.

Pots A and B ship full-resolution meshes (`_Mesh.obj`); C–J ship decimated ones
(`_Mesh_DS.obj`), which is why their edges are roughly twice as long. `--target-faces`
cannot buy resolution back: every SfS++ mesh already sits below the 200 000-face
budget, so no decimation happens at all (the logs read `16972 faces (from 16972,
budget 200000)`).

## 2. How the ground truth was read, and the check that it is right

`Ground Truth/Pot_<X>_Piece_<n>_T.txt` is a 4×4 matrix that maps the piece's own
file coordinates **into** the assembled frame. The direction was verified rather
than assumed (`python tools/stage_sfspp.py --verify`): applied forward, every pair
the adjacency graph calls neighbouring comes into contact, and non-neighbouring
pairs of the same pot do not.

| pot | adjacent pairs | median min-distance (mm) | worst | non-adjacent pairs | median min-distance (mm) |
|---|---|---|---|---|---|
| A | 15 | 0.027 | 0.043 | 13 | 75.1 |
| B | 15 | 0.015 | 0.046 | 21 | 59.5 |
| C | 5 | 0.074 | 0.102 | 1 | 12.1 |
| D | 69 | 0.056 | 0.496 | 309 | 103.2 |
| E | 62 | 0.051 | 10.879 | 403 | 78.1 |
| F | 9 | 0.056 | 0.069 | 6 | 34.1 |
| G | 10 | 0.060 | 0.092 | 11 | 15.8 |
| H | 17 | 0.057 | 0.130 | 38 | 50.2 |
| I | 65 | 0.066 | 4.996 | 286 | 69.3 |
| J | 21 | 0.068 | 0.247 | 34 | 35.3 |

Inverting the matrices instead scatters pot A over an 872 × 380 × 610 mm box and
puts the closest "adjacent" pair 26 mm apart, so the direction is settled.

Two caveats the table shows. The graph is a *simple* adjacency, not a contact
list, so a few non-adjacent pairs also touch (pot B has one at 0.015 mm, pot D one
at 0.041 mm) — pieces can graze without sharing a seam. And a few graph edges are
not in contact at all (pot E has one at 10.9 mm, pot I one at 5.0 mm), so a recall
of exactly 1.0 is unreachable on those two pots.

**Pieces without ground truth.** Several pots ship more meshes than the ground
truth covers. Those pieces have no `_T.txt` and are either an all-zero row of the
adjacency matrix or beyond its size: C 3 of 7, D 4 of 32, E 3 of 34, F 1 of 7,
I 3 of 30, J 8 of 19 (142 of 164 pieces have a pose). They are staged anyway —
they are part of the collection a real user would hand over — and listed under
`unknown` in `ground_truth.json`. Joins touching them are counted in their own
bucket and never scored.

## 3. Results

Five pots were run with the default parameters, `sherd-refit run input/sfspp/pot_<X>
--out output/sfspp/pot_<X>`, on a 10-core machine (9 worker processes).

| pot | pieces | joins used | correct | wrong pose | non-adjacent | GT pairs | precision | recall | fragment accuracy | SfS++ published | runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 8 | 0 | 0 | 0 | 0 | 15 | – | 0.00 | 0 / 8 = 0 % | 100 % | 3 min 37 s |
| B | 9 | 0 | 0 | 0 | 0 | 15 | – | 0.00 | 0 / 9 = 0 % | 100 % | 3 min 17 s |
| C | 7 | 0 | 0 | 0 | 0 | 5 | – | 0.00 | 0 / 4 = 0 % | 100 % | 57 s |
| G | 7 | 0 | 0 | 0 | 0 | 10 | – | 0.00 | 0 / 7 = 0 % | 100 % | 11 s |
| H | 11 | 7 | 4 | 3 | 0 | 17 | 0.57 | 0.24 | 6 / 11 = 55 % | 90.9 % | 11 min 34 s |

"Fragment accuracy" is a fragment with at least one correct join, over the
fragments that have a ground-truth pose — the same quantity as the SfS++ "Sherd
Accuracy", so the last two columns are comparable. A join is correct when the pair
is really adjacent *and* the relative pose is within 5° and 0.5 `t`.

Four of the five pots produce **no assembly at all**: every fragment ends in its
own group with the report line "not assembled (no confident join)". Pot H builds
one group of eight fragments, and every join it uses is a genuinely adjacent pair —
there is not a single false pairing anywhere in these five runs — but three of the
seven place the neighbour with a slide of 0.67 – 0.97 `t` along the seam.

Pot H's used joins:

| A | B | verdict | rot | trans (`t`) | seam | tight A/B | gap | pen |
|---|---|---|---|---|---|---|---|---|
| 08 | 10 | correct | 0.1° | 0.44 | 12.0 | 0.50 / 0.47 | 0.042 | 0.0000 |
| 07 | 08 | correct | 2.0° | 0.46 | 10.7 | 0.43 / 0.42 | 0.046 | 0.0000 |
| 09 | 10 | correct | 1.7° | 0.41 | 8.7 | 0.53 / 0.48 | 0.041 | 0.0000 |
| 03 | 05 | correct | 3.6° | 0.48 | 8.3 | 0.38 / 0.29 | 0.058 | 0.0000 |
| 09 | 11 | wrong pose | 28.5° | 0.97 | 11.3 | 0.65 / 0.50 | 0.040 | 0.0000 |
| 03 | 07 | wrong pose | 1.2° | 0.87 | 9.0 | 0.41 / 0.31 | 0.056 | 0.0000 |
| 03 | 04 | wrong pose | 3.1° | 0.67 | 9.0 | 0.31 / 0.27 | 0.059 | 0.0000 |

The three wrong ones are not marginal calls by the metric: placing pieces 03 and 04
at the pipeline's pose leaves 151 of 30 000 sampled vertices within 0.5 mm of the
neighbour, against 1 352 at the ground-truth pose; for 09 – 11 it is 190 against
1 008. The pipeline's pose really is the worse fit, and it scores better only on
the labelled fracture points (§4.3).

### The translation half of the pose test

`tools/evaluate.py` measures the translation error as the displacement of the
fragment's **centroid**, not of the two matrices' translation columns. These meshes
sit 300 – 500 mm from their file origin, so measured at the origin a 0.6° rotation
error alone shows up as 4 mm of "translation" and every correct pose fails. The
origin form is still recorded as `trans_origin_t` and `--translation origin`
restores it as the thresholded quantity.

## 4. Failure modes, in the order they bite

### 4.1 The verification thresholds are unreachable — the fatal one

Take every true adjacent pair, place the two fragments at their **exact
ground-truth relative pose**, and run the pipeline's own `verify()` on them. No
matcher can do better than that, so any threshold the true joins fail here can
never be met on this data.

| pot | true pairs | pass `gap ≤ 0.065` | pass `tight ≥ 0.25` | pass `seam ≥ 3` | pass `pen ≤ 0.005` | pass `normal agr. ≥ 0.8` | pass all five |
|---|---|---|---|---|---|---|---|
| A | 15 | **0** | **0** | 14 | 10 | 13 | **0** |
| B | 15 | **0** | **0** | 15 | 13 | 14 | **0** |
| C | 5 | **0** | **0** | 5 | 3 | 5 | **0** |
| G | 10 | **0** | **0** | 0 | 0 | 9 | **0** |
| H | 17 | **0** | **0** | 3 | 0 | 15 | **0** |

Median over the true pairs at the ground-truth pose:

| pot | gap (`t`) | best gap | tight | best tight |
|---|---|---|---|---|
| A | 0.120 | 0.076 | 0.05 | 0.14 |
| B | 0.105 | 0.070 | 0.06 | 0.17 |
| C | 0.105 | 0.099 | 0.05 | 0.07 |
| G | 0.178 | 0.146 | 0.02 | 0.03 |
| H | 0.176 | 0.151 | 0.03 | 0.09 |

The thresholds are `gap ≤ 0.065` and `tight ≥ 0.25`. The *best* true join in the
whole set reaches gap 0.070 and tight 0.17.

**Why.** Both scores are nearest-neighbour distances between two independent
30 000-point area-weighted samples of the two surfaces (`matching.py`
`fracture_scores`). For two samples of the same surface at density λ the median
nearest-neighbour distance is about `0.5 / sqrt(λ)`, which in units of `t` is
`0.5 · sqrt(area) / (t · sqrt(30000))` — it grows with the fragment's size-to-wall-thickness
ratio. Measured per fragment, that floor is 0.075 `t` on pot A and 0.069 `t` on
pot B, against 0.043 `t` on the terracotta test set. `tight` counts points closer
than `0.04 t`, which is 0.145 mm on pot A — well under one 0.39 mm triangle edge,
so a large fraction can never land there.

The reference set passes only because it is thick-walled, and it passes with almost
no margin: its two accepted joins score tight 0.27 / 0.28 against the 0.25
threshold and gap 0.057 / 0.060 against 0.065, i.e. by 8 – 12 %. The thresholds are
not scale-free in practice, even though they are written in units of `t`.

**Confirmation by relaxing them.** Re-running three pots with `--min-tight 0.10
--max-gap 0.11` and nothing else changed:

| pot | joins used | correct | wrong pose | non-adjacent | cross-object | precision | recall | fragment accuracy | default | SfS++ published | runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 5 | 5 | 0 | 0 | 0 | 1.00 | 0.33 | **7 / 8 = 87.5 %** | 0 % | 100 % | 4 min 16 s |
| B | 7 | 5 | 2 | 0 | 0 | 0.71 | 0.33 | **6 / 9 = 66.7 %** | 0 % | 100 % | 3 min 46 s |
| C | 3 | 3 | 0 | 0 | 0 | 1.00 | 0.60 | **4 / 4 = 100 %** | 0 % | 100 % | 1 min 05 s |

Pot C reaches the published SfS++ figure exactly, and pot A reaches 7 of 8 — which
is the ceiling left by §4.2, since piece 01 has no matchable pair at all. Still no
false pairing appears: 15 of the 15 joins used across the three runs are on
genuinely adjacent pairs, and every group is pure. The two wrong ones on pot B are
the same pair of seams solved with a 9° twist.

So the loosened thresholds do not buy accuracy with false positives here — they
simply stop discarding the answer. That is not a recommendation to ship these two
numbers as defaults; it is the evidence that the thresholds, not the matcher, are
what fails on thin walls. A defensible fix would make both scores independent of
the sampling density (a point-to-*surface* distance rather than
point-to-point-sample, or a sample count proportional to fracture area in `t²`
rather than a flat 30 000 per fragment) instead of moving the constants.

Relaxing them does nothing for pot G, whose candidates never reach verification
(§4.4).

### 4.2 The pot's mouth wrecks the thickness estimate and skips the pairs

The fragment carrying the pot's mouth has a thicker collar, and the modal
inward-ray distance latches onto it: pot A piece 01 measures `t` = 6.01 against a
collection median of 3.63, pot B piece 01 measures 5.59 against 3.50. The
pipeline then refuses to match any pair whose wall thicknesses differ by more than
1.5× (`pipeline.py:130-138`). On pots A and B the odd piece out is the largest in
the pot and the best connected in the ground truth, so the loss is heavy:

| pot | piece | own `t` | median `t` | pairs skipped | GT adjacent pairs lost | recall ceiling |
|---|---|---|---|---|---|---|
| A | 01 | 6.01 | 3.63 | 7 of 28 | 6 of 15 | 9 / 15 |
| B | 01 | 5.59 | 3.50 | 8 of 36 | 6 of 15 | 9 / 15 |
| C | 05, 07 | 9.83, 9.98 | 5.90 | 10 of 21 | 0 of 5 | 5 / 5 |
| G | 07 | 4.13 | 2.79 | 3 of 21 | 1 of 10 | 9 / 10 |
| H | 01 | 6.69 | 5.69 | 1 of 55 | 1 of 17 | 16 / 17 |

Pot A therefore starts with a recall ceiling of 9/15 and can never place piece 01,
before any matching happens; that is exactly the fragment the loosened run of §4.1
fails to place. On pots C, G and H the same filter costs little, and on C it is
arguably right — pieces 05 and 07 have no ground-truth pose and are twice the
pot's wall thickness. The same mouth also fools the segmentation: piece 01's
fracture surface comes out in four connected components totalling 5 646 mm², of
which only the 1 973 mm² one is the real break band — the other three are rings
around the mouth, 65 % of the labelled fracture area. Its reported fracture
fraction is 24.3 % against 9 – 11 % for the pot's other mid-size pieces.

### 4.3 On the decimated meshes the fracture band grows over the shell

Segmentation on the thin sherds is *mostly* right: `preview_segmentation.png` for
pots A, B and C shows a narrow red band that follows the break edges, which is the
correct answer for a 3.5 mm wall. But the band gets much wider on the decimated
pots:

| pot | median fracture area | range | fragments over 25 % |
|---|---|---|---|
| A | 10.4 % | 9.1 – 24.3 | 0 / 8 |
| B | 9.0 % | 6.3 – 22.1 | 0 / 9 |
| C | 22.2 % | 12.2 – 38.8 | 3 / 7 |
| D | 23.2 % | 10.7 – **100.0** | 15 / 32 |
| E | 23.7 % | 10.1 – 52.4 | 16 / 34 |
| F | 26.8 % | 22.3 – 48.0 | 5 / 7 |
| G | 22.2 % | 10.1 – 41.9 | 3 / 7 |
| H | 25.5 % | 17.2 – 54.8 | 8 / 11 |
| I | 22.0 % | 13.3 – 58.4 | 12 / 30 |
| J | 31.8 % | 10.8 – 75.5 | 14 / 19 |

The reference terracotta runs 12 – 20 %. On pot D one fragment is labelled 100 %
fracture — no shell at all. Pot H's preview shows the shape of the error: wide red
*arcs across the middle of the sherd faces*, not just the perimeter, so this is the
shell test failing on intact surface rather than a slightly generous boundary.

The shell test accepts a face when a cone of rays hits the opposite wall at
0.5 – 1.8 `t`, which assumes the wall is locally as thick as the fragment's modal
`t`. Pot H piece 10 has a modal `t` of 6.51 mm but a median inward-ray distance of
8.39 mm, and 54.8 % of its area is called fracture. **The cause is not isolated
here:** the eight affected pots are exactly the eight that ship decimated meshes,
so coarser triangles and a more variable wall are confounded, and this note does
not separate them. What is certain is the split: the two full-resolution pots sit at
a 9 – 10 % median, near the reference set's 12 – 20 %, and all eight decimated ones
sit at 22 – 32 %.

This is what produces pot H's three wrong-pose joins. The matcher aligns the
labelled fracture points, and when those include long arcs of intact shell it can
find a pose that scores far better on them than the truth does: for pieces 08 – 10
the pipeline's pose scores gap 0.042 and tight 0.50/0.47 where the ground-truth
pose scores gap 0.161 and tight 0.09/0.09, while the raw mesh contact goes the
other way (1 193 vertices within 0.5 mm against 1 691).

### 4.4 On the thinnest pot no candidate is even generated

Pot G (`t` = 2.79 mm, median edge 0.58 mm) finished matching in 2.8 seconds
because nothing survived to be scored. Of its 18 matched pairs, 8 logged "nothing
passed the coarse stage" and the other 10 produced 0 candidates for full ICP. The
two gates are distance thresholds far below the mesh resolution: the coarse
breakline test uses `0.15 t` = 0.42 mm and the stage-1 breakline score uses
`0.06 t` = 0.17 mm, against a 0.58 mm triangle edge, so breakline points quantised
to edge midpoints essentially cannot land inside them. Pot H shows the same gate
biting more mildly — 3 of its 54 pairs returned zero candidates.

Relaxing the verification thresholds cannot help pot G: the candidates never reach
verification.

### 4.5 Watertightness

Pot B has 3 of 9 fragments non-watertight (pieces 02, 07 and 08) and pot C 1 of 7,
which disables the penetration test for every pair that touches them. No wrong join
in any run here came from such a pair — pot B's two wrong-pose joins are 04 – 06 and
06 – 09, all three watertight — but it removes the main defence against a false join,
and it matters more the further the other thresholds are loosened.

## 5. What was not run

**`mixed_all` was skipped.** Pot H alone took 11 min 34 s, well over the 3-minute
budget that would have justified the mixed run. The full collection is 164
fragments = 13 366 pairs, of which 8 232 survive the 1.5× wall-thickness filter.
At the per-pair cost measured here — 40 s of CPU per pair on pot C, 87 s on pot A,
110 s on pot H — that is **10 to 28 hours** on 9 processes, against a 2-hour
budget. The folder and its combined `ground_truth.json` are staged and ready
(`input/sfspp/mixed_all/`), and `tools/evaluate.py` already reports cross-object
joins and group purity, so the run only needs the pairwise cost to come down.

Pots D, E, F, I and J were not assembled either; they were preprocessed with
`sherd-refit segment` for the facts in §1 and §4.3. D, E and I are 28 – 31 pieces,
which at pot H's rate is 6 – 8 hours each.

## 6. Reproducing

```bash
python tools/stage_sfspp.py                       # stage all ten pots + mixed_all
python tools/stage_sfspp.py --verify              # the ground-truth check of §2
sherd-refit run input/sfspp/pot_A --out output/sfspp/pot_A
python tools/evaluate.py output/sfspp/pot_A input/sfspp/pot_A

# the loosened-threshold run of §4.1
sherd-refit run input/sfspp/pot_A --out output/sfspp/relax_pot_A \
    --min-tight 0.10 --max-gap 0.11
python tools/evaluate.py output/sfspp/relax_pot_A input/sfspp/pot_A
```

`tools/evaluate.py` writes `OUT_DIR/evaluation.json` with every join bucketed, the
per-object breakdown, group purity, and the true adjacent pairs whose best
candidate was rejected together with the scores that rejected them.
