# A synthetic benchmark from a real digitized vessel

**Date:** 2026-09-05. Generator: `tools/make_synthetic.py`. Data: `input/synthetic_pingsdorf_{20,60,170}/`
(gitignored).

## The source object

A photogrammetry scan of a real pot: **074 Tongefäß / Potter Vessel**, red-painted Pingsdorf-type
earthenware from Meschede, Stiftskirche St. Walburga (Westphalia), made at Brühl-Pingsdorf around
900 AD, dendro-dated 897–913. Published by LWL-Archäologie für Westfalen.

| | |
|---|---|
| url | <https://doi.org/10.5281/zenodo.10332909> |
| licence | CC BY 4.0 |
| author | LWL-Archäologie für Westfalen / Florian Westphal (Nikon D850, RealityCapture) |
| mesh | GLB, 999 916 faces, watertight, one component |
| native size | 216 × 180 × 195 mm, wall 4.02 mm (mode of the inward-ray distance) |

The scan captured the inside of the pot through its mouth, so the solid enclosed by the mesh **is
the clay**: voxelizing the mesh interior gives the vessel wall directly, with the real, slightly
uneven thickness of the original (the base is about 12 mm against an 8 mm wall). No thickening or
offsetting was needed. The outer surface carries the throwing rings and the photogrammetry noise of
the original scan, which the fragments inherit.

Intruder fragments are cut from a second scan by the same team, **049 Kelch / Goblet**,
<https://doi.org/10.5281/zenodo.10354385>, also CC BY 4.0.

Both files are in `input/source_models/`. Downloading them needs no account: the Zenodo REST API
serves the GLB directly.

## Method

The whole object is scaled so that its measured wall thickness equals `--wall` (8 mm by default);
for this pot that is ×1986.6 from the file's metres, giving a vessel 430 × 358 × 387 mm. Every
threshold in the pipeline is expressed in `t`, so this fixes the scale of the benchmark.

1. **Occupancy.** `RaycastingScene.compute_occupancy` on a 0.6 mm grid, 724 × 603 × 652 cells,
   evaluated in z-slabs: 16.94 M solid voxels in 24 s. Cached next to the source mesh, so the
   three sets share one occupancy pass.
2. **Seeds.** `--fragments` seeds are dart-thrown over the solid voxels with a minimum spacing of
   0.62 of the mean cell width. 15 % of them get a multiplicative weight of 1/√3, so they claim
   about three times the area — a few large sherds among many small ones.
3. **Fracture.** Each solid voxel goes to the seed minimising `w_i · |x' − s_i|`, where `x'` is the
   voxel centre displaced by a coherent noise field: two octaves at about 0.9 of the cell width
   with an amplitude of 1.5 t (capped at 0.22 of the cell width so cells do not fall apart), plus
   two octaves at 1.2 t with an amplitude of 0.3 t for bumpiness at the wall scale. The noise is
   spline-interpolated white noise normalised to unit RMS.
4. **Wear.** Per fragment, a strength `w ~ U(0, --wear)`. The cell is eroded by
   `w · t · (0.01 + 0.99 · exp(−r / 0.06 t))` where `r` is the distance to the crease at which the
   fracture meets the shell, so the arris is chipped (up to 2 mm for the most worn fragment at
   `--wear 0.25`) while the middle of the fracture face recedes by under 0.1 mm. The outer and
   inner shell surfaces are untouched, which keeps the wall thickness exact.
5. **Surfaces.** Marching cubes at level 0.5 on the occupancy field smoothed with a 0.8-voxel
   Gaussian (this removes the voxel staircase and rounds the arris), six Taubin iterations, and
   quadric decimation to a 0.6 mm target edge. Vertex colour is terracotta modulated by
   low-frequency noise, with fracture vertices 4 % lighter.
6. **Poses and ground truth.** Each fragment is centred, given a uniform random rotation and a
   translation drawn from ±500 mm. `ground_truth.json` stores the **inverse**, the matrix that maps
   the file's coordinates back into the assembled frame, together with `adjacency` (pairs sharing
   at least 0.5 t² of fracture, counted as shared voxel faces on the labelling before wear),
   `missing` and `object_of`.

`--wear 0.5` was the first default and it is too much: eroding the arris by up to 4 mm on an 8 mm
wall destroys the breakline that the matcher keys on, and no true pair produced a usable candidate.
0.25 is the value used for all three sets.

Intruders are cut from the second vessel into pieces of the same size as the main ones (the piece
count is scaled by the volume ratio, 131 cells for the goblet against 170 for the pot); cutting the
goblet into only 18 pieces made intruders recognisable by size alone.

## What was generated

| set | files | intruders | withheld | adjacent pairs | generation |
|---|---|---|---|---|---|
| `synthetic_pingsdorf_20` | 20 | 0 | 0 | 50 | 174 s |
| `synthetic_pingsdorf_60` | 56 | 0 | 3 | 143 | 228 s |
| `synthetic_pingsdorf_170` | 164 | 6 | 8 | 407 | 211 s |

All fragments are watertight (edge-manifold, every edge used by exactly two triangles). Seed 0
throughout; the first occupancy pass adds 24 s to the first run only.

| set | faces (min/med/max) | area in t² | median edge | edges per t | thickness mm | total size |
|---|---|---|---|---|---|---|
| 20 | 53 360 / 401 464 / 717 672 | 101 / 726 / 1354 | 0.592 mm | 13.5 | 7.07 / 7.86 / 12.63 | 312 MB |
| 60 | 3 456 / 124 078 / 455 668 | 7 / 223 / 802 | 0.593 mm | 13.5 | 2.86 / 8.05 / 12.65 | 314 MB |
| 170 | 1 060 / 52 424 / 187 182 | 2 / 97 / 355 | 0.593 mm | 13.5 | 3.43 / 8.27 / 15.66 | 374 MB |

The thickness spread is real, not an artefact: it is the pot's own thick base and thin shoulder,
plus the estimator's ambiguity on the smallest chips. The pipeline reports a fracture area fraction
of 7–38 % per fragment, against 12–20 % measured on `test_fragments_1`.

Each folder also holds `README.md`, `preview_assembled.png` (the vessel rebuilt from the ground
truth, one colour per fragment) and `preview_fragments.png` (six fragments in their stored poses).

## Running the pipeline on the 20-fragment set

`sherd_refit/` was mid-edit by another agent while this ran and the CLI crashed inside the worker
pool, so the run used a pristine copy of the package at commit `4af8a68`. Command equivalent:

```
sherd-refit run input/synthetic_pingsdorf_20/fragments --out output/synthetic_pingsdorf_20 \
    --workers 3 --threads 2
```

Three workers rather than the default nine because the fragments are large and nine workers
exhausted 16 GB of RAM (`BrokenProcessPool`).

| stage | time |
|---|---|
| preprocess | 4.9 s |
| matching, 190 pairs | 216.9 s |
| assembly | 3.4 s |
| refinement | 1.9 s |

| | |
|---|---|
| candidates reaching verification | 52 |
| accepted joins | 2 |
| joins used in the assembly | 1 |
| joins used that are correct (≤ 5°, ≤ 0.5 t) | 1 of 1 — **precision 1.00** |
| ground-truth adjacent pairs recovered | 1 of 50 — **recall 0.02** |
| fragments placed | 2 of 20, leaving 19 groups |

The single join, `frag_008`–`frag_009`, is right to 0.01° and 0.09 mm (0.011 t).
`preview_segmentation.png` is correct: the fracture band is red and confined to the fragment edges,
the shell is grey.

## Why the recall is 0.02, and why it is not the data

Three measurements, in order.

**The fragments fit.** Two neighbouring cells extracted from the same labelling are 0.001 mm apart,
whatever the smoothing settings. At the exact ground-truth pose, the distance from one fragment's
vertices to the other's *surface* is 0.001–0.007 t, and 57–60 % of facing points are within
0.04 t. That is well inside the thresholds `gap ≤ 0.065`, `tight ≥ 0.25`.

**The search finds the pose.** Of the 22 adjacent pairs that produced any candidate, 17 had a
candidate within 0.13° and 1.0 mm of ground truth — usually within 0.03° and 0.2 mm. The
hypothesis generator and the ICP chain work on this data.

**Verification rejects those correct poses.** They score `tight` 0.07–0.15 and `gap` 0.086–0.108,
against thresholds 0.25 and 0.065. `fracture_scores` in `matching.py` measures both **point to
point** between the two fragments' fracture samples, and each fragment contributes a fixed
30 000 area-weighted surface samples regardless of its size. The sample spacing is therefore
`sqrt(area / 30000)`, which is not a quantity in `t`, and the nearest-neighbour distance between
two independent samples cannot fall below about half of it. Reproducing that measurement at the
exact ground-truth pose gives back the numbers the pipeline reports, and they track fragment area:

| pair | area A / B (t²) | point-to-point gap (t) | tight | point-to-surface gap (t) |
|---|---|---|---|---|
| 008–009 | 231 / 264 | 0.067 | 0.26 | 0.0066 |
| 002–015 | 551 / 440 | 0.087 | 0.15 | 0.0012 |
| 007–012 | 1022 / 375 | 0.112 | 0.09 | 0.0030 |
| 007–016 | 1022 / 947 | 0.117 | 0.08 | 0.0049 |
| 011–013 | 1047 / 490 | 0.118 | 0.08 | 0.0047 |
| 005–012 | 1286 / 375 | 0.127 | 0.07 | 0.0029 |

The only pair near the threshold is the smallest one, and it is the pair the pipeline accepted.
`test_fragments_1`, on which the thresholds were tuned, has fragments of roughly 100–300 t²; a
20-piece break of a whole 400 mm pot gives 100–1350 t², so the measurement floor rises above the
threshold. Nothing here says the thresholds are wrong — it says the point sampling that feeds them
is not scale-free. Raising `n_samples` with fragment area, or measuring point-to-surface, would
move these numbers.

A control run on a connected 6-fragment subset with `--target-faces 850000`, so that no fragment is
decimated at all, accepted one join (`frag_008`–`frag_012`, tight 0.26, gap 0.065) and rescued
`frag_001`–`frag_006` from producing no candidate at all to producing a correct-pose one. It did
**not** move `tight` or `gap` on the large pairs, which confirms that the working-mesh resolution
is a secondary effect and the sampling is the binding one.

The 60- and 170-fragment sets, whose fragments are 97–223 t² at the median, sit in the range the
thresholds were tuned for. They were not run through the pipeline (1540 and 13 366 pairs).

## Realism caveats

- **Fracture geometry is statistical.** The surfaces are noise-warped Voronoi boundaries: isotropic,
  with no conchoidal features and no preference for coil joins or other structure in the fabric.
  Real fracture in wheel-thrown pottery is not isotropic.
- **Wear is only an arris chip.** No abrasion of the shell, no encrustation, no lost surface, no
  scanner noise or holes on the fracture faces. Every fragment is watertight, which real scans
  often are not.
- **The fit is perfect.** Neighbours touch to 0.001 mm. Real refitting sherds have deformed and
  lost material, and the SfS++ pots show adjacent pieces 0.015–0.074 mm apart. This benchmark is
  therefore an upper bound on what a matcher can achieve, not a substitute for real data.
- **Colour is synthetic.** Low-frequency mottling on a terracotta base, not photogrammetric
  texture. The pipeline does not use colour, so this only affects how the previews look.
- **The 20-fragment set is unrealistically coarse.** No real refitting problem breaks a 400 mm pot
  into 20 pieces; it exists because 190 pairs is what fits in a pipeline run. The 60- and
  170-piece sets are the representative ones.
- **Intruder ground truth is in the other vessel's frame.** The matrices for `intruder_*` reassemble
  the goblet, not the pot; they are marked `intruder` in `object_of` and excluded from `adjacency`.
- **One source, one break.** Everything derives from a single pot and a single seed. Cross-object
  variation is not represented.
