# test_fragments_1 — result of the fracture-surface pipeline

**Date:** 2026-09-05. Command: `reassemble run input/test_fragments_1/fragments --out output/test_fragments_1`.
Apple M2 Pro, 9 workers: preprocessing 18 s, matching 120 s, assembly + refinement + outputs 20 s.

## Result

- **Group 0:** FY234104 – FY234094 – FY234021 in a row, FY234094 in the middle.
- **Not assembled:** FY234007 (no candidate passes verification).

This is the museum's manual assembly (three of four pieces; the photo in
`input/test_fragments_1/result_examples/result1.jpg` shows the same row, 094 being the tall middle
piece). It is the opposite of what Structure-from-Sherds++ claimed (007–021–104 with 094 rejected).

## Joins (distances in units of wall thickness t = 38.6)

| join | seam length | tight contact A/B | median gap | contact area | shell normal agreement | penetration |
|---|---|---|---|---|---|---|
| 021–094 | 21.3 t | 0.27 / 0.41 | 0.060 t | 8.7 t² | 0.94 | 0 |
| 094–104 | 11.3 t | 0.42 / 0.28 | 0.057 t | 4.3 t² | 0.88 | 0 |

Best rejected candidates: 021–104 tight 0.19 gap 0.077 pen 0.016; 007–021 tight 0.18 gap 0.076;
007–094 tight 0.12 gap 0.116; 007–104 tight 0.14 gap 0.090 pen 0.026. Acceptance needs tight ≥ 0.25,
gap ≤ 0.065, pen ≤ 0.005, seam ≥ 3.

## What the numbers mean

- "tight contact" is the share of fracture points facing the other fragment that lie within
  0.04 t (≈ 1.5 units) of it. Real joins on this eroded terracotta reach 0.3–0.45; every false
  candidate stays below 0.2. A perfect synthetic fit would be near 1.
- "seam length" is how much of the crack line on the shell coincides on both fragments; the
  021–094 seam runs the whole shared side (≈ 830 units).
- The two joins are consistent: 021 and 104 end up 4.6 t apart, no contact, no penetration, as in
  the manual assembly.

## On the fourth fragment

FY234007 has the same wall thickness (38.7) and colour as the others, so it may belong to the same
object without sharing an edge with these three. No hypothesis for it fits any of the three: the
best breakline alignments leave a median gap of 0.08–0.12 t and penetrate. A run with three times
more candidates (`--stage1 1200 --candidates 120`) is recorded in `output/test_fragments_1_exhaustive`.

## What was tried and discarded on the way (segmentation)

1. Multi-scale normal-dispersion roughness (Huang 2006 style): picks up the block-shaped
   decimation noise Geomagic left on the shells; fracture band comes out speckled.
2. Quadric-fit residual at 0.3 t: the same noise gives residuals of ~1 unit on the shells,
   indistinguishable from the fracture.
3. Inward-ray wall-thickness test with a **median** over a ray cone: flips at 50 % and leaves the
   band speckled where the smoothed normal is nearly tangent to the shells. A **vote** (≥ 5 of 7
   rays) plus the fixed-reference boundary growth gives a clean band and a boundary within ~0.1 t
   of the crease. This is what the package uses.

Matching by PCA-aligned fracture *facets* (SPPD style) was also tried first: on these long thin
facets the ICP has a sliding ambiguity and no candidate stood out. Pose hypotheses from breakline
point pairs with local frames fixed that.
