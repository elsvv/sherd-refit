# Reassembly report

Wall thickness (collection median): 30.15 units. All distances below are in units of thickness (t).

Every distance threshold is `max(k t, m res)`, with `res` the median edge length of the working mesh (column `edge`). The `tight` distance and the `gap` limit a pair was actually judged by are listed per join below; they equal the `k t` form on any mesh with enough edges across the wall.

## Fragments

| fragment | faces (orig) | thickness | ray mode | thickness/median | edge | edges per t | fracture area % | watertight | extent |
|---|---|---|---|---|---|---|---|---|---|
| pieceA | 45100 (45100) | 30.18 | 29.80 | 1.00 | 2.124 | 14.2 | 25.7 | True | 149 x 258 x 221 |
| pieceB | 35900 (35900) | 30.13 | 29.83 | 1.00 | 2.278 | 13.2 | 27.2 | True | 198 x 245 x 111 |

## Assembly

- group 0: pieceA, pieceB

## Joins used

| A | B | score | seam (t) | tight A/B | tight at (t) | gap (t) | gap limit (t) | contact (t²) | shell cont. | normal agr. | penetration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pieceA | pieceB | 17.64 | 21.0 | 0.90 / 0.84 | 0.011 | 0.001 | 0.034 | 7.0 | 0.026 | 0.99 | 0.0000 |

## Best candidate per pair

Acceptance requires tight ≥ 0.25, gap ≤ max(0.03 t, 0.45 res), penetration ≤ 0.005, seam ≥ 3.0, normal agreement ≥ 0.8; tight counts points within max(0.01 t, 0.15 res).

| A | B | accepted | score | seam (t) | tight A/B | tight at (t) | gap (t) | gap limit (t) | penetration | normal agr. |
|---|---|---|---|---|---|---|---|---|---|---|
| pieceA | pieceB | yes | 17.64 | 21.0 | 0.90 / 0.84 | 0.011 | 0.001 | 0.034 | 0.0000 | 0.99 |

## Timing

