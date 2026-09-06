# E1 — decimation: Open3D vs `meshopt` vs `baby_shark`

**Date:** 2026-09-06. Branch `rust-core`. Plan row E1, design `2026-09-06-rust-core-design.md` §3,
algorithm reference `2026-09-06-algorithm-reference.md` §3.2–3.4.
**Machine:** Apple M2 Pro (10 cores, 16 GB), macOS 24.6.0, cargo 1.88, `-O3` release build.
**Reference:** Open3D 0.19.0 `simplify_quadric_decimation`, numpy 2.5.2, scipy 1.18.1, Python 3.12.9.
**Candidates:** `meshopt` 0.6.2 (vendors meshoptimizer 0.25, built from C++ by `cc`, no system deps)
and `baby_shark` 0.3.12 (`decimation::EdgeDecimator`, pure Rust Garland–Heckbert).

**Recommendation: `meshopt` 0.6.2 with `SimplifyOptions::Regularize`.** It is the only candidate
that meets every criterion that can be met at all, and it is 20× faster than Open3D (34.7 s → 1.7 s
over the benchmark set, 8.15 s → 0.42 s on the largest scan) and 60–660× faster than `baby_shark`.
`baby_shark` is rejected. No own Garland–Heckbert is needed.

## 1. What was measured and how

For every mesh the harness reproduces `Fragment.from_mesh_file` exactly up to the decimation call:
read, `remove_duplicated_vertices` / `remove_degenerate_triangles` / `remove_unreferenced_vertices`,
largest connected component, `t` from the inward-ray histogram mode on the **full-resolution** mesh
with `rng(0)`, then the adaptive budget `target = clip(600·ΣA₀/t², 50000, 200000)`.

That one cleaned mesh is handed to each decimator. Everything after the decimation is Open3D's,
identical for all candidates — the post-decimation cleanup, `filter_smooth_taubin(3)`, the second
cleanup, and the whole Python segmentation (`segment_faces`: `t/8` grid, smoothed normals, the
7-ray cone test, majority filter, island removal, boundary growth) — so the only variable in the
comparison is the decimator. The Rust binary reads and writes raw f64/u32 mesh dumps, so no mesh
format is in the loop either.

Per mesh and method: faces reached vs target, boundary-edge count and fraction with the
`closed_enough` verdict (≤ 0.2 % of the unique edges), `res` (median unique-edge length),
the wall thickness re-estimated **on the working mesh**, the decimation wall time, and the
area-weighted agreement of the fracture mask against the Open3D-decimated working mesh after a
nearest-face-centroid transfer (reported as the mean of both transfer directions; the two never
differed by more than 0.003).

**Which meshes actually get decimated.** With the shipped budget only 5 of the 21 benchmark meshes
are above their target: the 4 terracotta scans (708 k–1.34 M faces → 76 k–156 k, an 8–9× cut) and
`Pot_A_Piece_01` (200 124 → 200 000, i.e. nothing). Every other SfS++ piece of pots A and B arrives
below the budget and is passed through untouched — for those the decimator choice is irrelevant in
production. To give the crates something to do on that data the ten pot pieces above 50 000 faces
were also run with the target forced to 50 000 (`MIN_FACES`, the budget's own lower clamp); those
rows are marked *forced target*.

## 2. Per-mesh results

**FY234007_reduced** — 1230314 faces in, `t` = 38.583, budget 147304, target 147304

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 147304 | +0.00 % | 0 (0.000 %) | yes | 2.2372 | — | 38.374 | — | 8.15 s | — |
| meshopt | 147304 | +0.00 % | 31 (0.014 %) | yes | 1.8938 | -15.3 % | 38.560 | +0.5 % | 0.44 s | 0.9839 |
| meshopt +LockBorder | 147304 | +0.00 % | 31 (0.014 %) | yes | 1.8938 | -15.3 % | 38.560 | +0.5 % | 0.41 s | 0.9839 |
| meshopt +Regularize | 147304 | +0.00 % | 0 (0.000 %) | yes | 2.4170 | +8.0 % | 38.405 | +0.1 % | 0.42 s | 0.9879 |
| baby_shark | 146908 | -0.27 % | 0 (0.000 %) | yes | 2.0749 | -7.3 % | 38.379 | +0.0 % | 25.07 s | 0.9883 |

**FY234021_reduced** — 1052554 faces in, `t` = 38.746, budget 122651, target 122651

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 122650 | -0.00 % | 0 (0.000 %) | yes | 2.2185 | — | 38.927 | — | 7.26 s | — |
| meshopt | 122650 | -0.00 % | 30 (0.016 %) | yes | 1.7899 | -19.3 % | 38.611 | -0.8 % | 0.38 s | 0.9916 |
| meshopt +LockBorder | 122650 | -0.00 % | 30 (0.016 %) | yes | 1.7899 | -19.3 % | 38.611 | -0.8 % | 0.35 s | 0.9916 |
| meshopt +Regularize | 122650 | -0.00 % | 1 (0.001 %) | yes | 2.4285 | +9.5 % | 38.710 | -0.6 % | 0.32 s | 0.9915 |
| baby_shark | 122254 | -0.32 % | 0 (0.000 %) | yes | 2.0231 | -8.8 % | 38.612 | -0.8 % | 20.62 s | 0.9916 |

**FY234094_reduced** — 1341472 faces in, `t` = 39.004, budget 155920, target 155920

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 155920 | +0.00 % | 0 (0.000 %) | yes | 2.2397 | — | 40.080 | — | 9.19 s | — |
| meshopt | 155918 | -0.00 % | 28 (0.012 %) | yes | 1.8666 | -16.7 % | 39.126 | -2.4 % | 0.48 s | 0.9913 |
| meshopt +LockBorder | 155918 | -0.00 % | 28 (0.012 %) | yes | 1.8666 | -16.7 % | 39.126 | -2.4 % | 0.46 s | 0.9913 |
| meshopt +Regularize | 155920 | +0.00 % | 1 (0.000 %) | yes | 2.4449 | +9.2 % | 40.619 | +1.3 % | 0.44 s | 0.9902 |
| baby_shark | 155524 | -0.25 % | 0 (0.000 %) | yes | 2.0512 | -8.4 % | 38.862 | -3.0 % | 48.45 s | 0.9908 |

**FY234104_reduced** — 708142 faces in, `t` = 40.568, budget 75526, target 75526

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 75526 | +0.00 % | 0 (0.000 %) | yes | 2.2255 | — | 39.845 | — | 7.69 s | — |
| meshopt | 75524 | -0.00 % | 17 (0.015 %) | yes | 1.9199 | -13.7 % | 39.811 | -0.1 % | 0.26 s | 0.9889 |
| meshopt +LockBorder | 75524 | -0.00 % | 17 (0.015 %) | yes | 1.9199 | -13.7 % | 39.811 | -0.1 % | 0.57 s | 0.9889 |
| meshopt +Regularize | 75526 | +0.00 % | 0 (0.000 %) | yes | 2.3867 | +7.2 % | 40.239 | +1.0 % | 0.38 s | 0.9885 |
| baby_shark | 75132 | -0.52 % | 0 (0.000 %) | yes | 1.9589 | -12.0 % | 39.949 | +0.3 % | 17.99 s | 0.9889 |

**Pot_A_Piece_01_Mesh** — 200124 faces in, `t` = 5.946, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 50000 | +0.00 % | 0 (0.000 %) | yes | 1.0216 | — | 5.860 | — | 0.81 s | — |
| meshopt | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.8129 | -20.4 % | 6.004 | +2.5 % | 0.05 s | 0.9765 |
| meshopt +LockBorder | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.8129 | -20.4 % | 6.004 | +2.5 % | 0.05 s | 0.9765 |
| meshopt +Regularize | 50000 | +0.00 % | 0 (0.000 %) | yes | 1.0398 | +1.8 % | 6.464 | +10.3 % | 0.04 s | 0.9801 |
| baby_shark | 49604 | -0.79 % | 0 (0.000 %) | yes | 0.8540 | -16.4 % | 6.020 | +2.7 % | 8.56 s | 0.9735 |

**Pot_A_Piece_02_Mesh** — 111434 faces in, `t` = 3.475, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 49999 | -0.00 % | 71 (0.095 %) | yes | 0.7453 | — | 3.512 | — | 0.28 s | — |
| meshopt | 49999 | -0.00 % | 132 (0.176 %) | yes | 0.6118 | -17.9 % | 3.505 | -0.2 % | 0.02 s | 0.9908 |
| meshopt +LockBorder | 50000 | +0.00 % | 209 (0.278 %) | **no** | 0.6109 | -18.0 % | 3.526 | +0.4 % | 0.02 s | 0.9908 |
| meshopt +Regularize | 49999 | -0.00 % | 55 (0.073 %) | yes | 0.7695 | +3.2 % | 3.515 | +0.1 % | 0.03 s | 0.9905 |
| baby_shark | 49602 | -0.80 % | 192 (0.258 %) | **no** | 0.6354 | -14.8 % | 3.515 | +0.1 % | 7.64 s | 0.9908 |

**Pot_A_Piece_03_Mesh** — 82255 faces in, `t` = 3.439, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 49999 | -0.00 % | 61 (0.081 %) | yes | 0.6528 | — | 3.450 | — | 0.19 s | — |
| meshopt | 50000 | +0.00 % | 122 (0.163 %) | yes | 0.5315 | -18.6 % | 3.448 | -0.1 % | 0.01 s | 0.9946 |
| meshopt +LockBorder | 49999 | -0.00 % | 149 (0.198 %) | yes | 0.5307 | -18.7 % | 3.452 | +0.1 % | 0.01 s | 0.9946 |
| meshopt +Regularize | 50000 | +0.00 % | 44 (0.059 %) | yes | 0.6560 | +0.5 % | 3.731 | +8.1 % | 0.01 s | 0.9935 |
| baby_shark | 49601 | -0.80 % | 137 (0.184 %) | yes | 0.5373 | -17.7 % | 3.494 | +1.3 % | 7.01 s | 0.9948 |

**Pot_A_Piece_04_Mesh** — 66940 faces in, `t` = 3.554, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 49999 | -0.00 % | 7 (0.009 %) | yes | 0.5388 | — | 3.487 | — | 0.10 s | — |
| meshopt | 50000 | +0.00 % | 8 (0.011 %) | yes | 0.4564 | -15.3 % | 3.715 | +6.5 % | 0.01 s | 0.9942 |
| meshopt +LockBorder | 50000 | +0.00 % | 8 (0.011 %) | yes | 0.4564 | -15.3 % | 3.715 | +6.5 % | 0.01 s | 0.9942 |
| meshopt +Regularize | 50000 | +0.00 % | 4 (0.005 %) | yes | 0.5346 | -0.8 % | 3.738 | +7.2 % | 0.01 s | 0.9932 |
| baby_shark | 49602 | -0.80 % | 8 (0.011 %) | yes | 0.4656 | -13.6 % | 3.729 | +7.0 % | 6.62 s | 0.9949 |

**Pot_A_Piece_05_Mesh** — 68332 faces in, `t` = 3.809, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.5665 | — | 3.814 | — | 0.14 s | — |
| meshopt | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.4719 | -16.7 % | 3.843 | +0.8 % | 0.01 s | 0.9941 |
| meshopt +LockBorder | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.4719 | -16.7 % | 3.843 | +0.8 % | 0.01 s | 0.9941 |
| meshopt +Regularize | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.5602 | -1.1 % | 3.821 | +0.2 % | 0.01 s | 0.9930 |
| baby_shark | 49602 | -0.80 % | 0 (0.000 %) | yes | 0.4859 | -14.2 % | 3.841 | +0.7 % | 7.67 s | 0.9946 |

**Pot_A_Piece_06_Mesh** — 58364 faces in, `t` = 3.659, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.4467 | — | 3.718 | — | 0.07 s | — |
| meshopt | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.4139 | -7.3 % | 3.675 | -1.2 % | 0.01 s | 0.9936 |
| meshopt +LockBorder | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.4139 | -7.3 % | 3.675 | -1.2 % | 0.01 s | 0.9936 |
| meshopt +Regularize | 50000 | +0.00 % | 0 (0.000 %) | yes | 0.4412 | -1.2 % | 3.666 | -1.4 % | 0.01 s | 0.9927 |
| baby_shark | 49602 | -0.80 % | 0 (0.000 %) | yes | 0.4146 | -7.2 % | 3.686 | -0.9 % | 6.45 s | 0.9948 |

**Pot_B_Piece_01_Mesh** — 157281 faces in, `t` = 5.623, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 49999 | -0.00 % | 7 (0.009 %) | yes | 0.8573 | — | 5.299 | — | 0.56 s | — |
| meshopt | 50000 | +0.00 % | 14 (0.019 %) | yes | 0.6550 | -23.6 % | 5.255 | -0.8 % | 0.03 s | 0.9832 |
| meshopt +LockBorder | 49999 | -0.00 % | 27 (0.036 %) | yes | 0.6531 | -23.8 % | 5.440 | +2.7 % | 0.04 s | 0.9832 |
| meshopt +Regularize | 49999 | -0.00 % | 5 (0.007 %) | yes | 0.8834 | +3.0 % | 5.365 | +1.2 % | 0.03 s | 0.9825 |
| baby_shark | 49603 | -0.79 % | 23 (0.031 %) | yes | 0.7161 | -16.5 % | 5.385 | +1.6 % | 7.65 s | 0.9823 |

**Pot_B_Piece_02_Mesh** — 75762 faces in, `t` = 3.109, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 49999 | -0.00 % | 215 (0.286 %) | **no** | 0.6165 | — | 3.052 | — | 0.13 s | — |
| meshopt | 50000 | +0.00 % | 510 (0.678 %) | **no** | 0.5229 | -15.2 % | 3.125 | +2.4 % | 0.01 s | 0.9942 |
| meshopt +LockBorder | 50000 | +0.00 % | 570 (0.757 %) | **no** | 0.5226 | -15.2 % | 3.085 | +1.1 % | 0.01 s | 0.9942 |
| meshopt +Regularize | 49999 | -0.00 % | 151 (0.201 %) | **no** | 0.6213 | +0.8 % | 3.101 | +1.6 % | 0.01 s | 0.9935 |
| baby_shark | 49602 | -0.80 % | 542 (0.726 %) | **no** | 0.5370 | -12.9 % | 3.120 | +2.2 % | 6.98 s | 0.9940 |

**Pot_B_Piece_03_Mesh** — 61231 faces in, `t` = 3.587, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 49999 | -0.00 % | 21 (0.028 %) | yes | 0.5268 | — | 3.540 | — | 0.08 s | — |
| meshopt | 50000 | +0.00 % | 42 (0.056 %) | yes | 0.4593 | -12.8 % | 3.537 | -0.1 % | 0.01 s | 0.9934 |
| meshopt +LockBorder | 49999 | -0.00 % | 43 (0.057 %) | yes | 0.4593 | -12.8 % | 3.596 | +1.6 % | 0.02 s | 0.9934 |
| meshopt +Regularize | 49999 | -0.00 % | 19 (0.025 %) | yes | 0.5181 | -1.7 % | 3.593 | +1.5 % | 0.01 s | 0.9923 |
| baby_shark | 49601 | -0.80 % | 43 (0.058 %) | yes | 0.4719 | -10.4 % | 3.531 | -0.3 % | 6.22 s | 0.9933 |

**Pot_B_Piece_04_Mesh** — 57553 faces in, `t` = 3.507, budget 200000, target 50000  *(forced target: the shipped budget does not decimate this mesh)*

| method | faces | vs target | boundary edges | closed_enough | res | Δres | t (working) | Δt | decim | agreement |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| Open3D | 50000 | +0.00 % | 84 (0.112 %) | yes | 0.4497 | — | 3.486 | — | 0.07 s | — |
| meshopt | 50000 | +0.00 % | 140 (0.186 %) | yes | 0.4022 | -10.6 % | 3.519 | +1.0 % | 0.01 s | 0.9938 |
| meshopt +LockBorder | 49999 | -0.00 % | 161 (0.214 %) | **no** | 0.4020 | -10.6 % | 3.525 | +1.1 % | 0.01 s | 0.9937 |
| meshopt +Regularize | 50000 | +0.00 % | 63 (0.084 %) | yes | 0.4436 | -1.3 % | 3.499 | +0.4 % | 0.01 s | 0.9934 |
| baby_shark | 49601 | -0.80 % | 147 (0.197 %) | yes | 0.4118 | -8.4 % | 3.498 | +0.4 % | 8.03 s | 0.9942 |

## 3. Controls: how much of this spread is the reference's own noise?

Two of the criteria the candidates fail sit below the noise floor of the reference itself; the
third does not, and it is the one that decides. The control re-runs **Open3D against Open3D**, on the same input with the face
order permuted (`rng(12345)`) — the same algorithm, a different collapse order, the situation
D §13.1 calls statistical parity:

| mesh | faces | res | t (working mesh) | segmentation agreement |
|---|---:|---:|---:|---:|
| FY234007_reduced | 147304 vs 147304 | +0.00 % | −0.03 % | 0.9853 |
| FY234104_reduced | 75526 vs 75526 | +0.00 % | +1.52 % | 0.9905 |
| Pot_A_Piece_01 (forced) | 50000 vs 50000 | +0.00 % | **+11.03 %** | 0.9916 |
| Pot_A_Piece_04 (forced) | 49999 vs 49999 | +0.00 % | **+8.48 %** | 0.9938 |
| Pot_B_Piece_01 (forced) | 49999 vs 49999 | +0.01 % | −0.22 % | 0.9911 |
| Pot_B_Piece_02 (forced) | 49999 vs 49999 | +0.00 % | +1.60 % | 0.9941 |

The permuted run is not a relabelling: the two decimated vertex sets differ (total area differs in
the 8th digit), so this is genuinely "the same algorithm, a different result".

**The ±2 % thickness gate cannot be met by anything.** `estimate_thickness` is the centre of the
first fullest bin of a 60-bin histogram over `[0, p90]`, and that bin is 1.8 % of `t` on the
terracotta and 2.2–5.7 % of `t` on the pot pieces — the gate is finer than the estimator's
quantisation. Re-drawing only the ray sample (seeds 0–9 on the *same* Open3D working mesh) already
moves the estimate by up to 6.1 % (`Pot_A_Piece_01`) and 8.1 % (`Pot_A_Piece_04`); permuting the
input faces moves it by 11.0 %. Every candidate's worst thickness deviation (meshopt +6.5 %,
+Regularize +10.3 %, baby_shark +7.0 %) is inside that band, and every one of them lands on a pot
piece — exactly where the control shows the reference itself moving by 8–11 %. The criterion is
reported per mesh above but carries no decision.

It also carries no consequence: in the frozen algorithm `t` is measured on the **pre-decimation**
mesh (R §3.2, `from_mesh_file` builds `scene0` over `V0/F0`), so the pipeline's `t` — the unit every
threshold is expressed in — is bit-identical for all candidates. The working-mesh thickness measured
here is only a geometry-preservation proxy.

**The ≥ 0.97 segmentation agreement gate is meaningful, and every candidate passes it**, but the
control shows the ceiling is 0.985–0.994, not 1.0: the segmentation's `t/8` grid picks the
lowest-index face per voxel (R §3.4.1, PMC-4), so any change of face order alone already moves
0.6–1.5 % of the area. The candidates (0.974–0.995) sit at that ceiling, not below it.

**The ±10 % `res` gate is real.** The permuted control moves `res` by +0.00 %, so `res` is a stable
property of the decimation algorithm and the differences below are genuine:

- plain `meshopt` returns **−7 to −24 %**: meshoptimizer's error-driven simplification is adaptive,
  it leaves flat areas coarse and detailed areas fine, so at equal face count the edge-length
  distribution is skewed and its median sits well below Open3D's.
- `baby_shark` returns **−7 to −18 %** (9 of 14 meshes outside ±10 %) for the same reason.
- `meshopt` with `SimplifyOptions::Regularize` ("more regular triangle sizes and shapes at some
  cost to geometric quality") returns **−1.7 to +9.5 %** — inside the gate on all 14 meshes, and it
  costs nothing in time and nothing that matters in agreement: it has the best worst case of the
  three (0.9801 against 0.9765 for plain meshopt and 0.9735 for baby_shark), it lifts the hardest
  terracotta scan from 0.9839 to 0.9879, and on the remaining meshes it runs about 0.001 below
  plain meshopt.

On the terracotta at the shipped budget `res/t` is 0.055–0.058 (Open3D) against 0.059–0.063
(`+Regularize`), both under the 0.065 crossover where `m·res` overtakes `k·t` in the pair scales
(R §1.2) and under the `0.1·t` line that would switch the shell test from 5 votes to 4 — so on the production case no
threshold and no vote count changes. The gate still matters for coarser inputs, and `+Regularize`
is the only candidate that holds it.

## 4. Summary over the 14 decimated meshes

| criterion (gate) | Open3D | meshopt | meshopt +LockBorder | meshopt +Regularize | baby_shark |
|---|---|---|---|---|---|
| faces vs target (±5 %) | 0–1 short | 0–2 short (≤ 0.003 %) | 0–2 short | 0–1 short | 394–399 short (0.25–0.80 %) |
| `closed_enough` verdict | — | matches on 14/14 | **flips on 2/14** | matches on 14/14 | **flips on 1/14** |
| boundary-edge fraction | 0.000–0.286 % | 0.000–0.678 % | 0.000–0.757 % | 0.000–0.201 % | 0.000–0.726 % |
| `res` (±10 %) | — | **−23.6…−7.3 %, 13/14 fail** | **−23.8…−7.3 %, 13/14 fail** | **−1.7…+9.5 %, 14/14 pass** | **−17.7…−7.2 %, 9/14 fail** |
| working-mesh `t` (±2 %) | — | −2.4…+6.5 %, 4 fail | −2.4…+6.5 %, 4 fail | −1.4…+10.3 %, 3 fail | −3.0…+7.0 %, 4 fail (gate unmeetable, §3) |
| segmentation agreement (≥ 0.97) | — | 0.9765–0.9948 pass | 0.9765–0.9946 pass | 0.9801–0.9935 pass | 0.9735–0.9949 pass |
| decimation time, 14 meshes | 34.70 s | 1.73 s | 1.96 s | **1.73 s** | 184.98 s |
| worst single mesh (1.34 M faces) | 9.19 s | 0.48 s | 0.46 s | **0.44 s** | 48.45 s |
| determinism (two runs) | — | byte-identical | byte-identical | byte-identical | byte-identical |

`LockBorder` is not worth taking: it changes nothing on closed meshes and on the meshes that do
have a rim it *raises* the boundary-edge count (`Pot_A_Piece_02` 0.176 % → 0.278 %,
`Pot_B_Piece_04` 0.186 % → 0.214 %), flipping the `closed_enough` verdict on two meshes. A fragment
that fails `closed_enough` gets no penetration test (R §3.3.2, §6.4), so a flip is a behaviour
change, not a cosmetic one.

## 5. Why `baby_shark` is rejected

- **Speed.** 185 s against Open3D's 35 s and meshopt's 1.7 s over the same 14 meshes; 25 s for the
  1.23 M-face terracotta scan against meshopt's 0.44 s. Worse, it spends 6–8 s even on a
  58 k → 50 k reduction, because `collapse_edges` breaks out of its inner loop after every collapse
  once the face count is at the target and then refills the whole priority queue, up to 200 times.
  A 170-fragment collection would pay tens of minutes for decimation alone.
- **`res` fails** on 9 of 14 meshes (−7.2 to −17.7 %), the same adaptive-sizing bias as plain
  meshopt, and it has no regularisation option.
- **It undershoots the target by a near-constant 394–399 faces** on every mesh, whatever its size
  (the same "one collapse per outer pass" artefact), and it collapses ~200 edges even when the
  mesh is *already* under the target — `CornerTable` + decimator on a 20 728-face mesh with a target of 86 969 returns 20 328
  faces after 2.4 s, where meshopt returns the input untouched in 1.3 ms.
- One `closed_enough` flip (`Pot_A_Piece_02`, 0.258 % boundary edges against Open3D's 0.095 %).
- `CornerTable::from_vertex_and_face_iters` returns an **empty mesh** on any builder error instead of
  reporting it (`unwrap_or_default`), which would be a silent data-loss path in a port.

Its only advantage — quadric placement, i.e. the same algorithm family as Open3D's — buys nothing
measurable: its segmentation agreement (0.9735–0.9949) is not better than meshopt's.

## 6. What the port should do (phase 1a, step S3)

```rust
// deps: meshopt = "0.6.2"   (vendors meshoptimizer 0.25, built with `cc`; no system deps)
let target = (600.0 * area0 / (t * t)).clamp(50_000.0, target_faces as f64) as usize;
if faces.len() > target {                       // R §3.3: only decimate above the budget
    let pos: Vec<f32> = /* xyz as f32, stride 12 */;
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&pos), 12, 0)?;
    let idx = meshopt::simplify(&indices, &adapter, target * 3, 1e9,
                                SimplifyOptions::Regularize, None);
    // idx references the ORIGINAL vertices; keep the f64 positions and drop the unreferenced ones
}
```

Notes that matter for S3:

- `target_error = 1e9` with `ErrorAbsolute` **off** (relative to the mesh extent) is effectively
  "no error cap", which is what Open3D does; the returned `result_error` was never the binding
  constraint — the face target always was.
- meshopt reads positions as **f32**, but only to decide collapses: the API returns an index buffer
  and nothing else, so vertices cannot move and the port keeps its `f64` coordinates exactly. This
  is a real advantage over `baby_shark`, which recomputes positions from the quadric.
- meshopt welds vertices by position internally, so the cleaned mesh can be handed over as is;
  it must be the **largest component** (R §3.1), and `Prune` must stay off.
- After `simplify`, apply R §3.3's cleanup in order — degenerate triangles, duplicated vertices,
  unreferenced vertices — then Taubin.
- Keep the `faces > target` guard: meshopt is a 1.3 ms no-op above the target, but the guard is what
  the reference does and it keeps the working mesh identical to the input in that case.
- The design table (D §3) lists `meshopt` 0.4 and `baby_shark` 0.3 as the fallback; the pin becomes
  **`meshopt = "0.6.2"`** — `SimplifyOptions::Regularize`, which is what makes the crate pass, is
  exposed by that version (meshoptimizer 0.25). `bindgen` is behind an optional feature there, so a
  default build needs only a C++ compiler. Drop `baby_shark`.

## 7. Residual risk

- E1 isolates the decimator: Open3D's Taubin and Open3D's segmentation were used for every
  candidate. The Rust Taubin (R §3.3.1, inverse-distance weights) and the Rust segmentation will add
  their own deviation on top of the 0.980–0.993 measured here; the phase-1b gate (≥ 0.97 native
  segmentation agreement) has roughly one to two points of headroom, not more. If a later stage
  comes in under the gate, the decimator is the first thing to re-measure with this harness.
- The working mesh will never be identical to Python's (PMC-2), so native-mode parity stays
  statistical (D §13.1). Injected-fixture parity is unaffected — it starts from the Python working
  mesh.
- Only the terracotta scans exercise the decimator at production settings; the SfS++ pots were
  measured at a forced target. Large-scan behaviour beyond 1.34 M faces (the 10 M-face case D §3
  worries about) is untested; meshopt's cost is linear in practice here (0.44 s at 1.23 M faces,
  0.48 s at 1.34 M), so 10 M faces should land near 4 s.
- Peak memory was not measured for any candidate.

## 8. Reproducing

Harness (not committed, it is experiment code):
`/private/tmp/claude-501/-Users-vaceslaveliseev--dev-ceramic-reassembling/0ffcf053-0fbe-4b23-9b9d-538d546e185e/scratchpad/rust/E1/`
— `decim/` (the Rust binary: `meshopt`,
`meshopt_lock`, `meshopt_reg`, `meshopt_reg_lock`, `baby_shark`, `baby_shark_keepb`), `e1.py`
(the measurement driver, `--force-target=N`), `control.py` (§3 controls), `results.jsonl`,
and the decimated working meshes as `meshes/<mesh>__<method>.ply`.
