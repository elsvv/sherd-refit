# Task Y — closing the phase-1d verification (V4-D1 … V4-D10)

**Date:** 2026-09-07. **Tree:** branch `rust-core`, from `4ca6750` ("V4: independent verification of
phase 1d"); working tree clean at the start. **Machine:** Apple M2 Pro, 10 cores (6P+4E), 16 GB,
macOS 24.6.0, rustc 1.97.0. **Reference:** `sherd_refit/` at `895a948`, Open3D 0.19.0, numpy 2.5.2,
Python 3.12.9; every Python run with `OMP_NUM_THREADS=1`.

Ten defects and two "recorded" items came out of V4. All twelve are closed. Three needed a
measurement rather than a fix, and the largest of those — **V4-D3**, "R §13's quality table is not
met natively on three of seven sets" — turned out to be a statement about the *reference*: R §13's
per-set figures are one draw of the reference's own randomised search, and the port's figures sit
inside the reference's own spread on every set that was swept (§3).

---

## 1. The twelve, and what each turned into

| defect | what it was | closed by |
|---|---|---|
| **V4-D1** | `pytest -q` red: `tools/dump_outputs.py` wrote 20 files the manifest never listed, and `--verify-checksums` hashed none of them | **Y1** — the tool rewrites the manifest; the slab dump goes 180 → 201 entries; `report.md` added to the dump; two new tests |
| **V4-D2** | `cargo test --release` failed 1 of 220, and CI only ever ran debug | **Y3** — `0..255_u8` plus the 255 assertion; a `test-release` CI job |
| **V4-D3** | R §13 not met natively on pot_B, pot_G, synthetic_20 | **Y6** — measured (§3); R §13, R §12.1 and D §10.3 restated on the measured spread |
| **V4-D4** | `timings` alphabetical, not in stage order | **Y4** — `report::Ordered<V>`; measured identical to the reference's own file |
| **V4-D5** | `transforms.json`'s `fragments` in name order, not placement order | **Y4** — R §8's insertion order recorded by the assembly |
| **V4-D6** | two of D §10.2's stated tolerances are not the ones the harness enforces | **Y5** — D §10.2 amended with the derivations (three rows) |
| **V4-D7** | `segment` wrote no preview and had no `--workers` | **Y4** — `preview_segmentation.png`, with a test; `--workers` added |
| **V4-D8** | `--workers` defaulted to one per core; the reference's default is cores − 1 | **Y4** — `pipeline::default_workers` |
| **V4-D9** | R §9's cap seeded from `params.seed`; R §10 seeds it with the literal `0` | **Y4** — `refine::CAP_SEED` |
| **V4-D10** | `pts.mean(0)` summed pairwise, and two doc comments said that was numpy's answer | **Y4** — measured against numpy 2.5.2; left-to-right per column; `recentre` goes to **0** |
| recorded | R §11.2 lists an `"output"` timing the reference never writes | **Y5** — R §11.2 amended (and it now states both files' key order) |
| recorded | R §13's terracotta seams, written "≈ 20.3 / 10.7" | **Y6** — the port's measured **20.667 / 12.333 t** recorded beside them |

### 1.1 Two of them are worth keeping in prose

**V4-D2's cause, as V4 reduced it.** The loop was `for k in 0..=255_u8 { …; if k < 255 {
assert_eq!(quantize_color(f64::from(k) + 0.5), k + 1); } }`. A standalone 17-line program with
nothing but `quantize_color` and that loop, built with rustc 1.97.0 for aarch64-apple-darwin,
panics at `-C opt-level=2` and `3` and passes at `0` and `1`: the optimiser folds the `k < 255`
guard into `RangeInclusive`'s own exhaustion test, whose increment carries the `add nuw i8` of
`Step::forward_unchecked`, so the last iteration runs the guarded body with a poisoned `k + 1` that
materialises as `0`. `quantize_color` itself was never wrong — `254.5 → 255`, `255.5 → 255`,
`-1 → 0`, `NaN → 0` all verified directly, and they are asserted after the loop now. The reason two
steps could ship with the release suite red is that `.github/workflows/rust.yml` only ran the debug
profile, which is the profile the pattern survives; a `test-release` job now runs it on the two
runners D §10.5 gives the `parity` job.

**What the fixes changed in the port's own output files.** `transforms.json` now keys `fragments`
in R §8's placement order and `report.json` / `report.md` list `timings` in stage order, so those
three files differ from the ones the port wrote before this task — deliberately, and towards the
reference's own byte order (measured against `output/fixtures/terracotta/_run/*` in §2.2). The
pose numbers moved too, by the 8.8e-15–7.0e-13 t that V4-D10's summation order was worth, and that
movement is *towards* the reference as well: `recentre` is now exact. No decision, group, join or
rejection changed anywhere — the gate table is V4's, row for row.

Commits: `156db42` (Y1, the manifest and `report.md`), `1d73e08` (Y2, the harness's `report md`
row), `3ca4e50` (Y3, the release suite and its CI job), `8c4b5ae` (Y4, the six code deviations),
`21992b4` (Y5, D §10.2 and R §11.2), `b75c737` (Y6, R §13, R §12.1 and D §10.3), and this note.

---

## 2. Gate table

`sherd-refit-rs run <input> --out <dir> --no-meshes` (previews **on**, which is more than
D §10.3's gate asks for), cold into an empty directory, then warm over its own cache; scored with
`tools/evaluate.py`. Release binary, nothing else running.

| set | port accuracy | reference (R §13, and its own spread) | precision | joins used (correct) | groups | cross-object | purity | cold | warm | peak RSS | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| terracotta | R §13's join row (§2.1) | — | — | 2 (both R §13's) | 1 + 1 singleton | 0 | — | 4.71 s | **2.61 s** | 1 500 MiB | ≤ 25 s, **9.6×** |
| `pot_A` | **100 % (8/8)** | 87.5 % (seed 0; not swept) | 1.000 | 7 (7) | one group of 8 | 0 | 1.000 | 10.37 s | **8.88 s** | 642 MiB | ≤ 35 s, **3.9×** |
| `pot_B` | 88.9 % (8/9) | 100 % (seed 0); **88.9–100 %** over seven runs | 1.000 | 7 (7) | 8 + 1 | 0 | 1.000 | 10.78 s | 9.32 s | 631 MiB | — |
| `pot_C` | 75.0 % (3/4) | 75 % | 0.667 | 3 (2, 1 unscorable) | 3 + 2 + 2 singletons | 0 | 1.000 | 5.69 s | 5.49 s | 345 MiB | — |
| `pot_G` | 0 % (0/7) | 0 % | 0.000 | 2 (0; **both wrong-pose on adjacent pairs**) | 2 + 2 + 3 singletons | 0 | 1.000 | 5.83 s | 5.32 s | 316 MiB | — |
| `pot_H` | 36.4 % (4/11) | 36.4 % | 0.429 | 7 (3) | 7 + 2 + 2 singletons | 0 | 1.000 | 11.00 s | **10.84 s** | 549 MiB | ≤ 40 s, **3.7×** |
| `synthetic_20` | 90.0 % (18/20) | 95 % (seed 0); **85–95 %** over seven runs | 1.000 | 19 (19) | 15 + 3 + 2 singletons | 0 | 1.000 | 49.06 s | **40.68 s** | 1 768 MiB | ≤ 120 s, **2.9×** |

Every row is inside R §13 as restated (§3.3): pot_B's 88.9 % and synthetic_20's 90 % are inside the
reference's own spread, pot_A is above the reference's own draw, pot_C and pot_H are equal to it,
and pot_G meets the restated clause (at most two joins, every one a wrong-pose join on a
ground-truth-adjacent pair). Cross-object joins are 0 and purity 1.000 everywhere. Peak RSS is
1.73 GiB at the worst, against D §8's 6 GB.

The figures are the ones V4 §4.1 measured, unchanged by any of Y1–Y5: the six defects fixed here
move file *order* and the last bits of a translation, not a decision.

### 2.1 R §13's terracotta row, exactly

| R §13 asks | measured |
|---|---|
| joins used exactly {021–094, 094–104} | **exactly those two, in that order; nothing rejected** |
| 007 unplaced | **unplaced**, its own group |
| both `pen` = 0 | 0.0000 / 0.0000 |
| tight of both ≥ 0.27 | 0.557 / 0.671 and 0.535 / 0.557 |
| 021–094 seam ≈ 20.3 t | **20.667 t** (+1.6 %) |
| 094–104 seam ≈ 10.7 t | **12.333 t** (+15.6 %) |

Both seam figures are now written into R §13 beside the reference's own 20.333 t and 10.667 t, and
the row states a 20 % band instead of an "≈" (the second recorded item of V4 §7).

### 2.2 Determinism

Two runs of the release binary into two directories, every file compared byte for byte.

| set | files | identical | differing | with the wall clock removed |
|---|---|---|---|---|
| terracotta | 5 | 4 | `report.json` | identical outside `timings`, same timing keys in the same order |
| `pot_C` | 5 | 3 | `report.json`, `report.md` | identical outside `timings`; `report.md` identical before `## Timing` |

Both PNGs and `transforms.json` are byte-identical on both sets — including, now, the placement
order of `transforms.json`'s keys.

---

## 3. V4-D3 — measured, not restated

### 3.1 (a) Does the reference's own R §13 table survive its own randomness?

R §10's streams are seeded from `Params.seed`. `sherd_refit/cli.py` exposes no `--seed`, so the
knob is the dataclass field, and the driver sets it directly:

```python
pipeline.run(input_dir, out, target_faces=tf, workers=5,
             params=Params(seed=k), preview=False, refine=True, write_meshes=False)
```

Seven runs per set: seeds 0–4 at the default 200 000-face budget, then the budget at 190 000 and
210 000 (±5 %) at seed 0. `OMP_NUM_THREADS=1` throughout, scored with `tools/evaluate.py`.
21 runs, 3.3 h of wall clock.

**pot_B** — R §13 says 100 %, precision 1.000.

| run | fragment accuracy | precision | joins used | groups |
|---|---|---|---|---|
| seed 0 | **100 % (9/9)** | 1.000 | 9, all correct | one group of 9 |
| seed 1 | **88.9 % (8/9)** | 1.000 | 8, all correct | 8 + 1 |
| seed 2 | **88.9 % (8/9)** | 1.000 | 8, all correct | 8 + 1 |
| seed 3 | **100 % (9/9)** | 1.000 | 8, all correct | one group of 9 |
| seed 4 | **100 % (9/9)** | 1.000 | 9, all correct | one group of 9 |
| 190 000 faces | 100 % | 1.000 | 9 | one group of 9 |
| 210 000 faces | 100 % | 1.000 | 9 | one group of 9 |

**pot_G** — R §13 says 0 %, "no join must be accepted".

| run | fragment accuracy | joins used | which, and how far off the ground truth |
|---|---|---|---|
| seed 0 | 0 % | **0** | — |
| seed 1 | 0 % | **1** | 02–05, wrong pose, 1.47° / 1.11 t |
| seed 2 | 0 % | **2** | 01–03 (2.42° / 1.12 t), 02–05 (1.47° / 1.11 t) |
| seed 3 | 0 % | **1** | 02–05, 2.46° / 1.20 t |
| seed 4 | 0 % | **1** | 02–05, 1.47° / 1.11 t |
| 190 000 faces | 0 % | 0 | — |
| 210 000 faces | 0 % | 0 | — |

Every join the reference accepted on pot_G is a **wrong-pose join on a ground-truth-adjacent pair**;
none is cross-object, none is non-adjacent, and the fragment accuracy is 0 % in all seven runs.
R §13's "no join must be accepted" is a property of seed 0, not of the collection.

**synthetic_20** — R §13 says 95 %, precision 1.000.

| run | fragment accuracy | precision | joins used |
|---|---|---|---|
| seed 0 | 95 % (19/20) | 1.000 | 23, all correct |
| seed 1 | 95 % | 1.000 | 24, all correct |
| seed 2 | 95 % | 1.000 | 20, all correct |
| seed 3 | **85 % (17/20)** | 1.000 | 20, all correct |
| seed 4 | 95 % | 1.000 | 23, all correct |
| 190 000 faces | 95 % | 1.000 | 23 |
| 210 000 faces | 95 % | 1.000 | 22 |

Cross-object joins are 0 and group purity 1.000 in all 21 runs; precision is 1.000 in all 14 runs
of pot_B and synthetic_20. What moves is the fragment accuracy (pot_B 88.9–100 %, synthetic_20
85–95 %) and, on pot_G, the number of accepted joins (0–2).

### 3.2 (b) pot_G's two joins, put to the reference

The port accepts two joins on pot_G: **04–07** and **02–05**. `tools/evaluate.py` calls both
`wrong_pose` on ground-truth-adjacent pairs:

| join | rotation off the ground-truth relative pose | translation (centroid) |
|---|---|---|
| 04–07 | **0.7°** | **1.05 t** (2.96 units) |
| 02–05 | **2.5°** | **1.20 t** (3.39 units) |

The tolerance `evaluate.py` scores against is 5° and 0.5 t, so both are **near misses of adjacent
fragments**: the rotation is right to within a degree or two and the piece is slid about a wall
thickness along the seam. They are the same failure the reference produces at four of its five
seeds (02–05 at 1.47–2.46° and 1.11–1.20 t), on the same pair.

**The reference's own R §6 accepts the port's two poses.** A driver built the reference's own
`MatchData` for each pair, its own `Scales.for_fragments`, and called
`sherd_refit.matching.verify(A, B, T, sc, p)` and `accept(s, p, sc)` on the port's `T` out of the
port's `report.json`:

| pair | pose | score | `accept` | seam | tight A/B | gap | pen | cont_n |
|---|---|---|---|---|---|---|---|---|
| 04–07 | **the port's** | 21.15 | **True** | 32.67 | 0.647 / 0.762 | 0.029 | 0.0000 | 0.901 |
| 04–07 | the reference's own best (dump, seed 0) | 2.07 | False | 10.67 | 0.209 / 0.194 | 0.112 | 0.1646 | 0.972 |
| 02–05 | **the port's** | 18.31 | **True** | 36.67 | 0.499 / 0.525 | 0.038 | 0.0000 | 0.898 |
| 02–05 | the reference's own best (dump, seed 0) | 0.41 | False | 2.67 | 0.178 / 0.153 | 0.140 | 0.0758 | 0.981 |

The port's pose is 49° and 130 t away from the reference's seed-0 best on 04–07, and 78° / 163 t
away on 02–05 — a different basin, not a refinement of the same one. The reference's seed-1…4 runs
find the 02–05 basin themselves and score it 16.6–18.1, next to the port's 18.31.

**And the port's own verification agrees with the reference's on the reference's own candidates.**
`parity --stage verify --injected` on pot_G scores the reference's ten stage-2 candidates of each
pair through the port's R §6: `accepted` **10 of 10 identical** on both pairs, and every score
inside D §10.2's tolerances (worst on these two pairs: `gap` 4.7e-8 t against 2e-3 t; `tight`,
`seam`, `pen` exactly 0 difference). Over the whole set: `verify` injected, 213 checks, 0 failed.

### 3.3 (c) What that means, and how the gates are restated

The port's R §6 is the reference's R §6 — 10-of-10 verdicts on the reference's own candidates,
and the reference's own verifier accepting the port's poses. What differs is **which poses reach
the verification**, which is precisely what PMC-6 (the coarse NMS tie order) and PMC-9 (the
samplers) are licensed to change, and the reference's own search changes it under nothing but a
seed: 0, 1, 2, 1, 1 accepted joins on pot_G, 88.9–100 % on pot_B, 85–95 % on synthetic_20.

So R §13's per-set rows are now the measured spread, R §12.1 carries the derivation under PMC-6 and
PMC-9, and D §10.3's quality gate reads: **within the reference's own spread on every listed set,
cross-object joins 0, group purity 1.000, and R §13's terracotta row exactly.** pot_G keeps a hard
clause, because a prohibition is what a spread cannot absorb: at most two joins, and every join
used must be a wrong-pose join on a ground-truth-adjacent pair.

---

## 4. Parity: `--stage all`, both modes, all eight sets

`sherd-refit-rs parity --fixtures <dump> --input <collection> --stage all [--injected]`.
The `worst/tol` column is the largest ratio of a measurement to its own tolerance inside a stage.

| set | injected checks | failed | worst stage rows | native checks | failed | worst stage rows |
|---|---|---|---|---|---|---|
| slab | 246 | 0 | nms 0.60, outputs 0.07, hypotheses 0.03 | 79 | 0 | samples 0.46, candidates 0.10, breakline 0.02 |
| terracotta | 626 | 0 | stage1 0.69, nms 0.67, outputs 0.19 | 141 | 0 | working mesh 0.94, breakline 0.45, segmentation 0.42 |
| `pot_A` | 2 033 | 0 | **load 1.00**, nms 0.81, verify 0.54 | 298 | 0 | **load 1.00**, candidates 0.71, samples 0.37 |
| `pot_B` | 2 517 | 0 | **load 1.00**, nms 0.76, stage2 0.75 | 345 | 0 | **load 1.00**, candidates 0.67, samples 0.37 |
| `pot_C` | 1 604 | 0 | nms 0.53, load 0.50, outputs 0.41 | 254 | 0 | load 0.50, samples 0.47, candidates 0.38 |
| `pot_G` | 1 589 | 0 | stage2 0.54, outputs 0.41, nms 0.37 | 241 | 0 | candidates 0.38, samples 0.33, load 0.25 |
| `pot_H` | 3 638 | 0 | nms 0.87, outputs 0.62, stage2 0.50 | 448 | 0 | candidates 0.58, samples 0.45, load 0.25 |
| `synthetic_20` | 8 582 | 0 | nms 0.74, outputs 0.72, stage1 0.56 | 1 020 | 0 | working mesh 0.93, breakline 0.76, segmentation 0.62 |
| **total** | **20 835** | **0** | | **2 826** | **0** | |

**23 661 checks, 0 failed**, against V4's 23 653: exactly **+1 per set**, which is the new
`report md` row. Nothing that used to run now skips. `load 1.00` on pot_A and pot_B is D §10.2's
own documented boundary (Assimp's `fast_atof`, one `f32` ULP) and has no headroom by construction.

Two rows moved, both because of V4-D10:

| row | before (V4) | after |
|---|---|---|
| `assembly` injected, `recentre` (the port's R §8.2 against the reference's own `transforms.json`) | 8.8e-15 t on terracotta, **7.0e-13 t on pot_A** | **0** on both — bit-identical |
| `outputs` injected, `transforms pose` | 8.8e-15 t | **0** |

So the answer to V4-D10's question is that the 7.0e-13 t figure does not merely tighten: with
numpy's own summation order and `apply_transform_fused`, R §8.2's recentring is now **exact**
against the reference's own file on all eight dumps. R §12.1 records it.

`parity --verify-checksums` now covers the whole dump: 201 files on the slab (was 180 of 200),
542 on terracotta, 1 338 on pot_G, 8 734 on synthetic_20 — every one matching.

---

## 5. Suites

| command | result |
|---|---|
| `cargo fmt --all --check` | **OK** |
| `cargo clippy --workspace --all-targets -- -D warnings` | **OK** |
| `cargo test --workspace --no-fail-fast` (debug) | **OK** — 311 passed, 0 failed, 2 ignored |
| `cargo test --workspace --release --no-fail-fast` | **OK** — 311 passed, 0 failed, 2 ignored (V4-D2: this was 1 failed) |
| `cargo test --workspace --release --no-fail-fast -- --ignored` | **OK** — 2 passed (R §13's terracotta CLI gate, 6.5 s in release; the terracotta cache determinism test, 4.8 s) |
| `python -m pytest -q` | **OK** — 60 passed (V4-D1: this was 1 failed of 58) |
| `parity --stage all` × 8 sets × 2 modes | **PASS** — 23 661 checks, 0 failed (§4) |
| `parity --verify-checksums` | every file of every dump, manifest included (§4) |

The debug and release suites are the same 311 tests: 309 of V4's plus
`segment_writes_the_segmentation_preview` and `timings_keep_the_order_the_stages_finished_in`.

---

## 6. What is not closed

* **`synthetic_170` and `mixed_all`** — D §10.3's two large CPU gates (≤ 2 h each, ≤ 6 GB) are
  still unrun, as in step D3 and V4 §8. Nothing here changes that.
* **pot_A, pot_C and pot_H were not swept.** The seed/budget sweep covered pot_B, pot_G and
  synthetic_20, which is what the task named; R §13 marks the other three rows as unswept single
  draws, and pot_C's own paragraph already measured its precision at five seeds of the *old*
  thickness estimator.
* **The GPU column of every gate** — there is no GPU executor yet (phase 2a).
* **`--second-pass-top` and `--screen-top-k`** are off by default and no run or fixture here
  exercises them.
