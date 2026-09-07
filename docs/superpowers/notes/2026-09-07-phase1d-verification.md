# Task V4 — independent verification of phase 1d (steps X, D1, D2, D3)

**Date:** 2026-09-07. **Tree:** branch `rust-core` at `a8897cf` ("D3: the run, and the five native
gates"), working tree clean at the start of the task and clean at the end apart from this file.
**Machine:** Apple M2 Pro, 10 cores, 16 GB, macOS 24.6.0. **Toolchain:** rustc 1.97.0 (pinned by
`rust-toolchain.toml`). **Reference:** `sherd_refit/` at `895a948`, Open3D 0.19.0, numpy 2.5.2,
Python 3.12.9.

Everything below was re-derived from R, D and the Python at the fixture commit. The notes of steps
X, D1, D2 and D3 were read for what they *claim* and then set aside; no number in this document is
copied from them. Where a claim of theirs is reproduced here it was measured again, and the two
places where a measurement of theirs does not survive contact with the tree are called out.

**Verdict: the phase does not pass its own gates.** Three of them are red:

* `pytest -q` has been red since step D2's commit — the committed slab fixture carries twenty files
  its manifest does not list, and `tests/test_fixtures.py` exists to catch exactly that (**V4-D1**);
* `cargo test --workspace --release` fails one test of 220 in `sherd-core` (**V4-D2**), a
  release-only failure that the debug CI command does not see;
* R §13's quality table is not met natively on `pot_B`, `pot_G` and `synthetic_20` (**V4-D3**), and
  `pot_G` fails the one row of that table that is written as a prohibition.

Everything else is green, and green with room: the whole parity sweep — sixteen stages, eight sets,
both modes, **23 653 checks** — passes with **0 failures**, two full runs are byte-identical outside
the wall clock, and every runtime gate of D §10.3 passes at three to seven times the margin.

---

## 1. What was run

| step | command | result |
|---|---|---|
| build | `cargo clean` (59 915 files, 12.1 GiB) then `cargo build --release --workspace --all-targets` | **OK**, 1 m 43 s wall, 963 MiB peak RSS, no warnings |
| fmt | `cargo fmt --all --check` | **OK** |
| clippy | `cargo clippy --workspace --all-targets -- -D warnings` | **OK** |
| tests (debug) | `cargo test --workspace --no-fail-fast` | **OK** — 309 passed, 0 failed, 2 ignored |
| tests (ignored) | `cargo test --workspace --no-fail-fast -- --ignored` | **OK** — 2 passed (R §13's terracotta CLI gate, 111 s; the terracotta cache determinism test, 36 s) |
| doc tests | `cargo test --workspace --doc` | **OK** — 0 tests |
| tests (release) | `cargo test --workspace --release --no-fail-fast` | **FAIL** — 1 of 220 in `sherd-core --lib` (**V4-D2**) |
| parity | `parity --stage all` × 8 sets × {injected, native} | **PASS** — 23 653 checks, 0 failed (§3) |
| runs | `sherd-refit-rs run` × 7 collections, cold and warm | **OK** (§4, §6) |
| evaluation | `python tools/evaluate.py OUT IN` × 6 | 4 of 6 rows match the Python table (§4) |
| reference control | `sherd-refit run input/sfspp/pot_A` + `evaluate.py` | Python table **is current** (§4.2) |
| determinism | two full runs on terracotta and on `pot_C`, file by file | **PASS** (§5) |
| Python | `pytest -q` | **FAIL** — 1 of 58 (**V4-D1**) |

The fixture dumps were checked against R's header before anything else: all seven
`output/fixtures/*` say `commit 895a948`, `dirty false`, and `fixtures/slab/dump` says `9cbcbbc` —
exactly what R's header and D §10.1 name. `synthetic_20` is at level `slim`, the other seven at
`full`.

---

## 2. Diff review against R and the Python at `895a948`

Read line by line against the reference: `assembly.py`, `refine.py`, `render.py`, `report.py`,
`pipeline.py` and `cli.py` on one side; `assembly/{greedy,groups,consistency}.rs`, `refine.rs`,
`render.rs`, `report.rs`, `pipeline.rs`, `params.rs` and `sherd-cli/src/main.rs` on the other.

### 2.1 What is faithful

**R §8, the greedy pass.** Every ordering decision the reference makes is the one the port makes.
`best_per_pair` keeps the *first* candidate at a pair's top score (`c.score() > best.score()`,
strictly) and sorts the survivors by descending score with a **stable** sort over insertion order,
which is R §4.1's `itertools.combinations` order — `greedy.rs:152-179`, against `assembly.py:43-45`.
The scan iterates a **snapshot** (`for here in remaining.clone()`, `greedy.rs:308`) exactly as
`for c in list(remaining)` does, and restarts on the first successful placement. `try_place` runs
the penetration test over `groups[g]` in placement order skipping the anchor, then walks **all** of
`accepted` for the consistency test with an identity test on the candidate rather than a field
comparison (`greedy.rs:243-266` against `assembly.py:69-83`). `group_thickness` counts the new
member before it is placed. The loop-closing and merge branches, `add_singletons` in collection
order and the stable `sort_by_size` are the reference's. `rel`, `rotation_angle_deg` and its clip
are the reference's; the rejection sentences are formatted with the same precisions, and Rust's and
Python's `{:.1}/{:.2}/{:.3}/{:.4}` were checked to agree digit for digit on ten adversarial values
including exact halves (both round half to even).

**R §9, the refinement.** The cloud predicate is `frac[j] ∧ d < max(0.15·t, 1.5·res)` over a
KD-tree of the working mesh's face centroids (`refine.rs:150-167`), the vertex normals are Open3D's
`ComputeVertexNormals` — the sum of *unnormalised* incident face normals over the face loop, one
normalisation, `(0,0,1)` where Eigen would leave a NaN (`refine.rs:109-141`; `load_mesh` does not
pre-compute triangle normals, so Open3D's `ComputeTriangleNormals(false)` is what the reference
runs). The tree order is R §9's: `edges` filtered out of `used` **in `used` order**, "the first
edge with exactly one endpoint in `done`", `done` seeded with `g[0]`, and
`poses[moving] ← T · poses[moving]` (`refine.rs:281-309`). The **radii are the reference's**:
`sc.icp_dist(0.05)` then `sc.icp_dist(0.02)`, 40 iterations each, point-to-plane, through the same
`icp::register` as every other stage, with `Scales::for_pair(min(t_fixed,t_moving),
max(res_fixed,res_moving))` (`refine.rs:322-348` against `refine.py:71-80`); Open3D's
`relative_fitness`/`relative_rmse` of 1e-6 are in `icp.rs:72-74`. The clouds are moved with
`apply_transform_fused`/`rotate_fused`, which is Open3D's `PointCloud::Transform`.

**R §11.1–11.4, the output schemas.** `Params` is 46 fields with the reference's names, order and
defaults — checked field by field against `asdict(Params())`. `FragmentStats` is `Fragment.stats()`
key for key and in its order. `report.json`'s top-level keys come out in the reference's own order
(`thickness, fragments, groups, params, timings, joins_used, joins_rejected, candidates`), verified
against `output/fixtures/terracotta/_run/report.json`, plus D §4.3's additive `engine`.
`transforms.json`'s top-level keys likewise. `report.md` was diffed against the reference's own
file for terracotta: **every heading, every column, every number format and the legend line are
identical**; the only structural difference is the order of the `## Timing` list (**V4-D4**).
`json.dump(..., indent=1)` is reproduced by a `PrettyFormatter::with_indent(b" ")`. R §11.4's
placed meshes are the *cleaned but not largest-component* original moved by `apply_transform_fused`,
and the parity harness compares them by SHA-256.

**R §11.5, the renderer.** All four arithmetic details are right and were re-derived: the two
`(n,3)@(3,3)` products are fused and `Nn @ light` is not (`render.rs:335-347`, `matmul_fused` at `render.rs:449-452`), `np.round` is
`round_ties_even` (`render.rs:403-404`), the z-buffer is `f32` while the depth compared against it
is `f64` (`render.rs:367-374`), and within one of the nine offsets the **last** of the deepest
points wins, which is `np.lexsort`'s stability plus the `last` mask (`render.rs:355-362`). The
framing, the camera basis and its degenerate escape, the 0.16 background, the clip-and-quantise and
the strip concatenation are the reference's. The captions are PMC-20 and the harness measures their
area rather than comparing them.

**Parity evidence for all of the above**, rather than reading alone: injected `preview px` is **0
differing pixels of 4 200 000** on every preview of every set, injected `placed ply` is SHA-256
identical on every file of the four PLY sets, and injected `report schema` and `report candidates`
are exact (1 324 JSON leaves and 30 candidates on terracotta alone).

### 2.2 Deviations that are **not** on R §12 or R §12.1

Ten, listed as defects in §7. Four of them have a measurable consequence in a file the user sees
(**V4-D3, V4-D4, V4-D5, V4-D7**); the rest are sub-tolerance arithmetic, unreachable branches or
documentation that asserts something that is not true.

### 2.3 One place where R is wrong and the code is right

R §11.2 lists `"output"` among `timings`' keys. The reference never writes it: `write_report`
serialises the dict with `json.dump` before `pipeline.run` sets `timings["output"]`, so
`report.json` and `report.md` carry every stage but that one. Verified on the reference's own
`_run/report.json` (`['preprocess','matching','assembly','refine']`). The port follows the code and
says so at `pipeline.rs:424-429`. R §11.2 should be amended.

---

## 3. Parity: `--stage all`, both modes, all eight sets

`sherd-refit-rs parity --fixtures <dump> --input <collection> --stage all [--injected]`.
No stage failed anywhere. The `worst/tol` column is the largest ratio of a measurement to its own
tolerance inside the stage, so `1.00` is a row sitting exactly on its limit and `0.00` is exact.

### 3.1 Injected

| set | checks | failed | worst stage rows (`worst/tol`) | skipped |
|---|---|---|---|---|
| slab | 245 | 0 | samples 7.3e-3, nms 0.60, breakline 4.9e-3, outputs 0.07 | 0 |
| terracotta | 625 | 0 | stage1 0.69, nms 0.67, outputs 0.19 | 0 |
| pot_A | 2 032 | 0 | **load 1.00**, nms 0.81, stage2 0.50, verify 0.54 | 1 |
| pot_B | 2 516 | 0 | **load 1.00**, stage2 0.75, nms 0.76 | 1 |
| pot_C | 1 603 | 0 | nms 0.53, load 0.50, outputs 0.41 | 1 |
| pot_G | 1 588 | 0 | stage2 0.54, nms 0.37, outputs 0.41 | 2 |
| pot_H | 3 637 | 0 | nms 0.87, outputs 0.62, stage2 0.50 | 1 |
| synthetic_20 | 8 581 | 0 | nms 0.74, outputs 0.72, stage1 0.56 | 148 |
| **total** | **20 827** | **0** | | |

`load 1.00` on pot_A and pot_B is D §10.2's own documented boundary (Assimp's `fast_atof`, exactly
one `f32` ULP) and it has no headroom by construction — the design says so, and this run confirms
the row is still *at* its limit rather than over it. The skips are all named by the harness:
injected thickness skips synthetic_20's twenty fragments because the `slim` dump has no `load.V0`;
`refine` skips pot_G because the reference assembled nothing there; the `placed ply` row skips the
five OBJ sets because their sources do not read back byte-identically.

### 3.2 Native

| set | checks | failed | worst stage rows (`worst/tol`) | skipped |
|---|---|---|---|---|
| slab | 79 | 0 | samples 0.46, refine 4.7e-14 | 1 |
| terracotta | 141 | 0 | working mesh 0.94, breakline 0.45, segmentation 0.42 | 6 |
| pot_A | 298 | 0 | **load 1.00**, samples 0.37, candidates 0.71 | 28 |
| pot_B | 345 | 0 | **load 1.00**, samples 0.37, candidates 0.67 | 36 |
| pot_C | 254 | 0 | load 0.50, samples 0.47, candidates 0.38 | 21 |
| pot_G | 241 | 0 | samples 0.33, candidates 0.38, load 0.25 | 22 |
| pot_H | 448 | 0 | candidates 0.58, samples 0.45, load 0.25 | 55 |
| synthetic_20 | 1 020 | 0 | working mesh 0.93, breakline 0.76, samples 0.57 | 210 |
| **total** | **2 826** | **0** | | |

`coarse`, `nms`, `stage1`, `stage2` and `verify` have no native column by design (PMC-9 gives the
two sides different samples, so there is nothing to compare a native pose against); the harness
prints the reason per pair.

### 3.3 The three rows phase 1d added

| set | `assembly` inj. | `assembly` nat. | `refine` inj. | `refine` nat. | `outputs` inj. | `outputs` nat. |
|---|---|---|---|---|---|---|
| slab | 6.9e-6 | 0.00 | 4.7e-14 | 4.7e-14 | 0.07 | 0.00 |
| terracotta | 8.8e-6 | 3.1e-6 | 0.03 | 0.20 | 0.19 | 0.00 |
| pot_A | 7.0e-4 | 0.26 | 8.9e-5 | 8.9e-5 | 0.46 | 0.00 |
| pot_B | 3.3e-4 | 0.17 | 7.4e-5 | 7.4e-5 | 0.59 | 0.00 |
| pot_C | 2.4e-5 | 0.25 | 3.4e-6 | 3.4e-6 | 0.41 | 0.00 |
| pot_G | 1.4e-4 | 0.17 | (skip) | (skip) | 0.41 | 0.00 |
| pot_H | 2.2e-4 | 0.20 | 1.7e-4 | 1.7e-4 | 0.62 | 0.00 |
| synthetic_20 | 2.9e-4 | 0.50 | 7.8e-5 | 7.8e-5 | 0.72 | 0.00 |

Terracotta, in detail, is the shape of all eight:

```
assembly (injected): groups 0/2 differing, used 0/2, rejected 0/0,
                     pose move 3.1e-15 t, pose move centred 2.1e-15 t, recentre 8.8e-15 t   (tol 1e-9 t)
refine   (injected): idx candidates 0/450 000 differing, idx overlap 1.3e-3 (tol 5e-2),
                     walk 0/2, dist 0/0, rung 1 rot 1.7e-6 deg, rung 2 rot 2.4e-6 deg (tol 0.2 deg),
                     rung trans 6.2e-15 t (tol 0.02 t), fitness 0, rmse 2.6e-17 t
outputs  (injected): transforms thickness exact, groups exact, flags 0/4, params exact,
                     pose 8.8e-15 t; report schema 0/1324, report candidates 0/30;
                     placed shape 0/5, placed ply 0/5 (SHA-256); preview px 0/4 200 000;
                     preview label px 1 490 (cap 8 000, PMC-20)
```

The native `assembly` column is dominated by `pmc8 pen` (0.26 on pot_A = 0.0078 against a 0.03
tolerance); the `pmc8 groups`, `pmc8 used` and `pmc8 rejected` rows are exact on all eight, so
PMC-8's re-verify column is met independently of step D1's note. Two of the constants behind the
native column are not the ones D §10.2 states — see **V4-D6**.

---

## 4. `run` on the seven collections, and `evaluate.py`

`sherd-refit-rs run <input> --out <dir>` with every default (previews and meshes **on**, which is
more than D §10.3's gate asks for), cold then warm into the same directory.

### 4.1 The quality table

| set | port fragment accuracy | Python (R §13) | port precision | Python | port joins used (correct) | port groups | purity |
|---|---|---|---|---|---|---|---|
| terracotta | — (R §13's join gate, §4.3) | — | — | — | 2 (both R §13's) | 1 + 1 singleton | — |
| `pot_A` | **100.0 % (8/8)** | 87.5 % | 1.000 | 1.000 | 7 (7) | one group of 8 | 1.000 |
| `pot_B` | **88.9 % (8/9)** | 100 % | 1.000 | 1.000 | 7 (7) | 8 + 1 | 1.000 |
| `pot_C` | 75.0 % (3/4) | 75 % | 0.667 | 0.667 | 3 (2, 1 unscorable) | 3 + 2 + 2 singletons | 1.000 |
| `pot_G` | 0 % (0/7) | 0 % | **0.000 (2 joins)** | — (**0 joins**) | 2 (0 correct, 2 wrong pose) | 2 + 2 + 3 singletons | 1.000 |
| `pot_H` | 36.4 % (4/11) | 36.4 % | 0.429 | 0.429 | 7 (3) | 7 + 2 + 2 singletons | 1.000 |
| `synthetic_20` | **90.0 % (18/20)** | 95 % | 1.000 | 1.000 | 19 (19) | 15 + 3 + 2 singletons | 1.000 |

Bold marks a row that does not match R §13. Cross-object joins are 0 everywhere and group purity is
1.000 everywhere, so the two "all sets" rows of R §13 pass. Three of the seven per-set rows do not:
`pot_B` and `synthetic_20` are below the reference, `pot_A` is above it, and **`pot_G` accepts two
joins where R §13 says "no join must be accepted"** — both land on ground-truth-adjacent pairs at a
wrong pose. This is **V4-D3**.

The measurements agree with step D3's own table, so this is a confirmation of that note and not a
new regression: D3 reported the same six rows. What D3's note frames as "four of the six rows match,
one is better and one worse" is, read against R §13 and D §10.3 ("Quality: exactly R §13 on every
listed set, run natively"), **three failing rows**, because pot_G's prohibition is a row too.

### 4.2 The Python table is current

Re-ran the reference itself: `sherd-refit run input/sfspp/pot_A --out …` with `OMP_NUM_THREADS=1`,
then `tools/evaluate.py`. Result: **7 of 8 = 87.5 %, precision 1.000, 6 joins all correct, one group
of 7** — exactly the figure R §13 and the task's table carry. 161.6 s wall, 912 MiB peak RSS
against the port's 11.7 s cold and 619 MiB, a **13.8×** speed-up on this set.

### 4.3 R §13's terracotta gate

Passes, and passes through two independent paths: the `#[ignore]`d integration test
`the_terracotta_assembles_the_two_joins_of_r_13` (111 s), and my own run.

| R §13 asks | measured |
|---|---|
| joins used exactly {021–094, 094–104} | **exactly those two, in that order; nothing rejected** |
| 007 unplaced | **unplaced**, its own group |
| both `pen` = 0 | **0.0000 / 0.0000** |
| tight of both ≥ 0.27 | **0.5566 / 0.5347** |
| 021–094 seam ≈ 20.3 t | 20.7 t (+2 %) |
| 094–104 seam ≈ 10.7 t | **12.3 t (+15 %)** |

The reference's own numbers for the same run, from `output/fixtures/terracotta/_run/report.md`:
score 10.79 / 6.06, seam 20.3 / 10.7 t, tight 0.53 / 0.69 and 0.59 / 0.57. The decisions are
identical; the second seam is the one measurement of R §13's terracotta row that "≈" is doing real
work for, and it is recorded here so the next reader does not have to rediscover it.

---

## 5. Determinism

Two full runs of the release binary into two separate output directories, every file compared by
SHA-256, caches excluded (they have their own gate in `segment_cli.rs`).

| set | files | byte-identical | differing | what differs |
|---|---|---|---|---|
| terracotta | 10 | **8** | 2 | `report.json`, `report.md` |
| `pot_C` | 15 | **13** | 2 | `report.json`, `report.md` |

Both differing files were then compared with the wall clock removed: `report.json` is **identical
outside `timings`** and its timing key set is identical; `report.md` is **identical before
`## Timing`** and its stage names are identical. So the only thing two runs disagree on is the
seconds, which is what D §10.4 layer 4 asks for. The byte-identical set includes both PNGs, all
`placed/*.ply`, `assembly_*.ply` and `transforms.json`.

The two `#[ignore]`-free CLI tests cover the same ground on the committed slab and passed in both
the debug and the release suites.

---

## 6. Time and memory

`/usr/bin/time -l`, release binary, previews and meshes written.

| set | cold wall | warm wall | cold peak RSS | warm peak RSS | D §10.3 gate | margin |
|---|---|---|---|---|---|---|
| terracotta (6 pairs) | 6.19 s | **3.87 s** | 1 141 MiB | 488 MiB | ≤ 25 s | 6.5× |
| `pot_A` (28 pairs) | 11.68 s | **10.14 s** | 619 MiB | 586 MiB | ≤ 35 s | 3.5× |
| `pot_B` (36 pairs) | 11.42 s | 10.36 s | 576 MiB | 600 MiB | — | — |
| `pot_C` (21 pairs) | 6.26 s | 6.17 s | 304 MiB | 298 MiB | — | — |
| `pot_G` (21 pairs) | 6.18 s | 6.04 s | 328 MiB | 311 MiB | — | — |
| `pot_H` (55 pairs) | 12.28 s | **11.11 s** | 538 MiB | 517 MiB | ≤ 40 s | 3.6× |
| `synthetic_20` (190 pairs) | 52.41 s | **44.28 s** | 1 577 MiB | 1 588 MiB | ≤ 120 s | 2.7× |

Every gate D §10.3 states for the CPU is met with the previews and the meshes on, i.e. under a
heavier load than the gate describes. Peak RSS stays under 1.6 GiB on the largest set, well inside
D §8's budget. The two large sets (`synthetic_170`, `mixed_all`) were not run, as in step D3.

The build itself: `cargo clean` removed 59 915 files / 12.1 GiB, and the release rebuild of the
whole workspace with all targets took **1 m 43 s** wall (697 s user) at 963 MiB peak RSS.

---

## 7. Defects

Ten, most severe first. Each gives `file:line`, what R or D says, and what the code does.

### V4-D1 — `pytest -q` is red: the committed slab dump and its manifest disagree (step D2)

* **Where.** `fixtures/slab/dump/manifest.json` (180 entries) against 200 files on disk;
  `tools/dump_outputs.py:226-241`, which writes into an existing dump and never rewrites the
  manifest. Test: `tests/test_fixtures.py:80-86`.
* **D says.** D §10.1 gives the dump layout as `DIR/manifest.json {commit, …, files: {path:
  {shape, dtype, sha256}}}` and then lists, inside that same layout, `DIR/outputs/ (added by
  tools/dump_outputs.py, step D2) placed.sha256.json, preview_index.json, preview_<k>.png,
  preview_<k>.nolabel.png, preview_<k>.meta.json, preview_<k>.<name>.{pick,u,v}.npy, and the same
  for preview_segmentation`. `tests/test_fixtures.py::test_committed_slab_dump_matches_its_manifest`
  asserts `{rel for _, rel in fixture.iter_files(SLAB_DUMP)} == set(man["files"])`.
* **Code does.** Commit `b9cc4b8` (D2) added twenty `fixtures/slab/dump/outputs/*` files and did not
  touch `manifest.json` (last changed at `f256074`, step C1). `pytest -q`: **1 failed, 57 passed**.
* **Second consequence, worse than the red test.** `sherd-refit-rs parity --verify-checksums` prints
  `checksums: all 180 files match the manifest` while the twenty files the `outputs` row actually
  reads — every `preview_*.png`, every `pick`/`u`/`v`, and `placed.sha256.json` — are unhashed. The
  same gap exists in all seven `output/fixtures` sets (32 extra files on terracotta, 56 on pot_A,
  128 on synthetic_20). A stale or hand-edited preview reference would pass the integrity check.
* **Fix.** Have `dump_outputs.py` extend `manifest.json["files"]` with the entries it writes (the
  sink's own hashing helper already exists), then re-commit the slab manifest.

### V4-D2 — `cargo test --workspace --release` fails one test

* **Where.** `crates/sherd-core/src/io/mod.rs:186-192`.
* **What happens.**
  `test io::tests::colours_quantise_the_way_open3d_writes_them ... FAILED` —
  `panicked at crates/sherd-core/src/io/mod.rs:190:17: assertion left == right failed, left: 255,
  right: 0`. Debug passes; release fails; it is the **only** failure in the release suite
  (219 of 220 pass in that binary, every other binary green).
* **Cause, reduced to 17 lines.** The loop is `for k in 0..=255_u8 { assert_eq!(…, k); if k < 255 {
  assert_eq!(quantize_color(f64::from(k) + 0.5), k + 1); } }`. A standalone program with only
  `quantize_color` and that loop, built with `rustc 1.97.0` (aarch64-apple-darwin) at
  `-C opt-level=2` and `-C opt-level=3`, panics identically; at `-C opt-level=0` and `1` it passes.
  In the optimised IR the `k < 255` guard has been folded into `RangeInclusive`'s own exhaustion
  test, whose increment carries `add nuw i8` from `Step::forward_unchecked`, and the last iteration
  runs the guarded body with a poisoned `k + 1` that materialises as `0`. **`quantize_color` itself
  is correct** — `254.5 → 255`, `255.5 → 255`, `-1 → 0`, `NaN → 0` all verified directly.
* **Why CI is green.** `.github/workflows/rust.yml:72` runs `cargo nextest run --workspace
  --all-targets` in the **debug** profile, which is the profile where the pattern survives.
* **Fix (both verified).** `for k in 0..255_u8 { … }` with the `k = 255` identity asserted once
  after the loop, or `for k16 in 0..=255_u16 { let k = k16 as u8; … }`.

### V4-D3 — R §13's quality table is not met natively on three of seven sets

* **Where.** The whole native path; the measurement is §4.1 above.
* **R says.** R §13: `pot_B` 100 %, `pot_G` "0 % (ground truth interpenetrates; **no join must be
  accepted**)", `synthetic_20` 95 %, `pot_A` 87.5 %. D §10.3: "Quality: **exactly** R §13 on every
  listed set, run natively (no injection)".
* **Code does.** `pot_B` 88.9 %, `synthetic_20` 90.0 %, `pot_A` 100.0 %, and `pot_G` **accepts two
  joins** (precision 0.000; both are ground-truth-adjacent pairs placed at a wrong pose).
* **What it is and is not.** It is not a defect of R §8, R §9 or R §11: the `assembly` row's PMC-8
  pass reproduces the reference's groups, used joins and rejections **exactly** on all eight dumps
  from the reference's own candidates, so the divergence is upstream, in the candidate sets that
  PMC-6 (the coarse NMS tie-break) and PMC-9 (the samplers) are licensed to change. What it *is* is
  the re-verification those two rows name: PMC-6's re-verify column is "pair gates" and PMC-9's is
  "injected-sample parity + **statistical gates**", and R §13 is the set of gates §12.1's C1
  addendum points at. **Those two licences are therefore still asserted rather than demonstrated at
  the collection level**, and `pot_G` is the clearest case because its row is a prohibition that no
  tolerance can absorb.
* **Not a new regression.** Step D3 measured the same six rows and attributed them the same way.
  What this note adds is that R §13 and D §10.3 state the gate as "exactly", that `pot_G`'s row is
  one of the three that fail, and that neither R §12.1 nor D §10.3 has been amended to record a
  weaker quality gate.

### V4-D4 — `timings` come out alphabetically, not in stage order

* **Where.** `crates/sherd-core/src/pipeline.rs:232` (`let mut timings: BTreeMap<String, f64>`),
  `crates/sherd-core/src/report.rs:208` (`pub timings: BTreeMap<String, Option<f64>>`), rendered at
  `report.rs:499-501`.
* **R says.** R §11.2: `"timings": {"preprocess", ["screen"], "matching", "assembly",
  ["second_pass"], ["refine"], "output"}` — an ordered list, and the reference's `dict` is in
  insertion order. R §11.3's `## Timing` prints that dict.
* **Code does.** A `BTreeMap`, so both `report.json` and `report.md` sort by name. Measured against
  the reference's own terracotta files: reference `preprocess, matching, assembly, refine`; port
  `assembly, matching, preprocess, refine`. This is the **only** difference between the two
  `report.md` files apart from the numbers themselves.
* **Fix.** Keep the insertion order (an `IndexMap`, or a `Vec<(String, f64)>` with a
  `serialize_with`), or emit the stages in R §11.2's fixed order.

### V4-D5 — `transforms.json`'s `fragments` object is keyed in name order, not placement order

* **Where.** `crates/sherd-core/src/report.rs:123` (`pub fragments: BTreeMap<String, Placement>`),
  filled at `report.rs:266-276` by walking `names` in collection order.
* **R says.** R §11.1: `"fragments": { name: {…} }`. The reference builds it from `poses.items()`,
  and `poses` is an insertion-ordered dict written by R §8's greedy loop (seed first, then each
  placement, then the singletons) and copied through `recenter`.
* **Code does.** Sorted by name. Measured on terracotta: reference
  `FY234021, FY234094, FY234104, FY234007` (placement order); port
  `FY234007, FY234021, FY234094, FY234104` (name order).
* **Impact.** None on any reader — `evaluate.py` and the parity harness both index by name — but the
  file is not the reference's file, and R §11.1 is the schema the port claims to write.

### V4-D6 — two of D §10.2's stated tolerances are not the ones the code enforces

* **Where.** `crates/sherd-parity/src/stages/assembly.rs:440` (`NATIVE_JOINS = 12.0`) and `:445`
  (`NATIVE_GROUP_SIZE = 8.0`); `crates/sherd-parity/src/stages/outputs.rs:73` (`POSE_T = 1e-9`).
* **D says.** D §10.2's `assembly` row, native column: "share of the used-join union belonging to
  one side **≤ 0.5 each**; largest group **within 4 fragments**". D §10.2's `outputs` row, injected
  column: `transforms.json` — "…, the poses" → "**exact**".
* **Code does.** An absolute **count** of ≤ 12 joins per side rather than a share, and a
  largest-group difference of ≤ 8 rather than 4 — twice the measured worst in both cases, with the
  reasoning written out in the constants' own doc comments (a share is degenerate on pot_G, where
  the reference used no join at all). `POSE_T` is `1e-9 t` rather than "exact", and its comment says
  D §10.2 asked for 0.02 t, which is not what the row printed either.
* **Why it matters.** The substitutions are defensible — the share form really is degenerate on
  pot_G, and a pose that is a chain of 4×4 products cannot be bit-exact through a different matrix
  kernel — but D §10.2 was not amended for either, and the team rule established in step X's own
  addendum (PMC-2's two breakline rows) is that a tolerance change is written into D with its
  derivation. As it stands, the document states a gate the harness does not run, and the code's
  count form is what lets pot_G's native assembly row pass at all.

### V4-D7 — `sherd-refit-rs segment` writes no segmentation preview, and has no `--workers`

* **Where.** `crates/sherd-cli/src/main.rs:361-452` (`fn segment`), `:204-224` (`SegmentArgs`).
* **R and D say.** R's `pipeline.segment_only` ends with `write_previews(out_dir, frags, {n: I},
  [[n] …])`, which writes `preview_segmentation.png`. D §9: "`sherd-refit run …` **and**
  `sherd-refit segment` accept **every** Python flag with the same name, default and meaning";
  `cli.py`'s `segment` has `--target-faces`, `--workers` and `-v`.
* **Code does.** Preprocesses, prints a table, writes caches, and stops. No preview is written even
  though `render.rs` and `pipeline::write_previews` both exist since step D2 — `main.rs:51-52` still
  carries the stale promise "The segmentation preview the reference's `segment` also produces
  arrives with the renderer (phase 1d)". `SegmentArgs` has `--threads`, `--force` and `--no-cache`
  but no `--workers`.

### V4-D8 — `--workers` defaults to one thread per core; the reference's default is cores − 1

* **Where.** `crates/sherd-cli/src/main.rs:79-81` and `:465`, `:477`;
  `crates/sherd-core/src/pipeline.rs:230` (`if options.workers == 0 { rayon::current_num_threads() }`).
* **R says.** R §1.4 makes `--workers` a pipeline option; `cli.py:18` declares it `default=None` and
  `pipeline.run` resolves `workers = workers or max(1, (os.cpu_count() or 2) - 1)` — nine on this
  machine, not ten.
* **Code does.** Ten. Checked for consequences: `block_size` returns the same value at 9 and at 10
  workers for every benchmark set (6, 21, 28, 36, 55 and 190 pairs), and the block schedule does not
  affect results in any case because the candidates are collected by index. So the effect is nil
  today and the deviation is D §9's "same default".

### V4-D9 — R §9's cap is seeded from `params.seed`, and R §10 seeds it `0`

* **Where.** `crates/sherd-core/src/pipeline.rs:688-694` passes `params.seed` into
  `fracture_cloud`; `crates/sherd-core/src/refine.rs:174-181` seeds the draw with it.
* **R says.** R §10's RNG inventory: "refinement (§9) | **0** | `choice(idx, 150000,
  replace=False)`". `refine.py:35` is `np.random.default_rng(0)`, a literal.
* **Code does.** `rng::seeded(params.seed)`. Unreachable today: neither CLI exposes `--seed`, so
  `p.seed` is always `0` and the two agree. Worth fixing before a `--seed` flag exists, since the
  reference's refinement would not move and the port's would.

### V4-D10 — R §8.2's centroid uses a summation order the reference does not, and two doc comments assert the opposite

* **Where.** `crates/sherd-core/src/assembly/groups.rs:194-195` and `:219`;
  `crates/sherd-core/src/render.rs:215-216` and `:229`; `crates/sherd-core/src/pipeline.rs:778`.
  Also `groups.rs:208`, which moves the points with `apply_transform` rather than
  `apply_transform_fused`.
* **R says.** R §8.2's `c` is `pts.mean(0)` over the concatenated `(N,3)` array; R §11.5's
  `principal_views` is `X = V − V.mean(0)`; the segmentation preview's offset is `P − P.mean(0)`.
* **Code does.** `pairwise_sum(&column) / n` in all three places, documented at `groups.rs:195` as
  "numpy reduces with `pairwise_sum` rather than left to right" and at `render.rs:216` as "which is
  how numpy reduces an axis". **Both claims are false for this shape.** Measured on numpy 2.5.2, a
  20 001 × 3 C-contiguous array: `a.mean(0)` is **bit-identical to a left-to-right per-column
  accumulation** on all three columns and differs from the pairwise sum by 8 ULP
  (`0x1.502532392cf5fp+10` against `0x1.502532392cf57p+10`). numpy's pairwise summation applies to a
  reduction over the *contiguous* axis; an axis-0 reduction of a 2-D array accumulates row by row.
* **Effect.** Below every gate: the `assembly` row's `recentre` check — the one place the port's
  `recenter` is compared against the reference's own `transforms.json` rather than against itself —
  measures **8.8e-15 t** on terracotta against a 1e-9 t tolerance, and the native `axis` row is
  0.000° against 1°. So this is an undeclared sub-tolerance arithmetic deviation plus two comments
  that will mislead the next reader. Note that the `pose move centred` row cannot catch it: it
  recentres **both** sides with the port's own `recenter`, so the arithmetic cancels.

---

## 8. What I could not check

* `synthetic_170` and `mixed_all` — D §10.3's two large gates. Not run here, and not run in step D3
  either; the note says so and this one repeats it.
* The GPU column of every gate: there is no GPU executor yet (phase 2a).
* `--second-pass-top` and `--screen-top-k` paths: off by default, so R §8.1 and R §4.3 were read but
  not exercised by any run or fixture here. The code was reviewed against the reference (the
  second-pass tie rule is `sorted(key=(-score, pair))` on both sides and has its own unit test).
* PMC-20's caption is given up by design and only its area is bounded; that is D §10.2's own choice
  and this note does not revisit it.

## 9. Commands, for reproduction

```
PATH="$HOME/.cargo/bin:$PATH" cargo clean
PATH="$HOME/.cargo/bin:$PATH" cargo build --release --workspace --all-targets
PATH="$HOME/.cargo/bin:$PATH" cargo fmt --all --check
PATH="$HOME/.cargo/bin:$PATH" cargo clippy --workspace --all-targets -- -D warnings
PATH="$HOME/.cargo/bin:$PATH" cargo test --workspace --no-fail-fast
PATH="$HOME/.cargo/bin:$PATH" cargo test --workspace --no-fail-fast -- --ignored
PATH="$HOME/.cargo/bin:$PATH" cargo test --workspace --release --no-fail-fast     # V4-D2

for m in "--injected" ""; do
  ./target/release/sherd-refit-rs parity --fixtures fixtures/slab/dump --input fixtures/slab/input --stage all $m
  ./target/release/sherd-refit-rs parity --fixtures output/fixtures/terracotta --input input/test_fragments_1/fragments --stage all $m
  ./target/release/sherd-refit-rs parity --fixtures output/fixtures/pot_A --input input/sfspp/pot_A --stage all $m
  # … pot_B, pot_C, pot_G, pot_H, and
  ./target/release/sherd-refit-rs parity --fixtures output/fixtures/synthetic_20 --input input/synthetic_pingsdorf_20/fragments --stage all $m
done

./target/release/sherd-refit-rs run input/sfspp/pot_A --out OUT && python tools/evaluate.py OUT input/sfspp/pot_A
source .venv/bin/activate && OMP_NUM_THREADS=1 sherd-refit run input/sfspp/pot_A --out PYOUT
source .venv/bin/activate && python -m pytest -q                                   # V4-D1
```
