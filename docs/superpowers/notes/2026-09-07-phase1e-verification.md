# V5 — independent verification of phase 1e (steps Y, E1, E2)

**Date:** 2026-09-07/08. **Tree:** branch `rust-core` at `fd5d5cf` ("E2 note: the file counts of the
byte-identity gate, counted"); `git status` clean at the start and nothing left over from an
interrupted attempt. **Machine:** Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 24.6.0,
rustc 1.97.0, `--release`. **Reference:** `sherd_refit/` at the working tree's own copy, Open3D
0.19.0, numpy 2.5.2, Python 3.12.9, every Python run with `OMP_NUM_THREADS=1`.

Everything below was re-derived rather than read off the notes: the suites were re-run from a
`cargo clean`, all sixteen parity stages were re-run on all eight dumps in both modes, all eight
development collections were re-run and re-scored, the byte-identity claim of task E2 was
reproduced against a **freshly built `7cec806` binary** (from `git archive`, its own target
directory) rather than against the copy E2 left in `output/bench_e2/bin/`, and two of task Y's
reference sweeps were re-run in Python.

**Verdict: the gates pass.** Every measurement of the three notes that this verification
re-derived came back the same, to the digit where the digit is deterministic: 23 661 parity checks
and 0 failures, the seven quality rows unchanged from V4 §4.1, E2 §9's four timing rows inside
11 %, and 70 output files byte-identical to phase 1d's binary. Eight defects are listed in §9; none
of them moves a result, and the largest — `mixed_ABG` — is a set that D §10.3, D §12 and README all
name as a development set and that no note in the phase ever ran (§4.4: neither implementation
meets D §10.3's "cross-object joins 0, purity 1.000 everywhere" on it, and the port is the cleaner
of the two).

Nothing above 27 fragments was started here, and nothing in the phase started one either (§1.4).

---

## 1. Diff review: `4ca6750..fd5d5cf`, against R and the Python

Nineteen commits, 35 files, +4 834 / −212; eighteen of the files are under `crates/sherd-core`
(+1 975 / −121). Read against R, D and `sherd_refit/`, the question being the task's: did anything
move a result, a default, a threshold, a key order or a seed?

### 1.1 What moved, and whether it should have

| what | commit | the reference says | verdict |
|---|---|---|---|
| `pipeline::default_workers()` = cores − 1 | Y4 `8c4b5ae` | `pipeline.py:295` and `:476` — `workers or max(1, (os.cpu_count() or 2) - 1)`, in **both** `run` and `segment_only` | **correct**; the port used one per core (V4-D8). Moves no result, and this was recomputed rather than taken from the note: the pair counts after R §4.1's wall-ratio filter are 6, 28, 36, 21, 21, 55, 273 and 190, and `block_size` returns the same value at 9 and at 10 workers for every one of them (only 37–40 and 144–159 pairs would separate them) |
| `refine::CAP_SEED = 0` | Y4 | `refine.py:35` is `np.random.default_rng(0)`, a literal, where every other stream takes `p.seed` | **correct**. The same number today (no CLI exposes `--seed`) |
| `transforms.json`'s `fragments` in R §8 placement order | Y4 | `assembly.py` fills `poses` seed-first (`:130`), then each placement (`:116`), then the singletons in collection order (`:135`); `json.dump` writes a dict in insertion order | **correct, and checked against the reference's own files**: `output/fixtures/*/_run/transforms.json` come out in exactly that order on all seven collections (§4.3) |
| `timings` in stage order | Y4 | same argument; `report.py:101` dumps the dict | **correct**, and the reference's own `_run/report.json` carry `['preprocess','matching','assembly','refine']` |
| `column_mean` replaces `pairwise_sum` in R §8.2 and in `render::principal_axes` | Y4 | `assembly.py:154` `pts.mean(0)` on a C-contiguous `(N,3)` array | **correct** for §8.2 — the `assembly` parity row's `recentre` check is exactly 0 on all eight dumps (§2). The second call site is inside PMC-10's own territory and is gated only by the `outputs` `axis` row at 1°; R §12.1's task-Y addendum names §8.2 alone (**V5-D7**) |
| `segment` writes `preview_segmentation.png`, gains `--workers` | Y4/Y7 | `pipeline.segment_only` ends with `write_previews(...)`; `cli.py:49` declares `--workers` | **correct** (V4-D7) |
| the `0..255_u8` loop in `io::tests` | Y3 `3ca4e50` | test-only | **correct**; the release suite is green in both profiles (§3) and CI now runs both (§3.3) |
| `MatchCache`, the memory semaphore, the bounding-box reject, the bounded centroid sweeps, the near mask, the parallel writers and views | E2 `d43614f`..`2df23e8` | — | **no result moves**: reproduced byte for byte against a freshly built `7cec806` on three collections, caches included, cold (§5) |

### 1.2 What did **not** move

`git diff 4ca6750..HEAD -- crates/ | grep 'const'` yields fourteen new constants and not one of
them is an algorithm threshold: `cache::CAPACITY = 64`; `memory::{BYTES_PER_FACE = 361,
PROCESS_FLOOR, FALLBACK_MEMORY}` and its five per-format face estimates; `grid::{CELLS = 64,
SLACK}`; `refine::CAP_SEED = 0`; `outputs::TIMING_HEADING`; and `GRID` in the new measurement
example. Not one line of the diff changes a number the algorithm reads. `crates/sherd-core/src/params.rs`
is untouched, so no `Params` default moved; `--target-faces`, `--min-tight`, `--max-gap`,
`--max-pen`, `--min-seam`, `--thick-ratio`, `--stage1`, `--candidates`, `--reg-points`,
`--margin-points`, `--surface-points`, `--frac-density`, the two screening flags and the three
second-pass flags all still print the Python's own defaults (`run --help` compared against
`cli.py` line by line).

R is amended in three places only, and `git diff -U0` on it confirms it: the four hunks land at
§11.2 (twice — the `"output"` key the reference never writes, and the insertion order of both
objects), §12.1 (one addendum block) and §13 (the restated table). **Not one line of R §1–§10
changed.** D is amended in §5, §8, §9's neighbourhood, §10.2 (three rows), §10.3 and
§12.

### 1.3 D §10.2's three amended rows are the *code's* numbers, not new ones

V4-D6 said two stated tolerances were not the ones the harness enforces. Y5 amended the document.
Checked: `crates/sherd-parity/src/stages/assembly.rs` carries `NATIVE_JOINS = 12.0` (line 440) and
`NATIVE_GROUP_SIZE = 8.0` (line 445), and `git log` on that file stops at `b9cc4b8` (step D2) —
the file was not touched by Y or by E. So the amendment moved the document to the code, which is
the right direction; **no parity gate was widened in this phase.** The third row (`outputs`
injected, `transforms.json` poses `1e-9 t` instead of "exact") is likewise the `assembly` row's own
existing tolerance, and it is met by seven orders of magnitude (measured 0).

### 1.4 Nothing above 27 fragments was run

* E1 §1 and the E2 header both state it. Verified independently: `grep -hoE 'input/[a-z_0-9/]+'
  output/bench_e2/*.sh` yields only `input/sfspp/pot_`, `input/synthetic_pingsdorf_20/fragments`
  and `input/test_fragments_1/fragments`; `output/bench_e1/` holds only `terra`, `potH` and
  `syn20` trees; no file under `output/bench_{e1,e2,y}` mentions `synthetic_pingsdorf_170`,
  `synthetic_pingsdorf_60` or `mixed_all`.
* The only large-collection outputs on this machine — `output/scale/{synth170,synth170_full,
  mixed_all}` — are dated 6 September, before the phase began on the 7th.
* The decision is recorded in **D §10.3** (the three rows marked "final acceptance only (decision
  2026-09-07)" plus the paragraph naming the development sets), in **D §12** (the phase-1e and
  phase-2d exit criteria), and in **README.md** line 429 ("Во время разработки порта гоняются
  только наборы до 27 фрагментов…"). All three agree.
* This verification started nothing larger either. The largest collection touched is `mixed_ABG`
  (24 fragments), which D §10.3 itself lists as a development set.

### 1.5 Deviations that are not on R §12 or R §12.1

R §12.1's own rule, from task X: "anything the port does differently that is not on it is an
undeclared deviation, however small its measured effect". Three of E2's changes are such
deviations, and both notes state that "R is untouched".

1. **The bounding-box reject in front of every bounded query** (`spatial/kdtree.rs:100`,
   `beyond_the_box`). Conservative and provably so — every point of the cloud is inside the box, so
   the box distance is a lower bound; the comparison is widened by `16 ε`. It has a brute-force
   test over 2 400 queries at four radii each. R §12.1's task-X addendum licenses the *bounded
   search* in front of R §5.2/§6.2/§6.3 and gives the proof; it does not mention a second filter in
   front of that.
2. **`spatial::grid::NearMask` as a near mask in front of R §5.2** (`matching/coarse.rs:215`).
   Same class, same argument, and `the_mask_never_hides_a_neighbour` checks it against brute force
   on 60 000 queries over three cloud shapes at five radii.
3. **The two bounded centroid sweeps** (`fragment/breakline.rs:257`, `fragment/samples.rs:230`).
   This one is *not* the same class: it does not filter a query, it changes the **array**. R §3.5's
   `distance` and R §3.5.6's `d_brk` now hold `∞` wherever the true distance is at or beyond the
   one threshold that reads them. The equality is of the predicates, not of the arrays, and one of
   the two halves has no standing parity row (**V5-D5**).

I verified all three empirically — §5's byte-identity is the check that actually settles them — but
they belong in R §12.1 beside the addendum whose rule they follow.

---

## 2. Parity: `--stage all`, both modes, all eight dumps

`sherd-refit-rs parity --fixtures <dump> --input <collection> --stage all [--injected]`, release
binary. `worst/tol` is the largest ratio of a measurement to its own tolerance inside a stage; the
three largest are quoted.

| set | native checks | failed | worst stage rows | injected checks | failed | worst stage rows |
|---|---:|---:|---|---:|---:|---|
| slab | 79 | **0** | samples 0.46, candidates 0.10, breakline 0.02 | 246 | **0** | nms 0.60, outputs 0.07, hypotheses 0.03 |
| terracotta | 141 | **0** | working mesh 0.94, breakline 0.45, segmentation 0.42 | 626 | **0** | stage1 0.69, nms 0.67, outputs 0.19 |
| `pot_A` | 298 | **0** | load 1.00, candidates 0.71, samples 0.37 | 2 033 | **0** | load 1.00, nms 0.81, verify 0.54 |
| `pot_B` | 345 | **0** | load 1.00, candidates 0.67, samples 0.37 | 2 517 | **0** | load 1.00, nms 0.76, stage2 0.75 |
| `pot_C` | 254 | **0** | load 0.50, samples 0.47, candidates 0.38 | 1 604 | **0** | nms 0.53, load 0.50, outputs 0.41 |
| `pot_G` | 241 | **0** | candidates 0.38, samples 0.33, load 0.25 | 1 589 | **0** | stage2 0.54, outputs 0.41, nms 0.37 |
| `pot_H` | 448 | **0** | candidates 0.58, samples 0.45, load 0.25 | 3 638 | **0** | nms 0.87, outputs 0.62, stage2 0.50 |
| `synthetic_20` | 1 020 | **0** | working mesh 0.93, breakline 0.76, segmentation 0.62 | 8 582 | **0** | nms 0.74, outputs 0.72, stage1 0.56 |
| **total** | **2 826** | **0** | | **20 835** | **0** | |

**23 661 checks over 16 stages × 8 sets × 2 modes, 0 failed** — Y §4's 23 661 to the check,
every set and both modes. `load 1.00` on pot_A and pot_B is D §10.2's own documented boundary
(Assimp's `fast_atof`, one `f32` ULP) and has no headroom by construction; nothing else is above
0.94.

`--verify-checksums` over every dump: **18 461 files, every one matching its manifest** — slab 201,
terracotta 542, pot_A 1 724, pot_B 2 126, pot_C 1 363, pot_G 1 338, pot_H 3 033, synthetic_20
8 734.

Two rows worth naming because V4 asked about them: the `assembly` injected `recentre` check (the
port's R §8.2 against the reference's own `transforms.json`) and the `outputs` injected
`transforms pose` row are both **0 differing** on all eight dumps, which is what V4-D10's fix was
for.

---

## 3. Build, suites, lints

From `cargo clean` (12.1 GiB removed), release build 48.5 s.

| command | result |
|---|---|
| `cargo build --release` | **OK** |
| `cargo fmt --all --check` | **OK** (exit 0) |
| `cargo clippy --workspace --all-targets -- -D warnings` | **OK** (exit 0) |
| `cargo test --workspace --no-fail-fast` (debug) | **OK** — **327 passed, 0 failed, 2 ignored** |
| `cargo test --workspace --release --no-fail-fast` | **OK** — **327 passed, 0 failed, 2 ignored** |
| `cargo test --workspace --release -- --ignored` | **OK** — 2 passed (`the_terracotta_assembles_the_two_joins_of_r_13`, `two_runs_on_the_terracotta_produce_byte_identical_caches`) |
| `python -m pytest -q` | **OK** — **60 passed** in 160.5 s |

The suite has grown 311 → 327 since task Y: E1 added none (it changed no line of `crates/`) and
E2 added exactly sixteen — `memory`'s seven, `matching::cache`'s four, `spatial::grid`'s three,
the KD-tree's box sweep and the bounded breakline sweep.

### 3.1 The CI workflow runs both profiles

`.github/workflows/rust.yml` has three jobs: `check` (fmt + clippy on ubuntu), `test`
(`cargo nextest run --workspace --all-targets` plus `cargo test --doc`, on ubuntu / macos-14 /
macos-13 / windows) and — added by Y3 — **`test-release`**, the same `nextest` run with `--release`
on the two runners D §10.5 gives the `parity` job. So V4-D2's cause (a release-only failure behind
a debug-only CI) is closed. Two small gaps remain and neither is a phase-1e regression:
`test-release` does not run `cargo test --doc`, and `nextest` skips `#[ignore]`d tests in both
jobs, so the two ignored tests of §3 run nowhere in CI.

---

## 4. Quality: `run` natively on all eight development collections

`sherd-refit-rs run <input> --out <dir> --no-meshes` (previews **on**, which is more than
D §10.3's gate asks for), scored with `tools/evaluate.py`. Release binary.

### 4.1 The seven sets of R §13's restated table

| set | port accuracy | R §13 as restated | precision | joins used (correct) | groups | cross-object | purity | verdict |
|---|---|---|---|---|---|---|---|---|
| terracotta | R §13's join row, exactly (§4.2) | the decisions | — | 2 (both R §13's), 0 rejected | 3 + 1 | **0** | 1.000 | **PASS** |
| `pot_A` | **100 % (8/8)** | ≥ 87.5 %, precision 1.000 | 1.000 | 7 (7) | one group of 8 | **0** | 1.000 | **PASS** |
| `pot_B` | 88.9 % (8/9) | 88.9–100 %, precision 1.000 | 1.000 | 7 (7) | 8 + 1 | **0** | 1.000 | **PASS** |
| `pot_C` | 75.0 % (3/4) | 75 %, precision 0.500–0.667 | 0.667 | 3 (2, 1 unscorable) | 3 + 2 + 2×1 | **0** | 1.000 | **PASS** |
| `pot_G` | 0 % (0/7) | 0 %; ≤ 2 joins, each a wrong-pose join on a ground-truth-adjacent pair | 0.000 | 2 (0; both wrong-pose, adjacent) | 2 + 2 + 3×1 | **0** | 1.000 | **PASS** |
| `pot_H` | 36.4 % (4/11) | 36.4 %, 0.429 | 0.429 | 7 (3) | 7 + 2 + 2×1 | **0** | 1.000 | **PASS** |
| `synthetic_20` | 90.0 % (18/20) | 85–95 %, precision 1.000 | 1.000 | 19 (19) | 15 + 3 + 2×1 | **0** | 1.000 | **PASS** |

Every figure is V4 §4.1's and Y §2's, unchanged. Nothing in phase 1e moved a decision on any of
the seven.

### 4.2 R §13's terracotta row, exactly

| R §13 asks | measured, from `report.json` |
|---|---|
| joins used exactly {021–094, 094–104} | **exactly those two, in that order; `joins_rejected` empty** |
| 007 unplaced | **unplaced**, its own group |
| both `pen` = 0 | 0.0 / 0.0 |
| tight of both ≥ 0.27 | 0.5566 / 0.6705 and 0.5347 / 0.5567 |
| 021–094 seam within 20 % of 20.3 t | **20.6667 t** (+1.6 %) |
| 094–104 seam within 20 % of 10.7 t | **12.3333 t** (+15.3 %) |

Scores 11.5025 and 6.5950, `cont_n` 0.995 and 0.997, `gap` 0.0080 and 0.0086 against a limit of
0.03 — the numbers R §13 now prints beside the reference's own.

### 4.3 The two key orders, against the reference's own files

`transforms.json`'s `fragments`: the port writes `FY234021, FY234094, FY234104, FY234007`; the
reference's `output/fixtures/terracotta/_run/transforms.json` writes the same four in the same
order. Checked on all seven collections the fixtures carry: the reference's key order is, on every
one, the concatenation of its groups' member lists in placement order followed by the singletons in
collection order — pot_A `01,03,05,04,07,06,08 | 02`, pot_C `02,03,04,01,07 | 05 | 06`, pot_H
`03,07,08,10,09,11,04,02 | 01 | 05 | 06`, synthetic_20 `000,017,014,013,011,…`. `timings`: the
reference's own runs carry `['preprocess','matching','assembly','refine']` and no `"output"`, and
so does the port's. Both of V4-D4 and V4-D5 are closed against the reference itself, not against a
reading of it.

### 4.4 `mixed_ABG` — the development set nobody ran (V5-D1)

`mixed_ABG` (24 fragments of three pots, 276 pairs) is named a phase-1e development set by D §10.3,
by D §12's exit criterion and by README line 429, and it appears in the E2 note's own header — but
no note in the phase ran it. It has no fixture dump and no row in R §13. Run here, on both sides:

| | **port** | **reference** (seed 0, `workers=5`, `OMP_NUM_THREADS=1`) |
|---|---|---|
| fragment accuracy | 14 / 24 = **58.3 %** | 14 / 24 = **58.3 %** |
| per object | A 75.0 %, B 88.9 %, G 0 % | A 62.5 %, B 100 %, G 0 % |
| precision | 0.667 | 0.750 |
| joins used | 18 — 12 correct, 3 wrong pose, **3 cross-object** | 16 — 12 correct, 3 wrong pose, **1 cross-object** |
| groups | 10 + 8 + 2 + 2 + 2×1 | 14 + 3 + 7×1 |
| **group purity** | **0.864** (0.80 and 0.88) | **0.706** (0.64 and 1.00) |
| matching stage | **64.0 s** | 613.7 s (**9.6×**) |

**The clause is the problem, not the port.** D §10.3's restated quality gate reads "within the
reference's own spread as R §13 states it on every listed set, with **cross-object joins 0 and
group purity 1.000 everywhere**". On the one mixed development set *neither implementation* meets
it: the reference makes a cross-object join too, its largest group is 14 fragments at purity 0.64
against the port's 10 at 0.80, and its overall purity is **lower** than the port's. Fragment
accuracy is the same on both sides, to the fragment. R §13's own "all sets" row is over R §13's seven
single-object collections and its 21 sweep runs were on pot_B, pot_G and synthetic_20 — a mixed
collection was never in its scope, and the "everywhere" in D §10.3 reads wider than the row it
restates.

So this is a **specification defect and a measurement gap**, not a regression: the port is not
worse than the reference here on any figure except the raw cross-object count (3 against 1), and it
is better on the one that measures the damage those joins do. What is missing is that a set the
phase declares a development set was scored by nobody, on either side, until now.

---

## 5. Determinism and byte identity

`compare_out.py` walks both trees and compares every file byte for byte, allowing exactly two
softenings: `report.json` with its `timings` object removed, and `report.md` above `## Timing`.

### 5.1 Determinism (D §7)

| comparison | files | byte-identical | differing |
|---|---:|---:|---|
| synthetic 20, two cold runs, **meshes and previews on** | 48 (20 `placed/*.ply`, 20 `cache/*.sherd`, 2 `assembly_*.ply`, 3 PNG, 3 text) | **46** | `report.json`, `report.md` — the wall clock alone |
| pot_H, default (9 threads) vs `--threads 1` | 30 | **28** | the same two |
| pot_H, default vs **`--workers 1`** | 30 | **28** | the same two |

The third row is not in E1 or E2 and is the stronger test: `--threads` only resizes the pool, while
`--workers` also feeds `pipeline::block_size`. pot_H is the one development set where that matters
— at 55 pairs `block_size(9, 55) = 1` and `block_size(1, 55) = 3`, so the default runs one pair per
block and `--workers 1` runs them as blocks of 3×3 fragments, a different traversal of R §4.1's
pair order. Same 30 files, byte for byte.

### 5.2 Byte identity against phase 1d (E2's own gate, re-derived)

The baseline is a **freshly built** `7cec806` ("E1: the profile, the memory model and the
development-set decision") — `git archive 7cec806` into a scratch tree with its own
`CARGO_TARGET_DIR`, `cargo build --release`. Not the binary E2 left behind. Both binaries report
the same `info` block (algorithm reference `9d4b9d3`, cache version 5).

| comparison | shape | files | identical | differing |
|---|---|---:|---:|---|
| terracotta | cold (`--force`), meshes and previews on | 14 | **12** | the wall clock alone |
| pot_H | cold (`--force`), meshes and previews on | 30 | **28** | the wall clock alone |
| synthetic 20 | cold (`--force`), previews on, `--no-meshes` | 26 | **24** | the wall clock alone |

70 files in all: **35 `cache/*.sherd`, every one rebuilt cold through E2.3's box and E2.5's two
bounded sweeps**, 15 `placed/*.ply`, 3 `assembly_*.ply`, 8 PNGs and 3 `transforms.json` — **64
byte-identical outright**, and the six `report.json` / `report.md` identical once the wall clock is
taken out. That is the check that settles §1.5's third deviation, and it is the only check that
can: the harness cannot see it (V5-D5).

### 5.3 D §9's `--memory-budget` on the terracotta

| run | budget | peak reserved | concurrent | waited | wall | peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| default | 8 094 MiB | 1 491 MiB | 4 | 0 | 4.01 s | 1 504 MiB |
| `--memory-budget 0.5` | 378 MiB | 461 MiB | **1** | **3** | 7.84 s | **1 012 MiB** |

The bounded run serialises the four scans and holds a third less memory, and its fourteen output
files — the four `cache/*.sherd` included — are byte-identical to the unbounded run's **and** to
the `7cec806` binary's. The default budget's own reservation, 1 491 MiB, is E1 §7.1's measurement
of the four-concurrent case (1 492 MiB) to a megabyte, which is the model checking itself. (E2 §4
reported 835 MiB for the bounded run against my 1 012; peak RSS on this set spans that much
run to run, and the direction and the file equality are what the row is for.)

---

## 6. Timing and memory, against D §10.3

One cold run (`--force`, every cache rebuilt) and two warm ones per set, on an otherwise idle
machine, `/usr/bin/time -l`. "warm, previews + meshes" is E2 §9's shape; "warm, gate shape" is
`--no-preview --no-meshes`, which is what D §10.3's rows are stated for.

| set | cold | E2 §9's cold | warm, previews + meshes | E2 §9's warm | warm, gate shape | D §10.3 gate | margin |
|---|---:|---:|---:|---:|---:|---|---:|
| terracotta, 6 pairs | 4.04 s | 4.46 | 2.06 s | 2.27 | **1.36 s** | ≤ 25 s | **18×** |
| `pot_A`, 28 pairs | 6.21 s | 6.25 | 5.10 s | 5.10 | **4.07 s** | ≤ 35 s | **8.6×** |
| `pot_H`, 55 pairs | 8.04 s | 8.19 | 7.49 s | 7.93 | **6.79 s** | ≤ 40 s | **5.9×** |
| `synthetic_20`, 190 pairs | 24.55 s | 27.46 | 17.13 s | 17.79 | **15.21 s** | ≤ 120 s | **7.9×** |

Every column reproduces E2 §9 within 1–11 %, the port being the faster side on this run of the
machine. CPU and peak RSS of the same runs:

| set | CPU cold | CPU warm | peak RSS cold | peak RSS warm | peak RSS gate shape |
|---|---:|---:|---:|---:|---:|
| terracotta | 19.4 core-s | 6.71 | 1 453 MiB | 587 | 456 |
| `pot_A` | 42.6 | 31.1 | 696 | 618 | 498 |
| `pot_H` | 63.8 | 60.2 | 652 | 681 | 319 |
| `synthetic_20` | 193.5 | **130.7** | 1 978 | **2 071** | 1 872 |

synthetic 20's warm CPU is **130.7 core-s** against E2 §9's 130.03, and its peak RSS **2 071 MiB =
2.02 GiB against D §8's 6 GB** — 2.9× of room. That is the largest figure anywhere in this
verification; the memory gate holds with the same margin E2 reported, and the +19 % E2 measured
against phase 1d is confirmed to be inside the run-to-run spread of the number.

The two large CPU rows of D §10.3 (`synthetic_170` and `mixed_all`, ≤ 2 h each) are **not**
discharged and were deliberately not attempted: they are the final acceptance after phase 2
(decision 2026-09-07), and D §10.3, D §12 and README all say so.

---

## 7. Python

* `python -m pytest -q` — **60 passed**, 160.5 s. (V4-D1 had this at 1 failed of 58; Y1 added the
  two tests that make it 60.)
* **The committed slab dump agrees with disk.** `fixtures/slab/dump/manifest.json` lists **201**
  files; `os.walk` over the dump finds exactly 201 (excluding `manifest.json` itself); the two sets
  are equal, and every entry's `sha256` **and** `size` match the file. The same is true through the
  port: `parity --verify-checksums` says "all 201 files match the manifest".
* `tools/dump_outputs.py` now writes `outputs/report.md` and rewrites the manifest, which is what
  took the dump from 180 hashed of 200 to 201 of 201.

---

## 8. Task Y's reference sweep, re-run

The task asked for two of the Y note's seeds on pot_G, re-run against the reference itself.
`pipeline.run(input, out, target_faces=200000, workers=5, params=Params(seed=k), preview=False,
refine=True, write_meshes=False)` with `OMP_NUM_THREADS=1`, scored with `tools/evaluate.py`.

| run | Y §3.1 says | measured here |
|---|---|---|
| pot_G, **seed 1** | 0 % accuracy, **1** join: 02–05, wrong pose, 1.47° / 1.11 t | 0 % (0/7), **1** join: `Pot_G_Piece_02 — Pot_G_Piece_05`, **wrong_pose, 1.5° / 1.11 t**; groups 2+1+1+1+1+1, cross-object 0, purity 1.000 |
| pot_G, **seed 2** | 0 % accuracy, **2** joins: 01–03 (2.42° / 1.12 t) and 02–05 (1.47° / 1.11 t) | 0 % (0/7), **2** joins: `01 — 03` **2.4° / 1.12 t** and `02 — 05` **1.5° / 1.11 t**, both wrong_pose; groups 2+2+1+1+1, cross-object 0, purity 1.000 |

Both rows reproduce. So R §13's pot_G clause — "at most two joins used, and every join used must be
a wrong-pose join on a ground-truth-adjacent pair" — is a statement about the *reference*, backed
by the reference: the prohibition "no join must be accepted" that the row used to carry is a
property of seed 0 and not of the collection, and the port's two joins (04–07 at 0.7 ° / 1.05 t and
02–05 at 2.5 ° / 1.20 t) are the same failure at the same magnitude on the same kind of pair.

The restated gate is therefore backed by measurement on the two rows where a *relaxation* was made
from a measurement (pot_G and, by the same sweep, pot_B and synthetic_20). It is **not** backed by
one where a relaxation was made without one: pot_A's row went from "87.5 %" to "≥ 87.5 %" and
pot_A was never swept (Y §6 says so). That is V5-D8.

---

## 9. Defects

None of the eight moves a result. One (V5-D3) is a code inconsistency, two are coverage gaps, and
the rest are documentation or specification.

| id | severity | where | R/D says | the code / the tree does |
|---|---|---|---|---|
| **V5-D1** | medium | D §10.3's quality clause; `input/sfspp/mixed_ABG` | D §10.3: "cross-object joins 0 and group purity 1.000 **everywhere**"; D §12, D §10.3 and README all name `mixed_ABG` a phase-1e development set | no note in the phase ran it, and neither implementation meets the clause on it: the port makes **3 cross-object joins** at purity **0.864**, the reference **1** at purity **0.706**, with identical fragment accuracy (58.3 %). The clause is R §13's, and R §13's scope is seven single-object collections; D §10.3's "everywhere" reads wider than what was ever measured (§4.4) |
| **V5-D2** | low | `crates/sherd-cli/src/main.rs:84` | `cli.py` resolves an unset `--workers`/`--threads` to `max(1, cpu_count() - 1)` (V4-D8) | `run --threads`'s help still says "(default: one per core)". The line above it was corrected by Y4; this one was not. Behaviour is right (`main.rs:495` uses `default_workers()`), the help is wrong |
| **V5-D3** | low | `crates/sherd-cli/src/main.rs:554` and `:562` | V4-D8; D §10.3 states its gates for a `--no-preview` run and names `bench` for them | `bench` resolves threads as `args.threads.unwrap_or(0)` (one per core) and passes `workers: 0`, so it runs the pool **and** R §4.2's block schedule at 10 where `run` uses 9. No result moves — `block_size(9, n) == block_size(10, n)` on all eight development sets — but the tool named for the gate is not the tool that produced D §10.3's numbers |
| **V5-D4** | low | D §5, first line | the default is cores − 1 since Y4 | "One process, one `rayon` pool sized `--threads` (**default: all cores**)" |
| **V5-D5** | medium | `crates/sherd-core/src/fragment/samples.rs:230` and `crates/sherd-parity/src/stages/samples.rs:227`, `:435` | D §10.2's `samples` row gates R §3.5.6's margin against the reference's | the pipeline computes `d_brk` with `breakline_distance_below(…, 1.5 t)` (∞ beyond the bound) while **both** harness call sites recompute it with the exact `breakline_distance`, so no standing parity row exercises the bounded form. Its only gate is a before/after byte comparison, which is not a standing gate. (The `breakline.rs:257` half *is* covered: the injected `breakline` row compares `ns`/`nf`, which read `distance` only through `>= inner`.) |
| **V5-D6** | low | `crates/sherd-parity/src/stages/outputs.rs:168` | V4-D5: `transforms.json` is written in R §8's placement order | the harness passes `order: &[]`, and the dump's own copy is written with `sort_keys=True`, so the key order has **no automatic gate**. Checked by hand here against `output/fixtures/*/_run/transforms.json` (§4.3) and it is right |
| **V5-D7** | low | R §12.1, task-Y addendum; `crates/sherd-core/src/render.rs:262` | the addendum records `column_mean` for R §8.2 alone | Y4 changed the mean in `render::principal_axes` too. It is inside PMC-10's own licence and gated by the `outputs` `axis` row at 1°, but the addendum does not say it moved. Same section: E2's three accelerations (§1.5) are not recorded anywhere in R §12/§12.1, although both notes state "R is untouched" |
| **V5-D8** | low | R §13, `pot_A` row | the row was relaxed from "87.5 %" to "**≥** 87.5 %" | pot_A was not swept (Y §6 lists it among the three that were not), so unlike pot_B, pot_G and synthetic_20 this relaxation rests on an argument rather than on the reference's own spread. Harmless today — the port scores 100 % — but it is the one row of the restated table with no measurement under it |

Also noted, below the defect line: `README.md`'s Rust flag list (line ~408) names `--backend`,
`--no-cache`, `--force` and `--dump-fixtures` and omits `--memory-budget`, which E2 added and D §9
lists.

---

## 10. Gate table

| gate | result |
|---|---|
| `cargo clean` + release build | **PASS** |
| `cargo fmt --all --check` | **PASS** |
| `cargo clippy --workspace --all-targets -D warnings` | **PASS** |
| `cargo test --workspace` (debug) | **PASS** — 327 / 0 / 2 |
| `cargo test --workspace --release` | **PASS** — 327 / 0 / 2 |
| `cargo test --release -- --ignored` | **PASS** — 2 |
| CI runs both profiles | **PASS** (`check`, `test`, `test-release`) |
| `parity --stage all`, 8 sets × 2 modes | **PASS** — 23 661 checks, **0 failed** |
| `parity --verify-checksums`, 8 dumps | **PASS** — 18 461 files |
| R §13 as restated, all seven sets | **PASS** (§4.1, §4.2) |
| cross-object 0 and purity 1.000 on those seven | **PASS**; on `mixed_ABG` neither side meets it — port 3 / 0.864, reference 1 / 0.706 (V5-D1) |
| determinism, two runs and two thread counts | **PASS** (§5.1), plus a `--workers 1` block-schedule run |
| byte identity against `7cec806` | **PASS** — 70 files on three sets, caches rebuilt cold (§5.2) |
| D §10.3's development-set time rows | **PASS** — 1.4 / 4.1 / 6.8 / 15.2 s against 25 / 35 / 40 / 120 |
| D §8's ≤ 6 GB | **PASS** — worst 2.02 GiB (synthetic 20, warm, previews + meshes) |
| `python -m pytest -q` | **PASS** — 60 |
| slab dump manifest vs disk | **PASS** — 201 = 201, every hash and size |
| D §10.3's three large CPU rows | **not run, by decision** (final acceptance after phase 2) |

**gates_ok: true.**

---

## 11. Housekeeping

Everything this verification wrote lives under `output/bench_v5/`; the trees were pruned to
`transforms.json`, `report.json`, `report.md`, `evaluation.json` and the logs once the numbers
above were written down. Nothing under `input/`, `output/fixtures/` or `fixtures/` was touched, the
branch never left `rust-core`, and the only file this task commits is this note. The scratch build
of `7cec806` lived outside the repository and was deleted.

Two pre-existing trees are worth a line for whoever next needs the disk: `output/bench_after` and
`output/bench_before` hold 196 MB each and date from 5 September, before this phase.
