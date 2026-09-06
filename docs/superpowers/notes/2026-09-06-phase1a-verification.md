# V — independent verification of phase 1a

**Date:** 2026-09-06. Branch `rust-core` at `5f9a89f`. Plan step V of
`docs/superpowers/plans/2026-09-06-rust-core-phase0-1a.md`. Design **D** =
`docs/superpowers/specs/2026-09-06-rust-core-design.md`, algorithm reference **R** =
`docs/superpowers/specs/2026-09-06-algorithm-reference.md`, ground truth = `sherd_refit/*.py`.

**Method.** Nothing in the S1–S4 notes was taken on trust. Every number below was re-measured on
this machine (Apple M2 Pro, 10 cores, 16 GB, shared with other agents; `-j4` / `RAYON_NUM_THREADS`
left at the default for the timed runs, which the note states where it matters), and every claim
about the reference was re-derived from the Python source or from a fresh Python run, not from the
implementers' tables. Toolchain: rustup, `rust-toolchain.toml`'s 1.97.0 (Homebrew's cargo 1.88 on
`PATH` cannot build the tree — it is below the 1.89 MSRV; `~/.cargo/bin` must come first).

**Verdict: gates green.** Build, tests, clippy, `fmt`, `--locked`, parity in both modes on all
eight fixture collections, and determinism all pass. Twelve findings follow, none of which is a
wrong answer in the port; the two that need a decision before phase 1b are **F1** (the harness
applies a native thickness tolerance D §10.2 does not sanction) and **F2** (a latent
`target_faces = 0` trap in the parity harness). The rest are documentation and spec drift.

---

## 1. Gates

| gate | command | result |
|---|---|---|
| from-scratch release build | `cargo build --workspace --release -j4`, empty target dir | **PASS** — 51.6 s wall, 160.2 s CPU, **zero warnings** |
| tests | `cargo test --workspace -j4` | **PASS** — **152 passed, 0 failed**, 1 ignored, 2 m 04 s |
| the ignored test, on the real scans | `cargo test -p sherd-cli -- --ignored` | **PASS** — `two_runs_on_the_terracotta_produce_byte_identical_caches`, 12.5 s |
| clippy | `cargo clippy --workspace -- -D warnings` | **PASS** — clean, 5.5 s |
| clippy, all targets (what CI runs) | `cargo clippy --workspace --all-targets -- -D warnings` | **PASS** — clean, 14.8 s |
| formatting | `cargo fmt --all --check` | **PASS** |
| lockfile | `cargo metadata --locked` | **PASS** (CI builds `--locked`) |
| parity, injected | `parity --stage all --injected`, 8 collections | **PASS** — 1207 comparisons, 0 failed |
| parity, native | `parity --stage all`, 8 collections | **PASS** — 776 comparisons, 0 failed |
| determinism | two cold `segment` runs, plus `--threads 1` and `--threads 3` | **PASS** — 4/4 caches byte-identical in all four pairings |
| Python unchanged, sink off | `pytest -q` | **PASS** — 54 passed, 71.6 s |

`cargo clean` was **not** run against the repository's own `target/`: other agents share this
working tree and the directory held 5.6 GB of their artifacts. Instead the build ran with
`CARGO_TARGET_DIR` pointing at an empty scratch directory, which is a strictly stronger
from-scratch check (no incremental cache and no stale fingerprints at all) and leaves their work
alone. Every command in this note used that scratch target.

Test counts by binary: `sherd-core` 103 unit + 3 (`io_open3d_parity`) + 4 (`ply_writer_bytes`) + 2
(`fragment_cache`); `sherd-parity` 20 unit + 3 (`slab_fixture`) + 5 (`stages_slab`) + 5
(`working_mesh_slab`); `sherd-cli` 5 unit + 2 (`segment_cli`, +1 ignored). Two doc-test targets,
0 doc tests.

## 2. Parity, both modes, every fixture

`sherd-refit-rs parity --fixtures DIR --input SRC --stage all [--injected]`. Fixture sets:
`output/fixtures/{terracotta,pot_A,pot_B,pot_C,pot_G,pot_H,synthetic_20}` and `fixtures/slab/dump`
(the committed one). None had to be regenerated. `worst/tol` is the largest
deviation/tolerance ratio in the stage; a value of 1.00 means a check sits exactly on its
tolerance and passes by the `<=` in `Check::passed`.

| set | frags | mode | load | thickness | working mesh |
|---|---:|---|---|---|---|
| terracotta | 4 | injected | 24 / 0.00 | 16 / 0.00 | 32 / 0.00 |
| | | native | 24 / 0.00 | 8 / 0.30 | 16 / **0.95** |
| pot_A | 8 | injected | 48 / **1.00** | 32 / 0.00 | 71 / 1.4e-3 |
| | | native | 48 / **1.00** | 16 / **0.95** | 32 / 1.8e-4 |
| pot_B | 9 | injected | 54 / **1.00** | 36 / 0.00 | 81 / 1.7e-3 |
| | | native | 54 / **1.00** | 18 / **0.91** | 36 / 4.8e-6 |
| pot_C | 7 | injected | 42 / 0.50 | 28 / 0.00 | 63 / 9.2e-4 |
| | | native | 42 / 0.50 | 14 / 0.19 | 28 / 1.3e-5 |
| pot_G | 7 | injected | 42 / 0.25 | 28 / 0.00 | 63 / 1.2e-3 |
| | | native | 42 / 0.25 | 14 / 0.36 | 28 / 6.6e-6 |
| pot_H | 11 | injected | 66 / 0.25 | 44 / 0.00 | 99 / 1.0e-3 |
| | | native | 66 / 0.25 | 22 / 0.70 | 44 / 8.8e-6 |
| synthetic 20 | 20 | injected | 80 / 0.00 (20 skipped) | 80 / 0.00 | 140 / 0.00 |
| | | native | 80 / 0.00 (20 skipped) | 40 / 0.73 | 80 / **0.94** |
| slab | 2 | injected | 12 / 0.00 | 8 / 0.00 | 18 / 1.1e-4 |
| | | native | 12 / 0.00 | 4 / 0.14 | 8 / 5.7e-7 |
| **total** | **68** | injected | **368** | **272** | **567** = **1207**, 0 failed |
| | | native | **368** | **136** | **272** = **776**, 0 failed |

The S4 note's headline (1207 / 776 / 68 / 8) reproduces exactly.

**Where the injected column is bit-exact and what that buys.** Every injected `load` count,
`thick.t`/`thick_mode` bit comparison, `face budget`, `area0`, `faces`, `vertices`, `res`, `area`,
`watertight` and `n_boundary` comes back at deviation 0.000e0 against a tolerance of 0. That is
strong, independently reproduced evidence that these are faithful to the reference and not merely
inside a tolerance:

* R §3.1 cleaning, largest component, and the tie rule in `Clusters::largest`;
* `face_geometry` (numpy's `cross`, `linalg.norm` reduction order and `V[F].mean(1)`), `median_edge`
  over unique edges, `np.median`'s even-count rule, and `pairwise_sum` (numpy's eight-register /
  128-block tree — I confirmed `np.sum` really is not left-to-right and that the port's tree
  matches);
* `closed_enough`'s "edges used by a number of faces other than two" and the 0.002 fraction;
* the face budget `int(np.clip(600·ΣA0/t², 50000, target_faces))`, including that numpy's `clip`
  applies the floor first so a cap below 50 000 wins;
* the whole R §3.2 histogram-mode chain in `f32` — `np.percentile(·,90)`'s virtual index and
  γ ≥ 0.5 branch, `np.histogram`'s `_get_outer_edges`, `linspace` with the endpoint assignment,
  the two ±1-ULP index corrections, and `argmax`'s first-maximum tie.

**Non-zero injected ratios, explained.**
`load` = **1.00 ULP of an `f32`** on the OBJ sets (pot_A, pot_B; 0.50 and 0.25 on C/G/H) and
**0.00** on every PLY set. I re-derived the cause: Open3D reads OBJ through Assimp, whose
`fast_atof` accumulates decimal digits itself instead of calling a correctly rounded `strtod`, so
the *reference* is one ULP low on some coordinates. Nothing on the port's side can close it. It is
worth knowing that these checks pass **at** their tolerance, not inside it.
`working mesh` injected ≤ 1.7e-3 is the Taubin check alone (`INJECTED_TAUBIN_RES = 1e-9 · res`),
i.e. Open3D's `std::unordered_set` neighbour-summation order against the port's ascending order.

**Native, where the margins actually are.** Two rows are close to their gate:

* **`res` is systematically high**, and it is the tightest gate in the whole phase. terracotta:
  +5.88 / +7.85 / +9.24 / **+9.54** % against ±10 %. synthetic 20: frag_002 +9.35 %, frag_003
  +9.13 %, frag_017 +8.29 %. This is `meshopt` + `Regularize` against Open3D's quadric decimation
  at the same face count (PMC-2), and it is one-sided — 17 of 20 synthetic fragments and 4 of 4
  terracotta come out *coarser*. There is roughly half a per cent of headroom on one fragment.
* **`area`** on the terracotta is −0.335…−0.354 % against ±0.5 % (ratio 0.71), also one-sided.

`faces` is comfortable (worst 2.22 %, gate 5 %) and `watertight` agrees 68 of 68.

## 3. Determinism

`sherd-refit-rs segment input/test_fragments_1/fragments --out DIR`, four runs into four
directories: two cold runs with the default pool, one with `--threads 1`, one with `--threads 3`.
All four produced **byte-identical** `.sherd` caches for all four fragments (`cmp -s`, 4/4 in each
of the three pairings against run 1). A fifth run over run 1's directory reported `hit` for every
fragment and rewrote nothing.

That covers D §7's "results must be identical for `--threads 1` and `--threads N`". The mechanism
holds up on reading: the only parallelism in phase 1a is `rayon::par_iter` over fragments
(`pipeline.rs:54`) and over rays (`thickness.rs:156`), both collected by index; the only unordered
containers in non-test code are `clean.rs:34` (a lookup whose iteration order is never read — the
output order is the input order) and `cache.rs:203` (a one-entry map), and the cache metadata
travels as a single `serde_json` struct precisely so that `safetensors`' `HashMap` ordering cannot
leak into the bytes.

## 4. Timing: `segment` on the four terracotta scans vs the Python

Release build, cold cache, 4 fragments, default pool. The Python numbers are from fresh runs made
for this note, with `SHERD_REFIT_FIXTURES` unset.

| what | wall | CPU |
|---|---:|---:|
| **Rust** `sherd-refit-rs segment` (R §3.1–3.3, writes the cache) | **1.36 s** | 4.33 s |
| Rust, warm cache (4 hits) | 0.00 s | 0.01 s |
| **Python, the same work** — R §3.1–3.3 only, 4 processes | **14.26 s** | 40.62 s |
| Python `sherd-refit segment` — the above **plus** R §3.4 segmentation and the previews | **21.17 s** | 57.03 s |

**10.5× on wall clock and 9.4× on CPU at the same stage boundary.** The plan's "16 s for 4 in
parallel" sits between the two Python numbers; the like-for-like figure is 14.26 s, and the extra
6.9 s of the full `segment` is the segmentation of R §3.4, which the Rust does not do yet — so the
comparison must not be read as a 15× speedup of the whole preprocessing. Per-fragment Python
working-mesh times were 11.71 / 9.75 / 12.80 / 6.35 s (1.23 M / 1.05 M / 1.34 M / 0.71 M input
faces).

Reference values from those Python runs, for the record: `t` = 38.583 / 38.746 / 39.004 / 40.568,
`res` = 2.237 / 2.218 / 2.240 / 2.225, faces = 147304 / 122650 / 155920 / 75526. The Rust gives
`t` = 38.522 / 38.779 / 39.039 / 40.126, `res` = 2.413 / 2.430 / 2.447 / 2.356, faces = 147770 /
122446 / 155638 / 77200.

## 5. The Python package is unchanged with the sink off

* `pytest -q`: **54 passed** in 71.6 s (the plan asked for 30+).
* `git diff 9d4b9d3..main -- sherd_refit/` is empty, so `main` *is* the frozen reference.
* I read every non-`fixture.put` line of `git diff main..rust-core -- sherd_refit/`. All of it is
  the sink: `from . import fixture`, `with fixture.scope(...)`, `dump=`/`tr=` parameters that
  default to off, and results bound to a local before being returned. The only two lines that
  execute differently with the sink off are both no-ops: `pipeline.py`'s
  `if os.path.exists(cache_path) and not fixture.enabled()` (`fixture.enabled()` is `False`, so
  the condition is the original), and `refine.py:51`'s `for n in sorted({...})` — an iteration
  order change over a `set` that only fills a dict by key, where each `fracture_cloud` reseeds
  `default_rng(0)` itself, so no result depends on the order. It is a determinism improvement, not
  a behaviour change, but it is a change to the reference that R does not describe.
* **Strongest check:** my independent Python run above reproduces the committed fixtures' numbers
  exactly — `t` = 38.582656860 / 38.746391296 / 39.003501892 / 40.568187714 in the fixture manifest
  against 38.583 / 38.746 / 39.004 / 40.568 fresh, and faces/`res` identical to the last printed
  digit. The fixtures are still reproducible from today's Python.

**CI** (`.github/workflows/rust.yml`) is coherent: the path filter covers `Cargo.*`,
`rust-toolchain.toml`, `rustfmt.toml`, `crates/**`, `fixtures/**` and itself; the `check` job runs
exactly the `fmt` and `clippy --all-targets --locked -- -D warnings` I ran green here, and the
`test` job runs `build`/`nextest`/`--doc` `--locked` on the four release platforms of D §10.5. It
installs the pinned toolchain rather than a floating one. There is no Python workflow, and there
was none on `main` either, so nothing regressed. **README is stale — see F9.**

## 6. Findings

Ordered by what needs a decision. "R says" quotes the algorithm reference; the Python is the
authority where they differ, and I checked the Python in every case.

### F1 — the native thickness gate is looser than D §10.2, and D was never amended

`crates/sherd-parity/src/stages/thickness.rs:41-43` (`NATIVE_BINS = 3.0`) and `:163-176`
(`push_native`).
**D §10.2 says** the native tolerance on `t` and `thick_mode` is **±2 %**.
**The code applies** `max(2 %, 3 bins of the reference's own histogram)`, which on this data is
5.1 % to 17.0 %.
**Measured:** **17 of the 136 native thickness comparisons are outside ±2 %.** Worst on `t`:
`Pot_A_Piece_04_Mesh` 6.58 %, `frag_019` 5.27 %, `frag_010` 4.54 %, `Pot_B_Piece_01_Mesh` 3.72 %,
`Pot_G_Piece_05_Mesh_DS` 2.27 %. Worst on `thick_mode`: `Pot_H_Piece_08_Mesh_DS` 5.79 %,
`Pot_B_Piece_08_Mesh` 5.76 %.

I re-derived the justification rather than accepting it, by running the reference's own
`estimate_thickness` with seeds 0–11 and nothing else changed:

```
Pot_A_Piece_04_Mesh   seed 0 = 3.554   seeds 1..11 = 3.774 … 3.795   (spread 6.78 % of the seed-0 value)
frag_019              seed 0 = 8.572   seeds 1..11 = 8.085 … 8.582   (spread 5.80 %)
```

So on exactly the two fragments that break ±2 %, **seed 0 is the outlier of the reference's own
estimator**, and the port's value is nearer the estimator's centre than the reference's is. ±2 %
is unreachable for anything that does not reproduce numpy's PCG64 stream, which PMC-9 explicitly
allows the port to abandon. The widening is scientifically right; the process is not — D §10.2
still says ±2 %, both the S3 and S4 notes only *recommend* the change, and a reader running the
harness against the design document would conclude the port passes a gate it does not.

**Action:** amend D §10.2's native thickness row to `max(2 %, 3 bins)` (or reproduce PCG64 and
keep ±2 %). **And carry the consequence forward:** `t` is the unit every threshold of R §1.2 is
expressed in, so a 6.6 % difference in `t` moves `coarse`, `stage1`, `tight`, `facing`, `gap`,
`seam`, `near`, `pen` and `nms` by 6.6 % for that fragment's pairs. No parity tolerance can absorb
that — it lands on the R §13 pair gates in phase 1c and should be on the phase-1b risk list.

### F2 — `target_faces = 0` in a manifest silently decimates every fragment to nothing

`crates/sherd-parity/src/manifest.rs:62-64` documents the field as "The face budget the run was
given; **0 means the adaptive budget of R §3.3**", and it is `#[serde(default)]`, so a manifest
missing the key also yields 0. `crates/sherd-parity/src/stages/mod.rs:160` reads it as
`usize::try_from(manifest.collection.target_faces).unwrap_or(200_000)` — a `u32 → usize`
conversion that cannot fail on any 64-bit target, so **the `unwrap_or(200_000)` fallback is dead
code**. Zero then reaches `face_budget` (`crates/sherd-core/src/mesh/decimate.rs:58-62`), where
`raw.max(50_000).min(0) == 0`.

Reproduced: I copied `fixtures/slab/dump`, set `collection.target_faces = 0`, and ran native
parity —

```
pieceA  faces  0.000  vs  45100.000   dev 1.000e0  tol 5.000e-2 rel  FAIL
pieceA  res    0.000  vs      2.124   dev 1.000e0  tol 1.000e-1 rel  FAIL
pieceA  area  -0.000  vs  86535.758   dev 1.000e0  tol 5.000e-3 rel  FAIL
… 6 of 24 comparisons outside their tolerance
```

Mitigating: it fails loudly, and every committed manifest carries 200000. Still, the sentinel the
doc promises does not exist. **Fix:** `if t == 0 { 200_000 } else { t }` at `stages/mod.rs:160`,
or delete the sentinel from the doc comment.

### F3 — the injected column is weaker than documented on `slim` dumps

`crates/sherd-parity/src/stages/mod.rs:92-93`: `FragmentFixture::original()` falls back to
recomputing `(V0, F0)` from the source file, justified in the comment as "legitimate for both
modes, because the `load` stage compares the two and the comparison is exact". At the `slim`/`min`
levels the load stage **skips** that comparison — `crates/sherd-parity/src/stages/load.rs:86-88`,
`if !fragment.has("load.V0.npy") { report.skip(…); continue; }`. `synthetic_20` is `slim`, so all
20 of its fragments took the fallback and its injected thickness and working-mesh rows ran on the
*port's own* `V0`/`F0` with nothing pinning them to the reference's. Harmless in fact — the
injected `t (bits)` check is bit-equal on all 20, which could not happen if `V0` differed — but the
justification does not hold and the table above overstates what `synthetic_20 / injected` proves.
**Fix:** the comment, and either fall back only when the load stage did compare, or say so in the
report.

### F4 — Taubin weights carry an `1e-12` R does not mention

`crates/sherd-core/src/mesh/taubin.rs:39,88`: `w = 1.0 / (dist + 1e-12)`.
**R §3.3.1 says** `w_j = 1 / |v − v_j|`, and the epsilon is not in R's PMC list (PMC-3 covers
switching to *uniform* weights, nothing else).
**The code is right and R is wrong:** Open3D's `FilterSmoothLaplacianHelper` computes
`weight = 1. / (dist + 1e-12)`, and the injected Taubin check confirms the port lands on Open3D's
mesh (worst 1.7e-3 of a `1e-9 · res` gate). **Fix R §3.3.1**, not the code.

### F5 — the working mesh is narrowed to `f32`; R has no PMC entry for it

`crates/sherd-core/src/types.rs:88-99` and `crates/sherd-core/src/fragment/mod.rs:168-173`. `V` is
stored as `f32`, `res` as `f32`, and `FN`/`A`/`C` are derived from the *narrowed* vertices (which
is the right call — it is what makes a cold and a warm run bit-identical).
**R §0 says** the reference is float64 for everything except Open3D's `RaycastingScene`.
**Authorised by D §4.1 and D §7**, so this is a spec-level decision rather than an oversight — but
it is a deviation from R and R §12 should carry it, because the ≈6e-8 relative error enters every
threshold of R §1.2 and every ICP residual downstream. Note also that the injected working-mesh
checks compute `res`/`area` with the `f64` helpers on the dump's own arrays and therefore do **not**
exercise the `f32` values the pipeline stores; only the native column does, at ±10 % / ±0.5 %.

### F6 — the OBB fallback is a different OBB

`crates/sherd-core/src/fragment/thickness.rs:330-374` (`obb_min_extent`, the function itself at `:342`).
**R §3.2 says** `t = min(extent of the PCA oriented bounding box) / 10`, i.e. Open3D's
`get_oriented_bounding_box()`, whose PCA runs over the **convex hull's** vertices.
**The code** runs the PCA over **all** vertices. Not marked PMC. The code documents it and puts the
difference at ~1 %. Latent, not active: the worst of the 68 fragments has 7 154 valid hits of
20 000 against a threshold of 100, so the fallback is unreachable on the benchmark.

### F7 — colours are quantised to `u8` at read time

`crates/sherd-core/src/io/mod.rs:12-18`, `crates/sherd-core/src/io/obj.rs:53-57`.
**R §3.1** keeps Open3D's `[0,1]` doubles and says colours are carried to the outputs only; R §11.4
writes `uchar`, so the written bytes should be identical. Not marked PMC. Latent: nothing in phase
1a compares an output mesh's colours end to end — `ply_writer_bytes.rs` pins the writer, not a
read → decimate → write round trip.

### F8 — Taubin does not smooth normals and colours

`crates/sherd-core/src/mesh/taubin.rs:43-45`. Open3D's `filter_smooth_taubin` smooths them;
the port does not. **R §3.3.1 itself says** they are not used downstream, so this is consistent
with R — recorded because the port's cached working mesh therefore carries *unsmoothed* colours
where the reference's carries smoothed ones, which would matter if a later phase ever wrote the
working mesh with colour (R §11.4 writes the cleaned original, so it should not).

### F9 — README is stale (last touched at S1)

`README.md:321-323`: "Сейчас готов каркас: … подкоманды `info` и `parity`. Стадии конвейера
добавляются шагами S2–S4 плана; `run` и `segment` пока сообщают, в какой фазе они появятся."
S2, S3 and S4 are committed, and `segment` is fully implemented
(`crates/sherd-cli/src/main.rs:171-244`) — it preprocesses a whole collection through the cache and
prints a table. Only `run` and `bench` bail (`main.rs:135,138`). `git log main..rust-core --
README.md` shows one commit, `5b60bbb` (S1). The rest of the section (crate list, toolchain note,
build commands) is accurate. CI needs no change.

### F10 — `sherd_core::fixture` claims to be filled in by S4 and is empty

`crates/sherd-core/src/fixture.rs:5`: "Filled in by plan step S4." The file is a five-line doc stub
with no code, and D §10.1's "The Rust CLI writes the same layout with `--dump-fixtures`" is not
implemented (`--dump-fixtures` is not a CLI flag). This is **not** a missed gate — the S4 row of
the plan does not name the writer — but the doc comment is wrong and should say phase 1d, or
whenever the writer is scheduled.

### F11 — small drifts between the code and D §4

`crates/sherd-core/src/fragment/mod.rs:49-52` stores `thick`/`thick_mode` as `f64`; **D §4.1**
declares them `f32`. The code is the better choice (`t` is the unit of every threshold), so D
should follow it. `Fragment` also gained `face_budget` and `area0`, which D does not list.
`crates/sherd-core/src/fragment/cache.rs` deliberately omits D §4.2's `created` metadata key so the
cache stays byte-reproducible, and folds the flat metadata map into one JSON object under the key
`sherd` because `safetensors` 0.8 serialises a `HashMap` in iteration order — both justified, both
documented in the S4 note, neither yet in D.

### F12 — `Check::relative` changes unit when the reference is 0

`crates/sherd-parity/src/report.rs:100-104`: when `reference == 0.0` the deviation becomes the
*absolute* difference but is still judged against a *relative* tolerance. Unreachable on real data
(no reference `res`, `area`, `t` or face count is zero) and it fails safe for large values.
Cosmetic; noted so it is not rediscovered.

### Checked and clean (no finding)

Things I expected to be wrong and were not, so that a later reader does not re-litigate them:

* **`percentile90`.** numpy keeps γ in `float64` and promotes the lerp to `float64`, while the port
  does the whole lerp in `f32`. I stress-tested the port's formula against `np.percentile` on
  **20 000 random `float32` samples of 101–20 000 elements: 0 mismatches**, on top of the 68
  bit-exact fixture fragments.
* **`hist_mode`.** `np.histogram`'s `bin_type` really does resolve to `float32` here (a Python
  `int` range endpoint is weak under NEP 50), so the port's `f32` edges and the two ±1-ULP index
  corrections are correct; verified indirectly by the bit-exact injected `t`/`thick_mode`.
* **Cleaning order.** Load is `duplicated → degenerate → unreferenced`; post-decimation is
  `degenerate → duplicated → unreferenced`. Both orders match `fragment.py:31-33` and
  `fragment.py:301`, and they are genuinely different — the code carries the right one in each
  place.
* **`n_orig_*` are taken before the largest-component pass** (`fragment.py:274` vs `:275`), and the
  port does the same (`fragment/mod.rs:103-105`).
* **Thickness is measured on the original component, before decimation**, because it sets the
  budget. Correct in the port.
* **`face_adjacency` tie order.** `np.lexsort` is stable and the reference's rows are slot-major
  (`F[:,[0,1]]` for all faces, then `F[:,[1,2]]`, then `F[:,[2,0]]`) with `fo = tile(arange(m), 3)`;
  the port sorts by `(key, position)` in exactly that order and takes `i % n`. Matches, including
  the three-faces-on-one-edge chain.
* **`largest_component` ties** go to the component owning the earliest triangle, which is what
  `np.argmax` over Open3D's first-appearance cluster numbering does.
* **Collection order.** `sorted(set(glob))` over full paths; the port sorts `OsStr` bytes, which is
  the same order for UTF-8. The documented mixed-case-extension difference (`.Ply`) cannot change a
  result, only accept one more file.
* **Cache validity** matches R §3.7 (absolute path, file exists, |Δmtime| < 1 s, same
  `target_faces`, same name) with D §4.3's two version fields added.
* **All 46 `Params` defaults** match R §1.1 field by field.
* **No `todo!`, `unimplemented!`, `dbg!`**, and the only `unsafe` is `bytemuck`'s derive in
  `vec3.rs` under a documented `#![allow(unsafe_code)]`.

## 7. What I would want before phase 1b

1. Decide **F1**: amend D §10.2 or reproduce PCG64. Whichever way it goes, put the `t`-propagation
   risk (6.6 % on one fragment's thresholds) on the phase-1c risk list, because the R §13 pair
   gates are exact-set gates and cannot be widened.
2. Fix **F2** (two lines) and **F3** (a comment plus a skip that is currently invisible).
3. `res` at 0.95 of its gate is the single tightest number in phase 1a and it is one-sided. The
   segmentation of R §3.4 votes on ray hits at `0.5 < d/t < 1.8` and switches its vote count at
   `res > 0.1·t`, so a 9.5 % coarser mesh is not neutral for it. The 0.97 native segmentation
   agreement gate should be measured early in 1b rather than at the end.
