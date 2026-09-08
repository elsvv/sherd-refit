# V6 — independent verification of phase 2 (steps Z, G1, G2, G3, G4)

**Date:** 2026-09-08. **Tree:** branch `rust-core` at `879e49c` ("G4.6: the note — what was tuned,
what was measured and not changed, and what cannot be measured here"); `git status` clean at the
start, nothing left over from an interrupted attempt, and the branch was never switched — the
`9bf35d6` baseline was built in a detached worktree outside the repository. **Range reviewed:**
`9bf35d6..879e49c`, 33 commits, 66 files, +13 795 / −563. **Machine:** Apple M2 Pro, 10 cores
(6P + 4E), 16-core GPU, 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`; one Metal adapter and no
software fallback. **Python:** the tree's own `.venv`, Open3D 0.19.0, `OMP_NUM_THREADS=1`.

Everything below was re-derived. The suites were re-run from a `cargo clean`; all sixteen parity
stages were re-run on all eight dumps in both modes; `gpu-check` was re-run on all eight sets, with
and without `--chaos`; all seven development collections were re-run on **both** backends and
re-scored with `tools/evaluate.py`; the CPU byte-identity claim was reproduced against a freshly
built `9bf35d6` binary from a scratch worktree; and the D §10.3 rows were re-timed cold and warm on
both backends with `/usr/bin/time -l` beside them.

**Verdict: the gates do not all pass.** Two of them fail.

1. **The CPU-only shape of the binary does not compile** (§3.2, defect **V6-D1**).
   `cargo build -p sherd-cli --no-default-features` and the matching `clippy -D warnings` both fail
   with `E0061`, and they are two steps of the CI workflow that this repository runs on four
   platforms. Broken since `d7bbdfe` (G3.1), unnoticed through G3 and G4.
2. **`gpu-check --stage all` is outside D §10.2 and exits non-zero on seven of the eight sets**
   (§5, defect **V6-D2**). This is the *recorded* state — D §6.7 and D §12's 2b row both say stage 2
   is inside on four of eight — but D §12's own exit criterion for phase 2b is "CPU/GPU cross-check
   within §10.2", and it is not met.

Everything else passes and reproduces the notes to the digit: **23 804 parity checks, 0 failed**;
the CPU path's outputs **byte-identical to `9bf35d6`** on all four sets; the coarse kernel
**bit-identical on 2 934 052 of 2 934 122 hypotheses**, the other 70 exactly one probe point away —
the same two integers the G2 note prints; **identical joins, groups, accepted sets and
`evaluate.py` scores on both backends on all seven collections**; two runs byte-identical on both
backends; every D §10.3 row inside its gate on both backends, warm and cold. Nine defects are
listed in §9; seven of them move no result.

Nothing above 27 fragments was run here, and nothing in phase 2 ran one either (§1.5).

---

## 1. Diff review: `9bf35d6..879e49c`

### 1.1 What moved

| area | commits | what |
|---|---|---|
| R §12.1, R §13 | `c7a2522`, `6f088b0` | documentation only: four undeclared phase-1e deviations recorded; three R §13 rows widened into bands **from fifteen new runs of the reference** |
| D | `d8e5166`, `7b42422`, `c082b53`, `548c16b`, `38ea84c`, `54932d6`, `bcc99c8` | §2, §3, §5, §6.1–§6.8, §7, §9, §10.2, §10.3, §10.4, §12 |
| `sherd-core` | `1cff49b`, `c5fd9bb`, `d7bbdfe`, … | the `Executor` boundary and `batch.rs`; `spatial::grid::HashGrid`; `progress.rs`; `pipeline::run_with` |
| `sherd-gpu` (new) | `c8f695a`, `1517f7f`, `146e70f`, `d798f66`, `5e0c535` | device, buffers, slots, self-test, two WGSL kernels, `Submitter`, `Allocations` |
| `sherd-cli` | `7f73eb2`, `7598344`, `9300116`, `03027dd` | `--backend`, `--gpu-adapter`, `--gpu-memory`, `gpu-check`, `info`'s self-test, Ctrl-C |
| `sherd-parity` | `bcc99c8`, `54932d6`, `93b9e72` | two **new** rows: `margin bound` and `transforms.json`'s key order |
| CI | `d8e5166` | a `test-release` job and a `gpu-check` job; the `check` job gains the `--no-default-features` clippy step that is now red |

### 1.2 No threshold moved, and no tolerance was widened

* **R.** The only change to the algorithm reference is prose: §12.1's addendum (the second
  `column_mean` call site and E2's three filters) and §13's bands. Every `Params` default, every
  scale of R §1.2, R §5's `0.7`, `0.15 t`, `0.06 t`, R §6.5's five limits and R §7's `1e-6`
  convergence pair are untouched — `git diff 9bf35d6..HEAD -- …algorithm-reference.md` contains no
  numeric change outside §13 and the new §12.1 paragraph.
* **R §13's three widened rows are the reference's own.** `pot_A` `≥ 87.5 %` → `87.5–100 %`,
  `pot_C` `75 %` → `50–75 %`, `pot_H` `36.4 %, 0.429` → `27.3–36.4 %, 0.333–0.500`. I checked the
  thing that matters: **the port's measured figures lie inside the *old* rows as well** (100 %,
  75 % / 0.667, 36.4 % / 0.429 — §6), so no band was widened to admit a number of the port's. The
  widening is a bottom edge, added from five reference seeds per set.
* **D §10.2.** Every tolerance in the table is unchanged. The two edits are *additions*: the
  `samples` row gains `margin bound` (bit-for-bit below `1.5 t`, `∞` at or above) and the `outputs`
  row gains `transforms.json`'s key order against the reference's own `_run/transforms.json`. Both
  are strictly stronger and both are met (§4).
* **GPU-side constants** (`coarse::MIN_QUERIES` 200 000, `coarse::MAX_QUERIES` 12 M,
  `icp::MIN_CANDIDATES` 16, `icp::MIN_WORK` 100 000, `icp::MAX_CANDIDATES` 512, `pipeline::DEPTH` 4)
  are scheduling policy: which executor answers a batch, never what the answer is. Each one is a
  fall-through to `CpuExecutor`, so a change to any of them can only change the *arithmetic path*,
  and the two backends' decisions are identical on every development set (§6).

### 1.3 The CPU executor still computes what it computed

The boundary was drawn without restating any arithmetic. Verified by reading, then by bytes (§7.1):

* `executor/cpu.rs::agreeing` is `matching/coarse.rs::agreeing_masked` moved verbatim — the same
  nine `f64` products in the same order, the same `nearest_below`, the same `dot > NORMAL_AGREE`;
  the empty-probe guard moved with it (`vec![0.0; poses.len()]`).
* `matching/icp.rs` is **unchanged except for three type aliases** (`Rotation`, `Translation`,
  `Pose`, `crates/sherd-core/src/matching/icp.rs:778-786`). `register`, `correspondences`,
  `umeyama`, `normal_equations`, `eigen_pivots`, `solve_ldlt`, `euler_zyx` are byte-for-byte the
  phase-1 code.
* `ladder::climb_all` and `pair::stage2_batch` transpose the loop nesting from *per candidate, all
  rungs* to *per rung, all candidates*. Each candidate's ladder reads only its own pose, and
  `CpuExecutor::icp_rung` calls `icp::register` once per candidate, so the transposition cannot
  move a result — and does not (§7.1).
* `verify::one_penetration` used to narrow the points once into a `Vec<[f32;3]>` and scan; it now
  calls `Executor::inside` and `Executor::bounded_distance(DistReduce::Min)`. The `Min` branch is
  the same shrinking-window scan in the same order from the same `f32::MAX`, and
  `executor/cpu.rs::apply` is arithmetically identical to `types::apply_transform`
  (`crates/sherd-core/src/types.rs:175`) which `verify.rs` still uses.
* `pipeline::with_device_slack` builds a deeper rayon pool **only when `Executor::device_slack() > 0`**,
  which `CpuExecutor` never does. R §4.2's block size still comes from `--workers`, and every
  parallel section collects by index.

### 1.4 The WGSL against its CPU mirror

Read line by line. `kernels/grid.wgsl` is prepended to both parity-critical kernels at build time
(`concat!(include_str!…)`), so the traversal exists once.

| question | `grid.wgsl` / `coarse.wgsl` / `icp.wgsl` | `sherd-core` mirror | verdict |
|---|---|---|---|
| grid visiting order | `dx`, `dy`, `dz` each `-1..=1`, `x` outermost; points of a cell in ascending `sorted_idx` | `crates/sherd-core/src/spatial/grid.rs:617-640`, same three loops, same order | identical |
| cell hash | `i32` wrapping `*`, then `^`, then `& (cap-1)` | `i64` `wrapping_mul`, `^`, mask, `& (cap-1)` | identical in the low 32 bits, which is all the mask reads |
| empty-slot marker | `slots[s].w < 0` | `start < 0` after the prefix sum | identical |
| radius rule | strict `d2 < r2` only (`kernels/grid.wgsl:115`); the inclusive variant deliberately absent | `nearest_below` (strict) and `nearest_within` (inclusive) | matches R §5.2/R §7's strict bound |
| tie rule | `d2 == best_d2 && j < best` | same expression | identical |
| squared distance | written out `ex*ex+ey*ey+ez*ez` | written out | identical |
| pose application | nine multiplies, three adds, written out | written out | identical |
| coarse reduction | integer `u32`, fixed 64 → 32 → … → 1 tree | `+= 1` in a scalar loop | exact either way |
| coarse score | kernel returns the **count**; `coarse::score_of` divides in `f64` on the host | `f64::from(agree) / points` | bit-identical wherever the counts agree — confirmed on 2 934 052 of 2 934 122 (§5.1) |
| ICP reduction | 29 accumulators, eight at a time, fixed 256 → 1 tree | `f64` sum in correspondence order | different by construction; D §10.2 governs |
| `JTJ`/`JTr` packing | `acc[0..21]` in `(k, l≤k)` order, `acc[21..27] = row[k]·r` | `jtj[k][l]`, `jtr[k]`, same enumeration | identical layout |
| LDLT pivots | `eigen_pivots` transcribed (selection sort by transposition, strict `>`) | `icp::eigen_pivots` | identical **before** the equilibration, which makes every diagonal 1 and the permutation the identity — a deliberate, documented departure (D §6.5 correction 3) |
| convergence test | `\|Δfitness\| < 1e-6 && \|Δrmse\| < 1e-6`, `done` latched, never iterated again | `icp::register`'s loop | identical rule; the kernel pays the remaining barriers because WGSL needs uniform control flow |
| iteration count | `it` runs `0..=max_iter`; `it == 0` only measures the initial pose | `correspondences` before the loop, then `max_iteration` updates | identical |
| the shifted frame | both clouds translated by their own centroids in `f64` on the host; `point_to_plane` adds `(R − I − ω̂)·c_t`; `point_to_point` adds nothing | — | matches D §7's last row and D §6.5 correction 2; `centroid()` sums in index order as `icp::centroid_of` does |

**No `dot`, `length`, `distance`, `normalize`, `inverseSqrt`, `subgroupAdd` or floating-point
atomic occurs in any of the five WGSL files.** Checked by grep over `crates/sherd-gpu/src/kernels/`;
`cross` and `det` are written out by hand (`cross3`, `det3`).

Two deliberate GPU-side departures from the CPU, both documented where they are:

* the **equilibration** of the 6×6 (`kernels/icp.wgsl:242-268`) — an exact reparameterisation that
  changes which permutation Eigen's rule picks;
* the moved point is **recomputed from the accumulated pose** rather than carried as a transformed
  cloud. D §6.5's own sketch does the same (`let p = rot(T, src_p[i]) + T.t`), so this is D's shape,
  not a drift from it.

### 1.5 Nothing above 27 fragments was run

`grep -rE 'synthetic_pingsdorf_(60|170)|sfspp/mixed_all'` over `output/bench_g1`, `bench_g2`,
`bench_g3`, `bench_g4` and `bench_z` returns **nothing**: no command, no log line, no output tree.
The three large rows appear only as projections, and the G4 note labels them as such
(`notes/2026-09-08-g4-tuning.md:330-347`: *"nothing above 27 fragments is run before the final
acceptance, so these are the measured per-pair cost multiplied by the pair count and nothing
more"*). This verification started nothing above 20 fragments either.

---

## 2. The 170-fragment projection, re-derived from my own measurements

D §10.3's projections are arithmetic over a measured per-pair cost. Both halves check out.

| row | D §10.3 | from my §8 timings | gate |
|---|---|---|---|
| matching per pair, `synthetic_20`, CPU | 83.2 ms | 16.21 s / 190 = **85.3 ms** | — |
| matching per pair, `synthetic_20`, GPU | 63.7 ms | 11.79 s / 190 = **62.1 ms** | — |
| matching per pair, `pot_H`, CPU | 127.5 ms | 7.15 s / 55 = **130.0 ms** | — |
| matching per pair, `pot_H`, GPU | 115.5 ms | 6.63 s / 55 = **120.5 ms** | — |
| synthetic 170 CPU | 12 800 × 0.523 / 6.43 = 17 min | 12 800 × 0.0853 = **18.2 min** | ≤ 2 h |
| synthetic 170 GPU | 12 800 × 0.0637 = 13.6 min | 12 800 × 0.0621 = **13.2 min** | ≤ 30 min |
| `mixed_all` CPU | 12 589 × 0.870 / 6.43 = 28 min | 12 589 × 0.1300 = **27.3 min** | ≤ 2 h |
| `mixed_all` GPU | 12 589 × 0.1155 = 24 min | 12 589 × 0.1205 = **25.3 min** | ≤ 30 min |
| synthetic 60 GPU | 1 770 × 0.0637 = 1.9 min | 1 770 × 0.0621 = **1.8 min** | ≤ 5 min |

So **yes**, D's projection is computed from numbers this machine produces, and it reproduces within
7 %. `mixed_all` on the GPU keeps under five minutes of margin against its 30-minute gate on my
figures as it does on D's — that row remains the one a projection could be wrong about, and D says
so. None of this is a run.

---

## 3. Build, suites, lints

`cargo clean` (18.5 GiB removed) and then, in order:

| command | result |
|---|---|
| `cargo build --release --workspace` | **ok**, 1 m 03 s |
| `cargo fmt --all --check` | **ok** |
| `cargo nextest run --workspace --all-targets --locked --release` | **379 run, 379 passed, 2 skipped**, 14.4 s |
| `cargo build --workspace --all-targets --locked` (debug) | **ok** |
| `cargo nextest run --workspace --all-targets --locked` (debug) | **379 run, 379 passed, 2 skipped**, 172.7 s |
| `cargo test --workspace --doc --locked` | **ok** (no doc tests) |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **ok** |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **FAIL, exit 101** |
| `cargo build -p sherd-cli --no-default-features --locked` | **FAIL, exit 101** |

### 3.1 The two skipped tests are the two `#[ignore]`d ones, and they pass

`run_cli.rs:144` and `segment_cli.rs:98` are ignored because they need
`input/test_fragments_1/fragments`, which is not in the repository. It is on this machine, so I ran
them: `cargo nextest … --run-ignored all` →
`the_terracotta_assembles_the_two_joins_of_r_13` **PASS**,
`two_runs_on_the_terracotta_produce_byte_identical_caches` **PASS**.

The fourteen adapter-dependent tests in `crates/sherd-gpu/tests/adapter.rs` all ran and passed on
this machine's Metal adapter in both profiles (`the_self_test_holds_on_this_machines_adapter`,
`the_icp_kernel_matches_the_cpu_executor_on_synthetic_batches`,
`a_cancelled_gpu_run_leaves_the_device_usable`, the two crossover tables, the chunking and TDR
tests, the memory-budget test).

### 3.2 The build without a GPU is broken — V6-D1

```
error[E0061]: this function takes 2 arguments but 3 arguments were supplied
   --> crates/sherd-cli/src/main.rs:682:20
682 |     let resolved = gpu::resolve(args.backend, args.gpu_adapter.as_deref(), args.gpu_memory)?;
note: function defined here
   --> crates/sherd-cli/src/gpu.rs:301:15
301 | pub(crate) fn resolve(backend: Backend, _adapter: Option<&str>) -> Result<Resolved> {
```

and the same at `main.rs:753`. The `#[cfg(feature = "gpu")]` `resolve` (`gpu.rs:196-200`) grew a
third parameter, `memory: Option<f64>`, in `d7bbdfe` (G3.1, `--gpu-memory`); the
`#[cfg(not(feature = "gpu"))]` stub did not. `git log -S` puts the break at that commit, so it has
been red through all of G3 and G4. The stub's own two-argument signature is still exercised by a
`#[cfg(not(feature = "gpu"))]` block inside `main.rs:1153-1158`, so the intent is unambiguous:
only the two production call sites were missed.

What this breaks:

* `.github/workflows/rust.yml:52` — the `check` job's third step, on ubuntu;
* `.github/workflows/rust.yml:81` — the `test` job's last step, on **all four** platforms;
* D §2/§3's claim that `sherd-gpu` is an optional feature and `--no-default-features` yields a
  wgpu-free binary, and the same sentence in `README.md`.

This is the "the workspace must still build on the four CI platforms without a GPU" rule, and it
does not.

### 3.3 The no-adapter paths that *are* reachable here

There is no hook to force the self-test to fail, so the failure branches were exercised the two ways
this machine allows, plus the unit tests that cover the rest:

| probe | result |
|---|---|
| `run --backend cpu` | runs, `report.json` `engine.backend = "cpu"`, no device opened |
| `run --backend gpu --gpu-adapter NoSuchAdapter` | **exit 1**, `--backend gpu: no adapter matches 'NoSuchAdapter'; this machine offers: [0] Metal Apple M2 Pro (IntegratedGpu)` |
| `run --backend auto --gpu-adapter NoSuchAdapter` | completes on the CPU, silently, `engine.backend = "cpu"` |
| `info --gpu-adapter NoSuchAdapter` | `gpu self-test: not run: no adapter matches …` and the adapter list |
| `selftest::Selection::decide` unit tests | cover a failed self-test, a software adapter, a ratio below and exactly at `AUTO_SPEEDUP`, and no adapter at all |

---

## 4. Parity: `--stage all`, both modes, all eight dumps

`sherd-refit-rs parity --fixtures … --input … --stage all [--injected]`, sixteen runs, every one
exit 0.

| stage | injected checks | failed | worst/tol | native checks | failed | worst/tol |
|---|---:|---:|---:|---:|---:|---:|
| load | 368 | 0 | 1.00 | 368 | 0 | 1.00 |
| thickness | 192 | 0 | 0.00 | 136 | 0 | 0.00 |
| working mesh | 567 | 0 | 0.00 | 272 | 0 | 0.94 |
| segmentation | 884 | 0 | 0.00 | 136 | 0 | 0.62 |
| breakline | 748 | 0 | 0.07 | 204 | 0 | 0.76 |
| samples | 1 156 | 0 | 0.35 | 476 | 0 | 0.57 |
| hypotheses | 2 148 | 0 | 0.03 | 1 074 | 0 | 0.25 |
| coarse | 1 790 | 0 | 0.00 | — | — | — |
| nms | 2 148 | 0 | 0.87 | — | — | — |
| stage1 | 3 654 | 0 | 0.69 | — | — | — |
| stage2 | 3 400 | 0 | 0.75 | — | — | — |
| verify | 2 508 | 0 | 0.54 | — | — | — |
| candidates | 1 123 | 0 | 0.00 | 46 | 0 | 0.71 |
| assembly | 48 | 0 | 0.00 | 80 | 0 | 0.50 |
| refine | 78 | 0 | 0.03 | 78 | 0 | 0.20 |
| outputs | 98 | 0 | 0.72 | 24 | 0 | 0.00 |
| **total** | **20 910** | **0** | | **2 894** | **0** | |

**23 804 checks, 0 failed, 259 skipped** — the number D §12's G1, G3 and G4 rows all quote, and
143 checks above the 23 661 of phase 1. The two new rows of task Z are both non-vacuous and both
pass: `margin bound` (0 differing entries) and `transforms.json`'s key order (0 differing places on
the seven dumps that carry a `_run/`; the slab skips with its reason printed). `worst/tol` is the
largest ratio any single check reached: the tightest are `load`'s vertex ULP row at 1.00 and
`native working mesh` at 0.94, both of which are phase-1 rows and unchanged.

The 259 skips are the documented ones: `verify` natively (D §10.2 has no native column there),
`synthetic_20`'s level-`min` pairs, `pot_G`'s absent `refine/`, the OBJ sets' meshes (D §10.2's
`load` row: Assimp's `fast_atof`), and the slab's absent `_run/`.

---

## 5. `gpu-check --stage all` on all eight sets

`gpu-check` forces the kernels on (it is a check of the *kernel*, not of the executor's size
policy), so every row below reached the device — `[device]` on all of them, and the counters confirm
it (terracotta: `coarse 8/8 on device`, `icp 24/24 on device`).

### 5.1 The self-test and the coarse kernel: clean

| check | measured here | D §6.8 says |
|---|---|---|
| limits | workgroup 1024, 32 768 B shared, 4 095 MB binding, 65 535 wg/dim | more than the wgpu defaults the kernels are written to |
| reduction, 1e7 `f32` | gpu `0x49a7230c` = cpu `0x49a7230c` | **bit-identical** |
| bounded nn, 384 000 queries | **1 differing (2.60e-6, limit 1e-5)**, a tie to 1.02e-8, 0 hit/miss disagreements, `max \|Δd\|` 1.32e-7 (limit 1e-6) | 1 of 384 000, tie to 1.0e-8, `max \|Δd\|` 1.3e-7 |
| throughput (idle machine) | 16.3 ns/query, **3.46×** the ten-core pool | 12.2–17.9 ns, 3.3–6.3× |

The coarse score, summed over all eight sets:

| set | hypotheses compared | differing | worst | tol |
|---|---:|---:|---|---|
| terracotta | 114 463 | **0** | 0 | 1/60 |
| pot_A | 988 382 | 27 | 1/60 | 1/60 |
| pot_B | 742 581 | 37 | 1/60 | 1/60 |
| pot_C | 265 556 | 1 | 1/60 | 1/60 |
| pot_G | 114 588 | 1 | 1/60 | 1/60 |
| pot_H | 205 068 | 2 | 1/60 | 1/60 |
| synthetic_20 | 487 147 | 2 | 1/60 | 1/60 |
| slab | 16 337 | **0** | 0 | 1/60 |
| **total** | **2 934 122** | **70** | | |

**2 934 122 hypotheses and 70 differing, each by exactly one probe point** — the two integers D
§6.5 and the G2 note print, reproduced independently and exactly, including "the two sets whose
fragments carry no near-tie agree on every bit" (terracotta and the slab at 0). The stage-1 re-score
row (`coarse s1`) is inside 1/60 on all eight, with 0–1 differing.

### 5.2 The ICP rungs: outside D §10.2, on four sets of eight

`--stage icp --pairs 4 --chaos` (the `--chaos` column re-climbs each candidate's ladder from the
twelve one-ULP neighbours of its own initial pose and excludes the ones whose own CPU answer moves
further than the row's tolerance — D §10.2's `chaotic` row). Tolerances: 0.05°, 0.01 t, 1e-4.

**Stage 1** (R §5.4, point-to-point, two rungs):

| set | `deg` | `t` (origin) | `t` @cloud | `fit` | `rmse` | `ctrl` | chaotic |
|---|---|---|---|---|---|---|---|
| terracotta | 2.30e-4 | 2.35e-5 | 1.54e-5 | 0 | 1.27e-6 | 2.88e-6 | 3 (worst over all 132.6°) |
| pot_A | 2.63e-2 | **4.32e-2** | 3.58e-3 | 0 | 9.66e-5 | 2.75e-6 | 0 |
| pot_B | 4.36e-2 | **7.76e-2** | 4.79e-3 | **3.11e-3** | **1.79e-3** | 3.03e-6 | 0 |
| pot_C | **9.14e-2** | **1.27e-1** | 7.34e-3 | **1.93e-3** | **2.58e-3** | 3.16e-6 | 0 |
| pot_G | 1.71e-4 | 5.56e-4 | 2.58e-5 | 0 | 9.41e-7 | 3.01e-6 | 0 |
| pot_H | 1.89e-4 | 3.40e-4 | 1.03e-5 | 0 | 9.97e-7 | 2.94e-6 | 0 |
| synthetic_20 | 5.04e-4 | 4.63e-4 | 8.92e-5 | 0 | 6.31e-7 | 3.09e-6 | 0 |
| slab | 7.89e-5 | 8.29e-6 | 5.03e-6 | 0 | 4.25e-7 | 2.67e-6 | 0 |
| **inside the row** | **7 of 8** | **5 of 8** | **8 of 8** | 6 of 8 | 6 of 8 | 8 of 8 | |

**Stage 2** (R §5.6, point-to-plane, four rungs):

| set | `deg` | `t` (origin) | `t` @cloud | `fit` | `rmse` | `ctrl` | attribution |
|---|---|---|---|---|---|---|---|
| terracotta | **2.113** | **2.01e-1** | **1.33e-1** | **1.60e-3** | **7.66e-4** | 5.08e-6 | **kernel** |
| pot_A | 1.08e-2 | **1.85e-2** | 1.40e-3 | **6.00e-4** | **1.50e-4** | 6.45e-6 | — |
| pot_B | 1.61e-2 | **1.68e-2** | 8.35e-4 | **5.21e-4** | 8.44e-5 | 6.99e-6 | — |
| pot_C | **5.030** | **7.274** | **3.47e-1** | **3.50e-3** | **2.33e-3** | **1.961** | ladder |
| pot_G | 2.75e-2 | **7.00e-2** | 1.80e-3 | 8.33e-5 | 3.22e-5 | 6.75e-6 | — |
| pot_H | **1.466** | **2.330** | **1.30e-1** | **3.33e-3** | **9.87e-4** | **7.84e-2** | ladder |
| synthetic_20 | **3.80e-1** | **2.65e-1** | **9.09e-2** | **3.996e-4** | **6.61e-4** | 4.81e-3 | **kernel** |
| slab | 4.03e-2 | 4.17e-3 | 1.38e-3 | 0 | 5.96e-7 | 5.89e-6 | — |
| **inside the row** | **4 of 8** | **1 of 8** | **4 of 8** | 2 of 8 | 3 of 8 | 6 of 8 | |

Every worst-case figure reproduces the G2 note's table (`notes/2026-09-08-g2-kernels.md:188-197`)
to the printed digit. `distance` and `inside` come back `delegated` on all eight sets, because
phase 2c was decided against (D §12's 2c row) and there is no kernel to compare.

`gpu-check --stage all` **exits 1 on seven of the eight sets** (terracotta 11 failing rows of 27
without `--chaos`, 5 of 23 with it; pot_A 4, pot_B 5, pot_C 10, pot_G 2, pot_H 6, synthetic_20 5;
slab 0). D §12's exit criterion for phase 2b is *"CPU/GPU cross-check within §10.2"* (**V6-D2**).

**The `ctrl` column does not support the split D §6.7 states** (**V6-D3**). D §6.7 and D §12's 2b
row say the tail is *"three sets"* where the ladder amplifies any `f32` input and *"one set, one
candidate of forty, at 2.1°"* where it is the kernel. The control — the same four `f64` rungs on
the CPU, started from the pose the device actually starts from — reads 1.961° on pot_C and 7.84e-2°
on pot_H (both themselves outside the 0.05° row: the ladder alone breaks it, so "the kernel is only
the messenger" holds), but **4.81e-3° on synthetic_20 — inside the row, and 79× under that set's
kernel deviation of 3.80e-1°.** synthetic_20's stage-2 tail is the kernel's, not the ladder's. Two
sets are kernel-attributable, not one.

**The translation rows are quoted at the cloud, and D §10.2's own row is origin-referenced**
(**V6-D4**). D §10.4 layer 3 introduces the cloud reading as *"a displacement read at the cloud
**beside** §10.2's origin-referenced one"*, so §10.2's `0.01 t` is the origin column. Stage 1 is
inside the origin column on **5 of 8** (pot_A 4.3e-2, pot_B 7.8e-2, pot_C 1.27e-1 are outside) and
stage 2 on **1 of 8**; D §6.7 and the notes report only the cloud reading for stage 1 ("inside its
translation row, read at the cloud, on all eight" — true) and do not say that the row D §10.2
actually writes is missed.

---

## 6. Quality: `run` on both backends, seven development collections

`run DIR --out … --backend {cpu,gpu} --no-preview --no-meshes`, warm, then `tools/evaluate.py`.

| set | backend | frag. accuracy | precision | cross-object | purity | joins used | groups | largest |
|---|---|---|---|---|---|---|---|---|
| pot_A | cpu / gpu | **100.0 %** / 100.0 % | 1.000 / 1.000 | 0 / 0 | 1.000 / 1.000 | 7 / 7 | 1 / 1 | 8 / 8 |
| pot_B | cpu / gpu | **88.9 %** / 88.9 % | 1.000 / 1.000 | 0 / 0 | 1.000 / 1.000 | 7 / 7 | 2 / 2 | 8 / 8 |
| pot_C | cpu / gpu | **75.0 %** / 75.0 % | 0.667 / 0.667 | 0 / 0 | 1.000 / 1.000 | 3 / 3 | 4 / 4 | 3 / 3 |
| pot_G | cpu / gpu | **0.0 %** / 0.0 % | 0.000 / 0.000 | 0 / 0 | 1.000 / 1.000 | 2 / 2 | 5 / 5 | 2 / 2 |
| pot_H | cpu / gpu | **36.4 %** / 36.4 % | 0.429 / 0.429 | 0 / 0 | 1.000 / 1.000 | 7 / 7 | 4 / 4 | 7 / 7 |
| synthetic_20 | cpu / gpu | **90.0 %** / 90.0 % | 1.000 / 1.000 | 0 / 0 | 1.000 / 1.000 | 19 / 19 | 4 / 4 | 15 / 15 |

Against the restated R §13 / D §10.3 gate: pot_A inside `87.5–100 %` at the top; pot_B inside
`88.9–100 %`; pot_C inside `50–75 %` and `0.500–0.667` at the top of both; pot_G 0 % with **two**
joins, both `wrong_pose` on ground-truth-adjacent pairs and neither `non_adjacent` nor
`cross_object` — which is exactly what the row permits; pot_H inside `27.3–36.4 %` and
`0.333–0.500`; synthetic_20 inside `85–95 %`. **Cross-object joins 0 and group purity 1.000 on all
seven single-object collections, on both backends.** Every one of these also lies inside the *old*
R §13 rows, so none of §1.2's widened bands is load-bearing for the port.

**Terracotta, R §13's own row, both backends, exactly:**

| decision | measured (identical on cpu and gpu) |
|---|---|
| joins used | exactly `{021–094, 094–104}` |
| 007 | unplaced (`placed: false`, its own group) |
| `pen` | 0.0 and 0.0 |
| `tight` | 0.5566 / 0.6705 and 0.5347 / 0.5567 — all ≥ 0.27 |
| seams | 20.667 t (1.8 % from 20.3) and 12.333 t (15.3 % from 10.7) — both inside 20 % |
| poses | **bit-identical** between the two backends |

### 6.1 Every CPU/GPU difference, traced

Comparing the two trees file by file:

| set | candidate list | accepted set | used joins | groups | `transforms.json` | worst pose gap |
|---|---|---|---|---|---|---|
| terracotta | identical, in order | identical (7) | identical | identical | **byte-identical** | 0 |
| pot_A | identical, in order | identical (71) | identical | identical | differs | **3.57e-2 °**, 5.53e-3 t |
| pot_B | identical, in order | identical (110) | identical | identical | byte-identical | 2.81e-2 °, 0 t |
| pot_C | identical, in order | identical (8) | identical | identical | differs | 2.59e-2 °, 5.49e-3 t |
| pot_G | identical, in order | identical (7) | identical | identical | byte-identical | 0 |
| pot_H | identical, in order | identical (52) | identical | identical | byte-identical | 3.93e-2 °, 0 t |
| synthetic_20 | identical, in order | identical (69) | identical | identical | differs | **6.96e-2 °**, 2.89e-4 t |

Every `cache/*.sherd` is byte-identical between the backends (preprocessing never touches the
device). The differences are confined to `report.json`, `report.md` and — on three sets —
`transforms.json`; they are the last bits of poses that went through the `f32` rungs, and each one
traces to a named candidate: on pot_A to `Pot_A_Piece_07_Mesh`, on pot_C to
`Pot_C_Piece_04_Mesh_DS`, on synthetic_20 to `frag_001`. **The worst final pose gap, 6.96e-2° and
2.89e-4 t, is inside D §10.2's `refine` row (0.2° / 0.02 t)**, which is the row a placed pose is
subject to, and no scalar score of any accepted candidate crosses a decision threshold: the accepted
sets, the used joins and the groups are identical everywhere.

`report.json`'s only *structural* difference is `engine.backend`: `"cpu"` against `"gpu"`.

---

## 7. Determinism and byte identity

### 7.1 The CPU path against `9bf35d6`

`9bf35d6` was checked out into a detached worktree **outside** the repository
(`git worktree add --detach <scratch>/base 9bf35d6`) and built there with its own target directory;
the branch was never switched. Both binaries then ran `run INPUT --out DIR` with the defaults —
previews **and** meshes on — into separate trees.

| set | files | byte-identical | identical once `timings` is removed | differing |
|---|---:|---:|---:|---:|
| terracotta | 14 | 12 | 2 (`report.json`, `report.md`) | **0** |
| pot_A | 22 | 20 | 2 | **0** |
| pot_H | 30 | 28 | 2 | **0** |
| synthetic_20 | 48 | 46 | 2 | **0** |
| **total** | **114** | **106** | **8** | **0** |

The eight are `report.json` and `report.md`, whose only differing top-level key is `timings`;
`engine` (including `backend`) is identical, and `report.md` is identical outside `## Timing`.
Caches, placed meshes, merged meshes, PNGs and `transforms.json` are byte-for-byte the ones
`9bf35d6` wrote. **The `Executor` boundary, the batch transposition, the deeper pool and the watch
moved nothing.**

### 7.2 Two runs, per backend

| set | backend | files | byte-identical | modulo `timings` | differing |
|---|---|---:|---:|---:|---:|
| pot_H | cpu | 30 | 28 | 2 | **0** |
| pot_H | gpu | 30 | 28 | 2 | **0** |
| synthetic_20 | cpu | 48 | 46 | 2 | **0** |
| synthetic_20 | gpu | 48 | 46 | 2 | **0** |

`report.json`'s only differing key is `timings`; `report.md` is identical outside `## Timing`.

### 7.3 Thread count (D §7's row, extended to the GPU)

`pot_H`, `--threads 1` against `--threads 9`: **identical on both backends** (12 files each,
`report.*` modulo timings). This matters more on the GPU than on the CPU, because
`Executor::device_slack` makes the pool a function of `--threads`; it still moves nothing, as the
trait's contract says.

### 7.4 The one latent hazard

`Allocations::reserve` (`crates/sherd-gpu/src/device.rs:309`) sends a call to the CPU when the
**live** total would exceed `--gpu-memory`. That is a function of how many calls are in flight, i.e.
of the schedule, and D §6.6 itself says a schedule-dependent split "would put a different set of
calls on the device in every run and §7's byte-identical gate would stop holding". It never fires
here — peak **74 MB of 1 000 MB, 0 refusals** on `synthetic_20`, 23 MB on `pot_H` — but the default
budget is a bound, not a switch, and the untested case is the large collections (**V6-D8**).

---

## 8. Timing and memory, both backends, cold and warm

`bench` (previews and meshes off, which is what D §10.3's gates ask for); cold is `--no-cache`;
`/usr/bin/time -l` beside every run; the device figures are the run's own accounting.

| set | backend | cold | warm | gate | verdict | peak RSS | peak device | device outstanding | submissions |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| terracotta | cpu | 3.94 s | **1.41 s** | ≤ 25 s | ok | 465 MiB | — | — | — |
| terracotta | gpu | 3.75 s | **1.37 s** | ≤ 15 s | ok | 485 MiB | 2 MB | 0.06 s | 10 |
| pot A | cpu | 5.69 s | **4.16 s** | ≤ 35 s | ok | 508 MiB | — | — | — |
| pot A | gpu | 4.82 s | **3.38 s** | ≤ 15 s | ok | 610 MiB | 14 MB | 1.00 s | 55 |
| pot H | cpu | 7.52 s | **7.25 s** | ≤ 40 s | ok | 339 MiB | — | — | — |
| pot H | gpu | 6.91 s | **6.73 s** | ≤ 15 s | ok | 471 MiB | 23 MB | 0.75 s | 59 |
| synthetic 20 | cpu | 28.51 s | **17.55 s** | ≤ 120 s | ok | 1 414 MiB | — | — | — |
| synthetic 20 | gpu | 20.61 s | **13.10 s** | ≤ 40 s | ok | 1 607 MiB | 74 MB | 5.97 s | 289 |

**Every row is inside its gate on both backends, warm and cold**; the tightest is synthetic 20's
GPU row at 13.10 s against 40 s (3.1× of margin). Peak RSS is 1.6 GiB against D §8's 3–6 GB row;
peak device memory is 74 MB against D §6.8's 1 GB.

Matching-stage speedup, from the same runs: terracotta 0.61/0.57 = **1.07×**, pot_A 3.71/2.95 =
**1.26×**, pot_H 7.15/6.63 = **1.08×**, synthetic_20 16.21/11.79 = **1.37×** — every one of the
four inside `selftest::STAGE_SPEEDUP`'s `1.07–1.43`, with `synthetic_20` above the 1.28× G4
measured for it. `bench --backend auto` takes the CPU on all four and, at `-v`, prints
why.

**GPU busy fraction.** On `synthetic_20` the device has work outstanding for **5.97 s of an 11.79 s
matching stage — 51 %**, not the 68 % D §6.8's `AUTO_ELIGIBLE` doc comment and D §6.6's G3
paragraph quote (**V6-D6**). Those figures are pre-G4: G4's own `MAX_QUERIES` paragraph says the
new ceiling *halves* occupancy (10.2 → 5.8 s), which is exactly what I measure, but the 68 % is
left standing unqualified where §6.6 and §6.8 summarise the phase. On `pot_H` the device is
outstanding 0.75 s of a 6.63 s stage (11 %) and **not one ICP rung reaches it** (`icp: 344 calls,
0 on device`): pot_H's 1.08× is the coarse kernel alone.

---

## 9. Cancellation, and what `info` prints

`run … --backend gpu` on `synthetic_20`, `SIGINT` 14 s in (mid-matching):

* prints `interrupted: finishing the units in flight, then stopping (Ctrl-C again to abort)`;
* stops at the next pair, exits **1** with `Caused by: cancelled`;
* leaves **no** `transforms.json`, `report.json` or `report.md` — nothing half-written;
* leaves 20 complete `cache/*.sherd`, **every one byte-identical** to the same run's caches from a
  completed run;
* leaves the device usable: a `--backend gpu` run started immediately afterwards completes normally
  (`0 batches refused`, 30 submissions).

A second `SIGINT` 200 ms after the first prints `interrupted again: stopping now` and exits **130**.

`info` on an idle machine opens the device, runs all four self-test checks and prints **both**
ratios with their names on them:

```
kernel:  16.3 ns per bounded-NN query, 3.46x the whole CPU pool on that batch (an idle device, E7 §5)
stage:   1.07-1.43x measured on the matching stage over the seven development collections (task G4)
--backend auto: cpu — … under the 1.5x D §6.8 asks for, because the device and the cores share one
power and bandwidth envelope (D §6.6); using the CPU
```

which is what D §9 asks for. One caution worth recording: the throughput check times the CPU side
over the live rayon pool, so the ratio it prints depends on what else the machine is doing — the
same binary reported **18.31×** while two `cargo` builds were running and **3.46×** idle. The
number the 1.5× bar is applied to is `STAGE_SPEEDUP`, not this one, so nothing decides on it; but
`info`'s "kernel" line is not reproducible under load.

---

## 10. Python

`OMP_NUM_THREADS=1 pytest -q` → **60 passed** in 106.8 s. No collection above 27 fragments was
touched.

---

## 11. Defects

| id | severity | where | R/D says | the code does |
|---|---|---|---|---|
| **V6-D1** | **blocking** | `crates/sherd-cli/src/gpu.rs:301` vs `crates/sherd-cli/src/main.rs:682,753` | D §2/§3 and `.github/workflows/rust.yml:52,81`: `sherd-gpu` is an optional feature and `--no-default-features` builds a CPU-only binary on all four CI platforms; README says the same | the `#[cfg(not(feature = "gpu"))]` `resolve` takes two arguments and both callers pass three (`args.gpu_memory`). `cargo build -p sherd-cli --no-default-features` and the matching clippy step fail with `E0061`. Broken since `d7bbdfe` (G3.1); two CI steps have been red through G3 and G4 |
| **V6-D2** | **major** | `gpu-check`, all eight sets | D §12's 2b exit criterion: "CPU/GPU cross-check within §10.2"; D §10.4 layer 3's rows carry §10.2's tolerances | `gpu-check --stage all` exits 1 on seven of eight sets. With `--chaos`, stage 2 is inside the 0.05° row on **4 of 8** and inside the 0.01 t row on **1 of 8**; stage 1 is inside 0.05° on 7 of 8 and inside 0.01 t on 5 of 8 (§5.2). D §6.7 records this rather than resolving it, so the phase closes on a criterion its own text says is unmet |
| **V6-D3** | medium | D §6.7 (¶ "Phase 2b measured the agreement"), D §12's 2b row, `notes/2026-09-08-g2-kernels.md:199-209` | "three sets" have a ladder that would move as far under any `f32` input, and "one set, one candidate of forty" is the kernel | the control reads **4.81e-3° on synthetic_20** — inside the 0.05° row, 79× under that set's own 3.80e-1° kernel deviation — so synthetic_20's tail is the kernel's. Two sets, not one, have a kernel-attributable stage-2 deviation outside the row |
| **V6-D4** | medium | D §6.7, D §12's 2b row | D §10.2's stage-1/stage-2 rows are `0.05° / 0.01 t`; D §10.4 layer 3 calls the cloud reading "beside §10.2's origin-referenced one" | only the cloud reading is quoted for the translation. On D §10.2's own origin-referenced column stage 1 is outside on pot_A, pot_B and pot_C and stage 2 on seven of eight; no note says so |
| **V6-D5** | low | `crates/sherd-gpu/src/executor.rs:366` (`device_slack`) | D §6.6: "the slack is now `min(⌈threads/2⌉, cores + 1 − threads)`" | the expression ends `.max(1)`. For `--threads > cores` D's formula gives 0 and the code gives 1, so the pool is `threads + 1` where D says `threads`. Harmless here (the default is `--threads 9` on ten cores, where both give 2) and undocumented in D |
| **V6-D6** | low | D §6.6 (G3 paragraph), D §6.8 (`AUTO_ELIGIBLE` doc) | "device occupancy 41 % → 68 %"; "the device now has work outstanding for 9.1 s of a 13.4 s stage" | measured at `879e49c`: **5.97 s of 11.79 s = 51 %**. The 68 % is pre-G4 and G4's `MAX_QUERIES` paragraph explains why it fell, but neither summary is qualified where it is quoted |
| **V6-D7** | low | D §6.8 vs D §4.3 vs `report.json` | §6.8: "`report.json` records the backend the run *asked* for". §4.3: `"engine": {"core": …, "algo_ref": …, "backend": "gpu:Apple M2 Pro"}` plus the git commit, in `report.json` **and** `transforms.json` | the file records the **resolved** backend (`--backend auto` writes `"cpu"`), the key is `core_version`, there is no git commit and no adapter name, and `transforms.json` carries no `engine` at all. The code follows §4.3's "backend used", so §6.8's sentence is the wrong one; the rest is inherited from phase 1 and never corrected |
| **V6-D8** | low (latent) | `crates/sherd-gpu/src/device.rs:309` (`Allocations::reserve`) | D §7: two runs must be byte-identical per backend. D §6.6: a schedule-dependent split "would put a different set of calls on the device in every run and §7's byte-identical gate would stop holding" | the default `--gpu-memory 1` refuses a call whose bytes do not fit **beside whatever else is in flight**, which is exactly such a split. It never fires on the development sets (peak 74 MB of 1 000 MB, 0 refusals), so determinism holds; the large collections are untested and are where it would first bite |
| **V6-D9** | informational | `gpu-check`'s `distance` and `inside` rows | D §10.4 layer 3 prints "one row per stage — items, worst deviation, §10.2's tolerance" | both rows read `delegated` on all eight sets: phase 2c was decided against (D §12's 2c row, on measurements), so R §6.1 and R §6.4 have no kernel and layer 3 has half of its table permanently empty. Correctly labelled, not silently zero — recorded here only so the gate table is read with it in view |

Two things I checked and found **not** to be defects, because both were worth checking:

* **`coarse.wgsl`'s bounding-box slack of `1.0001`** (`kernels/coarse.wgsl:112`) versus the CPU's
  `16·f64::EPSILON`. The file's comment justifies it by an *absolute* rounding of 1e-5 at 150-unit
  coordinates, which would make the slack marginal at `sc.stage1 = 0.06 t` on pot_G. The comment's
  reasoning is not what makes it safe; the filter is safe for a stronger reason. The box and the
  query are the same `f32` quantities the grid itself uses, `lo − q` is exact by Sterbenz where it
  is small, and the computed `gap²` therefore carries only ~1e-7 of *relative* error against the
  grid's own answer — 1000× inside the slack. Conservative at every radius.
* **The batch transposition of `climb_all`/`stage2_batch`.** Each candidate's ladder reads only its
  own pose and `CpuExecutor::icp_rung` calls `icp::register` once per candidate, so the nesting
  cannot move a result; §7.1's 114 byte-identical files are the proof.

---

## 12. Gate table

| gate | result |
|---|---|
| release build from `cargo clean` | **pass** |
| `cargo fmt --all --check` | **pass** |
| `cargo nextest`, **release** | **pass** — 379/379, 2 `#[ignore]`d, both pass when run |
| `cargo nextest`, **debug** | **pass** — 379/379 |
| `cargo test --doc` | **pass** |
| `cargo clippy --workspace --all-targets -D warnings` | **pass** |
| `cargo clippy -p sherd-cli --no-default-features -D warnings` | **FAIL** (V6-D1) |
| `cargo build -p sherd-cli --no-default-features` | **FAIL** (V6-D1) |
| `parity --stage all`, both modes, eight dumps | **pass** — 23 804 checks, 0 failed |
| `gpu-check --stage all`, eight sets, within D §10.2 | **FAIL** (V6-D2) — exit 1 on 7 of 8; stage 2 inside on 4 of 8 |
| D §10.3 quality, seven collections, **CPU** | **pass** |
| D §10.3 quality, seven collections, **GPU** | **pass** — identical decisions to the CPU |
| R §13's terracotta row exactly, both backends | **pass** |
| cross-object joins 0, purity 1.000, seven sets, both backends | **pass** |
| determinism, two runs, CPU | **pass** |
| determinism, two runs, GPU | **pass** |
| determinism, `--threads 1` vs `9`, both backends | **pass** |
| CPU outputs byte-identical to `9bf35d6` | **pass** — 114 files, 0 differing |
| D §10.3 runtime rows, both backends, cold and warm | **pass** — all eight inside |
| peak RSS and peak device memory | **pass** — 1.6 GiB of 6 GB, 74 MB of 1 GB |
| Ctrl-C on the GPU path | **pass** |
| `pytest -q` | **pass** — 60 passed |
| nothing above 27 fragments run, here or in the phase | **pass** |

**`gates_ok = false`**, on V6-D1 and V6-D2.

V6-D1 is a two-line fix and the CI job that catches it already exists. V6-D2 is not a fix at all:
it is the phase's own recorded measurement meeting an exit criterion that was written before the
measurement existed, and the team has to either restate D §12's 2b criterion in terms of what
D §6.7 now says a fast-math `f32` ladder can deliver, or ask for `f64` emulation in the fine rungs.
Nothing in either defect moves a decision on any development set: both backends place the same
fragments, use the same joins and score the same.

---

## 13. Housekeeping

Benchmark trees under the scratch directory, not under `output/`; nothing under `input/`,
`output/fixtures/` or `fixtures/` was written or deleted. The `9bf35d6` worktree is removed. Disk
at the end: 78 GiB free. The only file this task commits is this note.
