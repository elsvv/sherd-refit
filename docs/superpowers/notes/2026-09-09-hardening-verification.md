# V7 — independent verification of the hardening (H1–H4)

**Date:** 2026-09-09. **Tree:** branch `rust-core` at `d9ef901` (H4.5), clean, nothing uncommitted
at the start of the task and nothing but this note at the end. **Baseline:** `cdec069` (the audit),
built fresh in a detached worktree **outside the repository**
(`…/scratchpad/wt-cdec069`, its own `CARGO_TARGET_DIR`); the repository's branch was never
switched. **Machine:** Apple M2 Pro, 10 cores (6P + 4E), 16-core GPU, 16 GB, macOS 24.6.0,
rustc 1.97.0, `--release`; one Metal adapter, no software fallback. **Nothing above 24 fragments was
matched by this task**: `mixed_ABG` (24) inside `tools/quality_gate.py`, and everything else stops
at 20.

**Method.** Every number below was measured on this tree by this task. The H1–H4 notes were read
for *what they claim*, and then each claim was re-derived: the diff against R, D and the audit's
§A/§B; the eight standing gates; the W-D1 contention experiment with its own second process; the
byte-identity comparison against a binary built from `cdec069` here and now; the memory, seed and
quality measurements. Where a note's number and mine differ, mine is printed beside it.

---

## 0. The verdict in one paragraph

Twenty-one commits (`H1.1`…`H4.5`), 59 files, +5 310 / −2 254. **Nothing moved a result**: the CPU
outputs of terracotta, pot_A, pot_H and `synthetic_20` at seed 0 are byte-identical to `cdec069`'s
in 71 files of 71, and the only differences are the four the notes name. **No threshold moved**;
`MIN_CANDIDATES` and `MIN_WORK` are the same constants under a new predicate that is the same
expression. **The seed path is one parameter** threaded from `Params::seed` to `SampleParams::at`,
and it is 0 everywhere it used to be a literal 0. **No key was reordered** — `engine.seed` and
`report.json`'s `memory` are appended, and the check is programmatic, not visual. The removals are
complete: no code anywhere references `SlotTable`, `MatchCache`, `types::Pose`, `fixture.rs`,
`--dump-fixtures`, `Kernel::dispatch`, `--icp-precision/--icp-assembly` or `AUTO_ELIGIBLE`, and
none of them moved a byte. **Every standing gate passes**, including parity at **23 804 / 0**,
`cdec069`'s number to the check. **W-D1 reproduces exactly** and is accounted for by the port's own
refusals. **One gate this task was asked to confirm is not met, and the executor said so first**:
`gpu-check` exits 0 on **seven** of eight development sets, not eight — pot_C's three stage-1 rows
— which is the audit's plan §E step 3 gate and item (3) of this brief. It is stated in the H2 note
§1.3 and in D §12's 2b row, it reproduces to four digits, and it is not a regression; but the gate
as written is unmet, so **`gates_ok` is false**.

---

## 1. The diff, read against R, D and the audit

### 1.1 What the twenty-one commits touch

| area | files | note |
|---|---|---|
| `crates/sherd-gpu` | 16 | H1's nonce/validity/sentinel, H2's policy constant and `crosscheck` move, the two WGSL kernels |
| `crates/sherd-core` | 17 | H2's removals, H3's semaphore, BVH release, `RssMonitor`, the seed |
| `crates/sherd-cli` | 3 | `--seed`, `--force-device`, `info`, the removed flags |
| `crates/sherd-parity` | 13 | `DUMP_SEED`, the `pose_gap` consolidation, one signature |
| specs (R, D) | 2 | |
| `README.md` | 1 | the hand-over and the quality gate, in Russian |
| notes | 5 | H1–H4 and the wgpu report |
| `tools/quality_gate.py` | 1 | new |
| `Cargo.lock` | 1 | `+tracing-subscriber` (dev-dep of `examples/repeat_probe`) |

`sherd_refit/`, `input/`, `fixtures/` and `output/fixtures/` are untouched by the range
(`git diff --name-only cdec069..HEAD` matches none of them). `9cbcbbc` is verified as the last
commit that touched `sherd_refit/`, which is what H4 freezes it at. The workspace `Cargo.toml`
carries **no `[patch.crates-io]` stanza**, so H1's vendored wgpu-hal is not in the build.

### 1.2 Did anything move a result, a default, a threshold, a seed path or a key order?

| question | answer | evidence |
|---|---|---|
| a **result** | **no** on the CPU; on the GPU exactly the one difference H1 names | §4 below: 71/71 files identical. GPU: `synthetic_20`'s one stage-1 rung is refused and answered by the CPU (H1-D1), deterministically, on every run |
| a **default** | **yes, one, and the audit asked for it**: `gpu-check` used to force every rung (`--policy` opted into the thresholds); it now applies the thresholds and `--force-device` opts into forcing (audit §A.2.4 #1). `--seed` is new with default 0 | `crates/sherd-cli/src/main.rs:128-141`; verified `Params::default().seed == 0` (`params.rs:167`) and `info` printing `default seed: 0` |
| a **threshold** | **no.** `MIN_CANDIDATES = 16`, `MIN_WORK = 100_000` unchanged; `on_device(c, p) = c >= MIN_CANDIDATES && c*p >= MIN_WORK` is the same expression the old `if !force && (n < MIN_CANDIDATES \|\| n * n_src < MIN_WORK)` applied. `STAGE2_ON_DEVICE` is derived from them with a `const` assert. `ORTHONORMAL_TOLERANCE = 1e-3` is new and gates only a *refusal* | `crates/sherd-gpu/src/icp.rs:87-135` |
| a **gate's meaning** | **yes, one, and the audit asked for it**: the translation criterion is read at the cloud instead of at the origin. It changes which sets pass — measured on pot_A stage 1 here: **4.315e-2 t at the origin** (would fail 0.01) against **3.579e-3 t at the cloud**. Both are printed; only the second gates | `output/v7/gpucheck/pot_A.txt`; D §10.4 layer 3 |
| a **seed path** | **one parameter, and it is 0 where it used to be a literal 0.** `SampleParams::at(t, seed)`; `run`/`bench` pass `Params::seed`; `sherd-parity` passes `DUMP_SEED = 0`; `segment` passes `Params::default().seed`; R §9's cap keeps `refine::CAP_SEED` and R §11.5's previews keep their literal 0 (V4-D9) | `fragment/samples.rs:113-125`, `parity/stages/mod.rs:311`, `main.rs:562-570`, `pipeline.rs:1026-1034` |
| a **key order** | **no.** `engine.seed` is appended after `engine.backend`; `report.json`'s `memory` is appended after `engine`. Checked programmatically on all four sets: shared keys in the same order, `added=['memory']` / `engine added=['seed']`, nothing removed | §4 |

### 1.3 Are the removals complete, and did they move a byte?

`grep` over `crates/`, `tools/` and `sherd_refit/` for every removed name:

| name | hits in code | hits in prose |
|---|---|---|
| `SlotTable`, `slots::` | 0 | `sherd-gpu/src/lib.rs:22`, D §6.3 — both say it was removed |
| `MatchCache`, `matching::cache` | 0 | D §5 step 3 |
| `types::Pose` | 0 | `vec3.rs:7` points at `icp::Pose` instead |
| `--dump-fixtures`, `sherd_core::fixture` | 0 | D §9's struck row, D §10.1, `tools/compare_fixtures.py:7` |
| `Kernel::dispatch` | 0 | `examples/fastmath_probe.rs:130` |
| `--icp-precision`, `--icp-assembly`, `with_icp` | 0 | `parity/stages/mod.rs:332` |
| `AUTO_ELIGIBLE` | 0 | `selftest.rs:304`, `executor.rs:240`, D §6.8 |

`cli/gpu.rs` is **373** lines (was 1 230); `sherd-gpu/src/crosscheck.rs` is **1 073** with **six**
unit tests that need no adapter, plus one in `tests/adapter.rs`. The four `pose_gap`
implementations the audit §B.9 named are all routed through `icp::pose_gap` now
(`parity/stages/mod.rs:411`, `assembly/consistency.rs:75`, `crosscheck.rs:752`,
`tests/adapter.rs:420`), and `trace_deg`/`origin_t`/`angle_from_trace` are **term-for-term
transcriptions** of the loops they replaced — same order, same clamp — which is why the parity
total does not move. `centroid_of` is the single centroid. `IcpTarget::narrowed`,
`Precision::F32` and `Assembly::Centred` survive with exactly one caller, a unit test
(`matching/icp.rs:1127`), which is what the audit §B.5 asked for ("keep the enum behind a test").

Two §B.10 items are still open and H2's note says so in as many words: the second `pairwise_sum`
in `parity/stages/breakline.rs:446`, and the duplicated bounded-NN paragraph in
`spatial/kdtree.rs:128-135` / `176-181`. §B.11 and §B.12 were never in a brief.

**Did they move a byte?** No — §4 is the answer, and it is the strong form: the `MatchCache`
removal is the one that had to be argued (a cache whose hits are bit-identical to its misses), and
71 files of four collections say it was.

### 1.4 D §12's 2b criterion: stated at the policy, and met how far

The criterion **is** restated at the production policy, in D §12's own row and in D §10.4 layer 3,
and `gpu-check` applies it by default. Measured here, `--stage all --pairs 4`, idle adapter:

| set | exit | failing rows | coarse on device | icp on device |
|---|---|---|---|---|
| terracotta | **0** | 0 | 4 of 8 | 0 of 24 |
| pot_A | **0** | 0 | 0 of 8 (over `MAX_QUERIES`) | 6 of 24 |
| pot_B | **0** | 0 | 3 of 8 | 0 of 24 |
| **pot_C** | **1** | **3** | 4 of 8 | 4 of 24 |
| pot_G | **0** | 0 | 4 of 8 | 0 of 24 |
| pot_H | **0** | 0 | 4 of 8 | 0 of 24 |
| `synthetic_20` | **0** | 0 | 4 of 8 | 2 of 24 |
| slab | **0** | 0 | 1 of 2 | 0 of 6 |
| | **7 of 8** | | **7 of 8 sets** | **3 of 8 sets** |

pot_C's three rows, to the digit H2 quotes:

```
icp s1 deg    1000  9.142e-2  tol 5.000e-2  500 differ  FAIL — p50 0.000e0 p90 7.313e-5 p99 1.159e-4
icp s1 fit    1000  1.931e-3  tol 1.000e-4    1 differ  FAIL — p50 0.000e0 p90 0.000e0   p99 0.000e0
icp s1 rmse   1000  2.582e-3  tol 1.000e-4  500 differ  FAIL — p50 0.000e0 p90 5.762e-7  p99 1.836e-6
icp s1 ctrl   1000  3.163e-6       inf                   ok  (0 excused on the control alone)
icp s1 iter   1000  7.000e0        inf                   ok
icp s1 t@cloud 1000 7.340e-3  tol 1.000e-2                ok
```

The `ctrl` row confirms the diagnosis: 3.163e-6 over all thousand candidates, so the `f32` starting
pose is not what the ladder amplified — the rung itself stopped seven iterations apart. **No
tolerance was widened to make any row pass**, and the last column of the table above is the honest
size of what the criterion is stated over: the ICP kernel is exercised on three sets of eight.

The **run-level** half of the criterion is met on every set I checked, and its number reproduces:

| set | used joins | groups | candidates | worst final pose gap, CPU against GPU |
|---|---|---|---|---|
| `synthetic_20` | 19 = 19 | identical | 376 = 376 | **6.959e-2° / 2.891e-4 t** (H2 quotes 7.0e-2 / 2.9e-4) |
| pot_H | 7 = 7 | identical | 275 = 275 | 3.931e-2° / 0.000e0 t |
| pot_A | 7 = 7 | identical | 140 = 140 | 3.565e-2° / 5.534e-3 t |
| terracotta | 2 = 2 | identical | 30 = 30 | 0.000e0° / 0.000e0 t |

All four inside D §10.2's `refine` row (0.2° / 0.02 t).

---

## 2. The standing gates, on the final tree

| gate | result |
|---|---|
| `cargo test --workspace --locked`, **debug** | **pass** — 19 suites, **388 passed**, 0 failed, 2 ignored |
| `cargo test --release --workspace --locked` | **pass** — 19 suites, **388 passed**, 0 failed, 2 ignored |
| `cargo build -p sherd-cli --no-default-features --locked` | **pass** (exit 0) |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass** (exit 0) |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass** (exit 0) |
| `cargo fmt --all --check` | **pass** (exit 0) |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs, **all exit 0**, **256 rows** (213 PASS, 43 SKIP), **23 804 checks, 0 failed** — `cdec069`'s number to the check |
| `pytest -q` | **pass** — **60 passed in 102.12 s** |
| **CPU byte-identity against `cdec069`** | **pass** — 71 files, **0 differing** (§4) |
| `gpu-check --stage all --pairs 4`, eight sets | **7 of 8 exit 0** — the brief's own gate, **not met**; §1.4 |

The tests the H-notes introduce all ran and passed in both profiles:
`icp::tests::a_block_the_device_did_not_write_is_refused`,
`icp::tests::the_nonce_is_exact_and_never_repeats_inside_a_run`,
`pipeline::tests::the_refinement_reserves_its_scans_like_preprocessing`,
`releasing_the_scenes_frees_them_and_they_come_back`,
`the_seed_moves_the_sampled_arrays_and_nothing_else`,
`the_seed_flag_reaches_the_parameters_on_run_and_bench`,
`memory::tests::the_resident_set_can_be_read_and_a_window_closes`.

Parity per dump (checks, all failures 0):

| dump | native | injected | | dump | native | injected |
|---|---:|---:|---|---|---:|---:|
| terracotta | 145 | 631 | | pot_H | 459 | 3 650 |
| pot_A | 306 | 2 042 | | `synthetic_20` | 1 040 | 8 603 |
| pot_B | 354 | 2 527 | | slab | 81 | 248 |
| pot_C | 261 | 1 612 | | | | |
| pot_G | 248 | 1 597 | | **total** | | **23 804** |

---

## 3. W-D1 (audit §A.1), reproduced with H1's binary

`examples/repeat_probe 64 4000 64 sequential`, this tree's `--release` build.

| condition | PointToPlane | PointToPoint | readbacks refused | verdict the probe prints |
|---|---|---|---|---|
| **idle adapter** | **0 of 64** | **0 of 64** | 0 | "the device answer is a function of the batch" |
| **a `run --backend gpu` loop on `synthetic_20` beside it** | 0 of 64 | **1 of 64** | **1** | "every repeat that moved has a refused readback to account for it: the CPU answered it" |

So the gate — *0 differing, or exactly the delegations counted* — holds: **1 differing, 1 refused,
0 unexplained.** The single refusal's own message is E1's census and E6's receipts in one line:

```
candidate 0 of 64: nonce 108 came back as 108 rather than 109 (it reports 0 correspondences of
4000 after 0 iterations); 0 of 64 candidates finished the rung, 64 were never started, 0 carry
neither receipt; readback 1088 words: 0 still the sentinel, 511 zero, 0 not finite
```

which is exactly H1 §2's reading: the copy ran (no sentinel survives), the dispatch did not (the
nonce is the host's own), and the block is the uploaded initial state (511 of 1088 words zero
against the 512 that state's structure predicts — one candidate's near-identity row happened not to
be). Independently corroborated from the system log over the same window:
`log show --predicate 'eventMessage CONTAINS "command buffer was aborted"'` returns **5** lines.

**Two `--backend gpu` runs on an idle adapter, all outputs compared byte for byte:**

| set | files | byte-identical | exempt (`timings` / sampled `memory`) | **differing** |
|---|---:|---:|---:|---:|
| `synthetic_20` | 28 | 26 | 2 | **0** |
| pot_H | 19 | 18 | 1 | **0** |

**H1-D1 reproduces and is deterministic.** Both `synthetic_20` GPU runs print the same refusal on
the same candidate:

```
candidate 146 of 211: the rotation is 1.000e0 from orthonormal, over 1e-3
  (it reports 566 correspondences of 566 after 2 iterations);
  211 of 211 candidates finished the rung, 0 were never started, 0 carry neither receipt;
  readback 3587 words: 0 still the sentinel, 10 zero, 0 not finite
```

`icp: 1148 calls, 133 on device, 1015 delegated (0 host errors, 1 corrupt readbacks)`, and the run
still returns 376 candidates and 19 used joins. Full `--backend gpu` runs of pot_H, pot_A and the
terracotta report **0** corrupt readbacks, and every one of the eight `gpu-check` sweeps reports
`0 corrupt readbacks` on all four methods. H1-D1 — `umeyama_rotation`'s rank-1 completion in `kernels/icp.wgsl` — is open, as the
note says.

---

## 4. Byte identity against a freshly built `cdec069`

`cdec069` built in a detached worktree outside the repository (commit stamp verified from the
binary: `git commit: cdec069b8a6d…`, `cache version: 5`, same as this tree's). Both binaries run
`run <set> --backend cpu` with previews and meshes **on**, into their own fresh directories (so
each builds its own cache), seed 0.

| set | files | byte-identical | exempt | **differing** |
|---|---:|---:|---:|---:|
| terracotta | 10 | 7 | 3 | **0** |
| pot_A | 14 | 11 | 3 | **0** |
| pot_H | 19 | 16 | 3 | **0** |
| `synthetic_20` | 28 | 25 | 3 | **0** |
| **total** | **71** | **59** | **12** | **0** |

Every `placed/*.ply`, `assembly_*.ply` and both preview PNGs of every set are **byte-identical**.
The twelve exempt files are the same three per set, and the difference in each was enumerated
key by key rather than eyeballed:

| file | added keys | changed keys | key order of the shared keys |
|---|---|---|---|
| `transforms.json` | `engine.seed` | `engine.commit` | preserved |
| `report.json` | `memory`, `engine.seed` | `engine.commit`, `timings` | preserved |
| `report.md` | — | the `## Timing` section only (a wall clock rounding to a different tenth) | — |

Which is exactly the four differences H3 §4.1 names and no others. (H2/H3 counted "60
byte-identical, 11 exempt"; I count 59/12, because in their run one `report.md`'s `## Timing`
section happened to round to the same digits. H4 §7 predicts precisely that.)

---

## 5. Scaling: the semaphore, the release, the seed (audit §B.2, §B.3, §C.3)

**`refine` under `Budget::bytes(1)`.** `pipeline::tests::the_refinement_reserves_its_scans_like_preprocessing`
passes in debug and release. It is the right test: it asserts `stats.peak_running == 1` under a
budget that admits nothing, and compares the clouds' `idx`, `points` and `normals` **as bits**
against the unbounded run's, not with `==` (`pipeline.rs:1214-1246`). The release cannot move a
result for a structural reason I checked separately: the only runtime readers of either tree are
`matching::verify` (matching) and `assembly`'s `Piece::mesh` (assembly), both before the release
point, and R §8.2's `recenter` reads `s_pen` alone (`assembly/groups.rs:218-242`), which is why
passing it a `Piece` list with `mesh: None` is safe.

**Peak RSS per stage**, `run … --backend cpu`, previews and meshes on, **warm cache**, three runs
(MiB, from the printed table and `report.json`'s `memory` block, which agree):

| stage | run 1 | run 2 | run 3 | H3 §2.3 "trees released" |
|---|---:|---:|---:|---|
| preprocess | 335 | 342 | 336 | 322 (300–343) ✓ |
| matching | 1 230 | 1 379 | 1 677 | 1 664 (1 586–1 680) |
| assembly | 842 | 1 301 | 1 660 | 1 586 (1 491–1 604) |
| refine | 1 040 | 1 245 | 1 527 | 1 459 (1 394–1 532) |
| output | 975 | 1 518 | 1 534 | 1 605 (1 281–1 784) ✓ |
| whole-run peak | 1 230 | 1 518 | 1 677 | — |

The **shape** reproduces — preprocess is a tenth of the rest on a warm cache, refine sits below
matching in every run, and the run peak is matching's in two of three — and the **absolute band is
wider than H3's**: my matching runs 1 230–1 677 against its 1 586–1 680. That is what a 100 ms
sampler on a machine with a different allocator history gives, and the note's own §2.4 says so; it
is not a discrepancy between the code and the note, it is the honest precision of the instrument.
Anyone reading D §8's row should read it as "of the order of 1.2–1.7 GiB warm", not as a band.

**The release point**, `-v`, three warm runs:

```
BVHs released rss_before_mib=1135 rss_after_mib=934  reclaimed_mib=201
BVHs released rss_before_mib=1317 rss_after_mib=1044 reclaimed_mib=272
BVHs released rss_before_mib=1317 rss_after_mib=1086 reclaimed_mib=231
```

201–272 MiB, against **205–275** in the H3 note and **217–276** in D §8 — see defect V7-D3; the
order is confirmed and neither quoted band brackets all three of my readings.

**Seeds 0–4 on the terracotta**, `run --seed k --backend cpu --no-preview --no-meshes`:

| seed | joins used | tight | seam (t) | pen | accepted | groups |
|---|---|---|---|---|---:|---|
| 0 | 021–094, 094–104 | 0.557 / 0.535 | 20.67 / 12.33 | 0 / 0 | 7 | 3 + 1 alone |
| 1 | 021–094, 094–104 | 0.550 / 0.568 | 20.67 / 12.33 | 0 / 0 | 8 | 3 + 1 alone |
| 2 | 021–094, 094–104 | 0.542 / 0.553 | 20.67 / 12.00 | 0 / 0 | 7 | 3 + 1 alone |
| 3 | 021–094, 094–104 | 0.564 / 0.557 | 21.00 / 11.33 | 0 / 0 | 6 | 3 + 1 alone |
| 4 | 021–094, 094–104 | 0.543 / 0.521 | 21.00 / 12.33 | 0 / 0 | 9 | 3 + 1 alone |

**Identical to H3 §3.1 in every cell**, and R §13's terracotta row holds at all five seeds: the two
joins and no others, 007 unplaced, both penetrations 0, both `tight` far above 0.27, and the seams
inside 20 % of 20.3 t and 10.7 t (bands 16.3–24.4 and 8.5–12.8).

---

## 6. The quality gate (audit plan §E step 6)

`python tools/quality_gate.py`, this tree (`d9ef901`), cold caches, backend `cpu`:
**exit 0, 419.8 s = 7.0 min for 40 runs**, against the 25-minute budget — and against H4's
421.7 s at `f4466d6`.

**Its table is identical to H4 §3 in every quality column** — fragment accuracy, precision, the
four join buckets, purity and joins, on all 40 rows of all eight sets. The only column that moves
is `wall s`, which is a clock. Spot rows: pot_A 100.0/87.5/**75.0**/100.0/100.0 %; pot_B seeds 2
and 4 at 77.8 % / 0.857 with one non-adjacent join each; pot_C 75/75/50/75/75 %; pot_G 0 % with
1–2 wrong-pose joins at every seed; pot_H **0.0 %** at seed 2; `synthetic_20` 90/90/85/90/95 %;
`mixed_ABG` 58.3/50.0/54.2/66.7/58.3 % at purity 0.864/0.750/0.850/0.850/0.952.

The script's two verdicts behave as H4 describes: **gate pass on all eight sets** (cross-object 0
and purity 1.000 on the seven single-object collections at every seed, pot_G's rule, the
terracotta's decision row), and **band outside on pot_A, pot_B and pot_H** — reported, not gated,
with the seeds named. I checked that the band is genuinely not a gate: `quality_gate.py` exits 0
with three sets outside it, which is the H4 decision and is the opposite of widening a bound.

The eight sets it runs are terracotta (4), pot_A (8), pot_B (9), pot_C (7), pot_G (7), pot_H (11),
`synthetic_20` (20) and `mixed_ABG` (24) — **nothing above 27**, and `mixed_all` and
`synthetic_170` appear nowhere in `SETS`.

---

## 7. `pytest`, the small-set rule, housekeeping

* `pytest -q`: **60 passed in 102.12 s**, exit 0.
* **No note in this workflow matched a collection above 27 fragments.** H1, H2 and H3 say "nothing
  above 20", H4 says "nothing above 24"; the only run trees of a large collection anywhere under
  `output/` are `output/scale/{mixed_all,synth170,synth170_full}`, whose directory timestamps are
  **6 Sep**, from the scale-pairs task, three days before H1. Nothing in the H1–H4 range writes
  them, and `quality_gate.py`'s set list stops at 24.
* This task wrote `output/v7/` only: the sixteen parity reports, the eight `gpu-check` reports, the
  W-D1 probe logs, the quality table, and the small JSON/Markdown of the identity comparison. The
  eight identity run trees, the two GPU run trees, the cross-backend trees and the RSS trees were
  deleted once compared, which is what the comparison was for. The `cdec069` worktree and its
  target directory live outside the repository and the repository's branch was never switched.
  Nothing under `input/`, `fixtures/` or `output/fixtures/` was written or deleted. Disk ≥ 6.2 GiB
  free at every point (the minimum was during `synthetic_20`'s identity pair), 7.1 GiB at the end.

---

## 8. Defects

**V7-D1 — `gpu-check` exits 0 on seven of eight development sets; the plan's step-3 gate says
eight.** *Audit plan §E step 3 says:* "`gpu-check` exits 0 on all 8 development sets". *This
brief's item (3) says the same.* *The code does:* exit 1 on pot_C, three `icp s1` rows over
tolerance (`icp s1 deg` 9.142e-2 of 5.000e-2, `icp s1 fit` 1.931e-3 of 1e-4, `icp s1 rmse`
2.582e-3 of 1e-4). Reproduced here to four digits. **Not concealed** — H2 §1.3 names it, D §12's 2b
row (`docs/superpowers/specs/2026-09-06-rust-core-design.md:2015`) names it, and both explain that
widening 0.05° would have made the table green and the criterion worthless. It is not a regression
either: task W's table carried the same 9.1e-2 with forcing on. But the gate as written is unmet,
and that is why this note's `gates_ok` is false. **What would close it** is either fixing the
stopping-rule discontinuity (task W measured that more precision does not) or the step's gate being
restated to "seven of eight, pot_C named" — a decision for the user, not for an executor.

**V7-D2 — the orthonormality check's documentation understates the check.**
`crates/sherd-gpu/src/icp.rs:540` (the doc table row) and
`docs/superpowers/notes/2026-09-09-h1-wd1.md:207` (§3's table) both say the rotation must be "orthonormal to 1e-3
**with `det > 0`**". The code at `crates/sherd-gpu/src/icp.rs:636-637` requires
`|det − 1| ≤ ORTHONORMAL_TOLERANCE`, which is far stronger than `det > 0`. The code is right and
the prose is loose; a reader auditing the refusal rule from the doc would under-specify it.
Harmless today, wrong the moment someone re-derives the check from the note.

**V7-D3 — two different numbers for one measurement of the BVH release.**
`docs/superpowers/specs/2026-09-06-rust-core-design.md:1087` (D §8's peak-RSS row) says the release
"returns **217–276 MiB**"; `docs/superpowers/notes/2026-09-09-h3-scaling.md` §2.3 and §2.4 say
**205–275 MiB**, twice. My three warm runs give **201, 272 and 231 MiB**, so neither band brackets
the measurement. The right fix is one number in one place, quoted as an order ("about 200–275 MiB")
rather than as a band a re-run must land inside.

**V7-D4 — D still names a flag that no longer exists.**
`docs/superpowers/specs/2026-09-06-rust-core-design.md:2014` (the historical "2b, task G2" row)
lists what G2 built as "`gpu-check --pairs/--chaos/--policy`"; `--policy` was deleted in H2 and the
binary rejects it (`error: unexpected argument '--policy' found`). Line 1803 introduces `--policy` as a live flag one paragraph before line 1812
says it was removed. Both are narrative rather than normative, so nothing breaks — but a reader
following D §10.4 top to bottom meets a flag, then learns it is gone, and the §12 row never
corrects itself.

**V7-D5 — a residual duplicate the §B.9 consolidation left behind (minor).** Two test-local
`pose_gap` helpers still compute a pose angle by hand:
`crates/sherd-core/src/matching/icp.rs:958-969` (inside its own `mod tests`) and
`crates/sherd-core/tests/slab_pair.rs:80`. Neither is one of the four the audit named — they
measure the worst point displacement over a cloud, which `icp::pose_gap` does not offer — but the
angle line inside them is the same expression `angle_from_trace` now owns.

Nothing else. In particular I looked for and did **not** find: a moved threshold; a widened
tolerance; a reordered JSON key; a stale reference to a removed symbol in compiled code; a
`[patch.crates-io]` left in the workspace; a result that differs between the two backends' *decisions*;
a large collection matched by any of H1–H4.

---

## 9. What this verification did not do

* **It did not re-run the H1 experiments E2–E6 in their original form.** E4's patched wgpu-hal was
  not rebuilt; the abort was corroborated instead from the system log (five lines in the window)
  and from the port's own receipts, which is the same conclusion by two independent routes but not
  the same measurement.
* **It did not measure `--force-device`.** The eight `gpu-check` runs are at the policy, which is
  the criterion; task W's kernel table was not reproduced.
* **It did not re-measure the "before" column of H3's RSS table**, which would need the trees held;
  only the "after" column and the release-point delta were reproduced.
* **It did not touch H1-D1.** The rank-deficient `umeyama_rotation` is open, correctly recorded, and
  costs one `synthetic_20` stage-1 rung to the CPU on every `--backend gpu` run.
* **It matched nothing above 24 fragments itself** — `mixed_ABG`, inside `quality_gate.py`; every
  other command it ran stopped at 20.
