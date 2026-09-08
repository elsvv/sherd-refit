# G1 — phase 2a: the `sherd-gpu` crate, and everything that lets a kernel be trusted

**Date:** 2026-09-08. Branch `rust-core`, six commits from `8e4eaad` (the head of task Z).
Design references: D §2 (workspace), §3 (dependencies), §6.1 (the `Executor` interface), §6.2 (the
hash grid), §6.3 (buffers and slots), §6.4 (batch formation), §6.7 (CPU/GPU agreement), §6.8
(operational concerns), §9 (CLI), §10.4 (test layers), §10.5 (CI), §12 (phasing).
Machine: Apple M2 Pro, 10 CPU cores (6P+4E), 16-core GPU, 16 GB, one Metal adapter, no software
fallback.

**Verdict: the boundary is real and the CPU did not move.** 92 output files on the four development
sets are byte-identical to `9bf35d6`'s; the parity table is 23 804 checks / 0 failed, task Z's
number to the check. On the device, E7's two measured kernels reproduce: the fixed-order reduction
of 1e7 `f32` terms is **bit-identical** (`0x49a7230c` on both sides) and the bounded-NN kernel
disagrees with the CPU on 1 neighbour of 384 000, at a tie of 1.0e-8 and a worst distance
difference of 1.3e-7. The measured throughput is **12.2–17.9 ns/query, 3.3–6.3× the whole ten-core
CPU** — E7 §5's figure, independently reproduced, and nowhere near D §6.6's original assumption.

Two things went wrong on the way and are worth more than the things that went right; both are in §5.

---

## 1. What was built

| piece | where | what it is |
|---|---|---|
| D §6.2's hash grid | `sherd-core/src/spatial/grid.rs` | the structure both executors query, built on the CPU, `Pod` in the WGSL layout |
| the `Executor` trait, four batch structs, `CpuExecutor` | `sherd-core/src/executor/{mod,batch,cpu}.rs` | D §6.1 made real; the CPU path routed through it |
| `Engine<'e>` | `sherd-core/src/executor/mod.rs` | what the pipeline passes down: the executor plus D §7's numerics |
| device, adapter selection, limits | `sherd-gpu/src/device.rs` | D §6.8's first two bullets, `--gpu-adapter NAME|INDEX` |
| batch chunking, dispatch shape, upload, readback | `sherd-gpu/src/buffers.rs` | the 128 MB cap and the 65 535-workgroup wall |
| the fragment slots | `sherd-gpu/src/slots.rs` | D §6.3's LRU of 32, ≤ 400 MB |
| the self-test | `sherd-gpu/src/selftest.rs`, `kernels/{reduce,nn}.wgsl` | D §6.8, as E7 §8 corrects it |
| `GpuExecutor` | `sherd-gpu/src/executor.rs` | all four methods, all routed to the CPU, all counted |
| `--backend`, `--gpu-adapter`, `info`, `gpu-check` | `sherd-cli/src/{gpu,main}.rs` | D §9 and D §10.4 layer 3 |

`sherd-gpu` is an optional feature of `sherd-cli`, on by default; `--no-default-features` builds a
binary with no wgpu, no naga and no driver dependency, and CI builds both shapes.

## 2. The `Executor` boundary, and the one rule that shaped it

D §6.1's trait had been a comment since phase 1a. It is now what the pipeline calls, and the four
inner loops live in `executor::cpu` and nowhere else.

The rule that shaped every decision is the task's own: **the CPU executor's results must be
byte-identical to `9bf35d6`'s, and a batch boundary that would change them is the wrong boundary.**
Three consequences, each of which contradicts a plain reading of D §6.3.

**A batch carries `f64`, not the device's `f32`.** D §6.3 says the arrays are SoA `vec4<f32>`,
`bytemuck::Pod`, 16-byte aligned — and they are: `CoarseBatch::device_points`, `Poses::device`,
`PoseGpu`, `IcpBatch::device_inits`, `HashGrid`'s four buffers. But they are produced *from* a
batch by the executor that needs them, on the way to the device, not on formation. A batch that
narrowed on formation would change what the CPU computes, and experiment E5 measured by how much:
the *median* stage-2 pose moves 9.8 `t` on pot G with `f32` point loops. The narrowing belongs
inside the GPU executor, where D §10.2's tolerance-based parity is what governs.

**The hash grid is built by the executor that queries it.** `CoarseBatch::device_grid` and
`IcpBatch::device_grid` are on the batch, as the task asks — built on the CPU, uploaded with the
batch — but the CPU path never calls them. It searches with `kiddo`: E3 measured the grid at
0.4–2.0× `kiddo` on the CPU, never the ≥ 3× D §3 hoped for, and every parity row of D §10.2 is
measured through the KD-tree. Building a grid the CPU would not query would cost `synthetic_20`
real seconds to prove nothing. The grid is not decoration: it is what `nn.wgsl` binds, and its
Rust traversal is the mirror the kernel is checked against.

**R §6.4's fallback is a reduction, not an array.** When *nothing* is inside the other mesh, R §6.4
reads `min(sd)`, and the CPU finds it with a window that shrinks to the best distance so far. That
is an ordering, not a schedule, and reassociating it into "compute every distance, then take the
minimum" would be a different float. It is expressed as `DistReduce::Min` — a min-reduction with an
early-out — so the batch describes it rather than hiding it.

### What did change shape

Two loops were re-nested, and neither can move a result, because each candidate's ladder depends on
nothing but its own pose:

* **stage 1** hands R §5.3's whole kept list to one `icp_rung` per rung and one `coarse_scores` for
  the re-score, instead of climbing one ladder per candidate under `par_iter`. That is D §6.4
  step 4 exactly.
* **stage 2** does the same over R §5.5's candidates for the two `pc_reg` rungs, and then over the
  survivors of R §5.6's early rejection for the two `pc_frac` rungs — D §6.4 step 5.
  `Pair::stage2_candidate` is now the one-element case of `Pair::stage2_batch`, so there is one
  implementation of R §5.6 and not two.

R §9's refinement and every other single-pose caller form a batch of one, which `CpuExecutor`
answers without a rayon bridge.

### The gates

| gate | result |
|---|---|
| outputs byte-identical to `9bf35d6`'s | **PASS** — 92 files on terracotta, pot_A, pot_H and synthetic_20; 85 identical as raw bytes, the four `report.json` and three `report.md` identical once the wall clock is removed |
| `parity --stage all`, 8 sets × 2 modes | **PASS** — **23 804 checks, 0 failed**, task Z's number to the check |
| `fmt`, `clippy -D warnings`, tests, debug **and** release | **PASS** — 351 tests debug, 334 release, 0 failed; clippy clean in both profiles and with `--no-default-features` |

The identity gate is E2 §9's and Z §8.1's own script and comparison
(`output/bench_g1/{identity.sh,compare_out.py}`), re-run against this tree's binary after the CLI
changes, not only after the core refactor.

## 3. The device, and what the self-test actually asserts

D §6.8's self-test runs four checks. The two that touch the device are E7's own kernels, chosen for
exactly that reason: their expected answers are measured, not assumed.

```
limits        ok — workgroup 1024 lanes, 32768 B shared, 4095 MB binding, 65535 workgroups/dim
reduction     ok — 10000000 terms, 256 workgroups: gpu 1.3691855e6 (0x49a7230c), cpu … (0x49a7230c)
bounded nn    ok — 384000 queries, 1523 occupied cells, 13 per cell at most: 1 differing neighbours
                   (2.60e-6, limit 1e-5), each a tie to 1.0244548e-8; 0 hit/miss disagreements;
                   max |Δd| 1.322478e-7 (limit 1e-6)
throughput    ok — gpu 5.45 ms (14.2 ns/query), cpu 22.81 ms over 10 threads: 4.19x
```

**The reduction asserts 32 bits, not a tolerance.** `kernels/reduce.wgsl` is D §6.4's schedule
verbatim — 256 workgroups × 256 lanes, lane `l` accumulating `a[w·block + l], a[w·block + l + 256],
…`, then a shared-memory tree 256 → 128 → … → 1, then one pass over the partials — and
`reduce_mirror` is that WGSL transcribed into single-threaded Rust. E7 §3 measured the two
bit-identical twice; so does this. That is what makes a future kernel mismatch a signal instead of a
tolerance argument, and it is why the shape (workgroup count, lane stride, loop bound, tree) is
*data in the batch descriptor* on both sides rather than a constant on one and a divide on the
other.

**The bounded-NN check is deliberately not "no neighbour differs".** E7 §5.1 measured 2.3e-6 of
queries choosing a different point in stock configuration and called it inherent: Metal contracts
`R·p + τ` into FMAs, so the transformed point already differs, and D §7's "ties → the lowest index"
cannot fix a tie that is not exact on one side. A gate of zero would contradict E7's own
measurement. So the count is held to a **rate** — 1e-5, four times the measured 2.3e-6 — and the
substantive checks are that *every* disagreement is a genuine near-tie (`|Δd|` inside the distance
tolerance) and that neither side found a neighbour the other missed. Measured here: 1 of 384 000, a
tie to 1.0e-8, 0 hit/miss disagreements.

This is the one place a reader should look hard, because "hold the count to a rate" is what a
widened tolerance looks like. The difference is that the rate came from E7's measurement before this
task existed, that the count is not the criterion, and that a genuinely wrong neighbour — one whose
distance is not the CPU's — fails on `max |Δd|` whatever the count is.

**The ratio is the NN kernel's on both sides, and only a release build's is meaningful.** The
comparison is E7 §5's: D §6.2's grid traversal on the device against the same traversal on the CPU
under rayon — not `CpuExecutor::coarse_scores` against a coarse-score kernel, which does not exist
yet, and not the CPU's `kiddo`, which is a different algorithm. It is a kernel ratio, not a
matching-stage ratio, and D §6.6's speedup table still needs re-deriving from `icp_rung`'s own cost
in phase 2b. A second caveat found while writing the adapter test: the GPU side is a compiled
kernel in every profile, while the CPU side is `sherd-core`, a workspace member and therefore `-O0`
in a debug build. The same 384 000 queries take 22.8 ms release and 131.2 ms debug, so the same
device reports 4.2× or 12.6× depending only on how the CPU half was compiled. `Backend::Auto`
decides on what the release binary measures; the row prints both wall times so the reader can see
which is which.

**`Backend::Auto`'s rule is implemented and it still says CPU.** The rule is D §6.8's: an adapter,
a passing self-test, not a software implementation of the API, and a measured ratio ≥ 1.5×. It also
reads `GpuExecutor::HAS_KERNELS`, which is `false` until phase 2b. So on this machine the device
opens, the self-test passes at 4–6×, and the run says:

> the self-test passed at 4.37x on Apple M2 Pro, but phase 2a's GPU executor has no kernels yet
> (D §12: 2b and 2c) and routes every method to the CPU; using the CPU

`--backend gpu` is accepted (the device and the self-test are real) and prints, once, that the
numbers are the CPU's; `report.json` records `backend: "gpu"`, which is the *only* file that differs
between a `--backend gpu` tree and a `--backend cpu` one — checked: 6 of 7 identical,
`report.json` differing in `engine.backend` alone. With no adapter, or with a failing self-test,
`--backend gpu` fails and names what was tried; `--gpu-adapter nvidia` on this machine gives

```
Error: no adapter matches `nvidia`; this machine offers:
  [0] Metal Apple M2 Pro (IntegratedGpu)
```

## 4. `gpu-check`, and why four rows say `delegated`

`sherd-refit-rs gpu-check [--set DIR] --stage coarse|icp|distance|inside|all` is D §10.4 layer 3.
Without a collection it runs the self-test alone; with `--set` it builds the first matchable pair
and forms one batch of each kind from it, feeding the *same* batch to both executors.

```
stage             items        worst          tol     differ  status
coarse            25054      0.000e0     1.667e-2          0  delegated — phase 2a's GPU executor routes this method to the CPU
icp                  32      0.000e0     1.000e-4          0  delegated — …
distance           5122      0.000e0     2.000e-3          0  delegated — …
inside            20000      0.000e0     5.000e-4          0  delegated — …
```

The rows say `delegated` rather than showing a deviation of 0.000e0 as a result, because the
deviation *is* zero and it means nothing: both sides ran the same CPU code. A harness that reported
those four zeros as agreement would make every phase-2b cross-check vacuous before it was written.
What the run does prove is that D §6.3's batch-formation path works on a real collection at real
sizes — 25 054 hypotheses, 32 ICP candidates, 5 122 fracture-distance points, 20 000 inside tests —
and that is the half of layer 3 that can exist before a kernel does.

`--fixture DIR` is declared and refuses with a sentence rather than pretending: the batches it would
form are the parity harness's, and there is nothing yet to compare them against.

## 5. Two things that went wrong

### 5.1 The first throughput number was the shader compiler

The first self-test reported **196.9 ns/query and 0.29× the CPU** — the GPU 3.4× *slower*, on a
kernel E7 measured at 12.2 ns/query at exactly this batch size. Nothing was wrong with the kernel.
The timed region included `create_shader_module`, `create_compute_pipeline` (where Metal compiles
the shader), every buffer upload and the readback.

Fixed by timing what E7 §7.1 says to time: `submit` + `poll(Wait)` alone, with the pipeline built
and the buffers resident, after one warm-up dispatch. The same kernel then reports 14.2 ns/query and
4.19×. The lesson is not "warm up your benchmarks" — it is that a self-test whose *purpose* is to
decide `Backend::Auto` will silently pick the wrong backend if its measurement includes anything but
the kernel, and there is no second opinion available on this platform to catch it (E7 §6).

Quoted range across the runs of this task: **12.2–17.9 ns/query, 3.3–6.3×**, on a machine shared
with other jobs. E7 §9 warned that only the timings move; they moved by the amount it predicted.

### 5.2 The self-test initialised the rayon pool, and `--threads` then failed

`run --backend gpu` died with

```
Error: --threads 9: The global thread pool has already been initialized.
```

The self-test times the same batch on the CPU, over rayon, to get the ratio `Backend::Auto` needs.
Any rayon call initialises the global pool at its default size, and `pipeline::set_threads` can only
fail afterwards. The backend was being resolved before the pool was sized.

Fixed by sizing the pool first, in both `run` and `bench`. That also makes the ratio better: it is
now measured against the pool the run will actually use, which is nine threads and not ten
(V5-D2/D3/D4's own distinction).

Neither defect could have been found by a unit test — both need a device and a real subcommand —
which is the argument for wiring `--backend gpu` end to end in 2a rather than waiting for a kernel
to justify it.

## 6. What is untestable here, and said so rather than assumed

* **The discrete-GPU staging upload.** `buffers::upload` maps at creation, which is the whole of an
  upload on unified memory; on a discrete adapter `create_buffer_init` stages through
  `queue.write_buffer`. That path is written, is what D §6.3 specifies, and **has never run**: this
  machine has one integrated Metal adapter and no software fallback of any kind (E7 §6).
* **Every row of E8's matrix but Metal.** lavapipe is Linux, WARP is Windows; the new `gpu-check`
  CI job covers them, `continue-on-error`, because a runner with no adapter is a fact to report and
  not a failure to fix. macOS runners will most likely skip.
* **The 2-D dispatch fallback beyond 65 535 workgroups** is exercised by unit tests on E7's own
  96 000-workgroup case, but no kernel in this task dispatches that many; the first that will is
  phase 2b's coarse score.
* **`Chunking` has no consumer yet.** The split to the 128 MB binding cap is arithmetic, it is
  tested (contiguous, covering, one element never spanning two chunks, an oversized element still
  yielding one per chunk), and the batches report `device_bytes()` so that a scheduler can size
  itself — but nothing dispatches a chunked batch until a kernel needs one. The self-test's own
  buffers are far inside the cap: 40 MB for the reduction's 1e7 terms, 6 MB for the NN batch.
* **`gpu-check`'s four stage rows** compare nothing yet, and say so.
* **The adapter tests skip rather than fail** when there is none (`crates/sherd-gpu/tests/adapter.rs`),
  printing what they looked for and what they found, so a CI log says whether the platform was
  covered. Everything else in that crate is arithmetic and runs everywhere.

## 7. Numbers, for phase 2b to argue with

| quantity | E7 measured | G1 measured | note |
|---|---|---|---|
| fixed-order reduction, 1e7 terms | bit-identical, twice | **bit-identical** (`0x49a7230c`) | different data, same result |
| bounded NN, differing neighbours | 2.3e-6 of queries | **2.6e-6** (1 of 384 000) | each a tie to 1.0e-8 |
| bounded NN, `max |Δd|` | 2.4e-7 on a unit cloud | **1.3e-7** | limit 1e-6, D §10.2's tightest is 1e-4 `t` |
| saturated GPU throughput | 9.0–9.5 ns/query | **12.2–17.9 ns/query** at 64 poses | E7's own 64-pose row is 12.2 |
| GPU vs the whole CPU | ≈ 4–5× (extrapolated from 4 threads) | **3.3–6.3×** (measured against 10) | the extrapolation held |
| host→device | 4.6–4.9 GB/s | not re-measured | |

D §6.6's "0.5–1 G bounded NN queries/s" remains 5–10× optimistic, and the GPU column of its speedup
table still needs re-deriving — that is phase 2b's, once `icp_rung` has a kernel and a cost of its
own, since the rung carries the 6×6 solve as well as the correspondence search.

## 8. Artefacts

Kept under `output/bench_g1/` (pruned to ~2 MB): `identity.sh`, `compare_out.py`, the eight identity
trees pruned to `transforms.json` and `report.*`, `parity_all.sh`, `summarise_parity.py`, the
sixteen parity logs, `gpu_check_selftest.txt`, `gpu_check_terracotta.txt`, `test_debug.log` and
`test_release.log`.

Nothing above 27 fragments was run. Nothing under `input/`, `output/fixtures/` or `fixtures/` was
touched. The branch never left `rust-core` and nothing was pushed.
