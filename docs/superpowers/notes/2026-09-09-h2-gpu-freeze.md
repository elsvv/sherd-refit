# H2 — the 2b criterion read at the policy, the GPU frozen, and 1 941 lines gone

**Date:** 2026-09-09. **Tree:** branch `rust-core`, `5cf310e` (H1) → `H2.1…H2.6`. **Machine:**
Apple M2 Pro, 10 cores (6P + 4E), 16-core GPU, 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`; one
Metal adapter, no software fallback. Nothing above 20 fragments was matched. Task H2 executes
steps 3 and 5 of the audit's plan §E (`notes/2026-09-09-fable-audit.md`).

**The two sentences.** `gpu-check` used to force every rung onto the device, which measures the
kernel and not the port; read at the policy a `--backend gpu` run actually applies, it exits 0 on
**seven of the eight** development sets against **one of eight** before — and the eighth, pot_C, is
named here rather than excused, because the audit predicted eight and the measurement says seven.
`AUTO_ELIGIBLE = false` became a policy with a reason on it, D §12's 2c and 2e are struck as not
built, and nine things the audit's §B called dead weight are gone: 1 941 lines deleted against
1 716 added, of which 1 072 are the cross-check harness *moved* out of the CLI and given the first
unit tests it has ever had.

---

## 1. Step 3: the criterion at the production policy (audit §A.2.4)

### 1.1 What changed in the harness

| | change | why |
|---|---|---|
| 1 | **`--policy` deleted, `--force-device` added.** The thresholds a run applies are the default | the phase-2b criterion is a statement about the port, and the port never forces a rung |
| 2 | **`icp::STAGE2_ON_DEVICE`,** a policy constant, with `icp::on_device(candidates, points)` as the one predicate `IcpKernel::run` applies, and a unit test tying `STAGE2_CANDIDATES` to `Params::default().stage2` | "stage 2 is the CPU's" was a *consequence* of `MIN_CANDIDATES` and `MIN_WORK`; a tuning pass could have moved a verification criterion without anyone deciding to. Now the build breaks first |
| 3 | **stage-2 rows print `cpu by policy`** with the two numbers that make it so | a deviation of zero, computed by the CPU on both sides, is the lie task G1 built the `delegated` label to avoid |
| 4 | **the translation row is read at the cloud**; the origin-referenced one is printed beside it, ungated, and names the row that gates | D §10.2's origin form is a lever arm of 100–150 units times the rotation. Measured on pot_A's stage-1 poses: **4.3e-2 t at the origin, 3.6e-3 t at the fragment** — the same poses |
| 5 | **each stage's rows carry their own device count** (`stage1_calls`, `stage2_calls`) | one method-wide counter cannot say that stage 1 reached the device and stage 2 did not, which is the whole point of the restatement |

### 1.2 What the exit codes did

`gpu-check --stage all --pairs 4`, eight sets, idle adapter, 29 rows each:

| set | before (`--force-device`) | after (the policy) | which kernels the policy actually uses |
|---|---|---|---|
| terracotta | 5 failed | **0** | coarse 4 of 8 calls; every stage-1 rung the CPU's |
| pot_A | 4 failed | **0** | coarse delegated (over `MAX_QUERIES`); stage 1 6 of 8 |
| pot_B | 5 failed | **0** | coarse 3 of 8; every stage-1 rung the CPU's |
| pot_C | 9 failed | **3** | coarse 4 of 8; stage 1 4 of 8 |
| pot_G | 1 failed | **0** | coarse 4 of 8; every stage-1 rung the CPU's |
| pot_H | 5 failed | **0** | coarse 4 of 8; every stage-1 rung the CPU's |
| synthetic_20 | 5 failed | **0** | coarse 4 of 8; stage 1 2 of 8 |
| slab | 0 failed | **0** | coarse 1 of 2; every stage-1 rung the CPU's |
| **exits 0** | **1 of 8** | **7 of 8** | |

The last column is worth as much as the exit codes, and it is not flattering: at the production
policy the ICP kernel is exercised at all on **three** of eight sets and the coarse kernel on
**seven**. That is the honest size of what the 2b criterion is now stated over, and it is a
consequence of the port's own measured thresholds — a 200-point breakline cloud is under
`MIN_WORK`, a 15 M-query coarse batch is over `MAX_QUERIES`. `--force-device` still produces
D §6.7's table for anyone measuring the kernel itself.

### 1.3 pot_C: the audit predicted eight of eight and it is seven

The audit's §A.2.4 says *"under that statement the criterion is met"* on every set. It is not, and
this is the row that says so:

```
icp s1 deg    1000   9.142e-2  tol 5.000e-2   500 differ  FAIL — p50 0.000e0 p90 7.313e-5 p99 1.159e-4  [4 of 8 calls on the device]
icp s1 fit    1000   1.931e-3  tol 1.000e-4     1 differ  FAIL — p50 0.000e0 p90 0.000e0   p99 0.000e0
icp s1 rmse   1000   2.582e-3  tol 1.000e-4   500 differ  FAIL — p50 0.000e0 p90 5.762e-7  p99 1.836e-6
```

What it is, in the harness's own terms:

* **not the ladder.** The `ctrl` row — the same rungs, in `f64`, on the CPU, from the pose the
  device itself starts from — is **3.163e-6** over all thousand candidates, and 0 candidates were
  excused on it. Whatever the kernel did, its input did not do it.
* **not a chaotic candidate.** `icp s1 chaotic` is 0 of 1000 with `--chaos`; and the `--chaos` table
  of D §6.7 carries the same 9.142e-2, so the twelve one-ULP probes do not reach it either.
* **not forcing.** The number is identical with `--force-device` and without, so it is on a rung the
  policy really does send to the device.
* **one candidate.** p50 is exactly 0.0 and p99 is 1.2e-4; the worst is a single candidate of a
  thousand, and `icp s1 iter` puts it **seven iterations** apart.
* **the fragment barely moved.** `icp s1 t@cloud` is 7.340e-3 of 0.01 — inside the row.

So it is §6.7's stopping-rule discontinuity, one stage earlier than task W's table found it: a
threshold comparison (`|Δfitness| < 1e-6 ∧ |Δrmse| < 1e-6`) on `f32` sums whose own noise the
kernel's comment puts at 1.6e-6. Task W measured the remedy and rejected it — double-single
accumulation of the 27 partials, then of the 6×6 as well, moved nothing and cost up to 1.22× of
device time. **No tolerance was widened to make this row pass**, and none should be: 0.05° is
D §10.2's own number and the row is 1.8× over it.

What *is* met on all eight, and is the criterion the audit's own reasoning arrives at, is the
run level: identical used joins, identical groups, identical `evaluate.py` scores on both
backends, and a worst final pose gap of 7.0e-2° / 2.9e-4 t — inside D §10.2's `refine` row, which
is what a placed pose is held to. D §12's 2b row now states both halves and names pot_C.

### 1.4 The freeze

`GpuExecutor::AUTO_ELIGIBLE: bool = false` became
`GpuExecutor::AUTO: AutoPolicy = AutoPolicy::CpuUntilADiscreteAdapter`, and `Selection::decide`
takes the policy instead of a `bool`. The value was the same; what it *read* as was not. A `false`
with "1.07–1.43× is under 1.5×" beside it invites the next tuning pass to try again, and the
number is not that shape: the device and the ten cores share one power and bandwidth envelope, the
device loses 1.60× of its own throughput as the cores fill up (D §6.6, task G3), and no scheduler
moves that. `Selection`'s four branches are kept intact behind `AutoPolicy::MeasuredOnThisMachine`,
which is what a discrete adapter's measurement would set.

Struck from D §12, both as **not built, with the reason**:

* **2c, the BVH kernels.** Task G3 measured the move at *at most 1.2 s off a 13.38 s stage* on a
  device already 1.6× down whenever the CPU is busy. `gpu-check`'s `distance` and `inside` rows
  read `delegated` permanently and say so.
* **2e, the vendor matrix.** One Metal adapter, no software fallback (E7 §6): every cell would be
  written from a machine nobody has. Its two tuned constants are consequences of an *integrated*
  part, so a discrete card has to re-measure them rather than inherit them.

The README (Russian) now opens its GPU section with the operational rule — `--backend gpu` is an
opt-in for measuring the kernels and for a discrete adapter, `auto` is the CPU by policy — and
`info` prints the same two lines where an operator meets them.

---

## 2. Step 5: what was removed, in lines

| what | where | lines | why it went |
|---|---|---|---|
| `slots::SlotTable` | `sherd-gpu/src/slots.rs` | **−315** | D §6.3's resident set, built in phase 2a, never given a caller: a kernel uploads a *pair's* target at a *rung's* radius, not a fragment, and G3 §2 measured device-side preparation at 1.7 % of a run |
| `matching::cache::MatchCache` | `sherd-core/src/matching/cache.rs` | **−298** | E2 measured 0.9 % of CPU and nothing on the wall clock; kept for roadmap item 5, which is not scheduled. Its key carried the fragment's *address* beside its id |
| `types::Pose` | `sherd-core/src/types.rs` | **−80** | an `Isometry3` newtype used by nothing but its own test; the pipeline's pose is `Matrix4<f64>` under `icp::Pose` |
| `--icp-precision`, `--icp-assembly`, `Collection::with_icp` | `sherd-cli/src/main.rs`, `sherd-parity` | **−69** | experiment E5's instrument, and D §7 answered what it asked. The `Numerics`/`Precision`/`Assembly` enums stay: a unit test uses all four combinations |
| `Kernel::dispatch` | `sherd-gpu/src/shader.rs` | **−14** | one caller, `examples/fastmath_probe`; its four lines live there now |
| `fixture.rs` and `--dump-fixtures` | `sherd-core`, `sherd-cli` | **−12** | a twelve-line stub and a flag that only ever failed |
| four `pose_gap`s → one | `sherd-parity`, `sherd-core`, `sherd-cli`, `tests/adapter.rs` | −60 / +156 | both forms in `icp::pose_gap`, tested once, with the reason each exists |
| three `centroid`s → one | `sherd-gpu`, `sherd-cli` | −30 | `icp::centroid_of` is D §7's summation order, and the order is the thing |
| `check`/`compare_pair`/`determined_batch`/`Column` | `cli/gpu.rs` → `sherd_gpu::crosscheck` | −893 / +1 072 | 700 lines of analysis in a binary crate, running only with a device, with no unit test. `cli/gpu.rs` is 1 230 → **373** lines |

**Total over the task: 1 941 deleted, 1 716 added**, and 1 072 of the additions are the moved
harness. The net of what is *new* is the `pose_gap` module with its test, the stage-2 policy
constant with its test, `AutoPolicy`, and seven tests for the cross-check.

**What the moved harness bought**, which is why it was worth moving rather than deleting: six unit
tests that need no adapter — the row gates the determined candidates and prints the excused ones
with the worst they would have made; the three alarm rows never fail and name the row that does;
the `chaotic` row is bounded by its worst *pair* and not by the set (3 of 10 on one pair fails a
0.25 bound that 4 of 40 over the set would pass); `pose_deviation` is exactly zero for identical
bits — plus one in `tests/adapter.rs` that runs the columns and the exit rule over the synthetic
batch that file already builds, on both executors.

**Two things the audit listed and this task did not do**, deliberately: `buffers::Chunking` is not
dead (`coarse.rs` uses it, as the audit itself notes), and §B.10's third item — the second
`pairwise_sum` in `parity/stages/breakline.rs` and the duplicated `kdtree.rs` doc paragraph — is
not in this task's brief.

---

## 3. What did not move

The whole of the byte-identity gate, both backends, four development sets, against H1's tree:

| | files | byte-identical | identical modulo `timings` + `engine.commit` | **differing** |
|---|---|---|---|---|
| `--backend cpu` | 71 | 60 | 11 | **0** |
| `--backend gpu` | 71 | 59 | 12 | **0** |

`engine.commit` is in `report.json` *and* `transforms.json`, and it changes with every commit by
construction; `timings` is a clock. Nothing else moved a byte — which is the claim the `MatchCache`
removal had to support and does, a cache whose hits are bit-identical to its misses having no other
honest outcome.

The cost of removing it, measured on an **idle** machine (the byte-identity runs above were made
beside a `pytest`, so their stage times are not a comparison):
`bench input/synthetic_pingsdorf_20/fragments --backend cpu`, warm, three rounds —
14.59 / 14.79 / 14.95 s, **median 14.79 s** against H1's 14.58 s, **+1.4 %**, inside the
run-to-run spread and consistent with E2's 0.9 %. `--backend gpu`: 10.99 / 11.34 / 11.11,
**median 11.11 s** against H1's 11.10 s — unchanged, and 14.79 / 11.11 = **1.33×**, inside
`selftest::STAGE_SPEEDUP`'s 1.07–1.43×.

`synthetic_20` still reports **1 corrupt readback** on an idle adapter, deterministically: that is
H1-D1, the rank-deficient `umeyama_rotation` of `kernels/icp.wgsl`, still open and still costing
one 211-candidate stage-1 rung to the CPU executor.

---

## 4. The gates

| gate | result |
|---|---|
| `cargo test --workspace --locked`, **debug** | **pass** — 19 suites, **383 tests**, 0 failed |
| `cargo test --release --workspace --locked` | **pass** — 19 suites, **383 tests**, 0 failed |
| `cargo build -p sherd-cli --no-default-features --locked` | **pass** |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass** |
| `cargo fmt --all --check` | **pass** |
| `parity --stage all`, both modes, eight dumps | **pass** — 256 rows, **23 804 checks, 0 failed**, 16 runs exit 0, `cdec069`'s number to the check |
| `pytest -q` | **pass** — 60 passed in 116.5 s |
| **CPU byte-identity** against H1 (`5cf310e`) | **pass** — 71 files, **0 differing** |
| **GPU byte-identity** against H1 | **pass** — 71 files, **0 differing** |
| `gpu-check --stage all --pairs 4`, eight sets | **7 of 8 exit 0** (was 1 of 8); pot_C's three stage-1 rows are the miss, §1.3 |
| nothing above 27 fragments matched | **pass** — nothing above 20 |
| disk ≥ 4 GiB free | **pass** — 15 GiB at the end; `target/debug` was cleared mid-task to hold the row and rebuilt for the debug gate |

The parity total is unchanged because nothing this task touched is on a parity path: the harness
runs the CPU executor, `Pair::build` builds what it always built, and `pose_gap`'s two forms keep
every sum's terms in the order they had.

**The one gate not met is the brief's own**, and it is stated rather than worked around:
`gpu-check` exits 0 on seven sets and not on the eighth. Widening 0.05° or reading pot_C's rotation
somewhere kinder would have made the table green and the criterion worthless.

## 5. Housekeeping

What this task wrote is under `output/h2/`: the three `gpu-check` sweeps (`gpucheck-before` at
`--force-device`, `gpucheck-policy` and `gpucheck-final` at the policy), the sixteen parity reports
and the `bench` outputs. The four sets' run trees — both backends, before and after, four of them
at 753 MB each — were deleted once compared, which is what the comparison was for. The branch was never switched, nothing under `input/`,
`output/fixtures/` or `fixtures/` was written or deleted. The commits are `H2.1`…`H2.6`.
