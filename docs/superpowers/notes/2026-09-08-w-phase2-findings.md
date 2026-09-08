# W — the nine findings of the phase-2 verification, closed

**Date:** 2026-09-08. **Tree:** branch `rust-core`, `f44ecd0` (V6) → this note. **Machine:** Apple
M2 Pro, 10 cores (6P + 4E), 16-core GPU, 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`; one Metal
adapter, no software fallback. **Python:** the tree's own `.venv`, Open3D 0.19.0,
`OMP_NUM_THREADS=1`. Nothing above 20 fragments was run.

Task W closes V6-D1 … V6-D9 of `notes/2026-09-08-phase2-verification.md` §7. Two of the nine were
code (**V6-D1**, **V6-D8**), one was a gate whose criterion had to be decided before it could be
enforced (**V6-D2**), one was an output-format defect (**V6-D7**), and five were documentation
against measurements (**V6-D3 … D6, D9**). A tenth thing turned up while measuring the second of
those and is recorded here as **W-D1**, unfixed, with the probe that reproduces it.

## 0. What each of the nine became

| id | what it was | what it is now | commit |
|---|---|---|---|
| **V6-D1** | the CPU-only binary did not compile: the `#[cfg(not(feature = "gpu"))]` `resolve` stub kept its two-argument signature after G3.1 gave both callers a third | fixed; both `--no-default-features` commands added to D §10.4's "local gates, run before every commit" list and to the README | `W1` |
| **V6-D2** | `gpu-check --stage all` exits 1 on seven of eight sets, and D §12's 2b criterion is "CPU/GPU cross-check within §10.2" | **experiment first** (§2): double-single accumulation, then a double-single solve, measured on all eight sets and **not kept**. **Then the gate** (§3): C2's `chaotic` semantics extended to the control, counted and bounded, and the exit code is the criterion exactly. The answer is stated plainly in D §6.7 and §12's 2b row: **met on one set of eight**, with the per-set table beside it | `W2`, `W4`, `W7` |
| **V6-D3** | D §6.7 attributed the stage-2 tail to the ladder on three sets and to the kernel on one | measured **per candidate** rather than per column: the tail is the kernel's on **four** sets (§3.2). D §6.7, D §12's 2b row and the G2 note corrected | `W4`, `W7` |
| **V6-D4** | only the cloud-referenced translation was quoted; D §10.2's row is origin-referenced | both are quoted and the row is named: stage 1 inside on **5 of 8** at the origin (8 of 8 at the cloud), stage 2 on **1 of 8** (4 of 8). D §6.7 and the G2 note corrected | `W7` |
| **V6-D5** | `device_slack`'s `.max(1)` undocumented in D §6.6 | kept and documented, with the reason: past `--threads > cores` the inner expression is zero and nothing would cover a worker's wait | `W7` |
| **V6-D6** | "device occupancy 41 % → 68 %" quoted for the phase; G4 halved it deliberately | re-measured on the tuned tree: **48 %** on `synthetic_20` (5.44 s of 11.34 s), 11 % on pot_H. D §6.6, D §6.8's own comment, D §12's 2d row and the g3/g4 notes carry both numbers and the reason | `W7` |
| **V6-D7** | `engine` had three fields, no commit, no adapter name, and `transforms.json` had none | all five fields, both files, exactly D §4.3 (§5). The one deliberate difference from `9bf35d6`'s CPU trees, re-verified | `W6` |
| **V6-D8** | `Allocations::reserve` refused on a shortfall caused by other calls in flight — a schedule-dependent device/CPU split | only a batch larger than the whole budget is refused; a shortfall waits. Measured counterfactual in §4 | `W3` |
| **V6-D9** | `gpu-check`'s `distance`/`inside` rows read `delegated` and nothing said why | D §10.4 layer 3 says it: permanently `delegated` by the phase-2c decision, so half that table has nothing to compare and is read as empty rather than as four silent passes | `W7` |
| **W-D1** *(new)* | — | the ICP kernel's answer is **not a function of its input when a second process drives the same adapter** (§6). Reproduced by `examples/repeat_probe`, recorded in D §7, not fixed | `W5` |

---

## 1. What the fast-math probe had to establish first

Both variants of §2's experiment are compensated arithmetic, and Metal compiles every shader with
fast math on. Fast math may rewrite `(a + b) - a` into `b`, which is the expression every
compensation scheme is built out of. Whether it *takes* that licence decides whether the experiment
is measurable at all, so `crates/sherd-gpu/examples/fastmath_probe.rs` measures it: one lane, six
sums of 200 001 terms (`1.0` and 200 000 of `1e-8`) whose exact sum is `1.002` and whose plain
in-order `f32` sum is `1.0` exactly.

| scheme | reports | compensation |
|---|---|---|
| exact, in `f64` | 1.0020000000 | — |
| 0 plain `f32` | 1.0000000000 | — |
| 1 Kahan — the form `icp.wgsl`'s `search()` already carries | 1.0000000000 | **0** |
| 2 Neumaier | 1.0000000000 | **0** |
| 3 two-sum, two-float accumulator | 1.0000000000 | **0** |
| 4 two-sum with the split forced through an integer bit mask | 1.0000000000 | **0** |
| 5 every subtraction written `fma(-1.0, b, a)` | **1.0019999743** | 2.58e-8 |

So the Kahan chain `search()` carries is dead code on this device, exactly as its own comment says,
and no `f64` emulation can be written the obvious way — including the version whose split goes
through an integer mask, which the compiler still cancels algebraically. Written through
`fma(-1.0, b, a)` it survives whole: `-1.0 · b` is exact, the fused result is `a - b` correctly
rounded, and the rewriter leaves the intrinsic alone. Both variants below are written that way.

---

## 2. V6-D2, the experiment: double-single in the ICP kernel

The brief: try, in `icp.wgsl` only, **(a)** double-single (two-float, Dekker/Knuth) accumulation of
the 27 JTJ/JTr partials and of the residual sums, and **(a)+(b)** the 6×6 solve in double-single on
invocation 0 as well; keep the fixed-order tree. Measure `gpu-check --stage icp --pairs 4 --chaos`
on all eight sets after each, and keep what takes the kernel-attributed sets inside D §10.2's
stage-2 rows at ≤ 1.5× device time.

The kernel-attributed sets are V6 §5.2's: **terracotta** (2.11° with a `ctrl` of 5.1e-6) and
**synthetic_20** (0.380° with a `ctrl` of 4.8e-3); the control column is 5e-6/4.8e-3 there, so the
ladder is not what moves them.

### 2.1 Stage-2 rotation, `deg` (tolerance 0.05°)

| set | base | (a) accumulation | (a)+(b) with the solve |
|---|---|---|---|
| terracotta | **2.113** | **2.113** | **2.113** |
| pot_A | 1.083e-2 | 9.501e-3 | 9.497e-3 |
| pot_B | 1.610e-2 | 1.613e-2 | 1.611e-2 |
| pot_C | **5.030** | **4.996** | **1.466** |
| pot_G | 2.753e-2 | 2.311e-2 | 2.310e-2 |
| pot_H | **1.466** | **1.466** | **1.466** |
| synthetic_20 | **3.800e-1** | **3.799e-1** | **3.799e-1** |
| slab | 4.034e-2 | 4.033e-2 | 4.034e-2 |
| **failing rows, all eight sets** | 5/4/5/10/1/6/5/0 | 5/4/5/10/1/6/5/0 | 5/4/5/10/1/6/5/0 |

### 2.2 Stage-2 translation, `t` at the origin (tolerance 0.01 t)

| set | base | (a) | (a)+(b) |
|---|---|---|---|
| terracotta | 2.012e-1 | 2.012e-1 | 2.012e-1 |
| pot_A | 1.847e-2 | 1.706e-2 | 1.706e-2 |
| pot_B | 1.683e-2 | 1.686e-2 | 1.684e-2 |
| pot_C | 7.274 | 7.222 | **2.135** |
| pot_G | 7.003e-2 | 6.174e-2 | 6.171e-2 |
| pot_H | 2.330 | 2.330 | 2.330 |
| synthetic_20 | 2.650e-1 | 2.650e-1 | 2.650e-1 |
| slab | 4.167e-3 | 4.167e-3 | 4.167e-3 |

### 2.3 Device time per candidate (µs, from the run's own `icp:` counter)

| set | base | (a) | (a)+(b) |
|---|---|---|---|
| terracotta | 14.79 | 15.59 (1.05×) | 16.13 (1.09×) |
| pot_A | 16.53 | 17.86 (1.08×) | 18.05 (1.09×) |
| pot_B | 15.67 | 17.65 (1.13×) | 17.31 (1.11×) |
| pot_C | 30.29 | 30.94 (1.02×) | 31.50 (1.04×) |
| pot_G | 23.06 | 24.28 (1.05×) | 23.31 (1.01×) |
| pot_H | 31.27 | 35.27 (1.13×) | 32.51 (1.04×) |
| synthetic_20 | 23.27 | 24.91 (1.07×) | 26.70 (1.15×) |
| slab | 16.16 | 16.21 (1.00×) | 19.70 (1.22×) |

### 2.4 The decision rule, applied

* **Does either take the kernel-attributed sets inside the stage-2 rows?** No. terracotta stays at
  2.113° and synthetic_20 at 0.380/0.379° — unchanged to four digits under both variants. Not one
  failing row of any set becomes a passing row under either.
* **Fallback: keep the cheaper only if it halves the worst kernel-attributed deviation.** The worst
  is terracotta's 2.113°, and (a) leaves it at 2.113°. It does not halve it; it does not move it.
* **Therefore: revert.** Both variants are discarded. Cost was 1.00–1.13× (a) and 1.01–1.22×
  (a)+(b), so the rule's time bar was never the binding constraint — the accuracy was.

**What the experiment did establish**, and it is the useful result: the `f32` loss in the fine
rungs is **not in the accumulation and not in the solve**. Double-single in both makes no
difference to the sets that fail, which leaves only one place for it to be — the *number of
iterations*. `icp s2 iter` reports a difference of up to 30 between the two sides on the failing
sets: the two ladders stop at different rungs, and once they do, a tighter sum inside a rung cannot
bring them back together. `pot_C` is the one set the solve moves at all (5.030° → 1.466°, and its
translation 7.274 → 2.135) — its 6×6 is the ill-conditioned one the equilibration was written for,
and even there the row is missed by 29×. The remedy, if one is wanted, is a rung whose stopping
decision the two sides cannot disagree on, not more precision inside the one they have.

`W2` keeps `examples/fastmath_probe.rs`, which is what makes the measurement repeatable; the two
kernel variants themselves are not kept, and the WGSL is `f6fdfc4`'s.

---

## 3. V6-D2, the gate

### 3.1 What changed

C2 §5's `chaotic` semantics — excused from the pose rows, counted, bounded — now cover the case
this harness meets. A candidate is excused for either of two **measured** reasons:

* its **control** is already outside D §10.2's pose tolerances. The control is the same four rungs,
  in `f64`, on the CPU, from the pose the device itself starts from, so when it alone is outside,
  the ladder amplifies any `f32` input and a worst case over that candidate measures the ladder.
  One extra ladder per stage, so it is **always** applied;
* or C2's twelve one-ULP neighbours of its own initial pose move its own CPU answer out of the row.
  Thirteen extra climbs, so it stays behind `--chaos`. **The run without the flag is therefore the
  stricter of the two**, because it excuses fewer candidates.

Two rows keep it honest. `icp sN chaotic` reports how many candidates were excused, per pair and
over the set, against D §10.2's own **per-pair** bounds (0.06 and 0.4 — C2's, unchanged; the
per-dump bounds are stated over tens of thousands of candidates and four pairs are not a dump).
`icp sN ctrl` becomes what it always was — an alarm over *every* candidate with no tolerance of its
own, because gating the row that defines the exclusion would gate the same fact twice and would
guarantee its own verdict. `gpu-check` exits non-zero exactly when some row is over its tolerance,
which is now exactly D §12's 2b criterion.

### 3.2 The per-set table, before and after

`gpu-check --stage all --pairs 4 --chaos`, eight sets. "before" is `f44ecd0`, "after" is this tree.

| set | s2 `deg` before | after | s2 `t` before | after | excused before | after | failing rows before → after | exit |
|---|---|---|---|---|---:|---:|---|---:|
| terracotta | 2.113 | 2.113 | 2.012e-1 | 2.012e-1 | 0 | 0 | 5 → 5 | 1 |
| pot_A | 1.083e-2 | 1.083e-2 | 1.847e-2 | 1.847e-2 | 0 | 0 | 4 → 4 | 1 |
| pot_B | 1.610e-2 | 1.610e-2 | 1.683e-2 | 1.683e-2 | 0 | 0 | 5 → 5 | 1 |
| pot_C | 5.030 | 5.030 | 7.274 | 7.274 | 1 (ULP) | **4** | 10 → 9 | 1 |
| pot_G | 2.753e-2 | 2.753e-2 | 7.003e-2 | 7.003e-2 | 1 (ULP) | **1** | 2 → 1 | 1 |
| pot_H | 1.466 | 1.466 | 2.330 | 2.330 | 0 | **1** | 6 → 5 | 1 |
| synthetic_20 | 3.800e-1 | 3.800e-1 | 2.650e-1 | 2.650e-1 | 0 | 0 | 5 → 5 | 1 |
| slab | 4.034e-2 | 4.034e-2 | 4.167e-3 | 4.167e-3 | 0 | 0 | 0 → 0 | **0** |

The rows that stopped failing are the three `ctrl` rows, which are the ones the exclusion is *made*
of. **Excusing the chaotic candidates changes no set's verdict**, which is the point worth having:
the gate is honest about what it excuses and does not depend on it.

The whole table, at D §10.2's tolerances, is in D §6.7. Summarised: stage 1 is inside the 0.05° row
on **7 of 8** sets, inside the origin-referenced 0.01 t row on **5 of 8** (8 of 8 at the cloud) and
inside the 1e-4 rows on 6 of 8; stage 2 is inside on **4 of 8**, **1 of 8** (4 of 8 at the cloud),
2 of 8 and 3 of 8. The `chaotic` rows are inside their bound on all eight (stage 1 worst pair
8.0e-3 of 0.06; stage 2 worst pair 3.0e-1 of 0.4). **`gpu-check --stage all` exits 0 on one set of
eight, and D §12's 2b exit criterion is not met.** That sentence is now in D §6.7 and in D §12's own
table, with this table beside it.

### 3.3 V6-D3: the tail is the kernel's on four sets, not one

The old attribution compared a set's worst `ctrl` with its worst `deg`. Those are per-candidate
quantities and they are **not the same candidate**. Excluding per candidate, as D §10.2's `chaotic`
row asks:

| set | candidates whose own control is outside the row | worst `deg` among the rest | attribution |
|---|---:|---|---|
| terracotta | 0 of 40 | **2.113°** | kernel |
| pot_C | 4 of 40 | **5.030°** | kernel *and* ladder |
| pot_G | 1 of 40 | 2.75e-2° (inside) | ladder only |
| pot_H | 1 of 40 | **1.466°** | kernel *and* ladder |
| synthetic_20 | 0 of 40 | **3.80e-1°** | kernel |
| pot_A, pot_B, slab | 0 | inside | — |

synthetic_20 is the clearest case: its control worst is 4.8e-3°, **79× inside** the row, and no
candidate of it is excused at all — yet the kernel lands 0.380° away. pot_C and pot_H genuinely
have amplifying ladders *and* a kernel deviation on other candidates. Four sets carry a
kernel-attributable stage-2 deviation.

---

## 4. V6-D8: the split is a function of the batch and the machine

`Allocations::reserve` used to refuse whenever `live + bytes` crossed `--gpu-memory`, and `live` is
whatever other calls hold at that instant. It now has two branches and only one is a decision:
`bytes > budget` — would not fit on an empty device — is refused, delegated and counted, and that
reads the batch's size and the ceiling and nothing else; otherwise the caller **waits** on a
condition variable and is counted in a new `waited` column. Progress is guaranteed: a shortfall
means a reservation is live, every reservation is one `Executor` call's RAII guard, and no call
reserves twice.

**The counterfactual, `synthetic_20`, `--gpu-memory 0.03`** (a 30 MB ceiling under a 61 MB peak, so
the old rule fires; two runs of each binary):

| binary | run 1 | run 2 | two runs identical outside `timings`? |
|---|---|---|---|
| `f6fdfc4` (before) | **8 batches refused** | **5 batches refused** | yes, on this pair |
| this tree | 0 refused, 7 waited | 0 refused, 8 waited | yes |

The number of refusals moving from 8 to 5 between two runs of the same collection on the same
machine *is* the defect: which calls went to the CPU was a function of thread timing, and the two
executors are one probe point apart on 70 of 2.93 M hypotheses, so a different split is licence for
a different answer. It did not produce one here — but "it has not bitten yet on ≤ 20 fragments" was
V6's point, not a defence.

**The test the brief asked for**, `synthetic_20` at `--gpu-memory 0.05`: two runs identical outside
`timings`, `transforms.json` byte-identical to each other **and to the default budget's**, the same
19 used joins and the same groups, **0 refused**, 3 and 4 waits. `tests/adapter.rs` gains a
deterministic version of the same thing: one whole budget's worth of reservation is held on the
test thread while another thread submits, and the assertions are that the call has not answered,
is neither refused nor delegated, and comes back **bit-identical** to the same batch on an empty
device.

---

## 5. V6-D7: D §4.3's `engine`, and the byte-identity re-verification

Both files now carry all five fields:

```json
"engine": {"core_version": "0.1.0", "algo_ref": "2026-09-06/9d4b9d3", "cache_version": 5,
           "commit": "215f7618364f9b4432354824697140c180f95ef6", "backend": "gpu:Apple M2 Pro"}
```

`commit` is stamped at build time by a new `crates/sherd-core/build.rs`, which watches `HEAD` and
the reflog and yields `"unknown"` outside a git checkout. It claims `HEAD` at compile time and
deliberately does not claim the tree was clean: a build script cannot see an edit in another crate
without being re-run by it. `backend` is the resolved executor with the adapter's name —
`--backend auto` writes `cpu`, `--backend gpu` writes `gpu:Apple M2 Pro` — which the code already
did; it was D §6.8's sentence ("the backend the run *asked* for") that was wrong, and it is
corrected. `info` prints the same commit.

**This is the one deliberate difference from `9bf35d6`'s CPU-backend trees.** Re-verified against a
`9bf35d6` binary built in a detached worktree outside the repository, defaults (previews **and**
meshes on), `--backend cpu`:

| set | files | byte-identical | identical modulo `timings` + `engine` | **differing** |
|---|---:|---:|---:|---:|
| terracotta | 14 | 11 | 3 | **0** |
| pot_A | 22 | 20 | 2 | **0** |
| pot_H | 30 | 27 | 3 | **0** |
| synthetic_20 | 48 | 45 | 3 | **0** |
| **total** | **114** | **103** | **11** | **0** |

The files that are not byte-identical are `report.json`, `transforms.json` and (on three sets)
`report.md`; within them the only differences are `timings` and `engine`. Caches, placed meshes,
merged meshes, PNGs and every pose are the bytes `9bf35d6` wrote. Without the `engine` exemption
the same comparison leaves exactly two differing files per set — `report.json` and
`transforms.json` — and nothing else, which is the change named and bounded.

---

## 6. W-D1 (new): two processes on one adapter

Found while measuring §3, recorded rather than fixed.

**What was seen.** Two of eight `gpu-check --set input/sfspp/pot_H --stage all --pairs 4 --chaos`
runs printed `icp s2 deg` **4.081°** where the other six print 1.466°, with `icp s2 iter` at 30
instead of 9, `0 device errors` and `0 delegated`. Separately, two `run --backend gpu` of
`synthetic_20` returned **376 and 375 candidates** on the pair `frag_005__frag_015`. In both cases
the fragment caches were byte-identical, so the difference is in the matching stage and not in its
input; in both cases a second process was driving the same adapter.

**Narrowed by `examples/repeat_probe`**, which answers *one* `IcpBatch` value N times through one
`GpuExecutor` in one process and compares bit for bit:

| condition | PointToPlane | PointToPoint |
|---|---|---|
| idle adapter | 0 of 64 differ | 0 of 64 differ |
| a ten-core CPU load, **no** second device user | 0 of 64 | 0 of 64 |
| one other process running `run --backend gpu` | 0 of 64 | **5–6 of 64 differ** |

A differing repeat has **every** candidate of the batch moved, up to thirty iterations apart, and
its answer is not the initial pose either — so it is not a readback of un-run state in the obvious
sense. The kernel's output is therefore not a function of its input under that condition, and no
amount of host-side determinism covers it.

**What was ruled out by reading, and what was not.** `icp.wgsl`'s barriers (every lane reaches
`reduce8` unconditionally, and it opens and closes with one), `Submitter`'s one-submission-one-wait
pairing, and every buffer's lifetime against its own submission were re-read and none of them
explains it. Finding the cause is a task of its own; this note's contribution is the reproduction.
D §7 gains the row, and the operational consequence: **a run whose output has to be reproducible
must have the adapter to itself.** Every gate in §8 was run that way, as every earlier verification
was.

---

## 7. Timing, both backends, after this task

`bench` (previews and meshes off), warm, three interleaved rounds, matching-stage **median**:

| set | CPU | GPU | ratio | device outstanding | occupancy | peak device |
|---|---:|---:|---:|---:|---:|---:|
| `synthetic_20` | 14.96 s | **11.34 s** | **1.32×** | 5.44 s over 289 submissions | 48 % | 68 MB |
| `pot_H` | 7.05 s | **6.37 s** | **1.11×** | 0.70 s over 59 submissions | 11 % | 21 MB |

Both inside `selftest::STAGE_SPEEDUP`'s `1.07–1.43×`. Rounds: `synthetic_20` 14.87/14.96/14.96 s
CPU against 11.14/11.34/11.70 s GPU; `pot_H` 7.05/7.09/7.05 against 6.34/6.37/6.41. The occupancy
figures are V6-D6's correction measured on this tree — G3's 68 % is pre-G4, and G4's query ceiling
halved it on purpose. `pot_H`'s 11 % is the coarse kernel alone: not one of its ICP rungs reaches
the device, which is why its ratio is the smaller one. **0 batches refused and 0 waits at the
default budget on both sets**, so §4's change costs nothing where the ceiling is not reached.

---

## 8. The standing gates, on the final tree

| gate | result |
|---|---|
| `cargo build --release --workspace --locked` | **pass** |
| `cargo fmt --all --check` | **pass** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass** |
| `cargo build -p sherd-cli --no-default-features --locked` | **pass** (V6-D1) |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass** (V6-D1) |
| `cargo nextest run --workspace --all-targets --locked --release` | **pass** — 380 run, 380 passed, 2 skipped |
| `cargo nextest run --workspace --all-targets --locked` (debug) | **pass** — 380 run, 380 passed, 2 skipped |
| `cargo test --workspace --doc --locked` | **pass** |
| `parity --stage all`, both modes, eight dumps | **pass** — **23 804 checks, 0 failed**, 16 runs exit 0 |
| CPU outputs against `9bf35d6`, four sets | **pass** — 114 files, **0 differing** beside the named `engine` change (§5) |
| two GPU runs byte-identical, seven collections | **pass** — 7/11/12/10/10/14/23 files, 0 differing (outside `timings`) |
| CPU and GPU: used joins, groups, candidate counts, seven collections | **pass** — identical on every set (2/7/7/3/2/7/19 joins) |
| `pytest -q` | **pass** — 60 passed in 106.8 s |
| nothing above 27 fragments run | **pass** — nothing above 20 |
| disk ≥ 4 GiB free at the end | **pass** — 19 GiB |

**`gpu-check --stage all` is not in this list, and that is the point of §3**: it exits 0 on one set
of eight, D §12's 2b criterion is not met, and both the spec and this note say so in those words.

Two tests are `#[ignore]`d because they need `input/test_fragments_1/fragments`, which is not in the
repository; they are on this machine and both pass.

## 9. Housekeeping

The `9bf35d6` worktree was created outside the repository and removed; the branch was never
switched and nothing under `input/`, `output/fixtures/` or `fixtures/` was written or deleted. Every
benchmark and comparison tree was under the scratch directory and is deleted. The commits are
`W1`…`W7`.
