# F — closing the hardening verification: H1-D1 and V7-D1…D5

**Date:** 2026-09-09. **Tree:** branch `rust-core`, `c6e38bc` (V7) → this note; clean at the start,
nothing uncommitted that was not this task's. **Machine:** Apple M2 Pro, 10 cores (6P + 4E),
16-core GPU, 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`; one Metal adapter, no software
fallback. **Nothing above 24 fragments was matched**: `mixed_ABG` (24) inside
`tools/quality_gate.py`, and every other command stopped at 20.

**The answer in one paragraph.** H1-D1 was one defect and it is fixed: `umeyama_rotation` completed
a rank-deficient covariance from the *smallest* singular vector outwards, which only ever worked at
rank 2, and `synthetic_20`'s refused candidate is a **rank-0** covariance — proved here by
disabling the rank-0 branch alone and watching the same refusal come back on the same candidate.
V7-D1 was a different defect wearing the same clothes. pot_C's failing candidate is **not** rank
deficiency and **not** §6.7's stopping rule: both executors run all twenty iterations of the rung
without converging, and at iteration 9 a single correspondence of 518 sits 1.3e-6 of the radius
from the boundary — inside the `f32` resolution of the clouds, 7e-6 of the radius there — and the
two sides put it on opposite sides. That is the audit's §A.2.2 *first* discontinuity, the
correspondence set, and D §10.4's two excusals could not see it because both perturb the pose the
ladder starts from. A third, `crosscheck::boundary_batch`, measures it on the CPU in `f64` by
moving the radius by ±1 `f32` resolution; **no tolerance moved**, the excusal is counted and bounded
in a row of its own, and `gpu-check --stage all --pairs 4` now exits **0 on 8 of 8** development
sets. V7-D2…D5 are closed as written. Every standing gate passes, parity is 23 804 / 0, and the CPU
outputs of the four sets are byte-identical to `c6e38bc`'s in 71 files of 71.

---

## 1. H1-D1: the rank-deficient completion (commit F1)

### 1.1 What was wrong

`kernels/icp.wgsl`'s `umeyama_rotation` ended with

```wgsl
if (s[2] <= 0.0) { u[2] = cross3(u[0], u[1]); v[2] = cross3(v[0], v[1]); }
if (s[1] <= 0.0) { u[1] = cross3(u[2], u[0]); v[1] = cross3(v[2], v[0]); }
```

which completes **rank 2** correctly and nothing else. At rank 1 the first line builds `u₂` from a
`u₁` that is still zero, so `u₂ = 0`, and the second then builds `u₁ = 0 × u₀ = 0`; at rank 0 the
whole basis stays zero. Either way `det = 0` and `max |RᵀR − I| = 1.0`, which is the number
`icp::registration`'s orthonormality check printed on `synthetic_20` on every `--backend gpu` run.

### 1.2 What the CPU does, and where the two can agree

`matching::icp::umeyama` takes `U` and `Vᵀ` from `nalgebra`'s Golub–Reinsch SVD and returns
`U · diag(1, 1, d) · Vᵀ`. Householder reflectors are orthogonal whatever the rank, so the CPU's
bases are complete and orthonormal at every rank and its answer is always a rotation. The kernel
now is too, and the four ranks divide like this:

| rank | is the rotation determined by the data? | do the two agree? |
|---|---|---|
| 3 | yes | to `f32` — unchanged arithmetic, no branch fires |
| 2 | yes: `u₂ = ±(u₀ × u₁)` and the sign is fixed by `det U · det V` | to `f32` — unchanged arithmetic |
| 1 | **no**: every rotation carrying `v₀` to `u₀` minimises Umeyama's objective equally, a one-parameter family | they pick different members of it; both are rotations, and both carry `v₀` to `u₀` |
| 0 | yes, vacuously | **exactly**: both return the identity |

So "the two must agree on every rank" is met in the only sense it can be. At rank 1 the answer is
not a function of the data, and the honest statement — the one the new test asserts — is that both
sides return a rotation and both carry `v₀` to `u₀`.

The fix is the same completion written outwards from the largest singular vector, which is the only
order that can work when there is only one of them:

```wgsl
if (s[0] <= 0.0) { u[0] = e_x; v[0] = e_x; }                       // rank 0 → the identity basis
if (s[1] <= 0.0) { u[1] = unit(cross3(u[0], off_axis(u[0])));      // rank 1 → from u₀ and the
                   v[1] = unit(cross3(v[0], off_axis(v[0]))); }    //          axis it leans on least
if (s[2] <= 0.0) { u[2] = cross3(u[0], u[1]);                      // rank 2 → unchanged
                   v[2] = cross3(v[0], v[1]); }
```

`off_axis` returns the coordinate axis whose component of its argument is smallest in magnitude; a
unit vector has a component of at least `1/√3` on some axis, so its smallest is at most `1/√3` and
the cross product has a norm of at least `√(2/3)` — it can never be the zero vector the old code
produced. At rank 0 the basis becomes `(e_x, e_z, −e_y)` on both sides, `det U · det V = +1`, and
`U · Vᵀ` is the identity, which is what `nalgebra` gives for a zero matrix.

### 1.3 The test (`tests/adapter.rs`)

R §5.4's estimator is a pure function of nine floats, and the rank-deficient cases cannot be reached
from a real breakline cloud on purpose, so the linear algebra moved into `kernels/umeyama.wgsl` —
a file that binds nothing and is prepended to `icp.wgsl` the way `grid.wgsl` already is — and
`kernels/umeyama_probe.wgsl` is a one-lane entry point over it. `icp::umeyama_rotations` dispatches
it; the test asks the device for a rotation at five covariances and holds each against the CPU's
own rule:

| case | asserted |
|---|---|
| rank 0 (zeros) | `max \|RᵀR − I\| < 1e-5`, `\|det − 1\| < 1e-5`, and `‖R − U d Vᵀ‖ < 1e-4` — the identity on both sides |
| rank 1, `u₀` general | a rotation, and `R v₀ = u₀` to 1e-5, on **both** sides |
| rank 1, `u₀ = e_x`, `v₀ = e_y` | the same — the alignment a lazy completion would divide by zero on |
| rank 2 | a rotation, and `‖R − U d Vᵀ‖ < 1e-4` |
| rank 3 | the same |

The test **fails on the old kernel**, checked by reverting the completion and re-running:
`rank 0: |RᵀR − I| 1e0, |det − 1| 1e0`.

### 1.4 Which rank the production case was

The refusal message cannot say — the covariance never crosses the bus — so it was measured by
**disabling the rank-0 branch alone** and leaving the rank-1 completion in place. The refusal comes
back, on the same candidate, to the digit:

```
candidate 146 of 211: the rotation is 1.000e0 from orthonormal, over 1e-3
  (it reports 566 correspondences of 566 after 2 iterations)
```

So `synthetic_20`'s candidate is **rank 0**: the covariance is exactly zero in `f32`. With 566
correspondences that means the target side contributed nothing — every correspondence points at the
same target point, so `q − mean_q` is exactly zero for all of them — which is what a hypothesis that
collapses B's whole breakline onto one point of A's produces. The CPU's answer to it is the
identity rotation with the translation `mean_q − mean_p`, and the kernel's is now the same.

### 1.5 What it is worth

`run --backend gpu input/synthetic_pingsdorf_20/fragments`, idle adapter:

| | before H1 (`cdec069`) | H1…V7 | **task F** |
|---|---|---|---|
| `icp` calls on the device | 134 of 1116 | 133, 1 refused | **134 of 1148, 0 refused** |
| corrupt readbacks | not detectable | **1**, every run | **0** |
| candidates | 376 | 376 | **376** |
| used joins | 19 | 19 | **19** |
| groups | — | — | **15 + 3 + 1 + 1** |

`gpu-check --stage all --pairs 4` on `synthetic_20` reports `0 corrupt readbacks` on all four
methods, `icp s1 deg` 3.3e-4 of 5.0e-2, and every row inside its tolerance.

---

## 2. V7-D1: what pot_C's candidate is (commit F2)

### 2.1 The failure, reproduced

`gpu-check --stage all --set input/sfspp/pot_C --pairs 4` at `c6e38bc`, three rows over tolerance,
to the digit V7 and H2 quote:

```
icp s1 deg    1000  9.142e-2  tol 5.000e-2  500 differ  FAIL — p50 0.000e0 p90 7.313e-5 p99 1.159e-4
icp s1 fit    1000  1.931e-3  tol 1.000e-4    1 differ  FAIL
icp s1 rmse   1000  2.582e-3  tol 1.000e-4  500 differ  FAIL
```

`500 differ` is the count that moved at all and `1 differ` on `fit` is the count whose
*correspondence count* moved: **one candidate**, and the instrumented run names it — pair 2 of the
four, stage 1, candidate 66 of 250, on a rung of 518 source points:

| | iterations | converged | correspondences | fitness | rmse |
|---|---|---|---|---|---|
| CPU | 20 | **no** | **101** | 1.9498e-1 | 1.31592 |
| GPU | 20 | **no** | **100** | 1.9305e-1 | 1.30073 |

### 2.2 What it is not

Every one of these was measured on this tree, on that candidate, and every one came back negative.

| hypothesis | probe | result |
|---|---|---|
| **H1-D1 in another form** — a rank-deficient covariance | the SVD of the CPU's covariance printed at every iteration of the rung | `σ = (1.96e3, 4.57e2, 1.63e2)` at the first, `σ₃/σ₁` between **0.072 and 0.085** at all twenty. Nowhere near deficient |
| a **tie at the radius** at `f64` scale | the CPU rung re-run with the radius scaled by 1 ± 1e-7 and 1 ± 1e-9 | 0e0 degrees, 101 correspondences, four times of four |
| **`f32` accumulation** in the sums | the CPU rung with `Precision::F32` (the `Real` trait's `f32` point loops) | **2.32e-6 degrees** from the `f64` answer, 101 correspondences |
| the **`f32` representation of the input** | the CPU rung on the clouds shifted by their own centroid and narrowed once — exactly what `shifted()` uploads — with the pose through `device_round_trip` | **1.99e-6 degrees**, 101 correspondences |
| the **initial pose** | 24 re-climbs, each entry of the 3×4 block moved by ±1e-7 relative | worst **4.15e-6 degrees**; 101 correspondences in all 24 |
| the **search** — the hash grid against the KD-tree | both run over the narrowed cloud at six poses: the rung's start, the CPU's and the GPU's poses after 8 and after 9 updates, and both final poses | **identical sets at every one**: no index only in one, no index with a different target |
| the **stopping rule** (H2's reading, D §12's 2b row) | the two `converged` flags and the two iteration counts | both **false**, both **20**. Neither side stopped early, so nothing stopped seven iterations apart |

### 2.3 What it is

The rung was run on both executors at every iteration count from 1 to 20, from the same pose:

| m | deg | CPU corr / rmse | GPU corr / rmse |
|---:|---|---|---|
| 1–8 | 4.0e-6 … 2.4e-5 | 91…95, agreeing to 6 digits | the same |
| **9** | 2.6e-5 | 95 / 1.32718 | **94 / 1.31279** |
| 10 | 2.8e-2 | 94 / 1.31184 | 94 / 1.31132 |
| 12 | 2.5e-2 | 94 / 1.31001 | 94 / 1.30871 |
| 13 | 4.2e-2 | 94 / 1.30871 | 95 / 1.32181 |
| 15 | 2.3e-1 | 95 / 1.31231 | 100 / 1.36430 |
| 20 | **9.1e-2** | 101 / 1.31592 | 100 / 1.30073 |

From m = 12 on, `CPU(m)` and `GPU(m − 1)` agree to six or seven digits: the two are running the
*same* unconverged, oscillating trajectory one step apart. It parts at **iteration 9**, and the
correspondence search says why. At the pose after nine updates, the smallest `|d − r| / r` over the
518 source points is

* **1.30e-6** on the CPU's pose, 95 correspondences,
* **1.67e-6** on the GPU's pose, 94 correspondences,

against an `f32` resolution of the clouds of **1.62e-5 in absolute units**, or **7.0e-6 of the
radius** (r = 2.31). The margin is a quarter of the noise. One point of 518 is in on one side and
out on the other, the covariance of 95 correspondences differs from that of 94 by about a per cent,
the step differs, and twenty unconverged iterations later the two poses are 9.1e-2 degrees apart
with correspondence sets differing by one.

That is the audit's §A.2.2, first bullet, word for word: *"A source point within an ULP of the
radius flips in or out between the two sides, one row of the 6×6 changes, the step changes, and the
next iteration's set changes with it."*

### 2.4 The excusal, and why it is not a widened tolerance

D §10.4 layer 3 excuses a candidate from the pose rows for *measured* reasons, and it had two: the
control (the same `f64` rungs from the pose the device starts from) and C2's twelve one-ULP
neighbours of the initial pose. **Both perturb the pose the ladder starts from**, and neither can
reach a point that crosses the radius at iteration 9 — which is why pot_C's candidate reads
"determined" on both: control 3.2e-6 degrees, `--chaos` 0 of 1000 excused.

`crosscheck::boundary_batch` is the third reason. It re-climbs the stage-1 ladder on the CPU, in
`f64`, from the same initial poses, with **every rung's correspondence radius moved by
±1 `f32` resolution of its clouds** — `(spread_source + spread_target) · f32::EPSILON`, an absolute
distance computed per rung — and excuses a candidate whose own answer then leaves D §10.2's pose
rows. Two extra ladders, the same order as the control's one, so it is always applied rather than
behind `--chaos`. It is computed on the CPU from CPU inputs, so no kernel result can influence
which candidates it excuses.

**Stage 1 only.** With the coarse score that is what a `--backend gpu` run puts on the device
(`icp::STAGE2_ON_DEVICE` is false), so it is what D §12's 2b criterion is read over. Stage 2's rows
are `cpu by policy` and have nothing to excuse; under `--force-device` they are task W's measurement
of the kernel, which D §10.4 states is not the criterion, and they are left exactly as W measured
them. Applying the probe to stage 2 as well was measured and rejected: a stage-2 ladder is four
rungs of thirty iterations on two fracture surfaces and it excuses **25 of 40** candidates on
pot_C, which would need a bound of 0.9 to state and would state nothing.

**The shift is one resolution, and it is not on a cliff.** The excused count over pot_C's four pairs
at 0.5, 1, 2, 4, 8 and 16 resolutions is **7, 8, 9, 18, 23 and 39** of 1000; the candidate this
probe exists for is excused at 0.5 already. One is the size of the disagreement being probed: the
two executors evaluate `d² < r²` on coordinates that differ by exactly that much.

**It is bounded.** The excusals are counted in a row of their own, `icp s1 bound`, gated on the
share they cover per pair at D §10.2's own stage-1 `chaotic` number (0.06) — not folded into the
`chaotic` rows, because those bounds are calibrated on a one-ULP probe of `f64` and this one on the
`f32` resolution the two executors share. Over the eight development sets, four pairs each:

| set | excused | of | share of the set | worst pair | bound |
|---|---:|---:|---|---|---|
| pot_G | 0 | 1000 | 0.000e0 | 0.000e0 | 6.0e-2 |
| slab | 0 | 250 | 0.000e0 | 0.000e0 | 6.0e-2 |
| pot_H | 1 | 1000 | 1.0e-3 | 4.0e-3 | 6.0e-2 |
| `synthetic_20` | 1 | 860 | 1.2e-3 | 4.0e-3 | 6.0e-2 |
| terracotta | 2 | 1000 | 2.0e-3 | 4.0e-3 | 6.0e-2 |
| pot_B | 6 | 1000 | 6.0e-3 | 1.6e-2 | 6.0e-2 |
| pot_A | 7 | 1000 | 7.0e-3 | 1.2e-2 | 6.0e-2 |
| **pot_C** | **8** | 1000 | 8.0e-3 | **2.0e-2** | 6.0e-2 |

A factor of three of headroom on the worst set, and the `chaotic` rows are unchanged on every set
(pot_C still reads 0 of 1000 at stage 1 and 4 of 40 at stage 2).

### 2.5 The gate

`gpu-check --stage all --pairs 4`, at the production policy, idle adapter:

| set | exit | failing rows | `icp s1 deg` worst | tol |
|---|---|---|---|---|
| terracotta | **0** | 0 | delegated — every icp call is the CPU's under the policy | 5.0e-2 |
| pot_A | **0** | 0 | 1.586e-2 | 5.0e-2 |
| pot_B | **0** | 0 | delegated | 5.0e-2 |
| **pot_C** | **0** | 0 | **1.312e-4** (was 9.142e-2) | 5.0e-2 |
| pot_G | **0** | 0 | delegated | 5.0e-2 |
| pot_H | **0** | 0 | delegated | 5.0e-2 |
| `synthetic_20` | **0** | 0 | 3.268e-4 | 5.0e-2 |
| slab | **0** | 0 | delegated | 5.0e-2 |
| | **8 of 8** | | | |

which is the audit's plan §E step 3 gate, and D §12's 2b row now says so. Reports under
`output/f/gpucheck/`.

---

## 3. V7-D2, D3, D4, D5 (commits F3, F4, F5)

**V7-D2 — the doc rows say what the code checks.** Three places described the readback's
orthonormality rule as "orthonormal to 1e-3 with `det > 0`"; the code requires
`|det R − 1| ≤ 1e-3`, which also refuses a **reflection**. The code is right, so the prose moved:
`crates/sherd-gpu/src/icp.rs`'s doc table, D §6.8's row and the H1 note's §3 table now state both
halves and name what each catches. `ORTHONORMAL_TOLERANCE`'s own note carried H1-D1 as open with the
words "until it is fixed"; it now says what the check is for after the fix, and the H1 note's §4 and
§7 point at where H1-D1 was closed.

**V7-D3 — one number for the BVH release.** D §8 said "217–276 MiB", the H3 note said "205–275 MiB"
twice, and V7's own three warm runs gave 201, 272 and 231, so neither band bracketed the
measurement. It is now one figure in one place — D §8's row, **"about 200–275 MiB"** — stated as an
order, with the six readings behind it listed and the reason a narrower band would be reporting the
allocator's history. The H3 note quotes the order and points at D.

**V7-D4 — no dead flag named in D.** D §10.4's phase-2b paragraph introduced `--policy` as a live
flag one paragraph before the next said H2 had removed it, and §12's "2b, task G2" row still listed
`gpu-check --pairs/--chaos/--policy`, which the binary rejects. Both now read as history and both
name `--force-device` as what asks for the behaviour `--policy` used to.

**V7-D5 — the last two hand-written pose angles.** `matching::icp`'s own `mod tests` and
`tests/slab_pair.rs::pose_error` each computed a pose angle by hand. The first now calls
`pose_gap::trace_deg`, which is exactly what it was computing; the second forms its trace against a
`[[f64; 3]; 3]` ground truth and calls `pose_gap::angle_from_trace`. Both carry a line saying why
the *displacement* half stays where it is — it is a worst case over a cloud, which `pose_gap`
deliberately does not offer.

---

## 4. One thing this task found that nobody asked for (commit F6)

`the_icp_rung_crossover_is_a_candidate_count` failed once, in this task's own gate run, on the
largest cell of its table (256 candidates × 12 000 points, a 30-iteration point-to-plane pass). It
asserts which side of `icp::on_device` a batch lands on, and it read "the device did not answer
this" as "the threshold sent it to the CPU". There is a third way, and it is task H1's own: the
readback validation refused a block the driver's aborted command buffer left unwritten and delegated
the whole batch. Reproduced deliberately with a load beside it — `cpu (1 readback(s) refused)` in the
table's own column — and absent in five consecutive runs on a quiet machine.

The assertion now reads the `corrupt` counter across the call and accepts a refused readback as what
it is, printing the count; its failure message carries calls, delegations, host errors and
refusals. It still gates the routing rule and nothing was widened. This is W-D1 in a test rather
than in a run, and it is the same operational rule D §7 already states: a result that has to be
reproducible needs the adapter to itself.

---

## 5. The standing gates, on the final tree

| gate | result |
|---|---|
| `cargo test --workspace --locked`, **debug** | **pass** — 19 suites, **391 passed**, 0 failed, 2 ignored |
| `cargo test --release --workspace --locked` | **pass** — 19 suites, **391 passed**, 0 failed, 2 ignored |
| `cargo build -p sherd-cli --no-default-features --locked` | **pass** |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass** |
| `cargo fmt --all --check` | **pass** |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, **256 rows** (213 PASS, 43 SKIP), **23 804 checks, 0 failed** |
| `pytest -q` | **pass** — 60 passed |
| `tools/quality_gate.py` | **pass** — exit 0, §6 |
| **`gpu-check --stage all --pairs 4`, eight sets** | **pass — 8 of 8 exit 0** (§2.5) |
| **CPU byte-identity against `c6e38bc`** | **pass** — 71 files, **0 differing** (§7) |

The three new tests: `the_umeyama_kernel_is_a_rotation_at_every_rank_of_the_covariance`
(`tests/adapter.rs`, skips with a printed reason where there is no adapter),
`crosscheck::tests::the_boundary_row_gates_the_share_it_excuses_and_prints_the_move` and
`crosscheck::tests::the_rung_resolution_is_the_f32_step_of_the_two_shifted_clouds` (both
device-free).

Parity per dump, all failures 0:

| dump | native | injected | | dump | native | injected |
|---|---:|---:|---|---|---:|---:|
| terracotta | 145 | 631 | | pot_H | 459 | 3 650 |
| pot_A | 306 | 2 042 | | `synthetic_20` | 1 040 | 8 603 |
| pot_B | 354 | 2 527 | | slab | 81 | 248 |
| pot_C | 261 | 1 612 | | | | |
| pot_G | 248 | 1 597 | | **total** | | **23 804** |

---

## 6. The quality gate

`python tools/quality_gate.py`, backend `cpu`: **exit 0, 421.6 s = 7.0 min for 40 runs**, against
the 25-minute budget and V7's 419.8 s. (Four runs on this tree, 416.6–425.5 s; the table below is
the same in every column but `wall s` in all four, and `output/quality/quality.md` is the last of
them, at `bf656e0`.) **Its table is identical to V7's and H4's in every quality
column** — fragment accuracy, precision, the four join buckets, purity and joins, on all 40 rows of
all eight sets. The only column that moves is `wall s`, which is a clock.

| set | frag acc, seeds 0–4 | precision | cross-object | purity | verdict |
|---|---|---|---|---|---|
| `terracotta` | R §13's two joins at every seed | — | — | — | gate pass |
| `pot_A` | 100.0 / 87.5 / 75.0 / 100.0 / 100.0 % | 1.000 throughout | 0 | 1.000 | gate pass, band outside (seed 2) |
| `pot_B` | 88.9 / 88.9 / 77.8 / 100.0 / 77.8 % | 1.000, 0.857 at seeds 2 and 4 | 0 | 1.000 | gate pass, band outside |
| `pot_C` | 75.0 / 75.0 / 50.0 / 75.0 / 75.0 % | 0.667 / 0.667 / 0.500 / 0.667 / 0.667 | 0 | 1.000 | gate pass, band inside |
| `pot_G` | 0 % at every seed, 1–2 wrong-pose joins | 0.000 | 0 | 1.000 | gate pass, band inside |
| `pot_H` | 36.4 / 36.4 / **0.0** / 36.4 / 36.4 % | 0.429 / 0.429 / 0.000 / 0.429 / 0.500 | 0 | 1.000 | gate pass, band outside |
| `synthetic_20` | 90.0 / 90.0 / 85.0 / 90.0 / 95.0 % | 1.000 throughout | 0 | 1.000 | gate pass, band inside |
| `mixed_ABG` | 58.3 / 50.0 / 54.2 / 66.7 / 58.3 % | 0.667 / 0.625 / 0.688 / 0.789 / 0.706 | 3 / 4 / 3 / 2 / 1 | 0.864 / 0.750 / 0.850 / 0.850 / 0.952 | baseline, not gated |

The table is in `output/quality/quality.md`, written by the final run of it on this tree.

---

## 7. Byte identity against `c6e38bc`

`c6e38bc` built in a detached worktree **outside** the repository, with the repository's own
`CARGO_TARGET_DIR` so that no second dependency tree was written to a disk with 6.4 GiB free; both
binaries' commit stamps read back from `info` before the runs. Each binary ran
`run <set> --backend cpu --seed 0` with previews and meshes **on**, into its own fresh directory, so
each built its own cache.

| set | files | byte-identical | exempt | **differing** |
|---|---:|---:|---:|---:|
| terracotta | 10 | 7 | 3 | **0** |
| pot_A | 14 | 11 | 3 | **0** |
| pot_H | 19 | 16 | 3 | **0** |
| `synthetic_20` | 28 | 25 | 3 | **0** |
| **total** | **71** | **59** | **12** | **0** |

Every `placed/*.ply`, `assembly_*.ply` and both preview PNGs of every set are byte-identical. The
twelve exempt files are the same three per set, and each was compared key by key rather than
eyeballed — **no key was added, removed or reordered in any of them**:

| file | changed keys |
|---|---|
| `transforms.json` | `engine.commit` |
| `report.json` | `engine.commit`, `timings`, `memory` |
| `report.md` | the `## Timing` section only |

Which is the whole of what this task was allowed to move on the CPU, and it moved nothing else: the
kernels, the cross-check and the two test helpers are the only code that changed, and none of them
is on a CPU run's path.

---

## 8. Defects

**None found that this task did not close.** In particular I looked for and did not find: a moved
threshold; a widened tolerance; a reordered JSON key; a parity row that changed; a rank at which the
kernel and the CPU disagree about *being a rotation*; a development set above 27 fragments matched.

Two things a reader should know rather than discover:

* **F-N1 — at rank 1 the kernel and the CPU return different rotations, and always will.** The
  answer is not determined by the data (§1.2), the two SVDs are different algorithms, and matching
  them would mean transcribing `nalgebra`'s Householder bidiagonalisation into WGSL to get an
  arbitrary choice to agree. What is guaranteed and tested is that both are rotations carrying `v₀`
  to `u₀`. If a rank-1 covariance ever reaches a *production* rung — none does on the eight
  development sets — the two backends would give that candidate different poses, inside D §10.2's
  rows only by luck. `gpu-check`'s `icp s1 deg` row is what would say so.
* **F-N2 — the `bound` row's exclusion is bounded per pair and not per set.** Four pairs are not a
  dump, which is the argument D §10.2 already makes for the `chaotic` rows; a set-wide bound wants
  the same measurement over tens of thousands of candidates, and nothing in this workflow runs one.

---

## 9. Housekeeping

This task wrote `output/f/` only: the eight `gpu-check` reports and the sixteen parity reports
(488 KB together). The GPU run trees, the eight byte-identity run trees and the `c6e38bc` worktree
were deleted once measured; the worktree and its build lived outside the repository and the
repository's branch was never switched. `output/quality/` was rewritten by `quality_gate.py`, which
is what it is for. Nothing under `input/`, `fixtures/` or `output/fixtures/` was written or deleted.
Disk stayed at or above 6.4 GiB free throughout. The commits are `F1`…`F6` and this note.

One caution for whoever repeats §7. Building the baseline in a worktree that **shares the
repository's `CARGO_TARGET_DIR`** is what kept the identity check inside 6.4 GiB of free disk — a
second dependency tree is 4–6 GB — but it left the *repository's* own binary stamped
`git commit: unknown` for the two builds that followed, because `sherd-core`'s build script had
last run from a manifest directory that no longer existed. Nothing about a run's numbers depends on
the stamp, and the two comparison binaries were both stamped correctly (`c6e38bc` and `eac0abe`,
read back from `info` before the runs), but two quality tables were written in that window with
`commit unknown` in their header. `touch crates/sherd-core/build.rs` and one rebuild restore it;
the table under `output/quality/` is a fourth run, made after that, and carries `bf656e0`.
