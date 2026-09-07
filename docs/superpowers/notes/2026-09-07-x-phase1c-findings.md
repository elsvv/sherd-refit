# X — closing the phase-1c verification

**Date:** 2026-09-07. **Branch:** `rust-core`, from `7678722` ("V3: independent verification of
phase 1c"). **Machine:** Apple M2 Pro, 10 cores, 16 GB, macOS 15.6.1. **Toolchain:** 1.97.0 as
`rust-toolchain.toml` pins it, release profile. **Reference:** `sherd_refit/` at `895a948`
(byte-identical to the slab's `9cbcbbc`), Python 3.12.9, numpy 2.5.2, Open3D 0.19.0, scipy 1.18.1,
`OMP_NUM_THREADS=1` for every parity measurement.

`notes/2026-09-06-phase1c-verification.md` closed with twelve defects and a sweep that was not
clean. This note records what each defect became, what was measured to decide it, and where the
five native breakline failures went. **Every number below was measured in this session**; where a
measurement disagrees with an existing note the disagreement is stated rather than smoothed.

**In one line:** six of the twelve became code changes in `sherd-core` (D1, D2, D3, D4, D5, D9),
one a change to the harness (D7), two of them also needed a licence row R §12 did not have
(PMC-18 for D8, PMC-19 for what is left of D1), and four were documentation corrections
(D6, D10, D11, D12) — plus **a thirteenth defect found while writing the proof that closed D2**,
which turned out on measurement to be a false contract rather than a wrong answer, and this note
says so rather than keeping the better story. The native sweep is then green on all eight sets, on
a calibration of the two breakline rows whose derivation is in D §10.2 rather than on a tolerance
widened to fit.

---

## 1. The twelve, and the thirteenth

| # | what it was | what it became |
|---|---|---|
| D1 | `T⁻¹` by transpose where the reference factorises | **fixed** — measured at 6.2e-11 t, above the team's 1e-12 t bar, so `pose_inverse` runs the reference's own LU; the residual is **PMC-19** (§3) |
| D2 | R §6.2/§6.3's unbounded queries done bounded | **fixed** — `nearest_below` states the reference's semantics and carries the proof; writing the proof found **D13** (§6) |
| D3 | R §5.4's arg-max took the *last* tie, numpy takes the first | **fixed**, with a test (§2) |
| D4 | `nms` at `topk = 0` kept none, the reference keeps one | **fixed**, with a test, checked against the reference itself (§2) |
| D5 | the working-mesh BVHs were demanded above R §5.4's floor branch | **fixed** (§2) |
| D6 | R's header named the wrong fixture commits | corrected (§8) |
| D7 | the harness summed the reference's `frac_area` left to right | **fixed** — it uses `masked_area`, numpy's pairwise sum (§8) |
| D8 | R §7's Euler composition is not what Open3D evaluates | measured at 1.9e-13 t, under the bar → **PMC-18** (§4) |
| D9 | the LDLT pivot doc claimed a stable sort | **fixed** — the port runs Eigen's transpositions, and the doc says what is true (§4) |
| D10 | the injected `load` coordinate row is a gate with no headroom | documented in D §10.2 (§8) |
| D11 | the ten-thread cost figure was measured with OpenMP oversubscribed | **re-measured**, four configurations, on an idle machine (§5) |
| D12 | two `candidates` constants cited the wrong measured worsts | corrected to 0.417° and 0.108 t (§8) |
| **D13** | `nearest_within`'s documented radius test (`d ≤ r`) is not the one it performs | **found here**, and shown *not* to be a divergence — the contract was wrong, the answers were not (§6) |

---

## 2. The three transcription defects (D3, D4, D5)

All three are places where the port read a loop R freezes and wrote something else. None is
reachable with the shipped parameters, which is why no fixture caught any of them.

**D3.** R §5.4's floor branch returns the pose at `int(np.argmax(s1))`, and numpy's `argmax`
returns the **lowest** index among equal maxima. The port used `Iterator::max_by`, whose
documentation says the opposite: "if several elements are equally maximum, the last element is
returned". `s1` is a mean of booleans over `|brk_sub|` probe points, so its values are multiples of
`1/|brk_sub|` and exact ties are ordinary rather than exotic. `matching::pair::argmax` scans for the
first maximum; the test pins the rule on a tied list, an all-tied list and an all-zero list.

**D4.** R §5.3's loop appends the pose and *then* tests `if len(kept) >= topk: break`, so the
reference keeps **one** pose when `topk = 0`. Checked against the reference rather than read off the
source:

```
nms(order, R, tr, sc, 1.0, topk=0, floor=0.0) -> [0]
nms(order, R, tr, sc, 1.0, topk=0, floor=1.0) -> []
```

The port returned none in both cases. The early return is gone and the loop is R §5.3's shape.

**D5.** `pair.rs` demanded both working-mesh scenes — which builds two BVHs — above the
`stage1_floor` test, where the reference builds `frac_scene` lazily on first use inside R §6.1. A
pair whose working mesh has no triangle returned `[]` where R §5.4 returns the partial candidate.
The call moved below the branch; nothing else changed.

---

## 3. D1 — the inverse, measured (PMC-19)

R §6.1 and R §6.4 move A's points backwards through the pose and write `np.linalg.inv(T)`. The port
substituted `[Rᵀ | −Rᵀτ]`, which is the inverse of the *rotation* and equals the inverse of the
matrix only to the extent that `R` is orthonormal.

Measured over all **2 239 stage-2 poses** of the eight dumps (`s2.T_frac2`) applied to the A-side
`md.Pf` and `md.S` the reference itself dumped — 67.3 M point applications — as the largest
displacement of any point, in units of the pair's own `t`:

| | p50 | p99 | max |
|---|---:|---:|---:|
| transpose against `np.linalg.inv` | 4.6e-13 t | 2.6e-12 t | **6.2e-11 t** |
| partial-pivot LU against `np.linalg.inv` | **2.7e-14 t** | **1.5e-13 t** | 1.2e-11 t |

Per set, worst case, transpose / LU: terracotta 2.2e-13 / 6.7e-15, pot_A 2.3e-12 / 8.4e-14, pot_B
1.9e-11 / 5.9e-13, pot_C 1.6e-11 / 7.0e-13, pot_G 6.2e-11 / 1.9e-12, pot_H 3.7e-11 / 1.2e-11,
synthetic_20 6.0e-12 / 1.5e-13.

**Why the transpose costs as much as it does.** A pose that has climbed two ICP ladders is
orthonormal to `|RᵀR − I| ≤ 2.6e-14`, not to 1e-16, and a sample sits up to **885 units** from the
origin on these scans — so a 2.6e-14 error in the rotation is a 2e-11-unit displacement before `t`
divides it. The C3 note's "1e-14 on a point a hundred units from the origin" was an estimate at the
wrong orthonormality and the wrong radius.

6.2e-11 t is above the team's 1e-12 t bar, so `verify.rs::pose_inverse` now runs the reference's
own operation: `dgetf2` partial pivoting on the largest remaining column with the first index on a
tie, then a forward and a back substitution per column of `P·I`. It is written out rather than
delegated because nalgebra's `Matrix4::try_inverse` is a cofactor expansion — a different algorithm
from the reference's — and because a written loop is the same arithmetic on every machine (D §7). A
test pins it against numpy's own answer on a real stage-2 pose at 4 ULP per entry.

**What is left is PMC-19, and it is a library row.** LAPACK's blocked kernels against this loop
disagree by 1.2e-11 t at the worst, on candidates whose ICP diverged: the pot_H pose that produces
it has `‖τ‖ = 181 867` and `cond(T) = 3.3e10`. At the median the two agree to 2.7e-14 t.

---

## 4. D8 and D9 — the two Open3D internals (PMC-18, and a doc that is now true)

**D8, the Euler composition.** R §7 writes the update as `Rz(x₂)·Ry(x₁)·Rx(x₀)`, three explicit
matrices. Open3D's `TransformationMatrixFromPoseVector` writes
`(AngleAxisd(x₂,Z) * AngleAxisd(x₁,Y) * AngleAxisd(x₀,X)).matrix()`, and Eigen's `operator*` on two
`AngleAxis` converts both to quaternions, multiplies those, and converts the product to a matrix
once at the end. The two are the same rotation and not the same arithmetic.

Measured by transcribing Eigen's own expressions — `Quaternion(AngleAxis)`'s half-angle sines,
`quat_product`'s term order, `toRotationMatrix`'s `1 − (tyy + tzz)` — and sweeping the angles an
ICP update produces over eleven decades (1e-1 down to 1e-12 rad) and both signs:

* worst entrywise difference **3.33e-16 = 1.5 ULP of 1**;
* the largest displacement that can give a point 885 units from the origin is **4.4e-13 units**,
  which is **1.9e-13 t** on the thinnest benchmark wall (2.36 on pot_G).

Under the 1e-12 t bar, so the row is the answer: **PMC-18**. R §7's form is kept and R §12 now says
that Open3D does not evaluate it. The accumulated effect of an ULP over a whole ladder is what the
injected `stage 1` and `stage 2` pose rows already measure, and the `chaotic` rows are the honest
statement that an ULP is not bounded on 48 of 74 090 candidates.

**D9, the LDLT pivots.** `solve_ldlt`'s documentation claimed Eigen's permutation is "exactly a
stable descending sort of the original diagonal". The first half is right and the doc now says why:
`ldlt_inplace` is left-looking, `mat(k, k)` is decremented at step `k` *after* `k` has been chosen,
and the rank-1 update below it touches the column and not the diagonal — so the values compared are
the original diagonal's. The tie-break is not a stable sort. `maxCoeff` keeps the first of several
equal maxima, but the indices it walks are the current ones and the transpositions have already
reordered the tail; on `[3, 3, 5]` Eigen ends at the original indices `[2, 1, 0]` where a stable
descending sort gives `[2, 0, 1]`.

`eigen_pivots` runs Eigen's selection sort instead. On distinct diagonal entries it is the same
permutation, so nothing on the fixtures moves — the deviation was exactly zero and stays zero, and
the point is that the port no longer depends on a real 6×6 normal-equations matrix never having a
tie.

---

## 5. D11 — the cost figure, re-measured

Mean per pair over the terracotta's six pairs, preprocessing excluded on both sides, **on an idle
machine** (no parity sweep, no other job): the reference through
`matching.match_pair(A, B, p, n_threads=N)`, the port through `sherd-parity`'s new `pair_cost`
example, which is committed so the comparison can be repeated.

| reference configuration | mean | note |
|---|---:|---|
| `OMP_NUM_THREADS=1`, `n_threads=1` | **7.43 s** | D §10.2 said 7.62 s — reproduces to 3 % |
| OpenMP free, `n_threads=1` | **3.15 s** | D §10.2 said 3.40 s |
| OpenMP free, `n_threads=10` | **5.98 s** | D §10.2 said 5.83 s, and called it "ten threads" |
| **`OMP_NUM_THREADS=1`, `n_threads=10`** | **1.70 s** (1.82 s on a repeat) | **the reference's best; no row existed** |

Port: **1.031 s** at one thread (0.926–1.186 per pair), **0.193 s** at ten (0.152–0.248).

So the factors are **7.2× at one thread** and **8.8× at ten**, not the 29× D §10.2 claimed, and the
reference's best use of this machine for one pair is 1.70 s, not the 3.40 s the note called its
best. The cause is the one V3 identified: with OpenMP free, ten Python threads each open up to ten
OpenMP threads on ten cores and the reference runs slower than at one thread. Pinning
`OMP_NUM_THREADS=1` is what the reference's own pipeline does for its workers
(`pipeline.py::_worker_env`) and what `_map` already does for scipy.

Within-pair scaling from one thread to ten: **5.3× for the port, 4.4× for the reference** — not
1.3×. The reference's pipeline still takes most of its parallelism from worker processes over pairs;
that is a choice about memory and the GIL, not a scaling limit inside the pair, and D §10.2 now says
so.

---

## 6. D2 and D13 — the bounded query, and a claim I had to withdraw

D2 asked for the reference's semantics "and then keep any bounded fast path only if results are
provably identical". `PointTree::nearest_below(q, bound)` is that: the unbounded nearest followed by
the reference's own `d < bound`, computed through a bounded traversal, with the proof at the
function rather than a comment at each call site claiming it.

* **pruning cannot change the winner.** A node is dropped only when its box is further than the
  search radius, and no such node can hold a point at a minimum that is itself under the radius, so
  every candidate at the minimum — ties included — is visited either way.
* **rounding cannot change it either, once the square is widened.** `sqrt(x) < bound` implies
  `x < bound²` exactly, because `sqrt` is correctly rounded and monotone; and
  `(bound·bound)·(1 + 4ε)` is above `bound²` for every finite `bound`. So nothing the strict test
  would accept is ever outside the searched ball, and whatever the widening lets in beyond `bound`
  the strict test drops.

**D13, and the correction.** Writing the first version of that test found that
`nearest_within(q, r)` returns `None` when `d` is exactly `r`: it tests `d² ≤ fl(r·r)`, the product
rounds down, and `7² + 11² + 13² = 339` with `√339` squared rounding back to 338.99999999999994 is a
two-line demonstration. Its documentation claimed "the radius test here is inclusive (`d ≤ r`)",
which is false, and that is a real defect — in the contract.

**It is not a defect in the answers, and my first draft of this note said it was.** The claim was
that the port could drop a neighbour the reference keeps. Checked instead of asserted, it cannot:
`d² > fl(bound·bound)` forces `fl(sqrt(d²)) ≥ bound`, because the window between `fl(bound·bound)`
and `bound²` is at most `2⁻⁵³·bound²` wide, which after the square root is under half an ULP of
`bound` — so nothing can land in it and still test below `bound`. Searched over **2.4 M random
`(bound, d²)` pairs**, walking `d²` up from `fl(bound·bound)`: no counterexample. All three call
sites use a strict `<`, so the old form and the reference agreed all along, which is also why the
parity sweep is identical before and after the change.

So the change stands on a different ground than the one I first gave it. The old argument is true
and *fragile*: it holds only because every call site happens to use `<` rather than `≤`, it is
invisible at the call site, and it would have to be re-derived by anyone adding a fourth caller.
`nearest_below` needs no argument at all — it is the reference's two steps, in one call, and the
widening makes the traversal a hint rather than a semantic.

R §5.2's coarse probe and R §6.2's and R §6.3's queries use it. R §7's correspondence search does
not need it: it never leaves squared units, so its bound and its test are the same `f64` and
Open3D's strict `d² < r²` is reproduced exactly.

The new sweep test is on a cloud built to tie — 300 random points and their mirror images, queried
on the plane `x = 0`, so most queries are equidistant from two points — at five bounds each,
including the distance itself and the double above it: 2 000 assertions, most of them on a tie,
which is the case none of the fixtures produces.

---

## 7. The five native breakline failures, and the calibration that closes them

T1 open issue 1. The five are `p99 distance` on `frag_014` (0.888 t) and `frag_019` (0.901 t)
against `0.5 t`, and `dihedral KS` on `frag_010` (0.0636), `frag_014` (0.0655) and `frag_017`
(0.0512) against `0.05`. `t` is bit-identical on all five; what differs is `res`, +2.7 % to +8.3 %,
inside the working-mesh row's own ±10 %.

The team accepted that a decimator other than Open3D's cannot meet a `0.5 t` p99 row, and asked for
both rows to be calibrated to the reference's own measured sensitivity to PMC-2 plus a stated
margin. T1 had six data points for that; this is 99.

### 7.1 The experiment

The reference's own pipeline, twice per fragment, with **nothing changed but the face budget**, and
the two breaklines compared at the same `t` (R §3.2's estimator runs on the *original* mesh, so `t`
does not move when the budget does) with the harness's own two statistics: `percentile99`'s worse
directed 99th percentile, and the two-sample KS of the dihedral distributions.

The budgets are **relative** — 0.87, 0.825 and 0.75 of each fragment's own working-mesh face count
at the default — and that matters. T1 lowered `target_faces` from 200 000 to 174 000, and R §3.3's
budget is `clip(150 · area / t², MIN_FACES, target_faces)`: on every terracotta sherd and half of
pot B the adaptive term already sits below 174 000, so that change decimates nothing and measures
nothing. A first pass here made the same mistake and produced Δres = 0 on all four terracotta
fragments and four of twenty synthetic ones.

Three collections, 33 fragments, 99 comparisons, `res` gaps 3.8 % to 40.3 %:

| `res` gap | n | p99 distance p50 / p90 / max | dihedral KS p50 / p90 / max |
|---|---:|---|---|
| **≤ 10 %** — the working-mesh row's own allowance | 46 | 0.210 / 0.442 / **1.160 t** | 0.0213 / 0.0331 / **0.0431** |
| all measured | 99 | 0.262 / 1.884 / 8.081 t | 0.0244 / 0.0860 / 0.1691 |

The worst inside the window is `frag_011` at Δres +9.40 % (1.160 t) and `frag_018` at Δres +4.71 %
(KS 0.0431). Two collections reach into that window and they agree about the size of the effect:

| collection | comparisons ≤ 10 % | `res` gaps | p99 max | KS max |
|---|---:|---|---:|---:|
| synthetic_20 | 39 (20 fragments) | 3.8–9.8 % | 1.160 t | 0.0431 |
| terracotta | 7 (4 fragments) | 7.0–10.0 % | 0.280 t | 0.0397 |

pot B contributes nothing to the window: its sherds are small enough that the mildest perturbation
already moves `res` by 13 %. Its rows are in the second line above, and they are the reason the
second line matters — at a 30 % `res` gap the reference's own two breaklines are **8.1 t** apart
with a KS of **0.169**, so both statistics grow steeply with the gap and a threshold below the
row's own ±10 % is measuring the working mesh.

The T1 measurement is reproduced exactly where the two overlap: `frag_010` at 174 000 gives
Δres +5.46 %, p99 0.201 t, KS 0.0151 — T1's own three digits.

### 7.2 The gates

**`NATIVE_P99_T = 2.3` and `NATIVE_KS = 0.086`: twice the measured worst inside the window.**

The factor of two is a stated choice, not a measurement, and it is stated for two reasons that
point the same way. The experiment varies the **budget** of one decimator; PMC-2 licenses a
**different** decimator, which redistributes the same budget differently even at an identical
`res`, so the measured sensitivity is a lower bound on what the row has to tolerate. And 46
comparisons under-estimate a maximum.

A reader is entitled to ask whether the factor was chosen so that the port passes. It was not, and
the way to check is the port's own numbers against it:

| row | reference's own worst (≤ 10 % gap) | gate | port's worst | share of gate |
|---|---:|---:|---:|---:|
| `p99 distance` | 1.160 t | **2.3 t** | 0.901 t | **39 %** |
| `dihedral KS` | 0.0431 | **0.086** | 0.0655 | **76 %** |

Those two numbers, not the gates, are what a regression should be read against, and D §10.2 says so.

**The KS row has the least headroom of any native row, and the decimator is not the whole reason.**
The reference's own KS moves at most 0.0431 at these gaps and the port's moves 0.0655, so `res`
does not explain it; T1's explanation does, and it still holds. `frag_010`, `frag_014` and
`frag_017` carry three of the four lowest segmentation agreements of their set (0.9885, 0.9814,
0.9868), and a breakline is the **boundary** of the mask that agreement measures — a 1.9 % area
disagreement is a far larger fraction of a boundary than of a surface. Both rows are inside their
own gates and the pair of them is what to watch, which is the same non-independence D §10.2 already
describes for `Pf spacing`.

Two things the calibration deliberately does **not** do: it does not touch the `curve length` row
(worst 3.95 % of a 10 % allowance, unchanged), and it does not touch either injected column, which
is exact and stays exact.

---

## 8. The documentation corrections

**D6.** R's header said the reference and the fixtures were `09fb4d4`. `git diff 09fb4d4 895a948 --
sherd_refit/` is step C1's hoist of the two NMS walk orders, so both were one commit stale. The
header now names `895a948` for the reference and for the seven `output/fixtures` sets, and `9cbcbbc`
for the committed slab, and says that `git diff 9cbcbbc 895a948 -- sherd_refit/` is empty — which is
what D §10.1 already recorded and what all eight `manifest.json` say.

**D7.** `stages/pairs.rs` summed the reference's own face areas over the reference's own fracture
mask left to right, where `fragment.py:423` is `float(self.A[self.frac].sum())` — numpy's pairwise
summation, which `segment::masked_area` already reproduces for the port's side. The harness calls
`masked_area` now. Nothing D §10.2 gates moves; what moves is the injected `verify` row's claim that
not one threshold and not one sample is the port's, which is now true of the area too.

**D10.** D §10.2's `load` row gated the counts and nothing else, while the harness has always
compared every vertex of the largest component at one `f32` ULP. That column measures **exactly
1.000 ULP** on pot_A and pot_B (0.50 pot_C, 0.25 pot_G and pot_H, 0.000 on every PLY set), so the
row sits on its own limit by construction: Open3D reads OBJ through Assimp, whose `fast_atof`
accumulates decimal digits itself instead of calling a correctly rounded `strtod` and lands one ULP
low. D §10.2 now carries the row and the explanation, so that the next OBJ set failing it is read
as a parser change rather than as a port regression.

**D12.** `NATIVE_ROTATION_DEG` cited "measured worst median 0.28°" and `NATIVE_MOVE_T` "0.267 t";
the C3 note's own table says **0.417°** and **0.108 t**, both on pot_B, and this session measures
the same. Both alarms are far inside their 1.0° / 0.3 t limits either way, and both citations are
now the measured ones.

---

## 9. Gates

| gate | result |
|---|---|
| `cargo build --release --workspace --locked` | **ok** |
| `cargo fmt --all --check` | **ok** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **ok**, no warning |
| `cargo test --workspace --locked` | **ok**, 266 passed, 1 ignored (259 before) |
| `pytest -q` (the reference's own suite, untouched by this task) | **ok**, 58 passed in 103 s |
| parity, 13 stages × 8 sets, injected | **20 618 comparisons, 0 failures**, 250 skipped |
| parity, 13 stages × 8 sets, native | **2 644 comparisons, 0 failures**, 1 813 skipped |
| two `segment` runs byte-identical | **ok**, `cmp` clean on all four terracotta caches |
| `--threads 1` against the default, `segment` | **ok**, byte-identical caches |
| `match_pair` at one and at ten threads | **ok**, the same candidate list on all six terracotta pairs |
| two `parity --details` runs byte-identical | **ok** (slab, all thirteen stages) |
| the whole 16-file sweep re-run and compared | **ok**, identical once the log timestamps are stripped |
| `parity --details` at `RAYON_NUM_THREADS` 1 and 10 | **ok** (terracotta, native `candidates` and `breakline`) |

### 9.1 The sweep, set by set

| set | inj checks | inj fail | inj skip | nat checks | nat fail | nat skip |
|---|---:|---:|---:|---:|---:|---:|
| terracotta | 596 | 0 | 0 | 116 | **0** | 30 |
| pot_A | 2 005 | 0 | 0 | 274 | **0** | 140 |
| pot_B | 2 489 | 0 | 0 | 321 | **0** | 180 |
| pot_C | 1 576 | 0 | 0 | 230 | **0** | 105 |
| pot_G | 1 572 | 0 | 0 | 228 | **0** | 105 |
| pot_H | 3 610 | 0 | 0 | 424 | **0** | 275 |
| synthetic_20 | 8 553 | 0 | 250 | 996 | **0** | 973 |
| slab | 217 | 0 | 0 | 55 | **0** | 5 |
| **total** | **20 618** | **0** | 250 | **2 644** | **0** | 1 813 |

Every figure except the five that were failing reproduces V3's §3 table digit for digit, and that is
the point: the code changes of X1–X4 moved **nothing** the fixtures can see. The sweep was run twice
— once after X1–X4 with the old gates, which failed exactly the same five comparisons at exactly
the same values (0.063644396, 6.465510215, 0.065521982, 0.051237283, 7.302725871), and once after
the calibration, which passes them. The skips are unchanged and are what V3 described: 250 injected
on synthetic_20's `slim` level, and 1 790 of the 1 813 native are the five pair stages that have no
native column at all.

The native breakline row's worst, per set, as a share of its gate after the calibration: pot_B 0.02,
slab 0.02, pot_A 0.03, pot_C 0.04, pot_H 0.05, pot_G 0.07, terracotta 0.45, synthetic_20 0.76.

## 10. New tests (+7; 259 → 266 passing, 1 ignored)

* **`matching/nms.rs`** — `topk = 0` keeps one pose, none below the floor, none on an empty order:
  R §5.3's loop shape, checked against the reference's own `nms` first.
* **`matching/pair.rs`** — `argmax` on a tied list, an all-tied list, a single element and an
  all-zero list: numpy's first-maximum rule, and the all-zero case is the one a `>` scan seeded from
  zero would get wrong.
* **`matching/icp.rs`** — `eigen_pivots` on distinct, descending, tied and all-equal diagonals, with
  `[3, 3, 5]` written out because it is the case a stable sort gets wrong; and the quaternion
  composition against R §7's matrix product over eleven decades of angle, asserting both the ULP
  bound and the 1e-12 t displacement bound (it prints its own measurement under `--nocapture`).
* **`matching/verify.rs`** — `pose_inverse` against `np.linalg.inv`'s own answer on a real stage-2
  pose, at 4 ULP per entry.
* **`spatial/kdtree.rs`** — the bounded search against the unbounded one on a cloud built to tie,
  400 queries × 5 bounds including the distance itself; and the squared-radius boundary written out
  on `(7, 11, 13)`, where `√339` squared rounds below 339.

## 11. Commits

| commit | what |
|---|---|
| `a2873ad` | X1 — R §5.4's arg-max, R §5.3's stop rule, the order of R §5.4's floor branch (D3, D4, D5) |
| `5997172` | X2 — Eigen's LDLT transpositions (D9) |
| `1721254` | X3 — R §6's inverse and its nearest-neighbour queries (D1, D2, D13) |
| `02442f6` | X4 — the harness's pairwise `frac_area` (D7) |
| `0fce0d3` | X5 — PMC-18, PMC-19, and R's header (D8, D6) |
| `76e5d14` | X6 — the cost figure re-measured, two constants' citations, the `load` row (D11, D12, D10) |
| `9263c41` | X7 — the two native breakline rows calibrated to PMC-2 (T1 open issue 1) |
| `b8b0bdf` | X8 — this note |
| `eaad990` | X9 — D13's divergence claim withdrawn: the contract was wrong, the answers were not |

## 12. What this task did not do

* **The `curve length` row is untouched** and so is every injected column. The calibration is two
  constants of the native breakline row and nothing else.
* **PMC-19's tail is not reducible.** 1.2e-11 t is LAPACK against a written-out loop on a pose with
  `cond(T) = 3.3e10`; matching it would mean reproducing OpenBLAS's kernel selection, which is not a
  property of the reference so much as of the machine it ran on.
* **The `chaotic` rows still say what they said.** PMC-18 is an ULP, and on 48 of 74 090 candidates
  an ULP is not bounded; the row that gates their share and requires zero of them to have been kept
  by R §5.5 is what makes the pose rows sayable, and it is unchanged.
* **`pot_B` never entered the calibration window.** Its sherds are small enough that the mildest
  budget perturbation moves `res` by 13 %, so the 46 comparisons behind the two constants are
  terracotta's and synthetic_20's. A collection of small sherds where `res` can be perturbed gently
  would be the next thing to add.
