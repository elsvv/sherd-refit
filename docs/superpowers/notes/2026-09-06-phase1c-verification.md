# Phase 1c — independent verification (task V3)

**Date:** 2026-09-07. **Tree:** branch `rust-core` at `953128d` ("C3: R §6's verification, the
accept rule and `match_pair`, exact against the reference"), working tree clean.
**Machine:** Apple M2 Pro, 10 cores, 16 GB, macOS 15.6.1. **Toolchain:** cargo/rustc 1.97.0 as
`rust-toolchain.toml` pins them, release profile after `cargo clean`.
**Reference:** `sherd_refit/` at `953128d`, byte-identical to the fixture commit `895a948`
(`git diff 895a948 HEAD -- sherd_refit/` is empty), on Python 3.12.9, numpy 2.5.2, Open3D 0.19.0,
scipy 1.18.1.

This note re-derives the claims of the T1, C1, C2 and C3 notes instead of repeating them: every
figure below was measured in this session, and the places where a measurement disagrees with an
existing note are called out rather than smoothed over. Scope is task V3's six items — the diff
review of T1/C1/C2/C3 against R and the Python, the toolchain gates, the full parity sweep, the
determinism of `match_pair`, its cost against the reference, and the reference's own test suite.

**Verdict in one line: the port is correct where the fixtures can see it, and the sweep is not
clean.** 20 618 injected comparisons fail none; 2 644 native comparisons fail five, and those five
are the phase-1b decimator failures on `synthetic_20`, unchanged in fragment, row and value.
Nothing phase 1c added fails anything. Twelve defects are listed in §8, all of them small; two
(D1, D2) are undeclared deviations that R §12 has to grow a row for, one (D3) is a live tie-break
bug on an off-by-default path, and one (D11) is a cost figure in D §10.2 that this session could
not reproduce and that overstates the port's advantage by a factor of three.

---

## 1. What was checked, and against what

The diff review reads the port's phase-1c modules against R and against `sherd_refit/*.py` at the
fixture commit, in the order R §5–§6 states them:

| area | port | reference |
|---|---|---|
| frames, dihedral filter, pose | `crates/sherd-core/src/matching/hypotheses.rs` | `matching.py:175–200` (`hypotheses`) |
| coarse subset and score | `matching/coarse.rs`, `rng.rs` | `matching.py:202–223` (`coarse_score`) |
| suppression and its ties | `matching/nms.rs`, `matching/ladder.rs` | `matching.py:285–308` (`nms`) |
| ICP, update, convergence, radius | `matching/icp.rs` | Open3D 0.19 `registration_icp` via `matching.py:311–315` |
| the two ladders | `matching/ladder.rs` | `matching.py:528–533` (`stage1`), `matching.py:450–485` (`_stage2`) |
| verification and the accept rule | `matching/verify.rs` | `matching.py:328–427` |
| ranking, cut, threading | `matching/pair.rs` | `matching.py:500–578` (`_match_pair`) |
| thickness (task T1) | `fragment/thickness.rs` | `fragment.py:47–105` (`_hist_mode`, `estimate_thickness`) |

Everything that follows from that reading is in §8. §§2–7 are the measurements.

---

## 2. Toolchain gates

`cargo clean`, then, in this order:

| gate | command | result |
|---|---|---|
| release build | `cargo build --release --workspace --locked` | ok, 37.5 s from cold |
| formatting | `cargo fmt --all --check` | ok |
| lints | `cargo clippy --workspace --all-targets --locked -- -D warnings` | ok, no warning |
| tests | `cargo test --workspace --locked` | **259 passed, 0 failed, 1 ignored** |

The test counts by target: `sherd-core` 186 unit + 4 `fragment_cache` + 3 `io_open3d_parity` +
4 `ply_writer_bytes` + 3 `slab_pair`; `sherd-parity` 37 unit + 3 `slab_fixture` +
7 `stages_slab` + 5 `working_mesh_slab`; `sherd-cli` 5 unit + 2 `segment_cli` (1 ignored).

---

## 3. Parity — all thirteen stages, eight fixture sets, both modes

`sherd-refit-rs parity --fixtures DIR --input DIR --stage all [--injected]` on the seven
`output/fixtures` sets and on the committed `fixtures/slab/dump`. Every dump reports
`dirty: false`; the seven come from `895a948` and the slab from `9cbcbbc`, which is what D §10.1
records (and not what R's header says — defect D6).

### 3.1 Totals

| mode | comparisons | failed | skipped |
|---|---:|---:|---:|
| injected | **20 618** | **0** | 250 |
| native | **2 644** | **5** | 1 813 |

### 3.2 Per stage, summed over the eight sets

| stage | inj checks | inj fail | inj worst/tol | nat checks | nat fail | nat worst/tol |
|---|---:|---:|---:|---:|---:|---:|
| load | 368 | 0 | 1.00 (pot_A, pot_B) | 368 | 0 | 1.00 (pot_A, pot_B) |
| thickness | 192 | 0 | 0.00 | 136 | 0 | 4.4e-3 |
| working mesh | 567 | 0 | 1.7e-3 | 272 | 0 | 0.94 |
| segmentation | 884 | 0 | 1.0e-3 | 136 | 0 | 0.62 |
| breakline | 748 | 0 | 0.07 | 204 | **5** | **1.80** |
| samples | 1 088 | 0 | 0.35 | 408 | 0 | 0.57 |
| hypotheses | 2 148 | 0 | 0.03 | 1 074 | 0 | 0.25 |
| coarse | 1 790 | 0 | 0.00 | — | — | (no native column) |
| nms | 2 148 | 0 | 0.87 | — | — | (no native column) |
| stage 1 | 3 654 | 0 | 0.69 | — | — | (no native column) |
| stage 2 | 3 400 | 0 | 0.75 | — | — | (no native column) |
| verify | 2 508 | 0 | 0.54 | — | — | (the `candidates` row) |
| candidates | 1 123 | 0 | 0.00 | 46 | 0 | 0.71 |

The first six rows reproduce D §10.2's recorded totals **exactly**: 368 + 192 + 567 + 884 + 748 +
1 088 = **3 847** injected and 368 + 136 + 272 + 136 + 204 + 408 = **1 524** native with five
failures, which is the sentence D §10.2 already carries. The seven pair rows add 16 771 injected
and 1 120 native. The pair-stage check counts also match what C1, C2 and C3 claim, one for one:
2 148 + 1 790 + 2 148 = 6 086 (C1), 3 654 and 3 400 (C2), 2 508 (C3).

The dumps themselves were counted rather than taken on trust: **358 pairs, 71 841 stage-1
candidates, 2 249 stage-2 candidates** across the eight sets, which is what C2 and C3 state.

### 3.3 Per set

| set | inj checks | inj fail | inj skip | nat checks | nat fail | nat skip |
|---|---:|---:|---:|---:|---:|---:|
| terracotta | 596 | 0 | 0 | 116 | 0 | 30 |
| pot_A | 2 005 | 0 | 0 | 274 | 0 | 140 |
| pot_B | 2 489 | 0 | 0 | 321 | 0 | 180 |
| pot_C | 1 576 | 0 | 0 | 230 | 0 | 105 |
| pot_G | 1 572 | 0 | 0 | 228 | 0 | 105 |
| pot_H | 3 610 | 0 | 0 | 424 | 0 | 275 |
| synthetic_20 | 8 553 | 0 | 250 | 996 | **5** | 973 |
| slab | 217 | 0 | 0 | 55 | 0 | 5 |

`synthetic_20`'s 250 injected skips are the `slim` level: no `load.V0`, so the whole injected
thickness stage and the array half of `load` skip on all twenty fragments — finding F3 of the
phase-1a verification, working as designed. The native skips are almost all the five pair stages
that have no native column at all — `coarse`, `nms`, `stage1`, `stage2`, `verify`, 358 pairs × 5 =
**1 790** of the 1 813; the other 23 are `synthetic_20`'s twenty missing `load.V0` and the three
pairs whose R §5.5 kept nothing, so there is no `result.candidates.json` to compare against.

### 3.4 The pair stages, set by set (`checks/failed worst-tol`)

| set | hypotheses | coarse | nms | stage 1 | stage 2 | verify | candidates |
|---|---|---|---|---|---|---|---|
| terracotta | 36/0 0.03 | 30/0 0.00 | 36/0 0.67 | 73/0 0.69 | 87/0 1.1e-4 | 78/0 3.1e-5 | 24/0 0.00 |
| pot_A | 168/0 0.03 | 140/0 0.00 | 168/0 0.81 | 293/0 5.9e-5 | 373/0 0.50 | 280/0 0.54 | 112/0 0.00 |
| pot_B | 216/0 0.03 | 180/0 0.00 | 216/0 0.76 | 373/0 1.1e-4 | 477/0 0.75 | 352/0 0.03 | 144/0 0.00 |
| pot_C | 126/0 0.03 | 105/0 0.00 | 126/0 0.53 | 223/0 0.10 | 282/0 0.25 | 217/0 0.10 | 84/0 0.00 |
| pot_G | 126/0 0.03 | 105/0 0.00 | 126/0 0.37 | 223/0 7.2e-5 | 282/0 0.54 | 213/0 2.4e-5 | 84/0 0.00 |
| pot_H | 330/0 0.03 | 275/0 0.00 | 330/0 0.87 | 563/0 6.8e-5 | 724/0 0.50 | 519/0 0.10 | 220/0 0.00 |
| synthetic_20 | 1140/0 0.03 | 950/0 0.00 | 1140/0 0.74 | 1883/0 0.56 | 1153/0 0.50 | 816/0 0.10 | 451/0 0.00 |
| slab | 6/0 0.03 | 5/0 0.00 | 6/0 0.60 | 23/0 2.4e-5 | 22/0 7.9e-9 | 33/0 6.2e-14 | 4/0 0.00 |

`coarse` at 0.00 is the `cs exact` row: the port's score equals the reference's **bit for bit** on
every hypothesis of every pair, and the harness counts bit differences rather than measuring a
tolerance (`stages/coarse.rs:102–107`). `candidates` at 0.00 injected is the same shape: the
ranking, the cut, the `accepted` flags and `brk_best` are compared as identities. The `nms` row's
0.37–0.87 is **not** a parity number — those are the four PMC-6 tie measurements, which are alarms
on the size of the tie effect and are gated at their measured worst plus headroom; the parity row
in that stage (the kept list on the reference's own `nms1.order`) is exact and passes everywhere.

### 3.5 The five native failures, and why they are not phase 1c's

```
frag_010   dihedral KS    0.063644396   tol 5.000e-2   FAIL
frag_014   p99 distance   6.465510215   tol 3.640e0    FAIL      (= 0.888 t)
frag_014   dihedral KS    0.065521982   tol 5.000e-2   FAIL
frag_017   dihedral KS    0.051237283   tol 5.000e-2   FAIL
frag_019   p99 distance   7.302725871   tol 4.051e0    FAIL      (= 0.901 t)
```

Same five fragments, same two rows, same values as D §10.2 records after task T1 — 0.888 t and
0.901 t against 0.5 t, and KS 0.0512–0.0655 against 0.05. D §10.2 already explains them with a
measurement rather than an argument: running the *reference against itself* with only the face
budget changed (200 000 against 174 000, a `res` gap inside the ±10 % the working-mesh row allows)
puts its own two breaklines 0.866 t and 1.071 t apart at the 99th percentile on those same
fragments — **further apart than the port is from it**. The cause is `meshopt` against Open3D's
quadric decimator (PMC-2), it predates phase 1c by two steps, and none of the five is a `t`
difference: the native thickness row is 4.4e-3 of its tolerance at worst on this set.

I did not re-run that reference-against-itself experiment; what this session establishes is that
the failures are **unchanged**, fragment for fragment and digit for digit, so nothing C1, C2 or C3
did moved them.

---

## 4. Determinism

The port's whole per-pair matching was run twice per collection and compared **byte for byte** on
the candidate list: for every pair, every returned candidate's `accepted` flag, all sixteen
entries of its pose and all twenty of its scores, printed as raw IEEE-754 bit patterns. (The probe
was a throwaway `sherd-parity` example built against the workspace's own `Cargo.lock`; it is not
part of the tree.)

| collection | pairs | candidate lines | run 1 vs run 2 | 1 thread vs 10 threads |
|---|---:|---:|---|---|
| terracotta | 6 | 30 | identical (sha256 `a50a9875…`) | identical |
| pot_H | 55 | 275 | identical (sha256 `294771e6…`) | identical |

The second column is the stronger statement and it was not asked for: `RAYON_NUM_THREADS=1` and the
default ten-thread pool produce the **same bits**, so the candidate list, the ranking and the
accepted set are functions of the input alone and not of the schedule (D §7). That is what
`pair.rs`'s two `par_iter().collect()` are for — the results are collected by index, and the two
suppressions, the sort and `brk_best` all run on the collected vector.

The reference makes the same promise and keeps it at the level it can be checked at: its accepted
set on pot_H is identical at one and at ten threads (16 pairs either way).

---

## 5. Cost of `match_pair`, port against reference

Wall clock for `match_pair` alone, preprocessing excluded, means over the collection's pairs, this
machine. The reference is driven through `matching.match_pair(A, B, p, n_threads=N)` on
`MatchData` built at the pair's own `t`; the port through `matching::pair::match_pair`.

| collection | pairs | ref 1 thread | port 1 thread | ratio | ref 10 threads | port 10 threads | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| terracotta | 6 | **8.115 s** | **1.058 s** | 7.7× | **1.820 s** | **0.199 s** | 9.2× |
| pot_H | 55 | **7.316 s** | **1.196 s** | 6.1× | **1.853 s** | **0.225 s** | 8.2× |

Both sides run with `OMP_NUM_THREADS=1`, which is the configuration the fixtures were dumped under
and the only one in which Open3D's OpenMP does not multiply with the Python thread pool. Per-pair
spread on pot_H: reference 2.77–19.00 s at one thread and 0.80–4.02 s at ten; port 0.354–2.758 s
and 0.070–0.493 s. Totals for the 55 pairs: 402.4 s / 65.8 s / 101.9 s / 12.4 s.

**D §10.2's ten-thread row could not be reproduced, and the reason is the OpenMP setting** (defect
D11). D §10.2 and the C3 note record "5.83 s for the reference with ten threads inside the pair
against the port's 0.198 s, a factor of 29" and "the reference's best use of this machine for one
pair is 3.40 s (its own `_map` at one thread with Open3D's OpenMP free), still 17× the port's
ten-thread number". Measured here:

| reference configuration | terracotta mean |
|---|---:|
| `OMP_NUM_THREADS=1`, `n_threads=1` | 8.115 s (note: 7.62 s) |
| OpenMP free, `n_threads=1` | 3.735 s (note: 3.40 s) |
| OpenMP free, `n_threads=10` | 6.947 s (note: 5.83 s) |
| **`OMP_NUM_THREADS=1`, `n_threads=10`** | **1.820 s, 2.033 s on a repeat — no row in the note** |

Three of the note's four numbers reproduce to within 7 %. The fourth is the one the "factor of 29"
rests on: the note's ten-thread reference was measured with OpenMP *left free*, so ten Python
threads each opened up to ten OpenMP threads on ten cores and the run was slower than one thread's.
Pinning OpenMP to one thread — which is exactly what the pipeline's own `_worker_env` does for its
worker processes, and what `_map` already does for scipy — takes the reference to 1.82 s. The
port's advantage at ten threads is therefore **9.2× on the terracotta, not 29×**, and the
reference's within-pair scaling from one thread to ten is **4.5×, not 1.3×**. The single-thread
row (7.3×–7.7×) is unaffected and is the one R §13's "≈ 7 core-s per pair" speaks to.

---

## 6. The reference's own suite, and the terracotta joins

`python -m pytest -q`: **58 passed in 88.8 s**, no skip. That includes
`tests/test_integration_real.py::test_real_fragments_assemble_as_the_museum_did`, which is the
whole pipeline on `input/test_fragments_1` and asserts the join set.

Run separately for the numbers R §13 states:

```
groups: [['FY234021_reduced', 'FY234094_reduced', 'FY234104_reduced'], ['FY234007_reduced']]
join 021 - 094   seam=20.3333  tight=0.5307  gap=0.00899  pen=0.00000  cont_n=0.9928  score=10.7912
join 094 - 104   seam=10.6667  tight=0.5681  gap=0.00685  pen=0.00000  cont_n=0.9980  score=6.0600
```

Exactly R §13 after task T1: the two joins and no others, 007 unplaced, both penetrations zero,
seam 20.33 t and 10.67 t, score 10.79, tight 0.531 — every digit the algorithm-reference prints.
**Unchanged.**

The port agrees at the pair level on the same collection: run natively over all six pairs it
accepts a candidate on `021__094` and `094__104` and on no other pair, which is R §13's join set
found by a search that shares neither the sample nor the suppression tie-break with the reference.
On pot_H the port accepts 13 pairs against the reference's 16 (8 shared, 5 only the port's, 8 only
the reference's) — the C3 note's table for that set, reproduced independently.

---

## 7. Native `candidates` — the regression alarms, measured

D §10.2's native `pair result` row is an alarm, not a parity claim, and this is what it reads:

| set | `n_returned differs` | `accepted only ours` | `accepted only theirs` | `different placement` | `best rot p50` | `best move p50` |
|---|---:|---:|---:|---:|---:|---:|
| terracotta | 0.000 | 0.000 | 0.000 | 0.000 | 0.108° | 0.020 t |
| pot_A | 0.000 | **0.179** | 0.143 | 0.036 | 0.278° | 0.060 t |
| pot_B | 0.000 | 0.167 | 0.028 | 0.083 | **0.417°** | **0.108 t** |
| pot_C | 0.000 | 0.048 | 0.095 | 0.000 | 0.033° | 0.007 t |
| pot_G | 0.000 | 0.095 | 0.000 | 0.000 | — | — |
| pot_H | 0.000 | 0.091 | 0.145 | 0.036 | 0.183° | 0.035 t |
| synthetic_20 | **0.241** | 0.016 | 0.016 | 0.000 | 0.037° | 0.010 t |
| slab | 0.000 | 0.000 | 0.000 | 0.000 | 0.096° | 0.010 t |
| **alarm** | 0.4 | 0.25 | 0.25 | 0.25 | 1.0° | 0.3 t |

Every row is inside its alarm on every set, and the shares recover the C3 note's acceptance table
exactly: multiplied out, the port accepts 2/17/26/3/2/13/23 pairs against the reference's
2/16/21/4/0/16/23 on terracotta / pot_A / pot_B / pot_C / pot_G / pot_H / synthetic_20, and six of
the 354 pairs both sides accept are placed more than a wall apart (1 on pot_A, 3 on pot_B, 2 on
pot_H). pot_G has no `best rot`/`best move` because the reference accepts nothing there, so no pair
is accepted by both.

Two of the alarms sit at their own measured worst — `accepted only ours` 0.179 on pot_A and
`n_returned differs` 0.241 on `synthetic_20` are the numbers the constants in
`stages/candidates.rs` cite — so those rows have headroom by design and not by margin (D10 notes
the same pattern in `load`). Two others do **not** match the numbers their constants cite: see D12.

---

## 8. Defects

Ordered by how much they could cost, not by how likely they are. "R says" quotes the frozen
algorithm reference or the Python it follows; "code does" is the port or the harness.

### D1 — `T⁻¹` is a transpose where the reference factorises, and R §12 does not license it

`crates/sherd-core/src/matching/verify.rs:611` (`fn rigid_inverse`).
**R says:** R §6.1 `d2 = distance from apply(T⁻¹, A.Pf)`, R §6.4 `sdB = signed distance of
apply(T⁻¹, A.S)`; the reference is `np.linalg.inv(T)` — an LU factorisation of the full 4×4 —
at `sherd_refit/matching.py:343` and `:388`.
**Code does:** `[Rᵀ | −Rᵀτ]`, no factorisation.
The two agree only to the extent `R` is orthonormal, which after thirty ICP iterations is about
1e-16; the C3 note §2 measures the consequence at "1e-14 on a point a hundred units from the
origin" and calls it deliberate. The defect is not the substitution, it is where it is written
down. R §12.1's own T1 addendum sets the rule: *"R §12 is the port's licence and anything the port
does differently that is not on it is an undeclared deviation, however small its measured effect"*
— and PMC-16 and PMC-17 were promoted to rows for exactly this reason. `rigid_inverse` has no row
and no addendum. **Fix: a PMC row, not a code change.**

### D2 — the bounded KD-tree queries of R §6.2 and R §6.3 are undeclared

`crates/sherd-core/src/matching/verify.rs:390` (seam) and `:424` (continuity).
**R says:** R §6.2 "`(dA, jA) = nearest transformed B point (KD-tree over p, **no bound**)`" then
`dA < sc.seam`; R §6.3 "`(dm, jm) = nearest A margin point per p (**unbounded**)`" then
`dm < sc.near`. The reference is `cKDTree(...).query(...)` with no `distance_upper_bound`
(`matching.py:361`, `:374`).
**Code does:** `tree.nearest_within(point, sc.seam)` / `(…, sc.near)`, a bounded search, followed
by the same strict test.
The answers cannot differ — the nearest point inside the radius *is* the nearest point whenever
there is one, and when there is none the reference's threshold rejects it too — and both call
sites say so in a comment. PMC-12 licenses exactly this substitution but its text names only "the
fracture distances"; two more query sites use it. **Fix: widen PMC-12's row, or add one.**

### D3 — R §5.4's partial candidate picks the *last* tie, the reference picks the first

`crates/sherd-core/src/matching/pair.rs:250–253`.
**R says:** R §5.4, "return one partial candidate (**the arg-max pose**)"; the reference is
`k = int(np.argmax(s1))` (`matching.py:548`), and numpy's `argmax` returns the **first** index of
the maximum.
**Code does:** `stage1.iter().max_by(|x, y| x.score.partial_cmp(&y.score)…)`, and Rust's
`Iterator::max_by` documents that "if several elements are equally maximum, the **last** element is
returned".
`s1` is a mean of booleans over `|brk_sub|` points, so exact ties are the normal case, not a corner
one; two implementations would then return different poses for the same pair. It is not reachable
with the shipped parameters — `stage1_floor` defaults to 0.0 and the branch is dead, which is why
no fixture catches it — but it is a real divergence for anyone who raises `stage1_floor`, which
the reference's own CLI exposes as `--stage1-floor` (`sherd_refit/cli.py:35`).
**Fix:** `max_by` with an index tie-break to the lowest, or a manual first-max scan.

### D4 — `nms` disagrees with the reference at `topk = 0`

`crates/sherd-core/src/matching/nms.rs:66–68`.
**R says:** R §5.3's loop appends and *then* tests `if len(kept) ≥ topk: break`, so the reference
(`matching.py:285–308`) keeps **one** pose when `topk = 0`.
**Code does:** `if topk == 0 { return kept; }` — none.
Unreachable with `stage1 = 250` and `stage2 = 10`, and the port's behaviour is the sane one; it is
listed because it is a difference in the transcription of a loop R freezes, and because a
parameter sweep that sets either count to zero would see it.

### D5 — the working-mesh scenes are demanded before R §5.4's floor branch

`crates/sherd-core/src/matching/pair.rs:245`.
**R says:** R §5.4, `if stage1_floor > 0 and best1 < stage1_floor`: return the partial candidate.
Nothing in R §5.4 needs a BVH; the reference builds `frac_scene` lazily, on first use inside
R §6.1.
**Code does:** `let Some(surfaces) = self.surfaces() else { return Vec::new() };` **above** the
floor test, so a pair whose working mesh has no triangle returns `[]` where R returns the partial
candidate. Both the trigger (a mesh R §3.1 would have rejected) and the branch (off by default) are
unreachable on real data; the ordering is still wrong, and moving the call below the branch costs
nothing.

### D6 — R's header names the wrong fixture commits

`docs/superpowers/specs/2026-09-06-algorithm-reference.md:3–9`.
**R says:** "**The parity fixtures are regenerated from `09fb4d4`** — the committed
`fixtures/slab/dump` from `f0da041`".
**The dumps say:** `895a948` for all seven `output/fixtures` sets and `9cbcbbc` for the slab, which
is what D §10.1 records (`2026-09-06-rust-core-design.md:577–581`) and what step C1's commits
`f256074` and `895a948` did. The same paragraph's "**Reference implementation:** `sherd_refit/*.py`
at commit `09fb4d4`" is one commit behind for the same reason — `09fb4d4 → 895a948` touches
`sherd_refit/matching.py` (the `nms1.order` / `nms2.order` hoist). R §12.1's C1 addendum describes
that change, so nothing is *wrong*; the header simply was not brought with it, and a reader diffing
R against the Python is pointed at the wrong commit.

### D7 — the harness's reference-side `frac_area` is not summed the way the reference sums it

`crates/sherd-parity/src/stages/pairs.rs:350–356`.
**R says:** `fracture_area = float(self.A[self.frac].sum())` (`fragment.py:423`) — numpy's
**pairwise** summation, which the port's own `masked_area`
(`crates/sherd-core/src/fragment/segment.rs:635–638`) deliberately reproduces.
**Code does:** `geom.areas.iter().zip(&frac).filter(…).map(…).sum()`, a sequential left-to-right
sum.
The value feeds `contactA`, `contactB` and `contact` only, and D §10.2 gates none of the three, so
nothing measured moves; but the injected `verify` row's claim is "not one threshold and not one
sample is the port's", and this is one number that is. Using `masked_area` here would make the
sentence true.

### D8 — R §7's Euler composition is not the expression Open3D evaluates

`crates/sherd-core/src/matching/icp.rs:720` (`fn euler_zyx`).
**R says:** `U = [ Rz(x₂) · Ry(x₁) · Rx(x₀) | (x₃, x₄, x₅) ]`, written as three explicit matrices.
**Open3D does:** `utility::TransformationMatrixFromPoseVector` composes three `Eigen::AngleAxisd`
*as quaternions* and converts the product once. The two are the same rotation to about an ulp per
iteration, which is exactly the order of the residual the injected stage-1 and stage-2 rows carry
(0.000° and 1e-14–1e-11 t at the median).
The port follows R faithfully; R is the document that does not say what the reference computes.
R §7 already carries a paragraph on the exponential map being *not* equivalent — the quaternion
composition deserves a sentence in the same place.

### D9 — the LDLT pivot claim is stronger than Eigen guarantees

`crates/sherd-core/src/matching/icp.rs:638–654` (`fn solve_ldlt` and its doc).
**The doc says:** "the permutation is exactly a **stable descending sort of the original
diagonal**".
**Eigen does:** the first half is right — `ldlt_inplace` selects the pivot with
`mat.diagonal().tail(size-k).cwiseAbs().maxCoeff()` *before* the rank-k update touches entry `k`,
and it never touches a trailing diagonal, so the values compared are the original ones. The
tie-break is not: `maxCoeff` returns the first index **in the current permuted order**, and the
transpositions have already reordered the tail, so on an exact tie Eigen and the port can choose
different pivots. A tie between two diagonal entries of a real 6×6 normal-equations matrix is
measure-zero and nothing on the fixtures exercises it; the sentence should still say "on the
values, with a tie-break Eigen does not fix".

### D10 — the injected `load` coordinate row is a gate with no headroom

`crates/sherd-parity/src/stages/load.rs:38` (`COORDINATE_ULPS = 1.0`).
D §10.2's `load` row gates "counts after cleaning, largest component | exact | exact" and nothing
else; the harness adds a vertex-by-vertex comparison at one `f32` ULP, which is a much stronger
check and a good one. On the five SfS++ OBJ sets it measures **exactly 1.000 ULP** on pot_A and
pot_B (0.50 on pot_C, 0.25 on pot_G and pot_H) — the row sits on its own limit. The cause is
documented and is the reference's: Open3D reads OBJ through Assimp, whose `fast_atof` accumulates
decimal digits itself instead of calling a correctly rounded `strtod` and lands one ULP low. Every
PLY set is at 0.000. Nothing is wrong; what is missing is a line in D §10.2 saying the row exists
and that it is a boundary gate by construction, so that the next OBJ set failing it is read as a
parser change rather than as a port regression.

### D11 — D §10.2's ten-thread cost figure was measured with OpenMP oversubscribed

`docs/superpowers/specs/2026-09-06-rust-core-design.md` §10.2, last paragraph, and
`notes/2026-09-06-c3-verify.md` §6. See §5 above: "5.83 s for the reference with ten threads
inside the pair … a factor of 29" and "the reference's best use of this machine for one pair is
3.40 s … still 17× the port's ten-thread number". With `OMP_NUM_THREADS=1` and ten Python threads —
the configuration the pipeline itself uses for its workers (`pipeline.py::_worker_env`) — the
reference takes **1.82–2.03 s** per terracotta pair, so the port's factor at ten threads is **9.2×**
and the reference's within-pair scaling is **4.5×**, not 1.3×. The three other rows of the note's
cost paragraph reproduce to within 7 %. The port is still comfortably faster; the claim is the
thing that needs correcting.

### D12 — two `candidates` constants cite measured worsts that are not the measured worsts

`crates/sherd-parity/src/stages/candidates.rs:88–89` and `:93–94`.
**The doc comments say:** `NATIVE_ROTATION_DEG` — "measured worst median **0.28°** over the six
sets"; `NATIVE_MOVE_T` — "measured worst median **0.267 t**, on `pot_B`".
**Measured here:** 0.417° and 0.108 t, both on pot_B — which is what the C3 note's own table
(`notes/2026-09-06-c3-verify.md:270–271`) prints. The other two constants' citations (`0.179`,
`0.241`) are right. Both rows are far inside their 1.0° / 0.3 t alarms either way, so nothing
fails; the comments are simply stale against the note they point at, and a reader tuning these
alarms would tighten `NATIVE_MOVE_T` toward a number the tree does not produce.

### Not defects, checked and cleared

* **Frame construction and the dihedral filter.** `hypotheses.rs::pose` builds
  `R[i][k] = t_A[i]·(−t_B[k]) + ns_A[i]·ns_B[k] + f_A[i]·(−f_B[k])` and `τ = P_A − R·P_B` in
  R §5.1's own summation order, with `ns` alone unnegated; `build_with` walks `(ia, ib)` row-major,
  which is `np.where`'s order on a 2-D mask, and the test is `< dihedral_tol`, strict, on the sum.
  `pa`/`pb` are positions into the subsets in the order given, which is what makes PMC-4's
  substitution visible instead of silent.
* **The coarse subset and score.** The probe is `min(60, |brk_sub|)` drawn without replacement from
  `B.brk_sub`; the target is A's *full* breakline; the radius test is scipy's exclusive
  `distance_upper_bound` applied at the call site; the mean is a **division** by 60 and not a
  multiplication by `1/60` (`coarse.rs:121–125`), which is the ulp-level defect step C1 found. Fed
  the reference's own `coarse.idx` the scores are bit-identical on all **38 126 133** scored
  hypotheses of the eight dumps — counted here rather than taken from C1's "38.1 M".
* **ICP.** The correspondence radius is strict on the *squared* distance — Open3D 0.19's
  `KDTreeFlann::SearchHybrid` does a `knnSearch` and then cuts with `std::lower_bound(.., r²)`,
  which keeps only `d² < r²`, so R §7's C2 correction is right and `icp.rs:380` implements it.
  `fitness` divides by the source count and `inlier_rmse` by the correspondence count, from the
  squared distances the tree returns. The loop is Open3D's: transform first, measure, then
  `U ← update; T ← U·T; P ← U·P`, re-measure, break on `|Δfit| < 1e-6 ∧ |Δrmse| < 1e-6` **after**
  the update. Point-to-plane assembles `JᵀJ` over the lower triangle and mirrors it, which is
  identical to Eigen's full outer product because `a_k·a_l = a_l·a_k`; the solve reproduces
  Eigen's LDLT pseudo-inverse down to its `> numeric_limits<double>::min()` cut-off
  (`icp.rs:700`), which is the exact constant Eigen uses.
* **Score definitions and the accept rule.** All nine of R §6.1's numbers, R §6.2's voxel count on
  the *unmoved* A points in a world-origin grid, R §6.3's `> 20` near count and its two medians,
  R §6.4's two statistics read off a signed distance that is never materialised (PMC-11's
  "equivalent formulation"), and R §6.5's five conjuncts with `gap · t ≤ sc.gap` rather than
  `p.max_gap`. `2·sc.tight ≤ sc.facing` holds identically for every `(t, res)` — `max(0.02t, 0.3res)
  ≤ max(0.3t, 1.0res)` — so PMC-12's bounded distance window can never truncate a value `contact`
  needs. The `frac_scene` fallback to `F[:1]` is reproduced.
* **Threading order.** Both `par_iter().collect()` in `pair.rs` and the one in `coarse.rs` collect
  by index; nothing reduces across threads. §4's cross-thread bit-identity is the evidence.
* **Thickness (T1).** `stride = ceil(n_faces / 300 000)`, `arange(0, n_faces, stride)`, origin
  `C + d·1e-3`, the `> 0.7` filter evaluated on the `f64` normals and not on the `f32` ray
  directions, and a 60-bin histogram over `(0, p90]` reproduced down to numpy's two bin-edge
  corrections and its `argmax`-picks-the-first rule. `np.percentile` on an `f32` array returns
  `float32` (checked against numpy 2.5.2 directly), which is what `percentile90` assumes. Injected
  thickness is exact on all 48 fragments whose dump carries `load.V0`; native is 4.4e-3 of the
  ±2 % row at worst.
* **Parameters.** All 46 fields of `Params` match the reference's dataclass by name and by default
  value (checked programmatically, no difference).
* **Tolerances.** Every constant in `crates/sherd-parity/src/stages/*.rs` was read against
  D §10.2's table. None is wider than the row it implements; several rows are stricter than the
  table (`cs exact`, the injected `candidates` identities, `load`'s coordinates, breakline frames).

---

## 9. Verdict

* Build, fmt, clippy `-D warnings`, 259 tests: green.
* Injected parity: **20 618 comparisons, 0 failures**, on 13 stages × 8 sets. The pair stages that
  phase 1c added are exact where they can be — the coarse score bit for bit, the two suppressions
  on the reference's own walk orders, R §5.7's ranking and cut as identities, `tight` and `seam`
  bit for bit at the reference's own poses.
* Native parity: **2 644 comparisons, 5 failures** — the phase-1b `synthetic_20` breakline rows,
  unchanged in fragment, row and value, whose cause D §10.2 already measures on the reference
  itself. Nothing phase 1c added fails.
* `match_pair` is a function of its input: two runs and two thread counts give byte-identical
  candidate lists and scores on terracotta and pot_H.
* The reference's suite passes (58 tests) and the terracotta assembly is digit-for-digit R §13's.
* Twelve defects, none of them a wrong answer on the benchmarks: two undeclared deviations that
  R §12 must record (D1, D2), one live tie-break bug on an off-by-default path (D3), three
  transcription or ordering nits (D4, D5, D9), four documentation corrections (D6, D8, D10, D12),
  one harness inconsistency (D7), and one cost figure that overstates the port's advantage at ten
  threads by 3× (D11).

**Gate: not clean.** The native sweep fails five comparisons, and V3's own bar is "everything
passes, including native parity for all stages". They are inherited, explained and unchanged — but
they are failures, and this note is not the place to reclassify them.
