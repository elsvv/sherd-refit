# T1 — the wall thickness stops being a sample

**Date:** 2026-09-07. **Branch:** `rust-core`. **Commits:** `fbfebca` (the estimator),
`9347cfe` (the fixture uniforms and the Python tests), `03b3149` (the port and four defects),
`98bc319` (R), `09fb4d4` (the dump tool), and the documentation and fixture commits that follow.
**Machine:** Apple M2 Pro, 10 cores, 16 GB; toolchain 1.97.0; Open3D 0.19, numpy 2.5.2,
`OMP_NUM_THREADS=1` for every Python measurement.

The phase-1b verification (`2026-09-06-phase1b-verification.md`) closed with 43 of 1 478 native
parity checks failing across three of six stages, and traced all of them, through two intermediate
stages, to one number: R §3.2's wall thickness `t`. The estimator took the mode of a histogram over
rays cast from **20 000 randomly chosen faces**, so the port — which PMC-9 allows not to reproduce
numpy's PCG64 — was evaluating the same estimator on a different sample. Re-running the reference
with nothing but the seed changed moved `t` by up to 6.8 %, and `t` is the unit of every threshold
in R §1.2.

The decision was to remove the randomness from the **algorithm** rather than to replicate numpy.
R §3.2 now casts a ray from every face of the original largest component when there are at most
300 000 of them, and from `arange(0, n_faces, ceil(n_faces / 300000))` otherwise. Nothing else about
the estimator changed: the same `1e-3` origin offset, the same `> 0.7` looks-back filter, the same
60 bins over `(0, p90]`, the same unweighted per-face mode.

**The result, in one line:** native `t` is now **bit-identical on 41 of 68 fixture fragments** and
never more than **7.1e-6** relative anywhere, against a gate that went back from
`max(2 %, 3 bins)` to ±2 %. Native parity fails **5 of 1 524** checks instead of 43 of 1 478, and
all five are on one stage of one set.

---

## 1. What the old estimator was doing

Twelve seeds of the reference's own `estimate_thickness`, nothing else changed, on the fragments
where it mattered. The deterministic value is in the last column.

| fragment | seed 0 (the fixture's) | seeds 0–11: min / median / max | deterministic |
|---|---:|---|---:|
| `Pot_A_Piece_07_Mesh` | 3.5313 | 3.531 / **4.042** / 4.094 | **4.0880** |
| `Pot_A_Piece_04_Mesh` | 3.5539 | 3.554 / 3.785 / 3.795 | **3.7844** |
| `Pot_B_Piece_01_Mesh` | 5.6226 | 5.368 / 5.398 / 5.623 | **5.3984** |
| `frag_019` | 8.5725 | 8.085 / 8.120 / 8.582 | **8.1015** |
| `frag_010` | 6.2671 | 6.267 / 6.326 / 6.668 | **6.3355** |
| `Pot_G_Piece_05_Mesh_DS` | 2.4191 | 2.359 / 2.371 / 2.425 | **2.3629** |

`Pot_A_Piece_07_Mesh` is the clearest case and it is not a spread but a **coin flip**: its twelve
draws are 3.531, 4.037, 4.059, 4.093, 4.090, 4.089, 4.047, 4.013, 3.532, 3.539, 4.094, 3.533 —
two modes 15 % apart, four draws in one and eight in the other, and the fixture's seed 0 landed in
the smaller one. A fragment whose filtered distances form a plateau rather than a peak puts several
near-equal bins in contention and `argmax` picks by a count that another sample reorders. On five of
the six fragments above the deterministic value lands on the seed cloud's median or in its larger
mode; on `frag_004` it lands at the cloud's minimum, 2.0 % below the median.

**What `t` moved by across the benchmark.** 66 fragments of the seven collections, the fixture
value before against after:

| | value |
|---|---|
| fragments whose `t` moved by less than 1 % | **54 of 66** |
| median \|Δt\| | 0.12 % |
| mean \|Δt\| | 0.82 % |
| worst | 15.76 % (`Pot_A_Piece_07_Mesh`, the bimodal one) |

The largest movers, all of them fragments this project's notes had already named as thickness
outliers:

| set | fragment | before | after | Δ |
|---|---|---:|---:|---:|
| pot_A | `Pot_A_Piece_07_Mesh` | 3.5313 | 4.0880 | +15.76 % |
| pot_A | `Pot_A_Piece_04_Mesh` | 3.5539 | 3.7844 | +6.48 % |
| synthetic_20 | `frag_019` | 8.5725 | 8.1015 | −5.49 % |
| pot_B | `Pot_B_Piece_01_Mesh` | 5.6226 | 5.3984 | −3.99 % |
| synthetic_20 | `frag_004` | 8.1620 | 7.9499 | −2.60 % |
| pot_G | `Pot_G_Piece_05_Mesh_DS` | 2.4191 | 2.3629 | −2.33 % |

**Cost.** The stride bounds it: 0.40 s for the 1.23 M-face terracotta scan (246 063 rays at stride
5), 0.03 s for a 67 k-face pot sherd (every face). On the port, cold-cache `segment` over the four
terracotta fragments moved from 1.94 s wall / 6.19 s CPU to **2.71–3.38 s wall / 12.6–13.1 s CPU** —
the ray count on that set goes up twelvefold and the wall clock does not, because the casts are the
one embarrassingly parallel part of preprocessing. Two runs are still byte-identical, and so is
`--threads 1` against the default (§7).

## 2. The Python side: the quality gates

Everything here is the reference alone, at `OMP_NUM_THREADS=1`, evaluated with `tools/evaluate.py`
against the staged ground truth.

| set | before (scale-pairs note) | after T1 | verdict |
|---|---|---|---|
| terracotta | joins {021–094, 094–104}, 007 unplaced | **same**, both `pen` 0 | pass |
| pot A | 87.5 %, precision 1.000 | **87.5 %, 1.000** | pass |
| pot B | 100 %, 1.000 | **100 %, 1.000** | pass |
| pot C | 75 %, 0.667 | 75 %, **0.500** | accuracy equal, precision lower — §3 |
| pot G | 0 %, no join accepted | **0 %, no join accepted** | pass |
| pot H | 36.4 %, 0.429 | **36.4 %, 0.429** | pass |
| synthetic 20 | 95 %, 1.000 | **95 %, 1.000** | pass |

Cross-object joins 0 and group purity 1.000 on every set, before and after. `pytest -q`: **58
passed** (54 before; the four new tests are in §6).

**The terracotta scores, since the task asks for them.** Same two joins, same group, 007 unplaced,
both penetrations 0:

| join | score | tight | seam | gap | cont |
|---|---|---|---|---|---|
| 021–094 before | 11.701 | 0.548 | 21.33 t | 0.0083 | 0.011 |
| 021–094 **after** | **10.791** | **0.531** | **20.33 t** | 0.0090 | 0.014 |
| 094–104 before | 6.220 | 0.583 | 10.67 t | 0.0066 | 0.017 |
| 094–104 **after** | **6.060** | **0.568** | **10.67 t** | 0.0068 | 0.017 |

R §13's gate for this set is the join set, the unplaced fragment, `pen = 0` and `tight ≥ 0.27`, and
all four hold with room. Its two seam figures were stale and are now the measured ones (R §13).

## 3. pot C, where the precision column moved, and why it is not a regression

pot C used 3 joins before and uses 4 now; `correct` is 2 either way, so `correct / used` falls from
0.667 to 0.500. Fragment accuracy is 75.0 % before and after.

Two facts about this set make the precision column almost meaningless. First, **only four of its
seven fragments have a ground-truth pose** — 05, 06 and 07 are `unknown` — so a join to any of the
other three is scored `unscorable` and still counts in precision's denominator. Second, **piece 01
has no correct pose available in any run measured**: it attaches to a genuine neighbour at a wrong
pose (49° out) both before and after, and its `tight` sits exactly on `min_tight`, 0.255 before and
0.250 after.

Running the *old* estimator at five seeds settles it:

| run | fragment accuracy | precision | joins used | verdicts |
|---|---|---|---|---|
| old, seed 0 (the published number) | 75.0 % | 0.667 | 3 | 2 correct, 1 wrong pose |
| old, seed 1 | 75.0 % | 0.667 | 3 | 2 correct, 1 unscorable |
| old, seed 2 | 75.0 % | **0.500** | 4 | 2 correct, 2 unscorable |
| old, seed 3 | 75.0 % | 0.667 | 3 | 2 correct, 1 unscorable |
| old, seed 4 | 75.0 % | 0.667 | 3 | 2 correct, 1 wrong pose |
| **T1, deterministic** | **75.0 %** | **0.500** | 4 | 2 correct, 1 wrong pose, 1 unscorable |

`t` on pot C barely moves at all (piece 01 −1.52 %, piece 02 +1.09 %, the rest under 0.02 %), and
the join set still flips, because piece 01's acceptance is a coin balanced on `min_tight`. **0.500
is inside the old estimator's own cloud**: 1 of its 5 seeds produces it. Fragment accuracy — the
SfS++-comparable number every row of R §13 leads with — is 75.0 % on all six runs. R §13 now says
so rather than printing 0.667 as if it were a property of the algorithm.

This is the one place where a number in the scale-pairs table is lower after T1, and it is reported
rather than absorbed.

## 4. Native parity: before and after

`sherd-refit-rs parity --fixtures DIR --input DIR --stage all [--injected]`, release build, all
seven generated sets plus the committed slab. Cells are `checks/failed`, and the "before" column is
the phase-1b verification's §2.2 table.

| set | thickness | segmentation | breakline | samples |
|---|---|---|---|---|
| terracotta | 8/0 → **8/0** | 8/0 → **8/0** | 12/**1** → **12/0** | 24/0 → **24/0** |
| pot_A | 16/0 → **16/0** | 16/0 → **16/0** | 24/**1** → **24/0** | 48/**1** → **48/0** |
| pot_B | 18/0 → **18/0** | 18/**2** → **18/0** | 27/**7** → **27/0** | 54/**3** → **54/0** |
| pot_C | 14/0 → **14/0** | 14/0 → **14/0** | 21/0 → **21/0** | 42/0 → **42/0** |
| pot_G | 14/0 → **14/0** | 14/0 → **14/0** | 21/**3** → **21/0** | 42/**1** → **42/0** |
| pot_H | 22/0 → **22/0** | 22/0 → **22/0** | 33/0 → **33/0** | 66/0 → **66/0** |
| synthetic_20 | 40/0 → **40/0** | 40/**2** → **40/0** | 60/**21** → 60/**5** | 120/**1** → **120/0** |
| slab | 4/0 → **4/0** | 4/0 → **4/0** | 6/0 → **6/0** | 12/0 → **12/0** |
| **total** | 136/0 | 136/**4** → **136/0** | 204/**33** → 204/**5** | 408/**6** → **408/0** |

Load and working mesh pass everywhere in both tables and are omitted. The phase-1b totals quoted
above exclude the slab, which failed nothing then either, so the totals below include it on both
sides. Across all six stages: **1 524 native comparisons, 5 failures** (the phase-1b sweep: 1 478
with 43, excluding the slab's 46); **3 847 injected comparisons, 0 failures** (3 333 with 0, before
the six D6 columns per fragment existed).

Per quantity, native, over all 68 fragments:

| quantity | n | bit-exact | worst | tolerance | fails |
|---|---:|---:|---|---|---:|
| `t` | 68 | **41** | 7.08e-6 rel (`FY234021_reduced`) | 2 % | 0 |
| `thick_mode` | 68 | 28 | 1.20e-4 rel (`frag_004`) | 2 % | 0 |
| segmentation agreement | 68 | **39** | 0.9814 (`frag_014`) | ≥ 0.97 | 0 |
| fracture fraction | 68 | 1 | 1.14 pp (`frag_014`) | ±2 pp | 0 |
| breakline curve length | 68 | 0 | 3.95 % (`frag_014`) | ±10 % | 0 |
| breakline p99 distance | 68 | 0 | 0.901 t (`frag_019`) | 0.5 t | **2** |
| breakline dihedral KS | 68 | 0 | 0.0655 (`frag_014`) | 0.05 | **3** |
| `n_frac` | 68 | 58 | 2.00 % | ±10 % | 0 |
| margin fraction | 68 | 1 | 1.57 pp | ±5 pp | 0 |
| `S` / `Pf` spacing | 68 | 0 | 1.34 % / 1.23× | factor 2 | 0 |
| `res` | 68 | 0 | 9.39 % (`FY234021_reduced`) | ±10 % | 0 |
| `faces`, `area`, `watertight` | 68 each | 66 / 0 / **68** | 9e-5, 0.363 %, — | ±5 %, ±0.5 %, exact | 0 |

Two things are worth naming. **`t` is bit-identical on 41 of 68 fragments and within 7.1e-6
everywhere**, which is `parry3d` against Embree (PMC-17) and the `f32` histogram, and nothing else:
both sides now cast from the same faces of the same mesh. And **39 of 68 fragments have label
arrays equal to the reference's entry for entry** natively, where before T1 the number that could be
said was "66 of 68 inside the gate".

## 5. The breakline `count` row becomes a curve-length row — and the estimator matters

The team decision was to replace the native `count` gate (±10 % on the number of breakline points)
with a curve-length gate, `count × median nearest-neighbour spacing` at ±10 %, because a ±10 % gate
on the point count underneath a ±10 % gate on `res` has no headroom by construction: a breakline
crossing a mesh with longer edges has proportionally fewer edges to cross.

Implemented literally, that row **fails**: `count × median spacing` disagrees between port and
reference by 4.46 % on average and 11.17 % at worst over synthetic_20, and the separate `spacing`
row (which was the density, and which I had added beside it) failed on 4 of 4 terracotta and 14 of
20 synthetic fragments — rebuilding exactly the contradiction the decision was meant to remove.

The measurement says why. The spacing along a decimated mesh's breakline is heterogeneous, so the
median is a robust summary of the *typical* gap rather than an additive one; a length has to add up.
Over the twenty synthetic_20 fragments, port against reference:

| estimator of the curve's length | mean \|Δ\| | worst \|Δ\| |
|---|---:|---:|
| `count × median` nearest-neighbour spacing | 4.46 % | 11.17 % |
| `count × mean` spacing = **Σ nearest-neighbour distances** | **1.42 %** | **3.95 %** |

The row therefore measures the **sum of every point's distance to its nearest other point**, at the
decision's own ±10 %. Over all 68 fragments the worst is 3.95 % — under half the allowance. The
point *density* is not gated at all: it is a function of `res`, which the working-mesh row already
gates at ±10 %, and gating it twice is the contradiction. On the terracotta the port's breakline
spacing is 10.7–14.7 % wider than the reference's while the curve's length agrees to 0.2–1.4 %.

This is a change to the quantity, not to the tolerance, and it is recorded here because it departs
from the letter of the decision.

## 6. The five failures that are left, and why neither side is wrong

All five are the native breakline row on synthetic_20: `p99 distance` on `frag_014` (0.888 t) and
`frag_019` (0.901 t) against `0.5 t`, and `dihedral KS` on `frag_010` (0.0636), `frag_014` (0.0655)
and `frag_017` (0.0512) against 0.05. **`t` is bit-identical on all five.** What differs is `res`:
+2.7 % to +8.3 % on these fragments, inside the working-mesh row's ±10 %, because `meshopt` and
Open3D's quadric decimators spend the same face budget differently (PMC-2).

**The `p99 distance` row is not survivable by the reference either.** Running the reference's own
pipeline twice on the same fragment with only the face budget changed — 200 000 against 174 000, a
`res` gap of 5.4–8.4 % — and comparing its two breaklines at the same `t`:

| fragment | Δ`res` | reference vs itself: p99 | KS | port vs reference: p99 | KS |
|---|---:|---:|---:|---:|---:|
| `frag_010` | +5.46 % | 0.201 t | 0.0151 | 0.321 t | **0.0636** |
| `frag_014` | +6.61 % | **0.866 t** | 0.0129 | **0.888 t** | **0.0655** |
| `frag_017` | +8.39 % | 0.244 t | 0.0136 | 0.388 t | **0.0512** |
| `frag_019` | +5.39 % | 0.251 t | 0.0123 | **0.901 t** | 0.0235 |
| `frag_005` | +8.22 % | 0.471 t | 0.0153 | 0.348 t | 0.0444 |
| `frag_019` at −11 % faces | +8.90 % | **1.071 t** | 0.0334 | — | — |

The reference against itself puts `frag_014`'s two breaklines `0.866 t` apart and `frag_019`'s
`1.071 t` apart at the 99th percentile, at `res` gaps the working-mesh row allows — **larger than
the port's own 0.888 t and 0.901 t**. A `0.5 t` gate on the 99th percentile is unreachable by any
implementation whose decimator is not Open3D's; it measures the working-mesh row. The row is not
widened; D §10.2 now says what it measures.

Over all 68 fragments, **313 of 126 687** point-to-set distances exceed `0.5 t` (0.247 %), against
**3 115 of 271 592** (1.15 %) before T1.

**The three `dihedral KS` failures are the segmentation row leaking into a curve statistic.** The
same experiment moves the reference's own dihedral distribution by a KS of only 0.012–0.033, so
resolution alone does not explain 0.051–0.066. What does: `frag_010`, `frag_014` and `frag_017`
carry three of the four lowest segmentation agreements of the set (0.9885, 0.9814, 0.9868), and the
breakline is the *boundary* of the mask the agreement measures — a 1.9 % area disagreement is a far
larger fraction of a boundary than of a surface. Both rows are inside their gates and the pair is
not, which is the non-independence D §10.2 already describes for `Pf spacing`. It belongs with
R §13's pair gates, not with a widened tolerance.

## 7. The other defects of the phase-1b verification

| defect | what was done |
|---|---|
| **D1** — native parity red on three stages | closed by the estimator change: 43 → 5 failures, on one stage of one set, none of them a `t` difference (§4, §6). |
| **D2** — PMC-9 rewritten inside the frozen text | R §12's PMC-9 row is restored word for word to its wording at `9d4b9d3`; the stream-split decision moves to a new **R §12.1**, a dated and attributed addendum list, which also records that the amendment's own re-verification clause points at R §13 and is therefore asserted rather than demonstrated until phase 1c runs it. |
| **D3** — two `valid` predicates | `Breaklines::valid` is a stored field now, computed once in `f64` before the frames are narrowed and used to filter `sub`; the `f32` accessor is gone, replaced by `n_valid()`. The cache gains `brk_valid` (BOOL `[k]`) and a length check; `CACHE_VERSION` 4 → 5. |
| **D4** — the `near` tie rule unlisted | **PMC-16** in R §12: `cKDTree.query`'s tie rule is unspecified, `kiddo` resolves to the lowest index, re-verified by the injected `rep face` check that measures exactly 0 on all 68 fragments — no fixture has a tie, so the first symmetric synthetic mesh is what will exercise it. |
| **D5** — the ray-cast substitution unlisted | **PMC-17** in R §12: `parry3d` 0.30 against Embree, with E4's 2 hit/miss disagreements over 7.87 M cone rays as the evidence and the injected `votes/face` gate as the re-verification. |
| **D6** — injected mode never ran the port's sampler | the dump carries `md.S_u`, `md.S_v`, `md.Pf_u`, `md.Pf_v` (the uniforms before R §3.5.1's fold) and the injected samples stage rebuilds the reference's own `md.S` and `md.Pf` from them through the port's own fold and barycentric expression: **1 962 406 points over 68 fragments, every one bit-identical**. The face pick is checked without uniforms, by requiring the midpoint of every face's cdf interval to pick that face: **7 404 452 faces probed, none wrong**. |
| **D7** — injected thickness skips synthetic_20 | stated in D §10.2's preamble instead of being papered over: at level `slim` there is no `load.V0`, injected thickness skips those 20 fragments by design (finding F3), and the injected claims rest on the other 48. Native thickness covers all 68 and is now nearly as strong a check, both sides casting from the same faces. |
| **D8** — D §10.2 described an estimator the harness does not use | D §10.2's preamble now describes the quadrature the harness actually performs — one centroid per reference face, area-weighted, labelled by its own label on one side and by `closest_face` on the other — and says what it gives up (a wide face-area spread) rather than describing a 200 000-point Monte-Carlo sample that was never implemented. Two further narrowings are stated there: what `exact` means under PMC-9, and that a stage's injected column only covers the fragments whose dump carries its inputs. |

## 8. Gates

| gate | result |
|---|---|
| `cargo fmt --all --check` | **ok** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **ok** |
| `cargo test --workspace --locked` | **ok**, 194 passed, 1 ignored |
| `pytest -q` (sink off) | **ok**, 58 passed |
| two `segment` runs byte-identical | **ok**, `cmp` clean on all four terracotta caches |
| `--threads 1` vs default byte-identical | **ok** |
| injected parity, 6 stages × 8 sets | **3 847 checks, 0 failures** |
| native parity, 6 stages × 8 sets | **1 524 checks, 5 failures** (§6) |

## 9. New tests

* Rust, `fragment/thickness.rs`: `face_indices` against the reference's expression at, below and
  above the cap, including the degenerate `n_faces = 0` and `cap = 0`; the estimator asserted to
  return the identical float when called twice (there is no seed to vary any more).
* Rust, `stages/breakline.rs`: `curve_length` as the sum of the nearest-neighbour distances,
  including the scaling property and the far-outlier path that exercises the radius doubling.
* Rust, `stages/samples.rs`: the two D6 columns asserted **exact**, not merely inside a tolerance.
* Rust, `fragment/cache.rs`: `brk_valid` round-trips and a file without it is refused by its
  version.
* Python, `tests/test_synthetic.py`: five seeds through `Fragment.from_mesh_file` give the identical
  `t`, `thick_mode`, `res` and face count; the ray count is `n_faces` below the cap and
  `ceil(n_faces / stride)` above it and never exceeds `THICK_FACES`; `sample_on_faces(...,
  return_uv=True)` returns the pre-fold uniforms, changes neither the points nor the face ids, and
  the reference's own points are rebuilt from them exactly.

## 10. What T1 did not do

* **E6 (replicating numpy's PCG64) is no longer needed for `t`** — preprocessing draws nothing at
  random at all — but PMC-9 still covers R §3.5's three samplers, whose arrays remain incomparable
  point by point. What changed is that the incomparability no longer reaches `t` or anything
  upstream of the samplers.
* **PMC-2, the decimator, is now the largest source of native divergence.** Every one of the five
  remaining failures is downstream of `res`, and `res` runs +2.7 % to +9.4 % on the port. That is
  the next thing worth measuring if R §13's pair gates ever fail for a reason that is not `t`.
* **The fixture dump grew.** `thick.idx` / `thick.t_hit` / `thick.prim` are the stride set rather
  than 20 000 entries, and `md.*_u` / `md.*_v` are new; the seven generated sets went from 2.3 GB to
  2.6 GB and the committed slab from 18 MB to 21 MB. `docs/superpowers/notes/2026-09-06-p0-fixtures.md`
  §2's inventory predates both and its `(20000,)` shapes should be read as "the R §3.2 ray set".
