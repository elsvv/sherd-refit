# C1 — pair scales, hypotheses, the coarse score and NMS (R §1.2, §4.2, §5.1–5.3)

**Date:** 2026-09-07. **Branch:** `rust-core`. **Step:** phase 1c, C1, the first half of
`match_pair`; the previous step is T1 (the deterministic wall-thickness estimator).
**Code:** `crates/sherd-core/src/matching/{scales,hypotheses,coarse,nms,pair}.rs`,
`crates/sherd-core/src/spatial/kdtree.rs` (one new query), `crates/sherd-parity/src/stages/
{pairs,hypotheses,coarse,nms}.rs`, and one dump-only line in `sherd_refit/matching.py`.
**Reference:** `sherd_refit/matching.py::{Scales, hypotheses, coarse_score, nms}` at the fixture
commit. **Machine:** Apple M2 Pro (10 cores, 16 GB), macOS 15.6.1, rustc 1.97.0, Open3D 0.19,
numpy 2.5.2.

## Result in one line

This is the first step where two fragments meet, and **everything the reference computes without a
random number, the port computes bit for bit**: on 358 pairs of the eight fixture sets the injected
column is **6 086 comparisons with no failure** — the frame-pair set *and its order*, all twelve
fields of `Scales`, the coarse score of **every one of 38 126 133 hypotheses**, and the 250 poses
the suppression keeps. Native mode adds **1 074 comparisons with no failure**. Two stages cost
0.17 s and 0.80 s of a single core on the two pairs the reference takes 0.484 s and 2.318 s over.

| stage | mode | checks | failed | worst / tolerance |
|---|---|---:|---:|---|
| hypotheses | injected | 2 148 | 0 | 0.03 (`rotation`, 2.7e-6° of 1e-4°) |
| coarse | injected | 1 790 | 0 | 0.00 (bit-exact everywhere) |
| nms | injected | 2 148 | 0 | 0.87 (a PMC-6 tie row, see §5) |
| hypotheses | native | 1 074 | 0 | 0.25 (`n_hyp`, 7.5 % of ±30 %) |
| coarse, nms | native | — | — | no native column (D §10.2) |

## 1. What was ported

Four modules and the pair that drives them.

1. **`Scales` (R §1.2, §4.2).** `f(k, m) = max(k·t_pair, m·res_pair)` with `t_pair = min(t_A, t_B)`
   — a fragment carrying the rim measures a thicker wall than the body it broke off, and the wall
   is the thinner of the two — and `res_pair = max(res_A, res_B)`, the coarser mesh being what
   limits how precisely the pair can be told to fit. `nms` is the one threshold with no resolution
   floor; `icp` is a dimensionless stretch of the whole ladder rather than a rung-by-rung floor.
2. **Hypotheses (R §5.1).** `|dih_A + dih_B − 180| < 25°` over `brk_sub × brk_sub`, row-major, then
   `R = RA·RBᵀ`, `τ = P_A − R·P_B` with `RA = [t, ns, f]` and `RB = [−t, ns, −f]` as columns. `ns`
   is not negated and the other two are: two shells face the same way across a seam, while the two
   fracture surfaces face each other and the tangents run in opposite senses along the shared
   curve. 16 000 – 250 000 hypotheses per pair on these sets.
3. **The coarse score (R §5.2).** Sixty points of B's breakline, drawn once per pair from
   `brk_sub`, moved by each pose; a point scores when A's **full** breakline has a point within
   `sc.coarse` whose shell normal agrees to `cos > 0.7`. The score is that fraction.
4. **NMS (R §5.3).** A greedy walk best score first, dropping a pose when a kept one is nearer than
   `sc.nms = 0.5 t` **and** turned by less than `trace(RᵀR') > 2.9` ≈ 18.19°. `topk = 250`,
   `floor = 0.1`, order truncated to the first 5 000.
5. **`Pair` (R §4.1–4.2).** The wall-ratio skip, both `MatchData` built at `t_pair` (one of the two
   fragments rebuilds R §3.5, which B3's `MatchData::at` already did), the frames widened to `f64`
   once, and the three stages above as methods. The ICP ladder, the verification and the ranking
   are step C2's, on exactly this state.

`Frames` is the one new shape: the port stores the breakline in `f32` because it is cached and
every stored array must be a function of the other stored arrays (D §4.1, PMC-15), while the
matcher works in `f64` throughout. The widening therefore happens once per pair, and the tangent is
recomputed wide (`Breaklines::tangents_f64`) rather than widened from the narrowed one — a cross
product of two unit vectors costs nothing to compute twice and the reference keeps it wide.

## 2. Two defects the exact rows caught, and neither would have failed a tolerance

**`k · (1/60)` is not `k / 60`.** The port first wrote R §5.2's mean as a multiplication by the
reciprocal. `1/60` has no exact double, so for some counts the two differ in the last bit: 52 of
one pot_G pair's 40 029 scores came out `5.6e-17` away from the reference's. D §10.2's row allows
`1/60 + 1e-6`, three hundred million times that, so the tolerance would have passed it in silence —
and the ranking underneath is a sort, where a last-bit difference is a different order and can be a
different candidate. The `cs exact` row is in the harness for exactly this class of thing, and it
is the row that found it.

**The bounded query is not an optimisation, it is the algorithm.** R §5.2's radius is scipy's
`distance_upper_bound`, and the port's first version asked `kiddo` for the unbounded nearest
neighbour and then compared the distance itself. Most probe points of most poses land nowhere near
A's breakline, and an unbounded search has to find the true nearest however far away it is: 1.05 µs
per query against scipy's 0.22 µs, which made the port **slower than the reference** on this stage.
`PointTree::nearest_within` passes the radius into the traversal (`nearest_n(1).within(r²)`), and
the coarse stage over synthetic_20's 190 pairs went from **1 800 core-seconds to 178** with
bit-identical scores. scipy's bound is *exclusive* — a neighbour exactly at `delta` comes back as
`inf` — and the caller applies that itself, so the rule is stated where it is used.

## 3. The reference's fixture sink grew one line, and why

R §5.3's **walk order is an input**, and the reference's is `np.argsort(cs)[::-1]`: an unstable
quicksort over scores that are multiples of `1/60`, where thousands of hypotheses tie at every
level and the permutation among them is an artefact of numpy's partitioning. PMC-6 says the port
sorts stably instead. Comparing the two kept sets under two different orders therefore measures
numpy's sort, not the port's greedy loop.

`_match_pair` now hoists the expression it already evaluated and the sink writes it as
`nms1.order` (and `nms2.order`, which step C2 will need). Nothing the reference computes changes:
re-dumping the slab from the same `sherd_refit/` reproduced all **177** arrays and JSON files of
the previous dump byte for byte, and only `manifest.json` differs — in the commit it names and in
the two new arrays of its file list. With the order injected, the `nms` row is
exact — the port keeps the reference's 250 hypotheses, in the reference's order, on all 358 pairs.

All eight fixture sets were regenerated for it (`tools/dump_fixtures.py`, ≈ 20 min at
`--workers 4`), and the committed `fixtures/slab/dump` with them.

## 4. Parity: injected — 6 086 of 6 086 on 358 pairs

`sherd-refit-rs parity --fixtures output/fixtures/<set> --stage {hypotheses,coarse,nms} --injected`.
Each stage runs on the reference's own arrays at the pair's own `t`: `hyp.ia`/`hyp.ib` (in Open3D's
hash order — PMC-4's first real exercise), `md.brk_*` from the fragment directory or from the
`md_t/` rebuild the pair actually used, `scales.json`, `coarse.idx`, `nms1.order`.

| set | pairs | hypotheses | hypotheses | coarse | nms |
|---|---:|---:|---|---|---|
| terracotta | 6 | 166 635 | 36/0, 0.03 | 30/0, 0.00 | 36/0, 0.67 |
| pot_A | 28 | 3 893 002 | 168/0, 0.03 | 140/0, 0.00 | 168/0, 0.81 |
| pot_B | 36 | 3 396 743 | 216/0, 0.03 | 180/0, 0.00 | 216/0, 0.76 |
| pot_C | 21 | 685 427 | 126/0, 0.03 | 105/0, 0.00 | 126/0, 0.53 |
| pot_G | 21 | 468 554 | 126/0, 0.03 | 105/0, 0.00 | 126/0, 0.37 |
| pot_H | 55 | 1 659 645 | 330/0, 0.03 | 275/0, 0.00 | 330/0, 0.87 |
| synthetic_20 | 190 | 27 839 790 | 1140/0, 0.03 | 950/0, 0.00 | 1140/0, 0.74 |
| slab (committed) | 1 | 16 337 | 6/0, 0.03 | 5/0, 0.00 | 6/0, 0.60 |

(cells are "checks/failed, worst ratio of the tolerance"; 1.00 is the gate.)

Per pair the rows are:

| quantity | gate | measured |
|---|---|---|
| `scales` — the twelve fields of R §1.2 against `scales.json` | exact | 0 differing fields on 358 pairs |
| `n_hyp` | exact | 0 |
| `pairs` — the `(pa, pb)` set, as a merge of two sorted lists | exact | 0 |
| `order` — the same lists, position by position | exact | 0 |
| `rotation` — each pose against R §5.1 re-derived in the harness | 1e-4° | 2.7e-6° |
| `translation` — likewise | 1e-5 t | 0 (bit-identical) |
| `probe count`, `probe pool` — R §5.2's `min(60, \|brk_sub\|)` from B's subset | exact | 0 |
| `cs` — the score of every hypothesis | 1/60 + 1e-6 | 0 |
| `cs exact` — the same, bit for bit | exact | 0 of 38 126 133 |
| `kept count`, `kept` — R §5.3 on `nms1.order` | exact | 0 |

Three of those are worth reading twice.

* **`order` is exact and it is not the same claim as `pairs`.** PMC-4 lets the two implementations
  carry `brk_sub` in different orders, and injected mode removes that freedom by handing the port
  the reference's own subsets — so the row-major walk has to produce the reference's own row-major
  order, and it does. A port that iterated columns first would pass `pairs` and fail `order`.
* **`translation` is bit-identical and `rotation` is 2.7e-6°, and the asymmetry is `arccos`.** Both
  are measured against a second reading of R §5.1's formulas written in the parity crate, which
  assembles the frames as explicit column matrices and multiplies them with a plain triple loop.
  The translation agrees to the last bit; the rotation's residual is `f64` round-off of about
  `2e-15` in the trace, and `θ = arccos((trace−1)/2)` near zero turns `ε` into `√ε` — `4e-8` rad,
  which is `2.4e-6°`. There is no arithmetic difference behind it, only the derivative of `arccos`.
* **The load-bearing check on `R` and `τ` is the `coarse` row, not the `rotation` row.** A pose
  enters the coarse score through sixty transformed points and a nearest-neighbour test against a
  breakline whose spacing is a fraction of `t`; reproducing the reference's score bit for bit on
  38 million hypotheses is a far stronger statement about the frame mapping than agreeing with a
  second transcription of the same three lines. The `rotation` row is the cheap cross-check that
  says *which* of the two is wrong when they disagree.

## 5. PMC-6, measured: the ties move the membership and not the quality

With the port's own tie-break — a stable descending sort with the hypothesis index breaking ties —
the same suppression over the same scores keeps a different set. The `nms` stage measures four
things about that, on all 358 pairs:

| row | mean | worst | tolerance |
|---|---:|---:|---|
| `own order count` — kept count, relative | 0.03 % | 3.33 % | 5 % |
| `own order kept` — share of the reference's kept set the port misses | 14.3 % | 43.6 % | 50 % |
| `own order scores` — score gap at equal rank, both sets sorted | 0.0051 | 0.0333 (= 2/60) | 3/60 |
| `own order cover` — share of the reference's kept poses outside every ball of the port's | 10.0 % | 38.8 % | 50 % |

**The membership differs by up to 44 %; the quality does not differ at all.** Sorted by score and
compared rank by rank, the two kept sets are within one probe point of each other on average and
two at worst — which is the strongest statement available about a greedy walk whose order is
arbitrary among ties, and it is the one that matters, because what stage 1 receives is 250 poses to
refine and not a particular 250. The counts agree to 3.3 %, and they differ at all only on a pair
whose walk ends before `topk` (at the floor of 0.1, or at the five-thousandth hypothesis), where
how many poses survive depends on which member of each tie came first.

The four rows are gated at their measured worst plus headroom and they are **not parity claims** —
D §10.2 marks them as such. They are a regression alarm on the size of the tie effect: if a later
change made the port's order keep poses two probe points worse, this is where it would show.

## 6. Parity: native — 1 074 of 1 074

`sherd-refit-rs parity --fixtures output/fixtures/<set> --input <src> --stage hypotheses`. The port
preprocesses both fragments from their files, picks its own `t_pair`, rebuilds R §3.5 at it and
forms its own subsets; only the hypothesis count can then be compared, and D §10.2 allows ±30 %.

| set | pairs | `matched` | `matchable` | `n_hyp` worst |
|---|---:|---|---|---|
| terracotta | 6 | 6/6 | 6/6 | 6.9 % |
| pot_A | 28 | 28/28 | 28/28 | 0.15 % |
| pot_B | 36 | 36/36 | 36/36 | 0.30 % |
| pot_C | 21 | 21/21 | 21/21 | 0.01 % |
| pot_G | 21 | 21/21 | 21/21 | 0.03 % |
| pot_H | 55 | 55/55 | 55/55 | 0.004 % |
| synthetic_20 | 190 | 190/190 | 190/190 | 7.5 % |
| slab | 1 | 1/1 | 1/1 | 0.0 % |

Two rows are exact rather than statistical, and deliberately: **every pair the reference matched,
the port matches** (R §4.1's wall ratio) and **every pair the reference could match, the port can**
(R §5's first exit — both fragments need a fracture sample and a breakline). A pair that disappears
is not a tolerance question.

The counts agree far better than the row allows because, since T1, `t` is bit-identical on 41 of
the 68 fixture fragments and within `7.1e-6` on the rest, so `brk_sub` is nearly the same subset
and the dihedral filter is nearly the same filter. **The two sets whose counts move at all are
exactly the two whose working meshes do** (PMC-2, the decimator): the worst native `res` is
`7.0e-2` – `9.4e-2` on terracotta and up to `9.3e-2` on synthetic_20, against `1.8e-5` on pot_A and
`1.2e-7` or less on pot_B, C, G and H — where the port's decimation lands on the reference's mesh
and the hypothesis count follows it to four decimal places.

## 7. Cost

Against the reference on the same machine, the same pairs and the same single thread:

| pair | hypotheses | Python, 1 thread | port, 1 thread | port, 10 threads |
|---|---:|---:|---:|---:|
| terracotta `021__094` | 35 374 | 0.484 s | **0.17 s** | 0.03 s |
| synthetic_20 `frag_010__frag_015` | 163 098 | 2.318 s | **0.80 s** | 0.12 s |

The Python figure is `matching.hypotheses` plus `matching.coarse_score` called directly on the
dump's own arrays with the dump's own `coarse.idx`, so both sides score exactly the same
hypotheses on exactly the same sixty points (and the Python reproduces `coarse.cs` exactly, which
is the check that the timing harness is running the real thing). The port's figure is the whole
`parity --stage hypotheses --stage coarse --injected` process on a one-pair dump, so it includes
reading the arrays off disk and building the hypothesis set *twice* — it is an upper bound on the
two stages, and it is still 2.8–2.9× under the reference.

R §13's cost table puts the coarse score at 14 % of a mid-size pair's 6.94 s. On that basis this
step accounts for about a fifth of a pair, and the ICP ladder of C2 is two thirds of what remains.

Whole-set parity runs, for scale: `coarse --injected` over synthetic_20's 190 pairs and 27.8 M
hypotheses is 22 s of wall clock (178 core-seconds); `hypotheses --injected` over the same is 3 s;
native `hypotheses`, which preprocesses all twenty fragments from their files first, is 25 s.

## 8. Tests (+26; 207 → 233 passing, 1 ignored)

* **`Scales`**: the terracotta ratio (17.2 edges per `t`) leaves every floor inert and the slab's
  13.2 lifts all of them, the slab pair's twelve numbers are reproduced to the last bit from
  `scales.json`, the ICP rungs keep their ratios under a stretch, and `nms` has no floor.
* **The frame mapping**: a matching frame pair gives the identity pose; a partner displaced and
  turned is brought back onto the seam and its outward axis lands on the opposite of A's; the
  dihedral filter is on the *sum* and is strict; `pa`/`pb` are positions into the subsets in the
  order given, and an empty subset gives no hypothesis.
* **The coarse score**: the identity pose over two coincident curves scores 1 and the same probe
  moved past `delta` scores 0; a point exactly at the radius is a miss (scipy's exclusive bound);
  opposed shell normals never agree and the threshold is exercised on both sides of `0.7`; the
  score is the fraction that lands; the probe is drawn from the pool without replacement, is
  seeded, and takes the whole pool when it is smaller than the budget.
* **NMS**: the order is descending with the index breaking ties; a cluster collapses to its best
  member while a pose turned by 40° or standing 5 units away survives beside it; `topk` stops the
  walk and `floor` breaks it; the duplicate test is a conjunction; `trace > 2.9` is 18.19° and is
  tested in degrees on both sides of it.
* **The pair**: R §4.1's ratio is symmetric and refuses a fragment with no wall; a pair is built at
  the thinner wall and both `MatchData` come out at it.
* **`PointTree::nearest_within`**: inside its radius it is the unbounded answer on a 200-point
  sweep, outside it is nothing, and a negative radius answers nothing.
* **Ground truth** (`crates/sherd-core/tests/slab_pair.rs`): the whole of R §5.1–5.3 on the
  committed slab, whose true relative pose is known to machine precision. The best of the 250 kept
  poses is **3.20° and 0.305 t from the truth, at rank 0** — the highest-scoring pose the
  suppression keeps is the true one. This is the claim parity cannot make: a port could reproduce
  the reference's arrays exactly with the frame mapping transposed on both sides, and only a ground
  truth would notice.
* **Parity harness**: the pair directory and both sides' arrays resolve (including the `md_t`
  rebuild, matched on `md.params.json`'s `t` bit for bit rather than on Python's `%.9g` in a path
  name); all nine stages pass on the slab in both modes, with `coarse` and `nms` skipping natively
  by design; a dump with `scales.coarse` moved by 5 %, five hypotheses dropped, one coarse score
  raised by 0.05 and one kept index changed fails exactly the three stages that measure those; and
  a dump with no `nms1.order` skips rather than comparing against its own order.

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings` and
`cargo test --workspace --locked` are green (233 passed, 1 ignored). `pytest -q` is 58 passed,
unchanged. Determinism: the ignored terracotta test passes (two `segment` runs, byte-identical
caches) and `--threads 1` against `--threads 10` gives byte-identical caches too.

## 9. Corrections to the documents

* **D §10.1** gains `nms1.order`, `nms2.order`, `md_used.json` and `hyp.ia`/`hyp.ib` in the pair
  listing, and a paragraph on why the two orders are *inputs*.
* **D §10.2**'s `hypotheses` and `coarse` rows are filled in as implemented, an `nms` row and a
  PMC-6 tie row are added, and five paragraphs carry the measurements of §2, §4, §5 and §7.
* **D §12** gains a C1 row.
* **R §12.1** gains two addenda: the sink's two new arrays (a dump-only change, with the byte-level
  evidence that it changed nothing), and the re-verification of PMC-6, PMC-9's coarse draw and
  PMC-4.
* **`Scales` is `serde`-readable** because the harness reads `scales.json` back into it.
* The README's Rust section and `sherd-core`'s crate documentation follow.

## 10. What is next

Step C2: the ICP ladder of R §5.4–5.6 (experiment E5's point-to-point and point-to-plane), the
stage-1 re-score, the second NMS — for which `nms2.order` is already in the dump — the verification
of R §6 and the ranking of R §5.7. The `stage 1`, `stage 2` and `pair result` rows of D §10.2 are
what it has to meet, and `s1.T`, `s1.score`, `s2.T_*`, `s2.scores` and `result.candidates.json` are
already in every fixture.
