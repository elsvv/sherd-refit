# R1 — the rule, chosen on a table that contains the museum's own collection

**Date:** 2026-09-11. **Tree:** branch `rust-core`, `373520e` (task V10) → `R1.1`…`R1.8`.
**Machine:** Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`,
Python 3.12 in `.venv` (open3d 0.19.0, numpy 2.5.2, scipy 1.18.1). Working tree at the start:
clean, nothing to keep.

**The five sentences.** *The rule had to be chosen again, on a table that contains the museum's own
collection*: task S3 chose its rule on nine development collections where it is perfectly clean, and
task A2 then measured **thirteen** false confirmed joins on `input/sfspp/mixed_all` — ten real pots,
164 sherds — so this step built the table with that collection in it, 171 rules over 49 runs, and
chose from that. *What it chose adds three witnesses and not one number*: **both** re-searches must
land on the placement, the second placement must be one **R §6.5 itself refuses**, and the strict
half must hold on the **two redraws** — three booleans over evidence every run has computed and
reported since task S3, because task A2's finding was that a *threshold* fitted on one pot does not
transfer, and this step re-measured that on the corrected quantity: the rival distance separates
correct joins from false ones at AUC 0.776 on the development runs and **0.562** on `mixed_all`.
*The result is 13 false confirmed joins taken to 1 and the cross-object count to 0*, at 41 correct
against 62 on `mixed_all` and 174 against 225 on the 45 development runs — a museum trading recall on
collections it already trusted for precision on the one it did not, with the joins it loses still on
the probable list with the reason printed. *There are single-run rules in the table with **zero**
false joins everywhere and not one of them is shippable*: every one asks the second placement to be
25–30 walls away, and the terracotta's own two museum joins have rivals at 9.98–26.21 t, so audit
§E step 8's row reads two of ten at 25 t and none at 30 — which is why the thing that does reach zero
ships as a **mode**, `--tier-agree-seeds N`, measured at **11 correct joins and no false one** on
`mixed_all` for the price of a second whole run (78.6 min against 39.5). *And the binary agreed with
the table to the join*: the confirmation runs confirm **19** pairs at seed 0 and **11** under two
agreeing seeds, which is what §4 and §5 predicted, with group purity 1.000 and cross-object 0 on
both.

---

## 0. What this step was given

Task S3 chose the shipped rule on the nine development collections at five seeds — 45 runs, 225
correct confirmed joins, **no false one** — and task A2 then ran it on `input/sfspp/mixed_all`, the
museum's own ten pots and 164 real sherds, where it confirms **62 correct joins and 13 false ones**
over seeds 0 and 1. Every one of the thirteen is on the margin arm with `support` 0, and the
conjunct that made the rule clean on the development sets — the second placement being at least
five wall thicknesses away — separates nothing there: A2 §4.1 measured the false joins' rivals at a
median of 15.85 t and 14.19 t against 18.04 t and 12.59 t for the correct ones.

So the rule has to be chosen again, on a table that **contains** that collection. That is this
step, and everything below is on that table.

---

## 1. V10's observation, closed: the margin arm's two halves at one pose

Task V10 §11 wrote it down as an observation and not a defect, because it changes no number of that
cycle: `WideRival::moved_t` is measured from the pair's **best** candidate (`pair.rs:483`), while
`Probes::wide_margin` is each candidate's **own** `score / rival.score` (`tiers.rs:1021`). For a
pair's non-best accepted candidate the margin arm's two conjuncts were therefore statements about
two different poses. It was immaterial there because the scored population is `representatives`,
which takes a pair's best.

**It is not immaterial to a rule table.** Over the 45 development runs, **2 323 of the 3 184**
accepted candidates are not their pair's best (rank 0 holds 861 of them; the ranks run
861 / 719 / 604 / 535 / 465), and `s3_pairs` — like `tiers::classify` — reads every one of them:
a pair is confirmed when *any* of its candidates clears the rule.

**The decision is to measure both halves at the candidate under test**, which is
the pose the margin is already about and the pose the *kept* rival has always been measured from
(`tiers::rival`, `tiers.rs:1055`). After the fix the two margins differ only in which population
the rival is drawn from, which is the thing task S3's note claims about them. `WideRival` gains the
rival's own `transform`; `Probes` gains `wide_rival_moved_t`, computed per candidate by the same
`placement_gap`; `Thresholds::rival_moved_of` reads that. The pair-level number keeps its meaning —
it is what *selected* that pose as the pair's second placement, a property of the pair — and is
reported beside the new one as `Evidence::wide_rival_pair_t`, so nothing is lost.

**What it changes on the 45 development runs: the number, and no join.**
`output/r1/same_pose_effect.py` applies **task S3's rule** offline to two sets of dumps of the same
45 runs — task S3's own (`output/s3/measure/rules`) and this step's (`output/r1/measure/rules`) —
so the only quantity that moved between them is the one the margin arm reads:

| | |
|---|---:|
| runs compared | **45** |
| accepted candidates | **3 184** |
| … on which the wide rival's distance moved | **1 655 (52.0 %)** |
| runs whose confirmed set changed | **0** |
| confirmed pairs gained | **0** |
| confirmed pairs lost | **0** |

The distance moves on half the population and moves **no** confirmed join. That is the honest shape
of this fix: it is a correctness repair to what a number means, not a change of behaviour, and the
rule table below is read on the repaired quantity because a table read on the other one would be
measuring something the binary does not test.

---

## 2. The evidence task A2 left behind, and why it was not enough

Task R1's item 2 asks whether `output/acceptance2/` already carries what
`tools/measure_tiers.py --stage rules` needs. **Two questions with two different answers**, and
`output/r1/acceptance2-check/` is the check.

**Does it carry the fields? Yes, every one of them.** A run with the tier pass on writes an
`evidence` block on every accepted candidate of `report.json`, and it holds `arm`, `margin`,
`rival_moved_t`, `wide_margin`, `wide_rival_moved_t`, `wide_rival_source`, `wide_rival_accepted`,
`research` (agreed of run), `support`, `slide_t`, `placements`, the three redraw summaries,
`colour`, and the five scores on the candidate itself. `adapt_report_to_dump.py` turns a
`report.json` into the shape `--stage rules` parses, and with it **eleven of the twelve rows of
task A2 §4.2 reproduce exactly** from the saved runs alone — M1's 51 / 41 / 10, S3's 75 / 62 / 13,
`support ≥ 1`'s 16 / 16 / 0, the wide arm alone at 63 / 50 / 13, and the rest.

**One row of A2 §4.2 does not reproduce, and it is worth naming.**
`support ∨ (wide ≥ 10, rival ≥ 5 t, re-search 1)` is printed there as **35 / 32 / 3** and is
**40 / 37 / 3** when re-derived from A2's own runs. The false count — the number the row exists for
— is the same, and no decision of either task rests on the row; the correct count is five higher
than the note says. Nothing else in that table differs.

**Is it enough to choose task R1's rule on? No.** Of the **17 267** accepted candidates in those
four saved runs, **12 231 (70.8 %)** are not their pair's best-scoring candidate, and for every one
of them the saved `wide_rival_moved_t` is §1's *other* quantity. So the two collections were re-run
**once each at seeds 0 and 1** with `--tiers on --resample-seeds 2 --measure` on the tree that has
the fix — which is the run item 2 allows, and the only large runs of this step besides item 5's.

---

## 3. What is on the table, and what a rule may be built out of

`tools/measure_tiers.py --stage rule-runs` made every dump this table is read off, with
`--tiers on --resample-seeds 2 --measure`, so both re-search outcomes are recorded on every
accepted candidate and the run's own verdict constrains no row. `--stage r1` then evaluates every
rule **offline** from those dumps: `output/r1/measure/r1.md` is the whole table and
`output/r1/measure/r1.json` its data.

| collection | runs | accepted candidates | ground-truth adjacent pairs |
|---|---:|---:|---:|
| the nine development sets, seeds 0–4 (**dev45**) | 45 | 3 184 | 970 |
| `input/sfspp/mixed_all`, seeds 0 and 1 | 2 | 16 910 | 576 |
| `input/synthetic_mix3_60`, seeds 0 and 1 | 2 | 357 | 254 |

**The strict half is task M1's and never moves** — `tight ≥ 0.35`, `gap ≤ 0.015 t`, `seam ≥ 5 t`,
`cont_n ≥ 0.90`, `pen ≤ 0` and measurable, `slide ≤ 0.1 t`. What every row varies is the **second
arm**, and the five things it can be built out of are all quantities a run has computed and
reported since task S3:

| witness | what it asks | where it came from |
|---|---|---|
| `support ≥ n` | do *n* independent joins of the collection agree with this placement? | task M1 |
| `margin(wide) ≥ m` | does this candidate beat the pair's best second placement by *m*? | task S3 |
| the rival is ≥ *d* `t` away | is that second placement a genuinely different fit, or the same seam slid along itself? | task S3 |
| `re-search ≥ k` of 2 | do *k* independent searches of the pair, on other draws, land here? | task S3 |
| **the rival is one R §6.5 refuses** | would the search itself have believed the alternative? | recorded by task S3 (`WideRival::accepted`), read by nothing until now |
| **the strict half holds on the two redraws** | would the same sherds, sampled again, still clear the bar? | measured by M1 §5.8, left out because on the development sets it removed no false join |

The last two are **booleans**, not thresholds. That matters more than it sounds: task A2 §4.1's
finding was that `RIVAL_FAR_T = 5`, a number fitted on 89 false candidates of which 69 were one
pot, does not transfer to a second body of sherds. A rule assembled out of booleans over
already-reported evidence cannot fail that way, because there is no number in it to fit.

---

## 4. The rule table

Every rule is evaluated **offline** from the dumps of §3, so the runs' own verdicts constrain no row
and 49 runs answer for a table of 171 rules (`output/r1/measure/r1.md`). A pair counts once per
run, under its best-scoring candidate that clears the rule — `tiers::representatives`, in Python.
The rows that decide:

| rule | dev45 correct / **false** | terracotta slots | `mixed_all` correct / **false** / cross | `synthetic_mix3_60` correct / **false** |
|---|---:|---:|---:|---:|
| `support ≥ 1` alone | 105 / **0** | **0** | 16 / **0** / 0 | 29 / **0** |
| M1's rule — `support ∨ margin(kept) ≥ 2` | 169 / **0** | 9 | 41 / **10** / 3 | 63 / **0** |
| **S3's rule** — `support ∨ (wide ≥ 2, rival ≥ 5 t, re-search 1)` | 225 / **0** | 10 | 62 / **13** / 2 | 79 / **0** |
| … with **both** re-searches | 206 / **0** | 10 | 55 / **7** / 1 | 74 / **0** |
| … with the rival **refused** | 211 / **0** | 10 | 47 / **7** / 2 | 79 / **0** |
| … with the strict half on the **redraws** | 198 / **0** | 10 | 61 / **10** / **0** | 68 / **0** |
| … both re-searches + rival refused | 196 / **0** | 10 | 42 / **2** / 1 | 74 / **0** |
| … both re-searches + redraws | 184 / **0** | 10 | 54 / **5** / **0** | 63 / **0** |
| … rival refused + redraws | 184 / **0** | 10 | 46 / **5** / **0** | 68 / **0** |
| **R1, and what ships** — both + refused + redraws | **174** / **0** | **10** | **41** / **1** / **0** | **63** / **0** |
| S3 with the rival at ≥ 8 t | 216 / **0** | 10 | 54 / **12** / 1 | 76 / **0** |
| S3 with the rival at ≥ 10 t | 209 / **0** | 9 | 48 / **12** / 1 | 71 / **0** |
| S3 with the rival at ≥ 15 t | 202 / **0** | 9 | 41 / **7** / 1 | 65 / **0** |
| S3 with the rival at ≥ 20 t | 172 / **0** | 4 | 24 / **2** / 0 | 59 / **0** |
| S3 with the rival at ≥ 25 t | 157 / **0** | 2 | 19 / **1** / 0 | 54 / **0** |
| S3 with the rival at ≥ 30 t | 144 / **0** | **0** | 18 / **0** / 0 | 46 / **0** |
| both re-searches, rival ≥ 25 t | 142 / **0** | 2 | 18 / **0** / 0 | 51 / **0** |
| both re-searches, rival ≥ 25 t, rival refused | 142 / **0** | 2 | 18 / **0** / 0 | 51 / **0** |

Four things in that table, and the fourth is the decision.

**(a) `support ≥ 1` alone is still the only *simple* clean rule, and it is still unusable.** 16
correct joins on `mixed_all` and no false one — the trade task A1 §3.5 priced and task A2 confirmed.
What the table adds is the column that kills it: **the terracotta confirms none of its two museum
joins at any seed**, because those three sherds are a chain and a chain gives the support arm no
second path. `tools/quality_gate.py` exits non-zero on that row (audit §E step 8), so a default of
`support ≥ 1` is a default that fails the project's own gate.

**(b) Each of the three new conjuncts earns its place on its own, and they are not the same
conjunct.** Against S3's 13 false joins: both re-searches take it to 7, the rival's verdict to 7,
the redraws to 10 — and the three overlap only partly, because pairwise they reach 2, 5 and 5 and
together 1. What each removes is a different family. The **redraws** are the only one of the three
that takes the **cross-object** count to zero on its own: `Pot_E_16–Pot_I_25` is a join between two
pots whose `gap` is 0.0149 t on the draw the run made and 0.0162 t on the second redraw, over the
0.015 t limit, and no other witness in this table sees it.

**(c) The rival distance still does not transfer, and now there is a third collection to say so.**
§5 of `r1.md` re-derives it over the population the margin arm answers for — candidates clearing the
strict set with **no support**:

| collection | candidates | correct | false | correct: min / 5 % / med / 95 % / max | false: the same | AUC | any threshold separates? |
|---|---:|---:|---:|---|---|---:|---|
| dev45 | 654 | 610 | 44 | 1.36 / 1.97 / **18.81** / 66.47 / 323.47 | 1.75 / 1.75 / **2.98** / 21.90 / 21.90 | 0.776 | no |
| `mixed_all` | 413 | 303 | 110 | 1.15 / 3.43 / **16.29** / 33.46 / 354.95 | 0.00 / 1.56 / **13.11** / 26.24 / 29.17 | **0.562** | no |
| `synthetic_mix3_60` | 156 | 156 | 0 | 2.80 / 7.92 / 19.27 / 46.50 / 67.79 | — | — | — |

AUC **0.776** on the development runs and **0.562** on the museum's own collection: on the
collection the rule exists for, the distance is barely better than a coin. That is A2 §4.1's
finding re-measured on the corrected quantity of §1, and it is why not one number in the shipped
rule moved.

**(d) There *are* single-run rules with zero false joins everywhere, and not one of them is
shippable.** Every one asks the rival to be 25 or 30 walls away — and the terracotta's own two
museum joins have rivals at 9.98, 16.45, 19.19, 19.49, 19.57, 19.69, 23.16, 23.36, 26.18 and
26.21 t. At 25 t the row that audit §E step 8 states is met **twice of ten**; at 30 t, **none of
ten**. A rule that reaches zero on `mixed_all` by fitting a distance to `mixed_all` and losing the
museum's own four scans on the way is the mistake this whole cycle exists to stop making.

**So the default is the rule with no fitted number in it**, and it is the best single-run rule this
table has that holds the terracotta:

```
Confirmed  ⇔  R §6.5 accepted it
              ∧ tight ≥ 0.35 ∧ gap ≤ 0.015 t ∧ seam ≥ 5 t ∧ cont_n ≥ 0.90
              ∧ pen ≤ 0 (and measurable) ∧ slide ≤ 0.1 t          — M1's strict half, unmoved
              ∧ every one of those limits holding on the two redraws of R §3.5's samples
              ∧ ( support ≥ 1
                  ∨ ( margin(wide) ≥ 2
                      ∧ the wide rival is ≥ 5 t from **this** candidate
                      ∧ both independent re-searches land on this placement
                      ∧ that rival is a placement R §6.5 itself refuses ) )
```

**What it costs and what it buys, stated as a trade.** On the 45 development runs it confirms
**174 correct joins against S3's 225** — per set 10 / 22 / 16 / 3 / 0 / 2 / 41 / 38 / 42 against
S3's 10 / 26 / 22 / 4 / 0 / 2 / 59 / 48 / 54, the terracotta's ten held and `pot_G`'s zero still
zero — and on the museum's ten pots it takes **13 false confirmed joins to 1 and the cross-object
count from 2 to 0**, at 41 correct against 62. A museum trades 51 correct joins on collections it
already trusts for twelve false ones on the collection it does not.
The note says that plainly because it is the whole content of the step.

**The one that is left, named.** `Pot_I_Piece_07–Pot_I_Piece_11`, seed 1: `wrong_pose` at
**1.064° and 0.570 t** from the truth, against `evaluate.py`'s 5° and 0.5 t — a miss by **seven
hundredths of a wall**. Its geometry is a correct join's: `seam` 39.0 t, `tight` 0.843,
`cont_n` 0.996, `slide` 4.6e-14 t, a wide margin of 16.33 over a refused rival 23.59 t away, and
both re-searches agreeing. No threshold on any quantity in this table separates it from the joins
that are right, because at four hundredths of a wall past the scorer's line it very nearly *is*
one. What separates it is that the other seed does not confirm it — which is §5.

---

## 5. `--tier-agree-seeds N`: the mode that does reach zero, and what it costs

Item 4 asks for zero on `mixed_all` at **both** seeds. §4 (d) says no single-run rule reaches it
without losing the terracotta, so the thing that does reach it ships as a documented mode instead.

`--tier-agree-seeds N` matches and tiers the whole collection **N times**, at N seeds, and a join
keeps the confirmed band only where every run confirmed that pair **at the same placement** —
`RESEARCH_T = 0.5 t` at `placement_gap`'s worst vertex and `RESEARCH_DEG = 2°`, which are the
re-search's own tolerances asked of a whole collection instead of one pair. It never promotes: a
join no run confirmed stays where it was, the extra runs' candidates and poses are thrown away, and
a demotion writes its reason into the same `failed` list every other refusal goes into, naming the
seed that disagreed.

**One answer per collection**, which is the only fair way to compare a rule that costs one run with
one that costs two — a single-run row read at the collection's first seed, an agreement row read
across its first two:

| rule | answer from | dev45 correct / **false** | `mixed_all` correct / **false** / cross | `mix3_60` correct / **false** |
|---|---|---:|---:|---:|
| `support ≥ 1` | 1 run | 20 / **0** | 8 / **0** / 0 | 12 / **0** |
| M1's | 1 run | 34 / **0** | 19 / **4** / 0 | 29 / **0** |
| S3's | 1 run | 49 / **0** | 32 / **5** / 0 | 38 / **0** |
| **R1's, shipped** | **1 run** | **38** / **0** | **19** / **0** / 0 | **32** / **0** |
| `support ≥ 1` | 2 runs agreeing | 11 / **0** | 5 / **0** / 0 | 9 / **0** |
| M1's | 2 runs agreeing | 17 / **0** | 12 / **0** / 0 | 21 / **0** |
| S3's | 2 runs agreeing | 26 / **0** | **19 / 2 / 0** | 30 / **0** |
| **R1's + `--tier-agree-seeds 2`** | **2 runs agreeing** | **21** / **0** | **11** / **0** / 0 | **26** / **0** |

Three readings, and the middle one is the honest one.

* **The shipped rule is already clean at seed 0** — 19 correct joins and no false one — and the
  single false join of §4 is at seed 1. Two seeds agreeing removes it and takes the answer to
  **11 correct joins and none wrong**.
* **Agreement is not a substitute for the rule.** S3's rule at two agreeing seeds still confirms
  **two** false joins, which is task A2 §4.3's finding re-measured: a more determined wrong answer
  survives more independent draws. Agreement helps a rule that is already nearly clean; it does not
  rescue one that is not.
* **The price is the whole run again.** The matching stage and the tier pass run once per seed, and
  §6 measures it: on `mixed_all` `--tier-agree-seeds 2` costs **78.6 minutes against 39.5** for the
  ordinary run — **1.99×**, which is what "the matching and the tier pass once per seed" means
  measured. It is a mode a museum turns on for a final answer
  on a collection it is going to publish, and it is off by default for exactly that reason.

**And it halves the recall.** 41 correct joins over two seeds becomes 11 from the pair of them.
A conservator who wants the longer list runs once and reads `probable` with the pictures; a
conservator who wants the shorter list they can put in a catalogue runs twice.

---

## 6. The confirmation runs: the table predicted, the binary agreed

Item 5. Two runs of `input/sfspp/mixed_all` at seed 0, CPU, on an otherwise idle machine, the
default rule in the first and `--tier-agree-seeds 2` in the second:

```
/usr/bin/time -l target/release/sherd-refit-rs run input/sfspp/mixed_all \
    --out output/r1/confirm/<tag> --backend cpu --seed 0 --no-preview --no-meshes \
    --memory-budget 4.83 -v [--tier-agree-seeds 2]
```

| | the shipped default | `--tier-agree-seeds 2` |
|---|---:|---:|
| **confirmed pairs** | **19** | **11** |
| … the table's prediction | **19** | **11** |
| correct / wrong pose / non-adjacent / **cross-object** | 19 / 0 / 0 / **0** | 11 / 0 / 0 / **0** |
| confirmed recall of 288 adjacent pairs | 6.6 % | 3.8 % |
| probable pairs / correct in the two bands | 2 765 / 80 | 2 773 / 80 |
| joins R §8 used / groups (> 1 fragment) / largest | 17 / 12 / 4 | 9 / 6 / 4 |
| fragment accuracy / precision / **group purity** | 20.4 % / 1.000 / **1.000** | 10.6 % / 1.000 / **1.000** |
| **real** | **2 369.4 s (39.5 min)** | **4 717.0 s (78.6 min)** |
| `user / real` | 8.5× | 8.5× |
| preprocess / matching / tiers / **tier_agree** | 10.2 / 1 233.8 / 1 124.1 / — | 10.0 / 1 230.0 / 1 137.6 / **2 338.5** |
| the tier pass's share of the stage clock | **47.5 %** | **73.7 %** (with the agreement runs) |
| peak RSS | 1 986 MiB | 2 237 MiB |

**Both numbers are the table's, to the join.** §4 predicts 19 confirmed pairs at seed 0 with no
false one, and §5 predicts 11 under two agreeing seeds; the binary confirms 19 and 11. That is the
check the offline table has to pass to be worth anything, and it passes on the collection the whole
step is about.

**The wall clock.** 39.5 min against D §10.3's 2 h gate — **3.0×** to spare, and slightly faster
than task A2's 42.3 min under task S3's rule, because the rule only sorts what R §6.5 accepted and
the two runs' matching stages are 1 233.8 s and 1 323.2 s of the same work on a quieter machine.
`--tier-agree-seeds 2` is **1.99×** the default, which is what "the matching and the tier pass once
per seed" means measured; it is still **1.5×** inside the 2 h gate, and D §10.3 now carries both
rows.

**Group purity is 1.000 on both, and cross-object is 0 on both** — audit §E step 12's prohibition,
held at seed 0 for the first time on this collection by a default rule.

---

## 7. The quality gate, and the four defects of task V10

`python tools/quality_gate.py`, shipped defaults, nine development sets × seeds 0–4, 45 runs.
**Exit 0**, wall **930.4 s (15.5 min)** against D §10.4's 25-minute budget.

| set | correct confirmed, by seed | total | **false** | **cross-obj** | recall | S3's total |
|---|---|---:|---:|---:|---:|---:|
| `terracotta` | 2 / 2 / 2 / 2 / 2 | **10** | **0** | 0 | **100 %** | 10 |
| `pot_A` | 4 / 4 / 4 / 3 / 7 | **22** | **0** | 0 | 29.3 % | 26 |
| `pot_B` | 3 / 3 / 4 / 3 / 3 | **16** | **0** | 0 | 21.3 % | 22 |
| `pot_C` | 1 / 1 / 0 / 1 / 0 | **3** | **0** | 0 | 12.0 % | 4 |
| `pot_G` | 0 / 0 / 0 / 0 / 0 | **0** | **0** | 0 | 0.0 % | 0 |
| `pot_H` | 0 / 1 / 0 / 1 / 0 | **2** | **0** | 0 | 2.4 % | 2 |
| `synthetic_20` | 10 / 5 / 7 / 10 / 9 | **41** | **0** | 0 | 16.4 % | 59 |
| `mixed_ABG` | 7 / 7 / 8 / 6 / 10 | **38** | **0** | **0** | 19.0 % | 48 |
| `synthetic_mix3_24` | 11 / 9 / 5 / 8 / 9 | **42** | **0** | **0** | 21.0 % | 54 |
| **total** | | **174** | **0** | **0** | **17.9 %** | 225 |

* **Zero false confirmed joins in all 45 runs**, every prohibition held, and **the terracotta's ten
  (seed, join) slots met** — the row that turns the gate's exit code from 1 to 0, and the row §4 (d)
  shows every "clean everywhere" single-run rule would have lost.
* **Recall is below task S3's on six of the nine sets, and the note owes a reason.** It is §4's
  trade, stated once: the rule that keeps S3's 225 confirms thirteen false joins on the museum's own
  collection, and there is no rule in the table that keeps both. The correct joins lost are not
  discarded — they are in the **probable** band with the test each one failed printed beside them,
  which is what that band is for.
* **The offline table and the binary agree exactly**: 174 and 0, per set 10 / 22 / 16 / 3 / 0 / 2 /
  41 / 38 / 42, which is what §4's table predicts cell for cell.

### 7.1 Task V10's four defects, closed

| # | what it said | what was done |
|---|---|---|
| **V10-D1** | `README.md` described the object-disagreement arm with task O1's numbers, measured under M1's rule and reading as current: nine sets called eight, and "27 of 38 confirmed joins removed on `mixed_ABG` (8/7/7/7/9 → 2/3/2/4/0)" | **re-measured under the shipped rule** — ten runs of `mixed_ABG`, `--object-disagreement off` and `on` at seeds 0–4: **38 confirmed joins (7/7/8/6/10) become 15 (0/7/2/3/3)**, 23 removed, every one correct. The paragraph now carries those numbers, says the gate is nine sets, and says which task measured what |
| **V10-D2** | three blocks in `tools/quality_gate.py` said the terracotta row reaches nine of ten and that the gate fails on it | corrected (R1.1): the tree reaches ten and has since task S3, and the comment says which change closed it |
| **V10-D3** | `--stage rules` printed M1's rule under the name `shipped` and measured every other rule against it | corrected (R1.3): `shipped` is the rule the binary ships, `m1` is M1's, no number in that stage moved |
| **V10-D4** | the review-image caption was described as one line | corrected (R1.1): all three lines are listed, and the fourth `COLOUR:` line that appears on a coloured collection, with its `NOT PART OF THE BAND` said out loud |
| **the observation** | the wide margin's two conjuncts were referenced to two different poses | closed (R1.2), measured in §1: the distance moves on **52.0 %** of the 3 184 accepted candidates of the 45 runs and changes **no** confirmed join under task S3's rule |

---

## 8. Standing gates, on the final tree

| gate | result |
|---|---|
| `cargo fmt --check` | **pass** — no output |
| `cargo clippy --workspace --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo clippy -p sherd-cli --no-default-features --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo build -p sherd-cli --no-default-features` | **pass** |
| `cargo test --workspace` (debug) | **pass** — 19 targets, **456 passed, 0 failed**, 3 ignored |
| `cargo test --workspace --release` | **pass** — 19 targets, **456 passed, 0 failed**, 3 ignored |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, **256 rows** (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed** — the frozen total, unmoved |
| `pytest -q` | **pass** — **60 passed in 76.0 s** |
| `tools/quality_gate.py`, 9 sets × seeds 0–4 | **exit 0** — §7: 930.4 s, **174 correct confirmed joins, 0 false**, terracotta ten of ten |
| byte identity, seed 0, four sets, `--tiers off --objects off`, against a fresh `373520e` built outside the repository | **pass** — **35 187 leaves compared, 0 non-exempt differences**: 8 `engine.commit`, 16 `timings`, 20 `memory`, and four `- <stage>: N.N s` lines in `report.md`. **There is no fifth bucket** |

Three tests were added and three were corrected; the three additions are named in R1.2, R1.4 and
R1.5's own commit messages and the corrections in R1.6's.

---

## 9. What is open

**1. One false confirmed join is left on `mixed_all`, and it is a scorer's boundary as much as a
rule's.** `Pot_I_Piece_07–Pot_I_Piece_11` misses by 1.064° and 0.570 t against `evaluate.py`'s 5°
and 0.5 t. A step that wants it gone without a second run has to find a witness that separates a
pose 0.57 t out from one 0.45 t out, and this table has none. What removes it is
`--tier-agree-seeds 2`, and what would also remove it is a scorer that reported the margin rather
than a verdict — which is a change to `evaluate.py` and not to the algorithm, and is not made here.

**2. The three conjuncts were chosen on a table that contains the collection they are judged on.**
That is task R1's brief and it is also its risk. The mitigation is structural rather than
statistical: **not one of the three is a number**, so there is nothing in them fitted to
`mixed_all`'s particular distances — but "turn this boolean on" is still a choice made with that
collection in view, and the honest statement is that the rule has been *validated* on two
collections (`mixed_ABG`, `synthetic_mix3_60`) and *chosen* with a third in the table. A third real
museum collection would be the thing that settles it.

**3. `strict_on_redraws` costs the most recall of the three and is the least examined.** It takes
the development runs from 196 to 174 correct joins — 22, against the rival verdict's 15 and the
second re-search's 10 — and on `mixed_all` it removes exactly one false join, the cross-object one.
It is a good trade on the axis step 12 gates, and a thin one on every other. Two draws is what the
stability probe happens to make; nobody has asked what one draw or four would do.

**4. The agreement mode reads only the confirmed band of the other run.** A pair the other run put
in the **probable** band at the same placement counts as a disagreement, which is strict and
deliberate — but it means the mode's recall is bounded below by the intersection of two confirmed
bands, and §5 measures that as roughly half. A softer reading ("no other run *contradicts* the
placement") has not been measured and might keep most of the recall; it would also be a weaker
statement, and this step did not have the runs to price it.

**5. Colour still reads nothing, and the collection that needs it still has none.** Unchanged from
task S2 §10 and task A2 §5.2. The two false-join families `mixed_all` has left are Pot_I against
itself, which colour could not see even if the scans carried it.

**6. Task A2 §4.2 has one row that does not reproduce.** §2: `support ∨ (wide ≥ 10, rival ≥ 5 t,
re-search 1)` is printed there as 35 / 32 / 3 and is 40 / 37 / 3 when re-derived from that task's own
saved runs. The false count is the same and no decision rests on the row. It is written down here
because a table nobody re-derives is a table nobody checks.

---

## 10. Housekeeping

**Disk.** 83 GiB free at the start, never below 82 GiB, 85 GiB now. Written and kept under
`output/r1/`: the 49 `--measure` dumps and their report copies (`measure/rules`, 1.1 GB), the rule
table (`measure/r1.{md,json}`), the two confirmation runs without their caches, the quality gate's
table and 45 run files, the object-disagreement pair of gate runs, the parity logs, the
byte-identity runs and their comparison, `acceptance2-check/`, and the seven scripts every number
above came from. Deleted once measured: the `--measure` work directories (the tool removes them),
both confirmation caches, and the two byte-identity caches.

**Runs.** Development matching stayed inside D §12's small-set rule: 45 `--measure` runs for the
table, 45 for the quality gate, 10 for the object-disagreement measurement, 8 for byte identity, and
the test suite's own. **Six** runs above 27 fragments, which is what task R1's text allows and no
more: `synthetic_mix3_60` at seeds 0 and 1 and `mixed_all` at seeds 0 and 1 for the table (item 2),
and `mixed_all` at seed 0 twice for the confirmation (item 5). A seventh was started by mistake and
stopped two minutes in — `--stage rule-runs` with no `--sets` swept every collection the tool knows,
`mixed_all` included; that is R1.3a, and it is why `runnable_sets` now exists.

**Untouched.** `sherd_refit/` (frozen), `input/`, `output/fixtures/`, `fixtures/`,
`output/acceptance/`, `output/acceptance2/`, `output/quality/`, `output/s1`–`s3`, `output/showcase/`,
`output/v10/`. The branch was never switched, nothing was pushed, nothing was stashed.

**Commits.** `R1.1` V10-D2 and V10-D4 · `R1.2` V10's observation · `R1.3` V10-D3 and the rule stage ·
`R1.3a` the small-set rule in the tool · `R1.4` the rule · `R1.5` `--tier-agree-seeds` · `R1.6` the
table read where the binary reads · `R1.7` the documents · `R1.8` this note.
