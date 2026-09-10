# A2 — the second acceptance: task S3's rule on the two large collections

**Date:** 2026-09-11. **Tree:** branch `rust-core`, `1e34e79` (task S3) → `A2.1`…`A2.6`. **Machine:**
Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`, Python 3.12 in
`.venv` (open3d 0.19.0, numpy 2.5.2, scipy 1.18.1). This is the second time this project has matched
a collection above 27 fragments: audit §E's step 12 was task A1's, and task S3 replaced the rule
that step judged, so the step has to be judged again.

**The five sentences.** *On the coloured collection the tier is clean and the object report answers
audit §D.2's question in one column*: `synthetic_mix3_60` — three vessels, 58 fragments, a real
photograph on every sherd — confirms **38 and 41 joins at seeds 0 and 1, every one correct**,
cross-object 0 and group purity 1.000, and the 25 rows of `## Objects` separate the three vessels by
`frac_lab_a` into **2.40–4.13 / 6.64–7.62 / 13.44–17.18** while `thick` overlaps them almost
completely. *On `mixed_all` the shipped rule raises recall by half and does **not** clean the band*:
**62 correct confirmed joins against task A1's 41** over the two seeds — recall 7.1 % → 10.8 % — and
**13 false ones against A1's 10**, so audit §E step 12's prohibition is breached again, at both
seeds, and two of the joins are still Pot_E against Pot_I. *The reason is measured and it is exactly
the thing task S3 §9 item 3 said to re-read*: the rival-distance conjunct that made the rule clean on
the development sets separates **nothing** here — the false joins' second placements sit at a median
of **15.85 t and 14.19 t**, against **18.04 t and 12.59 t** for the correct ones, where on the
development sets the same two medians were 2.98 t and 19.83 t. *Every one of the 13 is on the margin
arm with `support` 0, and `support ≥ 1` alone is still perfectly clean at collection scale* — 16
confirmed joins over the two seeds, all correct, which is the same trade A1 §3.5 priced and it has
not moved. *The re-search does earn its keep here, which the development sets could not show*:
dropping it takes the band from 13 false joins to **15**, and the two it removes are **cross-object**
ones — so task S3 §9 item 2's proposed simplification is refused by this run, at a measured cost of
**569 s, 22 % of a `mixed_all` run** (the tier pass as a whole is now 1 204 s, 47 % of it, against
task A1's 580 s and 32 %).

---

## 0. What was on the tree when I picked the task up

`git status` at `1e34e79` was **clean** — task S3 committed everything it touched — and this task
changed **no crate code at all**. Every run below therefore carries the same binary,
**`1e34e79146092129205fd16d0713b2ddc3d18ae2`**, and `engine.commit` in each `report.json` and
`transforms.json` says so; byte-identity with any of these files is reachable from that commit, or
from any later one after substituting that one 40-character string (the caveat task A1 §1 records
for `486c919`).

---

## 1. What was run, and under what

**Seven large runs, one after another on an otherwise idle machine**, CPU backend, tiers and objects
at their shipped defaults, previews and meshes off:

```
/usr/bin/time -l target/release/sherd-refit-rs run <input> --out output/acceptance2/work/<set> \
    --backend cpu --seed <0|1> --no-preview --no-meshes --memory-budget 4.83 -v
```

Four of them are that command exactly — the shipped rule, `Thresholds::S3` — and three are the same
command with **`--tier-research 0 --resample-seeds 0`**, which prices task S3's open question 2
(§6): it removes the re-search from the rule *and* from the run, so one paired run measures both
what the re-search costs in wall clock and what it changes in the band. The queue was stopped after
the first `mixed_all` paired run: its wall clock is the measurement, and its verdicts are
reproducible offline (§4.1) at both seeds without a second half-hour.

**The seed-0 run of each set is cold** (empty work directory: R §3's whole preprocessing pass runs
and writes `cache/<name>.sherd` per fragment); **the seed-1 run is warm** (same directory, so
R §3.5's three sampled arrays are redrawn and nothing else, R §3.7). `--memory-budget 4.83` is
D §8's row for this machine, in GB of 10^9 — the same number task A1 used, so that the two
acceptances are comparable.

### 1.1 The two collections

| set | input | fragments | pairs matched | ground truth | objects |
|---|---|---:|---:|---|---|
| `synthetic_mix3_60` | `input/synthetic_mix3_60/fragments` | **58** | 1 584 (69 skipped by `--thick-ratio`) | **127** adjacent pairs | **3** — `V012` 19 sherds / 43 pairs, `V049` 19 / 40, `V094` 20 / 44 |
| `mixed_all` | `input/sfspp/mixed_all` | **164** real sherds | 12 612 | 288 adjacent pairs over the 142 fragments with a staged pose; the other 22 have an object but no pose, so a join touching one is `unscorable` | **10 pots** — A 8, B 9, C 7, D 32, E 34, F 7, G 7, H 11, I 30, J 19 |

`synthetic_mix3_60` is task S1's collection and **this is the first time it has been matched**: it is
above D §12's 27-fragment rule, so the small-set rule kept it out of every development step, and it
is the only collection this project has with **three objects and real photographic colour on every
fragment** at once. `mixed_all` is task A1's, unchanged, and it is the collection the whole of
task S3 was aimed at: the ten false confirmed joins this project had produced were all there.

**The search is the same search task A1 ran.** `mixed_all` at seed 0 reports `candidates=60372
accepted=8444` — A1's figures to the candidate — and applying M1's rule offline to this run's own
evidence reproduces A1's confirmed band exactly: **51 confirmed joins, 41 correct, 10 false**
(§4.1). Everything that differs below is task S3's rule and nothing else, which is what a
`--tier-rule m1` run of this collection would have cost 90 minutes to say.

---

## 2. Wall clock and memory, and where the re-search went

`real` and the peaks are `/usr/bin/time -l`'s; the stage columns are `report.json`'s own `timings`,
in seconds. `user / real` is the parallel speed-up the run got out of ten cores.

| set | seed | cache | **real** | user | user/real | preprocess | matching | **tiers** | assembly | objects | refine | **peak RSS** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_mix3_60` | 0 | cold | **131.8 s** (2.2 min) | 992.6 | 7.5× | 16.31 | 94.12 | **19.96** | 0.07 | 0.00 | 1.17 | **1 749 MiB** |
| `synthetic_mix3_60` | 1 | warm | **118.2 s** (2.0 min) | 883.2 | 7.5× | 0.59 | 96.68 | **19.41** | 0.07 | 0.00 | 1.31 | **1 827 MiB** |
| `mixed_all` | 0 | cold | **2 539.1 s** (42.3 min) | 19 883.1 | 7.8× | 10.78 | 1 323.15 | **1 203.74** | 0.06 | 0.00 | 0.82 | **1 413 MiB** |
| `mixed_all` | 1 | warm | **2 626.2 s** (43.8 min) | 20 290.5 | 7.7× | 1.22 | 1 384.71 | **1 238.79** | 0.06 | 0.00 | 0.73 | **1 109 MiB** |

**Both collections are inside D §10.3's gate with room.** `synthetic_mix3_60` at 2.2 min against
≤ 15 min is **6.8×**; `mixed_all` at 42.3 min against ≤ 2 h is **2.8×**. The worst peak resident set
of the four is **1 827 MiB**, 30 % of D §1's 6 GB row, and the memory semaphore never blocked
(`budget_mib=4508`, `waited=0`).

### 2.1 What task S3 cost, against task A1's run of the same collection

A1 ran `mixed_all` at 1 818.5 s cold with `matching` 1 227.3 s and `tiers` 580.5 s. This run is
**2 539.1 s — 39.6 % longer** — and the two stages say where it went:

| stage | task A1 (`486c919`, M1's rule) | **task A2 (`1e34e79`, S3's rule)** | Δ |
|---|---:|---:|---:|
| `matching` | 1 227.3 s | **1 323.2 s** | **+7.8 %** — the wide rival is read off R §5.6's full list inside `match_pair_wide`, before `keep` truncates it |
| `tiers` | 580.5 s (32 % of the run) | **1 203.7 s (47 % of the run)** | **+107 %** — two independent re-searches of every accepted pair |
| whole run, cold | 1 818.5 s | **2 539.1 s** | **+39.6 %** |

**The paired run prices the second row exactly**, because it is the same binary on the same
collection with `--resample-seeds 0`:

| set | seed | `tiers` with two re-searches | `tiers` with none | **the re-search costs** | as a share of `matching` |
|---|---:|---:|---:|---:|---:|
| `synthetic_mix3_60` | 0 | 19.96 s | **8.20 s** | **+11.76 s** | +12.5 % |
| `synthetic_mix3_60` | 1 | 19.41 s | **7.07 s** | **+12.34 s** | +12.8 % |
| `mixed_all` | 0 | 1 203.74 s | **634.27 s** | **+569.47 s** | **+43.0 %** |

On a collection of 58 fragments the re-search is an eighth of the matching stage; on a ten-pot
collection of 164 it is **nearly half of it**, and **47 % of the whole tier pass**. The reason is
the one D §10.3 already records for the tier pass as a whole: the cost is linear in the **accepted**
list, and R §6.5 accepts **8 444** candidates on `mixed_all` against **181** on
`synthetic_mix3_60` — forty-seven times as many, on four times the fragments.

The paired `mixed_all` run's own `matching` stage reads 1 430.5 s rather than 1 323.2 s, because the
byte-identity pass of §10 and two of §4's offline analyses ran during it; its `tiers` stage, which is
the measurement, ran on an idle machine and is the 634.27 s above. The tier pass without the
re-search is still 9 % above task A1's 580.5 s, and that difference is the wide rival's stage-1
fallback plus task S2's colour agreement.

---

## 3. Quality — the two bands, scored by `evaluate.py`

Through `tools/quality_gate.py --score-only`, which is the same `score()` the nine development sets
go through; the table and the JSON are in `output/acceptance2/quality/`. A candidate is classified on
**its own** pose, so a confirmed join R §8 refused is scored too.

### 3.1 The confirmed tier

| set | seed | GT adjacent pairs | confirmed | correct | wrong pose | non-adj | **cross-object** | unscorable | **recall** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_mix3_60` | 0 | 127 | 38 | **38** | 0 | 0 | **0** | 0 | **0.299** |
| `synthetic_mix3_60` | 1 | 127 | 41 | **41** | 0 | 0 | **0** | 0 | **0.323** |
| `mixed_all` | 0 | 288 | 37 | **32** | 4 | 1 | **0** | 0 | **0.111** |
| `mixed_all` | 1 | 288 | 38 | **30** | 4 | 2 | **2** | 0 | **0.104** |

### 3.2 The confirmed and probable bands together

| set | seed | probable | correct | wrong pose | non-adj | cross-object | unscorable | **conf+prob correct** | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_mix3_60` | 0 | 23 | **23** | 0 | 0 | 0 | 0 | **61** | 0.480 |
| `synthetic_mix3_60` | 1 | 21 | **21** | 0 | 0 | 0 | 0 | **62** | 0.488 |
| `mixed_all` | 0 | 2 747 | **48** | 48 | 394 | 2 224 | 33 | **80** | 0.278 |
| `mixed_all` | 1 | 2 785 | **54** | 51 | 384 | 2 264 | 32 | **84** | 0.292 |

**`synthetic_mix3_60`'s probable band is 100 % true joins at both seeds** — 23 of 23 and 21 of 21 —
which is a thing no collection of this project has done before, and it is a statement about the
collection rather than about the tier: three vessels of 19–20 sherds each simply do not offer many
false pairings that clear R §6.5. `mixed_all`'s band behaves exactly as A1 §3.2 measured it: 2 747
rows carrying 48 true joins, four fifths of the rest pairs of *different pots*.

### 3.3 The assembly R §8 built from the confirmed band

| set | seed | fragment accuracy | precision | joins used | correct | wrong pose | non-adj | **cross-object** | **purity** | groups (> 1) | largest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_mix3_60` | 0 | **69.0 %** | **1.000** | 38 | 38 | 0 | 0 | **0** | **1.000** | 25 (7) | 11 |
| `synthetic_mix3_60` | 1 | **77.6 %** | **1.000** | 36 | 36 | 0 | 0 | **0** | **1.000** | 25 (12) | 15 |
| `mixed_all` | 0 | 33.8 % | 0.857 | 35 | 30 | 4 | 1 | **0** | **1.000** | 129 (18) | 6 |
| `mixed_all` | 1 | 33.1 % | 0.784 | 37 | 29 | 4 | 2 | **2** | **0.966** | 128 (22) | 7 |

### 3.4 Against task A1, on the collection both tasks ran

| `mixed_all`, both seeds | task A1 (M1's rule) | **task A2 (S3's rule)** | Δ |
|---|---:|---:|---:|
| confirmed joins | 51 | **75** | +24 |
| **correct** confirmed joins | 41 | **62** | **+21 (+51 %)** |
| **false** confirmed joins | **10** | **13** | **+3** |
| confirmed-tier precision | 0.804 | **0.827** | +0.023 |
| confirmed recall of 288 adjacent pairs | 0.066 / 0.076 | **0.111 / 0.104** | 7.1 % → **10.8 %** |
| cross-object confirmed joins | 0 / **3** | 0 / **2** | −1 |
| group purity | 1.000 / **0.930** | 1.000 / **0.966** | +0.036 |
| correct joins in the **probable** band | 61 / 62 | 48 / 54 | −21, all of them promoted |
| correct joins in the **two bands together** | 80 / 84 | **80 / 84** | **0** |
| fragment accuracy | 21.8 / 23.2 % | **33.8 / 33.1 %** | **+10 pts** |

**Recall rose by half and precision rose by two points, and the prohibition is still breached.** The
brief this cycle works to asks for both halves — more recall *and* more precision — and on the
collection that matters the shipped rule delivers a great deal of the first, a little of the second,
and not the thing step 12 gates on, which is **zero**. Three lines of that table deserve naming
rather than reading past.

* **The last two rows are one fact.** The tier does not *find* joins; it sorts what R §6.5 accepted.
  The two bands together hold **80 and 84 correct joins on both tasks' runs, to the join** — every
  one of the 21 the confirmed band gained came out of the probable band below it. What changed is
  the strength of the claim the tool makes about those joins, which is the whole of what a
  confidence tier is.
* **The cross-object count fell from three to two and purity rose to 0.966.** That is real, it is on
  the one axis step 12 cares most about, and §6 shows it is the re-search's doing.
* **Fragment accuracy is up ten points** — 21.8/23.2 % → 33.8/33.1 % — because R §7 now refines
  35 and 37 joins instead of 21 and 27, and a bigger assembly places more sherds.

---

## 4. The thirteen false confirmed joins, and why the rule did not see them

**Every one of the thirteen is on the margin arm with `support` 0.** That is the same sentence
task A1 §3.4 wrote about its ten, and it is the one thing about this failure that has not moved in
two acceptances.

| seed | verdict | pair | objects | score | support | margin (wide) | **rival distance** | re-search | arm |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | non-adjacent | Pot_D 24 + Pot_D 26 | D/D | 8.86 | **0** | 5.01 | **11.85 t** | 2/2 | margin |
| 0 | wrong pose | Pot_E 13 + Pot_E 27 | E/E | 8.08 | **0** | 3.31 | **15.85 t** | 2/2 | margin |
| 0 | wrong pose | Pot_I 02 + Pot_I 18 | I/I | 27.41 | **0** | 67.90 | **12.87 t** | 1/2 | margin |
| 0 | wrong pose | Pot_I 03 + Pot_I 14 | I/I | 12.09 | **0** | 6.90 | **16.65 t** | 1/2 | margin |
| 0 | wrong pose | Pot_I 07 + Pot_I 08 | I/I | 32.21 | **0** | 6.61 | **19.29 t** | 2/2 | margin |
| 1 | non-adjacent | Pot_D 24 + Pot_D 26 | D/D | 8.87 | **0** | 5.03 | **11.84 t** | 2/2 | margin |
| 1 | non-adjacent | Pot_D 26 + Pot_D 29 | D/D | 5.25 | **0** | 6.99 | **10.30 t** | 1/2 | margin |
| 1 | wrong pose | Pot_E 13 + Pot_E 27 | E/E | 8.48 | **0** | 6.39 | **18.28 t** | 2/2 | margin |
| 1 | **cross-object** | Pot_E 16 + Pot_I 25 | E/I | 6.22 | **0** | 5.87 | **15.27 t** | 2/2 | margin |
| 1 | **cross-object** | Pot_E 23 + Pot_I 20 | E/I | 10.63 | **0** | 5.80 | **5.17 t** | 1/2 | margin |
| 1 | wrong pose | Pot_I 06 + Pot_I 13 | I/I | 18.94 | **0** | 23.57 | **29.17 t** | 1/2 | margin |
| 1 | wrong pose | Pot_I 07 + Pot_I 11 | I/I | 32.88 | **0** | 16.33 | **23.59 t** | 2/2 | margin |
| 1 | wrong pose | Pot_I 19 + Pot_I 21 | I/I | 17.69 | **0** | 3.13 | **13.11 t** | 1/2 | margin |

### 4.1 The rival distance separates nothing here, and that was the open question

Task S3 §3 chose `RIVAL_FAR_T = 5` on a population of 89 false candidates over the 45 development
runs, of which 69 were `pot_H`, and §9 item 3 wrote: *"it has not been measured on a second body of
real sherds, and it should be re-read on the acceptance sets before it is trusted as a constant
rather than as this project's current best guess."* Re-read:

| population | rival distance of **correct** confirmed joins | of **false** confirmed joins |
|---|---|---|
| the 45 development runs, candidates clearing the strict set (S3 §3) | median **19.83 t** | median **2.98 t** — AUC 0.806 |
| `mixed_all` seed 0, the confirmed band | min 5.65 / median **18.04** / max 51.52 (n = 27) | min 11.85 / median **15.85** / max 19.29 (n = 5) |
| `mixed_all` seed 1, the confirmed band | min 1.75 / median **12.59** / max 13 467 (n = 27) | min 5.17 / median **14.19** / max 29.17 (n = 8) |

**The two distributions are on top of each other, and at seed 1 the false joins' median is the
higher of the two.** The wrong-pose family of `pot_H` was a sherd slid three walls along a straight
break; the wrong-pose family of `Pot_I` is a sherd that fits a genuinely different place on the pot,
with its own best alternative a dozen walls away. Both are `wrong_pose` to `evaluate.py` and they
are not the same failure. `RIVAL_FAR_T = 5` was fitted on the first and this collection is the
second.

### 4.2 Every rule, read off these two runs

`evaluate.py`'s verdict applied to **every accepted candidate** of the two runs — the same
classifier `quality_gate.tier_score` uses, over the whole accepted list rather than the two top
bands, so a rule that is not the run's own can be scored too. Each rule is scored the way
`representatives` scores a pair: once, under its best-scoring candidate that clears it.

| rule | confirmed | correct | **false** | where the false ones are |
|---|---:|---:|---:|---|
| **M1's rule** — `support ≥ 1 OR margin(kept) ≥ 2` | 51 | 41 | **10** | 2 non-adj, 5 wrong-pose, **3 cross-object** |
| `margin(kept) ≥ 2` alone | 35 | 25 | 10 | the same ten |
| **S3's rule, shipped** — `support ∨ (wide ≥ 2, rival ≥ 5 t, re-search 1)` | **75** | **62** | **13** | 3 non-adj, 8 wrong-pose, **2 cross-object** |
| S3 without the re-search conjunct | 82 | 67 | **15** | 3 non-adj, 8 wrong-pose, **4 cross-object** |
| S3 with **two** re-searches | 62 | 55 | **7** | 2 non-adj, 4 wrong-pose, **1 cross-object** |
| `support ∨ (wide ≥ 2, rival ≥ 10 t, re-search 1)` | 60 | 48 | 12 | 3 non-adj, 8 wrong-pose, 1 cross-object |
| `support ∨ (wide ≥ 2, rival ≥ 25 t, re-search 1)` | 20 | 19 | **1** | 1 wrong-pose |
| `support ∨ (wide ≥ 5, rival ≥ 5 t, re-search 1)` | 62 | 51 | 11 | 3 non-adj, 6 wrong-pose, 2 cross-object |
| `support ∨ (wide ≥ 10, rival ≥ 5 t, re-search 1)` | 35 | 32 | **3** | 3 wrong-pose |
| **`support ≥ 1` alone** | 16 | **16** | **0** | — |
| `support ≥ 2` alone | 1 | 1 | 0 | — |
| the wide margin arm **alone** | 63 | 50 | 13 | 3 non-adj, 8 wrong-pose, 2 cross-object |

**M1's row is task A1's run to the join** — 51 confirmed, 41 correct, 10 false — which is the check
that the simulation is the rule and not an approximation of it, and it is why no `--tier-rule m1`
run of this collection was needed.

**There is exactly one clean rule on this collection and it is the one A1 §3.5 already priced.**
`support ≥ 1` alone confirms 16 joins over the two seeds, all correct — recall **2.8 %** of the 576
(pair, seed) slots against the shipped rule's 10.8 %. Nothing between the two is clean: raising the
margin to 10 leaves 3 false joins, pushing the rival to 25 walls leaves 1 and takes the band to 20
joins. **Nothing was changed on the strength of this table.** It is a table read off the collection
that judges the rule, and a threshold fitted there is not a threshold; it is here so that the step
which takes the decision has it, exactly as A1 §3.5 is.

### 4.3 Two-seed agreement is weaker than task C measured it

Task C §3 and A1 §5 measured that **none** of A1's ten false confirmed joins was confirmed at both
seeds, and proposed two-seed agreement as the rule that would remove them all. On this run:

| | `synthetic_mix3_60` | `mixed_all` |
|---|---:|---:|
| confirmed at seed 0 / seed 1 | 38 / 41 | 37 / 38 |
| confirmed at **both** | **30** | **21** |
| distinct false confirmed pairs | 0 | **11** |
| of those, confirmed at **both** seeds | — | **2** — `Pot_D 24–26` and `Pot_E 13–27` |
| correct joins confirmed at both seeds | 30 | 19 |

Counted as distinct pairs rather than as (pair, seed) slots, the two runs confirm **54** pairs of
which **43 are correct and 11 false**. Two-seed agreement keeps 21 of the 54 — it removes **9 of the
11** false pairs and **24 of the 43** correct ones — and it is **no longer a clean rule**:
`Pot_D_Piece_24–Pot_D_Piece_26` is a non-adjacent pair confirmed at both seeds with both
re-searches agreeing at both, and `Pot_E_Piece_13–Pot_E_Piece_27` is a wrong-pose join of the same
kind. A more determined wrong answer survives more independent draws, which is the
same lesson §4.1 states about the rival distance.

---

## 5. Colour, on the first collection with three objects and a photograph on every sherd

### 5.1 The object report separates the three vessels, and only the colour column does it

The tool calls **every assembled group an object**, so `## Objects` names 25 of them at either seed
on `synthetic_mix3_60` — 7 assembled groups and 18 single sherds at seed 0, 12 and 13 at seed 1.
What matters is that those 25 rows **partition the three vessels and never mix them**: not one group
at either seed holds fragments of two vessels. Reading the consensus rows across the 25 objects:

| vessel | groups (seed 0 / 1) | sherds | **`frac_lab_a` median, across its groups** | `thick` median, across its groups |
|---|---|---:|---|---|
| `V012` | 13 / 13 | 19 | **2.40 – 4.13** | 4.44 – 10.77 |
| `V094` | 2 / 3 | 20 | **6.64 – 7.62** | 7.99 – 9.54 |
| `V049` | 10 / 9 | 19 | **13.44 – 17.18** | 5.78 – 29.02 |

**Three bands that do not touch on the colour of the clay, and three that lie on top of each other
on the wall.** `V094`'s 7.99–9.54 mm sits inside `V049`'s 5.78–29.02 and overlaps `V012`'s
4.44–10.77. This is audit §D.2's shortlist question answered on a collection that has three objects:
the feature that separates them is the one the SfS++ scans do not carry, and the geometric feature
the scans do carry separates nothing. It is task S2's AUC of 0.985 seen from the other end — not a
number on a sweep, but three rows a conservator reads off `report.md`.

### 5.2 And colour still refuses nothing, for the reason S2 gave

On `synthetic_mix3_60` R §6.5 accepted **zero** cross-object candidates at either seed — not one in
the confirmed band, not one in the probable band, out of the **181 and 176** candidates R §6.5
accepted — so a colour veto has nothing to remove and `--object-demote frac_lab_a` demotes
nothing. On `mixed_all`, which is
where the false joins are, the fragments are SfS++ OBJs with no vertex colours at all: `report.md`'s
`fracture dE` and `shell hist` columns are absent there, and S2 §10's trap is why a *positive*
colour arm must not be written — a collection with no colour reports perfect agreement on every
pair. **The two halves of the colour question are still on two different collections**, and this run
is the sharpest statement of that: the collection where colour separates has nothing to separate,
and the collection that needs separating has no colour.

---

## 6. The re-search: task S3's open question 2, answered

S3 §9 item 2 wrote: *"the re-search is the expensive half and it earns one false join … If the
acceptance run shows the re-search removing nothing on `mixed_all` either, the honest next step is
to drop it and keep the rival-distance conjunct alone."* Both halves of that are now measured, and
they point opposite ways.

**What it costs** is §2.1's paired runs: **569 s on `mixed_all`** — 43 % of the matching stage,
47 % of the tier pass and 22 % of the whole run — and 12 s on `synthetic_mix3_60`. It is the single
most expensive thing this project added in the S-cycle.

**What it buys**, from §4.2's table, is the row a museum cares about most:

| | confirmed | correct | false | **of which cross-object** |
|---|---:|---:|---:|---:|
| `support ∨ (wide ≥ 2, rival ≥ 5 t)` — **no re-search** | 82 | 67 | **15** | **4** |
| `support ∨ (wide ≥ 2, rival ≥ 5 t, re-search 1)` — **shipped** | 75 | 62 | **13** | **2** |
| `support ∨ (wide ≥ 2, rival ≥ 5 t, re-search 2)` | 62 | 55 | **7** | **1** |

**Dropping the re-search costs 2 false joins and both of them are cross-object**, which is the
failure step 12 states as a prohibition and the one roadmap item 4 exists for. On the development
sets the conjunct removed one wrong-pose join of `pot_H` for five correct ones and that was a thin
justification; on a real ten-pot collection it removes two Pot_E–Pot_I joins for five correct ones,
and that is not thin. **S3's proposed simplification is refused by this run.**

**The offline table and the paired run agree to the join, which is why one paired run was enough.**
At seed 0 the run made with `--tier-research 0 --resample-seeds 0` confirms **40 joins, 34 correct,
6 false, one of them cross-object, group purity 0.983** — and §4.2's simulation of the same rule on
the *shipped* run's own evidence predicts 40, 34, 6 and the same one cross-object join
(`Pot_D_Piece_16–Pot_I_Piece_23`). The same check on `synthetic_mix3_60` gives 38 and 41 confirmed
joins with and without the re-search alike, at both seeds. So the second seed of the paired run was
not run: its wall clock would have repeated the first's and its verdicts are already known exactly.

The second row of that table is the one a next step should read. Asking for **both** re-searches
takes the band from 13 false joins to 7 and from 62 correct to 55 — it halves the false joins for
one correct join in nine, and it takes the cross-object count to 1. It is not shipped here, for the
reason §4.2 gives: `--tier-research 2` fitted on the collection that judges it is not a threshold.
But unlike the rival distance it is a knob that already exists, it costs nothing extra to compute on
a run that already performs two re-searches, and S3 §5 measured its cost on the development sets as
19 correct joins for no false one. It is the first thing to put on a development-set table.

---

## 7. The showcase run, for the bench rather than for a table

One ordinary run of `synthetic_mix3_24` — the 24-fragment twin of the coloured collection — with
**everything on**: meshes, previews and review images, nothing suppressed.

```
target/release/sherd-refit-rs run input/synthetic_mix3_24/fragments \
    --out output/showcase/synthetic_mix3_24 --backend cpu --seed 0 --review-images -v
```

**`output/showcase/synthetic_mix3_24/` — 563 MB, 36.1 s, kept.** What is in it:

| | |
|---|---|
| `report.md` (40 kB) | the whole worksheet: `## Objects` with the three clay bodies, `## Confirmed joins` with the generated rule paragraph and the `arm` column, `## Probable joins`, `## Rejected` with a reason per pair, and `## Candidates by fragment` with an `image` link per row |
| `review/` (18 PNGs, 5.5 MB) | one per confirmed and probable join — three views, A grey and B orange, the seam's voxels white, B's fracture samples coloured by distance, and a caption carrying the scores, the band, both margins, the re-search and **ARM** |
| `assembly_0..3.ply` (169 MB) | the four assembled groups — 5, 4, 3 and 2 fragments — each a single mesh carrying the scans' own vertex colours |
| `preview_0..3.png`, `preview_segmentation.png` | four views of each group, and R §3.4's shell/fracture segmentation |
| `placed/` (24 PLYs, 303 MB) | every fragment at its found pose, colours included |

The run matches 275 pairs, finds 505 candidates of which R §6.5 accepts 56, confirms **12 joins and
lists 6 as probable**, and assembles four groups (5 + 4 + 3 + 2 of 24 fragments); its stage clock
is preprocess 10.09 s, matching 15.26 s, tiers 6.01 s, review 0.59 s, refine 1.02 s. Scored for the
record — it is a development set, so this costs nothing the small-set rule forbids — **12 confirmed
joins, all correct, 0 false, cross-object 0, purity 1.000, fragment accuracy 58.3 %**, confirmed
recall 30.0 % of the set's 40 adjacent pairs and 45.0 % with the probable band. It is the artefact
to open first: the review images are what audit §D.1 built the band for, and this is the only
collection where they show a photograph rather than one flat terracotta.

---

## 8. The development quality gate, on this tree

`tools/quality_gate.py`, the nine development sets at seeds 0–4, shipped defaults, the same binary
every run above carries. **Exit 0**, and every cell reproduces task S3's to the join — the gate is
deterministic given a binary and a seed, and this task changed no crate code.

**45 runs in 988.0 s (16.5 min)**; `output/acceptance2/quality-dev/quality.{md,json}` has all 45
rows.

| set | adjacent pairs (5 seeds) | correct confirmed, by seed | total | **false** | recall | correct in both bands |
|---|---:|---|---:|---:|---:|---:|
| `terracotta` | 10 | 2 / 2 / 2 / 2 / 2 | **10** | **0** | 100.0 % | 10 (100.0 %) |
| `pot_A` | 75 | 6 / 5 / 4 / 4 / 7 | **26** | **0** | 34.7 % | 39 (52.0 %) |
| `pot_B` | 75 | 6 / 4 / 4 / 3 / 5 | **22** | **0** | 29.3 % | 56 (74.7 %) |
| `pot_C` | 25 | 1 / 1 / 0 / 1 / 1 | **4** | **0** | 16.0 % | 9 (36.0 %) |
| `pot_G` | 50 | 0 / 0 / 0 / 0 / 0 | **0** | **0** | 0.0 % | 0 (0.0 %) |
| `pot_H` | 85 | 0 / 1 / 0 / 1 / 0 | **2** | **0** | 2.4 % | 14 (16.5 %) |
| `synthetic_20` | 250 | 10 / 11 / 11 / 13 / 14 | **59** | **0** | 23.6 % | 119 (47.6 %) |
| `mixed_ABG` | 200 | 12 / 9 / 8 / 7 / 12 | **48** | **0** | 24.0 % | 95 (47.5 %) |
| `synthetic_mix3_24` | 200 | 12 / 10 / 7 / 12 / 13 | **54** | **0** | 27.0 % | 86 (43.0 %) |
| **total** | **970** | | **225** | **0** | **23.2 %** | **428 (44.1 %)** |

**Zero false confirmed joins in all 45 runs, every prohibition held, and the terracotta's two museum
joins confirmed in all ten (seed, join) slots** — the row audit §E's step 8 states, which the gate
failed by design from task G until task S3 and now passes. `pot_G` is 0 on both bands and R §13 says
it should be. The gate's own line reads *"every set holds R §13's prohibitions"*.

**The development sets and `mixed_all` now say opposite things about the same rule, and that is the
finding of this task.** 45 runs of nine collections: 225 correct, **0 false**. Two runs of one real
ten-pot collection: 62 correct, **13 false**. Nothing is wrong with either measurement; what is wrong
is the inference from the first to the second, and §4.1 names the quantity that breaks it.

---

## 9. Standing gates, on the final tree

| gate | result |
|---|---|
| `cargo fmt --check` | **pass** — no output |
| `cargo clippy --workspace --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo clippy -p sherd-cli --no-default-features --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo build -p sherd-cli --no-default-features` | **pass** |
| `cargo test --workspace` (debug) | **pass** — 19 targets, **453 passed, 0 failed**, 3 ignored |
| `cargo test --workspace --release` | **pass** — 19 targets, **453 passed, 0 failed**, 3 ignored |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, **256 rows** (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed** — the frozen total, unmoved |
| `pytest -q` | **pass** — **60 passed in 78.0 s** |
| `tools/quality_gate.py`, 9 sets × seeds 0–4 | **exit 0** — 988.0 s (16.5 min), 45 runs, **225 correct confirmed joins, 0 false**, 23.2 % of the 970 adjacent pairs, 428 correct with the probable band (44.1 %). Every prohibition holds and the terracotta's ten slots are met (§8) |
| byte identity, seed 0, four sets, before and after this task's commits | **pass** — §10 |

This task changed **no crate code**: the whole diff against `1e34e79` is `README.md`,
`docs/superpowers/specs/2026-09-06-rust-core-design.md` and this note. That is why the quality gate
reproduces task S3's 225/0 to the join and the parity total is the frozen one to the check. The
eight gates above were run at `2a34ffd`, before this note was committed; no test in the workspace
reads a Markdown file (the only `include_str!` in the tree pulls in WGSL kernels) and `pytest` reads
none either, so a Markdown-only commit on top cannot move any of them.

---

## 10. Byte-reproducibility

Seed 0, CPU, `--no-preview --no-meshes`, the four standing sets, twice with the shipped defaults and
twice with `--tiers off`: once **before** this task's commits (binary stamped `1e34e79`, the binary
every run above carries) and once **after** them and a rebuild (stamped `2a34ffd`). 12 files per
comparison — `report.json`, `transforms.json` and `report.md` × 4 sets.

| comparison | bucket | leaves | where |
|---|---|---:|---|
| `--tiers off`, pre vs post | `engine.commit` | 8 | the binary was rebuilt at a new commit |
| | `timings` / `memory` | 20 / 24 | wall clock and RSS |
| | `report.md` timing lines | 13 | the same, printed |
| the shipped defaults, pre vs post | `engine.commit` | 8 | as above |
| | `timings` / `memory` | 23 / 27 | as above |
| | `report.md` timing lines | 16 | as above |

**No fourth bucket, and no leaf anywhere else** — every differing `report.md` line is a
`- <stage>: N.N s` line, checked one by one. That is the strongest form the statement can take here
and it follows from the tree: `git diff --name-only 1e34e79..HEAD -- crates/ Cargo.* rust-toolchain.toml tools/ sherd_refit/`
is **empty**. This task changed two Markdown files and added a third; nothing that computes anything
moved, which is why the quality gate reproduces task S3's 225/0 and parity reproduces the frozen
23 804/0. The pre-pass numbers are the slower of the two because the `mixed_all` paired run of §2.1
was still in flight while it ran.

---

## 11. What the documents now say

**D §10.3** keeps task A1's rows and adds this task's beside them, with a sentence saying that
`synthetic_60` and `synthetic_170` were *not* re-run and that their rows are A1's under M1's rule.
The new material is the four-run timing table with its two gate margins (6.8× and **2.8×**, against
A1's 4.0× on the same collection), the stage-by-stage account of the 39.6 % the S3 rule costs, the
quality paragraph with both directions in it, and the paragraph that names what the run refutes —
`RIVAL_FAR_T` — and what it confirms — the re-search. The benchmark-gate table gains a
`synthetic_mix3_60` row and a pointer on the `mixed_all` row.

**D §10.4** had two rows that had been wrong since task S1 and task S3 respectively: the gate is
**45 runs of nine sets in 16.5 min**, not 40 of eight in 10.0–10.3, and it **exits 0** — audit §E's
step-8 terracotta row is ten of ten slots, and the entry now says which algorithm change closed it
(task G predicted it would need one; task S3's wide rival is it). `--score-only`'s list of
acceptance collections gains `synthetic_mix3_60`.

**D §11** item 4 said "the measurement said none" about vetoing features and stopped at M1's 0.740.
It now records the whole colour story in one cell: `frac_lab_a` at 0.985/0.990, the split shell and
fracture colours in `CACHE_VERSION` 7, and the three separate measured reasons it still vetoes
nothing.

**D §12** gains **step 12a**, the second acceptance, with its exit criterion and its risk column;
step 8's recall figures are re-stated (23.2 % on the development sets, 29.9–32.3 % and 10.4–11.1 %
on the two acceptance collections) and its exit criterion is corrected to nine sets and to a
terracotta row that is now met — with the sentence that neither half transfers to `mixed_all`.

**README** (Russian, museum sections). The colour paragraph claimed colour could not be measured at
all; it now carries S1's collections, S2's AUC and three measured bullets for why colour still
refuses nothing. `#### Настоящая коллекция` is re-measured: a two-column table with
`synthetic_mix3_60` beside `mixed_all`, the six numbered points on the new figures, and point 6
re-written around the thing that changed direction — two of the eleven false joins are confirmed at
*both* seeds where task C measured none of ten. The benchmark table gains the new collection and the
new `mixed_all` row with a paragraph on where the 40 % went, and the quality-gate section is
corrected on three stale facts (nine sets, forty-five runs, exit 0).

---

## 12. What is open

**1. The rule is chosen on the wrong population, and this run is the proof.** Nine development
collections at five seeds say 225 correct and **0** false; one real ten-pot collection at two seeds
says 62 correct and **13** false. The development sets contain one wrong-pose family — `pot_H`'s
slide along a straight break — and `mixed_all` contains a different one — `Pot_I` fitting a
genuinely wrong place on its own pot — and a threshold fitted on the first does not see the second
(§4.1). The next rule step should not be taken on `tools/quality_gate.py` alone. The cheapest fix
available is not a new collection: it is to add **`mixed_all`'s two runs as a scored, non-gating
row** of whatever table chooses the next rule, exactly as `tools/measure_tiers.py --stage rules`
already does for the development runs — one 42-minute run per seed, against the 90 minutes this task
spent discovering the gap by hand.

**2. `support ≥ 1` alone is the only clean rule anyone has found at collection scale, twice.**
A1 §3.5 measured it removing all ten false joins for 124 of 189 correct ones; this task measures it
removing all thirteen for 46 of 62. It is not shippable at 2.8 % recall and it is not nothing
either: **the support arm has never produced a false confirmed join on any collection, at any seed,
in either acceptance.** A tier with *two* confirmed grades — "supported" and "argued" — is a design
this project has not considered and the data now asks for it.

**3. `--tier-research 2` is the one knob that would help here and it has never been on a table with
`mixed_all` in it.** §6: two agreeing re-searches take the band from 13 false joins to 7 and the
cross-object count from 2 to 1, for 7 of the 62 correct joins. S3 §5 priced it on the development
sets at 19 correct joins for no false one. It costs nothing extra to compute. It was not turned on
here because this collection is the judge, not the fitting set.

**4. Two-seed agreement is no longer clean and the museum should be told before it is automated.**
§4.3: two of the eleven false pairs survive both seeds, both with re-searches agreeing at both. The
README now says so. An automatic two-seed mode is still worth building — it removes 9 of 11 — but it
must not be sold as a guarantee.

**5. The re-search costs 569 s on `mixed_all` — 22 % of the run — to remove two cross-object joins.**
That is the honest statement of a trade nobody has been asked to accept. A museum that runs two
seeds anyway may prefer to spend those 19 minutes on the second seed instead; a museum that runs
once may not. It is a `--resample-seeds` flag either way, and the number is now on record.

**6. Colour's two halves are still on two different collections.** §5.2. Nothing in this project can
close that except colour on a real mixed collection — audit §C.4's museum acceptance set, or a
re-scan of the SfS++ sherds. `frac_lab_a` is ready, measured and switched off, waiting for material.

**7. `synthetic_60` and `synthetic_170` were not re-run under the S3 rule.** Their rows in D §10.3
are A1's. Nothing suggests they would fail — both were clean under the weaker rule and the S3 rule
is strictly harder on the margin arm — but "nothing suggests" is not a measurement, and the wall
clock would certainly have moved (`synthetic_170` accepted only 464 candidates, so its tier pass
should grow far less than `mixed_all`'s did).

**8. The `mixed_all` gate margin is 2.8× and falling.** A1 had 4.0×; the S3 rule spent 30 % of it.
The next thing added to the tier pass has 47 minutes of headroom on a 164-fragment collection, and
the cost is linear in the accepted list, which on a real mixed collection is eighteen times longer
than on a synthetic one of the same size (D §10.3).

---

## 13. Housekeeping

**Disk.** The task started with 8.4 GiB free and never fell below **8.2 GiB**; it stands at
**10 GiB** now. `input/`, `output/fixtures/`, `fixtures/` and the earlier tasks' measured trees
(`output/acceptance/`, `output/s1`–`s3`, `output/quality`) were not touched. Nothing was written
under `input/`.

**What is kept**, gitignored:

| path | size | what |
|---|---:|---|
| `output/acceptance2/mixed_all/seed{0,1,0_nore}/` | 210 MB | `report.json`, `report.md`, `transforms.json` and the `/usr/bin/time -l` block of each run |
| `output/acceptance2/synthetic_mix3_60/seed{0,1,0_nore,1_nore}/` | 19 MB | the same |
| `output/acceptance2/logs/` | 13 MB | every run's `-v` stage log, the quality-gate log and the eight standing-gate logs |
| `output/acceptance2/quality{,-nore,-dev,-showcase}/` | 1.0 MB | `quality.{md,json}` and the per-run `evaluate.py` dumps of every table above |
| `output/acceptance2/byteid/{pre,post}/` | 4.6 MB | §10's twelve-file comparison, both passes |
| **`output/showcase/synthetic_mix3_24/`** | **563 MB** | **§7 — the run to open**, meshes, previews and review images included |

**What was deleted after it was measured**: the fragment caches of both acceptance collections
(they are rebuilt in 11 s and 16 s), and the duplicate copies of the three reports the shared work
directories held. The showcase run keeps its cache so that re-running it is instant.

**One disclosure about the timings.** The eight large runs were made on an otherwise idle machine
with two exceptions, both named where they matter: a 25-second single-core read of `report.json`
(§4's first check) ran during `mixed_all` seed 1's matching stage, and §10's byte-identity pre-pass
ran during the paired run's matching stage, which is why that stage reads 1 430.5 s rather than
1 323.2 s. **No `tiers` stage — the measurement this task turns on — had anything else running
beside it.** Task A1 measured 1 818.5 s and 1 815.3 s for two identical `mixed_all` runs, so this
machine's run-to-run noise is about 0.2 %.

**The queue was stopped after the first paired `mixed_all` run** rather than running the second: its
wall clock would have repeated the first's, and §6 shows the offline simulation predicts its verdicts
exactly — 40 confirmed, 34 correct, 6 false, one cross-object, which is what the run that *was* made
produced. That saved 35 minutes and cost nothing measurable.

**Nothing was tuned.** Every threshold in force is the one task S3 shipped. §4.2's rule table and
§6's re-search table are read off the collection that judges the rule and are recorded, not acted
on — the same discipline A1 §3.5 states.

`sherd_refit/` was not touched. The branch was never switched, nothing was pushed, nothing was
stashed, and nothing was sent anywhere.

**Commits.** `A2.1` D §10.3, §10.4, §11 and §12; `A2.2` the README's museum sections; `A2.3` a
one-word correction to step 12a's run count; `A2.4` this note; `A2.5` the cache version the object
features live in, which the README still gave as 6 and which task S2 moved to 7; `A2.6` this list.
