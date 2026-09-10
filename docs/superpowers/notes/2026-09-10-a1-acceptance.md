# A1 — the final acceptance: the three large collections, measured

**Date:** 2026-09-10. **Tree:** branch `rust-core`, `486c919` (task G) → `A1.1`…`A1.4`. **Machine:**
Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`. This is step 12
of the audit's plan §E (`notes/2026-09-09-fable-audit.md`) and of D §12 — **the only step that
matches a collection above 27 fragments**, and the step at which D §10.3's three projected rows
become measurements.

**The four sentences.** *Runtime and memory are met with room to spare*: 2.1, 18.5 and 30.3 minutes
cold against gates of 15 min, 2 h and 2 h — 7.0×, 6.5× and 4.0× — and a worst peak resident set of
2 647 MiB against D §1's 6 GB, with the memory semaphore never blocking once. *The confirmed tier is
clean on both synthetic collections*: 148 confirmed joins over four runs, every one `correct`,
cross-object 0 and group purity 1.000 — including on `synthetic_170`, which carries six pieces of a
second vessel so that a cross-object join is possible there. *On `mixed_all` — 164 real sherds of
ten pots — it is not*: 10 of the 51 confirmed joins over two seeds are false, three of them
cross-object at seed 1, which takes purity to 0.930 and **breaches step 12's prohibition**; every
one of the ten was confirmed on the margin arm with `support` 0, and the three cross-object ones are
roadmap item 4's empty feature shortlist arriving where it matters — the Pot_I sherd of each impure
group agrees with its Pot_E group to within that group's own MAD on every feature the cache carries.
*Nothing was tuned on the acceptance sets*: the one rule that removes all ten — requiring
`support ≥ 1` — was simulated exactly and costs 124 of the 189 correct confirmed joins, and the
number is recorded for the step that takes the decision rather than acted on here.

The two things D could not have known before this run: the tier pass costs **32 % of `mixed_all`**
because R §6.5 accepts 8 444 candidates there, and D §8's 4.4 GB BVH extrapolation is untested,
because it assumes 34 M working-mesh faces and the largest of these collections has 9.0 M.

---

## 1. What was run, and under what

Six runs, CPU backend, tiers and objects at their shipped defaults, previews and meshes off, one
after another on an otherwise idle machine:

```
/usr/bin/time -l target/release/sherd-refit-rs run <input> --out output/acceptance/<set> \
    --backend cpu --seed <0|1> --no-preview --no-meshes --memory-budget 4.83 -v
```

**The binary all six runs carried is stamped `486c91906ae1348acd8f1492d5b1eef2064db968`** — task
G's HEAD, not any commit of this task: it was built at the start of the step and this task changed
no crate code, so nothing rebuilt it. `engine.commit` in each `report.json` and `transforms.json`
says so, and this matters to anyone re-running them: **byte-identity with these files is reachable
only from `486c919`**, or from any later commit after substituting that one 40-character string.
Task V9 did exactly that and reproduced both large collections — `transforms.json` byte for byte
and the whole of `report.json` — from `ec72ad4` (V9 §2.2, V9-D19). This task's own forty-run gate
(§6) was made with a binary stamped `58704d5`, its second commit.

**The seed-0 run of each set is cold** — the work directory is empty, so R §3's whole preprocessing
pass runs and writes `cache/<name>.sherd` for every fragment. **The seed-1 run is warm**: the same
work directory, so the meshes, the labels and the breaklines are read from the cache and only
R §3.5's three sampled arrays are redrawn (R §3.7). That is the same cold/warm pair D §10.3's
measured table uses on the development sets, and `report.json`'s `timings` block separates it
exactly: the cold−warm difference is the `preprocess` stage and nothing else.

**`--memory-budget 4.83`** is D §8's row read for this machine. The flag is in GB of `10^9`
(`Budget::gigabytes`), and D §8 says that holding the "≤ 6 GB peak RSS" row with 170 fragments
resident wants a budget of **≈ 4.5 GiB** — 4.83 × 10^9 bytes. The default would have been half of
physical memory, 8.6 GB, which is the budget D §8 says is too generous at this collection size.

### 1.1 The three collections

| set | input | fragments | pairs | ground truth | objects |
|---|---|---:|---:|---|---|
| `synthetic_60` | `input/synthetic_pingsdorf_60/fragments` | **56** | 1 540 | 143 adjacent pairs | 1 (one pot) |
| `synthetic_170` | `input/synthetic_pingsdorf_170/fragments` | **164** | 13 366 | 407 adjacent pairs | **2** — 158 pieces of the pot plus **6 `intruder_*` pieces of another vessel**, which is what makes cross-object joins measurable on it |
| `mixed_all` | `input/sfspp/mixed_all` | **164** real sherds | 13 366 | 288 adjacent pairs over the 142 fragments that have a staged pose; the other **22 have an object but no pose**, so a join that touches one is `unscorable` and never `correct` | **10 pots** — A 8, B 9, C 7, D 32, E 34, F 7, G 7, H 11, I 30, J 19 |

Two of the three therefore test the prohibition step 12 is about. `synthetic_60` is a single pot and
its purity is 1.000 by construction; what it tests is the wall clock and the tier's own
zero-false-joins rule. D §10.3's row for `mixed_all` says 12 589 pairs, which is the pair count
*after* R §4.3's thickness screen; the collection itself has 13 366.

`synthetic_170`'s six intruders are **not adjacent to each other** in the ground truth — they are
distractors, not a second pot to reassemble — so every cross-object join it can produce pairs an
intruder with a piece of the main pot.


---

## 2. Wall clock, and where it goes

Two clocks are reported and they answer different questions. **`real`** is `/usr/bin/time -l`'s,
the process from `exec` to exit — what a museum waits. **The stage sum** is `report.json`'s
`timings` block, the pipeline's own accounting; the two differ by process start-up and by the JSON
and Markdown writers, which is under a second on every run below. `user` is the CPU time the ten
cores did between them, so `user / real` is the parallel speed-up the run actually got.

### 2.1 The six runs

`real` and the peaks are `/usr/bin/time -l`'s; the stage columns are `report.json`'s `timings`,
in seconds.

| set | seed | cache | **real** | user | user/real | preprocess | matching | tiers | assembly | objects | refine | **peak RSS** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_60` | 0 | cold | **128.1 s** (2.1 min) | 1 072.3 | 8.4× | 12.6 | 106.3 | 8.6 | 0.02 | 0.00 | 0.5 | **2 266 MiB** |
| `synthetic_60` | 1 | warm | **118.5 s** (2.0 min) | 994.0 | 8.4× | 0.5 | 109.0 | 8.4 | 0.02 | 0.00 | 0.5 | **2 288 MiB** |
| `synthetic_170` | 0 | cold | **1 107.0 s** (18.5 min) | 9 449.0 | 8.5× | 17.9 | 1 071.3 | 16.8 | 0.06 | 0.00 | 0.6 | **2 569 MiB** |
| `synthetic_170` | 1 | warm | **1 104.4 s** (18.4 min) | 9 404.3 | 8.5× | 1.3 | 1 085.5 | 16.6 | 0.05 | 0.00 | 0.6 | **2 715 MiB** |
| `mixed_all` | 0 | cold | **1 818.5 s** (30.3 min) | 15 493.5 | 8.5× | 9.8 | 1 227.3 | **580.5** | 0.04 | 0.00 | 0.5 | **1 754 MiB** |
| `mixed_all` | 1 | warm | **1 815.3 s** (30.3 min) | 15 517.5 | 8.5× | 1.0 | 1 227.9 | **585.4** | 0.03 | 0.00 | 0.6 | **1 795 MiB** |

Six runs, **101.5 minutes** of wall clock between them.

**Cold minus warm is the preprocessing stage and nothing else** — 12.1 s on `synthetic_60`, 16.6 s
on `synthetic_170`, 8.8 s on `mixed_all` — so on collections whose matching stage is a quarter of an
hour the cache is worth about one per cent of the run, and the cold row is the one to quote. It is
small because these fragments are small (§2.3) and because the second seed still re-reads the cache
and redraws R §3.5's three arrays; that redraw is what the warm `preprocess` cell (0.5–1.3 s)
measures.

**The pool returns 8.4–8.5× of ten cores** on every one of the six, against the **6.43×** D §10.3's
projection assumed. That is measured on runs an order of magnitude longer than the ones the
6.43× came from, so it is the better number of the two.

### 2.2 Against D §10.3's projections

| set | D §10.3 projected (CPU) | measured, cold | measured, warm | CPU gate | margin |
|---|---:|---:|---:|---:|---:|
| `synthetic_60` | 2.4 min | **2.1 min** | **2.0 min** | ≤ 15 min | **7.0×** |
| `synthetic_170` | 17 min | **18.5 min** | **18.4 min** | ≤ 2 h | **6.5×** |
| `mixed_all` | 28 min | **30.3 min** | **30.3 min** | ≤ 2 h | **4.0×** |

**Every row is inside its gate**, and the tightest of the three still has four times the room it
needs. The projections were of the *matching stage* alone, from a per-pair cost, and they are worth
comparing stage to stage:

| set | pairs matched | projected matching | measured matching | per pair, measured |
|---|---:|---:|---:|---:|
| `synthetic_60` | 1 540 | 2.4 min | **1.8 min** | 0.069 s wall, 0.58 core-s |
| `synthetic_170` | 12 839 | 17 min | **17.9 min** | 0.083 s wall, 0.71 core-s |
| `mixed_all` | 12 612 | 28 min | **20.5 min** | 0.097 s wall, 0.83 core-s |

The per-pair costs the projection used — 0.523 core-s synthetic, 0.870 core-s real — are **+10.5 %,
+36.2 % and −4.7 %** of what the three runs measured (0.58, 0.71 and 0.83 core-s in the table above)
at a hundred times the pair count. *(Corrected in task C, V9-D11: this paragraph read "within 11 %
and 5 %", which is `synthetic_60` and `mixed_all`; the collection it is about, `synthetic_170`, is
36 % off. What held to 5–12 % is the end-to-end **wall** prediction, and it held because the
per-pair cost was 36 % low and the 6.43× pool speed-up 33 % low in the other direction.)*

What the projection could not see is the stage that did not exist when it was written: **R §6.5's
accepted list on a real ten-pot collection is 8 444 candidates, and the tier pass probes every one
of them, which costs 580 s — 32 % of `mixed_all`'s run.** On `synthetic_170` the same stage is
16.8 s, because **464** candidates were accepted there rather than 8 444. So the tier's cost is
linear in the *accepted* list and not in the pair count, and on real sherds the accepted list is
**eighteen times** longer. Per accepted candidate the probe costs **36 ms** there and **69 ms**
here, so a real ten-pot collection is about twice as dear a candidate *and* has eighteen times as
many of them. D §12's step-8 risk column asked that the probes stay "a few per cent of a pair"; per
candidate they do — 580 s over 8 444 is 69 ms — but the collection-level share is a third of the
run, and that is worth knowing before the next collection. *(Corrected in task C, V9-D4: this
paragraph said 145 and "fifty times". 145 is `synthetic_170`'s count of pair **representatives**,
51 confirmed + 94 probable — the unit of §3.1's table — while 8 444 is a **candidate** count; the
run's own log line reads `matching done … candidates=54045 accepted=464`, and `report.json` holds
150 confirmed + 314 probable candidates.)*

### 2.3 Memory

| set | seed | preprocess | matching | tiers | assembly | objects | refine | **peak** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_60` | 0 | 1 400 | 2 066 | **2 266** | 2 256 | 2 256 | 2 131 | **2 266 MiB** |
| `synthetic_60` | 1 | 509 | 1 901 | **2 287** | 2 210 | 2 210 | 2 100 | **2 287 MiB** |
| `synthetic_170` | 0 | 1 034 | 2 074 | **2 569** | 2 362 | 2 362 | 2 472 | **2 569 MiB** |
| `synthetic_170` | 1 | 632 | 2 217 | 2 553 | 2 560 | 2 560 | **2 645** | **2 647 MiB** |
| `mixed_all` | 0 | 694 | 1 563 | **1 754** | 1 269 | 1 269 | 1 473 | **1 754 MiB** |
| `mixed_all` | 1 | 415 | 1 503 | **1 794** | 1 344 | 1 344 | 1 472 | **1 794 MiB** |

MiB, `memory::RssMonitor` at 100 ms, from `report.json`'s `memory` block; the process-level peak
`/usr/bin/time -l` reports agrees with it to within 70 MiB on every run.

**The worst of the six is 2 647 MiB — 44 % of D §1's ≤ 6 GB row**, and the gate holds with more
than twice the room it needs. Three things are worth saying about *why*, because two of them make
this a weaker test than D §8 expected:

* **The peak is the tier pass's, not matching's**, on five of six runs. H3 measured matching as the
  peak on `synthetic_20`; the tier pass did not exist then. The probes hold two independent
  resamples of R §3.5's arrays while the fracture BVHs of the pair are still live.
* **§8's 4.4 GB whole-mesh BVH total is an extrapolation from 170 × 200 000 faces, and no real
  170-fragment collection here reaches that face count.** Measured working-mesh totals:
  `synthetic_60` **6.48 M** faces (mean 116 k), `synthetic_170` **9.01 M** (mean 55 k), `mixed_all`
  **4.66 M** (mean 28 k) — against the 34 M the extrapolation assumed. At H3's measured 129 B a
  face that is 0.84, 1.16 and **0.60 GB** of whole-mesh tree, not 4.4 GB. The per-face constant is
  not challenged by this run; the face count is, and D §8 should say so.
* **The memory semaphore never blocked.** `budget_mib=4508` (the flag's 4.83 GB less D §5's 98 MiB
  process floor) on every run, `waited=0` on every one, with peak concurrency 12–17 in preprocessing
  and 9 in R §9's refinement, holding 360–1 143 MiB. A budget of 4.5 GiB is not binding at this
  fragment size, which is the same statement as the bullet above from the other end.


---

## 3. Quality — the confirmed tier and the confirmed-plus-probable list

Scored by `tools/evaluate.py` through `tools/quality_gate.py --score-only` (§A1.1), which is the
same `score()` the eight development sets go through. A candidate is classified on **its own** pose,
so a confirmed join R §8 refused is scored too. The table and the JSON are in
`output/acceptance/quality/`.

### 3.1 The confirmed tier

| set | seed | GT adjacent pairs | confirmed | correct | wrong pose | non-adj | **cross-object** | unscorable | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_60` | 0 | 143 | 26 | **26** | 0 | 0 | **0** | 0 | 0.182 |
| `synthetic_60` | 1 | 143 | 23 | **23** | 0 | 0 | **0** | 0 | 0.161 |
| `synthetic_170` | 0 | 407 | 51 | **51** | 0 | 0 | **0** | 0 | 0.125 |
| `synthetic_170` | 1 | 407 | 48 | **48** | 0 | 0 | **0** | 0 | 0.118 |
| `mixed_all` | 0 | 288 | 23 | **19** | 3 | 1 | **0** | 0 | 0.066 |
| `mixed_all` | 1 | 288 | 28 | **22** | 2 | 1 | **3** | 0 | 0.076 |

**On the two synthetic collections the confirmed tier is clean at both seeds: 148 confirmed joins,
every one `correct`, zero wrong-pose, zero non-adjacent, zero cross-object.** That includes
`synthetic_170`, which carries six pieces of a second vessel precisely so that a cross-object join
is possible there — and none was made, at either seed.

**On `mixed_all` it is not.** Ten of the 51 confirmed joins over the two seeds are false: four at
seed 0 (one non-adjacent, three wrong-pose) and six at seed 1 (one non-adjacent, three
**cross-object**, two wrong-pose). §3.4 takes them apart one by one.

### 3.2 The confirmed and probable bands together

| set | seed | probable | correct | wrong pose | non-adj | cross-object | unscorable | **conf+prob correct** | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_60` | 0 | 38 | **38** | 0 | 0 | 0 | 0 | **64** | 0.448 |
| `synthetic_60` | 1 | 43 | **43** | 0 | 0 | 0 | 0 | **66** | 0.462 |
| `synthetic_170` | 0 | 94 | **81** | 0 | 13 | 0 | 0 | **132** | 0.324 |
| `synthetic_170` | 1 | 92 | **88** | 0 | 4 | 0 | 0 | **136** | 0.334 |
| `mixed_all` | 0 | 2 761 | **61** | 49 | 394 | **2 224** | 33 | **80** | 0.278 |
| `mixed_all` | 1 | 2 795 | **62** | 53 | 385 | **2 263** | 32 | **84** | 0.292 |

**The probable band behaves completely differently on a mixed collection, and this is the finding
the museum has to be told about.** On one pot it is nearly all true — 38 of 38 and 43 of 43 on
`synthetic_60`, 81 of 94 and 88 of 92 on `synthetic_170`. On ten pots it is **2 761 rows with 61
true joins in them**: 2.2 % of the list, and four fifths of the rest are pairs from *different
pots*. R §6.5's own thresholds cannot tell a Pot_E fracture from a Pot_I one, and with 164 sherds of
ten vessels there are simply many more wrong pairings available than right ones.

The band is nevertheless **ordered**, which makes it usable. Sorting the probable rows by
`score`:

| K, the top rows by score | correct in them, seed 0 | seed 1 | precision | share of the band's true joins |
|---:|---:|---:|---:|---:|
| 25 | 21 | 19 | 76–84 % | 31–34 % |
| 50 | 33 | 36 | 66–72 % | 54–58 % |
| **100** | **49** | **51** | **49–51 %** | **80–82 %** |
| 200 | 56 | 58 | 28–29 % | 92–94 % |
| 400 | 59 | 61 | 15 % | 97–98 % |
| all (2 761 / 2 795) | 61 | 62 | 2 % | 100 % |

**The first hundred rows hold four fifths of everything true in the band, one in two of them is
real, and the other 2 661 rows are worth twelve more joins at seed 0 and eleven at seed 1.** That is
a bench day, not a bench month, and it is the number the README's museum section now quotes — and,
since task C, the number `--probable-top` defaults to.

### 3.3 The assembly R §8 built from the confirmed band

| set | seed | fragment accuracy | precision | joins used | correct | wrong pose | non-adj | **cross-object** | **purity** | groups (of which > 1) | largest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_60` | 0 | 46.4 % | 1.000 | 25 | 25 | 0 | 0 | **0** | **1.000** | 35 (5) | 8 |
| `synthetic_60` | 1 | 48.2 % | 1.000 | 22 | 22 | 0 | 0 | **0** | **1.000** | 36 (7) | 8 |
| `synthetic_170` | 0 | 42.7 % | 1.000 | 51 | 51 | 0 | 0 | **0** | **1.000** | 114 (20) | 9 |
| `synthetic_170` | 1 | 41.5 % | 1.000 | 48 | 48 | 0 | 0 | **0** | **1.000** | 117 (21) | 8 |
| `mixed_all` | 0 | 21.8 % | 0.810 | 21 | 17 | 3 | 1 | **0** | **1.000** | 143 (16) | 5 |
| `mixed_all` | 1 | 23.2 % | 0.778 | 27 | 21 | 2 | 1 | **3** | **0.930** | 138 (17) | 4 |

**Groups against objects.** `mixed_all` is ten pots and the run reports 143 and 138 groups — but
127 and 121 of those are single sherds that nothing was confident enough to join. The assembled
groups are **16 and 17**, none larger than five sherds, and at seed 0 every one of them is pure. At
seed 1 three of the seventeen are not: each mixes Pot_E with Pot_I (§3.4). `synthetic_170` is one
pot plus six distractors and reports 114 and 117 groups, 20 and 21 of them assembled, largest 9 —
no group ever mixes the intruders with the pot.

### 3.4 The ten false confirmed joins of `mixed_all`, one by one

Read off each candidate's own evidence block. **Every single one of the ten was confirmed by the
margin arm with `support` 0** — not one of them had a second, independent join of the collection
agreeing with where it put the sherd.

| seed | verdict | pair | objects | score | seam | tight | gap | cont_n | pen | slide | margin | placements | support |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | non-adjacent | Pot_D 20 + Pot_D 26 | D/D | 5.64 | 10.3 t | 0.55 | 0.0123 | 0.924 | 0 | 2.9e-3 | 2.24 | 4 | **0** |
| 0 | wrong pose | Pot_E 13 + Pot_E 27 | E/E | 8.08 | 12.0 t | 0.67 | 0.0106 | 0.934 | 0 | 1.1e-3 | 3.31 | 2 | **0** |
| 0 | wrong pose | Pot_I 03 + Pot_I 14 | I/I | 12.09 | 17.7 t | 0.68 | 0.0065 | 0.906 | 0 | 3.4e-14 | 6.90 | 2 | **0** |
| 0 | wrong pose | Pot_I 07 + Pot_I 08 | I/I | 32.21 | 36.7 t | 0.88 | 0.0033 | 0.996 | 0 | 1.3e-14 | 6.61 | 3 | **0** |
| 1 | non-adjacent | Pot_D 26 + Pot_D 29 | D/D | 5.25 | 8.3 t | 0.63 | 0.0044 | 0.990 | 0 | 6.4e-14 | 6.99 | 3 | **0** |
| 1 | **cross-object** | Pot_E 16 + Pot_I 25 | E/I | 6.22 | 12.0 t | 0.52 | 0.0149 | 0.975 | 0 | 1.7e-2 | 5.87 | 2 | **0** |
| 1 | **cross-object** | Pot_E 19 + Pot_I 23 | E/I | 10.59 | 16.3 t | 0.65 | 0.0130 | 0.976 | 0 | 3.8e-2 | 2.38 | 5 | **0** |
| 1 | **cross-object** | Pot_E 23 + Pot_I 20 | E/I | 10.63 | 18.0 t | 0.59 | 0.0137 | 0.975 | 0 | 1.1e-2 | 5.80 | 3 | **0** |
| 1 | wrong pose | Pot_I 06 + Pot_I 13 | I/I | 18.94 | 24.3 t | 0.78 | 0.0064 | 0.991 | 0 | 5.4e-14 | 23.57 | 5 | **0** |
| 1 | wrong pose | Pot_I 19 + Pot_I 21 | I/I | 17.69 | 22.7 t | 0.78 | 0.0050 | 0.952 | 0 | 1.2e-3 | 3.13 | 2 | **0** |

They are three different failures and they need three different answers.

**(a) The five wrong-pose joins are the sherd sliding along the seam, and they miss by a hair.**
Their pose errors are **0.65, 0.63, 0.58, 0.54 and 0.60 wall thicknesses** at 8.9°, 2.6°, 1.4°,
1.3° and 1.6°, against `evaluate.py`'s 0.5 t. Four of the five are inside 2.7° of the truth. This
is the failure mode task M1 measured on the development sets — a fracture surface that fits its
neighbour a little further along the crack than it should — and the tier's own slide probe cannot
see it: it reports 1.3e-14 to 1.2e-3 t, meaning the pose is **stable** under a ±0.5 t push. It is
determined; it is simply not the ground truth's. A conservator looking at the review image would see
a join that is a few millimetres off, which is exactly the kind of thing the image exists for.

**(b) The two non-adjacent joins are not a ground-truth artefact.** Pot_D 20 and 26 sit **28.1 t
apart** in the real pot and the run puts them together **139.1° and 22.6 t** from the truth; Pot_D
26 and 29 sit **31.7 t** apart and the run is **160.7° and 32.4 t** out. These are two sherds of the
same vessel whose fracture surfaces happen to mate, which is what a pot with 32 pieces offers a lot
of. Nothing about them is close.

**(c) The three cross-object joins are the twin-feature problem, arriving exactly where task M1
predicted it.** All three are Pot_E against Pot_I — the two largest pots in the collection, 34 and
30 sherds. Roadmap item 4's consensus is supposed to catch this, and here is what it saw:

| group | members | thick (median ± MAD) | shell radius | fracture roughness | members outside the consensus |
|---|---|---|---|---|---|
| 3 | Pot_E 07, Pot_E 16, **Pot_I 25** | 7.308 ± 0.270 | 33.602 ± 0.643 | 0.047 ± 0.000 | **none** |
| 5 | Pot_E 20, Pot_E 23, **Pot_I 20** | 5.766 ± 0.205 | 62.519 ± 6.521 | 0.029 ± 0.000 | **none** |
| 14 | Pot_E 19, **Pot_I 23** | 6.643 ± 0.046 | 99.294 ± 48.537 | 0.041 ± 0.004 | **none** |

**The Pot_I sherd agrees with its Pot_E group to within the group's own MAD on every feature the
cache carries** — 0.046 mm of wall thickness in group 14. There is no `k` at which a `k·MAD` rule
demotes these and does not demote half the true joins with them, and the shipped `--object-demote`
is empty for exactly this reason: task M1 §4 measured the best feature at AUC 0.740 against audit
§D.2's own bar of 0.800. Pot_E's wall is 6.45 ± 0.70 mm over 34 sherds and Pot_I's is 7.52 ± 0.57 mm
over 30, ranges 4.25–12.04 and 4.00–8.80 — two distributions that overlap almost completely. **This
is the measurement of §D.2's shortlist being empty, made on the collection it matters on.**

### 3.5 What the support arm would cost, measured and not applied

Every one of the ten was confirmed on the margin arm alone. Requiring `support ≥ 1` — i.e.
`--tier-support 1` with the margin arm removed — was simulated exactly over the six runs:

| set | seed | confirmed | support ≥ 1 | margin-only | of those false | **correct joins lost** | **false joins removed** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_60` | 0 | 26 | 14 | 12 | 0 | 12 | 0 |
| `synthetic_60` | 1 | 23 | 12 | 11 | 0 | 11 | 0 |
| `synthetic_170` | 0 | 51 | 16 | 35 | 0 | 35 | 0 |
| `synthetic_170` | 1 | 48 | 7 | 41 | 0 | 41 | 0 |
| `mixed_all` | 0 | 23 | 8 | 15 | 4 | 11 | **4** |
| `mixed_all` | 1 | 28 | 8 | 20 | 6 | 14 | **6** |

It removes **all ten** false joins and **124 of the 189 correct** ones — the confirmed tier's recall
over these six runs would fall from 11.3 % of the ground truth's adjacent pairs to 3.9 %. That is
the same trade task G measured on the development sets (`notes/2026-09-10-g-tiers-findings.md`:
51 of 130 confirmed joins were margin-only there), now measured at collection scale. **Nothing was
changed on the strength of it**: the brief for this step forbids tuning on the acceptance sets, and
a threshold fitted to the set it is judged on is not a threshold. The number is here so that the
step which does take this decision has it.

### 3.6 What the watertight rule costs on real sherds

Task G's rule — a join whose penetration cannot be measured is never confirmed — bites on
`mixed_all`, where **5 of 164 sherds are not watertight** (Pot_B 02, 07, 08; Pot_C 01; Pot_J 01).
**16 of the collection's 288 ground-truth adjacent pairs (5.6 %) touch one of them** and can
therefore never be confirmed, whatever their geometry. `synthetic_60` and `synthetic_170` are
watertight throughout and the rule moves nothing there.

---

## 4. The gates of D §10.3 and of audit §E's step 12

| gate | `synthetic_60` | `synthetic_170` | `mixed_all` | verdict |
|---|---|---|---|---|
| wall, CPU, inside D §10.3's row | 2.1 min of ≤ 15 min | 18.5 min of ≤ 2 h | 30.3 min of ≤ 2 h | **pass**, at both seeds, cold and warm |
| peak RSS ≤ 6 GB (D §1, D §8) | 2 288 MiB | 2 647 MiB | 1 794 MiB | **pass** — worst is 44 % of the row |
| confirmed tier: cross-object joins **0** | 0 / 0 (one object) | **0 / 0** | 0 at seed 0, **3 at seed 1** | **FAIL on `mixed_all`** |
| confirmed tier: group purity **1.000** | 1.000 / 1.000 | **1.000 / 1.000** | 1.000 at seed 0, **0.930 at seed 1** | **FAIL on `mixed_all`** |
| confirmed tier: zero false joins (audit §D.1) | 0 / 0 | 0 / 0 | **4 at seed 0, 6 at seed 1** | **FAIL on `mixed_all`** |
| confirmed recall reported | 0.161–0.182 | 0.118–0.125 | 0.066–0.076 | reported (§3.1) |
| probable list size reported | 38 / 43 | 94 / 92 | **2 761 / 2 795** | reported (§3.2) |
| groups against the number of objects | 5 and 7 assembled groups, 1 pot | 20 and 21, 1 pot + 6 distractors, never mixed | 16 and 17 assembled of **10 pots**; pure at seed 0, three impure at seed 1 | reported (§3.3) |
| the two seeds agree on the confirmed joins | §5 | §5 | §5 | reported (§5) |

**The runtime and memory rows of D §10.3 are discharged.** All three collections are inside their
CPU gates at both seeds, cold and warm, with 4.0–7.0× of margin, and the peak resident set never
reaches half of D §1's 6 GB row. Those were the two things the projection could not answer, and both
answers are comfortable.

**The quality row is not discharged on `mixed_all`, and it fails as a prohibition and not as a
recall row.** Three cross-object joins at seed 1 is the breach; the seven other false confirmed
joins are the same tier admitting two failures the geometry alone cannot see. It holds on
`synthetic_170`, which is the other collection step 12 names, at both seeds and with a second vessel
present to make the failure possible. §3.4 says what each of the ten is and §3.5 says what the one
rule that would remove them costs. **Nothing was tuned**: the step's own brief forbids fitting a
threshold on the set that judges it, and the failure is reported rather than papered over.

---

## 5. The two seeds

| set | confirmed at seed 0 | at seed 1 | **at both** | at one only | of those, **probable** at the other | of those, not accepted at the other |
|---|---:|---:|---:|---:|---:|---:|
| `synthetic_60` | 26 | 23 | **15** | 19 | 15 | 4 |
| `synthetic_170` | 51 | 48 | **33** | 33 | 21 | 12 |
| `mixed_all` | 23 | 28 | **12** | 27 | 20 | 7 |

**About three fifths of a confirmed list survives a change of seed, and four fifths of what does not
is still *probable* at the other seed.** Nothing disappears from the conservator's list because of
a seed; what a seed moves is the band a pair lands in. The pairs that vanish entirely — 4, 12 and 7
of them — are ones R §5.7's search did not return at all at the other draw, which is the same
chaotic quantity R §13's own bands are made of.

This matters for how the tier is read: **a confirmed join is a statement about one draw**, not a
property of the two sherds. The museum's answer to that is `--seed`: two runs at two seeds, and a
pair confirmed at both is a stronger claim than a pair confirmed at one. Nothing in the tool does
that automatically today, and this table is the argument for a step that makes it.

---

## 6. The development quality gate, unmoved

`tools/quality_gate.py`, the eight development sets at seeds 0–4, on the final tree. **40 runs in
10.0 min** (602.2 s) — and **every cell of the table below is task G's**. *(Corrected in task C,
V9-D9: this sentence read "task G measured 421.7 s for the same forty … the difference is thermal".
It is neither G's number nor a difference. G's own run of the same forty took **610.9 s**
(`output/g/quality/quality.json`, `meta.wall_total_s`), so this run is 1.4 % **faster** than G's,
not 43 % slower, and no thermal story is needed. 421.7 s is task **H4**'s figure, measured before
the tier pass was the default; the five measured runs of the shipped-default gate are 602.2, 610.9,
611.7, 614.9 and 602.6 s.)*

| set | seed | frag acc | precision | correct | wrong pose | non-adj | cross-obj | purity | joins | confirmed | conf correct | conf false | conf recall | probable | correct found |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `terracotta` | 0 | — | — | — | — | — | — | — | 2 | 2 | 2 | 0 | 1.000 | 0 | 2 |
| `terracotta` | 1 | — | — | — | — | — | — | — | 2 | 2 | 2 | 0 | 1.000 | 0 | 2 |
| `terracotta` | 2 | — | — | — | — | — | — | — | 2 | 2 | 2 | 0 | 1.000 | 0 | 2 |
| `terracotta` | 3 | — | — | — | — | — | — | — | 2 | 2 | 2 | 0 | 1.000 | 0 | 2 |
| `terracotta` | 4 | — | — | — | — | — | — | — | 1 | 1 | 1 | 0 | 0.500 | 1 | 2 |
| `pot_A` | 0 | 62.5 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | 0 | 0.267 | 13 | 8 |
| `pot_A` | 1 | 50.0 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | 0 | 0.267 | 14 | 8 |
| `pot_A` | 2 | 37.5 % | 1.000 | 2 | 0 | 0 | 0 | 1.000 | 2 | 3 | 3 | 0 | 0.200 | 9 | 6 |
| `pot_A` | 3 | 62.5 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | 0 | 0.267 | 8 | 8 |
| `pot_A` | 4 | 62.5 % | 1.000 | 4 | 0 | 0 | 0 | 1.000 | 4 | 6 | 6 | 0 | 0.400 | 9 | 9 |
| `pot_B` | 0 | 55.6 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | 0 | 0.267 | 22 | 13 |
| `pot_B` | 1 | 33.3 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 3 | 3 | 0 | 0.200 | 18 | 11 |
| `pot_B` | 2 | 55.6 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | 0 | 0.267 | 17 | 11 |
| `pot_B` | 3 | 33.3 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 3 | 3 | 0 | 0.200 | 22 | 11 |
| `pot_B` | 4 | 33.3 % | 1.000 | 2 | 0 | 0 | 0 | 1.000 | 2 | 3 | 3 | 0 | 0.200 | 19 | 10 |
| `pot_C` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 3 | 2 |
| `pot_C` | 1 | 50.0 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 1 | 1 | 0 | 0.200 | 3 | 2 |
| `pot_C` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 2 | 1 |
| `pot_C` | 3 | 50.0 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 1 | 1 | 0 | 0.200 | 2 | 2 |
| `pot_C` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 3 | 2 |
| `pot_G` | 0–4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 1–2 | 0 |
| `pot_H` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 13 | 3 |
| `pot_H` | 1 | 18.2 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 1 | 1 | 0 | 0.059 | 19 | 3 |
| `pot_H` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 11 | 2 |
| `pot_H` | 3 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 20 | 3 |
| `pot_H` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 | 0.000 | 13 | 3 |
| `synthetic_20` | 0 | 35.0 % | 1.000 | 5 | 0 | 0 | 0 | 1.000 | 5 | 5 | 5 | 0 | 0.100 | 18 | 23 |
| `synthetic_20` | 1 | 60.0 % | 1.000 | 9 | 0 | 0 | 0 | 1.000 | 9 | 9 | 9 | 0 | 0.180 | 15 | 24 |
| `synthetic_20` | 2 | 50.0 % | 1.000 | 9 | 0 | 0 | 0 | 1.000 | 9 | 9 | 9 | 0 | 0.180 | 14 | 23 |
| `synthetic_20` | 3 | 45.0 % | 1.000 | 8 | 0 | 0 | 0 | 1.000 | 8 | 8 | 8 | 0 | 0.160 | 14 | 22 |
| `synthetic_20` | 4 | 55.0 % | 1.000 | 10 | 0 | 0 | 0 | 1.000 | 10 | 11 | 11 | 0 | 0.220 | 16 | 27 |
| `mixed_ABG` | 0 | 41.7 % | 1.000 | 6 | 0 | 0 | 0 | 1.000 | 6 | 8 | 8 | 0 | 0.200 | 79 | 21 |
| `mixed_ABG` | 1 | 29.2 % | 1.000 | 6 | 0 | 0 | 0 | 1.000 | 6 | 7 | 7 | 0 | 0.175 | 73 | 19 |
| `mixed_ABG` | 2 | 33.3 % | 1.000 | 5 | 0 | 0 | 0 | 1.000 | 5 | 7 | 7 | 0 | 0.175 | 68 | 17 |
| `mixed_ABG` | 3 | 33.3 % | 1.000 | 6 | 0 | 0 | 0 | 1.000 | 6 | 7 | 7 | 0 | 0.175 | 65 | 19 |
| `mixed_ABG` | 4 | 33.3 % | 1.000 | 6 | 0 | 0 | 0 | 1.000 | 6 | 9 | 9 | 0 | 0.225 | 72 | 19 |

`pot_G`'s five rows are identical and are collapsed into one; the full table is
`output/a1/gates/quality_gate.log`.

**Confirmed tier over the 40 runs: 130 correct, 0 false, 16.9 % of the 770 ground-truth adjacent
pairs; 342 correct with the probable band, 44.4 %** — G's numbers to the join. The one failure is
G's too and it is the same recall row: the terracotta reaches **9 of the 10** (seed, join) slots
because 021–094 is only probable at seed 4, "no prohibition was breached".

**Read this table beside §3.** The eight development sets are seven single-object collections and
one three-pot collection of 24 fragments, and on all of them the confirmed tier has never made a
false join. `mixed_all` is 164 sherds of ten pots, and it makes ten. The development gate was not
wrong; it was never asked the question that collection asks.

---

## 7. Standing gates, on the final tree

| gate | result |
|---|---|
| `cargo fmt --check` | **pass** |
| `cargo clippy --workspace --all-targets -- -D warnings` | **pass**, no warning |
| `cargo build --no-default-features -p sherd-cli` | **pass** |
| `cargo clippy --no-default-features -p sherd-cli --all-targets -- -D warnings` | **pass** |
| `cargo test --workspace` (debug) | **442 passed, 0 failed, 3 ignored** over 19 targets |
| `cargo test --workspace --release` | **442 passed, 0 failed, 3 ignored** |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, **256 rows** (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed**, the frozen number to the check and every dump's own figure unmoved (145/631 terracotta, 306/2042 `pot_A`, 354/2527 `pot_B`, 261/1612 `pot_C`, 248/1597 `pot_G`, 459/3650 `pot_H`, 1040/8603 `synthetic_20`, 81/248 slab) |
| `pytest -q` | **pass** |
| `tools/quality_gate.py`, 8 sets × seeds 0–4 | §6 — 130 correct, **0 false**, exit 1 on the terracotta's recall row alone, as at task G |
| byte identity, seed 0, `--tiers off --objects off` | **pass** — 114 files, 104 byte-identical, 10 exempt, **0 differing** |

**The byte-identity check** ran the four sets — terracotta, `pot_A`, `pot_H`, `synthetic_20` — at
seed 0, CPU, `--tiers off --objects off`, previews and meshes **on**, once at the start of this task
(binary built at `486c919`, task G's tree) and once at the end (binary rebuilt at `58704d5`, this
task's second commit). Nothing under `crates/` was touched, so the two binaries differ only in the
commit string their build stamps in:

| set | files | byte-identical | exempt | **differing** |
|---|---:|---:|---:|---:|
| terracotta | 14 | 12 | 2 | **0** |
| `pot_A` | 22 | 20 | 2 | **0** |
| `pot_H` | 30 | 27 | 3 | **0** |
| `synthetic_20` | 48 | 45 | 3 | **0** |
| **total** | **114** | **104** | **10** | **0** |

Every cache, every placed and merged PLY and every PNG is byte-identical. The ten exempt files are
`report.json` and `transforms.json` (compared key by key with `engine.commit`, `timings` and
`memory` removed — the standing gate excludes the last two and the commit differs by construction)
and `report.md` on `pot_H` and `synthetic_20` (compared above the `## Timing` block; on the other
two sets it is byte-identical because their timings round to the same printed values). No key was
added, removed or reordered.

---

## 8. Housekeeping

**Disk.** The task started with **6.6 GiB** free, which is under the 4 GiB floor plus what a
164-fragment run needs, so the measured trees of earlier tasks were deleted first —
`output/scale`, `output/thin`, `output/t2`, `output/synthetic_pingsdorf_20*`,
`output/test_fragments_1*`, `output/bench_*`, `output/sfspp`, `output/h2`, `output/h3`,
`output/h4`, `output/measure` — which returned **3.5 GiB**. Free disk never fell below **8.3 GiB**
during the six runs and stands at **9.7 GiB** now. `input/`, `output/fixtures/` and `fixtures/`
were not touched.

**What is kept**, under `output/acceptance/` (gitignored, **271 MB**): for each of the six runs
`transforms.json`, `report.json`, `report.md` and `time.log` (the `/usr/bin/time -l` block), plus
`quality/quality.{md,json}` and the six per-run `evaluate.py` dumps, plus `collect.json`. The
`report.json` files are 5–66 MB each because they carry every candidate of the run — 60 372 of them
on `mixed_all` — and that is the file every number in §3 was read from, so it is worth the space.
`output/a1/gates/` (520 KB) holds the twenty-six gate logs of §7.

**What was deleted afterwards**: the three fragment caches and the six `-v` stage logs of the runs
(569 MB between them), the duplicate copies of the three reports the work directories held
(135 MB), and the two byte-identity trees (753 MB each). Deleting the last of those was a mistake of
housekeeping order rather than of judgement — the brief asked for the timing logs and those are
kept, but the `-v` logs were where the per-stage `preprocessing memory` and `refinement memory`
lines lived. Everything they carried is quoted in §2.3 and the same peaks are in each run's
`report.json` `memory` block, so nothing is unrecoverable; a future step that wants the per-pair
lines has to re-run.

**Nothing above 24 fragments had been matched before this step and this step matched three
collections of 56 and 164** — which is what step 12 is, and it is the only step allowed to.
`sherd_refit/` was not touched. The branch was never switched, nothing was pushed, nothing was
stashed, and nothing was sent anywhere.

**Commits.** `A1.1` the `--score-only` mode of the quality gate with D §10.4 and the README's gate
section; `A1.2` D §1, §10.3 and §12; `A1.3` the README's benchmark table and its new museum
section; `A1.4` this note; `A1.5` a paragraph split in the README's tier section, where `A1.3`'s
caveat had been spliced into the middle of another sentence.

---

## 9. What this step leaves for the next one

1. **`mixed_all`'s three cross-object joins are roadmap item 4's unfinished business, and they now
   have a measurement.** §D.2's rule — a feature may veto only above AUC 0.800 — leaves
   `--object-demote` empty, and §3.4 shows what that costs: the Pot_I sherd of each impure group
   agrees with its Pot_E group on every feature the cache carries. The next move is a feature that
   separates these two pots, not a smaller `k`. Colour is the obvious candidate and the SfS++ OBJs
   do not carry it (audit §C.4).
2. **The margin arm is where every false confirmed join came from, on this collection and on the
   development sets.** §3.5 prices the one rule that removes them. Somewhere between "the margin
   arm always confirms" and "only support confirms" there is a rule worth measuring — task G
   already rejected two candidates for it, and this note adds the collection-scale numbers those
   two were missing.
3. **The probable list needs a length.** 2 761 rows is not a review list; the top 100 by score is,
   and it holds four fifths of what is there. A `--probable-top N` that truncates `report.md`'s
   section and the review images would make the default output usable on a real collection, and
   the number to default it to is measured in §3.2.
4. **The tier pass is 32 % of a real run.** It probes every accepted candidate, and R §6.5 accepts
   8 444 of them on ten pots. If the probable list gets a length, the probes can respect it.
5. **The museum's own acceptance set has not arrived** (audit §C.4: 20–30 conservator-confirmed
   joins, a handful of confirmed non-joins, fragments with known object attribution). §E's step 12
   asks for it to be run the same way when it does, and that run is still owed.
