# V9 — independent verification of G and A1, and of the acceptance itself

**Date:** 2026-09-10. **Tree:** branch `rust-core`, `ec72ad4` (A1.5) plus this note. **Range under
review:** every commit since `956c0ca` (V8's note) — G1–G5 and A1.1–A1.5, ten commits. **Machine:**
Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`, backend `cpu`.
**Method:** the notes were read *after* the numbers were re-derived. Every figure below comes from
an artefact on disk or from a run made here — `output/g/sweep/`'s forty `report.json`,
`output/v8/quality/quality.json`, `output/acceptance/`'s six runs, and two fresh large-set runs of
my own.

**The verdict in five sentences.** *The code half of G is one decision and it is the one the brief
names*: `git diff 956c0ca..486c919 -- crates/` touches three files, of which exactly one changes
behaviour — the `pen_unavailable` refusal in `Thresholds::refusals` — and `git diff 486c919..HEAD
-- crates/` is **empty**, so A1 changed no algorithm at all. *The restated gate rows are honest*:
`TERRACOTTA_CONFIRMED_SLOTS` is the audit's own `2 * 5` and the run fails on it, `recall_floor` is
the audit's own 12, `pure_tiers` fails on `mixed_all` — three rows that can fail and two that do —
and over the forty development runs **not one of the 795 probed candidates carrying
`pen_unavailable` is Confirmed**. *The acceptance reproduces further than the brief asks*:
`synthetic_170` in 1 109.5 s against A1's 1 107.0 (+0.22 %) and `mixed_all` in 1 812.4 against
1 818.5 (−0.34 %), every stage over a second inside 1.3 %, both peak resident sets inside 8.1 %,
and **the whole of `report.json` — 54 045 and 60 372 candidates with every score and evidence
block — equal to A1's after removing `timings`, `memory` and `engine.commit` and nothing else**,
`transforms.json` included. *Every standing gate holds*: fmt, clippy and both
`--no-default-features` commands clean, 442 tests passed in each profile, parity **23 804 / 0**,
pytest 60 passed, the forty-run gate **cell for cell** G's and A1's (760 cells, 0 differing).
*Nineteen defects are open*, none of them a wrong decision and none a breached prohibition — two
sit in the shipped binary's own `--help`, one in the module doc of the file G edited, one is a
docstring in the gate script, and fifteen are documents.

**`gates_ok` is false**, on items 4 and 5 of the brief. D and the README do **not** all carry A1's
measured numbers: `synthetic_170`'s accepted-candidate count is wrong in three places (V9-D4), the
gate's own cost is documented at a pre-tier figure (V9-D10), D §12 quotes a warm time no run
produced (V9-D12), D §8 was never given the measurement A1's note asks it for (V9-D14), and
`tiers.rs`'s own headline still advertises the number G1 changed (V9-D1). The museum README fails
on four statements the tree contradicts (V9-D5, V9-D7, V9-D8, V9-D18). **Everything a conservator is
told about safety is correct**; what is wrong is arithmetic in the commentary and one piece of
advice in the README.

---

## 0. The six things this task was asked to check

| # | what the brief asked | answer | where |
|---|---|---|---|
| 1 | did G change any decision beyond the pen-unavailable refusal and the gate rows? | **no** — one behavioural line in `tiers.rs`; everything else is doc comments and one test | §1.1 |
| 1 | are the restated rows honest — no constant fitted to a measured answer? | **yes**, and two of the three fail on this tree | §1.2 |
| 1 | is the 40-run quality table what the G note prints? | **yes** — 760 cells compared, **0 differing**, and the run's own headline is G's | §3 |
| 2 | all standing gates, both profiles, both `--no-default-features`, the parity total, pytest | **all pass**; the quality gate exits 1 on the terracotta recall row alone, which is the intended state since G | §4 |
| 3 | the acceptance re-measured once per collection | **wall and RSS inside 8.1 %**, most stages inside 1 %; the scorer's rows and the Confirmed *sets* identical; `transforms.json` byte-identical **after substituting one 40-character commit string** | §2, V9-D19 |
| 4 | D §10.3 / §1 / §12 / README carry A1's measured numbers, no projection presented as a measurement | **no** — five documentation defects, one of them in `tiers.rs`'s own module doc | §7 |
| 5 | is the museum README section readable by a non-programmer? | **mostly** — every flag it names works on the terracotta, but four statements mislead and its only example file aborts the run | §6 |
| 6 | disk ≥ 4 GiB, the large sets' caches deleted | **yes** — never below 7.0 GiB, 7.4 GiB now, both caches deleted (413 MiB) | §8 |

---

## 1. The diff, read before the notes

### 1.1 Task G — one decision, two gate rows, four documentation fixes

`git diff 956c0ca..486c919 --stat` touches nine files. The three under `crates/`:

| file | what changed | a decision? |
|---|---|---|
| `crates/sherd-core/src/tiers.rs` | `Thresholds::refusals` pushes a refusal when `scores.pen_unavailable` instead of running `ceiling("pen", …)`; two doc comments; one new test and a sixth row in `each_threshold_refuses_on_its_own` | **yes** — the one V8-D1 asked for |
| `crates/sherd-core/tests/assembly_graphs.rs` | doc comment only ("it cannot" → argued and sampled) | no |
| `crates/sherd-parity/src/stages/outputs.rs` | a twelve-line doc comment moved back above `markdown_row` | no |

**Nothing else in the range touches behaviour.** `verify::accept` is untouched — the new test
asserts it (`crate::matching::verify::accept(&holes, …)` is still `true`) — which is why the join
lands in Probable rather than Rejected and why the parity total is unmoved.

**`--tiers off` cannot reach the change at all, and that is a proof rather than a measurement.**
`crates/sherd-core/src/pipeline.rs:1091` is `let thresholds = params.tiers?;` — `tier_pass` returns
`None` before `tiers::classify`, and therefore before `Thresholds::refusals`, is ever called. Worth
saying because G's own byte-identity check (its §6) ran terracotta, `pot_A`, `pot_H` and
`synthetic_20`, and **not one of those four has a fragment that is not watertight**, so the
measurement G quotes could not have caught a regression in the one place G1 changed even if there
had been one. I added `pot_B` (3 of 9 not watertight) and `pot_C` (1 of 7) to this step's own
identity check for that reason — §4. Measured on the set where it would show:
`output/v9/id_old/pot_B` has **105** candidates carrying `pen_unavailable`, emits **no `tier` field
at all**, and its assembly still uses `01-02` and `02-08`, two of the three joins G1 demoted from
Confirmed under `--tiers on`.

I re-derived the effect rather than reading it. Comparing `output/v8/quality/quality.json` with
`output/g/quality/quality.json` cell by cell, **two of the forty runs move and thirty-eight do
not**:

| run | confirmed | conf correct | probable | joins used | frag acc |
|---|---|---|---|---|---|
| `pot_B` seed 0 | 7 → **4** | 7 → **4** | 19 → 22 | 5 → 3 | 77.8 % → 55.6 % |
| `mixed_ABG` seed 0 | 11 → **8** | 11 → **8** | 76 → 79 | 8 → 6 | 50.0 % → 41.7 % |

The joins given up are the same three in both — `Pot_B_Piece_01+02`, `01+08`, `02+08` — and the
confirmed-false total is **0 on both sides**. G's §1.3 is exact.

The rule fires where G says it does, and **everywhere else too**. Over the forty runs of
`output/g/sweep/`, **2 325** candidates carry `pen_unavailable`, **795** of them are accepted and
therefore probed, and **not one is Confirmed**. On `pot_B` seed 0 alone it is 105 / 72 / 0, all 72
`probable`, and `Pot_B_Piece_01+Pot_B_Piece_02` carries the single reason
`pen: penetration not measurable, a fragment is not watertight`. The watertight census read off
`fragments[].watertight` is G's table to the fragment: terracotta 0 of 4, `pot_A` 0 of 8, **`pot_B`
3 of 9** (02, 07, 08), **`pot_C` 1 of 7** (01), `pot_G` 0 of 7, `pot_H` 0 of 11, `synthetic_20`
0 of 20, **`mixed_ABG` 3 of 24**. `ls input/sfspp/mixed_ABG/*.obj` is **24** (V8-D6) and
`const N: u32 = 8` in `assembly_graphs.rs` (V8-D5).

### 1.2 Are the restated rows fitted to a measured answer?

**No, and the strongest evidence is that two of the three fail.**

* `TERRACOTTA_CONFIRMED_SLOTS = 2 * 5` (`tools/quality_gate.py:218`). Audit §E step 8's brief is
  *"terracotta's two joins confirmed"* at seeds 0–4; two joins over five seeds is ten. The tree
  reaches nine and the gate exits 1. A fitted constant is exactly what the old `= 9` was.
* `recall_floor = 12` on `mixed_ABG`. Audit §E step 10's brief is *"correct joins ≥ 12 (the
  baseline's)"*, and D §10.3's baseline row is the twelve correct joins the pre-tier assembly used.
  Re-derived from `output/g/sweep/` over the confirmed **and** probable bands: **21 / 19 / 17 / 19 /
  19**. The floor has 5 to 9 joins of headroom, which makes it loose, but it is the audit's number
  and not one read off the run.
* `pure_tiers` (cross-object 0, purity 1.000) on `mixed_ABG` and on the three acceptance sets. It
  holds on `mixed_ABG` at all five seeds and **fails on `mixed_all`** — which is what a row that is
  not fitted looks like.
* Correct **confirmed** joins are reported per seed (`8 / 7 / 7 / 7 / 9` on `mixed_ABG`, re-derived
  here) and deliberately not gated. G's reason — a floor at today's number fails every future step
  that trades a confirmed join for a safer rule — is the right one, and G1 is such a step.

### 1.3 The two rules G measured and refused, re-measured

A refusal nobody re-checks is a claim, so both were recomputed from `output/g/sweep/`'s forty runs.

* **Rule (A)**, "a single placement means nothing to beat, so the margin arm passes." Every probable
  representative whose only refusal is `neither arm: support 0 < 1 and no second placement to beat`
  and whose `placements` is 1, classified on its own pose: **118 correct, 17 wrong-pose** —
  terracotta 1/0, `pot_A` 16/2, `pot_B` 15/0, `pot_C` 5/0, `pot_G` 0/0, **`pot_H` 13/13**,
  `synthetic_20` 37/0, `mixed_ABG` 31/2. Every wrong-pose pair G lists is in my list and no other:
  `pot_H` 03–07 at all five seeds, 02–04 at 1/2/3, 03–04 at 1/3/4, 02–11 at 3, 09–11 at 0; `pot_A`
  and `mixed_ABG` 02–05 at seeds 1 and 2. The rule does break the prohibition.
* **Rule (B)**, "a rival the tier would itself refuse is no rival." Of the 51 margin-only confirmed
  joins, the rival implied by `score / margin` is a candidate **R §6.5 refused in 47** of them and
  one it **accepted in 4** — 0 unmatched. G's 47 / 4 exactly.
* **The confirming arm of each of the 130 confirmed joins**: **68 support-only, 51 margin-only, 11
  both**. G's §3.2 to the join.
* **G §3.1's terracotta evidence**, seed 0 against seed 4: `tight` 0.5566 / 0.5427, `gap` 0.0080 /
  0.0082, `seam` 20.67 / 21.00, `cont_n` 0.9950 / 0.9904, `pen` 0 measured on both, slide 7.5e-15 /
  1.1e-14, redraws 3 of 3 on both, **`placements` 3 / 1**, **`margin` 9.84 / none**, `support` 0 on
  both. The kept list holds five at both seeds; at seed 4 all five sit at one placement with `brk`
  0.138 / 0.095 / 0.087 / 0.087 / 0.071 and **all five accepted**, while at seed 0 three are at the
  true placement and **two the matcher refused** (`tight` 0.117 and 0.093, `gap` 0.070 and 0.075).
  Exact.

### 1.4 Task A1 — no algorithm at all

`git diff --stat 486c919..HEAD -- crates/` is **empty**. A1 touched `README.md`, D,
`tools/quality_gate.py` (the `--score-only` mode and the three `ACCEPTANCE` entries) and its own
note. `--score-only` cannot change what the eight development sets do: `--sets` still validates
against `SETS`, the three acceptance names are absent from it, and `BANDS.update` only adds rows for
names an ordinary run never scores.

### 1.5 The one judgement call I want on the record

V8-D2 offered two acceptable outcomes: gate confirmed-correct at "the number recorded, per set", or
amend audit §E. G took neither — it gated **confirmed and probable together** at the audit's own 12
and reported confirmed-correct per seed ungated — and its reason is the better one: "the number
recorded, per set" *is* a constant fitted to a measured answer, which is what V8-D3 asked to be
removed one defect later. The tier sorts what R §6.5 accepted into bands and adds no join, so the
population the pre-tier baseline's 12 was counted on is the two upper bands together. I accept the
reading. What a reader should know, and G does print it: **under the narrow reading the audit's
step-10 row is still unmet** — correct *confirmed* joins on `mixed_ABG` are **8 / 7 / 7 / 7 / 9**
against 12. D §12's own step-10 row still states the narrow reading; that is **V9-D16**.

---

## 2. The acceptance, re-measured

Two runs, one per collection, seed 0, A1's own flags, one after another on an otherwise idle
machine — the only two large-set runs this task is allowed:

```
/usr/bin/time -l target/release/sherd-refit-rs run <input> --out output/v9/<set>/seed0 \
    --backend cpu --seed 0 --no-preview --no-meshes --memory-budget 4.83 -v
```

**A word on "warm".** Audit §E step 12 asks for a warm cache and so does this task's brief, but A1
deleted the three fragment caches when it finished (its §8), so there is no warm work directory left
to run into and **both of my runs are cold**. That is the right comparison anyway: **A1's seed-0
runs were cold too** (its §1), seed 0 is the row the brief points at, and cold is the slower of the
pair on all six of A1's runs, so it is the conservative direction.

### 2.1 Wall clock and peak resident set — everything inside 15 %, most inside 1 %

| set | quantity | A1, seed 0 | **V9, seed 0** | delta |
|---|---|---:|---:|---:|
| `synthetic_170` | **`real`** (`/usr/bin/time -l`) | 1 107.04 s | **1 109.52 s** | **+0.22 %** |
| | `user` | 9 449.03 s | 9 469.94 s | +0.22 % |
| | preprocess | 17.921 s | 18.057 s | +0.76 % |
| | matching | 1 071.312 s | 1 073.621 s | +0.22 % |
| | tiers | 16.789 s | 16.778 s | −0.07 % |
| | assembly / objects / refine | 0.060 / 0.000 / 0.648 s | 0.057 / 0.000 / 0.648 s | ±5 % on 60 ms |
| | **peak RSS**, `/usr/bin/time -l` | 2 569.4 MiB | **2 750.9 MiB** | **+7.1 %** |
| | peak RSS, `RssMonitor` | 2 569.3 MiB | 2 707 MiB | +5.4 % |
| `mixed_all` | **`real`** | 1 818.51 s | **1 812.38 s** | **−0.34 %** |
| | `user` | 15 493.55 s | 15 487.91 s | −0.04 % |
| | preprocess | 9.805 s | 9.690 s | −1.18 % |
| | matching | 1 227.335 s | 1 221.901 s | −0.44 % |
| | **tiers** | 580.534 s | **579.913 s** | −0.11 % |
| | assembly / objects / refine | 0.021 / 0.000 / 0.526 s | 0.019 / 0.000 / 0.533 s | ±7 % on 20 ms |
| | **peak RSS**, `/usr/bin/time -l` | 1 753.7 MiB | **1 896.0 MiB** | **+8.1 %** |
| | peak RSS, `RssMonitor` | 1 753.7 MiB | 1 854 MiB | +5.7 % |

Every stage that takes more than a second is inside **1.3 %** of A1's; the two peaks are inside
**8.1 %**, which is the allocator's history and not a change — the same two instruments disagree
with each other by 2.7 % on A1's own `synthetic_170` seed 1 (V9-D13). **Both wall clocks are deep
inside D §10.3's two-hour gate**: 18.49 min at **6.49×** and 30.21 min at **3.97×**, which are
A1's 6.5× and 4.0× to the decimal.

The runs also reproduce the two collection-level counts D §10.3 states, from the runs' own logs:
`synthetic_170` matches **12 839** pairs (527 skipped of 13 366 by R §4.3's thickness screen) and
`mixed_all` matches **12 612** (754 skipped), thickness 6.2573. And the log line that settles
V9-D4: `matching done … candidates=54045 accepted=464` on `synthetic_170`,
`candidates=60372 accepted=8444` on `mixed_all`.

### 2.2 `transforms.json` and `report.json` — identical, once one 40-character string is accounted for

Neither `transforms.json` is byte-identical to A1's as it stands, and the reason is one field.
**A1's six acceptance runs were all made with a binary stamped
`486c91906ae1348acd8f1492d5b1eef2064db968` — task G's HEAD, not any A1 commit** (their
`engine.commit`; A1's forty-run gate is stamped `58704d5`, A1.2). `engine.commit` is inside
`transforms.json`, so the bytes must differ from mine, which is stamped `ec72ad40…`. Substituting
that one string:

| set | A1's `transforms.json` | mine, with the commit string substituted |
|---|---|---|
| `synthetic_170` | `df7bce78f5b9031e8eb8329f08e24bd0` | **`df7bce78f5b9031e8eb8329f08e24bd0`** — `cmp` reports no difference |
| `mixed_all` | `caab3d681406daf6ac037c3a268fd5ca` | **`caab3d681406daf6ac037c3a268fd5ca`** — `cmp` reports no difference |

And the check A1 never claimed: for both collections **the whole of `report.json` is equal to A1's
after removing `timings`, `memory` and `engine.commit` and nothing else** — all 54 045 and 60 372
candidates with every score and every evidence block, the fragments, the groups, the object
consensus, `params`, `thickness` and `tiers`. **The literal criterion "byte-identical to A1's seed-0
file" cannot be met from A1's own final tree either**; that is V9-D19.

### 2.3 The Confirmed tier — identical, join for join

Scored through `tools/quality_gate.py --score-only`, the same `score()` A1 used:

| set | seed | frag acc | prec | correct | wrong pose | non-adj | **cross-obj** | **purity** | joins | confirmed | conf correct | conf false | conf recall | probable | correct found |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthetic_170` | 0 | 42.7 % | 1.000 | 51 | 0 | 0 | **0** | **1.000** | 51 | 51 | 51 | **0** | 0.125 | 94 | 132 |
| `mixed_all` | 0 | 21.8 % | 0.810 | 17 | 3 | 1 | **0** | **1.000** | 21 | 23 | 19 | **4** | 0.066 | 2 761 | 80 |

**Every cell is A1's seed-0 row**, and the *sets* are equal and not only the counts: the confirmed
representatives are the same 51 and the same 23 pairs (symmetric difference 0), and the probable
representatives the same 94 and the same 2 761. The four false confirmed joins on `mixed_all` at
seed 0 are the four A1 names — Pot_D 20+26 non-adjacent, Pot_E 13+27, Pot_I 03+14 and Pot_I 07+08
wrong-pose. **Cross-object 0 and purity 1.000 on both collections at seed 0**, which is what the
brief asks; the breach A1 reports is at seed **1** on `mixed_all`, which this task did not re-run
and which A1's own artefacts still carry.

`--score-only` exits **1**, on `mixed_all`'s four false confirmed joins, and prints "at least one
failure above is a prohibition, not a recall row". That is the verdict A1's own table records.

---

## 3. The forty-run quality gate, run here

`python tools/quality_gate.py --out output/v9/quality`, shipped defaults (`--tiers on --objects on
--object-disagreement off`), backend `cpu`, seeds 0–4, the eight development sets. **602.6 s
(10.0 min)**, header stamp `2026-09-10 09:13`, commit `ec72ad40…`. **Exit 1**, on the terracotta's
recall row alone:

```
total wall 602.6 s (10.0 min); FAILED: terracotta
  terracotta    recall row -- audit §E step 8 ("terracotta's two joins confirmed"): 9 of the 10
  (seed, join) slots confirmed -- 021-094 at seed 4 only probable. …
  no prohibition was breached: every failure above is a recall row …
confirmed tier over 40 runs: 130 correct, 0 false, 16.9 % of the 770 ground-truth adjacent pairs;
with the probable band 342 correct, 44.4 %
```

**Is the table G's?** Compared cell by cell over all forty rows and nineteen fields — `frag_acc`,
`precision`, the four join buckets, `purity`, `joins_used`, `confirmed`, `conf_correct`, the three
confirmed-false buckets, `conf_recall`, `probable`, `found_correct`, `merges`, `demoted`,
`consensus_rejects` — **760 cells**:

| pair | cells differing |
|---|---:|
| task G's run (`output/g/quality/quality.json`) vs task A1's re-run (`output/quality/quality.json`) | **0 of 760** |
| task G's run vs **this** one | **0 of 760** |
| task A1's run vs **this** one | **0 of 760** |

Verdicts: `terracotta` **FAIL**, the other seven **pass** — G's and A1's exactly.

| set | seed | frag acc | precision | correct | wrong pose | non-adj | cross-obj | purity | joins | wall s | confirmed | conf correct | conf false | conf recall | probable | correct found | merges | demoted | outside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `terracotta` | 0 | — | — | — | — | — | — | — | 2 | 3.8 | 2 | 2 | 0 | 1.000 | 0 | 2 | 0 | 0 | 0 |
| `terracotta` | 1 | — | — | — | — | — | — | — | 2 | 1.9 | 2 | 2 | 0 | 1.000 | 0 | 2 | 0 | 0 | 0 |
| `terracotta` | 2 | — | — | — | — | — | — | — | 2 | 2.3 | 2 | 2 | 0 | 1.000 | 0 | 2 | 0 | 0 | 0 |
| `terracotta` | 3 | — | — | — | — | — | — | — | 2 | 2.3 | 2 | 2 | 0 | 1.000 | 0 | 2 | 0 | 0 | 0 |
| `terracotta` | 4 | — | — | — | — | — | — | — | 1 | 2.2 | 1 | 1 | 0 | 0.500 | 1 | 2 | 0 | 0 | 0 |
| `pot_A` | 0 | 62.5 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 10.1 | 4 | 4 | 0 | 0.267 | 13 | 8 | 0 | 0 | 0 |
| `pot_A` | 1 | 50.0 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 7.8 | 4 | 4 | 0 | 0.267 | 14 | 8 | 0 | 0 | 3 |
| `pot_A` | 2 | 37.5 % | 1.000 | 2 | 0 | 0 | 0 | 1.000 | 2 | 7.6 | 3 | 3 | 0 | 0.200 | 9 | 6 | 0 | 0 | 0 |
| `pot_A` | 3 | 62.5 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 7.8 | 4 | 4 | 0 | 0.267 | 8 | 8 | 0 | 0 | 0 |
| `pot_A` | 4 | 62.5 % | 1.000 | 4 | 0 | 0 | 0 | 1.000 | 4 | 7.4 | 6 | 6 | 0 | 0.400 | 9 | 9 | 0 | 0 | 3 |
| `pot_B` | 0 | 55.6 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 12.8 | **4** | **4** | 0 | 0.267 | 22 | 13 | 0 | 0 | 0 |
| `pot_B` | 1 | 33.3 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 10.7 | 3 | 3 | 0 | 0.200 | 18 | 11 | 0 | 0 | 0 |
| `pot_B` | 2 | 55.6 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 10.2 | 4 | 4 | 0 | 0.267 | 17 | 11 | 0 | 0 | 0 |
| `pot_B` | 3 | 33.3 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 10.5 | 3 | 3 | 0 | 0.200 | 22 | 11 | 0 | 0 | 0 |
| `pot_B` | 4 | 33.3 % | 1.000 | 2 | 0 | 0 | 0 | 1.000 | 2 | 11.6 | 3 | 3 | 0 | 0.200 | 19 | 10 | 0 | 0 | 0 |
| `pot_C` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 5.7 | 0 | 0 | 0 | 0.000 | 3 | 2 | 0 | 0 | 0 |
| `pot_C` | 1 | 50.0 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 4.8 | 1 | 1 | 0 | 0.200 | 3 | 2 | 0 | 0 | 0 |
| `pot_C` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 5.0 | 0 | 0 | 0 | 0.000 | 2 | 1 | 0 | 0 | 0 |
| `pot_C` | 3 | 50.0 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 5.8 | 1 | 1 | 0 | 0.200 | 2 | 2 | 0 | 0 | 0 |
| `pot_C` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 5.6 | 0 | 0 | 0 | 0.000 | 3 | 2 | 0 | 0 | 0 |
| `pot_G` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 5.2 | 0 | 0 | 0 | 0.000 | 2 | 0 | 0 | 0 | 0 |
| `pot_G` | 1 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 5.5 | 0 | 0 | 0 | 0.000 | 1 | 0 | 0 | 0 | 0 |
| `pot_G` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 4.5 | 0 | 0 | 0 | 0.000 | 1 | 0 | 0 | 0 | 0 |
| `pot_G` | 3 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 5.3 | 0 | 0 | 0 | 0.000 | 1 | 0 | 0 | 0 | 0 |
| `pot_G` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 4.7 | 0 | 0 | 0 | 0.000 | 2 | 0 | 0 | 0 | 0 |
| `pot_H` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 11.5 | 0 | 0 | 0 | 0.000 | 13 | 3 | 0 | 0 | 0 |
| `pot_H` | 1 | 18.2 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 12.8 | 1 | 1 | 0 | 0.059 | 19 | 3 | 0 | 0 | 0 |
| `pot_H` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 10.9 | 0 | 0 | 0 | 0.000 | 11 | 2 | 0 | 0 | 0 |
| `pot_H` | 3 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 14.2 | 0 | 0 | 0 | 0.000 | 20 | 3 | 0 | 0 | 0 |
| `pot_H` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | 11.8 | 0 | 0 | 0 | 0.000 | 13 | 3 | 0 | 0 | 0 |
| `synthetic_20` | 0 | 35.0 % | 1.000 | 5 | 0 | 0 | 0 | 1.000 | 5 | 27.4 | 5 | 5 | 0 | 0.100 | 18 | 23 | 0 | 0 | 4 |
| `synthetic_20` | 1 | 60.0 % | 1.000 | 9 | 0 | 0 | 0 | 1.000 | 9 | 18.9 | 9 | 9 | 0 | 0.180 | 15 | 24 | 0 | 0 | 3 |
| `synthetic_20` | 2 | 50.0 % | 1.000 | 9 | 0 | 0 | 0 | 1.000 | 9 | 19.5 | 9 | 9 | 0 | 0.180 | 14 | 23 | 0 | 0 | 4 |
| `synthetic_20` | 3 | 45.0 % | 1.000 | 8 | 0 | 0 | 0 | 1.000 | 8 | 18.0 | 8 | 8 | 0 | 0.160 | 14 | 22 | 0 | 0 | 7 |
| `synthetic_20` | 4 | 55.0 % | 1.000 | 10 | 0 | 0 | 0 | 1.000 | 10 | 20.1 | 11 | 11 | 0 | 0.220 | 16 | 27 | 0 | 0 | 9 |
| `mixed_ABG` | 0 | 41.7 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | 58.1 | **8** | **8** | 0 | 0.200 | 79 | **21** | 0 | 0 | 0 |
| `mixed_ABG` | 1 | 29.2 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | 54.7 | 7 | 7 | 0 | 0.175 | 73 | **19** | 0 | 0 | 3 |
| `mixed_ABG` | 2 | 33.3 % | 1.000 | 5 | 0 | 0 | **0** | **1.000** | 5 | 51.2 | 7 | 7 | 0 | 0.175 | 68 | **17** | 0 | 0 | 0 |
| `mixed_ABG` | 3 | 33.3 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | 52.3 | 7 | 7 | 0 | 0.175 | 65 | **19** | 0 | 0 | 0 |
| `mixed_ABG` | 4 | 33.3 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | 53.9 | 9 | 9 | 0 | 0.225 | 72 | **19** | 0 | 0 | 3 |

`merges` and `demoted` are 0 in all forty; the consensus reports 0–9 members "outside" and acts on
none. `mixed_ABG` reads **gate pass** with all three of its checks named in the run's own line:
"Zero false joins in the confirmed tier; cross-object 0 and group purity 1.000; 12 correct joins in
the confirmed and probable bands together at every seed."

---

## 4. Standing gates, on the final tree

| gate | result |
|---|---|
| `cargo fmt --all --check` | **pass** — no output |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass** — 0 warnings |
| `cargo build -p sherd-cli --no-default-features --locked` | **pass** |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass** — 0 warnings |
| `cargo test --workspace --locked` (debug) | **pass** — 19 targets, **442 passed, 0 failed, 3 ignored** |
| `cargo test --release --workspace --locked` | **pass** — **442 passed, 0 failed, 3 ignored** |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, **256 rows** (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed** |
| `pytest -q` | **pass** — **60 passed in 76.9 s** |
| `tools/quality_gate.py`, 8 sets × seeds 0–4 | **exit 1** on the terracotta's recall row alone — the intended state since G — 130 correct confirmed, **0 false**; §3 |
| byte identity, seed 0, `--tiers off --objects off` | **pass** — **160 files over six sets, 143 byte-identical, 17 exempt, 0 differing** |

**Parity, tallied from the sixteen logs.** Every dump's own two figures are G's and A1's, and the
total is the frozen one:

| dump | native | injected | | dump | native | injected |
|---|---:|---:|---|---|---:|---:|
| terracotta | 145 | 631 | | pot_H | 459 | 3 650 |
| `pot_A` | 306 | 2 042 | | `synthetic_20` | 1 040 | 8 603 |
| `pot_B` | 354 | 2 527 | | slab | 81 | 248 |
| `pot_C` | 261 | 1 612 | | | | |
| `pot_G` | 248 | 1 597 | | **total** | | **23 804 / 0** |

**Byte identity, seed 0, `--tiers off --objects off`, previews and meshes on.** The previous step is
A1 (`ec72ad4`) and this step adds one Markdown file under `docs/`, so the two binaries differ only
in the commit string `build.rs` stamps; `id_old` was run before the commit and `id_new` after it,
with a rebuild in between so the stamp actually moved.

**I added `pot_B` and `pot_C` to the four sets the standing gate names**, because none of
terracotta, `pot_A`, `pot_H` and `synthetic_20` has a fragment that is not watertight — so the
standing set cannot exercise the one condition G1 changed, and task G's own identity check had the
same blind spot. `pot_B` carries 3 of 9 and `pot_C` 1 of 7.

| set | files | byte-identical | exempt | **differing** |
|---|---:|---:|---:|---:|
| `terracotta` | 14 | 11 | 3 | **0** |
| `pot_A` | 22 | 20 | 2 | **0** |
| **`pot_B`** | 24 | 21 | 3 | **0** |
| **`pot_C`** | 22 | 19 | 3 | **0** |
| `pot_H` | 30 | 27 | 3 | **0** |
| `synthetic_20` | 48 | 45 | 3 | **0** |
| **total** | **160** | **143** | **17** | **0** |

Every cache, every placed and merged PLY and every PNG is byte-identical. The seventeen exempt files
are `report.json` and `transforms.json` (compared key by key with `engine.commit`, `timings` and
`memory` removed — the standing gate excludes the last two and the commit differs by construction:
`id_old` was stamped `ec72ad4` and `id_new` `099721e`, and `sherd-refit-rs info` was read on both to
confirm the stamp actually moved) and `report.md` (compared above its `## Timing` block). **No key
was added, removed or reordered in any of them.** On A1's own four sets the split is 114 / 103 / 11
/ 0, against A1's 114 / 104 / 10 / 0 — one `report.md` whose timings happened to round the same for
A1 and not for me, which is the block both comparisons exclude.

*(This note's own commit is amended once, to write the three numbers above and the free-disk figure
into the file; the binary that produced `id_new` therefore carries the pre-amend hash. Nothing under
`crates/` moved, so the comparison is unaffected.)*

---

## 5. What I re-derived from A1's own artefacts

A1's `output/acceptance/` survives — six `report.json`, six `transforms.json`, the six
`/usr/bin/time -l` blocks and the scorer's `quality.{md,json}`. Every number of §§1–5 of A1's note
was recomputed from them rather than read:

| A1 § | claim | re-derived | verdict |
|---|---|---|---|
| §1.1 | `synthetic_60` 56 fragments, 143 adjacent pairs, one object | 56 / 143 / `main` only | **exact** |
| §1.1 | `synthetic_170` 164 fragments, 407 pairs, 158 + **6 intruders**, intruders never adjacent to each other | 164 / 407 / 158 + 6, 0 intruder–intruder pairs | **exact** |
| §1.1 | `mixed_all` 164 sherds, 288 pairs, 142 posed, 22 with an object and no pose, ten pots A 8 B 9 C 7 D 32 E 34 F 7 G 7 H 11 I 30 J 19 | identical, pot for pot | **exact** |
| §2.1 | six wall clocks and six `/usr/bin/time -l` peaks | 128.1 / 118.5 / 1 107.0 / 1 104.4 / 1 818.5 / 1 815.3 s; 2 265.6 / 2 288.1 / 2 569.4 / **2 714.9** / 1 753.7 / 1 794.7 MiB | **exact** |
| §2.1 | the six runs' stage timings | `report.json`'s `timings` to three decimals | **exact** |
| §2.3 | per-stage peaks, `RssMonitor`, worst **2 646.6 MiB** | `report.json`'s `memory` block | **exact** |
| §2.3 | working-mesh faces 6.48 M / 9.01 M / 4.66 M, means 116 k / 55 k / 28 k | identical | **exact** |
| §2.2 | `mixed_all` accepts **8 444** candidates | 79 + 8 365 = 8 444, and the run's own log line says `accepted=8444` | **exact** |
| §2.2 | `synthetic_170` accepts **145** | 150 + 314 = **464**, and the log line says `accepted=464`; 145 is the *representative* count (51 + 94) | **wrong unit — V9-D4** |
| §2.2 | the per-pair costs "are within 11 % and 5 %" | +10.5 % (`synthetic_60`), **+36.2 %** (`synthetic_170`), −4.7 % (`mixed_all`) | **true of one row only — V9-D11** |
| §3.1 §3.2 §3.3 | the three quality tables | equal to `output/acceptance/quality/quality.json` row for row | **exact** |
| §3.2 | the probable band's top-K table | 25: 21/19, 50: 33/36, 100: 49/51, 200: 56/58, 400: 59/61, all: 61/62 | **exact** |
| §3.3 | groups 35(5)/8, 36(7)/8, 114(20)/9, 117(21)/8, 143(16)/5, 138(17)/4, singletons 127 and 121 | identical | **exact** |
| §3.4 | the ten false confirmed joins, every score, every arm, every pose error | reproduced field by field, incl. 0.65 / 0.63 / 0.58 / 0.54 / 0.60 t at 8.9° / 2.6° / 1.4° / 1.3° / 1.6°, and 139.1° / 22.6 t and 160.7° / 32.4 t for the two non-adjacent ones | **exact** |
| §3.4 (c) | the three impure groups' consensus, `rejects` empty on all three | 7.308 ± 0.270, 5.766 ± 0.205, 6.643 ± **0.046**; `rejects` `None` | **exact** |
| §3.4 | Pot_E 6.45 ± 0.70 mm over 34, Pot_I 7.52 ± 0.57 over 30, ranges 4.25–12.04 and 4.00–8.80 | **median ± MAD**, not mean ± sd (means are 6.89 and 7.11) — the note's own convention elsewhere, and the ranges are exact | **exact, unit implicit** |
| §3.5 | the support-arm simulation, six rows | 26/14/12, 23/12/11, 51/16/35, 48/7/41, 23/8/15, 28/8/20 — 124 correct lost, 10 false removed | **exact** |
| §3.6 | 5 of 164 not watertight, 16 of 288 pairs (5.6 %) touch one | identical, and the five are Pot_B 02/07/08, Pot_C 01, Pot_J 01 | **exact** |
| §5 | the two-seed table | 15 / 19 / 15 / 4, 33 / 33 / 21 / 12, 12 / 27 / 20 / 7 | **exact** |
| §6 | "every cell of the table below is G's" | 760 cells compared: **0 differing** | **exact** |
| §6 | "task G measured 421.7 s for the same forty" | G measured **610.9 s** (`output/g/quality/quality.json`); 421.7 s is task **H4**'s pre-tier run | **wrong — V9-D9** |
| §7 | parity 256 rows, 213 PASS, 43 SKIP, 0 FAIL, 23 804 / 0 | tallied from A1's own sixteen logs: identical | **exact** |

Of about ninety separate figures recomputed, **three are wrong and one leaves a unit implicit**, and
all three wrong ones are in the commentary rather than in a safety number.

---

## 6. The museum README, tried on the terracotta

The museum-facing block is `README.md:339-486` — the three bands, the review images, the constraints
file, and A1.3's new *«Настоящая коллекция: чего ждать от прогона»*. It gives no single copy-paste
command; it gives flags. I ran every flag it names on `input/test_fragments_1/fragments`, the four
museum scans, and read the output the way a conservator would.

| what the README tells a museum to do | what happened |
|---|---|
| the default run, three bands (`:343-386`) | exit 0. `report.md` has `## Confirmed joins`, `## Probable joins`, `## Rejected` and `## Candidates by fragment` — the four sections the text names. **R §13's two joins and no others, both Confirmed**, 007 unplaced, 3 of 4 fragments placed |
| `--review-images` (`:388-403`) | exit 0, **two** PNGs in `review/`, one per confirmed join, `<A>__<B>.png`, linked from the `image` column of `## Candidates by fragment`. "One per confirmed or probable join", exactly as documented |
| the constraints file **copied verbatim from `:409-416`** | **exit 1** — `constraints.json: no fragment named 'pot_A_01' in this collection`. The README's only worked example mixes terracotta names with `pot_A_01`/`pot_B_03` placeholders and cannot run on any collection. **V9-D18** |
| `must_not_join [021, 094]` (`:426`) | exit 0, **094–104 only**, and `## Constraints` says «the pair was removed before matching and was never scored». Audit §E step 9's exit criterion, met |
| `must_join [007, 021]` (`:427`) | exit 0 — **the run still succeeds** — and `## Constraints` says «**unsatisfiable**: the pair was matched with the second-pass budget and R §6.5 accepted no candidate» |
| `different_object [021, 094]` (`:429`, and the museum section's own advice at `:472-475`) | exit 0, **all six pairs matched** (`at=6 of=6` in the `-v` log), the pair Confirmed by the tier and then refused a placement: «3 of 5 candidate(s) reached the assembly and the pair was refused», with `## Confirmed joins not used` naming it. **It did not filter before matching and it did not save a second of wall clock. V9-D5** |
| `--seed N`, «один сид — это одна попытка» (`:477-481`) | exit 0 with `--seed 4`: **021–094 drops to `probable`** and 094–104 stays Confirmed — G §3's seed-4 story, reproduced from the museum's own instruction |
| `--tiers off` (`:374`) | exit 0, both joins used, no bands — the pre-item-3 behaviour |

**What reads well.** The three-band table (`:349-355`) is the right shape: one row per band, one
column for what it means, one for what the tool does with it. The numbered list of the new section
is written for a person — «работать сверху вниз и остановиться примерно на двухсотой строке; это
рабочий день, а не рабочий месяц» is the sentence a bench needs, and the table behind it reproduces
exactly (§5). `## Candidates by fragment` really is a bench worksheet: best band first, a `why not
confirmed` column, and a clickable image per row. The watertight rule is explained twice, in both
places with the reason rather than the flag.

**What does not.** Five things, two of which would send a museum down the wrong path: **V9-D5** —
the `--constraints` advice, wrong about both what `different_object` does and what it can name;
**V9-D7** — the limitations list, which still promises that fragments of different objects "stay
unassembled"; **V9-D8** — the report-reading table, which still says a zero penetration means the
two sherds do not intersect; **V9-D17** — the two-seed figures, quoted at the three-collection
average rather than at `mixed_all`'s own; and **V9-D18** — the one example `constraints.json`,
which aborts the run.

---

## 7. Defects

Nineteen. **None is a wrong decision and none is a breached prohibition.** Two sit in the shipped
binary's own `--help`, one in the module doc of the file G edited, one is a docstring in the gate
script, and the rest are documents that say something the tree does not do. Ordered by how far they
could send a reader.

### V9-D1 — `tiers.rs`'s own headline still advertises the number G1 changed

`crates/sherd-core/src/tiers.rs:48-49`, the module doc of the file G1 edited:

```
//! **136 confirmed correct joins and 0 false ones** over the eight development sets at seeds 0–4 —
//! 17.7 % of the 770 ground-truth adjacent pairs those forty runs could have found.
```

*What the tree does:* **130 and 0 false, 16.9 %** — the gate prints it on every run (§3), G's note
§5 carries it, `README.md:371` carries it, D carries it. The module that implements the change is
the one place that still advertises the old figure. (The neighbouring `136`s at `:181`, `:268`,
`:287` and `crates/sherd-cli/src/main.rs:336` are **M1's** threshold-search results and are
correctly attributed; only the headline is a claim about the shipped tier. That M1's resample rule
also cost "136 → 130" makes the page harder to read, not wrong.)

### V9-D2 — five places, two of them shipped `--help` text, still say `mixed_ABG` seed 0 has 11 confirmed joins

| where | what it says |
|---|---|
| `crates/sherd-cli/src/main.rs:400` (**`run --help`**) | "switching the arm on takes mixed_ABG seed 0 from **11 confirmed joins to 2**" |
| `crates/sherd-core/src/objects.rs:274` | the same sentence |
| `tools/quality_gate.py:737-739` (**`--help`**) | "costs **nine of mixed_ABG seed 0's eleven**" |
| `docs/superpowers/specs/2026-09-06-rust-core-design.md:1107` | "takes `mixed_ABG` seed 0 from **11 confirmed joins to 2**" |
| `README.md:525-527` | "снимает на `mixed_ABG` (сид 0) **девять подтверждённых стыков из одиннадцати**" |

*What the tree does:* since G1, `mixed_ABG` seed 0 has **8** confirmed joins (§3), and **three of
the eleven are exactly the three G1 removed** (`Pot_B_Piece_01+02`, `01+08`, `02+08` — §1.1). The
denominator is stale in all five places and the numerator is **unmeasured on this tree**: what
`--object-disagreement on` costs today is between 0 and 8, and nobody has run it. This is the
argument that keeps a shipped default off, and it now rests on a run the tree no longer produces.

### V9-D3 — `--tier-max-pen`'s help does not say the flag cannot widen G1's refusal

`crates/sherd-cli/src/main.rs:316`: "Penetrating surface fraction a confirmed join may not exceed
(M1 §3; R §6.5 ships 0.005)." `crates/sherd-core/src/tiers.rs:280-281` says the same of the field.

*What the code does:* `tiers.rs:344-348` pushes `pen: penetration not measurable, a fragment is not
watertight` **before** the ceiling is consulted, and G's own test asserts that with `max_pen: 1.0`
the pair is still refused. A conservator who raises `--tier-max-pen` to admit a borderline
penetration will find that every pair touching a fragment with holes stays Probable regardless —
16 of `mixed_all`'s 288 true pairs, 5.6 % — with nothing in the flag's own help to say why. The
README's tier section (`README.md:366-372`) explains the rule; the flag does not point at it.

### V9-D4 — `synthetic_170` accepted **464** candidates, not 145, and the comparison mixes two units

| where | what it says |
|---|---|
| `docs/superpowers/notes/2026-09-10-a1-acceptance.md:129` | "the same stage is 16.8 s, because **145 candidates were accepted** there rather than 8 444" |
| `docs/superpowers/notes/2026-09-10-a1-acceptance.md:131` | "on real sherds the accepted list is **fifty times longer**" |
| `docs/superpowers/specs/2026-09-06-rust-core-design.md:1739` | "On synthetic 170 the same stage is 16.8 s, because **145 candidates were accepted** there" |
| `docs/superpowers/specs/2026-09-06-rust-core-design.md:1741` | "the accepted list is **fifty times longer** than on a synthetic one-pot collection of the same size" |
| `README.md:788` | "потому что там **принято 145 кандидатов**" |

*What the code does:* `crates/sherd-core/src/tiers.rs:540` — the probes run "for every **accepted**
candidate". `output/acceptance/synthetic_170/seed0/report.json` holds **150 confirmed + 314
probable = 464** accepted candidates, and my re-run of the same seed prints it in one line:

```
INFO matching done seconds=1073.620977 candidates=54045 accepted=464
```

**145** is the count of *pair representatives* (51 confirmed + 94 probable) — the unit of the
quality table, not of the tier pass. The 8 444 it is set against **is** a candidate count
(`mixed_all`: 79 + 8 365, and its own log line says `accepted=8444`); that collection's
representative count is 2 784. The paragraph compares `mixed_all`'s candidates with
`synthetic_170`'s representatives.

*What follows:* the accepted list is **18× longer**, not fifty times. Per accepted candidate the
probe costs 580.5 / 8 444 = **69 ms** on `mixed_all` and 16.789 / 464 = **36 ms** on
`synthetic_170`, so the true statement is that a real ten-pot collection costs about **twice** as
much per candidate *and* has **eighteen times** as many; 145 implies the opposite (16.8 / 145 =
116 ms, i.e. that the probe is three times *cheaper* on ten pots). The 32 % share of `mixed_all`'s
run and the 8 444 are measured and correct.

### V9-D5 — the museum section says `different_object` filters before matching and speeds the run up; it does neither

`README.md:472-475`: "**`--constraints` — главный инструмент на смешанной коллекции.** Список
`different_object` **для пар горшков** … убирает именно тот класс ошибок, который геометрия здесь
сделать не может, — и **делает это до сопоставления, то есть ещё и ускоряет прогон**."

*What the code does:* `crates/sherd-core/src/pipeline.rs:488` — the pre-matching pair filter calls
`forbids`, and `crates/sherd-core/src/assembly/constraints.rs:243` documents `forbids` as "the test
that removes it before matching — **`must_not_join`**". `different_object` is read by
`greedy.rs:402` and `:536`, inside the assembly; `constraints.rs:381` says so: "nothing to do here
either; `assemble_under` refuses the pair." *Measured on the terracotta* (§6): with
`different_object [021, 094]` the run matches **all six pairs**, the pair is **Confirmed** by the
tier, and only the placement is refused. It costs the same wall clock as no constraint at all.
`README.md:426` and the CLI help both say this correctly two hundred lines away.

The same sentence has a second problem for a bench: `different_object` takes **pairs of fragment
names**, validated against the collection — there is no way to name a pot. Separating Pot_E from
Pot_I on `mixed_all`, which is what the paragraph proposes, means writing out **34 × 30 = 1 020**
pairs by hand. This is the lever the museum section recommends most strongly and both halves of the
recommendation are wrong.

*(A related nuance the tree handles honestly: a `different_object` pair stays in `## Confirmed
joins`, and `## Confirmed joins not used` names it with the reason immediately below. Only the
Confirmed section's blanket sentence — "The assembly above is built from these and from nothing
else" — is then untrue of that row.)*

### V9-D6 — D §12 carries a shippability clause the shipped tree does not meet

`docs/superpowers/specs/2026-09-06-rust-core-design.md:2142`, step 8's risk column: "**If confirmed
recall collapses below half the true joins the tier is not shippable** and the thresholds go back to
step 7's table rather than being widened to pass."

Measured on this tree: **130 of the 770 ground-truth adjacent pairs — 16.9 %** over the forty
development runs, and **189 of 1 676 — 11.3 %** over A1's six acceptance runs (0.161–0.182,
0.118–0.125, 0.066–0.076 by collection). On T1's other denominator, the joins R §6.5 accepts, it was
40 % at T1 and is lower now. Audit §D.1's own expectation was 60–80 %.

Nothing in the tree hides this: T1 §4, V8's question 2, G's README paragraph and A1 §3.1 all print
it, and V8 says in as many words "it is not a defect, but it is the number a museum will ask about
first". What no step in the range does is go back to the sentence in D §12 that says a tier at this
recall does not ship — and **G1 moved the number further from it** (136 → 130) without touching the
cell. Either the clause is stale and should say so, or it is live and the thresholds are a
step-12 question; today D §12 contradicts the tree.

### V9-D7 — the README's limitations list promises the tool never joins two objects

`README.md:258`: "**Фрагменты от разных предметов**, попавшие в одну папку, остаются несобранными
или образуют отдельные группы. **Это нормальный исход, а не ошибка.**"

A1's own acceptance measured **three confirmed cross-object joins** on `mixed_all` at seed 1 and
group purity **0.930**, and A1.3's museum section says so in bold 200 lines further down. The
limitations list is where a conservator looks first when a run surprises them, and it still promises
the failure mode the acceptance found.

### V9-D8 — the report-reading table still reads a zero penetration as "they do not intersect"

`README.md:99`: "`penetration` … порог 0.005; **0.0000 означает, что фрагменты не пересекаются**."
`README.md:262`: "**Незамкнутые меши** лишаются проверки на проникновение … и ложный стык отсеять
сложнее."

That is exactly the reading G1 removed from the Confirmed band: where a fragment is not watertight
R §6.4 could not run and `pen` is 0 because the question was **refused**
(`crates/sherd-core/src/tiers.rs:136-142`). G put the correct reading in the tier section
(`README.md:366-372`) and in `ScoreRow::pen_unavailable`'s doc, and left the table a conservator
actually reads `report.md` with. The limitations bullet also states the old consequence: since G1
the consequence is not that a false join is harder to reject, it is that the pair can never be
Confirmed at all.

### V9-D9 — A1 §6 attributes 421.7 s to task G and explains a difference that is not there

`docs/superpowers/notes/2026-09-10-a1-acceptance.md:389`: "**40 runs in 10.0 min** (602.2 s; task G
measured **421.7 s** for the same forty on the same machine — this run followed two hours of
full-load acceptance work and **the difference is thermal, not a change** …)".

G's own note §5 says **610.9 s** and `output/g/quality/quality.json`'s `meta.wall_total_s` is
610.9419597. A1's 602.2 s is **1.4 % faster** than G's, not 43 % slower, and no thermal story is
needed — this task's own run of the same forty took **602.6 s**. 421.7 s is task **H4**'s figure,
measured before the tier pass was the default: the four shipped-default runs on disk are
602.2 / 610.9 / 611.7 / 615.0 s and the two `--tiers off` runs are 412.2 and 459.8 s.

### V9-D10 — the quality gate's cost is documented at H4's pre-tier number

`docs/superpowers/specs/2026-09-06-rust-core-design.md:1952` and `README.md:599` both say
"**Measured: 40 runs in 7.0 min** … (421.7 s, cold caches) … against the 25-minute budget".
`quality_gate.py`'s defaults are `--tiers on --objects on`, and every measured run of the
shipped-default gate is **602.2 / 610.9 / 611.7 / 615.0 s — 10.0 to 10.3 min**, this task's 602.6 s
among them. A1.1 edited both paragraphs — it inserted the `--score-only` text immediately after each
— and left the figure. The 25-minute budget is still met with room.

### V9-D11 — "the per-pair costs held to 5–11 %" is true of `synthetic_60` only

`docs/superpowers/notes/2026-09-10-a1-acceptance.md:125`: "The per-pair costs the projection used —
0.523 core-s synthetic, 0.870 core-s real — are **within 11 % and 5 %** of what the run measured."
Repeated at `docs/superpowers/specs/2026-09-06-rust-core-design.md:2146` ("held to 5–11 %") and
`README.md:784` ("ошибся на 5–11 % по цене пары").

From A1's own table three lines above the sentence: `synthetic_60` **0.58** core-s against 0.523 =
**+10.5 %**, `synthetic_170` **0.71** against 0.523 = **+36.2 %**, `mixed_all` **0.83** against
0.870 = **−4.7 %**. The sentence names "0.523 core-s synthetic" and pairs it with the best-fitting
of the two synthetic rows; the collection the paragraph is about — `synthetic_170`, whose 17-minute
projection it is discussing — is **36 % off**. The honest statement is that the *end-to-end wall*
predictions were close (−12.5 %, +8.8 %, +8.2 %), and that they were close because the projection's
per-pair cost was 36 % low and its 6.43× pool speed-up 33 % low in the other direction.

### V9-D12 — D §12 quotes a warm figure no run produced

`docs/superpowers/specs/2026-09-06-rust-core-design.md:2146`, step 12's gate cell: "… and
**2.1–2.6 min** / 18.4 min / 30.3 min warm". A1 §2.1, A1 §2.2 and D §10.3's own table (`…:1721`)
all give `synthetic_60` warm as **2.0 min** (118.5 s), and none of the six runs measured 2.6 minutes
of anything.

### V9-D13 — the worst peak RSS is quoted from two instruments in one paragraph

`docs/superpowers/specs/2026-09-06-rust-core-design.md:1719-1726` (§10.3) introduces its table with
"Peak RSS from `/usr/bin/time -l`", carries **2 715 MiB** for `synthetic_170` warm, and then says
two lines later "the worst peak resident set of the six runs is **2 647 MiB**" — `RssMonitor`'s
number from `report.json`, not the table's. D §1's measured cell (`…:36`) says 2 647 too, and A1
§4's row mixes them inside one line (`synthetic_60` 2 288 is `/usr/bin/time -l`'s; 2 647 and 1 794
are `RssMonitor`'s). Both instruments are far under the 6 GB row — the worst *process* peak of A1's
six runs is **2 715 MiB**, and my re-run of `synthetic_170` seed 0 measured **2 751 MiB** — so
nothing is breached; the paragraph should pick one instrument and say which.

### V9-D14 — D §8 was not given the measurement A1's own note asks it for

A1 §2.3: "§8's 4.4 GB whole-mesh BVH total is an extrapolation from 170 × 200 000 faces … The
per-face constant is not challenged by this run; the face count is, and **D §8 should say so**."

`docs/superpowers/specs/2026-09-06-rust-core-design.md:1082` still reads "≈ **4.4 GB** at
170 × 200 k faces" with no note, and §8's peak-RSS row (`…:1087`) still reads "measured on the
largest development set, warm … **1.9 GiB**" — the largest *development* set being 24 fragments.
D §10.3 carries the correction (6.48 / 9.01 / 4.66 M working-mesh faces, 0.84 / 1.16 / 0.60 GB of
tree, worst peak 2 647 MiB over 164 fragments); §8, the section a reader goes to for the memory
model, does not.

### V9-D15 — `tier_verdict`'s docstring still states the constant G deleted

`tools/quality_gate.py:456-458`:

```
Plus, for the terracotta, M1 §5.6's restatement of audit §E's step-8 gate: the two joins are
at least probable at every seed, nothing else is ever confirmed, and nine of the ten
(seed, join) slots are confirmed.
```

Twenty lines below, the code checks `slots < TERRACOTTA_CONFIRMED_SLOTS` with
`TERRACOTTA_CONFIRMED_SLOTS = 2 * 5` (`…:218`), whose comment says in as many words that a gate
fitted at nine "cannot fail, and a gate that cannot fail is not a gate". The module docstring
(`…:79-82`) and D §10.4 both say ten. This docstring is the one place left in the tree that asserts
nine, and it cites M1 §5.6 — the fitted reading V8-D3 asked to be removed.

*(In the same family, and true as written but easy to misread: D `…:1962` and `README.md:555` both
say the gate's prohibitions "hold at all five seeds on this tree", one sentence after "a failure of
one of those exits non-zero". Both are correct — the three prohibitions do hold and the run exits 1
on the *recall* row, which each document explains further down — but a reader checking "is the gate
green?" gets the wrong answer from that sentence alone.)*

### V9-D16 — D §12's step-10 row still states the narrow reading G restated in §10.4

`docs/superpowers/specs/2026-09-06-rust-core-design.md:2144`: "`mixed_ABG`: cross-object joins **0**
in the confirmed tier, group purity **1.000**, **correct joins ≥ 12** (§10.3's baseline) at seeds
0–4." Under that reading the row is unmet — correct *confirmed* joins are **8 / 7 / 7 / 7 / 9**.
D §10.4 (`…:1986`) and `quality_gate.py` gate the 12 over the **confirmed and probable bands
together** (measured 21 / 19 / 17 / 19 / 19) and give the argument; D §12's own plan row does not
carry it, so D says two things about the same row. Audit §E's step-10 row
(`notes/2026-09-09-fable-audit.md:571`) is likewise unmarked, where the other two notes G corrected
(`o1-objects.md`, `t2-review-constraints.md`) both carry an explicit *"corrected in task G"* marker.

### V9-D17 — the museum section quotes three-collection averages for the one collection it is about

`README.md:477-479` (item 6): "Между сидом 0 и сидом 1 совпадает около **трёх пятых** полосы
confirmed, а **четыре пятых** расхождения — это стыки, которые на другом сиде всего лишь
`probable`." Those are A1 §5's averages over all three collections. On `mixed_all` — the collection
that section is about — it is **12 of 23 and 12 of 28, 52 % and 43 %**, and 20 of 27 (74 %) of the
difference. The section's whole point is that a mixed collection behaves worse; this line makes it
look better.

One line up, `README.md:466` and A1 §3.2 (`…:229`) say the rows below the top hundred are worth
"**eleven** more joins". At seed 1 it is eleven (62 − 51); at seed 0 it is **twelve** (61 − 49).

### V9-D18 — the README's only `constraints.json` example cannot be run

`README.md:409-416` prints one worked file and it is the only one a museum has. The first two lists
name terracotta scans, the last two name `pot_A`/`pot_B` placeholders that exist in no collection.
Copied verbatim onto the terracotta it **fails the run**:

```
Error: assembling input/test_fragments_1/fragments
Caused by:
    constraints.json: no fragment named `pot_A_01` in this collection
```

which is the validator behaving exactly as `README.md:418-421` promises ("неизвестное имя фрагмента
— **ошибка**"). The rule is right and the message is excellent; the example is the problem. Split it
in two, or use one collection's names throughout, and a conservator's first attempt succeeds.

*(Smaller, same family: the `--score-only` example at `README.md:538-540` writes into `--out
output/acceptance/quality`, which is where task A1's own acceptance table lives. Run verbatim it
replaces that six-row table with a one-row one.)*

### V9-D19 — A1's acceptance runs cannot be byte-reproduced from A1's own tree

Not a mistake, but it defeats the check this task was asked to make, so it belongs in the list. All
six of A1's acceptance runs stamp `engine.commit` `486c91906ae1348acd8f1492d5b1eef2064db968` —
**task G's HEAD** — because the binary was built at the start of A1 and A1 changed no crate code;
its forty-run gate stamps `58704d5` (A1.2) for the same reason. `transforms.json` carries
`engine.commit`, so "`transforms.json` byte-identical to A1's seed-0 file" is unmeetable from
`ec72ad4`, or from any commit but `486c919`. What *is* meetable, and what §2.2 measures, is
byte-identity after substituting that one 40-character string; both collections meet it, and the
whole of `report.json` meets it too. A1 §7 says this about the two binaries it used for its own
byte-identity check and does not say it about the six acceptance runs; one sentence there would have
closed it.

---

## 8. Housekeeping

**Disk.** The task began with **9.7 GiB** free and never fell below **7.0 GiB**. What it wrote is
`output/v9/`: the two acceptance runs, the terracotta trees of §6, the two byte-identity trees, the
forty-run gate and the acceptance scoring, and the gate logs under `output/v9/gates/`. **The two
large collections' fragment caches were deleted once the runs were scored** — 243 MiB and 164 MiB,
413 MiB with their `-v` logs — which is the item the brief asks for; `report.json`, `report.md`,
`transforms.json` and `time.log` of both runs are kept, because §2 and §5 are computed from them.
Free disk now: **8.2 GiB**, after the two byte-identity trees (1.6 GiB) were compared and removed.

Nothing under `input/`, `fixtures/` or `output/fixtures/` was written or deleted, and no other
task's `output/` tree was touched — `output/acceptance/`, `output/g/`, `output/v8/`, `output/o1/`,
`output/quality/` and `output/a1/` are all as I found them, because every number in §1 and §5 is
read from them (the acceptance scoring of §2.3 went to `output/v9/quality_acceptance/`, not to
`output/acceptance/quality/`, for that reason). `sherd_refit/` was not touched, the branch was never
switched, nothing was pushed, nothing was stashed and nothing was sent anywhere.

**Large-set runs.** Exactly two, both at seed 0, both cold: `input/synthetic_pingsdorf_170/fragments`
and `input/sfspp/mixed_all`. That is the allowance the brief gives, and this note is part of the
final acceptance §E step 12 names. Nothing else above 27 fragments was matched; `mixed_ABG` (24
meshes) inside the forty-run gate is the largest of the rest.

**Commit.** One, this note. No file under `crates/`, `tools/`, `docs/superpowers/specs/` or
`README.md` was edited by this task: a verification that fixes what it finds cannot report on it,
and all nineteen defects of §7 are left for the step that takes them.
