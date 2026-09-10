# G — V8's six defects closed, and the one that stays open on purpose

**Date:** 2026-09-10. **Tree:** branch `rust-core`, from `956c0ca` (V8's note) to `ba7b738`, four
commits G1–G4 plus this note. **Machine:** Apple M2 Pro, 10 cores, 16 GB, rustc 1.97.0, `--release`,
backend `cpu`. **Sets:** the eight development sets only; the largest collection matched is
`mixed_ABG`, **24** meshes.

**What this task did.** V8-D1 is a change to the confirmed tier: a pair whose penetration cannot be
measured is no longer confirmed. V8-D2 and V8-D3 are the two rows of audit §E that no script could
fail; both are now stated in `tools/quality_gate.py` on the tier they are about, and one of them
**fails on this tree and is meant to**. V8-D4, D5 and D6 are documentation.

**The headline number moves and the prohibition does not.** Over the forty runs of the gate the
confirmed tier now holds **130 correct joins and 0 false** (was 136 and 0), 16.9 % of the 770
ground-truth adjacent pairs. Every one of the six joins it gave up is a *correct* join on a fragment
with holes. Two of the forty runs changed; the other thirty-eight are cell for cell what V8 measured.

---

## 1. V8-D1 — `pen ≤ 0` is a test, so missing evidence refuses it

### 1.1 The rule

`Thresholds::refusals` (`crates/sherd-core/src/tiers.rs`) states its own reading in a doc comment:
*"A test whose evidence is missing fails: a candidate with no seam to slide along has not passed the
slide probe, it has refused it."* The slide obeyed it and so did the margin. The penetration did
not: `refusals` read `scores.pen`, and `verify::penetration_scores` returns `(0.0, 0.0, true)` when
either fragment is not watertight — the third element being `pen_unavailable`, the reference's own
refusal. A `pen ≤ 0` ceiling reads that `0` as a pass. The clause was therefore vacuous on exactly
the pairs where R §6.4 could not run.

The fix is the one line V8 named, with the reason a conservator reads:

```rust
if scores.pen_unavailable {
    failed.push("pen: penetration not measurable, a fragment is not watertight".to_owned());
} else {
    ceiling("pen", scores.pen, self.max_pen, &mut failed);
}
```

**R §6.5 is untouched.** `verify::accept` keeps reading the `0` as "no penetration found": that is
the reference's arithmetic, it is what the parity harness freezes at 23 804 checks, and it is why
the join lands in **Probable** and not in Rejected, and why `--tiers off` is byte-identical. The
band a museum is told to trust is the only thing that moved.

**It is not a threshold a flag can widen.** `--tier-max-pen` names a limit; the refusal is that
there is nothing to compare against it. The test asserts this: with `max_pen: 1.0` the pair is still
refused, and a *measured* `pen = 0` still passes.

Test: `a_pair_whose_penetration_cannot_be_measured_is_never_confirmed`, plus a sixth row in
`each_threshold_refuses_on_its_own` so that the new clause is one of the tests proved to refuse on
its own. 442 test functions per profile now (was 441), 3 `#[ignore]`d.

### 1.2 Which sets have fragments that are not watertight

Read off `report.json`'s `fragments[].watertight` on a fresh run of each set:

| set | not watertight | which |
|---|---:|---|
| terracotta | 0 of 4 | — |
| `pot_A` | 0 of 8 | — |
| `pot_B` | **3 of 9** | `Pot_B_Piece_02`, `Pot_B_Piece_07`, `Pot_B_Piece_08` |
| `pot_C` | **1 of 7** | `Pot_C_Piece_01` |
| `pot_G` | 0 of 7 | — |
| `pot_H` | 0 of 11 | — |
| `synthetic_20` | 0 of 20 | — |
| `mixed_ABG` | **3 of 24** | the same three `pot_B` pieces |

Five of the eight sets are watertight throughout, and on those five **nothing moved** — which is
what the brief expected.

### 1.3 Which confirmed joins moved, and what they were

Every one of the forty runs was compared, cell by cell, against V8's own `quality.json`. **Two runs
differ and thirty-eight do not:**

| run | confirmed | conf correct | conf false | joins used | frag acc |
|---|---|---|---|---|---|
| `pot_B` seed 0 | 7 → **4** | 7 → **4** | 0 → **0** | 5 → 3 | 77.8 % → 55.6 % |
| `mixed_ABG` seed 0 | 11 → **8** | 11 → **8** | 0 → **0** | 8 → 6 | 50.0 % → 41.7 % |

The joins are the same three in both, and they are the three V8 named:
`Pot_B_Piece_01+Pot_B_Piece_02`, `Pot_B_Piece_01+Pot_B_Piece_08`, `Pot_B_Piece_02+Pot_B_Piece_08`.
All three are **`correct`** by `evaluate.py` — true adjacent joins at the true pose. So the price is
recall and not safety, and it is paid on the two runs where the tier was previously asserting a join
it had four measured tests for and called five.

At seeds 1–4 the same pot_B pairs are not confirmed anyway (they fail `tight` or `gap`), which is
why only seed 0 moves.

**What the refusal did *not* do is remove a false join, because there were none to remove.** The
worst false join the module's own doc names — `Pot_A_Piece_08–Pot_B_Piece_07` on `mixed_ABG`
seed 1 — is refused by the gap and always was. The honest statement of the change is therefore about
the *reason* the tier is clean, not about the number: on a pair with a fragment that has holes, the
confirmed band was a four-test conjunction presented as a five-test one, and now it is five.

---

## 2. V8-D2 — audit §E's step-10 row, restated on the tier it gates

**The row:** *"`mixed_ABG`: cross-object joins 0 in the confirmed tier, purity 1.000, correct joins
≥ 12 (the baseline's) at seeds 0–4."* `BANDS["mixed_ABG"]` carried `pure=False` with the note
"roadmap item 4's baseline (D §10.3), not a gate", so neither half was checked, and O1's restatement
of the recall half lived in prose.

It is now three things in `quality_gate.py`, and the run fails on the first two:

* **`pure_tiers`** — cross-object joins **0** and group purity **1.000** at every seed, gated under
  `--tiers on` (under `--tiers off` the set stays D §10.3's baseline, unchanged). Measured: 0 and
  1.000 at all five seeds. This is the half roadmap item 4 exists to deliver and the half that must
  never regress.
* **`recall_floor = 12`** — correct joins over the **confirmed and probable bands together**, at
  every seed. Measured **21 / 19 / 17 / 19 / 19**, comfortably above.
* **correct confirmed joins, reported per seed** — **8 / 7 / 7 / 7 / 9** — and deliberately *not*
  gated.

**Why the floor is the two bands and not the confirmed one.** The audit's 12 is a count of joins
R §6.5's own assembly *used*, from D §10.3's baseline row, taken before a tier existed. The tier
finds no join R §6.5 refused; it sorts what R §6.5 accepted into bands. So confirmed-and-probable is
the same population the 12 was counted on, and it is also the list a conservator is handed — the
probable section of `report.md` with a reason on every line, plus its review images. Gating
confirmed-correct at a number instead would be a constant fitted to what the tier reaches today: it
would fail every future step that trades a confirmed join for a safer rule (G1 is exactly such a
step, on other sets), and pass every step that loosens one. Reporting it per seed gives a reader the
number without giving a ratchet the wrong direction.

The half of the row that did move under G1 is the confirmed count on `mixed_ABG` seed 0, 11 → 8; the
recall floor did not notice, because the three joins are still on the probable list. That is the
floor behaving as designed.

---

## 3. V8-D3 — why 021–094 is only Probable at seed 4, and why the gate now asks for ten anyway

### 3.1 Which probe refuses, with the numbers

Both joins of R §13 are confirmed at seeds 0–3. At seed 4, `094–104` is confirmed and `021–094` is
**probable**, with one line of evidence in `report.json`:

```
"failed": ["neither arm: support 0 < 1 and no second placement to beat"]
```

Nothing else fails. The candidate's own scores at seed 4 are as good as at seed 0:

| | seed 0 (confirmed) | seed 4 (probable) |
|---|---:|---:|
| `tight` | 0.5566 | 0.5427 |
| `gap`, in `t` | 0.0080 | 0.0082 |
| `seam`, in `t` | 20.67 | 21.00 |
| `cont_n` | 0.9950 | 0.9904 |
| `pen` | 0 (measured) | 0 (measured) |
| slide, in `t` | 7.5e-15 | 1.1e-14 |
| redraws accepting | 3 of 3 | 3 of 3 |
| `placements` | **3** | **1** |
| `margin` | **9.84** | **none** |
| `support` | 0 | 0 |

The pair's kept list holds five candidates at both seeds. At seed 4 all five sit at **one**
placement — five different breakline correspondences (`brk` 0.138, 0.095, 0.087, 0.087, 0.071) that
converge on the same pose. At seed 0 the list holds three at the true placement and **two the
matcher itself rejected**, 23.4 t away: `tight` 0.117 and 0.093 against R §6.5's 0.25, `gap` 0.070
and 0.075 t against a 0.030 t limit. The better of those two is what the margin arm beats by a factor
of 9.84.

So the join is not confirmed at seed 4 because the search produced **no junk placement to beat**, and
it is confirmed at seeds 0–3 because it did. The terracotta is a three-fragment chain (007 never
places), so the support arm has no second path either.

### 3.2 Rule (A): "one placement means nothing to beat, so the margin arm passes" — measured, refused

This is the principled-looking reading, and it is the brief's own candidate in a different form: if
the pair's kept list makes exactly one placement, the lead over a second placement is unbounded, and
an unbounded margin is the *strongest* margin, not a missing one.

It was simulated exactly, over the forty runs: every probable representative whose only refusal is
that line and whose `placements` is 1, classified by `evaluate.py` on its own pose.

| set | would become confirmed: correct | wrong pose | the wrong-pose pairs |
|---|---:|---:|---|
| terracotta | 1 | 0 | — |
| `pot_A` | 16 | **2** | 02–05 at seeds 1, 2 |
| `pot_B` | 15 | 0 | — |
| `pot_C` | 5 | 0 | — |
| `pot_G` | 0 | 0 | — |
| `pot_H` | 13 | **13** | 03–07 at every seed; 02–04 at 1, 2, 3; 03–04 at 1, 3, 4; 02–11 at 3; 09–11 at 0 |
| `synthetic_20` | 37 | 0 | — |
| `mixed_ABG` | 31 | **2** | 02–05 at seeds 1, 2 (`pot_A`'s pair inside the mixed set) |
| **total** | **118** | **17** | |

**Rule (A) would confirm seventeen wrong-pose joins and break the tier's own prohibition.** It buys
the terracotta's tenth slot and 117 other true joins, and it costs the one property the confirmed
band exists for. M1 §3 had already named the family in a comment in `tiers.rs` — *"the pot_H and
pot_A wrong-pose family has a single placement, so the margin can never speak for it, and every
other quantity R §6 measures calls it a true join"* — and this is that comment, measured. The
support arm was moved from audit step 10 into step 8 precisely because of this family, and rule (A)
would undo the move.

### 3.3 Rule (B): "a rival the tier would refuse is no rival" — does not apply, and points the other way

The brief's parenthetical suggests the second placement at seed 4 might be a candidate the tier
itself would refuse. It is not: at seed 4 there is **no** second placement (`placements: 1`). Where
the rule *would* apply is seeds 0–3, and there it makes the margin easier to pass, not harder.

Measured over the 130 confirmed joins of the forty runs, by which arm confirms them:

| arm | joins |
|---|---:|
| support alone (`support ≥ 1`, no qualifying margin) | 68 |
| margin alone (`support = 0`, margin ≥ 2) | **51** |
| both | 11 |

and of the 51 that rest on the margin alone, **47 beat a second placement R §6.5 itself refused** and
only **4 beat one it accepted**. Rule (B) would turn those 47 margins into "no rival worth beating"
— which, on rule (A)'s reading, is a pass, and the two rules compose into exactly the 17 false joins
above. Neither rule survives its own measurement.

### 3.4 What the gate says now

`TERRACOTTA_CONFIRMED_SLOTS` was **9** — the number M1 measured on this tree. A constant fitted to
the observed answer cannot fail: it recorded the audit's row as met while the row was unmet, and it
could not tell nine-at-seed-4 from nine-at-another-seed. It is now `2 * 5`, the brief's ten, and the
run fails on it:

```
* `terracotta` — gate FAIL. recall row -- audit §E step 8 ("terracotta's two joins confirmed"):
  9 of the 10 (seed, join) slots confirmed -- 021-094 at seed 4 only probable. Nothing outside
  R §13's two is confirmed at any seed and both are at least probable at every seed, so no
  prohibition is breached: what is unmet is the audit's recall row.
...
total wall 610.9 s (10.2 min); FAILED: terracotta
  no prohibition was breached: every failure above is a recall row -- a true join the tier declined
  to assert, on the probable list with its reason
```

**The gate exits non-zero from this commit onward, and that is the intended state.** So that the
failure stays readable rather than becoming noise, a gate reason now carries a `recall row --`
marker and the run's last line separates the two kinds of failure: a breached *prohibition* is a
join the tool asserts and the pot denies; an unmet *recall row* is a true join the tool declined to
assert and put on the conservator's probable list with the reason beside it. A reader of a red run
can tell in one line which of the two happened.

Closing the slot for real needs an algorithm change and not a threshold: R §5.7's `keep` returning a
second placement for that pair, or a fifth kind of evidence the margin and the support do not carry.
That is a step-12 question and nobody has made it.

---

## 4. V8-D4, D5, D6 — the documentation

* **D4.** The twelve-line doc comment of `markdown_row` (`crates/sherd-parity/src/stages/outputs.rs`)
  had `fn group_ids` inserted between it and the function it described. Moved back onto
  `markdown_row`; `group_ids` keeps its own two-line comment. No arithmetic — the parity total is the
  frozen 23 804 / 0 per dump, re-run below.
* **D5.** `crates/sherd-core/tests/assembly_graphs.rs` said "four thousand random graphs of **six**
  fragments" where the code is `const N: u32 = 8`. Corrected, and both the test's doc and O1 §3's
  paragraph now claim what the evidence supports: the argument from R §8's greedy rule shows no
  reachable state has an accepted join spanning two groups, and four thousand graphs are a sample
  that finds no hole in it. "It cannot" is gone from both.
* **D6.** T2 §7 called `mixed_ABG` a 27-fragment set in the same sentence that called it 24.
  `ls input/sfspp/mixed_ABG/*.obj` is **24**; the sentence is corrected with a note saying so. Every
  other "27" in the tree is the small-set *rule*, not this set's size, and is left alone.

---

## 5. The quality gate, forty runs

`python tools/quality_gate.py --out output/g/quality`, shipped defaults (`--tiers on --objects on
--object-disagreement off`), backend `cpu`, seeds 0–4, eight sets. **610.9 s (10.2 min)**, header
stamp `2026-09-10 04:47`, commit `956c0ca…`: the binary was built from the working tree after G1's
and G4's code changes and before either was committed, so `report.json` stamps the parent commit.
No crate source changed after that build — everything later in this task is `tools/`, `docs/` and
`README.md` — so the table is the final tree's behaviour, and §6's gates were re-run on it.

**Exit 1**, on the terracotta's recall row alone. **130 correct confirmed joins, 0 false**, 16.9 % of
the 770 ground-truth adjacent pairs; with the probable band, 342 correct, 44.4 %.

| set | seed | frag acc | prec | correct | wrong pose | non-adj | cross-obj | purity | joins | confirmed | conf correct | conf false | conf recall | probable | correct found |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `terracotta` | 0 | — | — | — | — | — | — | — | 2 | 2 | 2 | **0** | 1.000 | 0 | 2 |
| `terracotta` | 1 | — | — | — | — | — | — | — | 2 | 2 | 2 | **0** | 1.000 | 0 | 2 |
| `terracotta` | 2 | — | — | — | — | — | — | — | 2 | 2 | 2 | **0** | 1.000 | 0 | 2 |
| `terracotta` | 3 | — | — | — | — | — | — | — | 2 | 2 | 2 | **0** | 1.000 | 0 | 2 |
| `terracotta` | 4 | — | — | — | — | — | — | — | 1 | 1 | 1 | **0** | 0.500 | 1 | 2 |
| `pot_A` | 0 | 62.5 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | **0** | 0.267 | 13 | 8 |
| `pot_A` | 1 | 50.0 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | **0** | 0.267 | 14 | 8 |
| `pot_A` | 2 | 37.5 % | 1.000 | 2 | 0 | 0 | 0 | 1.000 | 2 | 3 | 3 | **0** | 0.200 | 9 | 6 |
| `pot_A` | 3 | 62.5 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | **0** | 0.267 | 8 | 8 |
| `pot_A` | 4 | 62.5 % | 1.000 | 4 | 0 | 0 | 0 | 1.000 | 4 | 6 | 6 | **0** | 0.400 | 9 | 9 |
| `pot_B` | 0 | 55.6 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | **4** | **4** | **0** | 0.267 | 22 | 13 |
| `pot_B` | 1 | 33.3 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 3 | 3 | **0** | 0.200 | 18 | 11 |
| `pot_B` | 2 | 55.6 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 4 | 4 | **0** | 0.267 | 17 | 11 |
| `pot_B` | 3 | 33.3 % | 1.000 | 3 | 0 | 0 | 0 | 1.000 | 3 | 3 | 3 | **0** | 0.200 | 22 | 11 |
| `pot_B` | 4 | 33.3 % | 1.000 | 2 | 0 | 0 | 0 | 1.000 | 2 | 3 | 3 | **0** | 0.200 | 19 | 10 |
| `pot_C` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 3 | 2 |
| `pot_C` | 1 | 50.0 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 1 | 1 | **0** | 0.200 | 3 | 2 |
| `pot_C` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 2 | 1 |
| `pot_C` | 3 | 50.0 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 1 | 1 | **0** | 0.200 | 2 | 2 |
| `pot_C` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 3 | 2 |
| `pot_G` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 2 | 0 |
| `pot_G` | 1 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 1 | 0 |
| `pot_G` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 1 | 0 |
| `pot_G` | 3 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 1 | 0 |
| `pot_G` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 2 | 0 |
| `pot_H` | 0 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 13 | 3 |
| `pot_H` | 1 | 18.2 % | 1.000 | 1 | 0 | 0 | 0 | 1.000 | 1 | 1 | 1 | **0** | 0.059 | 19 | 3 |
| `pot_H` | 2 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 11 | 2 |
| `pot_H` | 3 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 20 | 3 |
| `pot_H` | 4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 | **0** | 0.000 | 13 | 3 |
| `synthetic_20` | 0 | 35.0 % | 1.000 | 5 | 0 | 0 | 0 | 1.000 | 5 | 5 | 5 | **0** | 0.100 | 18 | 23 |
| `synthetic_20` | 1 | 60.0 % | 1.000 | 9 | 0 | 0 | 0 | 1.000 | 9 | 9 | 9 | **0** | 0.180 | 15 | 24 |
| `synthetic_20` | 2 | 50.0 % | 1.000 | 9 | 0 | 0 | 0 | 1.000 | 9 | 9 | 9 | **0** | 0.180 | 14 | 23 |
| `synthetic_20` | 3 | 45.0 % | 1.000 | 8 | 0 | 0 | 0 | 1.000 | 8 | 8 | 8 | **0** | 0.160 | 14 | 22 |
| `synthetic_20` | 4 | 55.0 % | 1.000 | 10 | 0 | 0 | 0 | 1.000 | 10 | 11 | 11 | **0** | 0.220 | 16 | 27 |
| `mixed_ABG` | 0 | 41.7 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | **8** | **8** | **0** | 0.200 | 79 | **21** |
| `mixed_ABG` | 1 | 29.2 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | 7 | 7 | **0** | 0.175 | 73 | **19** |
| `mixed_ABG` | 2 | 33.3 % | 1.000 | 5 | 0 | 0 | **0** | **1.000** | 5 | 7 | 7 | **0** | 0.175 | 68 | **17** |
| `mixed_ABG` | 3 | 33.3 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | 7 | 7 | **0** | 0.175 | 65 | **19** |
| `mixed_ABG` | 4 | 33.3 % | 1.000 | 6 | 0 | 0 | **0** | **1.000** | 6 | 9 | 9 | **0** | 0.225 | 72 | **19** |

`merges` is 0 in all forty, `demoted` is 0 in all forty, and the consensus reports 0–9 members
"outside" per run and acts on none — unchanged from V8.

**Verdicts.** `pot_A`, `pot_B`, `pot_C`, `pot_G`, `pot_H`, `synthetic_20`, `mixed_ABG`: **gate
pass**. `terracotta`: **gate FAIL** on audit §E step 8's recall row, 9 of 10 slots, no prohibition
breached.

---

## 6. Standing gates, on the final tree

| gate | result |
|---|---|
| `cargo fmt --all --check` | **pass** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass** |
| `cargo build -p sherd-cli --no-default-features --locked` | **pass** |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass** |
| `cargo test --workspace --locked` (debug) | **pass** — 442 passed, 3 ignored |
| `cargo test --release --workspace --locked` | **pass** — 442 passed, 3 ignored |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, 256 rows (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed** |
| `pytest -q` | **pass** — 60 passed in 79.4 s |
| `tools/quality_gate.py` | **exit 1** on the terracotta's recall row alone; 130 correct confirmed, 0 false |
| byte identity, seed 0, `--tiers off --objects off` | **pass** — 71 files, 60 byte-identical, 11 exempt, **0 differing** |

Per-dump parity, every one equal to task F's and V8's own row:

| dump | native | injected | | dump | native | injected |
|---|---:|---:|---|---|---:|---:|
| terracotta | 145 | 631 | | pot_H | 459 | 3 650 |
| pot_A | 306 | 2 042 | | `synthetic_20` | 1 040 | 8 603 |
| pot_B | 354 | 2 527 | | slab | 81 | 248 |
| pot_C | 261 | 1 612 | | | | |
| pot_G | 248 | 1 597 | | **total** | | **23 804 / 0** |

**The byte-identity check** was made against a fresh build of `956c0ca` (V8's tree) exported with
`git archive` into the session scratchpad and built with its own `CARGO_TARGET_DIR`; the repository's
branch was never switched and its binary's commit stamp was never disturbed. Both binaries ran
terracotta, `pot_A`, `pot_H` and `synthetic_20` at seed 0, CPU, `--tiers off --objects off`, previews
and meshes **on**, into fresh directories:

| set | files | byte-identical | exempt | **differing** |
|---|---:|---:|---:|---:|
| terracotta | 10 | 8 | 2 | **0** |
| `pot_A` | 14 | 11 | 3 | **0** |
| `pot_H` | 19 | 16 | 3 | **0** |
| `synthetic_20` | 28 | 25 | 3 | **0** |
| **total** | **71** | **60** | **11** | **0** |

Every PLY and every PNG is byte-identical. The eleven exempt files are `report.json` and
`transforms.json` (compared key by key with `engine.commit`, `timings` and `memory` removed — the
standing gate excludes the last two, and the commit differs by construction) and `report.md`
(compared above its `## Timing` block). No key was added, removed or reordered in any of them. That
is the same 71 / 60 / 11 / 0 split V8 measured against `c6e38bc`, which is what "`--tiers off` is the
previous algorithm" means after a change to the tier.

---

## 7. Housekeeping

This task wrote `output/g/` only: the forty-run sweep's `report.json` and `transforms.json`
(`sweep/`, kept — it is what §1.3 and §3.2 are computed from), the quality gate's table and its
per-run `evaluate.py` dumps (`quality/`), and the five logs. The byte-identity trees and the
`956c0ca` build under the scratchpad were compared, measured and deleted. Free disk stayed above
**5.8 GiB** throughout. Nothing under `input/`, `fixtures/` or `output/fixtures/` was written or
deleted; no other task's `output/` tree was touched; `sherd_refit/` was not touched, and no commit
in this range names it.

Nothing above 24 fragments was matched: `mixed_ABG` (24 meshes) inside the sweep and inside the gate
is the largest collection this task ran, twice over. No `segment --features` pass was run.

**What a next step inherits.** The quality gate exits 1 until the terracotta's tenth slot is earned
by an algorithm change; §3 says what has already been measured and refused, so that nobody re-finds
it. Both of the rules tested here are cheap to re-test — the sweep under `output/g/sweep/` carries
every candidate's evidence, and `tiers::Probes` already computes `placements` and `rival_score`.
