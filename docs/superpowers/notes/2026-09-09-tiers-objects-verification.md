# V8 — independent verification of F, M1, T1, T2 and O1

**Date:** 2026-09-10. **Tree:** branch `rust-core` at `59be405` (O1's last commit), clean at the
start and clean apart from this note at the end. **Range verified:** every commit after `c6e38bc`
(V7) — `b9d8345`…`59be405`, 35 commits: task F (H1-D1 and V7-D1…D5), M1 (audit §E step 7), T1
(step 8), T2 (step 9), O1 (step 10). **Machine:** Apple M2 Pro, 10 cores, 16 GB, macOS 24.6.0,
rustc 1.97.0, `--release`.

**Method.** Nothing below is quoted from the five notes. Every gate was re-run on this tree, every
table was recomputed from its own inputs, and the two claims that a note could only assert — the
byte identity of the off switches and the AUC table the veto decision rests on — were re-derived
from a fresh build of `c6e38bc` outside the repository and from a fresh `segment --features` pass
respectively. Where a number below agrees with a note, it agrees because it was measured again.

**Verdict: the work is sound and two of the brief's own gate rows do not hold.** The confirmed tier
is 0 false joins over 8 sets × 5 seeds, reproduced; the off switches are byte-exact; the parity
total is the frozen 23 804 / 0; `gpu-check` is 8 of 8. What fails is audit §E's *step-10* row
"correct confirmed joins ≥ 12 on `mixed_ABG`" (measured 11/7/7/7/9) and its *step-8* row
"terracotta's two joins confirmed" (nine of ten (seed, join) slots; seed 4 leaves 021–094
probable). Both were found by the executing tasks themselves and restated in their notes; neither
restatement is checkable by a gate that can fail, which is defect **V8-D2**. Six defects in all,
one of substance (**V8-D1**), three of documentation, two of gate coverage.

---

## 1. The tier: is it the evidence the audit names, at the thresholds M1's table chose?

### 1.1 The thresholds, value for value

`Thresholds::default` (`crates/sherd-core/src/tiers.rs:299-315`) against M1 §3's chosen row
(`notes/2026-09-09-m1-measure.md:262-275`) and against audit §D.1's own candidate list
(`notes/2026-09-09-fable-audit.md:428-430`):

| test | audit §D.1 proposes | M1 §3 chose | `tiers.rs` ships | CLI flag |
|---|---|---|---|---|
| `tight` | `min_tight 0.35` | 0.35 | `min_tight: 0.35` | `--tier-min-tight` |
| `gap` | `gap 0.02 t` | **0.015 t** | `max_gap_t: 0.015` | `--tier-max-gap` |
| `seam` | `seam 5 t` | 5 t | `min_seam: 5.0` | `--tier-min-seam` |
| `cont_n` | `cont_n 0.9` | 0.90 | `min_cont_n: 0.90` | `--tier-min-cont-n` |
| `pen` | `pen 0` | 0 | `max_pen: 0.0` | `--tier-max-pen` |
| slide | "converge back within 0.1 t" | 0.1 t | `max_slide_t: SLIDE_BACK_T = 0.1` | `--tier-max-slide` |
| margin | `m = 2` | 2, read **strictly** | `min_margin: 2.0`, `None` rival fails | `--tier-margin` |
| support | (audit puts it in step 10, "two consistent paths") | **1**, as a disjunct with the margin | `min_support: 1` | `--tier-support` |
| determinedness | in the conjunction | **dropped**, reported | `max_determined_deg: None` | `--tier-max-determined-deg` |
| stability (3 draws) | in the conjunction | **dropped**, reported | `min_resample_accept: None` | `--tier-resample-accept` |

Every value the code ships is the value M1's table chose, and every one is a flag. The comparisons
are exact (`verify::under`/`over`, `verify.rs:646-655`, `partial_cmp` with NaN failing) — no
epsilon widens a threshold behind the number.

### 1.2 The four kinds of evidence audit §D.1 names are all computed

`tiers::probe` (`tiers.rs:544-600`) computes all four and the fifth M1 added:

* **margin** — `rival` (`tiers.rs:690-707`), the best candidate of the pair placing B more than
  `SAME_PLACEMENT_T = 1.0` wall away, measured as the parity harness measures it (worst
  displacement over every twentieth vertex of B's working mesh, `placement_gap`, `tiers.rs:737-752`);
* **stability under resampling** — `resampled_scores` (`tiers.rs:927-955`) re-runs R §6 at the same
  pose on two redraws at `seed + 1e6` and `seed + 2e6` (`RESAMPLE_OFFSETS`, `tiers.rs:107`);
* **determinedness** — `determinedness` (`tiers.rs:758-…`) over `rungs[2..]`, which is R §5.6's
  **last two** rungs (`SurfaceLadder::rungs` returns four, `matching/pair.rs:429`) — exactly the
  audit's wording;
* **slide** — `slide_probe` (`tiers.rs:811-845`) pushes ±0.5 t along the seam's principal axis and
  re-climbs **the whole** four-rung ladder, not its last rung; the note argues that the last rung
  alone cannot answer (its radius is 0.04 t against a 0.5 t push) and the code matches the argument;
* **support** — `support_count` (`tiers.rs:969-1006`), `T_ax · T_xb` against R §8's own tolerances
  (`consistency::agrees`, 10°, 0.5 t, `assembly/consistency.rs:20-22,80-82`), read at B's fracture
  cloud rather than at the file origin.

**Three structural departures from audit §D.1, all argued from M1's table, all switchable:**
determinedness and the three-draw stability are *reported and not gated* (M1: the whole
determinedness range over 808 candidates is 1.6e-14…4.1e-7 degrees, so there is no threshold in it;
requiring three draws costs 136 → 130 confirmed and removes no false join); the support count is
moved from step 10 into step 8 (M1: no conjunction of the other quantities reaches zero false joins
over 4 478 976 searched); and the support arm fires at 1 rather than the audit's two paths. Each
departure is a flag away from the audit's own wording (`--tier-max-determined-deg`,
`--tier-resample-accept`, `--tier-support 2`). This is the audit's own methodology — *"thresholds
are chosen on the measurement table, not guessed"* — applied to the shape of the rule and not only
to its numbers, and the notes say so. I record it as a departure, not as a defect.

### 1.3 The measurement and the shipped rule are the same code

`measure.rs` calls `tiers::probe` (`measure.rs:60`, `use crate::tiers::{self, Probes}`), so the
probes M1's table was computed with and the probes T1 ships are one implementation. That is the
right structure: a threshold cannot come to mean something else than it was chosen on.

### 1.4 No ground truth and no object id reaches a decision

`grep -rn --include='*.rs' 'ground_truth|object_of|evaluate' crates/` over the whole workspace:

| term | hits in `crates/` | what they are |
|---|---|---|
| `ground_truth` | 9 | 8 doc comments; 1 real read, `crates/sherd-core/tests/slab_pair.rs:35` — a **test** reading the committed slab's `ground_truth.json` |
| `object_of` | **0** | — |
| `evaluate` | 19 | all prose ("the two executors evaluate `d² < r²`", "`tools/evaluate.py` scores a Rust run"); no call, no path |

No pipeline path reads a `ground_truth.json`, an adjacency list or an object id
(`fragment/cache.rs`, `io/*`, `collection.rs` and `assembly/constraints.rs:152` are the only file
reads in `sherd-core`, and the last is the operator's own `constraints.json`). `same_object` /
`different_object` are operator input, not truth. `tools/quality_gate.py:614-618` builds the run
command from the set directory, the backend, the seed and the three switches and passes no
constraint file and no ground truth; the truth is opened only by `score()` afterwards
(`quality_gate.py:339-343`). No fragment name appears in a decision: the only `FY234…` and
`Pot_?_Piece_…` strings under `crates/*/src` are in doc comments and in `constraints.rs`'s JSON
example.

**Object ids are read by `evaluate.py` (and by `quality_gate.py`/`measure_tiers.py`, which import
it) and by nothing else.** Validation is leak-free.

### 1.5 What the object pass is allowed to do, against audit §D.2's own rule

Audit §D.2: *"a feature may veto only where its measured AUC exceeds 0.8, otherwise it reports."*
`ObjectParams::demote` ships **empty** (`objects.rs`, `--object-demote` default empty,
`crates/sherd-cli/src/main.rs:406`+), so no feature vetoes. §6 below reproduces the AUC table that
decision rests on: the best AUC on any collection with real object ids is **0.740**. The rule is
honoured.

Two further arms: `--object-merge` ships **on** (audit §D.2 (c)) and `--object-disagreement` ships
**off** — a departure from §D.2 (b)'s wording, measured (136 → 76 correct confirmed joins, 0 false
joins removed, because there were none to remove). Both are flags; both are argued in O1 §2.2.

---

## 2. The standing gates, re-run on this tree

| gate | command | result |
|---|---|---|
| `cargo test --workspace`, **debug** | `cargo test --workspace --locked` | **pass**, exit 0 |
| `cargo test --workspace`, **release** | `cargo test --release --workspace --locked` | **pass**, exit 0 |
| test inventory | `cargo test --workspace --locked -- --list` | **444** test functions, **3** `#[ignore]`d → **441** run per profile; O1's "441 passed, 3 ignored" confirmed |
| no-default-features build | `cargo build -p sherd-cli --no-default-features --locked` | **pass**, exit 0 |
| no-default-features clippy | `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass**, exit 0 |
| clippy | `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass**, exit 0 |
| fmt | `cargo fmt --all --check` | **pass**, exit 0 |
| `parity --stage all`, both modes, eight dumps | 16 runs | **pass** — all exit 0, **256 rows** (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed** |
| `pytest -q` | | **pass** — **60 passed** in 77.8 s |
| the three `#[ignore]`d tests | `cargo test --release --workspace -- --ignored` | **pass** — `the_terracotta_assembles_the_two_joins_of_r_13`, `the_terracotta_honours_its_two_constraints`, `two_runs_on_the_terracotta_produce_byte_identical_caches` |

The parity total is the frozen number **to the check**, and so is every dump:

| dump | native | injected | | dump | native | injected |
|---|---:|---:|---|---|---:|---:|
| terracotta | 145 | 631 | | pot_H | 459 | 3 650 |
| pot_A | 306 | 2 042 | | `synthetic_20` | 1 040 | 8 603 |
| pot_B | 354 | 2 527 | | slab | 81 | 248 |
| pot_C | 261 | 1 612 | | | | |
| pot_G | 248 | 1 597 | | **total** | | **23 804 / 0** |

Every one of the sixteen equals task F's own per-dump row, which is what "a frozen stage's
arithmetic did not move" means. The only parity-crate changes since `c6e38bc` are signature
adaptations (`None` for the four new optional fields of `Outcome`, and
`Tier::of_accept(c.accepted)` on the reference's own candidate list) plus one extracted helper —
no arithmetic (`git diff c6e38bc..HEAD -- crates/sherd-parity/`).

### The byte-identity check, against a fresh build of `c6e38bc` **outside** the repository

`git archive c6e38bc | tar -x` into the scratchpad, built with its **own** `CARGO_TARGET_DIR` (so
the repository's binary and its commit stamp were never disturbed — task F's §9 caution does not
apply here). The repository's branch was never switched. Both binaries then ran each set into its
own fresh directory, CPU, seed 0, previews and meshes **on**:

* old: `run <set> --out … --backend cpu --seed 0`
* new: `run <set> --out … --backend cpu --seed 0 --tiers off --objects off` (no `--constraints`,
  no `--review-images`, no `--measure`)

| set | files | byte-identical | exempt | **differing** |
|---|---:|---:|---:|---:|
| terracotta | 10 | 8 | 2 | **0** |
| pot_A | 14 | 11 | 3 | **0** |
| pot_H | 19 | 16 | 3 | **0** |
| `synthetic_20` | 28 | 25 | 3 | **0** |
| **total** | **71** | **60** | **11** | **0** |

Every PLY and every PNG of every set is byte-identical. The eleven exempt files were compared as
values, key by key, and the only keys that differ are:

| file | differing keys | why it is exempt |
|---|---|---|
| `transforms.json` (4) | `engine.commit`, `engine.cache_version` | two commits by construction; **5 → 6** is audit §D.2's own instruction |
| `report.json` (4) | the same two, plus `timings` and `memory` | the standing gate excludes both |
| `report.md` (3) | the `## Timing` section only | the standing gate excludes it |

No key was added, removed or reordered in any of the eleven. `report.md` of the terracotta is
identical **including** its timing section, which is why its row reads 8 + 2 rather than 7 + 3.

---

## 3. The quality gate, run here

`python tools/quality_gate.py --out output/v8/quality`, shipped defaults (`--tiers on --objects on
--object-disagreement off`), backend `cpu`, seeds 0–4, eight sets: **exit 0**, **615.0 s (10.2 min)**
for the forty runs, against the 25-minute budget. Header stamp `2026-09-10 03:39`, commit
`59be4054…`.

**136 correct confirmed joins, 0 false, over the eight development sets at seeds 0–4** — the same
136 and the same 0 T1 and O1 report, set for set and seed for seed. Every scored column of my forty
rows equals O1's table; only `wall s` moves.

| set | seed | frag acc | prec | correct | wrong pose | non-adj | cross-obj | purity | joins | confirmed | conf correct | conf false | conf recall | probable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `terracotta` | 0–3 | — | — | — | — | — | — | — | 2 | 2 | 2 | **0** | 1.000 | 0 |
| `terracotta` | 4 | — | — | — | — | — | — | — | 1 | 1 | 1 | **0** | 0.500 | 1 |
| `pot_A` | 0–4 | 37.5–62.5 % | 1.000 | 2–4 | 0 | 0 | 0 | 1.000 | 2–4 | 3–6 | 3–6 | **0** | 0.200–0.400 | 8–14 |
| `pot_B` | 0–4 | 33.3–77.8 % | 1.000 | 2–5 | 0 | 0 | 0 | 1.000 | 2–5 | 3–7 | 3–7 | **0** | 0.200–0.467 | 17–22 |
| `pot_C` | 0–4 | 0–50.0 % | 0/1.000 | 0–1 | 0 | 0 | 0 | 1.000/– | 0–1 | 0–1 | 0–1 | **0** | 0.000–0.200 | 2–3 |
| `pot_G` | 0–4 | 0.0 % | 0.000 | 0 | 0 | 0 | 0 | – | 0 | **0** | 0 | **0** | 0.000 | 1–2 |
| `pot_H` | 0–4 | 0–18.2 % | 0/1.000 | 0–1 | 0 | 0 | 0 | 1.000/– | 0–1 | 0–1 | 0–1 | **0** | 0.000–0.059 | 11–20 |
| `synthetic_20` | 0–4 | 35.0–60.0 % | 1.000 | 5–10 | 0 | 0 | 0 | 1.000 | 5–10 | 5–11 | 5–11 | **0** | 0.100–0.220 | 14–18 |
| `mixed_ABG` | 0–4 | 29.2–50.0 % | 1.000 | 5–8 | 0 | 0 | **0** | **1.000** | 5–8 | 7–11 | 7–11 | **0** | 0.175–0.275 | 65–76 |

Totals over the forty runs: **conf correct 136, conf wrong_pose 0, conf non_adjacent 0,
conf cross_object 0**. `merges` is 0 in all forty; `demoted` is 0 in all forty; the consensus reports
0–9 members "outside" per run and acts on none.

### The five questions the brief asks of this table

1. **Zero false joins in the Confirmed tier on 8 sets × 5 seeds — PASS.** 0 wrong-pose, 0
   non-adjacent, 0 cross-object, in every one of the forty runs, scored on each confirmed
   candidate's *own* pose (`quality_gate.py:194-250`) rather than on the assembly, so a confirmed
   join R §8 refused is still scored.
2. **Confirmed recall per set** — terracotta 0.500–1.000, pot_A 0.200–0.400, pot_B 0.200–0.467,
   pot_C 0.000–0.200, pot_G 0.000, pot_H 0.000–0.059, `synthetic_20` 0.100–0.220, `mixed_ABG`
   0.175–0.275. Over all forty: **136 of the 770 ground-truth adjacent pairs, 17.7 %**, against
   audit §D.1's own expectation of 60–80 %. The recall loss is the product and the notes say so;
   it is not a defect, but it is the number a museum will ask about first.
3. **pot_B's non-adjacent join.** Measured directly (five runs, `output/v8/potB`, every accepted
   candidate classified against `ground_truth.json`'s adjacency): pot_B has **9–13 accepted
   candidates on non-adjacent pairs at every seed, 0–4**, and **every one of them is `probable`**.
   None is confirmed at any seed. At **seeds 1 and 3** the non-adjacent accepted pairs are
   01-06, 01-07, 02-05, 02-07, 02-09, 03-04, 03-06, 03-08, 07-08 (seed 1) and 01-06, 01-07, 02-03,
   02-05, 02-06, 02-07, 02-09, 03-04, 03-06, 03-08, 04-09, 07-08 (seed 3) — all **probable**,
   `tight` 0.25–0.47 against the tier's 0.35 and `gap` 0.024–0.049 t against its 0.015 t. The
   audit's "seeds 1 and 3" is about the seeds at which R §8 *used* such a join under R §6.5; on
   this tree that is seeds 2 and 4 (M1 §2 and my own `--tiers off` sweep, §7). Either way the
   answer to the question asked is the same: **probable, at every seed**.
4. **pot_G's near-misses.** All accepted candidates of pot_G's five runs are wrong-pose joins on
   adjacent pairs (R §13's own row). Their band: **probable**, 1–2 per seed (02-05 at every seed,
   plus 04-07 at seed 0 and 03-06 at seed 4). **0 confirmed at every seed**, and pot_G's assembly
   is five singletons — which is the right answer for a set whose every candidate is 0.7°–2.2° and
   0.86–1.07 t from the truth.
5. **The terracotta.** Confirmed at seeds 0, 1, 2 and 3: both of R §13's joins (021–094 and
   094–104). **At seed 4: 094–104 confirmed, 021–094 probable.** Nine of the ten (seed, join) slots.
   Nothing outside R §13's two is ever confirmed, and both joins are at least probable at every
   seed. **The brief's row "terracotta Confirmed at every seed" therefore does not hold** — see
   **V8-D3**.
6. **`mixed_ABG`.** cross-object **0** at every seed ✔; group purity **1.000** at every seed ✔;
   correct confirmed joins **11 / 7 / 7 / 7 / 9** — **below 12 at every seed** ✘. See **V8-D2**.

---

## 4. The GPU

`gpu-check --stage all --pairs 4` at the production policy, idle adapter, one at a time, on the
eight sets the brief names:

| set | exit | rows | failed | `icp s1 deg` worst / tol |
|---|---|---:|---:|---|
| terracotta | **0** | 30 | 0 | delegated (every icp call the CPU's under the policy) |
| pot_A | **0** | 30 | 0 | 1.586e-2 / 5.0e-2 |
| pot_B | **0** | 30 | 0 | delegated |
| pot_C | **0** | 30 | 0 | **1.312e-4** / 5.0e-2 (task F's excusal holding) |
| pot_G | **0** | 30 | 0 | delegated |
| pot_H | **0** | 30 | 0 | delegated |
| `synthetic_20` | **0** | 30 | 0 | 3.268e-4 / 5.0e-2 |
| slab | **0** | 30 | 0 | delegated |
| | **8 of 8** | | **0** | |

**Corrupt readbacks.** `run input/synthetic_pingsdorf_20/fragments --backend gpu --seed 0` on an
idle adapter, exit 0:

```
coarse:   380 calls, 151 on device, 229 delegated (0 host errors, 0 corrupt readbacks)
icp:     1126 calls, 134 on device, 992 delegated (0 host errors, 0 corrupt readbacks)
distance:1572 calls,   0 on device (D §12's 2c, struck)
inside:  1552 calls,   0 on device
```

**0 corrupt readbacks** over 285 device calls and 285 dispatches. The nonce-and-validity path of
task F's A.1.6 is live (`crates/sherd-gpu/src/icp.rs:605-613` documents what each check catches) and
refused nothing on a quiet machine, which is what it should do.

---

## 5. Constraints and review images

### The four scenarios audit §E states for step 9

| scenario | how it was checked here | result |
|---|---|---|
| **images deterministic between two runs** | two full terracotta runs with `--review-images`, seed 0, into two fresh directories; both PNGs compared with `cmp` | **pass** — `FY234021__FY234094.png` and `FY234094__FY234104.png` byte-identical between the runs. (The suite's own check is on the slab; this is the terracotta) |
| **terracotta `must_not_join [021, 094]` yields 094–104 only and reports it** | `the_terracotta_honours_its_two_constraints`, run explicitly (`-- --ignored`) | **pass** — `joins_used` is exactly `[094, 104]`, no candidate for 021–094 exists at all (the pair never matched), and `constraints.entries[0]` reads `must_not_join`, `satisfied: true` |
| **`must_join [007, 021]` reports unsatisfiable and the run still succeeds** | the same test | **pass** — `satisfied: false`, outcome contains "unsatisfiable", exit 0 |
| **a pinned pose places without matching** | `a_pinned_pose_places_without_matching` (slab), in both profiles | **pass** — one candidate for the pair instead of R §5.7's five, band `confirmed`, at the pinned pose |

Three more, all green in both profiles: `must_not_join_removes_the_pair_before_matching_and_the_report_lists_it`
(a candidate list of length zero, not merely an empty assembly),
`different_object_vetoes_a_join_the_geometry_accepted`, and
`an_unknown_name_in_the_constraints_file_fails_the_run`.

**An unknown name is an error.** Verified in code as well as in the test: `constraints::load` →
`Resolved` resolves every name immediately after preprocessing and fails the run
(`assembly/constraints.rs:36-42` states the rule; the validation also refuses a pair naming one
fragment twice, a pair in two of the four lists, a `version` other than 1, and a `pose` that is not
a rigid transform).

**A constraint changes no score.** `assemble_under`'s only two entry points are the veto at the top
of the loop (`greedy.rs:536-540`, before any test of R §8's own) and the `forced` sort key
(`greedy.rs:237-248`), which is **constant when there are no constraints**, so the stable sort
leaves R §8's tie rules exactly where they were — which is what the byte identity of §2 measures.
The one place a constraint changes a *verdict* rather than a filter is a pinned pose:
`pinned_candidate` (`pipeline.rs:957-989`) sets `accepted: true` and `Tier::Confirmed` by fiat while
carrying R §6's honest scores at that pose. That is audit §D.1's own semantics ("the pose is a
confirmed candidate") and the report says a person pinned it.

### The report a conservator reads

The terracotta run with `--review-images` writes, in order: `## Fragments`, `## Assembly`,
`## Objects`, `## Joins used`, `## Confirmed joins`, `## Probable joins`, `## Rejected`,
`## Candidates by fragment`, `## Best candidate per pair`, `## Timing` — the per-fragment index
carries the `image` column and four rows linking the two PNGs, one under each of the pair's two
fragments, exactly as audit §C.8(a) asks.

---

## 6. The object features: M1's AUC table reproduced on two collections

Re-derived end to end, not read: `segment --features FILE --features-colour` was re-run on pots A–H
and on `mixed_ABG` with this tree's binary (preprocessing only — `segment` runs R §3 and stops), and
the resulting tables were compared with M1's stored ones and then fed to an **independently written**
AUC script (`P(|Δf| of a different-object pair > |Δf| of a same-object pair)`, ties at ½; MAD as the
median absolute deviation from the median).

| collection | fragments | my `segment --features` table vs `output/measure/features/` |
|---|---:|---|
| pot_A…pot_H | 8, 9, 7, 32, 34, 7, 7, 11 | **JSON-equal, all eight** |
| `mixed_ABG` | 24 | **JSON-equal** |

**`mixed_ABG` (3 objects, real ids):**

| feature | AUC (mine) | AUC (M1 §4) | MAD | 2·MAD pairs/adj | 3·MAD pairs/adj |
|---|---:|---:|---:|---|---|
| `thick` | **0.620** | 0.620 | 0.3152 | 0.478 / 0.425 | 0.351 / 0.375 |
| `thick_mode` | **0.641** | 0.641 | 0.2859 | 0.467 / 0.425 | 0.359 / 0.400 |
| `shell_radius` | **0.740** | 0.740 | 15.99 | 0.435 / 0.125 | 0.159 / 0.100 |
| `frac_rough` | **0.632** | 0.632 | 0.005852 | 0.312 / 0.175 | 0.134 / 0.075 |
| `axis_diameter` | **0.528** | 0.528 | 17.36 | 0.522 / 0.450 | 0.362 / 0.300 |
| `axis_residual` | **0.524** | 0.524 | 0.0107 | 0.435 / 0.375 | 0.250 / 0.275 |
| `shell_rms` | **0.642** | 0.642 | 0.04442 | 0.344 / 0.250 | 0.156 / 0.100 |
| `axis_rms` | **0.546** | 0.546 | 0.1976 | 0.370 / 0.375 | 0.192 / 0.125 |

**`pots_A_H` (115 fragments pooled, 8 objects):**

| feature | AUC (mine) | AUC (M1 §4) | MAD |
|---|---:|---:|---:|
| `thick` | **0.694** | 0.694 | 1.687 |
| `thick_mode` | **0.695** | 0.695 | 1.525 |
| `shell_radius` | **0.552** | 0.552 | 33.36 |
| `frac_rough` | **0.509** | 0.509 | 0.007308 |
| `axis_diameter` | **0.538** | 0.538 | 22.96 |
| `axis_residual` | **0.521** | 0.521 | 0.01325 |
| `shell_rms` | **0.469** | 0.469 | 0.135 |
| `axis_rms` | **0.571** | 0.571 | 0.1713 |
| `lab_L` / `lab_a` / `lab_b` / `lab_spread_L` | **0.489 / 0.500 / 0.500 / 0.494** | same | 7.5e-12 / 0 / 0 / 0 |

Every cell reproduces, including the 2·MAD and 3·MAD veto costs. The best AUC on a collection with
real object ids is **0.740**, below audit §D.2's own **0.800**, so **no feature earns a veto** and
`--object-demote` is rightly empty. The colour rows confirm the audit's §C.4 prediction: the SfS++
files carry one flat grey (`lab_a`/`lab_b` MAD exactly 0), so colour has nothing to be calibrated on
and a `k·MAD` rule on it would be a division by zero — which `Consensus::mads`
(`objects.rs:318-320`) refuses rather than answering infinity.

### Collection sizes: was anything above 27 fragments matched?

No. The largest collection **matched** by any of the five tasks is `mixed_ABG`, and it holds **24**
fragments (`ls input/sfspp/mixed_ABG` = 24 meshes), inside the gate and inside M1's forty
measurement runs. The only pass that saw a larger collection is M1's `segment --features` on
`input/synthetic_pingsdorf_170` (164) and `input/sfspp/mixed_all` (164) — preprocessing only, which
is the audit's one stated exception. M1 also ran `segment --features` on `pot_D` (32) and `pot_E`
(34); that is above 27 but it is *not matching*, and the brief's prohibition is on matching. My own
runs matched nothing larger than `mixed_ABG` either.

*(The brief and T2's note §7 both call `mixed_ABG` a 27-fragment set; it is 24. See **V8-D6**.)*

---

## 7. `pytest` and the README

`pytest -q`: **60 passed**, exit 0. The frozen Python was not touched by any commit in the range
(`git diff --name-only c6e38bc..HEAD` has no `sherd_refit/` entry), which is the reference-flip rule
holding.

**The README's museum section describes what the binary does.** `README.md:339` — *"Что программа
говорит музею: три полосы уверенности, картинки и файл ограничений"* — and `README.md:420` —
*"Что программа говорит об объектах"*. Checked claim by claim against `run --help` and against the
measured behaviour: `--tiers on|off` default on ✔; the three bands and "the assembly is built from
confirmed alone" ✔; the five probes named correctly ✔; the eight tier flags listed are the eight
that exist ✔; "136 стыков и ни одного ложного, 17.7 %" is the number I measured ✔;
`--review-images` writes `review/<A>__<B>.png` for confirmed **and** probable joins, three views,
A grey / B orange, seam white, contact green-yellow-red, deterministic ✔; the constraints table's
five rows are the five semantics the code implements, and "неизвестное имя — ошибка" ✔;
`--objects on|off` default on, `--object-demote` empty, `--object-k-mad 3`, `--object-min-members
3`, `--object-merge on`, `--object-disagreement off` — all six match the CLI defaults ✔; the AUC
argument (0.740 against 0.800) is the number §6 reproduces ✔; "`--objects off` возвращает поведение
до пункта 4, файл в файл и байт в байт" is what §2's identity check measures ✔.

One thing the museum section does **not** say and, on this evidence, should: that the confirmed
tier finds **17.7 % of the true joins** — it gives the figure, but beside "136 стыков и ни одного
ложного" rather than as the headline a conservator needs. That is an editorial remark, not a defect.

---

## 8. The off switch, over forty runs and not only four files

§2's byte-identity check is four sets at seed 0. The stronger statement is what
`tools/quality_gate.py --tiers off --objects off` does to the whole sweep, and it was run here:
**412.2 s (6.9 min), exit 0**, header `2026-09-10 04:09`, commit `59be4054…`.

Its forty rows were compared **cell by cell** against M1 §6's own `--tiers off` table (which is task
F7's, which is V7's, which is the behaviour before roadmap item 3 existed) — fragment accuracy,
precision, the four join buckets, group purity and the join count:

**35 scored rows compared, 0 differ**, and the terracotta's decision row passes at all five seeds
with 2 joins used at each. `mixed_ABG` comes back to its documented baseline: cross-object **1–4**,
purity **0.750–0.952**, correct used joins **10–15** — which is the state item 4's gate row was
written against, and which the confirmed tier replaces with 0 and 1.000.

So the two switches together restore the previous algorithm exactly, on every set and every seed,
not merely on the four files the standing gate names. The band verdicts also come back
(`pot_A`, `pot_B`, `pot_H` outside R §13's reference band; `pot_C`, `pot_G`, `synthetic_20` inside),
which is a further sign that the same arithmetic is running.

---

## 9. Defects

Six, in the brief's own form: **file:line — what the audit or D says — what the code does.**

### V8-D1 — the confirmed tier's `pen ≤ 0` is not a test where R §6.4 could not run *(substance, medium)*

**`crates/sherd-core/src/tiers.rs:325-345` (`Thresholds::refusals`), against
`crates/sherd-core/src/tiers.rs:320-323` and `crates/sherd-core/src/matching/verify.rs:490-493`.**

*The module says:* "A test whose evidence is **missing** fails: a candidate with no seam to slide
along has not passed the slide probe, it has refused it. That is the same reading as the margin's,
and M1's threshold table was computed under it" (`tiers.rs:320-323`). Audit §D.1 puts `pen 0` in the
conjunction, and T1 tightened it from R §6.5's 0.005 to **0**.

*The code does:* `refusals` reads `scores.pen` alone. `penetration_scores` returns `(0.0, 0.0,
true)` when either fragment is not watertight (`verify.rs:490-493`) — the question was refused, not
answered — and the third element is carried as `ScoreRow::pen_unavailable`, whose own doc says in as
many words that "a `pen ≤ 0` veto passes it for free" (`tiers.rs:138-141`). Nothing reads that flag:
`grep -n 'pen_unavailable' crates/sherd-core/src/tiers.rs` finds the field and its comment and no
use. So the tier's strictest penetration clause is **vacuous exactly on the pairs where penetration
cannot be checked**, and the "missing evidence fails" rule the module states for the slide and the
margin is not applied to it.

*Measured, not argued.* `pot_B`'s fragments **02, 07 and 08 are not watertight** (a fresh run's
`report.json`). At **seed 0**, three of pot_B's seven confirmed joins — **01–02, 01–08, 02–08** —
carry `pen_unavailable: true`; the other four do not. And the worst false join the module itself
names, `Pot_A_Piece_08–Pot_B_Piece_07` on `mixed_ABG` seed 1 (`tiers.rs:50-52`), is on one of the
same three fragments: what refused it was the gap, and the penetration clause could not have.

*Severity.* No false join resulted in the forty runs, so the measured claim ("0 false") stands. What
does not stand is the *reason* it stands: on a pair with a non-watertight fragment the confirmed
tier is a four-test conjunction plus an arm, not a five-test one. The hole is inherited from
R §6.5's `accept` (`verify.rs:632-638` has the same shape), so T1 did not create it — but T1 is
where `max_pen` became `0` and was presented as a test. **The fix is one line** (`if
scores.pen_unavailable { failed.push("pen: not watertight, R §6.4 could not run") }`), and it should
be made with a gate run beside it, because it will cost confirmed joins on pot_B and `mixed_ABG`.

### V8-D2 — audit §E's step-10 row "correct confirmed joins ≥ 12" fails, and no gate can say so *(gate coverage)*

**`tools/quality_gate.py:147-148` and `:355-393`, against
`docs/superpowers/notes/2026-09-09-fable-audit.md:571` (§E, step 10).**

*The audit says:* "`mixed_ABG`: cross-object joins 0 in the confirmed tier, purity 1.000, correct
joins ≥ 12 (the baseline's) at seeds 0–4."

*Measured here:* cross-object **0** ✔, purity **1.000** ✔, correct confirmed **11 / 7 / 7 / 7 / 9**
✘ — below 12 at every seed, and the best seed is 11.

*What the gate does:* `BANDS["mixed_ABG"]` has `pure=False` with the note "roadmap item 4's baseline
(D §10.3), not a gate", so neither the purity row nor the cross-object row is gated for that set
(they are covered indirectly, because `tier_verdict` counts `conf_cross_object` among the false
joins on all eight sets); and **nothing anywhere counts the confirmed-correct joins against 12**.
O1 §4 records the failure honestly and restates the row as "no fewer than the tier already had".
That restatement is reasonable — the 12 is a count of *used* joins under R §6.5 and object
separation adds no confirmed join by construction — but it lives in prose. **A restated gate that
no script can fail is not a gate.** Either `quality_gate.py` should carry the restated row
(confirmed-correct ≥ the number recorded, per set) or audit §E's table should be amended; on this
tree the brief's row is unmet.

### V8-D3 — audit §E's step-8 row "terracotta's two joins confirmed" fails at seed 4 *(gate coverage)*

**`tools/quality_gate.py:163` (`TERRACOTTA_CONFIRMED_SLOTS = 9`), against
`docs/superpowers/notes/2026-09-09-fable-audit.md:569` (§E, step 8).**

*The audit says:* "terracotta's two joins confirmed" — with the gate stated over seeds 0–4.

*Measured here:* seeds 0–3 confirm both; **seed 4 confirms 094–104 and leaves 021–094 probable**.
Nine of ten (seed, join) slots.

*What the gate does:* it asks for nine, a constant chosen from M1 §5.6's measurement of exactly this
tree. The three things it does check are all true and all worth checking (nothing outside R §13's
two is ever confirmed; both are at least probable at every seed; nine of ten slots). But the
constant was fitted to the observed answer, so the gate cannot notice a regression from nine to
nine-with-a-different-seed, and it records the brief's row as met when the brief's row is not.
T1 §7.6 names the cause correctly — the pair's search returns a single placement at seed 4, so the
margin arm has nothing to beat, and a three-fragment chain gives the support arm nothing — and says
the fix is R §5.7's `keep`, an algorithm change nobody has made. That analysis is right; the gate
wording is what is wrong.

### V8-D4 — a doc comment was orphaned onto the wrong function *(documentation)*

**`crates/sherd-parity/src/stages/outputs.rs:356-371`.** The twelve-line doc comment that documents
`markdown_row` — "R §11.3's `report.md`, rendered by the port from the reference's own
`report.json` … The `## Timing` block is where the comparison stops" — now sits immediately above
the newly inserted `fn group_ids` (`:371`), and `fn markdown_row` (`:383`) has no doc comment at
all. Commit `bfce0bd`/`13b27af` inserted the helper between the comment and the function it
describes. Cosmetic; the arithmetic is untouched (§2's parity total proves it).

### V8-D5 — the unreachability test's doc says six fragments, the test uses eight *(documentation)*

**`crates/sherd-core/tests/assembly_graphs.rs:211` against `:217`.** The doc reads "four thousand
random graphs of **six** fragments"; the code is `const N: u32 = 8`, and O1 §3 says eight. The
sample is otherwise as described and it passed here.

*A related over-claim worth a sentence:* O1 §3's heading is "It never fires, and the reason is
R §8's own rule", and the note then says the branch "cannot" be reached. The prose argument
(§3's paragraph, and `assembly_graphs.rs:200-209`) is sound and I re-read the greedy loop against
it — a second group is seeded only after a full scan in which nothing was placed
(`greedy.rs:601-616`), so at that moment every surviving join has both fragments free — but four
thousand random graphs are a sample, not a proof, and the note should say "argued, and sampled at
four thousand graphs" rather than "cannot".

### V8-D6 — `mixed_ABG` is called a 27-fragment set *(documentation)*

**`docs/superpowers/notes/2026-09-09-t2-review-constraints.md:296-297`:** "Nothing above 24
fragments was matched by this task's own runs; the quality gate's `mixed_ABG` (**27**) is the
largest collection" — the same sentence contradicts itself, and the set holds **24** meshes
(`ls input/sfspp/mixed_ABG`). M1 §4 and O1 §7 both say 24. Harmless, but the number is the one the
small-set rule is read against.

### Not defects, recorded so the next reader does not re-find them

* **The tier's structure departs from audit §D.1's conjunction** (determinedness and the three-draw
  stability report instead of gating; the support count moves from step 10 into step 8 and fires at
  1 rather than 2). Every departure is measured in M1 §2–§3, every one is a flag, and the audit's
  own rule is that the table decides. §1.2 above.
* **`--object-disagreement` ships off**, against §D.2 (b)'s wording. Measured: 136 → 76 correct
  confirmed joins and no false join removed. A flag.
* **R itself was not updated.** No commit in the range touches
  `specs/2026-09-06-algorithm-reference.md`, so the confidence tier, `constraints.json`, the review
  images and the object features exist in the code and in D §9/§11/§12 but not in R — while R §13
  is still what `quality_gate.py`'s prohibitions are read from. That is consistent with task H4's
  decision ("`sherd_refit/` is frozen as the parity oracle **for the stages R already describes**"),
  but a reader of R alone would not learn that R §8 no longer assembles from `accepted`. Worth one
  paragraph in R saying which of its sections the code has moved past.
* **`segment --features` ran on `pot_D` (32) and `pot_E` (34)** in task M1, above the 27-fragment
  line. It is preprocessing only and matched nothing; the rule is on matching. Recorded because the
  brief's stated exception names only `synthetic_170` and `mixed_all`.

---

## 10. Housekeeping

This task wrote `output/v8/` only — the two quality tables and their per-run `evaluate.py` dumps
(`quality/`, `quality_off/`), the eight `gpu-check` reports and the GPU run's log (`gpucheck/`), the
nine `segment --features` tables (`features/`), the five pot_B runs (`potB/`) and the two terracotta
review runs with their four PNGs (`review1/`, `review2/`) — 3.6 MB after the caches and the
`segment` work trees were removed — and this note. The byte-identity trees (1.5 GB of placed
meshes and previews), the `--backend gpu` run tree and the two gates' work trees were compared,
measured and deleted; `target/debug/incremental` (2.3 GB) was dropped before the first long run to
make room. The `c6e38bc` build lives **outside** the repository, under the session scratchpad, with
its own `CARGO_TARGET_DIR`; the repository's branch was never switched and its own binary's commit
stamp was never disturbed. Free disk stayed above **6.3 GiB** throughout. Nothing under `input/`,
`fixtures/` or `output/fixtures/` was written or deleted, and `sherd_refit/` was not touched.

Nothing above 24 fragments was matched: `mixed_ABG` inside the two gate runs is the largest
collection this task ran, twice over. `segment --features` saw pots A–H (largest 34) and
`mixed_ABG`, preprocessing only.

Only this note is committed.

