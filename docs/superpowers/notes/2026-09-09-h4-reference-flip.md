# H4 — the reference flips to Rust, and the first five-seed sweep of the port

**Date:** 2026-09-09. **Tree:** branch `rust-core`, `30a0e4e` (H3) → `H4.1…H4.4`. **Machine:**
Apple M2 Pro, 10 cores (6P + 4E), 16-core GPU, 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`.
Nothing above 24 fragments was matched. Task H4 executes step 6 of the audit's plan §E
(`notes/2026-09-09-fable-audit.md`), which is §C.1's recommendation.

**The three sentences.** From `f4466d6` the Rust core is the algorithm's reference: changes are
made in `crates/`, `sherd_refit/` is frozen at `9cbcbbc` as the parity oracle for the stages R
already describes, the parity harness is kept green at 23 804 / 0 and never extended, and new work
is judged by `tools/evaluate.py` over the ground-truth sets at seeds 0–4. That judgement now has a
harness — `tools/quality_gate.py`, **40 runs in 7.0 min** against a 25-minute budget — and its first
table is the first five-seed sweep the *port* has ever had. It found something: every prohibition
R §13 states holds at every seed, and the port's **spread** is wider than the reference's on three
sets of seven — `pot_B` accepts the same non-adjacent pair at two of five seeds, `pot_H` keeps no
correct join at all at seed 2, `pot_A` loses two joins at seed 2 — which is exactly the material
steps 7 and 8 of the plan exist to deal with.

---

## 1. The decision, and what it is not

D §13's question 7 asked "at which gate does the Rust core become the algorithm's reference", and
proposed the end of phase 3a — after `pyo3`. That proposal inherited roadmap item 7's premise:
*"Rust/GPU is deliberately last: it must port a settled algorithm."* Events reversed the order.
Item 2's partner search hit a geometric ceiling — one seam is 5–22 % of a fragment's breakline, so
the best of thousands of poses reaches that on pairs that never touched
(`notes/2026-09-06-scale-pairs.md` §4.4) — all-pairs became the design, and the port was built
before items 3–6 on an algorithm whose known blocker, fracture-mask precision on thin walls, is
unsettled. Under the old rule every one of roadmap items 3–6 would be written in Python, dumped as
fixtures, ported and re-gated: three implementations of every change, for a Python path that is
9.6× slower on the one mixed development set (D §10.3) and that nothing downstream uses.

Four terms, and they are what "the reference" now means:

* **Algorithm changes are made in Rust.** Items 3–6 and everything after them are designed, written
  and gated in `crates/`. Nothing is prototyped in `sherd_refit/` first.
* **`sherd_refit/` is frozen at `9cbcbbc`** — the last commit that touched it, and the one the
  committed slab dump comes from (D §10.1) — and stays as the parity oracle **for the stages R
  already describes, and for nothing else**. It is not deleted and not retired. `ALGO_REF` names
  `9d4b9d3`'s algorithm; the sixteen parity rows are the port's evidence that its sixteen stages do
  what R says; that evidence is worth keeping. What ends is the *direction*: no change flows
  Python → fixtures → Rust again.
* **The parity harness is frozen with it.** `sherd-parity`'s sixteen stages, `dump_fixtures.py`,
  `dump_outputs.py`, `compare_fixtures.py` and the eight dumps are kept green at **23 804 / 0** and
  **never extended**. A new stage has no Python side, so a new row would have nothing to compare
  against. A row whose count changes from now on is a regression in a stage R describes, which is
  what the harness is still for.
* **The quality gate for new work is `tools/evaluate.py` at seeds 0–4**, run by
  `tools/quality_gate.py`. `parity` answers "does the port still reproduce the reference";
  `quality_gate.py` answers "is the algorithm better", and after this commit only the second
  question can be asked of new work.

Two consequences. **Phase 3a (`pyo3`) is struck**: its exit criterion — "the Python pipeline with
Rust kernels reproduces the Rust CLI" — is a bridge back to a package that no longer holds the
algorithm; it is rebuilt only if the museum asks for a Python API. And **D §13's question 6 is
due**: `sherd-refit` is the Rust binary's name to take whenever the team wants it, because the
Python is now a test oracle rather than the product.

What the flip is *not*: it is not a licence to change R. R remains the written algorithm and the
port remains verified against it; what changes is where the *next* algorithm is written.

## 2. `tools/quality_gate.py`

```
python tools/quality_gate.py                                   # eight sets, seeds 0-4, CPU
python tools/quality_gate.py --sets pot_C pot_H --seeds 0 1    # a subset
python tools/quality_gate.py --render-only                     # re-render the table from JSON
```

For each of the eight development sets and each of five seeds it runs
`sherd-refit-rs run <input> --out <work> --backend cpu --no-preview --no-meshes --seed s`, scores
the result with `tools/evaluate.py`, and writes one table — fragment accuracy, precision, the four
join buckets, group purity, wall — as `output/quality/quality.md` and `quality.json`, one row per
(set, seed).

**Why Python and not a subcommand of the binary.** The score *is* `tools/evaluate.py`. A Rust twin
would have to reimplement its buckets, its centroid-referenced pose error (the meshes sit 300–500
units from the file origin, so a 0.5° rotation error alone reads as millimetres of "translation" —
`evaluate.py`'s own docstring) and its fragment-weighted purity. Two scorers that can disagree are
worse than one scorer in an environment the metric already requires: `evaluate.py` needs numpy and,
for the centroid form, Open3D. The museum runs the *binary* for the algorithm and this one file for
the number. It imports `evaluate` as a module rather than shelling out to it, which is what makes
forty runs affordable: Open3D loads once instead of forty times and each set's vertex centroids are
read once instead of five times.

**Cost, measured.** 40 runs, **421.7 s = 7.0 min**, cold caches, against the 25-minute budget the
hand-over was made under. Per set, the sum of its five runs:

| set | fragments | five runs | | set | fragments | five runs |
|---|---:|---:|---|---|---:|---:|
| terracotta | 4 | 10.4 s | | pot_H | 11 | 38.7 s |
| pot_A | 8 | 22.7 s | | `synthetic_20` | 20 | 92.8 s |
| pot_B | 9 | 26.5 s | | `mixed_ABG` | 24 | 182.9 s |
| pot_C | 7 | 20.1 s | | | | |
| pot_G | 7 | 20.9 s | | **total** | | **421.7 s** |

The set's work directory is reused across its five seeds, so only the first pays for the fragment
cache: R §3.7 says a cache written at one seed has R §3.5's three sampled arrays recomputed by the
run that wants another and nothing else, and that is what makes the second through fifth seed of
`synthetic_20` cost 17 s against the first one's 24 s. The trees are deleted per set once scored
(`--keep-work` keeps them), because forty of them do not fit beside the caches they came from.

**The terracotta has no ground truth to score.** `input/test_fragments_1` carries no
`ground_truth.json` — the museum's assembly was never staged as poses — so `evaluate.py` cannot be
run on it, and its columns read `—` in the table. That is a missing ground truth, not a missing
measurement: R §13's terracotta row is a set of *decisions*, and the script checks them straight off
`report.json` — the two joins 021–094 and 094–104 and no others, 007 unplaced, both penetrations 0,
both tight contacts ≥ 0.27, and the two seams within 20 % of 20.3 t and 10.7 t.

## 3. The five-seed table

`tools/quality_gate.py`, commit `f4466d6`, backend `cpu`, `--no-preview --no-meshes`,
`evaluate.py` at 5° / 0.5 t with the translation read at the fragment centroid.

| set | seed | frag acc | precision | correct | wrong pose | non-adj | cross-obj | purity | joins | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `terracotta` | 0 | — | — | — | — | — | — | — | 2 | 3.5 |
| `terracotta` | 1 | — | — | — | — | — | — | — | 2 | 1.5 |
| `terracotta` | 2 | — | — | — | — | — | — | — | 2 | 2.0 |
| `terracotta` | 3 | — | — | — | — | — | — | — | 2 | 1.9 |
| `terracotta` | 4 | — | — | — | — | — | — | — | 2 | 1.4 |
| `pot_A` | 0 | 100.0 % | 1.000 | 7 | 0 | 0 | 0 | 1.000 | 7 | 5.6 |
| `pot_A` | 1 | 87.5 % | 1.000 | 6 | 0 | 0 | 0 | 1.000 | 6 | 4.1 |
| `pot_A` | 2 | **75.0 %** | 1.000 | 5 | 0 | 0 | 0 | 1.000 | 5 | 4.2 |
| `pot_A` | 3 | 100.0 % | 1.000 | 7 | 0 | 0 | 0 | 1.000 | 7 | 4.6 |
| `pot_A` | 4 | 100.0 % | 1.000 | 7 | 0 | 0 | 0 | 1.000 | 7 | 4.3 |
| `pot_B` | 0 | 88.9 % | 1.000 | 7 | 0 | 0 | 0 | 1.000 | 7 | 6.2 |
| `pot_B` | 1 | 88.9 % | 1.000 | 7 | 0 | 0 | 0 | 1.000 | 7 | 5.2 |
| `pot_B` | 2 | **77.8 %** | **0.857** | 6 | 0 | **1** | 0 | 1.000 | 7 | 4.8 |
| `pot_B` | 3 | 100.0 % | 1.000 | 9 | 0 | 0 | 0 | 1.000 | 9 | 5.2 |
| `pot_B` | 4 | **77.8 %** | **0.857** | 6 | 0 | **1** | 0 | 1.000 | 7 | 5.1 |
| `pot_C` | 0 | 75.0 % | 0.667 | 2 | 0 | 0 | 0 | 1.000 | 3 | 4.3 |
| `pot_C` | 1 | 75.0 % | 0.667 | 2 | 1 | 0 | 0 | 1.000 | 3 | 3.8 |
| `pot_C` | 2 | 50.0 % | 0.500 | 1 | 0 | 0 | 0 | 1.000 | 2 | 3.9 |
| `pot_C` | 3 | 75.0 % | 0.667 | 2 | 1 | 0 | 0 | 1.000 | 3 | 4.1 |
| `pot_C` | 4 | 75.0 % | 0.667 | 2 | 0 | 0 | 0 | 1.000 | 3 | 4.0 |
| `pot_G` | 0 | 0.0 % | 0.000 | 0 | 2 | 0 | 0 | 1.000 | 2 | 4.2 |
| `pot_G` | 1 | 0.0 % | 0.000 | 0 | 1 | 0 | 0 | 1.000 | 1 | 4.6 |
| `pot_G` | 2 | 0.0 % | 0.000 | 0 | 1 | 0 | 0 | 1.000 | 1 | 4.2 |
| `pot_G` | 3 | 0.0 % | 0.000 | 0 | 1 | 0 | 0 | 1.000 | 1 | 4.1 |
| `pot_G` | 4 | 0.0 % | 0.000 | 0 | 2 | 0 | 0 | 1.000 | 2 | 3.8 |
| `pot_H` | 0 | 36.4 % | 0.429 | 3 | 3 | 1 | 0 | 1.000 | 7 | 7.7 |
| `pot_H` | 1 | 36.4 % | 0.429 | 3 | 3 | 1 | 0 | 1.000 | 7 | 7.2 |
| `pot_H` | 2 | **0.0 %** | **0.000** | 0 | 3 | 3 | 0 | 1.000 | 6 | 7.5 |
| `pot_H` | 3 | 36.4 % | 0.429 | 3 | 4 | 0 | 0 | 1.000 | 7 | 8.5 |
| `pot_H` | 4 | 36.4 % | 0.500 | 3 | 3 | 0 | 0 | 1.000 | 6 | 7.7 |
| `synthetic_20` | 0 | 90.0 % | 1.000 | 19 | 0 | 0 | 0 | 1.000 | 19 | 23.8 |
| `synthetic_20` | 1 | 90.0 % | 1.000 | 21 | 0 | 0 | 0 | 1.000 | 21 | 17.1 |
| `synthetic_20` | 2 | 85.0 % | 1.000 | 19 | 0 | 0 | 0 | 1.000 | 19 | 18.1 |
| `synthetic_20` | 3 | 90.0 % | 1.000 | 19 | 0 | 0 | 0 | 1.000 | 19 | 16.3 |
| `synthetic_20` | 4 | 95.0 % | 1.000 | 27 | 0 | 0 | 0 | 1.000 | 27 | 17.5 |
| `mixed_ABG` | 0 | 58.3 % | 0.667 | 12 | 3 | 0 | 3 | 0.864 | 18 | 38.9 |
| `mixed_ABG` | 1 | 50.0 % | 0.625 | 10 | 2 | 0 | 4 | 0.750 | 16 | 36.6 |
| `mixed_ABG` | 2 | 54.2 % | 0.688 | 11 | 1 | 1 | 3 | 0.850 | 16 | 35.5 |
| `mixed_ABG` | 3 | 66.7 % | 0.789 | 15 | 2 | 0 | 2 | 0.850 | 19 | 35.4 |
| `mixed_ABG` | 4 | 58.3 % | 0.706 | 12 | 3 | 1 | 1 | 0.952 | 17 | 36.5 |

**The seed-0 rows reproduce the record exactly**, which is the harness's own check: D §10.3's
task-Y quality paragraph reads terracotta's row exactly (the two joins, 007 unplaced, both `pen` 0,
tight 0.56/0.67 and 0.54/0.56, seams 20.667 t and 12.333 t — this run's seed 0 gives 20.67/12.33 and
0.557/0.671, 0.535/0.557), pot_A 100 % / 1.000, pot_B 88.9 % / 1.000, pot_C 75 % / 0.667, pot_G 0 %
with two wrong-pose joins on adjacent pairs, pot_H 36.4 % / 0.429, `synthetic_20` 90 % / 1.000, and
`mixed_ABG` 58.3 % / 0.667 with 18 joins — 12 correct, 3 wrong pose, 3 cross-object — at purity
0.864, which is D §10.3's port baseline to every digit. Nothing in the table above seed 0 was known
before this run.

## 4. What the sweep found: the gate holds, the band does not

The script prints two verdicts per set, and the distinction is the point.

**Gate — what R §13 states as a prohibition. All of it holds, at every seed of every set.**
Cross-object joins **0** and group purity **1.000** on all seven single-object collections (35
runs). pot_G's rule — at most two joins used, every one a wrong-pose join on a ground-truth-adjacent
pair — holds at all five seeds (1 or 2 joins, all wrong-pose). The terracotta's decision row holds
at all five: the same two joins, 007 unplaced, both penetrations 0, tight 0.521–0.687, seams
20.67–21.00 t and 11.33–12.33 t, all inside 20 % of 20.3 t and 10.7 t.

**Band — R §13's quoted spread of accuracy and precision. The port is outside it on three sets of
seven.**

| set | R §13 (the reference's five draws) | the port's five draws | outside at |
|---|---|---|---|
| `pot_A` | 87.5–100 %, precision 1.000 | **75.0–100 %**, 1.000 | accuracy, seed 2 |
| `pot_B` | 88.9–100 %, precision 1.000 | **77.8–100 %**, **0.857–1.000** | both, seeds 2 and 4 |
| `pot_C` | 50–75 %, 0.500–0.667 | 50–75 %, 0.500–0.667 | — |
| `pot_G` | 0 %, ≤ 2 wrong-pose joins | 0 %, 1–2 wrong-pose joins | — |
| `pot_H` | 27.3–36.4 %, 0.333–0.500 | **0.0–36.4 %**, **0.000–0.500** | both, seed 2 |
| `synthetic_20` | 85–95 %, 1.000 | 85–95 %, 1.000 | — |

**This is not gated, and the reason is not leniency.** R §13's bands are the *reference's* own five
draws (tasks Y and Z, thirty-six runs). The port's five are a second sample of the same chaotic
quantity — R §13 already says pot_C's decisions "flip on nothing" and that the port's RNG streams
are not numpy's (PMC-9) — not a subset of the first. Gating one five-draw sample with another fails
a run for a *draw* rather than for a *change*, and widening the band to admit the port's draws would
be the other error the standing rules forbid. So the band is **reported**, with the seeds named, and
the rule for every later step is that a set moving out of the reference's spread, or back into it,
has to be said out loud. What removes the spread is steps 8 and 10 of the plan, not a wider number.

**Two of the three are false joins in substance, not lost recall.** Read one at a time:

* **`pot_A` seed 2 is recall alone.** Five joins used, **all five correct**, precision 1.000; the
  set simply places six of eight fragments instead of eight. Nothing wrong was accepted.
* **`pot_B` seeds 2 and 4 accept the same wrong pair.** `Pot_B_Piece_02` — `Pot_B_Piece_07`, a
  non-adjacent pair of the same pot, at **145.1° and 39.59 t** from the ground truth (145.2° and
  39.61 t at seed 4). It is the same pair at both seeds, so it is a candidate that sits near the
  acceptance boundary and crosses it on the draw, not a fluke of one run. The group is still one
  group of eight at purity 1.000 — the join is wrong, the *objects* are not confused. R §13 records
  precision 1.000 in all seven of the reference's `pot_B` runs, so this is new.
* **`pot_H` seed 2 keeps nothing correct.** Six joins: three wrong-pose, three non-adjacent, zero
  correct, against 36.4 % / 0.429 at the other four seeds. Two of the three wrong-pose joins miss by
  a hair — **0.8° / 0.56 t** and **1.7° / 0.57 t** against `evaluate.py`'s 0.5 t tolerance — so this
  row is partly a tolerance edge and partly a real collapse; the other four joins are 11.0° / 2.29 t,
  13.9° / 10.73 t, 102.3° / 52.06 t and 130.6° / 147.27 t, and those are not near anything. The
  groups are 5 and 3 fragments, both at purity 1.000.

That is the material step 7 measures and step 8 has to remove: a confirmed tier a `pot_B` seed-2
join cannot enter, and a margin that says why it cannot. The audit's §C.3 argued that "zero false
joins on all benchmarks" is not a testable statement until it is stated over the seed spread; this
table is the first measurement of that spread on the port's side, and it says the statement would
be false today on `pot_B`.

**`mixed_ABG` is reported, never gated** (D §10.3, decision 2026-09-08): 1–4 cross-object joins and
purity 0.750–0.952 over the five seeds, against the recorded baseline of 3 and 0.864 at seed 0.
Removing those is roadmap item 4, which is step 10.

## 5. D §12: phase 3 becomes steps 7–12

The three phase-3 rows — 3a `pyo3`, 3b the Tauri app, 3c packaging — were written on the assumption
this task falsified. They are replaced by the audit's plan §E, steps 7–12, each with its own gate:

| step | what | gate | h |
|---|---|---|---:|
| 7 | measure before the tiers: every accepted candidate's scores, margin, `determined`, stability under two resamples, slide, support count and `evaluate.py` class over the 40 runs; the same pass (**preprocessing only**) computes the per-fragment features and their same-vs-different-object AUC | the table exists; the strict thresholds, `m` and the feature shortlist are chosen **on it** and written down with their false-join margin | 8 |
| 8 | tiers: `Tier` on `Candidate`, strict `Thresholds`, the three probes, assembly from confirmed only, report sections and a per-fragment candidate index | zero false joins in the confirmed tier on 8 sets × 5 seeds, confirmed recall reported per set, terracotta's two joins confirmed, byte-identical with `--tiers off` | 10 |
| 9 | review images and `constraints.json` v1 | images deterministic; `must_not_join [021, 094]` yields 094–104 only; `must_join [007, 021]` reports unsatisfiable; a pinned pose places without matching | 8 |
| 10 | object separation: `Features` in the cache, group consensus, support counts, group merging through a confirmed join | `mixed_ABG` cross-object 0 and purity 1.000 in the confirmed tier, correct joins ≥ 12 at seeds 0–4; the seven single-object sets unmoved or explained | 12 |
| 11 | *optional* spike: seam segments on `mixed_all` (**preprocessing only plus segment matching**) | recall at K = 5/10/15 against 0.36/0.47/0.53; adopted only above 0.8 at K = 10 | 6 |
| 12 | final acceptance: `synthetic_60`, `synthetic_170`, `mixed_all` at seeds 0 and 1 | §10.3's CPU walls; cross-object 0 and purity 1.000 in the confirmed tier | 6 |

50 agent-hours, and D §12 now says the column is agent-hours for one Opus executor rather than
engineer-weeks, because that is how steps 1–6 were executed.

**The small-set rule is stated with the plan.** Nothing above 27 fragments is *matched* before step
12; `mixed_all` (164) and `synthetic_170` are matched exactly **once**, at step 12. There are two
stated exceptions and both are preprocessing only and linear in the fragment count: step 7's
per-fragment feature pass and step 11's segment matching, which reads breaklines and never forms a
pair hypothesis. A step that needs a large collection for any other reason is a step to re-plan.

3a is struck with its reason. The Tauri app and packaging are carried as one row **after** step 12
rather than deleted: the screen the app exists for is the probable-join review, which is step 9's
output, so it belongs behind the algorithm work and not beside it.

## 6. What the specs now say

* **D §13 question 7 — answered**, with the four terms, the reason the proposal failed, and the two
  consequences (3a struck, question 6 due).
* **D §12** — cross-cutting risk 1's second half rewritten (the Python-first route is over); the
  three phase-3 rows replaced by steps 7–12 with their gates; the preamble carries the small-set
  rule, the standing gates, and what step 7 starts from.
* **D §10.4** — layer 5 marked **frozen** (kept green, never extended); a new **layer 7**, the
  quality gate, with the gate/band distinction, the measured 7.0 min, the terracotta's decision row
  and the argument for Python.
* **README** (Russian) — a section on the hand-over with its four terms, and one on
  `tools/quality_gate.py` with the two verdicts and the measured cost.

## 7. Gates

| gate | result |
|---|---|
| `cargo test --workspace` (debug) | **pass** — 19 suites, **388 tests** |
| `cargo test --workspace --release` | **pass** — 19 suites, **388 tests** |
| `cargo build -p sherd-cli --no-default-features --locked` | **pass** |
| `cargo clippy -p sherd-cli --no-default-features --all-targets --locked -- -D warnings` | **pass** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **pass** |
| `cargo fmt --all --check` | **pass** |
| `parity --stage all`, both modes, eight dumps | **pass** — 256 rows, **23 804 checks, 0 failed**, `cdec069`'s number to the check |
| `pytest -q` | **pass** — 60 passed in 100.7 s |
| **byte-identity** against H3 (`30a0e4e`), both backends | **pass** — **142 files, 120 byte-identical, 22 exempt, 0 differing**; per backend 71 / 60 / 11, the same split H3 reported |
| `quality_gate.py` under 25 min | **pass** — 421.7 s = **7.0 min** |
| nothing above 27 fragments matched | **pass** — nothing above 24 (`mixed_ABG`) |
| disk ≥ 4 GiB free | **pass** — 8.4 GiB at the end |

**The test count is unchanged at 388 and so is every byte of the four development sets**, which is
the property this task had to have: H4 adds one Python file and four documents and touches no crate.
The eleven exempt files per backend are the same four differences H3 named — `report.json`'s sampled
`memory` block, `timings`, `report.md`'s `## Timing` section, and `engine.commit`, which changes with
every commit. `engine.seed` is 0 on both sides and identical this time, because it existed at H3 too.

## 8. What this task did not do

* **No algorithm change, and no threshold moved.** The three out-of-band sets are reported, not
  fixed; fixing them is steps 7–10, and doing it here would have been the change this task exists to
  make possible rather than to make.
* **The band was not widened to make a gate green.** The alternative — gating on R §13's bands and
  failing this tree — was rejected on the argument of §4, not on convenience, and the band comparison
  is printed at every run so that it cannot quietly stop being checked.
* **`sherd_refit/` was not touched.** Freezing is a statement about what happens next, not an edit;
  the package is byte-identical to `9cbcbbc` and `pytest -q` still runs it.
* **The parity harness was not extended, and not shortened either.** It runs exactly the sixteen
  stages on exactly the eight dumps and reports the same total as `cdec069`.
* **Terracotta still has no staged ground truth.** Writing one would mean writing into `input/`,
  which the standing rules forbid, and R §13's decision row is what the museum's assembly actually
  supports — an adjacency without poses would make every join `unscorable`.
* **`mixed_all` and `synthetic_170` were not run.** The small-set rule holds; step 12 is where they
  are matched.

## 9. Housekeeping

Under `output/h4/`: the sixteen parity reports, the H3 binary kept for the identity comparison, and
the logs. Under `output/quality/`: the gate's own `quality.md`, `quality.json` and the 35 per-run
`evaluation.json` files — data, not committed; the table of §3 is what the repository keeps. The
eight identity run trees (1.5 GB) were deleted once compared, which is what the comparison was for.
The branch was never switched, and nothing under `input/`, `output/fixtures/` or `fixtures/` was
written or deleted. The commits are `H4.1`…`H4.4`.
