# Z — closing the phase-1e verification (V5-D1 … V5-D8, and the README item)

**Date:** 2026-09-08. **Tree:** branch `rust-core`, starting from `9bf35d6` ("V5: independent
verification of phase 1e"); `git status` clean at the start, nothing left over from an interrupted
attempt. **Machine:** Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 24.6.0, rustc 1.97.0,
`--release`. **Reference:** the working tree's own `sherd_refit/`, Open3D 0.19.0, numpy 2.5.2,
Python 3.12.9, every Python run with `OMP_NUM_THREADS=1`.

`notes/2026-09-07-phase1e-verification.md` ("V5") listed eight defects and one item below the
defect line. All nine are closed here, in seven commits (Z1–Z7) and this note. Two of them become
new **gates** rather than new prose — the phase's two coverage holes — and both are exact and both
are green; two are corrections to claims the documents make and the measurement contradicts; the
rest record what the port does where R and D said something else.

Nothing above 27 fragments was run. The largest collection touched is `mixed_ABG` (24).

---

## 1. What each defect became

| id | what it was | commit | what closed it |
|---|---|---|---|
| **V5-D1** | D §10.3's "cross-object joins 0 and group purity 1.000 **everywhere**" is wider than R §13 measures, and `mixed_ABG` — a set D §10.3, D §12 and README all name a development set — was run by nobody | **Z5** `38ea84c` | the clause is scoped to R §13's seven single-object collections; `mixed_ABG` is carried as **roadmap item 4's measured baseline**, both implementations' numbers in D §10.3, D §11 and README (§6) |
| **V5-D2** | `run --threads`'s help said "(default: one per core)" after Y4 made it cores − 1 | **Z1** `d6dbbc0` | both `run`'s and `segment`'s help name the default `cli.py` resolves an unset flag to |
| **V5-D3** | `bench` sized the pool at one thread per core and passed `workers: 0`, so R §4.2's block schedule was computed at ten where `run` computes it at nine | **Z1** | `bench` gains `--workers`; all three subcommands resolve both flags through `pool_threads` and `schedule_workers`, with a test (§2) |
| **V5-D4** | D §5's first line said the pool defaults to all cores | **Z1** | it says cores − 1, and how the two flags are resolved |
| **V5-D5** | the harness recomputed R §3.5.6's band with the **unbounded** `breakline_distance`, so E2.5's bounded sweep — the one E2 change that alters an array — had no standing gate | **Z2** `bcc99c8` | both call sites take the band from the pipeline's own function, and a new `margin bound` row compares the two arrays directly: **0 differing of 1 360 000 entries** per mode (§3) |
| **V5-D6** | `transforms.json`'s key order had no automatic gate: the dump's own copy is written with `sort_keys=True` and the harness passed R §8's insertion order as an empty slice | **Z3** `54932d6` | the row reads `<dump>/_run/transforms.json` — the file the *pipeline* wrote — and compares place for place: **0 differing of 66** on the seven dumps that carry one (§4) |
| **V5-D7** | four changes of phase 1e are on neither R §12 nor R §12.1, while both notes state "R is untouched" | **Z4** `c7a2522` | R §12.1 records `column_mean` in `render::principal_axes` and E2's three filters, each with its argument, its test and its gate (§5) |
| **V5-D8** | R §13's `pot_A` row was relaxed from "87.5 %" to "≥ 87.5 %" on an argument; `pot_C` and `pot_H` were single seed-0 draws | **Z6** `6f088b0` | fifteen runs of the reference at seeds 0–4; **two of the three rows move**, and `pot_C`'s own paragraph is corrected (§7) |
| README item | the Rust flag list omitted `--memory-budget` | **Z5** | it is there, with D §9's meaning |
| — | the counts three tests assert follow the two new rows | **Z7** `93b9e72` | the slab's samples reports, and the one skip the committed slab dump is allowed |

---

## 2. V5-D2/D3/D4 — one resolution rule, in one place

`bench` is the tool D §10.3 names for its runtime gates and it was the one tool that did not
resolve the two flags: `set_threads(args.threads.unwrap_or(0))` gives rayon one thread per core,
and `workers: 0` is read by `pipeline::run` as `rayon::current_num_threads()`. Both come out **ten**
on this machine where `run` uses **nine**, and `--workers` is not a pool size alone — it also feeds
`block_size(workers, n_pairs)`, which decides whether the pairs are walked one at a time or in 3×3
blocks.

No result of the eight development sets moves, because `block_size(9, n) == block_size(10, n)` on
every one of their pair counts (6, 21, 21, 28, 36, 55, 190, 273 — only 37–40 and 144–159 pairs
separate nine from ten). But "happens not to" is not a resolution rule, and the schedule really is a
function of the number: at pot H's 55 pairs, nine workers walk one pair per block and one worker
walks 3×3 blocks, which is a different traversal of R §4.1's pair order.

Both lines now live in `pool_threads(threads, workers)` and `schedule_workers(workers)`, called by
`run`, `segment` and `bench`, and `bench_resolves_the_two_flags_the_way_run_does` asserts the
answers. Two copies of those two lines had already drifted apart once (V4-D8 on `run` and
`segment`), which is why the third copy is a function instead.

---

## 3. V5-D5 — the harness gates the array the pipeline computes

**The decision: the harness moves to the pipeline's function, not the pipeline back to the
reference's array.** The alternative — making `samples::build` compute the exact array again —
would give the row an easier subject and cost the port E2.5's whole measured gain (R §4.2's rebuild,
22.7 core-s of synthetic 20's 169), for nothing: the bounded form is *provably* the same band, and
what was missing was a standing measurement of that proof, not the proof.

What the pipeline computes is `breakline_distance_below(S, brk, 1.5 t)`: `∞` wherever the true
distance is at or beyond the band's outer edge. `margin_indices` reads the array only as
`0.12 t < d < 1.5 t` and `∞ < 1.5 t` is the same `false`, so the *predicate* is unchanged while the
*array* is not — and the harness recomputed the band with the unbounded `breakline_distance` at both
of its call sites, so `margin count`, `margin members` and `margin fraction` all gated an array the
pipeline does not build. Its only check was task E2's own before/after byte comparison, which is a
measurement of one tree at one commit and not a gate that runs again.

Both call sites now take the band from the pipeline's own function, and a `margin bound` row beside
them compares the two arrays directly, which is stronger than comparing the bands:

* where the true distance is **below** the bound, the two must agree **bit for bit** — both take
  `sqrt` of the same minimal squared distance, the bounded search having pruned only nodes whose
  box is further than the radius;
* where it is at or beyond the bound, the bounded array must be `∞`;
* anything else is a differing entry, and the gate is zero.

| mode | rows | entries | differing |
|---|---:|---:|---:|
| injected (the reference's own `S` and `brk_P`) | 68 | 1 360 000 | **0** |
| native (the port's own arrays) | 68 | 1 360 000 | **0** |

The unbounded sweep costs the harness one pass per fragment and the pipeline nothing. D §10.2's
`samples` row and its prose carry the row; R §12.1 carries the deviation itself (§5).

---

## 4. V5-D6 — the key order, against the reference's own file

R §11.1's `poses` is a Python dict and `json.dump` writes a dict in insertion order, so the order of
`transforms.json`'s keys is a **result** of R §8 — each group's seed, then its placements in the
order the greedy pass took them, then the singletons in collection order — and not a formatting
choice. Nothing inside the dump can gate it: every JSON file the fixture sink writes goes through
`json.dumps(…, sort_keys=True)`, its own copy of `transforms.json` included. The harness passed the
insertion order to the writer as `&[]`, so the port's own file came out in **collection** order, and
no row noticed, because every other row of the stage looks its fragment up by name.

The gate now reads `<dump>/_run/transforms.json` — the pipeline's own output of the very run that
wrote the dump, and the one file on disk that carries the order. The port's own order comes from its
own `assemble` over the reference's own candidate list, which is the run the `assembly` row already
compares group for group, join for join and rejection sentence for rejection sentence.

| dump | places | differing |
|---|---:|---:|
| terracotta | 4 | **0** |
| `pot_A` | 8 | **0** |
| `pot_B` | 9 | **0** |
| `pot_C` | 7 | **0** |
| `pot_G` | 7 | **0** |
| `pot_H` | 11 | **0** |
| `synthetic_20` | 20 | **0** |
| slab | — | skipped: the committed dump carries no `_run` |
| **total** | **66** | **0** |

**The row is not vacuous.** On the terracotta the reference's key order is
`021, 094, 104, 007` and the collection order is `007, 021, 094, 104` — four places out of four —
so the empty slice the harness used to pass would fail this row on the first set it runs.

Two things were considered and not done. Re-dumping the reference's `transforms.json` without
`sort_keys` would have meant re-running `tools/dump_fixtures.py` over eight collections to gain one
file, with every other fixture byte at risk; deriving the expected order from `assembly/groups.json`
would have gated the port against a restatement of the rule rather than against the reference,
and it is not even equivalent — `assemble` sorts the groups by size after the singletons are
appended, so the concatenation of the sorted groups is not the insertion order whenever a later
group ends up larger than an earlier one.

---

## 5. V5-D7 — four deviations, recorded where R §12.1 says they belong

R §12.1's rule, from task X: *anything the port does differently that is not on R §12 or in the
addenda is an undeclared deviation, however small its measured effect.* Phase 1e made four while
both notes stated that R was untouched. The addendum now names each with its argument, its unit test
and the gate that measures it.

| deviation | class | proof of result-identity |
|---|---|---|
| `column_mean` in `render::principal_axes` (Y4) | a summation order | the reference's `V - V.mean(0)` is the same C-contiguous `(N, 3)` axis-0 reduction as R §8.2's, which task Y measured bit-identical to a per-column left-to-right accumulation; D §10.2's `outputs` `axis` row, fed the **reference's own** preview points at its own poses, measures **0.000° on seven dumps and 8.5e-7° on pot_C** against a gate of 1° |
| the bounding-box reject in front of every bounded query (E2.3) | a filter | the box distance is a lower bound on the distance to any point of the cloud, and the comparison is widened by `16 ε`; `the_box_filter_never_rejects_a_neighbour_the_traversal_would_have_found` sweeps 2 400 queries at four radii each |
| `NearMask` in front of R §5.2 (E2.6) | a filter | `may_be_near` is `false` only when nothing is within the radius; a cell is never narrower than the radius and the mask is dilated once, so the 3×3×3 block covers every neighbour; `the_mask_never_hides_a_neighbour`, 60 000 queries, three cloud shapes, five radii |
| the two bounded centroid sweeps of R §3.5 (E2.5) | **not a filter — it changes an array** | the equality is of the predicates `distance[i] >= 0.15 t` and `0.12 t < d_brk[i] < 1.5 t`, and nothing else reads either array; the `breakline` row gates the first half through `ns`/`nf`, and Z2's `margin bound` row gates the second directly (§3) |

The empirical check over all four at once is E2's own, re-derived by V5 §5.2 against a freshly built
phase-1d binary: 70 output files on three collections, 64 byte-identical outright and the six
`report.*` identical once the wall clock is taken out. Task Z re-derived it again against
`9bf35d6` on four collections (§8).

---

## 6. V5-D1 — `mixed_ABG` is roadmap item 4's baseline, not a phase gate

D §10.3's quality clause read "within the reference's own spread as R §13 states it on every listed
set, with cross-object joins 0 and group purity 1.000 **everywhere**". R §13's scope is seven
**single-object** collections and its 21 sweep runs were on three of them; the one *mixed*
development set had never been scored on either side until V5 §4.4 did it. Neither implementation
meets the clause there.

Re-measured at this commit for the port (`run --no-meshes`, previews on, cold, scored with
`tools/evaluate.py`), and quoted from V5 §4.4 for the reference (seed 0, `workers=5`,
`OMP_NUM_THREADS=1`). **Every quality figure of V5 §4.4 reproduces to the digit**, which is what a
port whose `sherd-core` this task did not touch should do:

| `mixed_ABG`, 24 fragments, 276 pairs | **the port** | **the reference** |
|---|---|---|
| fragment accuracy | 14 / 24 = **58.3 %** | 14 / 24 = **58.3 %** |
| per object | A 75.0 %, B 88.9 %, G 0 % | A 62.5 %, B 100 %, G 0 % |
| precision | 0.667 | 0.750 |
| joins used | **18** — 12 correct, 3 wrong pose, **3 cross-object** | **16** — 12 correct, 3 wrong pose, **1 cross-object** |
| groups | 10 + 8 + 2 + 2 + 2×1 | 14 + 3 + 7×1 |
| **group purity** | **0.864** (0.80 and 0.88) | **0.706** (0.64 and 1.00) |
| matching stage | **36.1 s** here, cold (V5 measured 64.0 s in a session that was also running the reference) | 613.7 s, V5's measurement |

Cross-object joins are exactly what D §11's roadmap item 4 (*object separation*) exists to remove —
cycle consistency over whole groups, per-fragment `Features` and a group consensus — and none of it
is in phase 1 or phase 2. Holding a port of a frozen algorithm to a standard the algorithm itself
does not meet leaves two options, failing a gate for reproducing the reference or changing the
algorithm to pass it, and both are outside R §12's licence. So the table above is the **baseline
item 4 is measured against**, recorded in D §10.3, pointed at from D §11's item-4 row, and marked in
D §12's phase-1e and phase-2d exit criteria and in README. The phase gate on that set stays the one
every development set carries: the same decisions as the reference, inside D §10.2's tolerances, at
D §10.3's runtime.

The two runtimes are not a clean pair — mine is this machine idle and V5's was measured beside the
reference's own 10-minute run — so the ratio to quote is V5's own paired one, **9.6×**, and not
613.7 / 36.1. What the re-measurement is for is the quality column, and there every figure is V5's.

Worth keeping in view: on the one figure that measures what those joins *cost*, the port is ahead —
group purity 0.864 against 0.706 — and fragment accuracy is identical to the fragment.

---

## 7. V5-D8 — the reference at seeds 0–4 on pot_A, pot_C and pot_H

Task Y's driver, in its shape: `pipeline.run(input, out, target_faces=200000, workers=5,
params=Params(seed=k), preview=False, refine=True, write_meshes=False)`, `OMP_NUM_THREADS=1`,
scored with `tools/evaluate.py`. Fifteen runs, 19.2 minutes of measured run time. One output directory per
set is reused across the seeds, so R §3.7's cache keeps the mesh work and only the seeded arrays are
rebuilt; that the results differ by seed at all is the check that the invalidation works.

| set | seed | fragment accuracy | precision | joins used | of them | groups | cross-object | purity | wall |
|---|---:|---|---:|---:|---|---|---:|---:|---:|
| `pot_A` | 0 | 7/8 = **87.5 %** | 1.000 | 6 | 6 correct | 7+1 | 0 | 1.000 | 96.5 s |
| | 1 | 7/8 = **87.5 %** | 1.000 | 6 | 6 correct | 7+1 | 0 | 1.000 | 90.7 s |
| | 2 | 8/8 = **100 %** | 1.000 | 7 | 7 correct | 8 | 0 | 1.000 | 93.2 s |
| | 3 | 7/8 = **87.5 %** | 1.000 | 6 | 6 correct | 7+1 | 0 | 1.000 | 88.3 s |
| | 4 | 8/8 = **100 %** | 1.000 | 7 | 7 correct | 8 | 0 | 1.000 | 95.7 s |
| `pot_C` | 0 | 3/4 = 75.0 % | 0.500 | 4 | 2 correct, 1 wrong pose, 1 unscorable | 5+1+1 | 0 | 1.000 | 40.0 s |
| | 1 | 3/4 = 75.0 % | 0.667 | 3 | 2 correct, 1 wrong pose | 4+1+1+1 | 0 | 1.000 | 40.8 s |
| | 2 | 3/4 = 75.0 % | 0.667 | 3 | 2 correct, 1 unscorable | 3+2+1+1 | 0 | 1.000 | 37.9 s |
| | 3 | 3/4 = 75.0 % | 0.500 | 4 | 2 correct, 1 wrong pose, 1 unscorable | 5+1+1 | 0 | 1.000 | 39.1 s |
| | 4 | 2/4 = **50.0 %** | 0.500 | 2 | 1 correct, 1 unscorable | 2+2+1+1+1 | 0 | 1.000 | 45.3 s |
| `pot_H` | 0 | 4/11 = 36.4 % | 0.429 | 7 | 3 correct, 4 wrong pose | 8+1+1+1 | 0 | 1.000 | 97.1 s |
| | 1 | 4/11 = 36.4 % | 0.429 | 7 | 3 correct, 4 wrong pose | 8+1+1+1 | 0 | 1.000 | 96.9 s |
| | 2 | 3/11 = **27.3 %** | **0.333** | 6 | 2 correct, 2 wrong pose, 2 non-adjacent | 4+3+2+1+1 | 0 | 1.000 | 94.3 s |
| | 3 | 4/11 = 36.4 % | 0.429 | 7 | 3 correct, 4 wrong pose | 8+1+1+1 | 0 | 1.000 | 98.8 s |
| | 4 | 4/11 = 36.4 % | **0.500** | 6 | 3 correct, 3 wrong pose | 7+1+1+1+1 | 0 | 1.000 | 96.2 s |

**Two of the three rows move, and the third's relaxation is now backed.**

* **`pot_A`** — 87.5, 87.5, 100, 87.5, 100 %, precision 1.000 in all five, every join used correct.
  Y5 relaxed the row from "87.5 %" to "≥ 87.5 %" on an argument (the port scores 100 % and 100 % is
  not worse than 87.5 %); the reference's own spread is 87.5–100 %, so the relaxation is measured
  now, and R §13 states the band.
* **`pot_C`** — 75, 75, 75, 75, **50 %**. R §13's paragraph on this row said "fragment accuracy is
  what is stable here; the precision column on this set is not". That was true of the **old**
  thickness estimator at five seeds and is false of §3.2's deterministic one: at seed 4 the
  reference places two of the four scorable fragments instead of three, on two joins instead of
  four. The cause is the one the paragraph already names — piece 01 has no correct pose available
  and its `tight` sits on `min_tight` — and what the fifth seed shows is that the flip can take a
  *placement* with it and not only a precision digit. Both columns of the row are bands now.
* **`pot_H`** — 36.4 % at four seeds of five and **27.3 %** at seed 2, precision 0.333–0.500. The
  losing seed is the one where the 8-fragment group does not form: 4+3+2 instead of 8+1+1+1.

**No band was widened to admit a figure of the port's.** The port scores 100 % / 1.000 on pot_A,
75 % / 0.667 on pot_C and 36.4 % / 0.429 on pot_H — inside the old rows as well as the new ones,
and at the **top** of both of pot_C's bands. What the sweep moved is the bottom of each band, which
is the direction that cannot flatter the port.

Cross-object joins 0 and group purity 1.000 held in all fifteen runs, so that row of R §13 now rests
on 36 runs of the reference.

---

## 8. Gates

| gate | result |
|---|---|
| `cargo fmt --all --check` | **PASS** |
| `cargo clippy --workspace --all-targets -- -D warnings` | **PASS** |
| `cargo test --workspace --no-fail-fast` (debug) | **PASS** — **328 passed, 0 failed, 2 ignored** (V5's 327 plus `bench_resolves_the_two_flags_the_way_run_does`) |
| `cargo test --workspace --release --no-fail-fast` | **PASS** — **328 passed, 0 failed, 2 ignored** |
| `cargo test --workspace --release -- --ignored` | **PASS** — 2 |
| `parity --stage all`, 8 sets × 2 modes | **PASS** — 23 804 checks, **0 failed** |
| `parity --verify-checksums`, 8 dumps | **PASS** — **19 061 files**, every one matching its manifest |
| `python -m pytest -q` | **PASS** — **60 passed** in 130.2 s |
| outputs byte-identical to `9bf35d6`'s | **PASS** — 92 files on four collections (below) |

**One arithmetic correction to V5 §2.** Its `--verify-checksums` line reads "18 461 files, every
one matching its manifest" and then lists the eight dumps: 201, 542, 1 724, 2 126, 1 363, 1 338,
3 033, 8 734. That list sums to **19 061**, and 19 061 is what the eight dumps hash to today, file
for file. The per-dump figures are right and the total was not; nothing else in that section
depends on it.

### 8.1 Byte identity against `9bf35d6`

The baseline is a freshly built `9bf35d6` — `git archive` into a scratch tree outside the
repository, its own `CARGO_TARGET_DIR`, `cargo build --release` — not a binary left behind by an
earlier task. Both binaries report the same `info` block (algorithm reference `9d4b9d3`, cache
version 5). Every run is cold (`--force`, every `cache/*.sherd` rebuilt).

| set | shape | files | identical | differing |
|---|---|---:|---:|---|
| terracotta | meshes and previews on | 14 | **14** | — (13 as raw bytes; `report.json`'s clock) |
| `pot_A` | meshes and previews on | 22 | **22** | — (20 as raw bytes; `report.*`'s clock) |
| `pot_H` | meshes and previews on | 30 | **30** | — (28 as raw bytes) |
| `synthetic_20` | previews on, `--no-meshes` | 26 | **26** | — (24 as raw bytes) |

**92 files, 85 byte-identical as raw bytes and the remaining seven — four `report.json` and three
`report.md` — identical once the wall clock is removed.** It is the result the diff predicts:
`crates/sherd-core` is untouched by task Z (`git diff 9bf35d6..HEAD` reaches `sherd-cli`,
`sherd-parity` and three documents and nothing else), and `run`'s two resolution lines are
expression-identical to the ones they replaced. The prediction is not the gate; this is.

### 8.2 Parity

| set | native checks | failed | worst stage rows | injected checks | failed | worst stage rows |
|---|---:|---:|---|---:|---:|---|
| slab | 81 | **0** | samples 0.46, candidates 0.10, breakline 0.02 | 248 | **0** | nms 0.60, outputs 0.07, hypotheses 0.03 |
| terracotta | 145 | **0** | working mesh 0.94, breakline 0.45, segmentation 0.42 | 631 | **0** | stage1 0.69, nms 0.67, outputs 0.19 |
| pot_A | 306 | **0** | load 1.00, candidates 0.71, samples 0.37 | 2042 | **0** | load 1.00, nms 0.81, verify 0.54 |
| pot_B | 354 | **0** | load 1.00, candidates 0.67, samples 0.37 | 2527 | **0** | load 1.00, nms 0.76, stage2 0.75 |
| pot_C | 261 | **0** | load 0.50, samples 0.47, candidates 0.38 | 1612 | **0** | nms 0.53, load 0.50, outputs 0.41 |
| pot_G | 248 | **0** | candidates 0.38, samples 0.33, load 0.25 | 1597 | **0** | stage2 0.54, outputs 0.41, nms 0.37 |
| pot_H | 459 | **0** | candidates 0.58, samples 0.45, load 0.25 | 3650 | **0** | nms 0.87, outputs 0.62, stage2 0.50 |
| synthetic_20 | 1040 | **0** | working mesh 0.93, breakline 0.76, segmentation 0.62 | 8603 | **0** | nms 0.74, outputs 0.72, stage1 0.56 |
| **total** | **2894** | **0** | | **20910** | **0** | |

**23 804 checks over 16 stages × 8 sets × 2 modes, 0 failed.** V5 measured 23 661 on the same
dumps; the difference is exactly this task's two new rows — `margin bound` 68 fragments × 2 modes =
136, and `transforms order` 7 — and no existing row's count or worst case moved. The three worst
ratios of every set are quoted; `load 1.00` on pot_A and pot_B is D §10.2's own documented boundary
(Assimp's `fast_atof`, one `f32` ULP) and has no headroom by construction, and nothing else is above
0.94.

---

## 9. What is not closed

* **The three large collections** — `synthetic_pingsdorf_60`, `synthetic_pingsdorf_170` and
  `mixed_all` — remain the final acceptance after phase 2 (decision 2026-09-07). Nothing here
  started one.
* **`mixed_ABG`'s cross-object joins** are roadmap item 4's work, and the baseline in §6 is what
  item 4 will be measured against. Neither implementation is at 0 today.
* **The GPU column of every gate** — there is no GPU executor yet (phase 2a).
* **`--second-pass-top` and `--screen-top-k`** are off by default and no run or fixture here
  exercises them.
* **Two CI gaps V5 §3.1 named** are unchanged: `test-release` does not run `cargo test --doc`, and
  `nextest` skips `#[ignore]`d tests in both jobs, so the two ignored tests run nowhere in CI.
* **The slab dump cannot gate the key order** (§4): it carries no `_run`, and the row skips there
  with its reason printed rather than passing silently.

## 10. Housekeeping

Everything this task wrote lives under `output/bench_z/` (11 MB after pruning) — the sweep driver
and its `sweep.json`, the fifteen scored `score_<set>_seed<k>/` directories, the sixteen parity
logs, the byte-identity trees pruned to `transforms.json` and `report.*`, the `mixed_ABG` run
pruned the same way, and the five helper scripts (`sweep.py`, `parity_all.sh`, `identity.sh`,
`compare_out.py`, `summarise_parity.py`). The scratch build of `9bf35d6`
lived outside the repository and was deleted. Nothing under `input/`, `output/fixtures/` or
`fixtures/` was touched, the branch never left `rust-core`, and no run above 27 fragments was
started.
