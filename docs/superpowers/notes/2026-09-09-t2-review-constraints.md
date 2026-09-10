# T2: review images and constraints — what the tool shows a person, and what a person tells the tool

**Date:** 2026-09-09. **Tree:** branch `rust-core`, previous step `db6178e` (T1).
**Step:** 9 of the audit's amended plan (`notes/2026-09-09-fable-audit.md` §E; D §12 row 9).
**What it is:** the second half of roadmap item 3. T1 gave every accepted candidate a band and
built R §8 from the confirmed one; a band a conservator cannot see is a band they cannot argue
with, so this step gives them **a picture per join** and **a file to answer back with**.

Both are off. A run with no `constraints.json` and no `--review-images` writes the bytes T1 wrote.

---

## 0. What was built

| where | what |
|---|---|
| `crates/sherd-core/src/render.rs` | `render_pair`, `PairEvidence`, the seam and contact colours, and `splat_z` — `splat` with the z-buffer it already had returned beside the image |
| `crates/sherd-core/src/review.rs` | the half that measures: the seam cells, the contact classes, the three views, the caption, and `write_review_images` |
| `crates/sherd-core/src/matching/verify.rs` | `seam_points` and `seam_cells` — R §6.2 with the last division left off |
| `crates/sherd-core/src/tiers.rs` | `seam_direction` now calls `seam_points` instead of carrying a second copy of R §6.2's loop |
| `crates/sherd-core/src/assembly/constraints.rs` | `constraints.json` v1: the four lists, `resolve` against the collection, `apply` to the candidate list, and `Report` — every constraint and what it did |
| `crates/sherd-core/src/assembly/greedy.rs` | `Rejection::Constrained`, and `assemble_under` — R §8 under a resolved file |
| `crates/sherd-core/src/pipeline.rs` | the constraints resolved before R §4.1, the `must_join` rematch, the pinned candidates, `constraints::apply` after the tier pass, `assemble_under`, and the review-image pass |
| `crates/sherd-core/src/report.rs` | `## Constraints`, the `image` column on the per-fragment index, and `report.json`'s `constraints` block |
| `crates/sherd-cli/src/main.rs` | `--constraints FILE` and `--review-images` |
| `README.md` | the museum-facing section: the three bands, how to read a review image, and the file |

---

## 1. The review image

### What it draws is what the report computes

Audit §D.1 asks for a picture of a join. The temptation is to write a second notion of "the seam"
and a second notion of "contact" for the picture; the whole value of the picture is that it is the
*same* computation:

* **the seam, white** — `verify::seam_cells`, the `t/3` voxels of A's breakline whose nearest moved
  B point is within `sc.seam` with agreeing shell normals. The count of those cells **is** R §6.2's
  `seam` score, divided by three. The loop was written twice on this tree — once in `seam_score`,
  once in `tiers::seam_direction` — and is now written once, in `seam_points`;
* **B's fracture samples, green / yellow / red** — R §6.1's own `bounded_distance` over A's fracture
  BVH at the pair's `facing` window, classified by the two limits R §6.5 judges the join by:
  under `sc.tight`, under `sc.gap`, beyond. A point outside the window comes back `+∞` and is red;
* **A grey and B orange** — `PALETTE[0]` and `[1]`, the colours a group preview gives the first two
  members of a group, so the two kinds of picture read the same way.

The caption prints the five scores, the pair's own `tight` and gap limit, the band, the margin, the
support count, the slide and the seed — and the legend, so the colours do not have to be remembered.

### The three views, measured

Audit §D.1 names them: *"down the seam's mean shell normal, and the two shells"*. The seam's is the
mean of A's breakline macro-normals over the points R §6.2 counted as seam; a fragment's is the mean
of R §3.5.6's shell-margin normals `Nm`, B's rotated by the pose. Up is the seam's own direction.

Whether that is three views or one is a property of the pair, so it was measured rather than
assumed. On the terracotta's two joins the pairwise angles are

| pair | seam vs A | seam vs B | A vs B |
|---|---:|---:|---:|
| `FY234021 — FY234094` | 83.2° | 19.9° | 68.8° |
| `FY234094 — FY234104` | 9.2° | 82.5° | 75.5° |

— three genuinely different views, and on both pairs one of the three comes out edge-on, which is
the view where the wall's section and the whole contact band are visible at once. On the committed
slab, a flat plate whose outer and inner shell normals cancel in the mean, the three collapse
towards two. The angles are logged at `debug` on every image (`review views`) rather than asserted,
because a degenerate pair still deserves its picture.

### Why the annotations are composited and not splatted

A fracture surface is roughly perpendicular to the wall. Splatted into the same z-buffer as the two
shells, **every one of its points is hidden** from any view down a shell normal — the colouring
audit §D.1 asks for would have been invisible in two panels of three. So the geometry is drawn
first, the two annotations are drawn into a second image over the same `Frame`, and the second is
composited wherever it wrote a pixel. From outside the wall the fracture samples project into a
narrow band along the seam, which is exactly where a conservator looks; edge-on the same overlay is
the contact map face on. `splat` itself is untouched — `splat_z` is the same function returning the
buffer it already had — so the previews the parity harness compares pixel for pixel do not move.

One consequence is worth stating plainly, because it is the first thing a reader asks about the
picture: **B is outlined in red**. Those are B's own fracture edges that have no partner in A, which
is true of every sherd with more than one break, and the caption says so.

### Determinism

Each image starts its own `rng(seed)` (R §10's seed, which `--seed` sets), so the samples depend on
the pair and the seed and on nothing else — not on how many images came before it, not on the thread
rayon gave it. Two runs write the same PNG bytes, which is the gate audit §E states for this step
and which `review_images_are_the_same_bytes_twice_and_the_index_links_them` checks on the slab.

### Cost

100 000 samples per fragment, three views at 900 × 700, and one image per confirmed or probable
**pair** rather than per candidate.

| set | images | review stage | run | share | per image | PNG |
|---|---:|---:|---:|---:|---:|---:|
| `terracotta` | 2 | 0.11 s | 5.19 s | 2.1 % | 55 ms | 0.6 MB |
| `mixed_ABG` | 87 | 2.25 s | 65.42 s | 3.4 % | 26 ms | 56 MB |

The 87 are seed 0's 11 confirmed and 76 probable joins, which is the gate's own row for that set.
The flag is a tenth of what the tier pass costs on the same run (24.0 s on `mixed_ABG`); what it
costs in *disk* is the number worth watching, at 0.3–0.6 MB an image.

---

## 2. `constraints.json` v1

### Where each list acts, and why there

| list | where it acts | why there |
|---|---|---|
| `must_not_join` | R §4.1's pair enumeration, before matching | the pair is never scored, so it costs nothing and cannot be placed; R §8 refuses any candidate that reaches it anyway, which is the belt to that braces |
| `must_join` without a pose | a second `match_all` at R §8.1's budget, then the tier pass, then promotion | audit §D.1 says "matched with the second-pass budget", and the budget is the point of the sentence |
| `must_join` with a pose | R §4.1 skips the pair; the candidate is made from the pose | "matching is skipped" |
| `same_object` | stored | roadmap item 4 |
| `different_object` | R §8, as a refusal with its own reason | "already vetoes a join" |

The names are resolved **immediately after preprocessing**, before a single pair is matched: an
unknown name fails the run there, when it costs seconds, rather than after twenty minutes of
matching. A `must_join` pair is matched even where R §4.1's wall-ratio filter would have skipped it —
the operator's word outranks a heuristic about wall thicknesses, and the run logs how many pairs that
was.

### What a constraint may not do

**It never edits a score.** Every number in `report.json` is the number R §5–§6 computed. What a
constraint changes is which pairs are matched, which candidates R §8 may build with, and the order
it sees them in. A **pinned pose is verified at the pose it was given** — one `verify` call — so the
report carries R §6's honest opinion of it beside the fact that a person pinned it; acceptance is
the operator's, which is what pinning a pose means, and the constraints section says so.

### Validation

An unknown name is an **error, not a skip**: a typo that quietly became "no constraint" is the
failure this validation exists to prevent. So are a pair naming one fragment twice, a pair in two of
the four lists (the message names both), a `version` other than 1, and a `pose` that is not a rigid
transform — checked for a `[0,0,0,1]` last row, an orthonormal rotation to 1e-6 and a positive
determinant, because a mirrored or scaled matrix would place a fragment inside its partner and the
matcher would be blamed.

### Both spellings of `must_join`

`["a", "b"]` and `{"a": …, "b": …, "pose": …}` both parse. The pair form is what the phase-1 stub
accepted and what a person types by hand; the object form is what a review screen writes.

### R §8 sees exactly two things

`assemble_under` is `assemble_with` under a resolved file. A vetoed pair is refused with
`Rejection::Constrained` **before any test of R §8's own**, so the report says whose decision it was
rather than attributing it to penetration; and a `must_join` pair sorts before every other join,
whatever its score, which is audit §D.1's *"seeded first"*. With no constraints the sort key is
constant and the stable sort leaves R §8's two tie rules exactly as they are —
`no_constraints_is_the_assembly_r_8_always_made` asserts the joins, their order, the groups and the
poses.

---

## 3. What the outputs say

`report.md` gains **`## Constraints`** directly under `## Assembly`, because a reader who has just
seen which fragments were placed needs to know at once which of those decisions were a person's.
Every line of the file gets a row — including the ones that did nothing — with a `satisfied` column
and a sentence. From the terracotta run of §4:

```
| constraint          | A                | B                | satisfied | what it did |
| `must_join`         | FY234007_reduced | FY234021_reduced | **no**    | unsatisfiable: the pair was matched with the second-pass budget and R §6.5 accepted no candidate |
| `different_object`  | FY234007_reduced | FY234104_reduced | yes       | 5 candidate(s) were scored and none was good enough to reach the assembly |
| `same_object`       | FY234021_reduced | FY234094_reduced | yes       | recorded for the group consensus (roadmap item 4); no join was changed |
```

The two veto rows carry **two counts** — how many candidates the pair produced, and how many of
those reached the assembly under the gate this run is using — because a pair whose candidates R §6.5
refused was never offered to R §8, and saying "R §8 refused it" of that pair would be an overclaim
(T2.6; the first version said exactly that).

`## Candidates by fragment` gains an `image` column, present only on a run that wrote images:

```
| FY234021_reduced | FY234094_reduced | confirmed | 11.50 | 20.7 | 0.56 | 0.0080 | 7.54e-15 | 9.84 | 0 | — | [png](review/FY234021_reduced__FY234094_reduced.png) |
```

`report.json` gains a `constraints` block — `version` and one entry per constraint — and nothing
else. `transforms.json` is **not** changed by a constraints file: it already carries the band per
pair, and a downstream tool that needs to know a pose was pinned reads the block in `report.json`.
That is a decision step 10 may revisit if the desktop app wants the poses and the provenance in one
file.

---

## 4. The four gates audit §E states for step 9

| gate | where | result |
|---|---|---|
| images deterministic between two runs | `run_cli.rs`, slab | **pass** — one image, same bytes twice, and the index links it |
| terracotta with `must_not_join [021, 094]` yields 094–104 only and reports it | `run_cli.rs`, on demand | **pass** — the only join used, and the refused pair produced no candidate at all |
| `must_join [007, 021]` reports unsatisfiable | `run_cli.rs`, on demand | **pass** — `satisfied: false`, and the run exits 0 |
| a pinned pose places without matching | `run_cli.rs`, slab | **pass** — one candidate for the pair instead of R §5.7's five, band `confirmed`, at the pose that was pinned |

Three more were added where the slab could carry them: `must_not_join` removing the pair before
matching (a candidate list of length **zero**, not merely an assembly that placed nothing),
`different_object` vetoing a join R §6.5 accepted with the constraint named in the rejection
sentence, and an unknown name failing the run with the fragment's name in the message.

---

## 5. Gates

| gate | result |
|---|---|
| `cargo test --workspace` (debug) | **pass** — 19 suites, **426 passed**, 0 failed, 3 ignored (412/2 before) |
| `cargo test --workspace --release` | **pass** — 19 suites, 426 passed, 0 failed, 3 ignored |
| `cargo build -p sherd-cli --no-default-features` | **pass**, exit 0 |
| `cargo clippy -p sherd-cli --no-default-features --all-targets -- -D warnings` | **pass**, exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | **pass**, exit 0 |
| `cargo fmt --check` | **pass**, exit 0 |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs, all exit 0, **256 rows** (213 PASS, 43 SKIP, 0 FAIL), **23 804 checks, 0 failed** — the frozen number to the check |
| `pytest -q` | **pass** — 60 passed |
| `tools/quality_gate.py --tiers on` | **pass**, exit 0 — 657.6 s; **136 confirmed correct, 0 false** over the eight sets at seeds 0–4, and every scored column identical to T1 §4's table in all 40 rows |
| byte identity, four sets at seed 0 | **pass**, twice — **69 files, 0 differ** with the defaults on both sides and **71 files, 0 differ** with `--tiers off` on both sides |

The quality table did not move, and it could not have: this step adds no pass to a run that has no
`constraints.json` and no `--review-images`, and the gate gives it neither. The comparison was made
anyway, row by row against `output/t1/quality_on/quality.md` — fragment accuracy, precision, the
four join buckets, purity, the join count and the five confirmed columns, **40 rows of 40 equal**,
the wall seconds excluded because they are a measurement of the machine.

### The byte-identity check

The check is stated twice, because this step's off switch is not T1's.

* **The defaults on both sides** — no `constraints.json`, no `--review-images`, the tier on, which
  is what `run` does out of the box — is the statement that matters for step 9: **69 files, 0
  differ**.
* **`--tiers off` on both sides** is T1's switch still doing what T1 measured: **71 files, 0
  differ**, the same 71 (10 / 14 / 19 / 28) T1's own check compared. The two file counts differ
  from each other because the two runs build different assemblies and therefore a different number
  of `assembly_<k>.ply` and `preview_<k>.png`.

The baseline is a `db6178e` binary built in a git worktree sharing the repository's
`CARGO_TARGET_DIR` (task F7.2's method, which keeps the comparison inside the free disk). CPU,
seed 0, previews and meshes on: `transforms.json` and `report.json` compared as JSON values with
`timings`, `memory` **and `engine.commit`** removed, `report.md` above `## Timing`, every PLY and
every PNG byte for byte. `engine.commit` is excluded and named here because the two binaries are
two different commits by construction — T1's check could compare it only because nothing had been
committed yet when it ran — and it is the one field whose whole purpose is to differ.

---

## 6. What step 10 should know

1. **`same_object` and `different_object` are already in the file and already resolved.**
   `Resolved::same` and `Resolved::different` are `BTreeSet<(FragId, FragId)>` keyed with the
   smaller id first; item 4's group consensus reads them rather than re-parsing the file, and
   `different_object` already refuses a join through `Rejection::Constrained`.
2. **`assemble_under` is where group merging goes.** Audit §D.2 (c) replaces `Rejection::MergesGroups`
   with a merge through a confirmed join. The constraint veto is checked at the top of the same loop
   and before every test of R §8's own, which is the order a merge has to respect too: a merge
   proposed across a `different_object` pair must be refused for the operator's reason and not for a
   penetration.
3. **A pinned pose is the shape of an operator-supplied placement.** `pinned_candidate` builds a
   `Candidate` with R §6's scores at a given pose and the band set by fiat; anything item 4 or the
   desktop app wants to place without matching has that function to reuse.
4. **The review images are where the object features belong.** Audit §D.2's per-group consensus
   demotes a join on a feature; the caption is one `format!` away from carrying the feature and its
   `k·MAD` distance, and the picture is the place a conservator would argue with that demotion.
5. **`--review-images` with `--tiers off` writes images and no index.** The per-fragment index is
   part of `tier_sections`, which a run without a band does not write at all, so the PNGs are there
   under their predictable names and nothing in `report.md` links them. The flag combination is
   reachable and harmless; if step 10 or the desktop app wants the index without the band, that is
   where to put it.
6. **The candidate order has one addition.** Pinned candidates are appended after the pairs in
   `report.json`'s `candidates`, which is otherwise R §4.1's pair order. Nothing reads that order,
   but a tool that assumed it would be surprised.
7. **Image cost is per probable join, and the probable list is long.** `mixed_ABG` at seed 0 writes **87 images (11 confirmed, 76 probable) in
   2.25 s**, 3.4 % of its 65.4 s run and 26 ms an image — a tenth of what the tier pass itself
   costs on the same run (24.0 s) — for **56 MB** of PNG. The terracotta writes two in 0.11 s. The
   cost is linear in the probable list, and step 10's demotions make that list longer, not shorter.

---

## 7. Housekeeping

Nothing under `input/`, `fixtures/` or `output/fixtures/` was written or deleted. This task wrote
`output/t2/` (203 MB) — the two test logs, the sixteen parity logs, the quality table, the pytest
log, the demonstration run of §3 with its two review images, the `mixed_ABG` cost run without them,
and the two identity comparisons — and the quality gate refreshed `output/quality/`. The two
identity trees (1.2 GB each) and `mixed_ABG`'s 56 MB of PNGs were compared, measured and deleted;
`target/debug/incremental` (3.8 GB) was dropped to make room. Free disk stayed above **5 GiB**
throughout and ended above 7 GiB. The baseline worktree under the scratchpad is removed with
`git worktree remove`.

Nothing above 24 fragments was matched by this task's own runs; the quality gate's `mixed_ABG` (24
meshes, three pots) is the largest collection, and it was matched by the gate exactly as T1 ran it.
*(Corrected in task G, V8-D6: this sentence used to call the same set 27 fragments in its second
half. `ls input/sfspp/mixed_ABG/*.obj` is 24.)* The branch was never
switched, and `sherd_refit/` was not touched — the Python is the frozen parity oracle, and both of
this step's behaviours are algorithm changes, which since `f4466d6` are made in `crates/` alone.
