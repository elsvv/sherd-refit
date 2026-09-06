# F — the phase-1a verification's findings, applied

**Date:** 2026-09-06. Branch `rust-core`, from `782b372` (step V). Plan step F of
`docs/superpowers/plans/2026-09-06-rust-core-phase0-1a.md`. Inputs: the verification note
`2026-09-06-phase1a-verification.md` (findings F1–F12) and the open issues of the S1/S3/S4 notes.
**R** = `specs/2026-09-06-algorithm-reference.md`, **D** = `specs/2026-09-06-rust-core-design.md`.

**Every claim below was re-measured here, not carried over.** Where a finding said "about 1 %" or
"E2 measured it exact", the number was taken again against Open3D 0.19 and two of them turned out
to be wrong in a way that mattered.

---

## 1. What each finding cost, and what closed it

| # | decision | what it turned into | commit |
|---|---|---|---|
| F1 | accept `max(2 %, 3 bins)` | D §10.2's native thickness row, with the seed sweep behind it; R §3.2 and R §13 carry the propagation risk | `a8d84e2` |
| F2 | 0 and absent both mean the adaptive budget | `manifest::Collection::face_cap`, `NO_CAP`, the dead `unwrap_or(200_000)` gone, 2 tests | `1956af0` |
| F3 | no silent recompute; skip with a reason | `dumped_original` vs `original` + `OriginalSource`, injected thickness skips, 1 test | `1956af0` |
| F4 | correct R | R §3.3.1's weight is `1/(dist + 1e-12)` | `a8d84e2` |
| F5 | record the `f32` narrowing | R §12 PMC-15; D §4.1 spells out what it buys | `a8d84e2` |
| F6 | implement if under an hour | convex-hull PCA in `obb_min_extent`, 40 minutes, 1 test | `b1bcd0e` |
| F7 | measure Open3D, then match it | `quantize_color` rounds; the **GLB reader was actually wrong**; 3 tests | `5e6b65e` |
| F8 | state it in R | R §3.3.1: normals and colours are not smoothed, and the cache's colours are unsmoothed | `a8d84e2` |
| F9 | README | what S2–S4 actually built, with the parity numbers | `a8d84e2` |
| F10 | fix the comment | `sherd_core::fixture` says phase 1d, and why | `a8d84e2` |
| F11 | D follows the code | D §4.1 (`f64` thickness, `face_budget`, `area0`, the `f32` `WorkingMesh`), D §4.2 (one metadata key, no `created`) | `a8d84e2` |
| F12 | cosmetic | `Check::relative` changes its *unit* when it changes its meaning | `1956af0` |
| S1 open 2 | sync D §3 | the dependency table is the workspace now, with an honest "not in the workspace yet" | `a8d84e2` |
| S3/S4 open | D §10.2, D §4.2 | folded into F1 and F11 | `a8d84e2` |

## 2. Three things measured, two of which changed a verdict

### 2.1 F6 — the OBB fallback was 8 % out, not 1 %

The code's comment put the difference between Open3D's OBB (PCA over the convex hull's vertices)
and the port's (PCA over every vertex) at "about 1 %, measured on a blob with a thin arm". Measured
here, on real meshes, with `min(extent)` — the only number R §3.2 uses:

| mesh | Open3D `get_oriented_bounding_box()` | PCA over all vertices | error |
|---|---:|---:|---:|
| `fixtures/slab/input/pieceA.ply` (22 552 v) | 42.0619 | 40.8516 | **−3.0 %** |
| `fixtures/slab/input/pieceB.ply` (17 952 v) | 41.7494 | 38.3579 | **−8.1 %** |
| `FY234007_reduced.ply` (terracotta) | 93.803 | 102.075 | **+8.8 %** |

I also confirmed the mechanism rather than assuming it: projecting *all* the points onto the
hull-PCA axes reproduces Open3D's extents to the last digit (`[93.80312397, 427.87832032,
561.00273636]` against Open3D's, identically sorted), while the axes are what differ. So the
extents may be taken over either set; only the covariance has to be the hull's.

`t` is the unit of every threshold in R §1.2, so 8 % here is 8 % on nine thresholds of every pair
that fragment takes part in. That is worth an hour. The hull is
`parry3d::transformation::try_convex_hull`, already in the tree for the rays; it is `f32`, and the
PCA over its `f32` vertices costs **1.6e-9** relative on pieceA and **1.0e-8** on pieceB against
Open3D's `f64` Qhull. Both are pinned in a test.

Still unreachable on the benchmark — the worst of the 68 fixture fragments has 7 154 valid ray hits
of 20 000 against a threshold of 100 — so this changes no parity number. It changes what happens
the first time a fragment is scanned badly enough to need it.

### 2.2 F7 — Open3D rounds, and the GLB reader was truncating

The question the finding asked was "round or truncate". Answer, measured: an OBJ carrying the
colours below, read with `o3d.io.read_triangle_mesh` and written straight back with
`write_triangle_mesh(..., write_ascii=False)`:

| colour in the file | reaches the writer as (f32) | ×255 | Open3D's byte |
|---|---:|---:|---:|
| `0.5/255` | 0.00196078442968428 | 0.500000030 | **1** |
| `1.5/255` | 0.00588235305622220 | 1.500000029 | **2** |
| `2.5/255` | 0.00980392191559076 | 2.500000088 | **3** |
| `0.5` | 0.5 | 127.5 | **128** |
| `254.5/255` | 0.99803918600082397 | 254.499992 | **254** |
| `0.3` | 0.30000001192092896 | 76.500003 | **77** |
| `1.2`, `−0.1` (a second run, PLY doubles) | — | 306, −25.5 | **255**, **0** |

Every half lands up, so it is not truncation and not round-half-to-even; the last row confirms both
clamps. `254.5/255` comes back 254 because it is 254.49999 by the time it is multiplied — the
arithmetic, not the rule. `quantize_color` now calls `round` rather than `floor(x + 0.5)`; the two
differ only on the doubles immediately below a half (at `x = 0.49999999999999994`, `x + 0.5` itself
rounds up to 1.0), and `round` is what Open3D calls.

**The finding under-stated the problem.** The PLY and OBJ readers were already right. The **GLB**
reader was not: `gltf`'s `Normalize<u8> for f32` is `(self.max(0.0) * 255.0) as u8`, a truncation,
so `into_rgba_u8()` read a float `COLOR_0` one byte low wherever the value was not exactly `k/255`
— 0.3 became 76 where Open3D writes 77. E2's measurement was not wrong, it was narrow: I checked
all 256 exact `k/255` values as `f32` and truncation and rounding agree on every one of them, which
is what an exporter writes when it converts from bytes. Colours now come through
`into_rgba_f32()` and the same `quantize_color` as the other three readers, which also rescales a
`u16` `COLOR_0` by `255/65535` instead of `gltf`'s `>> 8`.

Latent either way — nothing in phase 1a compares an output mesh's colours end to end, and GLB is
outside R §2's discovery list — but it is a wrong byte, and it was one line.

### 2.3 F2 — the sentinel the doc promised did not exist, and 0 was destructive

Reproduced before fixing: copy `fixtures/slab/dump`, set `collection.target_faces = 0`, run native
parity — 6 of 24 comparisons fail at deviation 1.000, because numpy's `clip` applies the floor
first, so `clip(raw, 50000, 0)` is 0 and every fragment decimates to nothing. A manifest with no
`target_faces` key at all reaches the same 0 through `#[serde(default)]`.

`Collection::face_cap` now names the two readings there are. 0 and absent are both `NO_CAP`
(`u32::MAX`, so that it survives the round trip through `Fragment::target_faces`, which is a `u32`
because it is part of the cache key). The `usize::try_from(...).unwrap_or(200_000)` that could not
fire is gone, and a hidden default of 200 000 was rejected on purpose: it would be a *guess* at
what the reference ran with, and silently wrong for a dump made at any other cap.

The test that matters is not the unit test but the end-to-end one: a throwaway dump built from the
slab's manifest **with the key removed** still passes the native working-mesh row, 8 checks, 0
failures — because on this collection the adaptive budget is below 200 000 anyway, so removing the
cap changes no answer. That is the evidence that the new reading is right and not merely different.

## 3. F3 — what "injected" is allowed to mean

`FragmentFixture::original()` recomputed `(V0, F0)` from the source file when the dump did not carry
them, justified in a comment by "the `load` stage compares the two and the comparison is exact".
That holds at the `full` level only. `synthetic_20` is `slim`: its load stage skips the array
comparison for all 20 fragments, so the injected thickness stage was running R §3.2's `> 0.7`
normal filter on the **port's own** `(V0, F0)` and reporting the result in the injected column.

There are now two accessors. `dumped_original()` returns the reference's arrays or nothing, and is
the only one an injected comparison may use; `original()` still falls back but says which of the two
it returned (`OriginalSource::{Dump, Recomputed}`). Injected thickness skips a fragment whose `V0`
is not in the dump; the working-mesh stage takes the dumped arrays only (its two affected checks
were already guarded, so nothing there changed); and the load stage's own skip names the
consequence instead of just the missing file.

**The visible cost is 80 injected comparisons, and it is a cost worth paying.** They were the 80
that could not have failed for the reason the injected column exists to test. The stage now reports
`SKIP` on `synthetic_20 / injected / thickness` instead of `PASS`, which is the honest word for it.

## 4. Parity after all of it

`sherd-refit-rs parity --fixtures DIR --input SRC --stage all [--injected]`, release build, the
same eight collections as step V.

| set | frags | mode | load | thickness | working mesh |
|---|---:|---|---|---|---|
| terracotta | 4 | injected | 24 / 0.00 | 16 / 0.00 | 32 / 0.00 |
| | | native | 24 / 0.00 | 8 / 0.30 | 16 / 0.95 |
| pot_A | 8 | injected | 48 / 1.00 | 32 / 0.00 | 71 / 1.4e-3 |
| | | native | 48 / 1.00 | 16 / 0.95 | 32 / 1.8e-4 |
| pot_B | 9 | injected | 54 / 1.00 | 36 / 0.00 | 81 / 1.7e-3 |
| | | native | 54 / 1.00 | 18 / 0.91 | 36 / 4.8e-6 |
| pot_C | 7 | injected | 42 / 0.50 | 28 / 0.00 | 63 / 9.2e-4 |
| | | native | 42 / 0.50 | 14 / 0.19 | 28 / 1.3e-5 |
| pot_G | 7 | injected | 42 / 0.25 | 28 / 0.00 | 63 / 1.2e-3 |
| | | native | 42 / 0.25 | 14 / 0.36 | 28 / 6.6e-6 |
| pot_H | 11 | injected | 66 / 0.25 | 44 / 0.00 | 99 / 1.0e-3 |
| | | native | 66 / 0.25 | 22 / 0.70 | 44 / 8.8e-6 |
| synthetic 20 | 20 | injected | 80 / 0.00 (20 skipped) | **0 (20 skipped, SKIP)** | 140 / 0.00 |
| | | native | 80 / 0.00 (20 skipped) | 40 / 0.73 | 80 / 0.94 |
| slab | 2 | injected | 12 / 0.00 | 8 / 0.00 | 18 / 1.1e-4 |
| | | native | 12 / 0.00 | 4 / 0.14 | 8 / 5.7e-7 |
| **total** | **68** | injected | **1127**, 0 failed, 40 skipped |
| | | native | **776**, 0 failed, 20 skipped |

**Every deviation/tolerance ratio is identical to step V's**, down to the last digit printed —
`load` at exactly 1.00 ULP on the OBJ sets, `res` at 0.95 on `Pot_A_Piece_01_Mesh` and 0.94 on
`frag_002`, `t` at 0.91 and 0.95 of the widened gate. Nothing regressed; the only difference in the
whole table is the one bold cell, which is F3 telling the truth.

## 5. Gates

| gate | result |
|---|---|
| `cargo fmt --all --check` | PASS |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | PASS, clean |
| `cargo test --workspace --locked` | PASS — **160 passed**, 0 failed, 1 ignored (152 at step V; +8 here) |
| `cargo test -p sherd-cli -- --ignored` | PASS — `two_runs_on_the_terracotta_produce_byte_identical_caches`, 12.6 s |
| determinism, default pool vs `--threads 1` | PASS — 4/4 caches byte-identical |
| parity, all 8 collections, both modes | PASS — 1127 + 776 comparisons, 0 failed |

The eight new tests: the eight OBJ colour bytes Open3D writes, a hand-built float-colour GLB
(76/127/2 under the old path, 77/128/3 under Open3D's rule), the round-vs-floor edge case, both
slab pieces' OBB extents against Open3D's own numbers, `face_cap`'s two readings, what the uncapped
budget means for `face_budget`, a manifest with no `target_faces` whose native row still passes, and
a `slim` dump whose injected thickness skips.

## 6. What phase 1b inherits

* **The `t` risk is now written down in three places** (R §3.2, R §13, D §10.2) and it is the same
  risk: a fragment whose `t` differs by 6.6 % has nine thresholds shifted by 6.6 % for every pair it
  is in, and R §13's gates are exact-set gates. E6 (replicating PCG64, ≈ 2 days) is the only lever
  if they ever fail for this reason — it is not worth doing before they do.
* **`res` at 0.95 of its gate is still the tightest number in the phase**, and one-sided. Nothing
  in this step moved it. Step V's advice stands: measure R §3.4's segmentation agreement early in
  1b, not at the end of it.
* **`hull_or_all` and `principal_axes` are in `fragment::thickness`** and are not really thickness
  code. If anything else ever wants a convex hull or a PCA, they belong in `crate::spatial` and
  `crate::geometry` respectively.
* **A new stage still costs one module plus one line in `Stage::ALL`** — and now also a decision
  about what it may read: `dumped_original()` in injected mode, `original()` only where native mode
  would have read the file anyway.
