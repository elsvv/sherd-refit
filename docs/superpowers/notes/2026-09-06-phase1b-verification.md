# Independent verification of phase 1b (steps B1, B2, B3)

**Date:** 2026-09-06. **Branch:** `rust-core`. **Verified range:** `782b372..ec8fc19` — the five
findings-applied commits of step F (`5e6b65e`, `b1bcd0e`, `1956af0`, `a8d84e2`, `6f0e88b`) and the
three phase-1b steps `2ace51a` (B1, segmentation), `ae0314a` (B2, breaklines), `ec8fc19` (B3, sampled match arrays).
**Method:** the implementers' notes were not used as evidence. Every claim below was re-derived from
`sherd_refit/*.py` at the fixture commit, from R
(`docs/superpowers/specs/2026-09-06-algorithm-reference.md`) and from the Rust source, and every
number was re-measured on this machine (Apple M2 Pro, 10 cores, 16 GB; toolchain 1.97.0, release
profile, `cargo clean` first).

**Verdict: the gates do not all pass.** Build, fmt, clippy, the 206 Rust tests, the 54 Python tests
and determinism are green, and **injected parity is exact or near-exact on all six stages and all
seven fixture sets plus the committed slab**. **Native parity fails 43 of 1 478 checks** on three of
the six stages (injected: 3 333 checks, none failed). Those failures are disclosed in D §10.2 and
in the step notes rather than hidden, and the central diagnosis behind them was reproduced here to
the digit — but a failed row is a failed row, so `gates_ok = false`.

---

## 1. Build, lint and tests

| gate | command | result |
|---|---|---|
| clean release build | `cargo clean && cargo build --release --workspace --locked` | **ok**, 37.6 s |
| formatting | `cargo fmt --all --check` | **ok**, no diff |
| lints | `cargo clippy --workspace --all-targets --locked -- -D warnings` | **ok**, no warning |
| tests | `cargo test --workspace --locked` | **ok**, 206 passed, 0 failed, 1 ignored |

Per binary: `sherd-cli` unit 5, `segment_cli` 2 (+1 ignored), `sherd-core` unit 142,
`fragment_cache` 4, `io_open3d_parity` 3, `ply_writer_bytes` 4, `sherd-parity` unit 33,
`slab_fixture` 3, `stages_slab` 5, `working_mesh_slab` 5. Doc-tests: none.

Python package with the fixture sink off (`env -u SHERD_REFIT_FIXTURES python -m pytest -q`):
**54 passed in 74.9 s**.

---

## 2. Parity, every fixture set, both modes, six stages

`sherd-refit-rs parity --fixtures DIR --input DIR --stage {load,thickness,working-mesh,segmentation,breakline,samples} [--injected]`.
`worst/tol` is the largest `deviation / tolerance` over the stage's checks; a value below 1 is inside
the gate.

### 2.1 Injected

| set | load | thickness | working mesh | segmentation | breakline | samples |
|---|---|---|---|---|---|---|
| terracotta | 24/0, 0.00 | 16/0, 0.00 | 32/0, 0.00 | 52/0, 1.7e-6 | 44/0, 4.3e-3 | 40/0, 0.03 |
| pot_A | 48/0, 1.00 | 32/0, 0.00 | 71/0, 1.4e-3 | 104/0, 3.4e-3 | 88/0, 0.05 | 80/0, 0.18 |
| pot_B | 54/0, 1.00 | 36/0, 0.00 | 81/0, 1.7e-3 | 117/0, 1.7e-6 | 99/0, 0.05 | 90/0, 0.29 |
| pot_C | 42/0, 0.50 | 28/0, 0.00 | 63/0, 9.2e-4 | 91/0, 1.7e-6 | 77/0, 0.05 | 70/0, 0.17 |
| pot_G | 42/0, 0.25 | 28/0, 0.00 | 63/0, 1.2e-3 | 91/0, 1.7e-6 | 77/0, 0.07 | 70/0, 0.10 |
| pot_H | 66/0, 0.25 | 44/0, 0.00 | 99/0, 1.0e-3 | 143/0, 1.7e-6 | 121/0, 0.07 | 110/0, 0.12 |
| synthetic_20 | 80/0, 0.00 (20 skipped) | **0 checks, SKIP** (20 skipped) | 140/0, 0.00 | 260/0, 1.0e-3 | 220/0, 0.05 | 200/0, 0.38 |
| slab (committed) | 12/0, 0.00 | 8/0, 0.00 | 18/0, 1.1e-4 | 26/0, 1.7e-6 | 22/0, 4.9e-3 | 20/0, 7.3e-3 |

Cells are `checks/failed, worst/tol`. **No injected check fails anywhere: 3 333 checks, 0 failures.**
The `1.00` in the `load` column of pot_A and pot_B is not a near miss of a numeric gate but the
`V0` check sitting exactly on its allowance — the OBJ vertices agree to **1 `f32` ulp** against a
tolerance of 1 ulp, which is the reader's parse and cannot be tighter. Worst case per quantity
over all 66 fixture fragments:

| stage | quantity | n | worst | tolerance |
|---|---|---|---|---|
| segmentation | agreement, fracture, raw fraction, raw/majority/islands masks | 66 each | **0** (agreement `1.000000000`) | 5e-3 |
| segmentation | votes, smooth radius, growth angle, grid points, rep face | 66 each | **0** | exact / 5e-3 |
| segmentation | NS angle | 66 | 1.708e-6 ° | 1.0 ° |
| segmentation | votes/face | 66 | 1.713e-5 of faces (`Pot_A_Piece_06_Mesh`) | 5e-3 |
| breakline | count, valid, sub count, sub set | 66 each | **0** | exact |
| breakline | points, points in order | 66 | 3.659e-5 (`frag_011`) | 1e-4·t = 8.0e-4 |
| breakline | dihedral | 66 | 2.582e-3 ° | 0.1 ° |
| breakline | ns / nf / f / tangent | 66 each | 2.83 / 2.70 / 2.83 / 4.52 e-6 ° | 0.1 ° |
| samples | n_surface, n_frac, fracture faces, margin count, margin members | 66 each | **0** | exact |
| samples | S on face / Pf on face | 66 | 3.27e-11 / 1.61e-13 | 1e-9·t ≈ 7.8e-9 |
| samples | surface normal / fracture normal | 66 | 0.0292 ° / 0.0381 ° | 0.1 ° |
| samples | sliver samples | 66 | 3.262e-4 (`frag_008`) | 1e-3 |

Two things are worth naming. First, **injected segmentation is bit-exact, not merely inside the
gate**: the area-weighted agreement is `1.000000000` on all 66 fragments and on both slab pieces,
every intermediate mask agrees at 1.0, and the fracture fraction is the reference's to the last
printed digit. Second, the residuals that are non-zero are all the `f32` narrowing of D §4.1 and
nothing else: the breakline point arrays at 3.7e-5 against a gate of 1e-4·t, the frames at 4.5e-6 °,
the normals at 0.038 °. The `votes/face` line is the only place a *different library* shows through —
2 of 7.87 M cone rays disagree between parry3d and Embree (E4) — and the four cleanup passes absorb
it completely (`raw mask` still agrees at 1.0).

### 2.2 Native

| set | load | thickness | working mesh | segmentation | breakline | samples |
|---|---|---|---|---|---|---|
| terracotta | 24/0, 0.00 | 8/0, 0.30 | 16/0, 0.95 | 8/0, 0.41 | **12/1, 1.31** | 24/0, 0.27 |
| pot_A | 48/0, 1.00 | 16/0, 0.95 | 32/0, 1.8e-4 | 16/0, 0.27 | **24/1, 1.29** | **48/1, 1.18** |
| pot_B | 54/0, 1.00 | 18/0, 0.91 | 36/0, 4.8e-6 | **18/2, 4.08** | **27/7, 14.95** | **54/3, 14.28** |
| pot_C | 42/0, 0.50 | 14/0, 0.19 | 28/0, 1.3e-5 | 14/0, 0.22 | 21/0, 0.42 | 42/0, 0.50 |
| pot_G | 42/0, 0.25 | 14/0, 0.36 | 28/0, 6.6e-6 | 14/0, 0.56 | **21/3, 3.80** | **42/1, 1.89** |
| pot_H | 66/0, 0.25 | 22/0, 0.70 | 44/0, 8.8e-6 | 22/0, 0.49 | 33/0, 0.76 | 66/0, 0.49 |
| synthetic_20 | 80/0, 0.00 (20 skipped) | 40/0, 0.73 | 80/0, 0.94 | **40/2, 1.27** | **60/21, 2.71** | **120/1, 1.31** |
| slab (committed) | 12/0, 0.00 | 4/0, 0.14 | 8/0, 5.7e-7 | 4/0, 0.07 | 6/0, 0.65 | 12/0, 0.47 |

**43 of 1 478 native checks fail** (excluding the slab, which passes everything):
segmentation **4 of 132**, breakline **33 of 198**, samples **6 of 396**. Load, thickness and
working mesh pass everywhere. Broken down by quantity:

| stage | quantity | failures | where |
|---|---|---|---|
| segmentation | agreement, fracture | 2 + 2 | `Pot_B_Piece_01_Mesh` (0.9142, +8.17 pp), `frag_010` (0.9687, −2.55 pp) |
| breakline | count | 13 | synthetic_20 only: frag 000, 002, 003, 005, 006, 007, 011, 013, 015, 016, 017, 018, 019, 10.1–19.7 % low |
| breakline | p99 distance | 15 | pot_B ×7, pot_G ×3, synthetic_20 ×5 |
| breakline | dihedral KS | 5 | `FY234007_reduced`, `Pot_A_Piece_04_Mesh`, frag 010, 014, 017 |
| samples | n_frac | 1 | `Pot_A_Piece_04_Mesh`, 11.76 % against a 10 % gate |
| samples | fracture fraction | 2 | `Pot_B_Piece_01_Mesh` (+8.18 pp), `frag_010` (+2.63 pp) |
| samples | margin fraction | 1 | `Pot_B_Piece_01_Mesh` (−6.63 pp) |
| samples | Pf spacing | 2 | `Pot_B_Piece_01_Mesh` (14.3×), `Pot_G_Piece_05_Mesh_DS` (1.89×) |

These counts match D §10.2's own text exactly (33 of 198 breakline, 390 of 396 samples, 66 of 68
fragments on the segmentation row counting the two slab pieces). The reporting is accurate.

### 2.3 The central excuse, re-derived rather than believed

D §10.2 blames the 15 `p99 distance` failures on the segmentation upstream: "3 115 of 271 592
point-to-set distances exceed `0.5 t`, and on every fragment whose working mesh is identical to the
reference's each of them lies within `0.169 t` of a face the two segmentations label differently."
I re-computed that from scratch — the port's `brk_P` read straight out of the `.sherd` caches with a
30-line safetensors reader, the reference's from the fixture `.npy`, symmetric nearest-neighbour
distances through `scipy.spatial.cKDTree`:

```
total point-to-set distances 271592, beyond 0.5 t: 3115
```

and on every fragment whose face list is equal to the reference's entry for entry, **every far point
lies within `0.16946 t` of a face the two label differently** — the largest such distance over all
fragments, on `Pot_G_Piece_04_Mesh_DS`, which is the figure D quotes rounded down to `0.169 t`.
The count reproduces exactly and the radius to five digits. The diagnosis is sound.

The correlation is visible without the diagnosis too: `Pot_C_Piece_06/07`, `Pot_G_Piece_07`,
`Pot_H_Piece_01/05/06/10/11` have **label arrays identical to the reference's** and a p99 of exactly
0; every fragment with a non-zero p99 has at least one differing label. What the failure really
says is that the `p99 distance` row is **not independent of the segmentation row**: `Pot_B_Piece_02`
passes segmentation at 0.9970 agreement and still puts 118 of its 4 087 breakline distances beyond
`0.5 t`, because a 0.3 % area disagreement on a *surface* is a whole extra loop on a *curve*. The
same non-independence is what D §10.2 already argues for `Pf spacing`. Two of the three failing
native rows are therefore one failure, counted three times.

The 13 `count` failures are a different and self-inflicted one, and D §10.2 says so: `res` is allowed
±10 % natively, the port's decimator lands ~15 % coarse on synthetic_20's median breakline spacing,
and a coarser mesh crosses the same curve with proportionally fewer edges. A ±10 % count gate under
a ±10 % `res` gate has no headroom by construction.

---

## 3. Semantic review of the diff against R and the Python

Every item the task names was re-derived line by line against `sherd_refit/fragment.py` and
`sherd_refit/geometry.py` at the fixture commit. **No unmarked semantic deviation was found in any of
them.** What follows is the audit, not a summary of the notes.

| item | reference | port | verdict |
|---|---|---|---|
| grid bucket rule | `voxel_down_sample_and_trace(spacing, C.min−1, C.max+1)`, voxel `⌊(p−min)/s⌋`, `l[0]` = lowest index in the voxel | `segment.rs:217-259`: same floor, same `min−1`, run-first of a `(voxel, index)` sort, then `rep` sorted ascending | exact; the ordering is PMC-4, and injected `rep face` = 0 and `sub set` = 0 on all 66 fragments confirm the *set* and the *map* are the reference's |
| ray origin | `C − FN·1e-3` in `f64`, narrowed once with the direction | `segment.rs:338-342`, `f64` then `as f32` | exact; the port keeps the reference's absolute `1e-3` and does **not** take PMC-1's option |
| ray directions | `d_0 = −n`; `d_k = −cos15°·n + sin15°(cosφ e1 + sinφ e2)`, `φ = 2π(k−1)/6`, `e1 = normalise(n×a)`, `a = [1,0,0]` unless `\|n_x\| ≥ 0.9` | `segment.rs:344-357`, `cone_basis` at `:383-395` | exact, including the `a` switch and `e2 = n × e1` |
| dtype of the window | `dh` is `f32`; NEP 50 makes `0.1·t`, `t`, `0.5`, `1.8` weak, so `dh > 0.1t` and both `dh/t` tests run in `f32`; `hit_normals·d` runs in `f64` | `segment.rs:328-368`: `min_hit`/`thick_f32` cast once, ratio and window in `f32`, `al` in `f64` | exact; R gained this paragraph in B1 and the paragraph is right |
| vote threshold | `votes = 4 if res > 0.1·t else 5`; `shell = good ≥ votes` | `segment.rs:546,549` (`frac = good < votes`) | exact |
| cleanup order | majority (`t/4`) → drop(frac, 0.5t²) → drop(shell, 2.0t²) → grow(25°, ≤60 passes) → drop(frac, 0.5t²) | `segment.rs:560-615`, same order, same constants | exact; `0.5·t·t` and `2.0·t·t` are exact rescalings of `0.5·t**2` because both factors are powers of two |
| boundary growth | candidates = the fracture side of every mixed adjacency, `np.unique`; flip on `FN·ref > cos25°` with a **fixed** `ref`; simultaneous assignment | `segment.rs:438-479`, sorted candidate list, `ref` computed once before the loop | exact; writing `frac[f]` inside the pass is safe because the predicate reads only `has_ref`, `FN`, `ref` |
| annulus radii | `dist ≥ 0.15·t` (inclusive), ball `≤ 0.60·t` (inclusive), fallback over the whole mask at `0.60·t` | `breakline.rs:287-311`; `PointTree::within` is `d ≤ r`, matching `query_ball_point` | exact |
| macro-normal masks | `ns` over `¬frac`, `nf` over `frac`; zero vector when the mask or the point set is empty | `breakline.rs:232-233,292-294` | exact |
| frame | `f = nf − (nf·ns)ns`, `/max(\|f\|,1e-9)`; `valid = \|ns\|>0.5 ∧ \|nf\|>0.5 ∧ \|ns×f\|>0.5` | `breakline.rs:235-254` | exact in `f64` (see defect **D3** for the `f32` accessor of the same name) |
| subsample voxel | `brk_voxel·t` with `brk_voxel = 0.5` — the code's own comment says `t/3` and is stale | `breakline.rs:257`, `BRK_VOXEL = 0.5` | exact; R was corrected this phase and the correction follows the code, which is the right direction |
| sample count rules | `n_surface = 20000` unconditionally; `n_frac = int(clip(150·A_f/t², 5000, 12000))`; `margin_points = 6000` | `samples.rs:209,241-254,229` | exact — same grouping `(150·A)/(t·t)`, same truncation after the clamp; the only spelling difference is `t**2` against `t*t`, which agrees bit for bit on every libm that special-cases an integer exponent of two. Injected `n_frac` is bit-exact on all 66 fragments, 32 of which sit strictly between the clamps |
| margin band | `¬frac[sp] ∧ 0.12t < d_brk < 1.5t`, both strict, no `res` floor (PMC-5), `d_brk = ∞` with no breakline | `samples.rs:365-376` | exact; injected `margin count` and `margin members` are both exact on all 66 |
| `sample_on_faces` | `p = A/ΣA` (pairwise `Σ`), sequential `cumsum`, `/cdf[-1]`, `searchsorted(...,'right')`; then `u`, then `v`; fold `u+v>1 → (1−u,1−v)`; `V0 + u(V1−V0) + v(V2−V0)` | `samples.rs:272-342`, `partition_point(c ≤ u)` | exact in structure and association; **but see defect D6 — nothing in the parity harness tests it** |
| `pc_reg` split | `nf' = max(1, round(nf·R/(nf+nm)))` (round-half-even), `nm' = max(0, R−nf')`, prefixes | `samples.rs:527-546` | exact; R's `max(0, …)` was added this phase and matches the code |
| RNG use | one `default_rng(p.seed)` per fragment, consumed by the three samplers in order; `_subsample` draws nothing when the margin already fits | `rng.rs:41-98`, `samples.rs:207,218,228`: four `ChaCha8Rng` streams, seed XOR a per-draw FNV-1a tag, numpy's `(word>>11)·2⁻⁵³` | **deviation, PMC-9** — see defect **D2**. Everything *inside* each sampler is the reference's order, and `subsample` still draws nothing on a small margin |
| thickness seed | `rng_pre = rng(0)`, hard-coded | `mod.rs:129` `seeded(thickness::SEED)`, `SEED = 0`, `Draw::Thickness.tag() = 0` | exact |
| `face_adjacency` | `[F[:,01]; F[:,12]; F[:,20]]` slot-major, stable lexsort, consecutive equal pairs | `adjacency.rs:44-51,138-156`, sorted by `(key, position)` | exact |
| island removal | components over adjacency edges whose *both* faces are in the set; isolated faces are their own component; `size < min_area` strict | `components.rs:135-191` | exact |
| `res`, `ΣA`, `face_geometry` before the narrowing | `f64` | `mod.rs:127-128,169`; per-face arrays for §3.4–3.5 re-derived from the **narrowed** vertices | exact under **PMC-15**, which spells this split out |

Two smaller checks, both clean: `raw_fraction` is taken before the majority filter as the reference
takes it (`segment.rs:550`), and `Fragment::load_or_build` rebuilds *both* halves of the match arrays
when either is stale (`mod.rs:250-265`), which is faithful because the reference's `match_arrays`
computes them in one call.

---

## 4. Defect list

`file:line` — what R says — what the code does.

### D1 — native parity is red on three of six stages (gate failure)

**Where:** the whole native column of §2.2. **R/D says:** D §10.2 states the six rows and says three
times, in as many words, that none of them is being widened. **Code does:** 43 of 1 478 native
checks fail. Nothing here is hidden — D §10.2 and the three step notes name every failing fragment and give
a diagnosis for each — and §2.3 above confirms the diagnosis. But the phase closes with three rows
red, and the task's own gate ("parity in both modes on all fixtures for all six stages") is not met.
The right next move is the one D already argues for: **E6, replicating PCG64**, which removes the `t`
divergence that feeds the segmentation row, which feeds the breakline and samples rows. Until then
the native column measures the sampler, not the port.

### D2 — PMC-9 was rewritten inside the frozen reference to legalise the port's stream split

**Where:** `docs/superpowers/specs/2026-09-06-algorithm-reference.md` §12, PMC-9 row;
`crates/sherd-core/src/rng.rs:41-98`. **R said at `9d4b9d3`:** "numpy PCG64 sampling | library |
portable RNG, **same draw structure** | injected-sample parity + statistical gates". **R says now:**
"… one stream per draw site rather than one per fragment, so that a rebuild at another `t` or
`surface_points` (§4.2, §8) cannot move a sampler that did not change …". **Code does:**
`seeded_for(seed, draw) = ChaCha8Rng::seed_from_u64(seed ^ draw.tag())` with four independent tagged
streams per fragment.

The engineering argument is good — a shared stream really does make the fracture samples a function
of `surface_points`, and R §4.2 and §8 do rebuild the arrays at another `t` — and the change is
declared in R, in D §10.2 and in the B3 note rather than smuggled. Two things are still owed. First,
R's own preamble says "every PMC change must be re-verified against the parity gates in §13", and
§13 is the pair and assembly gate set, which phase 1b cannot run; so PMC-9 is now *widened but not
yet re-verified by the thing that is supposed to re-verify it*. Second, the amendment is undated and
unattributed inside a document whose header calls itself frozen at `9d4b9d3`; a reader diffing R
against the Python will not see which clause is the reference's and which is the port's. Both are
cheap to fix: a dated marginal note in the PMC-9 row, and an explicit entry on the phase-1c risk list.

### D3 — `Breaklines::valid()` and the mask that filters `brk_sub` are two different predicates

**Where:** `crates/sherd-core/src/fragment/breakline.rs:179-188` against `:248-254` and `:257-260`.
**R §3.5.4 says:** one mask, `valid = |ns| > 0.5 ∧ |nf| > 0.5 ∧ |ns × f| > 0.5`, computed from the
`f64` macro normals, and `brk_sub = sub[valid[sub]]`. **Code does:** `build_with` computes `valid`
from the `f64` arrays (right) and filters `sub` with it; the public accessor `Breaklines::valid()`
recomputes the same formula from the **`f32`-narrowed** `ns`, `nf` and `f`. For a point whose
`|ns|`, `|nf|` or `|ns × f|` sits within an `f32` ulp of 0.5 the two disagree, and then a point that
`sub` contains is one that `valid()` calls invalid.

No effect today: `valid()` feeds a `tracing::info!` count (`mod.rs:415`) and the parity harness
(`stages/breakline.rs:124`), and the harness's `valid` check is exact on all 66 fragments, so the two
predicates agree on every fixture. It is a trap for phase 1c, where R §5.1 pairs breakline points:
the hypotheses must be built from `sub`, never re-filtered through `valid()`. Either document
`valid()` as a diagnostic only, or store the mask beside `sub` in the cache.

### D4 — the `near` tie rule of R §3.4.1 is an unlisted deviation

**Where:** `crates/sherd-core/src/spatial/kdtree.rs:49-51`, used by `segment.rs:196-199`.
**R §3.4.1 says:** "`near[i]` = index into `rep` of the representative whose centroid is nearest to
`C[i]` (KD-tree)" — and says nothing about a face equidistant from two representatives. **Code
does:** `kiddo`'s `nearest_one`, whose tie behaviour the module's own doc calls undocumented ("kiddo
resolves exact ties to the lowest index on every case E3 could construct, though it does not document
that as a guarantee"); the reference's `scipy.spatial.cKDTree.query` has its own, also unspecified.
`near` selects `NS`, the majority vote and the shell reference normal, so a tie that resolves the
other way moves a face's smoothed normal to another representative's.

Measured harm: none — the injected `rep face` check is **exactly 0** on all 66 fragments and both
slab pieces, so no tie occurs on any fixture. But this is a tie rule on a result path that R does not
pin and that PMC-4 does not cover (PMC-4 is about the *order of `rep`*, not ties in `near`), and it
will bite the first symmetric synthetic mesh. It belongs in R §12 as a PMC entry, with the harness's
existing 0.005 `rep face` gate named as its re-verification.

### D5 — the ray-cast library substitution is not in R §12 either

**Where:** `crates/sherd-core/src/spatial/bvh.rs:37-59`. **R §3.2 and §3.4.3 say:** the first hit
comes from Open3D's `RaycastingScene` — Embree, `f32`. **Code does:** `parry3d` 0.30
(`CompositeShapeRef::cast_local_ray`), which E4 measured at 2 hit/miss disagreements and 5 differing
primitive ids over 7.87 M cone rays. The effect is visible in the parity table — `votes/face` differs
on up to 1.7e-5 of the faces of `Pot_A_Piece_06_Mesh` — and is fully absorbed by the cleanup
(`raw mask` agreement 1.0). PMC-7 covers only the *inside test*; the ray cast itself is a library
substitution that R does not list. One more PMC row, with the injected `votes/face` gate as its
re-verification.

### D6 — injected mode never exercises the port's own sampler

**Where:** `crates/sherd-parity/src/stages/samples.rs:120-190`. **R §3.5.1 says:** three constructions
the port had to re-derive from `sherd_refit/geometry.py:119-130` — the `p = A/ΣA`, `cumsum`,
`/cdf[-1]`, `searchsorted(..., 'right')` face pick; the `u + v > 1 → (1−u, 1−v)` fold; and
`V0 + u(V1−V0) + v(V2−V0)`. **Code does:** injected mode compares `n_surface`, `n_frac`, the
`fp`-on-fracture test, the margin count and membership, and measures **the reference's** points
against the port's geometry (`S on face`, `Pf on face`). Those pin the *index convention* and the
*count rule*; they do not touch the port's own construction, which is covered only by the unit tests
in `samples.rs`. A port that folded the wrong way, or used `side='left'`, would pass every injected
check in the table.

PMC-9 makes a direct comparison impossible today because the dump carries no uniforms — but that is
a fixture limitation, not a law: dumping `pick`'s uniforms, `u` and `v` beside `md.S` would make
`sample_on_faces` injectable exactly, and it is the cheapest remaining increase in coverage on this
stage. It belongs in the phase-1c plan.

### D7 — injected thickness has no coverage at all on synthetic_20

**Where:** `crates/sherd-parity/src/stages/thickness.rs:56-71` (the F3 guard) and the `slim` dump.
**Result:** the injected thickness stage reports `0 checks, SKIP` with all 20 fragments skipped,
because the dump has no `load.V0` and F3 correctly refuses to run "injected" mode on a mesh the port
reconstructed. That is the right call. Its consequence should be stated where the claim is made:
D §10.2's "the injected row is untouched and is met bit-exactly" rests on **46 of the 66** fixture
fragments plus the slab's two, not on all of them. Either re-dump synthetic_20 at level `full`, or
say so in D §10.2.

### D8 — D §10.2 describes a segmentation measurement the harness does not make

**Where:** `crates/sherd-parity/src/stages/segmentation.rs:226-240` against D §10.2's preamble.
**D says:** "Segmentation agreement is measured by sampling 200 000 area-weighted points on the
Python working mesh and labelling each by its nearest face on each mesh." **Code does:** one
quadrature point per *reference face* — its centroid — weighted by that face's area, labelled on the
reference side by the face's **own** label (not by a nearest-face lookup) and on the port's side by
`RayScene::closest_face`. The module documents the substitution and it is the better estimator (no
RNG in the comparison, exact on the reference side, one point per face rather than a Monte-Carlo
draw), but it is not the same estimator on a mesh with a wide face-area spread, and D §10.2 still
describes the old one. Correct the sentence in D rather than the code.

**Not a defect, checked and cleared:** `res` is computed by `median_edge` on the pre-narrowing `f64`
vertices (`mod.rs:169`) while the per-face arrays come from the narrowed ones (`mod.rs:198-199`).
That looks like an inconsistency and is not: **PMC-15** prescribes exactly this split — "everything up
to the narrowing — Taubin, `face_geometry`, `median_edge`, `ΣA` — stays float64".

---

## 5. Determinism

Two `segment` runs per collection, cold cache each time, `shasum -a 256` over every `.sherd`:

| collection | fragments | run 1 vs run 2 | `--threads 1` vs `--threads 4` | `--threads 1` vs default |
|---|---|---|---|---|
| terracotta | 4 | **byte-identical** | **byte-identical** | **byte-identical** |
| pot_H | 11 | **byte-identical** | — | — |

`cmp` reports no difference on any of the 15 files, and the digests match pairwise. The thread-count
run is the extra one D §10.4 layer 4 asks for and phase 1b now has a reason to want: the segmentation
adds two `par_iter` ray/ball loops and the samples add a third, and all three are indexed collects,
so the schedule cannot reach the result. Confirmed rather than assumed.

---

## 6. Timing against the Python

Warm cache means the `.sherd` files are already on disk; the Python numbers are its own
`ProcessPoolExecutor` preprocessing (`pipeline.segment_only` minus the preview render, which the port
does not do yet), 9 worker processes, and they include pool start-up.

| collection | Rust cold, wall | Rust cold, CPU work | Rust warm | Python, wall (9 procs) | speed-up |
|---|---|---|---|---|---|
| terracotta (4 fragments, 0.7–1.3 M faces each) | **1.94–1.99 s** | 6.19 s | 0.01 s | 16.75–16.94 s | **8.5×** |
| synthetic_20 (20 fragments) | **6.71 s** | 73.57 s | 0.02 s | 24.78 s | **3.7×** |

Single fragment, one process, all cores available to it — this removes the pool start-up from the
Python side:

| fragment | Python `from_mesh_file` + `match_arrays` | Rust `segment --no-cache` | speed-up |
|---|---|---|---|
| `FY234007_reduced` (1.23 M → 148 k faces) | 11.50 s + 0.43 s = **11.93 s** | **1.41 s** | 8.5× |
| `frag_000` (630 k → 200 k faces) | 4.77 s + 0.32 s = **5.10 s** | **0.92 s** | 5.5× |

The narrower margin on synthetic_20 is the shape of the set, not a regression: 20 small fragments
give the Python's nine-process pool almost perfect parallelism, and its start-up is amortised over
five times as many fragments as terracotta's. The per-fragment table above is the fairer comparison
and it holds at 5.5–8.5× on both shapes. Against D §10.3's whole-run CPU budgets — 25 s for
terracotta, 120 s for synthetic 20 — preprocessing now costs 2.0 s and 6.7 s cold and effectively
nothing warm, so it is not the thing that will spend those budgets.

---

## 7. What I would ask for before phase 1c

1. **E6.** Three of the six native rows are red and all three trace back, through §2.3, to `t` — that
   is, to PMC-9's sample. Every phase-1c gate in R §13 is an exact-set gate on pairs, and a 6.6 % `t`
   moves all nine thresholds of R §1.2 for every pair a fragment takes part in. The port cannot be
   said to reproduce the reference's *decisions* until the estimator draws the same sample.
2. **Three PMC entries** — a dated marginal note on the amended PMC-9 (D2), plus new rows for D4
   (`near` ties) and D5 (the BVH substitution). R §12 is the port's licence; anything the port does differently that is
   not on it is an undeclared deviation, however small its measured effect.
3. **D6's fixture change**, which is the cheapest untested surface left in R §3: three arrays in the
   dump and `sample_on_faces` becomes injectable exactly.
4. **D3's one-line decision** before the hypotheses of R §5.1 are written.
