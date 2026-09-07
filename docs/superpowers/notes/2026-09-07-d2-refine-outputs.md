# Step D2 — R §9's full-resolution refinement and the whole of R §11

*2026-09-07. Phase 1d, step D2. Branch `rust-core`.*

R §9 and R §11 are the last two stages of the pipeline and the last two rows of D §10.2's table.
This step implements them — `crates/sherd-core/src/{refine,report,render}.rs` — adds the parity
stages `refine` and `outputs` that judge them, and runs both on all eight fixture sets. The port
now reproduces the reference's refined poses to **1e-15 t** where the inputs are the same,
`placed/<name>.ply` **byte for byte** on every collection the two implementations read the same
way, and R §11.5's previews **pixel for pixel — 0 differing pixels of 31 080 000**.

The headline is not that it passes. It is what it took to make an image and a binary mesh
reproducible at all, which is three arithmetic facts nobody had written down (§2), and one thing
that cannot be reproduced and is now a PMC (§7).

---

## 1. What was built

| file | what it is |
|---|---|
| `crates/sherd-core/src/refine.rs` | R §9: `vertex_normals` (Open3D's `ComputeVertexNormals`), `select_candidates` + `cap_selection` + `fracture_cloud`, `refine_joins`'s spanning walk over each group, the two point-to-plane rungs through the same `icp::register` every other stage uses |
| `crates/sherd-core/src/report.rs` | R §11.1–11.4: `transforms.json`, `report.json` (plus D §4.3's additive `engine`), `report.md` section for section, `write_placed_meshes` and the streamed merged `assembly_<k>.ply` |
| `crates/sherd-core/src/render.rs` | R §11.5: the palette, `principal_views`, the camera basis, the z-buffered 3×3 splat, the PNG, and the caption in the port's own 5×7 font |
| `crates/sherd-core/src/types.rs` | `apply_transform_fused` and `rotate_fused` — §2 below |
| `crates/sherd-parity/src/stages/refine.rs` | D §10.2's `refine` row |
| `crates/sherd-parity/src/stages/outputs.rs` | D §10.2's `outputs` row |
| `crates/sherd-parity/examples/write_outputs.rs` | R §8 + R §9 + R §11 in one pass, into a directory `tools/evaluate.py` can score (§6) |
| `tools/dump_outputs.py` | writes the reference's own R §11.4 meshes and R §11.5 previews for an existing dump (§5) |

---

## 2. Three arithmetic facts, measured

### 2.1 numpy's `@` and Eigen's `Matrix4d * Vector4d` fuse; a scalar transcription does not

`types::apply_transform` carried this sentence: "`a + b + c + d` associates to the left in Rust
exactly as numpy's matmul-then-add does". It is false on this machine, and it had never been
checked against a *coordinate* — step C3 checked it against R §6's *scores*, which are thresholds
on distances and survive a few ULP.

Measured: a random cloud 300 units from the origin, a random pose, Open3D's own
`TriangleMesh::Transform` as the reference and four candidate expressions as the port.

| expression | coordinates matching Open3D, of 900 |
|---|---|
| `fma(t₂, z, fma(t₁, y, t₀·x)) + τ` | **900** |
| `fma(t₃, 1, fma(t₂, z, fma(t₁, y, t₀·x)))` | **900** |
| `t₀x + t₁y + t₂z + τ` (the port's `apply_transform`) | 609 |
| `fma(t₂, z, fma(t₁, y, fma(t₀, x, τ)))` | 580 |
| `fma(t₀, x, fma(t₁, y, fma(t₂, z, τ)))` | 460 |

numpy's `P @ T[:3,:3].T + T[:3,3]` matches Open3D on all 900 as well, so the reference has one
answer and the port had another. The reason is that Eigen evaluates a 4×4 by a 4-vector as a
linear combination of the matrix's *columns* with `pmadd`, which on this machine's NEON packets is
`vfmaq_f64`; OpenBLAS's `dgemm` reaches the same three roundings. The port now has
`types::apply_transform_fused` for every place a coordinate has to come back the same — R §9's
clouds, R §11.4's placed meshes, R §11.5's samples — and `apply_transform` is unchanged, because
R §6's numbers were verified through it on 2 249 candidates and a score is not a coordinate. Its
doc comment now says which is which and why.

**Without this, `placed/<name>.ply` cannot be byte-identical**: every `double` in the file would
differ in its last bits.

### 2.2 the same is true inside the renderer, but only for one of its two products

* `(V − centre) @ R` and `N @ R` are `(n,3) @ (3,3)` — `dgemm` — and **fuse**. Measured: the fused
  form matches numpy on 1 200 of 1 200 coordinates, the unfused one on 784.
* `Nn · light` is `(n,3) @ (3,)` — a matrix-*vector* product — and does **not**. Measured: the
  unfused form matches on 400 of 400, the fused one on 244.

Getting either of the two wrong moves a point across a pixel boundary often enough to be visible:
this is what the `preview px` row is sensitive to.

### 2.3 `np.round` is round-half-to-even, and the z-buffer is `f32` while the comparison is `f64`

`px = np.round(P_x·scale + W/2)` uses banker's rounding (`f64::round_ties_even`, not `f64::round`).
The z-buffer is `np.float32` and the depth compared against it is `float64`, so a candidate depth
is compared against the *narrowed* value already stored and then stored narrowed itself. And
`np.lexsort` is stable while the reference's `last` mask takes the final entry of each pixel's run,
so a tie inside one of the nine offsets goes to the *later* point in concatenation order.

All three are in `render.rs`'s module documentation with the reason, because none of them is
visible from R §11.5's pseudocode alone.

---

## 3. The `refine` row, measured

Injected mode: the reference's own starting poses (`assembly/poses.json`), groups, join order
(`assembly/used.json`) and vertex selection (`refine/<name>.idx`); the port supplies R §9's walk,
its scales, its two rungs and the pose update. The full-resolution mesh is not in any dump — it is
the input file — so it is read from `--input`, which the `load` row pins at one `f32` ULP.

| set | `idx` | `walk` | `dist` | rung 1 rot / trans | rung 2 rot / trans | `fitness` | `rmse` | pose rot / trans |
|---|---|---|---|---|---|---|---|---|
| slab | 0 / 9 952 | 0 / 1 | 0 | 0° / 6.5e-16 t | 0° / 8.3e-16 t | 0 | 9.5e-17 | 0° / 9.3e-16 t |
| terracotta | *capped* (§4) | 0 / 2 | 0 | 1.7e-6° / 2.2e-15 t | 2.4e-6° / 6.2e-15 t | 0 | 2.6e-17 | 0° / 3.1e-15 t |
| pot_A | 0 / 93 923 | 0 / 6 | 0 | 4.0e-6° / 1.8e-6 t | 4.4e-6° / 1.3e-6 t | 0 | 2.3e-9 | 3.2e-6° / 1.3e-6 t |
| pot_B | 0 / 90 657 | 0 / 8 | 0 | 2.4e-6° / 1.5e-6 t | 3.0e-6° / 1.2e-6 t | 0 | 3.8e-9 | 0° / 1.2e-6 t |
| pot_C | 0 / 14 776 | 0 / 4 | 0 | 0° / 5.9e-8 t | 0° / 6.8e-8 t | 0 | 7.9e-10 | 0° / 6.4e-8 t |
| pot_G | — (no group of two or more; the stage skips and says so) |
| pot_H | 0 / 18 720 | 0 / 7 | 0 | 2.7e-6° / 3.3e-6 t | 3.4e-6° / 3.4e-6 t | 0 | 1.4e-8 | 8.5e-6° / 2.7e-6 t |
| synthetic_20 | 0 / 642 861 | 0 / 18 | 0 | 2.1e-6° / 3.3e-13 t | 2.1e-6° / 1.3e-13 t | 0 | 5.4e-16 | 1.6e-5° / 1.2e-13 t |

The row's tolerance is 0.2° / 0.02 t. The worst entry anywhere is **8.5e-6° and 3.4e-6 t**, which
is 4.3e-5 and 1.7e-4 of the tolerance. `fitness` is bit-identical on every join of every set, and
the correspondence radii are exact.

**Why the ULP-scale rows are not all zero.** The rungs on the slab, terracotta, pot_C and
synthetic_20 come out at 1e-13 t or better; pot_A, pot_B and pot_H sit at 1e-6 t. The difference is
the input, not the code: those three are the OBJ collections, whose *original mesh* — the one R §9
builds its cloud from — the two readers disagree on by one `f32` ULP (D §10.2's `load` row,
Assimp's `fast_atof`). One ULP on a vertex at 30 mm is 2e-6 mm, and 40 iterations of ICP on
90 000 such points carries it to 1e-6 t. pot_C is an OBJ set too and lands at 6e-8 t because its
joins converge in fewer iterations.

Native mode is the same ladder from the same poses on the port's own selection, which differs from
the reference's only above R §9's cap. Every set is identical to its injected column except
terracotta, which is §4.

---

## 4. PMC-9's one draw in R §9, and the row that survives it

R §9 caps a fracture cloud at 150 000 vertices with `rng(0).choice(idx, 150000, replace=False)`,
whose *order* is the ICP's summation order. PMC-9 forbids reproducing numpy's draw. Three of the
eight sets' fragments are over the cap — terracotta's three, at about 253 000 candidates each — and
everything else in the whole benchmark is under it and reproduced **exactly**: 0 differing entries
of 9 952 + 14 776 + 18 720 + 90 657 + 93 923 + 642 861 = **870 889 indices**.

For the capped three the row splits in two, and the exact half survives:

* **`idx candidates (capped)`** — every index the reference kept must be one the port's own
  predicate accepted. Measured **0 outside, of 450 000**. This is the whole of R §9's selection
  rule compared exactly; only the draw from it differs.
* **`idx overlap (capped)`** — `|A ∩ B| / |B|` for two independent draws of `n` from `k` is `n/k`
  and nothing else. Measured overlap 0.5932 against an expected 0.5945 (`150000/252 350`), a
  deviation of **0.0013** against a gate of 0.05.

The consequence for the pose is measured, not argued. Refining terracotta from the port's own
150 000 instead of the reference's moves the answer by **0.025° and 0.0027 t** — 12 % and 13 % of
D §10.2's row, which is the same tolerance in both columns and is met with room to spare.

`fitness` is the one number that cannot survive it: it is `|correspondences| / n_source`, a
Bernoulli fraction *of the cloud*, and the two sides count it on different draws. At the terracotta
joins' `p ≈ 0.19` and `n = 150 000`, one σ of the difference between two independent draws is
`√2·√(p(1−p)/n)` = **1.4e-3**. Measured: **9.5e-4**, under one σ. The row therefore renames itself
`fitness (redrawn)` and carries a tolerance of 0.008 (5.6 σ) whenever the cloud was redrawn, and
keeps the ICP rows' 1e-4 whenever it was not. The parity claim in that case is the pose, and the
pose is the row D §10.2 states.

---

## 5. `tools/dump_outputs.py`, and why there had to be one

`tools/dump_fixtures.py` runs the pipeline with `preview=False` and `write_meshes=False`: neither
output is on the algorithm's critical path and both are large. So no dump carries the two files
this step has to reproduce.

Re-dumping the eight sets to get them would cost an eight-hour sweep. `tools/dump_outputs.py DUMP
INPUT` produces them from the dump that already exists instead, and re-derives nothing the
reference computed: the fragments come out of the dump's own `mesh.V`, `mesh.F` and
`seg.frac_final` with `FN` and `A` from the reference's own `face_geometry`; the poses and the
groups out of the dump's own `outputs/transforms.json`; and the two writers called are
`report.write_placed_meshes` and the body of `pipeline.write_previews`, transcribed only so that
each sample's `pick`, `u` and `v` can be written out on the way past. The generator is consumed in
the pipeline's order, so the samples are the ones the pipeline would have drawn.

It leaves behind, per dump:

* `outputs/placed.sha256.json` — a SHA-256, a size, both counts and the colour flag per file, plus
  the hash of every fragment's mesh **as `load_mesh` leaves it**, before any pose. The meshes
  themselves are hundreds of megabytes and are hashed and dropped (`--keep-ply DIR` keeps them).
* `outputs/preview_<k>.png`, the same render with the caption left off, `preview_<k>.meta.json`
  (the views, which PMC-10 makes library-defined) and the `pick`/`u`/`v` of every sample.

The committed `fixtures/slab/dump` now carries all of it at 20 000 samples per fragment (2.0 MB),
so the renderer and the PLY writer have a regression fixture in the repository and the two new
stages have something to run on in `cargo test`.

---

## 6. The `outputs` row, measured

| set | `transforms pose` | `report schema` | `report candidates` | `placed shape` | `placed ply` | `preview px` | `label px` |
|---|---|---|---|---|---|---|---|
| slab | 6.9e-15 t | 0 / 308 | 0 / 5 | 0 / 3 | **0 / 3** | 0 / 4 200 000 | 566 |
| terracotta | 8.8e-15 t | 0 / 1 324 | 0 / 30 | 0 / 5 | **0 / 5** | 0 / 4 200 000 | 1 490 |
| pot_A | 7.0e-13 t | 0 / 6 101 | 0 / 140 | 0 / 9 | (reader, §6.1) | 0 / 4 200 000 | 3 688 |
| pot_B | 3.3e-13 t | 0 / 7 945 | 0 / 180 | 0 / 10 | (reader) | 0 / 4 200 000 | 4 720 |
| pot_C | 2.4e-14 t | 0 / 4 324 | 0 / 105 | 0 / 8 | (reader) | 0 / 4 200 000 | 3 248 |
| pot_G | 1.4e-13 t | 0 / 4 140 | 0 / 105 | 0 / 7 | (reader) | 0 / 1 680 000 | 3 283 |
| pot_H | 2.2e-13 t | 0 / 11 272 | 0 / 275 | 0 / 12 | (reader) | 0 / 4 200 000 | 4 943 |
| synthetic_20 | 2.9e-13 t | 0 / 13 936 | 0 / 335 | 0 / 21 | **0 / 21** | 0 / 4 200 000 | 5 795 |

* **`transforms pose`** is gated at 1e-9 t, which is the `assembly` row's own `recentre` tolerance
  rather than D §10.2's "as refine" (0.02 t): the port recentres the reference's own poses here, so
  the only arithmetic between the two files is R §8.2's mean and a subtraction. The worst measured
  is **7.0e-13 t**, 7e-4 of the gate. D §10.2's row is updated to say so.
* **`report schema`** reads the reference's own `report.json` into the port's `ReportJson`, writes
  it back and diffs the two JSON trees: a key the port's type does not know is a key it would
  silently drop. **0 differing leaves of 49 350** across the eight sets. `timings` are excluded —
  the reference's own dump nulls them for the same reason — and `engine` is D §4.3's addition.
* **`report candidates`** renders the reference's own candidate list through the port's own
  serialiser and compares it entry for entry: the pose, the verdict, `seam·tight` and every one of
  R §6's twenty score names. **0 differing of 1 175 candidates.**
* **`preview px`** is the whole point of `render.rs`: 31 080 000 pixels compared over the eight
  sets, **0 differ**.

### 6.1 `placed ply` is byte-for-byte on four sets and cannot be on five, and the row says which

A placed mesh is the input file transformed, so it can only be byte-identical when the input file
*reads back* byte-identically — and on the five OBJ collections it does not. D §10.2's `load` row
already measures Open3D's reader (Assimp's `fast_atof`) exactly one `f32` ULP from the port's. The
stage therefore hashes both sides' cleaned source meshes first and compares the placed files only
where those agree, skipping the rest with the reason spelled out. What is compared on **every**
set is `placed shape` — size, vertex count, face count and colour flag — which gates the header,
the merge, the index offsets and Open3D's colour rule (`operator+=` keeps colours only while every
member has had them) without the reader in the way.

The OBJ difference was measured rather than assumed. `SHERD_PARITY_KEEP_PLACED=/tmp/ours` keeps the
port's own files; on pot_C's smallest fragment (`Pot_C_Piece_06_Mesh_DS`, 3 579 vertices,
7 154 faces):

* the two files are **the same size** and their headers are **byte-identical**;
* the colour bytes are byte-identical;
* 209 payload bytes differ, in **178 of 10 737 coordinates**, every one of them by exactly
  9.5367431640625e-07 — one `f32` ULP at that magnitude — and the largest relative difference is
  2.9e-4 on a coordinate near zero.

That is the reader, and nothing of R §11.4.

### 6.2 `report.md` against the reference's own, on terracotta

R §11.3 is not in the parity table (no dump carries `report.md`), so it was checked by writing one
and diffing. `examples/write_outputs` on the terracotta dump produces a `report.md` that is
**identical to the reference's, line for line**, except for the `## Timing` block, which R §11.6
puts outside the contract. Getting there needed one fix: Python's `str(3.0)` is `"3.0"` and Rust's
`{}` is `"3"`, so the legend's `seam ≥ 3.0` came out as `seam ≥ 3`. `report::python_float` prints
an integral float the way Python does, and the diff is now empty.

### 6.3 the port's outputs, through the reference's own evaluator

`cargo run --release -p sherd-parity --example write_outputs -- DUMP INPUT OUT` runs R §8, R §9 and
all five of R §11's writers on the reference's own candidates and samples, and
`python tools/evaluate.py OUT INPUT` scores what comes out. Every number is the Python result:

| set | fragment accuracy | precision | largest group |
|---|---|---|---|
| terracotta | joins used exactly {021–094, 094–104}, 007 unplaced, both `pen` 0, seams 20.3 t and 10.7 t (R §13's gate, no ground truth file) |
| pot_A | **87.5 %** (7/8) | **1.000** | 7 |
| pot_B | **100 %** (9/9) | **1.000** | 9 |
| pot_C | **75 %** (3/4) | 0.500 | 5 |
| pot_G | **0 %** (0/7) | 0.000 | 1 |
| pot_H | **36.4 %** (4/11) | 0.429 | 8 |
| synthetic_20 | **95 %** (19/20) | **1.000** | 19 |

---

## 7. PMC-20 — the caption

R §11.5 draws a caption at `(10, 10)` with PIL's `ImageDraw.text` and `ImageFont.load_default()`,
which in Pillow 12 is an **anti-aliased FreeType face** (Aileron), not the classic PIL bitmap font.
Reproducing it would mean embedding that font and FreeType's rasteriser, and the font is under a
licence of its own. The port draws the same text with its own 5×7 bitmap glyphs, upper case only.

This is the one part of R §11 the port does not reproduce, so it is a PMC and it is *measured*
rather than asserted away. The `outputs` row compares against the reference's own **unlabelled**
render — which is why `tools/dump_outputs.py` writes both — and reports the caption's own size as
`preview label px`: how many pixels of the reference's image the caption covers. Measured over the
eight sets: **566 to 5 795 pixels**, against a cap of 8 000, on images of 1 680 000 to 4 200 000
pixels. So between 0.03 % and 0.35 % of a preview is given up, and the row says exactly how much.

R §12.1 carries PMC-20 as a new row of R §12's table, with the same numbers.

---

## 8. Tests

`cargo test --workspace`: **282 → 300 passing, 1 ignored** (+18).

* `sherd-core::refine` (4): Open3D's area-weighted vertex normal against a hand-computed
  `(0, 10, 1)/√101` (the *unweighted* mean of the same two unit normals is 39° away), the selection
  predicate and its resolution floor, the walk's order and one-correction-per-fragment on three
  saddle patches, and a join whose cloud is missing.
* `sherd-core::render` (7): the background's 40 (not 41 — the narrowing to `f32` and the
  truncation both matter), a single point splatting exactly nine pixels with the reference's own
  shade, the z-buffer picking the same winner whichever order the two points arrive in, the camera
  basis staying orthonormal through the degenerate `up × z`, `principal_views` looking along the
  flattest axis, the caption's pixels, and **two renders of one input being the same bytes**.
* `sherd-core::report` (4): `transforms.json` round-tripping with R §11.1's `placed` rule, a
  candidate flattening into `report.json` exactly as `Candidate.to_json()` does, R §11.3's seven
  sections and their number formats, and the writer placing every mesh, merging the groups with the
  second member's indices offset, and differing from the reference's file only in the header
  comment.
* `sherd-parity` (3, in `tests/stages_slab.rs`): the `refine` row made to fail by a pose and by a
  selection, the `outputs` row made to fail by a pose, a mesh hash and a pixel, and the two stages
  skipping what the dump does not carry. Two existing tests were updated: `outputs` in native mode
  needs no input directory (it compares the port's renderer and writer against themselves), and
  `refine` in *injected* mode does need one, because R §9's input is the original file and no dump
  carries it.

---

## 9. Documents

* **R §12.1** gained a step-D2 addendum: PMC-20 as a new row of §12's table, the FMA measurement of
  §2.1 with its five-way table, the renderer's four arithmetic details, and R §9's cap with the
  numbers of §4.
* **D §10.1** gained `tools/dump_outputs.py` and the files it writes, in the layout block and in a
  paragraph of its own.
* **D §10.2**'s `refine` and `outputs` rows are written out in full — every quantity, both columns
  — with two paragraphs under the table: what the second Python tool is for, and why `placed ply`
  is byte-for-byte on four sets and `placed shape` on all eight.
* `crates/sherd-core/src/lib.rs`'s module map now names D2 against `refine`, `report` and `render`,
  and says what is left of phase 1d: the run itself.

---

## 10. What is left

`pipeline.rs` still holds only preprocessing, and `sherd-refit-rs run` still does nothing. Every
piece it needs now exists and is gated: R §3 through R §6 (steps S2–C3), R §8 (D1) and R §9 and
R §11 (this step). `examples/write_outputs` is three quarters of that run already — it is missing
only the pairwise matching, which it takes from the dump instead — so the remaining work is the
driver, its threading and its progress reporting, not another algorithm.

Two smaller things this step leaves behind:

* `report.md` is not in any fixture dump, so §6.2's diff is a one-off rather than a gate. Adding
  `report.md` to `tools/dump_outputs.py` would make it one, and would cost a few kilobytes per set.
* `preview label px` bounds the caption at 8 000 pixels. If a collection ever has enough fragments
  that the caption wraps past the image edge, the number stops growing rather than failing —
  which is the right behaviour for a bound on what is given up, but worth remembering.
