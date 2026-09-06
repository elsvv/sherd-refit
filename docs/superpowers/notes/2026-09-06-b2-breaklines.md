# B2 — breaklines and frames (R §3.5.3–3.5.5)

**Date:** 2026-09-06. **Branch:** `rust-core`. **Step:** phase 1b, B2, after B1 (segmentation).
**Code:** `crates/sherd-core/src/fragment/breakline.rs`, the five `brk_*` tensors of
`fragment/cache.rs`, `crates/sherd-parity/src/stages/breakline.rs`.
**Reference:** `sherd_refit/fragment.py::match_arrays` / `macro_normals` at `9d4b9d3`.
**Machine:** Apple M2 Pro (10 cores, 16 GB), macOS 15.6.1, rustc 1.97.0, Open3D 0.19, numpy 2.5.2.

## Result in one line

**Injected: exact.** On all 66 fragments of the seven fixture collections the count and the
hypothesis subset are equal entry for entry, and *every* residual in the arrays — 3.7e-5 on a
coordinate, 4.5e-6° on a frame, 0.0026° on a dihedral — is reproduced to the last digit by
rounding the reference's own `f64` arrays to `f32` and back. There is no arithmetic difference
left over. **Native: 165 of 198 checks**, and all 33 failures are measured to be inherited from
`t` and the segmentation, or to sit in a row that contradicts the row above it (§4).

| mode | collections | fragments | checks | failed | worst / tolerance |
|---|---|---:|---:|---:|---|
| injected | 7 | 66 | 726 | 0 | 0.07 (`points`, `f32` narrowing) |
| native | 7 | 66 | 198 | 33 | 14.95 (`p99 distance`, Pot_B_Piece_01) |

## 1. What was ported

The breakline is the closed curve where a fragment's fracture meets its shell, and the frame that
sits on every point of it. R §3.5.3–3.5.5 in five steps, all of them in `breakline::build`:

1. **Points.** The midpoint of every edge whose two faces carry different labels, in
   face-adjacency order (`0.5·(V[e0] + V[e1])`, the reference's expression in the reference's
   order).
2. **Macro normals.** `dist[f]` = the distance from every face centroid to the nearest breakline
   point; `ns[q]` = the unit of `Σ A·FN / Σ A` over the **shell** faces with `dist ≥ 0.15 t` that
   lie within `0.60 t` of the point, `nf[q]` the same over the fracture faces. A point whose
   annulus is empty falls back to the whole `0.60 t` neighbourhood of that mask; a point that
   finds nothing even then keeps the zero vector.
3. **The frame.** `f = nf − (nf·ns) ns`, divided by `max(|f|, 1e-9)`; `tangent = ns × f` and
   `dih = degrees(arccos(clip(ns·nf)))` are derived (R §3.6), not stored.
4. **Validity.** `|ns| > 0.5 ∧ |nf| > 0.5 ∧ |ns × f| > 0.5`.
5. **The hypothesis subset.** The breakline points voxel-downsampled at `0.5 t` with Open3D's
   bounds convention, one representative (the lowest index) per occupied voxel, filtered to the
   valid frames. Ascending, which is PMC-4.

The inner radius is the point of the annulus and it is not a detail: it is what keeps the two
dihedrals of a mating pair summing to 180° rather than 141° (`2026-09-06-scale-pairs.md`), and
the slab test below fails by 20° without it.

**Shared with R §3.4.** Two pieces of the segmentation are literally the same computation and are
now called rather than copied: `segment::ball_normals` (the reference's `ball_matrix` followed by
`smoothed_normals`) and `segment::voxel_representatives`, lifted out of `coarse_grid` so that
R §3.4.1's `t/8` grid over face centroids and R §3.5.5's `0.5 t` grid over breakline points are
one function with one bucket rule. `PointTree` gained `nearest_distance`, the `cKDTree.query`
that returns the distance as well as the index, which is what selects the annulus.

**Deviations, all of them already in R or D.** PMC-4 (the subset is sorted, not in hash order);
the neighbourhood summation order (D §7 — ascending here, unspecified in scipy, and §3 now shows
it costs nothing measurable); and the `f32` storage of D §4.1, which is the only one this step
can put a number on — see §3.

## 2. Where it lives

`Fragment.brk: Breaklines` is built at the fragment's own `t` right after the labels, from the
same `f64` geometry the segmentation ran on (the geometry of the **narrowed** working mesh, for
the reason `WorkingMesh::from_parts` exists). Whole-collection preprocessing including it:
terracotta 4 fragments in 1.87 s wall, synthetic_20 in 6.65 s (20 fragments, 200 000 faces each,
79.65 s of work over 10 cores).

The cache gains `brk_P`, `brk_ns`, `brk_nf`, `brk_f` (`f32[k,3]`) and `brk_sub` (`u32[j]`), and
the metadata gains `brk_params` — `t` and the three radii, the breakline half of R §3.7's
`mdp_*`. `CACHE_VERSION` 2 → 3. Two rules the reader enforces that the writer does not: the four
frame arrays must describe the same points, and `brk_sub` must index them. And R §3.7's other
half is implemented rather than deferred: a cache that is valid but carries other `brk_params`
has its **breaklines** recomputed and rewritten, not the whole fragment thrown away.

`sherd-refit-rs segment` prints `brk` and `sub` columns beside `fracture`, and logs the point
count, the valid count and the subsample per fragment.

## 3. Parity: injected — exact on all 66 fragments of all seven sets

`sherd-refit-rs parity --fixtures output/fixtures/<set> --stage breakline --injected`, on
terracotta, pot_A, pot_B, pot_C, pot_G, pot_H and synthetic_20: **726 of 726 comparisons pass**,
worst ratio 0.07 of tolerance. The stage runs R §3.5.3–3.5.5 on the dump's own working mesh, its
own `seg.frac_final` labels and its own `md.params.json` knobs, so nothing upstream can move the
answer.

| quantity | fragments | worst deviation | gate |
|---|---:|---|---|
| `count` | 66 | 0 | exact (D §10.2) |
| `points` (symmetric Hausdorff) | 66 | 3.659e-5 | `1e-4 t` = 2.3e-4 … 3.9e-3 |
| `points in order` (elementwise) | 66 | 3.659e-5 | same |
| `dihedral` (worst per point) | 66 | 0.00258° | 0.1° (D §10.2) |
| `ns` / `nf` / `f` / `tangent` (worst angle) | 66 | 2.8e-6 / 2.7e-6 / 2.8e-6 / 4.5e-6 ° | 0.1° (diagnostic) |
| `valid` (entries differing) | 66 | 0 | exact |
| `sub count`, `sub set` (sorted) | 66 | 0 | exact |

**Every one of those three residuals is the `f32` cache narrowing, and each was reproduced on the
Python side rather than argued for.**

* `points` — the coordinates round trip through `f32` (D §4.1); 3.659e-5 is one `f32` ulp at
  pot and synthetic coordinate magnitudes.
* `dihedral` — rounding the reference's *own* `md.brk_ns` and `md.brk_nf` to `f32` and back and
  recomputing `degrees(arccos(clip(ns·nf)))` gives **0.00258193°** on `Pot_A_Piece_01_Mesh`, which
  is the port's number to six significant figures. It lands at the point whose dihedral is
  0.0569°, where `arccos` has an infinite derivative; 71 of that fragment's 6 446 points are
  within 1° of 0 or 180.
* `ns`, `nf`, `f`, `tangent` — the port's macro normals are equal to the reference's *bit for
  bit* before narrowing. Dumping the port's `ns` for `Pot_A_Piece_02_Mesh` and comparing with the
  dump: max component difference 2.82e-8 (one `f32` ulp at 0.6), max angle after normalising both
  sides 2.70e-6°, which is the noise floor of `arccos` near 1, and the median angle over all
  3 306 points is exactly 0.

### 3.1 A parity metric that was measuring the wrong thing

The frame checks first read 0.015–0.026°, and the draft of this note attributed that to the
summation order. It was not: `arccos` of a **raw** dot product between a unit `f64` vector and an
`f32`-narrowed one reads the narrowed vector's *length* error `δ` as a spurious angle of
`sqrt(2δ)`. For `δ = 4.68e-8` that is 0.0175°, which is the number that appeared, to six figures.
The direction error underneath it is 2.7e-6°, four orders of magnitude smaller — and the square
root means the artefact *grows* as the storage improves more slowly than the reader would expect.
`worst_angle` now normalises both sides, and its doc comment says why.

This is not a defect in the `segmentation` stage's identical-looking `NS angle` check: there both
sides are unit in `f64`, so there is no length error to amplify, and B1's 1.708e-6° is the same
`arccos` noise floor — which, read correctly, says B1's `NS` was bit-identical too rather than
merely close.

Three rows of the table are worth reading twice.

* **`points in order` is not `points`.** Both implementations emit the points in face-adjacency
  order, so the arrays are comparable entry by entry. A port that produced the right *cloud* in
  the wrong *order* would pass a Hausdorff gate and break every hypothesis of R §5.1; only the
  ordered check sees it.
* **`sub set` is exact on all 66 fragments,** which settles the second half of PMC-4 empirically.
  B1 established by reading Open3D that the bucket rule (`l[0]` is the lowest index in the voxel)
  is exact and only the voxel *order* is a hash artefact. Sorting both subsets and comparing them
  entry for entry is the test of that claim on real data, and it passes everywhere.
* **`valid` is exact on all 66 fragments,** so the two implementations agree not only on the
  frames but on which of them exist.

## 4. Parity: native — 165 of 198, and every failure is inherited

Native mode runs the port's own pipeline from the file. 33 of 198 comparisons fall outside
D §10.2's native column (ratios below are deviation / tolerance; 1.00 is the gate):

| set | fragments | `count` (±10 %) | `p99 distance` (0.5 t) | `dihedral KS` (0.05) |
|---|---:|---|---|---|
| terracotta | 4 | 0 fail, worst 0.87 | 0, 0.55 | **1**, 1.31 |
| pot_A | 8 | 0, 0.32 | 0, 0.52 | **1**, 1.29 |
| pot_B | 9 | 0, 0.86 | **7**, 14.95 | 0, 0.97 |
| pot_C | 7 | 0, 0.29 | 0, 0.38 | 0, 0.42 |
| pot_G | 7 | 0, 0.32 | **3**, 3.80 | 0, 0.68 |
| pot_H | 11 | 0, 0.29 | 0, 0.58 | 0, 0.76 |
| synthetic_20 | 20 | **13**, 1.97 | **5**, 2.70 | **3**, 1.55 |

The measurements below come from the port's own caches (`sherd-refit-rs segment` into a scratch
directory, then the `.sherd` tensors read straight out of the safetensors file) against the
dump's `mesh.*`, `seg.frac_final` and `md.brk_P` — so both sides are the finished artefacts, not
a reconstruction.

### 4.1 `p99 distance`: the far points sit on the segmentation's disagreement

Over the 66 fragments there are 271 592 point-to-set distances (both directions). **3 115 of them
— 1.15 % — exceed `0.5 t`.** The curves themselves coincide: the median distance is 0.000–0.057 t
per fragment, and 25 of the 66 fragments have not one point beyond `0.5 t`.

**44 of the 66 native working meshes are identical to the reference's** — same `F` entry for
entry, `V` equal to 3.05e-5, which is the `f32` narrowing again. On pot_B all nine are, on pot_G
all seven, on pot_H all eleven. On such a fragment the *only* thing that can differ between the
two breaklines is the labels, and §3 has already proved the breakline code exact on given labels.
So the question reduces to one measurement: how far is a far point from a face the two
segmentations label differently? Over the 12 identical-mesh fragments that have far points at all
(2 334 of the 3 115), **the worst is 0.169 t** and on seven of pot_B's eight it is ≤ 0.052 t.

The chain is short and was already written down in B1. The port's `t` comes from the same
estimator on a different sample (PMC-9); on pot_B it moves by −3.7 % to +1.2 %, and the labels
move with it:

| pot_B fragment | `t` Δ | area agreement | our far points | their far points | our count / theirs |
|---|---:|---:|---:|---:|---|
| Piece_01 | −3.72 % | 0.9142 | 182 | 446 | 5 329 / 5 616 |
| Piece_02 | −0.72 % | 0.9970 | 118 | 0 | 2 093 / 1 994 |
| Piece_03 | −0.08 % | 0.9943 | 302 | 180 | 2 665 / 2 535 |
| Piece_04 | +1.10 % | 0.9917 | 280 | 220 | 2 509 / 2 498 |
| Piece_05 | +1.20 % | 0.9955 | 0 | 28 | 1 770 / 1 889 |
| Piece_06 | +0.33 % | 0.9913 | 41 | 45 | 2 155 / 2 120 |
| Piece_07 | +0.19 % | 0.9921 | 130 | 187 | 1 716 / 1 849 |
| Piece_08 | −0.55 % | 0.9892 | 0 | 19 | 2 285 / 2 104 |
| Piece_09 | −0.47 % | 0.9945 | 0 | 0 | 1 500 / 1 532 |

Piece_02 is the clean case: 0.3 % of the area labelled differently, 118 of our 2 093 points more
than `0.5 t` from the reference's curve and **zero** reference points far from ours — one extra
loop around a small patch, nothing displaced.

So the port's breakline is right for the mask it is given (§3 proves that exactly) and the mask is
right for the `t` it is given (B1 proves that exactly), and what fails is the third link. This is
B1's finding one stage further on and amplified: a p99 **point** gate cannot be met when the
segmentation row upstream allows 3 % of the **area** to disagree, because the boundary of a 1 %
area difference is far more than 1 % of the boundary.

### 4.2 `count`: the curve is the same length, sampled more coarsely

The 13 `count` failures are all synthetic_20, all in the same direction (the port has 5–20 %
fewer points), and none of them is a missing piece of curve — `frag_000` fails `count` by 17.1 %
with **3** far points out of 6 127. The reason is the sampling density, not the curve:

| over the 20 synthetic fragments | mean | worst |
|---|---:|---:|
| point count, port / reference | −10.0 % | −19.7 % |
| median nearest-neighbour spacing along the breakline | +15.0 % | +30.8 % |
| **`count × spacing`** — the curve's length | **+2.9 %** | +11.6 % |

The port's decimator leaves a mesh with the same face count (200 000 = 200 000 on most of these)
but longer edges near the break, and a breakline crossing a coarser mesh has fewer edges to cross.
D §10.2 already allows `res` ±10 % natively, and a count gate of ±10 % on top of that has no
headroom left: the two rows contradict each other. The curve's *length* agrees to 2.9 % on
average, which is what "the same breakline" means when neither mesh is the other's.

### 4.3 `dihedral KS`: five fragments, 0.052–0.078 against 0.05

Two of the five are the thickness outliers B1 already named — `Pot_A_Piece_04` at `t` +6.58 % and
`frag_010` at +4.54 %. The annulus is `0.15–0.60 t`, so a 5 % larger `t` measures the macro
normals over a 5 % wider ring and the whole distribution shifts. The other three (`FY234007`,
`frag_014`, `frag_017`) have `t` within 0.4 % but a working mesh that is *not* identical to the
reference's. A KS statistic of 0.05 between two distributions sampled on two different meshes is
a tight ask; these are at 0.052–0.078.

### 4.4 What was not done about it

Nothing — the row is not widened. B1's precedent is followed exactly: when the evidence says the
port is right and the input is not, the gate stays and the cause is named (D §10.2's
t-propagation paragraph, D §13 question 2, E6). Two of the three native breakline quantities are
now measured to be *inherited* failures with the arithmetic to prove it, and the third (`count`)
is a row that contradicts the `res` row above it in the same table. Both belong to the parity
standard question, not to this step's code, and D §10.2 and D §13 now carry the numbers.

## 5. Tests (+10; 171 → 181 passing, 1 ignored)

* **A welded slab** — 36 × 36 × 6 at 0.5 per edge, six sides sharing their vertices (the slab in
  `segment`'s own tests is built side by side and shares none, so it has no breakline at all):
  two rims of exactly 288 points each, every point on the boundary rectangle of its own plane,
  the subsample between 80 and 96 points for two 48-voxel rims and strictly ascending.
* **The frames on it**: `ns` = ∓z to 1e-5, `nf` in the wall, `f ⟂ ns`, the dihedral 90° to 1e-3,
  the tangent a unit vector in the shell's plane across `nf`, and `f × tangent = ns` — R's
  right-handed convention, which is what lets one point of A and one of B fix a pose.
* **The fallback and the failure**: an inner radius larger than the slab leaves every annulus
  empty and the frames still come out right (only the fallback can do that); an outer radius below
  the mesh's resolution leaves every macro normal zero, every frame invalid and the subset empty.
* **No breakline at all**: an all-shell and an all-fracture mesh both produce an empty
  `Breaklines` without dividing by anything.
* **Cache**: the breakline arrays survive the round trip bit for bit and a warm run equals a cold
  run on the real slab; a file whose frames do not describe its points, or whose `brk_sub` points
  outside them, is refused; a cache with other `brk_params` has its breaklines rebuilt and
  rewritten, with the mesh and the labels still coming from the cache.
* **Parity harness**: the slab dump passes the new stage in both modes (22 injected checks, 6
  native), and the percentile and the two-sample KS statistic are tested against hand-computed
  values.

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings` and
`cargo test --workspace --locked` are green; two `segment` runs on the terracotta still produce
byte-identical caches, now including the five breakline tensors.

## 6. One correction to the reference

`sherd_refit/fragment.py` comments its subsample as `# subsample for hypotheses (voxel t/3)`
while the code uses `p["brk_voxel"] * t` with `brk_voxel = 0.5`. R §3.5.5 already said `0.5·t`,
which is the code; the comment is stale, and R now says so, so that the next reader does not port
the comment. R §3.5.5 also records that the sorted subsets were measured equal on all 66
fragments, which is the empirical half of PMC-4.

## 7. What is next

The sampled half of the match arrays — R §3.5.1 (`S`, `sp`), §3.5.2 (`Pf`, `fp`) and §3.5.6
(`margin_idx`) — which is where the RNG enters and where PMC-9's seed question is decided for the
sampling as well as for the thickness. The breakline arrays are the half that needs no RNG at all,
which is why this step could be exact.
