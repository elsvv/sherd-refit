# S3 — mesh operations up to the working mesh

**Date:** 2026-09-06. Branch `rust-core`. Plan step S3 of
`docs/superpowers/plans/2026-09-06-rust-core-phase0-1a.md`; algorithm reference R §3.1–3.3, design
D §3, §4.1, §7, §10.2. Toolchain: rustup 1.97.0 (the tree's `rust-toolchain.toml`; Homebrew's
cargo 1.88 is below the 1.89 MSRV). Machine: Apple M2 Pro, 10 cores, 16 GB, shared with other
agents; every timing below was taken with `RAYON_NUM_THREADS=4`.

**Exit criterion (plan): "fixtures *working mesh* stage within D §10.2 native tolerances on all
fixtures". Met for faces, `res`, area and `watertight` on all 68 fragments of the seven fixture
collections. Not met for `t`/`thick_mode` at D §10.2's ±2 %, and §5 below shows that ±2 % is not
reachable by any implementation that does not reproduce numpy's PCG64 — the reference's own
estimator moves by up to 14.5 % when only the seed changes. Everything else in the stage is
bit-identical to the reference.**

## 1. What was implemented

| file | R | contents |
|---|---|---|
| `crates/sherd-core/src/mesh/geometry.rs` | §0, §3.3 | `face_geometry` (FN, A, C), `median_edge` (`res`), numpy's `median`, numpy's `pairwise_sum` |
| `crates/sherd-core/src/mesh/adjacency.rs` | §3.3.2, §3.4 | `unique_edges`, `closed_enough`, `face_adjacency` (the reference's `(fa, fb)` pairs), `vertex_adjacency` (CSR, for Taubin) |
| `crates/sherd-core/src/mesh/decimate.rs` | §3.3 | `face_budget` (`clip(600·ΣA0/t², 50000, target_faces)`), `decimate` (meshopt + R's cleanup order) |
| `crates/sherd-core/src/mesh/taubin.rs` | §3.3.1 | `taubin` (3 iterations, λ = 0.5, μ = −0.53, inverse-distance weights, Jacobi) |
| `crates/sherd-core/src/fragment/thickness.rs` | §3.2 | `estimate_thickness`, `thickness_from_hits`, `hist_mode`, `percentile90`, `sample_face_indices`, `obb_min_extent`, and `RayScene` — a temporary parry3d BVH |
| `crates/sherd-core/src/fragment/mod.rs` | §3.1–3.3 | the `Fragment` type and `Fragment::from_mesh_file`, the reference's order of operations |
| `crates/sherd-parity/tests/working_mesh_slab.rs` | — | five integration tests against the committed slab fixture, injected and native |

Crates used, all of them the ones phase 0 chose, none of them replaced by own code: `meshopt`
0.6.2 with `SimplifyOptions::Regularize` (E1), `parry3d` 0.30.2 without `TriMeshFlags::ORIENTED`
(E3/E4), `rand_chacha` (D §7), `rayon` for the ray loop. **The only own code S3 adds beyond the
algorithm itself is numpy's pairwise summation and numpy's `float32` percentile/histogram chain
(§2), which no crate provides and which the exact comparisons depend on.**

## 2. Five things measured against the reference, not assumed

1. **Taubin's weight is `1 / (dist + 1e-12)`, additive, not `1 / max(dist, 1e-12)`.** Against
   `filter_smooth_taubin(number_of_iterations=3)` on a jittered 62-vertex sphere: the additive
   form agrees to 4.4e-16, the plain `1/dist` is off by 4.1e-14 — a hundredfold difference that
   only that constant explains. Uniform weights are off by 2.8e-2 (PMC-3 stays closed: the port
   uses inverse-distance weights).
2. **The whole thickness chain is `float32`.** Open3D's `RaycastingScene` returns `float32`
   `t_hit`; `np.percentile` on a `float32` array returns `float32` (numpy 2's weak scalars never
   promote it, and `_quantile`'s `weak_q` branch turns γ into a *Python* float so the lerp itself
   is `float32`); `np.histogram`'s `bin_type = result_type(0, p90, a)` is `float32`, so the bin
   edges, the index arithmetic and the bin centre are all `float32`. Reproducing that — including
   numpy's two ±1-ULP index corrections and its `linspace` endpoint assignment — is what makes `t`
   and `thick_mode` come out bit-identical on all 48 fragments that carry ray fixtures. A first
   version that interpolated the percentile in `f64` and rounded once was off by 1 ULP on 2 of
   those 48, which moved `t` by two ULP.
3. **`np.sum` is a pairwise tree, not a left-to-right loop.** `ΣA0` is the numerator of the face
   budget and the `area` the parity table compares. Eight accumulators over blocks of 128,
   splitting anything longer in half on a multiple of eight (numpy's `PW_BLOCKSIZE`): reproduced,
   and `ΣA0` is then bit-identical to the reference's on all 48 fragments. A naive sum differs by
   up to 6.3e-14 relative — never enough to move `target` by one triangle on this data, but there
   is no reason to carry the difference.
4. **Open3D's `get_oriented_bounding_box()` runs its PCA over the convex hull, not over the
   vertices.** Measured on a Gaussian blob with a thin arm: hull-PCA extents
   (7.257, 7.518, 23.613) against all-points-PCA (7.193, 7.341, 23.553), ≈ 1 % apart. `obb_min_extent`
   does the all-points PCA and says so in its documentation. This is the one knowing deviation in
   S3 and it is **unreachable on the benchmark**: the fallback needs fewer than 100 of 20 000 rays
   to hit, and the worst fragment of the 68 has 7 154.
5. **`serde_json`'s default float parser misrounds by one ULP.** `29.864871978759766` — the exact
   decimal of a `float32` the reference wrote — parses to `0x403ddd6840000001`, one ULP above the
   value `f64::from_str` and Python's `float()` both give. Every exact scalar comparison against a
   fixture would fail on that. Fixed at the root: the workspace pins
   `serde_json = { version = "1.0.151", features = ["float_roundtrip"] }`. **S4's parity reader
   must keep that feature on.**

## 3. Injected parity — the Rust stage on the Python stage's own arrays

Harness: `scratchpad/rust/s3` (not committed), over `output/fixtures/{terracotta, pot_A, pot_B,
pot_C, pot_G, pot_H, synthetic_20}` and `fixtures/slab/dump` — 68 fragments, 22 of which the
reference decimated.

| comparison | fragments | result |
|---|---|---|
| `res` from `mesh.V`/`mesh.F` vs `mesh.res` | 68 | **bit-identical**, relative difference exactly 0 |
| `ΣA` of the working mesh vs `mesh.stats.area` | 68 | **bit-identical** |
| `watertight`, `n_boundary` vs `mesh.watertight` | 68 | identical on all |
| `ΣA0` of the original component vs `thick.target.area0` | 48 | **bit-identical** |
| `face_budget(ΣA0, t, 200000)` vs `thick.target.target` | 48 | identical on all |
| `t`, `thick_mode` from `thick.idx`/`thick.t_hit`/`thick.prim` | 48 | **bit-identical**, both values |
| Taubin on `load.V0`/`load.F0` vs `mesh.V` | 43 | max abs deviation **7.96e-13** (1.68e-12 of `res`) |

464 exact comparisons, 0 failures. The Taubin row is the only inexact one and its cause is known
and bounded: Open3D accumulates each vertex's neighbour sum in `std::unordered_set` order, this
port in ascending index order, so six Laplacian steps accumulate a few ULP of round-off. Per
collection the worst deviation is 2.3e-13 (slab), 5.7e-13 (pot A), 6.3e-13 (pot B), 8.0e-13
(pot C), 6.8e-13 (pot G, pot H) — never more than 1.7e-12 of one edge length.

The 43 are the fragments the reference did *not* decimate, where its working mesh is exactly
`Taubin(V0, F0)` and the comparison is therefore meaningful. There is no fixture for the
intermediate decimated mesh, so injected Taubin cannot be checked on the other 25; §4 covers them
statistically.

## 4. Native parity — `Fragment::from_mesh_file` from the file

D §10.2's native column, worst deviation over each collection (positive = Rust above the
reference):

| set | n | decimated | faces | `res` | area | `watertight` | `t` | `thick_mode` | Rust s | Python s |
|---|---:|---:|---|---|---|---|---|---|---:|---:|
| terracotta | 4 | 4 | −0.18…+2.22 % | +5.88…+9.54 % | −0.35…−0.33 % | 4/4 agree | −1.09…+0.09 % | −1.54…+1.13 % | 3.52 | 33.65 |
| pot A | 8 | 1 | 0.00 % | 0.00 % | −0.00 % | 8/8 | −0.11…+6.58 % | −3.12…+2.23 % | 0.92 | 3.00 |
| pot B | 9 | 0 | 0.00 % | 0.00 % | −0.00 % | 9/9 | −3.72…+1.20 % | −5.76…+2.44 % | 0.75 | — |
| pot C | 7 | 0 | 0.00 % | −0.00 % | +0.00 % | 7/7 | −1.54…+0.23 % | −1.35…+0.51 % | 0.23 | — |
| pot G | 7 | 0 | 0.00 % | 0.00 % | +0.00 % | 7/7 | −2.27…+0.24 % | −2.35…+3.41 % | 0.24 | — |
| pot H | 11 | 0 | 0.00 % | 0.00 % | −0.00 % | 11/11 | −0.38…+0.94 % | −1.00…+5.80 % | 0.35 | 1.07 |
| synthetic 20 | 20 | 17 | −0.12…+0.22 % | −1.79…+9.35 % | −0.18…+0.00 % | 20/20 | −5.27…+4.54 % | −3.37…+2.67 % | 5.66 | 35.23 |
| slab | 2 | 0 | 0.00 % | 0.00 % | +0.00 % | 2/2 | −0.13…−0.01 % | +0.62…+0.75 % | 0.06 | 0.21 |
| **worst** | **68** | **22** | **2.22 %** (gate 5) | **9.54 %** (gate 10) | **0.35 %** (gate 0.5) | **0 mismatches** | **6.58 %** (gate 2) | **5.80 %** (gate 2) | **11.7** | — |

* **faces, `res`, area, `watertight`: pass.** On the 46 fragments the budget does not decimate,
  faces and area are exact and `res` differs by at most 4.7e-5 % — that is the `f32` narrowing of
  `WorkingMesh` (D §4.1) and nothing else. Every non-zero number in the table comes from one of
  the 22 decimated fragments.
* **`res` has the least headroom.** +9.54 % against a ±10 % gate, on the terracotta scans, and it
  is `meshopt`'s known signature: E1 measured −1.7…+9.5 % for exactly these meshes against Open3D.
  S3 changes nothing there; if a later stage needs more headroom, the decimator is the knob.
* **The decimator hits its budget.** On the 22 decimated fragments the face count lands 0 to 18
  triangles *below* the budget it was given, never above. The face deviations in the table are
  therefore not the decimator: they are the budget moving with `t` (`target ∝ 1/t²`, so
  FY234104's −1.09 % thickness buys +2.22 % faces).
* **Speed.** Read → clean → largest component → thickness → budget → decimate → Taubin →
  geometry: 11.7 s for all 68 fragments on 4 threads, 1.11 s for the largest scan (1.34 M faces).
  The same span in the Python (Open3D with `OMP_NUM_THREADS=4`, timed with the same steps and
  without segmentation) is 33.65 s for the four terracotta scans against 3.52 s, and 35.23 s for
  synthetic 20 against 5.66 s — **3.1× to 9.6× faster**, with the widest margins where decimation
  dominates, as E1 predicted.
* **Determinism.** Two runs of `from_mesh_file` on all 68 fragments: identical bit for bit
  (vertices, faces, per-face arrays, `res`, `t`, `thick_mode`, `watertight`, `n_boundary`). The
  ray loop is `rayon` but collects by index; nothing else is parallel.

## 5. Why `t` is outside D §10.2's ±2 %, and what the right criterion is

Five of 68 fragments exceed ±2 % on `t` (Pot_A_Piece_04 +6.58 %, frag_019 −5.27 %, frag_010
+4.54 %, Pot_B_Piece_01 −3.72 %, Pot_G_Piece_05 −2.27 %) and twelve on `thick_mode`. Three
measurements say this is the estimator's sampling noise and not a defect of the port:

1. **Given the reference's own sample, the answer is the reference's answer.** Casting parry rays
   from the faces the reference sampled (`thick.idx`) reproduces `t` to within **3.1e-5 %** and
   `thick_mode` to within **2.5e-3 %** on all 48 fragments — bit-identical on 9 of them, the rest
   differing only because the bin centre is a multiple of `p90/120` and `p90` is one order
   statistic that a 8.8e-3 ray difference can move. parry disagrees with Embree on the hit
   *triangle* for at most 0.005 % of rays (grazing hits on shared edges, exactly what E3/E4
   measured) and never on whether the ray hits.
2. **The estimator moves that much on its own.** Running the Rust rays with five different
   `ChaCha8Rng` seeds moves `t` by up to **14.5 %** (Pot_A_Piece_07: 3.531…4.044). E1's control —
   Open3D against Open3D with only the seed changed — measured the same effect at up to 8.1 %.
   One histogram bin is 1.7 % to 5.7 % of `t`.
3. **The deviation is unbiased.** The reference's `t` falls inside the range of the five Rust
   seeds on 41 of 68 fragments; for an unbiased estimator the expected count is 68 · 4/6 ≈ 45
   (sd 3.9), so 41 is 1.1 standard deviations low — noise, not a shift. Expressed in the
   reference's own histogram bins, the worst deviation over all 68 fragments is **2.86 bins**.

The cause is visible in the histogram. Pot_A_Piece_04's filtered distances form a plateau, not a
peak: bins 42–47 hold 1313, 1457, 1292, 1434, 1384 and 934 hits out of 13 638. The mode picks the
tallest of six near-equal bins, and a different 20 000-face sample of the same 66 940 faces picks
a different one — 3.55 or 3.79, 6.6 % apart. Nothing but numpy's PCG64 stream reproduces the
reference's choice, and D §7 deliberately does not (PMC-9).

**Recommendation for D §10.2.** Replace the native thickness tolerance `±2 %` with
`max(2 %, 3 bins of the reference's histogram)`, or equivalently "within the reference's own
seed-to-seed spread". The bin width is derivable from the fixture (`p90/60` over the filtered
distances) and the harness already computes it. The injected tolerance ("same bin, or ±1 bin on a
count tie") needs no change — S3 meets it bit-exactly.

## 6. Deviations from D and R, all deliberate and all recorded

| where | deviation | why |
|---|---|---|
| D §4.1 `Fragment { thick: f32, thick_mode: f32 }` | stored as `f64` | R computes every `k·t` threshold in `f64`, and the OBB fallback is a genuine `f64`. The ray estimate is a `float32` *value*, which an `f64` holds exactly, so nothing is lost either way; `f64` is the strict superset. The cache (D §4.2) carries `thick` as metadata text, so no layout changes. |
| D §3 decimation row | `meshopt` 0.6.2 only, `baby_shark` dropped | E1's recommendation; unchanged by S3. |
| R §3.2 OBB fallback | PCA over all vertices, not over the convex hull | §2.4. Unreachable on the benchmark; documented in the function. |
| R §3.3.1 | colours are not smoothed | Open3D smooths vertex colours alongside the positions; the working mesh carries none into any later stage and R §11.4 writes the *original* cleaned mesh. |
| R §3.3.1 | a vertex with no neighbours keeps its position | Open3D divides by a zero total weight and produces NaN. R §3.1 and R §3.3 both remove unreferenced vertices before Taubin runs, so the case is unreachable in the pipeline; it only makes a hand-built mesh usable. |
| D §7 (unordered containers) | Taubin sums in ascending neighbour order | Open3D's order is a `std::unordered_set`'s. Cost measured: ≤ 1.7e-12 of one edge length (§3). |

`WorkingMesh` keeps D §4.1's `f32` vertices and `f32` `res`; everything up to the narrowing —
Taubin, `face_geometry`, `median_edge`, `ΣA` — runs in `f64` on the readers' `f64` coordinates,
which is what makes the injected comparisons exact. The narrowing costs 4.7e-5 % on `res` (§4).

## 7. Tests

`cargo test --workspace` is green: **89 unit tests** in `sherd-core` (61 before S3) and 5 new
integration tests, plus the S1/S2 suites unchanged.

* `mesh::adjacency` — every edge twice and three neighbours per face on a closed tetrahedron; an
  open mesh's boundary count; the fraction rule, not a count; a non-manifold hinge counted as
  boundary and paired in a chain; the `(fa, fb)` order checked against
  `sherd_refit.geometry.face_adjacency` on the two-triangle case (it returns `(1, 0)`, not
  `(0, 1)`, and so does this port); sorted unique vertex neighbours.
* `mesh::geometry` — normal/area/centroid of a right triangle, winding, the 1e-12 floor on a
  zero-area face; numpy's pairwise sum against four values taken from numpy 2.5.2; the median for
  both parities; `res` over unique edges.
* `mesh::taubin` — **the volume of `icosphere(2)` after three iterations must equal Open3D's**
  4.076543327369995 (from 4.047044679978849) to 1e-9 relative, and three λ-only iterations must
  shrink it by Open3D's 33.46 %; a noisy sphere's roughness drops by more than 40 % while its mean
  radius stays; a planar mesh stays planar; an isolated vertex does not become NaN.
* `mesh::decimate` — the budget clipped at both ends, `int()` truncation, and numpy's floor-first
  clip order; a mesh at or below the budget left byte-identical; a decimated mesh within the
  target with **every surviving vertex bit-identical to an input vertex**; `simplify` twice gives
  the same buffer.
* `fragment::thickness` — numpy's percentile on a known ten-value case; the mode as a bin centre
  and `argmax`'s first-maximum tie rule; a hollow subdivided box measuring its own 10-unit wall
  end to end (rays included); fewer than 100 hits refusing an estimate; a side-on hit dropped from
  the filtered mode; sampling without replacement, seeded, capped at the face count; the OBB
  fallback on a rotated 2 × 6 × 30 box; first hits and misses from `RayScene`.
* `crates/sherd-parity/tests/working_mesh_slab.rs` — the injected and native comparisons of §3–§4
  on the committed slab fixture, plus a bit-reproducibility test. 0.73 s in debug.

## 8. What S4 and phase 1b inherit

* `serde_json`'s `float_roundtrip` feature is now on and **must stay on** for the fixture reader.
* `fragment::thickness::RayScene` is a placeholder: D §6.2's shared BVH belongs in
  `crate::spatial`, and R §3.4's cone of seven rays wants the same structure. Moving it is a
  rename plus the `Fragment`-level cache (`bvh_full`, D §4.1).
* `sample_face_indices` (partial Fisher–Yates over `ChaCha8Rng`, with Lemire's unbiased range) is
  the only uniform-draw helper in the tree; R §3.5's draws want the same one, at which point it
  belongs in `crate::rng`.
* `Fragment` carries `area0` and `face_budget` beyond D §4.1's field list, because the fixture
  sink dumps `thick.target` as `{target, area0, faces0, target_faces}` and S4's `--dump-fixtures`
  needs both.
* `mesh::adjacency::face_adjacency` and `vertex_adjacency` are already the shapes R §3.4's
  majority filter, island removal and boundary refinement want.
