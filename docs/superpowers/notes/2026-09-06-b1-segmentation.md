# B1 — shell/fracture segmentation in Rust (R §3.4)

**Date:** 2026-09-06. Branch `rust-core`, step B1 of phase 1b (D §12).
**Reference:** `sherd_refit/fragment.py` at the fixture commit `9d4b9d3` (`segment_faces`,
`classify_faces`, `coarse_grid`, `refine_boundary`) and `sherd_refit/geometry.py`
(`ball_matrix`, `smoothed_normals`, `components_of`, `drop_small_components`), against
`docs/superpowers/specs/2026-09-06-algorithm-reference.md` (R) §1.3 and §3.4.
**Machine:** Apple M2 Pro (10 cores, 16 GB), macOS 15.6.1, rustc 1.97.0, Open3D 0.19, numpy 2.5.2.

## Result in one line

**Injected: exact.** On all 68 fragments of the eight fixture collections the port's labels are
*identical* to the reference's — area-weighted agreement `1.000000000` on every fragment, and the
same at every intermediate stage (raw vote, majority filter, island removal). D §10.2 asks for
≥ 0.995. **Native: 66 of 68 fragments pass** ≥ 0.97, median agreement 0.9935; the two that do not
are explained, measured and *not* a segmentation defect — both have a wall thickness that the
reference's own estimator moves by more than that much between random seeds (§5).

| mode | collections | fragments | checks | failed | worst / tolerance |
|---|---:|---:|---:|---:|---|
| injected | 8 | 68 | 884 | 0 | 3.4e-3 (a single vote count, §4) |
| native | 8 | 68 | 136 | 4 (2 fragments) | 4.08 (Pot_B_Piece_01, §5) |

## 1. What was implemented

`crates/sherd-core/src/fragment/segment.rs` — R §3.4 end to end, in the reference's order:

| R | step | where |
|---|---|---|
| §3.4.1 | `t/8` voxel grid, lowest-index face per voxel, nearest representative per face | `coarse_grid` |
| §3.4.2 | area-weighted smoothed normal `NS` over `t/3` balls of the representatives | `ball_normals` |
| §3.4.3 | the seven-ray 15° cone, `votes = 4 if res > 0.1·t else 5` | `classify_faces` |
| §3.4.4 | majority filter over `t/4` balls | `majority_filter` |
| §3.4.5–6 | edge adjacency, then `drop(frac, true, 0.5 t²)` and `drop(frac, false, 2.0 t²)` | `mesh::components::drop_small_components` |
| §3.4.7 | boundary growth against the shell's fixed reference normal, 25°, ≤ 60 passes | `refine_boundary` |
| §3.4.8 | `drop(frac, true, 0.5 t²)` once more | as above |

Supporting work, all of it named by D §6.2 and experiments E3/E4:

* `crates/sherd-core/src/spatial/bvh.rs` — `RayScene` moved here out of `fragment::thickness`
  (where it had been parked as "a placeholder home"), plus `closest_face` for the native label
  transfer. `parry3d` 0.30 without `TriMeshFlags::ORIENTED`, as E4 concluded.
* `crates/sherd-core/src/spatial/kdtree.rs` — `PointTree`, the one KD-tree wrapper the algorithm
  needs: `nearest` (R §3.4.1's `cKDTree.query`) and `within` (R's `ball_matrix`).
* `crates/sherd-core/src/mesh/components.rs` — `mask_components` and `drop_small_components` over
  the existing union–find and the existing `face_adjacency`.
* `Fragment.labels`, the `labels u8[m]` tensor of D §4.2, and `CACHE_VERSION` 1 → 2.
* `sherd-refit-rs segment` now prints a `fracture` column; `sherd-refit-rs parity --stage
  segmentation` is the new row of D §10.2.

Two deliberate departures, both already in R:

* **PMC-4.** `rep` comes out of an Open3D `unordered_map` in the reference and is sorted ascending
  here. The *map from a face to its representative* is what is read downstream, and it is compared
  directly (`rep face` below): identical on all 5 992 585 faces.
* **Summation order.** `query_ball_point(..., return_sorted=False)` hands the reference its
  neighbourhoods in an unspecified order; `PointTree::within` returns them ascending by index, so
  this port's sums are reproducible where the reference's are not (D §7). The cost is round-off,
  measured at ≤ 1.7e-6 degrees on `NS` (§4).

## 2. Two decisions that had to be got right, and how

**Open3D's voxel bucket.** `coarse_grid` calls `voxel_down_sample_and_trace(spacing, C.min−1,
C.max+1)` and takes `l[0]` of each trace list. Open3D computes `voxel = ⌊(p − min_bound)/voxel⌋`
per axis and `AccumulatedPointForTrace::AddPoint` appends indices in increasing `i`, so `l[0]` is
the lowest-indexed face in the voxel — an exact rule, not a hash artefact. Only the *order of the
voxels* is a hash artefact, and that is PMC-4. The port sorts `(voxel, face)` pairs instead of
hashing, which also removes any dependence on a hasher (D §7). Grid point counts match the
reference exactly on all 68 fragments (7 866 on `pieceA`, 102 473 on `frag_017`).

**numpy's NEP 50 in the vote.** In `classify_faces` the hit distance `dh` is a `float32` array and
`thick` a Python float, so `dh > 0.1·t` and `dh/t` are evaluated *in float32*: numpy 2 treats a
Python scalar as weak and casts it down. The dot product against the hit face's normal is the
other way round — `np.einsum` over `float64` normals and the `float64` ray directions. The port
does both exactly so (`min_hit = (0.1·thick) as f32`, `ratio = dh / (thick as f32)`,
`al` in `f64`). R §3.4.3 now says this; it did not before.

## 3. Injected parity — all eight collections

`sherd-refit-rs parity --fixtures DUMP --stage segmentation --injected --details`, run on the
reference's own working mesh, `t` and `res`.

| collection | fragments | checks | agreement (min) | fracture fraction | worst / tol |
|---|---:|---:|---|---|---|
| slab (committed) | 2 | 26 | 1.000000000 | bit-identical | 1.7e-6 |
| terracotta | 4 | 52 | 1.000000000 | bit-identical | 1.7e-6 |
| pot_A | 8 | 104 | 1.000000000 | bit-identical | 3.4e-3 |
| pot_B | 9 | 117 | 1.000000000 | bit-identical | 1.7e-6 |
| pot_C | 7 | 91 | 1.000000000 | bit-identical | 1.7e-6 |
| pot_G | 7 | 91 | 1.000000000 | bit-identical | 1.7e-6 |
| pot_H | 11 | 143 | 1.000000000 | bit-identical | 1.7e-6 |
| synthetic_20 | 20 | 260 | 1.000000000 | bit-identical | 1.0e-3 |

Every one of the following was **exactly** equal on all 68 fragments: `votes` (4 or 5),
`smooth_radius`, `boundary_angle`, `raw_fraction`, the face → representative map, the number of
grid points, and the agreement of the raw vote mask, the majority-filtered mask and the
post-island mask. The two quantities that are not exactly equal are in §4.

## 4. The only two residuals, both accounted for

**`NS` angle ≤ 1.708e-6 degrees** on every fragment — the summation order of §1. Three degrees of
freedom below anything the 15° cone or the 25° growth rule can notice.

**Vote counts differ on 3 faces out of 5 992 585** (41.9 M cone rays), one face each on
`Pot_A_Piece_06_Mesh`, `frag_012` and `frag_018`. This is exactly E4's measured ray behaviour:
7.87 M cone rays gave 2 hit/miss disagreements against Open3D and 5 differing primitive ids, all
of them grazing hits at a silhouette or a shared edge. **Not one of the three changed a label** —
the agreement, the raw mask and every later mask are all still 1.000000000, because a vote moving
between (say) 6 and 7 does not cross the threshold of 5.

## 5. Native parity — and the two fragments that fail

`Fragment::from_mesh_file` from the source file, then every face of the *reference's* working mesh
looked up on the port's by closest point (`RayScene::closest_face`) and the two labels compared,
weighted by the reference face's area. That is D §10.2's "sample points on the Python working mesh
and label each by its nearest face on each mesh", with the reference's own faces as the quadrature
points — one point per face, weighted by area, and no RNG in the comparison.

| collection | fragments | min agreement | median | max | failures |
|---|---:|---:|---:|---:|---:|
| slab | 2 | 0.9980 | — | 0.9998 | 0 |
| terracotta | 4 | 0.9877 | 0.9896 | 0.9917 | 0 |
| pot_A | 8 | 0.9920 | 0.9950 | 0.9993 | 0 |
| pot_B | 9 | **0.9142** | 0.9921 | 0.9970 | **1** |
| pot_C | 7 | 0.9934 | 0.9995 | 1.0000 | 0 |
| pot_G | 7 | 0.9833 | 0.9949 | 1.0000 | 0 |
| pot_H | 11 | 0.9852 | 0.9987 | 1.0000 | 0 |
| synthetic_20 | 20 | **0.9687** | 0.9899 | 0.9948 | **1** |
| **all** | **68** | **0.9142** | **0.9935** | **1.0000** | **2** |

`Pot_B_Piece_01_Mesh` (0.9142, fracture 0.2963 against 0.2147) and `frag_010` (0.9687, 0.2120
against 0.2375) are outside the ≥ 0.97 / ±0.02 gate. **The gate was not touched.** What the cause
is was measured twice:

**Isolation.** Feed the port's *own* native `t` into the *reference's own* working mesh and run the
injected comparison — everything else the reference's:

| fragment | reference `t` | port `t` | agreement, native | agreement, reference mesh at the port's `t` |
|---|---:|---:|---:|---:|
| `Pot_B_Piece_01_Mesh` | 5.6226 | 5.4132 (−3.7 %) | 0.914161 | 0.914160 |
| `frag_010` | 6.2671 | 6.5516 (+4.5 %) | 0.968677 | 0.970653 |

So 100 % of the first gap and 94 % of the second is the wall thickness. Nothing else about the
port's segmentation is involved: at the reference's `t` the same code agrees to the last bit (§3).

**Which `t` is right.** The reference's `estimate_thickness` was run with seeds 0–11 and nothing
else changed (the pipeline uses seed 0):

| fragment | seeds 0–11 | seed 0 (the fixture) | median | port |
|---|---|---:|---:|---:|
| `Pot_B_Piece_01_Mesh` | 5.6226 5.3869 5.3805 5.4175 5.3981 5.4367 5.3985 5.4218 5.4245 5.3870 5.3676 5.3795 | **5.6226 — the maximum of the twelve** | 5.3983 | 5.4132 (**+0.3 %** of the median) |
| `frag_010` | 6.2671 6.3059 6.2908 6.5703 6.5602 6.6675 6.3156 6.5457 6.3491 6.3223 6.3299 6.3002 | **6.2671 — the minimum of the twelve** | 6.3261 | 6.5516 (+3.6 %, inside the cloud; 3 of 12 seeds are higher) |

On `Pot_B_Piece_01` the reference's seed-0 value is 4.2 % above *every other seed* and the port's
value sits 0.3 % from the estimator's centre: the port's `t` is the better estimate, and the
fixture's is the outlier. On `frag_010` both values are inside a 6.4 % seed cloud.

This is finding F1 of the phase-1a verification arriving where D §10.2 said it would: "`t` is the
unit of every threshold in R §1.2 … No row of this table can absorb that." **The segmentation row
is the first gate the thickness spread has actually broken**, which turns D §10.2's conditional
argument for E6 (replicating numpy's PCG64 so the port draws the *same* 20 000 faces, ≈ 2 days)
into an argument from evidence. Recorded as an open issue below, not worked around.

## 6. Timings

`sherd-refit-rs segment` on the four terracotta scans (75 k–156 k working faces), segmentation
only, from the `seconds` field of the per-fragment log:

| fragment | faces | 1 thread | 10 cores, one fragment | 10 cores, 4 fragments at once |
|---|---:|---:|---:|---:|
| FY234007_reduced | 147 770 | 1.42 s | 0.23 s | 0.37 s |
| FY234021_reduced | 122 446 | 1.17 s | — | 0.23 s |
| FY234094_reduced | 155 638 | 1.49 s | — | 0.28 s |
| FY234104_reduced | 77 200 | 0.72 s | — | 0.17 s |

The Python's `segment_faces` on the same meshes, using all cores throughout (scipy `workers=-1`,
Open3D's multithreaded ray casting): **1.06 s** for FY234007 (147 304 faces) and **0.46 s** for
FY234104 (75 526). So the port is **4.6× faster wall-clock** on the same machine, and within 1.4×
of Python's all-core time on *one* thread — which is the expected shape, since E4 measured
`parry3d` at 2.6–3.0× slower than Embree per ray and the port makes that back on the KD-tree side.

Whole collection, cold cache, `segment input/test_fragments_1/fragments`: **1.86 s wall** for four
scans of 0.7–1.3 M input triangles, segmentation included.

## 7. Tests

* `fragment::segment` — a closed 36 × 36 × 6 slab at 0.5 per edge (12 edges across the wall, which
  is what R §3.3's budget aims for): the four sides must come out fracture and the two large faces
  shell. They do, with **zero** faces on the wrong side and a fracture fraction equal to the
  sides' geometric share; the vote test asserts 7 of 7 on an interior shell face and 0 of 7 on a
  rim face. Plus the grid rule (lowest index per voxel, `rep` ascending), the trace, and the
  area-weighted agreement.
* `spatial::kdtree` — the ball is inclusive at `d = r` and comes back ascending by index;
  `spatial::bvh` — first hit, miss, and the closest face from inside and outside.
* `mesh::components` — island removal on hand-built masks.
* `sherd-parity` — the injected and native stages on the committed `fixtures/slab`, in
  `stages_slab.rs` (every stage, both modes) and in the module's own tests.
* `types::FaceLabel::from_u8`, and a cache whose `labels` tensor does not describe its face list is
  refused rather than read.

Gates: `cargo fmt --all --check` clean, `cargo clippy --workspace --all-targets --locked
-- -D warnings` clean, `cargo test --workspace --locked` **171 passed / 0 failed / 1 ignored**
(160 before this step), and the ignored terracotta determinism test passes — two `segment` runs on
the real scans still produce byte-identical caches, now with the labels in them.

## 8. Open issues

1. **The native segmentation gate fails on 2 of 68 fragments, and the cause is upstream `t`**
   (§5). Not a segmentation defect and not worked around: the gate stands at D §10.2's ≥ 0.97 /
   ±0.02 and the two fragments are a known, measured failure. The fix is E6 (replicate PCG64 for
   R §3.2's face sample, ≈ 2 days), which would make the port draw the reference's own 20 000
   faces and remove the whole class. Until then, `--stage segmentation` exits non-zero on `pot_B`
   and `synthetic_20` in native mode. Whether to spend those two days is the team's call
   (D §13); the argument for it is now evidence rather than anticipation.
2. **`SegParams` is not reachable from the command line.** R §1.3's knobs are all defaults and
   `Fragment::from_mesh_file` hard-codes `SegParams::default()`. The reference exposes them only
   through `tools/eval_segmentation.py`, so nothing is lost yet; when that tool is ported the
   parameter has to be threaded through `Fragment` and the cache key.
3. **The segmentation preview** (`--no-preview`'s other half, R §11.5) still has no Rust side; it
   belongs with the renderer in phase 1d.
