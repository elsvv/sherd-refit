# sherd-refit — frozen algorithm reference

**Date:** 2026-09-07. **Reference implementation:** `sherd_refit/*.py` at commit `895a948`
(branch `rust-core`), running on Open3D 0.19.0, numpy 2.5.2, scipy ≥ 1.11, Python 3.12.
The document was frozen at `9d4b9d3`; the one algorithm change made since is `fbfebca`, §3.2's
deterministic ray set (task T1, dated addendum at the end of §12). Two later commits changed the
fixture sink and not the pipeline: `9347cfe` added the sample uniforms, and `895a948` hoisted the
two NMS walk orders into the dump (§12.1's C1 addendum). **The parity fixtures are the seven
`output/fixtures` sets regenerated from `895a948`, and the committed `fixtures/slab/dump` from
`9cbcbbc`** — `git diff 9cbcbbc 895a948 -- sherd_refit/` is empty, so both carry the same
reference. Each dump's `manifest.json` records which commit wrote it, and all eight say
`dirty: false` (D §10.1).
**Purpose:** the algorithm exactly as the Python computes it, stage by stage, so that the Rust
port can be implemented and verified from this document alone. Where the design spec
(`2026-09-05-fracture-reassembly-design.md`) and the code differ, the code is authoritative and
this document follows the code.

Items marked **PMC** ("port may change, must re-verify") are places where the Python does
something for historical or library-specific reasons. The port may do them differently, but every
PMC change must be re-verified against the parity gates in §13.

---

## 0. Conventions

| symbol | meaning |
|---|---|
| `V`, `F` | vertices (n×3 float64), triangles (m×3 int64, counter-clockwise, outward normals assumed) |
| `FN`, `A`, `C` | unit face normals, face areas, face centroids of the working mesh |
| `t` | wall thickness of a fragment (units of the scan) |
| `res` | median length of the unique edges of the working mesh |
| `t_pair`, `res_pair` | `min(t_A, t_B)`, `max(res_A, res_B)` for a pair |
| `T` | 4×4 rigid transform; `apply(T, P) = P·Rᵀ + τ` with `R = T[:3,:3]`, `τ = T[:3,3]` (points are rows) |
| candidate `T` | maps fragment **B** into **A**'s frame: `p_A = R p_B + τ` |
| `⌊x⌋` | floor; `median` is numpy's (mean of the two middle values for even counts); `percentile` uses linear interpolation |
| `rng(seed)` | numpy `default_rng(seed)` = PCG64 with `SeedSequence(seed)`; §10 lists every draw |

Every distance threshold is `max(k·t_pair, m·res_pair)` (§1.2). All distances are Euclidean.

Angles compare unit vectors by dot product; "agree" means `dot > 0.7` (≈ 45.6°) unless stated.

Float precision in the reference: numpy/scipy in float64; Open3D ICP point clouds in float64;
Open3D `RaycastingScene` (ray casts, unsigned/signed distances) in **float32** (vertices and
queries are cast to float32).

---

## 1. Parameters

### 1.1 `Params` (matching) — every field, default, meaning

| field | default | unit | used in |
|---|---|---|---|
| `dihedral_tol` | 25.0 | degrees | §5.1 hypotheses |
| `coarse_delta` / `coarse_res` | 0.15 / 2.3 | t / edges | §5.2 coarse score radius |
| `coarse_points` | 60 | count | §5.2 |
| `stage1` | 250 | count | §5.3 poses kept for breakline ICP |
| `stage2` | 10 | count | §5.5 candidates for full ICP |
| `stage1_delta` / `stage1_res` | 0.06 / 0.9 | t / edges | §5.4 re-score radius |
| `tight_delta` / `tight_res` | 0.01 / 0.15 | t / edges | §6.1 tight contact |
| `facing_delta` / `facing_res` | 0.3 / 1.0 | t / edges | §6.1 facing window |
| `max_gap` / `gap_res` | 0.03 / 0.45 | t / edges | §6.5 gap limit |
| `seam_delta` / `seam_res` | 0.12 / 1.8 | t / edges | §6.2 seam |
| `near_delta` / `near_res` | 0.5 / 4.0 | t / edges | §6.3 continuity |
| `pen_delta` / `pen_res` | 0.06 / 0.9 | t / edges | §6.4 penetration depth |
| `nms_delta` | 0.5 | t (no floor) | §5.3 NMS translation radius |
| `icp_delta` / `icp_res` | 0.04 / 0.6 | t / edges | §5.4–5.6 ICP ladder stretch |
| `min_tight` | 0.25 | fraction | §6.5 |
| `max_pen` | 0.005 | fraction | §6.5, §7 |
| `min_seam` | 3.0 | t | §6.5 |
| `min_cont_n` | 0.8 | cosine | §6.5 |
| `early_reject_tight` | 0.0 (off) | fraction | §5.6 |
| `stage1_floor` | 0.0 (off) | fraction | §5.4 |
| `thick_ratio` | 2.5 | ratio | §4.1 |
| `screen_top_k` / `screen_points` / `screen_min_pairs` | 0 (off) / 150 / 200 | | §4.3 |
| `second_pass_top` / `second_pass_stage1` / `second_pass_stage2` | 0 (off) / 400 / 40 | | §8.1 |
| `margin_points` | 6000 | count | §3.5.6 |
| `reg_points` | 6000 | count | §3.6 |
| `surface_points` | 20000 | count | §3.5.1 |
| `macro_inner` / `macro_outer` | 0.15 / 0.60 | t | §3.5.4 |
| `brk_voxel` | 0.5 | t | §3.5.5 |
| `frac_per_t2` / `min_frac_points` / `max_frac_points` | 150.0 / 5000 / 12000 | per t² / count | §3.5.2 |
| `seed` | 0 | | §10 |

### 1.2 Pair scales (`Scales.for_pair`)

```
f(k, m) = max(k · t_pair, m · res_pair)
coarse = f(0.15, 2.3)      stage1 = f(0.06, 0.9)     tight = f(0.01, 0.15)
facing = f(0.3, 1.0)       gap    = f(0.03, 0.45)    seam  = f(0.12, 1.8)
near   = f(0.5, 4.0)       pen    = f(0.06, 0.9)     nms   = 0.5 · t_pair
icp    = f(0.04, 0.6) / (0.04 · t_pair)            (≥ 1, a dimensionless stretch)
icp_dist(k) = k · t_pair · icp                     (one rung of the ICP ladder)
limits: gap_limit = gap / t_pair,  tight_delta = tight / t_pair   (reported per candidate)
```

### 1.3 Segmentation (`SegParams`) and module constants

| name | value | meaning |
|---|---|---|
| `votes` / `votes_coarse` / `coarse_at` | 5 / 4 / 0.1 | cone rays out of 7 that must hit the far wall; 4 when `res > 0.1·t` |
| `smooth_res` | 0.0 | smoothing radius is `max(t/3, smooth_res·res)` = `t/3` |
| `smoothed_hit_normal` | False | judge the hit face by its raw normal |
| `boundary_angle` / `boundary_angle_auto` | 25° / False | shell-growth angle |
| `FACES_PER_T2` | 600 | working-mesh face budget per t² of surface |
| `MIN_FACES` | 50000 | lower clamp of the budget |
| `target_faces` (CLI `--target-faces`) | 200000 | upper clamp of the budget |
| `CACHE_VERSION` | 7 | fragment cache format version |
| `MESH_EXT` | `.ply .obj .stl .off` (case-insensitive) | accepted inputs |
| `MD_LRU_MAX` | 6 | per-worker `MatchData` cache entries (performance only) |
| cone half-angle / rays | 15° / 7 | §3.4.3 |

### 1.4 CLI → `Params` mapping

`--candidates → stage2`, `--frac-density → frac_per_t2`, `--second-pass-candidates →
second_pass_stage2`; every other flag maps to the field of the same name with `-` → `_`.
`--target-faces`, `--workers`, `--threads`, `--no-preview`, `--no-refine`, `--no-meshes`, `-v`
are pipeline options, not `Params`. `keep_per_pair` = 5 (candidates returned per pair, not a flag).

---

## 2. Collection input

1. `find_meshes(dir)`: all files with an extension in `MESH_EXT` (either case), `sorted(set(...))`
   by full path. Fewer than two files → error exit.
2. Names (`fragment_names`): the file stem; if two stems collide, the full basename with `.`
   replaced by `_`.
3. Collection order = sorted file order. It fixes the pair order (§4.1) and the group seeding order.

---

## 3. Per-fragment preprocessing

All of §3 is a pure function of (file, `target_faces`, `Params` sampling fields, `seed`) and is
cached (§3.7). One RNG stream is used: `rng_md = rng(p.seed)` for the match arrays of §3.5.
§3.2's wall thickness draws nothing — its ray set is fixed by the mesh.

### 3.1 Load and clean

1. Read the mesh with vertex colours if present (colours are carried to the outputs only).
   Zero triangles → error.
2. `remove_duplicated_vertices` (exact coordinate equality), `remove_degenerate_triangles`
   (a triangle with a repeated vertex index), `remove_unreferenced_vertices`.
3. `n_orig_vertices`, `n_orig_faces` recorded after step 2.
4. **Largest component:** connected components of triangles by **shared edge** (Open3D
   `cluster_connected_triangles`; two triangles sharing only a vertex are not connected). Keep
   the component with the most triangles; drop the rest; remove unreferenced vertices.
   The result is `(V0, F0)`.

### 3.2 Wall thickness `t`

Inputs: `FN0, A0, C0 = face_geometry(V0, F0)`; a raycasting scene over `(V0, F0)` in float32.

```
stride = max(1, ceil(len(C0) / 300000))                                 # THICK_FACES = 300000
idx  = arange(0, len(C0), stride)                                       # face indices
dvec = −FN0[idx]
origin = C0[idx] + dvec · 1e-3                                           # PMC-1
(d, prim) = first hit along (origin, dvec)          # d = inf, prim = 0xFFFFFFFF on a miss
ok   = isfinite(d) & (prim < n_faces)
if ok.sum() < 100: t = None (fallback below)
raw  = hist_mode(d[ok])
far  = d[ok][ FN0[prim[ok]] · dvec[ok] > 0.7 ]      # hit face looks back along the ray
t    = hist_mode(far) if len(far) ≥ 100 else raw
thick_mode = raw
hist_mode(x): 60 equal bins on [0, percentile(x, 90)]; value = centre of the first bin with the maximal count
```

Fallback when `t` is None or ≤ 0: `t = min(extent of the PCA oriented bounding box) / 10`, where
the OBB is Open3D's `get_oriented_bounding_box()` = `OrientedBoundingBox::CreateFromPoints`: the
**convex hull** of the vertices, a PCA over the *hull's* vertices, and the extents of the points
along those three axes. Running the PCA over every vertex instead is a different box, and not by a
little: measured against Open3D 0.19, `min(extent)` comes out 3.0 % low on `fixtures/slab`'s
`pieceA`, 8.1 % low on `pieceB` and 8.8 % high on a terracotta scan (finding F6). Since `t` is the
unit of every threshold in §1.2, that error would move all of them.
If `thick_mode` is unavailable it is set to `t`. (`thick_mode > 1.15·t` only produces a log line.)

`face_geometry`: `n = (V[F1]−V[F0]) × (V[F2]−V[F0])`; `FN = n/max(|n|, 1e-12)`; `A = |n|/2`;
`C = mean of the three vertices`.

**The ray set is deterministic, and it was not always.** Until commit `fbfebca` the faces came
from `rng_pre.choice(len(C0), min(20000, len(C0)), replace=False)` with the seed hard-coded to 0,
and that sample was the estimate's largest error term rather than its cost saving: re-running
`estimate_thickness` with seeds 0–11 and nothing else changed moved `t` by up to 6.8 % of the
seed-0 value on `Pot_A_Piece_04_Mesh` (3.554 at seed 0 against 3.774–3.795 at seeds 1–11) and
5.8 % on `frag_019`, because a fragment whose filtered distances form a plateau rather than a peak
puts several near-equal bins in contention and `argmax` picks by a count that another sample
reorders. Since `t` is the unit of every threshold of §1.2, that spread moved nine thresholds for
every pair the fragment took part in.

The fix was to take the randomness out of the *algorithm* rather than to reproduce numpy's PCG64
in the port: every face casts a ray when there are at most 300 000 of them, and a fixed stride
picks them otherwise. Nothing else about the estimator changed — the same origin offset, the same
`> 0.7` filter, the same 60 bins over `(0, p90]`, the same unweighted per-face mode — and the
value it now returns lands on the median of the old estimator's seed cloud rather than on any one
draw of it (`Pot_A_Piece_04_Mesh` 3.7844 against a 12-seed median of 3.7852; `Pot_B_Piece_01_Mesh`
5.3984 against 5.3983; `frag_019` 8.1015 against 8.1196). The cost is bounded by the stride:
0.40 s for the 1.23 M-face terracotta scan. D §10.2's native tolerance on `t` is ±2 % again as a
result, and there is no `rng_pre` any more.

### 3.3 Working mesh

```
target = clip( 600 · ΣA0 / t², 50000, target_faces )
if n_faces(F0) > target:
    mesh = quadric_decimation(mesh, target)        # PMC-2: Open3D Garland–Heckbert, boundary_weight 1, no error cap
    remove_degenerate_triangles; remove_duplicated_vertices; remove_unreferenced_vertices
mesh = taubin(mesh, iterations=3, λ=0.5, μ=−0.53)  # §3.3.1
remove_degenerate_triangles; remove_unreferenced_vertices
(V, F) = mesh
watertight, n_boundary = closed_enough(F)          # §3.3.2
FN, A, C = face_geometry(V, F)
res = median over the UNIQUE undirected edges of F of |V[e0] − V[e1]|
scene = raycasting scene over (V, F) in float32     # used by §3.4 and §6.4
```

**3.3.1 Taubin smoothing (Open3D semantics, verified).** One iteration = one Laplacian step with
`λ = 0.5` followed by one with `μ = −0.53`. A Laplacian step with factor `s` moves every vertex
(boundary vertices included) to

```
v' = v + s · ( Σ_j w_j v_j / Σ_j w_j − v ),   w_j = 1 / (|v − v_j| + 1e-12),  j over the vertex's edge neighbours
```

using the positions from before the step (Jacobi, not Gauss–Seidel). The `1e-12` is Open3D's, not
a rounding of the formula: `FilterSmoothLaplacianHelper` computes `weight = 1. / (dist + 1e-12)`,
so a coincident neighbour gives a weight of 1e12 rather than an infinity, and every other weight is
a hair below `1/|v − v_j|`. It was missing from this document until the phase-1a verification
(finding F4) and it is *not* a PMC item: the port reproduces it, and the injected Taubin check
lands within 1.7e-12 of one edge length of Open3D's own mesh.

Open3D smooths the vertex normals and vertex colours by the same recurrence. **The Rust port does
not** (finding F8), which is consistent with this section — they are not used downstream, R §11.4
writes the *cleaned original* mesh rather than the working mesh, and the segmentation of §3.4 uses
face normals recomputed from the smoothed positions. The one consequence to state out loud: the
port's cached working mesh therefore carries **unsmoothed** vertex colours where the reference's
carries smoothed ones. Nothing in R reads them; anything later that writes the *working* mesh with
colour would have to.

**PMC-3:** the port may use uniform weights only if re-verified; inverse-distance weights are what
the reference uses.

**3.3.2 `closed_enough`.** Count the unique undirected edges of `F` and how many of them are used
by a number of faces ≠ 2 (`n_boundary`, which also counts non-manifold edges). `watertight =
n_boundary ≤ 0.002 · n_unique_edges`. A non-watertight fragment gets no penetration test (§6.4).

### 3.4 Shell / fracture segmentation (`segment_faces`)

Output: boolean `frac` per working-mesh face (True = fracture). Steps, in order:

**3.4.1 Grid.** `rep, near = coarse_grid(C, t/8)`: voxel-downsample the face centroids with voxel
`t/8` (Open3D `voxel_down_sample_and_trace`, `min_bound = C.min − 1`, `max_bound = C.max + 1`;
a point's voxel is `⌊(c − min_bound)/voxel⌋` per axis); `rep` = for each occupied voxel the face
with the **lowest index** in it; `near[i]` = index into `rep` of the representative whose centroid
is nearest to `C[i]` (KD-tree). The order of `rep` is a hash-map order (**PMC-4**: the port sorts
`rep` ascending; only `near`-indexed values are used downstream, so results are unchanged).

**3.4.2 Smoothed normals.** `radius = t/3`. For each representative `r`: `NS_g[r]` = normalised
`Σ_{faces f with |C[f] − C[rep[r]]| ≤ radius} A[f]·FN[f]` (zero vector if the sum is zero).
`NS = NS_g[near]` (per face).

**3.4.3 Shell test (`classify_faces`).** `votes = 4 if res > 0.1·t else 5`. For each face `i`
with `n = NS[i]`:

```
a  = [1,0,0] if |n_x| < 0.9 else [0,1,0]
e1 = normalise(n × a);  e2 = n × e1
directions k = 0..6:  d_0 = −n;  d_k = −cos(15°)·n + sin(15°)·(cos φ_k e1 + sin φ_k e2),  φ_k = 2π(k−1)/6
origin = C[i] − FN[i]·1e-3                                   # PMC-1 (raw face normal)
for each k: (dh, prim) = first hit of (origin, d_k) on the working-mesh scene
    ok   = isfinite(dh) & (prim < n_faces) & (dh > 0.1·t)
    hit  = ok & (0.5 < dh/t < 1.8) & (FN[prim] · d_k > 0.7)    # far wall seen from behind
good[i] = number of k with hit
shell[i] = good[i] ≥ votes;  frac = ¬shell
raw_fraction = ΣA[frac] / ΣA                                   # diagnostic only
```

**Dtypes, and they are load-bearing.** `dh` is a `float32` array (Open3D's `t_hit`) and `t` a
Python float, so under numpy 2's NEP 50 the scalar is *weak*: `dh > 0.1·t` and the two `dh/t`
comparisons are evaluated in **float32**, with `0.1·t` and `t` cast down first. The
`hit_normals[prim] · d_k` test is the other way round — `np.einsum` over the `float64` face
normals and the `float64` direction array, not over the `float32` rays that were handed to the
scene. Origin and direction are computed in `float64` and narrowed once
(`np.concatenate([...]).astype(np.float32)`). A port that evaluates the window in `float64` moves
faces across it. (Step B1; found by reading the code, not by a failing gate.)

**3.4.4 Majority filter.** With `Wm` = membership of faces within `t/4` of each representative:
`frac_g[r] = Σ_{f∈ball(r)} A[f]·frac[f] > 0.5 · Σ_{f∈ball(r)} A[f]`; `frac = frac_g[near]`.

**3.4.5 Face adjacency.** `fa, fb, ke = face_adjacency(F)`: list every directed edge of every
face as a sorted vertex pair; lexsort all `3m` pairs by (v0, v1) with a **stable** sort; every two
consecutive equal pairs give one adjacency `(fa, fb)` (the face earlier in the sorted order first)
and the shared edge `ke = (v0, v1)`. Edges used by 3+ faces yield only consecutive pairs.

**3.4.6 Island removal.** `drop_small_components(mask, target, min_area)`: connected components
of the faces with `mask == target` over adjacency edges whose both faces are in that set; every
component with `Σ A < min_area` is flipped to `¬target`. Apply:
`frac = drop(frac, True, 0.5·t²)` then `frac = drop(frac, False, 2.0·t²)`.

**3.4.7 Boundary growth (`refine_boundary`, angle 25°).**

```
shell0 = ¬frac                              # after 3.4.6; the fixed reference
ref_g[r] = unit area-weighted mean of FN over shell0 faces within t/2 of C[rep[r]]
has_ref_g[r] = that ball is non-empty;   ref = ref_g[near];  has_ref = has_ref_g[near]
repeat up to 60 passes:
    cand = unique( { fa where frac[fa] ∧ ¬frac[fb] } ∪ { fb where frac[fb] ∧ ¬frac[fa] } )   # fracture faces adjacent to shell
    if cand is empty: stop
    flip = has_ref[cand] ∧ (FN[cand] · ref[cand] > cos 25°)
    if no flip: stop
    frac[cand[flip]] = False
```

**3.4.8** `frac = drop(frac, True, 0.5·t²)` once more. Done.

`fracture_area = ΣA[frac]`, `area = ΣA`.

### 3.5 Match arrays (`match_arrays(fr, t, …)`, RNG `rng_md = rng(seed)`)

Computed at the fragment's own `t` in preprocessing and cached; recomputed at `t_pair` for the
thicker fragment of a pair (§4.2). Draws from `rng_md` happen in exactly this order.

**3.5.1 Surface samples.** `S, sp = sample_on_faces(all faces, 20000)`:

```
sample_on_faces(mask, n):
    idx = faces with mask;  p = A[idx] / ΣA[idx]
    pick = idx[ rng.choice(len(idx), n, p=p) ]      # n uniforms u_i; pick_i = searchsorted(cumsum(p)/cumsum(p)[-1], u_i, 'right')
    u = rng.random(n); v = rng.random(n); where u+v > 1: (u,v) ← (1−u, 1−v)
    P = V[F[pick,0]] + u·(V[F[pick,1]] − V[F[pick,0]]) + v·(V[F[pick,2]] − V[F[pick,0]])
    return P (n×3), pick (face index per point)
```

**3.5.2 Fracture samples.** `n_frac = clip(⌊150 · fracture_area / t²⌋, 5000, 12000)`;
`Pf, fp = sample_on_faces(frac, n_frac)`.

**3.5.3 Breakline points.** `cross = frac[fa] ≠ frac[fb]`; `P_brk = (V[ke[cross,0]] + V[ke[cross,1]])/2`
in adjacency order. `brk_tree` = KD-tree over `P_brk` (None when empty).

**3.5.4 Macro normals (annulus 0.15 t – 0.60 t).** For `mask ∈ {¬frac (shell), frac (fracture)}`:

```
macro(mask):
    dist[f] = distance from C[f] to the nearest breakline point (all of P_brk)
    far  = { f : mask[f] ∧ dist[f] ≥ 0.15·t }
    n[q] = unit( Σ_{f∈far, |C[f]−P_brk[q]| ≤ 0.60·t} A[f]·FN[f] )      # zero vector if empty
    bad  = |n[q]| < 0.5                                               # i.e. the annulus ball was empty
    n[bad] = unit( Σ_{f: mask[f], |C[f]−P_brk[q]| ≤ 0.60·t} A[f]·FN[f] )   # fallback: whole neighbourhood
ns = macro(¬frac);  nf = macro(frac)
f  = nf − (nf·ns) ns;  f = f / max(|f|, 1e-9)
valid = |ns| > 0.5 ∧ |nf| > 0.5 ∧ |ns × f| > 0.5
```

If there are no breakline points, `ns = nf = 0`.

**3.5.5 Hypothesis subset.** Voxel-downsample `P_brk` with voxel `0.5·t` (same Open3D call and
bounds convention as §3.4.1 — the code's comment there says `voxel t/3` and is stale, the code
reads `brk_voxel · t` with `brk_voxel = 0.5`); `sub` = lowest-index point per occupied voxel in
hash-map order (**PMC-4**: port sorts ascending — this changes hypothesis order and therefore
tie-breaking in §5.2–5.3; must re-verify. The *bucket* rule is exact, as §3.4.1 records, and the
sorted subsets were measured equal entry for entry on all 66 fixture fragments in step B2);
`brk_sub = sub[valid[sub]]`.

**3.5.6 Shell margin.** `d_brk[i]` = distance from `S[i]` to the nearest breakline point (inf if
none). `margin = ¬frac[sp] ∧ (d_brk > 0.12·t) ∧ (d_brk < 1.5·t)` (**PMC-5**: these two are not
resolution-floored; both bounds are strict, and a fragment with no breakline has `d_brk = ∞`,
which fails the outer test, so it has no margin at all). `margin_idx =
sort(rng.choice(where(margin), 6000, replace=False))` if more than 6000, else `where(margin)` —
which draws nothing at all when the margin already fits.

Stored arrays (`MD_ARRAYS`): `S (n_s×3), sp (int32), Pf, fp (int32), brk_P, brk_ns, brk_nf,
brk_f, brk_sub (int32), margin_idx (int32)` plus the parameter dict
`{t, seed, surface_points, frac_per_t2, min_frac_points, max_frac_points, margin_points,
macro_inner, macro_outer, brk_voxel}`.

### 3.6 Runtime `MatchData` (derived, not cached)

```
SN = FN[sp];  Nf = FN[fp];  S_pen = S
brk_t   = ns × f                                   # tangent
brk_dih = degrees( arccos( clip(ns·nf, −1, 1) ) )  # per breakline point
Pm = S[margin_idx];  Nm = SN[margin_idx]
pc_reg:  nf = |Pf|, nm = |Pm|;  if reg_points > 0 and nf+nm > reg_points:
             nf' = max(1, round(nf·reg_points/(nf+nm))), nm' = max(0, reg_points − nf')   else nf' = nf, nm' = nm
         points = Pf[:nf'] ++ Pm[:nm'],  normals = Nf[:nf'] ++ Nm[:nm']        (prefixes, in sample order)
pc_frac: (Pf, Nf);   pc_brk: (P_brk[brk_sub], ns[brk_sub]);   pc_brk_full: (P_brk, ns)
brk_tree: KD-tree over P_brk;   tree_margin: KD-tree over Pm;   tree_frac: KD-tree over Pf (built, unused)
has_frac = |Pf| > 0;   frac_area = fracture_area
frac_scene: raycasting scene over the fracture faces only (V, F[frac]) in float32, built on first use
```

`round` is numpy's round-half-to-even.

### 3.7 Fragment cache (`<out>/cache/<name>.npz`)

Keys: `name, path (absolute), V (float64), F (int64), frac (bool), thick, res, watertight,
n_orig_vertices, n_orig_faces, target_faces, thick_mode, cache_version (=7), mtime (source file
mtime)`, `mdp_<param>` for each match-array parameter, `md_<array>` for each of `MD_ARRAYS`.

Validity (`cache_valid_for`): same absolute path, file exists, `|mtime_cached − mtime_file| < 1 s`,
same `target_faces`, `cache_version == 7`, same name. If valid but the `mdp_*` differ from the
current parameters (at the fragment's own `t`), only the match arrays are recomputed.

`Fragment.stats()` (report): `name, faces, orig_faces, orig_vertices, thickness, thickness_mode,
resolution, watertight, extent (V.max − V.min per axis), area, fracture_area_fraction`.

---

## 4. Pairs

### 4.1 Enumeration

Collection median `t_med = median(t_i)` and `res_med` are computed for the report/log only
(a fragment with `|t_i/t_med − 1| > 0.4` is flagged "differs"; nothing is decided by it).

Pairs `(a, b)` in `itertools.combinations(names, 2)` order (collection order, `a` before `b`).
A pair is **skipped** when `t_a/t_b > 2.5` or `< 1/2.5`. Skipped pairs produce no candidates.

### 4.2 Pair data

`t_pair = min(t_a, t_b)`, `res_pair = max(res_a, res_b)`, `Scales` from §1.2. Both `MatchData`
are built at `t_pair`: the fragment whose own `t == t_pair` uses its cached arrays; the other
recomputes §3.5 at `t_pair` with a fresh `rng(seed)` (identical to a from-scratch build).

### 4.3 Partner screening (off by default; `screen_top_k > 0` and `n_pairs ≥ screen_min_pairs`)

Each fragment's `MatchData` at its **own** `t`. For a pair: `ia = cap(A.brk_sub, 150)`,
`ib = cap(B.brk_sub, 150)` with `cap(idx, n) = sort(rng(seed).choice(idx, n, replace=False))`
if `len(idx) > n`; hypotheses (§5.1) over `(ia, ib)`; coarse score (§5.2) with `pool = ib` and a
fresh `rng(seed)`; the pair's score is `max` (0 if no breakline or no hypotheses).
`top_partners`: for every fragment keep its `k` best-scoring partners (ties by partner name);
a pair is kept if either endpoint keeps it. Pairs not kept are dropped from matching.

---

## 5. Pair matching (`match_pair(A, B, p, keep=5)`)

`rng_pair = rng(p.seed)`. Return `[]` immediately if either fragment has no fracture samples or no
breakline. All stages below use `sc = Scales` of the pair.

### 5.1 Hypotheses

```
ia = A.brk_sub, ib = B.brk_sub;  if either is empty: no hypotheses
ok[i, j] = | dih_A[ia[i]] + dih_B[ib[j]] − 180 | < 25          (i over ia, j over ib)
(pa, pb) = indices where ok, in row-major order (i outer, j inner)
RA = [ t_A | ns_A | f_A ]   (columns, at ia[pa])
RB = [ −t_B | ns_B | −f_B ] (columns, at ib[pb])
R  = RA · RBᵀ ;   τ = P_A − R · P_B                              (per hypothesis)
```

`n_hyp` is typically 25k–150k. A pair with zero hypotheses returns `[]`.

### 5.2 Coarse score

```
idx = rng_pair.choice(B.brk_sub, min(60, |B.brk_sub|), replace=False)     # first draw of rng_pair
Q = P_B[idx], QN = ns_B[idx]
for each hypothesis h: for each q: p = R_h q + τ_h; n = R_h QN_q
    j = nearest point of A's FULL breakline P_A with |p − P_A[j]| ≤ sc.coarse   (else miss)
    agree = hit ∧ (ns_A[j] · n > 0.7)
cs[h] = mean over the 60 points of agree
```

### 5.3 Non-maximum suppression

```
nms(order, R, τ, score, trans_tol, topk, floor):
    kept = []
    for h in order:
        if score[h] < floor: break
        dup = ∃ k ∈ kept: |τ_h − τ_k| < trans_tol ∧ trace(R_hᵀ R_k) > 2.9       # rotation within ≈ 18.2°
        if not dup: kept.append(h)
        if len(kept) ≥ topk: break
    return kept
```

Coarse NMS: `order = argsort(cs) descending, truncated to the first 5000`, `trans_tol = sc.nms`,
`topk = stage1 (250)`, `floor = 0.1`. numpy's `argsort` is an unstable quicksort; ties among
equal scores (common: scores are multiples of 1/60) are broken arbitrarily. **PMC-6:** the port
uses a stable descending sort with ascending hypothesis index as tie-break; expect small
candidate-set differences on ties, verified by the pair-level gates (§13).

### 5.4 Stage 1 — breakline ICP

For each kept hypothesis (independent; result order = `kept` order):

```
T0 = [R_h | τ_h]
T  = ICP_p2p(src = B.pc_brk, tgt = A.pc_brk_full, T0, d = sc.icp_dist(0.2), iters = 20)
T  = ICP_p2p(src = B.pc_brk, tgt = A.pc_brk_full, T,  d = sc.icp_dist(0.08), iters = 20)
s1 = brk_score(T, sc.stage1)
brk_score(T, δ): p = apply(T, P_B[brk_sub]); n = ns_B[brk_sub]·Rᵀ;  nearest A breakline point within δ;
                 score = mean( hit ∧ ns_A[j]·n > 0.7 )
```

`best1 = max(s1)`. If `stage1_floor > 0` and `best1 < stage1_floor`: return one partial
candidate (the arg-max pose) with scores `tightA=tightB=tight=0, gapA=gapB=gap=1,
contactA=contactB=contact=0, seam=0, cont=1, cont_n=−1, pen=0, pen_depth=0, partial=1,
brk=brk_best=best1` plus `limits`; never accepted.

### 5.5 Stage-1 NMS

`kept2 = nms(order = argsort(s1) descending, R, τ of the stage-1 poses, s1, sc.nms, topk = stage2
(10), floor = 0.05)`.

### 5.6 Stage 2 — full ICP and verification (per candidate `k ∈ kept2`, independent)

```
T = ICP_p2plane(B.pc_reg,  A.pc_reg,  T_k, sc.icp_dist(0.2),  30)
T = ICP_p2plane(B.pc_reg,  A.pc_reg,  T,   sc.icp_dist(0.08), 30)
if early_reject_tight > 0:                                   # off by default
    fs = fracture_scores(T);  if fs.tight < early_reject_tight:
        scores = fs ∪ seam ∪ limits ∪ {cont=1, cont_n=−1, pen=0, pen_depth=0, partial=1, brk=s1[k]}; not accepted; done
T = ICP_p2plane(B.pc_frac, A.pc_frac, T,   sc.icp_dist(0.08), 30)
T = ICP_p2plane(B.pc_frac, A.pc_frac, T,   sc.icp_dist(0.04), 30)
scores = verify(T) ∪ {brk = s1[k]}                            # §6
accepted = accept(scores)                                     # §6.5
```

### 5.7 Ranking and return

`score = seam · tight`. Candidates sorted by `score` descending (stable; ties keep `kept2`
order). Every candidate gets `brk_best = best1`. Return the first `keep = 5`. The pipeline keeps
all returned candidates for the report; the assembly uses accepted ones only.

---

## 6. Verification (`verify(A, B, T, sc)`)

### 6.1 Fracture contact (`fracture_scores`) — point-to-surface

```
d1 = distance from apply(T, B.Pf)      to A's fracture triangles   (A.frac_scene, float32)
d2 = distance from apply(T⁻¹, A.Pf)    to B's fracture triangles
for (tag, d, area) in (("A", d2, A.frac_area), ("B", d1, B.frac_area)):
    face = d < sc.facing
    if count(face) < 20: tight_tag = 0, gap_tag = 1, contact_tag = 0
    else: tight_tag   = mean( d[face] < sc.tight )
          gap_tag     = median( d[face] ) / t_pair
          contact_tag = mean( d < 2·sc.tight ) · area / t_pair²        # mean over ALL points of that fragment
tight = min(tightA, tightB);  gap = max(gapA, gapB);  contact = min(contactA, contactB)
```

### 6.2 Seam length (`_seam_score`)

```
p = apply(T, P_B) (all of B's breakline), n = ns_B · Rᵀ
for every A breakline point i: (dA, jA) = nearest transformed B point (KD-tree over p, no bound)
seamA[i] = dA < sc.seam ∧ ns_A[i] · n[jA] > 0.7
if none: seam = 0
else: vox = unique rows of ⌊ P_A[seamA] / (t_pair/3) ⌋  (integer voxel coords, world-origin grid);  seam = count(vox) / 3
```

### 6.3 Shell continuity (`_continuity_scores`)

```
if A has margin points and B has margin points:
    p = apply(T, B.Pm), n = B.Nm · Rᵀ;  (dm, jm) = nearest A margin point per p (unbounded)
    near = dm < sc.near
    if count(near) > 20:
        Am = A.Pm[jm[near]], An = A.Nm[jm[near]]
        cont   = median( | (p[near] − Am) · An | ) / t_pair
        cont_n = median( n[near] · An )
        return
cont = 1.0, cont_n = −1.0   (otherwise)
```

### 6.4 Penetration (`_penetration_scores`)

```
if not (A.watertight and B.watertight): pen = 0, pen_depth = 0, pen_unavailable = 1
else:
    sdA = signed distance of apply(T, B.S)    to A's working mesh (negative inside; sign by ray parity, float32)
    sdB = signed distance of apply(T⁻¹, A.S)  to B's working mesh
    pen = max( mean(sdA < −sc.pen), mean(sdB < −sc.pen) )
    pen_depth = max( −min(sdA), −min(sdB) ) / t_pair
```

`frac_scene` is built over the faces `frac` selects, and over `F[:1]` — the mesh's **first face** —
when it selects none: a raycasting scene needs a triangle, and a fragment with no fracture face has
no candidate to score anyway (it never reaches §5). A port has to reproduce the fallback or refuse
the same fragments.

Open3D's sign: closest-point distance, sign from the parity of intersections of one ray from the
query point (**PMC-7**: the port may use a different ray direction or the angle-weighted
pseudo-normal; `pen` must agree within 0.0005 on the parity gates).

**On a mesh that §3.3.2 accepts but that has boundary edges, that sign is not a function of the
geometry**, and the two implementations cannot be compared there. `closed_enough` tolerates up to
0.2 % boundary edges so that a decimated scan with a few holes still gets a penetration test; a ray
that leaves through a hole crosses the surface one time fewer and the parity flips. Measured on
`Pot_A_Piece_03_Mesh` (149 boundary edges, wall 3.45), Open3D calls 54 of `Pot_A_Piece_08_Mesh`'s
20 000 samples inside it at depths of 7.6–7.8 units — 2.2 t, inside a solid whose half wall is 1.72
— while its own `count_intersections` reports an **even** number of crossings for every one of them
along all six axis directions. D §10.2's `pen` row therefore applies to pairs of closed meshes, and
on an open one it is the *decision* (`pen ≤ max_pen`, and `accepted`) that is compared instead
(task C3, `notes/2026-09-06-c3-verify.md` §4).

### 6.5 Acceptance

```
accept ⇔ tight ≥ 0.25 ∧ gap · t_pair ≤ sc.gap ∧ pen ≤ 0.005 ∧ seam ≥ 3.0 ∧ cont_n ≥ 0.8
```

Score keys of a full candidate: `tightA tightB tight gapA gapB gap contactA contactB contact seam
gap_limit tight_delta cont cont_n pen pen_depth [pen_unavailable] brk brk_best [partial]`.

---

## 7. ICP reference (Open3D `registration_icp`, verified to 1e-16 against a re-implementation)

Inputs: source points `S` (n×3), target points `Q` (m×3) with target normals `Nq` (used only by
point-to-plane), initial `T0`, correspondence distance `d`, `max_iter`; convergence
`relative_fitness = relative_rmse = 1e-6`. Source normals are never used.

```
T ← T0;   P ← apply(T0, S)
corr(P): for each i: j = nearest target point (KD-tree) with |P_i − Q_j| < d, else none   # strict, see below
         C = {(i, j)};  fitness = |C| / n;  rmse = sqrt( Σ_{C} |P_i − Q_j|² / |C| )   (rmse = 0 if C empty)
(fit, rmse, C) ← corr(P)
for it in 1..max_iter:
    U ← update(P, C)                          # 4×4, below
    T ← U · T;   P ← apply(U, P)              # transform the already-transformed source
    (fit', rmse', C) ← corr(P)
    if |fit' − fit| < 1e-6 and |rmse' − rmse| < 1e-6: break
    fit, rmse ← fit', rmse'
return T
```

**Point-to-plane update** (`C` non-empty; identity if empty):

```
for (i, j) ∈ C:  a = P_i × Nq_j;  J_row = [a_x a_y a_z  Nq_jx Nq_jy Nq_jz];  r = (P_i − Q_j) · Nq_j
JTJ = Σ J_rowᵀ J_row  (6×6);  JTr = Σ J_rowᵀ · r  (6)      # plain sums, weight 1
x = solve(JTJ, −JTr)                                       # Open3D: LDLT; any stable solver
U = [ Rz(x₂) · Ry(x₁) · Rx(x₀)  |  (x₃, x₄, x₅) ]           # Euler composition, NOT the exponential map
Rx(a) = [[1,0,0],[0,cos a,−sin a],[0,sin a,cos a]];  Ry(b) = [[cos b,0,sin b],[0,1,0],[−sin b,0,cos b]];  Rz(g) = [[cos g,−sin g,0],[sin g,cos g,0],[0,0,1]]
```

(Using the exponential map instead changes one iteration by ~3e-4 on a 0.05 rad step; it is
**not** equivalent at the tolerances of §13 for single iterations, though the converged result
is; the port implements the Euler form.)

**Point-to-point update** (Umeyama without scaling):

```
μp = mean P_i, μq = mean Q_j over C;  Σ = Σ (Q_j − μq)(P_i − μp)ᵀ / |C|
SVD Σ = U S Vᵀ;  D = diag(1, 1, sign(det U · det V));  R = U D Vᵀ;  τ = μq − R μp;  U = [R | τ]
```

**The radius is strict.** Open3D hands FLANN `d²` and FLANN's `KNNRadiusResultSet` drops a
candidate at `dist >= radius`, so a source point exactly `d` from its nearest target is *not* a
correspondence. Measured against Open3D 0.19 with one source point and `max_correspondence_distance
= d`: one ulp inside gives `fitness = 1`, exactly `d` and one ulp outside give `fitness = 0`
(task C2, `notes/2026-09-07-c2-icp.md` §2). This is the same shape as §5.2's radius, which is
scipy's exclusive `distance_upper_bound`; the two libraries agree by coincidence and both bounds
are exclusive.

Correspondence ties (two targets at exactly the same distance) are resolved by the KD-tree
implementation; the port resolves them by lowest index.

---

## 8. Global assembly (`assemble(md, cands, p)`, `rot_tol = 10°`, `trans_tol = 0.5 t`)

Assembly-stage `MatchData` for every fragment is built at `t = t_med` (collection median) with
`surface_points = 15000` (**PMC-8**: this recomputes the samples with a fresh `rng(seed)`; the
penetration test here therefore sees different points than §6.4 did; the port may build at each
fragment's own `t` with the cached 20000-point sample, re-verifying `pen` decisions).

```
best_per_pair: for each accepted candidate keep the highest `score` per (a, b)
accepted = best_per_pair values sorted by score descending (stable)
poses = {}, group_of = {}, groups = [], used = [], rejected = []
group_thickness(names) = median( t_n for n in names )
rel(c, x) = c.T if x == c.a else c.T⁻¹                      # partner of x → x's frame

try_place(c, placed, new):
    T_new = poses[placed] · rel(c, placed)
    for other in groups[g] except placed:                   # penetration against every placed member
        T_rel = poses[other]⁻¹ · T_new                       # new → other
        pen = penetration(md[other], md[new], T_rel)         # §6.4 with sc.pen of that pair; 0 if either not watertight
        if pen > 0.005: reject "penetrates other (pen)"
    tg = group_thickness(groups[g] + [new])
    for c2 in accepted, c2 ≠ c, new ∈ {c2.a, c2.b}:          # consistency with other accepted joins into the group
        other = the other endpoint;  skip unless other is placed in group g
        T_alt = poses[other] · rel(c2, other);  D = T_alt⁻¹ · T_new
        ang = rotation angle of D, dist = |D_τ| / tg
        if (ang > 10° or dist > 0.5) and c2.score > c.score: reject "inconsistent with stronger join"
    return T_new

remaining = copy of accepted
loop:
    progressed = False
    for c in remaining (in order):
        a_in, b_in = c.a placed?, c.b placed?
        if a_in and b_in:
            remove c
            if same group: D = (poses[c.a] · c.T)⁻¹ · poses[c.b];  tg = group_thickness(group)
                           if ang(D) ≤ 10° and |D_τ|/tg ≤ 0.5: used.append(c)  else rejected "inconsistent with the assembled poses"
            else: rejected "would merge two groups (not supported)"
            continue
        if neither placed: continue
        (placed, new) = (c.a, c.b) if a_in else (c.b, c.a)
        T_new / why = try_place(c, placed, new);  remove c
        if rejected: rejected.append((c, why)); continue
        poses[new] = T_new; join group; used.append(c); progressed = True; break   # restart the scan from the best remaining
    if not progressed and remaining:
        seed = first c in remaining with both endpoints unplaced;  if none: break
        remove seed; groups.append([seed.a, seed.b]); poses[seed.a] = I; poses[seed.b] = seed.T; used.append(seed)
    if remaining empty: exit loop
every unplaced fragment: poses = I, its own singleton group
groups sorted by size descending (stable)
```

`rotation angle of D = degrees(arccos(clip((trace(R_D) − 1)/2, −1, 1)))`.

### 8.1 Second pass (off by default; `second_pass_top > 0`)

For every fragment in a singleton group, its `second_pass_top` best partners by the pair's
`brk_best` (ties by pair) are rematched with `stage1 = second_pass_stage1`, `stage2 =
second_pass_stage2`, `stage1_floor = 0`; their candidates replace the first-pass ones; assembly
runs again.

### 8.2 Recentre (`recenter`)

For **every** group (singletons too): `c = mean over members n of apply(poses[n], md[n].S[::10])`
(assembly-stage `S`, every tenth point); subtract `c` from every member's translation.

---

## 9. Full-resolution refinement (`refine_joins`, unless `--no-refine`)

For every fragment in a group of size ≥ 2, `fracture_cloud`:

```
mesh = load_mesh(original file)    (cleaned as §3.1 steps 1–2, NOT reduced to the largest component)
vertex normals = unit( Σ over incident faces of the unnormalised face normal )   # area-weighted
(d, j) = nearest working-mesh face centroid per vertex
sel = frac[j] ∧ d < max(0.15·t_fr, 1.5·res_fr)                                 # the fragment's own t, res
idx = where(sel);  if |idx| > 150000: idx = rng(0).choice(idx, 150000, replace=False)   (unsorted)
cloud = (V[idx], N[idx])
```

Then per group, a spanning walk from `g[0]`:

```
done = {g[0]};  edges = [c in used with c.a, c.b ∈ g] in `used` order
while ∃ edge with exactly one endpoint in done (take the first such in `edges`):
    fixed, moving = the done endpoint, the other
    src = cloud[moving] transformed by poses[moving];  tgt = cloud[fixed] transformed by poses[fixed]
    sc = Scales.for_pair(min(t_fixed, t_moving), max(res_fixed, res_moving))
    T = I;  T = ICP_p2plane(src, tgt, T, sc.icp_dist(0.05), 40);  T = ICP_p2plane(src, tgt, T, sc.icp_dist(0.02), 40)
    poses[moving] = T · poses[moving];  done.add(moving);  edges.remove(edge)
```

Recentre (§8.2) runs **after** refinement.

---

## 10. RNG inventory and determinism

| stream | seed | draws, in order |
|---|---|---|
| `rng_md` (§3.5) | `p.seed` | `choice(len(idx), 20000, p)`, `random(20000)`, `random(20000)`; `choice(len(idx_frac), n_frac, p)`, `random(n_frac)`, `random(n_frac)`; `choice(margin, 6000, replace=False)` only if `|margin| > 6000` |
| `rng_pair` (§5.2) | `p.seed` | 1 `choice(B.brk_sub, ≤60, replace=False)` |
| screening `cap` (§4.3) | `p.seed` | fresh generator per call |
| assembly `MatchData` (§8) | `p.seed` | as `rng_md` with 15000 surface points, at `t_med` |
| refinement (§9) | 0 | `choice(idx, 150000, replace=False)` only if `|idx| > 150000` |
| previews (§11.5) | 0 | one generator for the whole preview pass |

§3.2 has no stream: since `fbfebca` its ray set is `arange(0, n_faces, stride)` and
`Fragment.from_mesh_file`'s `seed` argument is accepted and unused. The `rng_pre` row that stood
here — `0 (hard-coded)`, one `choice(n_faces0, 20000, replace=False)` — is gone with it.

Reproducibility of the reference: two runs give byte-identical `report.json` (verified,
performance note §3b). Results do not depend on the number of processes or threads.

**PMC-9 (RNG).** The port does not reproduce numpy's PCG64/SeedSequence/`choice` bit-for-bit.
It uses its own portable generator with the same draw structure (§3.5.1); parity against the
reference is established by (a) injecting the reference's sampled indices/points in the fixture
harness, and (b) statistical tolerances on the natively sampled path (§13).

---

## 11. Outputs

### 11.1 `transforms.json`

```
{ "thickness": t_med, "params": {every Params field}, 
  "fragments": { name: { "matrix": 4×4 nested lists (world pose after recentre), "group": k, "placed": bool } },
  "groups": [[names…], …] }
```
`placed` = member of a group of size ≥ 2.

### 11.2 `report.json`

```
{ "thickness", "fragments": [stats() per fragment in collection order], "groups", "params",
  "timings": {"preprocess", ["screen"], "matching", "assembly", ["second_pass"], ["refine"]},
  "joins_used": [candidate JSON…], "joins_rejected": [candidate JSON + "reason"], "candidates": [candidate JSON…] }
candidate JSON = { "a", "b", "T": 4×4 lists, "accepted", "score", <every score key as float> }
```
`candidates` are in pair order, best first within a pair.

**`"output"` is not written, and this line used to say it was.** `write_report` serialises the
`timings` dict with `json.dump` *before* `pipeline.run` sets `timings["output"]`, so no
`report.json` and no `report.md` the reference has ever written carries that key. Verified on the
reference's own runs (`output/fixtures/*/_run/report.json` hold
`['preprocess', 'matching', 'assembly', 'refine']`), and the port follows the code rather than this
list (V4's §2.3, task Y).

Both objects are Python dicts written by `json.dump`, so **their key order is insertion order**:
`timings` comes out in the order the stages finished — `preprocess`, then `screen` when it ran,
`matching`, `assembly`, `second_pass`, `refine` — and §11.1's `fragments` in the order §8's greedy
loop filled `poses` (each seed's two fragments, then every placement, then the singletons in
collection order). A port that sorts either by name writes a different file (V4-D4, V4-D5).

### 11.3 `report.md`

Sections in order: title `# Reassembly report`; one line with the collection thickness; one
paragraph on the `max(k t, m res)` rule; `## Fragments` table (columns: fragment | faces (orig) |
thickness [+ ` **(differs)**` when > 40 % off the median] | ray mode | thickness/median | edge |
edges per t | fracture area % | watertight | extent); `## Assembly` (`- group k: names` per group of
size ≥ 2; `- not assembled (no confident join): names`); `## Joins used` table (A | B | score |
seam (t) | tight A/B | tight at (t) | gap (t) | gap limit (t) | contact (t²) | shell cont. | normal
agr. | penetration); `## Accepted joins not used` (only if any; `- a – b (score s): reason`);
`## Best candidate per pair` with a legend line built from the params and a table (A | B |
accepted | score | seam (t) | tight A/B | tight at (t) | gap (t) | gap limit (t) | penetration |
normal agr.), pairs sorted by name, `n/a` for penetration/normal agreement of partial candidates;
`## Timing` (`- stage: s`). Number formats: score `.2f`, seam `.1f`, tight `.2f`, tight at `.3f`,
gap `.3f`, gap limit `.3f`, contact `.1f`, cont `.3f`, cont_n `.2f`, pen `.4f`.

### 11.4 Meshes (unless `--no-meshes`)

`placed/<name>.ply` for **every** fragment: the original file (cleaned as §3.1 steps 1–2, all
components) transformed by its pose, binary little-endian PLY with vertex colours if present, no
normals. `assembly_<k>.ply` for each group of size ≥ 2: the concatenation of its members' placed
meshes (vertex indices offset), same format.

### 11.5 Previews (unless `--no-preview`) — software renderer

`write_previews` with one `rng(0)` for the whole pass. For each group `k` of size ≥ 2, in order,
and each member `i` in group order: `P, pick = sample_on_faces(all faces, 250000)` on the working
mesh, transformed by the pose; normals `FN[pick]·Rᵀ`; colour `PALETTE[i mod 10]`. Views =
`principal_views` of all points; image 900×700 per view, label `"name=colourname | …"` with
colour names `grey orange blue green yellow purple cyan pink indigo tan`. File `preview_<k>.png`.

Segmentation preview (`preview_segmentation.png`, 1400×600, first two principal views, label =
names joined by spaces): every fragment, 125000 samples, colour 0.8 grey or `(0.9, 0.2, 0.2)` for
fracture faces, centred at the sample mean and offset along x by `i · 1.3 · extent_x(V_i)`.

```
PALETTE = [[.78,.78,.78],[.95,.55,.25],[.40,.70,.95],[.50,.85,.50],[.90,.80,.40],[.80,.50,.90],[.35,.85,.85],[.90,.45,.55],[.60,.60,.95],[.75,.60,.40]]
principal_views(V): X = V − mean;  eigen-decompose XᵀX, eigenvalues ascending: u = ev[:,0], e1 = ev[:,2], e2 = ev[:,1]
                    views = [(u, e2), (−u, e2), (e1 + 0.35u, u), (e2 + 0.35u, u)]     # (eye_dir, up); PMC-10: eigenvector signs are library-defined
render_views(meshes, views, W, H, label):
    center = (min + max)/2 of all points;  ext = |max − min|;  scale = 0.9·min(W,H)/ext;  light = unit(0.3, 0.4, 1.0)
    images side by side; label drawn at (10, 10) in white with the default bitmap font
splat(V, N, C, eye_dir, up):
    z = unit(eye_dir);  x = up × z (if |x| < 1e-6: x = [1,0,0] × z);  x = unit(x);  y = z × x;  R = [x y z] (columns)
    P = (V − center)·R;  Nn = N·R;  shade = clip(Nn·light, 0, 1)·0.75 + 0.25;  shade[Nn_z < 0] ·= 0.5;  col = C·shade
    px = round(P_x·scale + W/2), py = round(−P_y·scale + H/2);  keep 1 ≤ px < W−1, 1 ≤ py < H−1
    background 0.16 grey; z-buffer = −inf; depth = P_z (larger is nearer)
    for dx, dy ∈ {−1,0,1}²: for each pixel (px+dx, py+dy) take the point with the largest depth; write colour if depth > z-buffer
    output = clip(img, 0, 1)·255 as uint8
```

### 11.6 Timing and logging

Not part of the contract. `timings` keys are listed in §11.2.

---

## 12. Consolidated PMC list (port may change, must re-verify)

| id | what the reference does | why | what the port may do | re-verify with |
|---|---|---|---|---|
| PMC-1 | ray origins offset by an absolute `1e-3` (thickness rays and cone rays) | historical; negligible on mm-scale data | offset `max(1e-6, 1e-4·res)` along −FN | thickness within 1 %, mask agreement (§13) |
| PMC-2 | Open3D quadric decimation | library | any quadric-error decimator to the same face budget | segmentation IoU, thickness, `res` within 10 %, pair gates |
| PMC-3 | Taubin with inverse-distance Laplacian weights | Open3D | same (uniform weights only with re-verification) | mask agreement |
| PMC-4 | hash-map order of voxel representatives (`rep`, `brk_sub`) | Open3D | sorted ascending | pair gates (tie effects) |
| PMC-5 | margin band `0.12 t < d < 1.5 t` without `res` floors | oversight | keep for parity; flag as a future threshold change | — |
| PMC-6 | unstable `argsort` for coarse and stage-1 ranking | numpy | stable sort, index tie-break | pair gates |
| PMC-7 | signed-distance sign by one-ray parity (Embree) | Open3D | robust inside test (several rays or winding number) | `pen` within 0.0005 |
| PMC-8 | assembly `MatchData` at `t_med` with 15000 samples | historical | own `t`, cached 20000 samples | assembly `pen` decisions on all benchmarks |
| PMC-9 | numpy PCG64 sampling | library | portable RNG, same draw structure | injected-sample parity + statistical gates |
| PMC-10 | eigenvector sign convention in `principal_views` | numpy | sign fixed by convention (largest component positive) | visual only |
| PMC-11 | penetration counts `sd < −pen` via full signed distance on all 20000 samples | direct | equivalent formulation: inside ∧ unsigned distance > pen, with AABB/early-exit prefilters | `pen` identical up to PMC-7 |
| PMC-12 | fracture distances computed for all samples | direct | bounded closest-point query with early exit at `sc.facing` (exact for points inside the window; `≥ facing` otherwise); `contact` needs `d < 2·tight` which lies inside the window | identical scores |
| PMC-13 | mesh orientation assumed outward | data | check signed volume and flip if negative | none on current data (all outward) |
| PMC-14 | `tree_frac` built and never used | dead code | drop | — |
| PMC-15 | working mesh, `res` and everything derived from them in float64 (§0) | numpy | store `V` and `res` as **float32** and derive `FN`, `A`, `C` from the *narrowed* vertices, so a cold run and a cache hit are bit-identical (D §4.1, D §7); everything up to the narrowing — Taubin, `face_geometry`, `median_edge`, `ΣA` — stays float64 | working-mesh row of D §10.2 in native mode (`res` ±10 %, area ±0.5 %); the ≈6e-8 relative error enters every §1.2 threshold and every ICP residual, so the pair gates of §13 are the real check |
| PMC-16 | `near[i]` (§3.4.1) from `scipy.spatial.cKDTree.query`, whose tie rule between two equidistant representatives is unspecified | library | any KD-tree, ties resolved by the lowest index (`kiddo`'s observed behaviour, which it does not document as a guarantee) | injected `rep face` and `near` agreement in D §10.2's segmentation row (measured exactly 0 on all 68 fixture fragments — no fixture has a tie — so the first symmetric synthetic mesh is what will exercise it) |
| PMC-17 | first-hit ray casts (§3.2, §3.4.3) through Open3D's `RaycastingScene`, i.e. Embree in `float32` | library | any `f32` BVH ray cast (`parry3d` 0.30 `CompositeShapeRef::cast_local_ray`), which disagrees with Embree on hit/miss or on the primitive id for a ray that grazes an edge | injected `votes/face` in D §10.2's segmentation row and the injected thickness row; measured at 2 hit/miss disagreements and 5 differing primitive ids over 7.87 M cone rays (experiment E4), absorbed completely by §3.4's cleanup (`raw mask` agreement 1.0) |
| PMC-18 | §7's pose update is composed as **quaternions**: `utility::TransformationMatrixFromPoseVector` writes `(AngleAxisd(x₂,Z) · AngleAxisd(x₁,Y) · AngleAxisd(x₀,X)).matrix()`, and Eigen's `operator*` on two `AngleAxis` converts both to quaternions, multiplies, and converts once at the end | library | §7's matrix product `Rz(x₂)·Ry(x₁)·Rx(x₀)`, which is the same rotation and not the same arithmetic | the two compositions swept over the angles an ICP update produces (1e-1 down to 1e-12 rad, both signs): worst entry **3.3e-16, 1.5 ULP of 1**, which moves a sample 885 units from the origin by 4.4e-13 units = **1.9e-13 t** on the thinnest benchmark wall (`matching/icp.rs`'s own test). The accumulated effect over a whole ladder is the injected `stage 1` and `stage 2` pose rows of D §10.2, and the `chaotic` rows are where an ULP is not bounded at all |
| PMC-19 | `np.linalg.inv(T)` (§6.1, §6.4) is LAPACK's `dgetrf` + `dgetri` through numpy | library | the same factorisation — partial pivoting on the largest remaining column, first index on a tie, then two triangular solves per column — written out in `matching/verify.rs::pose_inverse` so that it is the same arithmetic on every machine (D §7) | the two inverses applied to the fixtures' own clouds at the fixtures' own 2 239 stage-2 poses: **2.7e-14 t at the median, 1.5e-13 t at p99, 1.2e-11 t at the worst**, the tail belonging to candidates whose ICP diverged to `‖τ‖ = 1.8e5` where `cond(T)` reaches 3.3e10. A test pins one real stage-2 pose against numpy's own answer at 4 ULP |

### 12.1 Addenda (changes made after the freeze at `9d4b9d3`)

The table above is the frozen text. Every amendment to it is recorded here instead, dated and
attributed, so that a reader diffing this document against the Python can tell which clause is the
reference's and which is the port's.

**2026-09-06, step B3 — PMC-9 is exercised more widely than its row says (defect D2).** The row
allows "portable RNG, same draw structure". The port
(`crates/sherd-core/src/rng.rs`) keeps the reference's draw *order inside* each sampler and its
uniform construction — numpy's own `(word >> 11)·2⁻⁵³` — but gives **each draw site its own
stream** (`ChaCha8Rng::seed_from_u64(seed ^ tag)`) instead of consuming one `rng_md` per fragment
through §3.5's samplers in order. The reason is §4.2 and §8: both rebuild the match arrays at
another `t` or another `surface_points`, and with one shared stream a change to the *first*
sampler silently moves the other two. The reference's own behaviour is unchanged; only the port's
is. **Not yet re-verified by the gate its own row names:** "injected-sample parity + statistical
gates" means §13, which is the pair and assembly gate set, and phase 1b cannot run it. It is on
the phase-1c risk list, and until §13 runs, this amendment is asserted rather than demonstrated.

**2026-09-06, task T1 — §3.2's ray set is no longer sampled (an algorithm change, not a PMC).**
Commit `fbfebca` replaced `rng_pre.choice(n_faces0, 20000, replace=False)` with
`arange(0, n_faces0, max(1, ceil(n_faces0 / 300000)))` **in the reference itself**, and this
document's §3, §3.2 and §10 follow it. It is recorded here rather than only in §3.2 because it is
the one change to the frozen algorithm: the port did not gain a licence, the algorithm lost a
random number. Consequences: `rng_pre` is gone from §10, `Fragment.from_mesh_file`'s `seed` is
accepted and unused, `CACHE_VERSION` moved 7 → 8 on the Python side and the fixture commit in this
document's header moved to `fbfebca`. D §10.2's native tolerance on `t` returns to ±2 %.

**2026-09-07, step C1 — the fixture sink dumps the two NMS walk orders (a dump-only change).**
`_match_pair` now hoists `np.argsort(cs)[::-1][:5000]` and `np.argsort(s1)[::-1]` into variables
and writes them as `nms1.order` and `nms2.order` (D §10.1). Nothing the reference computes changes
— re-dumping the slab at the same `sherd_refit/` reproduced every previous file byte for byte —
and §5.3's text is unchanged. The reason is PMC-6: the walk order is an **input** of `nms`, the
reference's is an unstable quicksort over scores that are multiples of `1/60`, and without it in
the dump an injected comparison of the kept set would be measuring numpy's tie-breaking rather
than the port's greedy loop. With it, the port keeps the reference's 250 hypotheses in the
reference's order on all 358 pairs of the eight fixture sets.

**2026-09-07, step C1 — PMC-6 and PMC-9's coarse draw are re-verified, and PMC-4 is exercised.**
PMC-6's row says "expect small candidate-set differences on ties, verified by the pair-level gates
(§13)". Measured, the differences are not small and the gates are not the right instrument: with
the port's stable tie-break the coarse NMS misses 14.3 % of the reference's kept hypotheses on
average and 43.6 % at worst, while keeping poses of the same quality — the two kept sets agree
rank for rank to one probe point on average and two at worst (D §10.2). PMC-9's row gains R §5.2's
probe, which is the reference's first draw of `rng_pair` and the port's own `Draw::CoarsePoints`
stream; fed the reference's own `coarse.idx` the port reproduces every score bit for bit, so the
draw is the only thing between the two implementations at this stage. PMC-4 is exercised for the
first time: injected mode hands the port the reference's own `hyp.ia`/`hyp.ib`, in Open3D's hash
order, and `(pa, pb)` comes out in the reference's own order on every pair.

**2026-09-06, task T1 — two library substitutions were promoted to rows (defects D4 and D5).**
PMC-16 (the `near` tie rule) and PMC-17 (the ray-cast library) are new rows in the table above
rather than addenda, because they describe things the port has done since step B1 and neither
changes what the reference computes. They are listed because R §12 is the port's licence and
anything the port does differently that is not on it is an undeclared deviation, however small its
measured effect.

---

**2026-09-07, step C2 — §7's correspondence radius is strict (a correction, not a change).** The
frozen text wrote the correspondence test as `|P_i − Q_j| ≤ d`. Open3D's is `<`: FLANN rejects a
candidate at `dist >= radius`, measured against Open3D 0.19 as described in §7. Nothing the
reference computes changes — the case is measure-zero on real data — but a port that implements
`≤` disagrees with Open3D on a point that lands exactly on the radius, and the document now says
which one it is. The two ladders of §5.4 and §5.6, R §7's two estimators and R §5.5's suppression
are implemented in `crates/sherd-core/src/matching/{icp,ladder}.rs` and verified against the dump
on all eight fixture sets (`notes/2026-09-07-c2-icp.md`); the only other amendment C2 makes is to
D §7's ill-conditioning row, which is the port's own numerics and not the reference's.

**2026-09-07, step C3 — §6 is implemented and PMC-7, PMC-11 and PMC-12 are re-verified.**
`crates/sherd-core/src/matching/verify.rs` computes every score of §6 and §6.5's rule, and
`crates/sherd-core/src/matching/pair.rs` closes `match_pair` around it. Fed the reference's own
stage-2 poses, samples and meshes, the port reproduces `tight` and `seam` **exactly** on all 2 249
candidates of the eight fixture sets, `gap` to 3e-6 t, `cont` to 2.7e-14 t, `cont_n` to 2.2e-16 and
`pen` to 5e-5 wherever both working meshes are closed; §6.5's verdict is identical on every
candidate, and §5.7's ranking and cut return the reference's own five candidates in the reference's
own order on every pair. PMC-11's and PMC-12's "equivalent formulation" clauses are what the port
implements (an AABB reject and a parity test before any distance; a bounded closest point with
`r_max = sc.facing`), and both come out identical rather than merely close. PMC-7 is the one row
with a residual, and §6.4 above now says where it lives.

**2026-09-07, task X — closing the phase-1c verification: three corrections, two new rows, and
one boundary bug.** `notes/2026-09-06-phase1c-verification.md` listed twelve defects and
`notes/2026-09-07-x-phase1c-findings.md` records what each became. Three were transcriptions of
loops this document freezes, and the port now reads them as written: §5.4's `int(np.argmax(s1))`
takes the **first** maximum as numpy does (the port took the last, and `s1` is a mean of booleans
so ties are ordinary); §5.3's loop appends before it tests `len(kept) ≥ topk`, so the reference
keeps **one** pose at `topk = 0` and the port kept none; and §5.4's floor branch precedes any use
of a BVH, as the reference's lazy `frac_scene` does. None is reachable with the shipped parameters.

Two substitutions the port had made without a row are now measured, and the team rule was to
record anything at or under 1e-12 t and to match the reference above it. §7's Euler composition
came in at 1.9e-13 t and is **PMC-18**. §6's inverse did not: `[Rᵀ | −Rᵀτ]` sat 6.2e-11 t from
`np.linalg.inv` at the worst, because a pose that has climbed two ICP ladders is orthonormal only
to 2.6e-14 and a sample is up to 885 units from the origin, so the port now runs the reference's
own factorisation and **PMC-19** carries what is left of it (1.2e-11 t at the worst, on candidates
whose ICP diverged).

The third substitution — a radius-bounded nearest-neighbour search where §5.2, §6.2 and §6.3 read
an unbounded `cKDTree.query` and then test `d < r` — needs no row, because it is provably the same
answer and the proof is now in `spatial/kdtree.rs::nearest_below` rather than in three call-site
comments: pruning cannot change the winner, since a node is dropped only when its box is further
than the radius and no such node can hold a point at a minimum that is itself under the radius, and
the square is widened by its own rounding so that the traversal is a hint and not a semantic.

Writing that proof out found a false claim in the old form's documentation — `nearest_within` says
its radius test is inclusive (`d ≤ r`) and it is not, because `radius · radius` rounds and a point
at exactly `radius` comes back as a miss. It is **not** a difference in the answers, and the first
draft of this addendum said it was: `d² > fl(r·r)` forces `fl(sqrt(d²)) ≥ r`, the window between
the two being under half an ULP of `r` after the square root, so a distance the old form dropped
could not have tested below `r` either (searched over 2.4 M `(r, d²)` pairs, no counterexample).
The old argument was true and fragile — it depended on all three call sites using a strict `<` and
was invisible at each of them; the new one depends on nothing.

**2026-09-07, task X — PMC-2 gets a number, and two of D §10.2's rows are calibrated to it.**
PMC-2's re-verify column says "`res` within 10 %, pair gates" and says nothing about how far a
different decimator may move the *breakline*, which is where its whole effect lands. D §10.2's two
native breakline rows stood at `0.5 t` on the 99th percentile and `0.05` on the dihedral KS, and
step T1 showed the reference cannot meet them against itself. Measured over three collections, 33
fragments and three budget perturbations each (99 comparisons): at the `res` gaps the working-mesh
row already allows, the reference's own two breaklines move **1.160 t** at the 99th percentile and
**0.0431** in KS, and at a 30 % gap they move 8.1 t and 0.169. The two rows are now twice that —
`2.3 t` and `0.086` — with the derivation, the margin's justification and the port's own distance
from them (39 % and 76 %) in D §10.2. This is a change to two harness tolerances, not to anything
this document specifies; it is recorded here because PMC-2 is the row it belongs to.

Also corrected: this document's header named the wrong fixture commits (see it), and R §7's
`solve_ldlt` claimed Eigen's pivot permutation was a *stable* descending sort. It is a selection
sort by transpositions, which agrees with a stable sort on distinct diagonal entries and not on a
tie; the port runs Eigen's loop now. What was right about the claim is kept and now says why: the
left-looking factorisation decrements `mat(k, k)` only at step `k`, after `k` has been chosen, so
the values compared are the original diagonal's.

**2026-09-07, step D1 — §8 is implemented, PMC-8 is re-verified, and four tie rules §8 states
loosely are written out.** `crates/sherd-core/src/assembly/` implements §8's greedy pass, its two
tests and §8.2's recentring, and D §10.2's `assembly` row runs it on the reference's own candidate
lists: over the eight fixture dumps the groups, the joins used, the rejections — **with the
reference's own English reason, digit for digit** — come back identical, and the poses land
**7.0e-13 t** from the reference's at the worst (`notes/2026-09-07-d1-assembly.md`).

**PMC-8 is now measured rather than allowed.** Its row lets the port skip the reference's rebuild
at `t_med` with 15 000 surface points and use the fragment's own cached 20 000-point sample at its
own `t`, and its re-verify column asks for "assembly `pen` decisions on all benchmarks". Fed the
reference's own candidates with the port's own samples, the groups, the used joins and the poses
are **identical on all eight sets**, and so is every rejection's kind and subject. What moves is
the *number* printed inside a `penetrates` sentence, because that number is R §6.4's fraction
measured on whichever sample set was drawn: on pot_A it reads 0.129 against the reference's 0.127
and 0.180 against 0.188, and on pot_H four sentences move likewise; the worst movement anywhere is
**0.0078**, against a `max_pen` of 0.005 that these rejections clear by twenty to thirty times. No decision changes on
any set. The row's re-verification is therefore met, and D §10.2 states it as two rows: the
decision compared exactly, the fraction reported beside it.

**Four things §8's pseudocode above leaves to the reader, which a port has to get right.** None is
a change to what the reference computes; each is a tie or an iteration order that the text does not
pin down and the Python does.

* `best_per_pair` keeps a candidate only on `c.score > best.score`, strictly, so the **first**
  candidate at a pair's top score represents it — and `accepted`'s stable sort is over
  `dict.values()`, whose order is *insertion* order, so two pairs of equal score come back in the
  order the matcher produced them (R §4.1's `itertools.combinations` order).
* `for c in remaining (in order)` is `for c in list(remaining)` in the Python: the pass iterates a
  **snapshot**, so a join removed during the pass does not shorten that pass. Walking the live list
  instead skips the candidate after every removal, and that alone changes the answer on **four of
  the eight fixture sets** — measured, by running the port both ways on the reference's own
  candidates: pot_A, pot_B, pot_H and synthetic_20 come out with different groups, different used
  joins or different rejections; the slab, the terracotta, pot_C and pot_G are unaffected.
* The consistency loop walks **all** of `accepted`, not the joins still remaining, and `c2 ≠ c` is
  an *identity* test on the candidate object rather than a comparison of its fields.
* §8.2's `c` is the mean over the **concatenation** of every member's `S[::10]`, not the mean of
  the members' means. The two coincide here because every fragment carries the same
  `surface_points`, and they would not if that ever stopped being true.

**The `would merge two groups (not supported)` branch is unreachable under §8's own loop, and no
fixture exercises it.** A second group is seeded only after a whole pass has made no progress, and
such a pass has scanned and removed every join with exactly one placed endpoint; what is left all
has both endpoints unplaced, so it touches no existing group. The first group is therefore frozen
at the moment the second is seeded, by induction no join ever has its two endpoints in different
groups, and the branch cannot fire. That matches the dumps: **none of the eight** has a rejection
of that kind. The port keeps the branch — it is R §8's text and R §8.1's second pass reruns the
loop on another candidate list — and its own test exercises the sentence rather than the path.

**2026-09-07, step D2 — PMC-20 (the preview caption), a new row of §12's table.**

| id | what the reference does | why | what the port may do | re-verify with |
|---|---|---|---|---|
| PMC-20 | §11.5's caption is drawn by PIL's `ImageDraw.text` with `ImageFont.load_default()`, which in Pillow 12 is an anti-aliased FreeType face | library, and a font file | the same text in the port's own 5×7 bitmap font (`render.rs`'s `FONT`), upper case only | the pixels **outside** the caption compared exactly, and the caption's own area reported as a row: `preview label px`, measured 566–4 943 pixels per image over the eight dumps against a cap of 8 000 |

The rest of §11.5 is *not* a PMC and is not approximated: fed the reference's own samples and the
reference's own views, the port reproduces the splat pixel for pixel — **0 differing pixels of
4 200 000** on every preview of every set (D §10.2's `outputs` row). What that costs is four
arithmetic details, and each of them was measured rather than assumed:

* `(V − centre) @ R` and `N @ R` are numpy `(n,3) @ (3,3)` products and reach OpenBLAS's `dgemm`,
  which **fuses**: the port evaluates them as `fma(p₂, m₂ⱼ, fma(p₁, m₁ⱼ, p₀·m₀ⱼ))`. `Nn · light` is
  a `(n,3) @ (3,)` product and does **not** fuse; the port evaluates it left to right. Measured on
  400–1 200 random coordinates: the fused form matches numpy on 1 200 of 1 200 for the matrix
  product and misses 156 of 400 for the vector one, and the unfused form is the other way round.
* `np.round` is round-half-to-**even** (`f64::round_ties_even`), not half away from zero.
* the z-buffer is `float32` while the depth compared against it is `float64`.
* `np.lexsort` is stable and the reference takes the **last** entry of each pixel's run, so a tie
  in depth inside one of the nine offsets goes to the later point in concatenation order.

**2026-09-07, step D2 — `apply_transform` is not bit-identical to the reference's *coordinates*,
and R §11.4 needs one that is.** The port's `types::apply_transform` was documented as reproducing
`P @ T[:3,:3].T + T[:3,3]` because "`a + b + c + d` associates to the left in Rust exactly as
numpy's matmul-then-add does". Measured, that is false on this machine: numpy's `@` is a BLAS call
and Eigen's `Matrix4d * Vector4d` is a column combination, and both fuse, so the reference
accumulates three roundings where the scalar expression takes five. Over 900 coordinates of a
random cloud 300 units from the origin under a random pose, the fused form
`fma(t₂, z, fma(t₁, y, t₀·x)) + τ` reproduces **Open3D's transformed vertices and numpy's matmul,
900 of 900**, while the unfused one misses 291 and the two other plausible FMA associations miss
320 and 440. The port therefore has both: `apply_transform` unchanged, because §6's scores were
verified through it on all 2 249 candidates, and `apply_transform_fused` wherever a *coordinate*
has to come back the same — §9's clouds, §11.4's placed meshes and §11.5's samples. That is what
makes `placed/<name>.ply` byte-identical to the reference's file on every collection the two
implementations read the same way (the four PLY sets; on the five OBJ sets D §10.2's `load` row
already measures Assimp's `fast_atof` one `f32` ULP away, and a placed mesh cannot be
byte-identical when its input is not).

**2026-09-07, step D2 — §9's cap is the only draw in the refinement, and it is the only thing
between the two implementations there.** Under 150 000 candidates both sides compute
`np.where(sel)[0]` and the port reproduces the reference's selection **exactly**: 0 differing
entries of 9 952 (slab), 14 776 (pot_C), 18 720 (pot_H), 93 923 (pot_A), 40 565+ (pot_B) and
389 000 (synthetic_20). Above it, PMC-9 decides the order and the port draws its own 150 000; what
is still exact there is that **every index the reference kept is one the port's own predicate
accepted** (450 000 of 450 000 on terracotta's three capped fragments), and the overlap of the two
draws sits within 0.0013 of `150000/|candidates|`, the ratio two independent draws share in
expectation.

**2026-09-07, task Y — PMC-6 and PMC-9 are re-verified at the collection level, and §13 is
restated on what the measurement found.** Both rows' re-verify columns point at §13 ("pair gates",
"injected-sample parity + **statistical gates**"), and until now that meant comparing the port's
per-set figures against a table holding **one draw** of the reference's own randomised search. The
sweep behind §13's new ranges (five seeds and ±5 % of the working-mesh budget, three sets, 21 runs
of the reference; `notes/2026-09-07-y-phase1d-findings.md` §3) shows the table itself moving:
`pot_B` 88.9–100 %, `synthetic_20` 85–95 %, and `pot_G` — whose row was written as the prohibition
"no join must be accepted" — accepting **0, 1, 2, 1 and 1** joins as the seed changes. The port's
figures lie inside that spread on every swept set.

`pot_G` was taken further, because a prohibition is the one kind of row a spread cannot absorb.
The port's two joins there are 04–07 and 02–05; both are **ground-truth-adjacent pairs at a wrong
pose** (0.7° / 1.05 t and 2.5° / 1.20 t), which is the same failure the reference itself produces
at four of the five seeds (02–05 at 1.5° / 1.11 t, and 01–03 at 2.4° / 1.12 t at seed 2). Run the
**reference's own** §6 verification on the port's two poses and it accepts both, at scores 21.15
and 18.31 against 2.07 and 0.41 for the best candidate its own seed-0 search found for those pairs;
run the **port's** verification on the reference's own ten candidates for each of those pairs and
the verdicts agree 10 of 10 with every score inside D §10.2's tolerances. So §6 is not where the
two differ: the difference is which poses reach it, which is exactly what PMC-6 and PMC-9 are
licensed to change, and the reference's own search changes it under nothing but a seed.

**§8.2's mean has a summation order, and it is not the pairwise one.** The port summed the
concatenated `(N, 3)` cloud with numpy's pairwise blocking, on the assumption that `mean(0)` is
`np.sum` along an axis. Measured against numpy 2.5.2 on the fixtures' own arrays (4 500 × 3,
15 000 × 3 and 61 130 × 3): `a.mean(0)` of a C-contiguous array is **bit-identical to a
left-to-right accumulation per column** — numpy blocks along the axis it walks contiguously, and an
axis-0 reduction walks the rows — and the pairwise form is up to 8 ULP away. With that corrected
and the points moved by `apply_transform_fused`, the `assembly` row's `recentre` check, which
compares the port's §8.2 against the reference's own `transforms.json`, is **exactly 0** on all
eight dumps where it stood at 7.0e-13 t.

**2026-09-08, task Z — the second `column_mean` call site, and the three filters task E2 put in
front of R §3.5, §5.2 and §6 (V5-D7).** §12.1's rule, from task X: anything the port does
differently that is not on §12 or here is an undeclared deviation, however small its measured
effect. Four such changes were made in phase 1e while both notes stated that R was untouched.

**The mean of `render::principal_axes`.** Task Y's addendum above records `column_mean` for §8.2
alone; step Y4 changed the same summation in the renderer's own PCA, where the reference computes
`X = V - V.mean(0)` over a C-contiguous `(N, 3)` array (`render.principal_views`) — the same shape
and the same reduction as §8.2's, so the same measurement covers it: `a.mean(0)` of such an array
is bit-identical to a left-to-right accumulation per column and up to 8 ULP from the pairwise form
the port used before. The call site is inside PMC-10's own licence, since the eigenvector signs are
library-defined either way, and D §10.2's `outputs` `axis` row is what measures it: fed the
**reference's own** preview points at the reference's own poses, the angle between the port's first
principal axis and the reference's own first view is **0.000° on seven of the eight dumps and
8.5e-7° on pot_C**, against a gate of 1°. `XᵀX` is still accumulated per entry with `pairwise_sum`,
which is neither the reference's blocked `dgemm` nor anything numpy does for this shape; that
remains PMC-10's, and the row above is what bounds it.

**Three filters in front of bounded queries (task E2).** None of the three answers a query, resolves
a tie or changes an array the algorithm reads out; each is licensed by the same argument the
task-X addendum gives for the bounded search itself, and each has its own proof. They are recorded
here because they are on the critical path of §5.2, §6.2, §6.3, §7 and §3.5 and because the port's
arithmetic, not only its speed, is what this document is the licence for.

* **A bounding-box reject in front of every bounded query** (`spatial/kdtree.rs::beyond_the_box`,
  commit `f5f25a2`). `PointTree` carries the cloud's box and tests it before the traversal: every
  point of the cloud is inside the box, so the distance from the query to the box is a lower bound
  on the distance to any of them, and a box further than the radius means there is nothing to find.
  The argument holds whatever the tie rules are, because the filter never resolves a tie — a query
  it lets through is answered by exactly the code that answered it before — and the comparison is
  widened by `16 ε` so that the *true* box distance clears the radius even if every rounding in it
  went the other way, which makes the filter reject slightly less often than it could and never
  once too often. Test:
  `the_box_filter_never_rejects_a_neighbour_the_traversal_would_have_found`, 400 queries walking
  onto the box and 2 000 random ones at four radii each, including the exact distance and one ULP
  either side. It sits inside every bounded query in the port: §5.2, §5.4's re-score, §6.2's seam,
  §6.3's continuity, §7's correspondence search and §3.4's ball queries.
* **`spatial::grid::NearMask` as a near mask in front of §5.2** (`matching/coarse.rs`, commit
  `114455a`). One dilated bit per cell of a ≤ 64³ grid over A's breakline, built once per pair:
  `may_be_near` returns `false` **only** when nothing is within the radius, and `true` means "ask
  the tree", so `agreeing` falls through to the same `nearest_below` and computes the same count.
  A cell is never narrower than the radius and the mask is dilated once at build time, so a
  neighbour within the radius is inside the query cell's own 3×3×3 block; a query outside the grid
  is clamped to the boundary cell, whose block is a superset of the true one, so clamping only ever
  makes the answer more permissive. Test: `the_mask_never_hides_a_neighbour`, 60 000 queries over
  three cloud shapes — random, a curve, a plane — at five radii from a twentieth of a cell to
  larger than the cloud.
* **The two bounded centroid sweeps of §3.5** (`fragment/breakline.rs`, `fragment/samples.rs`,
  commit `77ce026`). This one is **not** the same class, and the distinction is the reason it is
  written out here: it does not filter a query, it changes an *array*. §3.5.4's `distance` and
  §3.5.6's `d_brk` now hold `∞` wherever the true distance is at or beyond the one threshold that
  reads them — `nearest_below(q, r)` answers `Some(d)` exactly when `d < r` — so the equality is of
  the **predicates** `distance[i] >= 0.15 t` and `0.12 t < d_brk[i] < 1.5 t`, not of the arrays.
  Nothing else in the port reads either array. Both halves are gated: the injected `breakline` row
  compares `ns`/`nf`, which read `distance` only through that predicate, and D §10.2's `samples`
  row gains a `margin bound` check in task Z that compares the bounded array against the unbounded
  one directly — bit for bit below the bound, `∞` at or above it — measured **0 differing of
  1 360 000 entries** on the eight dumps in each mode (V5-D5). The unit test is
  `the_bounded_breakline_distance_is_the_exact_one_under_its_bound`, 2 000 queries at five bounds
  against the exact sweep.

The empirical check that covers all four at once is task E2's own and the phase-1e verification
re-derived it against a freshly built phase-1d binary: **70 output files on three collections —
35 `cache/*.sherd` rebuilt cold through the box and both bounded sweeps, 15 `placed/*.ply`, 3
merged meshes, 8 PNGs and 3 `transforms.json` — 64 byte-identical outright and the six
`report.json`/`report.md` identical once the wall clock is taken out** (`notes/2026-09-07-phase1e-verification.md` §5.2).

## 13. Reference results and parity gates

Numbers the port must reproduce on the benchmark sets, with the defaults above (from the notes
`2026-09-06-scale-pairs.md` §3 and `2026-09-05-test-set-result.md`):

**The per-set rows are the *spread* the reference itself produces, not a single number.** R §10's
streams are seeded from `Params.seed`, and neither CLI exposed it, so the table used to print
whatever one draw gave. The **port** has exposed it since task H3 — `run --seed N` and
`bench --seed N`, recorded in `engine` (D §9, audit §C.3) — so the spread below can now be asked
for on the port's side as well as constructed by hand on the reference's; the reference's `cli.py`
still has no such flag, and the thirty-six runs of this table were made by setting `Params(seed=k)`
in a script. Running the reference with `Params(seed = 0…4)` and with the working-mesh budget at
±5 % (190 000 and 210 000 faces, seed 0) moves three of the seven rows — including the one written
as a prohibition (task Y, `notes/2026-09-07-y-phase1d-findings.md` §3) — and a five-seed sweep of
the three rows task Y left alone moves two more (task Z,
`notes/2026-09-08-z-phase1e-findings.md` §5). The runs per set are what the ranges below are made
of: seven for `pot_B`, `pot_G` and `synthetic_20` (five seeds and the two budgets), five for
`pot_A`, `pot_C` and `pot_H` (seeds alone). **Every one of the thirty-six is the reference's own
run**, and no range here was widened to admit a figure of the port's: the port's own numbers lie
inside the *old* rows as well, and what the sweep moved is the bottom of the band, not the top.

| set | gate | the reference's own seven runs |
|---|---|---|
| `input/test_fragments_1` | joins used exactly {021–094, 094–104}; 007 unplaced; both `pen` = 0; tight of both ≥ 0.27; the two seams within 20 % of 20.3 t and 10.7 t | not swept (the row is a set of decisions, and they are stable — see below) |
| `input/sfspp/pot_A` | fragment accuracy **87.5–100 %**, precision 1.000 | 87.5, 87.5, 100, 87.5, 100 % by seed; precision 1.000 in all five; 6 or 7 joins used and **every one correct**; groups 7+1 or one of 8 (task Z) |
| `pot_B` | fragment accuracy 88.9–100 %, precision 1.000 | 100, 88.9, 88.9, 100, 100 % by seed; 100, 100 % at ±5 % faces; precision 1.000 in all seven |
| `pot_C` | fragment accuracy **50–75 %**, precision 0.500–0.667 | 75, 75, 75, 75, **50** % by seed; precision 0.500, 0.667, 0.667, 0.500, 0.500; 2–4 joins used, of which 1–2 correct and the rest wrong-pose or unscorable (task Z; the note below measured the precision column at five seeds of the *old* thickness estimator and the accuracy column is now measured too) |
| `pot_G` | fragment accuracy 0 %; **at most two joins used, and every join used must be a wrong-pose join on a ground-truth-adjacent pair** (the ground truth interpenetrates) | 0, 1, 2, 1, 1 joins by seed; 0, 0 at ±5 % faces; 0 % accuracy in all seven, and every join used was a wrong-pose join on an adjacent pair (1.5–2.5°, 1.11–1.20 t) |
| `pot_H` | fragment accuracy **27.3–36.4 %**, precision **0.333–0.500** | 36.4, 36.4, **27.3**, 36.4, 36.4 % by seed; precision 0.429, 0.429, 0.333, 0.429, 0.500; 6 or 7 joins used, 2 or 3 of them correct; largest group 8 at four seeds of five (task Z) |
| `input/synthetic_pingsdorf_20` | fragment accuracy 85–95 %, precision 1.000 | 95, 95, 95, 85, 95 % by seed; 95, 95 % at ±5 % faces; precision 1.000 in all seven |
| all sets | cross-object joins 0; group purity 1.000 | held in all **36** runs (task Y's 21 and task Z's 15) — on these seven single-object collections. On the one *mixed* development set, `mixed_ABG`, neither implementation holds it, and D §10.3 carries that set as roadmap item 4's baseline rather than as a gate (decision 2026-09-08) |

**The two seam figures are measurements, and they have moved three times.** They were 21.3 t and
11.3 t in `2026-09-05-test-set-result.md`; the budget commits `991ff87` and `ca59c6a` took the
second to 10.7 t (p0 note §5); §3.2's deterministic ray set (task T1) took the first from 21.33 t to
20.33 t with the score 11.70 → 10.79 and the tight contact 0.548 → 0.531; and the **port**, whose
working mesh is PMC-2's own decimation rather than Open3D's, measures **20.667 t and 12.333 t**
(scores 11.50 and 6.60, tight 0.557/0.671 and 0.535/0.557) against the reference's own
20.333 t and 10.667 t (10.79 and 6.06, 0.531/0.687 and 0.593/0.568) on the same collection — +1.6 %
and +15.6 %, which is why the row above states a band rather than an "≈". The *decisions* have not
moved once: the same two joins, the same group, 007 unplaced, both penetrations 0, tight far above
0.27 on both. `pot_C`'s precision is the one number of this table that is not stable under §3.2's own
estimator — see the note below.

**`pot_C`'s precision of 0.667 is one draw of a coin, and this table should not have printed it
without saying so.** Only four of pot C's seven fragments have a ground-truth pose (05, 06 and 07
are `unknown`), so any join to the other three is *unscorable* and counts against precision by
construction, and piece 01 has no correct pose available at all: it is attached to a
ground-truth-adjacent neighbour at a wrong pose in every run measured. Its `tight` sits on
`min_tight` — 0.255 at the old estimator's seed 0, 0.250 with §3.2's deterministic rays — so which
of its marginal candidates is accepted flips on nothing. Running the *old* estimator at seeds 0–4
gives precision 0.667, 0.667, **0.500**, 0.667, 0.667 with fragment accuracy 75.0 % on all five;
the deterministic estimator gives 0.500 with the same 75.0 %.

**Task Z swept §3.2's deterministic estimator at the same five seeds, and fragment accuracy is not
stable either** — 75, 75, 75, 75 and **50 %**, with precision 0.500, 0.667, 0.667, 0.500, 0.500.
The sentence this paragraph used to end with ("fragment accuracy is what is stable here") was true
of the *old* estimator's five seeds and is now known to be false of the current one's: at seed 4 the
reference places two of the four scorable fragments instead of three, using two joins instead of
four. The cause is the same one the paragraph already names — piece 01 has no correct pose
available and its `tight` sits on `min_tight`, so which of its marginal candidates survives flips on
nothing — and what the fifth seed shows is that the flip can take a *placement* with it and not
only a precision digit. Neither column of this row is a single number; both are bands, and the port
sits at the top of both (75 % / 0.667).

Per-stage numeric tolerances for the fixture harness are defined in the port design
(`2026-09-06-rust-core-design.md`, §10.2). One of them used to be a property of the algorithm
rather than a rounding allowance: the native tolerance on `t` stood at
`max(2 %, 3 bins of the reference's own histogram)` because §3.2's estimator moved by up to 6.8 %
under a change of seed alone, and the consequence travelled — a fragment whose `t` differs by
6.6 % has every threshold of §1.2 shifted by 6.6 % for every pair it takes part in, which the
exact-set gates above cannot absorb. Task T1 removed the cause instead of carrying the risk:
§3.2's ray set is fixed by the mesh, both implementations evaluate the estimator on the same
faces, and the native tolerance is ±2 % again.

Measured cost structure of the reference (M2 Pro, single thread, one mid-size `mixed_all` pair,
42k/26k faces, scale-pairs note §3.1): stage-2 coarse ICPs 2.84 s (41 %), stage-2 fine ICPs
1.82 s (26 %), coarse score 0.99 s (14 %, 59 000 hypotheses), stage 1 0.55 s (8 %), penetration
0.30 s, `MatchData` 0.20 s, other verification 0.20 s, rest 0.03 s; total 6.94 s. A true pair
costs ~25 s (fine ICPs converge slowly). Preprocessing 5–15 s per fragment.
