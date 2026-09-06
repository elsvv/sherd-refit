# sherd-refit — frozen algorithm reference

**Date:** 2026-09-06. **Reference implementation:** `sherd_refit/*.py` at commit `9d4b9d3`
(branch `main`), running on Open3D 0.19.0, numpy 2.5.2, scipy ≥ 1.11, Python 3.12.
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
cached (§3.7). Two independent RNG streams are used: `rng_pre = rng(0)` for thickness rays
(seed is hard-coded 0), `rng_md = rng(p.seed)` for the match arrays.

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
idx  = rng_pre.choice(len(C0), min(20000, len(C0)), replace=False)      # face indices
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

**`t` is a sampled estimate, and its own spread is larger than one might assume.** Re-running the
reference's `estimate_thickness` with seeds 0–11 and nothing else changed moves it by up to 6.8 %
of the seed-0 value on `Pot_A_Piece_04_Mesh` (3.554 at seed 0 against 3.774–3.795 at seeds 1–11)
and 5.8 % on `frag_019`, because a fragment whose filtered distances form a plateau rather than a
peak puts several near-equal bins in contention and `argmax` picks by a count that a different
sample reorders. Any implementation that does not reproduce numpy's PCG64 stream — which PMC-9
allows the port not to — draws a *different sample of the same estimator*, and the difference it
must be allowed is this spread, not zero. D §10.2's native tolerance is set from it.

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
bounds convention as §3.4.1); `sub` = lowest-index point per occupied voxel in hash-map order
(**PMC-4**: port sorts ascending — this changes hypothesis order and therefore tie-breaking in
§5.2–5.3; must re-verify); `brk_sub = sub[valid[sub]]`.

**3.5.6 Shell margin.** `d_brk[i]` = distance from `S[i]` to the nearest breakline point (inf if
none). `margin = ¬frac[sp] ∧ (d_brk > 0.12·t) ∧ (d_brk < 1.5·t)` (**PMC-5**: these two are not
resolution-floored). `margin_idx = sort(rng.choice(where(margin), 6000, replace=False))` if more
than 6000, else `where(margin)`.

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
             nf' = max(1, round(nf·reg_points/(nf+nm))), nm' = reg_points − nf'   else nf' = nf, nm' = nm
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

Open3D's sign: closest-point distance, sign from the parity of intersections of one ray from the
query point (**PMC-7**: the port may use a different ray direction or the angle-weighted
pseudo-normal; `pen` must agree within 0.0005 on the parity gates).

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
corr(P): for each i: j = nearest target point (KD-tree) with |P_i − Q_j| ≤ d, else none
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
| `rng_pre` (§3.2) | 0 (hard-coded) | 1 `choice(n_faces0, 20000, replace=False)` |
| `rng_md` (§3.5) | `p.seed` | `choice(len(idx), 20000, p)`, `random(20000)`, `random(20000)`; `choice(len(idx_frac), n_frac, p)`, `random(n_frac)`, `random(n_frac)`; `choice(margin, 6000, replace=False)` only if `|margin| > 6000` |
| `rng_pair` (§5.2) | `p.seed` | 1 `choice(B.brk_sub, ≤60, replace=False)` |
| screening `cap` (§4.3) | `p.seed` | fresh generator per call |
| assembly `MatchData` (§8) | `p.seed` | as `rng_md` with 15000 surface points, at `t_med` |
| refinement (§9) | 0 | `choice(idx, 150000, replace=False)` only if `|idx| > 150000` |
| previews (§11.5) | 0 | one generator for the whole preview pass |

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
  "timings": {"preprocess", ["screen"], "matching", "assembly", ["second_pass"], ["refine"], "output"},
  "joins_used": [candidate JSON…], "joins_rejected": [candidate JSON + "reason"], "candidates": [candidate JSON…] }
candidate JSON = { "a", "b", "T": 4×4 lists, "accepted", "score", <every score key as float> }
```
`candidates` are in pair order, best first within a pair.

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

---

## 13. Reference results and parity gates

Numbers the port must reproduce on the benchmark sets, with the defaults above (from the notes
`2026-09-06-scale-pairs.md` §3 and `2026-09-05-test-set-result.md`):

| set | gate |
|---|---|
| `input/test_fragments_1` | joins used exactly {021–094, 094–104}; 007 unplaced; both `pen` = 0; 021–094 seam ≈ 21.3 t, 094–104 seam ≈ 11–12 t; tight of both ≥ 0.27 |
| `input/sfspp/pot_A` | fragment accuracy 87.5 %, precision 1.000 |
| `pot_B` | 100 %, 1.000 |
| `pot_C` | 75 %, 0.667 |
| `pot_G` | 0 % (ground truth interpenetrates; no join must be accepted) |
| `pot_H` | 36.4 %, 0.429 |
| `input/synthetic_pingsdorf_20` | 95 %, 1.000 |
| all sets | cross-object joins 0; group purity 1.000 |

Per-stage numeric tolerances for the fixture harness are defined in the port design
(`2026-09-06-rust-core-design.md`, §10.2). One of them is not a rounding allowance but a property
of the algorithm: the native tolerance on `t` is `max(2 %, 3 bins of the reference's own
histogram)`, because §3.2's estimator moves by up to 6.8 % under a change of seed alone. The
consequence travels — a fragment whose `t` differs by 6.6 % has every threshold of §1.2 shifted by
6.6 % for every pair it takes part in — and the gates in the table above are exact-set gates that
cannot be widened to absorb it. That is a risk to carry into the pair stages, not a tolerance to
add there.

Measured cost structure of the reference (M2 Pro, single thread, one mid-size `mixed_all` pair,
42k/26k faces, scale-pairs note §3.1): stage-2 coarse ICPs 2.84 s (41 %), stage-2 fine ICPs
1.82 s (26 %), coarse score 0.99 s (14 %, 59 000 hypotheses), stage 1 0.55 s (8 %), penetration
0.30 s, `MatchData` 0.20 s, other verification 0.20 s, rest 0.03 s; total 6.94 s. A true pair
costs ~25 s (fine ICPs converge slowly). Preprocessing 5–15 s per fragment.
