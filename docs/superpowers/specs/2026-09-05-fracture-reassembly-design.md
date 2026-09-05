# Reassembling 3D-scanned ceramic fragments — design

**Date:** 2026-09-05. **Status:** validated on `input/test_fragments_1` with prototypes before writing.

## Goal

A command-line tool that takes a folder of 3D-scanned fragments (coloured PLY meshes) of a
broken ceramic object and writes the most probable reassembly: a rigid transform per fragment,
the placed meshes, a merged model per assembled group, a preview image and a report of which
joins were accepted and why. No GUI, CPU only, Python 3.12, macOS.

It must work without per-object tuning on: 4 fragments or dozens; thick-walled sculptural
terracotta (the first test set), and pots or plates later; collections with missing pieces or
pieces from other objects (those must stay unplaced rather than be forced in).

## Why not the previous attempt

Structure-from-Sherds++ assumes an axially symmetric vessel and reassembles by aligning each
sherd's surface-of-revolution axis plus breaklines. The first test set is a sculptural object
(rectangular relief, no axis), so the axis estimate is meaningless and the result was wrong:
it claimed 007–021–104 and rejected 094, while the museum's manual assembly is 104–094–021 with
007 left over. See `docs/superpowers/notes/2026-09-05-papers-*.md` for the literature review;
the papers converge on fracture-surface matching (Huang et al. 2006 family) for thick fragments,
with breaking curves as the cheap pose generator. That is what this design implements.

## What the data looks like (measured)

| | value |
|---|---|
| fragments | watertight, single-component, 350k–670k vertices, ~0.65 units median edge |
| extent | 370–680 units; wall thickness `t` ≈ 39 units (mode of inward-ray hit distance) |
| fracture surface | 12–20 % of area, rough at the 5–10 unit scale |
| intact shells | smooth but with a block-pattern decimation noise from Geomagic |
| colour | uniform terracotta, not usable to separate fracture from shell |

Every threshold below is expressed in units of `t`, so the pipeline is scale-free.

## Pipeline

```
load ─► preprocess ─► segment shell/fracture ─► breaklines + frames
     ─► pairwise: hypotheses → coarse score → breakline ICP → full ICP → verify
     ─► global assembly (greedy, penetration + consistency) ─► full-res refinement
     ─► outputs
```

### 1. Preprocess (`preprocess.py`)

Quadric decimation to ≤ 200k faces (working mesh; the original is kept for output and final
refinement), 3 iterations of Taubin smoothing, face normals/areas/centroids, an Open3D
`RaycastingScene`. Wall thickness `t` = mode of the inward-ray hit distance over 20k random
faces. The collection thickness is the median over fragments; a fragment whose thickness
differs by more than 40 % is flagged in the report.

### 2. Shell / fracture segmentation (`segment.py`)

Validated criterion (the roughness criteria from the literature failed on this data because of
the block noise): a face belongs to a **shell** if a cone of 7 rays (15°) cast inward along its
smoothed normal (area-weighted, radius `t/3`) hits the opposite wall at `0.5t–1.8t` from behind
(hit-face normal aligned with the ray, cos > 0.7) for at least 5 of 7 rays. Everything else is
**fracture**. Then: majority filter (radius `t/4`), drop fracture islands `< 0.5 t²` and shell
islands `< 2 t²` (face-adjacency components), and refine the boundary by growing the shell into
the band while face normals stay within 25° of the original shell's normal within `t/2`.
Measured on the test set: fracture fraction 14–20 %, one connected band per fragment, boundary
within ~0.1 t of the crease.

### 3. Breaklines (`segment.py`)

Breakline points are midpoints of mesh edges between a shell face and a fracture face. Each
carries macro normals: `n_s` = area-weighted normal of shell faces within `0.35t`, `n_f` = same
for fracture faces. Frame `[tangent, n_s, f]` with `f` = `n_f` orthogonalised against `n_s`,
`tangent = n_s × f`. Dihedral = angle(`n_s`, `n_f`), typically 40–100°. Points are voxel
subsampled at `t/3` for hypothesis generation (250–400 per fragment).

Also kept per fragment: 30k area-weighted surface samples with normals, split into fracture
points and the shell margin (shell points within `1.5t` of the breakline).

### 4. Pairwise matching (`matching.py`)

For fragments A, B every pair of subsampled breakline points (p ∈ A, q ∈ B) with
`|dihedral_A + dihedral_B − 180°| < 40°` defines one rigid transform: B's frame
`[−tangent, n_s, −f]` is mapped onto A's `[tangent, n_s, f]` (same shell surface, curve traversed
in opposite directions, fracture normals opposed). 40k–75k hypotheses per pair.

1. **Coarse score**: fraction of 60 random B breakline points landing within `0.15t` of A's
   breakline with shell normals agreeing (cos > 0.7). Non-max suppression (0.5t / ~18°), keep 400.
2. **Stage 1**: point-to-point ICP of B's breakline onto A's (0.2t then 0.08t), re-score at
   `0.06t`, NMS, keep 40.
3. **Stage 2**: point-to-plane ICP on fracture + shell-margin points (0.2t, 0.08t), then on
   fracture points only (0.08t, 0.04t).
4. **Verification** per candidate:
   - `tight`: of B fracture points facing A (nearest A fracture point < 0.3t), fraction within
     `0.04t`; same for A; `tight = min`. `gap` = max of the two median facing distances.
   - `seam`: length (in `t`, counted in `t/3` voxels) of A's breakline within `0.12t` of B's with
     agreeing normals.
   - `pen`: fraction of either fragment's surface samples inside the other by more than `0.06t`
     (signed distance on the watertight working mesh).
   - `cont_n`: median normal agreement of B's shell margin with A's nearest margin points.
   - `contact` area = min(covered fracture area of A, of B) / t².

   A join is **accepted** if `tight ≥ 0.25`, `gap ≤ 0.065`, `pen ≤ 0.005`, `seam ≥ 3`,
   `cont_n ≥ 0.8`. Ranking score = `seam × tight`.

Measured on the test set: true joins 021–094 (seam 21, tight 0.29/0.40, gap 0.05) and 094–104
(seam 11, tight 0.46/0.31, gap 0.04); every false candidate has tight ≤ 0.17 on at least one
side or gap ≥ 0.065. Runtime ≈ 25 s per pair single-threaded; pairs run in a process pool.

### 5. Global assembly (`assembly.py`)

Accepted joins sorted by score. Greedy: seed with the best join, then repeatedly add the best
join that connects an unplaced fragment to a placed one, verifying against **every** placed
fragment that penetration stays ≤ 0.005 and that, if the new fragment also has an accepted join
to another placed fragment, the implied relative pose agrees within 10° / 0.5t (otherwise the
join is rejected and the next is tried). When no join can extend the group, start a new group
from the best remaining join. Fragments with no accepted join stay unplaced (identity pose,
listed as "not assembled"). Groups are connected components of the accepted graph; several
groups mean partial assemblies or mixed objects.

Optional loop closure: if the accepted graph has cycles, run Open3D pose-graph optimisation
with the pairwise transforms as edges before the final refinement.

### 6. Full-resolution refinement

For every accepted join, point-to-plane ICP between the original meshes' fracture vertices
(vertices whose nearest working-mesh face is fracture), 0.05t then 0.02t, applied along the
spanning tree from the group's root. This removes the ~0.5-unit bias of decimation/smoothing.

### 7. Outputs (`report.py`, `render.py`)

```
<out>/transforms.json      {fragment: {matrix 4x4, group, placed}}, thickness, parameters
<out>/report.md, report.json   accepted joins with all scores, rejected top candidates per
                           pair, groups, unplaced fragments, per-fragment thickness/fracture stats
<out>/placed/<name>.ply    original-resolution coloured mesh in its placed pose
<out>/assembly_<k>.ply     merged coloured mesh per group with ≥ 2 fragments
<out>/preview_<k>.png      software-rendered views (no GPU; Open3D offscreen is unavailable on macOS)
<out>/cache/<name>.npz     per-fragment preprocessing cache (restartable runs)
```

## CLI

```
sherd-refit run INPUT_DIR --out OUT_DIR [--target-faces 200000] [--workers N]
           [--candidates 40] [--min-tight 0.25] [--no-preview] [--no-refine]
sherd-refit segment INPUT_DIR --out OUT_DIR       # only preprocessing + segmentation previews
```

## Error handling

- Non-watertight or multi-component meshes: keep the largest component; if still not
  watertight, signed distance is unavailable → penetration test is skipped for that fragment and
  the report says so.
- Thickness estimation failure (no clear mode): fall back to 1/10 of the smallest OBB extent.
- A pair with zero hypotheses or zero accepted candidates is a normal outcome, reported as "no
  join".
- Every stage logs timing; caches let a crashed run resume.

## Testing

- Unit tests on synthetic data: a slab with a bumpy cut into two pieces (known transform):
  segmentation finds the fracture band, matching recovers the transform within 1° / 0.05t.
- Integration test on `input/test_fragments_1`: accepted joins are exactly {021–094, 094–104},
  007 unplaced, 021 and 104 do not penetrate.

## Non-goals

Axially-symmetric priors, deep learning, GUI, texture-based matching, hole filling. Colour is
loaded and carried to outputs but not used for matching.
