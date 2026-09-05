# Roadmap: large mixed collections without ground truth

**Date:** 2026-09-05. Agreed with the museum team.

## Operating conditions the tool must meet

- Collections of 170+ fragments, each scan larger than the first test set (millions of faces).
- Fragments come from excavations: several objects mixed together, pieces missing, no ground
  truth ever. The output must be a set of separate objects plus an honest statement of confidence.
- Thick sculptural terracotta and thin pots/plates alike; rims, bases, handles, worn edges.
- Runs on the museum's own machines (macOS, Windows, Linux), later as a desktop app.

## Metrics and benchmarks

Quality is measured only on sets with ground truth; every algorithm change runs on all of them:

| set | what it covers | ground truth |
|---|---|---|
| `input/test_fragments_1` | thick terracotta, 4 pieces | museum's manual assembly |
| `input/sfspp/pot_A..J` | thin pots, 4–31 pieces each, real scans | poses + adjacency (SfS++) |
| `input/sfspp/mixed_all` | 164 pieces of 10 different pots | poses + adjacency + object id |
| `input/synthetic_*` | real digitized vessel fractured into 20/60/170, worn edges, missing pieces, intruders | exact |

Metrics (`tools/evaluate.py`): join precision/recall, wrong-object joins (must be 0),
fragment accuracy (comparable to SfS++ "Sherd Accuracy"), group purity, runtime.

## Work items, in order

1. **Resolution-aware thresholds** (in progress, branch `thin-walls`): every distance is
   `max(k·t, m·res)` with `res` the mesh edge length; sampling density per t² of surface;
   pair thickness = min of the two; robust thickness against rims. Terracotta result unchanged.
2. **Scaling to thousands of pairs**: partner search by rigid-invariant breakline signatures
   (curvature, torsion, thickness and dihedral profiles along the crack line; 10–15 partners
   per fragment), early exit after the breakline stage when no candidate passes, precomputed
   matching data in the cache with an LRU per worker. Target: 164 mixed pieces in < 1 hour.
3. **Confidence tiers and human in the loop**: *confirmed* joins (strict thresholds, zero false
   joins on all benchmarks) build the assemblies; *probable* joins are listed with a rendered
   picture of each pair for verification at the bench; *rejected* joins carry the reason. A
   constraints file (`must_join`, `must_not_join`) is honoured as hard conditions on re-runs.
4. **Object separation**: cycle consistency inside a group before accepting a join; per-group
   consensus of wall thickness, shell curvature and colour/texture as extra evidence; groups
   reported as separate objects with their consensus features.
5. **Group-level matching**: after the first joins, match assembled groups (union of their
   breaklines minus internal seams) instead of single fragments.
6. **Special parts**: handles and knobs (solid, no opposite wall) excluded from the fracture mask
   by a volume test; rim/base awareness in thickness; memory-bounded preprocessing of very large
   scans (fewer workers or streaming decimation).
7. **Native core**: once 1–6 are stable on all benchmarks, port the core to Rust (`nalgebra`,
   `rayon`, `kiddo`, `parry3d`, decimation crate), verified against the Python reference through
   `pyo3` on the same benchmarks; then GPU kernels with `wgpu` (Metal / Vulkan / DX12) for
   hypothesis scoring, batched nearest neighbours and ICP, signed distance and ray casting,
   with a CPU fallback; Tauri app on the same crate, built for macOS, Windows and Linux.

Rust/GPU is deliberately last: it must port a settled algorithm, and items 2–5 give more speed
per effort than any rewrite (8 232 pairs → ~1 000).
