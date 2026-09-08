# sherd-refit — native core: Rust + wgpu port design

**Date:** 2026-09-06. **Status:** design, no code. Implements roadmap item 7 of
`2026-09-05-roadmap-scale-and-mixed.md`. The algorithm being ported is frozen in
`2026-09-06-algorithm-reference.md` ("the reference"); this document never restates it, only
cites it (§ numbers prefixed `R`).

## 0. Decisions in one table

| topic | decision |
|---|---|
| language / build | Rust (edition 2024, MSRV 1.85), Cargo workspace inside this repository under `crates/` |
| algorithm | byte-for-byte the reference; no algorithmic changes in phases 1–2 except the listed PMC items |
| precision | f32 storage and kernels; f64 for pose composition, 6×6 solves and Umeyama on the CPU path |
| parallelism | `rayon` over pairs and candidates in one process; no worker processes |
| GPU | `wgpu` + WGSL, backends Metal / Vulkan / DX12; kernels: coarse scoring, batched ICP (both estimators), bounded point-to-mesh distance, inside test; cone casts later if profiling asks |
| CPU fallback | the same kernels written once in Rust against the same buffer layouts; results within the tolerances of §10.2 |
| spatial structures | own hash grid (radius-bounded NN) and own flattened BVH (rays, closest point, parity), shared CPU/GPU layouts; `kiddo` only for unbounded nearest-neighbour queries on the CPU |
| decimation | `meshopt` (meshoptimizer bindings) by default, `baby_shark` as the fallback experiment |
| mesh IO | PLY (own reader/writer, binary+ASCII, colours), OBJ/STL via `tobj`/`stl_io`, GLB via `gltf` |
| cache | `safetensors` container per fragment, mmap-loaded, versioned |
| parity | Python-side fixture dumps (`.npy` + manifest) at every stage boundary; Rust runs each stage from injected inputs or natively; a Python comparison tool applies per-stage tolerances |
| determinism | fixed seeds, portable RNG, stable sorts with index tie-breaks, fixed-order reductions, per-candidate convergence flags; run-to-run identical per backend, CPU vs GPU within tolerance |
| bindings | `pyo3` + `maturin` wheel `sherd_refit_core` used by the Python package behind `SHERD_REFIT_BACKEND=rust` during the transition; Tauri 2 desktop app on the same crate afterwards |

## 1. Goals, targets, non-goals

Targets (from the brief and the notes):

| metric | Python today (M2 Pro, 9 workers) | target CPU-only | target with GPU |
|---|---|---|---|
| mid-size pair (42k/26k faces) | 6.9 core-s | ≤ 2.5 core-s | ≤ 0.15 s GPU + 0.05 core-s |
| full-resolution pair (200k faces each) | 20–25 core-s | ≤ 7 core-s | ≤ 0.4 s GPU |
| 170 fragments, ≈ 14 000 pairs, end to end | ≈ 4.6 h (projected) | ≤ 2 h (10 cores) | ≤ 30 min (M2 Pro 16-core GPU) |
| preprocessing per fragment (1–10 M faces) | 5–15 s | ≤ 4 s typical, ≤ 15 s for 10 M faces | same (CPU) |
| peak RSS, 170 fragments | 9 × 384 MB workers, but 16 GB exhausted on large scans | ≤ 6 GB (≤ 3 M faces/scan); ≤ 10 GB (10 M) | + ≤ 1 GB GPU |
| quality | reference §R13 | identical gates | identical gates |

Non-goals of the port: new matching algorithms, deep learning, changing thresholds, a GUI in
phases 1–2, OpenGL/WebGPU-in-browser backends, f16 or f64 GPU paths.

## 2. Workspace layout

```
sherd-refit/                      (this repo; Python package stays at the root during the transition)
  Cargo.toml                      workspace
  crates/
    sherd-core/                   library: everything below except GPU and bindings
      src/io/        ply.rs obj.rs stl.rs off.rs glb.rs writer.rs
      src/mesh/      clean.rs components.rs decimate.rs taubin.rs geometry.rs adjacency.rs
      src/spatial/   bvh.rs grid.rs kdtree.rs        (layouts shared with sherd-gpu)
      src/fragment/  thickness.rs segment.rs breakline.rs samples.rs cache.rs features.rs
      src/matching/  scales.rs hypotheses.rs coarse.rs nms.rs icp.rs verify.rs pair.rs screen.rs
      src/assembly/  greedy.rs consistency.rs groups.rs constraints.rs
      src/refine.rs  src/report.rs src/render.rs src/pipeline.rs src/executor.rs src/rng.rs src/fixture.rs
    sherd-gpu/                    wgpu executor: device.rs buffers.rs slots.rs selftest.rs executor.rs kernels/*.wgsl
                                  (scheduler.rs is phase 2d; the kernels are 2b and 2c)
    sherd-cli/                    binary `sherd-refit`: run, segment, parity, bench, info
    sherd-py/                     pyo3 module `sherd_refit_core` (maturin, its own pyproject.toml)
    sherd-parity/                 fixture reader/writer + stage runners used by `sherd-refit parity`
  apps/desktop/                   Tauri 2 app (phase 3)
  fixtures/slab/                  the synthetic slab pair (self-made, redistributable) + expected outputs
  tools/dump_fixtures.py          Python side of the parity harness (§10.1)
  tools/compare_fixtures.py       stage-by-stage comparison with tolerances (§10.2)
```

`sherd-core` has no GPU dependency and compiles on every target; `sherd-gpu` is an optional
feature of `sherd-cli` (`gpu`, in `default`), so `--no-default-features` builds a CPU-only binary
with no wgpu in the tree at all. Both shapes are built by the `check` and `test` jobs of §10.5.

## 3. Dependencies

**This table is the workspace, not a plan.** Every version below is what `Cargo.toml` pins and
`Cargo.lock` resolves at the head of `rust-core`; the phase-0 experiments moved several of them and
deleted two rows outright, and the table was re-synchronised with the tree after the phase-1a
verification (S1 open issue 2). Rows for crates that are *not* in the workspace yet say so: their
members (`sherd-py`, the desktop app) join in phase 3a, and pinning their dependencies before then
would be guessing. `sherd-gpu` joined in phase 2a and its row is now the tree's.

| need | crate, as pinned | why | what phase 0 changed | experiment |
|---|---|---|---|---|
| linear algebra | `nalgebra` 0.34.2 | f64 poses, 6×6 LDLT, 3×3 SVD (Umeyama), the 3×3 symmetric eigen of R §3.2's OBB fallback | 0.33 → 0.34.2 (current at S1) | — |
| small vectors in hot loops | own `Vec3f` (`#[repr(C)]`, `bytemuck` 1.25.2 `Pod`) | identical layout on CPU and GPU buffers | — | — |
| parallelism | `rayon` 1.12.0 | fragments, pairs, candidates, per-face loops | 1.10 → 1.12.0 | — |
| RNG | `rand_chacha` 0.10.0 (`ChaCha8Rng::seed_from_u64`) | portable, versioned stream guarantee across platforms | 0.9 → 0.10.0 | not numpy-compatible (PMC-9) |
| BVH: rays, closest point, inside | `parry3d` 0.30.2, feature `enhanced-determinism`, `TriMesh` **without** `TriMeshFlags::ORIENTED` | R §3.2's rays, R §3.4.3's cone, R §6.1's closest point, R §6.4's inside test; also the convex hull of R §3.2's OBB fallback | **the own flattened BVH was dropped.** E3/E4 measured parry against Open3D's `RaycastingScene` and it passed, saving ≈ 1.5 weeks; the `ORIENTED` flag is off because its pseudo-normals are wrong on decimated fracture surfaces (29 of 30 000 points on one closed fragment), so the inside test is ray parity only | E3/E4 |
| KD-tree and radius-bounded NN | `kiddo` 6.2.0 (`ImmutableKdTree<f32, 3>`) | seam and continuity tests, `near`, `d_brk`, margin, and ICP correspondences | **the own hash grid was dropped for the CPU**: E3 measured kiddo's bounded queries fast enough, so there is one structure to maintain instead of two. The GPU executor still wants a grid, and it arrives with it. kiddo's MSRV is what sets `rust-version = "1.89"` | E3 |
| quadric decimation | `meshopt` 0.6.2, `SimplifyOptions::Regularize` | fast, topology-preserving, cross-platform, and it never moves a vertex — the readers' f64 coordinates survive decimation exactly | 0.4 → 0.6.2, and **`baby_shark` was dropped**: E1 measured meshopt 20× faster (1.73 s against 34.70 s) and inside every gate, but *only* with `Regularize` — plain meshopt misses the `res` gate on 13 of 14 meshes | E1 |
| mesh read | `ply-rs-bw` 4.0.1 (PLY), `tobj` 4.0.5 (OBJ), `stl_io` 0.11.0 (STL), `gltf` 1.4.1 `default-features = false, features = ["utils"]` (GLB), own ~40-line OFF reader | PLY is the main format and needs colours and speed; the rest are the benchmark's | **no own PLY parser**: E2 measured `ply-rs-bw` bit-identical to Open3D on all eleven PLY variants and 0.057 s on a 25 MB scan. OFF has no crate at all, so that one is ours. `gltf`'s `import` feature is off: 36 → 17 transitive crates | E2 |
| mesh write | own PLY writer (binary LE, `uchar` RGB) | R §11.4's files must match Open3D's byte for byte, and they do | `gltf-json` is not in the workspace: GLB export is D §9 and arrives with the desktop app | E2 |
| cache | `safetensors` 0.8.0 | mmap-friendly, JSON header, readable from Python | 0.4 → 0.8.0. Its metadata is a `HashMap` serialised in iteration order, which is why §4.2's metadata is one JSON object under one key | — |
| fixtures | `npyz` 0.9.1 (read `.npy`), `serde` 1.0.229, `serde_json` 1.0.151 **with `float_roundtrip`** | Python writes `.npy` and JSON scalars; no zip needed | `float_roundtrip` is mandatory, not a preference: without it serde_json's fast float path misrounds `29.864871978759766` by one ULP, which is the whole difference between a fixture scalar and the float32 it was written from | — |
| fixture checksums | `sha2` 0.11.0 | `--verify-checksums` re-hashes a dump against its manifest | added in S1; not in the original table | — |
| images | `image` 0.25.10, `default-features = false, features = ["png"]` + an embedded 5×7 bitmap font | preview PNGs without a font stack | — | — |
| CLI / logging / errors | `clap` 4.6.6 (`derive`), `tracing` 0.1.44, `tracing-subscriber` 0.3.23 (`env-filter`), `anyhow` 1.0.104, `thiserror` 2.0.20 | — | thiserror 1 → 2 | — |
| tests | `proptest` 1.11.0, `approx` 0.5.1, `cargo nextest` in CI | — | `criterion` is not in the workspace: the benchmark harness is phase 1e | — |
| GPU | `wgpu` **`=30.0.1`** (`default-features = false`, features `metal`, `vulkan`, `dx12`, `wgsl` — no GL, D §6.8), `bytemuck` 1.25.2, `pollster` 0.4.0, `rayon` | Metal/Vulkan/DX12 from one WGSL source | **in the workspace since phase 2a (task G1)**, and the pin is exact rather than a floor: E7 read the shader-compile path of this `wgpu-hal` (`src/metal/device.rs:272`, which never touches `fastMathEnabled`) and measured the arithmetic that follows from it, so a minor bump can move the parity table. D §3 said "wgpu 24", which was six majors stale; 28.0.0 needs Rust 1.92 and will not build here. `sherd-gpu` is an **optional feature of `sherd-cli`, on by default** — `--no-default-features` gives a binary with no wgpu, no naga and no driver dependency, and CI builds both shapes | E7, E8 |
| Python bindings | `pyo3` 0.23 + `numpy` 0.23, `maturin` | transition and the parity harness | **not in the workspace yet** — `sherd-py` is phase 3a | — |
| desktop | `tauri` 2 | later | **not in the workspace yet** | — |

Workspace-wide settings that are part of the contract rather than taste: `edition = "2024"`,
`resolver = "3"`, `rust-version = "1.89"` (kiddo's MSRV, and `rust-toolchain.toml` pins 1.97.0),
`Cargo.lock` committed, `unsafe_code = "deny"` with a per-module `allow` and a reason,
`clippy::todo` / `unimplemented` / `dbg_macro` denied so a port of a frozen algorithm cannot ship a
hole, `lto = "thin"` and `codegen-units = 1` in release, and `opt-level = 2` for dependencies in
debug builds because the tests run on real scans.

Experiments (each is a small Rust or Python script, run before the phase that depends on it):

- **E1 decimation.** Decimate the 4 terracotta scans and pots A/B (full-resolution OBJs) with
  Open3D, `meshopt::simplify` (target index count = 3·target, `target_error = ∞`, lock-border
  off) and `baby_shark`. Compare: face count reached (must be within 5 % of the target),
  boundary-edge fraction (must satisfy `closed_enough`), `res` (within 10 % of Open3D's),
  thickness estimate on the working mesh (within 2 %), and the segmentation agreement after
  Taubin (≥ 0.97 area-weighted). Pick the fastest that passes; if none passes, own
  Garland–Heckbert (≈ 1 week).
- **E2 IO.** Round-trip every benchmark file (PLY binary/ASCII, OBJ with colours, GLB) through
  the readers and the PLY writer; compare vertex/face counts and colours with Open3D's reader.
- **E3 hash grid vs KD-tree.** ICP correspondence search on `pc_reg`/`pc_frac` clouds at the four
  ladder radii: expected ≥ 3× faster than `kiddo` bounded queries; if not, use a left-balanced
  implicit KD-tree for the CPU and keep the grid for the GPU.
- **E4 BVH parity.** Closest-point distances and signed distances of the terracotta samples
  against Open3D's `RaycastingScene`: |Δd| ≤ 1e-4·t; sign flips only at |d| < 1e-4·t.
- **E5 f32 ICP.** Run the injected-fixture ICPs of the terracotta pairs in f32 and f64; the f32
  result must stay within §10.2 tolerances (expected: 1e-5 t). **Done in task C2, and the answer is
  no** (`notes/2026-09-07-c2-icp.md` §6): with the point loops in `f32` and the pose and 6×6 solve
  still `f64`, terracotta's stage-2 pose is 1.9e-6 t out at the median but 0.257 t at p99, and on
  the thin-walled pots — where the same absolute `f32` step is many more wall thicknesses — the
  *median* is 0.115 t (pot B), 2.33 t (pot C) and 9.80 t (pot G). It is outside §10.2 on all seven
  sets. The stage-1 point-to-point rungs survive in the bulk (p99 inside 0.01 t on five of seven);
  stage 2 does not, and stage 2 is where the time goes. The ladders stay in `f64`; a GPU executor that wants
  `f32` has to shrink the coordinates first (§7).
- **E6 numpy RNG replication** (optional, only if bit-parity in native mode is demanded): port
  SeedSequence + PCG64 + `Generator.choice`/`random`; ≈ 2 days; recommendation: do not.
- **E7 naga fast-math.** Inspect generated MSL/SPIR-V/HLSL and the compile options wgpu-hal
  uses; a test kernel summing 1e7 terms in fixed order must match the CPU sum bit-for-bit
  when both use the same order and no FMA contraction is emitted.
- **E8 adapter matrix.** Run the self-test (§6.8) on: Apple M-series (Metal), NVIDIA (Vulkan
  and DX12), AMD (Vulkan/DX12), Intel Iris Xe (Vulkan/DX12), lavapipe, WARP.

## 4. Data model and cache format

### 4.1 In-memory types (`sherd-core`)

```rust
pub type FragId = u32;
pub struct SourceRef { path: PathBuf, size: u64, mtime_ns: i128, sha256: Option<[u8;32]> }
#[repr(u8)] pub enum FaceLabel { Shell = 0, Fracture = 1, Solid = 2, Rim = 3 }   // 2, 3 reserved for roadmap item 6
pub struct WorkingMesh { v: Vec<Vec3f>, f: Vec<[u32;3]>, fn_: Vec<Vec3f>, area: Vec<f32>, centroid: Vec<Vec3f>, res: f32 }
pub struct Fragment {
    id: FragId, name: String, source: SourceRef,
    thick: f64, thick_mode: f64,           // f64, not f32 -- see below
    watertight: bool, n_boundary: u32, n_orig_vertices: u32, n_orig_faces: u32, target_faces: u32,
    face_budget: u32, area0: f64,          // R§3.3's budget and its numerator; the fixtures carry both
    area: f64, frac_area: f64,             // R§3.4's two sums, kept rather than recomputed
    mesh: WorkingMesh, labels: Vec<FaceLabel>,
    brk: Breaklines, samples: Samples,     // both built at `thick`
    features: Features,                    // roadmap items 4 and 6, §11
    bvh_full: OnceLock<Arc<Bvh>>, bvh_frac: OnceLock<Arc<Bvh>>,   // built on first use, shared
}
pub struct BrkParams { t: f64, macro_inner: f64, macro_outer: f64, brk_voxel: f64 }
pub struct Breaklines { params: BrkParams, p: Vec<Vec3f>, ns: Vec<Vec3f>, nf: Vec<Vec3f>, f: Vec<Vec3f>, sub: Vec<u32> }
pub struct SampleParams { t: f64, seed: u64, surface_points: u32, frac_per_t2: f64, min_frac_points: u32, max_frac_points: u32, margin_points: u32 }
pub struct Samples { params: SampleParams, s: Vec<Vec3f>, sp: Vec<u32>, pf: Vec<Vec3f>, fp: Vec<u32>, margin_idx: Vec<u32> }
pub struct MatchData<'a> { fragment: &'a Fragment, t: f64, brk: Cow<'a, Breaklines>, samples: Cow<'a, Samples>,
                           surface_normals: Vec<Vec3f>, surface_fracture: Vec<bool>, fracture_normals: Vec<Vec3f>,
                           brk_tangent: Vec<Vec3f>, brk_dih: Vec<f64>, margin: Cloud,
                           pc_reg: Cloud, pc_frac: Cloud, pc_brk: Cloud, pc_brk_full: Cloud,
                           kd_brk: Option<PointTree>, kd_margin: Option<PointTree>, frac_area: f64 }
pub struct Cloud { p: Vec<Vec3f>, n: Vec<Vec3f> }                                // SoA-friendly, `Pod`
pub struct Pose(nalgebra::Isometry3<f64>);                                          // candidate T, poses
pub struct Scores { tight_a, tight_b, tight, gap_a, gap_b, gap, contact_a, contact_b, contact, seam, gap_limit, tight_delta, cont, cont_n, pen, pen_depth: f64, pen_unavailable: bool, partial: bool, brk: f64, brk_best: f64 }
pub struct Candidate { a: FragId, b: FragId, t: Pose, scores: Scores, accepted: bool, tier: Tier }
pub enum Tier { Confirmed, Probable, Rejected(RejectReason) }                       // item 3; phase 1 sets Confirmed ⇔ accepted
pub struct Constraints { must_join: Vec<(String, String)>, must_not_join: Vec<(String, String)> }   // item 3
pub struct Group { members: Vec<FragId>, consensus: GroupFeatures }                 // item 4
pub struct Params { /* every field of R§1.1, same names and defaults */ }
pub struct RunOptions { target_faces: u32, threads: Option<usize>, backend: Backend, memory_budget: Option<u64>, preview: bool, refine: bool, write_meshes: bool, keep_per_pair: usize, fixtures: Option<FixtureConfig>, constraints: Option<Constraints> }
```

`Scores` is a struct, not a map: the report writer serialises it to the same JSON keys as the
Python (`R§6.5`), including the optional ones only when set.

`MatchArrays`/`MdParams` were one struct in the original sketch; the implementation split them in
two along the line R §3.7 already draws, because the two halves are invalidated by different
knobs: `Breaklines`/`BrkParams` (steps B2, R §3.5.3–3.5.5) carry `t` and the three radii, and
`Samples`/`SampleParams` (step B3, R §3.5.1–3.5.2 and §3.5.6) carry `t`, the seed and the four
counts. A cache whose one half is stale has only that half recomputed (§4.2). `t` is `f64` in both,
for the reason `thick` is.

Three notes on the types above, all of them things the implementation settled and this document
was corrected to follow (phase-1a verification, finding F11):

* **`thick` and `thick_mode` are `f64`**, though the ray estimate is an `f32` value that an `f64`
  holds exactly. `t` is the unit of every threshold in R §1.2 and every `k·t` is computed in
  `f64`, and R §3.2's OBB fallback is a genuine `f64`; the wider type is the strict superset and
  costs nothing (the cache carries them as text either way).
* **`face_budget` and `area0` are part of the struct**, because the fixture sink dumps
  `thick.target` as `{target, area0, faces0, target_faces}` and the parity harness compares the
  first two directly.
* **The two BVHs are `OnceLock<Option<Arc<RayScene>>>`**, not `OnceLock<Arc<Bvh>>`: `parry3d`
  refuses a mesh with no triangle, so "built" and "buildable" are different states and the type
  says so. `Fragment::surface_scene` (R §6.4) and `Fragment::fracture_scene` (R §6.1) build them on
  first use and every pair the fragment takes part in borrows the same one; a 150 000-face mesh
  costs 40 ms to build and a fragment of a ten-fragment collection is in nine pairs (step C3).
* **`Candidate` has no `tier`** in phase 1. The field above belongs to roadmap item 3's confidence
  bands, and in phase 1 it would be `accepted` under a second name; it arrives with the constraint
  solver that reads it (§11).
* **`WorkingMesh` is `f32`** — `v`, and `res` with it — and `fn_`, `area` and `centroid` are
  derived from the *narrowed* vertices, not from the `f64` ones they came from. That is what makes
  a cold run and a cache hit bit-identical, since the cache stores `V`, `F` and `res` and both
  paths must derive the rest the same way. Everything upstream of the narrowing — the readers,
  cleaning, decimation, Taubin, `face_geometry`, `median_edge`, `ΣA` — is `f64`, which is what
  makes the injected parity comparisons exact. R §0 says the reference is `f64` throughout, so
  this is a deviation from it and R §12 carries it as **PMC-15**; the ≈6e-8 relative error enters
  every R §1.2 threshold and every ICP residual, and the native working-mesh row of §10.2 is what
  measures it.

### 4.2 Fragment cache: `<out>/cache/<name>.sherd`

A `safetensors` file. Tensors (all little-endian): `V f32[n,3]`, `F u32[m,3]`, `labels u8[m]`,
`S f32[20000,3]`, `sp u32`, `Pf f32`, `fp u32`, `brk_P/brk_ns/brk_nf/brk_f f32[k,3]`,
`brk_sub u32`, `margin_idx u32`, optional `features/*`. Phase 1a writes `V` and `F`, step B1
`labels`, step B2 the five `brk_*`, step B3 the five sampled ones; each later stage adds its own
tensors beside them, and the reader ignores what it does not know. `cache_version` moves when the
set changes *meaning* — which includes a tensor becoming one the reader requires: `labels` took it
from 1 to 2, the `brk_*` from 2 to 3 and `S`/`sp`/`Pf`/`fp`/`margin_idx` from 3 to 4, so a cache
written before any of them existed is refused and recomputed rather than read back half empty.
`S` and `Pf` are `f32` for the same reason `V` is (§4.1), and everything derived from them —
`d_brk`, R §3.5.6's band, `Pm` — is computed from the *narrowed* values, so every stored array is
a function of the other stored arrays. The reader checks what the writer cannot: the four frame
arrays must describe the same points and `brk_sub` must index them; `sp`/`fp` must describe
`S`/`Pf` and name faces of `F`; and `margin_idx` must index `S`.

Metadata: `format=sherd-cache`, `cache_version`, `algo_ref`, `core_version`, `name`,
`source_path`, `source_size`, `source_mtime_ns`, `source_sha256` (optional), `target_faces`,
`face_budget`, `area0`, `thick`, `thick_mode`, `res`, `watertight`, `n_boundary`,
`n_orig_vertices`, `n_orig_faces`, `area`, `frac_area` (R §3.4's two sums, so that neither
`stats()` nor R §3.5.2's sample count recomputes the face geometry on a cache hit), `brk_params`
(JSON: `t` and the three radii of R §3.5.4–3.5.5), `md_params` (JSON: `t`, the seed and the four
counts — the sampled half of R §3.7's `mdp_*`), `features` (JSON), `backend`. R §3.7's rule that a
valid cache with other match-array parameters has *only those arrays* recomputed is implemented
per half: `brk_params` that are not the run's rebuild the breaklines, `md_params` that are not the
run's redraw the samples, and the file is rewritten with the mesh and the labels still coming off
the disk. A rebuild of both runs the breaklines first, because the samples measure `d_brk` against
them.

**Two corrections the implementation forced, and this document follows it** (phase-1a
verification, finding F11):

* **the metadata is one key, not a flat map.** `safetensors` 0.8 takes the `__metadata__` block as
  a `HashMap<String, String>` and serialises it in iteration order, and that order is randomised
  per map instance — the same twenty-key map serialised four times inside one process gave four
  different headers. A cache written twice from the same input would then differ byte for byte,
  which is the one thing it must not do. So the whole block travels as a single JSON object under
  the key `sherd`, written by `serde_json`, which emits a struct's fields in declaration order.
  The field names inside it are the ones listed above, unchanged, and
  `safe_open(...).metadata()["sherd"]` is one `json.loads` away from the map this section
  originally described.
* **`created` is not written.** A timestamp makes two runs of the same input produce different
  files, which defeats the same requirement. The provenance that matters is
  `algo_ref` / `core_version` / `cache_version`, all three of which are there.

`face_budget` and `area0` are additions to the original list, for the reason §4.1 gives.

Validity rule = the reference's (`R§3.7`) with `cache_version` and `algo_ref` in place of
`CACHE_VERSION`; a mismatch of `md_params` alone recomputes only the match arrays. Loading is an
mmap plus header parse (< 1 ms); 170 fragments cost ≈ 0.5 GB of address space, paged on demand.

The Python package can read this file (`safetensors.numpy.load_file`) so `Fragment.load` can be
pointed at Rust caches during the transition; the reverse (Rust reading `.npz`) is not needed
(fixtures use `.npy`).

### 4.3 Versioning

`algo_ref` names the frozen algorithm; any algorithmic change bumps it and invalidates caches.
`cache_version` covers the file layout. `core_version` is informational. Reports carry all
three plus the git commit and the backend used (`"engine": {"core": "...", "algo_ref": "...", "backend": "gpu:Apple M2 Pro"}`),
added as a new top-level key in `report.json` and `transforms.json` (additive; the Python
readers ignore unknown keys).

## 5. Pipeline and threading model

One process, one `rayon` pool sized `--threads` (default: **cores − 1**, which is what `cli.py`
resolves an unset `--workers`/`--threads` to — `workers or max(1, (os.cpu_count() or 2) - 1)`, nine
on this ten-core machine; corrected in task Z from "all cores", which the port stopped doing at
step Y4 and this line went on saying). `run`, `segment` and `bench` all resolve the two flags
through the same two lines: the pool takes `--threads` or, failing that, `--workers`, and R §4.2's
block schedule takes `--workers`; both fall back to `pipeline::default_workers()`. Stages:

1. **Discover** files, names, pair order (R§2, R§4.1).
2. **Preprocess** (R§3): a `par_iter` over fragments **bounded by a memory-aware semaphore**:
   a job for a scan of `f` faces reserves `344 MiB per million faces` over a process floor of
   `98 MiB` from a budget of `--memory-budget` (default 50 % of physical RAM); jobs wait for the
   reservation. **Built in E2** (`sherd_core::memory`, `notes/2026-09-07-e2-tuning.md` §4): the
   admission rule is `running == 0 || in_flight + want <= budget`, whose first clause is what lets
   a scan larger than the whole budget run alone instead of waiting for a reservation that can
   never be released; the face count comes out of the PLY, OFF and binary-STL headers and is
   estimated from the file size for the rest. Verified on the terracotta at `--memory-budget 0.5`,
   which forces one scan at a time: the same four caches and the same fourteen files, byte for
   byte, as the unbounded run, at 835 MiB of peak RSS instead of 1 451. At the default budget on a
   16 GB machine nothing waits, and the reservation the unbounded run computes — 1 491 MiB for
   terracotta's four scans — is E1 §7.1's *measurement* of that same case to a megabyte. The same
   budget prices R§11.4's writers, which hold one original scan each (step 4).
   Large scans are read straight from the file into the vertex/face arrays (no
   intermediate copies). Cache hits skip everything. The constant used to read `60 MB + 110 B·f`
   and to say "re-measure in E1"; E1 measured it — seven scans of 53 k to 1.34 M faces, one at a
   time, give `peak RSS = 98 MiB + 361 B·f` with R² 0.978, and the fixed term is the process, not
   the scan, so the *reservation* is the slope alone (`notes/2026-09-07-e1-profile.md` §7). The
   model was checked at four concurrent scans (predicted 1 589 MiB, measured 1 492) and at nine
   (predicted 1 959 MiB on synthetic 20, measured 2 038 for the whole cold run).
3. **Match** (R§5–6): pairs are grouped into blocks of 3×3 fragments in collection order
   exactly as the Python does (`R` pipeline `_pair_blocks`), the blocks are the `par_iter` items,
   pairs inside a block run sequentially, candidates inside a pair run `par_iter` (nested
   parallelism is fine under rayon's work stealing; results are collected by index, so nothing
   depends on scheduling). Per-fragment derived data (`MatchData` at a given `t`) lives in a
   shared LRU (`moka`-free: a `Mutex<LruCache<(FragId, t_bits), Arc<MatchData>>>` of 64 entries)
   so a fragment recomputed at `t_pair` for one pair serves the next. **Built in E2**
   (`sherd_core::matching::cache`): a hit is bit-identical to the build it replaces, so it moves no
   result, and it is worth 0.9 % of CPU on synthetic 20 and nothing on wall clock or memory. **E1 measured what that is
   worth and the answer is almost nothing** (`notes/2026-09-07-e1-profile.md` §5): R §1.2 sets
   `t_pair = min(t_A, t_B)`, so of a pair's two builds exactly one is the fragment at its *own*
   `t` — free, its arrays are the cached ones — and the other is a rebuild at a thickness that
   belongs to that one partner and recurs nowhere else. On synthetic 20 the 380 builds hold 209
   distinct keys, but all 190 of the *expensive* ones are distinct: of `Pair::build`'s 30.4 core-s,
   all but at most 0.63 is R §3.5 rebuilt at the partner's `t`, and only that 0.63 — the clouds and
   the two KD-trees every build pays — can be cached at all. The LRU would drop 171 of those 190
   cheap builds: **at most 0.57 core-s of 319, 0.18 %** (pot H: 0.065 of 82.18, 0.08 %). E2 built it
   and measured it (`notes/2026-09-07-e2-tuning.md` §3): 160 hits of 380 lookups on synthetic 20 and
   45 of 110 on pot H — E1's counts exactly — for **0.9 %** of CPU, the bracket having been a lower
   bound because it counted only the stacks that kept the `Pair::build` frame through a work-steal.
   It is kept because group-level matching (§11, roadmap item 5) changes the arithmetic that makes
   every expensive key unique, not because 0.9 % justifies it.
   With the GPU executor the block loop becomes a software pipeline (§6.4).
4. **Assemble** (R§8), **refine** (R§9), **recentre**, **outputs** (R§11). Full-resolution meshes
   are read, transformed and written **in parallel, bounded by step 2's budget** (E2): the files
   are independent and their names are fixed, so the loop is a `par_iter` whose results are
   collected in fragment order, and `place` holds one original scan, which is exactly what
   preprocessing reserves for. The merged assembly PLY is written by first summing the members'
   header counts, then appending each transformed member. R§11.5's views are rendered in parallel
   too — one view is one image over its own z-buffer — while the *sampling* above them stays
   sequential, because R§10 walks one `rng(0)` in group and then collection order. Together those
   took synthetic 20's `output` stage from 5.04 s to 1.90 s
   (`notes/2026-09-07-e2-tuning.md` §8).

Cancellation: every stage checks an `AtomicBool` between units of work (a pair, a fragment); the
CLI wires Ctrl-C, the desktop app wires a button. Progress: a `Progress` trait with
`(stage, done, total)` callbacks; the CLI prints, the app emits events.

## 6. Executor abstraction and the GPU plan

### 6.1 The interface

```rust
pub trait Executor: Send + Sync + fmt::Debug {
    fn name(&self) -> &'static str;
    fn coarse_scores(&self, b: &CoarseBatch<'_>) -> Vec<f64>;   // R§5.2 and R§5.4, many poses
    fn icp_rung(&self, b: &IcpBatch<'_>) -> Vec<Registration>;  // R§7, one rung for many candidates
    fn bounded_distance(&self, b: &DistBatch<'_>) -> Vec<f64>;  // R§6.1, exact below `max_dist`, +inf above
    fn inside(&self, b: &InsideBatch<'_>) -> Vec<InsideOutcome>;// R§6.4, (inside, depth) per point
}
```

`CpuExecutor` is the reference implementation of every method (rayon inside); `GpuExecutor`
(crate `sherd-gpu`) implements the same methods with WGSL kernels and identical batch structs.
Everything else in the pipeline — hypotheses, NMS, seam/continuity via `kiddo`, scoring
arithmetic, assembly — stays on the CPU and is written once. `Backend::Auto` picks the GPU only
if an adapter exists, the self-test passes and its measured throughput beats the CPU by ≥ 1.5×
(§6.8).

**Built in phase 2a (task G1), with four things the original sketch did not say.**

* **`cone_cast` is not in the trait.** R §3.4.3's cone belongs to preprocessing, not to matching,
  and it has no CPU caller that would exercise the boundary; it joins the trait with the kernel
  that needs it (phase 2b, optional).
* **A batch carries `f64` and the CPU's own structures**, not a narrowing. E5 measured `f32` point
  loops far outside §10.2 — the *median* stage-2 pose moves 9.8 `t` on pot G (`matching::icp`) —
  so a batch that narrowed on formation would change what the CPU computes, and G1's contract is
  that it must not. §6.3's device layouts (SoA `vec4<f32>`, `bytemuck::Pod`, §6.2's hash grid) are
  produced **from** a batch by the executor that needs them, on the way to the device
  (`CoarseBatch::device_grid`, `IcpBatch::device_inits`, `Poses::device`, …). The CPU path never
  builds one: it searches with `kiddo`, and a grid it would not query costs `synthetic_20` real
  seconds to prove nothing.
* **The engine travels with the numerics.** `Engine<'e> { exec: &'e dyn Executor, numerics }` is
  what the pipeline passes down, because §7's two knobs reach the same functions and answer the
  same question from the other side. `pipeline::run_with` takes it; `pipeline::run` is that with
  `Engine::REFERENCE`. `sherd-core` cannot build a `GpuExecutor` — it has no GPU dependency — so
  the CLI resolves `--backend`, `sherd-gpu` builds the device, and the pipeline never learns which
  of the two it is holding.
* **Where a batch is a batch of one.** Stage 1 hands R §5.3's whole kept list to one `icp_rung`
  per rung and one `coarse_scores` for the re-score (§6.4 step 4), and stage 2 does the same over
  R §5.5's candidates for the two `pc_reg` rungs and then over the survivors of R §5.6's early
  rejection for the two `pc_frac` rungs. R §9's refinement and every single-pose caller form a
  batch of one, which the CPU executor answers without a rayon bridge. Each candidate's ladder
  depends on nothing but its own pose, so the grouping cannot move a result — and it does not:
  the CPU path's outputs are byte-identical to `9bf35d6`'s on all four development sets, and the
  parity table is unchanged at 23 804 / 0.

### 6.2 Spatial structures shared by both executors

**Hash grid** (radius-bounded NN over a cloud of `n ≤ 20 000` points at radius `r`): cell size
`r`; integer cell coords `⌊(p − origin)/r⌋`; keys hashed into a table of `cap = next_pow2(2n)`
slots by `(ix·73856093) ^ (iy·19349663) ^ (iz·83492791)`; open addressing; each slot stores
`(ix, iy, iz, start, count)` into a `sorted_idx` array (points sorted by cell, then by original
index). A query visits the 27 neighbour cells in fixed `(dx, dy, dz)` order and the points in
each cell in ascending index, keeping the nearest with `d ≤ r`; ties → lowest index. Build on
the CPU (counting sort, ≈ 50 µs for 6000 points), uploaded with the batch. Layout:
`GridHeader { origin: vec4<f32> (w = 1/r), cap, n, pad }`, `slots: array<vec4<i32>>`
(`ix, iy, iz, start`) + `counts: array<u32>`, `sorted_idx: array<u32>`.

**Built in phase 2a** (`sherd_core::spatial::grid::HashGrid`), in `f32` because the kernels are,
with the header carrying `1/r` rather than `r` so the kernel multiplies instead of dividing
(Metal's division is 2 ULP, E7 §4.2) and the squared distance written out rather than a `dot`
(`metal::dot` stays fused, E7 §4.2). Both radius conventions are provided, because D §6.2's rule is
inclusive (`d ≤ r`) while R §5.2's bound (scipy's `distance_upper_bound`) and R §7's (FLANN's) are
strict. **Nothing in the pipeline queries it**: the CPU searches with `kiddo` — E3 measured the
grid at 0.4–2.0× it, never the ≥ 3× §3 hoped for — and it is built from a batch by the GPU
executor. Its Rust traversal is the mirror `kernels/nn.wgsl` is cross-checked against, and it is
tested against brute force in its own `f32` arithmetic on 90 000 queries over three cloud shapes,
five radii and both conventions.

**Flattened BVH** over triangles (fracture faces only, or all faces): binned SAH build on the
CPU, leaves of ≤ 4 triangles, nodes `{ bmin: vec3, left_or_first: u32, bmax: vec3, count: u32 }`
(32 bytes), triangles stored as three `vec4<f32>` (`w` of the first carries the face index).
Queries, all with a short explicit stack (depth ≤ 48, in registers/private memory on the GPU):
`closest_point(q, r_max)` with node pruning by AABB distance and early exit when the best is
below a caller-supplied `r_exact` (see PMC-12), `first_hit(ray)` (returns `t`, face), and
`parity(q, dir)` counting all crossings (inside test, PMC-7: three axis rays, majority; the
reference uses one). Both executors traverse the same node array, so results differ only by
floating-point order.

### 6.3 Buffers and memory layout (GPU)

All arrays are SoA `vec4<f32>` (xyz + spare) or `u32`, 16-byte aligned, `bytemuck::Pod`.

- **Fragment slots** (resident on the device, LRU of 32 slots ≈ 400 MB): per fragment `S`,
  `Pf`, `Nf`, `brk_P`, `brk_ns`, `brk_sub`, `Pm`, `Nm`, fracture BVH, full BVH. Uploaded once
  per epoch; pairs reference slots by index. The scheduler orders pair blocks to maximise slot
  reuse (same block structure as the CPU path).
- **Per-pair descriptors** (`PairDesc`): slot indices, `t_pair`, `res_pair`, the `Scales`, and
  offsets of the pair's grids (built per pair per rung on the CPU, ≈ 1 MB per pair).
- **Per-candidate state** (`CandState { pair: u32, t: mat3x4<f32>, fitness, rmse: f32, done: u32, iter: u32 }`).
- **Batch buffers**: candidates of one stage across up to `P` pairs; `P` chosen so that grids +
  candidates ≤ 128 MB (the default `max_storage_buffer_binding_size`); typically `P = 64–256`.
- Uniform/unified memory (Apple): buffers are mapped-at-creation, no staging copies; discrete
  GPUs: staging via `queue.write_buffer`, readback through a mapped `MAP_READ` buffer once per
  batch. Readback volume is tiny (candidate states, scores).

**Built in phase 2a:** `sherd_gpu::slots::SlotTable` is the resident set (32 slots, 400 MB,
least-recently-used with ties to the lower slot index, a fragment larger than the budget refused
rather than admitted after emptying the table) and holds no wgpu type, so its eviction rule is
tested on every platform without an adapter; `sherd_gpu::buffers::{Chunking, Dispatch}` are the
binding-cap split and the 2-D dispatch fold; `upload`/`read_back` are the two transfer paths, of
which **only the mapped-at-creation one has ever run** — this machine has one integrated Metal
adapter and no software fallback (E7 §6). `Chunking` has no consumer until a kernel dispatches a
batch larger than one binding; the batch structs report `device_bytes()` so a scheduler can size
itself against it.

### 6.4 Batch formation and the software pipeline

For a block of pairs (CPU threads prepare, one GPU thread submits; double-buffered):

1. CPU: hypotheses for every pair (R§5.1), `idx` draw (R§5.2). Output: `R, τ` arrays per pair.
2. GPU `coarse_scores`: one invocation per (hypothesis, query point) pair → per-hypothesis
   `agree` counts via a fixed-order workgroup reduction (60 points = 64-lane workgroup, one
   hypothesis per workgroup). Chunked so that one dispatch ≤ 20 M point-queries (≈ 10 ms).
3. CPU: coarse NMS per pair (R§5.3), builds `T0` for the ≤ 250 kept poses.
4. GPU `icp_rung` ×2 (point-to-point, breakline clouds), CPU `brk_score` (or GPU
   `coarse_scores` with the stage-1 radius — same kernel), CPU stage-1 NMS.
5. GPU `icp_rung` ×4 (point-to-plane; `pc_reg`, `pc_reg`, `pc_frac`, `pc_frac`).
6. GPU `bounded_distance` ×2 and `inside` ×2 per candidate; CPU seam, continuity, `accept`.

Each `icp_rung` dispatch runs *all iterations* of that rung for a batch of candidates inside the
kernel: one workgroup (256 invocations) per candidate; per iteration each invocation handles
source points `tid, tid+256, …` (transform, grid query, accumulate its 27-float partial of
`JTJ`/`JTr` plus count and squared error), then a shared-memory tree reduction in fixed order,
then invocation 0 solves the 6×6 system (LDLT in f32 on the centred system, §7) or the 3×3
Umeyama (Jacobi SVD, ≤ 12 sweeps), updates `T`, evaluates the convergence test against the
previous `(fitness, rmse)` and sets `done`; barrier; next iteration skipped for done
candidates. This keeps the whole rung to one dispatch and one readback. To bound dispatch time
under Windows TDR (2 s) the batch is split into dispatches of ≤ 512 candidates; measured
per-candidate cost at 12 000 points × 30 iterations is expected at 1–3 ms of GPU time, so a
512-candidate dispatch is ≈ 100 ms at full occupancy.

### 6.5 Kernel semantics (WGSL sketches; the CPU code mirrors them line by line)

```wgsl
// coarse score and stage-1 re-score: one workgroup per pose, 64 lanes striding over the n_q query points
// (n_q = 60 for the coarse score, |brk_sub| ≤ ~800 for the re-score; same kernel, different radius and point set)
fn coarse_main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let h = wg.x + batch.first_pose;  let pair = pose_pair[h];
    var agree: u32 = 0u;
    for (var k = lane; k < n_q[pair]; k += 64u) {
        let p = rot(pose_R[h], q_p[pair][k]) + pose_t[h];  let n = rot(pose_R[h], q_n[pair][k]);
        let j = grid_nearest(pair_grid[pair], p, radius[pair]);           // u32::MAX on miss
        if (j != MAX && dot(brk_ns[pair][j], n) > 0.7) { agree += 1u; }
    }
    // fixed-order reduction of `agree` over 64 lanes in shared memory (integer, exact)
    if (lane == 0u) { score[h] = f32(sum) / f32(n_q[pair]); }
}
```

```wgsl
// icp rung: one workgroup per candidate; loops max_iter times; `plane` selects the estimator
for (var it = 0u; it < max_iter && !done; it++) {
    var acc: array<f32, 29>;                          // 21 (JTJ upper) + 6 (JTr) + count + err2
    for (var i = lane; i < n_src; i += 256u) {
        let p = rot(T, src_p[i]) + T.t;               // T kept in shared memory, f32
        let j = grid_nearest(grid, p, d_max);
        if (j != MAX) { let q = tgt_p[j]; let n = tgt_n[j]; let pc = p - centroid;   // centred coordinates (§7)
                        let a = cross(pc, n); let r = dot(p - q, n); accumulate(acc, a, n, r); }
    }
    reduce_fixed_order(acc);                          // shared memory, 256 → 1, same tree on the CPU
    if (lane == 0u) { let U = solve_and_compose(acc); T = U * T; update fitness/rmse; done = converged(); }
    workgroupBarrier();
}
```

`bounded_distance`: one invocation per query point, BVH closest-point with `r_max = facing` and
`r_exact = facing` (exact inside the window, `+inf` outside). `inside`: one invocation per
point: AABB reject → parity rays → for inside points, closest-point distance with
`r_max = pen_cutoff` for `pen_depth` (only needed for the maximum; computed exactly).

### 6.6 Expected speedups and their basis

**The GPU column of the table below rests on an assumption experiments E7 and G1 both measured as
5–10× optimistic, and it has not been re-derived.** The basis it names — "≈ 0.5–1 G bounded NN
queries/s on the hash grid" — is, on this Metal GPU and on this design's own grid at this design's
own sizes, **0.105 G queries/s** (9.0–9.5 ns/query at 46 candidate points per query, equivalently
4.9 G candidate distance tests/s); G1 re-measured 12.2–17.9 ns/query on a smaller batch and
**3.3–6.3× the whole ten-core CPU**, against the 15–50× the table implies. The GPU still clears
`Backend::Auto`'s 1.5× bar with room. Re-deriving the column needs `icp_rung`'s own cost, since
the rung carries the 6×6 solve as well as the correspondence search, and that is phase 2b's
measurement; until then read the GPU column as an upper bound that is known to be wrong by an
order of magnitude, and the CPU column — which phase 1e measured — as the real one.

Per mid-size pair (R§13 cost structure), single-thread Python core-seconds → estimated Rust CPU
core-seconds → estimated GPU seconds (M2 Pro 16-core GPU, ≈ 0.5–1 G bounded NN queries/s on the
hash grid — **not what E7 measured, see above** — ≈ 0.1–0.2 G BVH closest-point queries/s):

| stage | Python | Rust CPU | basis (CPU) | GPU | basis (GPU) |
|---|---|---|---|---|---|
| hypotheses | 0.02 | 0.003 | 60k × 12 floats | — (CPU) | |
| coarse score, 3.5 M bounded queries | 0.99 | 0.25 | hash grid ≈ 70 ns/query vs cKDTree + numpy ≈ 280 ns | 0.006 | 3.5 M / 0.6 G/s |
| NMS (both) | 0.01 | 0.005 | — | — (CPU) | |
| stage 1, 3 M queries + 10k Umeyama | 0.55 | 0.12 | as above; SVD negligible | 0.010 | one dispatch per rung |
| stage 2 coarse ICPs, 3.6 M queries | 2.84 | 0.50 | Open3D ≈ 0.8 µs/query (FLANN, double, per-call tree build) → ≈ 0.12 µs | 0.010 | |
| stage 2 fine ICPs, 7.2 M queries | 1.82 | 0.80 | larger clouds, more iterations until convergence | 0.020 | |
| verify: fracture distance, 240k BVH | 0.05 | 0.03 | bounded early exit | 0.003 | |
| verify: seam + continuity | 0.10 | 0.03 | kiddo | — (CPU) | |
| verify: penetration, 400k signed | 0.30 | 0.12 | AABB reject + parity only inside bbox | 0.005 | |
| `MatchData` setup | 0.20 | 0.02 | mmap cache; grids per pair | 0.02 (uploads) | |
| **total** | **6.9** | **≈ 1.9 (3.6×)** | | **≈ 0.08 GPU + 0.05 CPU** | |

Collection level, 14 000 pairs with the `mixed_all` size distribution (Python model 11.7 core-s
average per pair, 41 CPU-h): Rust CPU ≈ 3.2 core-s average → 12.5 CPU-h → **1.25 h on 10 cores**
(target ≤ 2 h; margin 1.6×, so the phase-1 benchmark gate is the real test); GPU ≈ 0.15 s
average per pair → **≈ 35 min of GPU time if serialised**, but the batches keep the GPU busy
while the CPU prepares the next block, and the fine ICP work is 60 % of the GPU time only on
true pairs (~3 % of pairs); realistic estimate **8–15 min matching + 1–2 min preprocessing**,
target ≤ 30 min. Preprocessing: meshopt 1 M → 200 k faces ≈ 0.6 s, Taubin ≈ 0.1 s, BVH build ≈
0.1 s, 1.4 M cone rays ≈ 0.4 s (rayon), grid balls ≈ 0.3 s, samples/breakline ≈ 0.2 s → ≈ 2 s
per typical fragment, 170 fragments ≈ 40 s on 10 cores; a 10 M-face scan ≈ 10 s.

Where the GPU does not pay: Intel iGPUs (≈ 1 TFLOPS, shared bandwidth) give an estimated 2–4×
over the CPU path; `Backend::Auto` measures rather than assumes (§6.8).

### 6.7 CPU fallback with results within tolerance

The CPU executor and the WGSL kernels share: data layouts, the grid/BVH traversal order, the
per-invocation striding (`i = lane + 256·k`), and the reduction tree (256 → 128 → … → 1).

**E7 corrected the sentence that used to stand here** ("no fast-math and no FMA contraction …
agree to a few ULPs"). Metal compiles every shader with fast math on and wgpu does not turn it
off, and the team decision is to accept that rather than patch or vendor `wgpu-hal`. What is
actually true, measured twice:

* an **addition-only reduction in a fixed order is bit-identical** — the self-test asserts the 32
  bits, not a tolerance, and reproduced `0x49a7230c` on both sides in phase 2a;
* anything with a multiply-add is not. On the hash-grid kernel that is `max |Δd| = 2.4e-7` of a
  unit cloud (1.3e-7 in G1's smaller sweep), 400× inside the tightest distance tolerance of
  §10.2, with **~2.3e-6 of queries choosing a different neighbour** because two candidates are
  within a ULP of each other and the transformed point already differs. §7's "ties → the lowest
  index" cannot fix that, since the tie is not exact on one side. A cross-check therefore gates on
  the *rate* and on every disagreement being a genuine near-tie, never on a count of zero.

Three rules follow for every parity-critical WGSL file, and they are enforced by reading rather
than by a lint: no `dot`/`length`/`distance`/`normalize` (`metal::dot` stays a fused chain even
with contraction off — write the sum out, and mirror it on the CPU); no `subgroupAdd` and no
floating-point atomics (neither has a defined order); no reliance on denormals (Apple GPUs flush
them in hardware and no compiler option changes that). The shape of a reduction — workgroup count,
lane stride, loop bound, tree — is *data in the batch descriptor*, never a constant on one side
and a divide on the other. The CI cross-check (§10.4) enforces the tolerances. The CPU path additionally offers `--precision f64` for the ICP (generic `Real`
type) as a diagnostic, not a production mode.

### 6.8 Operational GPU concerns

**Built in phase 2a (task G1); the numbers below are measured on this machine, not planned.**
`notes/2026-09-08-g1-gpu-foundation.md` carries the run.

- **Adapter selection:** `wgpu::Instance` over Metal | Vulkan | DX12 (no GL: the four backend
  features are named in `Cargo.toml`); `--gpu-adapter NAME|INDEX`, where an all-digit argument is
  an index and anything else a case-insensitive substring of the adapter's name or backend;
  `sherd-refit-rs info` lists adapters and `gpu-check` runs the self-test. On Metal `vendor`,
  `device`, `driver` and `driver_info` are all empty (E7 §7.6), so a name is all `--gpu-adapter`
  has to match on and a run report cannot record a driver version.
- **Self-test at start (`sherd_gpu::SelfTest`), four checks:**
  1. *limits* — the wgpu **defaults** of `Requirements`, not this adapter's own, because that is
     what a portable build gets;
  2. *reduction* — D §6.4's schedule over 1e7 `f32` terms (256 workgroups × 256 lanes, strided
     accumulate, then a 256 → 1 shared-memory tree, then one pass over the partials), asserted
     **bit-identical** to a single-threaded Rust mirror of the same WGSL. Measured
     `0x49a7230c` on both sides;
  3. *bounded nn* — D §6.2's hash-grid kernel on E7 §5's synthetic sheet, 64 poses × 6000 points.
     The criterion is **not** "no neighbour differs": E7 §5.1 measured `2.3e-6` of queries picking a
     different point in stock configuration and called it inherent, because the pose transform is
     contracted into FMAs and the transformed point already differs. The criterion is that every
     disagreement is a genuine near-tie (`|Δd|` within the distance tolerance), that neither side
     found a neighbour the other missed, and that the *rate* stays under `1e-5`. Measured: 1 of
     384 000 (2.6e-6), a tie to 1.0e-8, 0 hit/miss disagreements, `max |Δd|` 1.3e-7 against a
     limit of 1e-6;
  4. *throughput* — **host wall time of the dispatch alone**, `submit` + `poll(Wait)`, with the
     pipeline built and the buffers uploaded and after a warm-up dispatch. Never timestamp
     queries: this driver advertises `TIMESTAMP_QUERY`, resolves it without error and returns
     nonsense (E7 §7.1). The separation is not pedantry — timing the first dispatch includes
     Metal's shader compile and reported 197 ns/query for a kernel that runs at 14. Measured
     **12.2–17.9 ns/query, 3.3–6.3× the whole ten-core CPU on the same batch**, which brackets
     E7 §5's 12.2 ns at this size and its "4–5× over all ten cores".
- **`Backend::Auto`** takes the GPU only when the self-test passes, the adapter is not a software
  implementation of the API, the measured ratio is **≥ 1.5×**, *and* the executor has kernels.
  The last clause is `GpuExecutor::HAS_KERNELS`, `false` until phase 2b: in phase 2a the device
  opens, the self-test passes at 4–6× and `Auto` still runs on the CPU, saying so in one line.
- **`--backend gpu`** opens the device and runs the self-test, and **fails** — with the adapter
  list, the unmet limits or the failed checks — when either step fails, rather than falling back
  silently. A macOS self-test failure means the CPU with no second opinion: there is no software
  adapter on this platform at all (E7 §6). When it succeeds in phase 2a it says, once, that every
  `Executor` method is routed to the CPU implementation and the results are the CPU's;
  `report.json` records the backend the run *asked* for (D §4.3), which is the only difference
  between a `--backend gpu` tree and a `--backend cpu` one.
- **Limits:** the device requests `adapter.limits()` verbatim (accepted as such, E7 §2) and the
  kernels are written to the wgpu defaults: workgroup 256, ≤ 16 KB of workgroup storage, ≤ 8
  storage buffers per stage, 128 MB per binding. `Chunking` splits a batch to the binding cap and
  `Dispatch` folds a grid into two dimensions past `max_compute_workgroups_per_dimension = 65535`,
  with `wg_x` passed to every kernel as a uniform so that the CPU mirror agrees on the shape
  (E7 §2, §3). This adapter offers 1024 lanes, 32 KB, 29 buffers and a 4 GiB binding; none of that
  is used.
- **Timeouts:** dispatches ≤ 100 ms by construction (§6.4); `device.poll(PollType::wait_indefinitely())`
  with a watchdog; a device loss mid-run falls back to the CPU for the remaining blocks and is
  recorded in the report.
- **Memory:** slots + batches ≤ 1 GB. `SlotTable` is D §6.3's resident set — 32 slots, 400 MB,
  least-recently-used eviction with ties to the lower slot index, a fragment larger than the whole
  budget refused rather than admitted after emptying the table. It holds no wgpu type, so its
  eviction rule is tested on every CI platform without an adapter. On adapters reporting < 2 GB
  the slot count halves and `P` shrinks (`SlotTable::with_capacity`).
- **Precision:** f32 only (`f16`/`f64` unused; `SHADER_F64` is not available on Metal anyway);
  `Scales` and thresholds computed in f64 on the CPU and passed as f32.
- **Untestable here, and said so rather than assumed:** the staging upload path for discrete GPUs
  (this machine has one integrated Metal adapter and no software fallback, E7 §6), and every row
  of E8's vendor matrix but the Metal one.

## 7. Numerical determinism

| source of nondeterminism | policy |
|---|---|
| sampling | `ChaCha8Rng` seeded from `p.seed` (and hard-coded 0 where the reference does), draws in the reference's order (R§10), **one stream per draw site** rather than one per fragment: the seed carries a fixed 64-bit tag per `Draw` (`Thickness` tagged zero, so R §3.2's stream is unchanged), because three samplers sharing one seed would draw the same uniforms and put a surface sample and a fracture sample on the same point of the same face (PMC-9, `fragment::samples`) |
| unordered containers | none on any result path; voxel representatives sorted ascending (PMC-4) |
| sorting | `sort_by` with explicit keys and ascending-index tie-break; stable sorts only |
| parallel reductions | fixed-order trees on both executors; no floating-point atomics; rayon results collected by index |
| NN ties | lowest index (fixed traversal order) |
| ICP convergence | per-candidate `done` flag exactly as the sequential loop (R§7); converged candidates are never iterated further |
| thread count | results must be identical for `--threads 1` and `--threads N` (CI test) |
| CPU vs GPU | within §10.2; not bit-identical (different ULP behaviour); the report records the backend |
| platforms | same backend, same binary → identical; across OS/compilers → f32 ULP-level differences are possible in `libm` calls (`acos`, `sin`); tolerance-based |
| ill-conditioning | **measured in task C2 and not adopted.** Assembling the point-to-plane system about the target centroid and re-expressing the update about the origin is an exact re-parameterisation of the *linear* Gauss–Newton step but not of the finite update, which differs by `(R − I − ω̂)c = O(|ω|²·|c|)` per iteration. That was estimated at 1e-6 t; on terracotta it is 0.044 t at the median of stage 2 and 0.61 t at p90 (`|c| ≈ 100–150` units, `|ω| ≈ 0.1` rad, thirty unconverged iterations), so the poses of §10.2 are computed in world coordinates as R §7 writes them. Shrinking coordinates for an `f32` path means translating **both clouds** by `−c` — a rigid change of frame R §7 is equivariant under — not re-parameterising the Jacobian alone. `icp::Assembly` keeps both forms because it is what measured this |

## 8. Memory budget (170 scans)

| item | per unit | 170 fragments | notes |
|---|---|---|---|
| original scan in memory during preprocessing | **measured** (E1 §7): 344 MiB per million faces — 3 M faces ≈ 1.0 GiB, 10 M faces ≈ 3.4 GiB, over a 98 MiB process floor | bounded by the semaphore: `⌊(budget − 98 MiB) / cost⌋` concurrent | 16 GB laptop, budget 8 GiB: **23** concurrent 1 M-face scans, **11** at 2 M, **2** at 10 M. To hold the ≤ 6 GB row below with 170 fragments resident the budget has to be ≈ 4.5 GiB, and then it is 13 / **6** / 1 |
| cache file (mmap) | ≈ 6 MB (V f32 1.2, F 2.4, labels 0.2, arrays 2) | ≈ 1 GB address space, paged | resident only when touched |
| derived per fragment (FN, A, C, grids, fracture BVH) | ≈ 6 MB | ≈ 1 GB if all resident; LRU of 64 `MatchData` — **measured in E2 at no detectable cost**: the entries at a fragment's own `t` borrow its cached arrays rather than copying them, and synthetic 20's peak RSS did not move outside its own ±13 % spread when the cache was added | the 400 MB this row used to quote was an estimate for 10 M-face scans and has never been reached on a development set |
| full BVH (penetration) | ≈ 4 MB | in the LRU | |
| matching transient per pair | hypotheses 150k × 48 B ≈ 7 MB + grids 1 MB | ≈ 10 threads × 10 MB | |
| assembly | poses, candidates | negligible | |
| refinement clouds | ≤ 150k × 24 B = 3.6 MB per placed fragment | ≤ 0.6 GB (all placed) | freed per group |
| outputs | one original mesh **per writer in flight**, bounded by §5 step 2's budget | ≤ 1.2 GB peak (10 M faces) per writer | streaming PLY writer; E2 made the placed meshes parallel under that budget |
| **peak RSS** | | **≈ 3–6 GB** (≤ 3 M faces), **≤ 10 GB** (10 M-face scans, 3 preprocessing workers) | measured on the largest development set, warm, previews and meshes on: **1.9 GiB** after E2 against 1.6 GiB before it, the rise being E2's parallel writers and its `MatchData` cache (`notes/2026-09-07-e2-tuning.md` §9) |
| GPU | slots 32 × 12 MB + batch ≤ 256 MB | ≤ 1 GB | halves on small adapters |

## 9. CLI parity and outputs

`sherd-refit run INPUT --out OUT [flags]` and `sherd-refit segment` accept **every** Python flag
with the same name, default and meaning (R§1.4), plus:

| flag | meaning |
|---|---|
| `--backend auto|cpu|gpu` (default `auto`) | executor selection (§6.8) |
| `--gpu-adapter NAME|INDEX` | override adapter |
| `--memory-budget GB` | preprocessing budget (§5) |
| `--dump-fixtures DIR` | write the Rust-side fixture (§10.1) |
| `--inject-from DIR --inject-stages a,b,…` | parity mode: take the listed stage inputs from a Python fixture |
| `--constraints FILE` | roadmap item 3 (§11) |
| `--review-images` | roadmap item 3: render `review/<a>__<b>.png` for probable joins |
| `--export-glb` | additionally write `assembly_<k>.glb` for the desktop viewer |
| subcommands `parity`, `bench`, `info` | harness, timing gates, adapters |

Outputs are the reference's files (R§11) with identical names, JSON schemas (plus the additive
`engine` key) and PLY layout. `report.md` follows the same sections and number formats; the
software renderer is a line-by-line port of R§11.5 (PNG via `image`, label via an embedded
bitmap font; the exact glyphs differ from PIL's, which is acceptable since previews are not
compared). Log lines are free-form.

## 10. Verification

### 10.1 Fixture dumps (Python side)

`tools/dump_fixtures.py` is not a separate runner: it installs a sink into the reference package
(`sherd_refit/fixture.py`, enabled by `SHERD_REFIT_FIXTURES=DIR`, inherited by worker processes)
and the package calls `fixture.put(scope, stage, key, array)` at every stage boundary. Layout:

```
DIR/manifest.json                         {commit, open3d, numpy, params, target_faces, collection order, pairs, files: {path: {shape, dtype, sha256}}}
DIR/fragments/<name>/
   load.V0 load.F0 load.n_orig                              R§3.1
   thick.idx thick.t_hit thick.prim thick.t thick.thick_mode thick.target   R§3.2–3.3 (idx is the stride set, ≤ 300 000)
   mesh.V mesh.F mesh.res mesh.watertight                   R§3.3 (working mesh after Taubin)
   seg.rep seg.near seg.NS seg.good seg.frac_raw seg.frac_majority seg.frac_islands seg.ref seg.has_ref seg.frac_final   R§3.4
   md.<each MD_ARRAY> md.params md.brk_t md.brk_dih md.valid                 R§3.5–3.6
   md.S_u md.S_v md.Pf_u md.Pf_v      the uniforms behind each sample, before R§3.5.1's fold
DIR/pairs/<a>__<b>/
   scales.json md_used.json  hyp.ia hyp.ib hyp.pa hyp.pb  coarse.idx coarse.cs  nms1.order nms1.kept
   s1.T[250,4,4] s1.score  nms2.order nms2.kept
   s2.T_reg1 s2.T_reg2 s2.T_frac1 s2.T_frac2 (per candidate)  s2.scores.json  s2.accepted
   result.candidates.json (the 5 returned)
DIR/assembly/  md_t_median samples (S per fragment, 15000), poses.json, groups.json, used.json, rejected.json
DIR/refine/    <name>.idx (fracture cloud indices), per-join T after each rung, poses_final.json
DIR/outputs/   transforms.json report.json
DIR/outputs/   (added by tools/dump_outputs.py, step D2) placed.sha256.json, preview_index.json,
               preview_<k>.png, preview_<k>.nolabel.png, preview_<k>.meta.json,
               preview_<k>.<name>.{pick,u,v}.npy, and the same for preview_segmentation
```

Sizes: terracotta ≈ 240 MB, pot A ≈ 250 MB, synthetic 20 ≈ 850 MB at level `slim`; the p0 note's
table is the measured one and this line is the order of magnitude. For `mixed_all` and
`synthetic_170` only `mesh`, `seg.frac_final`, `md.*`, `result.candidates.json` and the assembly
are dumped (≈ 0.6 GB). Fixtures are stored outside git (§10.5) and regenerated whenever the
reference changes; the committed `fixtures/slab/dump` comes from
**`9cbcbbc`** — step C1's two NMS walk orders, carrying R §3.2's deterministic ray set (task T1) and
the sample uniforms of §10.2's D6 columns with it — and the seven `output/fixtures` sets from
**`895a948`**, the last commit of that step; each `manifest.json` records which, and all eight say
`dirty: false`.

**R §11.4's meshes and R §11.5's previews come from a second tool, because the dump does not run
them** (step D2). `dump_fixtures.py` calls the pipeline with `preview=False` and
`write_meshes=False`: neither output is on the algorithm's critical path and both are large.
`tools/dump_outputs.py DUMP INPUT` fills the gap without re-running anything — it rebuilds the
fragments from the dump's own `mesh.V`, `mesh.F` and `seg.frac_final`, takes the poses and the
groups from the dump's own `outputs/transforms.json`, and calls the reference's own
`report.write_placed_meshes` and the body of `pipeline.write_previews`, transcribed only so that
each sample's `pick`, `u` and `v` can be written out on the way past. The generator is consumed in
the pipeline's order, so the samples are the ones the pipeline would have drawn. The placed meshes
are hundreds of megabytes and are hashed and dropped rather than kept (`--keep-ply DIR` keeps
them); the previews are written twice, once as the pipeline writes them and once with the caption
left off, because the caption is PMC-20. §10.2's `outputs` row reads all of it.

**`nms1.order` and `nms2.order` are inputs, not outputs, and step C1 added them for that reason.**
R §5.3's suppression is a greedy walk over `np.argsort(score)[::-1]`, and numpy's `argsort` is an
unstable quicksort over scores that are multiples of `1/60`: thousands of hypotheses tie at every
level and the permutation among them is an artefact of numpy's partitioning, which PMC-6 already
says the port will not reproduce. Without the order in the dump, an injected NMS comparison would
be measuring numpy's sort. With it the port's greedy loop and duplicate test run on the
reference's own ranking and the kept list is compared exactly. The change is to the *sink* only —
`_match_pair` hoists the expression it already evaluated into a variable — so the reference's
results are unchanged, and re-dumping the slab at the same `sherd_refit/` reproduced all 178
previous files byte for byte. The Rust CLI writes the same layout with
`--dump-fixtures`.

### 10.2 Stage comparison and tolerances (`tools/compare_fixtures.py REF NEW`)

Two modes per stage: **injected** (the Rust stage ran on the Python stage's inputs) and
**native** (the Rust stage ran on Rust's own upstream results).

**What "agreement" means, as the harness measures it and not as an earlier draft described it**
(defect D8 of the phase-1b verification). Segmentation agreement is an **area-weighted quadrature
over the reference's own faces**: one point per reference face, its centroid, weighted by that
face's area, labelled on the reference's side by that face's own label and on the port's side by
the nearest face of the port's mesh (`RayScene::closest_face`). It is not the 200 000-point
Monte-Carlo sample this section used to describe: one point per face is exact on the reference's
side, has no generator in it at all, and does not need a tolerance for its own sampling noise.
What it gives up is fidelity on a mesh with a wide face-area spread, where a large face is
represented by its centroid alone; that is the price, and it is stated here rather than left in a
module comment. When the two working meshes are the same tessellation the harness compares the
label arrays directly instead and the question does not arise.

Two more places where the table is narrower than it sounds:

* **`exact` in the injected column means the port reproduced the reference's own array**, not that
  the two implementations drew the same random numbers — PMC-9 forbids the latter. Everything in
  the samples row that is marked exact is a count, an index or a membership; the point arrays
  themselves are compared only through the reference's own uniforms (the `md.*_u` / `md.*_v`
  columns, defect D6) and through statistics.
* **The injected column of a stage is only met on the fragments whose dump carries that stage's
  inputs.** At level `slim` there is no `load.V0`, so injected thickness *skips* those fragments
  rather than silently running natively (finding F3); today that is the twenty fragments of
  synthetic_20, and the injected thickness claims below rest on the other 46 plus the slab's two
  (defect D7). Native thickness covers all 68, and since T1 it is nearly as strong a check: both
  sides cast from the same faces of the same mesh, and only the ray caster differs.

| stage | quantity | injected tolerance | native tolerance |
|---|---|---|---|
| load | counts after cleaning, largest component; every vertex of the largest component | exact; 1 `f32` ULP | exact; 1 `f32` ULP |
| thickness | `t`, `thick_mode` | same bin, or ±1 bin on a count tie | ±2 %, with a floor of 1 bin of the reference's own histogram |
| working mesh | faces, `res`, area, `watertight` | (mesh is injected) | faces ±5 %, `res` ±10 %, area ±0.5 %, same `watertight` |
| segmentation | area-weighted label agreement; fracture fraction | ≥ 0.995; ±0.005 | ≥ 0.97; ±0.02 |
| breakline | count; point-set Hausdorff; `dih` per matched point | exact; 1e-4 t; 0.1° | curve length (the sum of the nearest-neighbour distances) ±10 %; **2.3 t on 99 %**; distribution **KS < 0.086** — the last two calibrated to PMC-2's own sensitivity, derived below |
| samples | `n_surface`, `n_frac`; sample-to-face residual; `fp` on fracture faces; margin count and membership; **`margin bound`** — the bounded `d_brk` the pipeline computes against the unbounded array it stands for; sample normals; **the reference's own points rebuilt from its own uniforms**; **the face pick over the reference's own cdf** | exact; 1e-9 t; exact; exact; **exact (bit for bit below `1.5 t`, `∞` at or above it)**; 0.1° over faces conditioned to 1000 f32 ulps, with ≤ 0.1 % of samples left out; exact (bit for bit); exact | `n_frac` ±10 %; fracture-sample fraction ±0.02; margin fraction ±0.05; **`margin bound` exact, on the port's own arrays**; cross-set nearest-distance p95 of `S` and of `Pf` within a factor of two of the Poisson expectation `0.977·√(A/n)` |
| hypotheses | `(pa, pb)` set **and order**; the pose of each; the twelve fields of R §1.2 | exact; 1e-4° / 1e-5 t; exact | pair matched at all; count ±30 % |
| coarse | `cs` per hypothesis, on the reference's own `coarse.idx` | ≤ 1/60 + 1e-6, **and bit-exact** (`cs exact`); probe count and pool exact | — |
| nms | kept hypotheses, on the reference's own walk order `nms1.order` | identical, in order | — |
| nms, PMC-6 tie effect (a measurement, not a parity requirement) | with the port's **own** tie-break: kept count; share of the reference's kept set missed; score at equal rank; share of its kept poses not covered by a kept pose of the port's | ±5 %; ≤ 0.5; ≤ 3/60; ≤ 0.5 | — |
| stage 1 | pose per kept hypothesis (by id); `s1`; the ICP's own fitness and rmse at both poses; `kept2` on the reference's own walk order `nms2.order` | 0.05° / 0.01 t; ±0.02; 1e-4 and 1e-4 t; identical, in order | — |
| stage 2 | pose per candidate (by stage-1 id) **on each of the four rungs**; fitness and rmse at both poses | 0.05° / 0.01 t; 1e-4 and 1e-4 t | — |
| verify (R §6, at the reference's own `s2.T_frac2`) | `tight`; `gap`; `seam`; `cont`; `cont_n`; `pen` **on a pair of closed meshes**; `pen` on a pair with an open one; `pen limit` and `accepted` | ±0.01; ±0.002 t; ±0.34 t; ±0.005 t; ±0.01; ±0.0005; ≤ `max_pen`; identical | — (the port scores its own candidates natively, which is the row below) |
| candidates (R §5.7) | how many candidates come back; which of the reference's own stage-2 candidates they are, in order; `accepted` of each; `brk_best` | exact; exact; exact; ±0.02 | see `pair result` |
| stage 1, stage 2 — distribution (a measurement beside the worst case) | `p50`, `p90`, `p99` and `max` of the same pose deviations over every candidate of the dump | the row's own tolerance | — |
| stage 1, stage 2 — `chaotic` (an alarm, not a parity requirement) | share of candidates whose own ladder moves further than the row's tolerance when the initial pose moves by one ULP; and, exactly, how many of those the reference itself kept | ≤ 0.002 / ≤ 0.06 per dump and ≤ 0.06 / ≤ 0.4 per pair; zero kept | — |
| pair result | (the `candidates` row above) | — | **a regression alarm, not a parity claim** (see below): share of pairs returning a different candidate count ≤ 0.4; share accepted by one side only ≤ 0.25 each; share of both-accepted pairs placed more than a wall apart ≤ 0.25; on the rest, median rotation ≤ 1° and median displacement of the moving fragment ≤ 0.3 t |
| assembly (R §8) | groups; joins used; rejections **with the reference's own reason string**; poses, before and after R §8.2; R §8.2's recentring against `transforms.json` | identical; identical; identical; 1e-9 t; 1e-9 t | **PMC-8 alone** — the reference's candidates on the port's own samples: identical, identical, identical, 1e-9 t. Then, from the port's own candidates, **a regression alarm**: **at most 12 joins** used by one side and not the other; largest group within **8** fragments; every used join inside a group |
| refine (R §9) | the vertex selection `refine/<name>.idx`; the walk's order; the two correspondence radii; the pose after **each** rung; `fitness` and `inlier_rmse`; relative poses within a group | exact; exact; exact; 0.2° / 0.02 t; 1e-4 and 0.02 t; 0.2° / 0.02 t | the same, with one row split: above R §9's 150 000-vertex cap PMC-9 gives the two sides different draws, so the selection is compared as "every index the reference kept is one the port's predicate accepted" (exact) plus an overlap within 0.05 of `150000/|candidates|`, and `fitness` — a Bernoulli fraction *of the cloud* — is gated at 0.008, about 5.6 σ of that sampling noise |
| outputs (R §11) | `transforms.json` — `thickness`, the groups, every `group` and `placed` flag, `params`, the poses, **and the file's own key order, against the reference's own `_run/transforms.json`**; `report.json` — the whole file through the port's own type, and the candidate list through the port's own serialiser; **`report.md` line for line**; `placed/<name>.ply` and `assembly_<k>.ply`; `preview_*.png` | exact; exact; **1e-9 t**; **identical, name for name and place for place**; **exact, down to `## Timing`**; **SHA-256 identical**, with size, both counts and the colour flag compared on every set; **pixel for pixel** | the port's own principal axis against the reference's (PMC-10, 1°); two renders of one input identical; a `transforms.json` written and read back |

The tool exits non-zero on any violation and prints a per-stage table.

**Three of those numbers were changed by the steps that measured them, and this is where they are
written down** (task Y, defect V4-D6 of the phase-1d verification; the rule is step X's — a
tolerance the harness enforces is a tolerance this section states, with its derivation).

* **`assembly`, native, the used-join alarm is an absolute count and not a share.** The row asked
  for "share of the used-join union belonging to one side ≤ 0.5 each". A share is degenerate where
  one side used *nothing*: on pot_G the reference accepts no join at seed 0, so any join of the
  port's own is 100 % of the union and the alarm fires on a collection where the reference itself
  has nothing to say. The harness enforces **≤ 12 joins** used by one side alone, twice the
  measured worst over the eight dumps (6 joins the reference uses and the port does not, on
  synthetic_20; 2 the other way on pot_G, pot_H and synthetic_20) — the same "twice the measured
  worst" shape the PMC-6 tie rows above already use.
* **`assembly`, native, the largest group is within 8 fragments and not 4.** Measured worst: 4, on
  synthetic_20 (the port's largest group holds 15 where the reference's holds 19); the gate is
  twice it, as everywhere else in this column.
* **`outputs`, injected, `transforms.json`'s poses are `1e-9 t` and not "exact".** A pose in that
  file is a chain of 4×4 products through R §8, R §9 and R §8.2, and no two matrix kernels give
  bit-identical chains; the row is the same `1e-9 t` the `assembly` row's own pose rows carry, and
  it is met by seven orders of magnitude (V4-D10 took the worst of the eight dumps from 7.0e-13 t
  to **0**).

**`outputs`, injected, `transforms.json`'s key order is gated against the reference's own file,
and it is read from outside the dump** (V5-D6, task Z). R §11.1's `poses` is a Python dict and
`json.dump` writes a dict in insertion order, so the order of the keys is a *result* of R §8 — each
group's seed, then its placements in the order the greedy pass took them, then the singletons in
collection order (V4-D5) — and not a formatting choice. The dump cannot gate it: every JSON file
the fixture sink writes goes through `json.dumps(..., sort_keys=True)`, its own copy of
`transforms.json` included, so the file in the dump is alphabetical by construction. The harness
therefore reads `<dump>/_run/transforms.json`, which is what the *pipeline* wrote in the very run
that produced the dump, and compares the port's key order against it name for name and place for
place; the port's own order comes from its own `assemble` over the reference's own candidate list,
the same run the `assembly` row compares group for group and join for join. The row skips, with the
reason printed, where there is no `_run` — the committed slab dump, whose two fragments would put
nothing at stake anyway. Measured, task Z: **0 differing of 66 places on the seven dumps that carry
one**, and the row is not vacuous — on the terracotta the reference's order
(`021, 094, 104, 007`) differs from the collection order (`007, 021, 094, 104`) in all four places,
which is exactly what the harness used to write, since it passed R §8's insertion order as an
empty slice and every other row of the stage looks its fragment up by name.

`report.md` joins the injected column with task Y: the port renders R §11.3 from the reference's
own `report.json` and the two files are diffed **line for line**, which gates every heading, every
column, every rounding and the legend's Python floats. The comparison stops at `## Timing` — those
are wall-clock seconds and the dump nulls them — while the *order* of that block is R §11.2's and
is checked in `sherd-core`'s own tests.

**The last two rows read the outputs the dump does not carry, and a second Python tool writes
them** (step D2). `tools/dump_fixtures.py` runs the pipeline with `preview=False` and
`write_meshes=False`, so a dump holds the poses that produce R §11.4's meshes and R §11.5's
previews and not the files. `tools/dump_outputs.py DUMP INPUT` produces them from the dump that
already exists — the fragments rebuilt out of `mesh.V`, `mesh.F` and `seg.frac_final`, the poses
out of `outputs/transforms.json`, and the two writers called being the reference's own — and
leaves behind `outputs/placed.sha256.json` (a SHA-256, a size and the two counts per file, plus
the hash of every fragment's mesh *as `load_mesh` leaves it*), `outputs/preview_<k>.png`, the same
render with the caption left off, `outputs/preview_<k>.meta.json` (the views, which PMC-10 makes
library-defined), the `pick`/`u`/`v` of every sample it drew, and `outputs/report.md` — R §11.3
rendered from the dump's own `report.json` by the reference's own writer, with an empty `## Timing`
block because the dump carries no wall clock. **It then rewrites `manifest.json`**, so that
`--verify-checksums` covers what it wrote; until task Y it did not, and the twenty files of the
committed slab dump went unhashed while the check reported "all 180 files match" (V4-D1). The port renders from those and
compares pixels; a dump without them makes the row skip, with the command to run in the message.
The committed `fixtures/slab/dump` carries them at 20 000 samples per fragment so that the
renderer and the PLY writer have a regression fixture in the repository.

**`placed ply` is byte-for-byte on four of the eight sets and cannot be on the other five, and the
row says which** (step D2). A placed mesh is the input file transformed, so it can only be
byte-identical when the input file *reads back* byte-identically, and on the five OBJ collections
it does not: the `load` row above already measures Open3D's reader (Assimp's `fast_atof`) exactly
one `f32` ULP from the port's. The stage therefore hashes both sides' cleaned source meshes first
and compares the placed files only where those agree — the slab, terracotta, synthetic_20 and any
other PLY set, where **every file matches to the byte** — while `placed shape` (size, vertex count,
face count, colour flag) is compared on *every* set and gates the header, the merge and the colour
rule of R §11.4 there too. What the OBJ sets' difference actually looks like was measured rather
than assumed: on pot_C's smallest fragment 178 of 10 737 coordinates differ, every one of them by
exactly one `f32` ULP of the source coordinate, the colour bytes and the face block are identical,
and the file sizes are equal. `SHERD_PARITY_KEEP_PLACED=DIR` keeps the port's own files so that
the next such difference can be looked at the same way.

**The `load` row's coordinate column is a boundary gate by construction, and it is stated here so
that the next OBJ set failing it is read as a parser change rather than as a port regression**
(defect D10 of the phase-1c verification). D §10.2 used to gate only the counts; the harness has
always compared the vertices as well, at one `f32` ULP (`stages/load.rs`'s `COORDINATE_ULPS`), and
that column measures **exactly 1.000 ULP on pot_A and pot_B** — 0.50 on pot_C, 0.25 on pot_G and
pot_H, 0.000 on every PLY set. The row therefore sits on its own limit on the OBJ sets and has no
headroom at all. The cause is the reference's reader, not the port's: Open3D reads OBJ through
Assimp, whose `fast_atof` accumulates the decimal digits itself instead of calling a correctly
rounded `strtod`, and lands one ULP low on the coordinates that need the 24th mantissa bit. The
gate is kept at one ULP because that is the whole of the observed difference and a wider one would
stop measuring anything; what it cannot absorb is a *second* rounding difference on top, which is
exactly the event worth failing on.

**The native thickness row is ±2 % again, and there is nothing left for it to absorb** (task T1,
`notes/2026-09-07-t1-deterministic-thickness.md`). Finding F1 widened it to
`max(2 %, 3 bins of the reference's own histogram)` on measured evidence: R §3.2's `t` was the mode
of a histogram over 20 000 *randomly sampled* rays, PMC-9 let the port draw that sample from
`ChaCha8Rng` rather than numpy's PCG64, and the two implementations were therefore evaluating the
same estimator on different samples — which moved `t` by up to 6.8 % between seeds of the reference
alone. T1 removed the sample instead of replicating PCG64: R §3.2 now casts a ray from every face
of the original largest component, or from `arange(0, n_faces, ceil(n_faces / 300000))` above
300 000, on both sides. The two implementations cast from the same faces of the same mesh, and what
is left between the numbers is `parry3d` against Embree (PMC-17) and nothing else.

Measured on all 68 fixture fragments, native mode: `t` is **bit-identical on 41** and the worst
relative difference anywhere is **7.1e-6** (`FY234021_reduced`), against the ±2 % row —
`3.5e-4` of the tolerance. `thick_mode` is bit-identical on 28 with a worst of 1.2e-4. A one-bin
floor is kept under the row, because two values that land in adjacent bins of a 60-bin histogram
differ by a bin's width whatever else is true; on these fragments the 2 % is what binds, the bin
being 1.7–5.7 % of `t`. The injected row is unchanged and still met bit-exactly on the 48
fragments whose dump carries `load.V0`.

**The measurements would allow a far tighter relative gate and the row is deliberately not taking
it.** 7.1e-6 is 3.5e-4 of the ±2 %, so a 0.1 % row would still pass with a hundredfold margin on
today's fixtures — but `t` is the *mode of a histogram*, a discontinuous function of the hits: on a
fragment whose filtered distances put two bins in contention, one grazing ray resolved differently
by `parry3d` and Embree (PMC-17) moves the answer by a whole bin, which is 1.7–5.7 % of `t` here.
The bin is therefore the unit this row can honestly be stated in, the floor already puts it there,
and the 2 % above it is the smaller of the two on every benchmark fragment. Tightening the relative
half would buy nothing and would fail the first fragment that has a tie.

**The consequence travelled, and it travelled the right way.** `t` is the unit of every threshold
in R §1.2, so the fragments whose `t` used to differ by 4–7 % had `coarse`, `stage1`, `tight`,
`facing`, `gap`, `seam`, `near`, `pen` and `nms` shifted by as much for every pair they took part
in — the risk this section used to carry into the pair stages, and the strongest argument for E6
(replicating PCG64). **E6 is no longer needed for `t`**: preprocessing draws nothing at random at
all, and R §13's exact-set pair gates now see the same thresholds on both sides. PMC-9 still covers
R §3.5's samplers, whose arrays remain incomparable point by point; what it no longer covers is
anything upstream of them.

**The segmentation row passes on all 68 fragments** (it was 66 of 68: `Pot_B_Piece_01_Mesh` at
0.9142 and `frag_010` at 0.9687 against ≥ 0.97). Step B1 had already shown those two gaps were
100 % and 94 % explained by `t`, and with `t` bit-identical they are gone: the worst native
agreement is now **0.9814** (`frag_014`, against ≥ 0.97), the median is 0.9989, and **39 of 68
fragments agree exactly** — their label arrays are the reference's entry for entry. The worst
fracture fraction is 1.14 pp against ±2 pp, also on `frag_014`. The injected column is unchanged:
agreement exactly 1.000000000 everywhere.

**The breakline row's `count` column is now a curve-length column, and that is the team decision of
task T1 rather than a widening** (step B2, `notes/2026-09-06-b2-breaklines.md` §4.2). `res` is
allowed ±10 % natively and a breakline crossing a mesh with longer edges has proportionally fewer
edges to cross, so a ±10 % gate on the *number of points* underneath a ±10 % gate on `res` had no
headroom by construction: on synthetic_20 the port carries 3–18 % fewer points than the reference
while tracing the same curve. The row now measures the curve's **length**, as the sum of every
point's distance to its nearest other point — equivalently `count × mean nearest-neighbour spacing`
— at the same ±10 %. The estimator is not the one the decision named, and the reason is measured:
`count × *median* spacing` disagrees between port and reference by 4.46 % on average and 11.17 % at
worst over synthetic_20, because the spacing along a decimated mesh's breakline is heterogeneous
and a median is not additive, while the sum of the nearest-neighbour distances disagrees by
**1.42 % on average and 3.95 % at worst**. Over all 68 fragments the worst curve length is 3.95 %
of a 10 % allowance. The point *density* is deliberately not gated here at all: it is a function of
`res`, which the working-mesh row already gates, and gating it twice rebuilds the contradiction.

**The `p99 distance` and `dihedral KS` rows are calibrated to PMC-2's own sensitivity, and this is
the derivation** (task X, `notes/2026-09-07-x-phase1c-findings.md` §7). They stood at `0.5 t` and
`0.05` — numbers written before anything was known about how far a decimator can move a breakline —
and they failed five of 204 native comparisons, all on synthetic_20: `p99 distance` on `frag_014`
(0.888 t) and `frag_019` (0.901 t), `dihedral KS` on `frag_010` (0.0636), `frag_014` (0.0655) and
`frag_017` (0.0512). None of the five is a `t` difference, `t` being bit-identical on all five; what
differs is `res`, +2.7 % to +8.3 % on these fragments, inside the working-mesh row's own ±10 %,
because `meshopt` and Open3D's quadric decimators distribute the same face budget differently
(PMC-2).

**Neither row is survivable by the reference at those thresholds, and that is measured rather than
argued.** Run the reference's own pipeline twice on one fragment with nothing changed but the face
budget, and compare its two breaklines at the same `t` — which is what these rows do to the port,
with the decimator's *distribution* of faces added on top. Three collections, 33 fragments, three
budget perturbations each (0.87, 0.825 and 0.75 of the fragment's own working-mesh face count, so
that R §3.3's adaptive budget binds on every fragment rather than only on the large ones), **99
comparisons**:

| `res` gap | n | p99 distance p50 / p90 / max | dihedral KS p50 / p90 / max |
|---|---:|---|---|
| **≤ 10 %** — what the working-mesh row allows | 46 | 0.210 / 0.442 / **1.160 t** | 0.0213 / 0.0331 / **0.0431** |
| all measured (up to 40 %) | 99 | 0.262 / 1.884 / 8.081 t | 0.0244 / 0.0860 / 0.1691 |

The 46 comparisons inside ±10 % are 24 fragments of **two** collections — synthetic_20 (39
comparisons, `res` gaps 3.8–9.8 %) and the terracotta (7, gaps 7.0–10.0 %) — and the two agree about
the size of the effect: the terracotta's own worsts are 0.280 t and 0.0397 against synthetic_20's
1.160 t and 0.0431, so 0.0431 is not an artefact of one set. pot B contributes nothing to that
window because its sherds are small enough that even the mildest perturbation moves `res` by 13 %;
its rows are in the second line, where a 30 % `res` gap puts the reference's own two breaklines
**8.1 t** apart with a KS of **0.169**, which is how steeply both statistics grow with the gap and
why a threshold under the row's own ±10 % measures the working mesh rather than the breakline.

**The gates are twice that sensitivity: `2.3 t` and `0.086`.** The factor of two is a stated choice
and not a measurement, for two reasons that both point the same way: this experiment varies the
*budget* of one decimator, while PMC-2 licenses a *different* decimator, which redistributes the
same budget differently even at an identical `res` — a strictly larger perturbation than the one
measured — and 46 comparisons under-estimate a maximum. What the rows are worth as alarms is the
port's distance from them, and that is the number to watch for a regression rather than the gate:
**0.901 t of 2.3 t (39 %)** and **0.0655 of 0.086 (76 %)**.

**The KS row has the least headroom of any native row, and the reason is not the decimator.** The
reference's own dihedral distribution moves by at most 0.0431 at these `res` gaps and the port's
moves by 0.0655, so resolution does not explain the difference; what does is that `frag_010`,
`frag_014` and `frag_017` carry three of the four lowest segmentation agreements of their set
(0.9885, 0.9814, 0.9868), and a breakline is the **boundary** of the mask that agreement measures —
a 1.9 % area disagreement is a far larger fraction of a boundary than of a surface. That is the same
non-independence this section already describes for `Pf spacing`: both rows are inside their own
gates and the pair of them is what to watch. Over all 68 fragments, 313 of 126 687 point-to-set
distances exceed `0.5 t` (0.247 %), down from 3 115 of 271 592 (1.15 %) before T1 — the old
threshold's own figure, kept because it is the finer instrument for a regression.

**The samples row passes natively on all 408 comparisons** (it was 390 of 396). The four fragments
that used to fail it — `Pot_A_Piece_04` on `n_frac`, `Pot_B_Piece_01` on three columns, `frag_010`
on the fracture fraction, `Pot_G_Piece_05` on `Pf spacing` — were all named in this section as `t`
consequences, and all four pass now. The worst native numbers are `n_frac` 2.0 % (±10 %), fracture
fraction 1.14 pp (±2 pp), margin fraction 1.6 pp (±5 pp) and `Pf spacing` 1.23× the Poisson
expectation (within a factor of two). The cross-set p95 of the 20 000 surface samples agrees with
the Poisson prediction `0.977·√(A/n)` to within **1.34 %** on every fragment, which is the
strongest statement available about a sample PMC-9 forbids comparing directly.

**The injected samples row now reaches the sampler itself** (defect D6 of the phase-1b
verification). It used to compare only what does not depend on which numbers the generator
produced — `n_surface`, `n_frac`, the `fp`-on-fracture test, the margin count and membership, all
exact, and the reference's own points sitting on the faces their own `sp`/`fp` name to 3.3e-11 —
so a port that folded `u + v > 1` the wrong way, or wrote the barycentric expression off one
corner, passed every column. The dump now carries the two uniforms behind each sample before the
fold (`md.S_u`, `md.S_v`, `md.Pf_u`, `md.Pf_v`), and the port rebuilds the reference's own `md.S`
and `md.Pf` from them through its own fold and its own barycentric expression: **1 962 406 points
over 68 fragments, every one bit-identical**. The face pick is checked without uniforms, by
requiring the midpoint of every face's interval of the reference's own cdf to pick that face:
**7 404 452 faces probed, none wrong**. A face whose interval is narrower than an ulp of the cdf is
skipped, because no uniform can distinguish it either; `frag_008` has one such sliver in 126 236
faces.

**`margin bound`: the row gates the array the pipeline computes, which until task Z it did not**
(V5-D5). Task E2 bounded R §3.5.6's `d_brk` sweep — `breakline_distance_below(..., 1.5 t)` reports
every distance at or beyond the outer edge of the band as `∞` instead of finding it, because
`margin_indices` reads the array only as `inner < d < outer` and `∞ < outer` is the same `false`.
That is an equality of *predicates*, not of arrays, and the harness recomputed the band with the
unbounded `breakline_distance` at both of its call sites, so the bounded sweep — a change to the
port's own arithmetic on the critical path — had no standing gate: its only check was E2's
before/after byte comparison of the outputs, which is a measurement of one tree at one commit and
not a gate that runs again. The band is now taken from the pipeline's own bounded array in both
modes, so `margin count`, `margin members` and `margin fraction` exercise it, and a row beside them
compares the two arrays directly, which is the stronger statement: **below `1.5 t` the two must
agree bit for bit** (both take `sqrt` of the same minimal squared distance, the bounded search
having pruned only nodes whose box is further than the radius) **and at or above it the bounded one
must be `∞`**. Anything else counts as a differing entry, and the gate is zero. The row costs one
unbounded sweep per fragment in the harness and nothing in the pipeline. Measured, task Z:
**0 differing of 1 360 000 entries** over the 68 fragments of the eight dumps, injected, and
0 of 1 360 000 natively.

**Its two normal columns needed a conditioning rule, and that rule is arithmetic rather than
fitted.** Narrowing a vertex to `f32` moves it by up to one ulp of its coordinate, and moving a
vertex by `δ` perpendicular to the opposite edge turns the face normal by `δ/h`, where `h` is that
vertex's altitude. The reference's own decimated meshes carry slivers — the worst face of
`frag_008` has edges 0.638, 0.0023, 0.640 at coordinates of 463, whose short edge is 77 `f32` ulps
— so on three of the twenty synthetic fragments the worst normal at a sample moves by 0.155–0.397°
with no port arithmetic involved. Measured against the model bound `ulp/h`, the observed angle is
at most **1.39×** on every one of the 66 fragments, so the bound is the whole story. The row
therefore takes the worst case over the faces with `h ≥ 1000 ulp` — which bounds the turn at 1e-3
rad = 0.057°, under the 0.1° gate *by construction* — and gates the share of samples left out at
0.1 % (measured maximum 0.033 %).

**The three pair rows of step C1 are met exactly, on 358 pairs of eight fixture sets**
(`notes/2026-09-07-c1-hypotheses.md`). Injected: **6 086 comparisons, none failed** — the `(pa, pb)` set
*and* its order are the reference's on every pair, the twelve fields of R §1.2 are bit-identical on
every pair, the coarse score is bit-identical on **every one of the 38.1 million hypotheses of
the eight sets**, and the NMS keeps the reference's 250 hypotheses in the reference's order. Native:
**1 074 comparisons, none failed**; every pair the reference matched the port matches, and the
hypothesis count is within 7.5 % of the reference's at worst against the ±30 % of the row above.

**`cs exact` is a stronger row than D §10.2's `1/60 + 1e-6`, and it is stated separately because it
found a defect the tolerance would have hidden.** The reference's `mean` is a division, and the
port first wrote it as a multiplication by `1/60` — which has no exact double, so `k · (1/60)` is
not `k / 60` for some `k`. That moved 52 of one pot_G pair's 40 029 scores by a single ulp:
`5.6e-17` against a tolerance of `1.7e-2`, three hundred million times inside the gate the row
prescribes, and a difference the ranking below it can still turn into a different candidate. Both
rows are kept: `cs` is the tolerance the algorithm can live with, `cs exact` is the statement the
port can actually make.

**The `nms` row exists only because the dump grew `nms1.order` (§10.1), and the PMC-6 rows say what
that bought.** Run on the port's own stable tie-break instead of the reference's ranking, the same
suppression over the same scores keeps a *different* set: over the 358 pairs it misses **14.3 % of
the reference's kept hypotheses on average and 43.6 % at worst**, and 10.0 % / 38.8 % of the
reference's kept poses lie outside every suppression ball of the port's. What does *not* differ is
their quality: sort both kept sets by score and compare rank by rank, and the two agree to
**one probe point on average and two at worst** (`own order scores`, mean 0.0051, worst 0.0333 =
2/60), while the kept counts agree to 3.3 %. That is the whole content of PMC-6 measured: which
member of a tie the suppression keeps is arbitrary, and the poses stage 1 receives are equally
good either way. The four tie rows are gated at their measured worst plus headroom — they are a
regression alarm on the size of the tie effect, not a parity claim.

**Cost, against the reference on the same machine and the same pairs** (single thread, R §13's own
comparison): a terracotta pair of 35 374 hypotheses takes the reference 0.484 s and the port
0.17 s; a synthetic_20 pair of 163 098 hypotheses takes 2.318 s and 0.80 s. With the default thread
pool the port's two stages are 0.03 s and 0.12 s of wall clock. The port's first version was
*slower* than the reference — 1.08 µs per breakline query against scipy's 0.22 — because it asked
`kiddo` for the unbounded nearest neighbour and then discarded it; R §5.2's radius is
`distance_upper_bound`, and passing it into the query (`PointTree::nearest_within`) took the
coarse stage over synthetic_20 from 1 800 core-seconds to 178 with bit-identical scores.

**The two refinement rows of step C2 are met on all eight sets** (`notes/2026-09-07-c2-icp.md`).
Injected: **3 654 comparisons of stage 1 and 3 400 of stage 2, none failed**, over 71 841 stage-1
and 2 249 stage-2 candidates. The distribution matters more than the worst case here and the rows
report both: the pose deviation is 0.000° and 1e-14 to 1e-11 t at the median, at worst 3e-10 t
(stage 1) and 7e-5 t (stage 2) at p99, and 5.4e-3 t at the maximum — 54 % of the row's 0.01 t, on a
candidate whose registration clouds never touch and whose `fitness` is zero on both sides. R §5.4's
re-score `s1` is identical on every candidate of every set, and R §5.5's suppression keeps the
reference's list exactly when it is walked in the reference's own `nms2.order`.

**The `chaotic` rows are the reason those two rows can be stated at all, and they are an alarm
rather than a claim.** 48 candidates of the 74 090 have ladders that are not functions of their
input at double precision, and step C2 measured that on *Open3D* rather than asserting it: on the
ten stage-2 candidates of `Pot_B_Piece_01__06`, fed the dump's own `s1.T`, Open3D reproduces its own
dumped pose exactly for all ten, and then moving one entry of `T0` by one ULP moves its answer for
candidates 1, 2 and 3 by 24.8°, 65.5° and 101.7° (39–145 t) while leaving the other seven at zero —
and running the same Open3D on ten OpenMP threads instead of one moves the same three by 10–22° and
the other seven by ≤ 8.5e-10 t. The harness excludes exactly those from the pose rows, gates their
share, and requires **exactly zero** of them to have been kept by the reference's own R §5.5 or
accepted by its own verification; on all eight sets that count is zero.

**Cost of the two ladders, single-threaded, against the reference on the same pairs and clouds:**
a terracotta pair's 250 breakline ladders and 10 surface ladders take the reference 0.810 s and
6.220 s and the port 0.078 s and 0.617 s (10.4× and 10.1×); a pot B pair's take 2.158 s and 1.710 s
against 0.344 s and 0.387 s (6.3× and 4.4×). One thread of the port also beats ten OpenMP threads of
the reference by 2.4–6.6×; part of that is structural, since Open3D rebuilds a `KDTreeFlann` inside
every `registration_icp` call and the port builds one tree per cloud and reuses it across the rungs
and the candidates.

**The whole table, as it stands after task X.** Thirteen stages, eight fixture sets, both modes:
**20 618 injected comparisons with no failure, and 2 644 native comparisons with no failure** (task
X, `notes/2026-09-07-x-phase1c-findings.md` §9). The preprocessing half of that is the six stages T1 measured — 3 847 injected and
1 524 native — and its five native failures are the two rows the paragraphs above calibrate to
PMC-2's own sensitivity; the seven pair stages add 16 771 injected and 1 120 native. Before T1 the
same six-stage sweep failed 43 native comparisons across three stages.

**The two verification rows of step C3 are met on all eight sets, and most of them are met
exactly** (`notes/2026-09-06-c3-verify.md`). Injected — R §6 run at the reference's own
`s2.T_frac2`, on the reference's own samples, meshes and `Scales` — **2 508 comparisons over 256
pairs and 2 249 candidates, none failed**. `tight` and `seam` are *bit-identical on every
candidate of every set*; `gap` is 0 at the median and 3.0e-6 t at its worst (pot B); `cont` is
2.7e-14 t at worst and `cont_n` 2.2e-16; `pen` is 0 on six of the eight sets and 5e-5 — one sample
of 20 000 — on the other two. R §6.5's verdict is identical on all 2 249. The `candidates` row is
the ranking on top of that: fed the same candidates the port returns the reference's own five, in
the reference's own order, with the same `accepted` flags and the same `brk_best`, on all 256
pairs.

**The `pen` row is split by whether the two working meshes are closed, and that is a statement
about Open3D rather than a widening.** R §3.3.2's `closed_enough` calls a mesh watertight at up to
0.2 % boundary edges, so R §6.4 asks "is this point inside" of meshes with holes, where a ray that
leaves through a hole flips the parity and the answer is not a function of the geometry. On
`Pot_A_Piece_03_Mesh` (149 boundary edges, wall 3.45) Open3D calls 54 of `Pot_A_Piece_08_Mesh`'s
20 000 samples inside it at depths of 7.6–7.8 units — **2.2 t**, inside a solid whose half wall is
1.72 — and its own `count_intersections` reports an **even** number of crossings for every one of
those points along **all six** axis directions. The port's majority-of-three says outside, and so
does the geometry. `pen` is therefore gated at 0.0005 on the 193 pairs whose two meshes are closed
(met: worst 5e-5) and reported against `max_pen` on the 63 that have an open one (worst 2.7e-3, on
pot A); on both kinds the *decision* is gated exactly and met exactly — no candidate of any set
crosses `max_pen` on one side and not the other, and no `accepted` flag differs.

**The native `pair result` row asked for something unreachable, and C3 measured what is reachable
instead.** "Accepted set identical" cannot hold: the port scores what the reference scores exactly,
but natively it *scores different poses*, because PMC-9 gives it different samples and PMC-6 lets
R §5.3's suppression keep a different set — 14.3 % of the reference's kept hypotheses differ on
average and 43.6 % at worst, which step C1 measured and this section already accepts for the stages
above. R §6.5 is a threshold on the output of that search, and R §13 already records that the
decision "flips on nothing" for a candidate sitting on `min_tight`. The row is now a regression
alarm gated at its measured worst plus headroom, the shape the PMC-6 tie rows already use, and the
*quality* claim is made against the ground truth rather than against the reference:

| set | pairs | port accepts | reference accepts | port true / false | reference true / false |
|---|---:|---:|---:|---:|---:|
| terracotta | 6 | 2 | 2 | exactly R §13's `{021–094, 094–104}` | the same two |
| pot_A | 28 | 17 | 16 | 9 / 8 | 10 / 6 |
| pot_B | 36 | 26 | 21 | 13 / 13 | 11 / 10 |
| pot_C | 21 | 3 | 4 | 2 / 1 | 3 / 1 |
| pot_G | 21 | 2 | 0 | **2 / 0** | 0 / 0 |
| pot_H | 55 | 13 | 16 | 7 / 6 | 7 / 9 |
| synthetic_20 | 187 | 23 | 23 | **23 / 0** | **23 / 0** |

Summed over the sets with an adjacency list the port accepts **56 true joins and 28 false** against
the reference's **54 and 26** — the same quality by any reading — and on `synthetic_20`, the one
set whose ground truth is complete and does not interpenetrate, both accept 23 of 23 true joins
with no false one at all, differing only in which three of them each search happened to find. The
two joins the port accepts on pot_G, where the reference accepts none, are both ground-truth
adjacent; R §13's "no join must be accepted" on that set is a statement about the *assembly* of a
collection whose ground truth interpenetrates, and phase 1d is where it is tested.

**Where both implementations accept a join, they place the sherd in the same spot.** Measured as
the largest displacement the two best candidates give any vertex of the moving fragment — not
`pose_gap`'s origin displacement, which a 0.19° difference 450 units from the origin inflates
tenfold — the median is 0.0096 t on synthetic_20 (max 0.065 t over its twenty shared joins),
0.020 t on the terracotta and 0.007–0.108 t on the pot sets. The tail belongs to the pairs both
sides accept *falsely*: six of the 354 pairs place the sherd more than a wall apart, and every one
of them is a pair with no ground-truth join, where "the best candidate" is one arbitrary way of
resting one sherd on another.

**Cost of a whole `match_pair`, on the terracotta's six pairs** (mean per pair, this machine,
preprocessing excluded on both sides; `sherd-parity`'s `pair_cost` example and the Python driven
through `matching.match_pair(A, B, p, n_threads=N)`). The reference's cost depends on *two* thread
knobs and they interact, so all four combinations are given rather than one:

| reference configuration | mean per pair | port | factor |
|---|---:|---:|---:|
| `OMP_NUM_THREADS=1`, `n_threads=1` | **7.43 s** | 1.03 s (1 thread) | **7.2×** |
| OpenMP free, `n_threads=1` | 3.15 s | — | — |
| OpenMP free, `n_threads=10` | 5.98 s | — | — |
| **`OMP_NUM_THREADS=1`, `n_threads=10`** — the reference's best | **1.70 s** (1.82 s on a repeat) | **0.193 s** (10 threads) | **8.8×** |

The single-thread row is R §13's "≈ 7 core-s per pair" confirmed. **The ten-thread row of this
paragraph used to read 5.83 s and a factor of 29, and it was measured in the wrong configuration**
(defect D11 of the phase-1c verification): with Open3D's OpenMP left free, ten Python threads each
open up to ten OpenMP threads on ten cores and the run comes out *slower than one thread's* — 5.98 s
here, which is the row above. Pinning OpenMP to one thread, which is exactly what the reference's
own pipeline does for its worker processes (`pipeline.py::_worker_env`) and what `_map` already
does for scipy, takes the reference to **1.70 s**. The port's advantage inside one pair is therefore
**8.8× at ten threads and 7.2× at one**, not 29× and not the 17× the old paragraph claimed against
3.40 s — 3.15 s is the OpenMP-free single-Python-thread row, not the reference's best. Within-pair
scaling from one thread to ten is **5.3× for the port and 4.4× for the reference**, not 1.3×; the
reference's pipeline still takes most of its parallelism from worker processes over pairs, but that
is a choice about memory and Python's GIL rather than a within-pair scaling limit.

### 10.3 Benchmark gates

Quality: **within the reference's own spread as R §13 states it** on every listed set, with
cross-object joins 0 and group purity 1.000 **on R §13's seven single-object collections** and
R §13's terracotta row met exactly; run natively (no injection), CPU and GPU. The clause used to
read "everywhere", which is wider than anything R §13 measures and wider than either implementation
achieves: see the `mixed_ABG` paragraph below (decision 2026-09-08, V5-D1). The gate used to read "exactly R §13", and R §13 used to
print one draw of a randomised search: task Y measured the reference at five seeds and at ±5 % of
the working-mesh budget and found three of its seven rows moving, pot_G's prohibition included
(`notes/2026-09-07-y-phase1d-findings.md` §3). Runtime (M2 Pro 10-core / 16-core GPU, warm cache,
`--no-preview`):

| set | CPU gate | GPU gate | when it is run |
|---|---|---|---|
| terracotta (6 pairs) | ≤ 25 s wall | ≤ 15 s | every step |
| pot A (28 pairs) | ≤ 35 s | ≤ 15 s | every step |
| pot H (55 pairs) | ≤ 40 s | ≤ 15 s | every step |
| synthetic 20 (190 pairs) | ≤ 120 s | ≤ 40 s | every step |
| synthetic 60 (≈ 1 770 pairs) | ≤ 15 min | ≤ 5 min | **final acceptance only** (decision 2026-09-07); projected **2.4 min** CPU |
| synthetic 170 (≈ 12 800 pairs) | ≤ 2 h | ≤ 30 min | **final acceptance only** (decision 2026-09-07); projected **17 min** CPU |
| `mixed_all` (12 589 pairs) | ≤ 2 h | ≤ 30 min | **final acceptance only** (decision 2026-09-07); projected **28 min** CPU |

The last three rows are the team's decision of 2026-09-07: the large collections are not run
during development — the development sets are the terracotta, pots A/B/C/G/H, `mixed_ABG` (a
roadmap-item-4 target, not a phase gate; see below) and synthetic 20, and nothing above 27
fragments is started — and the three large rows are checked
once, as the final acceptance after phase 2 and roadmap items 3–4. Until then they are carried as
a **projection from the measured per-pair cost**. E1 measured that cost and E2 halved it
(`notes/2026-09-07-e2-tuning.md` §10): one pair costs **0.523 core-s** on 200 000-face working
meshes (synthetic 20, mean over its 190 pairs, against E1's 1.34) and **0.870 core-s** on real
sherds (pot H, mean over 55, against 1.19), and the pool returns a measured **6.43×** of ten cores
on this machine. That gives 12 800 × 0.523 / 6.43 = **17 min** for synthetic 170, 12 589 × 0.870 /
6.43 = **28 min** for `mixed_all` and 1 770 × 0.523 / 6.43 = **2.4 min** for synthetic 60 — the
first two against the 2 h gate with 6.9× and 4.2× to spare, and the synthetic-60 row against
15 min with 6.3×. Preprocessing adds about 1.5 minutes of wall clock for 170 scans at the
concurrency §5's semaphore admits. The two synthetic figures are upper bounds in the one way that
matters: the 170-piece and 60-piece sets cut the *same* pot the 20-piece set cuts, so their
fragments are smaller than the ones the per-pair cost was measured on, and the coarse score is
still the term that grows with fragment size. **The projection is not a run and does not discharge
the gate**, and it cannot see anything that is not linear in the pair count — R §8's assembly,
R §9's refinement over 170 placed fragments, the merged writer — which on synthetic 20 are 0.03 s
and 1.14 s against 14.64 s of matching.

**Measured, task E2** (`notes/2026-09-07-e2-tuning.md` §9), CPU, warm cache and with the previews
**and** the meshes written — i.e. more work than the gate asks for, and on the same quiet machine
as the phase-1d column beside it:

| set | phase 1d | **E2** | gate |
|---|---:|---:|---|
| terracotta | 3.83 s | **2.27 s** | ≤ 25 s |
| pot A | 10.44 s | **5.10 s** | ≤ 35 s |
| pot H | 10.88 s | **7.93 s** | ≤ 40 s |
| synthetic 20 | 45.85 s | **17.79 s** | ≤ 120 s |

Cold (`--force`, every `cache/*.sherd` rebuilt) the same four are 5.77 → **4.46**, 11.51 →
**6.25**, 11.40 → **8.19** and 52.33 → **27.46 s**; CPU on the largest set falls 320.3 → **130.0**
core-seconds and peak RSS rises 1 576 → 1 879 MiB against §8's 6 GB. Every one of the 114 output
files of those four sets — caches, placed meshes, merged meshes, PNGs, `transforms.json`,
`report.*` outside its timings — is byte-identical to what phase 1d wrote, which is phase 1e's exit
criterion (§12). Against the reference, cold on both sides: 52.7 → **4.5 s** on the terracotta,
130.9 → **8.2 s** on pot H, 426.7 → **27.5 s** on synthetic 20. The three large sets have not been
run and will not be until the final acceptance.

**`mixed_ABG` is a roadmap-item-4 target, not a phase-1 or phase-2 quality gate** (team decision
2026-09-08, from V5-D1). It is the one *mixed* development set — 24 fragments of three pots, 276
pairs — and the phase-1e verification was the first time anybody ran it, on either side. Neither
implementation meets "cross-object joins 0 and group purity 1.000" on it, and the port is the
cleaner of the two on the figure that measures the damage:

| `mixed_ABG`, 24 fragments | **the port** | **the reference** (seed 0, `workers=5`, `OMP_NUM_THREADS=1`) |
|---|---|---|
| fragment accuracy | 14 / 24 = **58.3 %** | 14 / 24 = **58.3 %** |
| per object | A 75.0 %, B 88.9 %, G 0 % | A 62.5 %, B 100 %, G 0 % |
| precision | 0.667 | 0.750 |
| joins used | **18** — 12 correct, 3 wrong pose, **3 cross-object** | **16** — 12 correct, 3 wrong pose, **1 cross-object** |
| groups | 10 + 8 + 2 + 2 + 2×1 | 14 + 3 + 7×1 |
| **group purity** | **0.864** (0.80 and 0.88) | **0.706** (0.64 and 1.00) |
| matching stage | **64.0 s** | 613.7 s (**9.6×**) |

Cross-object joins are what §11's roadmap item 4 (*object separation*) exists to remove — cycle
consistency over whole groups, and per-fragment `Features` with a group consensus — and none of
that is in phase 1 or phase 2. Holding a port of a frozen algorithm to a standard the algorithm
itself does not meet would mean either failing a gate for reproducing the reference or changing
the algorithm to pass it, and both are outside the port's licence (R §12). So the row above is the
**baseline item 4 is measured against**, and the phase gate on this set is the one every other
development set carries: the same decisions as the reference, inside D §10.2's tolerances, at the
runtime §10.3 states. Fragment accuracy is already identical to the fragment, and the port's own
purity is 0.158 above the reference's.

Quality, task Y, all seven collections run natively: R §13's terracotta row **exactly** (the two
joins, 007 unplaced, both `pen` 0, tight 0.56/0.67 and 0.54/0.56, seams 20.667 t and 12.333 t);
pot_A **100 %** / 1.000 (reference 87.5–100 %, task Z), pot_B **88.9 %** / 1.000
(reference 88.9–100 %), pot_C 75 % / 0.667 (reference 50–75 % / 0.500–0.667, task Z: the port is at
the top of both bands), pot_G 0 % with **two** joins, both wrong-pose on
ground-truth-adjacent pairs (reference: 0–2 such joins, seed depending), pot_H 36.4 % / 0.429
(reference 27.3–36.4 % / 0.333–0.500, task Z), synthetic 20 **90 %** / 1.000 (reference 85–95 %);
cross-object joins **0** and group purity **1.000** on every one of these seven. Every row is inside
R §13's restated gate, and the three bands task Z added were measured on the reference at five
seeds each — 15 runs — not derived from the port.

### 10.4 Test layers

1. **Unit** (per crate, `cargo nextest`): geometry helpers (the Python `tests/test_geometry.py`
   cases ported one-to-one), BVH/grid vs brute force (`proptest`), ICP vs a naive f64
   implementation on random clouds, Umeyama vs known transforms, hash-grid tie rules.
2. **Slab fixture** (`fixtures/slab/`): the synthetic slab pair from `tests/test_synthetic.py`,
   generated once by the Python and committed (≈ 6 MB); the Rust tests reproduce
   `test_synthetic.py`'s assertions (pose error ≤ 2° / 0.1 t, segmentation bounds, acceptance).
3. **Kernel cross-checks**: `sherd-refit-rs gpu-check [--set DIR] --stage coarse|icp|distance|inside|all`
   feeds identical batches to both executors and prints one row per stage — items, worst
   deviation, §10.2's tolerance for that quantity, and the count of differing nearest-neighbour
   choices (E7 §5.1's form) — above the self-test's own rows. Run on software adapters in CI
   (`gpu-software` below); on a machine with no adapter it *fails* with "no GPU adapter found"
   rather than passing silently. **In phase 2a the four stage rows come back `delegated`**, because
   `GpuExecutor` routes every method to the CPU until 2b and 2c, and a row that printed a deviation
   of zero would be a lie; the batches are still formed from a real collection and fed to both
   executors, which is what exercises §6.3's formation path before the kernels exist.
4. **Determinism**: two runs, `--threads 1` vs `N`, byte-identical `report.json` per backend.
5. **Golden fixtures**: `sherd-refit parity` against the stored Python fixtures, injected and
   native, every stage (§10.2).
6. **Benchmark gates** (§10.3) on a self-hosted M2 Pro runner, manual/nightly.

### 10.5 CI matrix (GitHub Actions)

| job | runners | what |
|---|---|---|
| `check` | ubuntu-24.04 | `cargo fmt --check`, `clippy -D warnings`, `cargo deny` (licences: MIT/Apache/BSD/Zlib only; meshoptimizer is MIT) |
| `test` | ubuntu-24.04, macos-14 (arm64), macos-13 (x86_64), windows-2022 | layers 1, 2, 4 (CPU) |
| `gpu-software` | ubuntu-24.04 + `mesa-vulkan-drivers` (lavapipe), windows-2022 (WARP via `--gpu-adapter warp`), macos-14 (Metal if the VM exposes it, else skipped and reported) | layer 3 |
| `parity` | ubuntu-24.04, macos-14 | layer 5; fixtures fetched from a private bucket with a repository secret; nightly + `workflow_dispatch` |
| `bench` | self-hosted `m2pro` | layer 6; manual |
| `release` (tags) | the four `test` runners | CLI archives (macOS arm64 and x86_64 separately, Windows x64, Linux x64 built in an ubuntu-20.04 container for glibc 2.31), `maturin-action` abi3 wheels for py3.10+, later `tauri-action` bundles (dmg/msi/AppImage) |

Rust caches via `Swatinem/rust-cache`; `meshopt` needs a C compiler (present on all runners).
Test data licensing: the SfS++ fixtures are CC BY-NC-SA and stay in the private bucket; the
slab and the synthetic Pingsdorf sets (CC BY 4.0 sources) could be public but are large, so
only the slab is committed.

## 11. Where roadmap items 3–6 live

| item | architectural place | what phase 1 already provides |
|---|---|---|
| 3 confidence tiers, review images, constraints | `Tier` on `Candidate`; a second `Thresholds` set (`confirmed`, `probable`) in `Params`; `report.rs` gains "Probable joins" and per-join `tier`; `render.rs` gets `render_pair(a, b, T)` (both fragments, three views, seam highlighted) used by `--review-images`; `assembly/constraints.rs` reads `constraints.json` `{must_join, must_not_join}`: `must_not_join` removes pairs before matching and rejects candidates, `must_join` forces the pair through the second-pass budget and accepts its best candidate that passes the *probable* thresholds; the desktop app writes this file from the review screen | the types and the file format; `tier` = `Confirmed ⇔ accepted` |
| 4 object separation (**measured on `mixed_ABG`**, whose baseline both implementations set in §10.3: port 3 cross-object joins at purity 0.864, reference 1 at 0.706, fragment accuracy 58.3 % on both) | `assembly/consistency.rs`: cycle consistency is evaluated for every candidate join against all paths in the group (the reference checks only direct alternatives); `Features` per fragment computed in preprocessing and stored in the cache: `shell_radius` (quadric fit on the shell samples), `colour_lab_mean/std` (from vertex colours when present; the PLY reader keeps them), `thick`; `GroupFeatures` = medians + MAD; `groups.rs` reports every group as an object with its consensus and flags a join whose fragment deviates > k·MAD | `Features` struct and cache slots (empty in phase 1), the PLY colour path |
| 5 group-level matching | `matching::Matchable` trait implemented by `MatchData` (one fragment) and `GroupMatchData` (members with poses): breakline = union minus points within `seam` of another member's breakline; fracture samples = union minus points within `2·tight` of another member's fracture surface; BVH = two-level (member BVHs + transforms, no rebuild); `Executor` batches carry per-member transforms; the pipeline's block scheduler treats a group as one entity | the trait boundary; the BVH designed with a top level from day one |
| 6 special parts, memory | `FaceLabel::Solid` from a volume test in `segment.rs` (cone rays that hit no far wall but whose centroid is enclosed at > 2 t depth); `Rim` from the thickness estimator's second mode; both excluded from the fracture mask and from breaklines; memory-bounded preprocessing is §5 step 2; streaming decimation (out-of-core) remains an open question (§13) | u8 labels, the semaphore |

## 12. Phasing, effort, risks

Effort in engineer-weeks for one senior Rust engineer with geometry experience; two engineers
shorten phase 1+2 to ≈ 14 weeks because GPU work can start once the CPU ICP is verified.

| phase | content | weeks | exit criterion | main risks |
|---|---|---|---|---|
| 0 | Python fixture sink and dumps; `compare_fixtures.py`; fixtures for terracotta, pots A/B/C/G/H, synthetic 20, summary fixtures for the two large sets; slab fixture committed | 1 | fixtures reproducible from `9d4b9d3` twice, byte-identical | none |
| 1a | workspace, IO (E2), cleaning, components, thickness, decimation (E1), Taubin, working mesh, cache | 2.5 | native tolerances of §10.2 up to "working mesh" on all fixtures | decimation choice; PLY edge cases |
| 1b | BVH, hash grid (E3, E4), segmentation, breaklines, match arrays | 2.5 | segmentation ≥ 0.995 injected / ≥ 0.97 native; breakline gates | BVH correctness; `voxel_down_sample` semantics |
| 1b, step B1 | done: segmentation (R §3.4), `spatial::bvh`, `spatial::kdtree`, the `labels` tensor, the `segmentation` parity row | | injected 1.000000000 on all 68 fragments; native 66 of 68 (§10.2 on the two others) | `voxel_down_sample` semantics settled: the bucket rule is exact, only the voxel *order* is a hash artefact (PMC-4) |
| 1b, step B2 | done: breaklines and frames (R §3.5.3–3.5.5), the five `brk_*` tensors, the `breakline` parity row | | injected exact on all 66 fragments (the only residuals are the `f32` cache narrowing); native 165 of 198 checks, every failure inherited (§10.2) | none new; the native `count` row is found to contradict the `res` row above it |
| 1b, step B3 | done: the sampled match arrays (R §3.5.1–3.5.2, §3.5.6), `MatchData` (R §3.6), the five sampled tensors, the `samples` parity row | | injected 660 of 660 on all 66 fragments (`n_frac`, the margin and the face indices exact); native 390 of 396, six failures on four fragments the segmentation and thickness rows already name (§10.2) | none new; the native `Pf spacing` column is found to inherit the `segmentation` row above it |
| 1c | hypotheses, coarse, NMS, ICP (E5), verification, `match_pair`, screening flags | 2.5 | stage-2 injected tolerances on every fixture pair | ICP corner cases (empty correspondences), tie handling |
| 1c, step C1 | done: pair scales (R §1.2, §4.2), hypotheses (R §5.1), the coarse score (R §5.2) and the NMS (R §5.3), with the `hypotheses`, `coarse` and `nms` parity rows | | injected 6 086 of 6 086 on 358 pairs of eight sets — the `(pa, pb)` set and order exact, `Scales` bit-identical, `cs` bit-identical on 38.1 M hypotheses, the kept list identical on the reference's own walk order; native 1 074 of 1 074 (§10.2) | PMC-6's tie effect is now measured rather than assumed: membership differs by up to 43.6 %, quality by at most two probe points |
| 1c, step C2 | done: Open3D's ICP (R §7), the two refinement ladders (R §5.4–5.6), R §5.5's suppression and the `stage1`/`stage2` parity rows, with experiment E5 | | injected 3 654 (stage 1) and 3 400 (stage 2) with no failure on 358 pairs of eight sets — median pose deviation 0.000° / 1e-13 t, worst 5.4e-3 t of 0.01 t; no native column (§10.2) | 48 candidates of 74 090 have ladders neither implementation can reproduce, measured on Open3D itself; E5 answered: `f32` point loops and the centred assembly are both outside §10.2 (§3, §7) |
| 1c, step C3 | done: R §6's five verification scores and R §6.5's rule, R §5.7's ranking, the whole of `match_pair`, R §4.3's screening, and the `verify` and `candidates` parity rows | | injected 2 508 of 2 508 on 256 pairs of eight sets — `tight` and `seam` bit-identical on every candidate, `gap` ≤ 3e-6 t, `cont` ≤ 2.7e-14 t, `accepted` identical, the returned five in the reference's own order; natively the accepted sets differ and the row becomes an alarm (§10.2) | Open3D's own signed distance is self-contradictory on a mesh that `closed_enough` accepts (R §6.4); the native accepted set cannot be identical and is measured against the ground truth instead |
| 1d | assembly, refinement, recentre, report/transforms/meshes, renderer, CLI, determinism tests | 2 | R§13 gates natively; CI green on 4 OSs | none major |
| 1e | profiling and CPU tuning to §10.3 CPU gates | 1.5 | §10.3's CPU gates **on the development sets** — terracotta, pots A/B/C/G/H, `mixed_ABG` (a roadmap-item-4 target: its cross-object joins and group purity are **not** a phase gate, §10.3), synthetic 20 — with every output byte-identical to the one phase 1d wrote; the three large rows are carried as the projection §10.3 states and discharged at the final acceptance (decision 2026-09-07) | the 2 h collection gate is no longer measured here: E1 projects 39–44 min against it from the per-pair cost, which is a projection and not a run |
| **phase 1 total** | | **11** | | |
| 2a | wgpu device/adapter/self-test, buffers, slots, batch structs; E7, E8 | 1.5 | self-test passes on Metal + lavapipe | naga/driver issues |
| 2a, task G1 | done: the `Executor` boundary made real (§6.1) with the CPU path routed through it and byte-identical, `spatial::grid`'s hash grid (§6.2), the `sherd-gpu` crate — device and `--gpu-adapter` (§6.8), buffer chunking and the 2-D dispatch fallback, the 32-slot LRU (§6.3), the self-test on E7's two kernels, `gpu-check` (§10.4 layer 3) | | reduction bit-identical (`0x49a7230c`), bounded NN 1 differing of 384 000 at `max |Δd|` 1.3e-7, 12.2–17.9 ns/query and 3.3–6.3× the ten-core CPU; CPU outputs byte-identical to `9bf35d6` on the four sets (92 files) and parity 23 804 / 0 | lavapipe and WARP are CI's to answer; the discrete-GPU staging path is untestable here (E7 §6) |
| 2b | hash grid + `icp_rung` (both estimators, in-kernel solves), `coarse_scores` | 2.5 | CPU/GPU cross-check within §10.2 | shared-memory limits; f32 conditioning |
| 2c | BVH kernels (`bounded_distance`, `inside`) | 1.5 | cross-check | traversal stack in WGSL |
| 2d | scheduler, pipelining, TDR chunking, memory management | 1.5 | the GPU gates of §10.3 **on the development sets** (`mixed_ABG` included, on the same terms: it is roadmap item 4's baseline, not a quality gate), and §5's memory semaphore holding a projected 170-scan preprocessing budget (E1 §7); "synthetic 170 ≤ 30 min" is the final acceptance after phase 2, not a 2d exit (decision 2026-09-07) | overlap efficiency; a projection carried this long can be wrong in a way only the run shows |
| 2e | vendor matrix (NVIDIA/AMD/Intel/Apple), tuning | 2 | E8 matrix green | Intel/AMD driver quirks |
| **phase 2 total** | | **9** | | |
| 3a | pyo3 module, numpy interop, `SHERD_REFIT_BACKEND=rust` routing in the Python package, A/B on the fixtures | 2 | Python pipeline with Rust kernels reproduces the Rust CLI | packaging on Windows |
| 3b | Tauri 2 app: collection open, run with progress/cancel, report view, GLB viewer, probable-join review writing `constraints.json`, re-run | 3 | museum walkthrough on the terracotta set | signing/notarisation |
| 3c | packaging, code signing, release pipeline | 1 | tagged release with binaries, wheels, bundles | Apple developer account |
| **phase 3 total** | | **6** | | |

Cross-cutting risks: (1) the algorithm is still moving (segmentation precision is the known
blocker per the scale-pairs note) — mitigated by freezing at `9d4b9d3` and porting behind
fixtures; subsequent algorithm changes go into the Python first, produce new fixtures, then into
Rust, until phase 3a flips the reference; (2) decimation non-identity makes native-mode parity
statistical rather than exact — accepted and measured; (3) the 2 h CPU-only target is not
guaranteed by the estimates; (4) GPU driver diversity — mitigated by the self-test and the
automatic fallback.

## 13. Open questions for the team

1. **Decimator:** accept a working mesh that differs from Open3D's (statistical parity), or
   invest a week in an own Garland–Heckbert to get closer? (Recommendation: E1 first; accept.)
2. **Parity standard:** tolerance-based parity as in §10.2, or bit-parity in native mode via
   numpy RNG replication (E6, ≈ 2 days, fragile)? (Recommendation was: tolerance-based.
   **Now open again on evidence:** step B1 showed the thickness sample's seed spread breaking the
   native segmentation gate on 2 of 68 fragments, with the port's `t` nearer the estimator's
   centre than the fixture's on one of them — §10.2's t-propagation paragraph and
   `notes/2026-09-06-b1-segmentation.md` §5. E6 would remove the class rather than absorb it.
   Step B2 carried the same evidence one stage further: 3 115 of 271 592 native breakline
   point-to-set distances exceed `0.5 t`, and on the fragments whose working mesh is *identical*
   to the reference's — 44 of 66, all nine of pot_B — every one of them lies within `0.169 t` of a
   face the two segmentations label differently, so the whole of the native breakline shortfall is
   `t` arriving at the segmentation, not breakline code. `notes/2026-09-06-b2-breaklines.md` §4.)
3. **PMC items to apply in phase 1** (R§12): proposed to apply PMC-4, 6, 11, 12, 13, 14 in
   phase 1 (no result change expected), PMC-1, 7, 8 in phase 1 with re-verification, and to
   leave PMC-5 for a later algorithm change. Confirm.
4. **Fixture hosting and licensing:** where the private bucket lives (SfS++ data is
   CC BY-NC-SA), and whether the synthetic Pingsdorf fixtures may be public.
5. **Minimum GPU:** support Intel iGPUs and 2 GB discrete cards at reduced batch sizes, or
   require ≥ 4 GB and a 2019+ driver? (Recommendation: support with the automatic fallback.)
6. **Naming during the transition:** the Rust binary takes the name `sherd-refit` and the
   Python entry point becomes `sherd-refit-py` once phase 1 passes the gates; or keep
   `sherd-refit-rs` until phase 3?
7. **Reference hand-over:** at which gate does the Rust core become the algorithm's reference
   (proposed: end of phase 3a), after which algorithm changes are made in Rust and the Python
   package is retired?
8. **Out-of-core decimation** for scans above ≈ 10 M faces on 8 GB machines: needed for the
   museum's scans, or is a "fewer workers" budget enough?
9. **Code signing:** an Apple Developer account and a Windows certificate are needed for the
   desktop bundles in phase 3c.
10. **Target CPU for the 2 h gate:** confirm the M2 Pro 10-core (or specify the museum's
    workstation) so the gate is measurable.
