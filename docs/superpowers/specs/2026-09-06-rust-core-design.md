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
    sherd-gpu/                    wgpu executor: device.rs buffers.rs slots.rs kernels/*.wgsl scheduler.rs selftest.rs
    sherd-cli/                    binary `sherd-refit`: run, segment, parity, bench, info
    sherd-py/                     pyo3 module `sherd_refit_core` (maturin, its own pyproject.toml)
    sherd-parity/                 fixture reader/writer + stage runners used by `sherd-refit parity`
  apps/desktop/                   Tauri 2 app (phase 3)
  fixtures/slab/                  the synthetic slab pair (self-made, redistributable) + expected outputs
  tools/dump_fixtures.py          Python side of the parity harness (§10.1)
  tools/compare_fixtures.py       stage-by-stage comparison with tolerances (§10.2)
```

`sherd-core` has no GPU dependency and compiles on every target; `sherd-gpu` is an optional
feature of `sherd-cli` (`--features gpu`, on by default in release builds).

## 3. Dependencies

| need | crate (version to pin at start) | why | risk / unknown | experiment that resolves it |
|---|---|---|---|---|
| linear algebra | `nalgebra` 0.33 | f64 poses, 6×6 LDLT, 3×3 SVD (Umeyama) | none | — |
| small vectors in hot loops | own `Vec3f` newtype (`#[repr(C)]`, `bytemuck::Pod`) | identical layout on CPU and GPU buffers | none | — |
| parallelism | `rayon` 1.10 | pairs, candidates, per-face loops | none | — |
| GPU | `wgpu` 24 (pin the minor), `bytemuck`, `pollster` | Metal/Vulkan/DX12 from one WGSL source | driver bugs on Intel iGPU and older AMD; software adapters in CI; fast-math status of naga output | E7 (fast-math), E8 (adapter matrix) |
| KD-tree (unbounded NN on CPU) | `kiddo` 5 (`ImmutableKdTree<f32,3>`) | seam and continuity tests, `near`, `d_brk`, margin | API churn between majors; f32 only for the immutable tree (fine) | — |
| radius-bounded NN | own hash grid | ICP correspondences, coarse score, macro-normal balls; same layout as the GPU kernel | none beyond effort | unit tests vs brute force |
| BVH: rays, closest point, inside | own flattened BVH (SAH binned build, 4 tris/leaf, 32-byte nodes) | Embree-equivalent semantics for §R3.2, §R3.4.3, §R6.1, §R6.4, shared with GPU | effort ≈ 1.5 weeks; performance vs Embree (expect 3–5× slower single-thread, irrelevant at these sizes) | E4 (parity vs Open3D distances on the terracotta) |
| alternative considered | `parry3d` 0.17 | ready TriMesh QBVH with `project_point`, `cast_ray`, pseudo-normal inside test | second layout to maintain for the GPU; f32/f64 duplication; no bounded closest-point early exit | rejected unless the own BVH slips by > 2 weeks |
| quadric decimation | `meshopt` 0.4 (meshoptimizer C via `cc`) | fast, topology-preserving, cross-platform, actively maintained | different collapse order than Open3D → different working mesh (PMC-2); `simplify` may stop above the target when topology blocks it; border locking behaviour | E1 |
| decimation fallback | `baby_shark` 0.3 (`decimation::EdgeDecimator`) | pure Rust, Garland–Heckbert, closer to Open3D's algorithm | young API, speed unknown on 10 M faces | E1 |
| mesh read | own PLY parser; `tobj` 4 (OBJ), `stl_io` 0.8, own OFF, `gltf` 1 (GLB with `COLOR_0`) | PLY is the main format and needs colours + big files fast; Open3D's writer output must round-trip | `mesh-loader` was considered (one crate for PLY/OBJ/STL) — colour support in PLY unverified | E2 |
| mesh write | own PLY writer (binary LE, `uchar` RGB), `gltf-json` for GLB export | outputs must match Python's files byte-compatibly in layout | — | E2 |
| cache | `safetensors` 0.4 | mmap-friendly, JSON header with metadata, readable from Python | 2-byte alignment only; we align tensors ourselves by ordering | — |
| fixtures | `npyz` 0.8 (read `.npy`) + `serde_json` | Python writes `.npy`; no zip needed | — | — |
| images | `image` 0.25 (PNG) + embedded 5×7 bitmap font | preview PNGs without a font stack | — | — |
| CLI / logging / errors | `clap` 4, `tracing` + `tracing-subscriber`, `anyhow`, `thiserror` | — | — | — |
| RNG | `rand_chacha` 0.9 (`ChaCha8Rng::seed_from_u64`) | portable, versioned stream guarantee across platforms | not numpy-compatible (PMC-9) | — |
| Python bindings | `pyo3` 0.23 + `numpy` 0.23, `maturin` | transition and the parity harness | ABI3 wheels need `abi3-py310` feature | — |
| desktop | `tauri` 2 | later | — | — |
| tests | `cargo nextest`, `proptest`, `approx`, `criterion` | — | — | — |

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
  result must stay within §10.2 tolerances (expected: 1e-5 t).
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
    thick: f32, thick_mode: f32, watertight: bool, n_boundary: u32, n_orig_vertices: u32, n_orig_faces: u32, target_faces: u32,
    mesh: WorkingMesh, labels: Vec<FaceLabel>,
    md: Option<MatchArrays>,               // built at `thick`
    features: Features,                    // roadmap items 4 and 6, §11
    bvh_full: OnceLock<Arc<Bvh>>, bvh_frac: OnceLock<Arc<Bvh>>,   // built on first use, shared
}
pub struct MdParams { t: f32, seed: u64, surface_points: u32, frac_per_t2: f32, min_frac_points: u32, max_frac_points: u32, margin_points: u32, macro_inner: f32, macro_outer: f32, brk_voxel: f32 }
pub struct MatchArrays { params: MdParams, s: Vec<Vec3f>, sp: Vec<u32>, pf: Vec<Vec3f>, fp: Vec<u32>,
                         brk_p: Vec<Vec3f>, brk_ns: Vec<Vec3f>, brk_nf: Vec<Vec3f>, brk_f: Vec<Vec3f>, brk_sub: Vec<u32>, margin_idx: Vec<u32> }
pub struct MatchData<'a> { fr: &'a Fragment, t: f32, arrays: Cow<'a, MatchArrays>,
                           brk_t: Vec<Vec3f>, brk_dih: Vec<f32>, pc_reg: Cloud, pc_frac: Cloud, pc_brk: Cloud, pc_brk_full: Cloud,
                           pm: Cloud, kd_brk: KdTree, kd_margin: KdTree, grids: GridCache /* keyed by (cloud, radius) */ }
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

### 4.2 Fragment cache: `<out>/cache/<name>.sherd`

A `safetensors` file. Tensors (all little-endian): `V f32[n,3]`, `F u32[m,3]`, `labels u8[m]`,
`S f32[20000,3]`, `sp u32`, `Pf f32`, `fp u32`, `brk_P/brk_ns/brk_nf/brk_f f32[k,3]`,
`brk_sub u32`, `margin_idx u32`, optional `features/*`. Metadata (string map): `format=sherd-cache`,
`cache_version=1`, `algo_ref=2026-09-06/9d4b9d3`, `core_version`, `name`, `source_path`,
`source_size`, `source_mtime_ns`, `source_sha256` (optional), `target_faces`, `thick`,
`thick_mode`, `res`, `watertight`, `n_boundary`, `n_orig_vertices`, `n_orig_faces`,
`md_params` (JSON), `features` (JSON), `created`, `backend`.

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

One process, one `rayon` pool sized `--threads` (default: all cores). Stages:

1. **Discover** files, names, pair order (R§2, R§4.1).
2. **Preprocess** (R§3): a `par_iter` over fragments **bounded by a memory-aware semaphore**:
   a job for a scan of `f` faces reserves `60 MB + 110 B·f` (measured for meshopt + our arrays;
   re-measure in E1) from a budget of `--memory-budget` (default 50 % of physical RAM); jobs
   wait for the reservation. Large scans are read straight from the file into the vertex/face
   arrays (no intermediate copies). Cache hits skip everything.
3. **Match** (R§5–6): pairs are grouped into blocks of 3×3 fragments in collection order
   exactly as the Python does (`R` pipeline `_pair_blocks`), the blocks are the `par_iter` items,
   pairs inside a block run sequentially, candidates inside a pair run `par_iter` (nested
   parallelism is fine under rayon's work stealing; results are collected by index, so nothing
   depends on scheduling). Per-fragment derived data (`MatchData` at a given `t`) lives in a
   shared LRU (`moka`-free: a `Mutex<LruCache<(FragId, t_bits), Arc<MatchData>>>` of 64 entries)
   so a fragment recomputed at `t_pair` for one pair serves the next.
   With the GPU executor the block loop becomes a software pipeline (§6.4).
4. **Assemble** (R§8), **refine** (R§9), **recentre**, **outputs** (R§11). Full-resolution meshes
   are streamed one at a time; the merged assembly PLY is written by first summing the members'
   header counts, then appending each transformed member, so no merged mesh is ever in memory.

Cancellation: every stage checks an `AtomicBool` between units of work (a pair, a fragment); the
CLI wires Ctrl-C, the desktop app wires a button. Progress: a `Progress` trait with
`(stage, done, total)` callbacks; the CLI prints, the app emits events.

## 6. Executor abstraction and the GPU plan

### 6.1 The interface

```rust
pub trait Executor: Send + Sync {
    fn coarse_scores(&self, b: &CoarseBatch) -> Vec<f32>;                 // R§5.2, many pairs
    fn icp_rung(&self, b: &IcpBatch) -> Vec<IcpResult>;                    // R§7, one rung (up to max_iter) for many candidates
    fn bounded_distance(&self, b: &DistBatch) -> Vec<f32>;                 // point-to-triangle-set distance, exact below `radius`, +inf above
    fn inside(&self, b: &InsideBatch) -> Vec<InsideResult>;                // (inside: bool, dist: f32 for inside points beyond `pen`)
    fn cone_cast(&self, b: &ConeBatch) -> Vec<u8>;                         // R§3.4.3 vote counts (phase 2b, optional)
}
```

`CpuExecutor` is the reference implementation of every method (rayon inside); `GpuExecutor`
(crate `sherd-gpu`) implements the same methods with WGSL kernels and identical batch structs.
Everything else in the pipeline — hypotheses, NMS, seam/continuity via `kiddo`, scoring
arithmetic, assembly — stays on the CPU and is written once. `Backend::Auto` picks the GPU only
if an adapter exists, the self-test passes and its measured throughput beats the CPU by ≥ 1.5×
(§6.8).

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

Per mid-size pair (R§13 cost structure), single-thread Python core-seconds → estimated Rust CPU
core-seconds → estimated GPU seconds (M2 Pro 16-core GPU, ≈ 0.5–1 G bounded NN queries/s on the
hash grid, ≈ 0.1–0.2 G BVH closest-point queries/s):

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
per-invocation striding (`i = lane + 256·k`), and the reduction tree (256 → 128 → … → 1). With
E7 confirming no fast-math and no FMA contraction, the two are expected to agree to a few ULPs
per iteration and to the §10.2 tolerances after 30 iterations; the CI cross-check (§10.4)
enforces it. The CPU path additionally offers `--precision f64` for the ICP (generic `Real`
type) as a diagnostic, not a production mode.

### 6.8 Operational GPU concerns

- **Adapter selection:** `wgpu::Instance` over Metal | Vulkan | DX12 (no GL); default
  `HighPerformance` power preference; `--gpu-adapter NAME|INDEX`; `sherd-refit info` lists
  adapters, limits and the self-test result.
- **Self-test at start:** the slab pair's stage-2 ICPs and verification run on CPU and GPU;
  results must agree within §10.2; failure → warning + CPU fallback. Also measures throughput
  for `Backend::Auto`.
- **Limits:** request `max_storage_buffer_binding_size ≥ 256 MB` when available, else chunk at
  128 MB; workgroup size 256 (≤ `max_compute_invocations_per_workgroup`); shared memory ≤ 16 KB
  per workgroup (29 floats × 256 lanes = 29.7 KB → reduce in two passes of 128 lanes or use
  `f32` pairs; design the reduction for 16 KB from the start).
- **Timeouts:** dispatches ≤ 100 ms by construction (§6.4); `device.poll` with a watchdog; a
  device loss mid-run falls back to the CPU for the remaining blocks and is recorded in the
  report.
- **Memory:** slots + batches ≤ 1 GB; on adapters reporting < 2 GB the slot count halves and
  `P` shrinks.
- **Precision:** f32 only (`f16`/`f64` unused); `Scales` and thresholds computed in f64 on the
  CPU and passed as f32.

## 7. Numerical determinism

| source of nondeterminism | policy |
|---|---|
| sampling | `ChaCha8Rng` seeded from `p.seed` (and hard-coded 0 where the reference does), draws in the reference's order (R§10) |
| unordered containers | none on any result path; voxel representatives sorted ascending (PMC-4) |
| sorting | `sort_by` with explicit keys and ascending-index tie-break; stable sorts only |
| parallel reductions | fixed-order trees on both executors; no floating-point atomics; rayon results collected by index |
| NN ties | lowest index (fixed traversal order) |
| ICP convergence | per-candidate `done` flag exactly as the sequential loop (R§7); converged candidates are never iterated further |
| thread count | results must be identical for `--threads 1` and `--threads N` (CI test) |
| CPU vs GPU | within §10.2; not bit-identical (different ULP behaviour); the report records the backend |
| platforms | same backend, same binary → identical; across OS/compilers → f32 ULP-level differences are possible in `libm` calls (`acos`, `sin`); tolerance-based |
| ill-conditioning | point-to-plane systems are assembled in coordinates centred on the target centroid and the update re-expressed about the origin (an exact re-parameterisation of the same Gauss–Newton step; differences O(|ω|²·|c|) ≈ 1e-6 t); f32-safe |

## 8. Memory budget (170 scans)

| item | per unit | 170 fragments | notes |
|---|---|---|---|
| original scan in memory during preprocessing | 3 M faces: 60 MB + 110 B·f ≈ 400 MB; 10 M faces ≈ 1.2 GB | bounded by the semaphore: `⌊0.5·RAM / cost⌋` concurrent | 16 GB laptop: 10 concurrent 1 M-face scans, 3 concurrent 10 M-face scans |
| cache file (mmap) | ≈ 6 MB (V f32 1.2, F 2.4, labels 0.2, arrays 2) | ≈ 1 GB address space, paged | resident only when touched |
| derived per fragment (FN, A, C, grids, fracture BVH) | ≈ 6 MB | ≈ 1 GB if all resident; LRU of 64 `MatchData` ≈ 400 MB | |
| full BVH (penetration) | ≈ 4 MB | in the LRU | |
| matching transient per pair | hypotheses 150k × 48 B ≈ 7 MB + grids 1 MB | ≈ 10 threads × 10 MB | |
| assembly | poses, candidates | negligible | |
| refinement clouds | ≤ 150k × 24 B = 3.6 MB per placed fragment | ≤ 0.6 GB (all placed) | freed per group |
| outputs | one original mesh at a time | ≤ 1.2 GB peak (10 M faces) | streaming PLY writer |
| **peak RSS** | | **≈ 3–6 GB** (≤ 3 M faces), **≤ 10 GB** (10 M-face scans, 3 preprocessing workers) | |
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
   thick.idx thick.t_hit thick.prim thick.t thick.thick_mode thick.target   R§3.2–3.3
   mesh.V mesh.F mesh.res mesh.watertight                   R§3.3 (working mesh after Taubin)
   seg.rep seg.near seg.NS seg.good seg.frac_raw seg.frac_majority seg.frac_islands seg.ref seg.has_ref seg.frac_final   R§3.4
   md.<each MD_ARRAY> md.params md.brk_t md.brk_dih md.valid                 R§3.5–3.6
DIR/pairs/<a>__<b>/
   scales.json  hyp.pa hyp.pb  coarse.idx coarse.cs  nms1.kept
   s1.T[250,4,4] s1.score  nms2.kept
   s2.T_reg1 s2.T_reg2 s2.T_frac1 s2.T_frac2 (per candidate)  s2.scores.json  s2.accepted
   result.candidates.json (the 5 returned)
DIR/assembly/  md_t_median samples (S per fragment, 15000), poses.json, groups.json, used.json, rejected.json
DIR/refine/    <name>.idx (fracture cloud indices), per-join T after each rung, poses_final.json
DIR/outputs/   transforms.json report.json
```

Sizes: terracotta ≈ 40 MB, pot A ≈ 90 MB, synthetic 20 ≈ 350 MB (all pairs). For `mixed_all` and
`synthetic_170` only `mesh`, `seg.frac_final`, `md.*`, `result.candidates.json` and the assembly
are dumped (≈ 0.6 GB). Fixtures are generated once from commit `9d4b9d3` and stored outside
git (§10.5). The Rust CLI writes the same layout with `--dump-fixtures`.

### 10.2 Stage comparison and tolerances (`tools/compare_fixtures.py REF NEW`)

Two modes per stage: **injected** (the Rust stage ran on the Python stage's inputs) and
**native** (the Rust stage ran on Rust's own upstream results). Segmentation agreement is
measured by sampling 200 000 area-weighted points on the Python working mesh and labelling each
by its nearest face on each mesh.

| stage | quantity | injected tolerance | native tolerance |
|---|---|---|---|
| load | counts after cleaning, largest component | exact | exact |
| thickness | `t`, `thick_mode` | same bin, or ±1 bin on a count tie | ±2 % |
| working mesh | faces, `res`, area, `watertight` | (mesh is injected) | faces ±5 %, `res` ±10 %, area ±0.5 %, same `watertight` |
| segmentation | area-weighted label agreement; fracture fraction | ≥ 0.995; ±0.005 | ≥ 0.97; ±0.02 |
| breakline | count; point-set Hausdorff; `dih` per matched point | exact; 1e-4 t; 0.1° | ±10 %; 0.5 t on 99 %; distribution KS < 0.05 |
| hypotheses | `(pa, pb)` set | exact | count ±30 % |
| coarse | `cs` per hypothesis | ≤ 1/60 + 1e-6 | — |
| stage 1 | pose per kept hypothesis (by id); `s1` | 0.05° / 0.01 t; ±0.02 | — |
| stage 2 | pose per candidate (by stage-1 id); `tight`; `gap`; `seam`; `cont`; `cont_n`; `pen`; `accepted` | 0.05° / 0.01 t; ±0.01; ±0.002 t; ±0.34 t; ±0.005 t; ±0.01; ±0.0005; identical | — |
| pair result | accepted set; best candidate of pairs with an accepted join | identical; 1° / 0.05 t | identical; 1° / 0.05 t (perf-note criterion); no requirement on the best candidate of pairs without a join |
| assembly | groups, joins used, rejections | identical | identical |
| refine | relative poses within a group | 0.2° / 0.02 t | 0.2° / 0.02 t |
| outputs | `transforms.json` poses; `report.json` keys | as refine; schema | as refine; schema |

The tool exits non-zero on any violation and prints a per-stage table.

### 10.3 Benchmark gates

Quality: exactly R§13 on every listed set, run natively (no injection), CPU and GPU. Runtime
(M2 Pro 10-core / 16-core GPU, warm cache, `--no-preview`):

| set | CPU gate | GPU gate |
|---|---|---|
| terracotta (6 pairs) | ≤ 25 s wall | ≤ 15 s |
| pot A (28 pairs) | ≤ 35 s | ≤ 15 s |
| pot H (55 pairs) | ≤ 40 s | ≤ 15 s |
| synthetic 20 (190 pairs) | ≤ 120 s | ≤ 40 s |
| synthetic 170 (≈ 12 800 pairs) | ≤ 2 h | ≤ 30 min |
| `mixed_all` (12 589 pairs) | ≤ 2 h | ≤ 30 min |

### 10.4 Test layers

1. **Unit** (per crate, `cargo nextest`): geometry helpers (the Python `tests/test_geometry.py`
   cases ported one-to-one), BVH/grid vs brute force (`proptest`), ICP vs a naive f64
   implementation on random clouds, Umeyama vs known transforms, hash-grid tie rules.
2. **Slab fixture** (`fixtures/slab/`): the synthetic slab pair from `tests/test_synthetic.py`,
   generated once by the Python and committed (≈ 6 MB); the Rust tests reproduce
   `test_synthetic.py`'s assertions (pose error ≤ 2° / 0.1 t, segmentation bounds, acceptance).
3. **Kernel cross-checks**: every `Executor` method on random inputs and on the slab, CPU vs
   GPU, tolerances of §10.2; run on software adapters in CI.
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
| 4 object separation | `assembly/consistency.rs`: cycle consistency is evaluated for every candidate join against all paths in the group (the reference checks only direct alternatives); `Features` per fragment computed in preprocessing and stored in the cache: `shell_radius` (quadric fit on the shell samples), `colour_lab_mean/std` (from vertex colours when present; the PLY reader keeps them), `thick`; `GroupFeatures` = medians + MAD; `groups.rs` reports every group as an object with its consensus and flags a join whose fragment deviates > k·MAD | `Features` struct and cache slots (empty in phase 1), the PLY colour path |
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
| 1c | hypotheses, coarse, NMS, ICP (E5), verification, `match_pair`, screening flags | 2.5 | stage-2 injected tolerances on every fixture pair | ICP corner cases (empty correspondences), tie handling |
| 1d | assembly, refinement, recentre, report/transforms/meshes, renderer, CLI, determinism tests | 2 | R§13 gates natively; CI green on 4 OSs | none major |
| 1e | profiling and CPU tuning to §10.3 CPU gates | 1.5 | CPU gates | 2 h collection gate has 1.6× margin only |
| **phase 1 total** | | **11** | | |
| 2a | wgpu device/adapter/self-test, buffers, slots, batch structs; E7, E8 | 1.5 | self-test passes on Metal + lavapipe | naga/driver issues |
| 2b | hash grid + `icp_rung` (both estimators, in-kernel solves), `coarse_scores` | 2.5 | CPU/GPU cross-check within §10.2 | shared-memory limits; f32 conditioning |
| 2c | BVH kernels (`bounded_distance`, `inside`) | 1.5 | cross-check | traversal stack in WGSL |
| 2d | scheduler, pipelining, TDR chunking, memory management | 1.5 | synthetic 170 ≤ 30 min | overlap efficiency |
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
   numpy RNG replication (E6, ≈ 2 days, fragile)? (Recommendation: tolerance-based.)
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
