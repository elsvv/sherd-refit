// D §6.2's radius-bounded nearest neighbour over the hash grid — experiment E7 §5's kernel.
//
// One invocation per (pose, source point). The pose is applied, the query's cell is computed, the
// 27 neighbouring cells are visited in a fixed `(dx, dy, dz)` order, and the points of each cell
// are walked in ascending original index; the nearest inside the radius wins, ties go to the
// lowest index. That is D §6.2 verbatim and it is what
// `sherd_core::spatial::grid::HashGrid::search` mirrors, statement for statement.
//
// # The three rules E7 §4 leaves on this file
//
// 1. **No `dot`, `length`, `distance` or `normalize`.** `metal::dot` is a library function and
//    stays a fused chain regardless of the calling translation unit's contraction setting, so the
//    squared distance is written out as `dx*dx + dy*dy + dz*dz` and the transform as nine
//    multiplies and three adds — the same expressions, in the same order, as the CPU mirror.
// 2. **No subgroup reductions and no floating-point atomics.** Nothing here reduces, but the rule
//    is the file's, not the kernel's.
// 3. **The dispatch shape is data.** `params.wg_x` lets a 2-D dispatch recompute its linear
//    index; E7 §2 hit the 65 535-workgroup wall at 96 000 workgroups.
//
// E7 measured this kernel at 9.0–9.5 ns/query saturated, with 56 differing neighbour choices in
// 24 576 000 queries and `max |Δd| = 2.4e-7` on a unit cloud — the fast-math residue, 400× inside
// the tightest tolerance of D §10.2. Those are the numbers `selftest` holds it to.

struct GridHeader {
    // The cloud's minimum corner; `w` is `1/r`, so the kernel multiplies rather than divides
    // (Metal's division is 2 ULP, E7 §4.2).
    origin: vec4<f32>,
    cap: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

struct Params {
    // Poses in this dispatch.
    poses: u32,
    // Source points per pose.
    points: u32,
    // The search radius; the cell side is the same number.
    radius: f32,
    // Workgroups along x, for the 2-D fallback.
    wg_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> grid: GridHeader;
// `(ix, iy, iz, start)`; `start < 0` marks an empty slot.
@group(0) @binding(2) var<storage, read> slots: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<storage, read> sorted_idx: array<u32>;
// The target cloud, in grid order of index (i.e. the original order).
@group(0) @binding(5) var<storage, read> tgt_p: array<vec4<f32>>;
// The moving points, in their own frame.
@group(0) @binding(6) var<storage, read> source: array<vec4<f32>>;
// Row-major `[R | tau]`, three rows per pose.
@group(0) @binding(7) var<storage, read> poses: array<vec4<f32>>;
// `0xffffffff` for a miss, otherwise the target index.
@group(0) @binding(8) var<storage, read_write> found: array<u32>;
@group(0) @binding(9) var<storage, read_write> distance2: array<f32>;

const MISS: u32 = 0xffffffffu;

// D §6.2's cell hash, computed in `i32` with wrapping multiplication — WGSL's `*` on `i32` wraps,
// which is the same bit pattern the CPU mirror produces through `i64` and a mask.
fn hash_cell(c: vec3<i32>) -> u32 {
    let h = (c.x * 73856093) ^ (c.y * 19349663) ^ (c.z * 83492791);
    return u32(h) & (grid.cap - 1u);
}

// The slot holding `c`, or `cap` when the cell is empty. Linear probing, the build's own order.
fn find_slot(c: vec3<i32>) -> u32 {
    var slot = hash_cell(c);
    var probes = 0u;
    loop {
        if (probes >= grid.cap) {
            break;
        }
        let s = slots[slot];
        if (s.w < 0) {
            return grid.cap;
        }
        if (s.x == c.x && s.y == c.y && s.z == c.z) {
            return slot;
        }
        slot = (slot + 1u) & (grid.cap - 1u);
        probes = probes + 1u;
    }
    // naga rejects a function whose `loop` never falls through (E7 §7.4).
    return grid.cap;
}

@compute @workgroup_size(256)
fn nearest(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let group = wg.x + wg.y * params.wg_x;
    let id = group * 256u + lane;
    let total = params.poses * params.points;
    if (id >= total) {
        return;
    }
    let pose = id / params.points;
    let k = id - pose * params.points;

    let r0 = poses[pose * 3u + 0u];
    let r1 = poses[pose * 3u + 1u];
    let r2 = poses[pose * 3u + 2u];
    let p = source[k];
    // Written out: nine multiplies and three adds, in the CPU mirror's order.
    let q = vec3<f32>(
        r0.x * p.x + r0.y * p.y + r0.z * p.z + r0.w,
        r1.x * p.x + r1.y * p.y + r1.z * p.z + r1.w,
        r2.x * p.x + r2.y * p.y + r2.z * p.z + r2.w,
    );

    let inv_cell = grid.origin.w;
    let base = vec3<i32>(floor((q - grid.origin.xyz) * inv_cell));
    let r2sq = params.radius * params.radius;
    var best = MISS;
    var best_d2 = 3.4028235e38;

    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let cell = base + vec3<i32>(dx, dy, dz);
                let slot = find_slot(cell);
                if (slot >= grid.cap) {
                    continue;
                }
                let start = u32(slots[slot].w);
                let count = counts[slot];
                for (var m = 0u; m < count; m = m + 1u) {
                    let j = sorted_idx[start + m];
                    let t = tgt_p[j];
                    let ex = t.x - q.x;
                    let ey = t.y - q.y;
                    let ez = t.z - q.z;
                    let d2 = ex * ex + ey * ey + ez * ez;
                    // `<=` is D §6.2's inclusive rule; the exclusive bound of R §5.2 and R §7 is
                    // applied by the caller, which is where the two conventions differ.
                    if (d2 <= r2sq && (d2 < best_d2 || (d2 == best_d2 && j < best))) {
                        best = j;
                        best_d2 = d2;
                    }
                }
            }
        }
    }
    found[id] = best;
    if (best == MISS) {
        distance2[id] = -1.0;
    } else {
        distance2[id] = best_d2;
    }
}
