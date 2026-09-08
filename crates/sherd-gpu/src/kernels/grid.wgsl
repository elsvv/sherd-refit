// D §6.2's hash grid, as every kernel of this crate queries it.
//
// This file is not a kernel. It is prepended to `coarse.wgsl` and `icp.wgsl` at build time
// (`concat!(include_str!(...), ...)`, since WGSL has no include), so that the grid traversal — the
// one piece of arithmetic both of them share with `sherd_core::spatial::grid::HashGrid::search` —
// is written once and cannot drift between the two.
//
// **The binding numbers below are a contract.** Every kernel that includes this file binds
//
//   0  its own `Params` uniform
//   1  `GridHeader`
//   2  `slots`      `array<vec4<i32>>`, `(ix, iy, iz, start)`, `start < 0` for an empty slot
//   3  `counts`     `array<u32>`
//   4  `sorted_idx` `array<u32>`
//   5  `cloud`      `array<vec4<f32>>`, **interleaved**: `2j` is target point `j`, `2j + 1` is its
//                   normal. Interleaving is the binding budget, not a micro-optimisation: a
//                   portable compute stage has eight storage buffers (D §6.8) and the coarse
//                   kernel needs nine arrays as separate bindings.
//
// and is free to use 6, 7, 8 and 9 for its own.
//
// # The three rules E7 §4 leaves on every parity-critical file
//
// 1. **No `dot`, `length`, `distance` or `normalize`.** `metal::dot` is a library function and
//    stays a fused chain whatever the calling unit's contraction setting, so the squared distance
//    is written out as `dx*dx + dy*dy + dz*dz` — the same expression, in the same order, as
//    `HashGrid::search`.
// 2. **No subgroup reductions and no floating-point atomics.** Neither has a defined order.
// 3. **The dispatch shape is data** (`params.wg_x`), never a constant on one side and a divide on
//    the other.

struct GridHeader {
    // The cloud's minimum corner; `w` is `1/r`, so the kernel multiplies rather than divides
    // (Metal's division is 2 ULP, E7 §4.2) and the reciprocal is the host's own number.
    origin: vec4<f32>,
    cap: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(1) var<uniform> grid: GridHeader;
@group(0) @binding(2) var<storage, read> slots: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<storage, read> sorted_idx: array<u32>;
@group(0) @binding(5) var<storage, read> cloud: array<vec4<f32>>;

const MISS: u32 = 0xffffffffu;
const F32_MAX: f32 = 3.4028235e38;

// D §6.2's cell hash, computed in `i32` with wrapping multiplication — the same bit pattern the
// CPU mirror produces through `i64` and a mask.
fn hash_cell(c: vec3<i32>) -> u32 {
    let h = (c.x * 73856093) ^ (c.y * 19349663) ^ (c.z * 83492791);
    return u32(h) & (grid.cap - 1u);
}

// The slot holding `c`, or `cap` when the cell is empty. Linear probing in the build's own order.
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

// The cell `q` falls in: `floor((q − origin) · (1/r))`, exactly as `grid::cell_of` computes it.
fn cell_of(q: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor((q - grid.origin.xyz) * grid.origin.w));
}

// The nearest cloud point to `q` with `d² < r²`, or `MISS`.
//
// `HashGrid::nearest_below` line for line: the 27 neighbouring cells in `(dx, dy, dz)` order with
// `x` outermost, the points of each cell in ascending original index, `<` on the squared distance
// and the lowest index on an exact tie. `d²` is returned through `out_d2`, which is what the ICP
// kernel accumulates as its `error²` and what the cross-check compares.
//
// R §5.2's bound is scipy's `distance_upper_bound` and R §7's is FLANN's, and both are strict, so
// there is deliberately no inclusive variant here: the one caller that wants `d ≤ r` is D §6.2's
// own statement of the rule and no kernel of this crate has it.
fn nearest_below(q: vec3<f32>, r2: f32) -> vec2<u32> {
    let base = cell_of(q);
    var best = MISS;
    var best_d2 = F32_MAX;
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let slot = find_slot(base + vec3<i32>(dx, dy, dz));
                if (slot >= grid.cap) {
                    continue;
                }
                let start = u32(slots[slot].w);
                let count = counts[slot];
                for (var m = 0u; m < count; m = m + 1u) {
                    let j = sorted_idx[start + m];
                    let t = cloud[j * 2u];
                    let ex = t.x - q.x;
                    let ey = t.y - q.y;
                    let ez = t.z - q.z;
                    let d2 = ex * ex + ey * ey + ez * ez;
                    if (d2 < r2 && (d2 < best_d2 || (d2 == best_d2 && j < best))) {
                        best = j;
                        best_d2 = d2;
                    }
                }
            }
        }
    }
    // `bitcast` rather than a second output buffer: the pair travels in two registers and the
    // caller reads whichever half it needs.
    return vec2<u32>(best, bitcast<u32>(best_d2));
}
