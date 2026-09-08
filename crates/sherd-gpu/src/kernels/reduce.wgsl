// D §6.4's reduction schedule, and experiment E7 §3's pass criterion.
//
// 256 workgroups of 256 lanes. Lane `l` of workgroup `w` accumulates
// `a[w·block + l], a[w·block + l + 256], …` sequentially into a register, then the workgroup folds
// its 256 partials in a shared-memory tree 256 → 128 → … → 1. The host runs the same kernel a
// second time over the 256 partials.
//
// This is the *exact* striding and tree D §6.4 and D §6.7 specify, and E7 §3 measured it
// bit-identical to a single-threaded Rust mirror over 10 000 000 terms, in stock wgpu, twice
// (`0xcb01367b` both times). It is addition only: no multiply, no `dot`, no library call, so
// nothing here can be contracted into an FMA (E7 §4.2), and nothing is reassociated.
//
// The shape is *data*, not a derived constant — `params.workgroups`, `params.block` and
// `params.n` all come from the batch descriptor — because the CPU mirror has to agree on the
// workgroup count, the lane stride and the loop bound, and a constant on one side and a divide on
// the other is exactly how that drifts (E7 §3).

struct Params {
    // Terms in the input array.
    n: u32,
    // Terms one workgroup covers: `ceil(n / workgroups)`, rounded up to a multiple of nothing —
    // the loop bound handles the ragged tail.
    block: u32,
    // Workgroups this dispatch runs.
    workgroups: u32,
    // Workgroups along x, so a 2-D dispatch can recompute its linear index (E7 §2).
    wg_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> terms: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;

// 256 lanes × 4 bytes = 1 KiB, far inside the 16 KiB default (D §6.8).
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let group = wg.x + wg.y * params.wg_x;
    var acc: f32 = 0.0;
    if (group < params.workgroups) {
        let start = group * params.block;
        var end = start + params.block;
        if (end > params.n) {
            end = params.n;
        }
        // The strided accumulate of D §6.4: lane `l`, then `l + 256`, then `l + 512`, …
        var i = start + lane;
        loop {
            if (i >= end) {
                break;
            }
            acc = acc + terms[i];
            i = i + 256u;
        }
    }
    scratch[lane] = acc;
    workgroupBarrier();

    // The tree of D §6.7: 256 → 128 → 64 → … → 1, every level a plain addition of two lanes whose
    // indices are fixed. `stride` is a loop variable rather than an unrolled constant so that the
    // Rust mirror can be written from this text.
    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lane == 0u && group < params.workgroups) {
        partials[group] = scratch[0];
    }
    // naga rejects a function whose `loop` never falls through (E7 §7.4); this one does, so no
    // dead `return` is needed here.
}
