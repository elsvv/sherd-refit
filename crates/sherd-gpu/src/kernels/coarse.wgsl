// R §5.2's coarse score and R §5.4's re-score — D §6.4 step 2, D §6.5's `coarse_main`.
//
// One workgroup per hypothesis, 64 lanes striding over the probe. Lane `l` handles probe points
// `l, l + 64, …`: it moves the point by the pose, asks the hash grid for the nearest breakline
// point strictly inside the radius, turns the point's shell macro-normal by the same rotation and
// counts the pair when the two normals agree above `normal_agree`. The workgroup then folds its 64
// counts in a fixed-order tree and lane 0 writes the hypothesis' **integer count**.
//
// # Why the kernel returns a count and not a score
//
// The CPU executor's score is `f64::from(agree) / n_points` — a division, in `f64`, and step C1
// measured that `k · (1/60)` is not `k / 60` for every `k`. So the device computes the only part
// that is genuinely parallel, the count, and the host does that same `f64` division on the way
// out. Whenever the two counts agree the two **scores are bit-identical**, and the cross-check is
// then a comparison of integers rather than an argument about the last bit of a mean.
//
// # The near mask is not here, and that is a semantic claim
//
// `CoarseBatch::mask` is E2's conservative pre-filter: `may_be_near` returns `false` only when no
// breakline point is within the radius, so a caller that falls through on `true` computes exactly
// what it computes without the mask (`spatial::grid`'s own module documentation, and the test that
// asserts it on 60 000 queries). It is a speed structure for a KD-tree descent, and the grid's own
// answer to an empty neighbourhood is 27 hash probes, which is the same fast path by construction.
// Running it on the device would add a second `f32` rounding of a bound that is conservative in
// `f64` — a new way to disagree, bought for nothing. The kernel therefore ignores the mask and the
// note carries the measured cost of doing so.
//
// # Reduction
//
// The tally is `u32` and integer addition is exact and associative, so the tree cannot move a
// result whatever its order. It is written in the fixed 64 → 32 → … → 1 order anyway, because
// D §6.7's rule is the file's rather than the arithmetic's, and because `icp.wgsl`'s tree — which
// is `f32` and where the order does decide the bits — is the same code with a different type.

struct Params {
    // Poses in this dispatch.
    poses: u32,
    // Probe points per pose.
    points: u32,
    // The correspondence radius: `sc.coarse` for R §5.2, `sc.stage1` for R §5.4.
    radius: f32,
    // R §5.2's 0.7.
    normal_agree: f32,
    // Workgroups along x, for the 2-D fallback past 65 535 (E7 §2).
    wg_x: u32,
    // Where this chunk's first pose sits in the *output* array. The pose buffer itself holds
    // only this chunk, so a batch whose poses exceed one binding is uploaded in pieces while the
    // scores stay in one array at their own indices.
    first_pose: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Bindings 1–5 are `grid.wgsl`'s: the header, the three grid arrays and the interleaved target
// cloud (`2j` a breakline point, `2j + 1` its shell macro-normal).
// The probe, interleaved the same way: `2k` a point in B's frame, `2k + 1` its shell normal.
@group(0) @binding(6) var<storage, read> probe: array<vec4<f32>>;
// Row-major `[R | tau]`, three `vec4<f32>` per pose.
@group(0) @binding(7) var<storage, read> poses: array<vec4<f32>>;
// One agreement count per pose, written at the pose's own index.
@group(0) @binding(8) var<storage, read_write> agree_out: array<u32>;

const LANES: u32 = 64u;

var<workgroup> tally: array<u32, 64>;

@compute @workgroup_size(64)
fn coarse(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let group = wg.x + wg.y * params.wg_x;
    var agree = 0u;
    if (group < params.poses) {
        let r0 = poses[group * 3u + 0u];
        let r1 = poses[group * 3u + 1u];
        let r2 = poses[group * 3u + 2u];
        let radius2 = params.radius * params.radius;
        var k = lane;
        loop {
            if (k >= params.points) {
                break;
            }
            let p = probe[k * 2u];
            // Nine multiplies and three adds, written out in the CPU mirror's order.
            let q = vec3<f32>(
                r0.x * p.x + r0.y * p.y + r0.z * p.z + r0.w,
                r1.x * p.x + r1.y * p.y + r1.z * p.z + r1.w,
                r2.x * p.x + r2.y * p.y + r2.z * p.z + r2.w,
            );
            let hit = nearest_below(q, radius2);
            if (hit.x != MISS) {
                let n = probe[k * 2u + 1u];
                let turned = vec3<f32>(
                    r0.x * n.x + r0.y * n.y + r0.z * n.z,
                    r1.x * n.x + r1.y * n.y + r1.z * n.z,
                    r2.x * n.x + r2.y * n.y + r2.z * n.z,
                );
                let theirs = cloud[hit.x * 2u + 1u];
                let d = theirs.x * turned.x + theirs.y * turned.y + theirs.z * turned.z;
                if (d > params.normal_agree) {
                    agree = agree + 1u;
                }
            }
            k = k + LANES;
        }
    }
    tally[lane] = agree;
    workgroupBarrier();

    var stride = LANES >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            tally[lane] = tally[lane] + tally[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lane == 0u && group < params.poses) {
        agree_out[params.first_pose + group] = tally[0];
    }
}
