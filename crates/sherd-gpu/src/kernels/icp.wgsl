// One rung of R §7's ICP for a batch of candidates — D §6.4 steps 4 and 5, D §6.5's second sketch.
//
// One workgroup of 256 lanes per candidate, **every iteration of the rung inside the kernel**: one
// dispatch and one readback for a whole rung, which is what D §6.4 asks for and what makes the
// GPU path worth having at all (a rung is 20 or 30 iterations and a per-iteration round trip would
// be 30 submissions of 0.5 ms each).
//
// Two entry points, R §7's two estimators:
//
//   `point_to_point`  R §5.4's rungs: Eigen's `umeyama` without scaling, over the correspondence
//                     pairs. Two passes per iteration (the means, then the covariance) and a 3×3
//                     SVD by one-sided Jacobi on invocation 0.
//   `point_to_plane`  R §5.6's rungs: the 6×6 normal equations, an LDLT with Eigen's pivot rule
//                     and R §7's Euler composition, all on invocation 0.
//
// # The frame, and why it is not `Assembly::Centred`
//
// Both clouds arrive **translated**: the source by its own centroid `c_s`, the target by its own
// `c_t`, both computed in `f64` on the host. That is a rigid change of frame, and D §7's last row
// is explicit that it is the right way to shrink coordinates for an `f32` path — the scans sit
// 100–150 units from the origin and `f32` there has 1e-5 of resolution, while a fragment's own
// extent is tens of units.
//
// It is **not** free, and it is not `Assembly::Centred`. The linear Gauss–Newton step is an exact
// reparameterisation under the shift (`ω` is unchanged and `τ_world = τ' − ω × c`), but the finite
// update is not: composing `p ↦ R p + τ'` in the shifted frame is conjugate to composing
// `p ↦ R p + τ' + (R − I − ω̂)c` in the world frame, and D §7 measured that `O(|ω|²·|c|)` residue
// at **0.044 t at the median and 0.61 t at p90** on terracotta — outside D §10.2's 0.01 t row.
// So the kernel computes the residue and adds it: `point_to_plane`'s update is
//
//     τ_update = τ' + (R − I − ω̂)·c_t
//
// which is the world composition exactly, and is *better* conditioned than computing `R c − c`
// because `R − I − ω̂` is formed from `O(1)` entries before it ever meets `c`.
//
// `point_to_point` needs no such correction: Umeyama's translation is `mean_q − R·mean_p`, and
// shifting both clouds shifts the two means with them, so the shifted update is already the
// conjugate of the world one. That asymmetry is the whole content of D §7's "R §7 is equivariant
// under a rigid change of frame" — it is, for the estimator that reads its translation off the
// data, and it is not for the one that reads it off a linearisation about the origin.
//
// # The reduction
//
// 29 accumulators per lane (21 of `JTJ`'s lower triangle, 6 of `JTr`, the correspondence count and
// the squared error), folded eight at a time in a fixed-order 256 → 128 → … → 1 tree. Eight at a
// time because 32 × 256 × 4 B of workgroup storage is 32 KB and the portable budget is 16 KB
// (D §6.8); eight fits in 8 KB with room for the pose and the candidate state. The tree is the one
// `reduce.wgsl` uses and E7 §3 measured bit-identical to its Rust mirror — but here the terms are
// products, so the CPU's `f64` sum in source order and this `f32` sum in tree order are two
// different numbers, and D §10.2's tolerances are what govern (D §6.7).
//
// # Why every candidate runs `max_iter` times
//
// R §7 stops a candidate on its own convergence test and never iterates it again (D §7), and so
// does this kernel: `done` is set on invocation 0 and every later iteration skips the point loop
// and the update. What it cannot do is *leave* the loop, because the loop carries
// `workgroupBarrier()` and WGSL requires a barrier to sit in uniform control flow — a value read
// from workgroup storage is not uniform to the analysis, however uniform it is in fact. So the
// loop bound is `params.max_iter`, from a uniform buffer, and a converged candidate pays the
// barriers of the remaining iterations and none of the work.

struct Params {
    // The target centroid in world coordinates: the `c` of the correction above, and the shift the
    // host applied to the target cloud.
    centre: vec4<f32>,
    // Candidates in this dispatch.
    candidates: u32,
    // Source points, the same for every candidate of the batch.
    n_src: u32,
    // R §7's `max_correspondence_distance`.
    radius: f32,
    // `ICPConvergenceCriteria::max_iteration_`.
    max_iter: u32,
    // Workgroups along x, for the 2-D fallback past 65 535 (E7 §2).
    wg_x: u32,
    // Where this chunk's first candidate sits in the output array.
    first: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Bindings 1–5 are `grid.wgsl`'s: the header, the three grid arrays, and the interleaved target
// cloud (`2j` a point, `2j + 1` its normal), all in the shifted frame.
// The source cloud, shifted by its own centroid; `w` unused.
@group(0) @binding(6) var<storage, read> source: array<vec4<f32>>;
// Per candidate, seventeen words: twelve of the pose `[R | τ]` row-major (in and out), then the
// correspondence count, the squared error, the iterations applied, the convergence flag and the
// **nonce** — word 16, which the host writes and this kernel must give back as `nonce + 1`.
//
// The nonce is task H1's receipt (`notes/2026-09-09-h1-wd1.md`). A command buffer the driver
// aborts leaves the state buffer holding what the host uploaded and the staging buffer holding
// what it held before, and neither the fence nor `map_async` says so, so the host has no other way
// to tell "this block is the answer to my batch" from "this block is the answer to somebody
// else's, or to none". The host's nonce is an exact `f32` integer below 2^23, so `nonce + 1.0` is
// exact and one comparison decides it.
@group(0) @binding(7) var<storage, read_write> state: array<f32>;
// One target index per (candidate, source point), or `MISS`. Only `point_to_point` reads it back,
// in its second pass; writing it in both keeps one code path for the correspondence search.
@group(0) @binding(8) var<storage, read_write> corres: array<u32>;

const LANES: u32 = 256u;
const STATE_WORDS: u32 = 17u;
// Open3D's `ICPConvergenceCriteria` (R §7).
const RELATIVE_FITNESS: f32 = 1e-6;
const RELATIVE_RMSE: f32 = 1e-6;
// `f32::MIN_POSITIVE`, the pivot floor Eigen's pseudo-inverse of `D` uses.
const MIN_POSITIVE: f32 = 1.17549435e-38;

// Eight components of 256 lanes: 8 KB, half the portable workgroup budget (D §6.8).
var<workgroup> scratch: array<f32, 2048>;
// The reduced accumulators, 29 used of 32.
var<workgroup> total: array<f32, 32>;
// The candidate's pose, row-major `[R | τ]`, in the shifted frame.
var<workgroup> pose: array<f32, 12>;
var<workgroup> fitness: f32;
var<workgroup> rmse: f32;
var<workgroup> done: u32;
var<workgroup> applied: u32;
var<workgroup> corres_count: u32;
var<workgroup> error2: f32;
// The per-call nonce this candidate arrived with (word 16), given back incremented.
var<workgroup> nonce: f32;

// One fixed-order fold of eight per-lane values into `total[base .. base + 8]`.
//
// The tree is `reduce.wgsl`'s: 256 → 128 → … → 1, every level a plain addition of two lanes whose
// indices are fixed. Every lane of the workgroup reaches this function on every call, which is
// what makes its barriers legal.
fn reduce8(lane: u32, base: u32, v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) {
    scratch[0u * LANES + lane] = v0;
    scratch[1u * LANES + lane] = v1;
    scratch[2u * LANES + lane] = v2;
    scratch[3u * LANES + lane] = v3;
    scratch[4u * LANES + lane] = v4;
    scratch[5u * LANES + lane] = v5;
    scratch[6u * LANES + lane] = v6;
    scratch[7u * LANES + lane] = v7;
    workgroupBarrier();
    var stride = LANES >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            for (var k = 0u; k < 8u; k = k + 1u) {
                let at = k * LANES + lane;
                scratch[at] = scratch[at] + scratch[at + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lane == 0u) {
        for (var k = 0u; k < 8u; k = k + 1u) {
            total[base + k] = scratch[k * LANES];
        }
    }
    workgroupBarrier();
}

// The source point `i` under the workgroup's current pose, in the shifted target frame.
//
// Nine multiplies and three adds, written out. D §6.5 recomputes the moved point from the
// accumulated pose rather than carrying a moved cloud, which is one rounding instead of an
// accumulated chain of them.
fn moved(i: u32) -> vec3<f32> {
    let p = source[i];
    return vec3<f32>(
        pose[0] * p.x + pose[1] * p.y + pose[2] * p.z + pose[3],
        pose[4] * p.x + pose[5] * p.y + pose[6] * p.z + pose[7],
        pose[8] * p.x + pose[9] * p.y + pose[10] * p.z + pose[11],
    );
}

// `T ← U · T` for the workgroup's pose, with `U = [rot | tau]`. Invocation 0 only.
fn compose(rot: mat3x3<f32>, tau: vec3<f32>) {
    var out: array<f32, 12>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 4u; c = c + 1u) {
            // `rot` is column-major in WGSL: `rot[c][r]` is row `r`, column `c`.
            var v = rot[0u][r] * pose[0u * 4u + c]
                  + rot[1u][r] * pose[1u * 4u + c]
                  + rot[2u][r] * pose[2u * 4u + c];
            if (c == 3u) {
                v = v + tau[r];
            }
            out[r * 4u + c] = v;
        }
    }
    for (var k = 0u; k < 12u; k = k + 1u) {
        pose[k] = out[k];
    }
}

// R §7's `Rz(γ)·Ry(β)·Rx(α)`, as the matrix product R §7 writes rather than Eigen's quaternion
// chain (R §12.1 carries the row that licenses the substitution: 1.5 ULP of 1).
fn euler_zyx(a: f32, b: f32, g: f32) -> mat3x3<f32> {
    let sa = sin(a);
    let ca = cos(a);
    let sb = sin(b);
    let cb = cos(b);
    let sg = sin(g);
    let cg = cos(g);
    // Written out row by row; `mat3x3` takes columns.
    let r00 = cg * cb;
    let r01 = cg * sb * sa - sg * ca;
    let r02 = cg * sb * ca + sg * sa;
    let r10 = sg * cb;
    let r11 = sg * sb * sa + cg * ca;
    let r12 = sg * sb * ca - cg * sa;
    let r20 = -sb;
    let r21 = cb * sa;
    let r22 = cb * ca;
    return mat3x3<f32>(
        vec3<f32>(r00, r10, r20),
        vec3<f32>(r01, r11, r21),
        vec3<f32>(r02, r12, r22),
    );
}

// Eigen's pivot order for a symmetric 6×6 `LDLT`: a selection sort **by transposition** on the
// original diagonal, keeping the first of several equal maxima (`maxCoeff` tests `value > res`).
// `sherd_core::matching::icp::eigen_pivots`, transcribed.
fn eigen_pivots(a: ptr<function, array<f32, 36>>, perm: ptr<function, array<u32, 6>>) {
    for (var k = 0u; k < 6u; k = k + 1u) {
        (*perm)[k] = k;
    }
    for (var k = 0u; k < 6u; k = k + 1u) {
        var best = k;
        for (var i = k + 1u; i < 6u; i = i + 1u) {
            let pi = (*perm)[i];
            let pb = (*perm)[best];
            if (abs((*a)[pi * 6u + pi]) > abs((*a)[pb * 6u + pb])) {
                best = i;
            }
        }
        let tmp = (*perm)[k];
        (*perm)[k] = (*perm)[best];
        (*perm)[best] = tmp;
    }
}

// Eigen's `LDLT` on a symmetric 6×6, in `f32` — the solver Open3D's point-to-plane step uses, as
// `icp::solve_ldlt` reproduces it: the permutation up front, the plain unpivoted factorisation on
// the permuted matrix, and Eigen's pseudo-inverse of `D` (a pivot at or below the smallest normal
// float is set to zero rather than divided by, which is what keeps a rank-deficient system — two
// flat surfaces have three unconstrained degrees of freedom — from returning infinities).
//
// Returns `false` when the answer is not finite, which is the caller's cue for an identity update.
fn solve_ldlt(a: ptr<function, array<f32, 36>>, b: ptr<function, array<f32, 6>>, x: ptr<function, array<f32, 6>>) -> bool {
    // **Equilibration, and why this one deviation from the CPU's solve is here.**
    //
    // `JTJ`'s rotational block is `(p × n)ᵀ(p × n)` and its translational block is `nᵀn`, so with
    // the clouds shifted to their centroids the two differ by `|p|² ≈ 10³` — a condition number
    // the geometry never had, bought entirely by the choice of units. In `f64` that costs three of
    // sixteen digits and nobody notices; in `f32` it costs three of seven. Scaling by
    // `D = diag(1/√A_kk)` and solving `(D A D) y = D b`, `x = D y`, is an exact reparameterisation
    // — the same `x` in exact arithmetic — and it hands the factorisation a matrix with a unit
    // diagonal. Measured on terracotta's first two pairs, over the four stage-2 rungs: `p90` of
    // the pose deviation from the CPU rung moves from **6.5e-2° to 7.1e-4°** and of the
    // displacement from **4.8e-3 t to 7.8e-5 t**, while the median barely moves (3.5e-4° to
    // 3.3e-4°) — which is the signature of a conditioning fix rather than a change of answer.
    //
    // It does change which permutation Eigen's rule picks (every diagonal entry is 1 afterwards,
    // so the rule keeps the identity), and that is a real difference from `icp::solve_ldlt`. It is
    // the right one: the two orders factorise the same matrix and D §10.2's tolerances are what
    // this path is held to, so the version that lands closer to the `f64` answer is the version to
    // run. A row with a non-positive diagonal — which a rank-deficient system has — keeps its
    // scale at one and falls through to Eigen's pseudo-inverse of `D` as before.
    var scale: array<f32, 6>;
    for (var k = 0u; k < 6u; k = k + 1u) {
        let d = (*a)[k * 6u + k];
        if (d > 0.0) {
            scale[k] = 1.0 / sqrt(d);
        } else {
            scale[k] = 1.0;
        }
    }
    for (var k = 0u; k < 6u; k = k + 1u) {
        for (var l = 0u; l < 6u; l = l + 1u) {
            (*a)[k * 6u + l] = (*a)[k * 6u + l] * scale[k] * scale[l];
        }
        (*b)[k] = (*b)[k] * scale[k];
    }
    var perm: array<u32, 6>;
    eigen_pivots(a, &perm);
    var m: array<f32, 36>;
    for (var k = 0u; k < 6u; k = k + 1u) {
        for (var l = 0u; l < 6u; l = l + 1u) {
            m[k * 6u + l] = (*a)[perm[k] * 6u + perm[l]];
        }
    }
    for (var k = 0u; k < 6u; k = k + 1u) {
        if (k > 0u) {
            var temp: array<f32, 6>;
            for (var j = 0u; j < k; j = j + 1u) {
                temp[j] = m[j * 6u + j] * m[k * 6u + j];
            }
            var diagonal = m[k * 6u + k];
            for (var j = 0u; j < k; j = j + 1u) {
                diagonal = diagonal - m[k * 6u + j] * temp[j];
            }
            m[k * 6u + k] = diagonal;
            for (var row = k + 1u; row < 6u; row = row + 1u) {
                var below = m[row * 6u + k];
                for (var j = 0u; j < k; j = j + 1u) {
                    below = below - m[row * 6u + j] * temp[j];
                }
                m[row * 6u + k] = below;
            }
        }
        let pivot = m[k * 6u + k];
        if (pivot != 0.0) {
            for (var row = k + 1u; row < 6u; row = row + 1u) {
                m[row * 6u + k] = m[row * 6u + k] / pivot;
            }
        }
    }
    var y: array<f32, 6>;
    for (var k = 0u; k < 6u; k = k + 1u) {
        y[k] = (*b)[perm[k]];
    }
    for (var i = 0u; i < 6u; i = i + 1u) {
        var value = y[i];
        for (var j = 0u; j < i; j = j + 1u) {
            value = value - m[i * 6u + j] * y[j];
        }
        y[i] = value;
    }
    for (var i = 0u; i < 6u; i = i + 1u) {
        let pivot = m[i * 6u + i];
        if (abs(pivot) > MIN_POSITIVE) {
            y[i] = y[i] / pivot;
        } else {
            y[i] = 0.0;
        }
    }
    for (var k = 0u; k < 6u; k = k + 1u) {
        let i = 5u - k;
        var value = y[i];
        for (var j = i + 1u; j < 6u; j = j + 1u) {
            value = value - m[j * 6u + i] * y[j];
        }
        y[i] = value;
    }
    var finite = true;
    for (var k = 0u; k < 6u; k = k + 1u) {
        let v = y[k] * scale[perm[k]];
        (*x)[perm[k]] = v;
        // A NaN fails every comparison; an infinity fails the magnitude test.
        if (!(abs(v) < 3.4028235e38)) {
            finite = false;
        }
    }
    return finite;
}

// One-sided Jacobi SVD of a 3×3, at most `sweeps` sweeps of the three column pairs (D §6.4).
//
// The columns of `a` are orthogonalised in place; `v` accumulates the rotations, so
// `a_in = a_out · diag(1/s) · … ` — on return `a` holds `U · S` column by column, `v` holds `V`,
// and the caller reads the singular values off the column norms. This is not the algorithm
// `nalgebra` runs on the CPU side (Golub–Reinsch), and it does not need to be: what leaves
// `umeyama` is `U · d · Vᵀ`, and the two agree wherever the problem is well posed. Where it is not
// — a covariance of six correspondences on a curve, which C2 §5 measured as the chaotic case —
// they will not, and D §10.2's distribution rows are where that shows.
fn jacobi_svd(a: ptr<function, array<vec3<f32>, 3>>, v: ptr<function, array<vec3<f32>, 3>>, sweeps: u32) {
    (*v)[0] = vec3<f32>(1.0, 0.0, 0.0);
    (*v)[1] = vec3<f32>(0.0, 1.0, 0.0);
    (*v)[2] = vec3<f32>(0.0, 0.0, 1.0);
    for (var sweep = 0u; sweep < sweeps; sweep = sweep + 1u) {
        var off = 0.0;
        for (var p = 0u; p < 2u; p = p + 1u) {
            for (var q = p + 1u; q < 3u; q = q + 1u) {
                let cp = (*a)[p];
                let cq = (*a)[q];
                let alpha = cp.x * cp.x + cp.y * cp.y + cp.z * cp.z;
                let beta = cq.x * cq.x + cq.y * cq.y + cq.z * cq.z;
                let gamma = cp.x * cq.x + cp.y * cq.y + cp.z * cq.z;
                off = off + abs(gamma);
                if (gamma == 0.0 || abs(gamma) <= 1e-12 * sqrt(alpha * beta)) {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                var t = 1.0 / (abs(zeta) + sqrt(1.0 + zeta * zeta));
                if (zeta < 0.0) {
                    t = -t;
                }
                let c = 1.0 / sqrt(1.0 + t * t);
                let s = c * t;
                (*a)[p] = c * cp - s * cq;
                (*a)[q] = s * cp + c * cq;
                let vp = (*v)[p];
                let vq = (*v)[q];
                (*v)[p] = c * vp - s * vq;
                (*v)[q] = s * vp + c * vq;
            }
        }
        if (off == 0.0) {
            break;
        }
    }
}

// `a × b`, written out rather than through `cross`, which is a multiply-subtract chain the
// compiler is free to contract (E7 §4.2).
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// The determinant of a basis given by its three columns, written out.
fn det3(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> f32 {
    return c0.x * (c1.y * c2.z - c1.z * c2.y)
         - c1.x * (c0.y * c2.z - c0.z * c2.y)
         + c2.x * (c0.y * c1.z - c0.z * c1.y);
}

// Eigen's `umeyama(src, dst, with_scaling = false)` rotation from the 3×3 covariance `sigma`,
// given column by column.
//
// `SVD::new` sorts the singular values descending, which is what puts the reflection fix on the
// smallest of them; the sort here is the same selection sort over three columns. A singular value
// at zero leaves its `U` column undefined, and it is completed to the cross product of the other
// two rather than left at zero — an orthonormal completion, which is what keeps the result a
// rotation when the covariance is rank-deficient.
fn umeyama_rotation(s0: vec3<f32>, s1: vec3<f32>, s2: vec3<f32>) -> mat3x3<f32> {
    var a = array<vec3<f32>, 3>(s0, s1, s2);
    var v: array<vec3<f32>, 3>;
    jacobi_svd(&a, &v, 12u);
    var s: array<f32, 3>;
    var u: array<vec3<f32>, 3>;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let col = a[k];
        let norm = sqrt(col.x * col.x + col.y * col.y + col.z * col.z);
        s[k] = norm;
        if (norm > 0.0) {
            u[k] = col / norm;
        } else {
            u[k] = vec3<f32>(0.0, 0.0, 0.0);
        }
    }
    for (var i = 0u; i < 2u; i = i + 1u) {
        var best = i;
        for (var j = i + 1u; j < 3u; j = j + 1u) {
            if (s[j] > s[best]) {
                best = j;
            }
        }
        if (best != i) {
            let ts = s[i];
            s[i] = s[best];
            s[best] = ts;
            let tu = u[i];
            u[i] = u[best];
            u[best] = tu;
            let tv = v[i];
            v[i] = v[best];
            v[best] = tv;
        }
    }
    if (s[2] <= 0.0) {
        u[2] = cross3(u[0], u[1]);
        v[2] = cross3(v[0], v[1]);
    }
    if (s[1] <= 0.0) {
        u[1] = cross3(u[2], u[0]);
        v[1] = cross3(v[2], v[0]);
    }
    var d2 = 1.0;
    if (det3(u[0], u[1], u[2]) * det3(v[0], v[1], v[2]) < 0.0) {
        d2 = -1.0;
    }
    // `U · diag(1, 1, d2) · Vᵀ`: entry `(i, j)` is `Σ_k U[k][i] · d_k · V[k][j]`, so column `j` of
    // the result is `U[0]·V[0][j] + U[1]·V[1][j] + d2·U[2]·V[2][j]`.
    return mat3x3<f32>(
        u[0] * v[0].x + u[1] * v[1].x + d2 * u[2] * v[2].x,
        u[0] * v[0].y + u[1] * v[1].y + d2 * u[2] * v[2].y,
        u[0] * v[0].z + u[1] * v[1].z + d2 * u[2] * v[2].z,
    );
}

// Loads the candidate's pose and zeroes the per-candidate state. Invocation 0 only.
fn load_state(candidate: u32) {
    let base = candidate * STATE_WORDS;
    for (var k = 0u; k < 12u; k = k + 1u) {
        pose[k] = state[base + k];
    }
    nonce = state[base + 16u];
    fitness = 0.0;
    rmse = 0.0;
    done = 0u;
    applied = 0u;
    corres_count = 0u;
    error2 = 0.0;
}

// Writes the pose and the four result words back. Invocation 0 only.
fn store_state(candidate: u32) {
    let base = candidate * STATE_WORDS;
    for (var k = 0u; k < 12u; k = k + 1u) {
        state[base + k] = pose[k];
    }
    state[base + 12u] = f32(corres_count);
    state[base + 13u] = error2;
    state[base + 14u] = f32(applied);
    state[base + 15u] = f32(done);
    state[base + 16u] = nonce + 1.0;
}

// R §7's `corr`: the nearest target strictly inside the radius of each source point, its index
// stored for a second pass. Returns `(count, error²)` for this lane's stride.
fn search(lane: u32, candidate: u32) -> vec2<f32> {
    let r2 = params.radius * params.radius;
    let base = candidate * params.n_src;
    var count = 0.0;
    var err2 = 0.0;
    // Kahan compensation, and it is here for one measured reason: R §7's convergence test is
    // `|Δrmse| < 1e-6`, and a plain `f32` sum of forty-odd positive terms per lane carries about
    // `1.6e-6` of relative error — *above* the threshold. The rung then stops as soon as the pose
    // moves less than the summation noise, which is earlier than the `f64` rung stops, and it ends
    // somewhere else. Measured on pot_A's stage 2 before this: 30 of 40 candidates differed in
    // their iteration count, by 16 of 30 at p90.
    //
    // **On Metal this compensation is folded away, and the code stays for the other backends.**
    // `(next - err2) - term` is algebraically zero, and fast math is entitled to say so. E7 §3
    // measured this compiler leaving a fixed-order addition *tree* bit-identical to its Rust
    // mirror — it does not reassociate — but that is a different licence from an algebraic
    // identity, and this one it takes: every row of `gpu-check --set input/sfspp/pot_A --stage icp`
    // came back identical to the last printed digit with the compensation added, including
    // `icp s1 rmse`'s `p50` of 9.758e-8, which is exactly the quantity it would have moved.
    //
    // It is kept because it is correct WGSL and costs three flops, and a Vulkan or DX12 backend
    // that compiles without fast math gets the tighter sum. What it does *not* do is fix the
    // convergence test on this device — and, measured, that was not what needed fixing: pot_A's
    // stage 2 differs from the CPU rung by 1.1e-2 degrees at the worst while 30 of its 40
    // candidates stop at a different iteration, so stopping early costs almost nothing on a rung
    // that has already converged.
    var comp = 0.0;
    var i = lane;
    loop {
        if (i >= params.n_src) {
            break;
        }
        let hit = nearest_below(moved(i), r2);
        corres[base + i] = hit.x;
        if (hit.x != MISS) {
            count = count + 1.0;
            let term = bitcast<f32>(hit.y) - comp;
            let next = err2 + term;
            comp = (next - err2) - term;
            err2 = next;
        }
        i = i + LANES;
    }
    return vec2<f32>(count, err2);
}

// Invocation 0's end-of-iteration bookkeeping: the new `fitness` and `rmse` off the reduced count
// and squared error, and R §7's convergence test against the previous pair.
fn advance(first: bool) {
    let count = total[27];
    let err2 = total[28];
    var next_fitness = 0.0;
    var next_rmse = 0.0;
    if (count > 0.0) {
        next_fitness = count / f32(params.n_src);
        next_rmse = sqrt(err2 / count);
    }
    if (!first) {
        applied = applied + 1u;
        if (abs(next_fitness - fitness) < RELATIVE_FITNESS && abs(next_rmse - rmse) < RELATIVE_RMSE) {
            done = 1u;
        }
    }
    fitness = next_fitness;
    rmse = next_rmse;
    corres_count = u32(count);
    error2 = err2;
}

// ---------------------------------------------------------------------------------------------
// R §5.6's estimator: the 6×6 normal equations.
// ---------------------------------------------------------------------------------------------

@compute @workgroup_size(256)
fn point_to_plane(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let candidate = wg.x + wg.y * params.wg_x;
    if (candidate >= params.candidates) {
        return;
    }
    if (lane == 0u) {
        load_state(candidate);
    }
    workgroupBarrier();

    for (var it = 0u; it <= params.max_iter; it = it + 1u) {
        // The update from the accumulators of the previous pass; skipped on the first, which only
        // measures the initial pose (R §7 searches before it iterates).
        if (lane == 0u && it > 0u && done == 0u) {
            var a: array<f32, 36>;
            var b: array<f32, 6>;
            var x: array<f32, 6>;
            var at = 0u;
            for (var k = 0u; k < 6u; k = k + 1u) {
                for (var l = 0u; l <= k; l = l + 1u) {
                    a[k * 6u + l] = total[at];
                    a[l * 6u + k] = total[at];
                    at = at + 1u;
                }
            }
            for (var k = 0u; k < 6u; k = k + 1u) {
                b[k] = -total[21u + k];
            }
            var rot = mat3x3<f32>(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
            );
            var tau = vec3<f32>(0.0, 0.0, 0.0);
            if (total[27] > 0.0 && solve_ldlt(&a, &b, &x)) {
                rot = euler_zyx(x[0], x[1], x[2]);
                tau = vec3<f32>(x[3], x[4], x[5]);
                // The shift correction of the header: `(R − I − ω̂)·c`, formed from `O(1)` entries
                // before it meets `c`, so that composing in the shifted frame is the world
                // composition and not D §7's `Assembly::Centred`.
                let c = params.centre.xyz;
                let m00 = rot[0u][0u] - 1.0;
                let m01 = rot[1u][0u] + x[2];
                let m02 = rot[2u][0u] - x[1];
                let m10 = rot[0u][1u] - x[2];
                let m11 = rot[1u][1u] - 1.0;
                let m12 = rot[2u][1u] + x[0];
                let m20 = rot[0u][2u] + x[1];
                let m21 = rot[1u][2u] - x[0];
                let m22 = rot[2u][2u] - 1.0;
                tau = tau + vec3<f32>(
                    m00 * c.x + m01 * c.y + m02 * c.z,
                    m10 * c.x + m11 * c.y + m12 * c.z,
                    m20 * c.x + m21 * c.y + m22 * c.z,
                );
            }
            compose(rot, tau);
        }
        workgroupBarrier();

        // The correspondence search and the 29 accumulators, at the pose as it now stands.
        var acc: array<f32, 32>;
        for (var k = 0u; k < 32u; k = k + 1u) {
            acc[k] = 0.0;
        }
        if (done == 0u) {
            let counted = search(lane, candidate);
            acc[27] = counted.x;
            acc[28] = counted.y;
            let base = candidate * params.n_src;
            var i = lane;
            loop {
                if (i >= params.n_src) {
                    break;
                }
                let j = corres[base + i];
                if (j != MISS) {
                    let p = moved(i);
                    let q = cloud[j * 2u].xyz;
                    let n = cloud[j * 2u + 1u].xyz;
                    // `Assembly::World` in the shifted frame: `centre` is zero here because the
                    // host already subtracted it from both clouds.
                    let a0 = p.y * n.z - p.z * n.y;
                    let a1 = p.z * n.x - p.x * n.z;
                    let a2 = p.x * n.y - p.y * n.x;
                    let a3 = n.x;
                    let a4 = n.y;
                    let a5 = n.z;
                    let r = (p.x - q.x) * n.x + (p.y - q.y) * n.y + (p.z - q.z) * n.z;
                    acc[0] = acc[0] + a0 * a0;
                    acc[1] = acc[1] + a1 * a0;
                    acc[2] = acc[2] + a1 * a1;
                    acc[3] = acc[3] + a2 * a0;
                    acc[4] = acc[4] + a2 * a1;
                    acc[5] = acc[5] + a2 * a2;
                    acc[6] = acc[6] + a3 * a0;
                    acc[7] = acc[7] + a3 * a1;
                    acc[8] = acc[8] + a3 * a2;
                    acc[9] = acc[9] + a3 * a3;
                    acc[10] = acc[10] + a4 * a0;
                    acc[11] = acc[11] + a4 * a1;
                    acc[12] = acc[12] + a4 * a2;
                    acc[13] = acc[13] + a4 * a3;
                    acc[14] = acc[14] + a4 * a4;
                    acc[15] = acc[15] + a5 * a0;
                    acc[16] = acc[16] + a5 * a1;
                    acc[17] = acc[17] + a5 * a2;
                    acc[18] = acc[18] + a5 * a3;
                    acc[19] = acc[19] + a5 * a4;
                    acc[20] = acc[20] + a5 * a5;
                    acc[21] = acc[21] + a0 * r;
                    acc[22] = acc[22] + a1 * r;
                    acc[23] = acc[23] + a2 * r;
                    acc[24] = acc[24] + a3 * r;
                    acc[25] = acc[25] + a4 * r;
                    acc[26] = acc[26] + a5 * r;
                }
                i = i + LANES;
            }
        }
        reduce8(lane, 0u, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
        reduce8(lane, 8u, acc[8], acc[9], acc[10], acc[11], acc[12], acc[13], acc[14], acc[15]);
        reduce8(lane, 16u, acc[16], acc[17], acc[18], acc[19], acc[20], acc[21], acc[22], acc[23]);
        reduce8(lane, 24u, acc[24], acc[25], acc[26], acc[27], acc[28], 0.0, 0.0, 0.0);

        if (lane == 0u && done == 0u) {
            advance(it == 0u);
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        store_state(candidate);
    }
}

// ---------------------------------------------------------------------------------------------
// R §5.4's estimator: Eigen's `umeyama` without scaling.
// ---------------------------------------------------------------------------------------------

@compute @workgroup_size(256)
fn point_to_point(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let candidate = wg.x + wg.y * params.wg_x;
    if (candidate >= params.candidates) {
        return;
    }
    if (lane == 0u) {
        load_state(candidate);
    }
    workgroupBarrier();

    for (var it = 0u; it <= params.max_iter; it = it + 1u) {
        // The update from the previous pass's covariance.
        if (lane == 0u && it > 0u && done == 0u) {
            var rot = mat3x3<f32>(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
            );
            var tau = vec3<f32>(0.0, 0.0, 0.0);
            if (total[27] > 0.0) {
                // `sigma[r][c] = Σ b_r · a_c`, row-major in `total[0 .. 9]`.
                rot = umeyama_rotation(
                    vec3<f32>(total[0], total[3], total[6]),
                    vec3<f32>(total[1], total[4], total[7]),
                    vec3<f32>(total[2], total[5], total[8]),
                );
                let mean_p = vec3<f32>(total[9], total[10], total[11]);
                let mean_q = vec3<f32>(total[12], total[13], total[14]);
                tau = mean_q - rot * mean_p;
            }
            compose(rot, tau);
        }
        workgroupBarrier();

        // Pass one: the correspondences, their count and squared error, and the two means.
        var sum = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if (done == 0u) {
            let counted = search(lane, candidate);
            sum[6] = counted.x;
            sum[7] = counted.y;
            let base = candidate * params.n_src;
            var i = lane;
            loop {
                if (i >= params.n_src) {
                    break;
                }
                let j = corres[base + i];
                if (j != MISS) {
                    let p = moved(i);
                    let q = cloud[j * 2u].xyz;
                    sum[0] = sum[0] + p.x;
                    sum[1] = sum[1] + p.y;
                    sum[2] = sum[2] + p.z;
                    sum[3] = sum[3] + q.x;
                    sum[4] = sum[4] + q.y;
                    sum[5] = sum[5] + q.z;
                }
                i = i + LANES;
            }
        }
        reduce8(lane, 24u, sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], sum[7]);
        // `total[24 .. 30]` now holds `(sum_p, sum_q)` and `total[30], total[31]` the count and
        // the squared error; move them where `advance` and the update expect them.
        if (lane == 0u) {
            for (var k = 0u; k < 6u; k = k + 1u) {
                total[9u + k] = total[24u + k];
            }
            total[27] = total[30];
            total[28] = total[31];
            let count = total[27];
            var inv = 0.0;
            if (count > 0.0) {
                inv = 1.0 / count;
            }
            for (var k = 0u; k < 6u; k = k + 1u) {
                total[9u + k] = total[9u + k] * inv;
            }
        }
        workgroupBarrier();

        // Pass two: the covariance about the two means, with Eigen's `1/n` folded into each term.
        var cov = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if (done == 0u && total[27] > 0.0) {
            let mean_p = vec3<f32>(total[9], total[10], total[11]);
            let mean_q = vec3<f32>(total[12], total[13], total[14]);
            let inv = 1.0 / total[27];
            let base = candidate * params.n_src;
            var i = lane;
            loop {
                if (i >= params.n_src) {
                    break;
                }
                let j = corres[base + i];
                if (j != MISS) {
                    let p = moved(i);
                    let q = cloud[j * 2u].xyz;
                    let a = p - mean_p;
                    let b = (q - mean_q) * inv;
                    cov[0] = cov[0] + b.x * a.x;
                    cov[1] = cov[1] + b.x * a.y;
                    cov[2] = cov[2] + b.x * a.z;
                    cov[3] = cov[3] + b.y * a.x;
                    cov[4] = cov[4] + b.y * a.y;
                    cov[5] = cov[5] + b.y * a.z;
                    cov[6] = cov[6] + b.z * a.x;
                    cov[7] = cov[7] + b.z * a.y;
                    cov[8] = cov[8] + b.z * a.z;
                }
                i = i + LANES;
            }
        }
        reduce8(lane, 0u, cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6], cov[7]);
        reduce8(lane, 16u, cov[8], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if (lane == 0u) {
            total[8] = total[16];
        }
        workgroupBarrier();

        if (lane == 0u && done == 0u) {
            advance(it == 0u);
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        store_state(candidate);
    }
}
