// R §5.4's estimator's linear algebra: a 3×3 one-sided Jacobi SVD and the rotation Eigen's
// `umeyama(src, dst, with_scaling = false)` reads off it.
//
// This file is not a kernel and it binds nothing. Like `grid.wgsl` it is prepended to the kernel
// that uses it (`icp.wgsl`) at build time, for two reasons: the arithmetic is the one piece of
// `icp.wgsl` that is a pure function of nine floats, and a pure function of nine floats can be
// **tested on the device** against the CPU's own `umeyama` — `kernels/umeyama_probe.wgsl` is that
// test's entry point and `icp::umeyama_rotations` dispatches it.
//
// The three rules of E7 §4 hold here as they do in `grid.wgsl`: no `dot`, `length` or `normalize`
// on a parity-critical expression, no subgroup reductions, and the dispatch shape is data.

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

// The coordinate axis `a` leans on least: the one whose component of `a` is smallest in
// magnitude, ties to the lowest index. `a` is a unit vector, so that component is at most `1/√3`.
fn off_axis(a: vec3<f32>) -> vec3<f32> {
    let m = vec3<f32>(abs(a.x), abs(a.y), abs(a.z));
    if (m.x <= m.y && m.x <= m.z) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if (m.y <= m.z) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// `a / |a|`, written out rather than through `normalize` (E7 §4.2). A zero vector comes back
// unchanged, which is a case the two callers cannot reach: `off_axis` is never parallel to its
// argument.
fn unit(a: vec3<f32>) -> vec3<f32> {
    let n = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    if (n > 0.0) {
        return a / n;
    }
    return a;
}

// Eigen's `umeyama(src, dst, with_scaling = false)` rotation from the 3×3 covariance `sigma`,
// given column by column.
//
// `SVD::new` sorts the singular values descending, which is what puts the reflection fix on the
// smallest of them; the sort here is the same selection sort over three columns.
//
// # The rank-deficient cases (H1-D1)
//
// A singular value at zero leaves its `U` and `V` columns undefined, and the completion below is
// what keeps the result a rotation. It has to be written from the **largest** singular vector
// outwards, because at rank 1 there is only one of them:
//
// | rank | what is known | what is built |
// |---|---|---|
// | 3 | `u₀ u₁ u₂` | nothing |
// | 2 | `u₀ u₁` | `u₂ = u₀ × u₁` — determined, and the sign is fixed by `det U · det V` below |
// | 1 | `u₀` | `u₁ = normalise(u₀ × e)` for the axis `e` least parallel to `u₀`, then `u₂ = u₀ × u₁` |
// | 0 | nothing | the identity basis, on both sides |
//
// The rank-2 line is what this function always had; the other two are H1-D1, the defect the
// audit's orthonormality check found on its first production run
// (`notes/2026-09-09-h1-wd1.md` §4). Written the other way round — `u₂` first — rank 1 completes
// `u₂ = u₀ × 0 = 0` and then `u₁ = 0 × u₀ = 0`, and the function returns a matrix with `det = 0`
// that is 1.0 from orthonormal.
//
// **What the CPU does, and where the two agree.** `matching::icp::umeyama` takes `U` and `Vᵀ` from
// `nalgebra`'s Golub–Reinsch SVD, whose bases are products of Householder reflectors and are
// therefore complete and orthonormal at every rank, and returns `U · diag(1, 1, d) · Vᵀ` — always
// a rotation. So do we now, at every rank. At ranks 3 and 2 the rotation is *determined by the
// data* and the two agree to `f32`; at rank 0 both return the identity (`nalgebra` reflects a zero
// matrix with identity reflectors, so `U = V = I`). At **rank 1 the rotation is not determined by
// the data at all** — every rotation carrying `v₀` to `u₀` is an equal minimiser of Umeyama's
// objective, a one-parameter family — so the two pick different members of it, and what they agree
// on, which is what the caller needs and what `icp::registration` checks, is that the answer is a
// rotation.
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
    if (s[0] <= 0.0) {
        // Rank 0: the covariance is exactly zero and there is no singular vector to build from.
        // Both bases become the identity, so `U · d · Vᵀ` is the identity — `nalgebra`'s answer.
        u[0] = vec3<f32>(1.0, 0.0, 0.0);
        v[0] = vec3<f32>(1.0, 0.0, 0.0);
    }
    if (s[1] <= 0.0) {
        // Rank 1: only `u₀` and `v₀` are known. Complete each with the coordinate axis least
        // parallel to it, which cannot be parallel — a unit vector has a component of at least
        // `1/√3` on some axis, so its *smallest* component is at most `1/√3` and the cross
        // product below has a norm of at least `√(2/3)`.
        u[1] = unit(cross3(u[0], off_axis(u[0])));
        v[1] = unit(cross3(v[0], off_axis(v[0])));
    }
    if (s[2] <= 0.0) {
        u[2] = cross3(u[0], u[1]);
        v[2] = cross3(v[0], v[1]);
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
