//! Poses from paired breakline frames (R §5.1).
//!
//! Every point of a breakline carries an orthonormal frame (R §3.5.4): the shell macro-normal
//! `ns`, the in-plane axis `f` pointing away from the fracture, and the tangent `t = ns × f`
//! running along the break. Two fragments that were once one carry, at every point of the seam,
//! *the same* frame seen from two sides — so laying B's frame onto A's, flipped, is a candidate
//! pose, and the whole hypothesis set is one such pose per compatible pair of points.
//!
//! Two things make that set small enough to score.
//!
//! * **The dihedral filter.** The angle between the shell normal and the fracture normal is a
//!   property of the point, not of the pose, and at a true seam the two must add to a straight
//!   angle: `|dih_A + dih_B − 180| < 25°`. It costs one comparison per pair of points and throws
//!   away the great majority of them.
//! * **The subset.** Only `brk_sub` (R §3.5.5's voxel-thinned, frame-valid points) enters, not the
//!   whole breakline, so the pair count is thousands squared rather than tens of thousands
//!   squared. A typical pair keeps 25 000 – 150 000 hypotheses.
//!
//! The mapping itself is the one line worth reading twice:
//!
//! ```text
//! RA = [  t_A | ns_A |  f_A ]   (columns, at ia[pa])
//! RB = [ −t_B | ns_B | −f_B ]   (columns, at ib[pb])
//! R  = RA · RBᵀ                 τ = P_A − R · P_B
//! ```
//!
//! `ns` is *not* negated and the other two are: the two shells face the same way across a seam
//! (the pot's outside is the pot's outside on both sherds), while the two fracture surfaces face
//! each other and the two tangents run in opposite senses along the shared curve.
//!
//! # Numerics
//!
//! The arrays are `f64` — the reference's, and the port's own frames widened from the `f32` the
//! cache stores (D §4.1, PMC-15) — and the two contractions are written in the reference's
//! summation order, `j = 0, 1, 2`, so that a hypothesis built from the same frames is the same
//! hypothesis bit for bit.

use nalgebra::{Matrix3, Vector3};

use crate::fragment::samples::MatchData;
use crate::params::Params;

/// One fragment's breakline as R §5 reads it: the frames in `f64`, plus R §3.5.5's subset.
///
/// The port stores the breakline in `f32` (D §4.1), because it is cached and every stored array
/// has to be a function of the other stored arrays and of nothing wider. The matcher works in
/// `f64` throughout, so the widening happens once, here, and every pair the fragment takes part in
/// reads the same numbers. `tangent` is recomputed in `f64` from the widened `ns` and `f` rather
/// than widened from [`Breaklines::tangents`](crate::fragment::breakline::Breaklines::tangents),
/// which narrows it: a cross product of two unit vectors loses nothing by being computed twice and
/// the reference keeps it wide.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frames {
    /// `brk_P`: the breakline points, in adjacency order (R §3.5.3).
    pub p: Vec<[f64; 3]>,
    /// `brk_ns`: the shell macro-normal at each point (R §3.5.4).
    pub ns: Vec<[f64; 3]>,
    /// `brk_f`: the in-plane axis, pointing away from the fracture.
    pub f: Vec<[f64; 3]>,
    /// `brk_t = ns × f`: the tangent along the break (R §3.6).
    pub tangent: Vec<[f64; 3]>,
    /// `brk_dih`: the angle between the two macro normals, in degrees (R §3.6).
    pub dih: Vec<f64>,
    /// `brk_sub`: the hypothesis subset, indices into the arrays above (R §3.5.5).
    pub sub: Vec<u32>,
}

impl Frames {
    /// The frames of a [`MatchData`], widened to `f64`.
    pub fn of(md: &MatchData<'_>) -> Self {
        let brk = &md.brk;
        Self {
            p: brk.p.iter().map(|v| v.to_f64()).collect(),
            ns: brk.ns.iter().map(|v| v.to_f64()).collect(),
            f: brk.f.iter().map(|v| v.to_f64()).collect(),
            tangent: brk.tangents_f64(),
            dih: md.brk_dih.clone(),
            sub: brk.sub.clone(),
        }
    }

    /// Number of breakline points.
    #[inline]
    pub fn len(&self) -> usize {
        self.p.len()
    }

    /// True when the fragment has no breakline at all (R §5's first exit).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }
}

/// The hypothesis set of one pair: which frames were paired, and the pose each pairing implies.
///
/// `pa` and `pb` are positions **into the two subsets** `ia` and `ib`, not point indices — which
/// is what the reference's `np.where(ok)` returns and what its fixture dump carries, so the parity
/// harness compares like with like.
#[derive(Clone, Debug, Default)]
pub struct Hypotheses {
    /// Position into `ia` of the A-side frame, in row-major order over `(ia, ib)`.
    pub pa: Vec<u32>,
    /// Position into `ib` of the B-side frame.
    pub pb: Vec<u32>,
    /// `R`, mapping a direction of B into A's frame.
    pub r: Vec<Matrix3<f64>>,
    /// `τ`, so that a point of B lands at `R·q + τ`.
    pub tau: Vec<Vector3<f64>>,
}

impl Hypotheses {
    /// Number of hypotheses.
    #[inline]
    pub fn len(&self) -> usize {
        self.pa.len()
    }

    /// True when the dihedral filter left nothing — R §5.1's "a pair with zero hypotheses
    /// returns `[]`".
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pa.is_empty()
    }

    /// The pose of hypothesis `h` applied to a point of B.
    #[inline]
    pub fn apply(&self, h: usize, q: &[f64; 3]) -> [f64; 3] {
        let (r, t) = (&self.r[h], &self.tau[h]);
        [
            r[(0, 0)] * q[0] + r[(0, 1)] * q[1] + r[(0, 2)] * q[2] + t[0],
            r[(1, 0)] * q[0] + r[(1, 1)] * q[1] + r[(1, 2)] * q[2] + t[1],
            r[(2, 0)] * q[0] + r[(2, 1)] * q[1] + r[(2, 2)] * q[2] + t[2],
        ]
    }
}

/// R §5.1 over the two fragments' own subsets.
pub fn build(a: &Frames, b: &Frames, p: &Params) -> Hypotheses {
    build_with(a, b, &a.sub, &b.sub, p.dihedral_tol)
}

/// R §5.1 with the two subsets given — which is how R §4.3's screening pass buys a cheaper version
/// of the same test, and how the parity harness feeds in the reference's own `hyp.ia` / `hyp.ib`.
///
/// The subsets are used in the order given (PMC-4: the reference's is Open3D's hash order, the
/// port's is ascending), because `pa` and `pb` are positions into them and only the caller knows
/// which order it means.
pub fn build_with(a: &Frames, b: &Frames, ia: &[u32], ib: &[u32], dihedral_tol: f64) -> Hypotheses {
    if ia.is_empty() || ib.is_empty() {
        return Hypotheses::default();
    }
    let (mut pa, mut pb) = (Vec::new(), Vec::new());
    // Row-major over (i, j), which is `np.where`'s order on a 2-D mask.
    for (i, &pi) in ia.iter().enumerate() {
        let da = a.dih[pi as usize];
        for (j, &pj) in ib.iter().enumerate() {
            if (da + b.dih[pj as usize] - 180.0).abs() < dihedral_tol {
                pa.push(u32::try_from(i).unwrap_or(u32::MAX));
                pb.push(u32::try_from(j).unwrap_or(u32::MAX));
            }
        }
    }
    poses(a, b, ia, ib, &pa, &pb)
}

/// The poses of a *given* set of frame pairs — R §5.1's last two lines on their own.
///
/// The dihedral filter and the pose construction are separable, and the parity harness separates
/// them: fed the reference's own `hyp.pa` / `hyp.pb` this builds the reference's own hypothesis
/// set, so the coarse and NMS rows measure their own stage rather than inheriting the row above.
pub fn poses(a: &Frames, b: &Frames, ia: &[u32], ib: &[u32], pa: &[u32], pb: &[u32]) -> Hypotheses {
    let mut out = Hypotheses {
        pa: pa.to_vec(),
        pb: pb.to_vec(),
        r: Vec::with_capacity(pa.len()),
        tau: Vec::with_capacity(pa.len()),
    };
    for (&i, &j) in pa.iter().zip(pb) {
        let (ka, kb) = (ia[i as usize] as usize, ib[j as usize] as usize);
        let (r, tau) = pose(a, ka, b, kb);
        out.r.push(r);
        out.tau.push(tau);
    }
    out
}

/// The pose of one frame pair: `R = RA·RBᵀ`, `τ = P_A − R·P_B`.
///
/// Written out rather than assembled from matrix products so that the three-term sums happen in
/// the reference's `j = 0, 1, 2` order and the sign convention is visible in one place.
fn pose(a: &Frames, ka: usize, b: &Frames, kb: usize) -> (Matrix3<f64>, Vector3<f64>) {
    // Row `i` of `RA` is the `i`-th component of each of A's three axes; row `k` of `RB` likewise,
    // with the tangent and the in-plane axis negated (see the module documentation).
    let axes_a = [a.tangent[ka], a.ns[ka], a.f[ka]];
    let axes_b = [b.tangent[kb], b.ns[kb], b.f[kb]];
    let mut rot = Matrix3::zeros();
    for i in 0..3 {
        for k in 0..3 {
            rot[(i, k)] = axes_a[0][i] * -axes_b[0][k]
                + axes_a[1][i] * axes_b[1][k]
                + axes_a[2][i] * -axes_b[2][k];
        }
    }
    let (anchor_a, anchor_b) = (a.p[ka], b.p[kb]);
    let mut tau = Vector3::zeros();
    for i in 0..3 {
        tau[i] = anchor_a[i]
            - (rot[(i, 0)] * anchor_b[0] + rot[(i, 1)] * anchor_b[1] + rot[(i, 2)] * anchor_b[2]);
    }
    (rot, tau)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the frame algebra is exact on these unit vectors")]

    use super::{Frames, build_with};
    use crate::params::Params;

    /// A frame triple `(t, ns, f)` at a point, as R §3.5.4 leaves it.
    fn frame(p: [f64; 3], ns: [f64; 3], f: [f64; 3], dih: f64) -> Frames {
        let t =
            [ns[1] * f[2] - ns[2] * f[1], ns[2] * f[0] - ns[0] * f[2], ns[0] * f[1] - ns[1] * f[0]];
        Frames {
            p: vec![p],
            ns: vec![ns],
            f: vec![f],
            tangent: vec![t],
            dih: vec![dih],
            sub: vec![0],
        }
    }

    /// The hypothesis of a matching frame pair is the pose that *is* the ground truth: B's frame
    /// laid onto A's, its fracture facing A's and its shell continuing A's.
    ///
    /// A and B are two halves of a wall broken along the y axis at the origin: the shell of both
    /// points at `+z`, A's fracture faces `+x` (so A's outward in-plane axis is `−x`) and B's
    /// faces `−x`. Complementary dihedrals, one hypothesis, and the pose must be the identity
    /// rotation with no translation, because the two frames already coincide in world space.
    #[test]
    fn a_matching_frame_pair_gives_the_pose_that_joins_them() {
        let a = frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], 90.0);
        let b = frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 90.0);
        let h = build_with(&a, &b, &[0], &[0], Params::default().dihedral_tol);
        assert_eq!(h.len(), 1);
        assert_eq!(h.pa, vec![0]);
        assert_eq!(h.pb, vec![0]);
        for i in 0..3 {
            for k in 0..3 {
                let want = if i == k { 1.0 } else { 0.0 };
                assert!((h.r[0][(i, k)] - want).abs() < 1e-15, "R{i}{k} = {}", h.r[0][(i, k)]);
            }
        }
        assert!(h.tau[0].norm() < 1e-15);
        assert_eq!(h.apply(0, &[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    /// The same pair with B displaced and turned: the hypothesis has to bring it back, which is
    /// what "the frames are the pose" means.
    #[test]
    fn a_displaced_partner_is_brought_back_onto_the_seam() {
        let a = frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], 90.0);
        // B rotated 90° about z and moved to (5, 7, 11): shell still +z, in-plane axis now +y.
        let b = frame([5.0, 7.0, 11.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 90.0);
        let h = build_with(&a, &b, &[0], &[0], 25.0);
        assert_eq!(h.len(), 1);
        // B's own point lands on A's.
        let moved = h.apply(0, &[5.0, 7.0, 11.0]);
        assert!(moved.iter().all(|c| c.abs() < 1e-14), "{moved:?}");
        // Its outward in-plane axis lands on the opposite of A's, which is the seam condition.
        let dir = h.r[0] * nalgebra::Vector3::new(0.0, 1.0, 0.0);
        assert!((dir - nalgebra::Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-14, "{dir:?}");
    }

    /// The filter is on the *sum* of the two dihedrals and it is strict.
    #[test]
    fn the_dihedral_filter_keeps_only_complementary_frames() {
        let up = [0.0, 0.0, 1.0];
        let mut a = frame([0.0; 3], up, [-1.0, 0.0, 0.0], 70.0);
        a.dih = vec![70.0, 90.0];
        a.p.push([0.0; 3]);
        a.ns.push(up);
        a.f.push([-1.0, 0.0, 0.0]);
        a.tangent.push([0.0, 1.0, 0.0]);
        let mut b = frame([0.0; 3], up, [1.0, 0.0, 0.0], 100.0);
        b.dih = vec![100.0, 60.0];
        b.p.push([0.0; 3]);
        b.ns.push(up);
        b.f.push([1.0, 0.0, 0.0]);
        b.tangent.push([0.0, -1.0, 0.0]);

        // 70+100 = 170 (in), 70+60 = 130 (out), 90+100 = 190 (in), 90+60 = 150 (out) — and the
        // pairs come out row-major, `i` outer.
        let h = build_with(&a, &b, &[0, 1], &[0, 1], 25.0);
        assert_eq!(h.pa, vec![0, 1]);
        assert_eq!(h.pb, vec![0, 0]);

        // Exactly on the tolerance is out: the reference's test is `< tol`.
        a.dih = vec![90.0, 90.0];
        b.dih = vec![65.0, 115.0];
        let h = build_with(&a, &b, &[0, 1], &[0, 1], 25.0);
        assert!(h.is_empty(), "{:?}", h.pa);
    }

    /// The subsets are used in the order given, and `pa`/`pb` are positions into them — the
    /// property PMC-4 rests on, and the one the injected parity row depends on.
    #[test]
    fn the_pairs_are_positions_into_the_subsets_in_the_order_given() {
        let up = [0.0, 0.0, 1.0];
        let mut a = frame([0.0; 3], up, [-1.0, 0.0, 0.0], 90.0);
        for _ in 0..2 {
            a.p.push([0.0; 3]);
            a.ns.push(up);
            a.f.push([-1.0, 0.0, 0.0]);
            a.tangent.push([0.0, 1.0, 0.0]);
        }
        a.dih = vec![90.0, 0.0, 90.0];
        let b = frame([0.0; 3], up, [1.0, 0.0, 0.0], 90.0);

        // Descending subset, as Open3D's hash order can produce: position 0 is point 2.
        let h = build_with(&a, &b, &[2, 1, 0], &[0], 25.0);
        assert_eq!(h.pa, vec![0, 2], "point 1's dihedral is 0, so position 1 is filtered out");
        assert_eq!(h.pb, vec![0, 0]);
        assert!(build_with(&a, &b, &[], &[0], 25.0).is_empty());
        assert!(build_with(&a, &b, &[0], &[], 25.0).is_empty());
    }
}
