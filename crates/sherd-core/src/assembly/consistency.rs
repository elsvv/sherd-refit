//! The two tests R §8 puts a candidate join through before it may extend a group: penetration
//! against every fragment already placed, and agreement with the poses the group already implies.
//!
//! Both answer the same question — *does this join contradict what is already on the table?* — and
//! both are measured in the group's own wall (`groups::Grouping::thickness`), never in the
//! collection's median. Roadmap item 4 widens the second from the reference's direct alternatives
//! to all paths in the group (D §11); phase 1 is the reference's.

use nalgebra::Matrix4;

use crate::executor::Executor;
use crate::matching::scales::Scales;
use crate::matching::verify::{penetration_share, pose_inverse};
use crate::params::Params;

use super::groups::Piece;

/// R §8's `rot_tol_deg`: how far two poses of the same fragment may disagree in rotation.
pub const ROT_TOL_DEG: f64 = 10.0;
/// R §8's `trans_tol`: how far they may disagree in translation, in wall thicknesses.
pub const TRANS_TOL_T: f64 = 0.5;

/// R §8's `_penetration`: the fraction of either fragment's surface samples inside the other,
/// given the transform that maps `b` into `a`'s frame.
///
/// This is R §6.4's count, at R §6.4's own `pen` depth — *the pair's own*, resolved from the two
/// fragments' walls and resolutions through R §1.2 exactly as the matcher resolved it, and not
/// from the group's median wall. What R §6.4 has that this does not is `pen_depth`, which R §8
/// never looks at.
///
/// `0` when either mesh is open or missing: a signed distance against a mesh with a hole is not a
/// function of the geometry (PMC-7, R §6.4), and the reference refuses the question rather than
/// answering it wrongly.
pub fn penetration(
    exec: &dyn Executor,
    a: &Piece<'_>,
    b: &Piece<'_>,
    t_ab: &Matrix4<f64>,
    p: &Params,
) -> f64 {
    if !(a.watertight && b.watertight) {
        return 0.0;
    }
    let (Some(mesh_a), Some(mesh_b)) = (a.mesh, b.mesh) else { return 0.0 };
    let depth = Scales::for_pair(p, a.thick.min(b.thick), a.res.max(b.res)).pen;
    // `sd_a` is B's samples measured against A's mesh, which is the reference's own `sdA`.
    let sd_a = penetration_share(exec, b.s_pen, t_ab, mesh_a, depth);
    let sd_b = penetration_share(exec, a.s_pen, &pose_inverse(t_ab), mesh_b, depth);
    sd_a.max(sd_b)
}

/// How far apart two world poses of the same fragment are: `(degrees, wall thicknesses)`.
///
/// `d` is R §8's `D = T_alt⁻¹ · T_new` — the transform one pose implies relative to the other — so
/// the pair of numbers is the rotation angle of its 3×3 block and the length of its translation
/// divided by the group's wall `tg`. Both tolerances are absolute, so a fragment far from the
/// origin is *not* penalised for a small rotation the way an origin displacement would penalise it:
/// `D` is expressed in the fragment's own frame.
pub fn disagreement(d: &Matrix4<f64>, tg: f64) -> (f64, f64) {
    let mut translation = 0.0;
    for i in 0..3 {
        translation += d[(i, 3)] * d[(i, 3)];
    }
    (rotation_angle_deg(d), translation.sqrt() / tg)
}

/// R §8's `rotation_angle_deg`: `degrees(arccos(clip((trace(R) − 1)/2, −1, 1)))`.
///
/// Reads the 3×3 block of a 4×4, which is what every caller has. The clip is the reference's and
/// is not cosmetic: a rotation that has been through two ICP ladders is orthonormal only to about
/// 2.6e-14, so its trace can sit just above 3 and `arccos` would return `NaN` without it.
pub fn rotation_angle_deg(d: &Matrix4<f64>) -> f64 {
    let trace = d[(0, 0)] + d[(1, 1)] + d[(2, 2)];
    ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Whether two poses agree within R §8's tolerances.
#[inline]
pub fn agrees(angle: f64, distance: f64) -> bool {
    angle <= ROT_TOL_DEG && distance <= TRANS_TOL_T
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "R §8's clip is exact: the angle is zero or it is not")]

    use super::{ROT_TOL_DEG, TRANS_TOL_T, agrees, disagreement, rotation_angle_deg};
    use approx::assert_relative_eq;
    use nalgebra::{Matrix4, Rotation3, Vector3};

    fn pose(axis: Vector3<f64>, degrees: f64, tau: [f64; 3]) -> Matrix4<f64> {
        let r = Rotation3::from_scaled_axis(axis.normalize() * degrees.to_radians());
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
        for (i, t) in tau.into_iter().enumerate() {
            m[(i, 3)] = t;
        }
        m
    }

    #[test]
    fn the_angle_is_the_rotations_own_and_the_distance_is_in_walls() {
        let d = pose(Vector3::new(0.0, 0.0, 1.0), 7.5, [0.0, 3.0, 4.0]);
        let (angle, distance) = disagreement(&d, 20.0);
        assert_relative_eq!(angle, 7.5, epsilon = 1e-12);
        assert_relative_eq!(distance, 0.25, epsilon = 1e-15);
        assert!(agrees(angle, distance));
        assert!(!agrees(ROT_TOL_DEG + 1e-9, distance));
        assert!(!agrees(angle, TRANS_TOL_T + 1e-9));
        assert!(agrees(ROT_TOL_DEG, TRANS_TOL_T), "both tolerances are inclusive");
    }

    /// The clip is what stops a pose that is orthonormal only to 1e-14 from returning `NaN`.
    #[test]
    fn a_trace_just_over_three_is_zero_degrees_and_not_nan() {
        let mut m = Matrix4::identity();
        m[(0, 0)] = 1.0 + 4e-14;
        assert_eq!(rotation_angle_deg(&m), 0.0);
    }
}
