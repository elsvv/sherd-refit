//! The batch structs of D §6.1 and their device layouts (D §6.3).
//!
//! Every [`Executor`](super::Executor) method takes one of these and nothing else. A batch is the
//! whole of one kernel's input: the fixed side (a target cloud with its tree, a BVH), the moving
//! side (a point set), the poses to apply and the thresholds — enough for a GPU submission to be
//! formed from it without reaching back into the pipeline, and enough for the CPU executor to
//! compute exactly what the pipeline computed before the boundary existed.
//!
//! # Two layouts, and why the CPU one is `f64`
//!
//! D §6.3 says the device arrays are SoA `vec4<f32>`, 16-byte aligned and `bytemuck::Pod`, and
//! they are — [`CoarseBatch::device_points`], [`HashGrid`](crate::spatial::grid::HashGrid) and
//! [`PoseGpu`] below produce exactly that. But the batch *itself* carries `f64` slices and the
//! CPU's own structures, because the CPU executor is the reference implementation of every method
//! (D §6.1) and experiment E5 measured `f32` point loops far outside D §10.2 — the median stage-2
//! pose on pot G moves 9.8 `t` (`matching::icp`'s module documentation). A batch that narrowed on
//! formation would change what the CPU computes, and D's own rule is that it must not: the CPU
//! path's outputs stay byte-identical to the ones phase 1 verified. The narrowing therefore
//! happens **inside the GPU executor**, on the way to the device, where the tolerance-based parity
//! of D §10.2 is what governs.
//!
//! # The grid is built by the executor that needs it
//!
//! D §6.2's hash grid is built on the CPU and uploaded with the batch, and
//! [`CoarseBatch::device_grid`] / [`IcpBatch::device_grid`] are where that happens. They are not
//! called on the CPU path: the CPU searches with `kiddo` (E3 measured the grid at 0.4–2.0×
//! `kiddo`, never the ≥ 3× D §3 hoped for) and every parity row of D §10.2 is measured through the
//! KD-tree, so building a grid the CPU never queries would cost `synthetic_20` real seconds to
//! prove nothing.

use nalgebra::{Matrix3, Matrix4, Vector3};

use crate::matching::coarse::Target;
use crate::matching::icp::{IcpTarget, Options};
use crate::spatial::bvh::RayScene;
use crate::spatial::grid::{HashGrid, NearMask};

/// A pose as the kernels read it: three rows of `vec4<f32>`, translation in `w` (D §6.3's
/// `CandState::t: mat3x4<f32>`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PoseGpu {
    /// Row-major `[R | τ]`: `row[i] = (R[i][0], R[i][1], R[i][2], τ[i])`.
    pub rows: [[f32; 4]; 3],
}

impl PoseGpu {
    /// The `f32` pose a kernel applies, from the `f64` one the CPU keeps.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
    pub fn narrow(rotation: &Matrix3<f64>, translation: &Vector3<f64>) -> Self {
        let mut rows = [[0.0_f32; 4]; 3];
        for (i, row) in rows.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().take(3).enumerate() {
                *cell = rotation[(i, j)] as f32;
            }
            row[3] = translation[i] as f32;
        }
        Self { rows }
    }

    /// [`PoseGpu::narrow`] of a homogeneous 4×4.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
    pub fn of(transform: &Matrix4<f64>) -> Self {
        let mut rows = [[0.0_f32; 4]; 3];
        for (i, row) in rows.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = transform[(i, j)] as f32;
            }
        }
        Self { rows }
    }
}

/// A point array in the device's layout: `vec4<f32>`, `w` unused (D §6.3).
#[must_use]
#[allow(clippy::cast_possible_truncation, reason = "D §6.8: the kernels are f32 only")]
pub fn device_vec4(points: &[[f64; 3]]) -> Vec<[f32; 4]> {
    points.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0]).collect()
}

/// How a batch of poses is held: R §5.1 keeps the rotation and the translation apart, R §5.4 and
/// everything after it carry a homogeneous 4×4.
///
/// The two are the same poses; splitting them here rather than converting on formation keeps the
/// CPU executor's arithmetic the arithmetic it was before the boundary existed, down to which
/// `f64` is multiplied by which.
#[derive(Clone, Copy, Debug)]
pub enum Poses<'a> {
    /// R §5.1's hypotheses: a rotation array and a translation array.
    Split {
        /// The rotations.
        r: &'a [Matrix3<f64>],
        /// The translations.
        tau: &'a [Vector3<f64>],
    },
    /// One 4×4 per pose.
    Homogeneous(&'a [Matrix4<f64>]),
}

impl Poses<'_> {
    /// How many poses the batch scores.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Split { r, .. } => r.len(),
            Self::Homogeneous(t) => t.len(),
        }
    }

    /// True when there is no pose to score.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pose `i` as a rotation and a translation.
    ///
    /// The 4×4 case reads the twelve cells one at a time, which is what
    /// [`coarse::score_pose`](crate::matching::coarse::score_pose) did before this type existed.
    #[must_use]
    pub fn get(&self, i: usize) -> (Matrix3<f64>, Vector3<f64>) {
        match self {
            Self::Split { r, tau } => (r[i], tau[i]),
            Self::Homogeneous(t) => {
                let transform = &t[i];
                let mut rot = Matrix3::zeros();
                for i in 0..3 {
                    for j in 0..3 {
                        rot[(i, j)] = transform[(i, j)];
                    }
                }
                let tau = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
                (rot, tau)
            }
        }
    }

    /// The poses in the device layout of D §6.3.
    #[must_use]
    pub fn device(&self) -> Vec<PoseGpu> {
        (0..self.len())
            .map(|i| {
                let (r, tau) = self.get(i);
                PoseGpu::narrow(&r, &tau)
            })
            .collect()
    }
}

/// R §5.2's coarse score and R §5.4's re-score: many poses, one moving point set, one breakline.
///
/// The two stages differ only in the radius, the point set and how many poses there are — sixty
/// points against tens of thousands of hypotheses for R §5.2, a whole `brk_sub` against one pose
/// per candidate for R §5.4 — which is why D §6.5 gives them one kernel.
#[derive(Clone, Copy, Debug)]
pub struct CoarseBatch<'a> {
    /// A's breakline, its shell normals and the tree over them.
    pub target: Target<'a>,
    /// R §5.2's conservative pre-filter over the same points (`spatial::grid`, E2).
    ///
    /// A `NearMask` can only turn a query that would have missed into a query that is not made, so
    /// it moves no score; it is carried in the batch because whether it is worth building depends
    /// on how many poses share it, which is a property of the batch and not of the kernel.
    pub mask: Option<&'a NearMask>,
    /// The moving points, in B's own frame.
    pub points: &'a [[f64; 3]],
    /// Their shell macro-normals, in B's own frame.
    pub normals: &'a [[f64; 3]],
    /// The poses to score.
    pub poses: Poses<'a>,
    /// The correspondence radius: `sc.coarse` for R §5.2, `sc.stage1` for R §5.4.
    pub radius: f64,
    /// The cosine the two shell normals must exceed (R §5.2's 0.7).
    pub normal_agree: f64,
}

impl CoarseBatch<'_> {
    /// The moving points in D §6.3's layout.
    #[must_use]
    pub fn device_points(&self) -> Vec<[f32; 4]> {
        device_vec4(self.points)
    }

    /// The moving normals in D §6.3's layout.
    #[must_use]
    pub fn device_normals(&self) -> Vec<[f32; 4]> {
        device_vec4(self.normals)
    }

    /// D §6.2's hash grid over the target breakline at this batch's radius, for the kernel to
    /// query.
    ///
    /// Built here, on the CPU, and uploaded with the batch; `None` when the target is empty or
    /// the radius is not usable, which is the same case the CPU path answers zero for.
    #[must_use]
    pub fn device_grid(&self) -> Option<HashGrid> {
        HashGrid::build(self.target.points, self.radius)
    }

    /// Bytes the device side of this batch occupies, for the 128 MB binding cap of D §6.3.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let grid = self.device_grid().map_or(0, |g| g.device_bytes());
        grid + (self.points.len() + self.normals.len() + self.target.normals.len()) * 16
            + self.poses.len() * std::mem::size_of::<PoseGpu>()
    }
}

/// One rung of R §7's ICP for a batch of candidates that share the clouds, the estimator, the
/// radius and the iteration cap (D §6.4 steps 4 and 5).
///
/// Each candidate is an independent ladder — nothing crosses from one to another — so a batch of
/// `n` is `n` runs of `registration_icp` and the answers do not depend on how they were grouped.
/// That is what lets stage 1 hand the whole of R §5.3's kept list to one call while a caller with
/// a single pose hands over a slice of one.
#[derive(Clone, Copy, Debug)]
pub struct IcpBatch<'a> {
    /// The moving cloud, in B's own frame; every candidate registers this same cloud.
    pub source: &'a [[f64; 3]],
    /// The fixed cloud, with its tree and (for point-to-plane) its normals.
    pub target: &'a IcpTarget,
    /// One starting pose per candidate.
    pub inits: &'a [Matrix4<f64>],
    /// R §7's four knobs, shared by the batch.
    pub options: Options,
}

impl IcpBatch<'_> {
    /// How many candidates the rung refines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inits.len()
    }

    /// True when there is no candidate to refine.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inits.is_empty()
    }

    /// The moving cloud in D §6.3's layout.
    #[must_use]
    pub fn device_source(&self) -> Vec<[f32; 4]> {
        device_vec4(self.source)
    }

    /// The starting poses in D §6.3's layout.
    #[must_use]
    pub fn device_inits(&self) -> Vec<PoseGpu> {
        self.inits.iter().map(PoseGpu::of).collect()
    }

    /// D §6.2's hash grid over the target cloud at the rung's correspondence radius.
    #[must_use]
    pub fn device_grid(&self) -> Option<HashGrid> {
        HashGrid::build(self.target.points(), self.options.max_correspondence_distance)
    }

    /// Bytes the device side of this batch occupies (D §6.3's 128 MB cap).
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let grid = self.device_grid().map_or(0, |g| g.device_bytes());
        grid + (self.source.len() + self.target.len() * 2) * 16
            + self.inits.len() * std::mem::size_of::<PoseGpu>()
    }
}

/// What a [`DistBatch`] asks for: every distance, or only the smallest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistReduce {
    /// One distance per point, `+∞` beyond the window — R §6.1's `d1`/`d2` arrays.
    All,
    /// The smallest distance over the whole batch, as one value.
    ///
    /// R §6.4 needs it when *nothing* is inside the other mesh: `min(sd)` is then the smallest
    /// positive distance, and the CPU finds it with a window that shrinks to the best so far, so
    /// the batch is a min-reduction with an early-out rather than an array the caller reduces.
    Min,
}

/// R §6.1's bounded point-to-surface distance: a point set, a pose and a BVH (D §6.5).
#[derive(Clone, Copy, Debug)]
pub struct DistBatch<'a> {
    /// The points, in their own fragment's frame.
    pub points: &'a [[f64; 3]],
    /// The pose that moves them into the scene's frame.
    pub transform: &'a Matrix4<f64>,
    /// The surface to measure against.
    pub scene: &'a RayScene,
    /// The window: a surface further than this needs no number (`sc.facing` in R §6.1).
    pub max_dist: f64,
    /// Whether the caller wants the array or its minimum.
    pub reduce: DistReduce,
}

impl DistBatch<'_> {
    /// The points in D §6.3's layout, already moved by the pose — which is what the kernel
    /// receives, because the BVH is in the scene's frame.
    #[must_use]
    pub fn device_points(&self) -> Vec<[f32; 4]> {
        device_vec4(&moved(self.points, self.transform))
    }

    /// Bytes the device side of this batch occupies, excluding the resident BVH (D §6.3's
    /// fragment slots hold that).
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.points.len() * 16
    }
}

/// R §6.4's inside test for one point set against one mesh (D §6.5).
#[derive(Clone, Copy, Debug)]
pub struct InsideBatch<'a> {
    /// The points, in their own fragment's frame.
    pub points: &'a [[f64; 3]],
    /// The pose that moves them into the mesh's frame.
    pub transform: &'a Matrix4<f64>,
    /// The mesh, as a BVH; the answer is only meaningful when it is closed (R §6.4).
    pub scene: &'a RayScene,
}

impl InsideBatch<'_> {
    /// The points in D §6.3's layout, already moved by the pose.
    #[must_use]
    pub fn device_points(&self) -> Vec<[f32; 4]> {
        device_vec4(&moved(self.points, self.transform))
    }

    /// Bytes the device side of this batch occupies, excluding the resident BVH.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.points.len() * 16 + self.points.len() * std::mem::size_of::<InsideOutcome>()
    }
}

/// One point's answer to R §6.4: is it inside, and how deep (D §6.1's `InsideResult`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InsideOutcome {
    /// How far the point is from the surface, when it is inside; `0` when it is not.
    pub depth: f32,
    /// `1` when the point is inside the mesh, `0` when it is not.
    ///
    /// A `u32` rather than a `bool` because the array is read straight out of a device buffer,
    /// where WGSL has no one-byte type.
    pub inside: u32,
}

impl InsideOutcome {
    /// A point that is outside the mesh.
    pub const OUTSIDE: Self = Self { depth: 0.0, inside: 0 };

    /// A point inside the mesh, `depth` from its surface.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "the BVH is f32, as Open3D's scene is")]
    pub fn inside_at(depth: f64) -> Self {
        Self { depth: depth as f32, inside: 1 }
    }

    /// Whether the point is inside.
    #[must_use]
    pub fn is_inside(self) -> bool {
        self.inside != 0
    }
}

/// `transform · p` for every point, in `f64` — the way every caller of R §6 does it.
fn moved(points: &[[f64; 3]], transform: &Matrix4<f64>) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|p| {
            std::array::from_fn(|i| {
                transform[(i, 0)] * p[0]
                    + transform[(i, 1)] * p[1]
                    + transform[(i, 2)] * p[2]
                    + transform[(i, 3)]
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the device layout is exact or it is wrong")]

    use super::{CoarseBatch, DistReduce, InsideOutcome, PoseGpu, Poses, device_vec4};
    use crate::matching::coarse::Target;
    use crate::spatial::kdtree::PointTree;
    use nalgebra::{Matrix3, Matrix4, Vector3};

    /// The device structs are the byte layouts D §6.3 names: sixteen-byte aligned, no padding.
    #[test]
    fn the_device_structs_have_the_layout_the_kernels_bind() {
        assert_eq!(std::mem::size_of::<PoseGpu>(), 48);
        assert_eq!(std::mem::align_of::<PoseGpu>(), 4);
        assert_eq!(std::mem::size_of::<InsideOutcome>(), 8);
        let points = vec![[1.0, 2.0, 3.0], [-4.0, 5.0, 6.5]];
        let device = device_vec4(&points);
        assert_eq!(device, vec![[1.0, 2.0, 3.0, 0.0], [-4.0, 5.0, 6.5, 0.0]]);
        assert_eq!(bytemuck::cast_slice::<[f32; 4], u8>(&device).len(), 32);
    }

    /// The two pose layouts describe the same pose, and the device form is its `f32` narrowing.
    #[test]
    fn the_two_pose_layouts_agree() {
        let r = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let tau = Vector3::new(1.5, -2.5, 3.5);
        let homogeneous = crate::matching::icp::homogeneous(&r, &tau);
        let split = Poses::Split { r: std::slice::from_ref(&r), tau: std::slice::from_ref(&tau) };
        let dense = Poses::Homogeneous(std::slice::from_ref(&homogeneous));
        assert_eq!(split.len(), 1);
        assert_eq!(split.get(0), dense.get(0));
        assert_eq!(split.device(), dense.device());
        assert_eq!(dense.device()[0], PoseGpu::of(&homogeneous));
        assert_eq!(PoseGpu::of(&homogeneous).rows[0], [0.0, -1.0, 0.0, 1.5]);
        assert!(Poses::Homogeneous(&[]).is_empty());
    }

    /// A coarse batch reports the grid, the arrays and the byte count a submission would need.
    #[test]
    fn a_coarse_batch_carries_its_device_side() {
        let points: Vec<[f64; 3]> = (0..64).map(|k| [f64::from(k) * 0.5, 0.0, 0.0]).collect();
        let normals = vec![[0.0, 0.0, 1.0]; 64];
        let tree = PointTree::build(&points).expect("64 points");
        let target = Target { points: &points, normals: &normals, tree: &tree };
        let identity = Matrix4::identity();
        let batch = CoarseBatch {
            target,
            mask: None,
            points: &points,
            normals: &normals,
            poses: Poses::Homogeneous(std::slice::from_ref(&identity)),
            radius: 0.25,
            normal_agree: 0.7,
        };
        assert_eq!(batch.device_points().len(), 64);
        assert_eq!(batch.device_normals().len(), 64);
        let grid = batch.device_grid().expect("a non-empty target");
        assert_eq!(grid.points().len(), 64);
        assert!((grid.radius() - 0.25).abs() < 1e-9);
        assert!(batch.device_bytes() > grid.device_bytes());
        assert_eq!(DistReduce::All, DistReduce::All);
    }
}
