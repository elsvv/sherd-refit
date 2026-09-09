//! The data model shared by every stage (D §4.1).
//!
//! Only the types that more than one module needs live here. Stage-local structures — the match
//! arrays, the score set, the candidates, the groups — belong to the module that produces them
//! and appear as those modules are filled in.

use std::path::PathBuf;

use crate::Vec3f;

/// Index of a fragment inside one collection, assigned in the order R §2 discovers the files.
pub type FragId = u32;

/// What a face of the working mesh is (R §3.4).
///
/// `Shell` and `Fracture` are the two the reference produces; `Solid` and `Rim` are reserved for
/// roadmap item 6 (D §11) and are never assigned in phase 1.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FaceLabel {
    /// Original outer or inner surface of the vessel.
    #[default]
    Shell = 0,
    /// Fracture surface: where this fragment broke away from its neighbours.
    Fracture = 1,
    /// A solid part with no opposite wall (roadmap item 6).
    Solid = 2,
    /// The vessel's rim, thicker than the wall (roadmap item 6).
    Rim = 3,
}

impl FaceLabel {
    /// True for the faces the matcher works on.
    #[inline]
    pub fn is_fracture(self) -> bool {
        matches!(self, Self::Fracture)
    }

    /// The label a cache file's `labels` tensor byte stands for, or `None` for a byte no variant
    /// uses (D §4.2: the cache is validated, not trusted).
    #[inline]
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Shell),
            1 => Some(Self::Fracture),
            2 => Some(Self::Solid),
            3 => Some(Self::Rim),
            _ => None,
        }
    }
}

/// The file a fragment was read from, and enough of its metadata to tell whether a cache entry
/// still describes it (R §3.7, D §4.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRef {
    /// Path as given on the command line, not canonicalised.
    pub path: PathBuf,
    /// Size in bytes at load time.
    pub size: u64,
    /// Modification time in nanoseconds since the Unix epoch.
    pub mtime_ns: i128,
    /// Content hash, computed only when the caller asks for it (large scans make it expensive).
    pub sha256: Option<[u8; 32]>,
}

/// The decimated, smoothed mesh every later stage measures (R §3.3).
///
/// Vertices and faces come out of decimation; the three per-face arrays and `res` are derived
/// once and never recomputed, because every threshold of R §1.2 is expressed in `res`.
#[derive(Clone, Debug, Default)]
pub struct WorkingMesh {
    /// Vertices.
    pub v: Vec<Vec3f>,
    /// Triangles, counter-clockwise seen from outside.
    pub f: Vec<[u32; 3]>,
    /// Unit face normals, one per triangle.
    pub face_normals: Vec<Vec3f>,
    /// Face areas, one per triangle.
    pub face_areas: Vec<f32>,
    /// Face centroids, one per triangle.
    pub face_centroids: Vec<Vec3f>,
    /// Median length of the unique edges — the mesh resolution of R §0.
    pub res: f32,
}

impl WorkingMesh {
    /// The one way a working mesh is built (R §3.3): from its `f32` vertices, its triangles and
    /// its `res`, with the three per-face arrays derived here and nowhere else.
    ///
    /// The derivation runs `face_geometry` in `f64` over the vertices **after** they have been
    /// narrowed to `f32`, which is what makes a fragment read back from the cache
    /// (`fragment::cache`, D §4.2) bit-identical to the same fragment computed from the file: the
    /// cache stores `V`, `F` and `res` and nothing derived, so the two paths have to agree on how
    /// the rest follows from them. Computing the normals from the wider pre-narrowing coordinates
    /// instead would leave a cold run and a warm run a few ULP apart in every face normal.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the working mesh is f32 by design (D §4.1, §7)"
    )]
    pub fn from_parts(v: Vec<Vec3f>, f: Vec<[u32; 3]>, res: f32) -> Self {
        let v64: Vec<[f64; 3]> = v.iter().map(|p| p.to_f64()).collect();
        let geom = crate::mesh::geometry::face_geometry(&v64, &f);
        Self {
            v,
            f,
            face_normals: geom.normals.iter().copied().map(Vec3f::from_f64).collect(),
            face_areas: geom.areas.iter().map(|&a| a as f32).collect(),
            face_centroids: geom.centroids.iter().copied().map(Vec3f::from_f64).collect(),
            res,
        }
    }

    /// Number of vertices.
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.v.len()
    }

    /// Number of triangles.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.f.len()
    }

    /// True when the mesh carries no triangle.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.f.is_empty()
    }
}

/// A sampled point cloud with one normal per point (R §3.5, R §3.6).
///
/// The two vectors are always the same length; the samplers keep them in step and the ICP relies
/// on it.
#[derive(Clone, Debug, Default)]
pub struct Cloud {
    /// Points.
    pub p: Vec<Vec3f>,
    /// Unit normals, one per point.
    pub n: Vec<Vec3f>,
}

impl Cloud {
    /// Number of points.
    #[inline]
    pub fn len(&self) -> usize {
        self.p.len()
    }

    /// True when the cloud holds no point.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }
}

/// `R·p + τ` for one point — the reference's `apply_transform(T, P)` (`sherd_refit.geometry`),
/// which every stage that moves a point through a 4×4 goes through.
///
/// The reference writes it as `P @ T[:3, :3].T + T[:3, 3]`, so the translation is added after the
/// three-term dot product, and R §6's scores came out bit-identical to the reference's through
/// this expression on all 2 249 candidates of the eight fixture dumps (step C3).
///
/// It is **not** bit-identical to the reference's *coordinates*, and step D2 measured that rather
/// than leaving the old claim standing: numpy's `@` is a BLAS call and BLAS on this machine fuses,
/// so the reference accumulates `((c₀x) ⊕ c₁y) ⊕ c₂z` with three roundings instead of five and the
/// two answers part company in the last bits (3 ULP on a 300-unit coordinate, measured over 900
/// coordinates in the note of step D2). Where a *coordinate* has to come back the same — a placed
/// mesh written to PLY, a cloud handed to ICP, a preview pixel — the port calls
/// [`apply_transform_fused`] instead. This function stays as it is because R §6's numbers were
/// verified through it and because a score is a threshold on a distance, not a coordinate.
#[inline]
pub fn apply_transform(t: &nalgebra::Matrix4<f64>, p: [f64; 3]) -> [f64; 3] {
    [
        t[(0, 0)] * p[0] + t[(0, 1)] * p[1] + t[(0, 2)] * p[2] + t[(0, 3)],
        t[(1, 0)] * p[0] + t[(1, 1)] * p[1] + t[(1, 2)] * p[2] + t[(1, 3)],
        t[(2, 0)] * p[0] + t[(2, 1)] * p[1] + t[(2, 2)] * p[2] + t[(2, 3)],
    ]
}

/// `R·p + τ` as **Eigen and OpenBLAS** compute it: three fused multiply-adds, then the
/// translation.
///
/// Open3D's `TriangleMesh::Transform` and `PointCloud::Transform` multiply the homogeneous
/// 4×4 by `(p, 1)`, which Eigen evaluates as a linear combination of the matrix's *columns* —
/// `res = c₀x`, `res = pmadd(c₁, y, res)`, `res = pmadd(c₂, z, res)`, `res = pmadd(c₃, 1, res)` —
/// and `pmadd` on this machine's NEON packets is `vfmaq_f64`, a fused multiply-add. numpy's
/// `P @ T[:3,:3].T + T[:3,3]` reaches the same three roundings through OpenBLAS's `dgemm`.
///
/// **Measured, not assumed** (step D2): over 900 coordinates of a random cloud 300 units from the
/// origin under a random pose, this expression reproduces Open3D's transformed vertices *and*
/// numpy's matmul **bit for bit, 900 of 900**, where the unfused [`apply_transform`] misses 291 of
/// them and the two other plausible FMA associations miss 320 and 440. That is what makes
/// R §11.4's `placed/<name>.ply` byte-identical to the reference's file rather than merely equal
/// to within a rounding.
///
/// `mul_add` is one instruction on aarch64 and on any x86-64 with FMA; on a target without it,
/// Rust calls libm's correctly rounded `fma`, so the answer does not change with the machine.
#[inline]
pub fn apply_transform_fused(t: &nalgebra::Matrix4<f64>, p: [f64; 3]) -> [f64; 3] {
    let row =
        |i: usize| t[(i, 2)].mul_add(p[2], t[(i, 1)].mul_add(p[1], t[(i, 0)] * p[0])) + t[(i, 3)];
    [row(0), row(1), row(2)]
}

/// The rotation block of [`apply_transform_fused`] alone — Open3D's `TransformNormals`, whose
/// homogeneous fourth component is `0` rather than `1`, and the reference's `FN @ T[:3,:3].T`.
#[inline]
pub fn rotate_fused(t: &nalgebra::Matrix4<f64>, p: [f64; 3]) -> [f64; 3] {
    let row = |i: usize| t[(i, 2)].mul_add(p[2], t[(i, 1)].mul_add(p[1], t[(i, 0)] * p[0]));
    [row(0), row(1), row(2)]
}

#[cfg(test)]
mod tests {
    use super::{Cloud, FaceLabel, WorkingMesh};
    use crate::vec3::vec3;

    #[test]
    fn face_labels_are_one_byte_and_shell_is_zero() {
        assert_eq!(size_of::<FaceLabel>(), 1);
        assert_eq!(FaceLabel::default(), FaceLabel::Shell);
        assert_eq!(FaceLabel::Shell as u8, 0);
        assert_eq!(FaceLabel::Fracture as u8, 1);
        assert!(FaceLabel::Fracture.is_fracture());
        assert!(!FaceLabel::Shell.is_fracture());
        for label in [FaceLabel::Shell, FaceLabel::Fracture, FaceLabel::Solid, FaceLabel::Rim] {
            assert_eq!(FaceLabel::from_u8(label as u8), Some(label));
        }
        assert_eq!(FaceLabel::from_u8(4), None);
    }

    #[test]
    fn empty_containers_report_themselves() {
        let m = WorkingMesh::default();
        assert!(m.is_empty());
        assert_eq!(m.n_faces(), 0);
        assert_eq!(m.n_vertices(), 0);
        let c = Cloud::default();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        let c = Cloud { p: vec![vec3(0.0, 0.0, 0.0)], n: vec![vec3(0.0, 0.0, 1.0)] };
        assert_eq!(c.len(), 1);
    }
}
