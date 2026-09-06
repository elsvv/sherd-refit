//! Turning a scan into the working mesh (R §3.3).
//!
//! The order is the reference's: clean (duplicate vertices, degenerate faces, unreferenced
//! vertices), keep the largest connected component, decimate to the adaptive face budget when the
//! mesh is above it, Taubin-smooth, then derive the per-face arrays and `res`.
//!
//! [`clean`] and [`components`] are filled in by plan step S2 (they belong to R §3.1, the load
//! stage, and the IO tests exercise them); [`decimate`], [`taubin`], [`geometry`] and
//! [`adjacency`] by step S3.

pub mod adjacency;
pub mod clean;
pub mod components;
pub mod decimate;
pub mod geometry;
pub mod taubin;

/// A triangle mesh exactly as it comes off disk, before decimation (R §3.1's `(V0, F0)`).
///
/// This is the reference's `open3d.geometry.TriangleMesh` reduced to what the pipeline uses:
/// vertices, triangles and the optional per-vertex colours, which R §3.1 carries to the outputs
/// and nowhere else. Coordinates are `f64` because the reference's are (`np.asarray(m.vertices,
/// dtype=np.float64)`) and because thickness, area and the face budget of R §3.3 are computed
/// from them before anything is narrowed to `f32`.
///
/// Colours are stored as the three bytes a PLY carries rather than as Open3D's `[0, 1]` doubles:
/// every reader and the writer of R §11.4 speak `uchar`, and quantising once at read time makes
/// the output byte-identical to the reference's (see [`quantize_color`](crate::io::quantize_color)).
///
/// Invariants, upheld by every reader and by [`clean`]:
/// * every index in [`f`](Mesh::f) is `< v.len()`;
/// * `colors`, when present, has exactly one entry per vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertices.
    pub v: Vec<[f64; 3]>,
    /// Triangles, as indices into [`v`](Mesh::v).
    pub f: Vec<[u32; 3]>,
    /// Per-vertex RGB, or `None` when the file carries no colours.
    pub colors: Option<Vec<[u8; 3]>>,
}

impl Mesh {
    /// A mesh with no colours.
    pub fn new(v: Vec<[f64; 3]>, f: Vec<[u32; 3]>) -> Self {
        Self { v, f, colors: None }
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

    /// True when the mesh carries no triangle — what R §3.1 step 1 rejects.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.f.is_empty()
    }

    /// True when the mesh carries per-vertex colours.
    #[inline]
    pub fn has_colors(&self) -> bool {
        self.colors.is_some()
    }

    /// Checks the invariants a reader must produce: colours one per vertex, indices in range.
    ///
    /// Called by every reader, so that a malformed file becomes an [`Error`](crate::Error) rather
    /// than a panic several stages later.
    pub fn validate(&self, path: &std::path::Path) -> crate::Result<()> {
        if let Some(c) = &self.colors
            && c.len() != self.v.len()
        {
            return Err(crate::Error::read(
                path,
                format!("{} colours for {} vertices", c.len(), self.v.len()),
            ));
        }
        let n = self.v.len();
        for (i, t) in self.f.iter().enumerate() {
            for &k in t {
                if k as usize >= n {
                    return Err(crate::Error::read(
                        path,
                        format!("triangle {i} refers to vertex {k}, but the mesh has {n}"),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Mesh;
    use std::path::Path;

    #[test]
    fn an_empty_mesh_reports_itself() {
        let m = Mesh::default();
        assert!(m.is_empty());
        assert!(!m.has_colors());
        assert_eq!((m.n_vertices(), m.n_faces()), (0, 0));
        m.validate(Path::new("x.ply")).expect("an empty mesh is valid");
    }

    #[test]
    fn validation_catches_a_short_colour_list_and_a_stray_index() {
        let mut m = Mesh::new(vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], vec![[0, 1, 2]]);
        m.validate(Path::new("x.ply")).expect("a well-formed mesh is valid");

        m.colors = Some(vec![[0, 0, 0]]);
        let e = m.validate(Path::new("x.ply")).unwrap_err().to_string();
        assert!(e.contains("1 colours for 3 vertices"), "{e}");

        m.colors = None;
        m.f = vec![[0, 1, 7]];
        let e = m.validate(Path::new("x.ply")).unwrap_err().to_string();
        assert!(e.contains("refers to vertex 7"), "{e}");
    }
}
