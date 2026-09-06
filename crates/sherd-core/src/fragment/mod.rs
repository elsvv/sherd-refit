//! A fragment: everything derived from one file (R §3).
//!
//! Thickness (§3.2) sets the face budget and every scale-free threshold; the working mesh (§3.3)
//! is what the segmentation (§3.4) labels; the breaklines (§3.5.5) and the match arrays
//! (§3.5–3.6) are what the matcher compares. All of it is cached per fragment (§3.7, D §4.2), so
//! a rerun on the same collection starts at the matching stage.
//!
//! Step S3 fills in the preprocessing up to the working mesh — [`Fragment::from_mesh_file`] and
//! everything it calls; step S4 the cache ([`cache`], [`Fragment::load_or_build`]).
//! Segmentation, breaklines and the match arrays follow in phase 1b, each adding its own field to
//! [`Fragment`] and its own tensor to the cache.

pub mod breakline;
pub mod cache;
pub mod features;
pub mod samples;
pub mod segment;
pub mod thickness;

use std::path::Path;

use crate::error::{Error, Result};
use crate::mesh::adjacency::closed_enough;
use crate::mesh::clean::{remove_degenerate_triangles, remove_unreferenced_vertices};
use crate::mesh::decimate::{decimate, face_budget};
use crate::mesh::geometry::{face_geometry, median_edge};
use crate::mesh::taubin::taubin;
use crate::types::{FragId, SourceRef, WorkingMesh};
use crate::vec3::Vec3f;

/// One fragment of a collection, at the state plan step S3 leaves it in.
///
/// D §4.1 defines the finished struct; the fields below are the ones R §3.1–3.3 produce, which is
/// everything the matcher needs *about the mesh itself*. `labels` (R §3.4), the match arrays
/// (R §3.5) and the shared BVHs join them in phase 1b, and the cache that stores all of it in
/// step S4.
#[derive(Clone, Debug)]
pub struct Fragment {
    /// Index inside the collection, in the order R §2 discovers the files. `from_mesh_file`
    /// leaves it 0; the collection loader of phase 1d assigns it.
    pub id: FragId,
    /// The file's stem, which every report and output file is named after.
    pub name: String,
    /// Where the mesh came from, and enough metadata to invalidate a cache entry (R §3.7).
    pub source: SourceRef,
    /// The decimated, smoothed mesh with its per-face geometry and `res` (R §3.3).
    pub mesh: WorkingMesh,
    /// Wall thickness `t` (R §3.2) — the unit every scale-free threshold of R §1.2 is in.
    pub thick: f64,
    /// The unfiltered ray mode, reported beside `thick` so a fragment whose two values disagree
    /// (a rim, a collar) is visible in the report.
    pub thick_mode: f64,
    /// R §3.3.2: closed enough for a signed distance to be trusted. A fragment that fails this
    /// gets no penetration test (R §6.4).
    pub watertight: bool,
    /// Unique edges of the working mesh used by a number of faces other than two.
    pub n_boundary: u32,
    /// Vertices after cleaning and before the largest-component pass (R §3.1 step 3).
    pub n_orig_vertices: u32,
    /// Triangles after cleaning and before the largest-component pass.
    pub n_orig_faces: u32,
    /// The `--target-faces` cap this fragment was built with (part of the cache key, R §3.7).
    pub target_faces: u32,
    /// The adaptive budget R §3.3 actually asked the decimator for.
    pub face_budget: u32,
    /// Total area of the largest component *before* decimation — the numerator of the budget.
    pub area0: f64,
}

impl Fragment {
    /// R §3.1–3.3 for one file: load, clean, keep the largest component, measure the wall,
    /// decimate to the adaptive budget, smooth, and derive the per-face geometry.
    ///
    /// The name is the file's stem, as `sherd_refit.fragment.Fragment.from_mesh_file` takes it.
    ///
    /// The order is the reference's, and two things about it are load-bearing:
    ///
    /// * the thickness is measured on the **original** largest component, before decimation,
    ///   because it is what *sets* the face budget;
    /// * `n_orig_vertices` / `n_orig_faces` are counted after cleaning but before the
    ///   largest-component pass, so a fragment that arrives as two shells still reports what the
    ///   file held.
    pub fn from_mesh_file(path: impl AsRef<Path>, target_faces: usize) -> Result<Self> {
        let path = path.as_ref();
        let name = path.file_stem().map_or_else(
            || path.to_string_lossy().into_owned(),
            |s| s.to_string_lossy().into_owned(),
        );
        Self::from_mesh_file_named(path, target_faces, &name)
    }

    /// [`Fragment::from_mesh_file`] with the fragment's name given explicitly, which is how the
    /// pipeline calls it (two files in a collection can share a stem).
    pub fn from_mesh_file_named(
        path: impl AsRef<Path>,
        target_faces: usize,
        name: &str,
    ) -> Result<Self> {
        let path = path.as_ref();
        let source = source_ref(path)?;
        let mut mesh = crate::io::load_mesh(path)?;

        let n_orig_vertices = u32::try_from(mesh.v.len()).unwrap_or(u32::MAX);
        let n_orig_faces = u32::try_from(mesh.f.len()).unwrap_or(u32::MAX);
        crate::mesh::components::largest_component(&mut mesh);

        // --- R §3.2: the wall, measured on the original component ------------------------------
        let geom0 = face_geometry(&mesh.v, &mesh.f);
        let area0 = geom0.total_area();
        let mut rng = crate::rng::seeded(thickness::SEED);
        let estimate = thickness::estimate_thickness(&mesh.v, &mesh.f, &geom0, &mut rng);
        let (thick, raw_mode) = match estimate {
            Some((t, m)) if t > 0.0 => (f64::from(t), f64::from(m)),
            other => {
                let fallback = thickness::obb_min_extent(&mesh.v) / 10.0;
                tracing::warn!(
                    fragment = name,
                    thickness = fallback,
                    "thickness estimate failed, using the OBB fallback"
                );
                (fallback, other.map_or(0.0, |(_, m)| f64::from(m)))
            }
        };
        let thick_mode = if raw_mode == 0.0 { thick } else { raw_mode };
        if thick_mode > 1.15 * thick {
            tracing::info!(
                fragment = name,
                thick,
                thick_mode,
                "the plain ray mode disagrees with the wall -- a rim or a collar"
            );
        }

        // --- R §3.3: the working mesh ----------------------------------------------------------
        let budget = face_budget(area0, thick, target_faces);
        drop(geom0);
        decimate(&mut mesh, budget);
        taubin(&mut mesh);
        remove_degenerate_triangles(&mut mesh);
        remove_unreferenced_vertices(&mut mesh);

        let (watertight, n_boundary) = closed_enough(&mesh.f);
        if !watertight {
            tracing::warn!(
                fragment = name,
                n_boundary,
                "working mesh has boundary edges; penetration tests will be skipped for it"
            );
        }
        let res = median_edge(&mesh.v, &mesh.f);

        tracing::info!(
            fragment = name,
            faces = mesh.f.len(),
            from = n_orig_faces,
            budget,
            thick,
            res,
            watertight,
            "working mesh"
        );

        // The per-face arrays are derived by `WorkingMesh::from_parts` from the *narrowed*
        // vertices, not from `mesh.v`, so that this fragment and the same fragment read back from
        // the cache (`cache`, D §4.2) are bit-identical: the cache stores `V`, `F` and `res`, and
        // both paths have to derive the rest the same way. `geom` above is R §3.2's, over the
        // original component.
        #[allow(clippy::cast_possible_truncation, reason = "the working mesh is f32 (D §4.1, §7)")]
        let working = WorkingMesh::from_parts(
            mesh.v.iter().copied().map(Vec3f::from_f64).collect(),
            mesh.f,
            res as f32,
        );

        Ok(Self {
            id: 0,
            name: name.to_owned(),
            source,
            mesh: working,
            thick,
            thick_mode,
            watertight,
            n_boundary: u32::try_from(n_boundary).unwrap_or(u32::MAX),
            n_orig_vertices,
            n_orig_faces,
            target_faces: u32::try_from(target_faces).unwrap_or(u32::MAX),
            face_budget: u32::try_from(budget).unwrap_or(u32::MAX),
            area0,
        })
    }

    /// R §3.7's cache path through [`Fragment::from_mesh_file_named`]: the cached fragment when
    /// a valid cache describes this file, otherwise the file itself, with the result written to
    /// the cache.
    ///
    /// Returns the fragment and whether it came from the cache. Passing `None` for `cache`
    /// bypasses the cache in both directions, which is what `--no-cache` and the parity harness
    /// want. A cache that cannot be written is *not* an error — the fragment is already computed
    /// and the run continues without it — but it is logged.
    pub fn load_or_build(
        path: impl AsRef<Path>,
        target_faces: usize,
        name: &str,
        cache: Option<&Path>,
    ) -> Result<(Self, bool)> {
        let path = path.as_ref();
        let cap = u32::try_from(target_faces).unwrap_or(u32::MAX);
        if let Some(cache) = cache
            && let Some(fragment) = cache::load_valid(cache, path, cap, name)
        {
            tracing::debug!(fragment = name, cache = %cache.display(), "cache hit");
            return Ok((fragment, true));
        }
        let fragment = Self::from_mesh_file_named(path, target_faces, name)?;
        if let Some(cache) = cache
            && let Err(e) = cache::write(&fragment, cache)
        {
            tracing::warn!(fragment = name, "the cache could not be written: {e}");
        }
        Ok((fragment, false))
    }

    /// Number of faces of the working mesh.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.mesh.n_faces()
    }

    /// Number of vertices of the working mesh.
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.mesh.n_vertices()
    }

    /// `res` of the working mesh (R §0), as the `f64` every threshold of R §1.2 mixes with `t`.
    #[inline]
    pub fn res(&self) -> f64 {
        f64::from(self.mesh.res)
    }
}

/// The file's identity for the cache validity rule of R §3.7.
///
/// A missing modification time (some network filesystems) becomes 0 rather than an error: the
/// cache then simply never matches, which is the safe direction.
fn source_ref(path: &Path) -> Result<SourceRef> {
    let meta = std::fs::metadata(path).map_err(|e| Error::read(path, e))?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| i128::try_from(d.as_nanos()).unwrap_or(0));
    Ok(SourceRef { path: path.to_path_buf(), size: meta.len(), mtime_ns, sha256: None })
}
