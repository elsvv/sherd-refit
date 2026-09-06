//! A fragment: everything derived from one file (R §3).
//!
//! Thickness (§3.2) sets the face budget and every scale-free threshold; the working mesh (§3.3)
//! is what the segmentation (§3.4) labels; the breaklines (§3.5.5) and the match arrays
//! (§3.5–3.6) are what the matcher compares. All of it is cached per fragment (§3.7, D §4.2), so
//! a rerun on the same collection starts at the matching stage.
//!
//! Step S3 fills in the preprocessing up to the working mesh — [`Fragment::from_mesh_file`] and
//! everything it calls; step S4 the cache ([`cache`], [`Fragment::load_or_build`]); step B1 the
//! shell/fracture segmentation ([`segment`]), whose labels are the second tensor of the cache;
//! step B2 the breaklines and their frames ([`breakline`]), which are the next five; step B3 the
//! sampled match arrays of R §3.5.1–3.5.2 and §3.5.6 ([`samples`]), which are the last five and
//! which complete R §3. What a pair actually reads is
//! [`samples::MatchData`](samples::MatchData), built on those arrays and never stored.

pub mod breakline;
pub mod cache;
pub mod features;
pub mod samples;
pub mod segment;
pub mod thickness;

use std::path::Path;

use crate::error::{Error, Result};
use crate::fragment::breakline::{Breaklines, BrkParams};
use crate::fragment::samples::{SampleParams, Samples};
use crate::mesh::adjacency::closed_enough;
use crate::mesh::clean::{remove_degenerate_triangles, remove_unreferenced_vertices};
use crate::mesh::decimate::{decimate, face_budget};
use crate::mesh::geometry::{FaceGeometry, face_geometry, median_edge};
use crate::mesh::taubin::taubin;
use crate::spatial::bvh::RayScene;
use crate::types::{FaceLabel, FragId, SourceRef, WorkingMesh};
use crate::vec3::Vec3f;

/// One fragment of a collection, at the state plan step B3 leaves it in.
///
/// D §4.1 defines the finished struct; the fields below are everything R §3 produces — the mesh,
/// its wall, the shell/fracture labels the matcher works on, the breakline it anchors its
/// hypotheses on and the samples every score is measured over. The shared BVHs of R §6.1 and
/// §6.4 join them in phase 1c.
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
    /// Shell or fracture, one per face of `mesh`, in face order (R §3.4).
    pub labels: Vec<FaceLabel>,
    /// The breakline points, their frames and the hypothesis subset (R §3.5.3–3.5.5), built at
    /// this fragment's own `thick`.
    pub brk: Breaklines,
    /// The whole-surface, fracture and shell-margin samples (R §3.5.1–3.5.2, §3.5.6), built at
    /// this fragment's own `thick`.
    pub samples: Samples,
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
    /// Total area of the **working** mesh (R §3.4's `area`).
    pub area: f64,
    /// Area of the faces labelled fracture (R §3.4's `fracture_area`) — what R §6.1's `contact`
    /// scales by, and what R §3.5.2's sample count is a density over.
    pub frac_area: f64,
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

        // R §3.4 and R §3.5.3–3.5.5 both run on the working mesh's `f64` geometry, and both must
        // run on the geometry derived from the **narrowed** vertices, for the reason
        // `WorkingMesh::from_parts` derives its own arrays from them: a fragment computed from the
        // file and the same fragment read back from the cache have to be the same fragment.
        let v64: Vec<[f64; 3]> = working.v.iter().map(|p| p.to_f64()).collect();
        let geom = face_geometry(&v64, &working.f);
        let seg = segment_working_mesh(&working, &v64, &geom, thick, res, name);
        let brk = breaklines_of(&v64, &working.f, &geom, &seg.labels, thick, name);
        let samples =
            samples_of(&v64, &working.f, &geom, &seg.labels, &brk, SampleParams::at(thick), name);

        Ok(Self {
            id: 0,
            name: name.to_owned(),
            source,
            mesh: working,
            labels: seg.labels,
            brk,
            samples,
            thick,
            thick_mode,
            watertight,
            n_boundary: u32::try_from(n_boundary).unwrap_or(u32::MAX),
            n_orig_vertices,
            n_orig_faces,
            target_faces: u32::try_from(target_faces).unwrap_or(u32::MAX),
            face_budget: u32::try_from(budget).unwrap_or(u32::MAX),
            area0,
            area: seg.area,
            frac_area: seg.frac_area,
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
            && let Some(mut fragment) = cache::load_valid(cache, path, cap, name)
        {
            tracing::debug!(fragment = name, cache = %cache.display(), "cache hit");
            // R §3.7: a cache that is valid but was built with other match-array parameters has
            // those arrays recomputed, not the whole fragment. In phase 1 the knobs are
            // constants, so this can only fire on a cache written by another build.
            let stale_brk = fragment.brk.params != BrkParams::at(fragment.thick);
            let stale_md = stale_brk || fragment.samples.params != SampleParams::at(fragment.thick);
            if stale_brk {
                tracing::info!(
                    fragment = name,
                    "the cached breaklines were built with other parameters; recomputing them"
                );
                fragment.rebuild_breaklines();
            }
            if stale_md {
                tracing::info!(
                    fragment = name,
                    "the cached match arrays were built with other parameters; recomputing them"
                );
                fragment.rebuild_samples();
            }
            if (stale_brk || stale_md)
                && let Err(e) = cache::write(&fragment, cache)
            {
                tracing::warn!(fragment = name, "the cache could not be updated: {e}");
            }
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

    /// R §3.5.3–3.5.5 again, at this fragment's own `thick` and the shipped knobs.
    ///
    /// The one caller is [`Fragment::load_or_build`], for a cache whose `brk_params` are not this
    /// run's; a fragment built from its file already has them.
    pub fn rebuild_breaklines(&mut self) {
        let v64: Vec<[f64; 3]> = self.mesh.v.iter().map(|p| p.to_f64()).collect();
        let geom = face_geometry(&v64, &self.mesh.f);
        self.brk =
            breakline::build(&v64, &self.mesh.f, &geom, &self.labels, BrkParams::at(self.thick));
    }

    /// R §3.5.1–3.5.2 and §3.5.6 again, at this fragment's own `thick` and the shipped knobs.
    ///
    /// The one caller is [`Fragment::load_or_build`], for a cache whose `md_params` are not this
    /// run's. The breaklines are read from the fragment as they stand, so a run that has to
    /// rebuild both rebuilds them first — the samples measure `d_brk` against them.
    pub fn rebuild_samples(&mut self) {
        let v64: Vec<[f64; 3]> = self.mesh.v.iter().map(|p| p.to_f64()).collect();
        let geom = face_geometry(&v64, &self.mesh.f);
        self.samples = samples::build(
            &v64,
            &self.mesh.f,
            &geom,
            &self.labels,
            &self.brk.points_f64(),
            SampleParams::at(self.thick),
        );
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

    /// Total area of the faces labelled [`Fracture`](FaceLabel::Fracture) (R §3.4).
    ///
    /// Summed in `f64` over the face areas the segmentation itself used — not over the `f32` ones
    /// the working mesh stores — when the fragment was built, and carried in the cache since, so
    /// that reading it costs nothing on either path.
    #[inline]
    pub fn fracture_area(&self) -> f64 {
        self.frac_area
    }

    /// Fracture area over total area — the number the report and the parity table carry.
    #[inline]
    pub fn fracture_fraction(&self) -> f64 {
        if self.area <= 0.0 { 0.0 } else { self.frac_area / self.area }
    }
}

/// What [`segment_working_mesh`] hands back: R §3.4's labels and the two areas it summed on the
/// way, which every later stage would otherwise recompute (R §3.5.2, R §6.1).
struct Labelled {
    labels: Vec<FaceLabel>,
    area: f64,
    frac_area: f64,
}

/// R §3.4 for one working mesh: the shell/fracture label of every face.
///
/// The segmentation runs on the **narrowed** working mesh, for the same reason
/// [`WorkingMesh::from_parts`] derives its per-face arrays from it: a fragment computed from the
/// file and the same fragment read back from the cache must be the same fragment, and the cache
/// stores `f32` vertices. The `f64` geometry below is therefore the working mesh's own,
/// recomputed the way `from_parts` recomputes it — the reference's arithmetic, over vertices that
/// have been through an `f32`.
///
/// A mesh with no triangle gets no labels; nothing downstream reads them, and `RayScene` refuses
/// such a mesh anyway.
fn segment_working_mesh(
    working: &WorkingMesh,
    v64: &[[f64; 3]],
    geom: &FaceGeometry,
    thick: f64,
    res: f64,
    name: &str,
) -> Labelled {
    let Some(scene) = RayScene::new(v64, &working.f) else {
        return Labelled { labels: Vec::new(), area: 0.0, frac_area: 0.0 };
    };
    let started = std::time::Instant::now();
    let seg = segment::segment_faces(
        &scene,
        &working.f,
        geom,
        thick,
        res,
        &segment::SegParams::default(),
    );
    tracing::info!(
        fragment = name,
        raw = seg.raw_fraction,
        fracture = seg.fracture_fraction,
        votes = seg.votes,
        boundary_angle = seg.boundary_angle,
        seconds = started.elapsed().as_secs_f64(),
        "segmentation"
    );
    Labelled { labels: seg.labels, area: seg.area, frac_area: seg.fracture_area }
}

/// R §3.5.3–3.5.5 for one labelled working mesh, with the log line the pipeline prints.
///
/// A mesh with no label — the empty mesh `segment_working_mesh` refuses — has no breakline
/// either, and `build` would index the labels it does not have.
fn breaklines_of(
    v64: &[[f64; 3]],
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    labels: &[FaceLabel],
    thick: f64,
    name: &str,
) -> Breaklines {
    let params = BrkParams::at(thick);
    if labels.len() != f.len() {
        return Breaklines { params, ..Breaklines::default() };
    }
    let started = std::time::Instant::now();
    let brk = breakline::build(v64, f, geom, labels, params);
    let valid = brk.valid().iter().filter(|&&v| v).count();
    tracing::info!(
        fragment = name,
        points = brk.len(),
        valid,
        subsampled = brk.sub.len(),
        seconds = started.elapsed().as_secs_f64(),
        "breaklines"
    );
    brk
}

/// R §3.5.1–3.5.2 and §3.5.6 for one labelled working mesh, with the log line the pipeline prints.
///
/// A mesh whose labels do not describe it — the empty mesh `segment_working_mesh` refuses — is
/// sampled as nothing at all, for the same reason `breaklines_of` refuses it.
fn samples_of(
    v64: &[[f64; 3]],
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    labels: &[FaceLabel],
    brk: &Breaklines,
    params: SampleParams,
    name: &str,
) -> Samples {
    if labels.len() != f.len() {
        return Samples { params, ..Samples::default() };
    }
    let started = std::time::Instant::now();
    let md = samples::build(v64, f, geom, labels, &brk.points_f64(), params);
    tracing::info!(
        fragment = name,
        surface = md.n_surface(),
        fracture = md.n_fracture(),
        margin = md.n_margin(),
        seconds = started.elapsed().as_secs_f64(),
        "match arrays"
    );
    md
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
