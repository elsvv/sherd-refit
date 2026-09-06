//! The fragment cache `<out>/cache/<name>.sherd` (R §3.7, D §4.2).
//!
//! Everything R §3 derives from one file is a pure function of `(file, target_faces, the sampling
//! fields of Params, seed)`, so it is computed once and stored: a rerun on the same collection
//! starts at the matching stage. The reference does this with a compressed `.npz`; this port uses
//! a [`safetensors`] file, because its header is a single JSON object that can be read without
//! touching the tensor data, and because the Python side can read the tensors back
//! (`safetensors.numpy.load_file`) and the metadata (`safetensors.safe_open(...).metadata()`)
//! during the transition.
//!
//! # What is in the file
//!
//! | tensor | dtype | shape | R |
//! |---|---|---|---|
//! | `V` | f32 | `[n, 3]` | §3.3 working-mesh vertices |
//! | `F` | u32 | `[m, 3]` | §3.3 working-mesh triangles |
//! | `labels` | u8 | `[m]` | §3.4 shell (0) or fracture (1) per face |
//! | `brk_P` | f32 | `[k, 3]` | §3.5.3 breakline points |
//! | `brk_ns`, `brk_nf`, `brk_f` | f32 | `[k, 3]` | §3.5.4 macro normals and the in-plane axis |
//! | `brk_sub` | u32 | `[j]` | §3.5.5 hypothesis subset |
//! | `S` | f32 | `[n_s, 3]` | §3.5.1 whole-surface samples |
//! | `sp` | u32 | `[n_s]` | the face each surface sample came from |
//! | `Pf` | f32 | `[n_f, 3]` | §3.5.2 fracture samples |
//! | `fp` | u32 | `[n_f]` | the face each fracture sample came from |
//! | `margin_idx` | u32 | `[n_m]` | §3.5.6 shell margin, indices into `S` |
//!
//! D §4.2 lists the optional `features/*` beside them. Those stages are still ahead; the reader
//! treats an unknown tensor as data it does not need yet and the writer adds them when they exist,
//! so a newer cache does not confuse this build. A tensor the port *needs* is a different matter,
//! and moving [`crate::CACHE_VERSION`] is how it is announced: `labels` took it from 1 to 2 in
//! step B1, the five `brk_*` tensors from 2 to 3 in step B2 and the five sampled arrays from 3 to
//! 4 in step B3, so a cache written before any of them existed is refused by its version rather
//! than read back half empty.
//!
//! The `brk_*` tensors are `f32` because [`Breaklines`](crate::fragment::breakline::Breaklines)
//! is (D §4.1), so the arrays a warm run reads back are the arrays a cold run computed, bit for
//! bit — the same rule `V` and `WorkingMesh` follow.
//!
//! Everything else is metadata: the source's identity for the validity rule, the scalars of
//! R §3.2–3.3, and the version triple of D §4.3.
//!
//! # Why the metadata is one key
//!
//! D §4.2 asks for a flat string map (`format`, `cache_version`, …). `safetensors` 0.8 takes that
//! map as a `std::collections::HashMap` and serialises it in iteration order, and `HashMap`'s
//! iteration order is randomised per map instance: serialising the same twenty-key map four times
//! **inside one process** gave four different headers (measured, S4 note §2). A cache written
//! twice from the same input would then differ byte for byte, which is exactly what plan step S4
//! has to rule out. So the whole metadata block travels as one JSON object under the single key
//! `sherd` ([`METADATA_KEY`]), serialised by `serde_json`, which writes a struct's fields in
//! declaration order and a map's keys sorted — deterministic either way. The field names inside
//! it are D §4.2's, unchanged.
//!
//! `created` from D §4.2's list is deliberately **not** written, for the same reason: a timestamp
//! makes two runs of the same input produce different files.

use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};
use serde::{Deserialize, Serialize};

use super::Fragment;
use super::breakline::{Breaklines, BrkParams};
use super::samples::{SampleParams, Samples};
use crate::error::{Error, Result};
use crate::types::{FaceLabel, SourceRef, WorkingMesh};
use crate::vec3::Vec3f;
use crate::{ALGO_REF, CACHE_VERSION, CORE_VERSION};

/// The `format` field every cache file carries.
pub const FORMAT: &str = "sherd-cache";

/// File extension of a fragment cache, without the dot.
pub const EXTENSION: &str = "sherd";

/// The single `__metadata__` key the JSON block of [`CacheMeta`] travels under.
pub const METADATA_KEY: &str = "sherd";

/// How far a source file's modification time may have moved before the cache is stale, in
/// nanoseconds. R §3.7's rule is `|mtime_cached − mtime_file| < 1 s`, kept as it is: a file
/// restored from a backup or copied across a filesystem with a coarser clock must not silently
/// reuse a cache, and a sub-second jitter must not throw one away.
pub const MTIME_TOLERANCE_NS: i128 = 1_000_000_000;

/// `<out>/cache/<name>.sherd` (D §4.2).
pub fn cache_path(out_dir: impl AsRef<Path>, name: &str) -> PathBuf {
    out_dir.as_ref().join("cache").join(format!("{name}.{EXTENSION}"))
}

/// The metadata block of a cache file: D §4.2's string map, as one JSON object.
///
/// Floats are written by `serde_json`'s shortest-round-trip formatter and read back with the
/// `float_roundtrip` feature the workspace pins, so `thick`, `thick_mode`, `res` and `area0`
/// survive the round trip bit for bit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheMeta {
    /// Always [`FORMAT`]; a file whose `format` differs is not ours.
    pub format: String,
    /// Layout version of the file ([`crate::CACHE_VERSION`]).
    pub cache_version: u32,
    /// The frozen algorithm the contents were computed by ([`crate::ALGO_REF`], D §4.3).
    pub algo_ref: String,
    /// Version of `sherd-core` that wrote the file; informational (D §4.3).
    pub core_version: String,
    /// Fragment name — the collection's, which is not always the file's stem (R §2).
    pub name: String,
    /// Absolute path of the source file, as R §3.7's validity rule compares it.
    pub source_path: String,
    /// Size of the source file in bytes when it was read.
    pub source_size: u64,
    /// Modification time of the source file, nanoseconds since the Unix epoch.
    pub source_mtime_ns: i128,
    /// Content hash of the source, when the caller asked for one; lower-case hexadecimal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    /// The `--target-faces` cap the fragment was built with (part of the validity rule).
    pub target_faces: u32,
    /// The adaptive budget R §3.3 asked the decimator for. Additive to D §4.2: the fixture sink
    /// dumps it as `thick.target.target` and the parity harness compares it.
    pub face_budget: u32,
    /// Total area of the largest component before decimation. Additive to D §4.2, for the same
    /// reason (`thick.target.area0`).
    pub area0: f64,
    /// Wall thickness `t` (R §3.2).
    pub thick: f64,
    /// The unfiltered ray mode (R §3.2).
    pub thick_mode: f64,
    /// Median unique-edge length of the working mesh (R §0).
    pub res: f64,
    /// R §3.3.2's verdict.
    pub watertight: bool,
    /// Unique edges used by a number of faces other than two.
    pub n_boundary: u32,
    /// Total area of the working mesh (R §3.4's `area`), summed in `f64` where the segmentation
    /// summed it. Additive to D §4.2, so that neither `stats()` nor R §3.5.2's sample count has to
    /// recompute the face geometry on a cache hit.
    pub area: f64,
    /// Area of the faces labelled fracture (R §3.4's `fracture_area`), likewise.
    pub frac_area: f64,
    /// Vertices after cleaning, before the largest-component pass (R §3.1).
    pub n_orig_vertices: u32,
    /// Triangles after cleaning, before the largest-component pass.
    pub n_orig_faces: u32,
    /// Which executor produced the file (D §4.3). Preprocessing is CPU-only in phase 1.
    pub backend: String,
    /// The parameters the five `brk_*` tensors were built with — the breakline half of R §3.7's
    /// `mdp_*`. A cache whose `brk_params` differ from the run's has its breaklines recomputed and
    /// nothing else ([`Fragment::load_or_build`]), which is the reference's rule for the match
    /// arrays. Absent only in a cache written by a build that had no breaklines, which this
    /// build's `cache_version` refuses anyway.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brk_params: Option<BrkParams>,
    /// The parameters the five sampled tensors were built with — the rest of R §3.7's `mdp_*`.
    /// Same rule as `brk_params`: a cache whose `md_params` are not the run's has *those* arrays
    /// recomputed and rewritten, and nothing else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub md_params: Option<SampleParams>,
    /// Roadmap items 4 and 6 (D §11); absent in phase 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<serde_json::Value>,
}

impl CacheMeta {
    /// The metadata a fragment would be written with.
    pub fn of(fragment: &Fragment) -> Self {
        Self {
            format: FORMAT.to_owned(),
            cache_version: CACHE_VERSION,
            algo_ref: ALGO_REF.to_owned(),
            core_version: CORE_VERSION.to_owned(),
            name: fragment.name.clone(),
            source_path: absolute(&fragment.source.path).to_string_lossy().into_owned(),
            source_size: fragment.source.size,
            source_mtime_ns: fragment.source.mtime_ns,
            source_sha256: fragment.source.sha256.as_ref().map(hex),
            target_faces: fragment.target_faces,
            face_budget: fragment.face_budget,
            area0: fragment.area0,
            thick: fragment.thick,
            thick_mode: fragment.thick_mode,
            res: f64::from(fragment.mesh.res),
            watertight: fragment.watertight,
            n_boundary: fragment.n_boundary,
            area: fragment.area,
            frac_area: fragment.frac_area,
            n_orig_vertices: fragment.n_orig_vertices,
            n_orig_faces: fragment.n_orig_faces,
            backend: "cpu".to_owned(),
            brk_params: Some(fragment.brk.params),
            md_params: Some(fragment.samples.params),
            features: None,
        }
    }

    /// R §3.7's `cache_valid_for`, with D §4.3's two version fields in place of `CACHE_VERSION`.
    ///
    /// True when this cache describes `source` built with `target_faces` under `name`: the same
    /// absolute path, a file that still exists, a modification time within
    /// [`MTIME_TOLERANCE_NS`], the same face cap, the same name, and the same `cache_version` and
    /// `algo_ref` as this build. A cache that fails any of these is not an error — the caller
    /// recomputes the fragment.
    pub fn valid_for(&self, source: impl AsRef<Path>, target_faces: u32, name: &str) -> bool {
        let source = source.as_ref();
        if self.format != FORMAT
            || self.cache_version != CACHE_VERSION
            || self.algo_ref != ALGO_REF
            || self.name != name
            || self.target_faces != target_faces
        {
            return false;
        }
        if absolute(source).to_string_lossy() != self.source_path {
            return false;
        }
        let Ok(meta) = std::fs::metadata(source) else { return false };
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| i128::try_from(d.as_nanos()).unwrap_or(0));
        (mtime - self.source_mtime_ns).abs() < MTIME_TOLERANCE_NS
    }
}

/// Serialises a fragment to the bytes of a `.sherd` file.
///
/// The result depends on nothing but the fragment: no timestamp, no path of the cache itself, no
/// hash-map order (see the module documentation), so two runs on the same input produce the same
/// bytes.
pub fn to_bytes(fragment: &Fragment) -> Result<Vec<u8>> {
    let meta = CacheMeta::of(fragment);
    let json = serde_json::to_string(&meta).map_err(|e| {
        Error::cache(&fragment.source.path, format!("serialising the metadata: {e}"))
    })?;
    let data_info = std::collections::HashMap::from([(METADATA_KEY.to_owned(), json)]);

    let v: Vec<u8> =
        fragment.mesh.v.iter().flat_map(|p| p.to_array()).flat_map(f32::to_le_bytes).collect();
    let f: Vec<u8> = fragment.mesh.f.iter().flatten().copied().flat_map(u32::to_le_bytes).collect();
    let labels: Vec<u8> = fragment.labels.iter().map(|&l| l as u8).collect();
    let brk = &fragment.brk;
    let points = points_bytes(&brk.p);
    let ns = points_bytes(&brk.ns);
    let nf = points_bytes(&brk.nf);
    let axis = points_bytes(&brk.f);
    let sub: Vec<u8> = brk.sub.iter().copied().flat_map(u32::to_le_bytes).collect();
    let md = &fragment.samples;
    let surface = points_bytes(&md.s);
    let surface_faces: Vec<u8> = md.sp.iter().copied().flat_map(u32::to_le_bytes).collect();
    let fracture = points_bytes(&md.pf);
    let fracture_faces: Vec<u8> = md.fp.iter().copied().flat_map(u32::to_le_bytes).collect();
    let margin: Vec<u8> = md.margin_idx.iter().copied().flat_map(u32::to_le_bytes).collect();
    // Every shape is the array's own length rather than a shared `k`: the writer states what it
    // holds and the reader is what checks that the five hang together, which is the only order in
    // which a file written elsewhere is checked at all.
    let tensors = vec![
        ("F", TensorView::new(Dtype::U32, vec![fragment.mesh.f.len(), 3], &f)),
        ("Pf", TensorView::new(Dtype::F32, vec![md.pf.len(), 3], &fracture)),
        ("S", TensorView::new(Dtype::F32, vec![md.s.len(), 3], &surface)),
        ("V", TensorView::new(Dtype::F32, vec![fragment.mesh.v.len(), 3], &v)),
        ("brk_P", TensorView::new(Dtype::F32, vec![brk.p.len(), 3], &points)),
        ("brk_f", TensorView::new(Dtype::F32, vec![brk.f.len(), 3], &axis)),
        ("brk_nf", TensorView::new(Dtype::F32, vec![brk.nf.len(), 3], &nf)),
        ("brk_ns", TensorView::new(Dtype::F32, vec![brk.ns.len(), 3], &ns)),
        ("brk_sub", TensorView::new(Dtype::U32, vec![brk.sub.len()], &sub)),
        ("fp", TensorView::new(Dtype::U32, vec![md.fp.len()], &fracture_faces)),
        ("labels", TensorView::new(Dtype::U8, vec![fragment.labels.len()], &labels)),
        ("margin_idx", TensorView::new(Dtype::U32, vec![md.margin_idx.len()], &margin)),
        ("sp", TensorView::new(Dtype::U32, vec![md.sp.len()], &surface_faces)),
    ];
    let mut views = Vec::with_capacity(tensors.len());
    for (name, view) in tensors {
        let view = view.map_err(|e| Error::cache(&fragment.source.path, format!("{name}: {e}")))?;
        views.push((name.to_owned(), view));
    }
    safetensors::serialize(views, Some(data_info))
        .map_err(|e| Error::cache(&fragment.source.path, format!("serialising the cache: {e}")))
}

/// Writes `<out>/cache/<name>.sherd`, creating the directory when it does not exist.
///
/// The file is written to a temporary name in the same directory and renamed into place, so a
/// reader never sees a half-written cache and two workers cannot interleave.
pub fn write(fragment: &Fragment, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let bytes = to_bytes(fragment)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| Error::write(dir, e))?;
    }
    let tmp = path.with_extension(format!("{EXTENSION}.tmp{}", std::process::id()));
    std::fs::write(&tmp, &bytes).map_err(|e| Error::write(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| Error::write(path, e))?;
    Ok(())
}

/// Parses a `.sherd` file's bytes back into a fragment.
///
/// `path` is used only to name the file in error messages.
pub fn from_bytes(bytes: &[u8], path: impl AsRef<Path>) -> Result<Fragment> {
    let path = path.as_ref();
    let meta = meta_from_bytes(bytes, path)?;
    let file = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|e| Error::cache(path, format!("reading the tensors: {e}")))?;

    // Little-endian by the safetensors specification, and read as such rather than cast: the
    // buffer of a file read into a `Vec<u8>` carries no alignment guarantee.
    let v = read_points(&file, "V", path)?;
    let f = read_triangles(&file, "F", path)?;
    let n_v = u32::try_from(v.len()).unwrap_or(u32::MAX);
    if let Some(bad) = f.iter().flatten().find(|&&i| i >= n_v) {
        return Err(Error::cache(path, format!("triangle index {bad} is outside V ({n_v})")));
    }

    let l_view =
        file.tensor("labels").map_err(|e| Error::cache(path, format!("tensor labels: {e}")))?;
    if l_view.dtype() != Dtype::U8 || l_view.shape() != [f.len()] {
        return Err(Error::cache(
            path,
            format!(
                "tensor labels is {:?} {:?}, expected U8 [{}]",
                l_view.dtype(),
                l_view.shape(),
                f.len()
            ),
        ));
    }
    let labels = l_view
        .data()
        .iter()
        .map(|&b| FaceLabel::from_u8(b).ok_or_else(|| Error::cache(path, format!("label {b}"))))
        .collect::<Result<Vec<FaceLabel>>>()?;

    let brk_params = meta.brk_params.ok_or_else(|| Error::cache(path, "no brk_params"))?;
    let brk = Breaklines {
        params: brk_params,
        p: read_points(&file, "brk_P", path)?,
        ns: read_points(&file, "brk_ns", path)?,
        nf: read_points(&file, "brk_nf", path)?,
        f: read_points(&file, "brk_f", path)?,
        sub: read_u32(&file, "brk_sub", path)?,
    };
    if brk.ns.len() != brk.len() || brk.nf.len() != brk.len() || brk.f.len() != brk.len() {
        return Err(Error::cache(path, "the brk_* frames do not describe brk_P"));
    }
    let k = u32::try_from(brk.len()).unwrap_or(u32::MAX);
    if let Some(bad) = brk.sub.iter().find(|&&i| i >= k) {
        return Err(Error::cache(path, format!("brk_sub {bad} is outside brk_P ({k})")));
    }

    let md_params = meta.md_params.ok_or_else(|| Error::cache(path, "no md_params"))?;
    let samples = Samples {
        params: md_params,
        s: read_points(&file, "S", path)?,
        sp: read_u32(&file, "sp", path)?,
        pf: read_points(&file, "Pf", path)?,
        fp: read_u32(&file, "fp", path)?,
        margin_idx: read_u32(&file, "margin_idx", path)?,
    };
    // The three rules the writer cannot break but a file from elsewhere can: every sample has a
    // face, every face index is a face, and every margin index is a surface sample.
    if samples.sp.len() != samples.s.len() || samples.fp.len() != samples.pf.len() {
        return Err(Error::cache(path, "sp/fp do not describe S/Pf"));
    }
    let n_faces = u32::try_from(f.len()).unwrap_or(u32::MAX);
    if let Some(bad) = samples.sp.iter().chain(&samples.fp).find(|&&i| i >= n_faces) {
        return Err(Error::cache(path, format!("sample face {bad} is outside F ({n_faces})")));
    }
    let n_s = u32::try_from(samples.s.len()).unwrap_or(u32::MAX);
    if let Some(bad) = samples.margin_idx.iter().find(|&&i| i >= n_s) {
        return Err(Error::cache(path, format!("margin_idx {bad} is outside S ({n_s})")));
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "`res` was written from an f32 and round-trips exactly"
    )]
    let mesh = WorkingMesh::from_parts(v, f, meta.res as f32);
    Ok(Fragment {
        id: 0,
        name: meta.name,
        source: SourceRef {
            path: PathBuf::from(meta.source_path),
            size: meta.source_size,
            mtime_ns: meta.source_mtime_ns,
            sha256: meta.source_sha256.as_deref().and_then(unhex),
        },
        mesh,
        labels,
        brk,
        samples,
        thick: meta.thick,
        thick_mode: meta.thick_mode,
        watertight: meta.watertight,
        n_boundary: meta.n_boundary,
        n_orig_vertices: meta.n_orig_vertices,
        n_orig_faces: meta.n_orig_faces,
        target_faces: meta.target_faces,
        face_budget: meta.face_budget,
        area0: meta.area0,
        area: meta.area,
        frac_area: meta.frac_area,
    })
}

/// Reads `<out>/cache/<name>.sherd`.
///
/// D §4.2 sketched this as an mmap plus a header parse. `safetensors` 0.8 validates the header
/// against the *whole* buffer (`read_metadata` refuses a buffer that is not exactly header plus
/// data), so a header-only view is not available through the crate, and the file is read into
/// memory instead — a few megabytes per fragment. The mmap belongs with the memory work of phase
/// 1e, where the 170-scan budget of D §8 is measured rather than assumed.
pub fn read(path: impl AsRef<Path>) -> Result<Fragment> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::cache(path, e))?;
    from_bytes(&bytes, path)
}

/// Reads only the metadata block of a cache file — enough for the validity rule of R §3.7.
pub fn read_meta(path: impl AsRef<Path>) -> Result<CacheMeta> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::cache(path, e))?;
    meta_from_bytes(&bytes, path)
}

/// The fragment a valid cache holds, or `None` when there is no usable cache for this source.
///
/// A cache that is missing, unreadable, from another build or built from another file is not an
/// error here: R §3.7's rule is that the fragment is recomputed. The reason is logged at debug
/// level so a collection that recomputes everything on every run can be diagnosed.
pub fn load_valid(
    cache: impl AsRef<Path>,
    source: impl AsRef<Path>,
    target_faces: u32,
    name: &str,
) -> Option<Fragment> {
    let cache = cache.as_ref();
    let bytes = std::fs::read(cache).ok()?;
    let meta = match meta_from_bytes(&bytes, cache) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!(cache = %cache.display(), "{e}");
            return None;
        }
    };
    if !meta.valid_for(&source, target_faces, name) {
        tracing::debug!(cache = %cache.display(), "cache does not describe this file, recomputing");
        return None;
    }
    match from_bytes(&bytes, cache) {
        Ok(fragment) => Some(fragment),
        Err(e) => {
            tracing::debug!(cache = %cache.display(), "{e}");
            None
        }
    }
}

fn meta_from_bytes(bytes: &[u8], path: &Path) -> Result<CacheMeta> {
    let (_, header) = safetensors::SafeTensors::read_metadata(bytes)
        .map_err(|e| Error::cache(path, format!("reading the header: {e}")))?;
    let json = header
        .metadata()
        .as_ref()
        .and_then(|m| m.get(METADATA_KEY))
        .ok_or_else(|| Error::cache(path, format!("no `{METADATA_KEY}` metadata key")))?;
    let meta: CacheMeta = serde_json::from_str(json)
        .map_err(|e| Error::cache(path, format!("parsing the metadata: {e}")))?;
    if meta.format != FORMAT {
        return Err(Error::cache(path, format!("format `{}`, expected `{FORMAT}`", meta.format)));
    }
    if meta.cache_version != CACHE_VERSION {
        return Err(Error::cache(
            path,
            format!("cache_version {}, expected {CACHE_VERSION}", meta.cache_version),
        ));
    }
    if meta.algo_ref != ALGO_REF {
        return Err(Error::cache(
            path,
            format!("algo_ref `{}`, expected `{ALGO_REF}`", meta.algo_ref),
        ));
    }
    Ok(meta)
}

/// The little-endian `f32` bytes of a point array.
fn points_bytes(p: &[Vec3f]) -> Vec<u8> {
    p.iter().flat_map(|p| p.to_array()).flat_map(f32::to_le_bytes).collect()
}

/// One `u32[m, 3]` tensor as triangles, checking its dtype and shape.
fn read_triangles(
    file: &safetensors::SafeTensors<'_>,
    name: &str,
    path: &Path,
) -> Result<Vec<[u32; 3]>> {
    let view = file.tensor(name).map_err(|e| Error::cache(path, format!("tensor {name}: {e}")))?;
    if view.dtype() != Dtype::U32 || view.shape().len() != 2 || view.shape()[1] != 3 {
        return Err(Error::cache(
            path,
            format!("tensor {name} is {:?} {:?}, expected U32 [m, 3]", view.dtype(), view.shape()),
        ));
    }
    Ok(view
        .data()
        .chunks_exact(12)
        .map(|c| {
            [
                u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                u32::from_le_bytes([c[8], c[9], c[10], c[11]]),
            ]
        })
        .collect())
}

/// One `f32[n, 3]` tensor as points, checking its dtype and shape.
fn read_points(file: &safetensors::SafeTensors<'_>, name: &str, path: &Path) -> Result<Vec<Vec3f>> {
    let view = file.tensor(name).map_err(|e| Error::cache(path, format!("tensor {name}: {e}")))?;
    if view.dtype() != Dtype::F32 || view.shape().len() != 2 || view.shape()[1] != 3 {
        return Err(Error::cache(
            path,
            format!("tensor {name} is {:?} {:?}, expected F32 [n, 3]", view.dtype(), view.shape()),
        ));
    }
    Ok(view
        .data()
        .chunks_exact(12)
        .map(|c| {
            Vec3f::new(
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                f32::from_le_bytes([c[8], c[9], c[10], c[11]]),
            )
        })
        .collect())
}

/// One `u32[n]` tensor as indices, checking its dtype and shape.
fn read_u32(file: &safetensors::SafeTensors<'_>, name: &str, path: &Path) -> Result<Vec<u32>> {
    let view = file.tensor(name).map_err(|e| Error::cache(path, format!("tensor {name}: {e}")))?;
    if view.dtype() != Dtype::U32 || view.shape().len() != 1 {
        return Err(Error::cache(
            path,
            format!("tensor {name} is {:?} {:?}, expected U32 [n]", view.dtype(), view.shape()),
        ));
    }
    Ok(view.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// `os.path.abspath`'s Rust equivalent: the path against the working directory, unresolved.
///
/// Symbolic links are deliberately left alone — R §3.7 compares what the caller passed, not where
/// it eventually points.
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(64);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn unhex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{
        CacheMeta, EXTENSION, FORMAT, METADATA_KEY, absolute, cache_path, from_bytes, hex,
        load_valid, read, read_meta, to_bytes, unhex, write,
    };
    use crate::fragment::Fragment;
    use crate::fragment::breakline::{Breaklines, BrkParams};
    use crate::fragment::samples::{SampleParams, Samples};
    use crate::types::{FaceLabel, SourceRef, WorkingMesh};
    use crate::vec3::vec3;
    use crate::{ALGO_REF, CACHE_VERSION};

    /// A fragment with a small but non-trivial working mesh: a tetrahedron, its four faces, and
    /// scalars whose exact bits the round trip has to preserve.
    fn sample(source: &std::path::Path) -> Fragment {
        let v = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.1234_5678, 0.2, 0.987_654_3),
        ];
        let f = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        let meta = std::fs::metadata(source).expect("the source file exists");
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| i128::try_from(d.as_nanos()).unwrap_or(0));
        Fragment {
            id: 0,
            name: "pieceA".to_owned(),
            source: SourceRef {
                path: source.to_path_buf(),
                size: meta.len(),
                mtime_ns,
                sha256: Some([7u8; 32]),
            },
            mesh: WorkingMesh::from_parts(v, f, 0.123_456_79),
            labels: vec![
                FaceLabel::Shell,
                FaceLabel::Fracture,
                FaceLabel::Fracture,
                FaceLabel::Shell,
            ],
            brk: Breaklines {
                params: BrkParams::at(3.531_017_303_466_797),
                p: vec![vec3(0.5, 0.0, 0.0), vec3(0.0, 0.5, 0.123_456_79)],
                ns: vec![vec3(0.0, 0.0, -1.0), vec3(0.0, -1.0, 0.0)],
                nf: vec![vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)],
                f: vec![vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)],
                sub: vec![1],
            },
            samples: Samples {
                params: SampleParams::at(3.531_017_303_466_797),
                s: vec![vec3(0.25, 0.25, 0.0), vec3(0.3, 0.1, 0.2), vec3(0.1, 0.4, 0.1)],
                sp: vec![0, 1, 3],
                pf: vec![vec3(0.3, 0.1, 0.2)],
                fp: vec![1],
                margin_idx: vec![0, 2],
            },
            thick: 3.531_017_303_466_797,
            thick_mode: 4.044_1,
            watertight: true,
            n_boundary: 0,
            n_orig_vertices: 12,
            n_orig_faces: 20,
            target_faces: 200_000,
            face_budget: 50_000,
            area0: 17.529_384_756_291_3,
            area: 2.115_384_756_291_31,
            frac_area: 1.015_384_756_291_31,
        }
    }

    /// A directory of this test's own under the crate's target directory, so nothing is written
    /// where another test can see it.
    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sherd-cache-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    }

    #[test]
    fn a_fragment_survives_the_round_trip_bit_for_bit() {
        let dir = scratch("roundtrip");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"not really a mesh, but it has an mtime and a size").unwrap();
        let fr = sample(&source);

        let path = cache_path(&dir, &fr.name);
        assert_eq!(path, dir.join("cache").join("pieceA.sherd"));
        write(&fr, &path).expect("the cache is written");
        let back = read(&path).expect("the cache is read");

        assert_eq!(back.name, fr.name);
        assert_eq!(back.source.size, fr.source.size);
        assert_eq!(back.source.mtime_ns, fr.source.mtime_ns);
        assert_eq!(back.source.sha256, fr.source.sha256);
        assert_eq!(back.source.path, absolute(&fr.source.path));
        assert_eq!(back.target_faces, fr.target_faces);
        assert_eq!(back.face_budget, fr.face_budget);
        assert_eq!(back.n_boundary, fr.n_boundary);
        assert_eq!(back.n_orig_vertices, fr.n_orig_vertices);
        assert_eq!(back.n_orig_faces, fr.n_orig_faces);
        assert_eq!(back.watertight, fr.watertight);
        // Bit-identical, not merely close: a cached run and a cold run must be the same run.
        assert_eq!(back.thick.to_bits(), fr.thick.to_bits(), "thick");
        assert_eq!(back.thick_mode.to_bits(), fr.thick_mode.to_bits(), "thick_mode");
        assert_eq!(back.area0.to_bits(), fr.area0.to_bits(), "area0");
        assert_eq!(back.mesh.res.to_bits(), fr.mesh.res.to_bits(), "res");
        assert_eq!(back.mesh.v, fr.mesh.v, "vertices");
        assert_eq!(back.mesh.f, fr.mesh.f, "faces");
        assert_eq!(back.labels, fr.labels, "labels");
        assert_eq!(back.brk, fr.brk, "the breakline arrays, frames, subset and parameters");
        assert_eq!(back.samples, fr.samples, "the sampled arrays and their parameters");
        assert_eq!(back.area.to_bits(), fr.area.to_bits(), "area");
        assert_eq!(back.frac_area.to_bits(), fr.frac_area.to_bits(), "frac_area");
        assert_eq!(back.mesh.face_normals, fr.mesh.face_normals, "face normals");
        assert_eq!(back.mesh.face_areas, fr.mesh.face_areas, "face areas");
        assert_eq!(back.mesh.face_centroids, fr.mesh.face_centroids, "face centroids");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_writes_of_the_same_fragment_are_byte_identical() {
        let dir = scratch("determinism");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();
        let fr = sample(&source);
        // Four times, because the failure mode this guards against — `HashMap` ordering in the
        // safetensors header — is random per map instance and would show up intermittently.
        let first = to_bytes(&fr).unwrap();
        for _ in 0..3 {
            assert_eq!(to_bytes(&fr).unwrap(), first, "the cache bytes must not move between runs");
        }
        assert!(
            String::from_utf8_lossy(&first[..400.min(first.len())]).contains(METADATA_KEY),
            "the metadata block is in the header"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_validity_rule_is_the_references() {
        let dir = scratch("valid");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();
        let fr = sample(&source);
        let meta = CacheMeta::of(&fr);

        assert_eq!(meta.format, FORMAT);
        assert_eq!(meta.cache_version, CACHE_VERSION);
        assert_eq!(meta.algo_ref, ALGO_REF);
        assert!(meta.valid_for(&source, 200_000, "pieceA"));
        assert!(!meta.valid_for(&source, 100_000, "pieceA"), "a different face cap");
        assert!(!meta.valid_for(&source, 200_000, "pieceB"), "a different name");
        assert!(!meta.valid_for(dir.join("other.ply"), 200_000, "pieceA"), "a different file");

        let mut stale = meta.clone();
        stale.cache_version += 1;
        assert!(!stale.valid_for(&source, 200_000, "pieceA"), "another layout version");
        let mut other_algo = meta.clone();
        other_algo.algo_ref = "2026-01-01/deadbee".to_owned();
        assert!(!other_algo.valid_for(&source, 200_000, "pieceA"), "another algorithm reference");

        // R §3.7 tolerates a sub-second move of the modification time and nothing more.
        let mut near = meta.clone();
        near.source_mtime_ns -= 999_000_000;
        assert!(near.valid_for(&source, 200_000, "pieceA"));
        let mut far = meta;
        far.source_mtime_ns -= 1_001_000_000;
        assert!(!far.valid_for(&source, 200_000, "pieceA"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_valid_recomputes_rather_than_failing() {
        let dir = scratch("loadvalid");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();
        let fr = sample(&source);
        let path = cache_path(&dir, "pieceA");
        write(&fr, &path).unwrap();

        assert!(load_valid(&path, &source, 200_000, "pieceA").is_some());
        assert!(load_valid(&path, &source, 200_000, "pieceB").is_none(), "another name");
        assert!(load_valid(dir.join("nope.sherd"), &source, 200_000, "pieceA").is_none());

        // A truncated file is a miss, not a panic and not an error.
        let broken = dir.join("broken.sherd");
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&broken, &bytes[..bytes.len() / 2]).unwrap();
        assert!(load_valid(&broken, &source, 200_000, "pieceA").is_none());
        assert!(read(&broken).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_from_another_build_is_refused_with_its_reason() {
        let dir = scratch("versions");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();
        let fr = sample(&source);
        let mut bytes = to_bytes(&fr).unwrap();
        // Only the header is text; the tensor payload behind it is not, so the version is patched
        // in place inside the header and every length in the file stays as it was.
        let n = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
        let header = String::from_utf8(bytes[8..8 + n].to_vec()).unwrap();
        let old = format!("\\\"cache_version\\\":{CACHE_VERSION}");
        assert!(header.contains(&old), "{header}");
        let patched =
            header.replacen(&old, &format!("\\\"cache_version\\\":{}", CACHE_VERSION + 1), 1);
        assert_eq!(patched.len(), header.len(), "the patch must not move any offset");
        bytes.splice(8..8 + n, patched.into_bytes());

        let path = dir.join("bumped.sherd");
        std::fs::write(&path, &bytes).unwrap();
        let err = read_meta(&path).unwrap_err().to_string();
        assert!(err.contains("cache_version"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_hash_survives_hex_and_back() {
        let bytes = [
            0u8, 1, 15, 16, 255, 128, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 42,
        ];
        let s = hex(&bytes);
        assert_eq!(&s[..14], "00010f10ff8007");
        assert_eq!(unhex(&s), Some(bytes));
        assert_eq!(unhex("short"), None);
        assert_eq!(unhex(&"z".repeat(64)), None);
    }

    #[test]
    fn the_extension_is_the_designs() {
        assert_eq!(EXTENSION, "sherd");
        assert_eq!(cache_path("/out", "x").extension().unwrap(), "sherd");
    }

    /// A cache whose `brk_sub` points outside `brk_P`, or whose frames do not describe the points,
    /// is refused rather than read back and indexed.
    #[test]
    fn a_cache_whose_breaklines_do_not_hang_together_is_refused() {
        let dir = scratch("badbrk");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();

        let mut fr = sample(&source);
        fr.brk.sub = vec![7];
        let err = from_bytes(&to_bytes(&fr).unwrap(), &source).unwrap_err().to_string();
        assert!(err.contains("brk_sub 7 is outside brk_P"), "{err}");

        let mut fr = sample(&source);
        fr.brk.ns.pop();
        let err = from_bytes(&to_bytes(&fr).unwrap(), &source).unwrap_err().to_string();
        assert!(err.contains("frames do not describe brk_P"), "{err}");

        // And a fragment with no breakline at all round-trips as one, rather than as an error:
        // an all-shell mesh is a legitimate fragment (R §3.5.3).
        let mut fr = sample(&source);
        fr.brk = Breaklines { params: BrkParams::at(fr.thick), ..Breaklines::default() };
        let back = from_bytes(&to_bytes(&fr).unwrap(), &source).unwrap();
        assert!(back.brk.is_empty());
        assert_eq!(back.brk, fr.brk);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A cache whose sampled arrays do not hang together — a face index off the end of `F`, a
    /// `margin_idx` off the end of `S`, an `sp` of the wrong length — is refused rather than read
    /// back and indexed.
    #[test]
    fn a_cache_whose_samples_do_not_hang_together_is_refused() {
        let dir = scratch("badmd");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();

        let mut fr = sample(&source);
        fr.samples.sp.pop();
        let err = from_bytes(&to_bytes(&fr).unwrap(), &source).unwrap_err().to_string();
        assert!(err.contains("sp/fp do not describe S/Pf"), "{err}");

        let mut fr = sample(&source);
        fr.samples.fp = vec![9];
        let err = from_bytes(&to_bytes(&fr).unwrap(), &source).unwrap_err().to_string();
        assert!(err.contains("sample face 9 is outside F"), "{err}");

        let mut fr = sample(&source);
        fr.samples.margin_idx = vec![11];
        let err = from_bytes(&to_bytes(&fr).unwrap(), &source).unwrap_err().to_string();
        assert!(err.contains("margin_idx 11 is outside S"), "{err}");

        // A fragment with no fracture and no margin is a legitimate fragment, not an error.
        let mut fr = sample(&source);
        fr.samples = Samples { params: SampleParams::at(fr.thick), ..Samples::default() };
        let back = from_bytes(&to_bytes(&fr).unwrap(), &source).unwrap();
        assert_eq!(back.samples, fr.samples);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A cache whose `labels` tensor does not describe the face list is refused, not read.
    #[test]
    fn a_label_array_of_the_wrong_length_is_rejected() {
        let dir = scratch("badlabels");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();
        let mut fr = sample(&source);
        fr.labels.pop();
        let bytes = to_bytes(&fr).unwrap();
        let err = from_bytes(&bytes, &source).unwrap_err().to_string();
        assert!(err.contains("tensor labels"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_out_of_range_triangle_is_rejected() {
        let dir = scratch("badindex");
        let source = dir.join("pieceA.ply");
        std::fs::write(&source, b"x").unwrap();
        let mut fr = sample(&source);
        fr.mesh.f[0] = [0, 1, 99];
        let bytes = to_bytes(&fr).unwrap();
        let err = from_bytes(&bytes, &source).unwrap_err().to_string();
        assert!(err.contains("outside V"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
