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
//!
//! D §4.2 lists `labels u8[m]`, the match arrays (`S`, `sp`, `Pf`, `fp`, `brk_*`, `margin_idx`)
//! and the optional `features/*` beside them. Those stages are phase 1b; the reader treats an
//! unknown tensor as data it does not need yet and the writer adds them when they exist, so an
//! older cache stays readable and a newer one does not confuse this build (`cache_version` moves
//! when the *set* changes, not when a tensor is added — see [`crate::CACHE_VERSION`]).
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
use crate::error::{Error, Result};
use crate::types::{SourceRef, WorkingMesh};
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
    /// Vertices after cleaning, before the largest-component pass (R §3.1).
    pub n_orig_vertices: u32,
    /// Triangles after cleaning, before the largest-component pass.
    pub n_orig_faces: u32,
    /// Which executor produced the file (D §4.3). Preprocessing is CPU-only in phase 1.
    pub backend: String,
    /// The match-array parameters the `md_*` tensors were built with (R §3.7's `mdp_*`), once
    /// phase 1b writes them. A cache whose `md_params` differ from the run's recomputes only the
    /// match arrays, as the reference does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub md_params: Option<serde_json::Value>,
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
            n_orig_vertices: fragment.n_orig_vertices,
            n_orig_faces: fragment.n_orig_faces,
            backend: "cpu".to_owned(),
            md_params: None,
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
    let tensors = vec![
        ("F", TensorView::new(Dtype::U32, vec![fragment.mesh.f.len(), 3], &f)),
        ("V", TensorView::new(Dtype::F32, vec![fragment.mesh.v.len(), 3], &v)),
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

    let v_view = file.tensor("V").map_err(|e| Error::cache(path, format!("tensor V: {e}")))?;
    if v_view.dtype() != Dtype::F32 || v_view.shape().len() != 2 || v_view.shape()[1] != 3 {
        return Err(Error::cache(
            path,
            format!("tensor V is {:?} {:?}, expected F32 [n, 3]", v_view.dtype(), v_view.shape()),
        ));
    }
    let f_view = file.tensor("F").map_err(|e| Error::cache(path, format!("tensor F: {e}")))?;
    if f_view.dtype() != Dtype::U32 || f_view.shape().len() != 2 || f_view.shape()[1] != 3 {
        return Err(Error::cache(
            path,
            format!("tensor F is {:?} {:?}, expected U32 [m, 3]", f_view.dtype(), f_view.shape()),
        ));
    }

    // Little-endian by the safetensors specification, and read as such rather than cast: the
    // buffer of a file read into a `Vec<u8>` carries no alignment guarantee.
    let v: Vec<Vec3f> = v_view
        .data()
        .chunks_exact(12)
        .map(|c| {
            Vec3f::new(
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                f32::from_le_bytes([c[8], c[9], c[10], c[11]]),
            )
        })
        .collect();
    let f: Vec<[u32; 3]> = f_view
        .data()
        .chunks_exact(12)
        .map(|c| {
            [
                u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                u32::from_le_bytes([c[8], c[9], c[10], c[11]]),
            ]
        })
        .collect();
    let n_v = u32::try_from(v.len()).unwrap_or(u32::MAX);
    if let Some(bad) = f.iter().flatten().find(|&&i| i >= n_v) {
        return Err(Error::cache(path, format!("triangle index {bad} is outside V ({n_v})")));
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
        thick: meta.thick,
        thick_mode: meta.thick_mode,
        watertight: meta.watertight,
        n_boundary: meta.n_boundary,
        n_orig_vertices: meta.n_orig_vertices,
        n_orig_faces: meta.n_orig_faces,
        target_faces: meta.target_faces,
        face_budget: meta.face_budget,
        area0: meta.area0,
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
    use crate::types::{SourceRef, WorkingMesh};
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
            thick: 3.531_017_303_466_797,
            thick_mode: 4.044_1,
            watertight: true,
            n_boundary: 0,
            n_orig_vertices: 12,
            n_orig_faces: 20,
            target_faces: 200_000,
            face_budget: 50_000,
            area0: 17.529_384_756_291_3,
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
