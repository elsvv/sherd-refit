//! Reading what the Python sink wrote: `.npy` arrays and JSON scalars (D §10.1).
//!
//! `sherd_refit/fixture.py` writes every array with `numpy.save` — its own dtype, no compression,
//! C order — and every scalar block as JSON with sorted keys. This module reads both, and it is
//! the only place in the port that knows about numpy's file format: [`npyz`] does the parsing,
//! chosen in plan step S1 for exactly this.
//!
//! Two rules the readers keep:
//!
//! * **the dtype is checked, never guessed.** A fixture face array is `int64` and a fixture index
//!   array can be `int32` or `int64` depending on the stage that wrote it (P0's note lists them
//!   file by file); a reader that silently reinterpreted one as the other would compare rubbish
//!   and pass. Every function below names the dtypes it accepts and returns
//!   [`Error::Fixture`](sherd_core::Error::Fixture) with the file's name and the offending dtype
//!   otherwise.
//! * **floats are not re-rounded.** The JSON scalars are `float64` decimals of values the
//!   reference computed in `float32`; `serde_json`'s default float parser misrounds some of them
//!   by one ULP, which is exactly the size of the difference the injected comparisons look for,
//!   so the workspace pins `serde_json` with `float_roundtrip` (measured in plan step S3, §2.5 of
//!   its note). **That feature must stay on.**

use std::path::Path;

use npyz::{DType, NpyFile, TypeChar};
use serde::de::DeserializeOwned;
use sherd_core::error::{Error, Result};

/// One array as it was stored: its shape, and its values flattened in C order.
#[derive(Clone, Debug)]
pub struct Array<T> {
    /// Shape, as numpy wrote it.
    pub shape: Vec<u64>,
    /// The values, row-major.
    pub data: Vec<T>,
}

impl<T> Array<T> {
    /// Number of values.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the array holds nothing.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of rows: the first axis, or the length for a one-dimensional array.
    pub fn rows(&self) -> usize {
        usize::try_from(self.shape.first().copied().unwrap_or(0)).unwrap_or(0)
    }
}

/// True when the file exists and can be opened — the fixture levels of D §10.1 leave some arrays
/// out (`slim` drops the cleaned original mesh, `min` drops most of the segmentation), and a
/// stage that needs one of those must skip rather than fail.
pub fn exists(path: impl AsRef<Path>) -> bool {
    path.as_ref().is_file()
}

/// Reads an `.npy` file whose dtype matches `T` exactly.
pub fn read<T: npyz::Deserialize>(path: impl AsRef<Path>) -> Result<Array<T>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::fixture(path, e))?;
    let npy = NpyFile::new(std::io::BufReader::new(file))
        .map_err(|e| Error::fixture(path, format!("not a .npy file: {e}")))?;
    let shape = npy.shape().to_vec();
    let data = npy
        .into_vec::<T>()
        .map_err(|e| Error::fixture(path, format!("reading the values: {e}")))?;
    Ok(Array { shape, data })
}

/// The dtype of an `.npy` file, as numpy's descriptor string (`'<f8'`, `'|b1'`, …).
pub fn dtype(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::fixture(path, e))?;
    let npy = NpyFile::<std::io::BufReader<std::fs::File>>::new(std::io::BufReader::new(file))
        .map_err(|e| Error::fixture(path, format!("not a .npy file: {e}")))?;
    Ok(npy.dtype().descr())
}

/// `(n, 3)` `float64` — every point and normal array of the fixtures.
pub fn read_points(path: impl AsRef<Path>) -> Result<Vec<[f64; 3]>> {
    let path = path.as_ref();
    let a = read::<f64>(path)?;
    triples(path, &a)
}

/// `(n, 3)` `int64` — the fixtures' triangle arrays, narrowed to the `u32` the working mesh uses.
pub fn read_triangles(path: impl AsRef<Path>) -> Result<Vec<[u32; 3]>> {
    let path = path.as_ref();
    let a = read::<i64>(path)?;
    let rows = triples(path, &a)?;
    rows.into_iter()
        .map(|r| {
            let mut out = [0u32; 3];
            for (o, v) in out.iter_mut().zip(r) {
                *o = u32::try_from(v).map_err(|_| {
                    Error::fixture(path, format!("triangle index {v} is not a u32"))
                })?;
            }
            Ok(out)
        })
        .collect()
}

/// A one-dimensional index array of any integer dtype (`int32`, `int64`, `uint32`, `uint64`),
/// returned as `u32`.
///
/// The fixtures use `int64` for the thickness sample and the voxel representatives, `int32` for
/// the face ids of the match arrays and `uint32` for the ray primitive ids, so the caller must not
/// have to know which one a given file happens to be.
pub fn read_indices(path: impl AsRef<Path>) -> Result<Vec<u32>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::fixture(path, e))?;
    let npy = NpyFile::new(std::io::BufReader::new(file))
        .map_err(|e| Error::fixture(path, format!("not a .npy file: {e}")))?;
    let (kind, size) = match npy.dtype() {
        DType::Plain(ts) => (ts.type_char(), ts.size_field()),
        other => {
            return Err(Error::fixture(path, format!("dtype {} is not an index", other.descr())));
        }
    };
    let signed = match kind {
        TypeChar::Int => true,
        TypeChar::Uint => false,
        _ => {
            return Err(Error::fixture(
                path,
                format!("dtype {} is not an integer", npy.dtype().descr()),
            ));
        }
    };
    let values: Vec<i128> = match (signed, size) {
        (true, 4) => npy.into_vec::<i32>().map(|v| v.into_iter().map(i128::from).collect()),
        (true, 8) => npy.into_vec::<i64>().map(|v| v.into_iter().map(i128::from).collect()),
        (false, 4) => npy.into_vec::<u32>().map(|v| v.into_iter().map(i128::from).collect()),
        (false, 8) => npy.into_vec::<u64>().map(|v| v.into_iter().map(i128::from).collect()),
        _ => {
            return Err(Error::fixture(path, format!("{size}-byte integers are not read here")));
        }
    }
    .map_err(|e| Error::fixture(path, format!("reading the values: {e}")))?;
    values
        .into_iter()
        .map(|v| {
            u32::try_from(v).map_err(|_| Error::fixture(path, format!("index {v} is not a u32")))
        })
        .collect()
}

/// A one-dimensional `float32` array — the ray distances of R §3.2, which are `float32` all the
/// way through (S3 note §2.2) and must not be widened before the histogram sees them.
pub fn read_f32(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    Ok(read::<f32>(path)?.data)
}

/// A one-dimensional `float64` array.
pub fn read_f64(path: impl AsRef<Path>) -> Result<Vec<f64>> {
    Ok(read::<f64>(path)?.data)
}

/// A one-dimensional `bool` array — the segmentation masks of R §3.4, one byte per face.
pub fn read_bool(path: impl AsRef<Path>) -> Result<Vec<bool>> {
    Ok(read::<bool>(path)?.data)
}

/// Reads a JSON file of the dump.
pub fn read_json(path: impl AsRef<Path>) -> Result<serde_json::Value> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::fixture(path, e))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::fixture(path, e))
}

/// Reads a JSON file of the dump into a type.
pub fn read_json_as<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::fixture(path, e))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::fixture(path, e))
}

/// A `f64` field of a JSON object, with the file named when it is missing or of another type.
pub fn field_f64(value: &serde_json::Value, key: &str, path: impl AsRef<Path>) -> Result<f64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| Error::fixture(path, format!("no numeric field `{key}`")))
}

/// A `u64` field of a JSON object.
pub fn field_u64(value: &serde_json::Value, key: &str, path: impl AsRef<Path>) -> Result<u64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::fixture(path, format!("no integer field `{key}`")))
}

/// A `bool` field of a JSON object.
pub fn field_bool(value: &serde_json::Value, key: &str, path: impl AsRef<Path>) -> Result<bool> {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| Error::fixture(path, format!("no boolean field `{key}`")))
}

/// The whole file as one scalar — `thick.t.json` and `thick.thick_mode.json` are bare numbers.
pub fn read_scalar(path: impl AsRef<Path>) -> Result<f64> {
    let path = path.as_ref();
    read_json(path)?.as_f64().ok_or_else(|| Error::fixture(path, "the file is not a bare number"))
}

fn triples<T: Copy>(path: &Path, a: &Array<T>) -> Result<Vec<[T; 3]>> {
    if a.shape.len() != 2 || a.shape[1] != 3 {
        return Err(Error::fixture(path, format!("shape {:?}, expected (n, 3)", a.shape)));
    }
    Ok(a.data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

#[cfg(test)]
mod tests {
    use super::{
        dtype, exists, field_f64, field_u64, read, read_indices, read_json, read_points,
        read_scalar, read_triangles,
    };
    use std::path::{Path, PathBuf};

    /// The committed slab dump, which every checkout has.
    fn slab(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/slab/dump/fragments/pieceA")
            .join(name)
    }

    #[test]
    fn the_slabs_arrays_come_back_with_their_shapes() {
        let v = read_points(slab("mesh.V.npy")).unwrap();
        let f = read_triangles(slab("mesh.F.npy")).unwrap();
        assert!(!v.is_empty() && !f.is_empty());
        assert!(f.iter().flatten().all(|&i| (i as usize) < v.len()), "indices are inside V");

        let stats = read_json(slab("mesh.stats.json")).unwrap();
        assert_eq!(f.len() as u64, field_u64(&stats, "faces", slab("mesh.stats.json")).unwrap());
        assert_eq!(v.len() as u64, field_u64(&stats, "vertices", slab("mesh.stats.json")).unwrap());

        let raw = read::<f64>(slab("mesh.V.npy")).unwrap();
        assert_eq!(raw.shape, vec![v.len() as u64, 3]);
        assert_eq!(raw.rows(), v.len());
        assert!(!raw.is_empty());
        assert_eq!(raw.len(), 3 * v.len());
    }

    #[test]
    fn dtypes_are_read_not_guessed() {
        assert_eq!(dtype(slab("mesh.V.npy")).unwrap(), "'<f8'");
        assert_eq!(dtype(slab("mesh.F.npy")).unwrap(), "'<i8'");
        assert_eq!(dtype(slab("thick.t_hit.npy")).unwrap(), "'<f4'");
        assert_eq!(dtype(slab("thick.prim.npy")).unwrap(), "'<u4'");
        assert_eq!(dtype(slab("md.sp.npy")).unwrap(), "'<i4'");
        assert_eq!(dtype(slab("md.valid.npy")).unwrap(), "'|b1'");

        // int64 (`thick.idx`), uint32 (`thick.prim`) and int32 (`md.sp`) all read as indices.
        assert!(!read_indices(slab("thick.idx.npy")).unwrap().is_empty());
        assert!(!read_indices(slab("thick.prim.npy")).unwrap().is_empty());
        assert!(!read_indices(slab("md.sp.npy")).unwrap().is_empty());
        // A float array is not an index array, and the error says which file.
        let err = read_indices(slab("mesh.V.npy")).unwrap_err().to_string();
        assert!(err.contains("mesh.V.npy") && err.contains("not an integer"), "{err}");
    }

    #[test]
    fn a_scalar_file_is_a_bare_number() {
        let t = read_scalar(slab("thick.t.json")).unwrap();
        assert!(t > 0.0, "{t}");
        let target = read_json(slab("thick.target.json")).unwrap();
        assert!(field_f64(&target, "area0", slab("thick.target.json")).unwrap() > 0.0);
        let err = field_f64(&target, "nope", slab("thick.target.json")).unwrap_err().to_string();
        assert!(err.contains("nope"), "{err}");
    }

    #[test]
    fn a_missing_file_is_reported_with_its_name_not_panicked_on() {
        assert!(!exists(slab("no.such.npy")));
        let err = read_points(slab("no.such.npy")).unwrap_err().to_string();
        assert!(err.contains("no.such.npy"), "{err}");
        let err = read_triangles(slab("mesh.V.npy")).unwrap_err().to_string();
        assert!(err.contains("mesh.V.npy"), "{err}");
    }
}
