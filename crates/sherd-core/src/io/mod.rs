//! Reading meshes and writing the outputs (R §3.1, R §11; D §3, experiment E2).
//!
//! The crate choice is settled by `docs/superpowers/notes/2026-09-06-e2-io.md`: `ply-rs-bw` for
//! PLY (read and write; its writer reproduces Open3D's binary file byte for byte), `tobj` for
//! OBJ, `stl_io` for STL, `gltf` for GLB with `COLOR_0`, and an own ~40-line reader for OFF,
//! because no crate on crates.io reads OFF. Readers are filled in by plan step S2.

pub mod glb;
pub mod obj;
pub mod off;
pub mod ply;
pub mod stl;
pub mod writer;

use std::path::Path;

use crate::error::{Error, Result};

/// The extensions the reference scans a collection directory for, in its own order
/// (`sherd_refit.fragment.MESH_EXT`, R §2). Discovery must use exactly this list, or the
/// collection — and with it the pair order and every seeded draw — would differ from the
/// reference's.
pub const REFERENCE_EXTENSIONS: [&str; 4] = ["ply", "obj", "stl", "off"];

/// A mesh file format this crate can read.
///
/// [`Glb`](MeshFormat::Glb) is outside the reference's discovery list: the Python never reads it.
/// It is supported for explicitly named files and for the desktop app's exports (D §9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshFormat {
    /// Stanford PLY, ASCII or binary, little- or big-endian, with optional vertex colours.
    Ply,
    /// Wavefront OBJ.
    Obj,
    /// STL, ASCII or binary.
    Stl,
    /// Object File Format.
    Off,
    /// Binary glTF.
    Glb,
}

impl MeshFormat {
    /// The format a path's extension names.
    ///
    /// The extension is matched case-insensitively; `.gltf` is read as [`Glb`](MeshFormat::Glb)
    /// only when it is a binary container, which the reader checks.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "ply" => Ok(Self::Ply),
            "obj" => Ok(Self::Obj),
            "stl" => Ok(Self::Stl),
            "off" => Ok(Self::Off),
            "glb" | "gltf" => Ok(Self::Glb),
            _ => Err(Error::UnsupportedFormat { path: path.to_path_buf(), extension: ext }),
        }
    }

    /// True when a collection scan accepts this format (R §2).
    pub fn is_discovered(self) -> bool {
        matches!(self, Self::Ply | Self::Obj | Self::Stl | Self::Off)
    }
}

#[cfg(test)]
mod tests {
    use super::{MeshFormat, REFERENCE_EXTENSIONS};
    use crate::error::Error;

    #[test]
    fn extensions_map_to_formats() {
        assert_eq!(MeshFormat::from_path("a/b/frag_001.ply").unwrap(), MeshFormat::Ply);
        assert_eq!(MeshFormat::from_path("Piece.OBJ").unwrap(), MeshFormat::Obj);
        assert_eq!(MeshFormat::from_path("x.StL").unwrap(), MeshFormat::Stl);
        assert_eq!(MeshFormat::from_path("x.off").unwrap(), MeshFormat::Off);
        assert_eq!(MeshFormat::from_path("x.glb").unwrap(), MeshFormat::Glb);
    }

    #[test]
    fn other_files_are_rejected_by_name() {
        let err = MeshFormat::from_path("notes.txt").unwrap_err();
        assert!(matches!(&err, Error::UnsupportedFormat { extension, .. } if extension == "txt"));
        let err = MeshFormat::from_path("README").unwrap_err();
        assert!(matches!(&err, Error::UnsupportedFormat { extension, .. } if extension.is_empty()));
    }

    #[test]
    fn discovery_matches_the_reference_list() {
        for ext in REFERENCE_EXTENSIONS {
            let f = MeshFormat::from_path(format!("x.{ext}")).unwrap();
            assert!(f.is_discovered(), "{ext} is in MESH_EXT");
        }
        assert!(!MeshFormat::Glb.is_discovered());
    }
}
