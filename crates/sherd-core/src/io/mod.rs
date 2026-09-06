//! Reading meshes and writing the outputs (R §3.1, R §11; D §3, experiment E2).
//!
//! The crate choice is settled by `docs/superpowers/notes/2026-09-06-e2-io.md`: `ply-rs-bw` for
//! PLY (read and write; its writer reproduces Open3D's binary file byte for byte), `tobj` for
//! OBJ, `stl_io` for STL, `gltf` for GLB with `COLOR_0`, and an own ~40-line reader for OFF,
//! because no crate on crates.io reads OFF.
//!
//! [`read_mesh`] dispatches on the extension and returns the file as it is; [`load_mesh`] is the
//! reference's `fragment.load_mesh` — read, reject a mesh with no triangles, then run R §3.1's
//! three cleaning passes. Every stage after this one starts from a [`load_mesh`] result.
//!
//! # Colours
//!
//! The reference keeps Open3D's `[0, 1]` doubles; this port quantises to the three bytes a PLY
//! carries as soon as a file is read ([`quantize_color`]). Colours are carried to the outputs and
//! nowhere else (R §3.1), every reader's source is either a byte already or is quantised the same
//! way by Open3D on the way out, and R §11.4's writer emits `uchar` — so the bytes this port
//! writes are the bytes the reference writes, and a mesh costs 3 bytes a vertex instead of 24.
//!
//! All four readers go through [`quantize_color`] and nothing else does the arithmetic itself:
//! PLY hands it the number the file spells, OBJ and GLB the `[0, 1]` float scaled by 255. The rule
//! it applies was measured against Open3D 0.19 rather than assumed (finding F7 of the phase-1a
//! verification); the GLB path used to use `gltf`'s own `into_rgba_u8()`, which truncates.

pub mod glb;
pub mod obj;
pub mod off;
pub mod ply;
pub mod stl;
pub mod writer;

use std::path::Path;

use crate::error::{Error, Result};
use crate::mesh::{Mesh, clean};

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

/// Reads a mesh file, dispatching on its extension, with no cleaning.
///
/// The mesh comes back the way the file holds it: the file's own vertex list, the file's face
/// order, polygons fan-triangulated. Open3D's readers for OBJ, STL, OFF and GLB return a vertex
/// list that is neither of those (Assimp joins vertices on the way in), so raw counts are *not*
/// comparable with the reference's for those formats — only counts after [`load_mesh`] are, and
/// E2 measured them exact on every benchmark file.
pub fn read_mesh(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    match MeshFormat::from_path(path)? {
        MeshFormat::Ply => ply::read(path),
        MeshFormat::Obj => obj::read(path),
        MeshFormat::Stl => stl::read(path),
        MeshFormat::Off => off::read(path),
        MeshFormat::Glb => glb::read(path),
    }
}

/// Reads and cleans a mesh: `sherd_refit.fragment.load_mesh` (R §3.1 steps 1–2).
///
/// Rejects a file with no triangles — the reference raises `ValueError(f"{path}: no triangles")`
/// there — and then merges duplicate vertices, drops degenerate triangles and drops unreferenced
/// vertices, in that order. The largest-connected-component step of R §3.1 is *not* part of this:
/// the reference applies it to the fragment's own mesh only, while `report.write_placed_meshes`
/// writes all components of the cleaned original (R §11.4).
pub fn load_mesh(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let mut mesh = read_mesh(path)?;
    if mesh.is_empty() {
        return Err(Error::read(path, "no triangles"));
    }
    clean::clean(&mut mesh);
    Ok(mesh)
}

/// Open3D's own colour quantisation, applied once at read time.
///
/// Open3D stores a colour as a double in `[0, 1]` and writes it as
/// `round(min(255, max(0, c·255)))` — C's `round`, i.e. halves away from zero, which on a
/// non-negative value is halves up. Its readers divide whatever the file declares by 255 (a
/// `float` colour property is *not* treated as `[0, 1]`), so `raw` here is the number as the file
/// spells it and the round trip through Open3D is the identity on `0..=255`.
///
/// **Measured, not assumed** (finding F7 of the phase-1a verification, re-checked against Open3D
/// 0.19 on this machine): a mesh carrying the colours `k/255` for `k = 0.5, 1.5, 2.5, 127.5,
/// 128.5, 254.5` and the out-of-range `1.2` and `−0.1`, written with
/// `write_triangle_mesh(..., write_ascii=False)`, comes back as the bytes 1, 2, 3, 128, 128,
/// 254(*), 255, 0. Every half lands *up*, so the writer is not truncating and it is not rounding
/// half to even; the two out-of-range values confirm both clamps. (*) 254.5/255 is not exactly
/// representable and reaches the writer as 254.4999999, which rounds down — that is the
/// arithmetic, not the rule.
///
/// `round` and `floor(x + 0.5)` differ on the doubles immediately below a half-integer (at
/// `x = 0.49999999999999994`, `x + 0.5` rounds *up* to 1.0 and the floor is then 1, while `round`
/// gives 0). `round` is what Open3D calls, so it is what this does.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped to 0..=255")]
pub fn quantize_color(raw: f64) -> u8 {
    if raw.is_nan() {
        return 0;
    }
    raw.clamp(0.0, 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::{MeshFormat, REFERENCE_EXTENSIONS, quantize_color};
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

    #[test]
    fn colours_quantise_the_way_open3d_writes_them() {
        for k in 0..=255_u8 {
            assert_eq!(quantize_color(f64::from(k)), k);
            // Open3D rounds a half up, measured on all 255 exact half values.
            if k < 255 {
                assert_eq!(quantize_color(f64::from(k) + 0.5), k + 1);
            }
        }
        assert_eq!(quantize_color(-1.0), 0);
        assert_eq!(quantize_color(300.0), 255);
        assert_eq!(quantize_color(254.5), 255);
        assert_eq!(quantize_color(f64::NAN), 0);
    }

    /// The one input on which `round` and `floor(x + 0.5)` disagree: at the largest double below
    /// a half, `x + 0.5` rounds *up* to the next integer and its floor is 1 too high. Open3D calls
    /// `round`, so this must too (finding F7).
    #[test]
    #[allow(clippy::float_cmp, reason = "the point of the test is an exact float identity")]
    fn a_hair_below_a_half_rounds_down() {
        let just_under = 0.5_f64.next_down();
        assert!(just_under + 0.5 == 1.0, "the addition is what rounds up");
        assert_eq!(quantize_color(just_under), 0);
        assert_eq!(quantize_color(0.5), 1);
    }
}
