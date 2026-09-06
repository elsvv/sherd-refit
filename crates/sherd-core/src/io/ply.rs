//! PLY reader (R §3.1) — ASCII, binary little-endian and binary big-endian, `float` or `double`
//! coordinates, `vertex_indices` or `vertex_index` lists, optional `red`/`green`/`blue`
//! (roadmap item 4 keeps the colours), polygons triangulated as a fan.
//!
//! Crate: `ply-rs-bw` through its typed `PropertyAccess` interface rather than `DefaultElement`,
//! which E2 measured as bit-identical to Open3D on all eleven PLY variants of the benchmark and
//! reads a 25 MB scan in 0.057 s.
//!
//! Three details are Open3D's, not the format's, and the port copies them because the reference's
//! vertex and face lists are Open3D's:
//!
//! * a polygon of `n` corners becomes the `n − 2` triangles of a fan from its first corner, and a
//!   face with fewer than three corners is dropped;
//! * `vertex_indices` and `vertex_index` are both accepted (Open3D registers a callback for each);
//! * a colour property is divided by 255 whatever its declared type, so `float` colours are *not*
//!   read as `[0, 1]` — see [`quantize_color`](super::quantize_color).
//!
//! Filled in by plan step S2.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ply_rs_bw::parser::{Parser, Reader};
use ply_rs_bw::ply::{ElementDef, Header, Property, PropertyAccess, PropertyAccessResult};

use crate::error::{Error, Result};
use crate::io::quantize_color;
use crate::mesh::Mesh;

/// Reads a PLY file.
pub fn read(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let file = File::open(path)?;
    read_from(BufReader::with_capacity(1 << 20, file), path)
}

/// Reads a PLY from any buffered source; `path` only names the file in error messages.
pub fn read_from<R: std::io::BufRead>(source: R, path: &Path) -> Result<Mesh> {
    let mut reader = Reader::new(source);
    let vertex_parser = Parser::<PlyVertex>::new();
    let face_parser = Parser::<PlyFace>::new();
    let skip_parser = Parser::<Skip>::new();
    let header: Header =
        vertex_parser.read_header(&mut reader).map_err(|e| Error::read(path, e))?;

    let has_colors = element(&header, "vertex")
        .is_some_and(|e| ["red", "green", "blue"].iter().all(|p| e.properties.contains_key(*p)));

    let mut vertices: Vec<PlyVertex> = Vec::new();
    let mut faces: Vec<PlyFace> = Vec::new();
    for (_, element) in &header.elements {
        match element.name.as_str() {
            "vertex" => {
                vertices = vertex_parser
                    .read_payload_for_element(&mut reader, element, &header)
                    .map_err(|e| Error::read(path, e))?;
            }
            "face" => {
                faces = face_parser
                    .read_payload_for_element(&mut reader, element, &header)
                    .map_err(|e| Error::read(path, e))?;
            }
            // Anything else still has to be consumed, or the stream would be misaligned for the
            // elements after it.
            _ => {
                skip_parser
                    .read_payload_for_element(&mut reader, element, &header)
                    .map_err(|e| Error::read(path, e))?;
            }
        }
    }

    let mut mesh = Mesh {
        v: vertices.iter().map(|v| v.p).collect(),
        f: Vec::with_capacity(faces.len()),
        colors: has_colors.then(|| vertices.iter().map(|v| v.c).collect()),
    };
    for face in &faces {
        triangulate(&face.idx, &mut mesh.f, path)?;
    }
    mesh.validate(path)?;
    Ok(mesh)
}

/// Open3D's fan: `(i0, i[k-1], i[k])` for `k` from 2, and nothing at all below three corners.
fn triangulate(idx: &[i64], out: &mut Vec<[u32; 3]>, path: &Path) -> Result<()> {
    if idx.len() < 3 {
        return Ok(());
    }
    let corner = |v: i64| -> Result<u32> {
        u32::try_from(v).map_err(|_| Error::read(path, format!("negative vertex index {v}")))
    };
    let first = corner(idx[0])?;
    for k in 2..idx.len() {
        out.push([first, corner(idx[k - 1])?, corner(idx[k])?]);
    }
    Ok(())
}

fn element<'h>(header: &'h Header, name: &str) -> Option<&'h ElementDef> {
    header.elements.get(name)
}

/// One row of the `vertex` element: the coordinates, and the colours when the file has them.
struct PlyVertex {
    p: [f64; 3],
    c: [u8; 3],
}

impl PropertyAccess for PlyVertex {
    fn new() -> Self {
        Self { p: [0.0; 3], c: [0; 3] }
    }

    fn set_property(&mut self, key: &str, property: Property) -> PropertyAccessResult {
        let slot = match key {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            "red" | "green" | "blue" => {
                let Some(v) = scalar(&property) else {
                    return PropertyAccessResult::UnsupportedType;
                };
                let i = match key {
                    "red" => 0,
                    "green" => 1,
                    _ => 2,
                };
                self.c[i] = quantize_color(v);
                return PropertyAccessResult::Set;
            }
            // Normals, alpha, curvature, confidence, quality: read by nothing here, as in the
            // reference.
            _ => return PropertyAccessResult::Ignored,
        };
        match scalar(&property) {
            Some(v) => {
                self.p[slot] = v;
                PropertyAccessResult::Set
            }
            None => PropertyAccessResult::UnsupportedType,
        }
    }
}

/// One row of the `face` element: the corner list, whatever integer type it is declared with.
struct PlyFace {
    idx: Vec<i64>,
}

impl PropertyAccess for PlyFace {
    fn new() -> Self {
        Self { idx: Vec::new() }
    }

    fn set_property(&mut self, key: &str, property: Property) -> PropertyAccessResult {
        if key != "vertex_indices" && key != "vertex_index" {
            return PropertyAccessResult::Ignored;
        }
        match index_list(property) {
            Some(idx) => {
                self.idx = idx;
                PropertyAccessResult::Set
            }
            None => PropertyAccessResult::UnsupportedType,
        }
    }
}

/// An element whose payload has to be consumed but whose values nothing here wants.
struct Skip;

impl PropertyAccess for Skip {
    fn new() -> Self {
        Self
    }
}

fn scalar(p: &Property) -> Option<f64> {
    Some(match *p {
        Property::Char(v) => f64::from(v),
        Property::UChar(v) => f64::from(v),
        Property::Short(v) => f64::from(v),
        Property::UShort(v) => f64::from(v),
        Property::Int(v) => f64::from(v),
        Property::UInt(v) => f64::from(v),
        Property::Float(v) => f64::from(v),
        Property::Double(v) => v,
        _ => return None,
    })
}

fn index_list(p: Property) -> Option<Vec<i64>> {
    Some(match p {
        Property::ListChar(v) => v.into_iter().map(i64::from).collect(),
        Property::ListUChar(v) => v.into_iter().map(i64::from).collect(),
        Property::ListShort(v) => v.into_iter().map(i64::from).collect(),
        Property::ListUShort(v) => v.into_iter().map(i64::from).collect(),
        Property::ListInt(v) => v.into_iter().map(i64::from).collect(),
        Property::ListUInt(v) => v.into_iter().map(i64::from).collect(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these tests assert exact coordinates on purpose")]

    use super::read_from;
    use std::path::Path;

    fn parse(text: &str) -> crate::mesh::Mesh {
        read_from(std::io::Cursor::new(text.as_bytes().to_vec()), Path::new("<test>.ply"))
            .expect("the fixture parses")
    }

    const TRIANGLE: &str = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n\
        property float y\nproperty float z\nelement face 1\n\
        property list uchar int vertex_indices\nend_header\n\
        0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";

    #[test]
    fn a_minimal_ascii_file_reads() {
        let m = parse(TRIANGLE);
        assert_eq!(m.v, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(m.f, vec![[0, 1, 2]]);
        assert!(!m.has_colors());
    }

    #[test]
    fn polygons_fan_from_the_first_corner_and_short_faces_vanish() {
        let text = "ply\nformat ascii 1.0\nelement vertex 5\nproperty double x\n\
            property double y\nproperty double z\nelement face 3\n\
            property list uchar uint vertex_index\nend_header\n\
            0 0 0\n1 0 0\n1 1 0\n0 1 0\n2 2 0\n5 0 1 2 3 4\n2 0 1\n3 0 1 2\n";
        let m = parse(text);
        assert_eq!(m.f, vec![[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 1, 2]]);
    }

    #[test]
    fn colours_are_read_and_other_properties_skipped() {
        let text = "ply\nformat ascii 1.0\ncomment hello\nelement vertex 3\nproperty float x\n\
            property float y\nproperty float z\nproperty float nx\nproperty float ny\n\
            property float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n\
            property uchar alpha\nelement face 1\nproperty list uchar int vertex_indices\n\
            end_header\n0 0 0 0 0 1 255 0 0 255\n1 0 0 0 0 1 0 255 0 128\n\
            0 1 0 0 0 1 0 0 255 7\n3 0 1 2\n";
        let m = parse(text);
        assert_eq!(m.colors, Some(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]]));
        assert_eq!(m.v[1], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn an_element_between_vertex_and_face_is_consumed() {
        let text = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
            property float z\nelement stuff 2\nproperty int a\nproperty float b\n\
            element face 1\nproperty list uchar int vertex_indices\nend_header\n\
            0 0 0\n1 0 0\n0 1 0\n1 2.5\n3 4.5\n3 0 1 2\n";
        let m = parse(text);
        assert_eq!(m.f, vec![[0, 1, 2]]);
        assert_eq!(m.v.len(), 3);
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        let truncated = &TRIANGLE[..TRIANGLE.len() - 12];
        let e =
            read_from(std::io::Cursor::new(truncated.as_bytes().to_vec()), Path::new("<test>.ply"))
                .unwrap_err();
        assert!(matches!(e, crate::Error::Read { .. }), "{e}");
    }

    #[test]
    fn a_negative_index_is_an_error() {
        let text = TRIANGLE.replace("3 0 1 2\n", "3 0 -1 2\n");
        let e = read_from(std::io::Cursor::new(text.into_bytes()), Path::new("<test>.ply"))
            .unwrap_err();
        assert!(e.to_string().contains("negative vertex index"), "{e}");
    }
}
