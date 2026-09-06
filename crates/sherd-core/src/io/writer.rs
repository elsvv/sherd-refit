//! PLY writer (R §11.2–11.4).
//!
//! Binary little-endian, and the same header and element order as Open3D's
//! `write_triangle_mesh(write_ascii=False, compressed=False, write_vertex_normals=False)`, so that
//! the Rust outputs are byte-comparable with the reference's:
//!
//! ```text
//! ply
//! format binary_little_endian 1.0
//! comment Created by Open3D
//! element vertex N
//! property double x
//! property double y
//! property double z
//! property uchar red          <- only when the mesh has colours
//! property uchar green
//! property uchar blue
//! element face M
//! property list uchar uint vertex_indices
//! end_header
//! ```
//!
//! Two corrections to D §3's "mesh write" row, both measured in E2 and re-measured here: the
//! coordinates are `double`, not `float` (Open3D's vertices are `double` and it writes them as
//! they are), and the list values are `uint`, not `int`. A `float` writer would produce a file
//! half the size that is *not* byte-compatible with the reference's.
//!
//! [`PlyStream`] writes the file element by element, which is what R §11.4's merged
//! `assembly_<k>.ply` needs: the reference concatenates meshes in memory, and D §5 does not, so
//! the members are streamed one at a time with an index offset.
//!
//! Filled in by plan step S2.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ply_rs_bw::ply::{
    Addable, ElementDef, Encoding, Header, PropertyAccess, PropertyDef, PropertyType, ScalarType,
};
use ply_rs_bw::writer::Writer;

use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// The header comment Open3D stamps on every mesh it writes.
///
/// Passing it to [`PlyStream::begin`] makes the output byte-identical to the reference's file,
/// which is how the port's writer is regression-tested; the pipeline itself writes
/// [`DEFAULT_COMMENT`], because this port is not Open3D.
pub const OPEN3D_COMMENT: &str = "Created by Open3D";

/// The header comment this port writes.
pub const DEFAULT_COMMENT: &str = "Created by sherd-refit";

/// Writes a mesh as R §11.4's `placed/<name>.ply`.
pub fn write_ply(path: impl AsRef<Path>, mesh: &Mesh) -> Result<()> {
    write_ply_with_comment(path, mesh, DEFAULT_COMMENT)
}

/// Writes a mesh with a chosen header comment.
pub fn write_ply_with_comment(path: impl AsRef<Path>, mesh: &Mesh, comment: &str) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).map_err(|e| Error::write(path, e))?;
    let mut stream = PlyStream::begin(
        BufWriter::with_capacity(1 << 20, file),
        mesh.n_vertices(),
        mesh.n_faces(),
        mesh.has_colors(),
        comment,
    )
    .map_err(|e| Error::write(path, e))?;
    stream.write_mesh(mesh, 0).map_err(|e| Error::write(path, e))?;
    stream.finish().map_err(|e| Error::write(path, e))?;
    Ok(())
}

/// A PLY file written element by element, so that no merged mesh is ever held in memory (D §5).
///
/// The counts go into the header before the first vertex is written, so the caller has to know
/// them up front — R §11.4's merged mesh does, from the members' cached vertex and face counts.
#[derive(Debug)]
pub struct PlyStream<W: Write> {
    out: W,
    vertex_def: ElementDef,
    face_def: ElementDef,
    colors: bool,
    n_vertices: usize,
    n_faces: usize,
    wrote_vertices: usize,
    wrote_faces: usize,
}

impl<W: Write> PlyStream<W> {
    /// Writes the header and returns a stream expecting exactly `n_vertices` vertices followed by
    /// exactly `n_faces` faces.
    pub fn begin(
        mut out: W,
        n_vertices: usize,
        n_faces: usize,
        colors: bool,
        comment: &str,
    ) -> std::io::Result<Self> {
        let mut vertex_def = ElementDef::new("vertex".to_string());
        // `ElementDef::new` leaves `count` at 0 and `write_header` believes it: a file written
        // without this line claims zero vertices and is silently unreadable (E2's footgun).
        vertex_def.count = n_vertices;
        for axis in ["x", "y", "z"] {
            vertex_def
                .properties
                .add(PropertyDef::new(axis.to_string(), PropertyType::Scalar(ScalarType::Double)));
        }
        if colors {
            for channel in ["red", "green", "blue"] {
                vertex_def.properties.add(PropertyDef::new(
                    channel.to_string(),
                    PropertyType::Scalar(ScalarType::UChar),
                ));
            }
        }
        let mut face_def = ElementDef::new("face".to_string());
        face_def.count = n_faces;
        face_def.properties.add(PropertyDef::new(
            "vertex_indices".to_string(),
            PropertyType::List(ScalarType::UChar, ScalarType::UInt),
        ));

        let mut header = Header::new();
        header.encoding = Encoding::BinaryLittleEndian;
        header.comments.push(comment.to_string());
        header.elements.add(vertex_def.clone());
        header.elements.add(face_def.clone());
        Writer::<OutVertex>::new().write_header(&mut out, &header)?;

        Ok(Self {
            out,
            vertex_def,
            face_def,
            colors,
            n_vertices,
            n_faces,
            wrote_vertices: 0,
            wrote_faces: 0,
        })
    }

    /// Writes one mesh's vertices; call once per member, in order, before any face.
    ///
    /// A stream opened with colours accepts a mesh without them and writes white for it, which is
    /// what R §11.4's merged `assembly_<k>.ply` needs when its members disagree.
    pub fn write_vertices(&mut self, mesh: &Mesh) -> std::io::Result<()> {
        let writer = Writer::<OutVertex>::new();
        let mut vertex = OutVertex { p: [0.0; 3], c: [0; 3] };
        for i in 0..mesh.v.len() {
            vertex.p = mesh.v[i];
            if self.colors {
                vertex.c = mesh.colors.as_ref().map_or([255; 3], |c| c[i]);
            }
            writer.write_little_endian_element(&mut self.out, &vertex, &self.vertex_def)?;
            self.wrote_vertices += 1;
        }
        Ok(())
    }

    /// Writes one mesh's faces with `offset` added to every index; call once per member, in the
    /// same order as [`write_vertices`](Self::write_vertices), after all vertices.
    pub fn write_faces(&mut self, mesh: &Mesh, offset: u32) -> std::io::Result<()> {
        let writer = Writer::<OutFace>::new();
        let mut face = OutFace { idx: [0; 3] };
        for t in &mesh.f {
            face.idx = [t[0] + offset, t[1] + offset, t[2] + offset];
            writer.write_little_endian_element(&mut self.out, &face, &self.face_def)?;
            self.wrote_faces += 1;
        }
        Ok(())
    }

    /// A whole mesh at once: its vertices, then its faces. Only valid for a single-member file —
    /// a merged one has to write every member's vertices before the first face.
    pub fn write_mesh(&mut self, mesh: &Mesh, offset: u32) -> std::io::Result<()> {
        self.write_vertices(mesh)?;
        self.write_faces(mesh, offset)
    }

    /// Flushes, and checks that as many elements were written as the header promised.
    pub fn finish(mut self) -> std::io::Result<W> {
        if self.wrote_vertices != self.n_vertices || self.wrote_faces != self.n_faces {
            return Err(std::io::Error::other(format!(
                "header promised {} vertices and {} faces, {} and {} were written",
                self.n_vertices, self.n_faces, self.wrote_vertices, self.wrote_faces
            )));
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

/// One `vertex` row on the way out.
struct OutVertex {
    p: [f64; 3],
    c: [u8; 3],
}

impl PropertyAccess for OutVertex {
    fn new() -> Self {
        Self { p: [0.0; 3], c: [0; 3] }
    }

    fn get_double(&self, name: &str) -> Option<f64> {
        match name {
            "x" => Some(self.p[0]),
            "y" => Some(self.p[1]),
            "z" => Some(self.p[2]),
            _ => None,
        }
    }

    fn get_uchar(&self, name: &str) -> Option<u8> {
        match name {
            "red" => Some(self.c[0]),
            "green" => Some(self.c[1]),
            "blue" => Some(self.c[2]),
            _ => None,
        }
    }
}

/// One `face` row on the way out.
struct OutFace {
    idx: [u32; 3],
}

impl PropertyAccess for OutFace {
    fn new() -> Self {
        Self { idx: [0; 3] }
    }

    fn get_list_uint(&self, name: &str) -> Option<&[u32]> {
        (name == "vertex_indices").then_some(&self.idx)
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_COMMENT, OPEN3D_COMMENT, PlyStream, write_ply_with_comment};
    use crate::io::ply;
    use crate::mesh::Mesh;
    use std::path::Path;

    fn tetra() -> Mesh {
        Mesh {
            v: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            f: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            colors: Some(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]),
        }
    }

    fn bytes(mesh: &Mesh, comment: &str) -> Vec<u8> {
        let mut s = PlyStream::begin(
            Vec::new(),
            mesh.n_vertices(),
            mesh.n_faces(),
            mesh.has_colors(),
            comment,
        )
        .expect("the header is written");
        s.write_mesh(mesh, 0).expect("the payload is written");
        s.finish().expect("the counts match")
    }

    #[test]
    fn the_header_is_open3ds_header() {
        let out = bytes(&tetra(), OPEN3D_COMMENT);
        let end = b"end_header\n";
        let cut = out.windows(end.len()).position(|w| w == end).expect("a header") + end.len();
        assert_eq!(
            std::str::from_utf8(&out[..cut]).expect("the header is ASCII"),
            "ply\nformat binary_little_endian 1.0\ncomment Created by Open3D\nelement vertex 4\n\
             property double x\nproperty double y\nproperty double z\nproperty uchar red\n\
             property uchar green\nproperty uchar blue\nelement face 4\n\
             property list uchar uint vertex_indices\nend_header\n"
        );
        // 4 vertices of 3 doubles + 3 bytes, 4 faces of a length byte + 3 uints.
        assert_eq!(out.len() - cut, 4 * (24 + 3) + 4 * (1 + 12));
    }

    #[test]
    fn a_mesh_without_colours_has_no_colour_properties() {
        let mut m = tetra();
        m.colors = None;
        let out = bytes(&m, DEFAULT_COMMENT);
        let text = String::from_utf8_lossy(&out[..out.len().min(300)]).to_string();
        assert!(!text.contains("red"), "{text}");
        assert!(text.contains("comment Created by sherd-refit"), "{text}");
    }

    #[test]
    fn what_is_written_reads_back_identically() {
        let dir = std::env::temp_dir().join("sherd-core-writer");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = dir.join("round.ply");
        let m = tetra();
        write_ply_with_comment(&p, &m, DEFAULT_COMMENT).expect("the file is written");
        let back = ply::read(&p).expect("the file reads");
        assert_eq!(back, m);
    }

    #[test]
    fn a_merged_file_offsets_the_second_members_indices() {
        let m = tetra();
        let mut s = PlyStream::begin(Vec::new(), 8, 8, true, "merged").expect("header");
        s.write_vertices(&m).expect("member 1 vertices");
        s.write_vertices(&m).expect("member 2 vertices");
        s.write_faces(&m, 0).expect("member 1 faces");
        s.write_faces(&m, 4).expect("member 2 faces");
        let out = s.finish().expect("the counts match");
        let back = ply::read_from(std::io::Cursor::new(out), Path::new("<merged>.ply"))
            .expect("the file reads");
        assert_eq!(back.n_vertices(), 8);
        assert_eq!(back.f[4], [4, 6, 5]);
    }

    #[test]
    fn a_short_payload_is_refused_rather_than_written_as_a_broken_file() {
        let mut s = PlyStream::begin(Vec::new(), 4, 4, false, "short").expect("header");
        s.write_vertices(&tetra()).expect("vertices");
        let e = s.finish().unwrap_err();
        assert!(
            e.to_string().contains("0 and 0 were written") || e.to_string().contains("4 and 0")
        );
    }
}
