//! The PLY writer against Open3D's own file, byte for byte (plan step S2, R §11.4).
//!
//! `tests/data/tetra_binary_le.ply` and `tetra_binary_le_plain.ply` were written by
//! `open3d.io.write_triangle_mesh(path, mesh, write_ascii=False, compressed=False,
//! write_vertex_normals=False)` — the call `report.write_placed_meshes` makes. Reading one back and
//! writing it again with [`OPEN3D_COMMENT`] has to reproduce the file exactly: same header, same
//! `double` coordinates, same `uchar` colours, same `list uchar uint` faces, same order.
//!
//! That is the strongest statement available about the output schema, and it is what makes a
//! byte-diff of the Rust and Python outputs a usable regression test while the port is in flight.

use std::path::{Path, PathBuf};

use sherd_core::io::writer::{DEFAULT_COMMENT, OPEN3D_COMMENT, PlyStream, write_ply};
use sherd_core::io::{load_mesh, ply};
use sherd_core::mesh::Mesh;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn rewrite(mesh: &Mesh, comment: &str) -> Vec<u8> {
    let mut stream =
        PlyStream::begin(Vec::new(), mesh.n_vertices(), mesh.n_faces(), mesh.has_colors(), comment)
            .expect("the header is written");
    stream.write_mesh(mesh, 0).expect("the payload is written");
    stream.finish().expect("the promised counts were written")
}

#[test]
fn a_coloured_mesh_is_written_exactly_as_open3d_writes_it() {
    let path = data("tetra_binary_le.ply");
    let original = std::fs::read(&path).expect("the fixture is committed");
    let mesh = ply::read(&path).expect("the fixture reads");
    assert!(mesh.has_colors());
    assert_eq!(rewrite(&mesh, OPEN3D_COMMENT), original);
}

#[test]
fn a_mesh_without_colours_is_written_exactly_as_open3d_writes_it() {
    let path = data("tetra_binary_le_plain.ply");
    let original = std::fs::read(&path).expect("the fixture is committed");
    let mesh = ply::read(&path).expect("the fixture reads");
    assert!(!mesh.has_colors());
    assert_eq!(rewrite(&mesh, OPEN3D_COMMENT), original);
}

#[test]
fn the_port_stamps_its_own_comment_and_changes_nothing_else() {
    let path = data("tetra_binary_le.ply");
    let original = std::fs::read(&path).expect("the fixture is committed");
    let mesh = ply::read(&path).expect("the fixture reads");
    let ours = rewrite(&mesh, DEFAULT_COMMENT);
    assert_ne!(ours, original, "the comment line differs");
    assert_eq!(
        ours.len(),
        original.len() + DEFAULT_COMMENT.len() - OPEN3D_COMMENT.len(),
        "only the comment differs in length"
    );
    let cut = |b: &[u8]| {
        let end = b.windows(11).position(|w| w == b"end_header\n").expect("a header") + 11;
        b[end..].to_vec()
    };
    assert_eq!(cut(&ours), cut(&original), "the payload is untouched");
}

#[test]
fn every_benchmark_mesh_survives_a_write_and_a_read() {
    // The full loop the pipeline runs for `placed/<name>.ply`: load and clean a real scan, write
    // it, read it back. Cheap enough for one file per format, and it is the only place the writer
    // meets a mesh with hundreds of thousands of vertices.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let candidates = [
        "input/test_fragments_1/fragments/FY234104_reduced.ply",
        "input/sfspp/pot_C/Pot_C_Piece_01_Mesh_DS.obj",
        "input/synthetic_pingsdorf_20/fragments/frag_000.ply",
    ];
    let dir = std::env::temp_dir().join("sherd-core-ply-writer-benchmarks");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut ran = 0;
    for name in candidates {
        let path = root.join(name);
        if !path.exists() {
            continue;
        }
        let mesh = load_mesh(&path).expect("the scan loads");
        let out = dir.join("placed.ply");
        write_ply(&out, &mesh).expect("the mesh is written");
        let back = ply::read(&out).expect("what was written reads");
        assert_eq!(back, mesh, "{name}: the round trip is lossless");
        ran += 1;
    }
    eprintln!("{ran} of {} benchmark meshes were present", candidates.len());
}
