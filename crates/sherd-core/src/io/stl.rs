//! STL reader (R §3.1) via `stl_io`, binary and ASCII; E2 measured it identical to Open3D's
//! cleaned mesh in the same face order. STL carries no colours.
//!
//! `stl_io::read_stl` already does what the format needs: an STL is a bag of independent facets,
//! and it indexes them by the bit pattern of the three coordinates, which is exactly the vertex
//! merge R §3.1 runs next anyway. Coordinates are `f32` in the format (in the binary encoding by
//! definition, and Assimp — so Open3D — parses the ASCII encoding into `f32` too), so widening
//! them here loses nothing.
//!
//! Filled in by plan step S2.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// Reads an STL file, binary or ASCII — `stl_io` probes which it is.
pub fn read(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let mut file = BufReader::new(File::open(path)?);
    let indexed = stl_io::read_stl(&mut file).map_err(|e| Error::read(path, e))?;
    let mut mesh = Mesh {
        v: indexed
            .vertices
            .iter()
            .map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])])
            .collect(),
        f: Vec::with_capacity(indexed.faces.len()),
        colors: None,
    };
    for t in &indexed.faces {
        let mut corner = [0_u32; 3];
        for (out, &i) in corner.iter_mut().zip(t.vertices.iter()) {
            *out =
                u32::try_from(i).map_err(|_| Error::read(path, "more than 4 billion vertices"))?;
        }
        mesh.f.push(corner);
    }
    mesh.validate(path)?;
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these tests assert exact coordinates on purpose")]

    use super::read;

    const ASCII: &str = "solid t
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 1 0 0
    vertex 1 1 0
    vertex 0 1 0
  endloop
endfacet
endsolid t
";

    #[test]
    fn ascii_facets_share_their_repeated_vertices() {
        let dir = std::env::temp_dir().join("sherd-core-stl");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = dir.join("t.stl");
        std::fs::write(&p, ASCII).expect("the fixture is written");
        let m = read(&p).expect("the file reads");
        assert_eq!(m.n_faces(), 2);
        assert_eq!(m.n_vertices(), 4, "the two shared corners are indexed once");
        assert_eq!(m.v[0], [0.0, 0.0, 0.0]);
        assert!(!m.has_colors());
    }

    #[test]
    fn a_file_that_is_neither_encoding_is_an_error() {
        let dir = std::env::temp_dir().join("sherd-core-stl");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = dir.join("broken.stl");
        std::fs::write(&p, b"not an stl at all").expect("the fixture is written");
        assert!(read(&p).is_err());
    }
}
