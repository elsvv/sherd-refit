//! OBJ reader (R §3.1) via `tobj`, which E2 measured exact on counts after cleaning and within
//! 1.8e-8 relative on coordinates, including vertex colours.
//!
//! Two things the caller has to get right, both found by E2:
//!
//! * `load_obj` returns `(models, Result<materials>)` and the material half must be dropped — the
//!   SfS++ OBJs name a `.mtl` that does not exist, and the geometry loads fine without it;
//! * the file is split on `g` / `usemtl` into models with per-model vertex lists, so the models
//!   are concatenated here with an index offset.
//!
//! Vertex colours (`v x y z r g b`, what MeshLab writes) are `[0, 1]` floats in an OBJ, not the
//! `0..=255` a PLY carries, and Open3D — through Assimp — takes them as such without dividing by
//! 255. They are scaled here before [`quantize_color`](super::quantize_color) sees them.
//!
//! `tobj` parses coordinates into `f32`, its `use_f64` feature being off in E2's pin. At the scale
//! of these scans (~100 mm across) that resolves about 1e-5 mm, three orders of magnitude below
//! D §10.2's tightest tolerance, and — the one thing precision could actually change here — the
//! vertex counts after R §3.1's *exact*-equality merge came out equal to Open3D's on all four
//! benchmark OBJs, so no pair of distinct vertices collides at `f32`.
//!
//! Filled in by plan step S2.

use std::path::Path;

use crate::error::{Error, Result};
use crate::io::quantize_color;
use crate::mesh::Mesh;

/// Reads an OBJ file. Materials are not read; a missing or broken `.mtl` is not an error.
pub fn read(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let options = tobj::LoadOptions { triangulate: true, ..tobj::LoadOptions::default() };
    // The material half of the pair is deliberately dropped: R §3.1 reads geometry and colours.
    let (models, _materials) = tobj::load_obj(path, &options).map_err(|e| Error::read(path, e))?;

    let any_colors = models.iter().any(|m| !m.mesh.vertex_color.is_empty());
    let mut mesh = Mesh { v: Vec::new(), f: Vec::new(), colors: any_colors.then(Vec::new) };
    for model in &models {
        let m = &model.mesh;
        let offset = u32::try_from(mesh.v.len())
            .map_err(|_| Error::read(path, "more than 4 billion vertices"))?;
        if m.positions.len() % 3 != 0 {
            return Err(Error::read(path, "a vertex line with fewer than three coordinates"));
        }
        for p in m.positions.chunks_exact(3) {
            mesh.v.push([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
        }
        if let Some(colors) = &mut mesh.colors {
            let n = m.positions.len() / 3;
            if m.vertex_color.len() == m.positions.len() {
                for c in m.vertex_color.chunks_exact(3) {
                    colors.push([
                        quantize_color(f64::from(c[0]) * 255.0),
                        quantize_color(f64::from(c[1]) * 255.0),
                        quantize_color(f64::from(c[2]) * 255.0),
                    ]);
                }
            } else {
                // One group carries colours and another does not: keep the arrays in step rather
                // than dropping the colours that are there.
                colors.extend(std::iter::repeat_n([255_u8; 3], n));
            }
        }
        for t in m.indices.chunks_exact(3) {
            mesh.f.push([t[0] + offset, t[1] + offset, t[2] + offset]);
        }
    }
    mesh.validate(path)?;
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these tests assert exact coordinates on purpose")]

    use super::read;

    fn write(dir: &std::path::Path, name: &str, text: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).expect("the fixture is written");
        p
    }

    #[test]
    fn colours_are_scaled_from_zero_to_one() {
        let dir = std::env::temp_dir().join("sherd-core-obj-colours");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = write(
            &dir,
            "c.obj",
            "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0.752941 0.752941 0.752941\nf 1 2 3\n",
        );
        let m = read(&p).expect("the file reads");
        assert_eq!(m.colors, Some(vec![[255, 0, 0], [0, 255, 0], [192, 192, 192]]));
        assert_eq!(m.f, vec![[0, 1, 2]]);
    }

    #[test]
    fn a_missing_material_library_is_not_an_error() {
        let dir = std::env::temp_dir().join("sherd-core-obj-mtl");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = write(
            &dir,
            "m.obj",
            "mtllib nowhere.mtl\nusemtl none\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        );
        let m = read(&p).expect("the geometry reads without its materials");
        assert_eq!(m.n_faces(), 1);
        assert!(!m.has_colors());
    }

    #[test]
    fn groups_are_concatenated_with_an_index_offset() {
        let dir = std::env::temp_dir().join("sherd-core-obj-groups");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = write(
            &dir,
            "g.obj",
            "g one\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n\
             g two\nv 0 0 1\nv 1 0 1\nv 0 1 1\nf 4 5 6\n",
        );
        let m = read(&p).expect("the file reads");
        assert_eq!(m.n_vertices(), 6);
        assert_eq!(m.f, vec![[0, 1, 2], [3, 4, 5]]);
        assert_eq!(m.v[5], [0.0, 1.0, 1.0]);
    }

    #[test]
    fn a_quad_is_triangulated() {
        let dir = std::env::temp_dir().join("sherd-core-obj-quad");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let p = write(&dir, "q.obj", "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
        let m = read(&p).expect("the file reads");
        assert_eq!(m.n_faces(), 2);
    }
}
