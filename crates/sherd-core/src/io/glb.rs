//! GLB reader (R §3.1 is silent: the reference does not read GLB; D §9 wants it for the desktop
//! app) via `gltf` with `default-features = false, features = ["utils"]`.
//!
//! E2 measured two mandatory workarounds, both of them here:
//!
//! * **read without validation.** `Gltf::from_slice` refuses one museum file in five over
//!   `extensionsRequired[0] = "KHR_materials_pbrSpecularGlossiness"` — a *material* extension, in a
//!   file whose positions and indices are perfectly ordinary. `from_slice_without_validation`
//!   reads all five.
//! * **apply the node transforms.** Concatenating the primitives without them gives a mesh that
//!   does not match Open3D at all; walking the default scene and composing each node's matrix onto
//!   its children reproduces it exactly.
//!
//! The transform arithmetic is `f32`, as it is in glTF and in Assimp (so in Open3D). `COLOR_0` is
//! taken through `read_colors(0).into_rgba_u8()`, which E2 measured exact against Open3D for
//! colours stored as normalised `u8` and as `f32`.
//!
//! Filled in by plan step S2.

use std::path::Path;

use gltf::mesh::Mode;

use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// Reads a binary glTF file.
///
/// Only the self-contained form is supported: the geometry has to live in the GLB's own `BIN`
/// chunk, which is what an export is. A `.gltf` pointing at external buffers is an error.
pub fn read(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let gltf = gltf::Gltf::from_slice_without_validation(&bytes)
        .map_err(|e| Error::read(path, format!("not a readable glTF: {e}")))?;
    let blob = gltf.blob.as_deref();

    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| Error::read(path, "no scene"))?;

    let mut mesh = Mesh::default();
    let mut colored = false;
    for node in scene.nodes() {
        visit(&node, IDENTITY, blob, path, &mut mesh, &mut colored)?;
    }
    if !colored {
        mesh.colors = None;
    }
    mesh.validate(path)?;
    Ok(mesh)
}

type Mat4 = [[f32; 4]; 4];

const IDENTITY: Mat4 =
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

fn visit(
    node: &gltf::Node<'_>,
    parent: Mat4,
    blob: Option<&[u8]>,
    path: &Path,
    out: &mut Mesh,
    colored: &mut bool,
) -> Result<()> {
    let world = mul(parent, node.transform().matrix());
    if let Some(m) = node.mesh() {
        for primitive in m.primitives() {
            add_primitive(&primitive, world, blob, path, out, colored)?;
        }
    }
    for child in node.children() {
        visit(&child, world, blob, path, out, colored)?;
    }
    Ok(())
}

fn add_primitive(
    primitive: &gltf::Primitive<'_>,
    world: Mat4,
    blob: Option<&[u8]>,
    path: &Path,
    out: &mut Mesh,
    colored: &mut bool,
) -> Result<()> {
    let reader = primitive.reader(|buffer| match buffer.source() {
        gltf::buffer::Source::Bin => blob,
        gltf::buffer::Source::Uri(_) => None,
    });
    let Some(positions) = reader.read_positions() else {
        return Ok(()); // A primitive without POSITION carries no geometry.
    };
    let offset = u32::try_from(out.v.len())
        .map_err(|_| Error::read(path, "more than 4 billion vertices"))?;
    let first_new = out.v.len();
    for p in positions {
        out.v.push(transform(world, p));
    }
    let n = out.v.len() - first_new;

    if let Some(colors) = reader.read_colors(0) {
        let list = out.colors.get_or_insert_with(|| vec![[255_u8; 3]; first_new]);
        list.resize(first_new, [255_u8; 3]);
        for c in colors.into_rgba_u8() {
            list.push([c[0], c[1], c[2]]);
        }
        list.resize(out.v.len(), [255_u8; 3]);
        *colored = true;
    } else if let Some(list) = &mut out.colors {
        list.resize(out.v.len(), [255_u8; 3]);
    }

    let indices: Vec<u32> = match reader.read_indices() {
        Some(i) => i.into_u32().collect(),
        None => {
            (0..u32::try_from(n).map_err(|_| Error::read(path, "primitive too large"))?).collect()
        }
    };
    match primitive.mode() {
        Mode::Triangles => {
            for t in indices.chunks_exact(3) {
                out.f.push([t[0] + offset, t[1] + offset, t[2] + offset]);
            }
        }
        Mode::TriangleStrip => {
            for (k, w) in indices.windows(3).enumerate() {
                let (a, b, c) = if k % 2 == 0 { (w[0], w[1], w[2]) } else { (w[1], w[0], w[2]) };
                out.f.push([a + offset, b + offset, c + offset]);
            }
        }
        Mode::TriangleFan => {
            for w in indices.windows(2).skip(1) {
                out.f.push([indices[0] + offset, w[0] + offset, w[1] + offset]);
            }
        }
        // Points and lines carry no surface; the reference has no use for them.
        Mode::Points | Mode::Lines | Mode::LineLoop | Mode::LineStrip => {}
    }
    Ok(())
}

/// `parent · child`, both column-major as glTF stores them.
fn mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0_f32; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        for (r, cell) in col.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

fn transform(m: Mat4, p: [f32; 3]) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    for (r, o) in out.iter_mut().enumerate() {
        *o = f64::from(m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r]);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these tests assert exact coordinates on purpose")]

    use super::{IDENTITY, mul, transform};

    #[test]
    fn the_identity_leaves_a_point_alone() {
        assert_eq!(transform(IDENTITY, [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        assert_eq!(mul(IDENTITY, IDENTITY), IDENTITY);
    }

    #[test]
    fn a_parent_scale_composes_onto_a_child_translation() {
        // column-major: the translation lives in column 3.
        let mut scale = IDENTITY;
        scale[0][0] = 2.0;
        scale[1][1] = 2.0;
        scale[2][2] = 2.0;
        let mut shift = IDENTITY;
        shift[3] = [1.0, 0.0, 0.0, 1.0];
        // The child's translation must be scaled by the parent, not added after it.
        assert_eq!(transform(mul(scale, shift), [0.0, 0.0, 0.0]), [2.0, 0.0, 0.0]);
        assert_eq!(transform(mul(shift, scale), [0.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }
}
