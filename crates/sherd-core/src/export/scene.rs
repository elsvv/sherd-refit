//! `scene.glb`: the assembly as a glTF 2.0 scene a 3D program opens with every fragment named —
//! Blender, 3ds Max, any glTF viewer — and the scene [`viewer`](super::viewer) embeds.
//!
//! One node per fragment, named by its file stem and carrying **the fragment's own matrix** — the
//! one `transforms.json`, `transforms.csv` and `matrices/` hold — over a simplified copy of its
//! original mesh in the original's coordinates. Its parent is one node per group that adds a
//! translation and nothing else, so that the groups R §8.2 centred on one origin stand side by
//! side instead of inside each other: the assembled groups in a row, the unassembled fragments in
//! a grid below it. Drop the group node's translation and the placement is exact.
//!
//! The meshes are for looking at. `--viewer-faces` faces over the whole collection, shared out by
//! surface area between [`MIN_FACES`] and [`MAX_FACES`] a fragment; full resolution is `placed/`.
//! Colours are converted to linear light, which is what glTF's `COLOR_0` is, and stored as
//! 16-bit fractions so the dark end of a sherd does not band.

use std::path::{Path, PathBuf};

use meshopt::{SimplifyOptions, VertexDataAdapter};
use nalgebra::Matrix4;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde_json::{Value, json};

use super::Export;
use crate::error::{Error, Result};
use crate::memory::{self, Budget, MemorySemaphore, reservation};
use crate::mesh::Mesh;
use crate::{CORE_VERSION, GIT_COMMIT};

/// The file, under the output directory.
pub const SCENE_FILE: &str = "scene.glb";
/// `--viewer-faces`' default: the faces the whole scene may hold.
pub const DEFAULT_FACES: usize = 600_000;
/// The fewest faces a fragment is drawn with, however small its share.
pub const MIN_FACES: usize = 2_000;
/// The most faces a fragment is drawn with, however large its share.
pub const MAX_FACES: usize = 60_000;

/// `meshopt`'s error bound, set out of reach so that the face count is what stops it.
const NO_ERROR_CAP: f32 = 1e9;
const ARRAY_BUFFER: u32 = 34_962;
const ELEMENT_ARRAY_BUFFER: u32 = 34_963;
const UNSIGNED_SHORT: u32 = 5_123;
const UNSIGNED_INT: u32 = 5_125;
const FLOAT: u32 = 5_126;

/// One fragment's display mesh, in its original file's coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DisplayMesh {
    /// Vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Unit vertex normals, area-weighted over the simplified faces.
    pub normals: Vec<[f32; 3]>,
    /// Linear RGBA as 16-bit fractions, or `None` when the scan has no colours.
    pub colors: Option<Vec<[u16; 4]>>,
    /// Triangles, indexing `positions`.
    pub faces: Vec<[u32; 3]>,
}

/// What [`write_scene`] wrote, for the viewer that embeds it.
#[derive(Clone, Debug)]
pub struct Scene {
    /// Where the file is.
    pub path: PathBuf,
    /// The file's bytes.
    pub bytes: Vec<u8>,
    /// Whether each fragment's scan carried colours, by [`FragId`](crate::types::FragId).
    pub coloured: Vec<bool>,
    /// The faces the scene holds.
    pub faces: usize,
}

/// Loads every original, simplifies it, lays the groups out and writes `scene.glb`.
///
/// The originals are read in parallel under the same memory semaphore R §11.4's writer uses, and
/// collected by index, so two runs write the same bytes.
pub fn write_scene(
    out_dir: &Path,
    ex: &Export<'_>,
    areas: &[f64],
    total_faces: usize,
    budget: Budget,
) -> Result<Scene> {
    let targets = face_targets(areas, total_faces);
    let semaphore = MemorySemaphore::new(budget);
    let loaded: Vec<Result<DisplayMesh>> = (0..ex.sources.len())
        .into_par_iter()
        .map(|n| {
            let path = &ex.sources[n];
            let _permit = semaphore.acquire(memory::scan_faces(path).map_or(0, reservation));
            let mesh = crate::io::load_mesh(path)?;
            Ok(display_mesh(&mesh, targets[n]))
        })
        .collect();
    let mut meshes = Vec::with_capacity(loaded.len());
    for mesh in loaded {
        meshes.push(mesh?);
    }
    let offsets = layout(ex, &meshes);
    let bytes = glb(ex, &meshes, &offsets);
    let path = out_dir.join(SCENE_FILE);
    std::fs::write(&path, &bytes).map_err(|e| Error::write(&path, e))?;
    Ok(Scene {
        path,
        bytes,
        coloured: meshes.iter().map(|m| m.colors.is_some()).collect(),
        faces: meshes.iter().map(|m| m.faces.len()).sum(),
    })
}

/// Each fragment's face budget: its share of `total` by surface area, clamped to
/// [`MIN_FACES`]..=[`MAX_FACES`].
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a face count from a positive area share, clamped into a small window"
)]
pub fn face_targets(areas: &[f64], total: usize) -> Vec<usize> {
    let valid = |a: f64| a.is_finite() && a > 0.0;
    let sum: f64 = areas.iter().copied().filter(|&a| valid(a)).sum();
    let even = total as f64 / areas.len().max(1) as f64;
    areas
        .iter()
        .map(|&a| {
            let share = if sum > 0.0 && valid(a) { a / sum * total as f64 } else { even };
            (share as usize).clamp(MIN_FACES, MAX_FACES)
        })
        .collect()
}

/// `mesh` simplified to `target` faces when it has more, compacted, with normals and linear
/// colours. The kept vertices are the original's own: `meshopt` collapses edges onto existing
/// vertices, so a kept vertex keeps its scanned colour.
pub fn display_mesh(mesh: &Mesh, target: usize) -> DisplayMesh {
    let faces =
        if mesh.f.len() > target { simplify(&mesh.v, &mesh.f, target) } else { mesh.f.clone() };
    let mut slot = vec![u32::MAX; mesh.v.len()];
    let mut kept: Vec<usize> = Vec::new();
    let mut compact = Vec::with_capacity(faces.len());
    for t in &faces {
        let mut tri = [0_u32; 3];
        for (corner, &i) in tri.iter_mut().zip(t) {
            let i = i as usize;
            if slot[i] == u32::MAX {
                slot[i] = u32::try_from(kept.len()).expect("fewer than 2^32 vertices");
                kept.push(i);
            }
            *corner = slot[i];
        }
        compact.push(tri);
    }
    let positions = kept.iter().map(|&i| to_f32(mesh.v[i])).collect();
    let normals = vertex_normals(&mesh.v, &kept, &compact);
    let colors = mesh.colors.as_ref().map(|c| {
        let table = linear_table();
        kept.iter()
            .map(|&i| {
                let [r, g, b] = c[i];
                [table[usize::from(r)], table[usize::from(g)], table[usize::from(b)], u16::MAX]
            })
            .collect()
    });
    DisplayMesh { positions, normals, colors, faces: compact }
}

/// The translation each group's node adds, by group, so that no two groups overlap: the assembled
/// groups first, largest first, packed on shelves into a block about as wide as a screen is to its
/// height; the unassembled fragments on shelves below it, each at its own size, at least as wide
/// as the block. A row of seventeen groups is a strip no screen shows well, and one large sherd
/// must not set the pitch for every small one.
pub fn layout(ex: &Export<'_>, meshes: &[DisplayMesh]) -> Vec<[f64; 3]> {
    let bounds: Vec<Option<Bounds>> = ex
        .groups
        .iter()
        .map(|g| {
            let mut b = Bounds::EMPTY;
            for &n in g {
                let pose = &ex.poses[n as usize];
                for &p in &meshes[n as usize].positions {
                    b.add(apply(pose, p));
                }
            }
            b.valid().then_some(b)
        })
        .collect();
    let of = |assembled: bool| -> Vec<(usize, Bounds)> {
        (0..ex.groups.len())
            .filter(|&k| (ex.groups[k].len() > 1) == assembled)
            .filter_map(|k| bounds[k].map(|b| (k, b)))
            .collect()
    };
    let (groups, singles) = (of(true), of(false));
    let mut offsets = vec![[0.0; 3]; ex.groups.len()];
    let (mut top, mut width) = (0.0_f64, 0.0_f64);
    if !groups.is_empty() {
        let spacing = 0.2 * median_size(&groups);
        width = block_width(&groups, spacing);
        top = shelve(&groups, width, spacing, 0.0, &mut offsets) - spacing;
    }
    if !singles.is_empty() {
        let spacing = 0.25 * median_size(&singles);
        let width = width.max(block_width(&singles, spacing));
        shelve(&singles, width, spacing, top, &mut offsets);
    }
    offsets
}

/// The median of the items' largest extents.
fn median_size(items: &[(usize, Bounds)]) -> f64 {
    let mut sizes: Vec<f64> = items.iter().map(|(_, b)| b.size()).collect();
    sizes.sort_by(f64::total_cmp);
    sizes[sizes.len() / 2]
}

/// A shelf width that makes the packed block about 1.6 times as wide as it is tall, and never
/// narrower than the widest item.
fn block_width(items: &[(usize, Bounds)], spacing: f64) -> f64 {
    let area: f64 =
        items.iter().map(|(_, b)| (b.width(0) + spacing) * (b.width(1) + spacing)).sum();
    let widest = items.iter().map(|(_, b)| b.width(0)).fold(0.0, f64::max);
    (1.6 * area).sqrt().max(widest)
}

/// Packs `items` left to right on shelves `width` wide, downward from `top`, each centred in `z`;
/// returns the bottom of the last shelf.
fn shelve(
    items: &[(usize, Bounds)],
    width: f64,
    spacing: f64,
    top: f64,
    offsets: &mut [[f64; 3]],
) -> f64 {
    let (mut at_x, mut shelf_top, mut shelf_height) = (0.0_f64, top, 0.0_f64);
    for &(k, b) in items {
        let (w, h) = (b.width(0), b.width(1));
        if at_x > 0.0 && at_x + w > width {
            shelf_top -= shelf_height + spacing;
            at_x = 0.0;
            shelf_height = 0.0;
        }
        let centre = b.centre();
        offsets[k] = [at_x + w / 2.0 - centre[0], shelf_top - h / 2.0 - centre[1], -centre[2]];
        at_x += w + spacing;
        shelf_height = shelf_height.max(h);
    }
    shelf_top - shelf_height
}

/// The scene as GLB bytes: glTF 2.0's binary container, one JSON chunk and one buffer.
pub fn glb(ex: &Export<'_>, meshes: &[DisplayMesh], offsets: &[[f64; 3]]) -> Vec<u8> {
    let mut buffers = Buffers::default();
    let (gltf_meshes, mesh_of) = mesh_entries(ex, meshes, &mut buffers);
    let nodes = node_tree(ex, &mesh_of, offsets);
    let bin_length = buffers.bin.len();
    // Material 1, the plain clay, only exists where a scan without colours needs it.
    let mut materials = vec![material("scan colours", [1.0, 1.0, 1.0])];
    if meshes.iter().any(|m| !m.faces.is_empty() && m.colors.is_none()) {
        materials.push(material("clay", [0.52, 0.33, 0.22]));
    }
    let doc = json!({
        "asset": {
            "version": "2.0",
            "generator": format!("sherd-refit {CORE_VERSION} ({GIT_COMMIT})"),
        },
        "scene": 0,
        "scenes": [{
            "name": ex.collection,
            "nodes": [0],
            "extras": {
                "units": "the units of the input files",
                "wall_thickness": ex.thickness,
                "fragment_matrix": "each fragment node's matrix is its transforms.json matrix: \
                                    original file to assembly, p' = M p",
                "group_translation": "a group node only translates, to set the groups apart; \
                                      without it the placement is exact",
                "meshes": "simplified for display; full resolution is in placed/",
            },
        }],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "accessors": buffers.accessors,
        "bufferViews": buffers.views,
        "buffers": [{ "byteLength": bin_length }],
    });
    container(&doc, buffers.bin)
}

/// A matte, two-sided material: glTF's default is fully metallic, which renders a sherd black.
fn material(name: &str, colour: [f64; 3]) -> Value {
    json!({
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [colour[0], colour[1], colour[2], 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.9,
        },
        "doubleSided": true,
    })
}

/// One glTF mesh per fragment with faces, its arrays appended to `buffers`; and which mesh each
/// fragment got. Material 0 is the scan's colours, 1 the plain clay a colourless scan is drawn in.
fn mesh_entries(
    ex: &Export<'_>,
    meshes: &[DisplayMesh],
    buffers: &mut Buffers,
) -> (Vec<Value>, Vec<Option<usize>>) {
    let mut entries = Vec::new();
    let mut mesh_of = vec![None; meshes.len()];
    for (n, mesh) in meshes.iter().enumerate() {
        if mesh.faces.is_empty() {
            continue;
        }
        let (lo, hi) = extent(&mesh.positions);
        let view = buffers.view(&le_f32(&mesh.positions), ARRAY_BUFFER);
        let position = buffers.accessor(view, FLOAT, mesh.positions.len(), "VEC3", Some((lo, hi)));
        let view = buffers.view(&le_f32(&mesh.normals), ARRAY_BUFFER);
        let normal = buffers.accessor(view, FLOAT, mesh.normals.len(), "VEC3", None);
        let mut attributes = json!({ "POSITION": position, "NORMAL": normal });
        if let Some(colors) = &mesh.colors {
            let view = buffers.view(&le_u16(colors), ARRAY_BUFFER);
            let color = buffers.accessor(view, UNSIGNED_SHORT, colors.len(), "VEC4", None);
            buffers.accessors[color]["normalized"] = json!(true);
            attributes["COLOR_0"] = json!(color);
        }
        let (bytes, component) = indices(mesh);
        let view = buffers.view(&bytes, ELEMENT_ARRAY_BUFFER);
        let index = buffers.accessor(view, component, mesh.faces.len() * 3, "SCALAR", None);
        mesh_of[n] = Some(entries.len());
        entries.push(json!({
            "name": ex.names[n],
            "primitives": [{
                "attributes": attributes,
                "indices": index,
                "material": usize::from(mesh.colors.is_none()),
                "mode": 4,
            }],
        }));
    }
    (entries, mesh_of)
}

/// The node tree: the collection at the root, one node per assembled group under it (a
/// translation and nothing else), the singletons' group nodes under `unassembled`, and one node
/// per fragment carrying its own matrix.
fn node_tree(ex: &Export<'_>, mesh_of: &[Option<usize>], offsets: &[[f64; 3]]) -> Vec<Value> {
    let assembled = ex.assembled();
    let mut nodes: Vec<Value> = vec![Value::Null];
    let (mut row, mut singles) = (Vec::new(), Vec::new());
    for (k, members) in ex.groups.iter().enumerate() {
        let group_node = nodes.len();
        nodes.push(Value::Null);
        let mut children = Vec::with_capacity(members.len());
        for &n in members {
            let n = n as usize;
            children.push(nodes.len());
            let mut node = json!({
                "name": ex.names[n],
                "matrix": column_major(&ex.poses[n]),
                "extras": {
                    "fragment": ex.names[n],
                    "index": n,
                    "source_file": ex.source_name(n),
                    "group": k,
                    "assembled": assembled[n],
                },
            });
            if let Some(m) = mesh_of[n] {
                node["mesh"] = json!(m);
            }
            nodes.push(node);
        }
        let names: Vec<&String> = members.iter().map(|&n| &ex.names[n as usize]).collect();
        nodes[group_node] = json!({
            "name": format!("group_{k}"),
            "translation": offsets[k],
            "children": children,
            "extras": { "group": k, "members": names, "assembled": members.len() > 1 },
        });
        if members.len() > 1 { row.push(group_node) } else { singles.push(group_node) }
    }
    if !singles.is_empty() {
        row.push(nodes.len());
        nodes.push(json!({ "name": "unassembled", "children": singles }));
    }
    nodes[0] = json!({ "name": ex.collection, "children": row });
    nodes
}

/// The three GLB pieces: the 12-byte header, the JSON chunk padded with spaces and the binary
/// chunk padded with zeros, each to four bytes.
fn container(doc: &Value, mut bin: Vec<u8>) -> Vec<u8> {
    let mut json = serde_json::to_vec(doc).expect("a JSON tree serialises");
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let word = |x: usize| u32::try_from(x).expect("a scene under 4 GiB").to_le_bytes();
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2_u32.to_le_bytes());
    out.extend_from_slice(&word(total));
    out.extend_from_slice(&word(json.len()));
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&word(bin.len()));
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}

/// The buffer the scene's arrays share, and the views and accessors onto it.
#[derive(Debug, Default)]
struct Buffers {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Buffers {
    /// `bytes` appended at a four-byte boundary, as glTF wants every vertex attribute aligned.
    fn view(&mut self, bytes: &[u8], target: u32) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        self.views.push(json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len(),
            "target": target,
        }));
        self.views.len() - 1
    }

    fn accessor(
        &mut self,
        view: usize,
        component: u32,
        count: usize,
        kind: &str,
        bounds: Option<([f32; 3], [f32; 3])>,
    ) -> usize {
        let mut accessor =
            json!({ "bufferView": view, "componentType": component, "count": count, "type": kind });
        if let Some((lo, hi)) = bounds {
            accessor["min"] = json!(lo);
            accessor["max"] = json!(hi);
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }
}

/// The indices as 16-bit words while every index fits below glTF's reserved `u16::MAX`, 32-bit
/// otherwise.
fn indices(mesh: &DisplayMesh) -> (Vec<u8>, u32) {
    if u16::try_from(mesh.positions.len()).is_ok() {
        let mut out = Vec::with_capacity(mesh.faces.len() * 6);
        for &i in mesh.faces.as_flattened() {
            out.extend_from_slice(&u16::try_from(i).expect("below u16::MAX").to_le_bytes());
        }
        (out, UNSIGNED_SHORT)
    } else {
        let mut out = Vec::with_capacity(mesh.faces.len() * 12);
        for &i in mesh.faces.as_flattened() {
            out.extend_from_slice(&i.to_le_bytes());
        }
        (out, UNSIGNED_INT)
    }
}

fn le_f32(values: &[[f32; 3]]) -> Vec<u8> {
    values.as_flattened().iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_u16(values: &[[u16; 4]]) -> Vec<u8> {
    values.as_flattened().iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn extent(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in points {
        for ((l, h), &x) in lo.iter_mut().zip(hi.iter_mut()).zip(p) {
            *l = l.min(x);
            *h = h.max(x);
        }
    }
    (lo, hi)
}

/// glTF's `matrix`: sixteen numbers, column by column.
fn column_major(m: &Matrix4<f64>) -> [f64; 16] {
    std::array::from_fn(|i| m[(i % 4, i / 4)] + 0.0)
}

fn apply(pose: &Matrix4<f64>, point: [f32; 3]) -> [f64; 3] {
    let q = point.map(f64::from);
    [0, 1, 2]
        .map(|r| pose[(r, 0)] * q[0] + pose[(r, 1)] * q[1] + pose[(r, 2)] * q[2] + pose[(r, 3)])
}

#[allow(clippy::cast_possible_truncation, reason = "display coordinates are f32, as glTF's are")]
fn to_f32(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

fn simplify(v: &[[f64; 3]], f: &[[u32; 3]], target: usize) -> Vec<[u32; 3]> {
    let positions: Vec<f32> = v.iter().flat_map(|&p| to_f32(p)).collect();
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0)
        .expect("3 f32 per vertex is a stride-12 offset-0 buffer");
    let indices = meshopt::simplify(
        f.as_flattened(),
        &adapter,
        target * 3,
        NO_ERROR_CAP,
        SimplifyOptions::None,
        None,
    );
    indices.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Area-weighted vertex normals: each face adds its unnormalised cross product to its corners.
fn vertex_normals(vertices: &[[f64; 3]], kept: &[usize], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut sum = vec![[0.0_f64; 3]; kept.len()];
    for tri in faces {
        let [p0, p1, p2] = tri.map(|i| vertices[kept[i as usize]]);
        let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let w = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross =
            [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        for &i in tri {
            for (acc, x) in sum[i as usize].iter_mut().zip(cross) {
                *acc += x;
            }
        }
    }
    sum.into_iter()
        .map(|n| {
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if length > 0.0 { to_f32(n.map(|x| x / length)) } else { [0.0, 0.0, 1.0] }
        })
        .collect()
}

/// sRGB bytes to linear 16-bit fractions (IEC 61966-2-1's curve).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a fraction in [0, 1] scaled to u16 and rounded"
)]
fn linear_table() -> [u16; 256] {
    let mut table = [0_u16; 256];
    for c in 0..=255_u8 {
        let s = f64::from(c) / 255.0;
        let l = if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) };
        table[usize::from(c)] = (l * f64::from(u16::MAX)).round() as u16;
    }
    table
}

/// An axis-aligned box.
#[derive(Clone, Copy, Debug)]
struct Bounds {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Bounds {
    const EMPTY: Self = Self { lo: [f64::INFINITY; 3], hi: [f64::NEG_INFINITY; 3] };

    fn add(&mut self, p: [f64; 3]) {
        for ((l, h), x) in self.lo.iter_mut().zip(self.hi.iter_mut()).zip(p) {
            *l = l.min(x);
            *h = h.max(x);
        }
    }

    fn valid(&self) -> bool {
        self.lo[0] <= self.hi[0]
    }

    fn centre(&self) -> [f64; 3] {
        [0, 1, 2].map(|k| f64::midpoint(self.lo[k], self.hi[k]))
    }

    fn width(&self, axis: usize) -> f64 {
        self.hi[axis] - self.lo[axis]
    }

    fn size(&self) -> f64 {
        (0..3).map(|k| self.hi[k] - self.lo[k]).fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the translations compared here are exact")]

    use super::{DisplayMesh, MAX_FACES, MIN_FACES, display_mesh, face_targets, glb, layout};
    use crate::export::Export;
    use crate::mesh::Mesh;
    use nalgebra::Matrix4;
    use std::path::PathBuf;

    /// A closed tetrahedron with one colour per vertex, moved by `dx`.
    fn tetra(dx: f64) -> Mesh {
        let v = vec![[dx, 0.0, 0.0], [dx + 1.0, 0.0, 0.0], [dx, 1.0, 0.0], [dx, 0.0, 1.0]];
        let f = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        Mesh { v, f, colors: Some(vec![[0, 0, 0], [255, 255, 255], [128, 64, 32], [10, 20, 30]]) }
    }

    fn export<'a>(
        names: &'a [String],
        sources: &'a [PathBuf],
        poses: &'a [Matrix4<f64>],
        groups: &'a [Vec<u32>],
    ) -> Export<'a> {
        Export {
            collection: "test",
            names,
            sources,
            poses,
            groups,
            candidates: &[],
            used: &[],
            tiers: true,
            review: None,
            thickness: 1.0,
        }
    }

    #[test]
    fn shares_follow_the_area_inside_the_window() {
        let t = face_targets(&[1.0, 3.0], 400_000);
        assert_eq!(t, vec![MAX_FACES.min(100_000), MAX_FACES]);
        assert_eq!(face_targets(&[1.0, 1_000_000.0], 100_000)[0], MIN_FACES);
        // No usable area at all: an even share, inside the window.
        assert_eq!(face_targets(&[0.0, f64::NAN], 10_000), vec![5_000, 5_000]);
    }

    #[test]
    fn a_small_mesh_is_kept_whole_and_its_colours_go_linear() {
        let d: DisplayMesh = display_mesh(&tetra(0.0), 1000);
        assert_eq!(d.faces.len(), 4);
        assert_eq!(d.positions.len(), 4);
        let colors = d.colors.expect("the scan's colours travel");
        // The vertices are renumbered in order of first use, so each is found by its position.
        let at = |p: [f32; 3]| colors[d.positions.iter().position(|&q| q == p).expect("a vertex")];
        assert_eq!(at([0.0, 0.0, 0.0]), [0, 0, 0, u16::MAX]);
        assert_eq!(at([1.0, 0.0, 0.0]), [u16::MAX; 4]);
        // sRGB 128 is 21.6 % linear light, not half.
        assert!((f64::from(at([0.0, 1.0, 0.0])[0]) / 65_535.0 - 0.2158).abs() < 1e-3);
        for n in &d.normals {
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-5, "unit normals");
        }
    }

    /// A tetrahedron `size` across.
    fn tetra_of(size: f64) -> Mesh {
        let mut m = tetra(0.0);
        for v in &mut m.v {
            *v = v.map(|x| x * size);
        }
        m
    }

    #[test]
    fn unassembled_fragments_are_shelved_below_the_row_without_overlap() {
        let names: Vec<String> = ["a", "b", "c", "d", "e", "f"].map(String::from).to_vec();
        let sources: Vec<PathBuf> =
            names.iter().map(|n| PathBuf::from(format!("{n}.ply"))).collect();
        let poses = vec![Matrix4::identity(); 6];
        let groups = vec![vec![0, 1], vec![2], vec![3], vec![4], vec![5]];
        let ex = export(&names, &sources, &poses, &groups);
        let sizes = [1.0, 1.0, 5.0, 0.5, 2.0, 0.7];
        let meshes: Vec<DisplayMesh> =
            sizes.iter().map(|&s| display_mesh(&tetra_of(s), 100)).collect();
        let offsets = layout(&ex, &meshes);
        // Each group's footprint in x and y once its offset is applied.
        let foot = |k: usize| {
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for &n in &groups[k] {
                for p in &meshes[n as usize].positions {
                    for axis in 0..2 {
                        let x = f64::from(p[axis]) + offsets[k][axis];
                        lo[axis] = lo[axis].min(x);
                        hi[axis] = hi[axis].max(x);
                    }
                }
            }
            (lo, hi)
        };
        let row = foot(0);
        for i in 1..groups.len() {
            let (lo, hi) = foot(i);
            assert!(hi[1] < row.0[1], "group {i} sits below the assembled row");
            for j in i + 1..groups.len() {
                let (lo2, hi2) = foot(j);
                let apart =
                    hi[0] <= lo2[0] || hi2[0] <= lo[0] || hi[1] <= lo2[1] || hi2[1] <= lo[1];
                assert!(apart, "groups {i} and {j} overlap");
            }
        }
    }

    #[test]
    fn many_groups_make_a_block_and_not_a_strip() {
        let names: Vec<String> = (0..24).map(|i| format!("f{i}")).collect();
        let sources: Vec<PathBuf> =
            names.iter().map(|n| PathBuf::from(format!("{n}.ply"))).collect();
        let mut poses = vec![Matrix4::identity(); 24];
        for pose in poses.iter_mut().skip(1).step_by(2) {
            pose[(0, 3)] = 1.0;
        }
        let groups: Vec<Vec<u32>> = (0..12).map(|g| vec![2 * g, 2 * g + 1]).collect();
        let ex = export(&names, &sources, &poses, &groups);
        let meshes: Vec<DisplayMesh> = (0..24).map(|_| display_mesh(&tetra(0.0), 100)).collect();
        let offsets = layout(&ex, &meshes);
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        let mut boxes = Vec::new();
        for (k, g) in groups.iter().enumerate() {
            let (mut glo, mut ghi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for &n in g {
                for p in &meshes[n as usize].positions {
                    for axis in 0..2 {
                        let x =
                            f64::from(p[axis]) + poses[n as usize][(axis, 3)] + offsets[k][axis];
                        glo[axis] = glo[axis].min(x);
                        ghi[axis] = ghi[axis].max(x);
                    }
                }
            }
            for axis in 0..2 {
                lo[axis] = lo[axis].min(glo[axis]);
                hi[axis] = hi[axis].max(ghi[axis]);
            }
            boxes.push((glo, ghi));
        }
        let aspect = (hi[0] - lo[0]) / (hi[1] - lo[1]);
        assert!((0.8..3.0).contains(&aspect), "a block, not a strip: {aspect}");
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                let ((a0, a1), (b0, b1)) = (boxes[i], boxes[j]);
                let apart = a1[0] <= b0[0] || b1[0] <= a0[0] || a1[1] <= b0[1] || b1[1] <= a0[1];
                assert!(apart, "groups {i} and {j} overlap");
            }
        }
    }

    #[test]
    fn the_groups_stand_apart_and_the_file_reads_back_as_gltf() {
        let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        let sources: Vec<PathBuf> =
            names.iter().map(|n| PathBuf::from(format!("{n}.ply"))).collect();
        let mut moved = Matrix4::identity();
        moved[(0, 3)] = 0.5;
        moved[(1, 3)] = -2.0;
        let poses = vec![Matrix4::identity(), moved, Matrix4::identity()];
        let groups = vec![vec![0, 1], vec![2]];
        let ex = export(&names, &sources, &poses, &groups);
        let meshes: Vec<DisplayMesh> = (0..3).map(|_| display_mesh(&tetra(0.0), 1000)).collect();
        let offsets = layout(&ex, &meshes);
        assert!(offsets[0] != offsets[1], "the singleton does not sit inside the group");

        let bytes = glb(&ex, &meshes, &offsets);
        assert_eq!(&bytes[..4], b"glTF");
        let gltf = gltf::Gltf::from_slice(&bytes).expect("a valid GLB");
        // The reader is built without `gltf`'s `names` feature, so the names come from the JSON
        // chunk itself: its length is the word after the 12-byte header.
        let length = u32::from_le_bytes(bytes[12..16].try_into().expect("four bytes")) as usize;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes[20..20 + length]).expect("JSON");
        let nodes = json["nodes"].as_array().expect("nodes");
        let node_index = |name: &str| nodes.iter().position(|n| n["name"] == name).expect(name);
        let doc = &gltf.document;
        let b = doc.nodes().nth(node_index("b")).expect("node b");
        let m = b.transform().matrix();
        assert_eq!(m[3][0], 0.5, "column-major: the translation is the fourth column");
        assert_eq!(m[3][1], -2.0);
        assert!(nodes.iter().any(|n| n["name"] == "unassembled"));
        let group = doc.nodes().nth(node_index("group_1")).expect("the singleton's group");
        let (t, _, _) = group.transform().decomposed();
        #[allow(clippy::cast_possible_truncation, reason = "glTF's translation is f32")]
        let want = offsets[1].map(|x| x as f32);
        assert_eq!(t, want);
        let mesh = b.mesh().expect("b has a mesh");
        let primitive = mesh.primitives().next().expect("one primitive");
        assert_eq!(primitive.attributes().count(), 3, "POSITION, NORMAL, COLOR_0");
    }
}
