//! D §10.2's `outputs` row: `transforms.json`, `report.json`, the placed meshes of R §11.4 and the
//! previews of R §11.5.
//!
//! # Injected
//!
//! Four things are compared, and they are compared at four different strengths, because the four
//! outputs are four different kinds of object.
//!
//! * **`transforms.json`** — the port recentres the reference's own final poses (R §8.2 over
//!   `refine/poses_final.json`, or `assembly/poses.json` where R §9 had nothing to refine) with the
//!   reference's own groups and its own assembly-stage samples, and writes the file. `thickness`,
//!   the groups, every `group` index and every `placed` flag must be identical; the poses are
//!   measured the way every pose row of D §10.2 is, as the largest displacement they give a
//!   fragment's own samples, at [`POSE_T`].
//! * **`report.json`** — two rows. `schema` reads the reference's own file into the port's
//!   [`ReportJson`], writes it back out and diffs the two JSON
//!   trees: a key the port's type does not know about is a key it would silently drop, and this is
//!   what finds it. `candidates` renders the reference's own candidate list through the port's own
//!   serialiser and compares it against the file entry for entry, which checks the score, the
//!   verdict, the pose and every one of R §6's names.
//! * **`placed/<name>.ply` and `assembly_<k>.ply`** — written by the port from the reference's own
//!   poses and compared by **SHA-256**, which is the only honest way to say "byte-identical". The
//!   reference's own files are not in the dump (they are hundreds of megabytes per collection);
//!   `tools/dump_outputs.py` writes them into a scratch directory, hashes them and keeps
//!   `outputs/placed.sha256.json`. The port writes with Open3D's header comment for the comparison,
//!   because the comment is the one byte range that names the writer rather than the mesh.
//! * **`preview_<k>.png` and `preview_segmentation.png`** — rendered by the port from the
//!   reference's own samples (the `pick`/`u`/`v` the same tool dumps, which is the form D §10.2's
//!   `samples` row already compares points through) at the reference's own views (PMC-10 makes the
//!   eigenvector signs library-defined), and compared **pixel for pixel** against the reference's
//!   own render with the caption left off. The caption is PMC-20: PIL's default font is an
//!   anti-aliased FreeType face and the port draws its own 5×7 bitmap glyphs, so the row that
//!   would compare it instead *measures* it — how many pixels of the image the caption covers on
//!   the reference's side, against a cap.
//!
//! # Native
//!
//! Nothing here can be a parity claim: the port's own preview is drawn from its own samples
//! (PMC-9) at its own views (PMC-10). What native mode checks is that the two are the same *shape*
//! — the angle between the port's own principal axes and the reference's, with the sign divided
//! out — and that the renderer and the JSON writer are functions: two renders of one input are the
//! same bytes, and a `transforms.json` written and read back is the poses that went in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nalgebra::Matrix4;
use sherd_core::assembly::recenter;
use sherd_core::error::{Error, Result};
use sherd_core::fragment::samples::points_from_uniforms;
use sherd_core::mesh::geometry::{FaceGeometry, face_geometry, pairwise_sum};
use sherd_core::render::{
    self, PALETTE, Paint, Rgb, Splat, View, placed_splat, principal_axes, render_views,
};
use sherd_core::report::{self, ReportJson};
use sherd_core::types::FragId;

use super::Collection;
use super::assembly::{self, piece_views, read_poses, reference_pieces};
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// The scope every row of this stage is filed under.
const SCOPE: &str = "(collection)";

/// How far a pose in `transforms.json` may move a fragment's own samples, in wall thicknesses.
///
/// The same `1e-9 t` the `assembly` row's `recentre` check already meets end to end on all eight
/// dumps (step D1): the port recentres the reference's own poses here, so the only arithmetic
/// between the two files is R §8.2's mean and a subtraction. D §10.2's row said "as refine"
/// (0.02 t) and step D2 tightened it to what is measured, which is seven orders of magnitude
/// smaller.
pub const POSE_T: f64 = 1e-9;

/// How many pixels of a preview the caption may cover (PMC-20).
///
/// Not a tolerance on a difference — the caption is *given up*, not approximated — but a bound on
/// how much of the image is given up. A 900×700 view is 630 000 pixels and the reference's caption
/// is a single 11-pixel line; the cap is set at 8 000, which is roughly ten times the largest
/// caption the eight dumps produce and still 1.3 % of one view.
pub const LABEL_PIXELS: f64 = 8000.0;

/// How far the port's own principal axis may sit from the reference's, in degrees (PMC-10).
pub const AXIS_DEG: f64 = 1.0;

/// `outputs/placed.sha256.json`: what the reference's own R §11.4 writer produced.
#[derive(Debug, serde::Deserialize)]
struct RefMeshes {
    /// One entry per file written, keyed by its path under the output directory.
    files: BTreeMap<String, RefMesh>,
    /// The SHA-256 of every fragment's mesh as `load_mesh` leaves it, before any pose.
    sources: BTreeMap<String, String>,
}

/// One `placed/<name>.ply` or `assembly_<k>.ply`.
#[derive(Debug, serde::Deserialize)]
struct RefMesh {
    sha256: String,
    size: u64,
    vertices: u64,
    faces: u64,
    colors: bool,
    /// The fragments the file holds, in the order they were written.
    members: Vec<String>,
}

/// `preview_<tag>.meta.json`.
#[derive(Debug, serde::Deserialize)]
struct RefPreview {
    width: usize,
    height: usize,
    #[allow(dead_code, reason = "the caption is PMC-20; it is measured, not reproduced")]
    label: String,
    n_points: usize,
    views: Vec<[[f64; 3]; 2]>,
    members: Vec<String>,
    kind: String,
}

/// Runs R §11's writers against the dump.
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("outputs", mode);
    match mode {
        Mode::Injected => injected(collection, &mut report)?,
        Mode::Native => native(collection, &mut report)?,
    }
    Ok(report)
}

fn injected(collection: &Collection, report: &mut StageReport) -> Result<()> {
    transforms_rows(collection, report)?;
    report_rows(collection, report)?;
    mesh_rows(collection, report)?;
    preview_rows(collection, report)
}

// ---------------------------------------------------------------------------------------------
// transforms.json

fn transforms_rows(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let path = collection.dir.outputs_dir().join("transforms.json");
    if !path.is_file() {
        report.skip(SCOPE, "no outputs/transforms.json in the dump");
        return Ok(());
    }
    let Some(pieces) = reference_pieces(collection, report)? else { return Ok(()) };
    let views = piece_views(&pieces);
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let Some(groups) = reference_groups(collection, report, &names)? else { return Ok(()) };

    let refined = collection.dir.refine_dir().join("poses_final.json");
    let before = if refined.is_file() {
        read_poses(&refined, &names)?
    } else {
        read_poses(&collection.dir.assembly_dir().join("poses.json"), &names)?
    };
    let after = recenter(&before, &views, &groups);
    let thickness = collection.manifest.pairs.thickness_median;
    let ours = report::transforms(
        &names,
        &after,
        &groups,
        thickness,
        &collection.manifest.collection.params,
    );
    let theirs: report::Transforms = npy::read_json_as(&path)?;

    report.push(Check::exact(SCOPE, "transforms thickness", ours.thickness, theirs.thickness));
    report.push(Check::entries(
        SCOPE,
        "transforms groups",
        (0..ours.groups.len().max(theirs.groups.len()))
            .filter(|&i| ours.groups.get(i) != theirs.groups.get(i))
            .count(),
        ours.groups.len().max(theirs.groups.len()),
    ));
    let differing = names
        .iter()
        .filter(|n| match (ours.fragments.get(*n), theirs.fragments.get(*n)) {
            (Some(a), Some(b)) => a.group != b.group || a.placed != b.placed,
            _ => true,
        })
        .count();
    report.push(Check::entries(SCOPE, "transforms flags", differing, names.len()));
    report.push(Check::identical(SCOPE, "transforms params", ours.params == theirs.params, true));

    let theirs_poses: Vec<Matrix4<f64>> = names
        .iter()
        .map(|n| {
            theirs.fragments.get(n).map_or_else(Matrix4::identity, |e| assembly::matrix(&e.matrix))
        })
        .collect();
    report.push(Check::absolute(
        SCOPE,
        "transforms pose",
        assembly::worst_move(&after, &theirs_poses, &views, thickness),
        0.0,
        POSE_T,
    ));
    Ok(())
}

/// `assembly/groups.json` as [`FragId`] lists.
fn reference_groups(
    collection: &Collection,
    report: &mut StageReport,
    names: &[String],
) -> Result<Option<Vec<Vec<FragId>>>> {
    let path = collection.dir.assembly_dir().join("groups.json");
    if !path.is_file() {
        report.skip(SCOPE, "no assembly/groups.json in the dump (level min)");
        return Ok(None);
    }
    let raw: Vec<Vec<String>> = npy::read_json_as(path)?;
    let mut out = Vec::new();
    for group in &raw {
        let mut members = Vec::new();
        for name in group {
            let Some(i) = names.iter().position(|n| n == name) else {
                report.skip(SCOPE, "a group of the dump names a fragment the manifest does not");
                return Ok(None);
            };
            members.push(u32::try_from(i).expect("fewer than 2^32 fragments"));
        }
        out.push(members);
    }
    Ok(Some(out))
}

// ---------------------------------------------------------------------------------------------
// report.json

fn report_rows(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let path = collection.dir.outputs_dir().join("report.json");
    if !path.is_file() {
        report.skip(SCOPE, "no outputs/report.json in the dump");
        return Ok(());
    }
    let theirs: serde_json::Value = npy::read_json(&path)?;
    let typed: ReportJson = serde_json::from_value(theirs.clone())
        .map_err(|e| Error::fixture(&path, format!("report.json does not fit ReportJson: {e}")))?;
    let ours = serde_json::to_value(&typed)
        .map_err(|e| Error::fixture(&path, format!("ReportJson does not serialise: {e}")))?;
    let mut differences = Vec::new();
    diff_json(&ours, &theirs, "", &mut differences);
    for path in differences.iter().take(5) {
        tracing::warn!(path = %path, "report.json does not round-trip through ReportJson");
    }
    report.push(Check::entries(SCOPE, "report schema", differences.len(), json_leaves(&theirs)));

    // The candidate list, rendered by the port's own serialiser from the reference's own
    // candidates.
    let Some(candidates) = assembly::reference_candidates(collection, report)? else {
        return Ok(());
    };
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let ours: Vec<serde_json::Value> = candidates
        .iter()
        .map(|c| serde_json::to_value(report::CandidateJson::of(c, &names)))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| Error::fixture(&path, format!("a candidate does not serialise: {e}")))?;
    let theirs_candidates = theirs["candidates"].as_array().map_or(&[][..], Vec::as_slice);
    let mut differing = 0;
    for i in 0..ours.len().max(theirs_candidates.len()) {
        let mut paths = Vec::new();
        match (ours.get(i), theirs_candidates.get(i)) {
            (Some(a), Some(b)) => diff_json(a, b, "", &mut paths),
            _ => paths.push("(missing)".to_owned()),
        }
        if !paths.is_empty() {
            if differing < 5 {
                tracing::warn!(index = i, first = %paths[0], "candidate differs");
            }
            differing += 1;
        }
    }
    report.push(Check::entries(
        SCOPE,
        "report candidates",
        differing,
        ours.len().max(theirs_candidates.len()),
    ));
    Ok(())
}

/// Every leaf path where two JSON trees disagree, ignoring `timings` (wall clock) and the keys the
/// port adds on top of the reference's schema.
fn diff_json(
    ours: &serde_json::Value,
    theirs: &serde_json::Value,
    at: &str,
    out: &mut Vec<String>,
) {
    use serde_json::Value;
    match (ours, theirs) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                let path = format!("{at}/{k}");
                // `timings` are wall-clock seconds and the reference's dump nulls them; `engine`
                // is D §4.3's additive block, which the reference never wrote.
                if k == "timings" {
                    continue;
                }
                match a.get(k) {
                    Some(mine) => diff_json(mine, v, &path, out),
                    None => out.push(path),
                }
            }
            for k in a.keys() {
                if k != "engine" && k != "timings" && !b.contains_key(k) {
                    out.push(format!("{at}/{k} (only ours)"));
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                out.push(format!("{at}[] length {} against {}", a.len(), b.len()));
                return;
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                diff_json(x, y, &format!("{at}[{i}]"), out);
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let same = match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
                _ => a == b,
            };
            if !same {
                out.push(format!("{at} = {a} against {b}"));
            }
        }
        _ => {
            if ours != theirs {
                out.push(at.to_owned());
            }
        }
    }
}

/// How many leaves a JSON tree has — the denominator of the schema row.
fn json_leaves(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(m) => m.values().map(json_leaves).sum::<usize>().max(1),
        serde_json::Value::Array(a) => a.iter().map(json_leaves).sum::<usize>().max(1),
        _ => 1,
    }
}

// ---------------------------------------------------------------------------------------------
// placed meshes

#[allow(clippy::too_many_lines, reason = "one row per output file, and the reasons they skip")]
fn mesh_rows(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let index_path = collection.dir.outputs_dir().join("placed.sha256.json");
    if !index_path.is_file() {
        report.skip(
            SCOPE,
            "no outputs/placed.sha256.json in the dump: run tools/dump_outputs.py DUMP INPUT to \
             write the reference's own R §11.4 meshes and hash them",
        );
        return Ok(());
    }
    let theirs: RefMeshes = npy::read_json_as(&index_path)?;
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let Some(groups) = reference_groups(collection, report, &names)? else { return Ok(()) };

    let transforms_path = collection.dir.outputs_dir().join("transforms.json");
    if !transforms_path.is_file() {
        report.skip(SCOPE, "no outputs/transforms.json to place the meshes with");
        return Ok(());
    }
    let poses = assembly::read_transforms(&transforms_path, &names)?;

    let mut paths = Vec::new();
    for fragment in &collection.fragments {
        let Some(source) = &fragment.source else {
            report.skip(SCOPE, "no source file (pass --input DIR) to place the meshes from");
            return Ok(());
        };
        paths.push(source.clone());
    }

    // Which fragments the two implementations read the same way. A placed mesh cannot be
    // byte-identical when its input is not, and on the five OBJ collections it is not: D §10.2's
    // `load` row measures Open3D's reader (Assimp's `fast_atof`) exactly one `f32` ULP from the
    // port's on the coordinates that need the 24th mantissa bit.
    let mut source_same: BTreeMap<String, bool> = BTreeMap::new();
    for (n, name) in names.iter().enumerate() {
        let mesh = sherd_core::io::load_mesh(&paths[n])?;
        let same = theirs.sources.get(name).is_some_and(|h| *h == source_hash(&mesh));
        source_same.insert(name.clone(), same);
    }
    let unreadable = source_same.values().filter(|s| !**s).count();

    let keep = std::env::var_os("SHERD_PARITY_KEEP_PLACED").map(PathBuf::from);
    let scratch = keep.clone().unwrap_or_else(scratch_dir);
    let _ = std::fs::remove_dir_all(&scratch);
    report::write_placed_meshes(
        &scratch,
        &paths,
        &names,
        &poses,
        &groups,
        sherd_core::io::writer::OPEN3D_COMMENT,
    )?;

    let mut differing_bytes = 0_usize;
    let mut comparable = 0_usize;
    let mut differing_shape = 0_usize;
    for (relative, theirs) in &theirs.files {
        let ours = scratch.join(relative);
        let bytes = std::fs::read(&ours);
        let Ok(bytes) = bytes else {
            tracing::warn!(file = %relative, "the port wrote no such mesh");
            differing_shape += 1;
            continue;
        };
        // The shape of the file — its size, its two counts and whether it carries colours — is
        // comparable on every collection, because a one-ULP coordinate does not change any of
        // them. This is what gates the header, the merge and the colour rule of R §11.4 on the
        // OBJ sets, where the byte row below cannot run.
        let shape = ply_shape(&bytes);
        let same_shape = shape == Some((theirs.vertices, theirs.faces, theirs.colors))
            && bytes.len() as u64 == theirs.size;
        if !same_shape {
            tracing::warn!(
                file = %relative,
                ours = ?shape,
                our_size = bytes.len(),
                theirs = ?(theirs.vertices, theirs.faces, theirs.colors),
                their_size = theirs.size,
                "placed mesh shape differs"
            );
            differing_shape += 1;
        }
        if !theirs.members.iter().all(|m| source_same.get(m).copied().unwrap_or(false)) {
            continue;
        }
        comparable += 1;
        let hash = crate::layout::hex_sha256(&bytes);
        if hash != theirs.sha256 {
            tracing::warn!(
                file = %relative,
                ours = %hash,
                theirs = %theirs.sha256,
                "placed mesh differs byte for byte although its input does not"
            );
            differing_bytes += 1;
        }
    }
    report.push(Check::entries(SCOPE, "placed shape", differing_shape, theirs.files.len()));
    if comparable > 0 {
        report.push(Check::entries(SCOPE, "placed ply", differing_bytes, comparable));
    }
    if unreadable > 0 {
        report.skip(
            SCOPE,
            format!(
                "{unreadable} of {} fragments read back differently from the reference's own \
                 reader, so {} of {} meshes are not byte-comparable: this is D §10.2's `load` row \
                 (Assimp's fast_atof on OBJ, one f32 ULP), not R §11.4",
                names.len(),
                theirs.files.len() - comparable,
                theirs.files.len()
            ),
        );
    }
    if keep.is_none() {
        let _ = std::fs::remove_dir_all(&scratch);
    }
    Ok(())
}

/// SHA-256 of a mesh as `tools/dump_outputs.py` hashes it: `V` as `f64` then `F` as `i32`, both
/// little-endian and C order, which is what `np.asarray(...).tobytes()` produces.
fn source_hash(mesh: &sherd_core::mesh::Mesh) -> String {
    let mut bytes = Vec::with_capacity(mesh.v.len() * 24 + mesh.f.len() * 12);
    for v in &mesh.v {
        for c in v {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    for f in &mesh.f {
        for &i in f {
            #[allow(clippy::cast_possible_wrap, reason = "numpy holds the triangles as int32")]
            bytes.extend_from_slice(&(i as i32).to_le_bytes());
        }
    }
    crate::layout::hex_sha256(&bytes)
}

/// `(vertices, faces, colours)` out of a binary PLY header, or `None` when it is not one.
fn ply_shape(bytes: &[u8]) -> Option<(u64, u64, bool)> {
    let end = b"end_header\n";
    let at = bytes.windows(end.len()).position(|w| w == end)? + end.len();
    let header = std::str::from_utf8(&bytes[..at]).ok()?;
    let mut vertices = 0;
    let mut faces = 0;
    let mut colors = false;
    for line in header.lines() {
        let mut words = line.split_whitespace();
        match (words.next(), words.next(), words.next()) {
            (Some("element"), Some("vertex"), Some(n)) => vertices = n.parse().ok()?,
            (Some("element"), Some("face"), Some(n)) => faces = n.parse().ok()?,
            (Some("property"), Some("uchar"), Some("red")) => colors = true,
            _ => {}
        }
    }
    Some((vertices, faces, colors))
}

// ---------------------------------------------------------------------------------------------
// previews

fn preview_rows(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let out = collection.dir.outputs_dir();
    let index_path = out.join("preview_index.json");
    if !index_path.is_file() {
        report.skip(
            SCOPE,
            "no outputs/preview_index.json in the dump: run tools/dump_outputs.py DUMP INPUT to \
             write the reference's own R §11.5 previews",
        );
        return Ok(());
    }
    let tags: Vec<String> = npy::read_json_as(&index_path)?;
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let transforms_path = out.join("transforms.json");
    let poses = if transforms_path.is_file() {
        assembly::read_transforms(&transforms_path, &names)?
    } else {
        vec![Matrix4::identity(); names.len()]
    };

    let mut geometry: Vec<Option<(sherd_core::mesh::Mesh, FaceGeometry, Vec<bool>)>> = Vec::new();
    for fragment in &collection.fragments {
        let Some(mesh) = fragment.working()? else {
            geometry.push(None);
            continue;
        };
        let geom = face_geometry(&mesh.v, &mesh.f);
        let frac = if fragment.has("seg.frac_final.npy") {
            npy::read_bool(fragment.file("seg.frac_final.npy"))?
        } else {
            vec![false; mesh.f.len()]
        };
        geometry.push(Some((mesh, geom, frac)));
    }

    let mut differing_px = 0_usize;
    let mut total_px = 0_usize;
    let mut label_px = 0_usize;
    for tag in &tags {
        let meta_path = out.join(format!("{tag}.meta.json"));
        if !meta_path.is_file() {
            report.skip(SCOPE, format!("{tag}.meta.json is missing from the dump"));
            continue;
        }
        let meta: RefPreview = npy::read_json_as(&meta_path)?;
        let Some(splats) = build_splats(collection, &meta, tag, &names, &geometry, &poses)? else {
            report.skip(SCOPE, format!("{tag} is missing its sampled indices"));
            continue;
        };
        let views: Vec<View> = meta.views.iter().map(|v| View { eye: v[0], up: v[1] }).collect();
        let ours = render_views(&splats, &views, meta.width, meta.height);
        let plain = read_png(&out.join(format!("{tag}.nolabel.png")))?;
        let labelled = read_png(&out.join(format!("{tag}.png")))?;
        let (Some(plain), Some(labelled)) = (plain, labelled) else {
            report.skip(SCOPE, format!("{tag}: the dump has no reference render"));
            continue;
        };
        let (differing, total) = compare_images(&ours, &plain, tag);
        differing_px += differing;
        total_px += total;
        label_px = label_px.max(compare_images(&labelled, &plain, tag).0);
    }
    if total_px > 0 {
        report.push(Check::entries(SCOPE, "preview px", differing_px, total_px));
        #[allow(clippy::cast_precision_loss, reason = "pixel counts are far below 2^53")]
        report.push(Check::absolute(SCOPE, "preview label px", label_px as f64, 0.0, LABEL_PIXELS));
    }
    Ok(())
}

/// One preview's meshes, rebuilt from the reference's own samples.
fn build_splats(
    collection: &Collection,
    meta: &RefPreview,
    tag: &str,
    names: &[String],
    geometry: &[Option<(sherd_core::mesh::Mesh, FaceGeometry, Vec<bool>)>],
    poses: &[Matrix4<f64>],
) -> Result<Option<Vec<Splat>>> {
    let out = collection.dir.outputs_dir();
    let mut splats = Vec::with_capacity(meta.members.len());
    for (i, member) in meta.members.iter().enumerate() {
        let Some(n) = names.iter().position(|x| x == member) else { return Ok(None) };
        let Some((mesh, geom, frac)) = &geometry[n] else { return Ok(None) };
        let pick_path = out.join(format!("{tag}.{member}.pick.npy"));
        if !pick_path.is_file() {
            return Ok(None);
        }
        let picks = npy::read_indices(&pick_path)?;
        let first = npy::read_f64(out.join(format!("{tag}.{member}.u.npy")))?;
        let second = npy::read_f64(out.join(format!("{tag}.{member}.v.npy")))?;
        if picks.len() != meta.n_points || first.len() != picks.len() {
            return Err(Error::fixture(&pick_path, "the sample arrays do not agree in length"));
        }
        let points = points_from_uniforms(&mesh.v, &mesh.f, &picks, &first, &second);
        if meta.kind == "group" {
            splats.push(placed_splat(
                &points,
                &picks,
                &geom.normals,
                &poses[n],
                Paint::Uniform(PALETTE[i % PALETTE.len()]),
            ));
        } else {
            // The segmentation preview: no pose, colours by R §3.4's label, and each fragment
            // shifted along x by `i · 1.3 · extent_x` after being centred on its own samples.
            let mean = [0, 1, 2].map(|axis| {
                let column: Vec<f64> = points.iter().map(|p| p[axis]).collect();
                #[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
                let n = column.len() as f64;
                pairwise_sum(&column) / n
            });
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for v in &mesh.v {
                lo = lo.min(v[0]);
                hi = hi.max(v[0]);
            }
            #[allow(clippy::cast_precision_loss, reason = "a fragment index")]
            let offset = i as f64 * 1.3 * (hi - lo);
            let moved: Vec<[f64; 3]> = points
                .iter()
                .map(|p| [p[0] - mean[0] + offset, p[1] - mean[1], p[2] - mean[2]])
                .collect();
            let colours: Vec<[f64; 3]> = picks
                .iter()
                .map(|&f| {
                    if frac.get(f as usize).copied().unwrap_or(false) {
                        [0.9, 0.2, 0.2]
                    } else {
                        [0.8, 0.8, 0.8]
                    }
                })
                .collect();
            splats.push(Splat {
                points: moved,
                normals: picks.iter().map(|&f| geom.normals[f as usize]).collect(),
                paint: Paint::PerPoint(colours),
            });
        }
    }
    Ok(Some(splats))
}

/// A PNG as RGB bytes, or `None` when the file is not there.
fn read_png(path: &Path) -> Result<Option<Rgb>> {
    if !path.is_file() {
        return Ok(None);
    }
    let decoded = image::open(path).map_err(|e| Error::read(path, e))?.to_rgb8();
    Ok(Some(Rgb {
        width: decoded.width() as usize,
        height: decoded.height() as usize,
        pixels: decoded.into_raw(),
    }))
}

/// How many pixels of two images differ, and how many there are.
fn compare_images(ours: &Rgb, theirs: &Rgb, tag: &str) -> (usize, usize) {
    if ours.width != theirs.width || ours.height != theirs.height {
        tracing::warn!(
            preview = tag,
            ours = format!("{}x{}", ours.width, ours.height),
            theirs = format!("{}x{}", theirs.width, theirs.height),
            "preview size differs"
        );
        return (
            ours.width * ours.height + theirs.width * theirs.height,
            theirs.width * theirs.height,
        );
    }
    let mut differing = 0;
    let mut first: Option<(usize, usize)> = None;
    for y in 0..ours.height {
        for x in 0..ours.width {
            if ours.pixel(x, y) != theirs.pixel(x, y) {
                differing += 1;
                first.get_or_insert((x, y));
            }
        }
    }
    if let Some((x, y)) = first {
        tracing::warn!(
            preview = tag,
            x,
            y,
            ours = ?ours.pixel(x, y),
            theirs = ?theirs.pixel(x, y),
            differing,
            "preview pixels differ"
        );
    }
    (differing, ours.width * ours.height)
}

// ---------------------------------------------------------------------------------------------
// native

fn native(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let out = collection.dir.outputs_dir();
    let index_path = out.join("preview_index.json");
    if !index_path.is_file() {
        report.skip(SCOPE, "no outputs/preview_index.json in the dump");
        return Ok(());
    }
    let tags: Vec<String> = npy::read_json_as(&index_path)?;
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let transforms_path = out.join("transforms.json");
    let poses = if transforms_path.is_file() {
        assembly::read_transforms(&transforms_path, &names)?
    } else {
        vec![Matrix4::identity(); names.len()]
    };
    let mut geometry: Vec<Option<(sherd_core::mesh::Mesh, FaceGeometry, Vec<bool>)>> = Vec::new();
    for fragment in &collection.fragments {
        let Some(mesh) = fragment.working()? else {
            geometry.push(None);
            continue;
        };
        let geom = face_geometry(&mesh.v, &mesh.f);
        let frac = if fragment.has("seg.frac_final.npy") {
            npy::read_bool(fragment.file("seg.frac_final.npy"))?
        } else {
            vec![false; mesh.f.len()]
        };
        geometry.push(Some((mesh, geom, frac)));
    }

    let mut worst_axis = 0.0_f64;
    let mut unstable = 0_usize;
    let mut rendered = 0_usize;
    for tag in &tags {
        let meta_path = out.join(format!("{tag}.meta.json"));
        if !meta_path.is_file() {
            continue;
        }
        let meta: RefPreview = npy::read_json_as(&meta_path)?;
        let Some(splats) = build_splats(collection, &meta, tag, &names, &geometry, &poses)? else {
            continue;
        };
        let all: Vec<[f64; 3]> = splats.iter().flat_map(|s| s.points.iter().copied()).collect();
        // PMC-10: the port's own axes, against the reference's own first view. The sign is
        // library-defined, so what is compared is the angle between the two *lines*.
        let axes = principal_axes(&all);
        if let Some(theirs) = meta.views.first() {
            worst_axis = worst_axis.max(line_angle(axes[0], theirs[0]));
        }
        let ours = render::render_views(
            &splats,
            &meta.views.iter().map(|v| View { eye: v[0], up: v[1] }).collect::<Vec<_>>(),
            meta.width,
            meta.height,
        );
        let again = render::render_views(
            &splats,
            &meta.views.iter().map(|v| View { eye: v[0], up: v[1] }).collect::<Vec<_>>(),
            meta.width,
            meta.height,
        );
        unstable += usize::from(ours != again);
        rendered += 1;
    }
    if rendered > 0 {
        report.push(Check::absolute(SCOPE, "axis", worst_axis, 0.0, AXIS_DEG));
        report.push(Check::entries(SCOPE, "render determinism", unstable, rendered));
    }

    // A `transforms.json` written and read back is the poses that went in.
    if transforms_path.is_file()
        && let Some(groups) = reference_groups(collection, report, &names)?
    {
        let poses = assembly::read_transforms(&transforms_path, &names)?;
        let file = std::env::temp_dir()
            .join(format!("sherd-parity-transforms-{}.json", std::process::id()));
        report::write_transforms(
            &file,
            &names,
            &poses,
            &groups,
            collection.manifest.pairs.thickness_median,
            &collection.manifest.collection.params,
        )?;
        let back: report::Transforms = npy::read_json_as(&file)?;
        let differing = names
            .iter()
            .enumerate()
            .filter(|(n, name)| {
                back.fragments.get(*name).is_none_or(|e| {
                    (0..4).any(|i| {
                        (0..4).any(|j| e.matrix[i][j].to_bits() != poses[*n][(i, j)].to_bits())
                    })
                })
            })
            .count();
        report.push(Check::entries(SCOPE, "transforms round trip", differing, names.len()));
        let _ = std::fs::remove_file(&file);
    }
    Ok(())
}

/// The angle in degrees between two directions, taken as lines: the sign is PMC-10's.
fn line_angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot: f64 = (0..3).map(|k| a[k] * b[k]).sum();
    let na: f64 = (0..3).map(|k| a[k] * a[k]).sum::<f64>().sqrt();
    let nb: f64 = (0..3).map(|k| b[k] * b[k]).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot.abs() / (na * nb)).clamp(0.0, 1.0).acos().to_degrees()
}

/// The scratch directory this stage writes its placed meshes into.
///
/// `SHERD_PARITY_KEEP_PLACED=DIR` overrides it and stops the directory being removed.
pub fn scratch_dir() -> PathBuf {
    std::env::temp_dir().join(format!("sherd-parity-placed-{}", std::process::id()))
}
