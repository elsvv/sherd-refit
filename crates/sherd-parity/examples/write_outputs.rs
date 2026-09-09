//! R §8, R §9 and the whole of R §11 written by the port, into a directory `tools/evaluate.py`
//! can score.
//!
//! ```text
//! cargo run --release -p sherd-parity --example write_outputs -- DUMP INPUT_DIR OUT_DIR
//! python tools/evaluate.py OUT_DIR INPUT_DIR
//! ```
//!
//! The `outputs` row of D §10.2 compares the port's `transforms.json` and `report.json` against
//! the reference's field by field, which is the right gate and is not the whole claim: R §2's rule
//! is that the port's outputs are *readable by the reference's evaluator*, and the only way to
//! know that is to hand the evaluator a directory the port wrote. This example writes one.
//!
//! Everything upstream of R §8 is the reference's — the candidates come out of the dump's own
//! `result.candidates.json` and the samples out of its `assembly/md/<name>/md.S` — so what runs
//! here is R §8's greedy growth, R §9's full-resolution ladder over the original meshes, R §8.2's
//! recentring and R §11.1–11.5's five writers. On the eight fixture sets that reproduces the
//! reference's own joins exactly (the `assembly` row), so `evaluate.py` has to print the numbers
//! of `notes/2026-09-06-scale-pairs.md` and `notes/2026-09-07-t1-deterministic-thickness.md`.

use nalgebra::Matrix4;
use sherd_core::assembly::{assemble, recenter};
use sherd_core::error::Result;
use sherd_core::executor::CPU;
use sherd_core::mesh::geometry::face_geometry;
use sherd_core::refine::{FractureCloud, MAX_POINTS, RefinePiece, fracture_cloud, refine_joins};
use sherd_core::render::{PALETTE, Paint, Splat, group_label, principal_views, render_views};
use sherd_core::report::{
    FragmentStats, Outcome, Timings, write_placed_meshes, write_report, write_transforms,
};
use sherd_core::types::FragId;
use sherd_parity::FixtureDir;
use sherd_parity::report::{Mode, StageReport};
use sherd_parity::stages::Collection;
use sherd_parity::stages::assembly::{piece_views, reference_candidates, reference_pieces};

/// R §11.5's sample count per fragment.
const PREVIEW_POINTS: usize = 250_000;

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "a diagnostic that runs R §8, R §9 and R §11 in one pass; the casts are counts"
)]
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dump = args.next().expect("usage: write_outputs DUMP INPUT_DIR OUT_DIR");
    let input = args.next().expect("usage: write_outputs DUMP INPUT_DIR OUT_DIR");
    let out =
        std::path::PathBuf::from(args.next().expect("usage: write_outputs DUMP INPUT_DIR OUT_DIR"));

    let collection = Collection::open(FixtureDir::new(&dump), Some(std::path::Path::new(&input)))?;
    let params = collection.manifest.collection.params;
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let thickness = collection.manifest.pairs.thickness_median;
    let mut report = StageReport::new("write_outputs", Mode::Injected);

    // R §8 on the reference's own candidates and samples.
    let pieces =
        reference_pieces(&collection, &mut report)?.expect("the dump carries the assembly");
    let views = piece_views(&pieces);
    let candidates =
        reference_candidates(&collection, &mut report)?.expect("the dump's candidates");
    let assembly = assemble(&CPU, &views, &candidates, &params);
    let used: Vec<(FragId, FragId)> =
        assembly.used.iter().map(|&i| (candidates[i].a, candidates[i].b)).collect();

    // R §9 on the original meshes.
    let mut clouds: Vec<Option<FractureCloud>> = Vec::new();
    let mut stats: Vec<FragmentStats> = Vec::new();
    let mut paths = Vec::new();
    for (n, fragment) in collection.fragments.iter().enumerate() {
        let source = fragment.source.clone().expect("every fragment has a source file");
        paths.push(source.clone());
        let in_group = assembly.groups.iter().any(|g| {
            g.len() > 1 && g.contains(&u32::try_from(n).expect("fewer than 2^32 fragments"))
        });
        let mesh = fragment.working()?.expect("the dump carries a working mesh");
        let geom = face_geometry(&mesh.v, &mesh.f);
        let frac = sherd_parity::npy::read_bool(fragment.file("seg.frac_final.npy"))?;
        clouds.push(in_group.then(|| {
            let original = sherd_core::io::load_mesh(&source).expect("the source mesh reads");
            fracture_cloud(
                &original,
                &geom.centroids,
                &frac,
                pieces[n].thick,
                pieces[n].res,
                params.seed,
            )
        }));
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &mesh.v {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let n_orig = fragment.file("load.n_orig.json");
        let (orig_vertices, orig_faces) = if n_orig.is_file() {
            let value = sherd_parity::npy::read_json(&n_orig)?;
            (
                sherd_parity::npy::field_u64(&value, "n_orig_vertices", &n_orig)?,
                sherd_parity::npy::field_u64(&value, "n_orig_faces", &n_orig)?,
            )
        } else {
            (0, 0)
        };
        let stats_file = fragment.file("mesh.stats.json");
        let thick_mode = sherd_parity::npy::field_f64(
            &sherd_parity::npy::read_json(&stats_file)?,
            "thick_mode",
            &stats_file,
        )?;
        let area: f64 = geom.total_area();
        let frac_area = sherd_core::fragment::segment::masked_area(&geom.areas, &frac);
        stats.push(FragmentStats {
            name: fragment.name.clone(),
            faces: mesh.f.len() as u64,
            orig_faces,
            orig_vertices,
            thickness: pieces[n].thick,
            thickness_mode: thick_mode,
            resolution: pieces[n].res,
            watertight: pieces[n].watertight,
            extent: [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
            area,
            fracture_area_fraction: if area > 0.0 { frac_area / area } else { 0.0 },
        });
        eprintln!("preprocessed {} ({} of {})", fragment.name, n + 1, names.len());
    }
    let refine_pieces: Vec<RefinePiece<'_>> = pieces
        .iter()
        .zip(&clouds)
        .map(|(p, c)| RefinePiece { thick: p.thick, res: p.res, cloud: c.as_ref() })
        .collect();
    let refined = refine_joins(
        &refine_pieces,
        &assembly.poses,
        &assembly.groups,
        &used,
        &params,
        sherd_core::executor::Engine::REFERENCE,
    );
    let poses = recenter(&refined.poses, &views, &assembly.groups);

    // R §11.
    std::fs::create_dir_all(&out).expect("the output directory");
    write_transforms(
        out.join("transforms.json"),
        &names,
        &poses,
        &assembly.groups,
        &assembly.order,
        thickness,
        &params,
        Some("cpu"),
        None,
    )?;
    let rejected: Vec<(usize, String)> =
        assembly.rejected.iter().map(|r| (r.candidate, r.reason.message(&names))).collect();
    let outcome = Outcome {
        names: &names,
        candidates: &candidates,
        used: &assembly.used,
        rejected: &rejected,
        groups: &assembly.groups,
        tiers: None,
        constraints: None,
        review: None,
        objects: None,
    };
    let timings = Timings::from_iter([("output".to_owned(), 0.0)]);
    // No memory block: the harness rebuilds the reference's own file, and audit §B.3's
    // sampled peaks are the port's addition (D §4.3).
    write_report(&out, &stats, thickness, &outcome, &timings, &params, "cpu", None)?;
    write_placed_meshes(
        &out,
        &paths,
        &names,
        &poses,
        &assembly.groups,
        sherd_core::io::writer::DEFAULT_COMMENT,
        sherd_core::memory::Budget::default_for_machine(),
    )?;
    write_previews(&collection, &out, &names, &poses, &assembly.groups)?;
    println!("{} written", out.display());
    println!("MAX_POINTS = {MAX_POINTS}, previews at {PREVIEW_POINTS} samples per fragment");
    Ok(())
}

/// R §11.5 for every group of two or more, plus the segmentation preview.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "R §11.5's two passes, and the casts are counts"
)]
fn write_previews(
    collection: &Collection,
    out: &std::path::Path,
    names: &[String],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
) -> Result<()> {
    let mut rng = sherd_core::rng::seeded(0);
    let mut meshes = Vec::new();
    for fragment in &collection.fragments {
        let mesh = fragment.working()?.expect("the dump carries a working mesh");
        let geom = face_geometry(&mesh.v, &mesh.f);
        let frac = sherd_parity::npy::read_bool(fragment.file("seg.frac_final.npy"))?;
        meshes.push((mesh, geom, frac));
    }
    let all_faces = |n: usize| -> Vec<u32> {
        (0..u32::try_from(meshes[n].0.f.len()).expect("fewer than 2^32 faces")).collect()
    };

    for (k, group) in groups.iter().enumerate() {
        if group.len() < 2 {
            continue;
        }
        let mut splats = Vec::new();
        for (i, &n) in group.iter().enumerate() {
            let n = n as usize;
            let (mesh, geom, _) = &meshes[n];
            let (points, picks) = sherd_core::fragment::samples::sample_on_faces(
                &mesh.v,
                &mesh.f,
                &geom.areas,
                &all_faces(n),
                PREVIEW_POINTS,
                &mut rng,
            );
            splats.push(sherd_core::render::placed_splat(
                &points,
                &picks,
                &geom.normals,
                &poses[n],
                Paint::Uniform(PALETTE[i % PALETTE.len()]),
            ));
        }
        let all: Vec<[f64; 3]> = splats.iter().flat_map(|s| s.points.iter().copied()).collect();
        let member_names: Vec<String> = group.iter().map(|&n| names[n as usize].clone()).collect();
        let mut image = render_views(&splats, &principal_views(&all), 900, 700);
        sherd_core::render::draw_label(&mut image, 10, 10, &group_label(&member_names));
        image.write_png(out.join(format!("preview_{k}.png")))?;
    }

    let mut splats = Vec::new();
    for (n, (mesh, geom, frac)) in meshes.iter().enumerate() {
        let (points, picks) = sherd_core::fragment::samples::sample_on_faces(
            &mesh.v,
            &mesh.f,
            &geom.areas,
            &all_faces(n),
            PREVIEW_POINTS / 2,
            &mut rng,
        );
        let mean = [0, 1, 2].map(|axis| {
            let column: Vec<f64> = points.iter().map(|p| p[axis]).collect();
            sherd_core::mesh::geometry::pairwise_sum(&column) / column.len() as f64
        });
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in &mesh.v {
            lo = lo.min(v[0]);
            hi = hi.max(v[0]);
        }
        let offset = n as f64 * 1.3 * (hi - lo);
        splats.push(Splat {
            points: points
                .iter()
                .map(|p| [p[0] - mean[0] + offset, p[1] - mean[1], p[2] - mean[2]])
                .collect(),
            normals: picks.iter().map(|&f| geom.normals[f as usize]).collect(),
            paint: Paint::PerPoint(
                picks
                    .iter()
                    .map(|&f| if frac[f as usize] { [0.9, 0.2, 0.2] } else { [0.8, 0.8, 0.8] })
                    .collect(),
            ),
        });
    }
    let all: Vec<[f64; 3]> = splats.iter().flat_map(|s| s.points.iter().copied()).collect();
    let views = principal_views(&all);
    let mut image = render_views(&splats, &views[..2], 1400, 600);
    sherd_core::render::draw_label(&mut image, 10, 10, &names.join(" "));
    image.write_png(out.join("preview_segmentation.png"))?;
    Ok(())
}
