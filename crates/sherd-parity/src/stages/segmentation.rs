//! The segmentation stage: shell against fracture (R §3.4, D §10.2 row `segmentation`).
//!
//! # Injected
//!
//! R §3.4 runs on the dump's own working mesh (`mesh.V`, `mesh.F`), its own `t` and its own `res`,
//! so nothing upstream can move the answer, and the labels are compared with `seg.frac_final` by
//! **area-weighted agreement**: the fraction of the mesh's area whose two labels are the same.
//! D §10.2 asks for ≥ 0.995 and ±0.005 on the fracture fraction.
//!
//! Two things stop this from being bit-exact, and both are in R:
//!
//! * **PMC-4.** The reference's voxel representatives come out of an Open3D hash map; this port
//!   sorts them. The map from a face to *its* representative is what matters and is compared
//!   directly (`rep face`), but a face equidistant from two representatives may resolve either
//!   way, and then its smoothed normal is another face's.
//! * **Summation order.** `query_ball_point(..., return_sorted=False)` hands the reference its
//!   neighbourhoods in an unspecified order, so `NS`, the majority sums and the shell reference
//!   normal are accumulated differently here. The difference is round-off, and the intermediate
//!   checks below are what measures it rather than assuming it.
//!
//! The intermediate checks are the diagnosis, not the gate: when the agreement falls short they
//! say *where* — at the grid, at the smoothed normals, at the vote, or at one of the four cleanup
//! passes — which is the difference between a fix and a guess.
//!
//! # Native
//!
//! `Fragment::from_mesh_file` runs from the file, so the port's working mesh is its own: a
//! different decimation, a different vertex count, different faces. The labels are transferred
//! along the surface — every face of the *reference's* mesh is looked up on the port's mesh by
//! closest point, and the two labels compared, weighted by the reference face's area. That is
//! D §10.2's "sample points on the Python working mesh and label each by its nearest face on each
//! mesh" with the reference's own faces as the quadrature points: one point per face, weighted by
//! its area, and no RNG in the comparison.

use rayon::prelude::*;

use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::fragment::segment::{SegParams, label_agreement, segment_faces_traced};
use sherd_core::mesh::geometry::{face_geometry, pairwise_sum};
use sherd_core::spatial::bvh::RayScene;
use sherd_core::types::FaceLabel;

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport, Unit};

/// D §10.2, injected column: `1 − agreement`.
pub const INJECTED_DISAGREEMENT: f64 = 0.005;
/// D §10.2, injected column: the fracture fraction.
pub const INJECTED_FRACTION: f64 = 0.005;
/// D §10.2, native column: `1 − agreement` after the nearest-face transfer.
pub const NATIVE_DISAGREEMENT: f64 = 0.03;
/// D §10.2, native column: the fracture fraction.
pub const NATIVE_FRACTION: f64 = 0.02;

/// Diagnostic gate on the face → representative map (PMC-4 ties).
pub const INJECTED_REP: f64 = 0.005;
/// Diagnostic gate on the smoothed normal, in degrees.
pub const INJECTED_NS_DEG: f64 = 1.0;
/// Diagnostic gate on the cone vote: the fraction of faces whose count differs.
pub const INJECTED_VOTES: f64 = 0.005;
/// Diagnostic gate on each intermediate mask, as `1 − agreement`.
pub const INJECTED_STAGE: f64 = 0.005;

/// Runs R §3.4 for every fragment and compares the labels with the dump.
#[allow(clippy::too_many_lines, reason = "one arm per mode, each a flat list of comparisons")]
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("segmentation", mode);
    for fragment in &collection.fragments {
        let name = fragment.name.as_str();
        if !fragment.has("seg.frac_final.npy") {
            report.skip(name, "no seg.frac_final in the dump (level min)");
            continue;
        }
        let Some(mesh) = fragment.working()? else {
            report.skip(name, "no mesh.V in the dump: the reference's own labels have no mesh");
            continue;
        };
        let ref_frac = npy::read_bool(fragment.file("seg.frac_final.npy"))?;
        if ref_frac.len() != mesh.f.len() {
            report.skip(name, "seg.frac_final does not describe mesh.F");
            continue;
        }
        let ref_labels: Vec<FaceLabel> = ref_frac.iter().copied().map(label).collect();
        let ref_geom = face_geometry(&mesh.v, &mesh.f);
        let ref_fraction = fraction(&ref_geom.areas, &ref_labels);

        match mode {
            Mode::Injected => {
                let thick = npy::read_scalar(fragment.file("thick.t.json"))?;
                let res = npy::read_scalar(fragment.file("mesh.res.json"))?;
                let Some(scene) = RayScene::new(&mesh.v, &mesh.f) else {
                    report.skip(name, "the dump's working mesh has no triangle");
                    continue;
                };
                let (seg, trace) = segment_faces_traced(
                    &scene,
                    &mesh.f,
                    &ref_geom,
                    thick,
                    res,
                    &SegParams::default(),
                );

                // --- the gate D §10.2 states ---------------------------------------------------
                let agreement = label_agreement(&seg.labels, &ref_labels, &ref_geom.areas);
                report.push(agreement_check(name, "agreement", agreement, INJECTED_DISAGREEMENT));
                report.push(Check::absolute(
                    name,
                    "fracture",
                    seg.fracture_fraction,
                    ref_fraction,
                    INJECTED_FRACTION,
                ));

                // --- the diagnosis, step by step -----------------------------------------------
                let info_file = fragment.file("seg.info.json");
                if info_file.is_file() {
                    let info = npy::read_json(&info_file)?;
                    report.push(Check::count(
                        name,
                        "votes",
                        u64::from(seg.votes),
                        npy::field_u64(&info, "votes", &info_file)?,
                    ));
                    report.push(Check::exact(
                        name,
                        "smooth radius",
                        seg.smooth_radius,
                        npy::field_f64(&info, "smooth_radius", &info_file)?,
                    ));
                    report.push(Check::exact(
                        name,
                        "growth angle",
                        seg.boundary_angle,
                        npy::field_f64(&info, "boundary_angle", &info_file)?,
                    ));
                    report.push(Check::absolute(
                        name,
                        "raw fraction",
                        seg.raw_fraction,
                        npy::field_f64(&info, "raw_fraction", &info_file)?,
                        INJECTED_FRACTION,
                    ));
                }
                if fragment.has("seg.rep.npy") && fragment.has("seg.near.npy") {
                    let rep = npy::read_indices(fragment.file("seg.rep.npy"))?;
                    let near = npy::read_indices(fragment.file("seg.near.npy"))?;
                    if near.len() == mesh.f.len() {
                        // PMC-4: `rep` itself is in hash order in the dump and sorted here, so
                        // what is compared is the map a face is looked up through.
                        let differing = (0..mesh.f.len())
                            .filter(|&i| rep[near[i] as usize] != trace.rep[trace.near[i] as usize])
                            .count();
                        report.push(fraction_check(
                            name,
                            "rep face",
                            differing,
                            mesh.f.len(),
                            INJECTED_REP,
                        ));
                        report.push(Check::count(
                            name,
                            "grid points",
                            trace.rep.len() as u64,
                            rep.len() as u64,
                        ));
                    }
                }
                if fragment.has("seg.NS.npy") {
                    let ns = npy::read_points(fragment.file("seg.NS.npy"))?;
                    if ns.len() == mesh.f.len() {
                        let worst = worst_angle(&trace.ns, &ns);
                        let mut check =
                            Check::absolute(name, "NS angle", worst, 0.0, INJECTED_NS_DEG);
                        check.unit = Unit::Absolute;
                        report.push(check);
                    }
                }
                if fragment.has("seg.good.npy") {
                    let good = npy::read_indices(fragment.file("seg.good.npy"))?;
                    if good.len() == mesh.f.len() {
                        let differing = (0..mesh.f.len())
                            .filter(|&i| u32::from(trace.good[i]) != good[i])
                            .count();
                        report.push(fraction_check(
                            name,
                            "votes/face",
                            differing,
                            mesh.f.len(),
                            INJECTED_VOTES,
                        ));
                    }
                }
                for (file, quantity, ours) in [
                    ("seg.frac_raw.npy", "raw mask", &trace.frac_raw),
                    ("seg.frac_majority.npy", "majority", &trace.frac_majority),
                    ("seg.frac_islands.npy", "islands", &trace.frac_islands),
                ] {
                    if !fragment.has(file) {
                        continue;
                    }
                    let theirs = npy::read_bool(fragment.file(file))?;
                    if theirs.len() != mesh.f.len() {
                        continue;
                    }
                    let mine: Vec<FaceLabel> = ours.iter().copied().map(label).collect();
                    let yours: Vec<FaceLabel> = theirs.iter().copied().map(label).collect();
                    let a = label_agreement(&mine, &yours, &ref_geom.areas);
                    report.push(agreement_check(name, quantity, a, INJECTED_STAGE));
                }
            }
            Mode::Native => {
                let Some(source) = &fragment.source else {
                    report.skip(name, "no source file (pass --input DIR)");
                    continue;
                };
                let (fr, _) = Fragment::load_or_build(source, collection.target_faces, name, None)?;
                let v64: Vec<[f64; 3]> = fr.mesh.v.iter().map(|p| p.to_f64()).collect();
                let Some(scene) = RayScene::new(&v64, &fr.mesh.f) else {
                    report.skip(name, "the port's working mesh has no triangle");
                    continue;
                };
                // Every face of the reference's mesh, looked up on the port's by closest point.
                let transferred: Vec<FaceLabel> = ref_geom
                    .centroids
                    .par_iter()
                    .map(|c| {
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "the BVH is f32, as Open3D's scene is"
                        )]
                        let q = [c[0] as f32, c[1] as f32, c[2] as f32];
                        scene
                            .closest_face(q)
                            .map_or(FaceLabel::Shell, |(f, _)| fr.labels[f as usize])
                    })
                    .collect();
                let agreement = label_agreement(&transferred, &ref_labels, &ref_geom.areas);
                report.push(agreement_check(name, "agreement", agreement, NATIVE_DISAGREEMENT));
                report.push(Check::absolute(
                    name,
                    "fracture",
                    fr.fracture_fraction(),
                    ref_fraction,
                    NATIVE_FRACTION,
                ));
            }
        }
    }
    Ok(report)
}

/// `Fracture` for true, `Shell` for false — the reference's `frac` mask as labels.
fn label(is_fracture: bool) -> FaceLabel {
    if is_fracture { FaceLabel::Fracture } else { FaceLabel::Shell }
}

/// Fracture area over total area, summed the way numpy sums.
fn fraction(areas: &[f64], labels: &[FaceLabel]) -> f64 {
    let total = pairwise_sum(areas);
    if total <= 0.0 {
        return 0.0;
    }
    let selected: Vec<f64> =
        areas.iter().zip(labels).filter_map(|(&a, l)| l.is_fracture().then_some(a)).collect();
    pairwise_sum(&selected) / total
}

/// An agreement against 1: the deviation is what disagrees, the tolerance what may.
fn agreement_check(name: &str, quantity: &'static str, agreement: f64, tolerance: f64) -> Check {
    Check::absolute(name, quantity, agreement, 1.0, tolerance)
}

/// A count of differing faces, reported as the fraction of the mesh they are.
fn fraction_check(
    name: &str,
    quantity: &'static str,
    differing: usize,
    total: usize,
    tolerance: f64,
) -> Check {
    #[allow(clippy::cast_precision_loss, reason = "face counts are exact well past 2^53")]
    let f = if total == 0 { 0.0 } else { differing as f64 / total as f64 };
    let mut check = Check::absolute(name, quantity, f, 0.0, tolerance);
    check.unit = Unit::Relative;
    check
}

/// The largest angle, in degrees, between two arrays of unit vectors.
fn worst_angle(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x[0] * y[0] + x[1] * y[1]) + x[2] * y[2]).clamp(-1.0, 1.0).acos())
        .fold(0.0_f64, f64::max)
        .to_degrees()
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::layout::FixtureDir;
    use crate::report::{Check, Mode};
    use crate::stages::Collection;
    use crate::stages::tests::{slab_dump, slab_input};

    #[test]
    fn the_injected_labels_agree_with_the_reference_on_the_slab() {
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        let r = run(&c, Mode::Injected).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert!(r.checks.len() >= 20, "{} checks", r.checks.len());
        let agreement =
            r.checks.iter().find(|c| c.quantity == "agreement").expect("the gate D §10.2 states");
        assert!(agreement.measured >= 0.995, "{}", agreement.line());
    }

    #[test]
    fn the_native_labels_survive_the_transfer_onto_the_ports_own_mesh() {
        let c = Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).unwrap();
        let r = run(&c, Mode::Native).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert_eq!(r.checks.len(), 4, "agreement and fracture fraction for two fragments");
    }
}
