//! The working-mesh stage: decimation, Taubin, and everything derived from the result
//! (R §3.3, D §10.2 row `working mesh`).
//!
//! # Injected
//!
//! D §10.2 writes "(mesh is injected)" in the injected column, because the decimator is not the
//! reference's (PMC-2) and no fixture records the mesh between decimation and smoothing. What
//! *can* be injected is everything the stage derives, and all of it is exact:
//!
//! * `res`, the working mesh's area, `watertight` and `n_boundary` recomputed from the dump's own
//!   `mesh.V` / `mesh.F` — bit-identical, because `res` is a median over the same unique edges and
//!   the area a numpy pairwise sum over the same per-face areas (S3 note §2.3);
//! * `ΣA0` over `load.V0` / `load.F0`, the numerator of R §3.3's face budget;
//! * the face budget itself, from the reference's own `ΣA0` and `t`;
//! * Taubin smoothing, on the fragments the reference did **not** decimate — there the working
//!   mesh is exactly `Taubin(V0, F0)` and the comparison is meaningful. Open3D sums each vertex's
//!   neighbours in `std::unordered_set` order and this port in ascending index order, so what is
//!   left is round-off: measured at most 1.7e-12 of one edge length, gated here at 1e-9 of it.
//!
//! # Native
//!
//! `Fragment::from_mesh_file` runs from the file and D §10.2's native column applies: faces ±5 %,
//! `res` ±10 %, area ±0.5 %, the same `watertight` verdict. `meshopt` keeps a different mesh than
//! Open3D's quadric decimation by construction, which is what makes this column statistical.

use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::mesh::adjacency::closed_enough;
use sherd_core::mesh::decimate::face_budget;
use sherd_core::mesh::geometry::{face_geometry, median_edge};
use sherd_core::mesh::taubin::taubin;

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport, Unit};

/// D §10.2, native column: faces.
pub const NATIVE_FACES: f64 = 0.05;
/// D §10.2, native column: `res`.
pub const NATIVE_RES: f64 = 0.10;
/// D §10.2, native column: area.
pub const NATIVE_AREA: f64 = 0.005;
/// Injected Taubin, in units of `res`: what is left after the neighbour-summation order.
pub const INJECTED_TAUBIN_RES: f64 = 1e-9;

/// Runs R §3.3 for every fragment and compares the working mesh with the dump.
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("working mesh", mode);
    for fragment in &collection.fragments {
        let stats_file = fragment.file("mesh.stats.json");
        if !stats_file.is_file() {
            report.skip(&fragment.name, "no mesh.stats in the dump");
            continue;
        }
        let stats = npy::read_json(&stats_file)?;
        let ref_faces = npy::field_u64(&stats, "faces", &stats_file)?;
        let ref_vertices = npy::field_u64(&stats, "vertices", &stats_file)?;
        let ref_res = npy::field_f64(&stats, "res", &stats_file)?;
        let ref_area = npy::field_f64(&stats, "area", &stats_file)?;
        let ref_watertight = npy::field_bool(&stats, "watertight", &stats_file)?;
        let ref_boundary = npy::field_u64(&stats, "n_boundary", &stats_file)?;
        let name = fragment.name.as_str();

        match mode {
            Mode::Injected => {
                // --- the budget, from the reference's own area and thickness ------------------
                let target_file = fragment.file("thick.target.json");
                if target_file.is_file() {
                    let target = npy::read_json(&target_file)?;
                    let ref_area0 = npy::field_f64(&target, "area0", &target_file)?;
                    let ref_target = npy::field_u64(&target, "target", &target_file)?;
                    let ref_cap = npy::field_u64(&target, "target_faces", &target_file)?;
                    let ref_t = npy::read_scalar(fragment.file("thick.t.json"))?;
                    let budget =
                        face_budget(ref_area0, ref_t, usize::try_from(ref_cap).unwrap_or(0));
                    report.push(Check::count(name, "face budget", budget as u64, ref_target));
                    if let Some((v0, f0)) = fragment.original()? {
                        if fragment.has("load.V0.npy") {
                            let area0 = face_geometry(&v0, &f0).total_area();
                            report.push(Check::exact(name, "area0", area0, ref_area0));
                        }
                        // Taubin, where the reference's working mesh is exactly Taubin(V0, F0).
                        let faces0 = npy::field_u64(&target, "faces0", &target_file)?;
                        if faces0 <= ref_target
                            && fragment.has("load.V0.npy")
                            && let Some(mesh) = fragment.working()?
                        {
                            {
                                let mut smoothed = sherd_core::mesh::Mesh::new(v0, f0);
                                taubin(&mut smoothed);
                                let worst = if smoothed.v.len() == mesh.v.len() {
                                    smoothed
                                        .v
                                        .iter()
                                        .zip(&mesh.v)
                                        .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                                        .fold(0.0_f64, f64::max)
                                } else {
                                    f64::INFINITY
                                };
                                let mut check = Check::absolute(
                                    name,
                                    "taubin",
                                    worst,
                                    0.0,
                                    INJECTED_TAUBIN_RES * ref_res,
                                );
                                check.unit = Unit::Absolute;
                                report.push(check);
                            }
                        }
                    }
                }

                // --- everything derived from the reference's own working mesh -----------------
                let Some(mesh) = fragment.working()? else {
                    report.skip(name, "no mesh.V in the dump");
                    continue;
                };
                let geom = face_geometry(&mesh.v, &mesh.f);
                let res = median_edge(&mesh.v, &mesh.f);
                let (watertight, n_boundary) = closed_enough(&mesh.f);
                report.push(Check::count(name, "faces", mesh.f.len() as u64, ref_faces));
                report.push(Check::count(name, "vertices", mesh.v.len() as u64, ref_vertices));
                report.push(Check::exact(name, "res", res, ref_res));
                report.push(Check::exact(name, "area", geom.total_area(), ref_area));
                report.push(Check::flag(name, "watertight", watertight, ref_watertight));
                report.push(Check::count(name, "n_boundary", n_boundary as u64, ref_boundary));
            }
            Mode::Native => {
                let Some(source) = &fragment.source else {
                    report.skip(name, "no source file (pass --input DIR)");
                    continue;
                };
                let (fr, _) = Fragment::load_or_build(source, collection.target_faces, name, None)?;
                let area: f64 = fr.mesh.face_areas.iter().map(|&a| f64::from(a)).sum();
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "face counts are exact in f64 well past 2^53"
                )]
                let faces = (fr.n_faces() as f64, ref_faces as f64);
                report.push(Check::relative(name, "faces", faces.0, faces.1, NATIVE_FACES));
                report.push(Check::relative(name, "res", fr.res(), ref_res, NATIVE_RES));
                report.push(Check::relative(name, "area", area, ref_area, NATIVE_AREA));
                report.push(Check::flag(name, "watertight", fr.watertight, ref_watertight));
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::layout::FixtureDir;
    use crate::report::Check;
    use crate::report::Mode;
    use crate::stages::Collection;
    use crate::stages::tests::{slab_dump, slab_input};

    #[test]
    fn injected_derivations_are_bit_identical_on_the_slab() {
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        let r = run(&c, Mode::Injected).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        // Per fragment: budget, area0, taubin, faces, vertices, res, area, watertight, n_boundary.
        assert_eq!(r.checks.len(), 18);
        // Only Taubin has any headroom to speak of; everything else is exact and exactly equal.
        let worst = r.worst().unwrap();
        assert_eq!(worst.quantity, "taubin", "{}", worst.line());
        assert!(worst.ratio() < 0.01, "{}", worst.line());
    }

    #[test]
    fn the_native_stage_meets_the_design_tolerances_on_the_slab() {
        let c = Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).unwrap();
        let r = run(&c, Mode::Native).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert_eq!(r.checks.len(), 8);
    }
}
