//! The thickness stage: `t` and `thick_mode` (R §3.2, D §10.2 row `thickness`).
//!
//! # Injected
//!
//! The dump carries the rays themselves — the faces the reference sampled (`thick.idx`), the
//! distances Open3D's raycaster returned (`thick.t_hit`) and the primitives it hit
//! (`thick.prim`). Feeding those into [`thickness_from_hits`] runs the reference's own filter,
//! percentile and histogram over the reference's own numbers, so the comparison is exact: D §10.2
//! allows "the same bin, or ±1 bin on a count tie" and the port lands on the same `float32`, bit
//! for bit, on every fragment measured so far. Both statements are checked — the bin distance
//! against D §10.2's tolerance, and the bits as an exact check — because a port that started
//! rounding differently would still pass the bin test and should not.
//!
//! # Native
//!
//! The port draws its own 20 000 faces from `ChaCha8Rng` (PMC-9: numpy's PCG64 is not reproduced)
//! and casts its own rays through `parry3d`, so `t` is a *different sample of the same estimator*.
//! D §10.2's native tolerance is ±2 %, and plan step S3 measured that no implementation which does
//! not reproduce numpy's stream can hold it: the reference's own estimate moves by up to 14.5 %
//! when only the seed changes, because the filtered distances of a fragment with a plateau rather
//! than a peak put several near-equal bins in contention. The gate applied here is therefore
//! `max(2 %, 3 bins of the reference's own histogram)`, which is S3's recommendation to D §10.2,
//! and the per-cent deviation is reported beside it so the raw D §10.2 number stays visible.
//!
//! One bin is `percentile(far, 90) / 60` over the distances of R §3.2's filtered set, computed
//! from the dump's own rays (`filtered_distances`) — the reference's resolution, not the port's.

use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::fragment::thickness::{BINS, RayHits, filtered_distances, percentile90};
use sherd_core::mesh::geometry::face_geometry;

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport, Unit};

/// D §10.2's native tolerance on `t`, as a fraction.
pub const NATIVE_RELATIVE: f64 = 0.02;

/// The widening S3 recommends for D §10.2's native column: three bins of the reference's own
/// thickness histogram, whichever is larger. One bin is 1.7–5.7 % of `t` on the benchmark sets.
pub const NATIVE_BINS: f64 = 3.0;

/// D §10.2's injected tolerance: the same bin, or one bin away when the counts tie.
pub const INJECTED_BINS: f64 = 1.0;

/// Runs R §3.2 for every fragment and compares `t` and `thick_mode` with the dump.
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("thickness", mode);
    for fragment in &collection.fragments {
        if !fragment.has("thick.t.json") {
            report.skip(&fragment.name, "no thick.t in the dump");
            continue;
        }
        let ref_t = npy::read_scalar(fragment.file("thick.t.json"))?;
        let ref_mode = npy::read_scalar(fragment.file("thick.thick_mode.json"))?;

        // The reference's own rays, and the mesh they were cast on, are what the bin width and
        // the injected comparison both need.
        let Some((v0, f0)) = fragment.original()? else {
            report.skip(&fragment.name, "neither load.V0 nor a source file");
            continue;
        };
        if !fragment.has("thick.t_hit.npy") {
            report.skip(&fragment.name, "no thick.t_hit in the dump");
            continue;
        }
        let geom0 = face_geometry(&v0, &f0);
        let idx = npy::read_indices(fragment.file("thick.idx.npy"))?;
        let hits = RayHits {
            t_hit: npy::read_f32(fragment.file("thick.t_hit.npy"))?,
            prim: npy::read_indices(fragment.file("thick.prim.npy"))?,
        };
        let bin = bin_width(&geom0.normals, &idx, &hits);

        match mode {
            Mode::Injected => {
                let Some((t, m)) = sherd_core::fragment::thickness::thickness_from_hits(
                    &geom0.normals,
                    &idx,
                    &hits,
                ) else {
                    report.skip(&fragment.name, "fewer than 100 of the dump's rays hit");
                    continue;
                };
                push_bins(
                    &mut report,
                    &fragment.name,
                    "t",
                    f64::from(t),
                    ref_t,
                    bin,
                    INJECTED_BINS,
                );
                push_bins(
                    &mut report,
                    &fragment.name,
                    "thick_mode",
                    f64::from(m),
                    ref_mode,
                    bin,
                    INJECTED_BINS,
                );
                report.push(Check::exact(&fragment.name, "t (bits)", f64::from(t), ref_t));
                report.push(Check::exact(&fragment.name, "mode (bits)", f64::from(m), ref_mode));
            }
            Mode::Native => {
                let Some(source) = &fragment.source else {
                    report.skip(&fragment.name, "no source file (pass --input DIR)");
                    continue;
                };
                let (fr, _) =
                    Fragment::load_or_build(source, collection.target_faces, &fragment.name, None)?;
                push_native(&mut report, &fragment.name, "t", fr.thick, ref_t, bin);
                push_native(
                    &mut report,
                    &fragment.name,
                    "thick_mode",
                    fr.thick_mode,
                    ref_mode,
                    bin,
                );
            }
        }
    }
    Ok(report)
}

/// One bin of the reference's own thickness histogram, or `None` when its filtered set was too
/// small for the filtered mode to be taken at all (in which case R §3.2 falls back to the plain
/// mode, whose bin width is not what this reports).
fn bin_width(normals: &[[f64; 3]], idx: &[u32], hits: &RayHits) -> Option<f64> {
    let far = filtered_distances(normals, idx, hits);
    if far.len() < sherd_core::fragment::thickness::MIN_HITS {
        return None;
    }
    #[allow(clippy::cast_precision_loss, reason = "BINS is 60")]
    Some(f64::from(percentile90(&far)) / BINS as f64)
}

/// A comparison whose unit is a bin of the reference's histogram, with a relative fallback when
/// the bin width is unknown.
fn push_bins(
    report: &mut StageReport,
    scope: &str,
    quantity: &'static str,
    measured: f64,
    reference: f64,
    bin: Option<f64>,
    tolerance_bins: f64,
) {
    match bin {
        Some(bin) if bin > 0.0 => {
            let mut check = Check::absolute(scope, quantity, measured, reference, tolerance_bins);
            check.deviation = (measured - reference).abs() / bin;
            check.unit = Unit::Bins;
            report.push(check);
        }
        _ => report.push(Check::relative(scope, quantity, measured, reference, NATIVE_RELATIVE)),
    }
}

/// The native gate: `max(2 %, 3 bins)`, expressed as a relative check so the table shows the
/// per-cent deviation D §10.2 talks about.
fn push_native(
    report: &mut StageReport,
    scope: &str,
    quantity: &'static str,
    measured: f64,
    reference: f64,
    bin: Option<f64>,
) {
    let widened = match bin {
        Some(bin) if reference != 0.0 => (NATIVE_BINS * bin / reference.abs()).max(NATIVE_RELATIVE),
        _ => NATIVE_RELATIVE,
    };
    report.push(Check::relative(scope, quantity, measured, reference, widened));
}

#[cfg(test)]
mod tests {
    use super::{NATIVE_RELATIVE, run};
    use crate::layout::FixtureDir;
    use crate::report::Check;
    use crate::report::Mode;
    use crate::stages::Collection;
    use crate::stages::tests::{slab_dump, slab_input};

    #[test]
    fn injected_thickness_is_the_references_own_float() {
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        let r = run(&c, Mode::Injected).unwrap();
        assert_eq!(r.status(), "PASS");
        // Two fragments; `t`, `thick_mode` and the two bit comparisons each.
        assert_eq!(r.checks.len(), 8);
        assert!(r.worst().unwrap().ratio() < 1e-15, "bit-identical on both fragments");
    }

    #[test]
    fn native_thickness_is_inside_the_widened_gate() {
        let c = Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).unwrap();
        let r = run(&c, Mode::Native).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert_eq!(r.checks.len(), 4);
        // The slab is the easy case: a flat wall, a single-peaked histogram. Both fragments are
        // inside D §10.2's own ±2 % here, which the widened gate never narrows.
        for check in &r.checks {
            assert!(check.tolerance >= NATIVE_RELATIVE, "the gate never tightens below 2 %");
            assert!(check.deviation < NATIVE_RELATIVE, "{}", check.line());
        }
    }
}
