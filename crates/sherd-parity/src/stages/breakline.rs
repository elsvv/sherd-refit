//! The breakline stage: the points where fracture meets shell, and the frames on them
//! (R §3.5.3–3.5.5, D §10.2 row `breakline`).
//!
//! # Injected
//!
//! R §3.5.3–3.5.5 runs on the dump's own working mesh (`mesh.V`, `mesh.F`), the dump's own labels
//! (`seg.frac_final`) and the dump's own knobs (`md.params.json`), so nothing upstream can move
//! the answer and the comparison is with the reference's `md.brk_*` arrays directly.
//!
//! D §10.2 asks for three things — the count exactly, the point sets within `1e-4 t`, and the
//! dihedral within 0.1° per matched point — and "matched" here means *the same index*, because
//! both implementations emit the points in face-adjacency order (`mesh::adjacency::face_adjacency`
//! reproduces the reference's `np.lexsort`). The stage therefore checks the ordered distance as
//! well as the set distance: a port that produced the right cloud in the wrong order would pass
//! the Hausdorff gate and break every hypothesis downstream, and only the ordered check says so.
//!
//! Beside those, the frames and the subset:
//!
//! * `ns`, `nf`, `f` and the tangent `ns × f`, as the worst angle over the points whose frame both
//!   implementations call valid — both sides normalised first, because the port's are stored as
//!   `f32` and `arccos` of a raw dot product reads a length error of `delta` as an angle of
//!   `sqrt(2 delta)` (see [`worst_angle`]);
//! * `valid` itself, entry by entry;
//! * `brk_sub` **as a set**. The reference reads it out of an Open3D hash map, so its order is a
//!   hash artefact (PMC-4) and this port sorts it — but the *set* is exact, because the lowest
//!   index of each voxel is exact and the voxel occupancy is arithmetic (`voxel_representatives`).
//!   Sorted, the two must be equal entry for entry.
//!
//! # Native
//!
//! The port's working mesh is its own — a different decimation and a different `t` — so the two
//! breaklines are two different point sets on nearly the same curve, and D §10.2 gates them
//! loosely: the count within 10 %, the 99th percentile of the symmetric point-to-set distance
//! within `0.5 t`, and the two dihedral distributions within a KS statistic of 0.05. The `t` the
//! first two are measured in is the *reference's*, which is the fixture's own unit.

use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::fragment::breakline::{self, BrkParams};
use sherd_core::mesh::geometry::face_geometry;
use sherd_core::spatial::kdtree::PointTree;
use sherd_core::types::FaceLabel;

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport, Unit};

/// D §10.2, injected column: the point sets, in units of `t`.
pub const INJECTED_HAUSDORFF_T: f64 = 1e-4;
/// D §10.2, injected column: the dihedral per matched point, in degrees.
pub const INJECTED_DIH_DEG: f64 = 0.1;
/// Diagnostic gate on the macro normals and the frame they span, in degrees.
pub const INJECTED_FRAME_DEG: f64 = 0.1;
/// D §10.2, native column: the number of breakline points.
pub const NATIVE_COUNT: f64 = 0.10;
/// D §10.2, native column: the 99th percentile of the symmetric point-to-set distance, in `t`.
pub const NATIVE_P99_T: f64 = 0.5;
/// D §10.2, native column: the two-sample KS statistic of the dihedral distributions.
pub const NATIVE_KS: f64 = 0.05;

/// Runs R §3.5.3–3.5.5 for every fragment and compares it with the dump.
#[allow(clippy::too_many_lines, reason = "one arm per mode, each a flat list of comparisons")]
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("breakline", mode);
    for fragment in &collection.fragments {
        let name = fragment.name.as_str();
        if !fragment.has("md.brk_P.npy") {
            report.skip(name, "no md.brk_P in the dump (level min)");
            continue;
        }
        let params = match dump_params(fragment)? {
            Ok(params) => params,
            Err(reason) => {
                report.skip(name, reason);
                continue;
            }
        };
        let theirs = reference(fragment)?;

        match mode {
            Mode::Injected => {
                let Some(mesh) = fragment.working()? else {
                    report.skip(name, "no mesh.V in the dump: the reference's arrays have no mesh");
                    continue;
                };
                if !fragment.has("seg.frac_final.npy") {
                    report.skip(name, "no seg.frac_final in the dump: nothing to label the mesh");
                    continue;
                }
                let frac = npy::read_bool(fragment.file("seg.frac_final.npy"))?;
                if frac.len() != mesh.f.len() {
                    report.skip(name, "seg.frac_final does not describe mesh.F");
                    continue;
                }
                let labels: Vec<FaceLabel> = frac.iter().copied().map(label).collect();
                let geom = face_geometry(&mesh.v, &mesh.f);
                let ours = breakline::build(&mesh.v, &mesh.f, &geom, &labels, params);

                // --- the three gates D §10.2 states ------------------------------------------
                report.push(Check::count(name, "count", ours.len() as u64, theirs.len() as u64));
                let tolerance = INJECTED_HAUSDORFF_T * params.t;
                let points = ours.points_f64();
                report.push(distance_check(
                    name,
                    "points",
                    hausdorff(&points, &theirs.p),
                    tolerance,
                ));
                if ours.len() == theirs.len() {
                    report.push(distance_check(
                        name,
                        "points in order",
                        ordered_distance(&points, &theirs.p),
                        tolerance,
                    ));
                    report.push(worst_check(
                        name,
                        "dihedral",
                        worst_delta(&ours.dihedrals(), &theirs.dih),
                        INJECTED_DIH_DEG,
                    ));

                    // --- the diagnosis: which half of the frame moved -------------------------
                    let valid = ours.valid();
                    let both: Vec<usize> =
                        (0..ours.len()).filter(|&i| valid[i] && theirs.valid[i]).collect();
                    let differing =
                        (0..ours.len()).filter(|&i| valid[i] != theirs.valid[i]).count();
                    report.push(Check::entries(name, "valid", differing, ours.len()));
                    for (quantity, mine, yours) in [
                        ("ns", vectors(&ours.ns), &theirs.ns),
                        ("nf", vectors(&ours.nf), &theirs.nf),
                        ("f", vectors(&ours.f), &theirs.f),
                        ("tangent", vectors(&ours.tangents()), &theirs.tangent),
                    ] {
                        report.push(worst_check(
                            name,
                            quantity,
                            worst_angle(&mine, yours, &both),
                            INJECTED_FRAME_DEG,
                        ));
                    }
                }

                // --- the hypothesis subset, as the set PMC-4 leaves deterministic --------------
                let mut mine = ours.sub.clone();
                let mut yours = theirs.sub.clone();
                mine.sort_unstable();
                yours.sort_unstable();
                report.push(Check::count(name, "sub count", mine.len() as u64, yours.len() as u64));
                let differing = if mine.len() == yours.len() {
                    mine.iter().zip(&yours).filter(|(a, b)| a != b).count()
                } else {
                    mine.len().abs_diff(yours.len())
                };
                report.push(Check::entries(name, "sub set", differing, yours.len()));
            }
            Mode::Native => {
                let Some(source) = &fragment.source else {
                    report.skip(name, "no source file (pass --input DIR)");
                    continue;
                };
                let (fr, _) = Fragment::load_or_build(source, collection.target_faces, name, None)?;
                let ours = &fr.brk;

                #[allow(clippy::cast_precision_loss, reason = "point counts are far below 2^53")]
                report.push(Check::relative(
                    name,
                    "count",
                    ours.len() as f64,
                    theirs.len() as f64,
                    NATIVE_COUNT,
                ));
                if ours.is_empty() || theirs.p.is_empty() {
                    report.push(Check::flag(name, "has breakline", !ours.is_empty(), false));
                    continue;
                }
                report.push(distance_check(
                    name,
                    "p99 distance",
                    percentile99(&ours.points_f64(), &theirs.p),
                    NATIVE_P99_T * params.t,
                ));
                report.push(worst_check(
                    name,
                    "dihedral KS",
                    ks_statistic(&ours.dihedrals(), &theirs.dih),
                    NATIVE_KS,
                ));
            }
        }
    }
    Ok(report)
}

/// The reference's own breakline arrays, as the dump carries them.
struct Reference {
    p: Vec<[f64; 3]>,
    ns: Vec<[f64; 3]>,
    nf: Vec<[f64; 3]>,
    f: Vec<[f64; 3]>,
    tangent: Vec<[f64; 3]>,
    dih: Vec<f64>,
    valid: Vec<bool>,
    sub: Vec<u32>,
}

impl Reference {
    fn len(&self) -> usize {
        self.p.len()
    }
}

fn reference(fragment: &super::FragmentFixture) -> Result<Reference> {
    let p = npy::read_points(fragment.file("md.brk_P.npy"))?;
    let n = p.len();
    let mut out = Reference {
        p,
        ns: npy::read_points(fragment.file("md.brk_ns.npy"))?,
        nf: npy::read_points(fragment.file("md.brk_nf.npy"))?,
        f: npy::read_points(fragment.file("md.brk_f.npy"))?,
        tangent: Vec::new(),
        dih: Vec::new(),
        valid: Vec::new(),
        sub: npy::read_indices(fragment.file("md.brk_sub.npy"))?,
    };
    // `brk_t`, `brk_dih` and `valid` are derived (R §3.6), and the dump carries them; when it
    // does not, they are derived here from the arrays it does carry, by the same formulae.
    out.tangent = if fragment.has("md.brk_t.npy") {
        npy::read_points(fragment.file("md.brk_t.npy"))?
    } else {
        (0..n).map(|i| cross(out.ns[i], out.f[i])).collect()
    };
    out.dih = if fragment.has("md.brk_dih.npy") {
        npy::read_f64(fragment.file("md.brk_dih.npy"))?
    } else {
        (0..n).map(|i| dot(out.ns[i], out.nf[i]).clamp(-1.0, 1.0).acos().to_degrees()).collect()
    };
    out.valid = if fragment.has("md.valid.npy") {
        npy::read_bool(fragment.file("md.valid.npy"))?
    } else {
        (0..n)
            .map(|i| {
                norm(out.ns[i]) > breakline::VALID_MIN
                    && norm(out.nf[i]) > breakline::VALID_MIN
                    && norm(cross(out.ns[i], out.f[i])) > breakline::VALID_MIN
            })
            .collect()
    };
    Ok(out)
}

/// The knobs the dump's arrays were built with, so that the injected run is the reference's own
/// experiment rather than this port's defaults at the reference's `t`.
///
/// Returns the reason to skip when the dump has no `md.params.json` and no `thick.t.json` to fall
/// back on, or when its knobs are ones this build cannot be asked for.
fn dump_params(
    fragment: &super::FragmentFixture,
) -> Result<std::result::Result<BrkParams, &'static str>> {
    if fragment.has("md.params.json") {
        let file = fragment.file("md.params.json");
        let json = npy::read_json(&file)?;
        return Ok(Ok(BrkParams {
            t: npy::field_f64(&json, "t", &file)?,
            macro_inner: npy::field_f64(&json, "macro_inner", &file)?,
            macro_outer: npy::field_f64(&json, "macro_outer", &file)?,
            brk_voxel: npy::field_f64(&json, "brk_voxel", &file)?,
        }));
    }
    if fragment.has("thick.t.json") {
        return Ok(Ok(BrkParams::at(npy::read_scalar(fragment.file("thick.t.json"))?)));
    }
    Ok(Err("no md.params and no thick.t in the dump: the arrays have no `t`"))
}

/// `Fracture` for true, `Shell` for false — the reference's `frac` mask as labels.
fn label(is_fracture: bool) -> FaceLabel {
    if is_fracture { FaceLabel::Fracture } else { FaceLabel::Shell }
}

/// A `Vec3f` array widened for comparison with the reference's `f64` one.
fn vectors(v: &[sherd_core::vec3::Vec3f]) -> Vec<[f64; 3]> {
    v.iter().map(|p| p.to_f64()).collect()
}

/// The symmetric Hausdorff distance between two point sets.
fn hausdorff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let (mut worst, mut any) = (0.0_f64, false);
    for (from, to) in [(a, b), (b, a)] {
        if let Some(tree) = PointTree::build(to) {
            any = true;
            for q in from {
                worst = worst.max(tree.nearest_distance(q).1);
            }
        }
    }
    if any { worst } else { 0.0 }
}

/// The largest distance between two point sets *taken in order* — what makes the comparison a
/// comparison of arrays rather than of clouds.
fn ordered_distance(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            norm(d)
        })
        .fold(0.0_f64, f64::max)
}

/// The 99th percentile of the point-to-set distance, the worse of the two directions.
///
/// The percentile is numpy's: linear interpolation between the two neighbouring order statistics
/// (R §0), so the number is comparable with one computed on the Python side.
fn percentile99(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    for (from, to) in [(a, b), (b, a)] {
        let Some(tree) = PointTree::build(to) else { continue };
        let mut d: Vec<f64> = from.iter().map(|q| tree.nearest_distance(q).1).collect();
        d.sort_by(f64::total_cmp);
        worst = worst.max(percentile(&d, 0.99));
    }
    worst
}

/// numpy's `percentile(x, 100·q)` on an already sorted array: linear interpolation.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0 ≤ lo ≤ len − 1 by construction"
    )]
    let i = lo as usize;
    if i + 1 >= sorted.len() {
        return sorted[sorted.len() - 1];
    }
    sorted[i] + (pos - lo) * (sorted[i + 1] - sorted[i])
}

/// The two-sample Kolmogorov–Smirnov statistic: the largest gap between two empirical
/// distribution functions.
fn ks_statistic(first: &[f64], second: &[f64]) -> f64 {
    if first.is_empty() || second.is_empty() {
        return if first.is_empty() && second.is_empty() { 0.0 } else { 1.0 };
    }
    let mut left: Vec<f64> = first.to_vec();
    let mut right: Vec<f64> = second.to_vec();
    left.sort_by(f64::total_cmp);
    right.sort_by(f64::total_cmp);
    #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
    let (n_left, n_right) = (left.len() as f64, right.len() as f64);
    let (mut at_left, mut at_right, mut worst) = (0_usize, 0_usize, 0.0_f64);
    while at_left < left.len() && at_right < right.len() {
        let value = left[at_left].min(right[at_right]);
        while at_left < left.len() && left[at_left] <= value {
            at_left += 1;
        }
        while at_right < right.len() && right[at_right] <= value {
            at_right += 1;
        }
        #[allow(clippy::cast_precision_loss, reason = "as above")]
        let gap = (at_left as f64 / n_left - at_right as f64 / n_right).abs();
        worst = worst.max(gap);
    }
    worst
}

/// The largest absolute difference between two arrays of the same length.
fn worst_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

/// The largest angle, in degrees, between two arrays of vectors, over the given indices.
///
/// **Both sides are normalised first, and that is not a formality.** The reference's macro normals
/// are unit in `f64`; the port's are unit in `f64` and then stored as `f32` (D §4.1), so widening
/// them back gives a vector whose *direction* is the reference's to a part in 1e8 but whose
/// *length* is `1 ± 5e-8`. `arccos` of the raw dot product turns that length error into a
/// spurious angle of `sqrt(2 delta)` — 0.018 degrees for `delta = 4.7e-8`, four orders of
/// magnitude above the direction error it is supposed to be measuring, and growing as the square
/// root of a quantity the `points` check already reports directly. Normalising measures the angle.
fn worst_angle(a: &[[f64; 3]], b: &[[f64; 3]], at: &[usize]) -> f64 {
    at.iter()
        .map(|&i| dot(unit(a[i]), unit(b[i])).clamp(-1.0, 1.0).acos())
        .fold(0.0_f64, f64::max)
        .to_degrees()
}

/// `v / |v|`, leaving a zero vector alone.
fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n > 0.0 { [v[0] / n, v[1] / n, v[2] / n] } else { v }
}

/// A distance that must be under a tolerance, reported against a reference of zero.
fn distance_check(name: &str, quantity: &'static str, measured: f64, tolerance: f64) -> Check {
    Check::absolute(name, quantity, measured, 0.0, tolerance)
}

/// A worst-case deviation that must be under a tolerance, in the quantity's own units.
fn worst_check(name: &str, quantity: &'static str, worst: f64, tolerance: f64) -> Check {
    let mut check = Check::absolute(name, quantity, worst, 0.0, tolerance);
    check.unit = Unit::Absolute;
    check
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{ks_statistic, percentile, run};
    use crate::layout::FixtureDir;
    use crate::report::{Check, Mode};
    use crate::stages::Collection;
    use crate::stages::tests::{slab_dump, slab_input};

    #[test]
    fn the_injected_breakline_is_the_references_own() {
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        let r = run(&c, Mode::Injected).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert!(r.skips.is_empty());
        // Two fragments × (count, points, points in order, dihedral, valid, four frames, and the
        // subset's count and set).
        assert_eq!(r.checks.len(), 22);
        let count = r.checks.iter().find(|c| c.quantity == "count").expect("the first gate");
        assert!(count.measured > 100.0, "the slab has a breakline: {}", count.line());
    }

    #[test]
    fn the_native_breakline_lies_on_the_references_curve() {
        let c = Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).unwrap();
        let r = run(&c, Mode::Native).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert_eq!(r.checks.len(), 6, "count, p99 distance and KS for two fragments");
    }

    #[test]
    fn the_percentile_is_numpys_and_the_ks_statistic_is_the_two_sample_one() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&x, 0.0) - 0.0).abs() < 1e-12);
        assert!((percentile(&x, 1.0) - 4.0).abs() < 1e-12);
        // numpy: percentile([0..4], 99) = 3.96 by linear interpolation.
        assert!((percentile(&x, 0.99) - 3.96).abs() < 1e-12, "{}", percentile(&x, 0.99));
        assert!((percentile(&[], 0.5) - 0.0).abs() < 1e-12);

        // Two identical samples never separate; two disjoint ones separate completely.
        assert!((ks_statistic(&x, &x) - 0.0).abs() < 1e-12);
        assert!((ks_statistic(&x, &[10.0, 11.0]) - 1.0).abs() < 1e-12);
        // Half of one sample shifted past the other: the gap is a half.
        let y = [0.0, 1.0, 10.0, 11.0, 12.0];
        assert!((ks_statistic(&x, &y) - 0.6).abs() < 1e-12, "{}", ks_statistic(&x, &y));
        assert!((ks_statistic(&[], &[]) - 0.0).abs() < 1e-12);
        assert!((ks_statistic(&x, &[]) - 1.0).abs() < 1e-12);
    }
}
