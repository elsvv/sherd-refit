//! The samples stage: the whole-surface, fracture and shell-margin arrays (R §3.5.1–3.5.2,
//! §3.5.6, D §10.2 row `samples`).
//!
//! # Injected
//!
//! The arrays themselves cannot be compared point by point: PMC-9 lets the port draw from
//! `ChaCha8Rng` instead of numpy's PCG64, so the two implementations evaluate *the same estimator
//! on a different sample*. What can be compared exactly is everything that does **not** depend on
//! which numbers came out of the generator, and injected mode compares all of it, on the dump's
//! own mesh, labels, breakline and knobs:
//!
//! * **`n_frac`** — R §3.5.2's count rule, `clip(⌊150·fracture_area/t²⌋, 5000, 12000)`, recomputed
//!   from the dump's own labels and compared with the number of rows the reference actually drew.
//!   It is exact, and it is the one place the fracture area enters the sampling.
//! * **`S on face` / `Pf on face`** — the distance from each of the *reference's* points to the
//!   triangle its own `sp`/`fp` names, measured with this port's geometry. It tests the port's
//!   reading of `sample_on_faces`'s index convention against the reference's arrays: a port that
//!   built its points from the wrong two edges would report a residual here rather than an
//!   opinion.
//! * **`fracture faces`** — every `fp` names a face the dump labels fracture.
//! * **`margin count` / `margin members`** — R §3.5.6's band recomputed from the reference's `S`,
//!   `sp` and `brk_P`: the size of the mask must equal the reference's own `n_margin` exactly, and
//!   every index the reference kept must be in it. Equal counts and containment together mean the
//!   two masks are the same mask, which is as exact as a comparison of a *derived* array can be.
//! * **`surface normal` / `fracture normal`** — `SN = FN[sp]`, `Nf = FN[fp]`: the port stores the
//!   working mesh in `f32` (D §4.1) and the reference keeps it in `f64`, so this is the narrowing
//!   measured at the points the ICP will actually use. Both sides are normalised before the angle
//!   is taken, for the reason [`breakline::worst_angle`](super::breakline) documents, and the
//!   worst case is taken over the faces whose normal the narrowing *can* pin down — see
//!   [`CONDITIONED_ULPS`], and `sliver samples` for how few are left out.
//!
//! # Native
//!
//! Two independent area-weighted samples of the same surface are two different point sets, so the
//! native column is statistical throughout: the counts, the fraction of the sample that lands on
//! the fracture (which is the area weighting, measured), the fraction that lands in the margin
//! band, and the **spacing** — the 95th percentile of the nearest distance from each port sample
//! to the reference's set, against the value a Poisson process of the same density on the same
//! area would give, `0.977·√(A/n)`. Two i.i.d. samples of one surface land within a factor of one
//! of that; a sample set drawn on a *different* surface does not, and no per-fragment scale has to
//! be invented for the gate.

use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::fragment::samples::{self, SampleParams};
use sherd_core::mesh::geometry::{FaceGeometry, face_geometry};
use sherd_core::types::{FaceLabel, WorkingMesh};
use sherd_core::vec3::Vec3f;

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport, Unit};

/// D §10.2, injected column: how far a sample may sit off the triangle its index names, in `t`.
pub const INJECTED_ON_FACE_T: f64 = 1e-9;
/// D §10.2, injected column: the normal at a sample, in degrees (the `f32` narrowing of D §4.1).
pub const INJECTED_NORMAL_DEG: f64 = 0.1;
/// How many `f32` ulps of its own coordinate magnitude a triangle's smallest altitude must be
/// before its normal is well enough conditioned for [`INJECTED_NORMAL_DEG`] to mean anything.
///
/// Narrowing a vertex to `f32` moves it by up to one ulp of its coordinate, and moving a vertex by
/// `delta` perpendicular to the opposite edge turns the normal by `delta/h`, where `h` is that
/// vertex's altitude. So `h >= 1000 ulp` bounds the turn at `1e-3` radians = 0.057°, which is
/// under the gate **by construction** rather than by measurement — and a face below it is excluded
/// and counted instead of quietly setting the worst case.
pub const CONDITIONED_ULPS: f64 = 1000.0;
/// D §10.2, injected column: the share of samples that may land on a face too thin for that.
pub const INJECTED_SLIVER_FRACTION: f64 = 1e-3;
/// D §10.2, native column: the fracture sample count.
pub const NATIVE_COUNT: f64 = 0.10;
/// D §10.2, native column: the fraction of the surface sample that lands on the fracture.
pub const NATIVE_FRACTION: f64 = 0.02;
/// D §10.2, native column: the fraction of the surface sample that lands in the margin band.
pub const NATIVE_MARGIN_FRACTION: f64 = 0.05;
/// D §10.2, native column: the p95 nearest distance between the two sample sets, as a fraction of
/// the Poisson expectation for that density. `1.0` allows twice the expected spacing.
pub const NATIVE_SPACING: f64 = 1.0;
/// `√(ln 20 / π)`: the 95th percentile of the nearest-neighbour distance of a Poisson process of
/// unit density, which is what [`NATIVE_SPACING`] is a multiple of.
pub const POISSON_P95: f64 = 0.976_694_9;

/// Runs R §3.5.1–3.5.2 and §3.5.6 for every fragment and compares it with the dump.
#[allow(clippy::too_many_lines, reason = "one arm per mode, each a flat list of comparisons")]
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("samples", mode);
    for fragment in &collection.fragments {
        let name = fragment.name.as_str();
        if !fragment.has("md.S.npy") || !fragment.has("md.sp.npy") {
            report.skip(name, "no md.S in the dump (level min)");
            continue;
        }
        let params = match dump_params(fragment)? {
            Ok(params) => params,
            Err(reason) => {
                report.skip(name, reason);
                continue;
            }
        };
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
        let theirs = reference(fragment)?;
        let brk = if fragment.has("md.brk_P.npy") {
            npy::read_points(fragment.file("md.brk_P.npy"))?
        } else {
            Vec::new()
        };

        match mode {
            Mode::Injected => {
                // --- R §3.5.1–3.5.2: the counts ----------------------------------------------
                report.push(Check::count(
                    name,
                    "n_surface",
                    u64::from(params.surface_points),
                    theirs.s.len() as u64,
                ));
                let fracture_faces = faces_with(&labels, true);
                let area = samples::masked_area(&geom.areas, &fracture_faces);
                report.push(Check::count(
                    name,
                    "n_frac",
                    samples::fracture_count(area, params) as u64,
                    theirs.pf.len() as u64,
                ));

                // --- the points sit on the faces their own indices name -----------------------
                let tolerance = INJECTED_ON_FACE_T * params.t;
                report.push(distance_check(
                    name,
                    "S on face",
                    on_face(&theirs.s, &theirs.sp, &mesh.v, &mesh.f),
                    tolerance,
                ));
                report.push(distance_check(
                    name,
                    "Pf on face",
                    on_face(&theirs.pf, &theirs.fp, &mesh.v, &mesh.f),
                    tolerance,
                ));
                let stray = theirs.fp.iter().filter(|&&i| !frac[i as usize]).count();
                report.push(Check::entries(name, "fracture faces", stray, theirs.fp.len()));

                // --- R §3.5.6: the band, recomputed from the reference's own arrays -----------
                let d_brk = samples::breakline_distance(&theirs.s, &brk);
                let mask = samples::margin_indices(&theirs.sp, &labels, &d_brk, params);
                report.push(Check::count(
                    name,
                    "margin count",
                    mask.len() as u64,
                    theirs.n_margin as u64,
                ));
                // Both are ascending, so containment is a binary search rather than a scan.
                let outside =
                    theirs.margin_idx.iter().filter(|i| mask.binary_search(i).is_err()).count();
                report.push(Check::entries(
                    name,
                    "margin members",
                    outside,
                    theirs.margin_idx.len(),
                ));

                // --- R §3.6: the normals at the samples, which is the `f32` narrowing ---------
                let narrowed = narrowed_mesh(&mesh.v, &mesh.f);
                let conditioned = conditioned_faces(&mesh.v, &mesh.f, &geom);
                let mut slivers = 0;
                let mut total = 0;
                for (quantity, at) in
                    [("surface normal", &theirs.sp), ("fracture normal", &theirs.fp)]
                {
                    let (worst, excluded) = worst_normal_angle(&narrowed, &geom, at, &conditioned);
                    slivers += excluded;
                    total += at.len();
                    report.push(worst_check(name, quantity, worst, INJECTED_NORMAL_DEG));
                }
                #[allow(clippy::cast_precision_loss, reason = "sample counts are below 2^53")]
                report.push(Check::absolute(
                    name,
                    "sliver samples",
                    slivers as f64 / total.max(1) as f64,
                    0.0,
                    INJECTED_SLIVER_FRACTION,
                ));
            }
            Mode::Native => {
                let Some(source) = &fragment.source else {
                    report.skip(name, "no source file (pass --input DIR)");
                    continue;
                };
                let (fr, _) = Fragment::load_or_build(source, collection.target_faces, name, None)?;
                let ours = &fr.samples;

                report.push(Check::count(
                    name,
                    "n_surface",
                    ours.n_surface() as u64,
                    theirs.s.len() as u64,
                ));
                #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
                report.push(Check::relative(
                    name,
                    "n_frac",
                    ours.n_fracture() as f64,
                    theirs.pf.len() as f64,
                    NATIVE_COUNT,
                ));

                // The area weighting, measured: an area-weighted sample lands on the fracture as
                // often as the fracture is of the surface.
                report.push(Check::absolute(
                    name,
                    "fracture fraction",
                    fraction(&ours.sp, |i| fr.labels[i as usize].is_fracture()),
                    fraction(&theirs.sp, |i| frac[i as usize]),
                    NATIVE_FRACTION,
                ));

                // The margin *before* thinning on both sides: the port's is recomputed here, the
                // reference's comes from its own `md.rng.json`.
                #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
                let (ours_margin, theirs_margin) = (
                    native_margin(&fr) as f64 / ours.n_surface().max(1) as f64,
                    theirs.n_margin as f64 / theirs.s.len().max(1) as f64,
                );
                report.push(Check::absolute(
                    name,
                    "margin fraction",
                    ours_margin,
                    theirs_margin,
                    NATIVE_MARGIN_FRACTION,
                ));

                // The two clouds, against the spacing their own density implies.
                let total = geom.total_area();
                let fracture_area = samples::masked_area(&geom.areas, &faces_with(&labels, true));
                for (quantity, empty, mine, yours, area) in [
                    ("S spacing", "has surface", ours.surface_f64(), &theirs.s, total),
                    ("Pf spacing", "has fracture", ours.fracture_f64(), &theirs.pf, fracture_area),
                ] {
                    // A cloud one side has and the other has not is a difference in kind, not a
                    // spacing: there is no density to compare it against (R §5's first exit).
                    let expected = poisson_p95(area, yours.len());
                    if expected <= 0.0 || mine.is_empty() {
                        report.push(Check::flag(name, empty, !mine.is_empty(), !yours.is_empty()));
                        continue;
                    }
                    report.push(Check::relative(
                        name,
                        quantity,
                        percentile95(&mine, yours),
                        expected,
                        NATIVE_SPACING,
                    ));
                }
            }
        }
    }
    Ok(report)
}

/// The reference's own sampled arrays, as the dump carries them.
struct Reference {
    s: Vec<[f64; 3]>,
    sp: Vec<u32>,
    pf: Vec<[f64; 3]>,
    fp: Vec<u32>,
    margin_idx: Vec<u32>,
    /// The margin **before** thinning, from `md.rng.json`; the kept count when the dump has no
    /// such file, which is only ever right when nothing was thinned.
    n_margin: usize,
}

fn reference(fragment: &super::FragmentFixture) -> Result<Reference> {
    let margin_idx = if fragment.has("md.margin_idx.npy") {
        npy::read_indices(fragment.file("md.margin_idx.npy"))?
    } else {
        Vec::new()
    };
    let n_margin = if fragment.has("md.rng.json") {
        let file = fragment.file("md.rng.json");
        let json = npy::read_json(&file)?;
        usize::try_from(npy::field_u64(&json, "n_margin", &file)?).unwrap_or(usize::MAX)
    } else {
        margin_idx.len()
    };
    Ok(Reference {
        s: npy::read_points(fragment.file("md.S.npy"))?,
        sp: npy::read_indices(fragment.file("md.sp.npy"))?,
        pf: if fragment.has("md.Pf.npy") {
            npy::read_points(fragment.file("md.Pf.npy"))?
        } else {
            Vec::new()
        },
        fp: if fragment.has("md.fp.npy") {
            npy::read_indices(fragment.file("md.fp.npy"))?
        } else {
            Vec::new()
        },
        margin_idx,
        n_margin,
    })
}

/// The knobs the dump's arrays were drawn with, so that the injected run is the reference's own
/// experiment rather than this port's defaults at the reference's `t`.
fn dump_params(
    fragment: &super::FragmentFixture,
) -> Result<std::result::Result<SampleParams, &'static str>> {
    if !fragment.has("md.params.json") {
        return Ok(Err("no md.params in the dump: the arrays have no `t` and no counts"));
    }
    let file = fragment.file("md.params.json");
    let json = npy::read_json(&file)?;
    let count = |key: &str| -> Result<u32> {
        Ok(u32::try_from(npy::field_u64(&json, key, &file)?).unwrap_or(u32::MAX))
    };
    Ok(Ok(SampleParams {
        t: npy::field_f64(&json, "t", &file)?,
        seed: npy::field_u64(&json, "seed", &file)?,
        surface_points: count("surface_points")?,
        frac_per_t2: npy::field_f64(&json, "frac_per_t2", &file)?,
        min_frac_points: count("min_frac_points")?,
        max_frac_points: count("max_frac_points")?,
        margin_points: count("margin_points")?,
    }))
}

/// R §3.5.6's margin on the port's own fragment, **before** the thinning of `margin_idx`.
fn native_margin(fr: &Fragment) -> usize {
    let d_brk = samples::breakline_distance(&fr.samples.surface_f64(), &fr.brk.points_f64());
    samples::margin_indices(&fr.samples.sp, &fr.labels, &d_brk, fr.samples.params).len()
}

/// `Fracture` for true, `Shell` for false — the reference's `frac` mask as labels.
fn label(is_fracture: bool) -> FaceLabel {
    if is_fracture { FaceLabel::Fracture } else { FaceLabel::Shell }
}

/// The faces whose label matches, as the sampler's selection.
fn faces_with(labels: &[FaceLabel], fracture: bool) -> Vec<u32> {
    (0..labels.len())
        .filter(|&i| labels[i].is_fracture() == fracture)
        .map(|i| u32::try_from(i).expect("the face count fits in u32"))
        .collect()
}

/// The fraction of an index array whose entries satisfy a predicate.
#[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
fn fraction(idx: &[u32], predicate: impl Fn(u32) -> bool) -> f64 {
    if idx.is_empty() {
        return 0.0;
    }
    idx.iter().filter(|&&i| predicate(i)).count() as f64 / idx.len() as f64
}

/// The largest distance from a sample to the triangle its own index names.
fn on_face(points: &[[f64; 3]], faces: &[u32], v: &[[f64; 3]], f: &[[u32; 3]]) -> f64 {
    points
        .iter()
        .zip(faces)
        .map(|(p, &face)| {
            let tri = f[face as usize];
            point_triangle_distance(*p, v[tri[0] as usize], v[tri[1] as usize], v[tri[2] as usize])
        })
        .fold(0.0_f64, f64::max)
}

/// The distance from a point to a triangle, by clamping its barycentric coordinates.
///
/// Ericson's region test written out: the closest point of a triangle to `p` is either inside it,
/// on one of its edges, or one of its vertices, and the six cases are what the clamping below
/// enumerates.
#[allow(
    clippy::many_single_char_names,
    reason = "Ericson's d1..d6 and va/vb/vc are the published names of this test"
)]
fn point_triangle_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return norm(ap);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return norm(bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let s = d1 / (d1 - d3);
        return norm(sub(ap, scale(ab, s)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return norm(cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let s = d2 / (d2 - d6);
        return norm(sub(ap, scale(ac, s)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let s = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return norm(sub(cp, scale(sub(c, b), -s)));
    }
    let denom = 1.0 / (va + vb + vc);
    let closest = add(a, add(scale(ab, vb * denom), scale(ac, vc * denom)));
    norm(sub(p, closest))
}

/// The working mesh as the port stores it: the dump's vertices narrowed to `f32`, with the
/// per-face arrays derived from the narrowed values (D §4.1).
fn narrowed_mesh(v: &[[f64; 3]], f: &[[u32; 3]]) -> WorkingMesh {
    WorkingMesh::from_parts(v.iter().copied().map(Vec3f::from_f64).collect(), f.to_vec(), 0.0)
}

/// The largest angle, in degrees, between the port's stored face normal and the reference's `f64`
/// one over the faces an index array names, and how many of those samples were left out because
/// their face is too thin for the question ([`CONDITIONED_ULPS`]).
///
/// Both sides are normalised first, for the reason [`breakline`](super::breakline)'s
/// `worst_angle` documents.
fn worst_normal_angle(
    narrowed: &WorkingMesh,
    geom: &FaceGeometry,
    at: &[u32],
    conditioned: &[bool],
) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut excluded = 0;
    for &i in at {
        if !conditioned[i as usize] {
            excluded += 1;
            continue;
        }
        let mine = unit(narrowed.face_normals[i as usize].to_f64());
        worst = worst.max(dot(mine, unit(geom.normals[i as usize])).clamp(-1.0, 1.0).acos());
    }
    (worst.to_degrees(), excluded)
}

/// Which faces have a normal the `f32` narrowing cannot turn by more than `1/CONDITIONED_ULPS`
/// radians: those whose smallest altitude `2A/e_max` is at least that many `f32` ulps of the
/// largest coordinate of their own three vertices.
fn conditioned_faces(v: &[[f64; 3]], f: &[[u32; 3]], geom: &FaceGeometry) -> Vec<bool> {
    f.iter()
        .zip(&geom.areas)
        .map(|(tri, &area)| {
            let p = [v[tri[0] as usize], v[tri[1] as usize], v[tri[2] as usize]];
            let longest =
                norm(sub(p[1], p[0])).max(norm(sub(p[2], p[1]))).max(norm(sub(p[0], p[2])));
            let magnitude = p.iter().flatten().fold(0.0_f64, |m, c| m.max(c.abs()));
            if longest <= 0.0 {
                return false;
            }
            2.0 * area / longest >= CONDITIONED_ULPS * ulp_f32(magnitude)
        })
        .collect()
}

/// The distance from `x` to the next representable `f32` above it — one unit in the last place of
/// the value the port stores.
#[allow(clippy::cast_possible_truncation, reason = "the working mesh is f32 by design (D §4.1)")]
fn ulp_f32(x: f64) -> f64 {
    let narrowed = (x.abs() as f32).max(f32::MIN_POSITIVE);
    f64::from(f32::from_bits(narrowed.to_bits() + 1) - narrowed)
}

/// The 95th percentile of the nearest distance between two point sets, the worse of the two
/// directions.
fn percentile95(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    use sherd_core::spatial::kdtree::PointTree;
    let mut worst = 0.0_f64;
    for (from, to) in [(a, b), (b, a)] {
        let Some(tree) = PointTree::build(to) else { continue };
        let mut d: Vec<f64> = from.iter().map(|q| tree.nearest_distance(q).1).collect();
        d.sort_by(f64::total_cmp);
        worst = worst.max(super::breakline::percentile(&d, 0.95));
    }
    worst
}

/// The 95th percentile of the nearest-neighbour distance of `n` points spread over `area` as a
/// Poisson process: `√(ln 20 / (π·n/area))`.
#[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
fn poisson_p95(area: f64, n: usize) -> f64 {
    if n == 0 || area <= 0.0 {
        return 0.0;
    }
    POISSON_P95 * (area / n as f64).sqrt()
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

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n > 0.0 { [v[0] / n, v[1] / n, v[2] / n] } else { v }
}

#[cfg(test)]
mod tests {
    use super::{POISSON_P95, point_triangle_distance, poisson_p95, run};
    use crate::layout::FixtureDir;
    use crate::report::{Check, Mode};
    use crate::stages::Collection;
    use crate::stages::tests::{slab_dump, slab_input};

    #[test]
    fn the_injected_samples_are_the_references_own() {
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        let r = run(&c, Mode::Injected).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert!(r.skips.is_empty());
        // Two fragments × (n_surface, n_frac, two `on face`, fracture faces, margin count,
        // margin members, two normals, sliver samples).
        assert_eq!(r.checks.len(), 20);
        let n_frac = r.checks.iter().find(|c| c.quantity == "n_frac").expect("the count rule");
        assert!(n_frac.measured >= 5000.0, "the clamp is the floor: {}", n_frac.line());
    }

    #[test]
    fn the_native_samples_describe_the_same_surface() {
        let c = Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).unwrap();
        let r = run(&c, Mode::Native).unwrap();
        assert_eq!(r.status(), "PASS", "{:?}", r.failures().map(Check::line).collect::<Vec<_>>());
        assert_eq!(r.checks.len(), 12, "six comparisons for two fragments");
    }

    /// The distance to a triangle, in all three of its regions.
    #[test]
    fn the_point_to_triangle_distance_is_the_distance() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Above the interior.
        assert!((point_triangle_distance([0.25, 0.25, 2.0], a, b, c) - 2.0).abs() < 1e-12);
        // On it.
        assert!(point_triangle_distance([0.25, 0.25, 0.0], a, b, c) < 1e-12);
        // Past a vertex.
        assert!((point_triangle_distance([-1.0, 0.0, 0.0], a, b, c) - 1.0).abs() < 1e-12);
        assert!((point_triangle_distance([0.0, 2.0, 0.0], a, b, c) - 1.0).abs() < 1e-12);
        // Past an edge, in the plane.
        assert!((point_triangle_distance([0.5, -1.0, 0.0], a, b, c) - 1.0).abs() < 1e-12);
        // Past the hypotenuse.
        let d = point_triangle_distance([1.0, 1.0, 0.0], a, b, c);
        assert!((d - 0.5_f64.sqrt()).abs() < 1e-12, "{d}");
    }

    /// The Poisson expectation is a spacing: it scales as `√(area/n)` and vanishes for nothing.
    #[test]
    fn the_expected_spacing_scales_with_the_density() {
        assert!((poisson_p95(1.0, 1) - POISSON_P95).abs() < 1e-9);
        assert!((poisson_p95(400.0, 100) - POISSON_P95 * 2.0).abs() < 1e-9);
        assert!(poisson_p95(1.0, 0).abs() < 1e-12);
        assert!(poisson_p95(0.0, 10).abs() < 1e-12);
    }
}
