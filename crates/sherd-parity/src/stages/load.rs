//! The load stage: read, clean, keep the largest component (R §3.1, D §10.2 row `load`).
//!
//! D §10.2 asks for the counts after cleaning and after the largest-component pass, exactly, in
//! both modes — and there is nothing to inject here, because the stage's input *is* the file. So
//! injected and native run the same comparison, and injected mode adds the one thing the dump
//! makes possible: comparing the arrays themselves, vertex by vertex and triangle by triangle,
//! against `load.V0` / `load.F0`. That is a far stronger statement than the counts, and it is the
//! statement everything downstream rests on — every injected comparison of the later stages is
//! meaningless if the port and the reference do not agree on what the file contained.
//!
//! The coordinates are compared **in units of the last place of an `f32`**, not exactly, and one
//! ULP is the tolerance. Every PLY in the benchmark comes back at 0.000 ULP — bit-identical, which
//! is the statement that matters. The five SfS++ sets are OBJ, and there the port and Open3D
//! differ by exactly 1.000 ULP on 1.7–3.1 % of coordinates, because Open3D reads OBJ through
//! Assimp, whose `fast_atof` accumulates the decimal digits itself instead of calling the
//! platform's correctly rounded `strtod`, and lands one ULP low on some values. Nothing on this
//! side can be changed to close that gap (`tobj`'s `use_f64` would not: the difference is in the
//! *reference's* parser, not in ours), and one ULP of an `f32` at these scales is 1e-7 of the mesh
//! — five orders of magnitude below D §10.2's tightest downstream tolerance. Measured in S4's
//! note; a reader that actually mis-parsed a file would be thousands of ULPs out, not one.
//!
//! `F0` is compared exactly: an index is an integer and there is nothing to round.
//!
//! A dump written at the `slim` or `min` level does not carry `load.V0`; the counts are still
//! compared and the array comparison is skipped, with a reason that names the consequence — the
//! injected thickness stage skips that fragment too, because nothing pins `(V0, F0)` any more.

use sherd_core::error::Result;
use sherd_core::io;
use sherd_core::mesh::components::largest_component;

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport, Unit};

/// How far two readings of the same file may put a vertex: one unit in the last place of an
/// `f32`, which is what separates Assimp's `fast_atof` from a correctly rounded parse.
pub const COORDINATE_ULPS: f64 = 1.0;

/// Runs R §3.1 for every fragment of the collection and compares it with the dump.
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("load", mode);
    for fragment in &collection.fragments {
        let counts_file = fragment.file("load.n_orig.json");
        if !counts_file.is_file() {
            report.skip(&fragment.name, "no load.n_orig.json in the dump");
            continue;
        }
        let Some(source) = &fragment.source else {
            report.skip(&fragment.name, "no source file (pass --input DIR)");
            continue;
        };
        let counts = npy::read_json(&counts_file)?;

        let mut mesh = io::load_mesh(source)?;
        // R §3.1 step 3: the two `n_orig_*` counts are taken after cleaning and *before* the
        // component filter, so a fragment that arrives as two shells still reports what the file
        // held.
        report.push(Check::count(
            &fragment.name,
            "n_orig_vertices",
            mesh.v.len() as u64,
            npy::field_u64(&counts, "n_orig_vertices", &counts_file)?,
        ));
        report.push(Check::count(
            &fragment.name,
            "n_orig_faces",
            mesh.f.len() as u64,
            npy::field_u64(&counts, "n_orig_faces", &counts_file)?,
        ));

        largest_component(&mut mesh);
        report.push(Check::count(
            &fragment.name,
            "n_vertices",
            mesh.v.len() as u64,
            npy::field_u64(&counts, "n_vertices", &counts_file)?,
        ));
        report.push(Check::count(
            &fragment.name,
            "n_faces",
            mesh.f.len() as u64,
            npy::field_u64(&counts, "n_faces", &counts_file)?,
        ));

        // The arrays themselves, when the dump carries them.
        // F3: this is the comparison that pins the port's `(V0, F0)` to the reference's, and
        // every injected comparison downstream rests on it. When the dump does not carry the
        // arrays the skip has to say so, because the consequence travels: the injected thickness
        // stage skips the same fragment rather than quietly running on arrays of its own.
        if !fragment.has("load.V0.npy") {
            report.skip(
                &fragment.name,
                "no load.V0 in the dump (level slim or min): (V0, F0) is not pinned, and the \
                 injected thickness stage skips this fragment for the same reason",
            );
            continue;
        }
        let v0 = npy::read_points(fragment.file("load.V0.npy"))?;
        let f0 = npy::read_triangles(fragment.file("load.F0.npy"))?;
        let worst = if v0.len() == mesh.v.len() {
            let floor = extent(&v0);
            mesh.v
                .iter()
                .zip(&v0)
                .flat_map(|(a, b)| (0..3).map(move |c| ulps_apart(a[c], b[c], floor)))
                .fold(0.0_f64, f64::max)
        } else {
            f64::INFINITY
        };
        let mut check = Check::absolute(&fragment.name, "V0", worst, 0.0, COORDINATE_ULPS);
        check.unit = Unit::Ulps;
        report.push(check);

        let differing_f = if f0.len() == mesh.f.len() {
            mesh.f.iter().zip(&f0).filter(|(a, b)| a != b).count()
        } else {
            f0.len().abs_diff(mesh.f.len())
        };
        report.push(Check::entries(&fragment.name, "F0", differing_f, f0.len()));
    }
    Ok(report)
}

/// The largest span of the mesh along any axis — the scale at which a coordinate stops being
/// worth comparing relatively.
fn extent(v: &[[f64; 3]]) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in v {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    (0..3).map(|c| hi[c] - lo[c]).fold(0.0_f64, f64::max)
}

/// How many units in the last place of an `f32` separate two coordinates.
///
/// The magnitude the ULP is taken at is `max(|a|, |b|, floor)`: without the floor, two coordinates
/// a hair either side of zero would be millions of ULPs apart and mean nothing, since a mesh's
/// precision is set by its extent and not by how close a vertex happens to sit to the origin.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the magnitude is narrowed on purpose: the question is what an f32 can resolve there"
)]
fn ulps_apart(a: f64, b: f64, floor: f64) -> f64 {
    if a.to_bits() == b.to_bits() {
        return 0.0;
    }
    let magnitude = a.abs().max(b.abs()).max(floor) as f32;
    let ulp = f64::from(magnitude.next_up() - magnitude);
    if ulp > 0.0 { (a - b).abs() / ulp } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use super::{extent, run, ulps_apart};
    use crate::layout::FixtureDir;
    use crate::report::Check;
    use crate::report::Mode;
    use crate::stages::Collection;
    use crate::stages::tests::{slab_dump, slab_input};

    #[test]
    fn a_coordinate_is_measured_in_the_f32_ulps_that_separate_two_readers() {
        // A PLY is read bit for bit, and that is 0 ULPs.
        assert!(ulps_apart(1.0, 1.0, 1.0) < f64::EPSILON);
        // One ULP of an f32 near 1.0 is 2^-23, whoever the two readers are.
        let one_ulp = f64::from(f32::EPSILON);
        assert!((ulps_apart(1.0, 1.0 + one_ulp, 1.0) - 1.0).abs() < 1e-9);
        // The floor keeps a coordinate near the origin from dominating the metric.
        assert!(ulps_apart(0.0, 1e-9, 100.0) < 1.0);
        assert!(ulps_apart(0.0, 1e-9, 0.0).is_infinite() || ulps_apart(0.0, 1e-9, 0.0) > 1e6);
        // The extent is the largest span of any axis.
        assert!((extent(&[[0.0, 0.0, 0.0], [1.0, 5.0, 2.0]]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn the_slab_loads_exactly_as_open3d_loaded_it() {
        let c = Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).unwrap();
        for mode in [Mode::Injected, Mode::Native] {
            let r = run(&c, mode).unwrap();
            assert_eq!(r.status(), "PASS", "{:?}", r.failures().next().map(Check::line));
            // Four counts and two array comparisons per fragment.
            assert_eq!(r.checks.len(), 12);
            assert!(r.skips.is_empty());
            assert!(r.worst().unwrap().ratio() < 1e-15, "every load check is exact and equal");
        }
    }

    #[test]
    fn without_the_source_files_the_stage_skips_rather_than_failing() {
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        let r = run(&c, Mode::Native).unwrap();
        assert!(r.checks.is_empty());
        assert_eq!(r.skips.len(), 2);
        assert!(r.skips[0].reason.contains("--input"));
        assert_eq!(r.status(), "SKIP");
    }
}
