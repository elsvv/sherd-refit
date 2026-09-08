//! The verification half of D §10.2's `stage 2` row: R §6's scores and R §6.5's verdict.
//!
//! # Injected
//!
//! The pose is the reference's own — `s2.T_frac2`, the last rung of R §5.6 — so this row measures
//! R §6 and nothing above it: the same pose, the same samples, the same two meshes, and the only
//! thing that differs is who computed the distances. Everything comes out of the dump:
//!
//! * `mesh.V` / `mesh.F` and `seg.frac_final` build the two BVHs the reference builds
//!   (`Fragment.scene` and `Fragment.frac_scene`) and the `fracture_area` that `contact` scales by;
//! * `md.Pf`, `md.S`, `md.sp` and `md.margin_idx` **at the pair's own `t`** are R §6.1's and
//!   R §6.4's sample sets and R §6.3's shell margin;
//! * `md.brk_P` / `md.brk_ns` are R §6.2's whole breakline;
//! * `scales.json` is R §1.2 as the reference resolved it, so not one threshold is the port's.
//!
//! The rows are D §10.2's: `tight` ±0.01, `gap` ±0.002 t, `seam` ±0.34 t, `cont` ±0.005 t,
//! `cont_n` ±0.01, `pen` ±0.0005 and `accepted` identical. Each per-pair row is the worst over the
//! pair's candidates; the `(all pairs)` rows add the distribution of every one of them
//! ([`Spread`]), because a worst case cannot tell one candidate in a thousand from all of them.
//!
//! **What is left between the two sides, and why the tolerances are not zero.** Four things, all
//! of them in the geometry engines rather than in the arithmetic. `tight`, `gap` and `contact` are
//! point-to-surface distances through `parry3d` against Embree, which E4 measured at ≤ 3.1e-5 on
//! 300 000 queries (float32 summation order); `pen` is an inside test, ray parity here and one ray
//! in Open3D (PMC-7); `seam` counts voxels, so it moves by a whole `1/3 t` when one breakline
//! point sits on a voxel boundary; and `cont`/`cont_n` are medians over a nearest-neighbour set
//! that can differ by a point where two margin samples are equidistant.
//!
//! # The `pen` row and meshes that are not closed
//!
//! R §3.3.2's `closed_enough` calls a working mesh watertight when **under 0.2 % of its edges are
//! boundary edges**, deliberately, so that a decimated scan with a few holes still gets a
//! penetration test. On such a mesh the question "is this point inside" has no answer: a ray that
//! leaves through a hole crosses the surface one time fewer, and the parity flips. The two
//! implementations then disagree — not by rounding, but because they ask different rays.
//!
//! Measured on `Pot_A_Piece_03_Mesh` (149 boundary edges, wall 3.45): Open3D's
//! `compute_signed_distance` calls **54** of the 20 000 samples of `Pot_A_Piece_08_Mesh` inside it
//! at candidate 8's pose, at depths of 7.6 to 7.8 units — **2.2 t**, inside a wall whose half
//! thickness is 1.72, which no point of that solid can be. Open3D's own `count_intersections`
//! reports an **even** number of crossings for every one of those 54 points along **all six** axis
//! directions, so its own ray parity contradicts its own occupancy there; the port's
//! majority-of-three says outside, and so does the geometry.
//!
//! The row therefore applies where the question is well posed. `pen` is gated at D §10.2's 0.0005
//! on pairs whose two working meshes are **closed** (`n_boundary = 0`); on a pair with an open
//! mesh the deviation is reported as `pen (open mesh)` against `max_pen`, the algorithm's own
//! scale for this quantity — the claim being that the ambiguity never spans the decision itself —
//! and the decision is gated exactly, twice over: `pen limit` counts the candidates on which the
//! two sides disagree about `pen ≤ max_pen`, and `accepted` counts the ones where R §6.5's
//! verdicts differ. Both must be zero on every pair, open mesh or not.
//!
//! # Native
//!
//! There is none here. Natively the port has its own candidates to score, which is the
//! [`candidates`](super::candidates) stage — D §10.2's `pair result` row — rather than a
//! comparison of two numbers computed for different poses.

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use sherd_core::error::Result;
use sherd_core::executor::CPU;
use sherd_core::matching::verify::{self, Scores};

use super::{Collection, Spread};
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// D §10.2: the tight-contact fraction.
pub const TIGHT: f64 = 0.01;
/// D §10.2: the median gap, in wall thicknesses.
pub const GAP_T: f64 = 0.002;
/// D §10.2: the seam length, in wall thicknesses — one voxel of R §6.2's `t/3` grid.
pub const SEAM_T: f64 = 0.34;
/// D §10.2: the shell step, in wall thicknesses.
pub const CONT_T: f64 = 0.005;
/// D §10.2: the shell normal agreement, a cosine.
pub const CONT_N: f64 = 0.01;
/// D §10.2: the penetrating fraction, on a pair of closed meshes (PMC-7).
pub const PEN: f64 = 0.0005;

/// One gated score: the row's name, how to read it off [`Scores`], and D §10.2's tolerance.
type Row = (&'static str, fn(&Scores) -> f64, f64);

/// The five scores that are gated on every pair, in the order the table prints them.
///
/// `pen` is the sixth and is not here: which row it lands in depends on whether the pair's two
/// working meshes are closed (see the module documentation).
const ROWS: [Row; 5] = [
    ("tight", |s| s.tight, TIGHT),
    ("gap", |s| s.gap, GAP_T),
    ("seam", |s| s.seam, SEAM_T),
    ("cont", |s| s.cont, CONT_T),
    ("cont_n", |s| s.cont_n, CONT_N),
];

/// Runs R §6 at every candidate pose of every pair and compares it.
#[allow(clippy::too_many_lines, reason = "one flat list of comparisons per pair")]
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("verify", mode);
    let params = collection.manifest.collection.params;
    let mut geometry = super::pairs::GeometryCache::default();
    let mut spread: Vec<Spread> = (0..ROWS.len()).map(|_| Spread::default()).collect();
    let (mut closed_pen, mut open_pen) = (Spread::default(), Spread::default());
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        if mode == Mode::Native {
            report.skip(
                &scope,
                "D §10.2's native column for the verification is the `pair result` row, which is \
                 the `candidates` stage: natively the port scores its own candidates, not the \
                 reference's",
            );
            continue;
        }
        let Some((fa, fb, used)) = super::hypotheses::sides(collection, &pair, &mut report)? else {
            continue;
        };
        if !(pair.has("s2.T_frac2.npy") && pair.has("s2.scores.json")) {
            // A pair whose R §5.5 kept nothing has no candidate to verify, and the dump says so
            // itself rather than leaving something out.
            let empty = pair.has("nms2.kept.npy")
                && npy::read_indices(pair.file("nms2.kept.npy"))?.is_empty();
            report.skip(
                &scope,
                if empty {
                    "R §5.5 kept no candidate: the pair has nothing to verify"
                } else {
                    "no s2.T_frac2 or s2.scores.json in the dump (level min)"
                },
            );
            continue;
        }
        let (Some(a), Some(b)) = (collection.fragment(&pair.a), collection.fragment(&pair.b))
        else {
            report.skip(&scope, "a fragment of the pair is not in the collection");
            continue;
        };
        let (Some(geometry_a), Some(geometry_b)) = (geometry.get(a)?, geometry.get(b)?) else {
            report.skip(&scope, "the dump has no working mesh for a fragment of the pair");
            continue;
        };
        let (Some(sa), Some(sb)) = (
            super::pairs::reference_surfaces(a, &geometry_a, &fa, used.t, used.surface_points)?,
            super::pairs::reference_surfaces(b, &geometry_b, &fb, used.t, used.surface_points)?,
        ) else {
            report.skip(&scope, "the dump has no sample arrays at the pair's own t");
            continue;
        };
        let closed = geometry_a.n_boundary == 0 && geometry_b.n_boundary == 0;

        let poses = npy::read_transforms(pair.file("s2.T_frac2.npy"))?;
        let theirs: Vec<Scores> = npy::read_json_as(pair.file("s2.scores.json"))?;
        let accepted: Vec<bool> = if pair.has("s2.accepted.npy") {
            npy::read_bool(pair.file("s2.accepted.npy"))?
        } else {
            Vec::new()
        };
        report.push(Check::count(&scope, "n_candidates", poses.len() as u64, theirs.len() as u64));
        if poses.len() != theirs.len() || accepted.len() != poses.len() {
            report.skip(&scope, "s2.T_frac2, s2.scores and s2.accepted do not describe each other");
            continue;
        }
        let sc = pair.scales()?;

        let ours: Vec<Scores> =
            poses.par_iter().map(|t| verify::verify(&CPU, &sa, &sb, t, &sc, true, None)).collect();
        let mut worst = [0.0_f64; ROWS.len()];
        let (mut worst_pen, mut over_limit, mut differing) = (0.0_f64, 0_usize, 0_usize);
        for ((mine, theirs), &accepted) in ours.iter().zip(&theirs).zip(&accepted) {
            for ((row, out), samples) in ROWS.iter().zip(&mut worst).zip(&mut spread) {
                let deviation = (row.1(mine) - row.1(theirs)).abs();
                *out = out.max(deviation);
                samples.push(deviation);
            }
            let deviation = (mine.pen - theirs.pen).abs();
            worst_pen = worst_pen.max(deviation);
            if closed { &mut closed_pen } else { &mut open_pen }.push(deviation);
            if (mine.pen <= params.max_pen) != (theirs.pen <= params.max_pen) {
                over_limit += 1;
            }
            if verify::accept(mine, &params, &sc) != accepted {
                differing += 1;
            }
        }
        for ((name, _, tolerance), deviation) in ROWS.iter().zip(&worst) {
            report.push(Check::absolute(&scope, name, *deviation, 0.0, *tolerance));
        }
        if closed {
            report.push(Check::absolute(&scope, "pen", worst_pen, 0.0, PEN));
        } else {
            report.push(Check::absolute(&scope, "pen (open mesh)", worst_pen, 0.0, params.max_pen));
        }
        report.push(Check::entries(&scope, "pen limit", over_limit, ours.len()));
        report.push(Check::entries(&scope, "accepted", differing, ours.len()));
    }
    for ((name, _, tolerance), samples) in ROWS.iter().zip(&mut spread) {
        samples.report(&mut report, distribution_rows(name), *tolerance);
    }
    closed_pen.report(&mut report, distribution_rows("pen"), PEN);
    open_pen.report(&mut report, distribution_rows("pen (open mesh)"), params.max_pen);
    Ok(report)
}

/// The four `(all pairs)` row names for one score.
///
/// `Spread::report` takes `&'static str`s and the set is fixed, so they are spelled out rather
/// than formatted at runtime.
fn distribution_rows(name: &str) -> [&'static str; 4] {
    match name {
        "tight" => ["tight p50", "tight p90", "tight p99", "tight max"],
        "gap" => ["gap p50", "gap p90", "gap p99", "gap max"],
        "seam" => ["seam p50", "seam p90", "seam p99", "seam max"],
        "cont" => ["cont p50", "cont p90", "cont p99", "cont max"],
        "cont_n" => ["cont_n p50", "cont_n p90", "cont_n p99", "cont_n max"],
        "pen" => ["pen p50", "pen p90", "pen p99", "pen max"],
        _ => ["pen open p50", "pen open p90", "pen open p99", "pen open max"],
    }
}
