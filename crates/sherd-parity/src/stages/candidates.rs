//! D §10.2's `pair result` row: what `match_pair` returns — the ranking of R §5.7 and the
//! accepted set of R §6.5.
//!
//! # Injected
//!
//! The candidates are the reference's own stage-2 poses (`s2.T_frac2`) and its own stage-1 scores
//! (`s1.score`, `nms2.kept`); what the port supplies is R §6's scores, R §5.7's ranking and R
//! §6.5's verdict. The row therefore measures the *ordering and the cut*, which the
//! [`verify`](super::verify) stage cannot: two candidates whose scores agree to 1e-9 still have to
//! come back in the same order and the same five have to be returned.
//!
//! Four things are compared against `result.candidates.json`: how many candidates come back
//! (`keep = 5`), **which** ones and in what order (by the index of the reference's own stage-2
//! candidate each returned pose is), the `accepted` flag of each, and `brk_best` — the pair's best
//! stage-1 re-score, which R §5.7 writes onto every candidate after the sort.
//!
//! The order is compared by index rather than by pose, because the poses *are* the reference's:
//! the returned matrix is `s2.T_frac2[i]` for some `i`, so the ranking is a permutation and
//! comparing permutations is exact where comparing matrices would need a tolerance.
//!
//! # Native
//!
//! The port preprocesses both fragments from their files (R §3) and runs the whole of
//! [`match_pair`](sherd_core::matching::pair::match_pair) — its own samples, its own hypotheses,
//! its own suppression with the tie-break PMC-6 allows, its own ladders and its own verification.
//!
//! **Nothing here is a parity claim, and that is the finding rather than a shortfall.** D §10.2's
//! native `pair result` row asked for an identical accepted set and a best candidate within
//! 1° / 0.05 t; neither is reachable, and the reason is in the rows above this one rather than in
//! R §6. The port scores what the reference scores, exactly — the [`verify`](super::verify) row
//! reproduces `tight` and `seam` bit for bit and `gap` to 3e-6 t at the reference's own poses —
//! but natively the two implementations *score different poses*: PMC-9 gives them different
//! samples and therefore different breaklines, and PMC-6 lets R §5.3's suppression keep a
//! different set, which step C1 measured at 14.3 % of the reference's kept hypotheses on average
//! and 43.6 % at worst. R §6.5 is a threshold on the output of that search, and R §13 already
//! records that the decision "flips on nothing" for a candidate sitting on `min_tight`.
//!
//! So the rows here are regression alarms gated at their measured worst plus headroom — the shape
//! D §10.2 already uses for the PMC-6 tie rows of the `nms` stage — and the *quality* claim is
//! made against the ground truth in `notes/2026-09-06-c3-verify.md` §5 rather than against the
//! reference: over the six sets with an adjacency list the port accepts **56 true joins and 28
//! false** against the reference's **54 and 26**, and on `synthetic_20`, the one set whose ground
//! truth is complete and does not interpenetrate, both accept **23 of 23 true joins with no false
//! one at all**.
//!
//! | row | what it counts | alarm |
//! |---|---|---|
//! | `n_returned differs` | pairs where the two return a different number of candidates — `min(keep, \|kept2\|)`, and `kept2` is the search's | ≤ 0.4 |
//! | `accepted only ours` / `accepted only theirs` | pairs one side accepts and the other does not | ≤ 0.25 |
//! | `different placement` | pairs both accept whose best candidates place the sherd more than a wall apart — a *different join*, not a pose difference | ≤ 0.25 |
//! | `best rot p50`, `best move p50` | how far apart the two best candidates are on the rest | ≤ 1°, ≤ 0.3 t |
//!
//! `best move` is the largest displacement the two poses give any vertex of the moving fragment,
//! in wall thicknesses — `tests/test_synthetic.py::pose_error`'s measure, not
//! [`pose_gap`](super::pose_gap)'s origin displacement. A sherd 450 units from the origin whose
//! two placements differ by 0.19° moves its own points by 0.02 t and the origin by 0.19 t, and it
//! is where the *sherd* goes that a candidate has to get right.
//!
//! The per-pair wall time is logged at `info`, which is where the note's cost table comes from.

use std::collections::BTreeMap;

use nalgebra::Matrix4;
use sherd_core::error::Result;
use sherd_core::executor::CPU;
use sherd_core::fragment::Fragment;
use sherd_core::matching::pair::{self, Candidate};
use sherd_core::matching::verify::{self, Scores};

use super::{ALL_PAIRS, Collection, Spread, pose_gap};
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// Whether two poses are the same matrix — entry by entry, not through
/// [`pose_gap`](super::pose_gap).
///
/// The rotation half of `pose_gap` goes through `acos`, which has a square-root singularity at
/// zero: a rotation matrix that is orthonormal to 1e-16 — which every pose that has been through
/// an ICP is — gives a trace 1e-16 away from 3 and an angle of 8e-7 degrees against itself. That
/// is fine as a *deviation* and useless as an *identity*, and this row needs an identity.
fn same_pose(a: &Matrix4<f64>, b: &Matrix4<f64>) -> bool {
    (0..4).all(|i| (0..4).all(|j| (a[(i, j)] - b[(i, j)]).abs() <= 1e-9))
}

/// R §5.7's `keep`: how many candidates a pair returns.
pub const KEEP: usize = 5;
/// D §10.2, native column: the rotation of the best candidate of a pair both sides accept.
///
/// A regression alarm rather than a parity claim (see the module documentation). The measured worst
/// median is **0.417°**, on `pot_B`; the six other sets run 0.033°–0.278°, and `pot_G` has no
/// figure because the reference accepts nothing there. Measured in the phase-1c verification
/// (`notes/2026-09-06-phase1c-verification.md` §7) and again after the phase-1c fixes
/// (`notes/2026-09-07-x-phase1c-findings.md` §7).
pub const NATIVE_ROTATION_DEG: f64 = 1.0;
/// How far apart the two best candidates may place the sherd, in wall thicknesses, before they
/// are counted as *different placements* rather than as a pose difference.
pub const SAME_PLACEMENT_T: f64 = 1.0;
/// The median displacement between the two best candidates of a pair both sides accept, in wall
/// thicknesses (a regression alarm; measured worst median **0.108 t**, on `pot_B`, with the other
/// sets at 0.007–0.060 t).
pub const NATIVE_MOVE_T: f64 = 0.3;
/// The share of a collection's pairs whose acceptance may differ, and the share of the pairs both
/// sides accept that may be placed differently (a regression alarm; measured worst 0.179).
pub const ACCEPT_SHARE: f64 = 0.25;
/// The share of a collection's pairs that may return a different *number* of candidates.
///
/// R §5.7 returns `min(keep, |kept2|)`, and `|kept2|` is what R §5.5's suppression left of a
/// search the two implementations do not share (PMC-6, PMC-9). A regression alarm; measured worst
/// 0.241, on `synthetic_20`, where many pairs leave only one or two poses above R §5.5's floor.
pub const RETURNED_SHARE: f64 = 0.4;

/// One candidate as `result.candidates.json` carries it.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct RefCandidate {
    /// The pose, `b → a`.
    #[serde(rename = "T")]
    pub transform: [[f64; 4]; 4],
    /// R §6.5's verdict.
    pub accepted: bool,
    /// R §5.7's ranking key, `seam · tight`, as the reference computed it.
    pub score: f64,
    /// Every score of R §6, under the reference's own keys.
    #[serde(flatten)]
    pub scores: Scores,
}

impl RefCandidate {
    /// The pose as a matrix.
    pub fn pose(&self) -> Matrix4<f64> {
        let mut out = Matrix4::zeros();
        for (i, row) in self.transform.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                out[(i, j)] = *value;
            }
        }
        out
    }
}

/// Runs the ranking (injected) or the whole of `match_pair` (native) and compares it.
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("candidates", mode);
    match mode {
        Mode::Injected => injected(collection, &mut report)?,
        Mode::Native => native(collection, &mut report)?,
    }
    Ok(report)
}

/// R §5.7 over the reference's own stage-2 candidates.
#[allow(clippy::too_many_lines, reason = "one flat list of comparisons per pair")]
fn injected(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let params = collection.manifest.collection.params;
    let mut geometry = super::pairs::GeometryCache::default();
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        let Some((fa, fb, used)) = super::hypotheses::sides(collection, &pair, report)? else {
            continue;
        };
        if !pair.has("result.candidates.json") {
            report.skip(&scope, "no result.candidates.json in the dump (level min)");
            continue;
        }
        let theirs: Vec<RefCandidate> = npy::read_json_as(pair.file("result.candidates.json"))?;
        if !pair.has("s2.T_frac2.npy") {
            // R §5.5 kept nothing, so R §5.7 returned nothing (or R §5.4's partial candidate,
            // which has no stage-2 pose of its own and is the `stage1_floor` path — off by
            // default, so it cannot appear in a dump written with the shipped parameters).
            report.push(Check::count(&scope, "n_returned", 0, theirs.len() as u64));
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
        let poses = npy::read_transforms(pair.file("s2.T_frac2.npy"))?;
        let kept2 = npy::read_indices(pair.file("nms2.kept.npy"))?;
        let stage1 = npy::read_f64(pair.file("s1.score.npy"))?;
        if kept2.len() != poses.len() || kept2.iter().any(|&k| (k as usize) >= stage1.len()) {
            report.skip(&scope, "nms2.kept does not describe s2.T_frac2 and s1.score");
            continue;
        }
        let sc = pair.scales()?;

        // R §6 at each of the reference's own poses, then R §5.7's sort and cut.
        let mut ours: Vec<(usize, Scores)> = poses
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut scores = verify::verify(&CPU, &sa, &sb, t, &sc, true, None);
                scores.brk = stage1[kept2[i] as usize];
                (i, scores)
            })
            .collect();
        let best1 = stage1.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        ours.sort_by(|x, y| {
            y.1.score().partial_cmp(&x.1.score()).unwrap_or(std::cmp::Ordering::Equal)
        });
        ours.truncate(KEEP);
        for (_, scores) in &mut ours {
            scores.brk_best = best1;
        }

        report.push(Check::count(&scope, "n_returned", ours.len() as u64, theirs.len() as u64));
        if ours.len() != theirs.len() {
            continue;
        }
        // Which stage-2 candidate each returned pose is, on the reference's side: the poses are
        // its own, so this is a lookup rather than a match.
        let mut wrong_order = 0_usize;
        let mut wrong_accept = 0_usize;
        let mut worst_brk_best = 0.0_f64;
        for ((i, mine), theirs) in ours.iter().zip(&theirs) {
            // The reference's returned pose *is* one of `poses`, so this is a lookup by identity.
            // It collects every match rather than the first, because two stage-1 candidates can
            // converge on the same pose and then either index is a correct answer.
            let theirs_pose = theirs.pose();
            let matches = poses.iter().enumerate().filter(|(_, p)| same_pose(p, &theirs_pose));
            if !matches.map(|(k, _)| k).any(|k| k == *i) {
                wrong_order += 1;
            }
            if verify::accept(mine, &params, &sc) != theirs.accepted {
                wrong_accept += 1;
            }
            worst_brk_best = worst_brk_best.max((mine.brk_best - theirs.scores.brk_best).abs());
        }
        report.push(Check::entries(&scope, "order", wrong_order, ours.len()));
        report.push(Check::entries(&scope, "accepted", wrong_accept, ours.len()));
        report.push(Check::absolute(&scope, "brk_best", worst_brk_best, 0.0, 0.02));
    }
    Ok(())
}

/// The whole of `match_pair`, from the port's own preprocessing.
fn native(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let params = collection.manifest.collection.params;
    let fragments = native_fragments(collection, report)?;
    let (mut rotations, mut displacements) = (Spread::default(), Spread::default());
    let (mut pairs, mut ours_only, mut theirs_only) = (0_usize, 0_usize, 0_usize);
    let (mut apart, mut returned) = (0_usize, 0_usize);
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        if !pair.has("result.candidates.json") {
            report.skip(&scope, "no result.candidates.json in the dump (level min)");
            continue;
        }
        let (Some(a), Some(b)) = (fragments.get(&pair.a), fragments.get(&pair.b)) else {
            report.skip(&scope, "a fragment of the pair was not preprocessed");
            continue;
        };
        let theirs: Vec<RefCandidate> = npy::read_json_as(pair.file("result.candidates.json"))?;
        let started = std::time::Instant::now();
        let ours: Vec<Candidate> = pair::match_pair(a, b, &params, KEEP);
        let seconds = started.elapsed().as_secs_f64();
        let (mine, reference) =
            (ours.iter().any(|c| c.accepted), theirs.iter().any(|c| c.accepted));
        pairs += 1;
        ours_only += usize::from(mine && !reference);
        theirs_only += usize::from(reference && !mine);
        tracing::info!(
            pair = %scope,
            seconds,
            candidates = ours.len(),
            accepted = mine,
            reference_accepted = reference,
            best = ours.first().map_or(0.0, Candidate::score),
            "match_pair"
        );
        returned += usize::from(ours.len() != theirs.len());
        if mine
            && reference
            && let (Some(mine), Some(theirs)) = (ours.first(), theirs.first())
        {
            let t = pair.scales().map_or_else(|_| f64::NAN, |sc| sc.t);
            let (angle, _) = pose_gap(&mine.transform, &theirs.pose(), t);
            let moved = displacement(b, &mine.transform, &theirs.pose()) / t;
            if moved > SAME_PLACEMENT_T {
                // The two sides accepted the same *pair* at two different placements of it, which
                // is not a pose disagreement: it happens where the pair is not a join at all and
                // each search found its own way of resting one sherd on the other.
                apart += 1;
            } else {
                rotations.push(angle);
                displacements.push(moved);
            }
        }
    }
    if pairs == 0 {
        // Nothing ran — no input directory, or a `min`-level dump. The collection rows below are
        // shares of the pairs that ran, and a share of nothing is not a measurement.
        return Ok(());
    }
    #[allow(clippy::cast_precision_loss, reason = "pair counts are far below 2^53")]
    let share = |count: usize| count as f64 / pairs as f64;
    report.push(Check::absolute(
        ALL_PAIRS,
        "n_returned differs",
        share(returned),
        0.0,
        RETURNED_SHARE,
    ));
    report.push(Check::absolute(
        ALL_PAIRS,
        "accepted only ours",
        share(ours_only),
        0.0,
        ACCEPT_SHARE,
    ));
    report.push(Check::absolute(
        ALL_PAIRS,
        "accepted only theirs",
        share(theirs_only),
        0.0,
        ACCEPT_SHARE,
    ));
    report.push(Check::absolute(ALL_PAIRS, "different placement", share(apart), 0.0, ACCEPT_SHARE));
    if !rotations.is_empty() {
        report.push(Check::absolute(
            ALL_PAIRS,
            "best rot p50",
            rotations.percentile(0.5),
            0.0,
            NATIVE_ROTATION_DEG,
        ));
        report.push(Check::absolute(
            ALL_PAIRS,
            "best move p50",
            displacements.percentile(0.5),
            0.0,
            NATIVE_MOVE_T,
        ));
    }
    Ok(())
}

/// How far the two poses move the fragment they place: the largest displacement either gives any
/// of `fragment`'s own vertices, in the same units the mesh is in.
///
/// This is `tests/test_synthetic.py::pose_error`'s second number and **not**
/// [`pose_gap`](super::pose_gap)'s. The latter reports the displacement of the *origin*, which is
/// the right conservative reading when two poses agree to 1e-9 and the wrong instrument here: a
/// fragment 450 units from the origin whose two placements differ by 0.19° moves its own points by
/// 0.17 units and the origin by 1.5. What a candidate has to get right is where the *sherd* goes.
///
/// Every twentieth vertex, which is what the synthetic test measures over.
fn displacement(fragment: &Fragment, ours: &Matrix4<f64>, theirs: &Matrix4<f64>) -> f64 {
    let mut worst = 0.0_f64;
    for v in fragment.mesh.v.iter().step_by(20) {
        let p = v.to_f64();
        let mut d = 0.0;
        for i in 0..3 {
            let a = ours[(i, 0)] * p[0] + ours[(i, 1)] * p[1] + ours[(i, 2)] * p[2] + ours[(i, 3)];
            let b = theirs[(i, 0)] * p[0]
                + theirs[(i, 1)] * p[1]
                + theirs[(i, 2)] * p[2]
                + theirs[(i, 3)];
            d += (a - b) * (a - b);
        }
        worst = worst.max(d.sqrt());
    }
    worst
}

/// Every fragment of the collection, preprocessed by the port itself (R §3).
///
/// Built once and shared by every pair: a fragment of a 55-pair collection would otherwise be
/// segmented ten times over.
fn native_fragments(
    collection: &Collection,
    report: &mut StageReport,
) -> Result<BTreeMap<String, Fragment>> {
    let mut out = BTreeMap::new();
    for fragment in &collection.fragments {
        let Some(source) = &fragment.source else {
            report.skip(&fragment.name, "no source file (pass --input DIR)");
            continue;
        };
        let started = std::time::Instant::now();
        let (fr, _) =
            Fragment::load_or_build(source, collection.target_faces, &fragment.name, None)?;
        tracing::info!(
            fragment = %fragment.name,
            seconds = started.elapsed().as_secs_f64(),
            "preprocessed"
        );
        out.insert(fragment.name.clone(), fr);
    }
    Ok(out)
}
