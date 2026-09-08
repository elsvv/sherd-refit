//! The stage-1 row: the breakline ICP ladder, its re-score and R §5.5's suppression
//! (R §5.4–5.5, D §10.2 row `stage 1`).
//!
//! # Injected
//!
//! Everything the ladder needs is in the dump. The initial poses are the reference's own kept
//! hypotheses (`nms1.kept` into the hypothesis set rebuilt from `hyp.ia/ib/pa/pb`, whose poses the
//! `hypotheses` row already pins to 1e-4° / 1e-5 t); the clouds are the reference's own `brk_P`
//! and `brk_ns` at the pair's `t`; the radii come from the reference's own `scales.json`. Fed
//! those, the port's ladder has nothing of its own in it but arithmetic, and what the rows measure
//! is the ICP.
//!
//! * **`T rot` / `T trans`** — the worst pose deviation over the pair's 250 candidates, against
//!   `s1.T`. D §10.2 allows 0.05° and 0.01 t.
//! * **`s1`** — the worst deviation of R §5.4's re-score, against `s1.score`, ±0.02. The score is
//!   a fraction of `|brk_sub|`, so a step of one probe point is 1/200 to 1/900 on these pairs and
//!   the row is between two and ten of them wide.
//! * **`fitness` / `rmse`** — the ICP's own view of both poses, computed by the port at its own
//!   pose and at the reference's, on the last rung's radius and clouds. The dump carries no
//!   fitness of its own (Open3D's `RegistrationResult` is not part of R §5.4's output), so this is
//!   the one comparison available in that currency: it says that where the two poses differ, they
//!   are equally good under the objective the ICP was minimising, and it would catch a port that
//!   landed on the reference's pose by luck while optimising something else.
//! * **`kept2` / `kept2 count`** — R §5.5's suppression, run on the reference's own poses, its own
//!   scores and its own walk order (`nms2.order`, the input D §10.1 grew for exactly this reason),
//!   compared exactly against `nms2.kept`.
//!
//! The per-pair rows are worst-case; the `(all pairs)` rows add the distribution of the same
//! deviations over every candidate of every pair (`p50`, `p90`, `p99`, `max` — see [`Spread`]),
//! because a worst case alone does not say whether it is one candidate or all of them.
//!
//! # Native
//!
//! D §10.2 gives no native column here, and the reason is the one the `coarse` and `nms` rows
//! give: the poses this stage refines are the coarse ranking's, the coarse ranking is a function
//! of sixty points drawn from a generator, and PMC-9 forbids the port drawing the reference's.
//! What a native run produces is compared at the `pair result` row instead.

use nalgebra::Matrix4;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use sherd_core::error::Result;
use sherd_core::executor::Engine;
use sherd_core::matching::hypotheses::{self, Hypotheses};
use sherd_core::matching::icp::{self, Numerics, Options};
use sherd_core::matching::ladder;
use sherd_core::matching::scales::Scales;
use sherd_core::spatial::kdtree::PointTree;

use super::pairs::RefClouds;
use super::{ALL_PAIRS, Collection, Spread, pose_gap};
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// D §10.2, injected column: the rotation of a stage-1 pose, in degrees.
pub const INJECTED_ROTATION_DEG: f64 = 0.05;
/// D §10.2, injected column: the translation of a stage-1 pose, in wall thicknesses.
pub const INJECTED_TRANSLATION_T: f64 = 0.01;
/// D §10.2, injected column: R §5.4's re-score `s1`.
pub const INJECTED_SCORE: f64 = 0.02;
/// The ICP's own fitness at the two poses, as a fraction of the source cloud.
pub const INJECTED_FITNESS: f64 = 1e-4;
/// The ICP's own inlier RMSE at the two poses, in wall thicknesses.
pub const INJECTED_RMSE_T: f64 = 1e-4;
/// The share of **one dump's** candidates whose ladder is allowed to be chaotic (task C2).
///
/// Measured worst 1.4e-3 — terracotta, 2 of its 1 459 candidates; 11 of 71 591 over all eight
/// fixture sets, and five of the seven sets have none at all.
pub const CHAOTIC_SHARE_ALL: f64 = 0.002;
/// The share of *one pair's* candidates whose ladder is allowed to be chaotic
/// (task C2; measured worst 0.0333 — 1 of 30 — over the eight fixture sets).
///
/// See [`determined`](super::determined): a candidate whose own answer moves when its initial pose
/// moves by one ULP cannot be compared against another implementation's, and this row is the alarm
/// that says how often that happens. It is not a parity claim — it is set at the measured worst
/// over the eight fixture sets plus headroom, in the shape of the PMC-6 rows of the `nms` stage.
///
/// The row that keeps it honest is `chaotic kept`: **every** candidate excused here must be one
/// the reference itself discarded at R §5.5. A chaotic pose that the reference went on to refine
/// would be a difference that reaches the pipeline's answer, and it fails exactly. Over the eight
/// sets, all 11 excused candidates were discarded by the reference's own R §5.5.
pub const CHAOTIC_SHARE: f64 = 0.06;

/// Runs R §5.4–5.5 for every pair of the dump and compares it.
#[allow(clippy::too_many_lines, reason = "one flat list of comparisons per pair")]
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("stage1", mode);
    let params = collection.manifest.collection.params;
    let mut normals = super::pairs::NormalCache::default();
    let (mut all_chaotic, mut all_candidates) = (0_usize, 0_usize);
    let mut spread = Spreads::default();
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        if mode == Mode::Native {
            report.skip(
                &scope,
                "D §10.2 has no native column here: the poses this stage refines come from the \
                 coarse ranking, which has no native column either (PMC-9)",
            );
            continue;
        }
        let Some((fa, fb, used)) = super::hypotheses::sides(collection, &pair, &mut report)? else {
            continue;
        };
        if !(pair.has("nms1.kept.npy") && pair.has("hyp.pa.npy")) {
            report.skip(&scope, "no nms1.kept in the dump (level min)");
            continue;
        }
        if !(pair.has("s1.T.npy") && pair.has("s1.score.npy")) {
            // R §5.3 kept nothing, so R §5.4 refined nothing and the reference returned before it
            // dumped a pose. There is no stage 1 here to compare, and that is the dump agreeing
            // with itself rather than a level.
            let empty = npy::read_indices(pair.file("nms1.kept.npy"))?.is_empty();
            report.skip(
                &scope,
                if empty {
                    "R §5.3 kept no hypothesis above the coarse floor: the pair has no candidate"
                } else {
                    "no s1.T in the dump (level min)"
                },
            );
            continue;
        }
        let theirs_hyp = pair.hypotheses()?;
        if !theirs_hyp.describes(&fa, &fb) {
            report.skip(&scope, "the dump's hypothesis indices do not describe its own breaklines");
            continue;
        }
        let Some((clouds_a, clouds_b)) =
            super::clouds(collection, &pair, &fa, &fb, &used, &params, &mut normals, &mut report)?
        else {
            continue;
        };

        let hyp = hypotheses::poses(
            &fa,
            &fb,
            &theirs_hyp.ia,
            &theirs_hyp.ib,
            &theirs_hyp.pa,
            &theirs_hyp.pb,
        );
        let kept = npy::read_indices(pair.file("nms1.kept.npy"))?;
        if kept.iter().any(|&h| (h as usize) >= hyp.len()) {
            report.skip(&scope, "nms1.kept does not index the hypothesis set");
            continue;
        }
        let theirs_pose = npy::read_transforms(pair.file("s1.T.npy"))?;
        let theirs_score = npy::read_f64(pair.file("s1.score.npy"))?;
        report.push(Check::count(&scope, "n_kept", kept.len() as u64, theirs_pose.len() as u64));
        if kept.len() != theirs_pose.len() || kept.len() != theirs_score.len() {
            report.skip(&scope, "s1.T and s1.score do not describe nms1.kept");
            continue;
        }
        let sc = pair.scales()?;
        // Read ahead: a candidate the port cannot reproduce is only excusable if the reference
        // itself threw it away, and `nms2.kept` is where it says so.
        let their_kept2 = if pair.has("nms2.kept.npy") {
            npy::read_indices(pair.file("nms2.kept.npy"))?
        } else {
            Vec::new()
        };

        // --- the ladder, candidate by candidate --------------------------------------------
        let target = clouds_a.brk_full.target();
        let rungs = ladder::stage1_rungs(&clouds_b.brk_sub.p, &target);
        let ours = refine(&clouds_a, &clouds_b, &rungs, &hyp, &kept, &sc, collection.icp);
        let (mut rotation, mut translation, mut rescore) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut fitness, mut rmse) = (0.0_f64, 0.0_f64);
        let (mut chaotic, mut kept_chaotic) = (0_usize, 0_usize);
        let last = Options::point_to_point(
            sc.icp_dist(ladder::STAGE1_RUNGS[ladder::STAGE1_RUNGS.len() - 1]),
            0,
        )
        .with(collection.icp);
        for (i, mine) in ours.iter().enumerate() {
            let (angle, distance) = pose_gap(&mine.transform, &theirs_pose[i], sc.t);
            let moved = (mine.score - theirs_score[i]).abs();
            let theirs = icp::register(&clouds_b.brk_sub.p, &target, &theirs_pose[i], &last);
            let (gap_fitness, gap_rmse) = (
                (mine.fitness - theirs.fitness).abs(),
                (mine.inlier_rmse - theirs.inlier_rmse).abs() / sc.t,
            );
            let failed = angle > INJECTED_ROTATION_DEG
                || distance > INJECTED_TRANSLATION_T
                || moved > INJECTED_SCORE
                || gap_fitness > INJECTED_FITNESS
                || gap_rmse > INJECTED_RMSE_T;
            if failed {
                let init = icp::homogeneous(&hyp.r[kept[i] as usize], &hyp.tau[kept[i] as usize]);
                if !super::determined(
                    &rungs,
                    &init,
                    &sc,
                    collection.icp,
                    INJECTED_ROTATION_DEG,
                    INJECTED_TRANSLATION_T,
                ) {
                    chaotic += 1;
                    #[allow(clippy::cast_possible_truncation, reason = "at most `stage1` poses")]
                    let index = i as u32;
                    if their_kept2.contains(&index) {
                        kept_chaotic += 1;
                    }
                    continue;
                }
            }
            rotation = rotation.max(angle);
            translation = translation.max(distance);
            rescore = rescore.max(moved);
            fitness = fitness.max(gap_fitness);
            rmse = rmse.max(gap_rmse);
            spread.rotation.push(angle);
            spread.translation.push(distance);
            spread.rescore.push(moved);
        }
        report.push(Check::absolute(&scope, "T rot", rotation, 0.0, INJECTED_ROTATION_DEG));
        report.push(Check::absolute(&scope, "T trans", translation, 0.0, INJECTED_TRANSLATION_T));
        report.push(Check::absolute(&scope, "s1", rescore, 0.0, INJECTED_SCORE));
        report.push(Check::absolute(&scope, "fitness", fitness, 0.0, INJECTED_FITNESS));
        report.push(Check::absolute(&scope, "rmse", rmse, 0.0, INJECTED_RMSE_T));
        #[allow(clippy::cast_precision_loss, reason = "candidate counts are at most `stage1`")]
        report.push(Check::absolute(
            &scope,
            "chaotic",
            chaotic as f64 / ours.len().max(1) as f64,
            0.0,
            CHAOTIC_SHARE,
        ));
        report.push(Check::entries(&scope, "chaotic kept", kept_chaotic, chaotic));
        all_chaotic += chaotic;
        all_candidates += ours.len();

        // --- R §5.5's suppression, on the reference's own poses, scores and walk order -------
        if !(pair.has("nms2.kept.npy") && pair.has("nms2.order.npy")) {
            report.skip(&scope, "no nms2.order in the dump: this dump predates task C1");
            continue;
        }
        let order = npy::read_indices(pair.file("nms2.order.npy"))?;
        if order.iter().chain(&their_kept2).any(|&i| (i as usize) >= theirs_pose.len()) {
            report.skip(&scope, "nms2.order or nms2.kept does not index the stage-1 poses");
            continue;
        }
        let kept2 = ladder::suppress_stage1(
            &theirs_pose,
            &theirs_score,
            &order,
            sc.nms,
            params.stage2 as usize,
        );
        report.push(Check::count(
            &scope,
            "kept2 count",
            kept2.len() as u64,
            their_kept2.len() as u64,
        ));
        let differing = if kept2.len() == their_kept2.len() {
            kept2.iter().zip(&their_kept2).filter(|(a, b)| a != b).count()
        } else {
            kept2.len().max(their_kept2.len())
        };
        report.push(Check::entries(&scope, "kept2", differing, their_kept2.len()));
    }
    if all_candidates > 0 {
        #[allow(clippy::cast_precision_loss, reason = "candidate counts are far below 2^53")]
        report.push(Check::absolute(
            ALL_PAIRS,
            "chaotic",
            all_chaotic as f64 / all_candidates as f64,
            0.0,
            CHAOTIC_SHARE_ALL,
        ));
    }
    spread.rotation.report(
        &mut report,
        ["T rot p50", "T rot p90", "T rot p99", "T rot max"],
        INJECTED_ROTATION_DEG,
    );
    spread.translation.report(
        &mut report,
        ["T trans p50", "T trans p90", "T trans p99", "T trans max"],
        INJECTED_TRANSLATION_T,
    );
    spread.rescore.report(&mut report, ["s1 p50", "s1 p90", "s1 p99", "s1 max"], INJECTED_SCORE);
    Ok(report)
}

/// R §5.4 for every kept hypothesis: the two breakline rungs, then the re-score.
///
/// The candidates are independent, so this is `rayon`'s to spread; the results are collected by
/// index and the summation inside each ICP is sequential, so the answer does not depend on the
/// thread count (D §7).
fn refine(
    a: &RefClouds,
    b: &RefClouds,
    rungs: &[ladder::Rung<'_>],
    hyp: &Hypotheses,
    kept: &[u32],
    sc: &Scales,
    numerics: Numerics,
) -> Vec<Refined> {
    let Some(tree) = PointTree::build(&a.brk_full.p) else { return Vec::new() };
    let scoring = sherd_core::matching::coarse::Target {
        points: &a.brk_full.p,
        normals: &a.brk_full.n,
        tree: &tree,
    };
    kept.par_iter()
        .map(|&h| {
            let init = icp::homogeneous(&hyp.r[h as usize], &hyp.tau[h as usize]);
            let climbed = ladder::climb(Engine::cpu(numerics), rungs, &init, sc);
            let last = climbed.last().copied();
            let transform = last.map_or(init, |r| r.transform);
            Refined {
                score: ladder::brk_score(
                    Engine::cpu(numerics),
                    &scoring,
                    &b.brk_sub.p,
                    &b.brk_sub.n,
                    &transform,
                    sc.stage1,
                ),
                transform,
                fitness: last.map_or(0.0, |r| r.fitness),
                inlier_rmse: last.map_or(0.0, |r| r.inlier_rmse),
            }
        })
        .collect()
}

/// One refined candidate: the pose, R §5.4's re-score and the ICP's own view of the pose.
struct Refined {
    transform: Matrix4<f64>,
    score: f64,
    fitness: f64,
    inlier_rmse: f64,
}

/// The three per-candidate deviations whose distribution the stage reports (task C2).
#[derive(Debug, Default)]
struct Spreads {
    /// The pose rotation, in degrees.
    rotation: Spread,
    /// The pose translation, in wall thicknesses.
    translation: Spread,
    /// R §5.4's re-score.
    rescore: Spread,
}
