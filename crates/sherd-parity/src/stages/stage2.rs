//! The stage-2 row: the four point-to-plane rungs of R §5.6 (D §10.2 row `stage 2`, pose half).
//!
//! # Injected
//!
//! The candidates are the reference's own: `nms2.kept` names stage-1 poses and `s1.T` carries
//! them, so the ladder starts exactly where the reference's did, and it runs on the reference's
//! own `pc_reg` and `pc_frac` — the dump's `Pf`, `S`, `fp`, `sp` and `margin_idx` at the pair's
//! `t`, with the normals `FN[fp]` and `FN[sp[margin_idx]]` off the dump's own working mesh and
//! R §3.6's prefix split between the two halves of `pc_reg`.
//!
//! **Every rung is compared, not only the last.** The dump carries `s2.T_reg1`, `s2.T_reg2`,
//! `s2.T_frac1` and `s2.T_frac2`, and a ladder that arrives at the right pose by a different route
//! — a rung that converged early, a radius applied to the wrong cloud — is a different ladder
//! whatever its last pose says. Each row is the worst deviation over the pair's candidates, and
//! the `(all pairs)` rows add the distribution over every candidate of every rung ([`Spread`]).
//!
//! `fitness` and `rmse` are the ICP's own view of the port's final pose and of the reference's,
//! both computed here on `pc_frac` at the last rung's radius: the dump carries no fitness of its
//! own, and this is what says that two poses which differ are equally good under the objective
//! being minimised.
//!
//! The verification scores of R §6 — `tight`, `gap`, `seam`, `cont`, `pen`, `accepted` — are the
//! other half of D §10.2's stage-2 row and are not here: they need R §6, which is the step after
//! this one. `s2.scores.json` and `s2.accepted` are in the dump waiting for them.
//!
//! # Native
//!
//! None, for the reason the `stage 1` row gives.

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use sherd_core::error::Result;
use sherd_core::matching::icp::{self, Numerics, Options, Registration};
use sherd_core::matching::ladder;
use sherd_core::matching::scales::Scales;

use super::{ALL_PAIRS, Collection, Spread, pose_gap};
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// D §10.2, injected column: the rotation of a stage-2 pose, in degrees.
pub const INJECTED_ROTATION_DEG: f64 = 0.05;
/// D §10.2, injected column: the translation of a stage-2 pose, in wall thicknesses.
pub const INJECTED_TRANSLATION_T: f64 = 0.01;
/// The ICP's own fitness at the two poses, as a fraction of the source cloud.
pub const INJECTED_FITNESS: f64 = 1e-4;
/// The ICP's own inlier RMSE at the two poses, in wall thicknesses.
pub const INJECTED_RMSE_T: f64 = 1e-4;
/// The share of **one dump's** candidates whose ladder is allowed to be chaotic (task C2).
///
/// Measured worst 0.0333 — pot B, 12 of its 360 candidates; 37 of 2 239 over all eight fixture
/// sets.
pub const CHAOTIC_SHARE_ALL: f64 = 0.06;
/// The share of *one pair's* candidates whose ladder is allowed to be chaotic
/// (task C2; measured worst 0.3, which on ten candidates is three of them).
///
/// See [`determined`](super::determined) and the `stage1` row of the same name. Stage 2 registers
/// six-thousand-point surfaces, so its systems are over-determined wherever the two clouds *meet*;
/// what this row catches is the candidates where they do not. A stage-1 pose that is nowhere near
/// a join sends the ladder wandering — the measured cases end with an empty correspondence set, or
/// hit the thirty-iteration cap without converging — and after thirty unconverged iterations two
/// implementations of the same arithmetic are nowhere near each other.
///
/// The row that keeps it honest is `chaotic accepted`: **every** candidate excused here must be
/// one the reference itself refused (`s2.accepted`), and over the eight sets all 37 were. Every
/// one of them is refused with `tight = 0` and `seam = 0` — no contact and no seam at all, which
/// is what a wandering ladder produces — while the pairs that do have a join keep it: the one
/// accepted candidate of `Pot_B_Piece_01__07` (tight 0.312, seam 22.7) is not among the three the
/// probe refuses there.
pub const CHAOTIC_SHARE: f64 = 0.4;

/// The dump's name for each rung's pose, in ladder order.
const RUNG_FILES: [&str; 4] = ["s2.T_reg1", "s2.T_reg2", "s2.T_frac1", "s2.T_frac2"];
/// What the table calls each rung's two rows.
const RUNG_ROWS: [(&str, &str); 4] = [
    ("T_reg1 rot", "T_reg1 trans"),
    ("T_reg2 rot", "T_reg2 trans"),
    ("T_frac1 rot", "T_frac1 trans"),
    ("T_frac2 rot", "T_frac2 trans"),
];

/// Runs R §5.6's ICP chain for every candidate of every pair and compares it.
#[allow(clippy::too_many_lines, reason = "one flat list of comparisons per pair")]
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("stage2", mode);
    let params = collection.manifest.collection.params;
    let mut normals = super::pairs::NormalCache::default();
    let (mut all_chaotic, mut all_candidates) = (0_usize, 0_usize);
    let (mut rotations, mut translations) = (Spread::default(), Spread::default());
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        if mode == Mode::Native {
            report.skip(
                &scope,
                "D §10.2 has no native column here: the candidates come from the coarse ranking, \
                 which has no native column either (PMC-9)",
            );
            continue;
        }
        let Some((fa, fb, used)) = super::hypotheses::sides(collection, &pair, &mut report)? else {
            continue;
        };
        if !(pair.has("s1.T.npy") && pair.has("nms2.kept.npy")) {
            // A pair whose coarse suppression kept nothing never reached R §5.4, so the reference
            // dumped no pose for R §5.6 to start from. That is the dump agreeing with itself.
            let empty = pair.has("nms1.kept.npy")
                && npy::read_indices(pair.file("nms1.kept.npy"))?.is_empty();
            report.skip(
                &scope,
                if empty {
                    "R §5.3 kept no hypothesis above the coarse floor: the pair has no candidate"
                } else {
                    "no s1.T or nms2.kept in the dump (level min)"
                },
            );
            continue;
        }
        if !pair.has("s2.T_frac2.npy") {
            // R §5.5 kept nothing, so R §5.6 refined nothing: the reference dumped no rung. The
            // dump is agreeing with itself, not leaving something out.
            let empty = npy::read_indices(pair.file("nms2.kept.npy"))?.is_empty();
            report.skip(
                &scope,
                if empty {
                    "R §5.5 kept no candidate: every stage-1 score is below the floor"
                } else {
                    "no s2.T_* in the dump (level min)"
                },
            );
            continue;
        }
        let Some((clouds_a, clouds_b)) =
            super::clouds(collection, &pair, &fa, &fb, &used, &params, &mut normals, &mut report)?
        else {
            continue;
        };
        let stage1 = npy::read_transforms(pair.file("s1.T.npy"))?;
        let kept2 = npy::read_indices(pair.file("nms2.kept.npy"))?;
        if kept2.iter().any(|&k| (k as usize) >= stage1.len()) {
            report.skip(&scope, "nms2.kept does not index the stage-1 poses");
            continue;
        }
        let mut theirs = Vec::with_capacity(RUNG_FILES.len());
        for name in RUNG_FILES {
            let file = pair.file(&format!("{name}.npy"));
            if !file.is_file() {
                report.skip(&scope, format!("no {name} in the dump"));
                break;
            }
            theirs.push(npy::read_transforms(file)?);
        }
        if theirs.len() != RUNG_FILES.len() {
            continue;
        }
        report.push(Check::count(
            &scope,
            "n_candidates",
            kept2.len() as u64,
            theirs[0].len() as u64,
        ));
        if theirs.iter().any(|rung| rung.len() != kept2.len()) {
            report.skip(&scope, "the s2.T_* stacks do not describe nms2.kept");
            continue;
        }
        if clouds_a.reg.is_empty() || clouds_b.reg.is_empty() {
            report.skip(&scope, "a registration cloud of the dump is empty");
            continue;
        }
        let sc = pair.scales()?;
        let accepted = if pair.has("s2.accepted.npy") {
            npy::read_bool(pair.file("s2.accepted.npy"))?
        } else {
            Vec::new()
        };

        // --- the four rungs, candidate by candidate -----------------------------------------
        let reg_target = clouds_a.reg.target();
        let frac_target = clouds_a.frac.target();
        let rungs =
            ladder::stage2_rungs(&clouds_b.reg.p, &reg_target, &clouds_b.frac.p, &frac_target);
        let ours = refine(&rungs, &stage1, &kept2, &sc, collection.icp);
        let mut worst = [(0.0_f64, 0.0_f64); 4];
        let (mut fitness, mut rmse) = (0.0_f64, 0.0_f64);
        let (mut chaotic, mut accepted_chaotic) = (0_usize, 0_usize);
        let last = Options::point_to_plane(
            sc.icp_dist(ladder::STAGE2_FRAC_RUNGS[ladder::STAGE2_FRAC_RUNGS.len() - 1]),
            0,
        )
        .with(collection.icp);
        for (i, climbed) in ours.iter().enumerate() {
            let mut gaps = [(0.0_f64, 0.0_f64); 4];
            for ((rung, theirs), out) in climbed.iter().zip(&theirs).zip(&mut gaps) {
                *out = pose_gap(&rung.transform, &theirs[i], sc.t);
            }
            let Some(mine) = climbed.last() else { continue };
            let reference = icp::register(&clouds_b.frac.p, &frac_target, &theirs[3][i], &last);
            let (gap_fitness, gap_rmse) = (
                (mine.fitness - reference.fitness).abs(),
                (mine.inlier_rmse - reference.inlier_rmse).abs() / sc.t,
            );
            let failed = gaps.iter().any(|&(angle, distance)| {
                angle > INJECTED_ROTATION_DEG || distance > INJECTED_TRANSLATION_T
            }) || gap_fitness > INJECTED_FITNESS
                || gap_rmse > INJECTED_RMSE_T;
            if failed
                && !super::determined(
                    &rungs,
                    &stage1[kept2[i] as usize],
                    &sc,
                    collection.icp,
                    INJECTED_ROTATION_DEG,
                    INJECTED_TRANSLATION_T,
                )
            {
                chaotic += 1;
                if accepted.get(i).copied().unwrap_or(false) {
                    accepted_chaotic += 1;
                }
                continue;
            }
            for (out, &(angle, distance)) in worst.iter_mut().zip(&gaps) {
                out.0 = out.0.max(angle);
                out.1 = out.1.max(distance);
                rotations.push(angle);
                translations.push(distance);
            }
            fitness = fitness.max(gap_fitness);
            rmse = rmse.max(gap_rmse);
        }
        for (&(rotation_row, translation_row), &(rotation, translation)) in
            RUNG_ROWS.iter().zip(&worst)
        {
            report.push(Check::absolute(
                &scope,
                rotation_row,
                rotation,
                0.0,
                INJECTED_ROTATION_DEG,
            ));
            report.push(Check::absolute(
                &scope,
                translation_row,
                translation,
                0.0,
                INJECTED_TRANSLATION_T,
            ));
        }
        report.push(Check::absolute(&scope, "fitness", fitness, 0.0, INJECTED_FITNESS));
        report.push(Check::absolute(&scope, "rmse", rmse, 0.0, INJECTED_RMSE_T));
        #[allow(clippy::cast_precision_loss, reason = "candidate counts are at most `stage2`")]
        report.push(Check::absolute(
            &scope,
            "chaotic",
            chaotic as f64 / ours.len().max(1) as f64,
            0.0,
            CHAOTIC_SHARE,
        ));
        report.push(Check::entries(&scope, "chaotic accepted", accepted_chaotic, chaotic));
        all_chaotic += chaotic;
        all_candidates += ours.len();
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
    // Pooled over the four rungs as well as over the candidates: a rung that agreed on `T_reg1`
    // and drifted by `T_frac2` is the shape this distribution is here to show.
    rotations.report(
        &mut report,
        ["T rot p50", "T rot p90", "T rot p99", "T rot max"],
        INJECTED_ROTATION_DEG,
    );
    translations.report(
        &mut report,
        ["T trans p50", "T trans p90", "T trans p99", "T trans max"],
        INJECTED_TRANSLATION_T,
    );
    Ok(report)
}

/// R §5.6's four rungs for every candidate, one [`Registration`] per rung.
fn refine(
    rungs: &[ladder::Rung<'_>],
    stage1: &[nalgebra::Matrix4<f64>],
    kept2: &[u32],
    sc: &Scales,
    numerics: Numerics,
) -> Vec<Vec<Registration>> {
    kept2.par_iter().map(|&k| ladder::climb(rungs, &stage1[k as usize], sc, numerics)).collect()
}
