//! The NMS stage: which poses go on to the breakline ICP (R §5.3, D §10.2's `stage 1` inputs).
//!
//! # Injected, and why the dump had to grow a file
//!
//! NMS is a greedy walk, so its **order is an input** — and the reference's order is
//! `np.argsort(cs)[::-1]`, an unstable quicksort over scores that are multiples of `1/60`.
//! Thousands of hypotheses tie at every level and which of them numpy puts first is an artefact of
//! its partitioning; PMC-6 lets the port sort stably instead, and the two walks then meet
//! different candidates in a different order. Comparing the kept sets under two different orders
//! measures numpy's sort, not the port.
//!
//! The reference therefore dumps `nms1.order` (task C1), and this stage runs the port's greedy
//! loop and duplicate test on the reference's own ranking. What is compared is then exact: the
//! same kept hypotheses, in the same order, or a defect.
//!
//! Three further rows say what the port's *own* order does with the same scores, because that is
//! what a native run will do and PMC-6 asks for it to be measured rather than assumed:
//!
//! * **`own order count`** — its kept list is as long as the reference's (both stop at `stage1`).
//! * **`own order kept`** — how much of the reference's kept set it reproduces. This is *not* a
//!   parity claim: ties make the difference, and the tolerance is a measurement of how large the
//!   tie effect is on these fixtures, recorded in `notes/2026-09-07-c1-hypotheses.md`.
//! * **`own order cover`** — the fraction of the reference's kept poses that are *not* within the
//!   suppression ball of any pose the port kept. This is the invariant that survives the tie:
//!   NMS returns a cover of the high-scoring poses, and two runs that disagree only on which
//!   member of a cluster to keep still cover each other.

use nalgebra::Matrix3;
use sherd_core::error::Result;
use sherd_core::matching::hypotheses::{self, Hypotheses};
use sherd_core::matching::nms::{self, ROT_TOL};
use sherd_core::matching::pair::{COARSE_FLOOR, COARSE_ORDER_LIMIT};

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// PMC-6, measured: how far the port's own kept count may be from the reference's.
///
/// The two counts are both `stage1` whenever the walk reaches it, and differ only on a pair whose
/// walk ends early — at the floor, or at the five-thousandth hypothesis — where how many poses
/// survive depends on which member of each tie came first.
pub const OWN_ORDER_COUNT: f64 = 0.05;
/// PMC-6, measured: the share of the reference's kept hypotheses the port's own tie-break may
/// miss. See the module documentation — this is a tie measurement, not a parity gate.
pub const OWN_ORDER_KEPT: f64 = 0.5;
/// PMC-6, measured: how far apart the two kept sets' scores may be at equal rank.
///
/// This is the row that says the two tie-breaks keep poses of the *same quality*: sort both kept
/// sets by score and compare rank by rank. A step of `1/60` is one probe point out of sixty.
pub const OWN_ORDER_SCORES: f64 = 3.0 / 60.0 + 1e-6;
/// PMC-6, measured: the share of the reference's kept poses that may lie outside every
/// suppression ball of the port's own kept set.
pub const OWN_ORDER_COVER: f64 = 0.5;

/// Runs R §5.3 for every pair of the dump and compares it.
#[allow(clippy::too_many_lines, reason = "one flat list of comparisons per pair")]
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("nms", mode);
    let params = collection.manifest.collection.params;
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        if mode == Mode::Native {
            report.skip(
                &scope,
                "D §10.2 has no native column here: the walk order is the coarse ranking, which \
                 has no native column either (PMC-9)",
            );
            continue;
        }
        let Some((fa, fb, _used)) = super::hypotheses::sides(collection, &pair, &mut report)?
        else {
            continue;
        };
        if !pair.has("nms1.kept.npy") || !pair.has("coarse.cs.npy") || !pair.has("hyp.pa.npy") {
            report.skip(&scope, "no nms1.kept in the dump (level min)");
            continue;
        }
        if !pair.has("nms1.order.npy") {
            report.skip(
                &scope,
                "no nms1.order in the dump: this dump predates task C1 and the walk order — an \
                 input of R §5.3 — cannot be injected; regenerate it",
            );
            continue;
        }
        let theirs_hyp = pair.hypotheses()?;
        let cs = npy::read_f64(pair.file("coarse.cs.npy"))?;
        let order = npy::read_indices(pair.file("nms1.order.npy"))?;
        let theirs = npy::read_indices(pair.file("nms1.kept.npy"))?;
        let sc = pair.scales()?;
        let hyp = hypotheses::poses(
            &fa,
            &fb,
            &theirs_hyp.ia,
            &theirs_hyp.ib,
            &theirs_hyp.pa,
            &theirs_hyp.pb,
        );
        if hyp.len() != cs.len() {
            report.skip(&scope, "coarse.cs does not describe the hypothesis set");
            continue;
        }
        let topk = params.stage1 as usize;

        // --- the reference's own ranking: the greedy loop and the duplicate test, exactly ------
        let ours = nms::nms(&order, &hyp.r, &hyp.tau, &cs, sc.nms, topk, COARSE_FLOOR);
        report.push(Check::count(&scope, "kept count", ours.len() as u64, theirs.len() as u64));
        let differing = if ours.len() == theirs.len() {
            ours.iter().zip(&theirs).filter(|(a, b)| a != b).count()
        } else {
            ours.len().max(theirs.len())
        };
        report.push(Check::entries(&scope, "kept", differing, theirs.len()));

        // --- and what the port's own tie-break does with the same scores (PMC-6) ---------------
        let mine = nms::nms(
            &nms::order_by_score(&cs, COARSE_ORDER_LIMIT),
            &hyp.r,
            &hyp.tau,
            &cs,
            sc.nms,
            topk,
            COARSE_FLOOR,
        );
        #[allow(clippy::cast_precision_loss, reason = "kept counts are at most `stage1`")]
        report.push(Check::relative(
            &scope,
            "own order count",
            mine.len() as f64,
            theirs.len() as f64,
            OWN_ORDER_COUNT,
        ));
        #[allow(clippy::cast_precision_loss, reason = "kept counts are at most `stage1`")]
        let denominator = theirs.len().max(1) as f64;
        let mut sorted = mine.clone();
        sorted.sort_unstable();
        #[allow(clippy::cast_precision_loss, reason = "kept counts are at most `stage1`")]
        let missed = theirs.iter().filter(|h| sorted.binary_search(h).is_err()).count() as f64;
        report.push(Check::absolute(
            &scope,
            "own order kept",
            missed / denominator,
            0.0,
            OWN_ORDER_KEPT,
        ));
        report.push(Check::absolute(
            &scope,
            "own order scores",
            rank_gap(&mine, &theirs, &cs),
            0.0,
            OWN_ORDER_SCORES,
        ));
        #[allow(clippy::cast_precision_loss, reason = "kept counts are at most `stage1`")]
        let uncovered =
            theirs.iter().filter(|&&h| !covered(&hyp, h as usize, &mine, sc.nms)).count() as f64;
        report.push(Check::absolute(
            &scope,
            "own order cover",
            uncovered / denominator,
            0.0,
            OWN_ORDER_COVER,
        ));
    }
    Ok(report)
}

/// True when hypothesis `h` is inside the suppression ball of one of `kept` — the same predicate
/// R §5.3 drops a duplicate with.
fn covered(hyp: &Hypotheses, h: usize, kept: &[u32], trans_tol: f64) -> bool {
    kept.iter().any(|&k| {
        let k = k as usize;
        (hyp.tau[h] - hyp.tau[k]).norm() < trans_tol && trace(&hyp.r[h], &hyp.r[k]) > ROT_TOL
    })
}

/// The worst score difference at equal rank between two kept sets — both sorted by score,
/// descending, and compared entry for entry over the shorter one.
fn rank_gap(mine: &[u32], theirs: &[u32], cs: &[f64]) -> f64 {
    let sorted = |kept: &[u32]| {
        let mut v: Vec<f64> = kept.iter().map(|&h| cs[h as usize]).collect();
        v.sort_unstable_by(|a, b| b.total_cmp(a));
        v
    };
    sorted(mine).iter().zip(&sorted(theirs)).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max)
}

/// `trace(Aᵀ B)`.
fn trace(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        sum += a[(0, i)] * b[(0, i)] + a[(1, i)] * b[(1, i)] + a[(2, i)] * b[(2, i)];
    }
    sum
}
