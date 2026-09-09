//! Roadmap step 7 (audit §D.1, D §12 row 7): everything a confidence tier could be built on,
//! measured on every accepted candidate before any threshold is chosen.
//!
//! The audit's rule for step 8 is that "a tier is defined by evidence beyond the five scores", and
//! its rule for this step is that the evidence has to be *measured* first — "thresholds are chosen
//! on the measurement table, not guessed". This module is that measurement and nothing else. It
//! runs after R §8's assembly, reads what the run already computed, decides nothing, and writes one
//! JSON file; with `--measure` absent it does not run at all and no output of a run moves.
//!
//! # The six quantities, and what each one asks
//!
//! | quantity | the question | why a false join might fail it |
//! |---|---|---|
//! | [`CandidateMeasure::margin`] | how far ahead of the pair's **second placement** is this one? | a wrong-pose join is a slide along the seam with a near-equal score at the true pose, so its margin is near 1 |
//! | [`CandidateMeasure::determined_deg`] | is the pose a function of its input? | C2 §5: a chaotic candidate moves 25–177° under a one-ULP change and no implementation reproduces it |
//! | [`CandidateMeasure::resamples`] | do the scores survive two independent redraws of `Pf`, `S` and the margin? | R §13's own spread is a redraw: a candidate that only just cleared `min_tight` clears it on one draw in three |
//! | [`CandidateMeasure::slide_t`] | pushed half a wall along the seam, does the fit come back? | a seam that fits anywhere along its length does not come back, and that *is* the wrong-pose join |
//! | [`CandidateMeasure::support`] | how many independent accepted joins agree with this placement? | a coincidence has no second path to it; a true join in an assembled group usually has one |
//! | [`CandidateMeasure::used`] | did R §8 take it? | the join a conservator sees is a used join, and only used joins carry `evaluate.py`'s class today |
//!
//! # Where each probe restarts, and why it is that and not something else
//!
//! Two of the probes re-climb R §5.6's ladder, and both restart from the candidate's **own final
//! pose** rather than from the stage-1 pose it grew out of. That is deliberate:
//!
//! * the stage-1 pose is not on the candidate — R §5.7 keeps the pose, the scores and the verdict —
//!   so a probe that wanted it would have to re-run the whole of R §5.4 for every pair;
//! * the audit words the determinedness probe as *"a candidate whose **last two rungs** move under
//!   one-ULP perturbations is not confirmed"*, which is exactly the two `pc_frac` rungs
//!   ([`ladder::stage2_rungs`](crate::matching::ladder::stage2_rungs)`[2..]`) climbed from the pose
//!   that reached them;
//! * a tier computed inside `match_pair` in step 8 will have the stage-1 pose in hand and may use
//!   the four-rung form. The two are not the same probe, and this note says which one the table
//!   below was measured with so that step 8 does not quietly change the definition and keep the
//!   thresholds.
//!
//! `sherd_parity::stages::determined` and `sherd_gpu::crosscheck::determined_batch` are the same
//! twelve one-ULP neighbours; neither is reused here because both live above this crate and both
//! answer a *parity* question — "may this candidate's row be excused" — against a fixed tolerance,
//! where this one reports the deviation and lets the note choose.
//!
//! # Cost
//!
//! Per pair that has an accepted candidate: one [`Pair::build`], one
//! [`SurfaceLadder`](crate::matching::pair::SurfaceLadder), thirteen two-rung climbs and two
//! one-rung climbs over the accepted candidates as a batch, and two R §6 verifications each on the
//! two resampled collections. The resampling is done once for the whole collection rather than per
//! pair, and a cloned [`Fragment`] shares its two BVHs through their `Arc`s, so the redraw costs
//! R §3.5 and not R §3.3 or R §6.1's trees.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::executor::Engine;
use crate::fragment::Fragment;
use crate::matching::pair::Candidate;
use crate::params::Params;
use crate::tiers::{self, Probes};
use crate::types::FragId;

pub use crate::tiers::{RESAMPLE_OFFSETS, SAME_PLACEMENT_T, SLIDE_BACK_T, SLIDE_T, ScoreRow};

/// One accepted candidate, with everything a tier could read about it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateMeasure {
    /// Index into the run's own candidate list, so a row can be traced back to `report.json`.
    pub index: usize,
    /// A's name.
    pub a: String,
    /// B's name.
    pub b: String,
    /// The pair's `t_pair` (R §4.2) — the unit every distance below is in.
    pub t: f64,
    /// `T`, row-major: `p_A = T · p_B`. `evaluate.py`'s class is computed from this and the
    /// ground truth, so a candidate that R §8 never used still gets one.
    pub transform: [[f64; 4]; 4],
    /// R §6's scores at that pose.
    pub scores: ScoreRow,
    /// R §5.4's own re-score of the pose the ladder started from, and the pair's best.
    pub brk: f64,
    /// The best stage-1 score of the whole pair.
    pub brk_best: f64,
    /// This candidate's rank inside its pair's returned list (0 is R §5.7's best).
    pub rank: usize,
    /// How many candidates the pair returned.
    pub of_pair: usize,
    /// How many **distinct placements** those candidates make, clustered at
    /// [`SAME_PLACEMENT_T`](crate::tiers::SAME_PLACEMENT_T).
    ///
    /// Usually one: several stage-1 poses of a pair routinely converge on the same fit, and a pair
    /// whose kept list holds one placement has nothing for [`CandidateMeasure::margin`] to be
    /// ahead of. That is why the margin is so often unbounded, and it is worth its own column.
    pub placements: usize,
    /// `seam · tight` of the best candidate of this pair that places B more than one wall away,
    /// or `None` when the pair's kept list holds only one placement.
    pub rival_score: Option<f64>,
    /// How far that rival places B, in `t`.
    pub rival_moved_t: Option<f64>,
    /// `score / rival_score`; `None` when there is no rival, or when the rival scores zero — both
    /// of which mean "no second placement to be ahead of" and are read as an unbounded margin.
    pub margin: Option<f64>,
    /// Worst rotation, in degrees, over the twelve one-ULP neighbours of this pose re-climbed
    /// through R §5.6's last two rungs.
    pub determined_deg: Option<f64>,
    /// The same in translation at the fracture cloud's centroid, in `t`.
    pub determined_t: Option<f64>,
    /// How far the pose comes back after being pushed ±[`SLIDE_T`](crate::tiers::SLIDE_T) along
    /// the seam and given R §5.6's ladder again, in `t` — the worse of the two directions.
    pub slide_t: Option<f64>,
    /// R §6 again at this very pose on two independent redraws of `Pf`, `S` and the margin.
    pub resamples: Vec<ScoreRow>,
    /// Independent accepted joins that agree with this placement.
    pub support: u32,
    /// Whether R §8's assembly took this candidate as a join.
    pub used: bool,
    /// Whether this is the candidate R §8 would consider for the pair (its best accepted one).
    pub best_of_pair: bool,
}

/// The file `--measure` writes: one run, one seed, every accepted candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasureReport {
    /// The collection's names, in R §2's order.
    pub names: Vec<String>,
    /// The run's seed (R §10).
    pub seed: u64,
    /// The collection's median wall thickness.
    pub thickness: f64,
    /// Candidates the run produced, accepted or not.
    pub candidates: usize,
    /// Of those, how many R §6.5 accepted.
    pub accepted: usize,
    /// The joins R §8 used, as name pairs.
    pub used: Vec<[String; 2]>,
    /// The two seeds the stability redraws ran at.
    pub resample_seeds: [u64; 2],
    /// One row per accepted candidate.
    pub rows: Vec<CandidateMeasure>,
}

/// Measures every accepted candidate of a finished run.
///
/// `used` is R §8's `Assembly::used` — indices into `candidates` — and `seed` is `Params::seed`,
/// which the redraws are offset from. `ready` is the tier pass's own answer when the run made
/// one ([`crate::tiers::probe`]): the two passes ask for exactly the same numbers, so a run with
/// both `--tiers on` and `--measure` climbs the ladders once and not twice.
pub fn measure(
    engine: Engine<'_>,
    fragments: &[Fragment],
    candidates: &[Candidate],
    used: &[usize],
    params: &Params,
    thickness: f64,
    ready: Option<&[Option<Probes>]>,
) -> MeasureReport {
    let started = std::time::Instant::now();
    let names: Vec<String> = fragments.iter().map(|f| f.name.clone()).collect();
    let used_set: BTreeSet<usize> = used.iter().copied().collect();

    let mut by_pair: BTreeMap<(FragId, FragId), Vec<usize>> = BTreeMap::new();
    for (i, c) in candidates.iter().enumerate() {
        by_pair.entry((c.a, c.b)).or_default().push(i);
    }
    let best: BTreeMap<(FragId, FragId), usize> = by_pair
        .iter()
        .filter_map(|(&key, list)| tiers::best_of(candidates, list).map(|i| (key, i)))
        .collect();

    let computed;
    let probes: &[Option<Probes>] = if let Some(ready) = ready {
        ready
    } else {
        computed = tiers::probe(engine, fragments, candidates, params);
        &computed
    };

    let rows: Vec<CandidateMeasure> = candidates
        .iter()
        .enumerate()
        .zip(probes)
        .filter_map(|((i, c), probe)| {
            let p = probe.as_ref()?;
            let list = &by_pair[&(c.a, c.b)];
            Some(CandidateMeasure {
                index: i,
                a: names[c.a as usize].clone(),
                b: names[c.b as usize].clone(),
                t: p.t,
                transform: std::array::from_fn(|r| std::array::from_fn(|q| c.transform[(r, q)])),
                scores: ScoreRow::of(&c.scores, c.accepted),
                brk: c.scores.brk,
                brk_best: c.scores.brk_best,
                rank: list.iter().position(|&j| j == i).unwrap_or(0),
                of_pair: list.len(),
                placements: p.placements,
                rival_score: p.rival_score,
                rival_moved_t: p.rival_moved_t,
                margin: p.margin,
                determined_deg: p.determined_deg,
                determined_t: p.determined_t,
                slide_t: p.slide_t,
                resamples: p.resamples.clone(),
                support: p.support,
                used: used_set.contains(&i),
                best_of_pair: best.get(&(c.a, c.b)) == Some(&i),
            })
        })
        .collect();

    tracing::info!(rows = rows.len(), seconds = started.elapsed().as_secs_f64(), "measure");
    MeasureReport {
        seed: params.seed,
        thickness,
        candidates: candidates.len(),
        accepted: candidates.iter().filter(|c| c.accepted).count(),
        used: used
            .iter()
            .map(|&i| {
                [names[candidates[i].a as usize].clone(), names[candidates[i].b as usize].clone()]
            })
            .collect(),
        resample_seeds: RESAMPLE_OFFSETS.map(|o| params.seed.wrapping_add(o)),
        names,
        rows,
    }
}
