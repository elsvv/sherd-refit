//! Roadmap item 3, step 8 (audit §D.1, D §12 row 8): the confidence tier of a candidate, and the
//! evidence it is decided on.
//!
//! R §6.5 answers one question — *do these two fracture surfaces fit?* — with five thresholds, and
//! the audit's finding is that the answer is not enough to place a sherd in a museum: pot_C's
//! decisions flip on the seed, pot_G's every accepted candidate is a slide along the break, and a
//! conservator reading `report.md` cannot tell those from the terracotta's two real joins. A tier
//! is the second question: *is this fit the only place the sherd can go?* Audit §D.1 words it as
//! **"a tier is defined by evidence beyond the five scores"** and names four kinds of evidence —
//! the margin to the pair's second placement, stability under resampling, determinedness, and the
//! slide along the seam — to which task M1's measurement adds a fifth, audit §D.2's support count,
//! for the reason §D.2 could not know: *nothing else reaches zero false joins*.
//!
//! # What this module is, and what task M1 settled about it
//!
//! Everything below was measured before it was written
//! (`docs/superpowers/notes/2026-09-09-m1-measure.md`): 2 916 accepted candidates of the quality
//! gate's own forty runs, 812 of them the best of their pair, classified by `tools/evaluate.py`'s
//! rule applied to each candidate's **own** pose. The probes here are the probes that table was
//! measured with, line for line — [`measure`](crate::measure) and this module share them so that
//! the thresholds cannot quietly come to mean something else than they were chosen on.
//!
//! | evidence | AUC, correct vs false | what M1 decided |
//! |---|---:|---|
//! | `gap` | 0.989 | the best single number there is; the tier's own limit is 0.015 t, not R §6.5's 0.02–0.03 |
//! | `tight` | 0.967 | 0.35, audit §D.1's own proposal |
//! | `slide_t` | 0.935 | kept at 0.1 t, restarted on **the whole of R §5.6** and not on its last rung |
//! | `determined_deg` | 0.879 | **reported, never gated**: the whole range is 1.6e-14…4.1e-7 degrees |
//! | `seam` | 0.878 | 5 t |
//! | `cont_n` | 0.843 (`cont`) / 0.606 | 0.90, kept because R §6.5 already explains it, not because it works |
//! | `margin` | 0.674 over all rows, **0.921** over the rows that have a rival | read *strictly*: no second placement is a failed test, not an infinite margin |
//! | `support` | 0.635 | a one-sided certificate: **every one** of the 466 false joins has support 0 |
//!
//! # The tier
//!
//! ```text
//! Confirmed = tight  >= 0.35
//!         AND gap    <= 0.015 t
//!         AND seam   >= 5 t
//!         AND cont_n >= 0.90
//!         AND pen    <= 0
//!         AND slide  <= 0.1 t
//!         AND ( support >= 1  OR  margin >= 2 over a second placement that scores )
//! Probable  = R §6.5 accepted it and the line above did not confirm it
//! Rejected  = R §6.5 refused it, with the reason it has today
//! ```
//!
//! **136 confirmed correct joins and 0 false ones** over the eight development sets at seeds 0–4 —
//! 17.7 % of the 770 ground-truth adjacent pairs those forty runs could have found. The worst false
//! join over all forty runs sits **37 %** outside a boundary (`mixed_ABG` seed 1,
//! `Pot_A_Piece_08`–`Pot_B_Piece_07`, gap 0.0206 t against the limit of 0.015 t), which is what
//! makes these thresholds rather than a fit: the search's own best zero-false rule, `gap ≤ 0.0064 t`
//! alone, confirms five more and has its nearest false join 0.5 % of a boundary away.
//!
//! # Why the disjunction, and why the support count is here at all
//!
//! Audit §E puts the support count in step 10 and §D.1 does not mention it. M1 searched
//! 4 478 976 conjunctions of the five scores, the margin, the slide, determinedness and the
//! redraws and **none of them reaches zero false joins**: the survivors are one family of
//! wrong-pose joins on genuinely adjacent pairs (pot_H 02-04, 02-11, 03-04, 03-07, 09-11, pot_A
//! 02-05) whose two fracture surfaces really do match and whose pose is a slide along the break.
//! They are indistinguishable from a true join on every quantity R §6 measures, they come back
//! from the slide probe because the slid pose is itself a local optimum, they are determined to
//! 1e-13 degrees, and their pairs produce one placement so the margin has nothing to say. What
//! refuses them is that no third fragment agrees with where they put the sherd. The two arms are
//! the two independent ways a placement can be distinguished — a second path through the
//! collection agrees with it, or the pair's own search found another placement and this one beat
//! it — and there is no reason both should have to hold.
//!
//! # The off switch
//!
//! [`Params::tiers`](crate::params::Params::tiers) is `None` by default, and with it `None`
//! nothing here runs: [`crate::pipeline::run`] does not probe, R §8 assembles from R §6.5's
//! accepted joins as it always has, and every output is the byte it was before this module
//! existed. The CLI's `run` turns it on (`--tiers off` turns it back off); `bench` and the parity
//! harness leave it off, because a tier pass is not part of the arithmetic either of them measures.

use std::collections::BTreeMap;

use nalgebra::Matrix4;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::assembly::consistency::{agrees, rotation_angle_deg};
use crate::executor::Engine;
use crate::fragment::Fragment;
use crate::matching::coarse::NORMAL_AGREE;
use crate::matching::icp::{self, pose_gap};
use crate::matching::ladder;
use crate::matching::pair::{Candidate, Pair, SurfaceLadder};
use crate::matching::scales::Scales;
use crate::matching::verify::{self, Scores, Surfaces, over, pose_inverse, under};
use crate::params::Params;
use crate::spatial::kdtree::PointTree;
use crate::types::FragId;

/// How far apart two candidates must place the sherd before they count as **different
/// placements** rather than as two readings of one, in wall thicknesses.
///
/// One wall, which is `sherd_parity::stages::candidates::SAME_PLACEMENT_T` — the same number the
/// parity harness uses to tell "the two sides put the sherd somewhere else" from "the two sides
/// disagree about the pose", and it is measured the same way (see [`placement_gap`]).
pub const SAME_PLACEMENT_T: f64 = 1.0;

/// What the two stability redraws add to the run's own seed.
///
/// Far from R §13's own 0–4 so that a redraw can never coincide with another seed of the same
/// sweep, and fixed so that a tier is reproducible.
pub const RESAMPLE_OFFSETS: [u64; 2] = [1_000_000, 2_000_000];

/// How far the slide probe pushes the placed sherd along the seam before restarting, in `t`
/// (audit §D.1: "restart the last fracture rung from ±0.5 t along the breakline tangent").
pub const SLIDE_T: f64 = 0.5;

/// How close the slid pose must come back for the candidate to be slide-stable, in `t` — audit
/// §D.1's own number, and [`Thresholds::default`]'s `max_slide_t`.
pub const SLIDE_BACK_T: f64 = 0.1;

/// Every score of R §6 a tier could read, with the pair's own limits (a flattened [`Scores`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreRow {
    /// `min(tightA, tightB)`.
    pub tight: f64,
    /// `max(gapA, gapB)`, in `t`.
    pub gap: f64,
    /// The pair's own gap limit in `t`, which R §6.5 compares `gap` against.
    pub gap_limit: f64,
    /// `min(contactA, contactB)`, in `t²`.
    pub contact: f64,
    /// Shared seam length, in `t`.
    pub seam: f64,
    /// Median shell step across the seam, in `t`.
    pub cont: f64,
    /// Median shell-normal agreement across the seam.
    pub cont_n: f64,
    /// Penetrating surface fraction.
    pub pen: f64,
    /// Deepest excursion, in `t`.
    pub pen_depth: f64,
    /// Set when a fragment is not watertight and R §6.4 could not run — `pen` is then `0` because
    /// the question was refused, not because nothing penetrates, and a `pen ≤ 0` veto passes it
    /// for free.
    pub pen_unavailable: bool,
    /// R §5.7's ranking key, `seam · tight`.
    pub score: f64,
    /// R §6.5's verdict at these scores.
    pub accepted: bool,
}

impl ScoreRow {
    /// The row of one [`Scores`], with R §6.5's verdict beside it.
    ///
    /// The verdict is passed in rather than recomputed: on a candidate's own scores it is
    /// [`Candidate::accepted`], which the run already decided, and on a redraw it is
    /// [`verify::accept`] at that pair's own [`Scales`] — the two differ in where the resolution
    /// floor of R §1.2 binds, and a row that recomputed it from a unit scale would report a
    /// verdict no run ever made.
    pub fn of(s: &Scores, accepted: bool) -> Self {
        Self {
            tight: s.tight,
            gap: s.gap,
            gap_limit: s.gap_limit,
            contact: s.contact,
            seam: s.seam,
            cont: s.cont,
            cont_n: s.cont_n,
            pen: s.pen,
            pen_depth: s.pen_depth,
            pen_unavailable: s.pen_unavailable,
            score: s.score(),
            accepted,
        }
    }
}

/// The three probes of audit §D.1, the margin, and audit §D.2's support count, for one accepted
/// candidate.
///
/// Every field is reported whether or not the tier reads it: `determined_deg` is in the report
/// and out of the rule (M1 §5.4), and the resamples are in the report and out of the rule
/// (M1 §5.8) — they cost 136 → 130 confirmed joins and remove no false one.
#[derive(Clone, Debug, PartialEq)]
pub struct Probes {
    /// The pair's `t_pair` (R §4.2) — the unit every distance here is in.
    pub t: f64,
    /// How many **distinct placements** the pair's kept list makes, clustered at
    /// [`SAME_PLACEMENT_T`].
    ///
    /// Usually one: several stage-1 poses of a pair routinely converge on the same fit, and a pair
    /// whose kept list holds one placement has nothing for [`Probes::margin`] to be ahead of. That
    /// is the case on 270 of M1's 342 correct joins and 245 of its 466 false ones, and it is why
    /// the tier needs the support count as a second arm.
    pub placements: usize,
    /// `seam · tight` of the best candidate of this pair that places B more than
    /// [`SAME_PLACEMENT_T`] away, or `None` when the pair's kept list holds only one placement.
    pub rival_score: Option<f64>,
    /// How far that rival places B, in `t`.
    pub rival_moved_t: Option<f64>,
    /// `score / rival_score`, **strictly**: `None` when there is no second placement or when it
    /// scores zero, both of which are a failed margin test and not an unbounded one.
    ///
    /// Read the other way round the test is not merely weaker, it is inert: M1 measured that
    /// adding a loose `margin ≥ 2` to the rest of the tier removes neither a correct join nor a
    /// false one.
    pub margin: Option<f64>,
    /// Worst rotation, in degrees, over the twelve one-ULP neighbours of this pose re-climbed
    /// through R §5.6's last two rungs.
    pub determined_deg: Option<f64>,
    /// The same in translation at the fracture cloud's centroid, in `t`.
    pub determined_t: Option<f64>,
    /// How far the pose comes back after being pushed ±[`SLIDE_T`] along the seam and given
    /// R §5.6's ladder again, in `t` — the worse of the two directions. `None` when the candidate
    /// has no shared seam to slide along, which is a failed test.
    pub slide_t: Option<f64>,
    /// R §6 again at this very pose on two independent redraws of `Pf`, `S` and the margin.
    pub resamples: Vec<ScoreRow>,
    /// Independent accepted joins that agree with this placement (see [`support_count`]).
    pub support: u32,
}

/// A candidate's confidence band (audit §D.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// R §6.5 accepted it **and** the strict thresholds, the slide probe and one of the two
    /// distinguishing arms hold. R §8 assembles from these and from nothing else.
    Confirmed,
    /// R §6.5 accepted it and the line above did not confirm it. Listed, never placed.
    Probable,
    /// R §6.5 refused it, with the reason it has today.
    Rejected,
}

impl Tier {
    /// The band a candidate has before any probe runs: R §6.5's own verdict, under the two names
    /// audit §D.1 gives it.
    ///
    /// This is what every candidate keeps when the tier pass is off, and it is why `--tiers off`
    /// changes nothing: R §8's gate is then [`Tier::Probable`] ∪ [`Tier::Confirmed`], which is
    /// `accepted`.
    #[must_use]
    pub const fn of_accept(accepted: bool) -> Self {
        if accepted { Self::Probable } else { Self::Rejected }
    }

    /// The word `report.md` and `report.json` print.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Confirmed => "confirmed",
            Self::Probable => "probable",
            Self::Rejected => "rejected",
        }
    }
}

/// The strict threshold set a [`Tier::Confirmed`] join has to clear, chosen on M1's table.
///
/// Every value is the one the note chose and every one is a flag on `run`. Two of the arms audit
/// §D.1 proposes are here as `Option`s that default to **off**, because M1 measured that they
/// cannot do the work the audit expected of them:
///
/// * `max_determined_deg` — determinedness separates (AUC 0.879) and cannot be thresholded: the
///   whole range over 808 candidates is 1.6e-14 to 4.1e-7 degrees, so every candidate of every run
///   is determined by any tolerance a person would write down. It is a ranking, and the report
///   prints it.
/// * `min_resample_accept` — requiring all three draws to accept costs six confirmed joins
///   (136 → 130) and removes no false one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    /// Fraction; tight contact a confirmed join needs (R §6.5 ships 0.25).
    pub min_tight: f64,
    /// `t`; median fracture gap a confirmed join may not exceed (R §6.5's own limit is 0.02–0.03).
    pub max_gap_t: f64,
    /// `t`; shortest seam a confirmed join must share (R §6.5 ships 3).
    pub min_seam: f64,
    /// Cosine; shell-normal agreement across the seam (R §6.5 ships 0.8).
    pub min_cont_n: f64,
    /// Fraction of surface samples allowed inside the other fragment (R §6.5 ships 0.005).
    pub max_pen: f64,
    /// `t`; how far the pose may stay from where it started after a ±[`SLIDE_T`] push along the
    /// seam.
    pub max_slide_t: f64,
    /// The margin over a second placement that scores, when the margin is the arm that confirms.
    ///
    /// M1: 1.5 and 2 confirm the same 136 joins, 3 confirms 132, 4 confirms 130 and 5 confirms
    /// 128 — 2 sits on a plateau rather than on a cliff.
    pub min_margin: f64,
    /// Independent agreeing paths, when the support count is the arm that confirms. `0` makes
    /// that arm always true, which disables the disjunction.
    pub min_support: u32,
    /// Degrees; off by default (see the type's own note).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_determined_deg: Option<f64>,
    /// How many of the three draws must accept; off by default (see the type's own note).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_resample_accept: Option<u32>,
}

impl Default for Thresholds {
    /// M1 §3's chosen tier, value for value.
    fn default() -> Self {
        Self {
            min_tight: 0.35,
            max_gap_t: 0.015,
            min_seam: 5.0,
            min_cont_n: 0.90,
            max_pen: 0.0,
            max_slide_t: SLIDE_BACK_T,
            min_margin: 2.0,
            min_support: 1,
            max_determined_deg: None,
            min_resample_accept: None,
        }
    }
}

impl Thresholds {
    /// Which of this set's tests the candidate fails, in the order the note argues them; empty
    /// means [`Tier::Confirmed`].
    ///
    /// A test whose evidence is **missing** fails: a candidate with no seam to slide along has not
    /// passed the slide probe, it has refused it. That is the same reading as the margin's, and
    /// M1's threshold table was computed under it.
    #[must_use]
    pub fn refusals(&self, scores: &Scores, probes: &Probes) -> Vec<String> {
        let mut failed: Vec<String> = Vec::new();
        let floor = |name: &str, value: f64, limit: f64, failed: &mut Vec<String>| {
            if under(value, limit) {
                failed.push(format!("{name} {value:.4} < {limit}"));
            }
        };
        let ceiling = |name: &str, value: f64, limit: f64, failed: &mut Vec<String>| {
            if over(value, limit) {
                failed.push(format!("{name} {value:.4} > {limit}"));
            }
        };
        floor("tight", scores.tight, self.min_tight, &mut failed);
        ceiling("gap", scores.gap, self.max_gap_t, &mut failed);
        floor("seam", scores.seam, self.min_seam, &mut failed);
        floor("cont_n", scores.cont_n, self.min_cont_n, &mut failed);
        ceiling("pen", scores.pen, self.max_pen, &mut failed);
        match probes.slide_t {
            Some(slide) => ceiling("slide", slide, self.max_slide_t, &mut failed),
            None => failed.push("slide: no shared seam to slide along".to_owned()),
        }
        if let Some(limit) = self.max_determined_deg {
            match probes.determined_deg {
                // Exponent form: the whole measured range is 1.6e-14 to 4.1e-7 degrees, and four
                // decimals would print every one of them as zero.
                Some(deg) if over(deg, limit) => {
                    failed.push(format!("determined {deg:e} deg > {limit:e} deg"));
                }
                Some(_) => {}
                None => failed.push("determined: not probed".to_owned()),
            }
        }
        if let Some(least) = self.min_resample_accept {
            let accepted = resample_accept(probes);
            if accepted < least {
                failed.push(format!("redraws accepted {accepted} < {least}"));
            }
        }
        let by_support = probes.support >= self.min_support;
        let by_margin = probes.margin.is_some_and(|m| m >= self.min_margin);
        if !(by_support || by_margin) {
            failed.push(match probes.margin {
                Some(m) => format!(
                    "neither arm: support {} < {} and margin {m:.2} < {}",
                    probes.support, self.min_support, self.min_margin
                ),
                None => format!(
                    "neither arm: support {} < {} and no second placement to beat",
                    probes.support, self.min_support
                ),
            });
        }
        failed
    }
}

/// How many of the three draws (the run's own and the two redraws) R §6.5 accepts.
///
/// The run's own draw is accepted **by construction**: a [`Probes`] exists only for a candidate
/// R §6.5 accepted, so the count is one plus the redraws that agree with it.
#[must_use]
pub fn resample_accept(probes: &Probes) -> u32 {
    let redraws = probes.resamples.iter().filter(|row| row.accepted).count();
    u32::try_from(redraws).unwrap_or(u32::MAX).saturating_add(1)
}

/// What `report.md`, `report.json` and `transforms.json` print about one candidate's tier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// `score / rival_score` over a second placement that scores; absent when there is none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
    /// How far that second placement puts the sherd, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rival_moved_t: Option<f64>,
    /// Distinct placements the pair's kept list makes.
    pub placements: usize,
    /// Worst rotation over the twelve one-ULP neighbours, in degrees — reported, never gated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub determined_deg: Option<f64>,
    /// The same in translation, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub determined_t: Option<f64>,
    /// How far the pose stays away after a ±0.5 t push along the seam, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slide_t: Option<f64>,
    /// Smallest `tight` over the three draws.
    pub resample_tight_min: f64,
    /// Largest `gap` over the three draws, in `t`.
    pub resample_gap_max: f64,
    /// How many of the three draws R §6.5 accepts.
    pub resample_accept: u32,
    /// Independent accepted joins that agree with this placement.
    pub support: u32,
    /// The tier's tests this candidate failed; empty on a confirmed join.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub failed: Vec<String>,
}

impl Evidence {
    /// The evidence of one probed candidate, with the tier the thresholds give it.
    fn of(scores: &Scores, probes: &Probes, th: &Thresholds) -> Self {
        let failed = th.refusals(scores, probes);
        let draws = std::iter::once(scores.tight).chain(probes.resamples.iter().map(|r| r.tight));
        let gaps = std::iter::once(scores.gap).chain(probes.resamples.iter().map(|r| r.gap));
        Self {
            margin: probes.margin,
            rival_moved_t: probes.rival_moved_t,
            placements: probes.placements,
            determined_deg: probes.determined_deg,
            determined_t: probes.determined_t,
            slide_t: probes.slide_t,
            resample_tight_min: draws.fold(f64::INFINITY, f64::min),
            resample_gap_max: gaps.fold(f64::NEG_INFINITY, f64::max),
            resample_accept: resample_accept(probes),
            support: probes.support,
            failed,
        }
    }
}

/// What one tier pass decided, by candidate index.
#[derive(Clone, Debug)]
pub struct TierReport {
    /// The set the pass ran with — what `report.json` and `transforms.json` carry as "the tier set
    /// used".
    pub thresholds: Thresholds,
    /// One band per candidate, in the candidate list's own order.
    ///
    /// [`crate::pipeline::run`] copies this onto [`Candidate::tier`] as soon as the pass returns,
    /// and everything downstream — R §8's gate, [`representatives`], the report's sections — reads
    /// it from the candidate. The two are the same value; this is the answer, that is where the
    /// answer lives.
    pub tiers: Vec<Tier>,
    /// The evidence behind each band; `None` for a candidate R §6.5 refused, which is never probed.
    pub evidence: Vec<Option<Evidence>>,
    /// The raw probe answers, kept so that `--measure` in the same run reads the ladders this pass
    /// already climbed instead of climbing them again.
    pub probes: Vec<Option<Probes>>,
}

impl TierReport {
    /// How many **candidates** fell in each band.
    ///
    /// Several candidates of one pair routinely converge on one placement and are all confirmed,
    /// so this is not the number of joins; [`TierReport::pair_counts`] is.
    #[must_use]
    pub fn counts(&self) -> (usize, usize, usize) {
        let count = |want: Tier| self.tiers.iter().filter(|&&t| t == want).count();
        (count(Tier::Confirmed), count(Tier::Probable), count(Tier::Rejected))
    }

    /// How many **pairs** fell in each band — the number a person means by "confirmed joins",
    /// counted over [`representatives`].
    #[must_use]
    pub fn pair_counts(&self, candidates: &[Candidate]) -> TierCounts {
        let mut counts = TierCounts { confirmed: 0, probable: 0, rejected: 0 };
        for i in representatives(candidates) {
            match self.tiers.get(i).copied().unwrap_or(Tier::Rejected) {
                Tier::Confirmed => counts.confirmed += 1,
                Tier::Probable => counts.probable += 1,
                Tier::Rejected => counts.rejected += 1,
            }
        }
        counts
    }
}

/// The tier of every candidate of a finished match: the probes, then the thresholds.
///
/// Runs **before** R §8, because R §8 assembles from the confirmed joins; it needs R §6.1's
/// fracture BVHs, which are alive between the last pair and the assembly, and it needs nothing
/// from the assembly itself — the support count walks the accepted-candidate graph and not the
/// groups, which is what lets a join be confirmed before anything has been placed.
pub fn classify(
    engine: Engine<'_>,
    fragments: &[Fragment],
    candidates: &[Candidate],
    params: &Params,
    thresholds: &Thresholds,
) -> TierReport {
    let started = std::time::Instant::now();
    let probes = probe(engine, fragments, candidates, params);
    let evidence: Vec<Option<Evidence>> = candidates
        .iter()
        .zip(&probes)
        .map(|(c, p)| p.as_ref().map(|p| Evidence::of(&c.scores, p, thresholds)))
        .collect();
    let tiers: Vec<Tier> = candidates
        .iter()
        .zip(&evidence)
        .map(|(c, e)| match e {
            Some(e) if e.failed.is_empty() => Tier::Confirmed,
            _ => Tier::of_accept(c.accepted),
        })
        .collect();
    let report = TierReport { thresholds: *thresholds, tiers, evidence, probes };
    let (confirmed, probable, rejected) = report.counts();
    tracing::info!(
        confirmed,
        probable,
        rejected,
        seconds = started.elapsed().as_secs_f64(),
        "tiers"
    );
    report
}

/// Audit §D.1's three probes, the margin and audit §D.2's support count, for every **accepted**
/// candidate; `None` for the rest.
///
/// # Cost
///
/// Per pair that has an accepted candidate: one [`Pair::build`], one [`SurfaceLadder`], thirteen
/// two-rung climbs and two full-ladder climbs over the accepted candidates as a batch, and two
/// R §6 verifications each on the two resampled collections. The resampling is done once for the
/// whole collection rather than per pair, and a cloned [`Fragment`] shares its two BVHs through
/// their `Arc`s, so the redraw costs R §3.5 and not R §3.3 or R §6.1's trees. Measured over the
/// quality gate's forty runs: 634.8 s against 410.2 s without it, about half a run again.
pub fn probe(
    engine: Engine<'_>,
    fragments: &[Fragment],
    candidates: &[Candidate],
    params: &Params,
) -> Vec<Option<Probes>> {
    // The pair's whole returned list, in R §5.7's order, keyed by the pair.
    let mut by_pair: BTreeMap<(FragId, FragId), Vec<usize>> = BTreeMap::new();
    for (i, c) in candidates.iter().enumerate() {
        by_pair.entry((c.a, c.b)).or_default().push(i);
    }
    // R §8's own view: the best accepted candidate of each pair, which is what the support count
    // walks — a second candidate of the same pair is not an independent join.
    let best: BTreeMap<(FragId, FragId), usize> = by_pair
        .iter()
        .filter_map(|(&key, list)| best_of(candidates, list).map(|i| (key, i)))
        .collect();

    let redraws: Vec<Vec<Fragment>> = RESAMPLE_OFFSETS
        .iter()
        .map(|offset| {
            let seed = params.seed.wrapping_add(*offset);
            fragments
                .par_iter()
                .map(|f| {
                    let mut copy = f.clone();
                    copy.rebuild_samples(seed);
                    copy
                })
                .collect()
        })
        .collect();

    let work: Vec<((FragId, FragId), Vec<usize>)> = by_pair
        .iter()
        .filter(|(_, list)| list.iter().any(|&i| candidates[i].accepted))
        .map(|(&key, list)| (key, list.clone()))
        .collect();

    let found: Vec<(usize, Probes)> = work
        .par_iter()
        .flat_map(|(key, list)| {
            one_pair(
                engine,
                Collection { fragments, redraws: &redraws, candidates, best: &best },
                list,
                *key,
                params,
            )
        })
        .collect();

    let mut out: Vec<Option<Probes>> = vec![None; candidates.len()];
    for (index, probes) in found {
        out[index] = Some(probes);
    }
    out
}

/// R §8's `best_per_pair` rule for one pair's returned list: the **first** candidate at the top
/// score among the accepted ones.
///
/// Strictly the first, because `assembly::greedy::best_per_pair` compares with `>` and R §5.7
/// already returned the list sorted — a `max_by` would answer with the *last* of several equal
/// maxima and name a different candidate as the one R §8 would consider. Equal maxima are the
/// ordinary case here: several stage-1 poses of one pair routinely converge on one placement and
/// come back with identical scores.
pub(crate) fn best_of(candidates: &[Candidate], list: &[usize]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for &i in list {
        if !candidates[i].accepted {
            continue;
        }
        if best.is_none_or(|b| candidates[i].score() > candidates[b].score()) {
            best = Some(i);
        }
    }
    best
}

/// What every pair reads and none of them writes.
#[derive(Clone, Copy)]
struct Collection<'a> {
    fragments: &'a [Fragment],
    redraws: &'a [Vec<Fragment>],
    candidates: &'a [Candidate],
    best: &'a BTreeMap<(FragId, FragId), usize>,
}

/// Every accepted candidate of one pair: the probes that need the pair's own clouds, and the two
/// counts that need the whole collection.
fn one_pair(
    engine: Engine<'_>,
    all: Collection<'_>,
    list: &[usize],
    key: (FragId, FragId),
    params: &Params,
) -> Vec<(usize, Probes)> {
    let candidates = all.candidates;
    let (a, b) = (&all.fragments[key.0 as usize], &all.fragments[key.1 as usize]);
    let pair = Pair::build(a, b, params);
    let sc = pair.scales;
    let Some(surfaces) = pair.surfaces() else { return Vec::new() };
    let ladder = SurfaceLadder::of(&pair);
    let rungs = ladder.rungs();
    let centre = icp::centroid_of(ladder.fracture_source());

    let taken: Vec<usize> = list.iter().copied().filter(|&i| candidates[i].accepted).collect();
    let poses: Vec<Matrix4<f64>> = taken.iter().map(|&i| candidates[i].transform).collect();

    let determined = determinedness(&rungs[2..], &poses, &sc, &centre);
    let slide = slide_probe(&rungs, &surfaces, &poses, &sc, &centre);
    let resamples = resampled_scores(engine, all.redraws, key, &poses, params);
    let placements = distinct_placements(b, candidates, list, sc.t);

    taken
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let c = &candidates[i];
            let (rival_score, rival_moved_t) = rival(b, candidates, list, i, sc.t);
            (
                i,
                Probes {
                    t: sc.t,
                    placements,
                    margin: rival_score.and_then(|r| (r > 0.0).then(|| c.score() / r)),
                    rival_score,
                    rival_moved_t,
                    determined_deg: determined.get(k).map(|d| d.0),
                    determined_t: determined.get(k).map(|d| d.1),
                    slide_t: slide[k],
                    resamples: resamples.iter().map(|row| row[k]).collect(),
                    support: support_count(all, key, c, sc.t, &centre),
                },
            )
        })
        .collect()
}

/// The best candidate of the pair that places B somewhere else, and how far away that is.
///
/// "Somewhere else" is [`SAME_PLACEMENT_T`] measured as the parity harness measures it — the
/// **worst** displacement over every twentieth vertex of B's working mesh, in `t`. A maximum
/// rather than a mean, because a candidate that rotates the sherd about a point on its own surface
/// has a small mean displacement and is plainly a different placement.
fn rival(
    b: &Fragment,
    candidates: &[Candidate],
    list: &[usize],
    here: usize,
    t: f64,
) -> (Option<f64>, Option<f64>) {
    let mine = &candidates[here];
    let mut best: Option<(f64, f64)> = None;
    for &j in list {
        if j == here {
            continue;
        }
        let other = &candidates[j];
        let moved = placement_gap(b, &mine.transform, &other.transform) / t;
        if moved <= SAME_PLACEMENT_T {
            continue;
        }
        if best.is_none_or(|(score, _)| other.score() > score) {
            best = Some((other.score(), moved));
        }
    }
    (best.map(|(s, _)| s), best.map(|(_, m)| m))
}

/// How many distinct placements a pair's kept list makes, at [`SAME_PLACEMENT_T`].
///
/// Greedy single-link clustering in the order R §5.7 returned: a candidate joins the first cluster
/// whose representative it is within one wall of, and starts a new one otherwise. The clusters are
/// what "the second-best *placement* of the same pair" counts, and the count of them says whether
/// a margin exists at all.
fn distinct_placements(b: &Fragment, candidates: &[Candidate], list: &[usize], t: f64) -> usize {
    let mut representatives: Vec<usize> = Vec::new();
    for &i in list {
        let apart = representatives.iter().all(|&j| {
            placement_gap(b, &candidates[i].transform, &candidates[j].transform) / t
                > SAME_PLACEMENT_T
        });
        if apart {
            representatives.push(i);
        }
    }
    representatives.len()
}

/// The worst distance between two placements of one fragment, over every twentieth vertex of its
/// working mesh.
fn placement_gap(fragment: &Fragment, ours: &Matrix4<f64>, theirs: &Matrix4<f64>) -> f64 {
    let mut worst = 0.0_f64;
    for vertex in fragment.mesh.v.iter().step_by(20) {
        let point = vertex.to_f64();
        let here = crate::types::apply_transform(ours, point);
        let there = crate::types::apply_transform(theirs, point);
        let squared = (here[0] - there[0]).powi(2)
            + (here[1] - there[1]).powi(2)
            + (here[2] - there[2]).powi(2);
        worst = worst.max(squared.sqrt());
    }
    worst
}

/// C2 §5's twelve one-ULP neighbours, over a batch of poses at once: the worst rotation and the
/// worst displacement each pose moves by when R §5.6's last two rungs are climbed again from a
/// neighbour of it.
///
/// The rotation is [`pose_gap::frobenius_deg`] and not the trace form, for the reason
/// `sherd_gpu::crosscheck` gives: the trace form's floor on these poses is 3.6e-2 degrees, which
/// is larger than the differences being looked for.
fn determinedness(
    rungs: &[ladder::Rung<'_>],
    poses: &[Matrix4<f64>],
    sc: &Scales,
    centre: &[f64; 3],
) -> Vec<(f64, f64)> {
    let last = |climbed: Vec<Vec<icp::Registration>>| -> Vec<Matrix4<f64>> {
        climbed
            .into_iter()
            .zip(poses)
            .map(|(out, init)| out.last().map_or(*init, |r| r.transform))
            .collect()
    };
    let base = last(ladder::climb_all(Engine::REFERENCE, rungs, poses, sc));
    let mut worst = vec![(0.0_f64, 0.0_f64); poses.len()];
    for i in 0..3 {
        for j in 0..4 {
            let nudged: Vec<Matrix4<f64>> = poses
                .iter()
                .map(|pose| {
                    let mut near = *pose;
                    near[(i, j)] = near[(i, j)].next_up();
                    near
                })
                .collect();
            let other = last(ladder::climb_all(Engine::REFERENCE, rungs, &nudged, sc));
            for (k, (x, y)) in base.iter().zip(&other).enumerate() {
                worst[k].0 = worst[k].0.max(pose_gap::frobenius_deg(x, y));
                worst[k].1 = worst[k].1.max(pose_gap::cloud_t(x, y, centre, sc.t));
            }
        }
    }
    worst
}

/// Audit §D.1's slide probe: push the placed sherd ±[`SLIDE_T`] along the seam, refine it again,
/// and report how far the answer ends up from the pose it started at — the worse of the two
/// directions, read at the fracture cloud's centroid ([`pose_gap::cloud_t`]).
///
/// A true join sits in a well along the seam and comes back; a wrong-pose join *is* a slide along
/// that seam, where the surface is flat in exactly the direction the probe pushes and the ladder
/// has nothing to pull it back with.
///
/// # Which rungs, and why not the one the audit names
///
/// The audit says "restart the **last fracture rung**". Measured, that rung alone cannot answer
/// the question it is being asked: its correspondence radius is `0.04 t`
/// ([`STAGE2_FRAC_RUNGS`](crate::matching::ladder::STAGE2_FRAC_RUNGS)`[1]`) and the push is
/// `0.5 t`, so after the push almost no source point has a target inside the radius and the rung
/// does nothing — on the terracotta's two true joins it left the pose **0.49 t** away, which is
/// the push itself and not a measurement of the seam. R §5.6's ladder as a whole starts at
/// `0.2 t`, and from the same pushes it brings both of them back to **0.000 t**. So the probe is
/// the whole of R §5.6 restarted: two ladders per accepted candidate, batched over the pair.
fn slide_probe(
    rungs: &[ladder::Rung<'_>],
    surfaces: &(Surfaces<'_>, Surfaces<'_>),
    poses: &[Matrix4<f64>],
    sc: &Scales,
    centre: &[f64; 3],
) -> Vec<Option<f64>> {
    let directions: Vec<Option<[f64; 3]>> =
        poses.iter().map(|pose| seam_direction(&surfaces.0, &surfaces.1, pose, sc)).collect();
    let mut worst: Vec<Option<f64>> =
        directions.iter().map(|d| d.map(|_| 0.0_f64)).collect::<Vec<_>>();
    for sign in [1.0_f64, -1.0] {
        // One batch per direction: the shifted poses of every candidate that has a seam, climbed
        // rung by rung together, which is what makes the probe two ladders and not two per pose.
        let live: Vec<usize> = (0..poses.len()).filter(|&k| directions[k].is_some()).collect();
        let shifted: Vec<Matrix4<f64>> = live
            .iter()
            .map(|&k| {
                let direction = directions[k].expect("filtered to the ones that have a seam");
                let mut pose = poses[k];
                for i in 0..3 {
                    pose[(i, 3)] += sign * SLIDE_T * sc.t * direction[i];
                }
                pose
            })
            .collect();
        let climbed = ladder::climb_all(Engine::REFERENCE, rungs, &shifted, sc);
        for (n, &k) in live.iter().enumerate() {
            let back = climbed[n].last().map_or(shifted[n], |r| r.transform);
            let moved = pose_gap::cloud_t(&back, &poses[k], centre, sc.t);
            worst[k] = Some(worst[k].unwrap_or(0.0).max(moved));
        }
    }
    worst
}

/// The seam's own direction: the principal axis of the breakline points R §6.2 counts as shared.
///
/// The audit says "along the breakline tangent", and the tangent of a chain has an arbitrary sign
/// per point, so an average of `brk_tangent` over a seam cancels rather than adding up. The
/// principal axis of the seam's points is the same direction without the sign, and it is exactly
/// the set `seam_score` measures: A's breakline points with a moved B point inside `sc.seam` whose
/// shell normals agree.
fn seam_direction(
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
) -> Option<[f64; 3]> {
    let moved: Vec<[f64; 3]> =
        b.brk_p.iter().map(|p| crate::types::apply_transform(transform, *p)).collect();
    let normals: Vec<[f64; 3]> = b.brk_ns.iter().map(|n| rotate(transform, *n)).collect();
    let tree = PointTree::build(&moved)?;
    let mut seam: Vec<[f64; 3]> = Vec::new();
    for (point, normal) in a.brk_p.iter().zip(&a.brk_ns) {
        let Some((j, _)) = tree.nearest_below(point, sc.seam) else { continue };
        let n = normals[j as usize];
        if normal[0] * n[0] + normal[1] * n[1] + normal[2] * n[2] > NORMAL_AGREE {
            seam.push(*point);
        }
    }
    if seam.len() < 2 {
        return None;
    }
    principal_axis(&seam)
}

/// `R·n` as R §6.2 rotates a macro normal: the rotation block alone, not renormalised.
fn rotate(t: &Matrix4<f64>, n: [f64; 3]) -> [f64; 3] {
    [
        t[(0, 0)] * n[0] + t[(0, 1)] * n[1] + t[(0, 2)] * n[2],
        t[(1, 0)] * n[0] + t[(1, 1)] * n[1] + t[(1, 2)] * n[2],
        t[(2, 0)] * n[0] + t[(2, 1)] * n[1] + t[(2, 2)] * n[2],
    ]
}

/// The direction of largest variance of a point set, unit, with its largest component made
/// positive so that the two runs of one candidate cannot pick opposite signs.
pub(crate) fn principal_axis(points: &[[f64; 3]]) -> Option<[f64; 3]> {
    #[allow(clippy::cast_precision_loss, reason = "breakline lengths are far below 2^53")]
    let n = points.len() as f64;
    let mut centre = [0.0_f64; 3];
    for p in points {
        for i in 0..3 {
            centre[i] += p[i];
        }
    }
    for c in &mut centre {
        *c /= n;
    }
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    for p in points {
        let d = nalgebra::Vector3::new(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]);
        cov += d * d.transpose();
    }
    if !cov.iter().all(|x| x.is_finite()) {
        return None;
    }
    let eigen = cov.symmetric_eigen();
    let mut best = 0;
    for i in 1..3 {
        if eigen.eigenvalues[i] > eigen.eigenvalues[best] {
            best = i;
        }
    }
    if eigen.eigenvalues[best] <= 0.0 {
        return None;
    }
    let column = eigen.eigenvectors.column(best);
    let mut v = [column[0], column[1], column[2]];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return None;
    }
    let mut largest = 0;
    for i in 1..3 {
        if v[i].abs() > v[largest].abs() {
            largest = i;
        }
    }
    let sign = if v[largest] < 0.0 { -1.0 } else { 1.0 };
    for x in &mut v {
        *x = *x * sign / norm;
    }
    Some(v)
}

/// R §6 again at the same poses, on each of the two redrawn collections.
///
/// The pose is held fixed and only the samples move, which is the audit's own form of the probe:
/// *"a confirmed join keeps `tight` above the strict threshold and `gap` under its limit on all
/// three draws"*. Re-running the ICP as well would measure the ladder's basin instead, which is
/// what the determinedness and slide probes are for.
fn resampled_scores(
    engine: Engine<'_>,
    redraws: &[Vec<Fragment>],
    key: (FragId, FragId),
    poses: &[Matrix4<f64>],
    params: &Params,
) -> Vec<Vec<ScoreRow>> {
    redraws
        .iter()
        .map(|collection| {
            let (a, b) = (&collection[key.0 as usize], &collection[key.1 as usize]);
            let pair = Pair::build(a, b, params);
            let sc = pair.scales;
            let Some((sa, sb)) = pair.surfaces() else {
                let refused = Scores::default();
                let verdict = verify::accept(&refused, params, &sc);
                return vec![ScoreRow::of(&refused, verdict); poses.len()];
            };
            poses
                .iter()
                .map(|pose| {
                    let scores = verify::verify(engine.exec, &sa, &sb, pose, &sc, true, None);
                    let verdict = verify::accept(&scores, params, &sc);
                    ScoreRow::of(&scores, verdict)
                })
                .collect()
        })
        .collect()
}

/// How many independent accepted joins agree with this placement.
///
/// A *path* rather than a group: for every third fragment `x` the collection also accepted a join
/// with, `T_ax · T_xb` is a second, independent opinion about where B goes in A's frame, and it
/// supports this candidate when the two agree inside R §8's own tolerances
/// ([`agrees`], 10° and 0.5 `t`, measured in the pair's `t`). It needs no assembly, so a candidate
/// R §8 has not reached yet still has a support count — which is what lets the tier run *before*
/// R §8 and R §8 assemble from its answer. It is the quantity audit §D.2 asks the tier to read:
/// *"two consistent paths confirm a join whose own scores are probable"*.
///
/// Only the **best accepted candidate of each pair** counts, because that is the one R §8 would
/// consider; a second candidate of the same pair is the same evidence twice.
fn support_count(
    all: Collection<'_>,
    key: (FragId, FragId),
    here: &Candidate,
    t: f64,
    centre: &[f64; 3],
) -> u32 {
    if !(t.is_finite() && t > 0.0) {
        return 0;
    }
    let leg = |x: FragId, y: FragId| -> Option<Matrix4<f64>> {
        // A pair is stored under the order R §4.1 produced it in; either way round is a leg, and
        // the other way round is its inverse.
        if let Some(&i) = all.best.get(&(x, y)) {
            Some(all.candidates[i].transform)
        } else {
            all.best.get(&(y, x)).map(|&i| pose_inverse(&all.candidates[i].transform))
        }
    };
    let mut support = 0;
    for x in 0..all.fragments.len() {
        let Ok(x) = u32::try_from(x) else { continue };
        if x == key.0 || x == key.1 {
            continue;
        }
        let (Some(t_ax), Some(t_xb)) = (leg(key.0, x), leg(x, key.1)) else { continue };
        let path = t_ax * t_xb;
        // The rotation is R §8's own; the translation is read **at B's fracture cloud** rather
        // than at the file origin. Both transforms map B into A, so the origin form measures the
        // disagreement through a lever arm of B's distance from its own origin — 100–500 units on
        // these scans, where the tolerance is half a wall — and called every path a disagreement.
        let angle = rotation_angle_deg(&(pose_inverse(&path) * here.transform));
        let distance = pose_gap::cloud_t(&path, &here.transform, centre, t);
        if agrees(angle, distance) {
            support += 1;
        }
    }
    support
}

/// One pair's band, as `transforms.json` carries it: the tier per join.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierJoin {
    /// The fragment the pose maps into.
    pub a: String,
    /// The fragment the pose moves.
    pub b: String,
    /// The band of the candidate that represents the pair.
    pub tier: Tier,
    /// That candidate's `seam · tight`.
    pub score: f64,
}

/// The three counts over the collection's **pairs**, as `report.json` carries them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierCounts {
    /// Pairs R §8 may build with.
    pub confirmed: usize,
    /// Pairs R §6.5 accepted and the tier did not confirm.
    pub probable: usize,
    /// Pairs whose every candidate R §6.5 refused.
    pub rejected: usize,
}

/// The candidate that represents each pair: the best **band**, and inside it the best score.
///
/// Indices into `candidates`, one per pair, in the pair order R §4.1 produced. A pair's band is
/// the band of its best candidate under that order, which is the row a conservator reads: what is
/// the strongest thing this tool will say about these two sherds?
#[must_use]
pub fn representatives(candidates: &[Candidate]) -> Vec<usize> {
    let rank = |t: Tier| match t {
        Tier::Confirmed => 0_u8,
        Tier::Probable => 1,
        Tier::Rejected => 2,
    };
    let mut best: BTreeMap<(FragId, FragId), usize> = BTreeMap::new();
    let mut order: Vec<(FragId, FragId)> = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        match best.get(&(c.a, c.b)) {
            None => {
                best.insert((c.a, c.b), i);
                order.push((c.a, c.b));
            }
            Some(&held) => {
                let (held_c, mine) = (&candidates[held], c);
                let better =
                    (rank(mine.tier), -mine.score()) < (rank(held_c.tier), -held_c.score());
                if better {
                    best.insert((c.a, c.b), i);
                }
            }
        }
    }
    order.iter().map(|key| best[key]).collect()
}

/// The tier per join, for `transforms.json`.
#[must_use]
pub fn joins(candidates: &[Candidate], names: &[String]) -> Vec<TierJoin> {
    representatives(candidates)
        .into_iter()
        .map(|i| {
            let c = &candidates[i];
            TierJoin {
                a: names[c.a as usize].clone(),
                b: names[c.b as usize].clone(),
                tier: c.tier,
                score: c.score(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        Evidence, Probes, SAME_PLACEMENT_T, SLIDE_BACK_T, SLIDE_T, ScoreRow, Thresholds, Tier,
        principal_axis,
    };
    use crate::matching::verify::Scores;
    use approx::assert_relative_eq;

    /// The constants are the audit's own numbers, and `SAME_PLACEMENT_T` is the parity harness's.
    #[test]
    fn the_constants_are_the_ones_the_audit_names() {
        assert_relative_eq!(SAME_PLACEMENT_T, 1.0);
        assert_relative_eq!(SLIDE_T, 0.5);
        assert_relative_eq!(SLIDE_BACK_T, 0.1);
    }

    /// The principal axis of a line of points is the line, sign-normalised; a single point has
    /// no direction and the probe says so rather than inventing one.
    #[test]
    fn the_principal_axis_is_the_line_a_seam_lies_on() {
        let along: Vec<[f64; 3]> =
            (0..50).map(|i| [f64::from(i) * 0.1, -f64::from(i) * 0.2, 3.0]).collect();
        let axis = principal_axis(&along).expect("fifty collinear points have an axis");
        assert_relative_eq!(axis[0].hypot(axis[1]), 1.0, epsilon = 1e-12);
        assert_relative_eq!(axis[1] / axis[0], -2.0, epsilon = 1e-9);
        assert_relative_eq!(axis[2], 0.0, epsilon = 1e-12);
        // The sign rule: the largest component is positive, whichever way the chain was walked.
        let mut back = along.clone();
        back.reverse();
        assert_eq!(principal_axis(&back), Some(axis));
        assert_eq!(principal_axis(&[[1.0, 2.0, 3.0]]), None);
        assert_eq!(
            principal_axis(&[[1.0, 2.0, 3.0]; 8]),
            None,
            "a heap of one point is not a line"
        );
    }

    /// `Thresholds::default` is M1 §3's chosen tier, value for value, and the two arms the
    /// measurement retired are off.
    #[test]
    fn the_default_thresholds_are_the_ones_the_measurement_chose() {
        let th = Thresholds::default();
        assert_relative_eq!(th.min_tight, 0.35);
        assert_relative_eq!(th.max_gap_t, 0.015);
        assert_relative_eq!(th.min_seam, 5.0);
        assert_relative_eq!(th.min_cont_n, 0.90);
        assert_relative_eq!(th.max_pen, 0.0);
        assert_relative_eq!(th.max_slide_t, 0.1);
        assert_relative_eq!(th.min_margin, 2.0);
        assert_eq!(th.min_support, 1);
        assert_eq!(th.max_determined_deg, None, "M1 §5.4: a ranking, not a gate");
        assert_eq!(th.min_resample_accept, None, "M1 §5.8: 136 confirmed become 130, no false one");
        // The two retired arms are skipped on the way out, so the tier set a report carries is the
        // eight numbers the note prints and not ten with two nulls.
        let json = serde_json::to_value(th).expect("Thresholds serialises");
        assert_eq!(json.as_object().expect("an object").len(), 8);
        assert_eq!(serde_json::from_value::<Thresholds>(json).expect("round trip"), th);
    }

    /// A scores/probe pair that clears every test of the default set.
    fn confirmable() -> (Scores, Probes) {
        let scores = Scores {
            tight: 0.72,
            gap: 0.007,
            seam: 45.3,
            cont_n: 0.988,
            pen: 0.0,
            ..Scores::default()
        };
        let probes = Probes {
            t: 3.75,
            placements: 2,
            rival_score: Some(4.0),
            rival_moved_t: Some(8.2),
            margin: Some(8.15),
            determined_deg: Some(1.1e-13),
            determined_t: Some(7.0e-14),
            slide_t: Some(5.3e-14),
            resamples: vec![row(0.71, 0.0072, true), row(0.70, 0.0075, true)],
            support: 0,
        };
        (scores, probes)
    }

    fn row(tight: f64, gap: f64, accepted: bool) -> ScoreRow {
        ScoreRow {
            tight,
            gap,
            gap_limit: 0.03,
            contact: 16.0,
            seam: 45.0,
            cont: 0.04,
            cont_n: 0.98,
            pen: 0.0,
            pen_depth: 0.0,
            pen_unavailable: false,
            score: tight * 45.0,
            accepted,
        }
    }

    /// Every test of the chosen tier, one at a time: which one refuses, and what it says.
    #[test]
    fn each_threshold_refuses_on_its_own() {
        let th = Thresholds::default();
        let (scores, probes) = confirmable();
        assert!(th.refusals(&scores, &probes).is_empty(), "the fixture is a confirmed join");

        let cases: Vec<(Scores, Probes, &str)> = vec![
            (Scores { tight: 0.34, ..scores }, probes.clone(), "tight"),
            (Scores { gap: 0.0151, ..scores }, probes.clone(), "gap"),
            (Scores { seam: 4.9, ..scores }, probes.clone(), "seam"),
            (Scores { cont_n: 0.899, ..scores }, probes.clone(), "cont_n"),
            (Scores { pen: 0.001, ..scores }, probes.clone(), "pen"),
            (scores, Probes { slide_t: Some(0.11), ..probes.clone() }, "slide"),
            (scores, Probes { slide_t: None, ..probes.clone() }, "slide: no shared seam"),
            (scores, Probes { margin: None, rival_score: None, ..probes.clone() }, "neither arm"),
            (scores, Probes { margin: Some(1.99), ..probes.clone() }, "neither arm"),
        ];
        for (s, p, want) in cases {
            let failed = th.refusals(&s, &p);
            assert_eq!(failed.len(), 1, "one test refuses {want}: {failed:?}");
            assert!(failed[0].starts_with(want), "{want} is refused by {failed:?}");
            assert!(!Evidence::of(&s, &p, &th).failed.is_empty());
        }
    }

    /// The disjunction: either arm confirms alone, and neither arm refuses together.
    #[test]
    fn the_support_count_and_the_margin_are_two_arms_of_one_test() {
        let th = Thresholds::default();
        let (scores, probes) = confirmable();
        let no_rival = Probes { margin: None, rival_score: None, ..probes.clone() };
        assert_eq!(th.refusals(&scores, &no_rival).len(), 1, "no rival and no support refuses");
        let supported = Probes { support: 1, ..no_rival.clone() };
        assert!(th.refusals(&scores, &supported).is_empty(), "one agreeing path confirms it");
        assert!(th.refusals(&scores, &probes).is_empty(), "a margin of 8.15 confirms it");
        // This is the whole reason audit §D.2's support count had to move from step 10 into
        // step 8: the pot_H and pot_A wrong-pose family has a single placement, so the margin can
        // never speak for it, and every other quantity R §6 measures calls it a true join.
        let family = Scores { tight: 0.77, gap: 0.0091, seam: 20.0, cont_n: 0.995, ..scores };
        let alone = Probes { support: 0, margin: None, rival_score: None, ..probes };
        assert_eq!(th.refusals(&family, &alone).len(), 1);
        assert!(th.refusals(&family, &alone)[0].starts_with("neither arm"));
    }

    /// The two retired arms do work when they are switched on, and are silent when they are not.
    #[test]
    fn the_retired_arms_are_off_and_still_work() {
        let (scores, probes) = confirmable();
        let off = Thresholds::default();
        assert!(off.refusals(&scores, &probes).is_empty());
        let strict =
            Thresholds { max_determined_deg: Some(1e-14), min_resample_accept: Some(3), ..off };
        let refusals = strict.refusals(&scores, &probes);
        assert_eq!(refusals.len(), 1, "1.1e-13 deg is over 1e-14: {refusals:?}");
        assert!(refusals[0].starts_with("determined 1.1e-13 deg"), "{refusals:?}");
        let loose = Thresholds { max_determined_deg: Some(1e-9), ..strict };
        let one_draw = Probes { resamples: vec![row(0.2, 0.09, false)], ..probes };
        let refusals = loose.refusals(&scores, &one_draw);
        assert_eq!(refusals, ["redraws accepted 1 < 3"], "the run's own draw and no redraw");
    }

    /// `Tier::of_accept` is what a candidate carries when the pass never runs, and it is why
    /// `--tiers off` changes nothing: R §8's gate becomes `accepted` again.
    #[test]
    fn the_band_before_any_probe_is_r_6_5s_own_verdict() {
        assert_eq!(Tier::of_accept(true), Tier::Probable);
        assert_eq!(Tier::of_accept(false), Tier::Rejected);
        assert_eq!(Tier::Confirmed.label(), "confirmed");
        assert_eq!(serde_json::to_string(&Tier::Probable).expect("a tier"), "\"probable\"");
    }

    /// The evidence carries the three draws as one number each, and the smallest `tight` and
    /// largest `gap` include the run's own draw.
    #[test]
    fn the_evidence_summarises_the_three_draws() {
        let th = Thresholds::default();
        let (scores, probes) = confirmable();
        let e = Evidence::of(&scores, &probes, &th);
        assert_relative_eq!(e.resample_tight_min, 0.70);
        assert_relative_eq!(e.resample_gap_max, 0.0075);
        assert_eq!(e.resample_accept, 3, "the run's own draw and both redraws");
        assert!(e.failed.is_empty(), "which is what makes it confirmed");
    }
}
