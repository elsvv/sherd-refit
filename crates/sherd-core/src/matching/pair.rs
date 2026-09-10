//! One pair of fragments, up to the candidate set (R §4.1–4.2, R §5.1–5.3).
//!
//! A pair is the unit of work of the matcher, and R §4.2 says what a pair *is*: the two fragments'
//! match arrays rebuilt at `t_pair = min(t_A, t_B)`, plus the [`Scales`] both are measured with.
//! The fragment whose own `t` is `t_pair` uses its cached arrays; the other rebuilds R §3.5 at
//! `t_pair`, which is why this type exists at all — nothing below it may assume a fragment is at
//! its own wall thickness.
//!
//! Step C1 filled in the first half of `match_pair`: [`Pair::hypotheses`] (R §5.1),
//! [`Pair::coarse`] (R §5.2) and [`Pair::suppress`] (R §5.3). Step C2 added the two refinement
//! ladders on top of exactly that state: [`Pair::stage1`] (R §5.4's breakline ICP and its
//! re-score), [`Pair::suppress_stage1`] (R §5.5) and [`Pair::stage2`] (R §5.6's four rungs). Step
//! C3 closed the chain: [`Pair::match_pair`] drives all of it, R §6's verification
//! ([`verify`](super::verify)) scores each surviving pose, and R §5.7 ranks what comes out.
//!
//! # Where the threads are
//!
//! Two stages are spread over `rayon`, and both are spread over *candidates*: R §5.4's 250
//! breakline ladders and R §5.6's ten surface ladders with the verification behind each of them.
//! Everything else — the hypotheses, the coarse score, both suppressions — is either already
//! parallel inside itself or too cheap to matter. The results are collected by index, so the
//! candidate list, the ranking and the accepted set are what they are whatever the thread count
//! does (D §7); the reference makes the same promise through `ThreadPoolExecutor.map`.

use std::sync::Arc;

use nalgebra::Matrix4;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::executor::Engine;
use crate::fragment::Fragment;
use crate::fragment::samples::MatchData;
use crate::matching::coarse::{self, Probe, Target};
use crate::matching::hypotheses::{self, Frames, Hypotheses};
use crate::matching::icp::{IcpTarget, Registration, cloud_points, homogeneous};
use crate::matching::ladder::{self, Rung};
use crate::matching::nms;
use crate::matching::scales::Scales;
use crate::matching::verify::{self, Scores, Surfaces};
use crate::params::Params;
use crate::tiers::{SAME_PLACEMENT_T, Tier, placement_gap};
use crate::types::FragId;

/// R §5.3's cap on the hypotheses the coarse NMS walks.
///
/// The score is a fraction of sixty points, so below the five thousandth-best hypothesis the
/// ranking carries no information the ICP could use; the truncation is what keeps NMS linear in
/// the kept count rather than in the hypothesis count.
pub const COARSE_ORDER_LIMIT: usize = 5000;
/// R §5.3's floor under the coarse score.
pub const COARSE_FLOOR: f64 = 0.1;

/// Task S3: how many stage-1 poses the wide rival falls back on when R §5.6's own list makes one
/// placement.
///
/// Four, because the fallback is the *best* second placement and not all of them: the stage-1
/// poses are read in R §5.4's own score order, so the first one that is a second placement is
/// already the pair's best-supported alternative, and the three behind it are there for the case
/// where R §5.6's ladder pulls the leader back onto the winner (§S3 note §2.2 measures how often
/// it does). Each try costs one four-rung climb and one R §6 verification, batched, and the
/// fallback runs only on a pair that has an accepted candidate at all.
pub const WIDE_STAGE1_TRIES: usize = 4;

/// Where a pair's second placement was found (task S3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RivalSource {
    /// One of R §5.6's own verified candidates, taken from the pair's **full** list — the one that
    /// exists before R §5.7's `keep` truncates it to five. Free: the pose was climbed and scored
    /// by the run itself and then thrown away.
    Stage2,
    /// A stage-1 pose R §5.5 did not keep, refined through R §5.6's four rungs and scored by R §6
    /// here, so that its `seam · tight` is the same quantity the winner's is.
    Stage1,
}

/// Task S3's second placement of a pair: what the margin is read against when nothing in R §5.7's
/// kept list moved the sherd.
///
/// The kept list is five candidates and they routinely converge on one fit — 270 of M1's 342
/// correct joins make a single placement there — so the margin arm cannot fire for them and the
/// tier falls back on the support count alone. This is the same question asked of a wider
/// population, and it is asked **beside** R §5.7's answer and never instead of it: the pair still
/// returns the five candidates it returned, in the order it returned them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WideRival {
    /// `seam · tight` of that placement, measured by R §6 exactly as the winner's is.
    pub score: f64,
    /// The rival's own pose, row-major, in the convention every other transform of this crate
    /// uses (`p_A = T · p_B`).
    ///
    /// It is carried because the distance the **margin arm** reads is measured from the candidate
    /// being judged and not from the pair's best
    /// ([`Probes::wide_rival_moved_t`](crate::tiers::Probes::wide_rival_moved_t)), and that
    /// distance cannot be recovered from [`WideRival::moved_t`] alone.
    pub transform: [[f64; 4]; 4],
    /// How far it places B from the pair's **best** candidate, in `t`
    /// ([`placement_gap`](crate::tiers::placement_gap)); always above
    /// [`SAME_PLACEMENT_T`](crate::tiers::SAME_PLACEMENT_T).
    ///
    /// This is the distance that **selected** this pose as the pair's second placement, and it is
    /// a property of the pair. It is *not* the number the margin arm tests: a pair's non-best
    /// candidate is judged against this rival at its own pose, so
    /// [`Probes::wide_rival_moved_t`](crate::tiers::Probes::wide_rival_moved_t) — measured from
    /// the candidate under test, exactly as [`Probes::rival_moved_t`](crate::tiers::Probes::rival_moved_t)
    /// is — is what [`Thresholds::rival_moved_of`](crate::tiers::Thresholds::rival_moved_of)
    /// returns. Both are reported.
    pub moved_t: f64,
    /// Whether it came out of R §5.6's own list or out of the stage-1 fallback.
    pub source: RivalSource,
    /// Whether R §6.5 would accept the rival itself — a second placement the pipeline would have
    /// been willing to believe is a stronger objection than one it would refuse.
    pub accepted: bool,
}

/// Two fragments and everything R §5 measures them with.
///
/// The two `MatchData` are behind an `Arc` because D §5's cache hands the same arrays to every
/// pair that asks for that fragment at that `t`; without a cache the `Arc` is one allocation per
/// pair and nothing else.
#[derive(Debug)]
pub struct Pair<'a> {
    /// A's match arrays at `t_pair` (R §4.2).
    pub a: Arc<MatchData<'a>>,
    /// B's match arrays at `t_pair`.
    pub b: Arc<MatchData<'a>>,
    /// The pair's resolved distances (R §1.2).
    pub scales: Scales,
    /// A's breakline frames in `f64` (R §5.1).
    pub frames_a: Frames,
    /// B's breakline frames in `f64`.
    pub frames_b: Frames,
}

impl<'a> Pair<'a> {
    /// R §4.1's wall-ratio test: a pair whose walls differ by more than `thick_ratio` is not
    /// matched at all.
    ///
    /// Two sherds of one vessel have the same wall to within the accuracy of R §3.2; a factor of
    /// two and a half apart means one of them is a rim, a base or another pot, and the thresholds
    /// of R §1.2 — all of them in units of the *thinner* wall — would describe neither.
    pub fn skipped(a: &Fragment, b: &Fragment, p: &Params) -> bool {
        if !(a.thick > 0.0 && b.thick > 0.0) {
            return true;
        }
        let ratio = a.thick / b.thick;
        ratio > p.thick_ratio || ratio < 1.0 / p.thick_ratio
    }

    /// R §4.2: both fragments' arrays at `t_pair`, and the pair's scales.
    ///
    /// Building this is the expensive part of a pair — one of the two fragments usually has to
    /// redraw R §3.5 — and it is why the pipeline groups a fragment's pairs together (D §5).
    pub fn build(a: &'a Fragment, b: &'a Fragment, p: &Params) -> Self {
        let scales = Scales::for_fragments(p, a, b);
        let reg_points = p.reg_points as usize;
        let a = Arc::new(MatchData::at(a, scales.t, reg_points));
        let b = Arc::new(MatchData::at(b, scales.t, reg_points));
        Self { frames_a: Frames::of(&a), frames_b: Frames::of(&b), a, b, scales }
    }

    /// R §5's first exit: both fragments need a fracture sample and a breakline.
    #[inline]
    pub fn matchable(&self) -> bool {
        self.a.matchable() && self.b.matchable()
    }

    /// R §5.1's hypotheses over the two fragments' own subsets.
    pub fn hypotheses(&self, p: &Params) -> Hypotheses {
        hypotheses::build(&self.frames_a, &self.frames_b, p)
    }

    /// R §5.2's probe: the sixty points of B every hypothesis is scored on.
    pub fn probe(&self, p: &Params) -> Probe {
        Probe::draw(&self.frames_b, &self.frames_b.sub, p.coarse_points as usize, p.seed)
    }

    /// R §5.2's coarse score of every hypothesis.
    pub fn coarse(&self, engine: Engine<'_>, hyp: &Hypotheses, probe: &Probe) -> Vec<f64> {
        let Some(tree) = self.a.kd_brk.as_ref() else {
            return vec![0.0; hyp.len()];
        };
        let target = Target { points: &self.frames_a.p, normals: &self.frames_a.ns, tree };
        coarse::scores(engine.exec, &target, probe, hyp, self.scales.coarse)
    }

    /// R §5.3's non-maximum suppression over the coarse scores: the poses stage 1 refines.
    pub fn suppress(&self, hyp: &Hypotheses, cs: &[f64], p: &Params) -> Vec<u32> {
        let order = nms::order_by_score(cs, COARSE_ORDER_LIMIT);
        nms::nms(&order, &hyp.r, &hyp.tau, cs, self.scales.nms, p.stage1 as usize, COARSE_FLOOR)
    }

    /// R §5.4: the breakline ladder and its re-score for every kept hypothesis.
    ///
    /// The candidates are independent — each one starts from its own hypothesis and touches
    /// nothing the others touch — so the whole kept list is one batch per rung and one batch for
    /// the re-score (D §6.4 step 4); the results are collected by index, so the list is the
    /// reference's `kept` order whatever the executor does with it (D §7).
    pub fn stage1(
        &self,
        engine: Engine<'_>,
        hyp: &Hypotheses,
        kept: &[u32],
    ) -> Vec<Stage1Candidate> {
        let Some(tree) = self.a.kd_brk.as_ref() else { return Vec::new() };
        let target = IcpTarget::from_cloud(&self.a.pc_brk_full);
        let source = cloud_points(&self.b.pc_brk);
        let rungs = ladder::stage1_rungs(&source, &target);
        let scoring = Target { points: &self.frames_a.p, normals: &self.frames_a.ns, tree };
        let (brk_points, brk_normals) = self.b_subset();

        let inits: Vec<Matrix4<f64>> =
            kept.iter().map(|&h| homogeneous(&hyp.r[h as usize], &hyp.tau[h as usize])).collect();
        let climbed = ladder::climb_all(engine, &rungs, &inits, &self.scales);
        let transforms: Vec<Matrix4<f64>> = climbed
            .iter()
            .zip(&inits)
            .map(|(out, init)| out.last().map_or(*init, |r| r.transform))
            .collect();
        let scores = ladder::brk_scores(
            engine,
            &scoring,
            &brk_points,
            &brk_normals,
            &transforms,
            self.scales.stage1,
        );
        kept.iter()
            .zip(transforms)
            .zip(scores)
            .map(|((&hypothesis, transform), score)| Stage1Candidate {
                hypothesis,
                transform,
                score,
            })
            .collect()
    }

    /// R §5.5's suppression over the stage-1 poses: the candidates stage 2 refines.
    pub fn suppress_stage1(&self, stage1: &[Stage1Candidate], p: &Params) -> Vec<u32> {
        let poses: Vec<Matrix4<f64>> = stage1.iter().map(|c| c.transform).collect();
        let scores: Vec<f64> = stage1.iter().map(|c| c.score).collect();
        let order = ladder::stage1_order(&scores);
        ladder::suppress_stage1(&poses, &scores, &order, self.scales.nms, p.stage2 as usize)
    }

    /// R §5.6's four rungs from one stage-1 pose, one [`Registration`] per rung.
    ///
    /// The clouds and their trees are built for this one call, so a caller with several poses to
    /// refine should build a [`SurfaceLadder`] once and climb it instead — which is what
    /// [`Pair::match_pair`] does, and most of why one thread of the port outruns ten of Open3D's
    /// (C2's note §6: `registration_icp` rebuilds a `KDTreeFlann` on every call).
    pub fn stage2(&self, engine: Engine<'_>, init: &Matrix4<f64>) -> Vec<Registration> {
        let ladder = SurfaceLadder::of(self);
        ladder::climb(engine, &ladder.rungs(), init, &self.scales)
    }

    /// R §6's view of both fragments, built once for the whole pair.
    ///
    /// `None` when either working mesh has no triangle — the one case [`RayScene`] refuses, and
    /// one no fragment of a real collection reaches.
    ///
    /// [`RayScene`]: crate::spatial::bvh::RayScene
    pub fn surfaces(&self) -> Option<(Surfaces<'a>, Surfaces<'a>)> {
        Some((Surfaces::of(&self.a)?, Surfaces::of(&self.b)?))
    }

    /// R §5.6 and R §6 for one stage-1 pose: the four rungs, then the verification and R §6.5.
    ///
    /// `brk` is the pose's own stage-1 re-score, which travels with the candidate into the report.
    pub fn stage2_candidate(
        &self,
        engine: Engine<'_>,
        ladder: &SurfaceLadder,
        surfaces: &(Surfaces<'_>, Surfaces<'_>),
        init: &Matrix4<f64>,
        brk: f64,
        p: &Params,
    ) -> Candidate {
        self.stage2_batch(engine, ladder, surfaces, std::slice::from_ref(init), &[brk], p)
            .pop()
            .expect("one pose in, one candidate out")
    }

    /// R §5.6 and R §6 for a batch of stage-1 poses (D §6.4 steps 5 and 6).
    ///
    /// The four rungs are climbed a rung at a time over the whole batch, then R §6 scores each
    /// pose; per candidate the sequence is exactly the one [`Pair::stage2_candidate`] describes,
    /// because each ladder depends on nothing but its own pose.
    ///
    /// With `Params::early_reject_tight > 0` the fracture scores are taken after the two `pc_reg`
    /// rungs and a candidate below the threshold skips the two `pc_frac` rungs and the expensive
    /// half of R §6; it keeps its cheap scores, is marked `partial`, and can never be accepted.
    /// That is off by default, and R §5.6 says why: the estimate can still rise by 0.09 over the
    /// two remaining rungs, so a threshold safe against `min_tight` saves almost nothing. It is
    /// also what keeps the last two rungs a batch over the *survivors* rather than over the batch.
    pub fn stage2_batch(
        &self,
        engine: Engine<'_>,
        ladder: &SurfaceLadder,
        surfaces: &(Surfaces<'_>, Surfaces<'_>),
        inits: &[Matrix4<f64>],
        brk: &[f64],
        p: &Params,
    ) -> Vec<Candidate> {
        let (a, b) = surfaces;
        let rungs = ladder.rungs();
        let climbed = ladder::climb_all(engine, &rungs[..2], inits, &self.scales);
        let mut poses: Vec<Matrix4<f64>> = climbed
            .iter()
            .zip(inits)
            .map(|(out, init)| out.last().map_or(*init, |r| r.transform))
            .collect();

        // R §5.6's early rejection, if it is on: the candidates it rejects keep their cheap scores
        // and leave the batch here.
        let mut out: Vec<Option<Candidate>> = vec![None; inits.len()];
        let mut alive: Vec<usize> = (0..inits.len()).collect();
        if p.early_reject_tight > 0.0 {
            let mut kept = Vec::with_capacity(alive.len());
            for &i in &alive {
                let frac = verify::fracture_scores(engine.exec, a, b, &poses[i], &self.scales);
                if frac.tight[2] < p.early_reject_tight {
                    let mut scores = verify::verify(
                        engine.exec,
                        a,
                        b,
                        &poses[i],
                        &self.scales,
                        false,
                        Some(frac),
                    );
                    scores.brk = brk[i];
                    out[i] = Some(self.candidate(poses[i], scores, false));
                } else {
                    kept.push(i);
                }
            }
            alive = kept;
        }

        let fine_inits: Vec<Matrix4<f64>> = alive.iter().map(|&i| poses[i]).collect();
        let climbed = ladder::climb_all(engine, &rungs[2..], &fine_inits, &self.scales);
        for (&i, out_rungs) in alive.iter().zip(&climbed) {
            if let Some(last) = out_rungs.last() {
                poses[i] = last.transform;
            }
        }
        let scored: Vec<Candidate> = alive
            .par_iter()
            .map(|&i| {
                let mut scores =
                    verify::verify(engine.exec, a, b, &poses[i], &self.scales, true, None);
                scores.brk = brk[i];
                let accepted = verify::accept(&scores, p, &self.scales);
                self.candidate(poses[i], scores, accepted)
            })
            .collect();
        for (&i, candidate) in alive.iter().zip(scored) {
            out[i] = Some(candidate);
        }
        out.into_iter()
            .map(|c| c.expect("every candidate is scored on one path or the other"))
            .collect()
    }

    /// R §4–§6 for this pair: the best `keep` candidates, best first (R §5.7).
    ///
    /// The whole chain, in the reference's order — hypotheses, coarse score, suppression, the
    /// breakline ladder and its re-score, the second suppression, the surface ladder, R §6 and
    /// R §6.5 — and the three places it gives up early are the reference's too: a pair with no
    /// fracture sample or no breakline, a pair with no hypothesis, and a pair whose coarse
    /// suppression kept nothing.
    pub fn match_pair(&self, engine: Engine<'_>, p: &Params, keep: usize) -> Vec<Candidate> {
        self.match_pair_wide(engine, p, keep, false)
    }

    /// [`Pair::match_pair`], and with `wide` task S3's second placement stamped on every candidate
    /// it returns.
    ///
    /// **Additive by construction.** The returned list — which poses, in which order, with which
    /// scores and which verdicts — is the one [`Pair::match_pair`] returns, whatever `wide` says;
    /// the flag adds a reading of the list R §5.7 was about to throw away and, where that reading
    /// is empty, a few stage-1 poses climbed and scored on the side. Nothing it computes is fed
    /// back into the search, and none of it draws a random number, so a run with `wide` on and a
    /// run with it off produce the same candidates in the same order.
    ///
    /// It runs only on a pair that has an **accepted** candidate, which is the population the tier
    /// reads (§S3 note §2.3 measures what that bound costs in time).
    pub fn match_pair_wide(
        &self,
        engine: Engine<'_>,
        p: &Params,
        keep: usize,
        wide: bool,
    ) -> Vec<Candidate> {
        if !self.matchable() {
            return Vec::new();
        }
        let hyp = self.hypotheses(p);
        if hyp.is_empty() {
            tracing::info!(pair = self.name(), "no hypotheses");
            return Vec::new();
        }
        let cs = self.coarse(engine, &hyp, &self.probe(p));
        let kept = self.suppress(&hyp, &cs, p);
        let stage1 = self.stage1(engine, &hyp, &kept);
        let Some(best1) = stage1.iter().map(|c| c.score).reduce(f64::max) else {
            tracing::info!(pair = self.name(), "nothing passed the coarse stage");
            return Vec::new();
        };
        if p.stage1_floor > 0.0 && best1 < p.stage1_floor {
            // R §5.4: nothing the breakline stage found comes near a seam, and stage 2 is where
            // all the time goes. The pair keeps its best stage-1 pose, marked `partial` so that it
            // still appears in the report and can never be accepted.
            //
            // This branch runs **before** `surfaces()`, and that ordering is the reference's: R §5.4
            // needs no BVH, and the reference builds `frac_scene` lazily on first use inside R §6.1.
            // Demanding the two scenes above the floor test would turn a pair whose working mesh has
            // no triangle into an empty return where R §5.4 returns the partial candidate.
            let best = &stage1[argmax(&stage1)];
            tracing::info!(
                pair = self.name(),
                best1,
                floor = p.stage1_floor,
                "the best stage-1 score is below the floor; stage 2 skipped"
            );
            return vec![self.candidate(
                best.transform,
                Scores::partial(&self.scales, best1),
                false,
            )];
        }
        let Some(surfaces) = self.surfaces() else { return Vec::new() };
        let kept2 = self.suppress_stage1(&stage1, p);
        let ladder = SurfaceLadder::of(self);
        let inits: Vec<Matrix4<f64>> =
            kept2.iter().map(|&k| stage1[k as usize].transform).collect();
        let brk: Vec<f64> = kept2.iter().map(|&k| stage1[k as usize].score).collect();
        let mut candidates = self.stage2_batch(engine, &ladder, &surfaces, &inits, &brk, p);
        // R §5.7: `seam · tight`, descending, stable — ties keep the `kept2` order. `brk_best` is
        // the pair's own best stage-1 score and goes on every candidate, the reference's included.
        candidates.sort_by(|x, y| {
            y.scores.score().partial_cmp(&x.scores.score()).unwrap_or(std::cmp::Ordering::Equal)
        });
        for candidate in &mut candidates {
            candidate.scores.brk_best = best1;
        }
        // Task S3, and the one line of this function that is not R §5.7's: the second placement is
        // read **here**, off the full sorted list, because the next line is where the information
        // is lost.
        let rival = (wide && candidates.iter().any(|c| c.accepted))
            .then(|| self.wide_rival(engine, &ladder, &surfaces, &candidates, &stage1, &kept2, p))
            .flatten();
        candidates.truncate(keep);
        if rival.is_some() {
            for candidate in &mut candidates {
                candidate.wide = rival;
            }
        }
        candidates
    }

    /// Task S3's second placement of this pair: the best-scoring pose more than one wall from the
    /// pair's best, looked for in two places in order.
    ///
    /// 1. **R §5.6's own full list**, before R §5.7's `keep` truncated it. This costs nothing at
    ///    all — every one of those poses was climbed and verified by the run — and it is the half
    ///    that makes the margin *exist* where the kept five agreed with each other.
    /// 2. **The stage-1 poses R §5.5 did not keep**, at most [`WIDE_STAGE1_TRIES`] of them, in
    ///    R §5.4's own score order, refined through R §5.6's four rungs and scored by R §6 so that
    ///    the number the margin divides by is the same quantity as its numerator. A pose is only a
    ///    rival if it is *still* more than one wall away after the ladder: the ordinary reason a
    ///    pair makes one placement is that every start converges on it, and a fallback that
    ///    reported the pre-refinement distance would invent a rival out of that convergence.
    ///
    /// `None` when neither place holds a second placement, which is the honest answer and the one
    /// the margin test reads as a failure ([`Thresholds::refusals`](crate::tiers::Thresholds)).
    #[allow(clippy::too_many_arguments, reason = "the pair's whole stage-2 state, borrowed")]
    fn wide_rival(
        &self,
        engine: Engine<'_>,
        ladder: &SurfaceLadder,
        surfaces: &(Surfaces<'_>, Surfaces<'_>),
        candidates: &[Candidate],
        stage1: &[Stage1Candidate],
        kept2: &[u32],
        p: &Params,
    ) -> Option<WideRival> {
        let best = candidates.first()?;
        let t = self.scales.t;
        if !(t.is_finite() && t > 0.0) {
            return None;
        }
        let b = self.b.fragment;
        let mut found: Option<WideRival> = None;
        let offer = |c: &Candidate, source: RivalSource, found: &mut Option<WideRival>| {
            let moved = placement_gap(b, &best.transform, &c.transform) / t;
            if moved <= SAME_PLACEMENT_T || !moved.is_finite() {
                return;
            }
            if found.is_none_or(|w| c.score() > w.score) {
                *found = Some(WideRival {
                    score: c.score(),
                    transform: std::array::from_fn(|r| {
                        std::array::from_fn(|q| c.transform[(r, q)])
                    }),
                    moved_t: moved,
                    source,
                    accepted: c.accepted,
                });
            }
        };
        for c in candidates.iter().skip(1) {
            offer(c, RivalSource::Stage2, &mut found);
        }
        if found.is_some() {
            return found;
        }

        // The fallback. `kept2` is skipped because those poses *are* the candidates above.
        let taken: Vec<bool> = {
            let mut flags = vec![false; stage1.len()];
            for &k in kept2 {
                if let Some(flag) = flags.get_mut(k as usize) {
                    *flag = true;
                }
            }
            flags
        };
        let scores: Vec<f64> = stage1.iter().map(|c| c.score).collect();
        let mut inits: Vec<Matrix4<f64>> = Vec::new();
        let mut brk: Vec<f64> = Vec::new();
        for k in ladder::stage1_order(&scores) {
            if inits.len() >= WIDE_STAGE1_TRIES {
                break;
            }
            let k = k as usize;
            if taken.get(k).copied().unwrap_or(true) {
                continue;
            }
            let moved = placement_gap(b, &best.transform, &stage1[k].transform) / t;
            if moved <= SAME_PLACEMENT_T || !moved.is_finite() {
                continue;
            }
            inits.push(stage1[k].transform);
            brk.push(stage1[k].score);
        }
        if inits.is_empty() {
            return None;
        }
        for c in &self.stage2_batch(engine, ladder, surfaces, &inits, &brk, p) {
            offer(c, RivalSource::Stage1, &mut found);
        }
        found
    }

    /// One candidate of this pair, with the two fragment ids R §11.1 writes it under.
    fn candidate(&self, transform: Matrix4<f64>, scores: Scores, accepted: bool) -> Candidate {
        Candidate {
            a: self.a.fragment.id,
            b: self.b.fragment.id,
            transform,
            scores,
            accepted,
            tier: Tier::of_accept(accepted),
            wide: None,
        }
    }

    /// `a__b`, for the log lines and the fixture scope.
    fn name(&self) -> String {
        format!("{}__{}", self.a.fragment.name, self.b.fragment.name)
    }

    /// B's breakline subset and its shell normals, in `f64` — R §5.4's `P_B[brk_sub]`, `ns_B[brk_sub]`.
    pub fn b_subset(&self) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let take = |source: &[[f64; 3]]| -> Vec<[f64; 3]> {
            self.frames_b.sub.iter().map(|&i| source[i as usize]).collect()
        };
        (take(&self.frames_b.p), take(&self.frames_b.ns))
    }
}

/// R §5.6's two clouds, their trees and their sources, built once for a whole pair.
///
/// The four rungs register `pc_reg` twice and `pc_frac` twice, ten times over on a full pair, and
/// the trees behind the two targets are what makes each of those forty registrations cheap. They
/// are built here, once, and borrowed by every rung of every candidate.
#[derive(Debug)]
pub struct SurfaceLadder {
    reg_target: IcpTarget,
    frac_target: IcpTarget,
    reg_source: Vec<[f64; 3]>,
    frac_source: Vec<[f64; 3]>,
}

impl SurfaceLadder {
    /// The clouds of one pair: A's are the targets, B's move (R §0's convention).
    pub fn of(pair: &Pair<'_>) -> Self {
        Self {
            reg_target: IcpTarget::from_cloud(&pair.a.pc_reg),
            frac_target: IcpTarget::from_cloud(&pair.a.pc_frac),
            reg_source: cloud_points(&pair.b.pc_reg),
            frac_source: cloud_points(&pair.b.pc_frac),
        }
    }

    /// The moving cloud of the two fracture rungs — B's `pc_frac` in B's own frame.
    ///
    /// The two probes of `measure` read a pose difference *at the fragment* rather than at the
    /// file origin ([`pose_gap::cloud_t`](crate::matching::icp::pose_gap::cloud_t)), and this is
    /// the cloud whose centroid they read it at: the fracture samples are what the last rung
    /// registers and what a slide along the seam moves.
    pub fn fracture_source(&self) -> &[[f64; 3]] {
        &self.frac_source
    }

    /// R §5.6's four rungs over those clouds.
    pub fn rungs(&self) -> [Rung<'_>; 4] {
        ladder::stage2_rungs(
            &self.reg_source,
            &self.reg_target,
            &self.frac_source,
            &self.frac_target,
        )
    }
}

/// One verified candidate: a pose, R §6's scores, R §6.5's verdict and roadmap item 3's
/// confidence band (D §4.1).
///
/// `a` and `b` are the two fragments' ids in the collection, and the pose maps **b** into **a**'s
/// frame (R §0). [`Candidate::tier`] is R §6.5's verdict under another name
/// ([`Tier::of_accept`]) until [`crate::tiers::classify`] has run over the finished match; a run
/// with the tier pass off never calls it, and R §8's gate is then `accepted` exactly as before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The fragment the pose maps *into* (R §4.1's first name).
    pub a: FragId,
    /// The fragment the pose moves.
    pub b: FragId,
    /// `T`: `p_A = R·p_B + τ`.
    pub transform: Matrix4<f64>,
    /// Every score of R §6.
    pub scores: Scores,
    /// R §6.5's verdict.
    pub accepted: bool,
    /// Roadmap item 3's confidence band (audit §D.1), [`Tier::of_accept`] until the tier pass has
    /// spoken.
    pub tier: Tier,
    /// Task S3's second placement of this candidate's **pair**, or `None` — the default, and what
    /// every candidate of a run with the tier pass off carries.
    ///
    /// The same value on every candidate of a pair: it is a property of the pair's search and not
    /// of one pose, and [`WideRival::moved_t`] is measured from the pair's best candidate
    /// ([`Pair::match_pair_wide`]). The distance the margin arm reads is measured from the
    /// candidate being judged instead ([`Probes::wide_rival_moved_t`](crate::tiers::Probes::wide_rival_moved_t)).
    pub wide: Option<WideRival>,
}

impl Candidate {
    /// R §5.7's ranking key, `seam · tight`.
    #[inline]
    pub fn score(&self) -> f64 {
        self.scores.score()
    }

    /// Whether R §8 may build with this candidate, under the gate the run is using.
    #[inline]
    #[must_use]
    pub fn admitted(&self, gate: Gate) -> bool {
        match gate {
            Gate::Accepted => self.accepted,
            Gate::Confirmed => self.tier == Tier::Confirmed,
        }
    }
}

/// Which candidates R §8 is allowed to build an assembly from.
///
/// [`Gate::Accepted`] is R §8 as the reference wrote it and as `--tiers off` runs it;
/// [`Gate::Confirmed`] is audit §D.1's rule — *"the assembly is built from confirmed joins only;
/// probable joins are listed and rendered, never placed"*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Gate {
    /// R §6.5's own verdict.
    #[default]
    Accepted,
    /// [`Tier::Confirmed`] and nothing else.
    Confirmed,
}

/// R §4–§6 for one pair of fragments, from the two fragments themselves.
///
/// R §4.1's wall-ratio test is applied first: a pair whose walls differ by more than
/// `p.thick_ratio` produces no candidate and never builds its [`Pair`], which is the expensive
/// part. `keep` is R §5.7's `keep = 5`.
pub fn match_pair(a: &Fragment, b: &Fragment, p: &Params, keep: usize) -> Vec<Candidate> {
    match_pair_with(Engine::REFERENCE, a, b, p, keep)
}

/// [`match_pair`] on a given executor — what the pipeline calls, because `--backend` reaches
/// R §5.2's coarse score and R §7's rungs through it.
pub fn match_pair_with(
    engine: Engine<'_>,
    a: &Fragment,
    b: &Fragment,
    p: &Params,
    keep: usize,
) -> Vec<Candidate> {
    match_pair_wide_with(engine, a, b, p, keep, false)
}

/// [`match_pair_with`], with task S3's second placement recorded when `wide` is set.
///
/// The pipeline passes `params.tiers.is_some()`: the rival is evidence the tier reads, so a run
/// with the tier pass off does not pay for it and does not carry it.
pub fn match_pair_wide_with(
    engine: Engine<'_>,
    a: &Fragment,
    b: &Fragment,
    p: &Params,
    keep: usize,
    wide: bool,
) -> Vec<Candidate> {
    if Pair::skipped(a, b, p) {
        return Vec::new();
    }
    Pair::build(a, b, p).match_pair_wide(engine, p, keep, wide)
}

/// One pose surviving R §5.4, with the hypothesis it came from and its re-score.
///
/// Not to be confused with [`Candidate`], which is what R §6 makes of the ten of these that
/// R §5.5 keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1Candidate {
    /// Index into the hypothesis set the pose started as (R §5.1).
    pub hypothesis: u32,
    /// The pose after both breakline rungs.
    pub transform: Matrix4<f64>,
    /// R §5.4's `s1`: the re-score of that pose at `sc.stage1`.
    pub score: f64,
}

/// `int(np.argmax(s1))` over a stage-1 list: the index of the **first** maximum.
///
/// numpy's `argmax` returns the lowest index among equal maxima, and R §5.4 reads its answer
/// directly. That choice is not a detail here: `s1` is a mean of booleans over `|brk_sub|` probe
/// points, so its values are multiples of `1/|brk_sub|` and exact ties between poses are the
/// normal case rather than a corner one. Rust's [`Iterator::max_by`] documents the opposite rule —
/// "if several elements are equally maximum, the last element is returned" — so a transcription
/// through it returns a different pose for the same pair.
///
/// Panics on an empty list, which the one caller has already excluded.
#[allow(
    clippy::float_cmp,
    reason = "the maximum is one of these very scores; the equality is the search, not a tolerance"
)]
fn argmax(stage1: &[Stage1Candidate]) -> usize {
    let best = stage1.iter().map(|c| c.score).fold(f64::NEG_INFINITY, f64::max);
    stage1.iter().position(|c| c.score == best).expect("a non-empty list has a maximum")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "R §4.2 selects one of two doubles; there is no rounding")]

    use super::{Pair, Stage1Candidate, argmax};
    use crate::fragment::Fragment;
    use crate::params::Params;
    use nalgebra::Matrix4;
    use std::path::{Path, PathBuf};

    fn stage1(scores: &[f64]) -> Vec<Stage1Candidate> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &score)| Stage1Candidate {
                hypothesis: u32::try_from(i).unwrap(),
                transform: Matrix4::identity(),
                score,
            })
            .collect()
    }

    /// R §5.4's `int(np.argmax(s1))` picks the **first** maximum, as numpy does.
    ///
    /// `s1` is a mean over `|brk_sub|` booleans, so equal scores are ordinary rather than rare, and
    /// `Iterator::max_by` — which returns the *last* of several equal maxima — would hand R §5.4's
    /// floor branch a different pose from the reference's on any pair whose best score ties.
    #[test]
    fn the_stage_one_floor_picks_the_first_maximum_like_numpy() {
        assert_eq!(argmax(&stage1(&[0.1, 0.4, 0.4, 0.2, 0.4])), 1);
        assert_eq!(argmax(&stage1(&[0.5, 0.5, 0.5])), 0, "an all-tie list picks index 0");
        assert_eq!(argmax(&stage1(&[0.1, 0.2, 0.9])), 2);
        assert_eq!(argmax(&stage1(&[0.9])), 0);
        // A stage-1 score of exactly zero on every pose is the case a `>` scan would get wrong if
        // it started from zero rather than from the list's own first entry.
        assert_eq!(argmax(&stage1(&[0.0, 0.0, 0.0])), 0);
    }

    fn slab(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input").join(name)
    }

    /// R §4.1's ratio is on the wall thicknesses and it is symmetric in effect: the same pair is
    /// skipped whichever way round it is written.
    #[test]
    fn a_pair_of_very_different_walls_is_skipped() {
        let mut a = Fragment::from_mesh_file(slab("pieceA.ply"), 200_000, 0).unwrap();
        let mut b = a.clone();
        let p = Params::default();

        a.thick = 3.0;
        b.thick = 7.0;
        assert!(!Pair::skipped(&a, &b, &p), "7/3 = 2.33 is inside the ratio of 2.5");
        assert!(!Pair::skipped(&b, &a, &p));
        b.thick = 7.6;
        assert!(Pair::skipped(&a, &b, &p), "7.6/3 = 2.53 is outside it");
        assert!(Pair::skipped(&b, &a, &p), "and the test does not depend on the order");
        b.thick = 0.0;
        assert!(Pair::skipped(&a, &b, &p), "a fragment with no wall has no scale");
    }

    /// R §4.2: the pair is built at the thinner wall, and the fragment that is not at it has its
    /// arrays rebuilt while the other keeps its own.
    #[test]
    fn a_pair_is_built_at_the_thinner_wall() {
        let a = Fragment::from_mesh_file(slab("pieceA.ply"), 200_000, 0).unwrap();
        let b = Fragment::from_mesh_file(slab("pieceB.ply"), 200_000, 0).unwrap();
        let p = Params::default();
        let pair = Pair::build(&a, &b, &p);

        assert_eq!(pair.scales.t, a.thick.min(b.thick));
        assert_eq!(pair.scales.res, a.res().max(b.res()));
        assert_eq!(pair.a.t, pair.scales.t);
        assert_eq!(pair.b.t, pair.scales.t);
        assert!(pair.matchable());
        assert_eq!(pair.frames_a.len(), pair.a.brk.len());
        assert_eq!(pair.frames_b.sub, pair.b.brk.sub);
    }
}
