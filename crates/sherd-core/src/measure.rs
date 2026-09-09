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
use crate::matching::verify::{self, Scores, Surfaces, pose_inverse};
use crate::params::Params;
use crate::spatial::kdtree::PointTree;
use crate::types::FragId;

/// How far apart two candidates must place the sherd before they count as **different
/// placements** rather than as two readings of one, in wall thicknesses.
///
/// One wall, which is `sherd_parity::stages::candidates::SAME_PLACEMENT_T` — the same number the
/// parity harness uses to tell "the two sides put the sherd somewhere else" from "the two sides
/// disagree about the pose", and it is measured the same way (below).
pub const SAME_PLACEMENT_T: f64 = 1.0;

/// What the two stability redraws add to the run's own seed.
///
/// Far from the gate's own 0–4 so that a redraw can never coincide with another seed of the same
/// sweep, and fixed so that the table is reproducible.
pub const RESAMPLE_OFFSETS: [u64; 2] = [1_000_000, 2_000_000];

/// How far the slide probe pushes the placed sherd along the seam before restarting, in `t`
/// (audit §D.1: "restart the last fracture rung from ±0.5 t along the breakline tangent").
pub const SLIDE_T: f64 = 0.5;

/// How close the slid pose must come back for the audit to call the candidate slide-stable, in `t`.
///
/// Reported rather than applied: [`CandidateMeasure::slide_t`] carries the distance and the note
/// picks the threshold, but this is the audit's own proposal and the aggregate quotes it.
pub const SLIDE_BACK_T: f64 = 0.1;

/// Every score of R §6 the tier could read, with the pair's own limits (a flattened [`Scores`]).
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
    /// The row of one [`Scores`], with R §6.5 applied to it.
    pub fn of(s: &Scores, p: &Params, sc: &Scales) -> Self {
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
            accepted: verify::accept(s, p, sc),
        }
    }
}

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
    /// How many **distinct placements** those candidates make, clustered at [`SAME_PLACEMENT_T`].
    ///
    /// Usually one: several stage-1 poses of a pair routinely converge on the same fit, and a pair
    /// whose kept list holds one placement has nothing for [`CandidateMeasure::margin`] to be
    /// ahead of. That is why the margin is so often unbounded, and it is worth its own column.
    pub placements: usize,
    /// `seam · tight` of the best candidate of this pair that places B more than
    /// [`SAME_PLACEMENT_T`] away, or `None` when the pair's kept list holds only one placement.
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
    /// How far the pose comes back after being pushed ±[`SLIDE_T`] along the seam and given the
    /// last fracture rung again, in `t` — the worse of the two directions.
    pub slide_t: Option<f64>,
    /// R §6 again at this very pose on two independent redraws of `Pf`, `S` and the margin.
    pub resamples: Vec<ScoreRow>,
    /// Independent accepted joins that agree with this placement (see [`support_count`]).
    pub support: usize,
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
/// which the redraws are offset from.
pub fn measure(
    engine: Engine<'_>,
    fragments: &[Fragment],
    candidates: &[Candidate],
    used: &[usize],
    params: &Params,
    thickness: f64,
) -> MeasureReport {
    let started = std::time::Instant::now();
    let names: Vec<String> = fragments.iter().map(|f| f.name.clone()).collect();
    let used_set: BTreeSet<usize> = used.iter().copied().collect();

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

    let mut rows: Vec<CandidateMeasure> = work
        .par_iter()
        .flat_map(|(key, list)| {
            one_pair(
                engine,
                Collection {
                    fragments,
                    redraws: &redraws,
                    candidates,
                    best: &best,
                    used: &used_set,
                },
                list,
                *key,
                params,
            )
        })
        .collect();
    rows.sort_by_key(|r| r.index);

    tracing::info!(
        rows = rows.len(),
        pairs = work.len(),
        seconds = started.elapsed().as_secs_f64(),
        "measure"
    );
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

/// R §8's `best_per_pair` rule for one pair's returned list: the **first** candidate at the top
/// score among the accepted ones.
///
/// Strictly the first, because `assembly::greedy::best_per_pair` compares with `>` and R §5.7
/// already returned the list sorted — a `max_by` would answer with the *last* of several equal
/// maxima and name a different candidate as the one R §8 would consider. Equal maxima are the
/// ordinary case here: several stage-1 poses of one pair routinely converge on one placement and
/// come back with identical scores.
fn best_of(candidates: &[Candidate], list: &[usize]) -> Option<usize> {
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
    used: &'a BTreeSet<usize>,
}

/// Every accepted candidate of one pair: the probes that need the pair's own clouds, and the two
/// counts that need the whole collection.
fn one_pair(
    engine: Engine<'_>,
    all: Collection<'_>,
    list: &[usize],
    key: (FragId, FragId),
    params: &Params,
) -> Vec<CandidateMeasure> {
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

    taken
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let c = &candidates[i];
            let (rival_score, rival_moved_t) = rival(b, candidates, list, i, sc.t);
            CandidateMeasure {
                index: i,
                a: a.name.clone(),
                b: b.name.clone(),
                t: sc.t,
                transform: std::array::from_fn(|r| std::array::from_fn(|q| c.transform[(r, q)])),
                scores: ScoreRow::of(&c.scores, params, &sc),
                brk: c.scores.brk,
                brk_best: c.scores.brk_best,
                rank: list.iter().position(|&j| j == i).unwrap_or(0),
                of_pair: list.len(),
                placements: distinct_placements(b, candidates, list, sc.t),
                margin: rival_score.and_then(|r| (r > 0.0).then(|| c.score() / r)),
                rival_score,
                rival_moved_t,
                determined_deg: determined.get(k).map(|d| d.0),
                determined_t: determined.get(k).map(|d| d.1),
                slide_t: slide[k],
                resamples: resamples.iter().map(|row| row[k]).collect(),
                support: support_count(all, key, c, sc.t, &centre),
                used: all.used.contains(&i),
                best_of_pair: all.best.get(&key) == Some(&i),
            }
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
/// the whole of R §5.6 restarted, which is also what a run could afford in step 8: two ladders per
/// accepted candidate, batched over the pair.
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
fn principal_axis(points: &[[f64; 3]]) -> Option<[f64; 3]> {
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
                return vec![ScoreRow::of(&Scores::default(), params, &sc); poses.len()];
            };
            poses
                .iter()
                .map(|pose| {
                    let scores = verify::verify(engine.exec, &sa, &sb, pose, &sc, true, None);
                    ScoreRow::of(&scores, params, &sc)
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
/// R §8 never reached still has a support count, and it is the quantity audit §D.2 asks the tier
/// to read: *"two consistent paths confirm a join whose own scores are probable"*.
///
/// Only the **best accepted candidate of each pair** counts, because that is the one R §8 would
/// consider; a second candidate of the same pair is the same evidence twice.
fn support_count(
    all: Collection<'_>,
    key: (FragId, FragId),
    here: &Candidate,
    t: f64,
    centre: &[f64; 3],
) -> usize {
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

#[cfg(test)]
mod tests {
    use super::{SAME_PLACEMENT_T, SLIDE_BACK_T, SLIDE_T, principal_axis};
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
}
