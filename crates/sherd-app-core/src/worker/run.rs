//! `Run` (A §2.2): the whole pipeline into the run's folder, keeping the match (A §3.1) and
//! writing the window's index of it — and none of the heavy outputs, which Export writes on
//! demand (A §4): a run here is tens of megabytes, which is what makes a history affordable.

use std::collections::BTreeSet;
use std::path::Path;

use nalgebra::Matrix4;
use sherd_backend::Resolved;
use sherd_core::memory::Budget;
use sherd_core::pipeline::{self, RunOptions, RunSummary};
use sherd_core::session::MatchState;
use sherd_core::tiers::Tier;
use sherd_core::{FragId, collection};

use super::{Context, Failure, app_failure, io_failure};
use crate::atomic;
use crate::protocol::{
    AssemblyDto, CandidateRow, EngineInfo, Event, FailKind, GroupDto, JoinDto, RunCounts, RunJob,
    StageTime,
};

/// The window's index of the match, in the run's folder (A §4).
pub const CANDIDATES_FILE: &str = "candidates.json";
/// A §3.1's saved match, likewise — `reassemble` reads it when a review changes a decision.
pub const MATCH_STATE_FILE: &str = "match.state";
/// How many rejected candidates per fragment the index keeps (A §4).
pub const REJECTED_PER_FRAGMENT: usize = 10;

/// Runs the pipeline into `runs/<run_id>/` and answers with the run's `Done`.
///
/// What lands there is the engine's own output with its heavy half switched off — no `placed/`,
/// no previews, no `scene.glb` (A §4) — plus A §3.1's match and [`CANDIDATES_FILE`], the index
/// the review screen reads instead of the 54 000 candidate rows of `report.json`.
pub(crate) fn run(job: &RunJob, context: &Context) -> Result<Event, Failure> {
    // A worker is a fresh process and builds the pool here; an `Err` is an in-process test that
    // already shares one, and the size it asked for is then the size it gets.
    if let Err(already) = pipeline::set_threads(job.spec.workers) {
        tracing::debug!("the thread pool is built already: {already}");
    }
    let resolved = sherd_backend::resolve(
        job.spec.backend(),
        job.spec.adapter.as_deref(),
        job.spec.gpu_memory_gb,
    )
    .map_err(|e| Failure::new(FailKind::Gpu, format!("{e:#}")))?;
    let dir = job.workspace.join("runs").join(&job.run_id);
    std::fs::create_dir_all(&dir).map_err(|e| io_failure(&dir, &e))?;

    // R §2's refusal is counted here rather than inside `run_with`, which reports it as a read
    // error on the folder: A §10 gives «мало фрагментов» a class and an action of its own, and a
    // disk error earns neither.
    let entries = collection::discover_excluding(&job.input, &job.excluded)?;
    if entries.len() < 2 {
        let message = format!("need at least two mesh files, found {}", entries.len());
        return Err(Failure::new(FailKind::TooFew, message));
    }

    let options = run_options(job, &resolved, &dir, context);
    let summary = pipeline::run_with(&job.input, &dir, &options, resolved.engine)?;

    write_candidates(&dir, &summary)?;
    context.emitter.emit(&Event::Assembly(assembly_dto(
        &summary.names,
        &summary.groups,
        &summary.poses,
        &summary.used,
        options.refine,
    )));
    Ok(Event::Done {
        counts: Some(counts(&summary, &options)),
        engine: Some(EngineInfo {
            core_version: sherd_core::CORE_VERSION.to_owned(),
            algo_ref: sherd_core::ALGO_REF.to_owned(),
            commit: sherd_core::GIT_COMMIT.to_owned(),
            backend: options.backend_label(),
        }),
        params: Some(options.params),
    })
}

/// The launch sheet as the engine takes it (A §7.4): the eleven thresholds the window sets, the
/// workspace's shared cache, and everything that writes a gigabyte turned off (A §4).
fn run_options(job: &RunJob, resolved: &Resolved, dir: &Path, context: &Context) -> RunOptions {
    RunOptions {
        target_faces: job.spec.target_faces,
        params: job.spec.params(),
        preview: false,
        write_meshes: false,
        review_images: false,
        cache: Some(job.workspace.join("cache")),
        workers: job.spec.workers,
        backend: resolved.backend,
        adapter: resolved.adapter.clone(),
        memory: job.spec.memory_gb.map_or_else(Budget::default_for_machine, Budget::gigabytes),
        watch: context.watch.clone(),
        constraints: job.constraints.clone(),
        match_state: Some(dir.join(MATCH_STATE_FILE)),
        excluded: job.excluded.clone(),
        ..RunOptions::default()
    }
}

/// Writes [`CANDIDATES_FILE`] from the match the run has just saved.
///
/// The evidence comes from the state and the bands from the summary, and that is deliberate: the
/// state is A §3.1's snapshot, taken before the constraints and the object round moved anything,
/// while the summary's `tier` is the band the run acted on. The two lists are the same list by
/// construction — but a pinned pose is appended to the match, so where the lengths differ the
/// index says nothing about evidence rather than attribute one candidate's to another.
fn write_candidates(dir: &Path, summary: &RunSummary) -> Result<(), Failure> {
    let state = MatchState::load(&dir.join(MATCH_STATE_FILE))?;
    let tiers = (state.candidates.len() == summary.candidates.len())
        .then_some(state.tiers.as_ref())
        .flatten();
    let used: BTreeSet<(FragId, FragId)> = summary.used.iter().copied().collect();
    let rows: Vec<(FragId, FragId, f64, Tier)> =
        summary.candidates.iter().map(|c| (c.a, c.b, c.score(), c.tier)).collect();
    let index: Vec<CandidateRow> = select(&rows, REJECTED_PER_FRAGMENT)
        .into_iter()
        .map(|i| {
            let candidate = &summary.candidates[i];
            CandidateRow {
                a: summary.names[candidate.a as usize].clone(),
                b: summary.names[candidate.b as usize].clone(),
                pose: pose_rows(&candidate.transform),
                score: candidate.score(),
                scores: candidate.scores,
                tier: candidate.tier,
                used: used.contains(&(candidate.a, candidate.b)),
                evidence: tiers.and_then(|t| t.evidence.get(i).cloned().flatten()),
            }
        })
        .collect();
    atomic::write_json(&dir.join(CANDIDATES_FILE), &index).map_err(|e| app_failure(&e))
}

/// What the run found, for `run.json` and the history (A §4).
fn counts(summary: &RunSummary, options: &RunOptions) -> RunCounts {
    let band = |want: Tier| summary.candidates.iter().filter(|c| c.tier == want).count();
    // With the tier pass off every accepted candidate carries `Probable` (`Tier::of_accept`) and
    // R §8 builds from all of them, so what the confirmed column means is then `accepted`.
    let (confirmed, probable) = if options.params.tiers.is_some() {
        (band(Tier::Confirmed), band(Tier::Probable))
    } else {
        (summary.accepted(), 0)
    };
    RunCounts {
        fragments: summary.names.len(),
        pairs: summary.pairs,
        skipped_pairs: summary.skipped_pairs,
        candidates: summary.candidates.len(),
        confirmed,
        probable,
        groups: summary.assembled().count(),
        unassembled: summary.groups.iter().filter(|g| g.len() == 1).count(),
        timings: summary
            .timings
            .iter()
            .map(|(stage, &seconds)| StageTime { stage: stage.to_owned(), seconds })
            .collect(),
    }
}

/// What the window draws of the assembly (A §8): the groups as R §8 laid them out, singletons
/// included — a group of one is what «не собрано» counts — and one pose per fragment by name,
/// because the window has the meshes already and needs only the matrices.
pub(crate) fn assembly_dto(
    names: &[String],
    groups: &[Vec<FragId>],
    poses: &[Matrix4<f64>],
    used: &[(FragId, FragId)],
    refined: bool,
) -> AssemblyDto {
    let name = |id: FragId| names[id as usize].clone();
    AssemblyDto {
        groups: groups
            .iter()
            .map(|members| GroupDto {
                // R §9 refines the joins of the assembly in one pass, so a run's groups are all
                // refined or none are — and a group of one has no join to refine.
                refined: refined && members.len() > 1,
                members: members.iter().map(|&id| name(id)).collect(),
            })
            .collect(),
        poses: groups
            .iter()
            .flatten()
            .map(|&id| (name(id), pose_rows(&poses[id as usize])))
            .collect(),
        joins: used.iter().map(|&(a, b)| JoinDto { a: name(a), b: name(b) }).collect(),
        // A run places what it can and argues with nobody; `unplaced` is `reassemble`'s, where a
        // decision can conflict with another (A §8.4, milestone 5).
        unplaced: Vec::new(),
    }
}

/// A pose as the protocol carries it: row-major, which is the order `transforms.json` writes and
/// the viewer reads.
fn pose_rows(pose: &Matrix4<f64>) -> [[f64; 4]; 4] {
    std::array::from_fn(|r| std::array::from_fn(|c| pose[(r, c)]))
}

/// Which candidates the index keeps, as ascending indices: every confirmed and probable one, and
/// for each fragment the `per_fragment` best-scoring rejected candidates it takes part in.
///
/// A pure function over `(a, b, score, tier)` so that the rule — which is A §4's promise to the
/// reviewer and not the engine's business — is tested without a collection.
pub(crate) fn select(rows: &[(u32, u32, f64, Tier)], per_fragment: usize) -> Vec<usize> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut keep: BTreeSet<usize> = BTreeSet::new();
    let mut rejected: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &(a, b, _, tier)) in rows.iter().enumerate() {
        if tier == Tier::Rejected {
            rejected.entry(a).or_default().push(i);
            rejected.entry(b).or_default().push(i);
        } else {
            keep.insert(i);
        }
    }
    for list in rejected.values_mut() {
        list.sort_by(|&x, &y| rows[y].2.total_cmp(&rows[x].2).then(x.cmp(&y)));
        keep.extend(list.iter().take(per_fragment));
    }
    keep.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sherd_core::tiers::Tier;

    /// A §4: every confirmed and probable join, and per fragment its ten best rejected — a
    /// fragment's worksheet, not the 54 000 rows of `report.json`.
    #[test]
    fn the_index_keeps_every_live_candidate_and_each_fragments_best_rejected() {
        // (a, b, score, tier): fragment 0 has 12 rejected candidates against partners 1..=12
        let mut rows: Vec<(u32, u32, f64, Tier)> =
            (1..=12).map(|b| (0, b, f64::from(b), Tier::Rejected)).collect();
        rows.push((0, 13, 0.5, Tier::Probable));
        rows.push((1, 2, 0.1, Tier::Confirmed));
        let kept = select(&rows, 10);
        let kept_rows: Vec<_> = kept.iter().map(|&i| rows[i]).collect();
        assert!(kept_rows.contains(&(0, 13, 0.5, Tier::Probable)));
        assert!(kept_rows.contains(&(1, 2, 0.1, Tier::Confirmed)));
        // fragment 0's ten best rejected are scores 12 down to 3; but (0,1) and (0,2) are also
        // among fragment 1's and 2's own best, so they stay: the index is a union of worksheets.
        let rejected_of_zero: Vec<f64> =
            kept_rows.iter().filter(|r| r.3 == Tier::Rejected).map(|r| r.2).collect();
        assert_eq!(rejected_of_zero.len(), 12);
        assert!(kept.windows(2).all(|w| w[0] < w[1]), "in candidate order, each once");
        assert_eq!(select(&rows, 0).len(), 2, "with no worksheet only the live candidates stay");

        // Where the cap bites: twelve rejected poses of ONE pair are in both fragments' lists, and
        // both lists keep the same ten best.
        let one_pair: Vec<(u32, u32, f64, Tier)> =
            (1..=12).map(|k| (0, 1, f64::from(k), Tier::Rejected)).collect();
        let kept: Vec<f64> = select(&one_pair, 10).iter().map(|&i| one_pair[i].2).collect();
        assert_eq!(kept, (3..=12).map(f64::from).collect::<Vec<_>>());
    }
}
