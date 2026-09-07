//! The run: discovery, preprocessing, matching, assembly, refinement, outputs (D §5).
//!
//! One process, one rayon pool. Preprocessing is a `par_iter` over fragments bounded by a
//! memory-aware semaphore (a scan of `f` faces reserves `60 MB + 110 B·f` from `--memory-budget`,
//! default half of physical RAM). Matching walks the same 3×3 blocks of the collection order the
//! reference walks, so the pair order — and with it every seeded draw — is the reference's;
//! candidates inside a pair run in parallel and are collected by index, so nothing depends on the
//! schedule. Cancellation is an `AtomicBool` checked between units of work; progress is a
//! callback.
//!
//! Step S4 filled in the first stage: [`preprocess`], which is what `sherd-refit-rs segment`
//! drives — R §3.1–3.3 then, R §3.1–3.4 since step B1. The memory-aware semaphore is not part of it yet — that needs the per-fragment
//! high-water mark measured rather than guessed, which belongs with the memory work of phase 1e —
//! so the fan-out is a plain `par_iter` over the collection, with `--threads` sizing the pool.
//!
//! Step D3 filled in the rest: [`run`] is the reference's `sherd_refit.pipeline.run`, stage for
//! stage, and the schedule below it is the reference's too — [`pair_blocks`] is `_pair_blocks` and
//! [`block_size`] the one number `_match_workers` leaves for a single process to act on. The
//! per-fragment `MatchData` LRU of D §5 is not there: a pair rebuilds one of its two fragments'
//! arrays at `t_pair`, which R's cost table puts at 0.2 s against a pair's several seconds, and the
//! block order is already the one an LRU would want when it arrives.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nalgebra::Matrix4;
use rayon::prelude::*;

use crate::assembly::{Piece, assemble, recenter};
use crate::collection::{self, Entry};
use crate::error::{Error, Result};
use crate::executor::Backend;
use crate::fragment::{Fragment, cache, samples};
use crate::matching::hypotheses::Frames;
use crate::matching::icp::Numerics;
use crate::matching::pair::{self, Candidate};
use crate::matching::screen::{Screened, screen_pair, top_partners};
use crate::mesh::geometry;
use crate::params::Params;
use crate::refine::{FractureCloud, RefinePiece, fracture_cloud, refine_joins};
use crate::render::{self, PALETTE, Paint, Splat};
use crate::report::{FragmentStats, Outcome, write_placed_meshes, write_report, write_transforms};
use crate::spatial::kdtree::PointTree;
use crate::types::FragId;

/// What preprocessing one fragment produced.
#[derive(Debug)]
pub struct Preprocessed {
    /// The fragment, at the state R §3.4 leaves it in (the match arrays follow).
    pub fragment: Fragment,
    /// Whether it came from the cache rather than from the file (R §3.7).
    pub cached: bool,
    /// Wall-clock seconds the fragment took, cache hit or not.
    pub seconds: f64,
}

/// R §3.1–3.4 for a whole collection, in parallel, through the fragment cache (R §3.7, D §5
/// stage 2).
///
/// `out_dir` is the run's output directory: caches are written to `<out_dir>/cache/<name>.sherd`
/// and read from there when they describe the same file at the same `target_faces`. `None`
/// bypasses the cache entirely, which is what the parity harness wants and what `--no-cache`
/// gives.
///
/// Results come back in **collection order**, whatever order the pool finished them in, and every
/// fragment's own computation is single-threaded apart from the ray loops of R §3.2 and R §3.4.3
/// and the radius queries of R §3.4.2, all of which are indexed collects, so the result does not
/// depend on the thread count. A fragment that fails to load takes its error into the
/// result vector instead of aborting the collection: one unreadable file among a hundred scans is
/// a warning, not the end of the run.
pub fn preprocess(
    entries: &[Entry],
    target_faces: usize,
    out_dir: Option<&Path>,
) -> Vec<Result<Preprocessed>> {
    entries
        .par_iter()
        .map(|entry| {
            let started = std::time::Instant::now();
            let cache_path = out_dir.map(|dir| cache::cache_path(dir, &entry.name));
            let (fragment, cached) = Fragment::load_or_build(
                &entry.path,
                target_faces,
                &entry.name,
                cache_path.as_deref(),
            )?;
            Ok(Preprocessed { fragment, cached, seconds: started.elapsed().as_secs_f64() })
        })
        .collect()
}

/// Sizes the process-wide rayon pool (D §5's `--threads`); `0` leaves it at one per core.
///
/// Returns an error string when the pool has already been built, which can only happen if this is
/// called twice or after something else has already used rayon.
pub fn set_threads(threads: usize) -> std::result::Result<(), String> {
    if threads == 0 {
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().map_err(|e| e.to_string())
}

// =================================================================================================
// The run (D §5, R §2–§11)
// =================================================================================================

/// R §5.7's `keep`: how many candidates a pair returns. Not a flag on either side.
pub const KEEP_PER_PAIR: usize = 5;

/// D §5's block size: the collection is cut into blocks of this many fragments and a job is all
/// the pairs between two blocks.
///
/// The reference's `MD_LRU_MAX // 2` with `MD_LRU_MAX = 6`, and the number is not arbitrary on
/// either side: a worker holding a block of three against a block of three needs six `MatchData`
/// to serve the whole job without rebuilding one.
pub const BLOCK: usize = 3;

/// Everything `run` takes besides the two directories — the Python `pipeline.run`'s keyword
/// arguments, plus the port's own cache switch and backend.
#[derive(Clone, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "four of the reference's own `--no-*` switches, one field each"
)]
pub struct RunOptions {
    /// `--target-faces`: the working-mesh face budget per fragment (R §3.3).
    pub target_faces: usize,
    /// Every threshold of R §1.1.
    pub params: Params,
    /// R §5.7's `keep`; [`KEEP_PER_PAIR`] on both sides.
    pub keep_per_pair: usize,
    /// Write R §11.5's previews.
    pub preview: bool,
    /// Run R §9's full-resolution refinement.
    pub refine: bool,
    /// Write R §11.4's placed and merged meshes.
    pub write_meshes: bool,
    /// Read and write `<out>/cache/<name>.sherd` (R §3.7).
    pub cache: bool,
    /// How many pairs the machine can work on at once — the reference's `workers`, and here the
    /// size of the one rayon pool. Only the pair schedule reads it.
    pub workers: usize,
    /// Which executor ran, for `report.json`'s `engine` (D §4.3).
    pub backend: Backend,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            target_faces: 200_000,
            params: Params::default(),
            keep_per_pair: KEEP_PER_PAIR,
            preview: true,
            refine: true,
            write_meshes: true,
            cache: true,
            workers: 0,
            backend: Backend::Cpu,
        }
    }
}

/// What a finished run produced — everything the CLI prints and the tests assert on.
#[derive(Clone, Debug)]
pub struct RunSummary {
    /// The collection's names, in R §2's order.
    pub names: Vec<String>,
    /// One world pose per fragment after R §8.2, by [`FragId`].
    pub poses: Vec<Matrix4<f64>>,
    /// R §8's groups, largest first.
    pub groups: Vec<Vec<FragId>>,
    /// The joins the assembly took, in the order it took them.
    pub used: Vec<(FragId, FragId)>,
    /// Every candidate of every pair, in pair order (R §11.2's `candidates`).
    pub candidates: Vec<Candidate>,
    /// The collection's median wall thickness (R §4.1).
    pub thickness: f64,
    /// The collection's median working-mesh edge.
    pub resolution: f64,
    /// Pairs matched, after R §4.1's wall-ratio filter and R §4.3's screening.
    pub pairs: usize,
    /// Pairs R §4.1 skipped because the walls differ by more than `thick_ratio`.
    pub skipped_pairs: usize,
    /// `(screened, kept)` when R §4.3's partner search ran.
    pub screened: Option<(usize, usize)>,
    /// Pairs R §8.1 rematched with the larger budget.
    pub second_pass: usize,
    /// R §11.2's `timings`, in seconds.
    pub timings: BTreeMap<String, f64>,
    /// The files written, in the order they were written.
    pub written: Vec<PathBuf>,
}

impl RunSummary {
    /// How many candidates R §6.5 accepted.
    pub fn accepted(&self) -> usize {
        self.candidates.iter().filter(|c| c.accepted).count()
    }

    /// The groups of two or more, in R §8's order.
    pub fn assembled(&self) -> impl Iterator<Item = &Vec<FragId>> {
        self.groups.iter().filter(|g| g.len() > 1)
    }
}

/// R §2–§11 for one collection: the whole pipeline, in one process (D §5).
///
/// The stages are the reference's `sherd_refit.pipeline.run`, in its order and with its early
/// exits: discovery, preprocessing through the cache, R §4.1's pair enumeration and wall-ratio
/// filter, R §4.3's optional partner search, R §5–§6 over the pairs, R §8's assembly, R §8.1's
/// optional second pass, R §9's refinement, R §8.2's recentring and R §11's five writers.
///
/// The one structural difference from the reference is the one D §5 states: the reference forks a
/// process pool per stage and this is a single process with one rayon pool. Nothing about the
/// results depends on it — every parallel section collects by index — and the schedule below is
/// still the reference's, because the *order* the pairs are visited in is what fixes which
/// `MatchData` a worker already holds.
#[allow(clippy::too_many_lines, reason = "the reference's `pipeline.run`, stage for stage")]
pub fn run(input: &Path, out_dir: &Path, options: &RunOptions) -> Result<RunSummary> {
    let entries = collection::discover(input)?;
    if entries.len() < 2 {
        return Err(Error::read(
            input,
            format!("need at least two mesh files, found {}", entries.len()),
        ));
    }
    std::fs::create_dir_all(out_dir).map_err(|e| Error::write(out_dir, e))?;
    let workers = if options.workers == 0 { rayon::current_num_threads() } else { options.workers };
    let params = &options.params;
    let mut timings: BTreeMap<String, f64> = BTreeMap::new();
    tracing::info!(
        fragments = entries.len(),
        input = %input.display(),
        workers,
        "collection discovered"
    );

    // 1. preprocessing (R §3, through the cache of R §3.7)
    let started = Instant::now();
    let cache_dir = options.cache.then(|| out_dir.to_path_buf());
    let fragments = preprocess_collection(&entries, options.target_faces, cache_dir.as_deref())?;
    timings.insert("preprocess".to_owned(), started.elapsed().as_secs_f64());
    let names: Vec<String> = fragments.iter().map(|f| f.name.clone()).collect();
    let thickness = geometry::median(&fragments.iter().map(|f| f.thick).collect::<Vec<f64>>());
    let resolution = geometry::median(&fragments.iter().map(Fragment::res).collect::<Vec<f64>>());
    for fragment in &fragments {
        if (fragment.thick / thickness - 1.0).abs() > 0.4 {
            tracing::warn!(
                fragment = fragment.name,
                thickness = fragment.thick,
                median = thickness,
                "wall thickness differs from the collection median by more than 40%"
            );
        }
    }
    tracing::info!(
        seconds = timings["preprocess"],
        thickness,
        resolution,
        edges_per_t = thickness / resolution.max(1e-9),
        "preprocessing done"
    );

    // 2. pairs (R §4.1)
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut skipped = 0_usize;
    for a in 0..fragments.len() {
        for b in (a + 1)..fragments.len() {
            if pair::Pair::skipped(&fragments[a], &fragments[b], params) {
                skipped += 1;
            } else {
                pairs.push((a, b));
            }
        }
    }
    if skipped > 0 {
        tracing::info!(
            skipped,
            ratio = params.thick_ratio,
            "pairs skipped: the walls differ by more than the ratio"
        );
    }

    // 2a. partner search (R §4.3, off by default)
    let mut screened = None;
    if params.screen_top_k > 0 && pairs.len() >= params.screen_min_pairs as usize {
        let started = Instant::now();
        let before = pairs.len();
        pairs = screen(&fragments, &names, &pairs, params);
        timings.insert("screen".to_owned(), started.elapsed().as_secs_f64());
        screened = Some((before, pairs.len()));
        tracing::info!(
            screened = before,
            kept = pairs.len(),
            top_k = params.screen_top_k,
            seconds = timings["screen"],
            "partner search"
        );
    } else if pairs.len() >= params.screen_min_pairs as usize {
        tracing::info!(
            pairs = pairs.len(),
            "--screen-top-k K would keep only the K best-scoring partners per fragment; the cost \
             in true joins is measured in docs/superpowers/notes/2026-09-06-scale-pairs.md"
        );
    }

    // 2b. matching (R §5–§6)
    let started = Instant::now();
    let mut per_pair = match_all(&fragments, &pairs, params, options.keep_per_pair, workers, None);
    timings.insert("matching".to_owned(), started.elapsed().as_secs_f64());
    let mut candidates: Vec<Candidate> = per_pair.iter().flatten().copied().collect();
    tracing::info!(
        seconds = timings["matching"],
        candidates = candidates.len(),
        accepted = candidates.iter().filter(|c| c.accepted).count(),
        "matching done"
    );

    // 3. assembly (R §8). The timer starts here, not at `assemble`: the reference builds the
    // stage's own `MatchData` inside `timings["assembly"]`, and PMC-8's cheaper equivalent — the
    // fragments' own samples and their whole-mesh BVHs — belongs in the same place.
    let started = Instant::now();
    let samples: Vec<Vec<[f64; 3]>> = fragments.iter().map(|f| f.samples.surface_f64()).collect();
    fragments.par_iter().for_each(|f| {
        let _ = f.surface_scene();
    });
    let pieces: Vec<Piece<'_>> = fragments
        .iter()
        .zip(&samples)
        .map(|(f, s)| Piece {
            thick: f.thick,
            res: f.res(),
            watertight: f.watertight,
            mesh: f.surface_scene(),
            s_pen: s,
        })
        .collect();
    let mut assembly = assemble(&pieces, &candidates, params);
    timings.insert("assembly".to_owned(), started.elapsed().as_secs_f64());

    // 3a. second pass (R §8.1, off by default)
    let retry = second_pass_pairs(&names, &pairs, &candidates, &assembly.groups, params);
    if !retry.is_empty() {
        let started = Instant::now();
        let bigger = Params {
            stage1: params.second_pass_stage1,
            stage2: params.second_pass_stage2,
            stage1_floor: 0.0,
            ..*params
        };
        let again =
            match_all(&fragments, &retry, &bigger, options.keep_per_pair, workers, Some("second"));
        for (k, found) in retry.iter().zip(again) {
            let at = pairs.iter().position(|p| p == k).expect("a retried pair is a pair");
            per_pair[at] = found;
        }
        candidates = per_pair.iter().flatten().copied().collect();
        timings.insert("second_pass".to_owned(), started.elapsed().as_secs_f64());
        assembly = assemble(&pieces, &candidates, params);
        tracing::info!(
            pairs = retry.len(),
            seconds = timings["second_pass"],
            stage1 = bigger.stage1,
            stage2 = bigger.stage2,
            accepted = candidates.iter().filter(|c| c.accepted).count(),
            groups = assembly.groups.iter().filter(|g| g.len() > 1).count(),
            "second pass"
        );
    }
    let used: Vec<(FragId, FragId)> =
        assembly.used.iter().map(|&i| (candidates[i].a, candidates[i].b)).collect();

    // 4. full-resolution refinement (R §9) and R §8.2's recentring
    let mut poses = assembly.poses.clone();
    if options.refine && assembly.groups.iter().any(|g| g.len() > 1) {
        let started = Instant::now();
        poses = refine(&fragments, &assembly.groups, &poses, &used, params)?;
        timings.insert("refine".to_owned(), started.elapsed().as_secs_f64());
        tracing::info!(seconds = timings["refine"], joins = used.len(), "refinement done");
    }
    let poses = recenter(&poses, &pieces, &assembly.groups);

    // 5. outputs (R §11)
    let started = Instant::now();
    let mut written = Vec::new();
    let stats: Vec<FragmentStats> = fragments.iter().map(FragmentStats::of).collect();
    let rejected: Vec<(usize, String)> =
        assembly.rejected.iter().map(|r| (r.candidate, r.reason.message(&names))).collect();
    let outcome = Outcome {
        names: &names,
        candidates: &candidates,
        used: &assembly.used,
        rejected: &rejected,
        groups: &assembly.groups,
    };
    write_transforms(
        out_dir.join("transforms.json"),
        &names,
        &poses,
        &assembly.groups,
        thickness,
        params,
    )?;
    written.push(out_dir.join("transforms.json"));
    write_report(out_dir, &stats, thickness, &outcome, &timings, params, options.backend.as_str())?;
    written.push(out_dir.join("report.json"));
    written.push(out_dir.join("report.md"));
    if options.write_meshes {
        let paths: Vec<PathBuf> = fragments.iter().map(|f| f.source.path.clone()).collect();
        written.extend(write_placed_meshes(
            out_dir,
            &paths,
            &names,
            &poses,
            &assembly.groups,
            crate::io::writer::DEFAULT_COMMENT,
        )?);
    }
    if options.preview {
        written.extend(write_previews(out_dir, &fragments, &names, &poses, &assembly.groups)?);
    }
    // R §11.2's `timings` is written **before** this line on the reference's side too: the dict
    // handed to `write_report` is mutated after `json.dump` has already run, so `report.json`
    // carries every stage but this one. The fixture dumps confirm it — their `timings` hold
    // `preprocess`, `matching`, `assembly` and `refine` and nothing else — and the port keeps the
    // key out of the file for the same reason, reporting it only to the caller and the log.
    timings.insert("output".to_owned(), started.elapsed().as_secs_f64());
    tracing::info!(out = %out_dir.display(), files = written.len(), "outputs written");

    Ok(RunSummary {
        names,
        poses,
        groups: assembly.groups,
        used,
        candidates,
        thickness,
        resolution,
        pairs: pairs.len(),
        skipped_pairs: skipped,
        screened,
        second_pass: retry.len(),
        timings,
        written,
    })
}

/// [`preprocess`] with the ids assigned and the first failure turned into the run's failure.
///
/// The reference's `ProcessPoolExecutor.map` raises on the first worker that raised, and a
/// fragment that cannot be preprocessed has no pose, no pair and no row: there is nothing sensible
/// to assemble around it.
fn preprocess_collection(
    entries: &[Entry],
    target_faces: usize,
    out_dir: Option<&Path>,
) -> Result<Vec<Fragment>> {
    let results = preprocess(entries, target_faces, out_dir);
    let mut fragments = Vec::with_capacity(results.len());
    for (i, result) in results.into_iter().enumerate() {
        let mut fragment = result?.fragment;
        fragment.id = u32::try_from(i).expect("fewer than 2^32 fragments");
        fragments.push(fragment);
    }
    Ok(fragments)
}

/// D §5's block schedule: the pair indices grouped so that one job touches at most `2·block`
/// fragments, in the reference's `_pair_blocks` order.
///
/// `block = 1` degenerates to one job per pair, which is what the reference falls back to when
/// there are enough pairs to fill the machine on their own and load balance matters more than the
/// cache.
pub fn pair_blocks(pairs: &[(usize, usize)], block: usize) -> Vec<Vec<usize>> {
    let block = block.max(1);
    let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (k, &(a, b)) in pairs.iter().enumerate() {
        let (ba, bb) = (a / block, b / block);
        groups.entry((ba.min(bb), ba.max(bb))).or_default().push(k);
    }
    groups.into_values().collect()
}

/// The reference's `_match_workers`, reduced to the one number a single process can act on.
///
/// The reference returns `(processes, threads per process, fragments per block)`; here the pool is
/// one and the pair schedule is the whole of the decision. Blocking pays for the `MatchData` a job
/// reuses and costs load balance, so it is taken only when there are many more blocks than
/// workers — the reference's own `n_jobs >= 16 * workers`, measured on pot H (D §5).
pub fn block_size(workers: usize, n_pairs: usize) -> usize {
    let workers = workers.max(1);
    if n_pairs > 4 * workers && n_pairs < 16 * workers { 1 } else { BLOCK }
}

/// R §4–§6 over a list of pairs, in the reference's block order, candidates back in pair order.
fn match_all(
    fragments: &[Fragment],
    pairs: &[(usize, usize)],
    params: &Params,
    keep: usize,
    workers: usize,
    tag: Option<&str>,
) -> Vec<Vec<Candidate>> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let blocks = pair_blocks(pairs, block_size(workers, pairs.len()));
    tracing::info!(
        pairs = pairs.len(),
        blocks = blocks.len(),
        largest = blocks.iter().map(Vec::len).max().unwrap_or(0),
        pass = tag.unwrap_or("first"),
        "matching"
    );
    let done = AtomicUsize::new(0);
    let mut out: Vec<Vec<Candidate>> = vec![Vec::new(); pairs.len()];
    let found: Vec<Vec<(usize, Vec<Candidate>)>> = blocks
        .par_iter()
        .map(|block| {
            block
                .iter()
                .map(|&k| {
                    let (a, b) = pairs[k];
                    let started = Instant::now();
                    let cs = pair::match_pair(&fragments[a], &fragments[b], params, keep);
                    tracing::info!(
                        pair = %format!("{}__{}", fragments[a].name, fragments[b].name),
                        seconds = started.elapsed().as_secs_f64(),
                        candidates = cs.len(),
                        accepted = cs.iter().filter(|c| c.accepted).count(),
                        at = done.fetch_add(1, Ordering::Relaxed) + 1,
                        of = pairs.len(),
                        "pair matched"
                    );
                    (k, cs)
                })
                .collect()
        })
        .collect();
    for (k, cs) in found.into_iter().flatten() {
        out[k] = cs;
    }
    out
}

/// R §4.3's partner search: the pairs worth matching, in the order they were given.
fn screen(
    fragments: &[Fragment],
    names: &[String],
    pairs: &[(usize, usize)],
    params: &Params,
) -> Vec<(usize, usize)> {
    // Each fragment's frames and breakline tree at its **own** `t` — R §4.3 is explicit that the
    // pass may not build a pair, which is the whole point of it.
    let frames: Vec<Frames> = fragments
        .iter()
        .map(|f| Frames {
            p: f.brk.points_f64(),
            ns: f.brk.ns.iter().map(|v| v.to_f64()).collect(),
            f: f.brk.f.iter().map(|v| v.to_f64()).collect(),
            tangent: f.brk.tangents_f64(),
            dih: f.brk.dihedrals(),
            sub: f.brk.sub.clone(),
        })
        .collect();
    let trees: Vec<Option<PointTree>> = frames.par_iter().map(|f| PointTree::build(&f.p)).collect();
    let view = |n: usize| {
        trees[n].as_ref().map(|tree| Screened {
            frames: &frames[n],
            tree,
            thick: fragments[n].thick,
            res: fragments[n].res(),
        })
    };
    let blocks = pair_blocks(pairs, BLOCK);
    let scored: Vec<Vec<(usize, f64)>> = blocks
        .par_iter()
        .map(|block| {
            block
                .iter()
                .map(|&k| {
                    let (a, b) = pairs[k];
                    let s = match (view(a), view(b)) {
                        (Some(a), Some(b)) => screen_pair(&a, &b, params),
                        _ => 0.0,
                    };
                    (k, s)
                })
                .collect()
        })
        .collect();
    let mut score: BTreeMap<(String, String), f64> = BTreeMap::new();
    for (k, s) in scored.into_iter().flatten() {
        let (a, b) = pairs[k];
        score.insert((names[a].clone(), names[b].clone()), s);
    }
    let keep = top_partners(&score, names, params.screen_top_k as usize);
    pairs
        .iter()
        .copied()
        .filter(|&(a, b)| keep.contains(&(names[a].clone(), names[b].clone())))
        .collect()
}

/// R §8.1: the best-scoring partners of every fragment the assembly left on its own.
///
/// "Best-scoring" is the pair's own `brk_best` — the only number a pair that produced nothing
/// still has — and the ties go to the pair that comes first by name, which is the reference's
/// `sorted(..., key=lambda x: (-x[0], x[1]))` over `(a, b)` name tuples.
fn second_pass_pairs(
    names: &[String],
    pairs: &[(usize, usize)],
    candidates: &[Candidate],
    groups: &[Vec<FragId>],
    params: &Params,
) -> Vec<(usize, usize)> {
    if params.second_pass_top == 0 {
        return Vec::new();
    }
    let placed: BTreeSet<FragId> =
        groups.iter().filter(|g| g.len() > 1).flatten().copied().collect();
    let lonely: Vec<FragId> = groups
        .iter()
        .filter(|g| g.len() == 1)
        .flatten()
        .copied()
        .filter(|n| !placed.contains(n))
        .collect();
    if lonely.is_empty() {
        return Vec::new();
    }
    let mut best: BTreeMap<(String, String), f64> = BTreeMap::new();
    for c in candidates {
        let key = (names[c.a as usize].clone(), names[c.b as usize].clone());
        let entry = best.entry(key).or_insert(0.0);
        *entry = entry.max(c.scores.brk_best);
    }
    let mut want: BTreeSet<(String, String)> = BTreeSet::new();
    for n in lonely {
        let name = &names[n as usize];
        let mut row: Vec<(f64, &(String, String))> = best
            .iter()
            .filter(|((a, b), _)| a == name || b == name)
            .map(|(k, &v)| (v, k))
            .collect();
        row.sort_by(|x, y| {
            y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| x.1.cmp(y.1))
        });
        want.extend(row.into_iter().take(params.second_pass_top as usize).map(|(_, k)| k.clone()));
    }
    pairs
        .iter()
        .copied()
        .filter(|&(a, b)| want.contains(&(names[a].clone(), names[b].clone())))
        .collect()
}

/// R §9 for a whole collection: one fracture cloud per member of a group of two or more, then the
/// spanning walk.
fn refine(
    fragments: &[Fragment],
    groups: &[Vec<FragId>],
    poses: &[Matrix4<f64>],
    used: &[(FragId, FragId)],
    params: &Params,
) -> Result<Vec<Matrix4<f64>>> {
    let in_group: Vec<bool> = {
        let mut flags = vec![false; fragments.len()];
        for group in groups.iter().filter(|g| g.len() > 1) {
            for &n in group {
                flags[n as usize] = true;
            }
        }
        flags
    };
    let clouds: Vec<Option<FractureCloud>> = fragments
        .par_iter()
        .zip(&in_group)
        .map(|(fragment, &wanted)| {
            if !wanted {
                return Ok(None);
            }
            let v64: Vec<[f64; 3]> = fragment.mesh.v.iter().map(|v| v.to_f64()).collect();
            let centroids = geometry::face_geometry(&v64, &fragment.mesh.f).centroids;
            let fracture: Vec<bool> = fragment.labels.iter().map(|l| l.is_fracture()).collect();
            let original = crate::io::load_mesh(&fragment.source.path)?;
            Ok(Some(fracture_cloud(
                &original,
                &centroids,
                &fracture,
                fragment.thick,
                fragment.res(),
                params.seed,
            )))
        })
        .collect::<Result<Vec<Option<FractureCloud>>>>()?;
    let pieces: Vec<RefinePiece<'_>> = fragments
        .iter()
        .zip(&clouds)
        .map(|(f, c)| RefinePiece { thick: f.thick, res: f.res(), cloud: c.as_ref() })
        .collect();
    Ok(refine_joins(&pieces, poses, groups, used, params, Numerics::REFERENCE).poses)
}

/// R §11.5: one preview per group of two or more, then the segmentation preview.
///
/// One `rng(0)` for the whole pass, drawn on in group order and then in collection order, exactly
/// as the reference's `write_previews` consumes it (R §10).
#[allow(clippy::cast_precision_loss, reason = "the offset is a fragment index")]
pub fn write_previews(
    out_dir: &Path,
    fragments: &[Fragment],
    names: &[String],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
) -> Result<Vec<PathBuf>> {
    /// R §11.5's sample count per fragment; the segmentation preview takes half of it.
    const PREVIEW_POINTS: usize = 250_000;

    let mut rng = crate::rng::seeded(0);
    let mut written = Vec::new();
    let geometry_of = |n: usize| {
        let fragment: &Fragment = &fragments[n];
        let v64: Vec<[f64; 3]> = fragment.mesh.v.iter().map(|v| v.to_f64()).collect();
        let geom = geometry::face_geometry(&v64, &fragment.mesh.f);
        let faces: Vec<u32> =
            (0..u32::try_from(fragment.mesh.f.len()).expect("fewer than 2^32 faces")).collect();
        (v64, geom, faces)
    };

    for (k, group) in groups.iter().enumerate() {
        if group.len() < 2 {
            continue;
        }
        let mut splats = Vec::new();
        for (i, &n) in group.iter().enumerate() {
            let n = n as usize;
            let (v64, geom, faces) = geometry_of(n);
            let (points, picks) = samples::sample_on_faces(
                &v64,
                &fragments[n].mesh.f,
                &geom.areas,
                &faces,
                PREVIEW_POINTS,
                &mut rng,
            );
            splats.push(render::placed_splat(
                &points,
                &picks,
                &geom.normals,
                &poses[n],
                Paint::Uniform(PALETTE[i % PALETTE.len()]),
            ));
        }
        let all: Vec<[f64; 3]> = splats.iter().flat_map(|s| s.points.iter().copied()).collect();
        let members: Vec<String> = group.iter().map(|&n| names[n as usize].clone()).collect();
        let mut image = render::render_views(&splats, &render::principal_views(&all), 900, 700);
        render::draw_label(&mut image, 10, 10, &render::group_label(&members));
        let file = out_dir.join(format!("preview_{k}.png"));
        image.write_png(&file)?;
        written.push(file);
    }

    let mut splats = Vec::new();
    for (n, fragment) in fragments.iter().enumerate() {
        let (v64, geom, faces) = geometry_of(n);
        let (points, picks) = samples::sample_on_faces(
            &v64,
            &fragment.mesh.f,
            &geom.areas,
            &faces,
            PREVIEW_POINTS / 2,
            &mut rng,
        );
        let mean = [0, 1, 2].map(|axis| {
            let column: Vec<f64> = points.iter().map(|p| p[axis]).collect();
            geometry::pairwise_sum(&column) / column.len().max(1) as f64
        });
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in &v64 {
            lo = lo.min(v[0]);
            hi = hi.max(v[0]);
        }
        let offset = n as f64 * 1.3 * (hi - lo);
        let labels = &fragment.labels;
        splats.push(Splat {
            points: points
                .iter()
                .map(|p| [p[0] - mean[0] + offset, p[1] - mean[1], p[2] - mean[2]])
                .collect(),
            normals: picks.iter().map(|&f| geom.normals[f as usize]).collect(),
            paint: Paint::PerPoint(
                picks
                    .iter()
                    .map(|&f| {
                        if labels[f as usize].is_fracture() {
                            [0.9, 0.2, 0.2]
                        } else {
                            [0.8, 0.8, 0.8]
                        }
                    })
                    .collect(),
            ),
        });
    }
    let all: Vec<[f64; 3]> = splats.iter().flat_map(|s| s.points.iter().copied()).collect();
    let views = render::principal_views(&all);
    let mut image = render::render_views(&splats, &views[..2], 1400, 600);
    render::draw_label(&mut image, 10, 10, &names.join(" "));
    let file = out_dir.join("preview_segmentation.png");
    image.write_png(&file)?;
    written.push(file);
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK, RunOptions, block_size, pair_blocks, preprocess, run, second_pass_pairs, set_threads,
    };
    use crate::collection::Entry;
    use crate::matching::pair::Candidate;
    use crate::matching::verify::Scores;
    use crate::params::Params;
    use crate::types::FragId;
    use nalgebra::Matrix4;

    #[test]
    fn an_unreadable_file_fails_only_itself() {
        let dir = std::env::temp_dir().join(format!("sherd-preprocess-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let broken = dir.join("broken.ply");
        std::fs::write(&broken, b"ply\nformat ascii 1.0\nend_header\n").unwrap();
        let entries = vec![Entry { path: broken, name: "broken".to_owned() }];
        let results = preprocess(&entries, 200_000, None);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "a mesh with no triangles is R §3.1's error case");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every pair of `n` fragments, in R §4.1's `itertools.combinations` order.
    fn all_pairs(n: usize) -> Vec<(usize, usize)> {
        (0..n).flat_map(|a| ((a + 1)..n).map(move |b| (a, b))).collect()
    }

    /// D §5's block schedule is the reference's `_pair_blocks`: the collection cut into blocks of
    /// three, one job per pair of blocks, and the jobs in sorted block order.
    #[test]
    fn pairs_are_grouped_three_fragments_against_three() {
        let pairs = all_pairs(6);
        let blocks = pair_blocks(&pairs, BLOCK);
        assert_eq!(blocks.len(), 3, "blocks (0,0), (0,1) and (1,1)");
        let named = |b: &Vec<usize>| b.iter().map(|&k| pairs[k]).collect::<Vec<(usize, usize)>>();
        assert_eq!(named(&blocks[0]), [(0, 1), (0, 2), (1, 2)], "inside the first block");
        assert_eq!(blocks[1].len(), 9, "every pair between the two blocks");
        assert_eq!(named(&blocks[2]), [(3, 4), (3, 5), (4, 5)], "inside the second");

        // Every pair lands in exactly one job, and the pair indices inside a job ascend, which is
        // what lets the results be scattered back by index.
        let mut seen: Vec<usize> = blocks.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..pairs.len()).collect::<Vec<usize>>());
        for block in &blocks {
            assert!(block.windows(2).all(|w| w[0] < w[1]), "{block:?}");
        }

        // A block of one degenerates to one job per pair, in pair order — the reference's
        // unblocked case.
        let one = pair_blocks(&pairs, 1);
        assert_eq!(one.len(), pairs.len());
        assert!(one.iter().enumerate().all(|(k, b)| b.as_slice() == [k]), "{one:?}");
        assert!(pair_blocks(&[], BLOCK).is_empty());
    }

    /// The reference blocks when the pairs cannot fill the machine on their own (threads go inside
    /// a pair) and again when there are many more blocks than workers; between the two it takes
    /// one pair per worker, because the last worker's second block is the whole tail.
    #[test]
    fn the_block_size_is_the_references_own_rule() {
        assert_eq!(block_size(10, 6), BLOCK, "terracotta: fewer pairs than cores");
        assert_eq!(block_size(10, 28), BLOCK, "pot A: 28 pairs is still under 4 x 10");
        assert_eq!(block_size(10, 41), 1, "one pair past 4 x 10 and the blocking stops");
        assert_eq!(block_size(10, 55), 1, "pot H: the case the reference measured");
        assert_eq!(block_size(10, 190), BLOCK, "synthetic 20: 190 >= 16 x 10");
        assert_eq!(block_size(10, 40), BLOCK, "exactly 4 x workers is still the cache case");
        assert_eq!(block_size(0, 1), BLOCK, "a pool of none is a pool of one");
    }

    /// R §8.1: the lonely fragments' best partners by `brk_best`, ties by pair name, and the
    /// retried list in R §4.1's pair order.
    #[test]
    fn the_second_pass_takes_the_lonely_fragments_best_partners() {
        let names: Vec<String> = ["a", "b", "c", "d"].map(str::to_owned).to_vec();
        let pairs = all_pairs(4);
        let candidate = |a: FragId, b: FragId, brk_best: f64| Candidate {
            a,
            b,
            transform: Matrix4::identity(),
            scores: Scores { brk_best, ..Scores::default() },
            accepted: false,
        };
        let candidates = vec![
            candidate(0, 1, 0.1),
            candidate(0, 2, 0.4),
            candidate(0, 3, 0.2),
            candidate(1, 2, 0.9),
            candidate(1, 3, 0.3),
            candidate(2, 3, 0.5),
        ];
        // `b` and `c` were assembled; `a` and `d` were left on their own.
        let groups = vec![vec![1, 2], vec![0], vec![3]];
        let two = Params { second_pass_top: 2, ..Params::default() };
        assert_eq!(
            second_pass_pairs(&names, &pairs, &candidates, &groups, &two),
            [(0, 2), (0, 3), (1, 3), (2, 3)],
            "a's best two are c (0.4) and d (0.2), d's are c (0.5) and b (0.3)"
        );
        assert_eq!(
            second_pass_pairs(&names, &pairs, &candidates, &groups, &Params::default()),
            [],
            "second_pass_top = 0 is the default and disables the pass"
        );
        assert_eq!(
            second_pass_pairs(&names, &pairs, &candidates, &[vec![0, 1, 2, 3]], &two),
            [],
            "nothing is lonely, so there is nothing to retry"
        );

        // A tie between two partners of the same fragment goes to the one that comes first by
        // pair name, which is the reference's `sorted(..., key=(-score, pair))`. Only `a` is
        // lonely here, so the answer is that one fragment's row and nothing else.
        let tied = vec![candidate(0, 2, 0.5), candidate(0, 3, 0.5)];
        let one = Params { second_pass_top: 1, ..Params::default() };
        assert_eq!(
            second_pass_pairs(&names, &pairs, &tied, &[vec![1, 2, 3], vec![0]], &one),
            [(0, 2)],
            "`a__c` before `a__d`"
        );
    }

    /// R §2's error case for a run: a directory with fewer than two meshes names itself.
    #[test]
    fn a_collection_of_one_is_not_a_run() {
        let dir = std::env::temp_dir().join(format!("sherd-run-one-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("only.ply"), b"ply\n").unwrap();
        let out = dir.join("out");
        let err = run(&dir, &out, &RunOptions::default()).unwrap_err().to_string();
        assert!(err.contains("need at least two mesh files, found 1"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The defaults of `RunOptions` are the reference's `pipeline.run` defaults.
    #[test]
    fn the_run_defaults_are_the_references() {
        let options = RunOptions::default();
        assert_eq!(options.target_faces, 200_000);
        assert_eq!(options.keep_per_pair, 5, "R §5.7's `keep`, not a flag on either side");
        assert!(options.preview && options.refine && options.write_meshes && options.cache);
        assert_eq!(options.params, Params::default());
    }

    #[test]
    fn zero_threads_leaves_the_pool_alone() {
        assert!(set_threads(0).is_ok());
    }
}
