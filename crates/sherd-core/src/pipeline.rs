//! The run: discovery, preprocessing, matching, assembly, refinement, outputs (D §5).
//!
//! One process, one rayon pool. Preprocessing is a `par_iter` over fragments bounded by a
//! memory-aware semaphore ([`memory`](crate::memory)): a scan of `f` faces reserves E1's measured
//! `361 B·f` from `--memory-budget`, whose default is half of physical RAM. Matching walks the same 3×3 blocks of the collection order the
//! reference walks, so the pair order — and with it every seeded draw — is the reference's;
//! candidates inside a pair run in parallel and are collected by index, so nothing depends on the
//! schedule. Cancellation is an `AtomicBool` checked between units of work; progress is a
//! callback.
//!
//! Step S4 filled in the first stage: [`preprocess`], which is what `sherd-refit-rs segment`
//! drives — R §3.1–3.3 then, R §3.1–3.4 since step B1. Step E2 put the semaphore in front of it,
//! once E1 had measured the per-fragment high-water mark rather than guessing it: the fan-out is
//! still a `par_iter` over the collection with `--threads` sizing the pool, and the semaphore only
//! decides when a job may start.
//!
//! Step H3 added two things the audit's §B.2 and §B.3 asked for and one it asked for in §C.3:
//! R §9's refinement takes D §5 step 2's reservation for the original scan it reads, as
//! preprocessing and R §11.4's writers do; both BVHs are released once their last reader is done,
//! which is after the second pass and before the two stages that hold an original per job; and
//! every stage records the peak resident set beside its seconds ([`StageLog`]), which is the
//! measurement the second of those decisions was made on. R §10's seed reaches R §3.5's samplers
//! from `Params::seed`, which is what `--seed` sets.
//!
//! Step D3 filled in the rest: [`run`] is the reference's `sherd_refit.pipeline.run`, stage for
//! stage, and the schedule below it is the reference's too — [`pair_blocks`] is `_pair_blocks` and
//! [`block_size`] the one number `_match_workers` leaves for a single process to act on. Step E2
//! added D §5's per-fragment `MatchData` cache and task H2 removed it (audit §B.8): E1 had
//! measured what it can remove — the cheap half of a pair's two builds, because R §1.2's
//! `t_pair = min(t_A, t_B)` gives every *rebuild* a key that belongs to one partner alone — and E2
//! measured what it was worth, 0.9 % of CPU and nothing on the wall clock. The block order it was
//! written for stays: it is the reference's own pair order.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nalgebra::Matrix4;
use rayon::prelude::*;

use crate::assembly::constraints::{self, Constraints, Resolved};
use crate::assembly::{Piece, assemble_under, recenter};
use crate::collection::{self, Entry};
use crate::error::{Error, Result};
use crate::executor::{Backend, Engine, Executor};
use crate::fragment::{Fragment, cache, samples};
use crate::matching::hypotheses::Frames;
use crate::matching::pair::{self, Candidate, Gate};
use crate::matching::screen::{Screened, screen_pair, top_partners};
use crate::memory::{self, Budget, MemorySemaphore, reservation};
use crate::mesh::geometry;
use crate::objects;
use crate::params::Params;
use crate::progress::Watch;
use crate::refine::{self, FractureCloud, RefinePiece, fracture_cloud, refine_joins};
use crate::render::{self, PALETTE, Paint, Splat};
use crate::report::{
    FragmentStats, MemoryReport, Outcome, Timings, write_placed_meshes, write_report,
    write_transforms,
};
use crate::spatial::kdtree::PointTree;
use crate::tiers::Tier;
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
    budget: Budget,
    seed: u64,
) -> Vec<Result<Preprocessed>> {
    preprocess_watched(entries, target_faces, out_dir, budget, seed, &Watch::default())
}

/// [`preprocess`] under D §5's cancellation flag and progress callback.
///
/// The check is at the top of each fragment's job and the report is at the bottom of it, so a
/// cancelled run leaves no half-built fragment and writes no half-built cache: a job either ran
/// whole or did not start. Jobs already in flight when the flag goes up finish — which is what
/// makes the guarantee true rather than nearly true.
pub fn preprocess_watched(
    entries: &[Entry],
    target_faces: usize,
    out_dir: Option<&Path>,
    budget: Budget,
    seed: u64,
    watch: &Watch,
) -> Vec<Result<Preprocessed>> {
    let semaphore = MemorySemaphore::new(budget);
    let finished = AtomicUsize::new(0);
    let out: Vec<Result<Preprocessed>> = entries
        .par_iter()
        .map(|entry| {
            watch.check()?;
            let started = std::time::Instant::now();
            // D §5 step 2: reserve what E1's model says this scan will add to the process's peak
            // RSS, and wait for it. A file whose size cannot be read reserves nothing — the load
            // below is about to fail with the reader's own error, which is the better one.
            let permit = semaphore.acquire(memory::scan_faces(&entry.path).map_or(0, reservation));
            let cache_path = out_dir.map(|dir| cache::cache_path(dir, &entry.name));
            let (fragment, cached) = Fragment::load_or_build(
                &entry.path,
                target_faces,
                &entry.name,
                cache_path.as_deref(),
                seed,
            )?;
            drop(permit);
            watch.advance(
                "preprocess",
                finished.fetch_add(1, Ordering::Relaxed) + 1,
                entries.len(),
            );
            Ok(Preprocessed { fragment, cached, seconds: started.elapsed().as_secs_f64() })
        })
        .collect();
    if budget.is_bounded() {
        let stats = semaphore.stats();
        tracing::info!(
            budget_mib = budget.available() / (1024 * 1024),
            peak_mib = stats.peak / (1024 * 1024),
            peak_concurrent = stats.peak_running,
            waited = stats.waited,
            "preprocessing memory"
        );
    }
    out
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
    /// Which executor ran, for D §4.3's `engine` block.
    ///
    /// The **resolved** one: the CLI turns `--backend auto` into `Cpu` or `Gpu` before it gets
    /// here, because a file that recorded `auto` would say nothing about the arithmetic that
    /// produced it.
    pub backend: Backend,
    /// The adapter's own name when the run resolved to a device, for the same block.
    ///
    /// D §4.3 spells the field `gpu:Apple M2 Pro`, and the reason is attribution: the two backends
    /// agree within D §10.2 and not to the bit, so a pose that has to be explained years later
    /// needs to name the machine and not merely "the GPU".
    pub adapter: Option<String>,
    /// D §5 step 2's preprocessing memory budget (D §9's `--memory-budget`).
    pub memory: Budget,
    /// D §5's cancellation flag and progress callback; neither by default
    /// ([`progress`](crate::progress)).
    pub watch: Watch,
    /// Roadmap step 7's measurement file (audit §D.1, [`crate::measure`]), or `None`.
    ///
    /// `Some(path)` adds one pass **after** R §8's assembly and one file; it reads what the run
    /// has already computed and writes nothing else, so a run with it unset is byte for byte the
    /// run that came before the flag existed.
    pub measure: Option<PathBuf>,
    /// Roadmap item 3's operator constraints ([`crate::assembly::constraints`]), already parsed.
    ///
    /// The pipeline resolves the names against the collection and fails the run on an unknown one;
    /// the CLI reads the file, so that a caller with the constraints in hand needs no file at all.
    /// `None` is a run with no `constraints.json`, which is byte for byte the run before the flag.
    pub constraints: Option<Constraints>,
    /// Write audit §D.1's review images ([`crate::review`]) for the confirmed and probable joins.
    pub review_images: bool,
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
            adapter: None,
            memory: Budget::default_for_machine(),
            watch: Watch::default(),
            measure: None,
            constraints: None,
            review_images: false,
        }
    }
}

impl RunOptions {
    /// D §4.3's `backend` field: the resolved executor, with the adapter's name when it was a
    /// device — `cpu`, or `gpu:Apple M2 Pro`.
    #[must_use]
    pub fn backend_label(&self) -> String {
        match &self.adapter {
            Some(name) => format!("{}:{name}", self.backend.as_str()),
            None => self.backend.as_str().to_owned(),
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
    /// R §11.2's `timings`, in seconds, in the order the stages finished.
    pub timings: Timings,
    /// Peak resident set of the process, per stage, when this platform reports one (audit §B.3).
    pub memory: Option<MemoryReport>,
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

/// The reference's default worker count: **one per core minus one**.
///
/// `cli.py` declares `--workers default=None` and `pipeline.run` resolves it to
/// `workers or max(1, (os.cpu_count() or 2) - 1)` — nine on a ten-core machine, not ten. D §9 asks
/// for the same default and the port used one per core until V4-D8. Nothing about the results
/// depends on it (every parallel section collects by index, and R §4.2's block size comes out the
/// same at nine and at ten on all six benchmark sets); it is the flag's documented meaning that
/// has to match.
pub fn default_workers() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get).saturating_sub(1).max(1)
}

/// The two numbers a stage leaves behind: its wall clock, and the largest resident set seen while
/// it ran.
///
/// The pipeline used to write `timings.insert(stage, seconds)` at the end of every stage; the
/// audit's §B.3 wants the peak RSS beside it, on every run and not in a one-off experiment, and
/// one call is the way to keep the two lists in the same order and to make it impossible to record
/// a timing without its memory. A machine whose resident set cannot be read keeps the timings and
/// reports no memory at all.
#[derive(Debug, Default)]
struct StageLog {
    timings: Timings,
    peaks: crate::report::Ordered<u64>,
    monitor: Option<crate::memory::RssMonitor>,
}

impl StageLog {
    /// Starts the clock's companion. The sampler runs until this is dropped.
    fn start() -> Self {
        Self { monitor: crate::memory::RssMonitor::start(), ..Self::default() }
    }

    /// Closes `stage`: its seconds, its peak resident set, and the log line an operator reads.
    fn finish(&mut self, stage: &str, seconds: f64) {
        self.timings.insert(stage, seconds);
        let Some(monitor) = &self.monitor else {
            return;
        };
        let peak = monitor.take_peak();
        self.peaks.insert(stage, peak);
        tracing::info!(stage, seconds, peak_mib = peak / (1024 * 1024), "stage done");
    }

    /// `report.json`'s `memory` block — `timings`' neighbour — or `None` where the resident
    /// set cannot be read.
    fn memory(&self) -> Option<MemoryReport> {
        let monitor = self.monitor.as_ref()?;
        Some(MemoryReport { peak_rss: monitor.peak(), stages: self.peaks.clone() })
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
    run_with(input, out_dir, options, Engine::REFERENCE)
}

/// [`run`] on a chosen executor (D §6.1).
///
/// `sherd-core` has no GPU dependency and cannot build a `GpuExecutor`, so the executor is handed
/// in: the CLI resolves `--backend`, `sherd-gpu` builds the device, and the pipeline never learns
/// which of the two it is holding. `options.backend` is what the CLI *resolved* `--backend` to,
/// and with `options.adapter` it is what both output files record (D §4.3); `engine.exec` is the
/// executor those batches actually go to.
#[allow(clippy::too_many_lines, reason = "the reference's `pipeline.run`, stage for stage")]
pub fn run_with(
    input: &Path,
    out_dir: &Path,
    options: &RunOptions,
    engine: Engine<'_>,
) -> Result<RunSummary> {
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
    let mut stages = StageLog::start();
    tracing::info!(
        fragments = entries.len(),
        input = %input.display(),
        workers,
        "collection discovered"
    );

    // 1. preprocessing (R §3, through the cache of R §3.7)
    let started = Instant::now();
    let cache_dir = options.cache.then(|| out_dir.to_path_buf());
    let mut fragments = preprocess_collection(
        &entries,
        options.target_faces,
        cache_dir.as_deref(),
        options.memory,
        // R §10 seeds `rng_md` from `Params.seed`; `--seed` (task H3) is what sets it, and this is
        // the one place preprocessing learns it. A cache written at another seed has R §3.5's
        // three arrays recomputed and nothing else, which is R §3.7's rule.
        params.seed,
        &options.watch,
    )?;
    stages.finish("preprocess", started.elapsed().as_secs_f64());
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
        seconds = stages.timings["preprocess"],
        thickness,
        resolution,
        edges_per_t = thickness / resolution.max(1e-9),
        "preprocessing done"
    );

    // 1a. the operator's constraints (audit §D.1, roadmap item 3), resolved against the names the
    // collection actually has. An unknown name fails the run here, before a single pair is
    // matched, because a typo that quietly became "no constraint" is the failure this validation
    // exists to prevent.
    let plan: Option<Resolved> = match &options.constraints {
        Some(file) => {
            let resolved = constraints::resolve(file, &names)?;
            tracing::info!(
                must_join = resolved.forced.len(),
                must_not_join = resolved.forbidden.len(),
                same_object = resolved.same.len(),
                different_object = resolved.different.len(),
                "constraints"
            );
            Some(resolved)
        }
        None => None,
    };
    let id = |n: usize| u32::try_from(n).expect("fewer than 2^32 fragments");

    // 2. pairs (R §4.1), with the two lists that decide before R §4.1 does: `must_not_join` takes
    // its pairs out of the run entirely, a pinned `must_join` skips matching because it already
    // has its pose, and a `must_join` without one is matched even where R §4.1's wall-ratio filter
    // would have refused it — the operator's word outranks a heuristic about wall thicknesses.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut skipped = 0_usize;
    let mut forbidden = 0_usize;
    let mut pinned_pairs = 0_usize;
    let mut forced_over_ratio = 0_usize;
    for a in 0..fragments.len() {
        for b in (a + 1)..fragments.len() {
            let forced = plan.as_ref().and_then(|r| r.forced_pair(id(a), id(b)));
            if plan.as_ref().is_some_and(|r| r.forbids(id(a), id(b))) {
                forbidden += 1;
            } else if forced.is_some_and(|f| f.pose.is_some()) {
                pinned_pairs += 1;
            } else if pair::Pair::skipped(&fragments[a], &fragments[b], params) {
                if forced.is_some() {
                    forced_over_ratio += 1;
                    pairs.push((a, b));
                } else {
                    skipped += 1;
                }
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
    if forbidden + pinned_pairs + forced_over_ratio > 0 {
        tracing::info!(
            forbidden,
            pinned = pinned_pairs,
            matched_over_ratio = forced_over_ratio,
            "pairs decided by constraints.json"
        );
    }

    // 2a. partner search (R §4.3, off by default)
    let mut screened = None;
    if params.screen_top_k > 0 && pairs.len() >= params.screen_min_pairs as usize {
        let started = Instant::now();
        let before = pairs.len();
        pairs = screen(engine.exec, &fragments, &names, &pairs, params);
        stages.finish("screen", started.elapsed().as_secs_f64());
        screened = Some((before, pairs.len()));
        tracing::info!(
            screened = before,
            kept = pairs.len(),
            top_k = params.screen_top_k,
            seconds = stages.timings["screen"],
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
    let mut per_pair = match_all(
        engine,
        &fragments,
        &pairs,
        params,
        options.keep_per_pair,
        workers,
        None,
        &options.watch,
    )?;
    stages.finish("matching", started.elapsed().as_secs_f64());

    // 2b'. `must_join` without a pose: audit §D.1 asks for the pair to be *"matched with the
    // second-pass budget"*, which is R §8.1's larger `stage1`/`stage2` with the stage-1 floor
    // removed. It is a handful of pairs, matched a second time rather than promoted out of the
    // first pass, because the budget is the whole point of the sentence.
    let forced_retry: Vec<(usize, usize)> = plan
        .as_ref()
        .map(|r| {
            pairs
                .iter()
                .copied()
                .filter(|&(a, b)| r.forced_pair(id(a), id(b)).is_some_and(|f| f.pose.is_none()))
                .collect()
        })
        .unwrap_or_default();
    if !forced_retry.is_empty() {
        let started = Instant::now();
        let found = match_all(
            engine,
            &fragments,
            &forced_retry,
            &second_pass_params(params),
            options.keep_per_pair,
            workers,
            Some("must_join"),
            &options.watch,
        )?;
        for (k, list) in forced_retry.iter().zip(found) {
            let at = pairs.iter().position(|p| p == k).expect("a forced pair is a pair");
            per_pair[at] = list;
        }
        tracing::info!(
            pairs = forced_retry.len(),
            seconds = started.elapsed().as_secs_f64(),
            "must_join rematched with the second-pass budget"
        );
    }

    // 2b''. `must_join` with a pose: matching was skipped for the pair, so its candidate is made
    // here. R §6 is run **at the pinned pose** so that the report carries an honest opinion of it;
    // acceptance is the operator's, which is what pinning a pose means, and the constraints
    // section says so beside the numbers.
    let pinned: Vec<Candidate> = plan
        .as_ref()
        .map(|r| {
            r.forced
                .iter()
                .filter_map(|f| {
                    f.pose.map(|pose| pinned_candidate(engine, &fragments, f.a, f.b, &pose, params))
                })
                .collect()
        })
        .unwrap_or_default();
    let collect_candidates = |per_pair: &[Vec<Candidate>]| -> Vec<Candidate> {
        let mut all: Vec<Candidate> = per_pair.iter().flatten().copied().collect();
        all.extend(pinned.iter().copied());
        all
    };
    let mut candidates = collect_candidates(&per_pair);
    tracing::info!(
        seconds = stages.timings["matching"],
        candidates = candidates.len(),
        accepted = candidates.iter().filter(|c| c.accepted).count(),
        "matching done"
    );

    // 2c. the confidence tier (audit §D.1, roadmap item 3), off unless `Params::tiers` is set.
    //
    // Here and not after R §8: the assembly is built from the **confirmed** joins, so the tier has
    // to be decided first. It needs nothing the assembly produces — the support count walks the
    // accepted-candidate graph and not the groups — and it needs R §6.1's fracture BVHs, which are
    // alive from the last pair until the block that releases them below.
    let mut tiered = tier_pass(engine, &fragments, &mut candidates, params, &mut stages);
    let gate = if params.tiers.is_some() { Gate::Confirmed } else { Gate::Accepted };
    // 2d. and then the half of the constraints that acts on the candidate list: a `must_join` is
    // promoted to the confirmed band. `assemble_under` below does the other half.
    let mut honoured = plan.as_ref().map(|r| {
        constraints::apply(r, &names, &mut candidates, tiered.as_mut(), gate == Gate::Confirmed)
    });

    // 3. assembly (R §8). The timer starts here, not at `assemble`: the reference builds the
    // stage's own `MatchData` inside `timings["assembly"]`, and PMC-8's cheaper equivalent — the
    // fragments' own samples and their whole-mesh BVHs — belongs in the same place.
    let started = Instant::now();
    let samples: Vec<Vec<[f64; 3]>> = fragments.iter().map(|f| f.samples.surface_f64()).collect();
    // The whole-mesh BVHs are built eagerly and then held as `Arc`s of their own rather than as
    // borrows of the fragments. That is what lets the fracture trees be released between the last
    // pair and R §9 below (audit §B.3): a `Piece` list borrowed from `fragments` would keep the
    // collection immutably borrowed to the end of the run.
    let scenes: Vec<Option<std::sync::Arc<crate::spatial::bvh::RayScene>>> =
        fragments.par_iter().map(Fragment::surface_scene_arc).collect();
    let pieces: Vec<Piece<'_>> = fragments
        .iter()
        .zip(&samples)
        .zip(&scenes)
        .map(|((f, s), scene)| Piece {
            thick: f.thick,
            res: f.res(),
            watertight: f.watertight,
            mesh: scene.as_deref(),
            s_pen: s,
        })
        .collect();
    let mut assembly = assemble_under(
        engine.exec,
        &pieces,
        &candidates,
        params,
        gate,
        plan.as_ref(),
        params.objects.as_ref(),
    );
    stages.finish("assembly", started.elapsed().as_secs_f64());

    // 3'. roadmap item 4's object pass (audit §D.2), off unless `Params::objects` is set.
    //
    // Here and not before R §8, because a consensus needs the groups R §8 builds and a
    // disagreement needs the poses it placed. It runs **one** round: the demotions it finds are
    // applied to the bands and R §8 is run again on them, and no further, because a second round's
    // demotions would be computed on the groups the first round's demotions built and the answer
    // would then depend on how many rounds were run.
    let mut object_pass = object_round(
        &fragments,
        &names,
        &mut candidates,
        &mut assembly,
        &pieces,
        ObjectRound {
            params,
            gate,
            constraints: plan.as_ref(),
            tiers: tiered.as_mut(),
            exec: engine.exec,
            stages: &mut stages,
        },
    );

    // 3a. second pass (R §8.1, off by default)
    let retry = second_pass_pairs(&names, &pairs, &candidates, &assembly.groups, params);
    if !retry.is_empty() {
        let started = Instant::now();
        let bigger = second_pass_params(params);
        let again = match_all(
            engine,
            &fragments,
            &retry,
            &bigger,
            options.keep_per_pair,
            workers,
            Some("second"),
            &options.watch,
        )?;
        for (k, found) in retry.iter().zip(again) {
            let at = pairs.iter().position(|p| p == k).expect("a retried pair is a pair");
            per_pair[at] = found;
        }
        candidates = collect_candidates(&per_pair);
        stages.finish("second_pass", started.elapsed().as_secs_f64());
        // The candidate list is a different list, so its tiers are a different answer: the
        // margin, the placements and the support count are all statements about the list as a
        // whole, and a pair rematched with the larger budget changes them for pairs it never
        // touched. The pass is repeated rather than patched for that reason.
        tiered = tier_pass(engine, &fragments, &mut candidates, params, &mut stages);
        honoured = plan.as_ref().map(|r| {
            constraints::apply(r, &names, &mut candidates, tiered.as_mut(), gate == Gate::Confirmed)
        });
        assembly = assemble_under(
            engine.exec,
            &pieces,
            &candidates,
            params,
            gate,
            plan.as_ref(),
            params.objects.as_ref(),
        );
        object_pass = object_round(
            &fragments,
            &names,
            &mut candidates,
            &mut assembly,
            &pieces,
            ObjectRound {
                params,
                gate,
                constraints: plan.as_ref(),
                tiers: tiered.as_mut(),
                exec: engine.exec,
                stages: &mut stages,
            },
        );
        tracing::info!(
            pairs = retry.len(),
            seconds = stages.timings["second_pass"],
            stage1 = bigger.stage1,
            stage2 = bigger.stage2,
            accepted = candidates.iter().filter(|c| c.accepted).count(),
            groups = assembly.groups.iter().filter(|g| g.len() > 1).count(),
            "second pass"
        );
    }
    let used: Vec<(FragId, FragId)> =
        assembly.used.iter().map(|&i| (candidates[i].a, candidates[i].b)).collect();

    // 3b. roadmap step 7's measurement (audit §D.1), off unless `--measure` asked for it.
    //
    // Here and not later: it needs R §6.1's fracture BVHs, which the next block releases, and
    // R §8's `used`, which the line above has just resolved. It reads the run and writes one
    // file; nothing below it changes because of it.
    let mut measured = None;
    if let Some(path) = &options.measure {
        let started = Instant::now();
        let report = crate::measure::measure(
            engine,
            &fragments,
            &candidates,
            &assembly.used,
            params,
            thickness,
            tiered.as_ref().map(|t| t.probes.as_slice()),
        );
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| Error::write(path, std::io::Error::other(e)))?;
        std::fs::write(path, json).map_err(|e| Error::write(path, e))?;
        tracing::info!(
            rows = report.rows.len(),
            seconds = started.elapsed().as_secs_f64(),
            out = %path.display(),
            "measurement written"
        );
        measured = Some(path.clone());
    }

    // 3c. audit §D.1's review images, off unless `--review-images` asked for them.
    //
    // Here for the same reason the measurement is: the contact colouring is R §6.1's own distance
    // against A's fracture tree, and the next block releases it.
    let mut review = None;
    let mut review_files: Vec<PathBuf> = Vec::new();
    if options.review_images {
        let started = Instant::now();
        let (files, index) = crate::review::write_review_images(
            engine,
            out_dir,
            &fragments,
            &names,
            &candidates,
            tiered.as_ref(),
            object_pass.as_ref(),
            params,
        )?;
        stages.finish("review", started.elapsed().as_secs_f64());
        review_files = files;
        review = Some(index);
    }

    // Both BVHs have had their last reader (audit §B.3): R §6.1's fracture tree ended with the
    // last pair, R §6.4's whole-mesh tree with the last `try_place`. R §8.2's recentring reads
    // only the surface samples, so what follows — R §9's refinement and R §11's writers, the two
    // stages that hold one full-resolution original per job — does not have to make room for
    // 4.4 GB of trees on a 170-fragment collection. The `Piece` list is dropped with them, because
    // it holds the whole-mesh trees' `Arc`s; `placed` is R §8.2's own view, without them.
    let before = crate::memory::resident_memory();
    drop(pieces);
    drop(scenes);
    for fragment in &mut fragments {
        fragment.release_scenes();
    }
    if let (Some(before), Some(after)) = (before, crate::memory::resident_memory()) {
        tracing::info!(
            rss_before_mib = before / (1024 * 1024),
            rss_after_mib = after / (1024 * 1024),
            reclaimed_mib = before.saturating_sub(after) / (1024 * 1024),
            "BVHs released"
        );
    }
    let placed: Vec<Piece<'_>> = fragments
        .iter()
        .zip(&samples)
        .map(|(f, s)| Piece {
            thick: f.thick,
            res: f.res(),
            watertight: f.watertight,
            mesh: None,
            s_pen: s,
        })
        .collect();

    // 4. full-resolution refinement (R §9) and R §8.2's recentring
    let mut poses = assembly.poses.clone();
    if options.refine && assembly.groups.iter().any(|g| g.len() > 1) {
        let started = Instant::now();
        poses =
            refine(engine, &fragments, &assembly.groups, &poses, &used, params, options.memory)?;
        stages.finish("refine", started.elapsed().as_secs_f64());
        tracing::info!(seconds = stages.timings["refine"], joins = used.len(), "refinement done");
    }
    let poses = recenter(&poses, &placed, &assembly.groups);

    // 5. outputs (R §11)
    let started = Instant::now();
    let mut written = Vec::new();
    written.extend(measured);
    written.extend(review_files);
    let stats: Vec<FragmentStats> = fragments.iter().map(FragmentStats::of).collect();
    let rejected: Vec<(usize, String)> =
        assembly.rejected.iter().map(|r| (r.candidate, r.reason.message(&names))).collect();
    let outcome = Outcome {
        names: &names,
        candidates: &candidates,
        used: &assembly.used,
        rejected: &rejected,
        groups: &assembly.groups,
        tiers: tiered.as_ref(),
        constraints: honoured.as_ref(),
        review: review.as_ref(),
        objects: object_pass.as_ref(),
    };
    let tier_joins = tiered.as_ref().map(|_| crate::tiers::joins(&candidates, &names));
    write_transforms(
        out_dir.join("transforms.json"),
        &names,
        &poses,
        &assembly.groups,
        &assembly.order,
        thickness,
        params,
        Some(&options.backend_label()),
        tier_joins.as_deref(),
    )?;
    written.push(out_dir.join("transforms.json"));
    write_report(
        out_dir,
        &stats,
        thickness,
        &outcome,
        &stages.timings,
        params,
        &options.backend_label(),
        stages.memory().as_ref(),
    )?;
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
            options.memory,
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
    stages.finish("output", started.elapsed().as_secs_f64());
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
        memory: stages.memory(),
        timings: stages.timings,
        written,
    })
}

/// R §8.1's larger budget, which audit §D.1 also gives a `must_join` pair.
fn second_pass_params(params: &Params) -> Params {
    Params {
        stage1: params.second_pass_stage1,
        stage2: params.second_pass_stage2,
        stage1_floor: 0.0,
        ..*params
    }
}

/// The candidate a `must_join` with a pose stands for: R §6 at the pinned pose, accepted by the
/// operator's word and confirmed by it (audit §D.1).
///
/// The scores are measured and not invented — a conservator who pins a pose still wants to see
/// what the geometry thinks of it, and `report.md`'s constraints section prints the verdict R §6.5
/// would have reached beside the fact that the pose was pinned. A pair one of whose fragments has
/// no surface at all cannot be scored; the candidate then carries R §5.6's partial scores and
/// `accepted = false`, which the constraints section reports as unsatisfiable.
fn pinned_candidate(
    engine: Engine<'_>,
    fragments: &[Fragment],
    a: FragId,
    b: FragId,
    pose: &Matrix4<f64>,
    params: &Params,
) -> Candidate {
    use crate::matching::verify::{Scores, verify};
    let (fa, fb) = (&fragments[a as usize], &fragments[b as usize]);
    let built = pair::Pair::build(fa, fb, params);
    let sc = built.scales;
    match built.surfaces() {
        Some((sa, sb)) => {
            let scores = verify(engine.exec, &sa, &sb, pose, &sc, true, None);
            Candidate {
                a,
                b,
                transform: *pose,
                scores,
                accepted: true,
                tier: crate::tiers::Tier::Confirmed,
            }
        }
        None => Candidate {
            a,
            b,
            transform: *pose,
            scores: Scores::partial(&sc, 0.0),
            accepted: false,
            tier: crate::tiers::Tier::Rejected,
        },
    }
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
    budget: Budget,
    seed: u64,
    watch: &Watch,
) -> Result<Vec<Fragment>> {
    let results = preprocess_watched(entries, target_faces, out_dir, budget, seed, watch);
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

/// D §6.4's software pipeline: `body` on a pool deep enough that a block waiting on a device is
/// not a core standing still.
///
/// The design sentence is *"CPU threads prepare block k+1 while the GPU runs block k;
/// double-buffered"*, and this is what it comes to in a pipeline whose unit of work is a pair. A
/// worker that hands a batch to the device blocks until the device answers — measured on
/// `synthetic_20`, the device had work outstanding for 11.3 s of a 15.7 s matching stage while the
/// pool used 78 of the 157 core-seconds the stage could have burnt. Neither side was the limit;
/// the *alternation* was. [`Executor::device_slack`] says how many extra blocks have to be in
/// flight for the two to overlap, and the CPU executor says none.
///
/// It cannot move a result. Every parallel section under it collects by index, R §4.2's block size
/// comes from `--workers` rather than from the pool, and the pairs of a block still run in the
/// reference's order inside one task.
fn with_device_slack<T: Send>(exec: &dyn Executor, body: impl FnOnce() -> T + Send) -> T {
    let slack = exec.device_slack();
    if slack == 0 {
        return body();
    }
    let threads = rayon::current_num_threads() + slack;
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("sherd-match-{i}"))
        .build()
    {
        Ok(pool) => {
            tracing::info!(threads, slack, "matching on a device-slack pool");
            pool.install(body)
        }
        Err(e) => {
            tracing::warn!(error = %e, "could not build the device-slack pool; using the run's");
            body()
        }
    }
}

/// Audit §D.1's tier pass over a finished candidate list, writing each band onto its candidate.
///
/// `None` — and no stage, no log line and no cost — when [`Params::tiers`] is `None`, which is the
/// library default and what `--tiers off` sets: every candidate then keeps
/// [`Tier::of_accept`](crate::tiers::Tier::of_accept)'s band, R §8's gate is `accepted`, and the
/// run is byte for byte the run it was before this pass existed.
fn tier_pass(
    engine: Engine<'_>,
    fragments: &[Fragment],
    candidates: &mut [Candidate],
    params: &Params,
    stages: &mut StageLog,
) -> Option<crate::tiers::TierReport> {
    let thresholds = params.tiers?;
    let started = Instant::now();
    let report = crate::tiers::classify(engine, fragments, candidates, params, &thresholds);
    for (c, &tier) in candidates.iter_mut().zip(&report.tiers) {
        c.tier = tier;
    }
    stages.finish("tiers", started.elapsed().as_secs_f64());
    let (confirmed, probable, rejected) = report.counts();
    tracing::info!(
        confirmed,
        probable,
        rejected,
        seconds = stages.timings["tiers"],
        "tiers decided"
    );
    Some(report)
}

/// What one [`object_round`] reads besides the collection and the assembly.
///
/// A struct because the round needs the run's parameters, its gate, its constraints, its tier
/// report, its executor and its stage log, and eleven arguments in a row is where a pass stops
/// being readable.
struct ObjectRound<'a> {
    params: &'a Params,
    gate: Gate,
    constraints: Option<&'a Resolved>,
    tiers: Option<&'a mut crate::tiers::TierReport>,
    exec: &'a dyn crate::executor::Executor,
    stages: &'a mut StageLog,
}

/// Roadmap item 4's pass over a finished assembly (audit §D.2), off unless [`Params::objects`] is
/// set.
///
/// Three things happen here and in this order:
///
/// 1. **The demotions** ([`objects::demotions`]) — the per-group consensus and the
///    mutual-disagreement rule. A demotion moves a candidate from [`Tier::Confirmed`] to
///    [`Tier::Probable`] and writes its sentence into the candidate's evidence, so `report.md`'s
///    probable row names the object rule that refused it beside the tier tests it passed.
/// 2. **R §8 again**, but only if something was demoted: the assembly is built from confirmed
///    joins and fewer of them are confirmed now. One round, not to a fixed point.
/// 3. **The objects themselves** ([`objects::objects`]) — every group with its consensus and the
///    members that consensus rejects — and what the two operator lists said about them.
///
/// `None`, and no cost, when the pass is off: `--objects off` leaves `assembly` exactly as R §8
/// built it and writes no section.
fn object_round(
    fragments: &[Fragment],
    names: &[String],
    candidates: &mut [Candidate],
    assembly: &mut crate::assembly::Assembly,
    pieces: &[Piece<'_>],
    mut run: ObjectRound<'_>,
) -> Option<objects::ObjectReport> {
    let rules = run.params.objects.as_ref()?;
    let started = Instant::now();
    let demoted =
        objects::demotions(fragments, names, candidates, assembly, pieces, run.constraints, rules);
    // Per pair and not per candidate: a pair routinely has several candidates on one placement,
    // all of them confirmed, and R §8's `best_per_pair` would place the next one instead.
    let by_pair: std::collections::BTreeMap<(FragId, FragId), &objects::Demotion> =
        demoted.iter().map(|(key, d)| (*key, d)).collect();
    for (i, c) in candidates.iter_mut().enumerate() {
        let Some(demotion) = by_pair.get(&constraints::key(c.a, c.b)) else { continue };
        if c.tier != Tier::Confirmed {
            continue;
        }
        c.tier = Tier::Probable;
        if let Some(report) = run.tiers.as_deref_mut() {
            if let Some(tier) = report.tiers.get_mut(i) {
                *tier = Tier::Probable;
            }
            if let Some(Some(evidence)) = report.evidence.get_mut(i) {
                evidence.failed.push(format!("{}: {}", demotion.arm.label(), demotion.reason));
            }
        }
    }
    if !demoted.is_empty() {
        *assembly = assemble_under(
            run.exec,
            pieces,
            candidates,
            run.params,
            run.gate,
            run.constraints,
            Some(rules),
        );
    }
    let (same_object_split, different_object_together) =
        objects::operator_view(names, assembly, run.constraints);
    let report = objects::ObjectReport {
        objects: objects::objects(fragments, names, candidates, assembly, rules),
        demotions: demoted.into_iter().map(|(_, d)| d).collect(),
        same_object_split,
        different_object_together,
        merges: assembly.merges,
    };
    run.stages.finish("objects", started.elapsed().as_secs_f64());
    tracing::info!(
        objects = report.objects.len(),
        demoted = report.demotions.len(),
        merges = report.merges,
        rejects = report.objects.iter().map(|o| o.rejects.len()).sum::<usize>(),
        seconds = run.stages.timings["objects"],
        "objects"
    );
    Some(report)
}

/// R §4–§6 over a list of pairs, in the reference's block order, candidates back in pair order.
#[allow(
    clippy::too_many_arguments,
    reason = "the reference's `_match_all`: what to match, how, on how many workers, watched by               what"
)]
fn match_all(
    engine: Engine<'_>,
    fragments: &[Fragment],
    pairs: &[(usize, usize)],
    params: &Params,
    keep: usize,
    workers: usize,
    tag: Option<&str>,
    watch: &Watch,
) -> Result<Vec<Vec<Candidate>>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
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
    let found: Vec<Vec<(usize, Vec<Candidate>)>> = with_device_slack(engine.exec, || {
        #[allow(clippy::redundant_closure_for_method_calls, reason = "the collect needs the type")]
        blocks
            .par_iter()
            .map(|block| {
                block
                    .iter()
                    .map(|&k| {
                        // D §5's unit of work for the matching stage. A pair either ran whole or
                        // did not start, so the candidate list a cancelled run carries is a
                        // prefix of a complete one rather than a torn version of it.
                        watch.check()?;
                        let (a, b) = pairs[k];
                        let started = Instant::now();
                        let cs = pair::match_pair_with(
                            engine,
                            &fragments[a],
                            &fragments[b],
                            params,
                            keep,
                        );
                        tracing::info!(
                            pair = %format!("{}__{}", fragments[a].name, fragments[b].name),
                            seconds = started.elapsed().as_secs_f64(),
                            candidates = cs.len(),
                            accepted = cs.iter().filter(|c| c.accepted).count(),
                            at = done.fetch_add(1, Ordering::Relaxed) + 1,
                            of = pairs.len(),
                            "pair matched"
                        );
                        watch.advance("matching", done.load(Ordering::Relaxed), pairs.len());
                        Ok((k, cs))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
    })?;
    for (k, cs) in found.into_iter().flatten() {
        out[k] = cs;
    }
    Ok(out)
}

/// R §4.3's partner search: the pairs worth matching, in the order they were given.
fn screen(
    exec: &dyn Executor,
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
                        (Some(a), Some(b)) => screen_pair(exec, &a, &b, params),
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
    engine: Engine<'_>,
    fragments: &[Fragment],
    groups: &[Vec<FragId>],
    poses: &[Matrix4<f64>],
    used: &[(FragId, FragId)],
    params: &Params,
    budget: Budget,
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
    let clouds = fracture_clouds(fragments, &in_group, budget)?.0;
    let pieces: Vec<RefinePiece<'_>> = fragments
        .iter()
        .zip(&clouds)
        .map(|(f, c)| RefinePiece { thick: f.thick, res: f.res(), cloud: c.as_ref() })
        .collect();
    Ok(refine_joins(&pieces, poses, groups, used, params, engine).poses)
}

/// R §9's clouds: one per member of a group of two or more, under D §5 step 2's budget.
///
/// This is the audit's §B.2. R §9 reads the **original** scan of every grouped fragment — the
/// whole file, before R §3.3's decimation — so the stage's own high-water mark is
/// `concurrent jobs x (vertices + normals + the KD query)` of originals, which is the same peak
/// preprocessing has and is priced by the same model ([`memory::reservation`]). Preprocessing
/// (`preprocess_watched`) and R §11.4's placed writers ([`write_placed_meshes`]) have both gone
/// through the semaphore since E2; this stage did not, and on a 170-scan collection it is the one
/// place left where the pool's width alone decides how many originals are resident.
///
/// It moves no result, for the reason the semaphore never does (D §5, [`memory`]): the loop
/// collects by index, each job reads nothing but its own file, and the cap R §9 draws is seeded
/// from [`refine::CAP_SEED`] and not from the schedule. `Budget::bytes(1)` — one scan at a time —
/// produces the same clouds as an unbounded budget, bit for bit, which is what
/// `the_refinement_reserves_its_scans_like_preprocessing` asserts.
///
/// Returns the clouds and what the semaphore did, so the caller can log it as preprocessing does
/// and the test can read it.
fn fracture_clouds(
    fragments: &[Fragment],
    in_group: &[bool],
    budget: Budget,
) -> Result<(Vec<Option<FractureCloud>>, memory::SemaphoreStats)> {
    let semaphore = MemorySemaphore::new(budget);
    let clouds: Vec<Option<FractureCloud>> = fragments
        .par_iter()
        .zip(in_group)
        .map(|(fragment, &wanted)| {
            if !wanted {
                return Ok(None);
            }
            let v64: Vec<[f64; 3]> = fragment.mesh.v.iter().map(|v| v.to_f64()).collect();
            let centroids = geometry::face_geometry(&v64, &fragment.mesh.f).centroids;
            let fracture: Vec<bool> = fragment.labels.iter().map(|l| l.is_fracture()).collect();
            // The reservation covers the original and the cloud built from it, and is released
            // the moment the cloud is the only thing left — the same span `place` reserves for.
            let permit =
                semaphore.acquire(memory::scan_faces(&fragment.source.path).map_or(0, reservation));
            let original = crate::io::load_mesh(&fragment.source.path)?;
            let cloud = fracture_cloud(
                &original,
                &centroids,
                &fracture,
                fragment.thick,
                fragment.res(),
                // R §10's inventory gives the refinement stream the literal 0, and `refine.py:35`
                // is `np.random.default_rng(0)` — not `Params.seed`, which every other stream
                // takes. `--seed` (task H3) does not reach here for that reason: the reference's
                // refinement does not answer to its own seed, so neither does the port's (V4-D9).
                refine::CAP_SEED,
            );
            drop(original);
            drop(permit);
            Ok(Some(cloud))
        })
        .collect::<Result<Vec<Option<FractureCloud>>>>()?;
    if budget.is_bounded() {
        let stats = semaphore.stats();
        tracing::info!(
            budget_mib = budget.available() / (1024 * 1024),
            peak_mib = stats.peak / (1024 * 1024),
            peak_concurrent = stats.peak_running,
            waited = stats.waited,
            "refinement memory"
        );
    }
    Ok((clouds, semaphore.stats()))
}

/// The tail of the reference's `pipeline.segment_only`: the segmentation preview alone.
///
/// `write_previews(out_dir, frags, {n: I}, [[n] …])` — every fragment at the identity and in a
/// group of its own, so no group preview is drawn and `preview_segmentation.png` is the one file
/// written. `sherd-refit segment` writes it and so does this one (V4-D7).
pub fn write_segmentation_preview(out_dir: &Path, fragments: &[Fragment]) -> Result<Vec<PathBuf>> {
    let names: Vec<String> = fragments.iter().map(|f| f.name.clone()).collect();
    let poses = vec![Matrix4::identity(); fragments.len()];
    let groups: Vec<Vec<FragId>> = (0..fragments.len())
        .map(|n| vec![u32::try_from(n).expect("fewer than 2^32 fragments")])
        .collect();
    write_previews(out_dir, fragments, &names, &poses, &groups)
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
        let mean = geometry::column_mean(&points);
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
        BLOCK, Budget, RunOptions, block_size, pair_blocks, preprocess, run, second_pass_pairs,
        set_threads,
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
        let results = preprocess(&entries, 200_000, None, Budget::unbounded(), 0);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "a mesh with no triangles is R §3.1's error case");
        // The same file under a budget that admits nothing: the semaphore lets it through anyway
        // (nothing else is running) and it fails with the reader's error, not by waiting.
        let bounded = preprocess(&entries, 200_000, None, Budget::bytes(1), 0);
        assert!(bounded[0].is_err(), "the semaphore never turns a read error into a hang");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Audit §B.2: R §9 reads one original scan per grouped fragment, and it now reserves for
    /// them exactly as preprocessing and R §11.4's writers do.
    ///
    /// The two claims the fix has to support are here: under a budget that admits nothing
    /// (`Budget::bytes(1)`) the stage runs **one scan at a time**, and the clouds it produces are
    /// the unbounded run's, bit for bit — the semaphore reorders when a scan is read and never
    /// what is read from it.
    #[test]
    fn the_refinement_reserves_its_scans_like_preprocessing() {
        let input =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input");
        let entries = vec![
            Entry { path: input.join("pieceA.ply"), name: "pieceA".to_owned() },
            Entry { path: input.join("pieceB.ply"), name: "pieceB".to_owned() },
        ];
        let fragments: Vec<crate::fragment::Fragment> =
            preprocess(&entries, 200_000, None, Budget::unbounded(), 0)
                .into_iter()
                .map(|r| r.expect("the slab preprocesses").fragment)
                .collect();
        let in_group = vec![true; fragments.len()];

        let (free, _) = super::fracture_clouds(&fragments, &in_group, Budget::unbounded())
            .expect("the clouds build");
        let (bounded, stats) = super::fracture_clouds(&fragments, &in_group, Budget::bytes(1))
            .expect("the clouds build under a budget that admits nothing");

        assert_eq!(stats.peak_running, 1, "one scan at a time: {stats:?}");
        assert!(
            free.iter().all(Option::is_some) && free.len() == fragments.len(),
            "every member of a group gets a cloud"
        );
        for (a, b) in free.iter().zip(&bounded) {
            let (a, b) = (a.as_ref().expect("a cloud"), b.as_ref().expect("a cloud"));
            assert_eq!(a.idx, b.idx, "the selected vertices, in the ICP's summation order");
            assert_eq!(bits(&a.points), bits(&b.points), "the points, bit for bit");
            assert_eq!(bits(&a.normals), bits(&b.normals), "the normals, bit for bit");
        }
    }

    /// Every `f64` of a cloud as its bits, so "identical" means identical and not "equal".
    fn bits(points: &[[f64; 3]]) -> Vec<u64> {
        points.iter().flatten().map(|x| x.to_bits()).collect()
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
            tier: crate::tiers::Tier::of_accept(true),
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
