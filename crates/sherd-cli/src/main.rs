//! `sherd-refit-rs` — the command line front end of the Rust core.
//!
//! The subcommands are D §9's: `run` and `segment` mirror the Python's, flag for flag, and
//! `parity`, `bench` and `info` are new. Phase 1a implemented `info`, `segment` up to the working
//! mesh and `parity` for the stages the port computes; step B1 added R §3.4's shell/fracture
//! labels to `segment` and its own `parity` row, step B2 R §3.5's breaklines and theirs, step
//! B3 the sampled match arrays and the `samples` row, step C1 the first three pair rows —
//! `hypotheses`, `coarse` and `nms` — step C2 the two refinement rows and step C3 the last two,
//! `verify` and `candidates`, which is the whole of `match_pair`. Step D1 opened phase 1d with the
//! `assembly` row — R §8's groups, its used joins, its rejections and R §8.2's recentring — and
//! step D2 the last two, `refine` and `outputs`. Step D3 turned `run` and `bench` on: `run` is the
//! reference's own subcommand, flag for flag (R §1.4), and `bench` is a run with the previews off,
//! timed against D §10.3.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use sherd_core::fragment::cache;
use sherd_core::matching::icp::{Assembly, Numerics, Precision};
use sherd_core::{ALGO_REF, Backend, CACHE_VERSION, CORE_VERSION, Params, collection, pipeline};
use sherd_parity::FixtureDir;
use sherd_parity::report::{Mode, StageReport};
use sherd_parity::stages::{Collection, Stage};

/// Fracture-surface reassembly of 3D-scanned ceramic fragments.
#[derive(Debug, Parser)]
#[command(name = "sherd-refit-rs", version, about, long_about = None)]
struct Cli {
    /// Log level: repeat for more (`-v` info, `-vv` debug, `-vvv` trace).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Assemble a collection of fragments (R §2–§11).
    Run(RunArgs),

    /// Preprocess every fragment and write the fragment cache (R §3.1–3.7).
    ///
    /// Reads every mesh of INPUT in the reference's collection order, cleans it, keeps the largest
    /// component, measures the wall thickness, decimates to the adaptive face budget, smooths,
    /// labels every face shell or fracture, traces the breakline and its frames, draws the
    /// surface, fracture and shell-margin samples, and writes `<OUT>/cache/<name>.sherd`. A second
    /// run over the same files reuses those caches.
    ///
    /// The segmentation preview the reference's `segment` also produces arrives with the renderer
    /// (phase 1d).
    Segment(SegmentArgs),

    /// Run the port's stages against a Python fixture dump and report D §10.2's tolerances.
    Parity(ParityArgs),

    /// Time the pipeline against the gates of D §10.3.
    Bench(BenchArgs),

    /// Print what this build is: versions, algorithm reference, backends.
    Info,
}

/// Arguments of `run`: every flag of `sherd_refit/cli.py`, with its name and its default (R §1.4),
/// plus the four the port adds (D §9).
#[derive(Debug, Args)]
#[allow(clippy::struct_excessive_bools, reason = "the reference's own switches, one field each")]
struct RunArgs {
    /// Directory of fragment files (`.ply`, `.obj`, `.stl`, `.off`).
    input: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Working-mesh face budget per fragment.
    #[arg(long, default_value_t = 200_000)]
    target_faces: u32,
    /// Parallel workers; the reference's process count, and here the size of the one rayon pool
    /// (default: one per core).
    #[arg(long)]
    workers: Option<usize>,
    /// Threads per matching worker. One process here, so this sizes the same pool `--workers`
    /// does and wins when both are given (default: one per core).
    #[arg(long)]
    threads: Option<usize>,
    /// Candidates refined with full ICP per pair.
    #[arg(long, default_value_t = Params::default().stage2)]
    candidates: u32,
    /// Hypotheses refined with breakline ICP per pair.
    #[arg(long, default_value_t = Params::default().stage1)]
    stage1: u32,
    /// t, voxel the breakline is thinned to before frames are paired.
    #[arg(long, default_value_t = Params::default().brk_voxel)]
    brk_voxel: f64,
    /// Degrees, tolerance on |dih_A + dih_B - 180| for a hypothesis.
    #[arg(long, default_value_t = Params::default().dihedral_tol)]
    dihedral_tol: f64,
    /// Min tight-contact fraction to accept a join.
    #[arg(long, default_value_t = Params::default().min_tight)]
    min_tight: f64,
    /// k for the max median fracture gap; the pair's limit is max(k t, m res).
    #[arg(long, default_value_t = Params::default().max_gap)]
    max_gap: f64,
    /// Max penetrating surface fraction.
    #[arg(long, default_value_t = Params::default().max_pen)]
    max_pen: f64,
    /// Min seam length (in t).
    #[arg(long, default_value_t = Params::default().min_seam)]
    min_seam: f64,
    /// Skip a pair whose wall thicknesses differ by more than this factor.
    #[arg(long, default_value_t = Params::default().thick_ratio)]
    thick_ratio: f64,
    /// Partners kept per fragment by the partner search (0 disables it).
    #[arg(long, default_value_t = Params::default().screen_top_k)]
    screen_top_k: u32,
    /// Breakline points per fragment used by the partner search.
    #[arg(long, default_value_t = Params::default().screen_points)]
    screen_points: u32,
    /// The partner search is skipped below this many pairs.
    #[arg(long, default_value_t = Params::default().screen_min_pairs)]
    screen_min_pairs: u32,
    /// Partners of each unplaced fragment to match again with a larger budget (0 disables it).
    #[arg(long, default_value_t = Params::default().second_pass_top)]
    second_pass_top: u32,
    /// Hypotheses refined in the second pass.
    #[arg(long, default_value_t = Params::default().second_pass_stage1)]
    second_pass_stage1: u32,
    /// Candidates fully verified in the second pass.
    #[arg(long, default_value_t = Params::default().second_pass_stage2)]
    second_pass_candidates: u32,
    /// Skip stage 2 when the pair's best stage-1 breakline score is below this.
    #[arg(long, default_value_t = Params::default().stage1_floor)]
    stage1_floor: f64,
    /// Skip the fracture-only ICPs and the costly verification below this tight-contact fraction.
    #[arg(long, default_value_t = Params::default().early_reject_tight)]
    early_reject_tight: f64,
    /// Points in the cloud the two coarse stage-2 ICPs run on (0: all).
    #[arg(long, default_value_t = Params::default().reg_points)]
    reg_points: u32,
    /// Shell-margin points kept per fragment for ICP and the continuity test.
    #[arg(long, default_value_t = Params::default().margin_points)]
    margin_points: u32,
    /// Whole-surface samples per fragment (penetration test and shell margin).
    #[arg(long, default_value_t = Params::default().surface_points)]
    surface_points: u32,
    /// Fracture samples per t^2 of fracture area.
    #[arg(long, default_value_t = Params::default().frac_per_t2)]
    frac_density: f64,
    /// Do not write R §11.5's previews.
    #[arg(long)]
    no_preview: bool,
    /// Skip full-resolution refinement.
    #[arg(long)]
    no_refine: bool,
    /// Do not write placed/merged meshes.
    #[arg(long)]
    no_meshes: bool,
    /// Executor: `auto`, `cpu` or `gpu` (D §6.8).
    #[arg(long, default_value_t = Backend::Auto)]
    backend: Backend,
    /// Neither read nor write the fragment cache (R §3.7).
    #[arg(long)]
    no_cache: bool,
    /// Recompute every fragment and overwrite its cache, even when the cache is valid.
    #[arg(long)]
    force: bool,
    /// Write the Rust-side fixture dump of D §10.1 (not built yet).
    #[arg(long, value_name = "DIR")]
    dump_fixtures: Option<PathBuf>,
}

impl RunArgs {
    /// R §1.4's CLI-to-`Params` mapping: `--candidates → stage2`, `--frac-density → frac_per_t2`,
    /// `--second-pass-candidates → second_pass_stage2`, and every other flag to the field of the
    /// same name. Everything R §1.1 has no flag for keeps its default.
    fn params(&self) -> Params {
        Params {
            stage1: self.stage1,
            stage2: self.candidates,
            brk_voxel: self.brk_voxel,
            dihedral_tol: self.dihedral_tol,
            min_tight: self.min_tight,
            max_gap: self.max_gap,
            max_pen: self.max_pen,
            min_seam: self.min_seam,
            thick_ratio: self.thick_ratio,
            early_reject_tight: self.early_reject_tight,
            stage1_floor: self.stage1_floor,
            second_pass_top: self.second_pass_top,
            second_pass_stage1: self.second_pass_stage1,
            second_pass_stage2: self.second_pass_candidates,
            screen_top_k: self.screen_top_k,
            screen_points: self.screen_points,
            screen_min_pairs: self.screen_min_pairs,
            margin_points: self.margin_points,
            reg_points: self.reg_points,
            surface_points: self.surface_points,
            frac_per_t2: self.frac_density,
            ..Params::default()
        }
    }
}

/// Arguments of `segment`.
#[derive(Debug, Args)]
struct SegmentArgs {
    /// Directory of fragment files (`.ply`, `.obj`, `.stl`, `.off`).
    input: PathBuf,
    /// Output directory; the caches go to `<OUT>/cache`.
    #[arg(long)]
    out: PathBuf,
    /// Working-mesh face budget per fragment (R §3.3 caps its adaptive budget with this).
    #[arg(long, default_value_t = 200_000)]
    target_faces: u32,
    /// Worker threads; 0 means one per core.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Recompute every fragment and overwrite its cache, even when the cache is valid.
    #[arg(long)]
    force: bool,
    /// Neither read nor write the fragment cache.
    #[arg(long)]
    no_cache: bool,
}

/// Arguments of `parity`.
#[derive(Debug, Args)]
struct ParityArgs {
    /// A fixture dump written by `tools/dump_fixtures.py` (D §10.1).
    #[arg(long)]
    fixtures: PathBuf,
    /// The collection the dump was made from; needed by native mode and by any stage the dump
    /// itself does not carry (levels `slim` and `min`, D §10.1).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Stage to compare: `load`, `thickness`, `working-mesh`, `segmentation`, `breakline`,
    /// `samples`, `hypotheses`, `coarse`, `nms`, `stage1`, `stage2`, `verify`, `candidates`,
    /// `assembly`, `refine`, `outputs`, or `all`. Repeatable.
    #[arg(long, default_value = "all")]
    stage: Vec<String>,
    /// Feed each stage the Python stage's own inputs instead of the port's upstream results
    /// (D §10.2's injected column). Without it the stages run natively.
    #[arg(long)]
    injected: bool,
    /// Print every comparison, not only the per-stage summary.
    #[arg(long)]
    details: bool,
    /// Re-hash every file of the dump and compare against the manifest.
    #[arg(long)]
    verify_checksums: bool,
    /// Scalar the ICP point loops of `stage1` and `stage2` run in (D §7, experiment E5).
    /// `f64` is the reference's and the one D §10.2's rows are stated for.
    #[arg(long, value_enum, default_value_t = IcpPrecision::F64)]
    icp_precision: IcpPrecision,
    /// Frame the point-to-plane normal equations are assembled in (D §7, experiment E5).
    /// `world` is R §7 verbatim; `centred` is the re-parameterisation the GPU path will use.
    #[arg(long, value_enum, default_value_t = IcpAssembly::World)]
    icp_assembly: IcpAssembly,
}

/// `--icp-precision`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum IcpPrecision {
    /// The reference's: every point loop in `f64`.
    F64,
    /// The GPU executor's: the point loops in `f32`, the pose and the solve still `f64`.
    F32,
}

/// `--icp-assembly`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum IcpAssembly {
    /// R §7 verbatim: the normal equations in the meshes' own coordinates.
    World,
    /// D §7: about the target centroid, with the update re-expressed about the origin.
    Centred,
}

impl ParityArgs {
    /// D §7's two knobs as `sherd-core` states them.
    fn numerics(&self) -> Numerics {
        Numerics {
            precision: match self.icp_precision {
                IcpPrecision::F64 => Precision::F64,
                IcpPrecision::F32 => Precision::F32,
            },
            assembly: match self.icp_assembly {
                IcpAssembly::World => Assembly::World,
                IcpAssembly::Centred => Assembly::Centred,
            },
        }
    }
}

/// Arguments of `bench`: a run with the previews and the meshes off, timed against D §10.3.
#[derive(Debug, Args)]
struct BenchArgs {
    /// Directory of fragment files to time the pipeline on.
    input: PathBuf,
    /// Where the run's outputs go; `report.json` carries the same timings.
    #[arg(long)]
    out: PathBuf,
    /// Working-mesh face budget per fragment.
    #[arg(long, default_value_t = 200_000)]
    target_faces: u32,
    /// Worker threads; unset means one per core.
    #[arg(long)]
    threads: Option<usize>,
    /// D §10.3's wall-clock gate for this set, in seconds; without it the timings are only
    /// reported.
    #[arg(long)]
    gate: Option<f64>,
    /// Also write R §11.4's meshes, which D §10.3's gates do not include.
    #[arg(long)]
    meshes: bool,
    /// Neither read nor write the fragment cache; D §10.3's gates are stated for a warm one.
    #[arg(long)]
    no_cache: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    match cli.command {
        Command::Run(args) => run(&args),
        Command::Segment(args) => segment(&args),
        Command::Parity(args) => parity(&args),
        Command::Bench(args) => bench(&args),
        Command::Info => {
            info();
            Ok(())
        }
    }
}

fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(format!("sherd={level}")));
    tracing_subscriber::fmt().with_env_filter(filter).with_target(false).init();
}

/// Prints what this build is; the same three strings go into `report.json`'s `engine` key.
fn info() {
    println!("sherd-refit-rs {CORE_VERSION}");
    println!("  algorithm reference: {ALGO_REF}");
    println!("  cache version:       {CACHE_VERSION}");
    println!("  backends:            cpu (gpu arrives in phase 2)");
    let threads = std::thread::available_parallelism().map_or(0, std::num::NonZero::get);
    println!("  cores available:     {threads}");
    println!("  default seed:        {}", Params::default().seed);
}

/// R §3.1–3.5 for a whole collection, with the cache of R §3.7 (plan steps S4, B1, B2 and B3).
#[allow(clippy::cast_precision_loss, reason = "counts printed in a table")]
fn segment(args: &SegmentArgs) -> Result<()> {
    if let Err(e) = pipeline::set_threads(args.threads) {
        bail!("--threads {}: {e}", args.threads);
    }
    let entries = collection::discover(&args.input)
        .with_context(|| format!("scanning {}", args.input.display()))?;
    if entries.is_empty() {
        bail!("{}: no .ply, .obj, .stl or .off file", args.input.display());
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    // `--force` recomputes by writing to a cache directory it first empties of the names it is
    // about to write; `--no-cache` neither reads nor writes.
    if args.force && !args.no_cache {
        for entry in &entries {
            let path = cache::cache_path(&args.out, &entry.name);
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("removing {}", path.display()))?;
            }
        }
    }
    let out = if args.no_cache { None } else { Some(args.out.as_path()) };

    let started = std::time::Instant::now();
    let results = pipeline::preprocess(&entries, args.target_faces as usize, out);
    let wall = started.elapsed().as_secs_f64();

    println!(
        "{:<28} {:>9} {:>10} {:>9} {:>8} {:>7} {:>8} {:>7} {:>6} {:>7} {:>7} {:>6} {:>6}",
        "fragment",
        "faces",
        "from",
        "t",
        "res",
        "t/res",
        "fracture",
        "brk",
        "sub",
        "frac pts",
        "margin",
        "closed",
        "cache"
    );
    let mut failed = 0;
    let mut total = 0.0;
    for (entry, result) in entries.iter().zip(&results) {
        match result {
            Ok(p) => {
                let fr = &p.fragment;
                total += p.seconds;
                println!(
                    "{:<28} {:>9} {:>10} {:>9.3} {:>8.3} {:>7.1} {:>8.3} {:>7} {:>6} {:>7} {:>7} \
                     {:>6} {:>6}",
                    fr.name,
                    fr.n_faces(),
                    fr.n_orig_faces,
                    fr.thick,
                    fr.res(),
                    fr.thick / fr.res().max(1e-9),
                    fr.fracture_fraction(),
                    fr.brk.len(),
                    fr.brk.sub.len(),
                    fr.samples.n_fracture(),
                    fr.samples.n_margin(),
                    fr.watertight,
                    if p.cached { "hit" } else { "miss" }
                );
            }
            Err(e) => {
                failed += 1;
                println!("{:<28} {e}", entry.name);
            }
        }
    }
    println!(
        "{} fragments, {failed} failed, {wall:.2} s wall ({total:.2} s of work)",
        entries.len()
    );
    if !args.no_cache {
        println!("caches in {}", args.out.join("cache").display());
    }
    println!(
        "R §3.1-3.7 for every fragment: the working mesh, the labels, the breaklines and the \
         match arrays. Matching (R §4-§6) is phase 1c."
    );
    if failed > 0 {
        bail!("{failed} of {} fragments could not be preprocessed", entries.len());
    }
    Ok(())
}

/// R §2–§11 for a whole collection: the reference's `sherd-refit run`, flag for flag.
#[allow(clippy::cast_precision_loss, reason = "counts and seconds printed in a table")]
fn run(args: &RunArgs) -> Result<()> {
    if let Some(dir) = &args.dump_fixtures {
        bail!(
            "--dump-fixtures {} is D §10.1's Rust-side writer and is not built yet; the Python \
             side is `python tools/dump_fixtures.py INPUT OUT`",
            dir.display()
        );
    }
    let backend = resolve_backend(args.backend)?;
    let threads = args.threads.or(args.workers).unwrap_or(0);
    if let Err(e) = pipeline::set_threads(threads) {
        bail!("--threads {threads}: {e}");
    }
    let options = pipeline::RunOptions {
        target_faces: args.target_faces as usize,
        params: args.params(),
        keep_per_pair: pipeline::KEEP_PER_PAIR,
        preview: !args.no_preview,
        refine: !args.no_refine,
        write_meshes: !args.no_meshes,
        cache: !args.no_cache,
        workers: args.workers.unwrap_or(0),
        backend,
    };
    if args.force && !args.no_cache {
        clear_caches(&args.input, &args.out)?;
    }
    let started = std::time::Instant::now();
    let summary = pipeline::run(&args.input, &args.out, &options)
        .with_context(|| format!("assembling {}", args.input.display()))?;
    let wall = started.elapsed().as_secs_f64();

    println!(
        "{} fragments, {} pairs ({} skipped by --thick-ratio{}), {} candidates, {} accepted",
        summary.names.len(),
        summary.pairs,
        summary.skipped_pairs,
        match summary.screened {
            Some((seen, kept)) => format!(", {seen} screened to {kept}"),
            None => String::new(),
        },
        summary.candidates.len(),
        summary.accepted()
    );
    for (k, group) in summary.groups.iter().enumerate() {
        if group.len() > 1 {
            let members: Vec<&str> =
                group.iter().map(|&n| summary.names[n as usize].as_str()).collect();
            println!("  group {k}: {}", members.join(", "));
        }
    }
    let alone: Vec<&str> = summary
        .groups
        .iter()
        .filter(|g| g.len() == 1)
        .map(|g| summary.names[g[0] as usize].as_str())
        .collect();
    if !alone.is_empty() {
        println!("  not assembled: {}", alone.join(", "));
    }
    print_timings(&summary.timings, wall);
    println!("{} files in {}", summary.written.len(), args.out.display());
    Ok(())
}

/// D §10.3's timing gate: a run with the previews and the meshes off, timed stage by stage.
fn bench(args: &BenchArgs) -> Result<()> {
    if let Err(e) = pipeline::set_threads(args.threads.unwrap_or(0)) {
        bail!("--threads: {e}");
    }
    let options = pipeline::RunOptions {
        target_faces: args.target_faces as usize,
        preview: false,
        write_meshes: args.meshes,
        cache: !args.no_cache,
        workers: 0,
        backend: Backend::Cpu,
        ..pipeline::RunOptions::default()
    };
    let started = std::time::Instant::now();
    let summary = pipeline::run(&args.input, &args.out, &options)
        .with_context(|| format!("timing {}", args.input.display()))?;
    let wall = started.elapsed().as_secs_f64();
    println!(
        "{}: {} fragments, {} pairs, {} accepted, {} assembled group(s)",
        args.input.display(),
        summary.names.len(),
        summary.pairs,
        summary.accepted(),
        summary.assembled().count()
    );
    print_timings(&summary.timings, wall);
    match args.gate {
        Some(gate) if wall > gate => {
            bail!("{wall:.1} s wall is over the gate of {gate:.1} s")
        }
        Some(gate) => println!("within the gate of {gate:.1} s"),
        None => {}
    }
    Ok(())
}

/// The per-stage table both `run` and `bench` print; the same numbers `report.json` carries.
fn print_timings(timings: &std::collections::BTreeMap<String, f64>, wall: f64) {
    for (stage, seconds) in timings {
        println!("  {stage:<12} {seconds:>8.2} s");
    }
    println!("  {:<12} {wall:>8.2} s", "wall");
}

/// `--backend` resolved for a build with no GPU executor (D §6.8): `auto` falls back to the CPU
/// and `gpu` is an error rather than a silent fallback, which is what a benchmark asking for it
/// needs.
fn resolve_backend(backend: Backend) -> Result<Backend> {
    match backend {
        Backend::Gpu => bail!("--backend gpu: the GPU executor arrives in phase 2 (D §6)"),
        Backend::Auto | Backend::Cpu => Ok(Backend::Cpu),
    }
}

/// `--force`: removes the cache files of the collection about to be run, so every fragment is
/// recomputed and rewritten.
fn clear_caches(input: &std::path::Path, out: &std::path::Path) -> Result<()> {
    let entries =
        collection::discover(input).with_context(|| format!("scanning {}", input.display()))?;
    for entry in &entries {
        let path = cache::cache_path(out, &entry.name);
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
    }
    Ok(())
}

/// Reads a fixture dump, runs the requested stages against it and prints D §10.2's table.
fn parity(args: &ParityArgs) -> Result<()> {
    let dir = FixtureDir::new(&args.fixtures);
    let collection = Collection::open(dir, args.input.as_deref())
        .with_context(|| format!("reading the fixture in {}", args.fixtures.display()))?
        .with_icp(args.numerics());
    let manifest = &collection.manifest;

    println!("fixture:    {}", args.fixtures.display());
    println!("  commit:   {}{}", manifest.commit, if manifest.dirty { " (dirty)" } else { "" });
    println!("  level:    {}", manifest.level);
    println!("  open3d:   {}, numpy {}", manifest.open3d, manifest.numpy);
    println!("  files:    {} in {} fragments", manifest.files.len(), manifest.pairs.names.len());
    println!("  pairs:    {}", manifest.pairs.pairs.len());
    println!("  faces:    --target-faces {}", collection.target_faces);
    println!(
        "  params:   {}",
        if manifest.uses_default_params() {
            "the defaults of R §1.1".to_owned()
        } else {
            "not the defaults — the comparison must use the dump's own values".to_owned()
        }
    );
    match &args.input {
        Some(input) => println!("  input:    {}", input.display()),
        None => println!("  input:    none given (native mode will skip)"),
    }
    println!(
        "  icp:      {:?} point loops, {:?} assembly{}",
        collection.icp.precision,
        collection.icp.assembly,
        if collection.icp == Numerics::REFERENCE { "" } else { "  (not the reference's)" }
    );

    if args.verify_checksums {
        let bad = collection.dir.verify_checksums().context("verifying the fixture's checksums")?;
        if bad.is_empty() {
            println!("  checksums: all {} files match the manifest", manifest.files.len());
        } else {
            for path in &bad {
                println!("  MISMATCH: {path}");
            }
            bail!("{} of {} files do not match the manifest", bad.len(), manifest.files.len());
        }
    }

    let stages = requested_stages(&args.stage)?;
    let mode = if args.injected { Mode::Injected } else { Mode::Native };
    let reports = collection.run_all(&stages, mode).context("running the stages")?;

    println!();
    println!("{}", StageReport::summary_header());
    for report in &reports {
        println!("{}", report.summary_line());
    }

    if args.details {
        for report in &reports {
            println!();
            println!("--- {} ({}) ---", report.stage, report.mode);
            println!("{}", StageReport::detail_header());
            for check in &report.checks {
                println!("{}", check.line());
            }
        }
    }

    let skipped: usize = reports.iter().map(|r| r.skips.len()).sum();
    if skipped > 0 {
        println!();
        for report in &reports {
            for skip in &report.skips {
                println!("skipped {} in {}: {}", skip.scope, report.stage, skip.reason);
            }
        }
    }

    let failures: Vec<&sherd_parity::Check> =
        reports.iter().flat_map(StageReport::failures).collect();
    if !failures.is_empty() {
        println!();
        println!("{}", StageReport::detail_header());
        for check in &failures {
            println!("{}", check.line());
        }
        let total: usize = reports.iter().map(|r| r.checks.len()).sum();
        bail!("{} of {total} comparisons outside their tolerance", failures.len());
    }
    Ok(())
}

/// `--stage` values, in pipeline order and without duplicates; `all` is every stage this build
/// can run.
fn requested_stages(requested: &[String]) -> Result<Vec<Stage>> {
    let mut wanted = Vec::new();
    for name in requested {
        if name == "all" {
            wanted.extend(Stage::ALL);
            continue;
        }
        let stage = Stage::parse(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown stage `{name}`; this build compares {} or `all`",
                Stage::ALL.map(Stage::as_str).join(", ")
            )
        })?;
        wanted.push(stage);
    }
    let ordered: Vec<Stage> = Stage::ALL.into_iter().filter(|s| wanted.contains(s)).collect();
    if ordered.is_empty() {
        bail!("no stage requested");
    }
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::{Backend, Cli, Params, requested_stages};
    use clap::{CommandFactory, Parser};
    use sherd_parity::stages::Stage;

    #[test]
    fn the_command_line_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_binary_is_named_for_the_transition() {
        assert_eq!(Cli::command().get_name(), "sherd-refit-rs");
    }

    #[test]
    fn stages_come_back_in_pipeline_order_without_duplicates() {
        assert_eq!(requested_stages(&["all".to_owned()]).unwrap(), Stage::ALL.to_vec());
        assert_eq!(
            requested_stages(&["working-mesh".to_owned(), "load".to_owned()]).unwrap(),
            vec![Stage::Load, Stage::WorkingMesh]
        );
        assert_eq!(
            requested_stages(&["load".to_owned(), "load".to_owned()]).unwrap(),
            vec![Stage::Load]
        );
        assert_eq!(
            requested_stages(&["segmentation".to_owned()]).unwrap(),
            vec![Stage::Segmentation]
        );
        assert_eq!(requested_stages(&["breakline".to_owned()]).unwrap(), vec![Stage::Breakline]);
        assert_eq!(requested_stages(&["samples".to_owned()]).unwrap(), vec![Stage::Samples]);
        assert_eq!(
            requested_stages(&["nms".to_owned(), "hypotheses".to_owned(), "coarse".to_owned()])
                .unwrap(),
            vec![Stage::Hypotheses, Stage::Coarse, Stage::Nms],
            "the pair stages come back in pipeline order too"
        );
        assert_eq!(
            requested_stages(&["stage2".to_owned(), "stage1".to_owned()]).unwrap(),
            vec![Stage::Stage1, Stage::Stage2],
            "and so do the two refinement stages"
        );
        assert_eq!(
            requested_stages(&["candidates".to_owned(), "verify".to_owned()]).unwrap(),
            vec![Stage::Verify, Stage::Candidates],
            "and so do the two verification stages"
        );
        assert_eq!(
            requested_stages(&["assembly".to_owned(), "candidates".to_owned()]).unwrap(),
            vec![Stage::Candidates, Stage::Assembly],
            "R §8's row comes after the pair it is built from"
        );
        assert_eq!(
            requested_stages(&["outputs".to_owned(), "refine".to_owned()]).unwrap(),
            vec![Stage::Refine, Stage::Outputs],
            "and so do the last two rows of D §10.2"
        );
        // A stage this build does not have names the ones it does.
        let err = requested_stages(&["no such stage".to_owned()]).unwrap_err().to_string();
        assert!(err.contains("assembly") && err.contains("outputs"), "{err}");
    }

    #[test]
    fn segment_defaults_to_the_references_face_budget() {
        let cli = Cli::try_parse_from(["sherd-refit-rs", "segment", "in", "--out", "out"]).unwrap();
        match cli.command {
            super::Command::Segment(args) => {
                assert_eq!(args.target_faces, 200_000, "the Python's --target-faces default");
                assert_eq!(args.threads, 0);
                assert!(!args.force && !args.no_cache);
            }
            other => panic!("{other:?}"),
        }
    }

    /// Every flag of `sherd_refit/cli.py` is here, spelled the same, and every default is the
    /// reference's — which for the twenty-one that map to [`Params`] means `Params::default()`
    /// exactly (R §1.1, R §1.4).
    #[test]
    #[allow(clippy::float_cmp, reason = "the defaults are literals on both sides")]
    fn run_takes_every_python_flag_at_the_python_default() {
        let cli = Cli::try_parse_from(["sherd-refit-rs", "run", "in", "--out", "out"]).unwrap();
        match cli.command {
            super::Command::Run(args) => {
                assert_eq!(args.target_faces, 200_000, "the Python's --target-faces default");
                assert!(args.workers.is_none() && args.threads.is_none(), "both default to None");
                assert!(!args.no_preview && !args.no_refine && !args.no_meshes);
                assert!(!args.no_cache && !args.force && args.dump_fixtures.is_none());
                assert_eq!(args.backend, Backend::Auto);
                assert_eq!(
                    args.params(),
                    Params::default(),
                    "no flag given must leave every threshold of R §1.1 at its default"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// R §1.4's three renamed flags, and one that is not renamed, through the mapping.
    #[test]
    #[allow(clippy::float_cmp, reason = "the values are the literals just passed in")]
    fn run_renames_the_three_flags_r_1_4_names() {
        let cli = Cli::try_parse_from([
            "sherd-refit-rs",
            "run",
            "in",
            "--out",
            "out",
            "--candidates",
            "3",
            "--frac-density",
            "7.5",
            "--second-pass-candidates",
            "11",
            "--stage1",
            "4",
            "--no-preview",
        ])
        .unwrap();
        match cli.command {
            super::Command::Run(args) => {
                let p = args.params();
                assert_eq!(p.stage2, 3, "--candidates -> stage2");
                assert_eq!(p.frac_per_t2, 7.5, "--frac-density -> frac_per_t2");
                assert_eq!(p.second_pass_stage2, 11, "--second-pass-candidates");
                assert_eq!(p.stage1, 4, "and --stage1 keeps its name");
                assert!(args.no_preview);
            }
            other => panic!("{other:?}"),
        }
    }

    /// `--backend gpu` fails rather than falling back silently; `auto` resolves to the CPU while
    /// there is no other executor (D §6.8).
    #[test]
    fn the_backend_resolves_to_the_cpu_and_refuses_the_gpu() {
        assert_eq!(super::resolve_backend(Backend::Auto).unwrap(), Backend::Cpu);
        assert_eq!(super::resolve_backend(Backend::Cpu).unwrap(), Backend::Cpu);
        let err = super::resolve_backend(Backend::Gpu).unwrap_err().to_string();
        assert!(err.contains("phase 2"), "{err}");
    }

    #[test]
    fn parity_takes_the_flags_the_plan_names() {
        let cli = Cli::try_parse_from([
            "sherd-refit-rs",
            "parity",
            "--fixtures",
            "dump",
            "--stage",
            "working-mesh",
            "--injected",
        ])
        .unwrap();
        match cli.command {
            super::Command::Parity(args) => {
                assert_eq!(args.fixtures.to_str(), Some("dump"));
                assert_eq!(args.stage, ["working-mesh"]);
                assert!(args.injected);
                assert!(args.input.is_none());
            }
            other => panic!("{other:?}"),
        }
    }
}
