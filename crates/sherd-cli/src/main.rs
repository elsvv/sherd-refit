//! `sherd-refit-rs` — the command line front end of the Rust core.
//!
//! The subcommands are D §9's: `run` and `segment` mirror the Python's, flag for flag, and
//! `parity`, `bench` and `info` are new. Phase 1a implemented `info`, `segment` up to the working
//! mesh and `parity` for the stages the port computes; step B1 added R §3.4's shell/fracture
//! labels to `segment` and its own `parity` row. `run` and `bench` arrive with the pipeline they
//! drive (phase 1d) and report that plainly until then.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use sherd_core::fragment::cache;
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

    /// Preprocess every fragment and write the fragment cache (R §3.1–3.4).
    ///
    /// Reads every mesh of INPUT in the reference's collection order, cleans it, keeps the largest
    /// component, measures the wall thickness, decimates to the adaptive face budget, smooths,
    /// labels every face shell or fracture, and writes `<OUT>/cache/<name>.sherd`. A second run
    /// over the same files reuses those caches.
    ///
    /// The breaklines, the match arrays and the segmentation preview the reference's `segment`
    /// also produces are the remaining steps of phase 1b.
    Segment(SegmentArgs),

    /// Run the port's stages against a Python fixture dump and report D §10.2's tolerances.
    Parity(ParityArgs),

    /// Time the pipeline against the gates of D §10.3.
    Bench(BenchArgs),

    /// Print what this build is: versions, algorithm reference, backends.
    Info,
}

/// Arguments of `run`; the full flag set of R §1.4 lands in phase 1d.
#[derive(Debug, Args)]
struct RunArgs {
    /// Directory of fragment files (`.ply`, `.obj`, `.stl`, `.off`).
    input: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Working-mesh face budget per fragment (R §3.3 caps its adaptive budget with this).
    #[arg(long, default_value_t = 200_000)]
    target_faces: u32,
    /// Worker threads; 0 means one per core.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Executor: `auto`, `cpu` or `gpu` (D §6.8).
    #[arg(long, default_value_t = Backend::Auto)]
    backend: Backend,
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
    /// Stage to compare: `load`, `thickness`, `working-mesh`, `segmentation`, or `all`.
    /// Repeatable.
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
}

/// Arguments of `bench`.
#[derive(Debug, Args)]
struct BenchArgs {
    /// Directory of fragment files to time the pipeline on.
    input: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    match cli.command {
        Command::Run(_) => bail!("`run` lands in phase 1d, when every stage it drives exists"),
        Command::Segment(args) => segment(&args),
        Command::Parity(args) => parity(&args),
        Command::Bench(_) => bail!("`bench` lands in phase 1d, together with `run`"),
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

/// R §3.1–3.4 for a whole collection, with the cache of R §3.7 (plan steps S4 and B1).
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
        "{:<28} {:>9} {:>10} {:>9} {:>8} {:>7} {:>8} {:>6} {:>6}",
        "fragment", "faces", "from", "t", "res", "t/res", "fracture", "closed", "cache"
    );
    let mut failed = 0;
    let mut total = 0.0;
    for (entry, result) in entries.iter().zip(&results) {
        match result {
            Ok(p) => {
                let fr = &p.fragment;
                total += p.seconds;
                println!(
                    "{:<28} {:>9} {:>10} {:>9.3} {:>8.3} {:>7.1} {:>8.3} {:>6} {:>6}",
                    fr.name,
                    fr.n_faces(),
                    fr.n_orig_faces,
                    fr.thick,
                    fr.res(),
                    fr.thick / fr.res().max(1e-9),
                    fr.fracture_fraction(),
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
        "the breaklines and match arrays of R §3.5 are the next step of phase 1b; this run stopped \
         at the segmentation"
    );
    if failed > 0 {
        bail!("{failed} of {} fragments could not be preprocessed", entries.len());
    }
    Ok(())
}

/// Reads a fixture dump, runs the requested stages against it and prints D §10.2's table.
fn parity(args: &ParityArgs) -> Result<()> {
    let dir = FixtureDir::new(&args.fixtures);
    let collection = Collection::open(dir, args.input.as_deref())
        .with_context(|| format!("reading the fixture in {}", args.fixtures.display()))?;
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
                "unknown stage `{name}`; this build compares {} or `all` (the later stages of \
                 D §10.2 arrive with the stages they judge)",
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
    use super::{Cli, requested_stages};
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
        let err = requested_stages(&["breakline".to_owned()]).unwrap_err().to_string();
        assert!(err.contains("segmentation"), "{err}");
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
