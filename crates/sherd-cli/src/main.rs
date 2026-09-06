//! `sherd-refit-rs` — the command line front end of the Rust core.
//!
//! The subcommands are D §9's: `run` and `segment` mirror the Python's, flag for flag, and
//! `parity`, `bench` and `info` are new. Phase 1a implements `info` and the part of `parity`
//! that does not need the pipeline — reading a fixture dump and checking that it is intact;
//! `run`, `segment` and `bench` arrive with the stages they drive (plan steps S2–S4 and phase
//! 1d) and report that plainly until then.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use sherd_core::{ALGO_REF, Backend, CACHE_VERSION, CORE_VERSION, Params};
use sherd_parity::FixtureDir;

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
    /// Segment fragments into shell and fracture surface, without matching (R §3).
    Segment(RunArgs),
    /// Work with the parity fixtures of D §10.
    Parity(ParityArgs),
    /// Time the pipeline against the gates of D §10.3.
    Bench(BenchArgs),
    /// Print what this build is: versions, algorithm reference, backends.
    Info,
}

/// Arguments shared by `run` and `segment`; the full flag set of R §1.4 lands in phase 1d.
#[derive(Debug, Args)]
struct RunArgs {
    /// Directory of fragment files (`.ply`, `.obj`, `.stl`, `.off`).
    input: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Face budget of the working mesh; 0 keeps the adaptive budget of R §3.3.
    #[arg(long, default_value_t = 0)]
    target_faces: u32,
    /// Worker threads; 0 means one per core.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Executor: `auto`, `cpu` or `gpu` (D §6.8).
    #[arg(long, default_value_t = Backend::Auto)]
    backend: Backend,
}

/// Arguments of `parity`.
#[derive(Debug, Args)]
struct ParityArgs {
    /// A fixture dump written by `tools/dump_fixtures.py` (D §10.1).
    fixtures: PathBuf,
    /// Re-hash every file and compare against the manifest.
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
        Command::Segment(_) => bail!("`segment` lands in phase 1b, with the segmentation stage"),
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

/// Reads a fixture dump and reports what it holds — the first half of D §10.2's `parity`
/// subcommand; running the stages against it follows in plan step S4.
fn parity(args: &ParityArgs) -> Result<()> {
    let dir = FixtureDir::new(&args.fixtures);
    let manifest = dir
        .load_manifest()
        .with_context(|| format!("reading the fixture manifest in {}", args.fixtures.display()))?;

    println!("fixture:    {}", args.fixtures.display());
    println!("  commit:   {}{}", manifest.commit, if manifest.dirty { " (dirty)" } else { "" });
    println!("  level:    {}", manifest.level);
    println!("  open3d:   {}, numpy {}", manifest.open3d, manifest.numpy);
    println!("  files:    {} in {} fragments", manifest.files.len(), manifest.pairs.names.len());
    println!("  pairs:    {}", manifest.pairs.pairs.len());
    println!(
        "  params:   {}",
        if manifest.collection.params == Params::default() {
            "the defaults of R §1.1".to_owned()
        } else {
            "not the defaults — the comparison must use the dump's own values".to_owned()
        }
    );

    if args.verify_checksums {
        let bad = dir.verify_checksums().context("verifying the fixture's checksums")?;
        if bad.is_empty() {
            println!("  checksums: all {} files match the manifest", manifest.files.len());
        } else {
            for path in &bad {
                println!("  MISMATCH: {path}");
            }
            bail!("{} of {} files do not match the manifest", bad.len(), manifest.files.len());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::CommandFactory;

    #[test]
    fn the_command_line_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_binary_is_named_for_the_transition() {
        assert_eq!(Cli::command().get_name(), "sherd-refit-rs");
    }
}
