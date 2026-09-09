//! Wall clock of R §4–§6 per pair, against the reference's own `match_pair` (D §10.2's cost rows).
//!
//! ```text
//! cargo run --release -p sherd-parity --example pair_cost -- INPUT_DIR [THREADS]
//! ```
//!
//! Preprocessing is done first and excluded from the timing, exactly as the Python measurement
//! excludes it: what is timed is `matching::pair::match_pair` on two already-built fragments, so
//! that the two sides are compared on the same work. `THREADS` sizes the rayon pool for the run
//! (the default is rayon's own); the reference's corresponding knob is `match_pair(n_threads=…)`
//! with `OMP_NUM_THREADS=1`, which is the configuration its own pipeline gives its workers.
//!
//! The last line is the mean over the collection's pairs, which is the number D §10.2 quotes.

use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::matching::pair::match_pair;
use sherd_core::params::Params;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: pair_cost INPUT_DIR [THREADS]");
    if let Some(threads) = args.next() {
        let n: usize = threads.parse().expect("THREADS is a number");
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .expect("the global pool is built once");
    }
    let params = Params::default();
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("an input directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                matches!(e.to_ascii_lowercase().as_str(), "ply" | "obj" | "stl" | "off" | "glb")
            })
        })
        .collect();
    files.sort();

    let mut fragments = Vec::new();
    for path in &files {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("fragment").to_owned();
        let started = std::time::Instant::now();
        let (fragment, _) = Fragment::load_or_build(path, 200_000, &name, None, 0)?;
        println!("prep {name}: {:.2} s", started.elapsed().as_secs_f64());
        fragments.push((name, fragment));
    }

    let mut spent = Vec::new();
    for i in 0..fragments.len() {
        for j in i + 1..fragments.len() {
            let (a, b) = (&fragments[i], &fragments[j]);
            let started = std::time::Instant::now();
            let candidates = match_pair(&a.1, &b.1, &params, 5);
            let elapsed = started.elapsed().as_secs_f64();
            spent.push(elapsed);
            println!("  {}__{}  {elapsed:.3} s  ({} candidates)", a.0, b.0, candidates.len());
        }
    }
    let total: f64 = spent.iter().sum();
    #[allow(clippy::cast_precision_loss, reason = "a collection has few pairs")]
    let mean = total / spent.len() as f64;
    let worst = spent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let best = spent.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "MEAN {mean:.3} s over {} pairs  (min {best:.3}, max {worst:.3}, total {total:.1})",
        spent.len()
    );
    Ok(())
}
