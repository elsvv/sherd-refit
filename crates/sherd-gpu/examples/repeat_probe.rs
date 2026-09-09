//! Is one ICP batch's device answer a function of the batch alone? (Task W, defect W-D1; task H1.)
//!
//! D §7 asks for two runs of a collection to be byte-identical, and every verification of that so
//! far ran on a quiet machine. Task W caught two of eight `gpu-check --set input/sfspp/pot_H`
//! runs disagreeing with the other six on their **stage-2** rows — 4.081° against 1.466°, with a
//! candidate whose device iteration count moved by thirty — while a full `run --backend gpu` on
//! `synthetic_20` came back with 376 candidates once and 375 another time. Both happened only
//! while a second process was driving the same adapter; the fragment caches were byte-identical
//! in every case, so the difference is in the matching stage and not in the input to it.
//!
//! This probe cuts the question down to one submission. **The same `IcpBatch` value, answered `N`
//! times by the same `GpuExecutor` in one process**, compared bit for bit against the first
//! answer.
//!
//! # What task H1 made it say
//!
//! The cause is in `notes/2026-09-09-h1-wd1.md`: the driver aborts a command buffer — a second
//! process on the adapter produces several such aborts a minute — and neither `wgpu-hal`'s Metal
//! fence nor `map_async` reports it, so the host reads a buffer the device never wrote. The port
//! now refuses such a readback and answers the batch on the CPU, so a differing repeat is no
//! longer a wrong answer; it is the **CPU's** answer to a batch whose device readback was refused,
//! and the probe reconciles the two counts:
//!
//! * `differing` — repeats whose answer is not bit-identical to the first;
//! * `corrupt` — readbacks the decoder refused over the same repeats;
//! * the gate is `differing == 0` (an idle adapter) **or** `differing <= corrupt` (a contended
//!   one): every repeat that moved has a refusal to account for it.
//!
//! # Modes
//!
//! ```text
//! cargo run --release -p sherd-gpu --example repeat_probe -- [repeats] [points] [candidates] [mode]
//! ```
//!
//! `mode` is `sequential` (the default: every point-to-plane repeat, then every point-to-point
//! one) or `interleaved`, which alternates the two estimators and timestamps every call. The
//! second mode is the audit's experiment E3: task W's table read "PointToPlane 0 of 64,
//! PointToPoint 5–6 of 64" as a property of the estimator, and it is confounded with *when* the
//! other process was in its matching stage. Alternating separates the two.
//!
//! Run it a second time with the adapter under load — another `run --backend gpu` in a second
//! terminal is enough — because that is the condition the defect was seen under.
use std::time::Instant;

use sherd_core::executor::Executor;
use sherd_core::executor::batch::IcpBatch;
use sherd_core::matching::icp::{
    Estimation, IcpTarget, Numerics, Options, Pose, Registration, Rotation, Translation,
    homogeneous,
};
use sherd_gpu::{Gpu, GpuExecutor, SelfTest, device::AdapterChoice};

/// The same synthetic batch `tests/adapter.rs` builds: a warped sheet 130 units from the origin,
/// a source that overlaps it everywhere, and one small perturbation per candidate.
fn icp_batch(points: usize, candidates: usize) -> (Vec<[f64; 3]>, IcpTarget, Vec<Pose>) {
    let mut state = 0x6d2b_79f5_a1b3_c7d9_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss, reason = "24 bits into an f64 mantissa")]
        let unit = (state >> 40) as f64 * (1.0 / 16_777_216.0);
        unit
    };
    let sheet = |u: f64, v: f64| {
        let z = 3.0 * (u * 0.18).sin() * (v * 0.11).cos();
        [130.0 + u, -120.0 + v, 95.0 + z]
    };
    let normal_at = |u: f64, v: f64| {
        let du = 3.0 * 0.18 * (u * 0.18).cos() * (v * 0.11).cos();
        let dv = -3.0 * 0.11 * (u * 0.18).sin() * (v * 0.11).sin();
        let n = (du * du + dv * dv + 1.0).sqrt();
        [-du / n, -dv / n, 1.0 / n]
    };
    let mut target = Vec::with_capacity(points);
    let mut normals = Vec::with_capacity(points);
    let mut source = Vec::with_capacity(points);
    for _ in 0..points {
        let (u, v) = (next() * 40.0, next() * 40.0);
        target.push(sheet(u, v));
        normals.push(normal_at(u, v));
        let (u, v) = (next() * 40.0, next() * 40.0);
        let p = sheet(u, v);
        source.push([p[0] + 0.2, p[1] - 0.1, p[2] + 0.15]);
    }
    let icp_target = IcpTarget::new(target, normals);
    let inits: Vec<Pose> = (0..candidates)
        .map(|k| {
            #[allow(clippy::cast_precision_loss, reason = "candidate counts are small")]
            let angle = (k as f64 - 0.5) * 0.0009;
            let (s, c) = angle.sin_cos();
            let r = Rotation::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
            let centre = Translation::new(130.0, -120.0, 95.0);
            let tau = centre - r * centre + Translation::new(0.05, -0.04, 0.03);
            homogeneous(&r, &tau)
        })
        .collect();
    (source, icp_target, inits)
}

/// How far one answer is from another, in candidates and in iterations.
fn compare(first: &[Registration], again: &[Registration]) -> (usize, usize) {
    let mut moved = 0;
    let mut iterations = 0;
    for (a, b) in first.iter().zip(again) {
        let mut same = a.iterations == b.iterations && a.converged == b.converged;
        for i in 0..4 {
            for j in 0..4 {
                same &= a.transform[(i, j)].to_bits() == b.transform[(i, j)].to_bits();
            }
        }
        if !same {
            moved += 1;
            iterations = iterations.max(a.iterations.abs_diff(b.iterations));
        }
    }
    (moved, iterations)
}

/// What one estimator's repeats came to.
#[derive(Clone, Copy, Default)]
struct Tally {
    repeats: usize,
    differing: usize,
    corrupt: u64,
    worst_candidates: usize,
    worst_iterations: usize,
}

impl Tally {
    fn report(self, what: &str, candidates: usize, points: usize) {
        println!(
            "{what}: {} answers to one batch of {candidates} candidates × {points} points — {} \
             differ from the first, {} readbacks refused (worst: {} candidates, {} iterations \
             apart)",
            self.repeats,
            self.differing,
            self.corrupt,
            self.worst_candidates,
            self.worst_iterations,
        );
        let verdict = if self.differing == 0 {
            "the device answer is a function of the batch"
        } else if u64::try_from(self.differing).unwrap_or(u64::MAX) <= self.corrupt {
            "every repeat that moved has a refused readback to account for it: the CPU answered it"
        } else {
            "UNEXPLAINED: a repeat moved with no refusal to account for it"
        };
        println!("  {verdict}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The decoders' refusals are `tracing::warn!`s, and the census inside one is experiment E1's
    // measurement: how many words of a refused readback are still the sentinel, zero or NaN.
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).with_target(false).init();

    let args: Vec<String> = std::env::args().collect();
    let arg =
        |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let repeats = arg(1, 64);
    let points = arg(2, 4000);
    let candidates = arg(3, 64);
    let interleaved = args.get(4).is_some_and(|s| s.starts_with("inter"));

    let gpu = Gpu::open(&AdapterChoice::Default)?;
    println!("adapter: {}", gpu.entry());
    let test = SelfTest::run(&gpu)?;
    if !test.passed() {
        println!("the self-test failed: {}", test.failures().join("; "));
        return Ok(());
    }
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);
    executor.force_device(true);

    let (source, target, inits) = icp_batch(points, candidates);
    let estimators = [Estimation::PointToPlane, Estimation::PointToPoint];
    let batches: Vec<IcpBatch<'_>> = estimators
        .iter()
        .map(|&estimation| IcpBatch {
            source: &source,
            target: &target,
            inits: &inits,
            options: Options {
                estimation,
                max_correspondence_distance: 1.0,
                max_iteration: 30,
                numerics: Numerics::REFERENCE,
            },
        })
        .collect();

    let started = Instant::now();
    // The baseline has to be an answer the **device** gave. Under contention the very first call
    // is as likely to be refused as any other, and a baseline that is the CPU's answer makes every
    // later device answer look like a difference — which is what the first draft of this probe
    // reported (53 differing against 10 refusals) before it asked.
    let mut before = executor.stats().icp.snapshot().corrupt;
    let mut first = Vec::with_capacity(batches.len());
    for batch in &batches {
        loop {
            let answer = executor.icp_rung(batch);
            let after = executor.stats().icp.snapshot().corrupt;
            let refused = after - before;
            before = after;
            if refused == 0 {
                first.push(answer);
                break;
            }
            println!("  the baseline readback was refused; asking the device again");
        }
    }
    let mut tallies = [Tally { repeats, ..Tally::default() }; 2];

    // `sequential` runs every repeat of one estimator and then every repeat of the other, which is
    // what task W measured; `interleaved` alternates them (experiment E3).
    let order: Vec<usize> = if interleaved {
        (1..repeats).flat_map(|_| [0_usize, 1]).collect()
    } else {
        (0..2).flat_map(|k| std::iter::repeat_n(k, repeats - 1)).collect()
    };
    for which in order {
        let at = started.elapsed().as_secs_f64();
        let again = executor.icp_rung(&batches[which]);
        let after = executor.stats().icp.snapshot().corrupt;
        let refused = after - before;
        before = after;
        let (moved, iterations) = compare(&first[which], &again);
        let tally = &mut tallies[which];
        tally.corrupt += refused;
        if moved > 0 {
            if tally.differing == 0 || refused == 0 {
                println!(
                    "  {:>7.3} s {:?}: {moved} of {candidates} candidates moved, up to \
                     {iterations} iterations apart, {refused} readbacks refused",
                    at, estimators[which],
                );
            }
            tally.differing += 1;
            tally.worst_candidates = tally.worst_candidates.max(moved);
            tally.worst_iterations = tally.worst_iterations.max(iterations);
        }
    }
    for (tally, estimation) in tallies.into_iter().zip(estimators) {
        tally.report(&format!("{estimation:?}"), candidates, points);
    }
    for line in executor.stats().lines() {
        println!("  {line}");
    }
    Ok(())
}
