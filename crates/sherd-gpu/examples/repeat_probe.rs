//! Is one ICP batch's device answer a function of the batch alone? (Task W, defect W-D1.)
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
//! answer. If a repeat differs, the kernel's output is not a function of its input, and no amount
//! of host-side determinism can fix that; if every repeat agrees, the variation is upstream — in
//! which batch is formed, or which executor answers it.
//!
//! ```text
//! cargo run --release -p sherd-gpu --example repeat_probe -- [repeats] [points] [candidates]
//! ```
//!
//! Run it a second time with the adapter under load — another `run --backend gpu` in a second
//! terminal is enough — because that is the condition the defect was seen under.
use sherd_core::executor::Executor;
use sherd_core::executor::batch::IcpBatch;
use sherd_core::matching::icp::{
    Estimation, IcpTarget, Numerics, Options, Pose, Rotation, Translation, homogeneous,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let repeats = arg(1, 64);
    let points = arg(2, 4000);
    let candidates = arg(3, 64);

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
    for estimation in [Estimation::PointToPlane, Estimation::PointToPoint] {
        let options = Options {
            estimation,
            max_correspondence_distance: 1.0,
            max_iteration: 30,
            numerics: Numerics::REFERENCE,
        };
        let batch = IcpBatch { source: &source, target: &target, inits: &inits, options };
        let first = executor.icp_rung(&batch);
        let mut differing = 0_usize;
        let mut worst_candidates = 0_usize;
        let mut worst_iterations = 0_usize;
        for _ in 1..repeats {
            let again = executor.icp_rung(&batch);
            let mut moved = 0_usize;
            let mut iterations = 0_usize;
            for (a, b) in first.iter().zip(&again) {
                let mut same = a.iterations == b.iterations && a.converged == b.converged;
                for i in 0..4 {
                    for j in 0..4 {
                        same &= a.transform[(i, j)].to_bits() == b.transform[(i, j)].to_bits();
                    }
                }
                if !same {
                    moved += 1;
                    let delta = a.iterations.abs_diff(b.iterations);
                    iterations = iterations.max(delta);
                }
            }
            if moved > 0 {
                if differing == 0 {
                    for (k, (a, b)) in first.iter().zip(&again).enumerate().take(3) {
                        let untouched = inits[k]
                            .iter()
                            .zip(b.transform.iter())
                            .all(|(i, t)| i.to_bits() == t.to_bits());
                        println!(
                            "    candidate {k}: first {} it (converged {}), repeat {} it \
                             (converged {}), repeat == its own initial pose: {untouched}",
                            a.iterations, a.converged, b.iterations, b.converged,
                        );
                    }
                }
                differing += 1;
                worst_candidates = worst_candidates.max(moved);
                worst_iterations = worst_iterations.max(iterations);
            }
        }
        println!(
            "{estimation:?}: {repeats} answers to one batch of {candidates} candidates × {points} \
             points — {differing} differ from the first (worst: {worst_candidates} candidates, \
             {worst_iterations} iterations apart)",
        );
    }
    let stats = executor.stats();
    for line in stats.lines() {
        println!("  {line}");
    }
    Ok(())
}
