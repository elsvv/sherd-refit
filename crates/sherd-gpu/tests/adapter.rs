//! The tests that need a real adapter (D §10.4 layer 3).
//!
//! Every other test in this crate is arithmetic — chunking, the dispatch fold, the slot LRU, the
//! reduction's CPU mirror — and runs on all four CI platforms with no driver at all. These three
//! need a device, so they **skip with a printed reason** rather than failing when there is none:
//! the `check` and `test` jobs of D §10.5 run on hosted runners that usually have no GPU, and a
//! red test there would say nothing about the code.
//!
//! A skip is not silence. Each test prints what it looked for and what it found, so
//! `cargo test -- --nocapture` on a runner tells the operator whether the platform was covered.
//! That matters most on macOS, where there is no software adapter to fall back to for a second
//! opinion (E7 §6): a self-test failure there means the CPU path and nothing else.

use sherd_core::executor::CPU;
use sherd_core::executor::Engine;
use sherd_core::executor::Executor;
use sherd_core::executor::batch::IcpBatch;
use sherd_core::executor::batch::{CoarseBatch, Poses};
use sherd_core::matching::coarse::Target;
use sherd_core::matching::icp::{
    Estimation, IcpTarget, Options, Pose, Rotation, Translation, homogeneous, pose_gap,
};
use sherd_core::pipeline::RunOptions;
use sherd_core::progress::{Cancel, Watch};
use sherd_core::spatial::kdtree::PointTree;
use sherd_gpu::device::{AdapterChoice, Requirements};
use sherd_gpu::selftest::AUTO_SPEEDUP;
use sherd_gpu::{AutoPolicy, Gpu, GpuExecutor, Selection, SelfTest};

/// Opens the default adapter, or prints why it could not and returns `None`.
fn device(what: &str) -> Option<Gpu> {
    let adapters = Gpu::adapters();
    if adapters.is_empty() {
        println!("SKIP {what}: no Metal, Vulkan or DX12 adapter on this machine");
        return None;
    }
    match Gpu::open(&AdapterChoice::Default) {
        Ok(gpu) => {
            println!("{what}: running on {}", gpu.entry());
            Some(gpu)
        }
        Err(e) => {
            println!("SKIP {what}: {e}");
            None
        }
    }
}

/// The self-test on an open device, or `None` with a printed reason — every kernel test needs one
/// because `GpuExecutor::new` takes it, and a device whose self-test fails is not one to compare
/// a kernel on.
fn selftest(what: &str, gpu: &Gpu) -> Option<SelfTest> {
    match SelfTest::run(gpu) {
        Ok(test) if test.passed() => Some(test),
        Ok(test) => {
            println!("SKIP {what}: the self-test failed: {}", test.failures().join("; "));
            None
        }
        Err(e) => {
            println!("SKIP {what}: {e}");
            None
        }
    }
}

/// Enumeration is consistent with selection: `--gpu-adapter` by index and by name both land on an
/// adapter that is in the list, and every entry can be opened by its own index.
#[test]
fn the_adapters_the_instance_lists_are_the_ones_the_flag_selects() {
    let adapters = Gpu::adapters();
    if adapters.is_empty() {
        println!("SKIP adapter enumeration: no Metal, Vulkan or DX12 adapter on this machine");
        return;
    }
    for (i, entry) in adapters.iter().enumerate() {
        assert_eq!(entry.index, i, "the index is the position in the enumeration");
        assert!(!entry.name.is_empty(), "an adapter without a name cannot be selected by name");
        assert_eq!(AdapterChoice::parse(&i.to_string()).pick(&adapters).unwrap(), i);
        // Metal reports no vendor, device, driver or driver_info (E7 §7.6), so the name is all
        // `--gpu-adapter NAME` has; it must at least match itself.
        assert_eq!(AdapterChoice::Name(entry.name.clone()).pick(&adapters).unwrap(), i);
        println!("  {entry}");
    }
    assert_eq!(AdapterChoice::Default.pick(&adapters).unwrap(), 0);
    assert!(AdapterChoice::Name("no such adapter exists".to_owned()).pick(&adapters).is_err());
}

/// The device that opens clears the *portable* floor — the wgpu defaults — not merely its own
/// limits, and reports at least them back.
#[test]
fn an_opened_device_clears_the_portable_limits() {
    let Some(gpu) = device("limits") else { return };
    let unmet = Requirements::default().unmet(gpu.limits());
    assert!(unmet.is_empty(), "a device that opened is below D §6's floor: {unmet:?}");
    assert!(gpu.limits().max_compute_invocations_per_workgroup >= 256);
    assert!(gpu.limits().max_compute_workgroups_per_dimension >= 65_535);
    println!("  unified memory: {}, software adapter: {}", gpu.is_unified(), gpu.is_software());
}

/// D §6.8's self-test on whatever adapter this machine has: the reduction bit-identical, the
/// bounded-NN kernel agreeing inside E7 §5.1's measured residue, and a throughput figure.
///
/// The `Backend::Auto` rule is asserted on the result, including the clause that keeps it on the
/// CPU under `GpuExecutor::AUTO` — which is a policy about integrated adapters, and has nothing
/// to do with the self-test's own ratio.
#[test]
fn the_self_test_holds_on_this_machines_adapter() {
    let Some(gpu) = device("self-test") else { return };
    let test = SelfTest::run(&gpu).expect("the self-test runs or names its error");
    for check in &test.checks {
        println!(
            "  {:<12} {} — {}",
            check.name,
            if check.passed { "ok" } else { "FAIL" },
            check.detail
        );
    }
    assert!(test.passed(), "{:?}", test.failures());

    // The reduction row is the one that asserts bits rather than a tolerance (E7 §3).
    let reduction = test.checks.iter().find(|c| c.name == "reduction").expect("a reduction row");
    assert!(reduction.passed, "{}", reduction.detail);

    // The throughput row is a measurement, never a pass criterion: a slow GPU is a reason not to
    // use it, not a broken one.
    assert!(test.ns_per_query() > 0.0);
    assert!(test.queries > 0);

    // D §6.8's rule, on the real numbers: the executor is not `Auto`-eligible, whatever the
    // device did on the self-test's own batch.
    let today = Selection::decide(test.clone(), GpuExecutor::AUTO);
    assert!(!today.use_gpu, "the policy is the CPU until a discrete adapter: {}", today.reason);
    println!("  auto (as shipped): {}", today.reason);

    // And on a machine whose stage *has* been measured, the decision is the ratio against
    // D §6.8's 1.5x.
    let with_kernels = Selection::decide(test.clone(), AutoPolicy::MeasuredOnThisMachine);
    assert_eq!(
        with_kernels.use_gpu,
        test.speedup >= AUTO_SPEEDUP && !test.adapter.is_software(),
        "{}",
        with_kernels.reason
    );
    println!("  auto (with kernels): {}", with_kernels.reason);
}

/// The coarse kernel against the CPU executor on a batch built so that **nothing is a boundary
/// case**: every probe point is either 0.05 from its nearest breakline point or 0.43 from any of
/// them, against a radius of 0.2, and every normal pair scores 1 or −1 against a threshold of 0.7.
///
/// With no near-tie, no query at the radius and no dot at the threshold, the two executors must
/// agree **exactly** — the same integer count and therefore, since the host does the CPU's own
/// `f64` division, the same bits. That is the assertion, and it is what makes the second test's
/// tolerance meaningful: a disagreement there is a boundary case and not a broken traversal.
#[test]
fn the_coarse_kernel_is_exact_where_nothing_is_a_boundary_case() {
    let Some(gpu) = device("coarse kernel, clean batch") else { return };
    let Some(test) = selftest("coarse kernel, clean batch", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    // A 12³ lattice at spacing 0.5: the nearest neighbour of any point is unique by a wide margin.
    let mut target = Vec::new();
    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                target.push([f64::from(i) * 0.5, f64::from(j) * 0.5, f64::from(k) * 0.5]);
            }
        }
    }
    let target_normals = vec![[0.0, 0.0, 1.0]; target.len()];
    let tree = PointTree::build(&target).expect("a non-empty lattice");
    let view = Target { points: &target, normals: &target_normals, tree: &tree };

    // Sixty probe points: a third land 0.05 from a lattice point with an agreeing normal, a third
    // land there with an opposing one, and a third sit at a cell centre, 0.433 from anything.
    let mut points = Vec::new();
    let mut normals = Vec::new();
    for k in 0..60 {
        let base = [f64::from(k % 5) * 0.5 + 2.0, f64::from(k % 3) * 0.5 + 2.0, 2.0];
        match k % 3 {
            0 => {
                points.push([base[0] + 0.05, base[1], base[2]]);
                normals.push([0.0, 0.0, 1.0]);
            }
            1 => {
                points.push([base[0] + 0.05, base[1], base[2]]);
                normals.push([0.0, 0.0, -1.0]);
            }
            _ => {
                points.push([base[0] + 0.25, base[1] + 0.25, base[2] + 0.25]);
                normals.push([0.0, 0.0, 1.0]);
            }
        }
    }

    // Rotations about z leave the normals alone, so the agreement test stays at ±1; the
    // translations are small enough that "inside" stays inside and "outside" stays outside.
    // Four thousand poses × sixty points is 240 000 queries, over `coarse::MIN_QUERIES`, so the
    // batch actually reaches the device — the assertion at the end of this test is what checks it.
    let poses: Vec<_> = (0..4000)
        .map(|p| {
            let angle = f64::from(p % 200) * 0.0007;
            let (s, c) = angle.sin_cos();
            let r = Rotation::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
            let tau = Translation::new(f64::from(p % 7) * 0.004, 0.0, f64::from(p % 5) * 0.004);
            homogeneous(&r, &tau)
        })
        .collect();

    let batch = CoarseBatch {
        target: view,
        mask: None,
        points: &points,
        normals: &normals,
        poses: Poses::Homogeneous(&poses),
        radius: 0.2,
        normal_agree: 0.7,
    };
    let cpu = CPU.coarse_scores(&batch);
    let gpu_scores = executor.coarse_scores(&batch);
    assert_eq!(cpu.len(), gpu_scores.len());
    let differing = cpu.iter().zip(&gpu_scores).filter(|(c, g)| c.to_bits() != g.to_bits()).count();
    println!(
        "  {} poses, {} probe points: {differing} scores differ; first score {} vs {}",
        poses.len(),
        points.len(),
        cpu[0],
        gpu_scores[0]
    );
    assert!(cpu[0] > 0.0, "the batch has to score something for the test to mean anything");
    assert_eq!(differing, 0, "no boundary case, so the counts and the bits must agree");
    assert_eq!(executor.stats().coarse.snapshot().on_device, 1, "it ran on the device");
    assert_eq!(executor.stats().coarse.snapshot().delegated, 0);
}

/// The coarse kernel on E7 §5's own shape — a warped random sheet, where near-ties are the norm —
/// gated on D §10.2's `coarse` row (`1/60 + 1e-6`, one probe point of sixty) and on every
/// disagreement being a single probe point rather than a different traversal.
#[test]
fn the_coarse_kernel_agrees_within_one_probe_point_on_a_random_sheet() {
    let Some(gpu) = device("coarse kernel, random sheet") else { return };
    let Some(test) = selftest("coarse kernel, random sheet", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    let (sheet, probe, poses) = sherd_gpu::selftest::nn_cloud(4000, 4096);
    let widen = |p: &[f32; 3]| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
    let target: Vec<[f64; 3]> = sheet.iter().map(widen).collect();
    let points: Vec<[f64; 3]> = probe.iter().take(60).map(widen).collect();
    // Normals that vary over the sheet, so the 0.7 test is exercised rather than saturated.
    let normal_of = |p: &[f64; 3]| {
        let a = (p[0] * 3.1).sin() * 0.3;
        let b = (p[1] * 2.7).cos() * 0.3;
        let n = (1.0 + a * a + b * b).sqrt();
        [a / n, b / n, 1.0 / n]
    };
    let target_normals: Vec<[f64; 3]> = target.iter().map(normal_of).collect();
    let normals: Vec<[f64; 3]> = points.iter().map(normal_of).collect();
    let tree = PointTree::build(&target).expect("a non-empty sheet");
    let view = Target { points: &target, normals: &target_normals, tree: &tree };
    let transforms: Vec<_> = poses
        .iter()
        .map(|rows| {
            let r = Rotation::new(
                f64::from(rows[0][0]),
                f64::from(rows[0][1]),
                f64::from(rows[0][2]),
                f64::from(rows[1][0]),
                f64::from(rows[1][1]),
                f64::from(rows[1][2]),
                f64::from(rows[2][0]),
                f64::from(rows[2][1]),
                f64::from(rows[2][2]),
            );
            let tau = Translation::new(
                f64::from(rows[0][3]),
                f64::from(rows[1][3]),
                f64::from(rows[2][3]),
            );
            homogeneous(&r, &tau)
        })
        .collect();

    let batch = CoarseBatch {
        target: view,
        mask: None,
        points: &points,
        normals: &normals,
        poses: Poses::Homogeneous(&transforms),
        radius: 0.03,
        normal_agree: 0.7,
    };
    let cpu = CPU.coarse_scores(&batch);
    let gpu_scores = executor.coarse_scores(&batch);
    let step = 1.0 / 60.0;
    let mut worst = 0.0_f64;
    let mut differing = 0_usize;
    for (c, g) in cpu.iter().zip(&gpu_scores) {
        let delta = (c - g).abs();
        worst = worst.max(delta);
        differing += usize::from(c.to_bits() != g.to_bits());
    }
    let scored = cpu.iter().filter(|&&s| s > 0.0).count();
    println!(
        "  {} poses × 60 points: {differing} scores differ, worst {worst:.6} ({:.2} probe \
         points), {scored} poses score above zero",
        cpu.len(),
        worst / step
    );
    assert!(scored * 2 > cpu.len(), "most poses must score something: {scored} of {}", cpu.len());
    assert!(
        worst <= step + 1e-6,
        "D §10.2's coarse row is one probe point of sixty; this batch moved {worst}"
    );
}

/// A plausible wall thickness for the synthetic sheet below, which is 40 units across; D §10.2's
/// translation rows are stated in it.
const SYNTHETIC_T: f64 = 2.0;

/// A synthetic ICP batch: a warped sheet with normals as the target, a rigidly displaced copy as
/// the source, and one small perturbation per candidate as the initial poses.
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
    // A sheet 40 units across, 130 units from the origin — the magnitudes a scan actually has, so
    // that the shifted frame of `sherd_gpu::icp` is doing the work it exists to do.
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
        // The source sits 0.3 units off the sheet, well inside a 1.0 correspondence radius.
        source.push([p[0] + 0.2, p[1] - 0.1, p[2] + 0.15]);
    }
    let icp_target = IcpTarget::new(target, normals);
    let inits: Vec<Pose> = (0..candidates)
        .map(|k| {
            #[allow(clippy::cast_precision_loss, reason = "candidate counts are small")]
            let angle = (k as f64 - 0.5) * 0.0009;
            let (s, c) = angle.sin_cos();
            let r = Rotation::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
            // A rotation about the origin moves a cloud 130 units away by a lot, so the
            // translation puts it back: these are poses an ICP could plausibly be handed.
            let centre = Translation::new(130.0, -120.0, 95.0);
            let tau = centre - r * centre + Translation::new(0.05, -0.04, 0.03);
            homogeneous(&r, &tau)
        })
        .collect();
    (source, icp_target, inits)
}

/// The ICP kernel against the CPU executor on synthetic batches, both estimators.
///
/// The gate is D §10.2's stage-1/stage-2 rows — 0.05° and 0.01 t — over a batch built to be well
/// conditioned: a dense sheet, a source that overlaps it everywhere, and initial poses a few
/// hundredths of a degree apart. A batch like that has no chaotic ladder in it, so this test is
/// about the kernel's arithmetic and nothing else. The pairs that *are* chaotic are the
/// collections', and `gpu-check --chaos` is where they are measured.
///
/// The displacement is measured **at the cloud**, as the furthest a source point moves between the
/// two poses, and not at the origin. D §10.2's own translation column is the displacement of the
/// origin and says why ("a pure rotation difference shows up in both rows — which is the
/// conservative way round"); on this sheet, which sits 130 units from the origin exactly as a scan
/// does, that conservatism is the whole number: 6.2e-3 degrees of rotation about a point 130 units
/// away *is* 1.4e-2 units at the origin and 2e-3 units at the cloud. `gpu-check`'s table keeps
/// D §10.2's form because it is comparing against poses D §10.2's tolerances were calibrated on;
/// here, where the geometry is this file's own, the honest quantity is where the points go.
/// `T` is 2 units — a plausible wall for a 40-unit fragment — and the row is 0.01 t.
#[test]
fn the_icp_kernel_matches_the_cpu_executor_on_synthetic_batches() {
    let Some(gpu) = device("icp kernel") else { return };
    let Some(test) = selftest("icp kernel", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    for (estimation, iterations, name) in [
        (Estimation::PointToPoint, 20_usize, "point-to-point"),
        (Estimation::PointToPlane, 30, "point-to-plane"),
    ] {
        let (source, target, inits) = icp_batch(4000, 64);
        let options = Options {
            estimation,
            max_correspondence_distance: 1.0,
            max_iteration: iterations,
            numerics: sherd_core::matching::icp::Numerics::REFERENCE,
        };
        let batch = IcpBatch { source: &source, target: &target, inits: &inits, options };
        let cpu = CPU.icp_rung(&batch);
        let device_out = executor.icp_rung(&batch);
        assert_eq!(cpu.len(), device_out.len());

        let mut worst_deg = 0.0_f64;
        let mut worst_point = 0.0_f64;
        let mut worst_origin = 0.0_f64;
        let mut worst_fitness = 0.0_f64;
        let mut moved = 0_usize;
        for (c, g) in cpu.iter().zip(&device_out) {
            // The Frobenius form, and `t = 1` because this test reports the displacement in the
            // synthetic cloud's own units rather than in wall thicknesses.
            worst_deg = worst_deg.max(pose_gap::frobenius_deg(&c.transform, &g.transform));
            worst_origin = worst_origin.max(pose_gap::origin_t(&c.transform, &g.transform, 1.0));
            for p in &source {
                let mut moved2 = 0.0;
                for i in 0..3 {
                    let a = (0..3).map(|j| c.transform[(i, j)] * p[j]).sum::<f64>()
                        + c.transform[(i, 3)];
                    let b = (0..3).map(|j| g.transform[(i, j)] * p[j]).sum::<f64>()
                        + g.transform[(i, 3)];
                    moved2 += (a - b).powi(2);
                }
                worst_point = worst_point.max(moved2.sqrt());
            }
            worst_fitness = worst_fitness.max((c.fitness - g.fitness).abs());
            moved += usize::from(c.iterations != g.iterations);
        }
        println!(
            "  {name}: 64 candidates × 4000 points × {iterations} iterations — worst \
             {worst_deg:.3e} deg, {:.3e} t at the cloud ({worst_origin:.3e} units at the \
             origin), fitness {worst_fitness:.3e}; {moved} differing iteration counts; cpu \
             fitness {:.4}",
            worst_point / SYNTHETIC_T,
            cpu[0].fitness
        );
        assert!(cpu[0].fitness > 0.5, "the batch has to register for the test to mean anything");
        assert!(worst_deg <= 0.05, "{name}: D §10.2's stage rows are 0.05 deg, got {worst_deg}");
        assert!(
            worst_point <= 0.01 * SYNTHETIC_T,
            "{name}: D §10.2's stage rows are 0.01 t, got {} t",
            worst_point / SYNTHETIC_T
        );
        // D §10.2's `fitness` row is 1e-4, and it was calibrated on the clouds R §5.6 registers,
        // which are ten to thirty thousand points. On this 4 000-point sheet a *single*
        // correspondence at the radius boundary — the tie E7 §5.1 measured at 2.3e-6 of queries,
        // and the one thing an `f32` search is entitled to resolve differently — is already
        // 2.5e-4 of the fitness. The row here is therefore one correspondence or 1e-4, whichever
        // is larger, and the count is what it is really asserting.
        #[allow(clippy::cast_precision_loss, reason = "4 000 points")]
        let one_correspondence = 1.0 / source.len() as f64;
        assert!(
            worst_fitness <= 1e-4_f64.max(one_correspondence),
            "{name}: fitness moved {worst_fitness}, more than one correspondence of {}",
            source.len()
        );
    }
}

/// Where the ICP rung starts paying on this device: the CPU rung against the GPU rung at candidate
/// counts from one to 256.
///
/// The table `sherd_gpu::icp::MIN_CANDIDATES` was set from, re-run. D §6.4 puts one workgroup on
/// one candidate, so a rung of one candidate is one workgroup on a sixteen-core GPU while the CPU
/// has ten cores and SIMD: the GPU cannot win there, and before the threshold existed a
/// `--backend gpu` terracotta run's refine stage was 2.6× slower than the CPU's for exactly that
/// reason.
///
/// **The rows below the threshold now read ≈ 1.0×, and that is the assertion**: the executor sent
/// them to the CPU, so both columns are the same code. The rows above it are the device's. The
/// numbers the threshold came from, measured with the threshold removed, are in
/// `sherd_gpu::icp::MIN_CANDIDATES`'s own documentation.
#[test]
fn the_icp_rung_crossover_is_a_candidate_count() {
    // Not in a debug build. The GPU side is a compiled kernel in every profile while the CPU side
    // is `sherd-core`, a workspace member and therefore `-O0` here (G1 §3's caveat: the same batch
    // took 22.8 ms release and 131.2 ms debug), so the ratio would be the optimiser's and not the
    // device's — and the 256 × 12 000 cell would spend minutes saying so.
    if cfg!(debug_assertions) {
        println!("SKIP icp crossover: a debug build measures the optimiser, not the device");
        return;
    }
    let Some(gpu) = device("icp crossover") else { return };
    // 256 candidates × 12 000 points × 30 iterations is 92 M bounded queries. That is a second on
    // this Metal part and minutes on lavapipe or WARP, and the answer would be about the software
    // rasteriser rather than about a GPU either way (D §6.8: `Auto` never prefers one to the CPU).
    if gpu.is_software() {
        println!("SKIP icp crossover: {} is a software adapter", gpu.entry());
        return;
    }
    let Some(test) = selftest("icp crossover", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);
    println!("  candidates  points  cpu ms   gpu ms   ratio");
    for &points in &[2000_usize, 12000] {
        for &candidates in &[1_usize, 4, 16, 64, 256] {
            let (source, target, inits) = icp_batch(points, candidates);
            let options = Options {
                estimation: Estimation::PointToPlane,
                max_correspondence_distance: 1.0,
                max_iteration: 30,
                numerics: sherd_core::matching::icp::Numerics::REFERENCE,
            };
            let batch = IcpBatch { source: &source, target: &target, inits: &inits, options };
            let started = std::time::Instant::now();
            let cpu = CPU.icp_rung(&batch);
            let cpu_time = started.elapsed().as_secs_f64() * 1e3;
            let started = std::time::Instant::now();
            let before = executor.stats().icp.snapshot().on_device;
            let out = executor.icp_rung(&batch);
            let gpu_time = started.elapsed().as_secs_f64() * 1e3;
            let on_device = executor.stats().icp.snapshot().on_device > before;
            assert_eq!(cpu.len(), out.len());
            println!(
                "  {candidates:>10}  {points:>6}  {cpu_time:>6.1}   {gpu_time:>6.1}   {:>5.2}x  {}",
                cpu_time / gpu_time,
                if on_device { "device" } else { "cpu (below the threshold)" }
            );
            let wanted = sherd_gpu::icp::on_device(candidates, points);
            assert_eq!(on_device, wanted, "{candidates} × {points} went the wrong way");
            // The ratios are printed and not asserted. `cargo test` runs these tests in parallel
            // on the same ten cores and the same one GPU, so a timing assertion here would be
            // measuring the test harness; the numbers the threshold was set from are in
            // `sherd_gpu::icp::MIN_CANDIDATES` and were taken with `--test-threads=1`.
        }
    }
}

/// Where the coarse score starts paying on this device: the CPU against the GPU at pose counts
/// from one to 40 000, at R §5.2's sixty probe points and at R §5.4's larger point set.
///
/// The same shape as the ICP crossover, and the same conclusion in a different place:
/// `sherd_gpu::coarse::MIN_QUERIES` is set from this table, and the rows below it read ≈ 1.0×
/// because the executor sent them to the CPU.
///
/// Since task G4 the table has a **second** boundary in it, and the two are measured under
/// different conditions on purpose. This test runs on an idle device, and on an idle device the
/// biggest cell — 40 000 poses on 800 points, 32 M queries — is the *best* one, 3.6×. In a run it
/// is not, because the ten cores are busy beside the device (D §6.6), and `coarse::MAX_QUERIES`
/// is set from runs rather than from this table. So the assertion below reads both thresholds,
/// and the printed ratio of the last row is the measurement that says why a ceiling could never
/// have been derived here.
#[test]
fn the_coarse_crossover_is_a_query_count() {
    // Not in a debug build, for the reason the ICP crossover gives.
    if cfg!(debug_assertions) {
        println!("SKIP coarse crossover: a debug build measures the optimiser, not the device");
        return;
    }
    let Some(gpu) = device("coarse crossover") else { return };
    // 40 000 poses × 800 points is 32 M queries — the same reason as the ICP crossover above.
    if gpu.is_software() {
        println!("SKIP coarse crossover: {} is a software adapter", gpu.entry());
        return;
    }
    let Some(test) = selftest("coarse crossover", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    let (sheet, probe, poses) = sherd_gpu::selftest::nn_cloud(6000, 1);
    let widen = |p: &[f32; 3]| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
    let target: Vec<[f64; 3]> = sheet.iter().map(widen).collect();
    let target_normals = vec![[0.0, 0.0, 1.0]; target.len()];
    let tree = PointTree::build(&target).expect("a non-empty sheet");
    let view = Target { points: &target, normals: &target_normals, tree: &tree };
    let _ = poses;

    println!("  poses  points  queries   cpu ms   gpu ms   ratio");
    for &points in &[60_usize, 800] {
        let moving: Vec<[f64; 3]> = probe.iter().take(points).map(widen).collect();
        let normals = vec![[0.0, 0.0, 1.0]; moving.len()];
        for &n in &[1_usize, 64, 1000, 40_000] {
            let transforms: Vec<_> = (0..n)
                .map(|k| {
                    #[allow(clippy::cast_precision_loss, reason = "pose counts are small")]
                    let angle = k as f64 * 1e-5;
                    let (s, c) = angle.sin_cos();
                    let r = Rotation::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
                    homogeneous(&r, &Translation::new(0.0, 0.0, 0.0))
                })
                .collect();
            let batch = CoarseBatch {
                target: view,
                mask: None,
                points: &moving,
                normals: &normals,
                poses: Poses::Homogeneous(&transforms),
                radius: 0.03,
                normal_agree: 0.7,
            };
            let started = std::time::Instant::now();
            let cpu = CPU.coarse_scores(&batch);
            let cpu_time = started.elapsed().as_secs_f64() * 1e3;
            let before = executor.stats().coarse.snapshot().on_device;
            let started = std::time::Instant::now();
            let out = executor.coarse_scores(&batch);
            let gpu_time = started.elapsed().as_secs_f64() * 1e3;
            let on_device = executor.stats().coarse.snapshot().on_device > before;
            assert_eq!(cpu.len(), out.len());
            println!(
                "  {n:>5}  {points:>6}  {:>7}  {cpu_time:>6.1}   {gpu_time:>6.1}   {:>5.2}x  {}",
                n * points,
                cpu_time / gpu_time,
                if on_device {
                    "device"
                } else if n * points < sherd_gpu::coarse::MIN_QUERIES {
                    "cpu (under MIN_QUERIES)"
                } else {
                    "cpu (over MAX_QUERIES)"
                }
            );
            let queries = n * points;
            assert_eq!(
                on_device,
                (sherd_gpu::coarse::MIN_QUERIES..=sherd_gpu::coarse::MAX_QUERIES)
                    .contains(&queries),
                "{n} × {points} went the wrong way"
            );
        }
    }
}

/// The coarse chunking test's poses repeat with this period, which is what makes the fold
/// checkable across a chunk boundary and across a two-dimensional grid.
const PERIOD: usize = 200;

/// D §6.4's TDR bound and E7 §2's two-dimensional dispatch, both on one batch (task G3, item 4).
///
/// The coarse kernel has two caps and this batch crosses both at once: 400 000 poses on a
/// sixty-point probe is **24 M point-queries**, past `coarse::MAX_POINT_QUERIES`, so the call
/// becomes two dispatches; and the first of those is 333 333 workgroups, past
/// `MAX_WORKGROUPS_PER_DIM`, so its grid folds into two dimensions and the kernel has to recover
/// its pose index from `params.wg_x` (E7 §3: the shape is data, never a constant on one side).
///
/// Two assertions, and the second is the one that would catch a wrong fold:
///
/// * the call is **two** dispatches, so the chunking really ran;
/// * the poses repeat with period 200, so score `p` must equal score `p % 200` — across the
///   65 535-workgroup boundary, across the chunk boundary and across `first_pose` — and the first
///   200 of them must be exactly what the CPU executor computes.
///
/// The executor is put in `force_device` mode because 24 M queries is over `coarse::MAX_QUERIES`,
/// the batch **policy** task G4 measured: in a run a batch this size is the CPU's. This test is
/// about the *kernel*, so it asks for the kernel, exactly as `gpu-check` does — which is what
/// `force_device` exists for. The policy has its own test in `coarse.rs`.
#[test]
fn the_coarse_kernel_chunks_past_the_dispatch_caps_and_folds_the_grid() {
    let Some(gpu) = device("coarse chunking") else { return };
    let Some(test) = selftest("coarse chunking", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);
    executor.force_device(true);

    let mut target = Vec::new();
    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                target.push([f64::from(i) * 0.5, f64::from(j) * 0.5, f64::from(k) * 0.5]);
            }
        }
    }
    let target_normals = vec![[0.0, 0.0, 1.0]; target.len()];
    let tree = PointTree::build(&target).expect("a non-empty lattice");
    let view = Target { points: &target, normals: &target_normals, tree: &tree };

    let mut points = Vec::new();
    let mut normals = Vec::new();
    for k in 0..60 {
        let base = [f64::from(k % 5) * 0.5 + 2.0, f64::from(k % 3) * 0.5 + 2.0, 2.0];
        points.push([base[0] + 0.05, base[1], base[2]]);
        normals.push(if k % 2 == 0 { [0.0, 0.0, 1.0] } else { [0.0, 0.0, -1.0] });
    }

    let pose_of = |p: usize| {
        let step = f64::from(u32::try_from(p % PERIOD).unwrap_or(0));
        let angle = step * 0.0007;
        let (s, c) = angle.sin_cos();
        let r = Rotation::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
        let tau = Translation::new(step * 0.0001, 0.0, 0.0);
        homogeneous(&r, &tau)
    };
    let poses: Vec<_> = (0..400_000).map(pose_of).collect();
    let batch = CoarseBatch {
        target: view,
        mask: None,
        points: &points,
        normals: &normals,
        poses: Poses::Homogeneous(&poses),
        radius: 0.2,
        normal_agree: 0.7,
    };
    let expect_dispatches = poses.len().div_ceil(sherd_gpu::coarse::MAX_POINT_QUERIES / 60);
    let scores = executor.coarse_scores(&batch);
    let snapshot = executor.stats().coarse.snapshot();
    println!(
        "  {} poses × {} points = {} queries in {} dispatches ({} on device)",
        poses.len(),
        points.len(),
        poses.len() * points.len(),
        snapshot.dispatches,
        snapshot.on_device,
    );
    assert_eq!(snapshot.on_device, 1, "one call");
    assert_eq!(
        usize::try_from(snapshot.dispatches).unwrap_or(0),
        expect_dispatches,
        "the query cap has to have split it"
    );
    assert!(expect_dispatches > 1, "the batch is meant to cross the cap");

    // The head, against the reference implementation, on its own batch so that it is a comparison
    // of answers and not of schedules.
    let head: Vec<_> = poses[..PERIOD].to_vec();
    let head_batch = CoarseBatch { poses: Poses::Homogeneous(&head), ..batch };
    let cpu = CPU.coarse_scores(&head_batch);
    let differing =
        cpu.iter().zip(&scores[..PERIOD]).filter(|(c, g)| c.to_bits() != g.to_bits()).count();
    assert!(cpu.iter().any(|&s| s > 0.0), "the batch has to score something");
    assert_eq!(differing, 0, "the first chunk's head is the CPU's, to the bit");

    // And every pose is its own residue's score, which is what a wrong `first_pose` or a wrong
    // `gid.x + gid.y · wg_x` would break.
    let wrong = (0..poses.len())
        .filter(|&p| scores[p].to_bits() != scores[p % PERIOD].to_bits())
        .take(4)
        .collect::<Vec<_>>();
    assert!(wrong.is_empty(), "poses {wrong:?} do not match their own period-200 residue");
}

/// D §6.4's 512-candidate ceiling for Windows TDR, on the ICP kernel (task G3, item 4).
///
/// 600 candidates is two dispatches; the initial poses repeat with period 100, so candidate `k`
/// and candidate `k % 100` are the same problem and must come back the same registration — and
/// candidate 512, the first of the second dispatch, has a twin in the first. That is what checks
/// that a chunk's state buffer, its `first` and its slice of the one staging buffer line up.
///
/// The ICP grid never folds into two dimensions: one workgroup is one candidate and a dispatch is
/// capped at 512 of them, three orders below `MAX_WORKGROUPS_PER_DIM`. The fold is the coarse
/// kernel's, and the test above is where it is exercised.
/// Candidates in the ICP chunking test — past D §6.4's 512 ceiling, so it is two dispatches.
const CANDIDATES: usize = 600;
/// How often the ICP chunking test's initial poses repeat.
const ICP_PERIOD: usize = 100;

#[test]
fn the_icp_kernel_chunks_at_the_tdr_ceiling() {
    let Some(gpu) = device("icp chunking") else { return };
    let Some(test) = selftest("icp chunking", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    let (source, target, seed) = icp_batch(2000, ICP_PERIOD);
    let inits: Vec<Pose> = (0..CANDIDATES).map(|k| seed[k % ICP_PERIOD]).collect();
    let options = Options {
        estimation: Estimation::PointToPlane,
        max_correspondence_distance: 1.0,
        max_iteration: 5,
        numerics: sherd_core::matching::icp::Numerics::REFERENCE,
    };
    let batch = IcpBatch { source: &source, target: &target, inits: &inits, options };
    let out = executor.icp_rung(&batch);
    let snapshot = executor.stats().icp.snapshot();
    println!(
        "  {CANDIDATES} candidates × {} points in {} dispatches ({} on device)",
        source.len(),
        snapshot.dispatches,
        snapshot.on_device,
    );
    assert_eq!(snapshot.on_device, 1, "one call");
    assert_eq!(
        usize::try_from(snapshot.dispatches).unwrap_or(0),
        CANDIDATES.div_ceil(sherd_gpu::icp::MAX_CANDIDATES),
        "600 candidates is two dispatches at a ceiling of 512",
    );
    assert_eq!(out.len(), CANDIDATES);
    for k in 0..CANDIDATES {
        let (a, b) = (&out[k], &out[k % ICP_PERIOD]);
        assert_eq!(a.iterations, b.iterations, "candidate {k} took a different path");
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (a.transform[(i, j)] - b.transform[(i, j)]).abs() < 1e-12,
                    "candidate {k} and its twin {} differ at ({i},{j})",
                    k % ICP_PERIOD,
                );
            }
        }
    }
}

/// D §1's device-memory ceiling, exercised from both sides (task G3, item 5).
///
/// A batch that fits under the default 1 GB runs on the device; the *same* batch under a ceiling
/// of one megabyte is refused before a single buffer is created, answered by the CPU executor and
/// counted — as a delegation, which it is, and as a refusal, which says why. The two answers are
/// then compared, because a budget that changed a result would be a bug and not a policy.
#[test]
fn the_device_memory_budget_sends_what_it_cannot_hold_to_the_cpu() {
    let Some(gpu) = device("device memory budget") else { return };
    let Some(test) = selftest("device memory budget", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    let (source, target, inits) = icp_batch(4000, 64);
    let options = Options {
        estimation: Estimation::PointToPlane,
        max_correspondence_distance: 1.0,
        max_iteration: 5,
        numerics: sherd_core::matching::icp::Numerics::REFERENCE,
    };
    let batch = IcpBatch { source: &source, target: &target, inits: &inits, options };

    let generous = executor.icp_rung(&batch);
    let after = executor.stats().icp.snapshot();
    let peak = executor.gpu().allocations().peak();
    println!(
        "  under the default budget: {} on device, {} delegated, peak {} bytes",
        after.on_device, after.delegated, peak,
    );
    assert_eq!(after.on_device, 1, "the batch fits under D §1's 1 GB");
    assert!(peak > 0, "the reservation has to have been made");
    assert_eq!(executor.gpu().allocations().live(), 0, "and released when the call returned");

    executor.gpu().allocations().set_budget(1024 * 1024);
    let starved = executor.icp_rung(&batch);
    let after = executor.stats().icp.snapshot();
    println!(
        "  under 1 MB: {} on device, {} delegated, {} refused",
        after.on_device,
        after.delegated,
        executor.gpu().allocations().refused(),
    );
    assert_eq!(after.on_device, 1, "no second call reached the device");
    assert_eq!(after.delegated, 1, "the starved call was answered by the CPU");
    assert_eq!(executor.gpu().allocations().refused(), 1);

    // The CPU's answer is the reference implementation's, so the starved call is the CPU's own
    // output to the bit — and the device's is inside D §10.2, which its own test asserts.
    let cpu = CPU.icp_rung(&batch);
    assert_eq!(starved.len(), cpu.len());
    for (a, b) in starved.iter().zip(&cpu) {
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    a.transform[(i, j)].to_bits(),
                    b.transform[(i, j)].to_bits(),
                    "a refused batch is the CPU executor's own answer",
                );
            }
        }
    }
    assert_eq!(generous.len(), cpu.len());
}

/// A shortfall that is only what else is in flight **waits**, and the answer is the device's
/// (V6-D8).
///
/// The defect this closes is not a wrong answer, it is a schedule-dependent one: the old
/// `Allocations::reserve` refused whenever `live + bytes` crossed the ceiling, and `live` is
/// whatever other workers happen to hold at that instant. Which call went to the CPU was then a
/// function of thread timing, and D §7's byte-identical gate rests on it not being.
///
/// Two threads submit the same batch under a ceiling of exactly one batch. One of them must wait
/// for the other. The assertions are that **both** reach the device, that neither is refused, that
/// at least one waited, and that both answers are bit-identical to the same batch's answer on an
/// empty device — the wait moved the batch in time and in nothing else.
#[test]
fn a_shortfall_that_is_other_calls_in_flight_waits_rather_than_delegating() {
    let Some(gpu) = device("device memory shortfall") else { return };
    let Some(test) = selftest("device memory shortfall", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    let (source, target, inits) = icp_batch(4000, 64);
    let options = Options {
        estimation: Estimation::PointToPlane,
        max_correspondence_distance: 1.0,
        max_iteration: 5,
        numerics: sherd_core::matching::icp::Numerics::REFERENCE,
    };
    let batch = IcpBatch { source: &source, target: &target, inits: &inits, options };

    // One call on an empty device fixes both the reference answer and the batch's own size.
    let alone = executor.icp_rung(&batch);
    let one_batch = executor.gpu().allocations().peak();
    assert!(one_batch > 0, "the reservation has to have been made");
    assert_eq!(executor.stats().icp.snapshot().delegated, 0, "it fits under the default budget");

    // A ceiling of exactly one batch, with one batch's worth already held here. The shortfall is
    // therefore certain rather than raced: the spawned call cannot proceed until this thread
    // releases, and the only question the test asks is what it does in the meantime.
    let allocations = executor.gpu().allocations();
    allocations.set_budget(one_batch);
    let held = allocations.reserve(one_batch).expect("one batch fits in a budget of one batch");
    let finished = std::sync::atomic::AtomicBool::new(false);
    let waited = std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let answer = executor.icp_rung(&batch);
            finished.store(true, std::sync::atomic::Ordering::SeqCst);
            answer
        });
        // Nothing can release but this thread, so the call is still inside `reserve`.
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(
            !finished.load(std::sync::atomic::Ordering::SeqCst),
            "the call cannot have answered while the whole budget is held here",
        );
        assert_eq!(allocations.refused(), 0, "a shortfall is not a refusal");
        assert_eq!(executor.stats().icp.snapshot().delegated, 0, "nor a delegation");
        drop(held);
        worker.join().expect("the waiting call finished once the budget was released")
    });
    let after = executor.stats().icp.snapshot();
    println!(
        "  budget {one_batch} B (one batch), one held: {} on device, {} delegated, {} refused, \
         {} waited",
        after.on_device,
        after.delegated,
        allocations.refused(),
        allocations.waited(),
    );
    assert_eq!(after.on_device, 2, "both calls reached the device");
    assert_eq!(after.delegated, 0, "a shortfall is not a delegation");
    assert_eq!(allocations.refused(), 0, "and it is not a refusal");
    assert_eq!(allocations.waited(), 1, "the one that could not fit waited, and is counted");
    assert_eq!(allocations.live(), 0, "both reservations were released");

    assert_eq!(waited.len(), alone.len());
    for (a, b) in waited.iter().zip(&alone) {
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    a.transform[(i, j)].to_bits(),
                    b.transform[(i, j)].to_bits(),
                    "waiting for room moved the batch in time and in nothing else",
                );
            }
        }
    }
}

/// D §5's cancellation on the GPU path: the run stops, and the device is still there afterwards
/// (task G3, item 6).
///
/// The claim being tested is the one that could not be checked on the CPU: *"Ctrl-C mid-batch
/// leaves no device hang."* The run is cancelled from its own progress callback — deterministic,
/// no sleep — and then a real batch is put through the same device. If a cancelled run had left a
/// submission unretired, an unmapped buffer or a submitting thread waiting on a fence, that batch
/// would hang or fail; it does neither, because a dispatch is bounded by D §6.4's chunking and
/// `pipeline::Submitter` retires every command buffer it submitted.
#[test]
fn a_cancelled_gpu_run_leaves_the_device_usable() {
    let Some(gpu) = device("cancellation on the gpu path") else { return };
    let Some(test) = selftest("cancellation on the gpu path", &gpu) else { return };
    let executor = GpuExecutor::new(std::sync::Arc::new(gpu), test);

    let input = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input");
    let out = std::env::temp_dir().join(format!("sherd-gpu-cancel-{}", std::process::id()));
    std::fs::remove_dir_all(&out).ok();

    let cancel = Cancel::new();
    let progress = std::sync::Arc::new(CancelAtOnce(cancel.clone()));
    let watch = Watch { cancel: Some(cancel), progress: Some(progress) };
    let options = RunOptions {
        preview: false,
        write_meshes: false,
        cache: false,
        watch,
        ..RunOptions::default()
    };
    let engine = Engine::new(&executor, sherd_core::matching::icp::Numerics::REFERENCE);
    let result = sherd_core::pipeline::run_with(&input, &out, &options, engine);
    assert!(
        matches!(result, Err(sherd_core::error::Error::Cancelled)),
        "expected Cancelled, got {result:?}",
    );

    // And now the device, on a batch big enough to reach it.
    let before = executor.stats().icp.snapshot().on_device;
    let (source, target, inits) = icp_batch(4000, 64);
    let options = Options {
        estimation: Estimation::PointToPlane,
        max_correspondence_distance: 1.0,
        max_iteration: 5,
        numerics: sherd_core::matching::icp::Numerics::REFERENCE,
    };
    let batch = IcpBatch { source: &source, target: &target, inits: &inits, options };
    let out_poses = executor.icp_rung(&batch);
    assert_eq!(out_poses.len(), 64);
    assert_eq!(
        executor.stats().icp.snapshot().on_device,
        before + 1,
        "the device answered a batch after the cancelled run",
    );
    assert_eq!(executor.gpu().allocations().live(), 0, "nothing is still reserved");
    println!("  the run stopped and the device answered {} candidates afterwards", out_poses.len());
    std::fs::remove_dir_all(&out).ok();
}

/// A `Progress` whose only act is to raise the flag on the first unit it is told about.
#[derive(Debug)]
struct CancelAtOnce(Cancel);

impl sherd_core::progress::Progress for CancelAtOnce {
    fn advance(&self, _stage: &str, _done: usize, _total: usize) {
        self.0.cancel();
    }
}
