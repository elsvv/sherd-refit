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
use sherd_core::executor::Executor;
use sherd_core::executor::batch::IcpBatch;
use sherd_core::executor::batch::{CoarseBatch, Poses};
use sherd_core::matching::coarse::Target;
use sherd_core::matching::icp::{
    Estimation, IcpTarget, Options, Pose, Rotation, Translation, homogeneous,
};
use sherd_core::spatial::kdtree::PointTree;
use sherd_gpu::device::{AdapterChoice, Requirements};
use sherd_gpu::selftest::AUTO_SPEEDUP;
use sherd_gpu::{Gpu, GpuExecutor, Selection, SelfTest};

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
/// CPU while `GpuExecutor::AUTO_ELIGIBLE` is false — which it is on this machine, and for a
/// measured reason that has nothing to do with the self-test's own ratio.
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
    let today = Selection::decide(test.clone(), GpuExecutor::AUTO_ELIGIBLE);
    assert!(!today.use_gpu, "the stage is 1.0x on ten threads: {}", today.reason);
    println!("  auto (as shipped): {}", today.reason);

    // And when it is eligible, the decision is the measured ratio against D §6.8's 1.5x.
    let with_kernels = Selection::decide(test.clone(), true);
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
            let mut frobenius = 0.0;
            let mut origin = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    frobenius += (c.transform[(i, j)] - g.transform[(i, j)]).powi(2);
                }
                origin += (c.transform[(i, 3)] - g.transform[(i, 3)]).powi(2);
            }
            let half = (frobenius.sqrt() / (2.0 * std::f64::consts::SQRT_2)).clamp(-1.0, 1.0);
            worst_deg = worst_deg.max(2.0 * half.asin().to_degrees());
            worst_origin = worst_origin.max(origin.sqrt());
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
    let Some(gpu) = device("icp crossover") else { return };
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
            let wanted = candidates >= sherd_gpu::icp::MIN_CANDIDATES
                && candidates * points >= sherd_gpu::icp::MIN_WORK;
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
#[test]
fn the_coarse_crossover_is_a_query_count() {
    let Some(gpu) = device("coarse crossover") else { return };
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
                if on_device { "device" } else { "cpu (below the threshold)" }
            );
            assert_eq!(
                on_device,
                n * points >= sherd_gpu::coarse::MIN_QUERIES,
                "{n} × {points} went the wrong way"
            );
        }
    }
}
