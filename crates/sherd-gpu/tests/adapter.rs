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
use sherd_core::executor::batch::{CoarseBatch, Poses};
use sherd_core::matching::coarse::Target;
use sherd_core::matching::icp::{Rotation, Translation, homogeneous};
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
/// CPU while `GpuExecutor::HAS_KERNELS` is false.
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

    // D §6.8's rule, on the real numbers: in phase 2a it is the CPU whatever the device did.
    let phase2a = Selection::decide(test.clone(), GpuExecutor::HAS_KERNELS);
    assert!(!phase2a.use_gpu, "phase 2a has no kernels: {}", phase2a.reason);
    println!("  auto (phase 2a): {}", phase2a.reason);

    // And with kernels, the decision is the measured ratio against D §6.8's 1.5x.
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
    let poses: Vec<_> = (0..200)
        .map(|p| {
            let angle = f64::from(p) * 0.0007;
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

    let (sheet, probe, poses) = sherd_gpu::selftest::nn_cloud(4000, 256);
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
