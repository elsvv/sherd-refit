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
