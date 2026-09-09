//! What the CLI does about `--backend`, `--gpu-adapter` and `gpu-check` (D §6.8, D §9, D §10.4).
//!
//! The whole of the GPU crate's surface is used from here and nowhere else, so a build with
//! `--no-default-features` — no `sherd-gpu`, no wgpu, no `naga` — differs from the default build
//! in this file alone. That is D §2's "`sherd-gpu` is an optional feature of `sherd-cli`".
//!
//! # What `--backend` means
//!
//! | flag | with an adapter | without one |
//! |---|---|---|
//! | `auto` (default) | the device is opened, the self-test runs, and `Selection::decide` applies D §6.8's rule — an adapter that is not a software one, a passing self-test, ≥ 1.5×, **and** an executor with its matching kernels | the CPU, silently |
//! | `cpu` | the CPU; no device is opened at all | the CPU |
//! | `gpu` | the device is opened and the self-test must pass, or the run **fails**; every method without a kernel is then counted as delegated | the run **fails**, naming what was tried |
//!
//! The one thing this file must never do is let a run believe it used the GPU when it did not.
//! `--backend gpu` therefore prints, once, which of the four `Executor` methods are the device's
//! and which are the CPU's; `report.json` records the backend the *run asked for*, which is what
//! D §4.3 says it records.

use anyhow::{Result, bail};
use sherd_core::executor::{Backend, Engine};
#[cfg(feature = "gpu")]
use sherd_core::matching::icp::Numerics;

/// The lines `sherd-refit-rs info` prints under "backends".
#[must_use]
pub(crate) fn info_lines() -> Vec<String> {
    #[cfg(feature = "gpu")]
    {
        let adapters = sherd_gpu::Gpu::adapters();
        if adapters.is_empty() {
            return vec![
                "cpu; gpu built in (wgpu 30.0.1) but no Metal, Vulkan or DX12 adapter found"
                    .to_owned(),
            ];
        }
        // Which of D §6.1's four methods have a kernel is the thing an operator reading `info`
        // wants, and it is not the same question as "is there an adapter".
        let mut lines = vec![format!(
            "cpu, gpu (wgpu 30.0.1, {} adapter{}; {})",
            adapters.len(),
            if adapters.len() == 1 { "" } else { "s" },
            if sherd_gpu::GpuExecutor::HAS_KERNELS {
                "R §5.2's coarse score and R §5.4's stage-1 ICP rungs on the device, R §5.6's \
                 stage 2 and R §6's two methods on the CPU (policy; D §12's 2c struck)"
            } else {
                "no kernels yet"
            }
        )];
        lines.extend(adapters.iter().map(|a| format!("  {a}")));
        lines
    }
    #[cfg(not(feature = "gpu"))]
    {
        vec!["cpu (built without the `gpu` feature)".to_owned()]
    }
}

/// What `sherd-refit-rs info` prints under "gpu self-test": the adapter it opened, every check
/// D §6.8's self-test made, and the two ratios — the kernel's and the matching stage's.
///
/// The two are printed together and named apart on purpose. The self-test measures E7 §5's
/// bounded-NN kernel on an **idle** device and reports 3–6× here; the matching stage, where that
/// kernel's work actually lives, measures 1.0–1.5× because the device and the ten cores share one
/// power and bandwidth envelope (D §6.6, task G3 §5). Reading the first as the second is the
/// mistake task G2's note made and task G3 measured, so `info` prints both and says which is
/// which.
#[must_use]
pub(crate) fn selftest_lines(adapter: Option<&str>) -> Vec<String> {
    #[cfg(feature = "gpu")]
    {
        use sherd_gpu::{AdapterChoice, Gpu, GpuExecutor, Selection, SelfTest, selftest};

        let choice = adapter.map_or(AdapterChoice::Default, AdapterChoice::parse);
        let opened = Gpu::open(&choice).and_then(|gpu| {
            let test = SelfTest::run(&gpu)?;
            Ok(test)
        });
        let test = match opened {
            Ok(test) => test,
            Err(e) => return vec![format!("not run: {e}")],
        };
        let mut lines = vec![format!(
            "{} on {}",
            if test.passed() { "passed" } else { "FAILED" },
            test.adapter
        )];
        lines.extend(test.checks.iter().map(|c| {
            format!("  {} {}: {}", if c.passed { "ok  " } else { "FAIL" }, c.name, c.detail)
        }));
        lines.push(format!(
            "  kernel:  {:.1} ns per bounded-NN query, {:.2}x the whole CPU pool on that batch \
             (an idle device, E7 §5)",
            test.ns_per_query(),
            test.speedup,
        ));
        lines.push(format!(
            "  stage:   {:.2}-{:.2}x measured on the matching stage over the seven development \
             collections ({}) — the device and the cores share one envelope, D §6.6",
            selftest::STAGE_SPEEDUP.0,
            selftest::STAGE_SPEEDUP.1,
            selftest::STAGE_SPEEDUP_SOURCE,
        ));
        let selection = Selection::decide(test, GpuExecutor::AUTO);
        lines.push(format!(
            "  --backend auto: {} — {}",
            if selection.use_gpu { "gpu" } else { "cpu" },
            selection.reason,
        ));
        lines
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = adapter;
        vec!["not run: this binary was built without the `gpu` feature (D §2)".to_owned()]
    }
}

/// What a run resolved `--backend` to: the engine to pass down, the backend to record, and the
/// sentence to log.
#[derive(Debug)]
pub(crate) struct Resolved {
    /// The backend both output files record (D §4.3) — `--backend` **resolved**, never `auto`.
    pub(crate) backend: Backend,
    /// The executor the pipeline actually runs on, and D §7's numerics.
    pub(crate) engine: Engine<'static>,
    /// One sentence naming the deciding fact, for the log and for `--verbose`.
    pub(crate) reason: String,
    /// The adapter's own name when a device was opened and kept, for D §4.3's `backend` field.
    pub(crate) adapter: Option<String>,
    /// The GPU executor, when one was built — so that a run can print how much of it was used.
    #[cfg(feature = "gpu")]
    pub(crate) executor: Option<&'static sherd_gpu::GpuExecutor>,
}

impl Resolved {
    /// One line per `Executor` method: calls, how many reached the device, dispatches, GPU wall
    /// time and the method's own unit of work. Empty on the CPU path.
    ///
    /// This is what makes a timing claim about the GPU checkable. "The matching stage was 15.7 s"
    /// says nothing about *why*; "the device was busy for 6.1 s of it, over 1 520 dispatches" says
    /// whether the device was the limit or the queue in front of it was.
    #[must_use]
    #[cfg_attr(
        not(feature = "gpu"),
        allow(clippy::unused_self, reason = "a build without the crate has no device to report")
    )]
    pub(crate) fn device_lines(&self) -> Vec<String> {
        #[cfg(feature = "gpu")]
        {
            self.executor.map_or_else(Vec::new, |executor| {
                let stats = executor.stats();
                let mut lines = stats.lines();
                let occupancy = executor.gpu().occupancy();
                let allocations = executor.gpu().allocations();
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "byte counts far below 2^53, printed to the megabyte"
                )]
                lines.push(format!(
                    "device memory: {:.0} MB at the peak of a {} budget, {} batches refused (too \
                     big for the whole budget), {} waited for room",
                    allocations.peak() as f64 / (1024.0 * 1024.0),
                    if allocations.budget() == u64::MAX {
                        "no".to_owned()
                    } else {
                        format!(
                            "{:.2} GB",
                            allocations.budget() as f64 / (1024.0 * 1024.0 * 1024.0)
                        )
                    },
                    allocations.refused(),
                    allocations.waited(),
                ));
                lines.push(format!(
                    "device: {:.2} s with work outstanding over {} submissions, {} deep at the \
                     peak (the per-thread sum below counts one submission once per waiter: \
                     {:.2} s)",
                    occupancy.busy().as_secs_f64(),
                    occupancy.submissions(),
                    occupancy.peak_in_flight(),
                    stats.gpu_busy().as_secs_f64(),
                ));
                lines
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            Vec::new()
        }
    }
}

/// D §6.8's resolution of `--backend` and `--gpu-adapter`.
///
/// The GPU executor is leaked deliberately when one is built: it owns a `wgpu::Device` that must
/// outlive every batch of the run, the run *is* the process, and `Engine<'static>` is what the
/// pipeline's signature wants. One leak per process, of one device.
#[cfg(feature = "gpu")]
pub(crate) fn resolve(
    backend: Backend,
    adapter: Option<&str>,
    memory: Option<f64>,
) -> Result<Resolved> {
    use std::sync::Arc;

    use sherd_gpu::{AdapterChoice, Gpu, GpuExecutor, Selection, SelfTest};

    if backend == Backend::Cpu {
        return Ok(Resolved {
            backend: Backend::Cpu,
            engine: Engine::REFERENCE,
            reason: "--backend cpu".to_owned(),
            adapter: None,
            executor: None,
        });
    }
    let choice = adapter.map_or(AdapterChoice::Default, AdapterChoice::parse);
    let opened = Gpu::open(&choice).and_then(|gpu| {
        // D §1's ceiling, or whatever `--gpu-memory` puts in its place, before the self-test runs
        // its own batches through the same accounting.
        if let Some(gigabytes) = memory {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a byte count from a gigabyte flag"
            )]
            let bytes = (gigabytes.max(0.0) * 1024.0 * 1024.0 * 1024.0) as u64;
            gpu.allocations().set_budget(bytes);
        }
        let test = SelfTest::run(&gpu)?;
        Ok((gpu, test))
    });
    match (backend, opened) {
        // `--backend gpu` and something went wrong: fail, loudly, naming what was tried. D §6.8
        // wants a benchmark that asked for the GPU to stop rather than quietly time the CPU, and
        // on macOS there is no software adapter to get a second opinion from (E7 §6).
        (Backend::Gpu, Err(e)) => bail!("--backend gpu: {e}"),
        (Backend::Gpu, Ok((gpu, test))) => {
            if !test.passed() {
                bail!(
                    "--backend gpu: {}",
                    sherd_gpu::GpuError::SelfTest {
                        adapter: test.adapter.to_string(),
                        failures: test.failures(),
                    }
                );
            }
            let reason = format!(
                "--backend gpu on {}: the self-test passed ({:.1} ns/query, {:.2}x the CPU on its \
                 batch). R §5.2's coarse score and R §5.4's stage-1 ICP rungs run on the device, \
                 between the measured size thresholds; R §5.6's stage 2 is the CPU's by policy \
                 (icp::STAGE2_ON_DEVICE: ten candidates under a threshold of sixteen), and \
                 R §6.1's bounded distance and R §6.4's inside test are the CPU implementation's \
                 (D §12's 2c, struck). `gpu-check` says which is which. \
                 The self-test's ratio is a kernel on an idle device and not the stage's: the \
                 matching stage measures {:.2}-{:.2}x, and the device loses 1.6x of its own \
                 throughput as the ten cores fill up (tasks G3, G4).",
                test.adapter.name,
                test.ns_per_query(),
                test.speedup,
                sherd_gpu::selftest::STAGE_SPEEDUP.0,
                sherd_gpu::selftest::STAGE_SPEEDUP.1,
            );
            let name = test.adapter.name.clone();
            let executor: &'static GpuExecutor =
                Box::leak(Box::new(GpuExecutor::new(Arc::new(gpu), test)));
            Ok(Resolved {
                backend: Backend::Gpu,
                engine: Engine::new(executor, Numerics::default()),
                reason,
                adapter: Some(name),
                executor: Some(executor),
            })
        }
        // `auto`: never fails, and today never picks the GPU. `Selection` is the rule, and it is
        // the same rule that will pick the GPU on a machine whose stage has been measured
        // (`GpuExecutor::AUTO`).
        (_, Err(e)) => Ok(Resolved {
            backend: Backend::Cpu,
            engine: Engine::REFERENCE,
            reason: Selection::no_gpu(&e).reason,
            adapter: None,
            executor: None,
        }),
        (_, Ok((gpu, test))) => {
            let selection = Selection::decide(test, GpuExecutor::AUTO);
            if !selection.use_gpu {
                drop(gpu);
                return Ok(Resolved {
                    backend: Backend::Cpu,
                    engine: Engine::REFERENCE,
                    reason: selection.reason,
                    adapter: None,
                    executor: None,
                });
            }
            let test = selection.selftest.expect("a GPU selection carries its self-test");
            let name = test.adapter.name.clone();
            let executor: &'static GpuExecutor =
                Box::leak(Box::new(GpuExecutor::new(Arc::new(gpu), test)));
            Ok(Resolved {
                backend: Backend::Gpu,
                engine: Engine::new(executor, Numerics::default()),
                reason: selection.reason,
                adapter: Some(name),
                executor: Some(executor),
            })
        }
    }
}

/// [`resolve`] for a build without the `gpu` feature: `auto` is the CPU, `gpu` is an error.
///
/// The signature is the one the callers use, argument for argument — that is the whole contract
/// this stub has, and when it drifted from it (`--gpu-memory` added to the callers in G3.1 and not
/// here) the CPU-only build stopped compiling for two whole steps without anyone noticing, because
/// nothing local ran it. D §10.4's local gate list now runs both `--no-default-features` commands
/// before every commit (V6-D1).
#[cfg(not(feature = "gpu"))]
pub(crate) fn resolve(
    backend: Backend,
    _adapter: Option<&str>,
    _memory: Option<f64>,
) -> Result<Resolved> {
    match backend {
        Backend::Gpu => {
            bail!("--backend gpu: this binary was built without the `gpu` feature (D §2)")
        }
        _ => Ok(Resolved {
            backend: Backend::Cpu,
            engine: Engine::REFERENCE,
            reason: "built without the `gpu` feature".to_owned(),
            adapter: None,
        }),
    }
}

/// [`check`] for a build without the `gpu` feature.
/// `sherd-refit-rs gpu-check`: the cross-check table as lines, and how many rows failed.
///
/// The harness itself moved into `sherd_gpu::crosscheck` in task H2 (audit §B.7); what is left
/// here is the adapter between `clap` and it. It returns *lines* rather than rows so that a build
/// without the crate has no row type to mirror.
#[cfg(feature = "gpu")]
pub(crate) fn check(
    stage: &str,
    set: Option<&std::path::Path>,
    fixture: Option<&std::path::Path>,
    adapter: Option<&str>,
    pairs: usize,
    chaos: bool,
    force_device: bool,
) -> Result<(Vec<String>, usize)> {
    let rows =
        sherd_gpu::crosscheck::check(stage, set, fixture, adapter, pairs, chaos, force_device)?;
    let failed = rows.iter().filter(|row| row.failed()).count();
    Ok((sherd_gpu::crosscheck::table(&rows), failed))
}

/// [`check`] for a build without the `gpu` feature.
#[cfg(not(feature = "gpu"))]
pub(crate) fn check(
    _stage: &str,
    _set: Option<&std::path::Path>,
    _fixture: Option<&std::path::Path>,
    _adapter: Option<&str>,
    _pairs: usize,
    _chaos: bool,
    _force_device: bool,
) -> Result<(Vec<String>, usize)> {
    bail!("gpu-check: this binary was built without the `gpu` feature (D §2)")
}
