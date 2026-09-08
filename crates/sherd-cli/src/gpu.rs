//! What the CLI does about `--backend`, `--gpu-adapter` and `gpu-check` (D §6.8, D §9, D §10.4).
//!
//! The whole of the GPU crate's surface is used from here and nowhere else, so a build with
//! `--no-default-features` — no `sherd-gpu`, no wgpu, no `naga` — differs from the default build
//! in this file alone. That is D §2's "`sherd-gpu` is an optional feature of `sherd-cli`".
//!
//! # What `--backend` means in phase 2a
//!
//! | flag | with an adapter | without one |
//! |---|---|---|
//! | `auto` (default) | the device is opened, the self-test runs, and the CPU is used anyway — because phase 2a's executor has no kernels (D §12) — with the measured speedup logged | the CPU, silently |
//! | `cpu` | the CPU; no device is opened at all | the CPU |
//! | `gpu` | the device is opened and the self-test must pass, or the run **fails**; every `Executor` call is then counted as delegated | the run **fails**, naming what was tried |
//!
//! The one thing this file must never do is let a run believe it used the GPU when it did not.
//! `--backend gpu` therefore prints, once, that the executor has no kernels yet and that the
//! numbers are the CPU's; `report.json` still records the backend the *run asked for*, which is
//! what D §4.3 says it records.

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
        let mut lines = vec![format!(
            "cpu, gpu (wgpu 30.0.1, {} adapter{}; kernels arrive in phase 2b)",
            adapters.len(),
            if adapters.len() == 1 { "" } else { "s" }
        )];
        lines.extend(adapters.iter().map(|a| format!("  {a}")));
        lines
    }
    #[cfg(not(feature = "gpu"))]
    {
        vec!["cpu (built without the `gpu` feature)".to_owned()]
    }
}

/// What a run resolved `--backend` to: the engine to pass down, the backend to record, and the
/// sentence to log.
#[derive(Debug)]
pub(crate) struct Resolved {
    /// The backend `report.json` records (D §4.3) — what the run asked for, resolved.
    pub(crate) backend: Backend,
    /// The executor the pipeline actually runs on, and D §7's numerics.
    pub(crate) engine: Engine<'static>,
    /// One sentence naming the deciding fact, for the log and for `--verbose`.
    pub(crate) reason: String,
}

/// D §6.8's resolution of `--backend` and `--gpu-adapter`.
///
/// The GPU executor is leaked deliberately when one is built: it owns a `wgpu::Device` that must
/// outlive every batch of the run, the run *is* the process, and `Engine<'static>` is what the
/// pipeline's signature wants. One leak per process, of one device.
#[cfg(feature = "gpu")]
pub(crate) fn resolve(backend: Backend, adapter: Option<&str>) -> Result<Resolved> {
    use std::sync::Arc;

    use sherd_gpu::{AdapterChoice, Gpu, GpuExecutor, Selection, SelfTest};

    if backend == Backend::Cpu {
        return Ok(Resolved {
            backend: Backend::Cpu,
            engine: Engine::REFERENCE,
            reason: "--backend cpu".to_owned(),
        });
    }
    let choice = adapter.map_or(AdapterChoice::Default, AdapterChoice::parse);
    let opened = Gpu::open(&choice).and_then(|gpu| {
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
                 batch). Phase 2a's GPU executor has no kernels yet (D §12: 2b and 2c), so every \
                 batch is computed by the CPU implementation and the results are the CPU's.",
                test.adapter.name,
                test.ns_per_query(),
                test.speedup,
            );
            let executor: &'static GpuExecutor =
                Box::leak(Box::new(GpuExecutor::new(Arc::new(gpu), test)));
            Ok(Resolved {
                backend: Backend::Gpu,
                engine: Engine::new(executor, Numerics::default()),
                reason,
            })
        }
        // `auto`: never fails, and in phase 2a never picks the GPU. `Selection` is the rule, and
        // it is the same rule phase 2b will use with `HAS_KERNELS` flipped.
        (_, Err(e)) => Ok(Resolved {
            backend: Backend::Cpu,
            engine: Engine::REFERENCE,
            reason: Selection::no_gpu(&e).reason,
        }),
        (_, Ok((gpu, test))) => {
            let selection = Selection::decide(test, GpuExecutor::HAS_KERNELS);
            if !selection.use_gpu {
                drop(gpu);
                return Ok(Resolved {
                    backend: Backend::Cpu,
                    engine: Engine::REFERENCE,
                    reason: selection.reason,
                });
            }
            let test = selection.selftest.expect("a GPU selection carries its self-test");
            let executor: &'static GpuExecutor =
                Box::leak(Box::new(GpuExecutor::new(Arc::new(gpu), test)));
            Ok(Resolved {
                backend: Backend::Gpu,
                engine: Engine::new(executor, Numerics::default()),
                reason: selection.reason,
            })
        }
    }
}

/// [`resolve`] for a build without the `gpu` feature: `auto` is the CPU, `gpu` is an error.
#[cfg(not(feature = "gpu"))]
pub(crate) fn resolve(backend: Backend, _adapter: Option<&str>) -> Result<Resolved> {
    match backend {
        Backend::Gpu => {
            bail!("--backend gpu: this binary was built without the `gpu` feature (D §2)")
        }
        _ => Ok(Resolved {
            backend: Backend::Cpu,
            engine: Engine::REFERENCE,
            reason: "built without the `gpu` feature".to_owned(),
        }),
    }
}

/// One row of the `gpu-check` table (D §10.4 layer 3, E7 §5.1's form).
#[derive(Debug)]
pub(crate) struct CheckRow {
    /// The batch that was compared.
    pub(crate) stage: &'static str,
    /// How many values the row compares.
    pub(crate) items: usize,
    /// The largest absolute deviation between the two executors.
    pub(crate) worst: f64,
    /// D §10.2's tolerance for that quantity.
    pub(crate) tolerance: f64,
    /// How many nearest-neighbour choices differ (E7 §5.1's column that matters).
    pub(crate) differing: usize,
    /// `ok`, `FAIL`, `delegated` or `skipped`, with its reason.
    pub(crate) status: String,
}

impl CheckRow {
    /// Whether the row is a failure — a deviation over the tolerance, or a differing choice.
    #[must_use]
    pub(crate) fn failed(&self) -> bool {
        self.status.starts_with("FAIL")
    }
}

/// D §10.2's tolerances, for the four rows the kernels will fill in.
#[cfg(feature = "gpu")]
mod tolerance {
    /// `cs` per hypothesis: one probe point of sixty, plus rounding.
    pub(super) const COARSE: f64 = 1.0 / 60.0 + 1e-6;
    /// A rung's `fitness` and `inlier_rmse`.
    pub(super) const ICP: f64 = 1e-4;
    /// `gap`, in `t`; the tightest distance tolerance of the verification rows.
    pub(super) const DISTANCE: f64 = 0.002;
    /// `pen`, a fraction of the surface samples.
    pub(super) const INSIDE: f64 = 0.0005;
}

/// `sherd-refit-rs gpu-check`: feed identical batches to both executors and report the deviations.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_lines, reason = "one block per Executor method, each self-contained")]
pub(crate) fn check(
    stage: &str,
    set: Option<&std::path::Path>,
    fixture: Option<&std::path::Path>,
    adapter: Option<&str>,
) -> Result<Vec<CheckRow>> {
    use std::sync::Arc;

    use sherd_core::executor::batch::{
        CoarseBatch, DistBatch, DistReduce, IcpBatch, InsideBatch, Poses,
    };
    use sherd_core::executor::{CPU, Executor};
    use sherd_core::matching::coarse::{NORMAL_AGREE, Target};
    use sherd_core::matching::icp::{IcpTarget, cloud_points, homogeneous};
    use sherd_core::matching::ladder;
    use sherd_core::matching::pair::Pair;
    use sherd_core::{Params, collection, pipeline};
    use sherd_gpu::{AdapterChoice, Gpu, GpuExecutor, SelfTest};

    let choice = adapter.map_or(AdapterChoice::Default, AdapterChoice::parse);
    let gpu = Gpu::open(&choice)?;
    let selftest = SelfTest::run(&gpu)?;
    let mut rows: Vec<CheckRow> = selftest
        .checks
        .iter()
        .map(|c| CheckRow {
            stage: c.name,
            items: 0,
            worst: 0.0,
            tolerance: 0.0,
            differing: 0,
            status: format!("{} — {}", if c.passed { "ok" } else { "FAIL" }, c.detail),
        })
        .collect();
    let executor = GpuExecutor::new(Arc::new(gpu), selftest);

    let wanted = |name: &str| stage == "all" || stage == name;
    let delegated = |name: &'static str, tolerance: f64, items: usize| CheckRow {
        stage: name,
        items,
        worst: 0.0,
        tolerance,
        differing: 0,
        status: "delegated — phase 2a's GPU executor routes this method to the CPU (D §12: 2b, 2c)"
            .to_owned(),
    };

    // Without a collection there is nothing to form batches from, and the four rows say so rather
    // than reporting a zero nobody measured.
    let Some(input) = set.or(fixture) else {
        for (name, tol) in [
            ("coarse", tolerance::COARSE),
            ("icp", tolerance::ICP),
            ("distance", tolerance::DISTANCE),
            ("inside", tolerance::INSIDE),
        ] {
            if wanted(name) {
                rows.push(CheckRow {
                    stage: name,
                    items: 0,
                    worst: 0.0,
                    tolerance: tol,
                    differing: 0,
                    status: "skipped — no --set or --fixture, so no batch to compare".to_owned(),
                });
            }
        }
        return Ok(rows);
    };
    if fixture.is_some() && set.is_none() {
        // A fixture dump's own `_run` tree holds the collection the dump came from; pointing
        // `--fixture` at one is D §10.1's shape and it is not read here yet.
        bail!(
            "--fixture is not read by `gpu-check` yet: the batches it would form are the parity \
             harness's, and the four Executor rows have no kernels to compare in phase 2a. Use \
             `--set DIR` to exercise the batch formation on a real collection."
        );
    }

    // The first pair of the collection that R §4.1 does not skip: enough to form one batch of
    // every kind, and cheap enough to run at a console.
    let params = Params::default();
    let entries = collection::discover(input)?;
    if entries.len() < 2 {
        bail!("{}: need at least two mesh files", input.display());
    }
    let fragments: Vec<_> = pipeline::preprocess(
        &entries[..2],
        200_000,
        None,
        sherd_core::memory::Budget::default_for_machine(),
    )
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()?;
    let (a, b) = (&fragments[0].fragment, &fragments[1].fragment);
    let pair = Pair::build(a, b, &params);
    if !pair.matchable() {
        bail!("{}: the first two fragments have no breakline to match", input.display());
    }
    let hyp = pair.hypotheses(&params);
    let probe = pair.probe(&params);
    let tree = pair.a.kd_brk.as_ref().expect("a matchable pair has a breakline tree");
    let target = Target { points: &pair.frames_a.p, normals: &pair.frames_a.ns, tree };

    // --- R §5.2's coarse score -----------------------------------------------------------------
    if wanted("coarse") && !hyp.is_empty() {
        let batch = CoarseBatch {
            target,
            mask: None,
            points: &probe.q,
            normals: &probe.qn,
            poses: Poses::Split { r: &hyp.r, tau: &hyp.tau },
            radius: pair.scales.coarse,
            normal_agree: NORMAL_AGREE,
        };
        let (cpu, gpu) = (CPU.coarse_scores(&batch), executor.coarse_scores(&batch));
        let worst = worst_deviation(&cpu, &gpu);
        rows.push(CheckRow { worst, ..delegated("coarse", tolerance::COARSE, cpu.len()) });
    } else if wanted("coarse") {
        rows.push(delegated("coarse", tolerance::COARSE, 0));
    }

    // --- one rung of R §7's ICP ----------------------------------------------------------------
    let cs = pair.coarse(sherd_core::executor::Engine::REFERENCE, &hyp, &probe);
    let kept = pair.suppress(&hyp, &cs, &params);
    let icp_target = IcpTarget::from_cloud(&pair.a.pc_brk_full);
    let source = cloud_points(&pair.b.pc_brk);
    let inits: Vec<_> = kept
        .iter()
        .take(32)
        .map(|&h| homogeneous(&hyp.r[h as usize], &hyp.tau[h as usize]))
        .collect();
    if wanted("icp") && !inits.is_empty() {
        let rung = ladder::stage1_rungs(&source, &icp_target)[0];
        let batch = IcpBatch {
            source: rung.source,
            target: rung.target,
            inits: &inits,
            options: sherd_core::matching::icp::Options::point_to_point(
                pair.scales.icp_dist(rung.k),
                rung.iterations,
            ),
        };
        let (cpu, gpu) = (CPU.icp_rung(&batch), executor.icp_rung(&batch));
        let worst = cpu
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c.fitness - g.fitness).abs().max((c.inlier_rmse - g.inlier_rmse).abs()))
            .fold(0.0_f64, f64::max);
        let differing =
            cpu.iter().zip(&gpu).filter(|(c, g)| c.correspondences != g.correspondences).count();
        rows.push(CheckRow { worst, differing, ..delegated("icp", tolerance::ICP, cpu.len()) });
    } else if wanted("icp") {
        rows.push(delegated("icp", tolerance::ICP, 0));
    }

    // --- R §6.1's bounded distance and R §6.4's inside test ------------------------------------
    let identity = homogeneous(&nalgebra_identity(), &nalgebra_zero());
    let pose = inits.first().copied().unwrap_or(identity);
    if let Some((sa, sb)) = pair.surfaces() {
        if wanted("distance") {
            let batch = DistBatch {
                points: &sb.pf,
                transform: &pose,
                scene: sa.fracture,
                max_dist: pair.scales.facing,
                reduce: DistReduce::All,
            };
            let (cpu, gpu) = (CPU.bounded_distance(&batch), executor.bounded_distance(&batch));
            let worst = worst_deviation(&cpu, &gpu);
            let differing =
                cpu.iter().zip(&gpu).filter(|(c, g)| c.is_finite() != g.is_finite()).count();
            rows.push(CheckRow {
                worst: worst / pair.scales.t,
                differing,
                ..delegated("distance", tolerance::DISTANCE, cpu.len())
            });
        }
        if wanted("inside") {
            match sa.mesh {
                Some(mesh) => {
                    let batch = InsideBatch { points: &sb.s, transform: &pose, scene: mesh };
                    let (cpu, gpu) = (CPU.inside(&batch), executor.inside(&batch));
                    let differing =
                        cpu.iter().zip(&gpu).filter(|(c, g)| c.inside != g.inside).count();
                    let worst = cpu
                        .iter()
                        .zip(&gpu)
                        .map(|(c, g)| f64::from((c.depth - g.depth).abs()))
                        .fold(0.0_f64, f64::max);
                    rows.push(CheckRow {
                        worst: worst / pair.scales.t,
                        differing,
                        ..delegated("inside", tolerance::INSIDE, cpu.len())
                    });
                }
                None => rows.push(CheckRow {
                    status: "skipped — the first fragment has no surface mesh (R §6.4)".to_owned(),
                    ..delegated("inside", tolerance::INSIDE, 0)
                }),
            }
        }
    }
    Ok(rows)
}

/// [`check`] for a build without the `gpu` feature.
#[cfg(not(feature = "gpu"))]
pub(crate) fn check(
    _stage: &str,
    _set: Option<&std::path::Path>,
    _fixture: Option<&std::path::Path>,
    _adapter: Option<&str>,
) -> Result<Vec<CheckRow>> {
    bail!("gpu-check: this binary was built without the `gpu` feature (D §2)")
}

/// The 3×3 identity, without a `nalgebra` dependency of this crate's own.
#[cfg(feature = "gpu")]
fn nalgebra_identity() -> sherd_core::matching::icp::Rotation {
    sherd_core::matching::icp::Rotation::identity()
}

/// The zero translation.
#[cfg(feature = "gpu")]
fn nalgebra_zero() -> sherd_core::matching::icp::Translation {
    sherd_core::matching::icp::Translation::zeros()
}

/// The largest absolute difference between two aligned arrays; `∞` on both sides counts as zero.
#[cfg(feature = "gpu")]
fn worst_deviation(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| if x.is_infinite() && y.is_infinite() { 0.0 } else { (x - y).abs() })
        .fold(0.0_f64, f64::max)
}
