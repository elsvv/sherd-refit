//! `gpu-check`: the same batches, answered by both executors, as a table (D §10.4 layer 3).
//!
//! This is the harness the phase-2b exit criterion is read off, and until task H2 it lived in
//! `sherd-cli`'s `gpu.rs` — 700 lines of analysis in a binary crate, running only with a device
//! and covered by no unit test (audit §B.7). It belongs here: it is about the kernels, it uses
//! nothing of the CLI but its arguments, and here its pieces can be tested without one.
//!
//! # What the table is read at
//!
//! **The production policy** (audit §A.2.4). [`check`] applies the executor's own size thresholds
//! by default, so the rows describe what a `--backend gpu` run of the collection would do: the
//! device answers R §5.2's coarse score and R §5.4's two stage-1 breakline rungs, and R §5.6's
//! stage 2 is the CPU's by policy ([`icp::STAGE2_ON_DEVICE`](crate::icp::STAGE2_ON_DEVICE)), so
//! its rows print `cpu by policy`. `force_device` puts every rung on the device instead, which is
//! task W's measurement of the kernel and not the criterion.
//!
//! # What gates and what only reports
//!
//! [`CheckRow::failed`] is the whole exit rule, and three kinds of row deliberately never fail:
//! the `ctrl` row (it *defines* the exclusion the pose rows apply, so gating it would gate the
//! same fact twice), the origin-referenced translation row (the criterion is read at the cloud),
//! and the `iter` row (a rung-count difference is a measurement, and D §10.2 states no tolerance
//! for one). Everything else carries D §10.2's own number.

use std::path::Path;

use crate::GpuError;

/// One row of the `gpu-check` table (D §10.4 layer 3, E7 §5.1's form).
#[derive(Debug)]
pub struct CheckRow {
    /// The batch and the quantity that was compared.
    pub stage: String,
    /// How many values the row compares.
    pub items: usize,
    /// The largest absolute deviation between the two executors.
    pub worst: f64,
    /// D §10.2's tolerance for that quantity.
    pub tolerance: f64,
    /// How many values differ at all (E7 §5.1's column that matters).
    pub differing: usize,
    /// `ok`, `FAIL`, `delegated` or `skipped`, with its reason.
    pub status: String,
}

impl CheckRow {
    /// Whether the row is a failure — a deviation over the tolerance.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.status.starts_with("FAIL")
    }
}

/// D §10.2's tolerances, for the rows the kernels fill in.
pub mod tolerance {
    /// `cs` per hypothesis: one probe point of sixty, plus rounding (D §10.2's `coarse` row).
    pub const COARSE: f64 = 1.0 / 60.0 + 1e-6;
    /// A rung's pose, in degrees (D §10.2's `stage 1` and `stage 2` rows).
    pub const POSE_DEG: f64 = 0.05;
    /// A rung's pose, in wall thicknesses.
    pub const POSE_T: f64 = 0.01;
    /// A rung's `fitness` and `inlier_rmse`.
    pub const ICP: f64 = 1e-4;
    /// `gap`, in `t`; the tightest distance tolerance of the verification rows.
    pub const DISTANCE: f64 = 0.002;
    /// `pen`, a fraction of the surface samples.
    pub const INSIDE: f64 = 0.0005;
    /// D §10.2's `chaotic` alarm for stage 1, **per pair**: the share of a pair's candidates whose
    /// ladder is not a function of its input, so that a pose row cannot be applied to them.
    ///
    /// A `gpu-check` set is four pairs and not a dump, so the per-pair bound is the one that fits;
    /// D §10.2's per-dump bounds (0.002 and 0.06) are stated over tens of thousands of candidates.
    /// Neither number is task W's: both are C2 §5's, unchanged.
    pub const CHAOTIC_S1: f64 = 0.06;
    /// D §10.2's `chaotic` alarm for stage 2, per pair. C2 §5 measured 3 of 10 on the worst pair
    /// of pot_B and set the bound at 0.4.
    pub const CHAOTIC_S2: f64 = 0.4;
}

/// A deviation column: the worst, the count that moved at all, and the percentiles beside it.
///
/// D §10.2 asks for both — a tolerance on the worst case and a distribution beside it ("stage 1,
/// stage 2 — distribution (a measurement beside the worst case)") — and it asks for a third thing
/// as well, in the `chaotic` row: *how many of these candidates' ladders are not a function of
/// their input at double precision at all*. Task C2 §5 measured that on the reference: nudging one
/// entry of `T0` by one ULP moves Open3D's own answer for three of `Pot_B_Piece_01__06`'s ten
/// stage-2 candidates by 24.8°, 65.5° and 101.7°. A worst-case row over such a candidate measures
/// the chaos and not the kernel, so this column keeps the two apart: [`Column::determined`] holds
/// the deviations of the candidates whose ladder *is* a function of its input, and it is those the
/// tolerance is applied to.
///
/// **Two things make a candidate chaotic here, and both are measured** (task W, V6-D2):
///
/// * **the control** — the same rungs, in `f64`, on the CPU, from the pose the device itself
///   starts from. When *that* answer is further from the CPU's own than the row allows, the ladder
///   has amplified an `f32` input all by itself and the kernel is only the messenger. This one
///   costs one extra ladder per stage, so it is always measured and always applied.
/// * **the twelve one-ULP neighbours** of the candidate's own initial pose, C2's probe, under
///   `--chaos`. Thirteen extra climbs, so it is opt-in; without the flag the exclusion set is
///   smaller and the criterion therefore stricter.
///
/// Excluded is not ignored: [`Column::chaotic`] is counted, printed in every row's tail with the
/// worst deviation over *all* candidates beside it, and gated in its own row.
#[derive(Debug, Default)]
pub struct Column {
    all: Vec<f64>,
    determined: Vec<f64>,
    differing: usize,
    chaotic: usize,
}

impl Column {
    /// One candidate's deviation: whether it moved at all, and whether its ladder is a function
    /// of its input (a chaotic one is counted and excluded from the tolerance).
    pub fn push(&mut self, deviation: f64, differs: bool, determined: bool) {
        self.all.push(deviation);
        if determined {
            self.determined.push(deviation);
        } else {
            self.chaotic += 1;
        }
        self.differing += usize::from(differs);
    }

    fn worst(values: &[f64]) -> f64 {
        values.iter().copied().fold(0.0_f64, f64::max)
    }

    /// `p50`, `p90`, `p99` by nearest rank over the sorted deviations.
    fn percentiles(values: &[f64]) -> [f64; 3] {
        if values.is_empty() {
            return [0.0; 3];
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |q: f64| {
            #[allow(clippy::cast_precision_loss, reason = "candidate counts are small")]
            let n = sorted.len() as f64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the index is clamped into the vector"
            )]
            let i = ((q * n).ceil() as usize).clamp(1, sorted.len()) - 1;
            sorted[i]
        };
        [at(0.5), at(0.9), at(0.99)]
    }

    /// The column as a row: the worst of the determined candidates against `tolerance`, with the
    /// distribution over *all* of them and the excused count beside it.
    #[must_use]
    pub fn row(&self, stage: &str, tolerance: f64) -> CheckRow {
        let worst = Self::worst(&self.determined);
        let [p50, p90, p99] = Self::percentiles(&self.all);
        let over_all = Self::worst(&self.all);
        let tail = if self.chaotic > 0 {
            format!(" ({} chaotic excluded, worst over all {over_all:.3e})", self.chaotic)
        } else {
            String::new()
        };
        let status = if self.all.is_empty() {
            "skipped — nothing to compare".to_owned()
        } else if worst <= tolerance {
            format!("ok — p50 {p50:.3e} p90 {p90:.3e} p99 {p99:.3e}{tail}")
        } else {
            format!("FAIL — p50 {p50:.3e} p90 {p90:.3e} p99 {p99:.3e}{tail}")
        };
        CheckRow {
            stage: stage.to_owned(),
            items: self.all.len(),
            worst,
            tolerance,
            differing: self.differing,
            status,
        }
    }
}

/// Which candidates' ladders are a function of their input at double precision — C2's probe, over
/// a whole batch at once.
///
/// The ladder is re-climbed from the twelve initial poses one ULP from each candidate's own (each
/// entry of the 3×4 block moved to its next representable neighbour, one at a time), on the CPU,
/// and a candidate is *determined* when every one of those twelve answers lands within the row's
/// own tolerance of its unperturbed one. Thirteen `climb_all` calls over the whole array rather
/// than thirteen per candidate, so the cost is thirteen rungs and not thirteen ladders each.
///
/// `sherd_parity::stages::determined` is the same probe for one candidate; it is not reused here
/// because it measures with the trace form of `pose_gap`, whose floor on these poses is 3.6e-2
/// degrees — most of the 0.05 the row allows.
pub fn determined_batch(
    rungs: &[sherd_core::matching::ladder::Rung<'_>],
    inits: &[sherd_core::matching::icp::Pose],
    scales: &sherd_core::matching::scales::Scales,
    rotation: f64,
    translation: f64,
) -> Vec<bool> {
    use sherd_core::executor::Engine;
    use sherd_core::matching::ladder::climb_all;

    let last = |climbed: Vec<Vec<sherd_core::matching::icp::Registration>>| -> Vec<_> {
        climbed
            .into_iter()
            .zip(inits)
            .map(|(out, init)| out.last().map_or(*init, |r| r.transform))
            .collect::<Vec<_>>()
    };
    let base = last(climb_all(Engine::REFERENCE, rungs, inits, scales));
    let mut ok = vec![true; inits.len()];
    for i in 0..3 {
        for j in 0..4 {
            let nudged: Vec<_> = inits
                .iter()
                .map(|init| {
                    let mut near = *init;
                    near[(i, j)] = near[(i, j)].next_up();
                    near
                })
                .collect();
            let other = last(climb_all(Engine::REFERENCE, rungs, &nudged, scales));
            for (k, (a, b)) in base.iter().zip(&other).enumerate() {
                let (deg, units) = pose_deviation(a, b, scales.t);
                if deg > rotation || units > translation {
                    ok[k] = false;
                }
            }
        }
    }
    ok
}

/// `sherd-refit-rs gpu-check`: feed identical batches to both executors and report the deviations.
///
/// The harness preprocesses the collection once and then walks R §4.1's pair order, taking the
/// first `pairs` matchable pairs and forming, from each, exactly the batches the pipeline forms:
/// R §5.2's coarse batch over every hypothesis, R §5.4's re-score over the kept list, the two
/// stage-1 rungs and the four stage-2 rungs. Both executors see the **same batch value**, so a
/// deviation is the kernel's and not the caller's.
#[allow(clippy::too_many_lines, reason = "one block per Executor method, each self-contained")]
#[allow(
    clippy::float_cmp,
    reason = "a cross-check asks whether two executors returned the same bits, not whether they \
              are close: the tolerance is the row's, and the `differ` column is exact equality"
)]
pub fn check(
    stage: &str,
    set: Option<&Path>,
    fixture: Option<&Path>,
    adapter: Option<&str>,
    pairs: usize,
    chaos: bool,
    force_device: bool,
) -> Result<Vec<CheckRow>, GpuError> {
    use std::sync::Arc;

    use crate::{AdapterChoice, Gpu, GpuExecutor, SelfTest};
    use sherd_core::executor::batch::{DistBatch, DistReduce, InsideBatch};
    use sherd_core::executor::{CPU, Executor};
    use sherd_core::matching::pair::Pair;
    use sherd_core::{Params, collection, pipeline};

    let choice = adapter.map_or(AdapterChoice::Default, AdapterChoice::parse);
    let gpu = Gpu::open(&choice)?;
    let selftest = SelfTest::run(&gpu)?;
    let mut rows: Vec<CheckRow> = selftest
        .checks
        .iter()
        .map(|c| CheckRow {
            stage: c.name.to_owned(),
            items: 0,
            worst: 0.0,
            tolerance: 0.0,
            differing: 0,
            status: format!("{} — {}", if c.passed { "ok" } else { "FAIL" }, c.detail),
        })
        .collect();
    let executor = GpuExecutor::new(Arc::new(gpu), selftest);
    // D §12's 2b criterion is stated at the **production policy** (audit §A.2.4): the executor's
    // own size thresholds decide what reaches the device, which is what a `--backend gpu` run of
    // this collection would do. `--force-device` is task W's measurement of the kernel instead —
    // every rung on the device, including the ones no run puts there — and on a small collection
    // it is also what keeps a row from comparing the CPU with itself.
    executor.force_device(force_device);

    let wanted = |name: &str| stage == "all" || stage == name;
    let delegated = |name: &str, tolerance: f64, items: usize| CheckRow {
        stage: name.to_owned(),
        items,
        worst: 0.0,
        tolerance,
        differing: 0,
        status: "delegated — phase 2c's kernels; this method is the CPU's".to_owned(),
    };

    // Without a collection there is nothing to form batches from, and the rows say so rather than
    // reporting a zero nobody measured.
    let Some(input) = set.or(fixture) else {
        for (name, tol) in [
            ("coarse", tolerance::COARSE),
            ("icp", tolerance::POSE_DEG),
            ("distance", tolerance::DISTANCE),
            ("inside", tolerance::INSIDE),
        ] {
            if wanted(name) {
                rows.push(CheckRow {
                    stage: name.to_owned(),
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
        return Err(GpuError::CrossCheck(
            "--fixture is not read by `gpu-check`: the batches it would form are the parity \
             harness's, and D §10.2's own rows already compare those against the reference. Use \
             `--set DIR` to cross-check the two executors on a real collection."
                .to_owned(),
        ));
    }

    let params = Params::default();
    let entries = collection::discover(input)?;
    if entries.len() < 2 {
        return Err(GpuError::CrossCheck(format!(
            "{}: need at least two mesh files",
            input.display()
        )));
    }
    let fragments: Vec<_> = pipeline::preprocess(
        &entries,
        200_000,
        None,
        sherd_core::memory::Budget::default_for_machine(),
    )
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut coarse = Column::default();
    let mut rescore = Column::default();
    let mut s1_rot = Column::default();
    let mut s1_disp = Column::default();
    let mut s1_fit = Column::default();
    let mut s1_rmse = Column::default();
    let mut s2_rot = Column::default();
    let mut s2_disp = Column::default();
    let mut s2_fit = Column::default();
    let mut s2_rmse = Column::default();
    let mut s1_ctrl = Column::default();
    let mut s2_ctrl = Column::default();
    // `(chaotic, candidates)` per pair, for D §10.2's own `chaotic` alarm row.
    let mut s1_chaos: Vec<(usize, usize)> = Vec::new();
    let mut s2_chaos: Vec<(usize, usize)> = Vec::new();
    let mut s1_iter = Column::default();
    let mut s2_iter = Column::default();
    let mut s1_cloud = Column::default();
    let mut s2_cloud = Column::default();
    let mut distance = Column::default();
    let mut inside_depth = Column::default();
    let mut inside_flag = 0_usize;
    let mut inside_items = 0_usize;
    let mut used = 0_usize;
    let mut over_one_probe = 0_usize;
    // `(on_device, calls)` of `icp_rung` per stage, so that the stage-1 and stage-2 rows can each
    // say what the *policy* did with them rather than sharing one method-wide counter.
    let mut s1_calls = (0_u64, 0_u64);
    let mut s2_calls = (0_u64, 0_u64);

    'outer: for i in 0..fragments.len() {
        for j in i + 1..fragments.len() {
            if used >= pairs {
                break 'outer;
            }
            let (a, b) = (&fragments[i].fragment, &fragments[j].fragment);
            if Pair::skipped(a, b, &params) {
                continue;
            }
            let pair = Pair::build(a, b, &params);
            if !pair.matchable() {
                continue;
            }
            used += 1;
            let t = pair.scales.t;
            let report = compare_pair(&pair, &params, &executor, chaos);
            s1_calls = (s1_calls.0 + report.stage1_calls.0, s1_calls.1 + report.stage1_calls.1);
            s2_calls = (s2_calls.0 + report.stage2_calls.0, s2_calls.1 + report.stage2_calls.1);

            if wanted("coarse") {
                for (c, g) in report.coarse_host.iter().zip(&report.coarse_device) {
                    coarse.push((c - g).abs(), c.to_bits() != g.to_bits(), true);
                    over_one_probe += usize::from((c - g).abs() > tolerance::COARSE);
                }
                for (c, g) in report.rescore_host.iter().zip(&report.rescore_device) {
                    rescore.push((c - g).abs(), c.to_bits() != g.to_bits(), true);
                }
            }
            if wanted("icp") {
                for (which, cpu, gpu, determined) in [
                    (1_u8, &report.stage1_host, &report.stage1_device, &report.stage1_ok),
                    (2, &report.stage2_host, &report.stage2_device, &report.stage2_ok),
                ] {
                    let (rot, disp, fit, rmse) = if which == 1 {
                        (&mut s1_rot, &mut s1_disp, &mut s1_fit, &mut s1_rmse)
                    } else {
                        (&mut s2_rot, &mut s2_disp, &mut s2_fit, &mut s2_rmse)
                    };
                    let (control, poses, iters, cloud, centre) = if which == 1 {
                        (
                            &mut s1_ctrl,
                            &report.stage1_control,
                            &mut s1_iter,
                            &mut s1_cloud,
                            report.stage1_centre,
                        )
                    } else {
                        (
                            &mut s2_ctrl,
                            &report.stage2_control,
                            &mut s2_iter,
                            &mut s2_cloud,
                            report.stage2_centre,
                        )
                    };
                    // Which candidates a pose row may be applied to at all (V6-D2). Two things
                    // disqualify one, and both are C2 §5's `chaotic` case: the twelve one-ULP
                    // neighbours of its own initial pose move its own CPU answer further than the
                    // row allows (`determined`, measured only under `--chaos`), or the **control**
                    // does — the same rungs, in `f64`, on the CPU, from the pose the device itself
                    // starts from. When the control alone is outside the row, the ladder has
                    // amplified an `f32` starting pose by itself and the kernel is the messenger.
                    let sound: Vec<bool> = cpu
                        .iter()
                        .zip(poses)
                        .enumerate()
                        .map(|(k, (c, ctrl))| {
                            let (deg, units) = pose_deviation(&c.transform, ctrl, t);
                            determined.get(k).copied().unwrap_or(true)
                                && deg <= tolerance::POSE_DEG
                                && units <= tolerance::POSE_T
                        })
                        .collect();
                    let excused = sound.iter().filter(|ok| !**ok).count();
                    if which == 1 {
                        s1_chaos.push((excused, sound.len()));
                    } else {
                        s2_chaos.push((excused, sound.len()));
                    }
                    let sound_at = |k: usize| sound.get(k).copied().unwrap_or(true);
                    for (k, (c, g)) in cpu.iter().zip(gpu).enumerate() {
                        let moved = cloud_deviation(&c.transform, &g.transform, &centre, t);
                        cloud.push(moved, moved > 0.0, sound_at(k));
                    }
                    for (k, (c, g)) in cpu.iter().zip(gpu).enumerate() {
                        #[allow(clippy::cast_precision_loss, reason = "iteration caps are small")]
                        let delta = (c.iterations as f64 - g.iterations as f64).abs();
                        iters.push(
                            delta,
                            c.iterations != g.iterations || c.converged != g.converged,
                            sound_at(k),
                        );
                    }
                    // The control is the row that *defines* the exclusion, so it is reported over
                    // every candidate and gates nothing: an alarm, in D §10.2's own sense.
                    for (c, ctrl) in cpu.iter().zip(poses) {
                        let (deg, _) = pose_deviation(&c.transform, ctrl, t);
                        control.push(deg, deg > 0.0, true);
                    }
                    for (k, (c, g)) in cpu.iter().zip(gpu).enumerate() {
                        let sound = sound_at(k);
                        let moved = c.transform != g.transform;
                        let (deg, units) = pose_deviation(&c.transform, &g.transform, t);
                        rot.push(deg, moved, sound);
                        disp.push(units, moved, sound);
                        let df = (c.fitness - g.fitness).abs();
                        let dr = (c.inlier_rmse - g.inlier_rmse).abs() / t;
                        fit.push(df, c.correspondences != g.correspondences, sound);
                        rmse.push(dr, dr > 0.0, sound);
                    }
                }
            }
            if let Some((sa, sb)) = pair.surfaces() {
                let pose = report.pose;
                if wanted("distance") {
                    let batch = DistBatch {
                        points: &sb.pf,
                        transform: &pose,
                        scene: sa.fracture,
                        max_dist: pair.scales.facing,
                        reduce: DistReduce::All,
                    };
                    let (cpu, gpu) =
                        (CPU.bounded_distance(&batch), executor.bounded_distance(&batch));
                    for (c, g) in cpu.iter().zip(&gpu) {
                        let delta = if c.is_infinite() && g.is_infinite() {
                            0.0
                        } else {
                            (c - g).abs() / t
                        };
                        distance.push(delta, c.is_finite() != g.is_finite(), true);
                    }
                }
                if wanted("inside")
                    && let Some(mesh) = sa.mesh
                {
                    {
                        let batch = InsideBatch { points: &sb.s, transform: &pose, scene: mesh };
                        let (cpu, gpu) = (CPU.inside(&batch), executor.inside(&batch));
                        inside_items += cpu.len();
                        for (c, g) in cpu.iter().zip(&gpu) {
                            inside_flag += usize::from(c.inside != g.inside);
                            inside_depth.push(
                                f64::from((c.depth - g.depth).abs()) / t,
                                c.depth != g.depth,
                                true,
                            );
                        }
                    }
                }
            }
        }
    }
    if used == 0 {
        return Err(GpuError::CrossCheck(format!(
            "{}: no matchable pair among {} fragments",
            input.display(),
            fragments.len()
        )));
    }

    let coarse_rows = rows.len();
    if wanted("coarse") {
        let mut row = coarse.row("coarse cs", tolerance::COARSE);
        row.status = format!("{} ({over_one_probe} over one probe point)", row.status);
        rows.push(row);
        rows.push(rescore.row("coarse s1", tolerance::COARSE));
    }
    let s1_rows = rows.len();
    if wanted("icp") {
        rows.push(s1_rot.row("icp s1 deg", tolerance::POSE_DEG));
        rows.push(control_row(&s1_ctrl, "icp s1 ctrl", &s1_chaos));
        rows.push(s1_iter.row("icp s1 iter", f64::INFINITY));
        rows.push(s1_cloud.row("icp s1 t@cloud", tolerance::POSE_T));
        rows.push(origin_row(&s1_disp, "icp s1 t"));
        rows.push(s1_fit.row("icp s1 fit", tolerance::ICP));
        rows.push(s1_rmse.row("icp s1 rmse", tolerance::ICP));
    }
    let s2_rows = rows.len();
    if wanted("icp") {
        rows.push(s2_rot.row("icp s2 deg", tolerance::POSE_DEG));
        rows.push(control_row(&s2_ctrl, "icp s2 ctrl", &s2_chaos));
        rows.push(s2_iter.row("icp s2 iter", f64::INFINITY));
        rows.push(s2_cloud.row("icp s2 t@cloud", tolerance::POSE_T));
        rows.push(origin_row(&s2_disp, "icp s2 t"));
        rows.push(s2_fit.row("icp s2 fit", tolerance::ICP));
        rows.push(s2_rmse.row("icp s2 rmse", tolerance::ICP));
    }
    let icp_end = rows.len();
    if wanted("icp") {
        // Not a kernel comparison, so it stays outside the span `annotate` labels `[device]`:
        // which candidates the pose rows had to excuse is decided entirely on the CPU.
        rows.push(chaotic_row("icp s1 chaotic", &s1_chaos, tolerance::CHAOTIC_S1));
        rows.push(chaotic_row("icp s2 chaotic", &s2_chaos, tolerance::CHAOTIC_S2));
    }
    if wanted("distance") {
        rows.push(CheckRow {
            items: distance.all.len(),
            ..delegated("distance", tolerance::DISTANCE, distance.all.len())
        });
    }
    if wanted("inside") {
        let mut row = delegated("inside", tolerance::INSIDE, inside_items);
        row.differing = inside_flag + inside_depth.differing;
        rows.push(row);
    }
    // Every row that compares a kernel says how many of its calls actually reached the device.
    let stats = executor.stats();
    let annotate = |rows: &mut Vec<CheckRow>,
                    span: std::ops::Range<usize>,
                    (on_device, calls): (u64, u64)| {
        for row in &mut rows[span] {
            if on_device == calls {
                row.status = format!("{} [device]", row.status);
            } else if on_device == 0 {
                "delegated — every call was the CPU's, so this row compares nothing"
                    .clone_into(&mut row.status);
            } else {
                row.status = format!("{} [{on_device} of {calls} calls on the device]", row.status);
            }
        }
    };
    let coarse_snapshot = stats.coarse.snapshot();
    annotate(&mut rows, coarse_rows..s1_rows, (coarse_snapshot.on_device, coarse_snapshot.calls));
    annotate(&mut rows, s1_rows..s2_rows, s1_calls);
    // The stage-2 rows are the ones the audit's §A.2.4 restates. When the policy is in force and
    // no stage-2 rung reached the device — which is every production run, because R §5.6 climbs
    // ten candidates and `icp::MIN_CANDIDATES` is sixteen — the rows say so by name instead of
    // reporting a deviation of zero that both executors computed on the CPU.
    if !force_device && s2_calls.0 == 0 && !crate::icp::STAGE2_ON_DEVICE {
        for row in &mut rows[s2_rows..icp_end] {
            row.status = format!(
                "cpu by policy — R §5.6 climbs {} candidates, under icp::MIN_CANDIDATES ({}), so \
                 no run sends stage 2 to the device (audit §A.2.4); `--force-device` measures the \
                 kernel on it",
                crate::icp::STAGE2_CANDIDATES,
                crate::icp::MIN_CANDIDATES,
            );
        }
    } else {
        annotate(&mut rows, s2_rows..icp_end, s2_calls);
    }
    for line in stats.lines() {
        rows.push(CheckRow {
            stage: "counters".to_owned(),
            items: 0,
            worst: 0.0,
            tolerance: 0.0,
            differing: 0,
            status: line,
        });
    }
    rows.push(CheckRow {
        stage: "pairs".to_owned(),
        items: used,
        worst: 0.0,
        tolerance: 0.0,
        differing: 0,
        status: format!(
            "{used} pair(s) of {}, gpu busy {:.1} ms",
            fragments.len(),
            stats.gpu_busy().as_secs_f64() * 1e3
        ),
    });
    Ok(rows)
}

/// The table as lines, header first — what `sherd-refit-rs gpu-check` prints.
///
/// The formatting lives beside the rows so that a build without this crate has no row type to
/// mirror: the CLI asks for lines and a failure count, and its CPU-only stub fails before either.
#[must_use]
pub fn table(rows: &[CheckRow]) -> Vec<String> {
    let mut out = vec![format!(
        "{:<12} {:>10} {:>12} {:>12} {:>10}  status",
        "stage", "items", "worst", "tol", "differ"
    )];
    out.extend(rows.iter().map(|row| {
        let (worst, tol) = if row.tolerance > 0.0 {
            (format!("{:.3e}", row.worst), format!("{:.3e}", row.tolerance))
        } else {
            ("-".to_owned(), "-".to_owned())
        };
        format!(
            "{:<12} {:>10} {:>12} {:>12} {:>10}  {}",
            row.stage, row.items, worst, tol, row.differing, row.status
        )
    }));
    out
}

/// D §10.2's `chaotic` alarm as a `gpu-check` row: how many candidates the pose rows had to excuse.
///
/// D §10.2 states the shape of this row as well as its bound — *"an alarm, not a parity
/// requirement"*, whose job (C2 §5) is "to fail if a future change makes the port chaotic where the
/// reference is not". The bound applied is D §10.2's own **per-pair** share, because a `gpu-check`
/// set is four pairs and its per-dump bound is stated over tens of thousands of candidates; the
/// set-wide share is printed beside it and gates nothing.
pub fn chaotic_row(stage: &str, per_pair: &[(usize, usize)], tolerance: f64) -> CheckRow {
    let chaotic: usize = per_pair.iter().map(|&(c, _)| c).sum();
    let total: usize = per_pair.iter().map(|&(_, n)| n).sum();
    #[allow(clippy::cast_precision_loss, reason = "candidate counts are far below 2^53")]
    let share = |c: usize, n: usize| if n == 0 { 0.0 } else { c as f64 / n as f64 };
    let worst = per_pair.iter().map(|&(c, n)| share(c, n)).fold(0.0_f64, f64::max);
    let detail = format!(
        "{chaotic} of {total} candidates excused over {} pair(s) ({:.3e} of the set), worst pair \
         {worst:.3e}",
        per_pair.len(),
        share(chaotic, total),
    );
    let status = if total == 0 {
        "skipped — nothing to compare".to_owned()
    } else if worst <= tolerance {
        format!("ok — {detail}")
    } else {
        format!("FAIL — {detail}")
    };
    CheckRow { stage: stage.to_owned(), items: total, worst, tolerance, differing: chaotic, status }
}

/// The origin-referenced translation row, as an **alarm** beside the row that gates.
///
/// D §10.2 writes the translation tolerance over the displacement of the *origin*, and its own
/// note says why: *"a pure rotation difference shows up in both rows — which is the conservative
/// way round"*. On these scans that conservatism is the whole number rather than a rounding: the
/// fragments sit 100–150 units from the origin, so the row reads a lever arm times the rotation
/// and reports 4.3e-2 t where the fragment itself moved 1e-4 t (task W, D §6.7's third bullet).
///
/// The audit's §A.2.4 therefore reads the translation **at the cloud** — the centroid of the
/// moving points, where a reader asking "did the fragment move" is looking — and keeps this one
/// beside it, printed and ungated. Both are measured over the same candidates; only which of them
/// decides the exit code changed.
pub fn origin_row(column: &Column, stage: &str) -> CheckRow {
    let mut row = column.row(stage, f64::INFINITY);
    if !column.all.is_empty() {
        row.status = format!(
            "{} (the displacement of the origin, D §10.2's own form: a lever arm of 100-150 units \
             times the rotation on these scans; the row the criterion is read at is `{stage}@cloud`)",
            row.status,
        );
    }
    row
}

/// The `ctrl` row: the same rungs, in `f64`, on the CPU, from the pose the device itself starts
/// from — an alarm over **every** candidate, with no tolerance of its own.
///
/// It is the row that decides which candidates the pose rows may be applied to (a control outside
/// D §10.2's pose tolerances is C2's chaotic case), so gating it as well would be gating the same
/// fact twice and would guarantee its own verdict. What it reports instead is the size of the
/// effect: the worst the ladder alone moved, and how many candidates it moved out of the rows.
pub fn control_row(column: &Column, stage: &str, per_pair: &[(usize, usize)]) -> CheckRow {
    let excused: usize = per_pair.iter().map(|&(c, _)| c).sum();
    let mut row = column.row(stage, f64::INFINITY);
    if !column.all.is_empty() {
        row.status = format!(
            "{} ({excused} candidate(s) outside D §10.2's pose rows on the control alone, and \
             excused there)",
            row.status,
        );
    }
    row
}

/// The rotation (degrees) and the displacement (wall thicknesses) between two poses, in the units
/// every pose row of D §10.2 is stated in — through the **Frobenius** form rather than the trace.
///
/// Both forms live in `sherd_core::matching::icp::pose_gap`, with the reason each exists; this
/// harness needs the one that is exactly zero for identical bits, because R §5.1's hypothesis
/// rotations are orthonormal to about 1e-7 and the trace form reports 3.6e-2 degrees for a pose
/// against itself — most of the 0.05 the row allows.
pub fn pose_deviation(
    a: &sherd_core::matching::icp::Pose,
    b: &sherd_core::matching::icp::Pose,
    t: f64,
) -> (f64, f64) {
    use sherd_core::matching::icp::pose_gap;
    (pose_gap::frobenius_deg(a, b), pose_gap::origin_t(a, b, t))
}

/// How far a point of the moving cloud travels between two poses, in wall thicknesses — the row
/// `gpu-check`'s translation criterion is stated over (audit §A.2.4).
///
/// D §10.2's own translation column is the displacement of the **origin**, and its note says why:
/// "a pure rotation difference shows up in both rows — which is the conservative way round". These
/// scans sit 100–150 units from the origin, so that conservatism is not a rounding — it is the
/// whole number. On pot_A's stage-1 rungs the worst rotation is 2.6e-2 degrees, inside the row's
/// 0.05, and the origin-referenced displacement of the same poses is 4.3e-2 t, four times outside
/// the row's 0.01 — the same disagreement, read at a point 130 units away from where the fragment
/// is. Both are reported; this one is gated.
pub fn cloud_deviation(
    a: &sherd_core::matching::icp::Pose,
    b: &sherd_core::matching::icp::Pose,
    centre: &[f64; 3],
    t: f64,
) -> f64 {
    sherd_core::matching::icp::pose_gap::cloud_t(a, b, centre, t)
}

/// Everything one pair contributes to the table: the same batches, answered twice.
struct PairReport {
    coarse_host: Vec<f64>,
    coarse_device: Vec<f64>,
    rescore_host: Vec<f64>,
    rescore_device: Vec<f64>,
    stage1_host: Vec<sherd_core::matching::icp::Registration>,
    stage1_device: Vec<sherd_core::matching::icp::Registration>,
    stage2_host: Vec<sherd_core::matching::icp::Registration>,
    stage2_device: Vec<sherd_core::matching::icp::Registration>,
    /// Per candidate, whether its own CPU ladder survives all twelve one-ULP perturbations.
    stage1_ok: Vec<bool>,
    stage2_ok: Vec<bool>,
    /// The **control**: the same rungs, on the CPU, from the poses the device actually starts
    /// from (`init` through the shifted `f32` state and back). What separates the kernel's `f32`
    /// arithmetic from the ladder's own amplification of an `f32` starting pose.
    stage1_control: Vec<sherd_core::matching::icp::Pose>,
    stage2_control: Vec<sherd_core::matching::icp::Pose>,
    /// The centroid of each stage's moving cloud, for the displacement read at the fragment.
    stage1_centre: [f64; 3],
    stage2_centre: [f64; 3],
    /// `(on_device, calls)` of `icp_rung` for this pair's stage-1 and stage-2 rungs, so that each
    /// stage's rows report what the size policy did with them.
    stage1_calls: (u64, u64),
    stage2_calls: (u64, u64),
    pose: sherd_core::matching::icp::Pose,
}

/// Forms the pipeline's own batches for one pair and answers each of them on both executors.
///
/// The CPU is always the one that decides what the *next* batch is — the coarse scores that feed
/// R §5.3's suppression, the stage-1 poses that feed R §5.5's — so that both executors are asked
/// the same question at every rung. A harness that let each side pick its own candidates would be
/// comparing two different ladders and calling the difference a kernel deviation.
#[allow(clippy::too_many_lines, reason = "R §5.2 to R §5.6 in order, each block a batch")]
fn compare_pair(
    pair: &sherd_core::matching::pair::Pair<'_>,
    params: &sherd_core::Params,
    executor: &crate::GpuExecutor,
    chaos: bool,
) -> PairReport {
    use sherd_core::executor::batch::{CoarseBatch, IcpBatch, Poses};
    use sherd_core::executor::{CPU, Engine, Executor};
    use sherd_core::matching::coarse::{NORMAL_AGREE, Target};
    use sherd_core::matching::icp::{IcpTarget, Options, centroid_of, cloud_points, homogeneous};
    use sherd_core::matching::ladder;
    use sherd_core::spatial::grid::NearMask;

    let hyp = pair.hypotheses(params);
    let probe = pair.probe(params);
    let tree = pair.a.kd_brk.as_ref().expect("a matchable pair has a breakline tree");
    let target = Target { points: &pair.frames_a.p, normals: &pair.frames_a.ns, tree };

    // R §5.2's batch, exactly as `coarse::scores` forms it — near mask and all.
    let mask = NearMask::of(target.points, pair.scales.coarse);
    let batch = CoarseBatch {
        target,
        mask: mask.as_ref(),
        points: &probe.q,
        normals: &probe.qn,
        poses: Poses::Split { r: &hyp.r, tau: &hyp.tau },
        radius: pair.scales.coarse,
        normal_agree: NORMAL_AGREE,
    };
    let coarse_host = CPU.coarse_scores(&batch);
    let coarse_device = executor.coarse_scores(&batch);

    // R §5.3's suppression, on the CPU's scores, so both sides climb the same ladder.
    let kept = pair.suppress(&hyp, &coarse_host, params);
    let icp_target = IcpTarget::from_cloud(&pair.a.pc_brk_full);
    let source = cloud_points(&pair.b.pc_brk);
    let inits: Vec<_> =
        kept.iter().map(|&h| homogeneous(&hyp.r[h as usize], &hyp.tau[h as usize])).collect();
    let rungs = ladder::stage1_rungs(&source, &icp_target);
    let stage1_ok = if chaos {
        determined_batch(&rungs, &inits, &pair.scales, tolerance::POSE_DEG, tolerance::POSE_T)
    } else {
        Vec::new()
    };
    let mut stage1_host = Vec::new();
    let mut stage1_device = Vec::new();
    let mut poses = inits.clone();
    let mut control = crate::icp::device_round_trip(&inits, &source, &icp_target);
    let icp_before = executor.stats().icp.snapshot();
    for rung in &rungs {
        let options = Options {
            estimation: rung.estimation,
            max_correspondence_distance: pair.scales.icp_dist(rung.k),
            max_iteration: rung.iterations,
            numerics: Engine::REFERENCE.numerics,
        };
        let batch = IcpBatch { source: rung.source, target: rung.target, inits: &poses, options };
        stage1_host = CPU.icp_rung(&batch);
        stage1_device = executor.icp_rung(&batch);
        poses = stage1_host.iter().map(|r| r.transform).collect();
        let batch = IcpBatch { source: rung.source, target: rung.target, inits: &control, options };
        control = CPU.icp_rung(&batch).iter().map(|r| r.transform).collect();
    }
    let stage1_control = control;
    let stage1_centre = centroid_of(&source);
    let icp_after_s1 = executor.stats().icp.snapshot();

    // R §5.4's re-score, over the poses stage 1 actually produced: `Pair::stage1`'s own batch,
    // over B's whole breakline subset at `sc.stage1` rather than sixty points at `sc.coarse`.
    let (points, normals) = pair.b_subset();
    let rescore_batch = CoarseBatch {
        target,
        mask: None,
        points: &points,
        normals: &normals,
        poses: Poses::Homogeneous(&poses),
        radius: pair.scales.stage1,
        normal_agree: NORMAL_AGREE,
    };
    let rescore_host = CPU.coarse_scores(&rescore_batch);
    let rescore_device = executor.coarse_scores(&rescore_batch);

    // R §5.6's ladder, from the best few stage-1 poses.
    let ladder2 = sherd_core::matching::pair::SurfaceLadder::of(pair);
    let rungs2 = ladder2.rungs();
    let mut order: Vec<usize> = (0..rescore_host.len()).collect();
    order.sort_by(|&x, &y| {
        rescore_host[y].partial_cmp(&rescore_host[x]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let take = order.len().min(usize::try_from(params.stage2).unwrap_or(16));
    let mut poses2: Vec<_> = order[..take].iter().map(|&k| poses[k]).collect();
    let stage2_ok = if chaos {
        determined_batch(&rungs2, &poses2, &pair.scales, tolerance::POSE_DEG, tolerance::POSE_T)
    } else {
        Vec::new()
    };
    let mut stage2_host = Vec::new();
    let mut stage2_device = Vec::new();
    let mut control2 = poses2.clone();
    for rung in &rungs2 {
        let options = Options {
            estimation: rung.estimation,
            max_correspondence_distance: pair.scales.icp_dist(rung.k),
            max_iteration: rung.iterations,
            numerics: Engine::REFERENCE.numerics,
        };
        let batch = IcpBatch { source: rung.source, target: rung.target, inits: &poses2, options };
        stage2_host = CPU.icp_rung(&batch);
        stage2_device = executor.icp_rung(&batch);
        poses2 = stage2_host.iter().map(|r| r.transform).collect();
        let round = crate::icp::device_round_trip(&control2, rung.source, rung.target);
        let batch = IcpBatch { source: rung.source, target: rung.target, inits: &round, options };
        control2 = CPU.icp_rung(&batch).iter().map(|r| r.transform).collect();
    }
    let stage2_control = control2;
    let stage2_centre = rungs2.last().map_or([0.0; 3], |rung| centroid_of(rung.source));
    let icp_after_s2 = executor.stats().icp.snapshot();
    let pose = poses2
        .first()
        .copied()
        .or_else(|| poses.first().copied())
        .unwrap_or_else(sherd_core::matching::icp::Pose::identity);
    PairReport {
        coarse_host,
        coarse_device,
        rescore_host,
        rescore_device,
        stage1_host,
        stage1_device,
        stage2_host,
        stage2_device,
        stage1_ok,
        stage2_ok,
        stage1_control,
        stage2_control,
        stage1_centre,
        stage2_centre,
        stage1_calls: (
            icp_after_s1.on_device - icp_before.on_device,
            icp_after_s1.calls - icp_before.calls,
        ),
        stage2_calls: (
            icp_after_s2.on_device - icp_after_s1.on_device,
            icp_after_s2.calls - icp_after_s1.calls,
        ),
        pose,
    }
}

#[cfg(test)]
mod tests {
    use super::{CheckRow, Column, chaotic_row, control_row, origin_row, pose_deviation, table};
    use sherd_core::matching::icp::{Pose, Rotation, Translation, homogeneous};

    fn column(deviations: &[(f64, bool)]) -> Column {
        let mut c = Column::default();
        for &(d, determined) in deviations {
            c.push(d, d > 0.0, determined);
        }
        c
    }

    fn turn(deg: f64) -> Pose {
        let (s, c) = deg.to_radians().sin_cos();
        homogeneous(
            &Rotation::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
            &Translation::new(0.0, 0.0, 0.0),
        )
    }

    /// The tolerance is applied to the **determined** candidates and the chaotic ones are counted
    /// and printed, never silently dropped (V6-D2).
    #[test]
    fn the_row_gates_the_determined_and_reports_the_rest() {
        let c = column(&[(1e-3, true), (2e-3, true), (5.0, false)]);
        let row = c.row("icp s1 deg", 0.05);
        assert!(!row.failed(), "{}", row.status);
        assert_eq!(row.items, 3, "every candidate is in the item count");
        assert!((row.worst - 2e-3).abs() < 1e-12, "the worst is the determined one: {}", row.worst);
        assert!(
            row.status.contains("1 chaotic excluded") && row.status.contains("worst over all"),
            "an excluded candidate is printed with the worst it would have made: {}",
            row.status
        );

        // And a determined candidate over the tolerance fails, which is the whole exit rule.
        let over = column(&[(1e-3, true), (0.06, true)]).row("icp s1 deg", 0.05);
        assert!(over.failed(), "{}", over.status);

        // Nothing to compare is not a pass and not a failure.
        let empty = Column::default().row("icp s2 deg", 0.05);
        assert!(!empty.failed() && empty.status.starts_with("skipped"), "{}", empty.status);
    }

    /// Three rows report and never gate: `ctrl` defines the exclusion, the origin translation is
    /// read at the cloud instead, and both say so in words.
    #[test]
    fn the_alarm_rows_never_fail_and_name_the_row_that_does() {
        let wild = column(&[(1e6, true)]);
        let ctrl = control_row(&wild, "icp s1 ctrl", &[(1, 10)]);
        assert!(!ctrl.failed() && ctrl.tolerance.is_infinite(), "{}", ctrl.status);
        assert!(ctrl.status.contains("1 candidate(s) outside"), "{}", ctrl.status);

        let origin = origin_row(&wild, "icp s1 t");
        assert!(!origin.failed() && origin.tolerance.is_infinite(), "{}", origin.status);
        assert!(
            origin.status.contains("icp s1 t@cloud"),
            "the origin row names the row the criterion is read at: {}",
            origin.status
        );
    }

    /// D §10.2's `chaotic` alarm is bounded **per pair**, not over the set: four pairs are not a
    /// dump, and the per-dump bounds are stated over tens of thousands of candidates.
    #[test]
    fn the_chaotic_row_is_bounded_by_its_worst_pair() {
        // 4 of 40 over the set is 0.1, inside 0.4; but one pair is 3 of 10, which is not.
        let row = chaotic_row("icp s2 chaotic", &[(3, 10), (1, 10), (0, 10), (0, 10)], 0.25);
        assert!(row.failed(), "the worst pair is 0.3 of 0.25: {}", row.status);
        assert!(row.status.contains("worst pair"), "{}", row.status);
        let ok = chaotic_row("icp s2 chaotic", &[(3, 10), (1, 10), (0, 10), (0, 10)], 0.4);
        assert!(!ok.failed(), "{}", ok.status);
    }

    /// The Frobenius reading `gpu-check` needs: exactly zero for the same bits, and D §10.2's own
    /// units otherwise.
    #[test]
    fn the_pose_deviation_is_exactly_zero_for_identical_poses() {
        let a = turn(3.0);
        assert_eq!(pose_deviation(&a, &a, 2.0), (0.0, 0.0));
        let (deg, units) = pose_deviation(&turn(0.0), &turn(7.5), 2.0);
        assert!((deg - 7.5).abs() < 1e-9, "{deg}");
        assert!(units.abs() < 1e-12, "a pure rotation about the origin moves no origin: {units}");
    }

    /// The table is a header and one line per row, and the failure count is the exit code.
    #[test]
    fn the_table_is_a_header_and_one_line_per_row() {
        let rows = vec![
            column(&[(0.0, true)]).row("coarse cs", 1.0 / 60.0 + 1e-6),
            column(&[(1.0, true)]).row("icp s1 deg", 0.05),
        ];
        let lines = table(&rows);
        assert_eq!(lines.len(), rows.len() + 1);
        assert!(lines[0].contains("stage") && lines[0].contains("status"));
        assert_eq!(rows.iter().filter(|r| r.failed()).count(), 1);
        assert!(lines[2].contains("FAIL"), "{}", lines[2]);
    }

    /// A row whose status the executor's own counters rewrote is still readable as a row.
    #[test]
    fn a_row_is_only_a_failure_when_its_status_says_so() {
        let mut row = CheckRow {
            stage: "icp s2 deg".to_owned(),
            items: 40,
            worst: 0.0,
            tolerance: 0.05,
            differing: 0,
            status: "cpu by policy — R §5.6 climbs 10 candidates".to_owned(),
        };
        assert!(!row.failed());
        row.status = "FAIL — over the row".to_owned();
        assert!(row.failed());
    }
}
