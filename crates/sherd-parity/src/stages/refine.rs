//! D §10.2's `refine` row: R §9's full-resolution ladder, from the assembly's own poses to the
//! refined ones.
//!
//! # Injected
//!
//! Everything R §9 reads is the reference's own: the starting poses are `assembly/poses.json`, the
//! groups `assembly/groups.json`, the joins and their order `assembly/used.json`, the pair scales
//! come from each fragment's `mesh.stats.json`, and — the one array R §9 has that nothing else
//! does — the fracture cloud is built from the reference's own vertex selection,
//! `refine/<name>.idx`. The full-resolution mesh itself is not in the dump (it is the input file,
//! hundreds of megabytes per collection), so the port reads it from `--input`; the `load` row is
//! what pins those vertices to the reference's, at one `f32` ULP.
//!
//! Six things are compared:
//!
//! * `idx` — the port's **own** selection against the reference's, entry for entry. R §9's
//!   predicate is `frac[j] ∧ d < max(0.15 t, 1.5 res)` over a KD-tree of the working mesh's face
//!   centroids, and under [`MAX_POINTS`] both sides produce
//!   `np.where(sel)[0]`, ascending. A fragment **over** the cap is a different matter: R §10 draws
//!   `rng(0).choice(idx, 150000, replace=False)` there and PMC-9 forbids reproducing it, so those
//!   fragments report the *overlap* of the two selections instead and the row that would be exact
//!   is skipped for them by name.
//! * `dist` — the two correspondence radii, exact against `refine/joins.json`.
//! * `rung 1` and `rung 2` — the pose after each rung, at the row's own 0.2° / 0.02 t.
//! * `fitness` and `rmse` — Open3D's own two numbers at the last rung.
//! * `pose` — the refined poses against `refine/poses_final.json`, measured the way every pose row
//!   of D §10.2 is measured (the largest displacement they give a fragment's own samples) and
//!   **relative to each group's first fragment**, which is what the row asks for: R §9 never moves
//!   `g[0]`, so a group's absolute poses and its relative ones differ only by a fixed left factor.
//!
//! # Native
//!
//! The same ladder from the same poses, on the port's **own** vertex selection — which is the
//! reference's on every fragment under the cap and PMC-9's own draw on the three above it
//! (terracotta's, at 150 000 of about 200 000 candidates each). That isolates exactly one thing:
//! whether R §9's answer depends on *which* 150 000 of the fracture's vertices it is given. The
//! row's tolerance is the same in both columns, which is D §10.2's own statement.

use std::collections::BTreeMap;
use std::path::Path;

use nalgebra::Matrix4;
use sherd_core::error::Result;
use sherd_core::executor::Engine;
use sherd_core::mesh::Mesh;
use sherd_core::mesh::geometry::face_geometry;
use sherd_core::refine::{
    FractureCloud, MAX_POINTS, RefinePiece, Refinement, cap_selection, cloud_from_indices,
    refine_joins, select_candidates, vertex_normals,
};
use sherd_core::types::FragId;

use super::assembly::{RefJoin, matrix, read_poses};
use super::{Collection, FragmentFixture};
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// The scope every row of this stage is filed under.
const SCOPE: &str = "(collection)";

/// D §10.2's `refine` row, in degrees.
pub const ROT_DEG: f64 = 0.2;
/// D §10.2's `refine` row, in wall thicknesses.
pub const TRANS_T: f64 = 0.02;
/// How far the two `fitness` values may differ when both sides registered the **same** cloud —
/// the same 1e-4 the two ICP rows of D §10.2 use.
pub const FITNESS_TOL: f64 = 1e-4;

/// How far they may differ when they did not (native mode on a fragment above R §9's cap).
///
/// `fitness` is `|correspondences| / n_source`, a Bernoulli fraction over
/// [`MAX_POINTS`] points, and PMC-9 gives the two implementations
/// different draws of those points. At the terracotta joins' own `p ≈ 0.19` and `n = 150 000` one
/// σ of the *difference* between two independent draws is `√2·√(p(1−p)/n)` = **1.4e-3**, so this
/// gate is about 5.6 σ of the sampling noise the row cannot avoid. Measured worst over the eight
/// dumps: **9.5e-4**, which is under one σ. The row that carries the parity claim is the pose,
/// not this one.
pub const FITNESS_SAMPLED: f64 = 0.008;

/// `refine/joins.json`, one entry per join the reference's walk took.
#[derive(Debug, serde::Deserialize)]
struct RefRefined {
    fixed: String,
    moving: String,
    dist: [f64; 2],
    #[serde(rename = "T_rung")]
    rungs: Vec<[[f64; 4]; 4]>,
    fitness: f64,
    rmse_t: f64,
}

/// Everything the dump carries about R §9, resolved against the collection.
struct RefFixture {
    /// The poses R §8 left behind, by [`FragId`].
    before: Vec<Matrix4<f64>>,
    /// The poses R §9 produced.
    after: Vec<Matrix4<f64>>,
    /// The groups, as [`FragId`] lists in R §8's final order.
    groups: Vec<Vec<FragId>>,
    /// The joins R §8 used, in the order it took them.
    used: Vec<(FragId, FragId)>,
    /// The joins R §9 walked, in the order it walked them.
    joins: Vec<RefRefined>,
    /// `(thick, res)` per fragment.
    scales: Vec<(f64, f64)>,
}

/// Runs R §9 on the dump's own selection (injected) or on the port's (native).
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("refine", mode);
    let Some(fixture) = read_fixture(collection, &mut report)? else { return Ok(report) };
    let Some(originals) = read_originals(collection, &fixture, &mut report)? else {
        return Ok(report);
    };
    let clouds = build_clouds(collection, &fixture, &originals, mode, &mut report)?;
    let pieces: Vec<RefinePiece<'_>> = fixture
        .scales
        .iter()
        .enumerate()
        .map(|(n, &(thick, res))| RefinePiece {
            thick,
            res,
            cloud: clouds[n].as_ref().map(CloudPair::cloud),
        })
        .collect();
    let out = refine_joins(
        &pieces,
        &fixture.before,
        &fixture.groups,
        &fixture.used,
        &collection.manifest.collection.params,
        Engine::REFERENCE,
    );
    compare(collection, &mut report, &fixture, &out, &clouds);
    Ok(report)
}

/// The dump's own R §8 and R §9 boundaries, or `None` with the skip recorded.
fn read_fixture(collection: &Collection, report: &mut StageReport) -> Result<Option<RefFixture>> {
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let refine = collection.dir.refine_dir();
    if !refine.join("poses_final.json").is_file() || !refine.join("joins.json").is_file() {
        report.skip(
            SCOPE,
            "no refine/ in the dump: R §9 had nothing to refine (no group of two or more) or the \
             run was made with --no-refine",
        );
        return Ok(None);
    }
    let assembly = collection.dir.assembly_dir();
    if !assembly.join("poses.json").is_file() {
        report.skip(SCOPE, "no assembly/poses.json in the dump (level min)");
        return Ok(None);
    }
    let before = read_poses(&assembly.join("poses.json"), &names)?;
    let after = read_poses(&refine.join("poses_final.json"), &names)?;

    let raw_groups: Vec<Vec<String>> = npy::read_json_as(assembly.join("groups.json"))?;
    let index = |name: &str| names.iter().position(|n| n == name);
    let mut groups = Vec::new();
    for group in &raw_groups {
        let mut members = Vec::new();
        for name in group {
            let Some(i) = index(name) else {
                report.skip(SCOPE, "a group of the dump names a fragment the manifest does not");
                return Ok(None);
            };
            members.push(u32::try_from(i).expect("fewer than 2^32 fragments"));
        }
        groups.push(members);
    }
    let raw_used: Vec<RefJoin> = npy::read_json_as(assembly.join("used.json"))?;
    let mut used = Vec::new();
    for join in &raw_used {
        let (Some(a), Some(b)) = (index(&join.a), index(&join.b)) else {
            report.skip(SCOPE, "a used join of the dump names a fragment the manifest does not");
            return Ok(None);
        };
        used.push((
            u32::try_from(a).expect("fewer than 2^32 fragments"),
            u32::try_from(b).expect("fewer than 2^32 fragments"),
        ));
    }
    let joins: Vec<RefRefined> = npy::read_json_as(refine.join("joins.json"))?;

    let mut scales = Vec::new();
    for fragment in &collection.fragments {
        let stats = fragment.file("mesh.stats.json");
        if !stats.is_file() {
            report.skip(SCOPE, "the dump has no mesh.stats.json for a fragment");
            return Ok(None);
        }
        let value = npy::read_json(&stats)?;
        scales.push((
            npy::field_f64(&value, "thick", &stats)?,
            npy::field_f64(&value, "res", &stats)?,
        ));
    }
    Ok(Some(RefFixture { before, after, groups, used, joins, scales }))
}

/// The original mesh of every fragment R §9 touches, read from `--input`.
///
/// R §9's cloud is built on the file's own vertices *before* the largest-component pass, so this
/// is `io::load_mesh` and nothing more. Only the fragments in a group of two or more are read: on
/// a collection of 164 the rest are hundreds of megabytes that nothing looks at.
fn read_originals(
    collection: &Collection,
    fixture: &RefFixture,
    report: &mut StageReport,
) -> Result<Option<Vec<Option<Mesh>>>> {
    let mut wanted = vec![false; collection.fragments.len()];
    for group in &fixture.groups {
        if group.len() > 1 {
            for &n in group {
                wanted[n as usize] = true;
            }
        }
    }
    let mut out = Vec::with_capacity(collection.fragments.len());
    for (n, fragment) in collection.fragments.iter().enumerate() {
        if !wanted[n] {
            out.push(None);
            continue;
        }
        let Some(source) = &fragment.source else {
            report.skip(SCOPE, "no source file for a refined fragment (pass --input DIR)");
            return Ok(None);
        };
        out.push(Some(sherd_core::io::load_mesh(source)?));
    }
    Ok(Some(out))
}

/// One fragment's cloud, plus the reference's own selection for the comparison.
struct CloudPair {
    cloud: FractureCloud,
    /// The reference's `refine/<name>.idx`, when the dump carries it.
    theirs: Option<Vec<u32>>,
    /// The port's own selection, always computed.
    ours: Vec<u32>,
    /// The port's own selection **before** the cap — R §9's predicate on its own.
    candidates: Vec<u32>,
    /// True when either side hit R §9's cap and PMC-9 decides the order.
    capped: bool,
}

impl CloudPair {
    /// The cloud R §9 was run on.
    fn cloud(&self) -> &FractureCloud {
        &self.cloud
    }
}

/// The fracture cloud of every refined fragment, injected or native.
fn build_clouds(
    collection: &Collection,
    fixture: &RefFixture,
    originals: &[Option<Mesh>],
    mode: Mode,
    report: &mut StageReport,
) -> Result<Vec<Option<CloudPair>>> {
    let mut out = Vec::with_capacity(originals.len());
    for (n, original) in originals.iter().enumerate() {
        let Some(original) = original else {
            out.push(None);
            continue;
        };
        let fragment = &collection.fragments[n];
        let Some(mesh) = fragment.working()? else {
            report.skip(SCOPE, "the dump has no working mesh for a refined fragment");
            out.push(None);
            continue;
        };
        if !fragment.has("seg.frac_final.npy") {
            report.skip(SCOPE, "the dump has no seg.frac_final for a refined fragment");
            out.push(None);
            continue;
        }
        let fracture = npy::read_bool(fragment.file("seg.frac_final.npy"))?;
        let centroids = face_geometry(&mesh.v, &mesh.f).centroids;
        let (thick, res) = fixture.scales[n];
        let candidates = select_candidates(&original.v, &centroids, &fracture, thick, res);
        let ours = cap_selection(
            candidates.clone(),
            MAX_POINTS,
            collection.manifest.collection.params.seed,
        );
        let theirs = reference_idx(fragment)?;
        let capped =
            ours.len() >= MAX_POINTS || theirs.as_ref().is_some_and(|t| t.len() >= MAX_POINTS);
        let normals = vertex_normals(original);
        let chosen = match (mode, theirs.as_ref()) {
            (Mode::Injected, Some(theirs)) => theirs.clone(),
            _ => ours.clone(),
        };
        out.push(Some(CloudPair {
            cloud: cloud_from_indices(&original.v, &normals, &chosen),
            theirs,
            ours,
            candidates,
            capped,
        }));
    }
    Ok(out)
}

/// `refine/<name>.idx.npy`, when the dump carries it.
fn reference_idx(fragment: &FragmentFixture) -> Result<Option<Vec<u32>>> {
    let path = fragment
        .dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("refine").join(format!("{}.idx.npy", fragment.name)));
    match path {
        Some(p) if p.is_file() => Ok(Some(npy::read_indices(p)?)),
        _ => Ok(None),
    }
}

/// Every row of the stage.
#[allow(clippy::too_many_lines, reason = "the stage's rows are a list, and this is the list")]
fn compare(
    collection: &Collection,
    report: &mut StageReport,
    fixture: &RefFixture,
    out: &Refinement,
    clouds: &[Option<CloudPair>],
) {
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();

    // --- the selection ---------------------------------------------------------------------
    let mut differing = 0_usize;
    let mut total = 0_usize;
    let mut outside = 0_usize;
    let mut capped_total = 0_usize;
    let mut worst_overlap = 0.0_f64;
    for pair in clouds.iter().flatten() {
        let Some(theirs) = &pair.theirs else { continue };
        if pair.capped {
            // PMC-9: above the cap the two implementations draw different subsets, so what is
            // left to compare is the *candidate set* they drew from — every index the reference
            // kept has to be one R §9's predicate accepted here too — and the size of the overlap,
            // which for two independent draws of `max_points` from `k` candidates is
            // `max_points / k` and nothing else.
            let mine: std::collections::BTreeSet<u32> = pair.candidates.iter().copied().collect();
            let drawn: std::collections::BTreeSet<u32> = pair.ours.iter().copied().collect();
            outside += theirs.iter().filter(|i| !mine.contains(i)).count();
            capped_total += theirs.len();
            #[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
            let (shared, expected) = (
                theirs.iter().filter(|i| drawn.contains(i)).count() as f64,
                MAX_POINTS as f64 / pair.candidates.len().max(1) as f64,
            );
            #[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
            let overlap = if theirs.is_empty() { expected } else { shared / theirs.len() as f64 };
            worst_overlap = worst_overlap.max((overlap - expected).abs());
            continue;
        }
        total += theirs.len().max(pair.ours.len());
        differing += (0..theirs.len().max(pair.ours.len()))
            .filter(|&i| theirs.get(i) != pair.ours.get(i))
            .count();
    }
    if total > 0 {
        report.push(Check::entries(SCOPE, "idx", differing, total));
    }
    if capped_total > 0 {
        report.push(Check::entries(SCOPE, "idx candidates (capped)", outside, capped_total));
        report.push(Check::absolute(
            SCOPE,
            "idx overlap (capped)",
            worst_overlap,
            0.0,
            CAPPED_OVERLAP,
        ));
    }

    // --- the joins -------------------------------------------------------------------------
    let ours: Vec<String> = out
        .joins
        .iter()
        .map(|j| format!("{}->{}", names[j.moving as usize], names[j.fixed as usize]))
        .collect();
    let theirs: Vec<String> =
        fixture.joins.iter().map(|j| format!("{}->{}", j.moving, j.fixed)).collect();
    let mismatched =
        (0..ours.len().max(theirs.len())).filter(|&i| ours.get(i) != theirs.get(i)).count();
    if mismatched > 0 {
        for i in 0..ours.len().max(theirs.len()).min(5) {
            tracing::warn!(
                index = i,
                ours = ours.get(i).map_or("(none)", String::as_str),
                theirs = theirs.get(i).map_or("(none)", String::as_str),
                "refine walked a different join"
            );
        }
    }
    report.push(Check::entries(SCOPE, "walk", mismatched, ours.len().max(theirs.len())));

    let mut worst_dist = 0.0_f64;
    let mut worst_rung = [(0.0_f64, 0.0_f64); 2];
    let mut worst_fitness = 0.0_f64;
    let mut worst_rmse = 0.0_f64;
    // Whether R §9 ran on the reference's own vertex selection. Native mode above the cap draws
    // its own 150 000 (PMC-9), so `fitness` — a fraction *of the cloud* — is then a statistic on a
    // different sample and its row has to say so.
    let redrawn = clouds.iter().flatten().any(|pair| {
        pair.capped
            && pair.theirs.is_some()
            && pair.cloud.idx != *pair.theirs.as_ref().expect("checked")
    });
    for (mine, theirs) in out.joins.iter().zip(&fixture.joins) {
        if names[mine.moving as usize] != theirs.moving {
            break; // the walk row above has already failed; the rest would compare nothing.
        }
        let t = fixture.scales[mine.fixed as usize].0.min(fixture.scales[mine.moving as usize].0);
        for (k, worst) in worst_rung.iter_mut().enumerate() {
            worst_dist = worst_dist.max((mine.dist[k] - theirs.dist[k]).abs());
            if let Some(rung) = theirs.rungs.get(k) {
                let (rot, trans) = super::pose_gap(&mine.rungs[k], &matrix(rung), t);
                worst.0 = worst.0.max(rot);
                worst.1 = worst.1.max(trans);
            }
        }
        worst_fitness = worst_fitness.max((mine.fitness - theirs.fitness).abs());
        worst_rmse = worst_rmse.max((mine.rmse_t - theirs.rmse_t).abs());
    }
    if !out.joins.is_empty() {
        report.push(Check::absolute(SCOPE, "dist", worst_dist, 0.0, 0.0));
        for (k, row) in
            [("rung 1 rot", "rung 1 trans"), ("rung 2 rot", "rung 2 trans")].into_iter().enumerate()
        {
            report.push(Check::absolute(SCOPE, row.0, worst_rung[k].0, 0.0, ROT_DEG));
            report.push(Check::absolute(SCOPE, row.1, worst_rung[k].1, 0.0, TRANS_T));
        }
        let (row, tolerance) =
            if redrawn { ("fitness (redrawn)", FITNESS_SAMPLED) } else { ("fitness", FITNESS_TOL) };
        report.push(Check::absolute(SCOPE, row, worst_fitness, 0.0, tolerance));
        report.push(Check::absolute(SCOPE, "rmse", worst_rmse, 0.0, TRANS_T));
    }

    // --- the poses -------------------------------------------------------------------------
    //
    // D §10.2 asks for the *relative* poses within a group, which is `inv(P[g0]) · P[n]` on both
    // sides. R §9 never moves `g[0]`, so this is the absolute comparison with the group's own
    // frame divided out — and it is the quantity a reassembly is judged by.
    let mut worst = (0.0_f64, 0.0_f64);
    for group in &fixture.groups {
        if group.len() < 2 {
            continue;
        }
        let anchor = group[0] as usize;
        let (Some(mine0), Some(theirs0)) =
            (invert(&out.poses[anchor]), invert(&fixture.after[anchor]))
        else {
            continue;
        };
        for &n in &group[1..] {
            let t = fixture.scales[n as usize].0;
            let mine = mine0 * out.poses[n as usize];
            let theirs = theirs0 * fixture.after[n as usize];
            let (rot, trans) = super::pose_gap(&mine, &theirs, t);
            worst = (worst.0.max(rot), worst.1.max(trans));
        }
    }
    report.push(Check::absolute(SCOPE, "pose rot", worst.0, 0.0, ROT_DEG));
    report.push(Check::absolute(SCOPE, "pose trans", worst.1, 0.0, TRANS_T));
}

/// How far a capped selection's overlap with the reference's may sit from the ratio two
/// independent draws share in expectation (PMC-9).
///
/// Not a parity requirement — the two draws come from different generators — but a check that both
/// drew from the *same* candidate set of the *same* size: `|A ∩ B| / |B|` for two independent
/// draws of `max_points` from `k` candidates is `max_points / k`, and a predicate that accepted a
/// different set of vertices would move it. The exact half of that statement is the row above,
/// which requires every index the reference kept to be one the port's predicate accepted.
/// Measured on terracotta's three capped fragments: see `notes/2026-09-07-d2-refine-outputs.md`.
pub const CAPPED_OVERLAP: f64 = 0.05;

/// `inv(T)` for a rigid 4×4, or `None` when it is singular.
fn invert(t: &Matrix4<f64>) -> Option<Matrix4<f64>> {
    t.try_inverse()
}

/// The joins of `refine/joins.json`, keyed the way a caller reading them by name wants.
///
/// Exposed so that a diagnostic can read the same file the row does.
pub fn reference_joins(path: impl AsRef<Path>) -> Result<BTreeMap<String, [f64; 2]>> {
    let joins: Vec<RefRefined> = npy::read_json_as(path)?;
    Ok(joins.into_iter().map(|j| (format!("{}->{}", j.moving, j.fixed), j.dist)).collect())
}
