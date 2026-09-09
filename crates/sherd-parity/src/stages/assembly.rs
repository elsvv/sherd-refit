//! D §10.2's `assembly` row: R §8's greedy growth over the accepted joins — the groups, the joins
//! used and the accepted joins rejected, with the reason each was rejected.
//!
//! # Injected
//!
//! Everything R §8 reads is the reference's own: the candidate list is every pair directory's
//! `result.candidates.json`, the surface samples the penetration test casts are the dump's
//! `assembly/md/<name>/md.S` — the reference's rebuild at the collection median `t_med` with
//! `surface_points = 15000` (PMC-8) — and the two meshes behind every signed distance are
//! `mesh.V`/`mesh.F`. What the port supplies is the loop: which candidate represents a pair, the
//! order the joins are walked in, the penetration and consistency tests, the seeding of a new
//! group and the ordering of the groups at the end.
//!
//! Four things are compared, and the row calls for all four to be **identical**: the groups (as
//! lists of names, in order), the used joins (as pairs, in the order they were taken), the
//! rejections (as pairs *and* the reference's own English reason, digit for digit) and the poses.
//! Poses are the one row with a tolerance, `1e-9 t`, because a pose here is a product of up to a
//! dozen 4×4s and one `np.linalg.inv` per step, and numpy's `@` is BLAS rather than the triple
//! loop; it is measured as the largest displacement the two poses give the fragment's own surface
//! samples, before and after R §8.2's recentring.
//!
//! R §8.2 gets a check of its own on top of that, and it is end to end: `outputs/transforms.json`
//! is the reference's recentring of `refine/poses_final.json` — of the assembly's own poses on a
//! set where R §9 had nothing to refine — so the port recentres the reference's own input and
//! compares against the reference's own output.
//!
//! # Native
//!
//! Two runs, because there are two things to isolate.
//!
//! * **PMC-8** — the reference rebuilds every fragment's match arrays at `t_med` with 15 000
//!   surface points purely for this stage, and PMC-8 lets the port use the fragment's own cached
//!   20 000-point sample at its own `t` instead. That is one field of
//!   [`Piece`](sherd_core::assembly::Piece) and nothing else, so it can be measured on its own:
//!   the *reference's* candidates assembled with the *port's* samples, against the reference's own
//!   answer. PMC-8's re-verify column asks exactly this ("assembly `pen` decisions on all
//!   benchmarks") and the rows are gated identical.
//! * **The port's own candidates** — R §3 through R §6 run by the port, then R §8 on top. Nothing
//!   here is a parity claim, for the reason step C3 measured on the row above this one: PMC-9 gives
//!   the two implementations different samples and PMC-6 lets R §5.3's suppression keep a different
//!   set, so they *score different poses* and R §6.5 is a threshold on the output of that search.
//!   The accepted sets differ by construction, so the joins the assembly is built from differ, and
//!   these rows are regression alarms on how far apart the two assemblies end up.

use std::collections::BTreeMap;
use std::path::Path;

use nalgebra::Matrix4;
use sherd_core::assembly::{Assembly, Piece, assemble, recenter};
use sherd_core::error::{Error, Result};
use sherd_core::executor::CPU;
use sherd_core::fragment::Fragment;
use sherd_core::fragment::samples::MatchData;
use sherd_core::matching::pair::{self, Candidate};
use sherd_core::types::{FragId, apply_transform};

use super::Collection;
use super::DUMP_SEED;
use super::pairs::RefGeometry;
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// The scope every row of this stage is filed under: R §8 is about a collection, not about a pair.
const SCOPE: &str = "(collection)";

/// D §10.2's tolerance on an assembled pose, in wall thicknesses.
///
/// The row's three neighbours are exact; this one is not, because a pose here is a chain of 4×4
/// products and `np.linalg.inv`s — numpy's `@` is BLAS on the reference's side and `nalgebra`'s
/// triple loop on the port's — and neither `poses.json` nor `transforms.json` is reachable bit for
/// bit through a different matrix kernel. Measured worst over the eight dumps: see
/// `notes/2026-09-07-d1-assembly.md`.
pub const POSE_T: f64 = 1e-9;

/// One fragment as R §8 reads it, out of the dump: the two numbers R §1.2 resolves `pen` from,
/// R §3.3.2's verdict, the working mesh and the assembly-stage surface samples.
#[derive(Debug)]
pub struct RefPiece {
    /// R §3.2's wall thickness.
    pub thick: f64,
    /// R §3.3's `res`.
    pub res: f64,
    /// R §3.3.2's verdict.
    pub watertight: bool,
    /// The working mesh and its scenes, when the dump carries them.
    pub geometry: Option<RefGeometry>,
    /// `md.S` at `t_med` — PMC-8's own sample set.
    pub s_pen: Vec<[f64; 3]>,
}

/// `mesh.stats.json`: the fragment-level numbers R §8 needs and the sample arrays do not carry.
#[derive(serde::Deserialize)]
struct RefStats {
    thick: f64,
    res: f64,
    watertight: bool,
}

/// `assembly/md_t_median.json`: the `(t, surface_points)` the reference rebuilt every fragment at.
#[derive(Debug, serde::Deserialize)]
pub struct MdTMedian {
    /// The collection's median wall thickness.
    pub t: f64,
    /// How many surface points R §8's own `MatchData` was drawn with.
    pub surface_points: u64,
}

/// One entry of `assembly/used.json` or `assembly/rejected.json`.
#[derive(Debug, serde::Deserialize)]
pub struct RefJoin {
    /// The fragment the pose maps into.
    pub a: String,
    /// The fragment the pose moves.
    pub b: String,
    /// R §8's sentence, on a rejection.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Runs R §8 on the dump's own candidates (injected) or on the port's (native).
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("assembly", mode);
    match mode {
        Mode::Injected => injected(collection, &mut report)?,
        Mode::Native => native(collection, &mut report)?,
    }
    Ok(report)
}

/// R §8 over the reference's own candidates, samples and meshes.
fn injected(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let Some(pieces) = reference_pieces(collection, report)? else { return Ok(()) };
    let Some(candidates) = reference_candidates(collection, report)? else { return Ok(()) };
    let t = md_t_median(collection)?.map_or(f64::NAN, |md| md.t);
    let views = piece_views(&pieces);
    let out = assemble(&CPU, &views, &candidates, &collection.manifest.collection.params);
    compare(collection, report, &views, &candidates, &out, t, PLAIN)?;
    recentre_row(collection, report, &views, t)?;
    Ok(())
}

/// The port's own preprocessing and matching, with PMC-8's samples, twice over.
fn native(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let params = collection.manifest.collection.params;
    let Some(md) = md_t_median(collection)? else {
        report.skip(SCOPE, "no assembly/md_t_median.json in the dump (level min)");
        return Ok(());
    };
    let Some(fragments) = native_fragments(collection, report)? else { return Ok(()) };
    let samples: Vec<Vec<[f64; 3]>> =
        fragments.iter().map(|f| MatchData::own(f).samples.surface_f64()).collect();
    let pieces: Vec<Piece<'_>> = fragments
        .iter()
        .zip(&samples)
        .map(|(f, s)| Piece {
            thick: f.thick,
            res: f.res(),
            watertight: f.watertight,
            mesh: f.surface_scene(),
            s_pen: s,
        })
        .collect();
    let t = md.t;

    // PMC-8 alone: the reference's own candidates, the port's own samples.
    if let Some(candidates) = reference_candidates(collection, report)? {
        let out = assemble(&CPU, &pieces, &candidates, &params);
        compare(collection, report, &pieces, &candidates, &out, t, PMC8)?;
    }

    // The port's own candidates on top of that.
    let mut candidates: Vec<Candidate> = Vec::new();
    for names in &collection.manifest.pairs.pairs {
        let (Some(a), Some(b)) = (index_of(collection, &names[0]), index_of(collection, &names[1]))
        else {
            continue;
        };
        let started = std::time::Instant::now();
        let found =
            pair::match_pair(&fragments[a], &fragments[b], &params, super::candidates::KEEP);
        tracing::info!(
            pair = %format!("{}__{}", names[0], names[1]),
            seconds = started.elapsed().as_secs_f64(),
            accepted = found.iter().filter(|c| c.accepted).count(),
            "match_pair"
        );
        candidates.extend(found);
    }
    let out = assemble(&CPU, &pieces, &candidates, &params);
    alarms(collection, report, &candidates, &out)
}

/// What five rows of one comparison are called.
///
/// Native mode runs [`compare`] for its PMC-8 pass and injected mode for R §8 itself, so the same
/// five comparisons appear twice in one table under different names; a report row's quantity is a
/// `&'static str`, so the names travel as data.
#[derive(Clone, Copy)]
struct Rows {
    groups: &'static str,
    used: &'static str,
    rejected: &'static str,
    pose: &'static str,
    pose_centred: &'static str,
    /// The row that reports how far R §6.4's fraction moved, when there is one.
    pen: Option<&'static str>,
    /// Whether a rejection is compared by its whole sentence or only by its kind and subject.
    ///
    /// The sentence carries a *measured* number — R §6.4's fraction, printed to three decimals —
    /// and PMC-8 changes the sample it is measured on, so the PMC-8 pass compares the decision and
    /// reports the movement of the number as a row of its own.
    full_reason: bool,
}

/// The row names of a run that speaks for R §8 itself.
const PLAIN: Rows = Rows {
    groups: "groups",
    used: "used",
    rejected: "rejected",
    pose: "pose move",
    pose_centred: "pose move centred",
    pen: None,
    full_reason: true,
};

/// The row names of native mode's PMC-8 pass.
const PMC8: Rows = Rows {
    groups: "pmc8 groups",
    used: "pmc8 used",
    rejected: "pmc8 rejected",
    pose: "pmc8 pose move",
    pose_centred: "pmc8 pose move centred",
    pen: Some("pmc8 pen"),
    full_reason: false,
};

/// The rows that must be identical: the groups, the used joins, the rejections and the poses.
///
/// `views` are the pieces R §8 was run on — the reference's samples in injected mode, the port's
/// in native mode — and the poses are compared through them, since a pose is only as interesting
/// as where it puts the fragment's own points.
fn compare(
    collection: &Collection,
    report: &mut StageReport,
    views: &[Piece<'_>],
    candidates: &[Candidate],
    out: &Assembly,
    t: f64,
    rows: Rows,
) -> Result<()> {
    let dir = collection.dir.assembly_dir();
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();

    let theirs: Vec<Vec<String>> = npy::read_json_as(dir.join("groups.json"))?;
    let ours: Vec<Vec<String>> =
        out.groups.iter().map(|g| g.iter().map(|&n| names[n as usize].clone()).collect()).collect();
    push_sequence(report, rows.groups, &render(&ours), &render(&theirs));

    let used: Vec<RefJoin> = npy::read_json_as(dir.join("used.json"))?;
    let ours_used: Vec<String> =
        out.used.iter().map(|&i| join_name(&names, &candidates[i], None)).collect();
    let theirs_used: Vec<String> = used.iter().map(ref_join_name).collect();
    push_sequence(report, rows.used, &ours_used, &theirs_used);

    let rejected: Vec<RefJoin> = npy::read_json_as(dir.join("rejected.json"))?;
    let reason = |r: &sherd_core::assembly::Rejected| {
        if rows.full_reason { r.reason.message(&names) } else { r.reason.kind(&names) }
    };
    let ours_rejected: Vec<String> = out
        .rejected
        .iter()
        .map(|r| join_name(&names, &candidates[r.candidate], Some(&reason(r))))
        .collect();
    let theirs_rejected: Vec<String> = rejected
        .iter()
        .map(|r| if rows.full_reason { ref_join_name(r) } else { ref_join_kind(r) })
        .collect();
    push_sequence(report, rows.rejected, &ours_rejected, &theirs_rejected);
    if let Some(row) = rows.pen {
        // PMC-8's own re-verify column: how far the fraction R §6.4 counts moves when the samples
        // change, on the rejections both sides reach the same way. The decision is the row above;
        // this is the distance from `max_pen` that decision had to spare. The two lists are walked
        // in step, which means something only while that row passes — and when it does not, it is
        // the row that fails, not this one.
        let mut worst = 0.0_f64;
        for (mine, theirs) in out.rejected.iter().zip(&rejected) {
            let (Some(ours), Some(theirs)) = (mine.reason.penetration(), parse_pen(theirs)) else {
                continue;
            };
            worst = worst.max((ours - theirs).abs());
        }
        report.push(Check::absolute(SCOPE, row, worst, 0.0, PMC8_PEN));
    }

    let theirs_poses = read_poses(&dir.join("poses.json"), &names)?;
    report.push(Check::absolute(
        SCOPE,
        rows.pose,
        worst_move(&out.poses, &theirs_poses, views, t),
        0.0,
        POSE_T,
    ));
    // The same poses after R §8.2, which is what `transforms.json` carries and what a caller sees.
    // Both sides are recentred with the port's own groups, because the groups row above has just
    // required them to be the reference's.
    let ours_centred = recenter(&out.poses, views, &out.groups);
    let theirs_centred = recenter(&theirs_poses, views, &out.groups);
    report.push(Check::absolute(
        SCOPE,
        rows.pose_centred,
        worst_move(&ours_centred, &theirs_centred, views, t),
        0.0,
        POSE_T,
    ));
    Ok(())
}

/// R §8.2 on its own: the reference's own final poses, recentred by the port, against the
/// reference's own `transforms.json`.
///
/// The input is `refine/poses_final.json` where R §9 ran and `assembly/poses.json` where it had
/// nothing to refine (a collection of singletons), because the pipeline recentres **after**
/// refinement. The grouping is the dump's own, for the same reason.
fn recentre_row(
    collection: &Collection,
    report: &mut StageReport,
    views: &[Piece<'_>],
    t: f64,
) -> Result<()> {
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let outputs = collection.dir.outputs_dir().join("transforms.json");
    if !outputs.is_file() {
        report.skip(SCOPE, "no outputs/transforms.json in the dump");
        return Ok(());
    }
    let refined = collection.dir.refine_dir().join("poses_final.json");
    let before = if refined.is_file() {
        read_poses(&refined, &names)?
    } else {
        read_poses(&collection.dir.assembly_dir().join("poses.json"), &names)?
    };
    let groups: Vec<Vec<String>> =
        npy::read_json_as(collection.dir.assembly_dir().join("groups.json"))?;
    let mut ids: Vec<Vec<FragId>> = Vec::new();
    for group in &groups {
        let mut members = Vec::new();
        for name in group {
            let Some(i) = names.iter().position(|n| n == name) else {
                report.skip(SCOPE, "a group of the dump names a fragment the manifest does not");
                return Ok(());
            };
            members.push(u32::try_from(i).expect("fewer than 2^32 fragments"));
        }
        ids.push(members);
    }
    let ours = recenter(&before, views, &ids);
    let theirs = read_transforms(&outputs, &names)?;
    report.push(Check::absolute(
        SCOPE,
        "recentre",
        worst_move(&ours, &theirs, views, t),
        0.0,
        POSE_T,
    ));
    Ok(())
}

/// The native rows: how far the port's own assembly is from the reference's.
///
/// A regression alarm and not a parity claim (see the module documentation). The shares are of the
/// *union* of the two used-join sets, so a run that finds nothing and a run that finds twice as
/// much are both visible.
fn alarms(
    collection: &Collection,
    report: &mut StageReport,
    candidates: &[Candidate],
    out: &Assembly,
) -> Result<()> {
    let names: Vec<String> = collection.fragments.iter().map(|f| f.name.clone()).collect();
    let dir = collection.dir.assembly_dir();
    let used: Vec<RefJoin> = npy::read_json_as(dir.join("used.json"))?;
    let theirs: std::collections::BTreeSet<String> = used.iter().map(ref_join_name).collect();
    let ours: std::collections::BTreeSet<String> =
        out.used.iter().map(|&i| join_name(&names, &candidates[i], None)).collect();

    for &i in &out.used {
        tracing::info!(join = %join_name(&names, &candidates[i], None), "used");
    }
    for r in &out.rejected {
        tracing::info!(
            join = %join_name(&names, &candidates[r.candidate], Some(&r.reason.message(&names))),
            "rejected"
        );
    }
    #[allow(clippy::cast_precision_loss, reason = "join counts are far below 2^53")]
    let count = |n: usize| n as f64;
    report.push(Check::absolute(
        SCOPE,
        "used only ours",
        count(ours.difference(&theirs).count()),
        0.0,
        NATIVE_JOINS,
    ));
    report.push(Check::absolute(
        SCOPE,
        "used only theirs",
        count(theirs.difference(&ours).count()),
        0.0,
        NATIVE_JOINS,
    ));

    let theirs_groups: Vec<Vec<String>> = npy::read_json_as(dir.join("groups.json"))?;
    let largest = |groups: &[Vec<String>]| groups.first().map_or(0, Vec::len);
    let ours_groups: Vec<Vec<String>> =
        out.groups.iter().map(|g| g.iter().map(|&n| names[n as usize].clone()).collect()).collect();
    #[allow(clippy::cast_precision_loss, reason = "group sizes are far below 2^53")]
    report.push(Check::absolute(
        SCOPE,
        "largest group",
        largest(&ours_groups) as f64,
        largest(&theirs_groups) as f64,
        NATIVE_GROUP_SIZE,
    ));
    // Every join the port used has to be inside one of the port's own groups, which is a
    // consistency check on the loop rather than a comparison with the reference.
    let mut stray = 0_usize;
    for &i in &out.used {
        let c = &candidates[i];
        let inside = out.groups.iter().any(|g| g.contains(&c.a) && g.contains(&c.b));
        stray += usize::from(!inside);
    }
    report.push(Check::entries(SCOPE, "joins inside a group", stray, out.used.len()));
    Ok(())
}

/// How many joins of a collection may belong to one side's assembly alone (a regression alarm).
///
/// Measured worst over the eight dumps: 2 joins the port uses and the reference does not (pot_G,
/// pot_H, synthetic_20) and **6** the reference uses and the port does not (synthetic_20); the gate
/// is twice that, the shape the PMC-6 tie rows of D §10.2 already use. It is a count and not a
/// share because a share is degenerate where one side used nothing at all — which is pot_G.
pub const NATIVE_JOINS: f64 = 12.0;
/// How many fragments the two largest groups may differ by (a regression alarm).
///
/// Measured worst: **4**, on synthetic_20, where the port assembles 15 fragments into its largest
/// group against the reference's 19; the gate is twice that.
pub const NATIVE_GROUP_SIZE: f64 = 8.0;
/// How far R §6.4's fraction may move between the two sample sets PMC-8 allows, on a rejection
/// both sides reach the same way.
///
/// A measurement beside the decision, not the gate on it — the decision is the `pmc8 rejected` row
/// above, which is exact. Measured worst over the eight dumps: **0.0078** on pot_A (0.129 against
/// the reference's 0.127) and 0.0060 on pot_H; zero on the other six, which reject nothing for
/// penetration. The threshold is 0.03 because the statistic is a share of two *independent*
/// samples of 15 000 and 20 000 points, whose binomial spread is at most `√(0.25/15000)` = 0.004
/// at one σ, so 0.03 is roughly 7σ of the largest sampling noise it can carry.
pub const PMC8_PEN: f64 = 0.03;

/// Two sequences that must be equal entry for entry, as a `Check::entries` row.
fn push_sequence(report: &mut StageReport, name: &'static str, ours: &[String], theirs: &[String]) {
    let differing = (0..ours.len().max(theirs.len()))
        .filter(|&i| ours.get(i) != theirs.get(i))
        .collect::<Vec<usize>>();
    if !differing.is_empty() {
        let show = |v: &[String], i: usize| v.get(i).map_or("(none)", String::as_str).to_owned();
        for &i in differing.iter().take(5) {
            tracing::warn!(
                row = name,
                index = i,
                ours = show(ours, i),
                theirs = show(theirs, i),
                "assembly differs"
            );
        }
    }
    report.push(Check::entries(SCOPE, name, differing.len(), ours.len().max(theirs.len())));
}

/// A group list flattened into one comparable string per group.
fn render(groups: &[Vec<String>]) -> Vec<String> {
    groups.iter().map(|g| g.join(",")).collect()
}

/// `a-b`, with the rejection reason appended when there is one.
fn join_name(names: &[String], c: &Candidate, reason: Option<&str>) -> String {
    let pair = format!("{}-{}", names[c.a as usize], names[c.b as usize]);
    reason.map_or(pair.clone(), |why| format!("{pair}: {why}"))
}

/// The same, from the dump's own entry.
fn ref_join_name(join: &RefJoin) -> String {
    let pair = format!("{}-{}", join.a, join.b);
    join.reason.as_ref().map_or(pair.clone(), |why| format!("{pair}: {why}"))
}

/// The same as [`ref_join_name`] with the measured numbers taken out of the reason.
fn ref_join_kind(join: &RefJoin) -> String {
    let pair = format!("{}-{}", join.a, join.b);
    join.reason.as_ref().map_or(pair.clone(), |why| {
        let kind = why.split_once(" (").map_or(why.as_str(), |(head, _)| head);
        format!("{pair}: {kind}")
    })
}

/// R §6.4's fraction out of the reference's own sentence, `penetrates NAME (0.127)`.
fn parse_pen(join: &RefJoin) -> Option<f64> {
    let why = join.reason.as_ref()?;
    if !why.starts_with("penetrates ") {
        return None;
    }
    why.rsplit_once(" (")?.1.trim_end_matches(')').parse().ok()
}

/// The largest displacement the two pose sets give any of a fragment's own surface samples, in `t`.
///
/// Shared with the `refine` and `outputs` rows, which measure a pose the same way.
///
/// Every tenth sample, which is R §8.2's own stride and enough of a lever arm: a fragment 450 units
/// from the origin whose two poses differ by 1e-12 in a rotation entry moves its points by 4.5e-10,
/// and this measure sees that where a comparison of the translation column alone would not.
pub fn worst_move(
    ours: &[Matrix4<f64>],
    theirs: &[Matrix4<f64>],
    pieces: &[Piece<'_>],
    wall: f64,
) -> f64 {
    let mut worst = 0.0_f64;
    for (n, piece) in pieces.iter().enumerate() {
        let (Some(ours), Some(theirs)) = (ours.get(n), theirs.get(n)) else { continue };
        for point in piece.s_pen.iter().step_by(10) {
            let here = apply_transform(ours, *point);
            let there = apply_transform(theirs, *point);
            let squared: f64 = (0..3).map(|k| (here[k] - there[k]).powi(2)).sum();
            worst = worst.max(squared.sqrt() / wall);
        }
    }
    worst
}

/// `{name: 4x4}` out of a JSON file, in collection order.
pub fn read_poses(path: &Path, names: &[String]) -> Result<Vec<Matrix4<f64>>> {
    let raw: BTreeMap<String, [[f64; 4]; 4]> = npy::read_json_as(path)?;
    Ok(names.iter().map(|n| raw.get(n).map_or_else(Matrix4::identity, matrix)).collect())
}

/// The same out of `transforms.json`, whose poses sit under `fragments.<name>.matrix`.
pub fn read_transforms(path: &Path, names: &[String]) -> Result<Vec<Matrix4<f64>>> {
    #[derive(serde::Deserialize)]
    struct Entry {
        matrix: [[f64; 4]; 4],
    }
    #[derive(serde::Deserialize)]
    struct Transforms {
        fragments: BTreeMap<String, Entry>,
    }
    let raw: Transforms = npy::read_json_as(path)?;
    Ok(names
        .iter()
        .map(|n| raw.fragments.get(n).map_or_else(Matrix4::identity, |e| matrix(&e.matrix)))
        .collect())
}

/// A 4x4 out of the nested lists both JSON files carry.
pub fn matrix(rows: &[[f64; 4]; 4]) -> Matrix4<f64> {
    let mut out = Matrix4::zeros();
    for (i, r) in rows.iter().enumerate() {
        for (j, v) in r.iter().enumerate() {
            out[(i, j)] = *v;
        }
    }
    out
}

/// `assembly/md_t_median.json`, or `None` when the dump does not carry the assembly at all.
pub fn md_t_median(collection: &Collection) -> Result<Option<MdTMedian>> {
    let path = collection.dir.assembly_dir().join("md_t_median.json");
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(npy::read_json_as(path)?))
}

/// Position of a fragment in collection order.
pub fn index_of(collection: &Collection, name: &str) -> Option<usize> {
    collection.fragments.iter().position(|f| f.name == name)
}

/// Every fragment as the dump holds it, or `None` with the skip recorded.
pub fn reference_pieces(
    collection: &Collection,
    report: &mut StageReport,
) -> Result<Option<Vec<RefPiece>>> {
    let Some(md) = md_t_median(collection)? else {
        report.skip(SCOPE, "no assembly/md_t_median.json in the dump (level min)");
        return Ok(None);
    };
    let mut out = Vec::new();
    for fragment in &collection.fragments {
        let stats = fragment.file("mesh.stats.json");
        if !stats.is_file() {
            report.skip(SCOPE, "the dump has no mesh.stats.json for a fragment");
            return Ok(None);
        }
        let stats: RefStats = npy::read_json_as(stats)?;
        let dir = collection.dir.assembly_dir().join("md").join(&fragment.name);
        let params = dir.join("md.params.json");
        if !params.is_file() {
            report.skip(SCOPE, "the dump has no assembly-stage samples for a fragment");
            return Ok(None);
        }
        let value = npy::read_json(&params)?;
        let their_t = npy::field_f64(&value, "t", &params)?;
        let their_points = npy::field_u64(&value, "surface_points", &params)?;
        if their_t.to_bits() != md.t.to_bits() || their_points != md.surface_points {
            return Err(Error::fixture(
                params,
                "the assembly-stage samples were not drawn at md_t_median's own (t, points)",
            ));
        }
        out.push(RefPiece {
            thick: stats.thick,
            res: stats.res,
            watertight: stats.watertight,
            geometry: RefGeometry::of(fragment)?,
            s_pen: npy::read_points(dir.join("md.S.npy"))?,
        });
    }
    Ok(Some(out))
}

/// The borrowed view R §8 takes of [`reference_pieces`]'s owned arrays.
pub fn piece_views(pieces: &[RefPiece]) -> Vec<Piece<'_>> {
    pieces
        .iter()
        .map(|p| Piece {
            thick: p.thick,
            res: p.res,
            watertight: p.watertight,
            mesh: p.geometry.as_ref().and_then(|g| g.scene.as_ref()),
            s_pen: &p.s_pen,
        })
        .collect()
}

/// Every candidate of every pair, in the order the reference's own `cands` list holds them.
///
/// R §8 reads only the accepted ones and only the best of each pair, but the *order* of the list
/// is what breaks a tie between two pairs of equal score, so the whole list is rebuilt rather than
/// filtered here.
pub fn reference_candidates(
    collection: &Collection,
    report: &mut StageReport,
) -> Result<Option<Vec<Candidate>>> {
    let mut out = Vec::new();
    for names in &collection.manifest.pairs.pairs {
        let dir = collection.dir.pair_dir(&names[0], &names[1]);
        let file = dir.join("result.candidates.json");
        if !dir.is_dir() {
            // A pair R §4.1's wall-ratio test skipped has no directory and produced no candidate.
            continue;
        }
        if !file.is_file() {
            // R §5's three early exits return `[]` *before* the sink is written, so a pair with no
            // fracture sample, no hypothesis or nothing above R §5.3's floor has no
            // `result.candidates.json` and contributed nothing to the reference's own `cands`.
            // Each exit leaves its own evidence in the dump, and the absence of all three is a
            // truncated dump rather than an empty pair — three of synthetic_20's 190 pairs are
            // the third case, with `nms1.kept` written and empty.
            let no_arrays = !dir.join("md_used.json").is_file();
            let no_hypotheses =
                empty_array(&dir.join("hyp.pa.npy"))? || !dir.join("hyp.pa.npy").is_file();
            let nothing_kept = empty_array(&dir.join("nms1.kept.npy"))?;
            if no_arrays || no_hypotheses || nothing_kept {
                tracing::info!(
                    pair = %format!("{}__{}", names[0], names[1]),
                    "R §5 returned before stage 2: no candidate"
                );
                continue;
            }
            report.skip(SCOPE, "a pair of the dump has no result.candidates.json (level min)");
            return Ok(None);
        }
        let (Some(a), Some(b)) = (index_of(collection, &names[0]), index_of(collection, &names[1]))
        else {
            report.skip(SCOPE, "a pair of the dump names a fragment the manifest does not");
            return Ok(None);
        };
        let theirs: Vec<super::candidates::RefCandidate> = npy::read_json_as(file)?;
        for c in theirs {
            out.push(Candidate {
                a: u32::try_from(a).expect("fewer than 2^32 fragments"),
                b: u32::try_from(b).expect("fewer than 2^32 fragments"),
                transform: c.pose(),
                scores: c.scores,
                accepted: c.accepted,
            });
        }
    }
    Ok(Some(out))
}

/// True when the dump carries that array and it holds nothing — R §5's own evidence that a pair
/// stopped where it did.
fn empty_array(path: &Path) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    Ok(npy::read_indices(path)?.is_empty())
}

/// Every fragment preprocessed by the port, in collection order and with its [`FragId`] set.
pub fn native_fragments(
    collection: &Collection,
    report: &mut StageReport,
) -> Result<Option<Vec<Fragment>>> {
    let mut out = Vec::new();
    for (i, fragment) in collection.fragments.iter().enumerate() {
        let Some(source) = &fragment.source else {
            report.skip(SCOPE, "no source file (pass --input DIR)");
            return Ok(None);
        };
        let started = std::time::Instant::now();
        let (mut fr, _) = Fragment::load_or_build(
            source,
            collection.target_faces,
            &fragment.name,
            None,
            DUMP_SEED,
        )?;
        fr.id = u32::try_from(i).expect("fewer than 2^32 fragments");
        tracing::info!(
            fragment = %fragment.name,
            seconds = started.elapsed().as_secs_f64(),
            "preprocessed"
        );
        out.push(fr);
    }
    Ok(Some(out))
}
