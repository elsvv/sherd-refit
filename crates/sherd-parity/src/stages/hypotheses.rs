//! The hypotheses stage: frame pairs and the poses they imply (R §5.1, D §10.2 row `hypotheses`).
//!
//! # Injected
//!
//! The stage runs on the reference's own subsets (`hyp.ia`, `hyp.ib`) and the reference's own
//! frames at the pair's `t`, so what is compared is the dihedral filter and the frame mapping and
//! nothing upstream of them.
//!
//! * **`n_hyp` and `pairs`** are D §10.2's `(pa, pb)` set, exact: every frame pair the reference
//!   kept, the port keeps, and no other.
//! * **`order`** counts the positions where the two lists differ. It is separate from the set
//!   because PMC-4 lets the two implementations carry `brk_sub` in different orders — but not
//!   *here*: injected mode is handed the reference's own `ia`/`ib`, so the row-major walk has to
//!   produce the reference's own order too, and the check is exact. The row exists to say which
//!   property failed when one does.
//! * **`rotation` and `translation`** compare each pose against one built in this crate from
//!   R §5.1's formulas directly — a second reading of the same three lines, so that a transposed
//!   product or a mislabelled column shows up as a number rather than as agreement. The
//!   load-bearing check on `R` and `τ` is the `coarse` stage next door, whose scores the reference
//!   dumped for every hypothesis; this one is the cheap independent cross-check.
//! * **`scales`** counts the fields of R §1.2 the port resolves differently from the reference's
//!   own `scales.json`, from the pair's own `t` and `res`. It is exact, and it is here rather than
//!   in a stage of its own because every threshold below depends on it.
//!
//! # Native
//!
//! The port preprocesses both fragments from their files, picks its own `t_pair`, rebuilds R §3.5
//! at it and forms its own subsets: the hypothesis *count* is then a statistic of two independent
//! breaklines and D §10.2 allows it ±30 %. The pair must still be matched at all — a pair the
//! reference matched and the port skips on R §4.1's wall ratio is a failure, not a tolerance.

use std::collections::BTreeMap;

use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::matching::hypotheses::{self, Frames, Hypotheses};
use sherd_core::matching::pair::Pair;
use sherd_core::matching::scales::Scales;

use super::Collection;
use super::pairs::{PairFixture, reference_frames};
use crate::report::{Check, Mode, StageReport};

/// D §10.2, injected column: the rotation of a hypothesis, in degrees.
pub const INJECTED_ROTATION_DEG: f64 = 1e-4;
/// D §10.2, injected column: the translation of a hypothesis, in wall thicknesses.
pub const INJECTED_TRANSLATION_T: f64 = 1e-5;
/// D §10.2, native column: the hypothesis count.
pub const NATIVE_COUNT: f64 = 0.30;

/// Runs R §5.1 for every pair of the dump and compares it.
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("hypotheses", mode);
    match mode {
        Mode::Injected => injected(collection, &mut report)?,
        Mode::Native => native(collection, &mut report)?,
    }
    Ok(report)
}

fn injected(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let params = collection.manifest.collection.params;
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        let Some((fa, fb, used)) = sides(collection, &pair, report)? else { continue };
        if !pair.has("hyp.pa.npy") {
            report.skip(&scope, "no hyp.pa in the dump (level min)");
            continue;
        }
        let theirs = pair.hypotheses()?;
        if !theirs.describes(&fa, &fb) {
            report.skip(&scope, "the dump's hypothesis indices do not describe its own breaklines");
            continue;
        }

        // R §1.2 from the pair's own two numbers, against the reference's own resolution of it.
        let sc = Scales::for_pair(&params, used.t, used.res_a.max(used.res_b));
        report.push(Check::entries(&scope, "scales", scales_differing(&sc, &pair.scales()?), 12));

        let ours = hypotheses::build_with(&fa, &fb, &theirs.ia, &theirs.ib, params.dihedral_tol);
        report.push(Check::count(&scope, "n_hyp", ours.len() as u64, theirs.pa.len() as u64));

        // Both lists are row-major over `(i, j)` and so lexicographically ascending; a merge is
        // therefore enough for the set difference and for matching like with like.
        let (only_ours, only_theirs, common) = merge(&ours.pa, &ours.pb, &theirs.pa, &theirs.pb);
        report.push(Check::entries(
            &scope,
            "pairs",
            only_ours + only_theirs,
            ours.len().max(theirs.pa.len()),
        ));
        let misordered = positional_differences(&ours, &theirs.pa, &theirs.pb);
        report.push(Check::entries(&scope, "order", misordered, theirs.pa.len()));

        let (rotation, translation) = pose_deviation(&fa, &fb, &theirs, &ours, &common, used.t);
        report.push(Check::absolute(&scope, "rotation", rotation, 0.0, INJECTED_ROTATION_DEG));
        report.push(Check::absolute(
            &scope,
            "translation",
            translation,
            0.0,
            INJECTED_TRANSLATION_T,
        ));
    }
    Ok(())
}

fn native(collection: &Collection, report: &mut StageReport) -> Result<()> {
    let params = collection.manifest.collection.params;
    let fragments = native_fragments(collection, report)?;
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        if !pair.has("hyp.pa.npy") {
            report.skip(&scope, "no hyp.pa in the dump (level min)");
            continue;
        }
        let (Some(a), Some(b)) = (fragments.get(&pair.a), fragments.get(&pair.b)) else {
            report.skip(&scope, "a fragment of the pair has no source file (pass --input DIR)");
            continue;
        };
        let theirs = hypotheses_count(&pair)?;

        // The reference matched this pair, so R §4.1 kept it; a port that drops it has no count
        // to compare and the disagreement is about the pair, not about its size.
        let matched = !Pair::skipped(a, b, &params);
        report.push(Check::flag(&scope, "matched", matched, true));
        if !matched {
            continue;
        }
        let built = Pair::build(a, b, &params);
        report.push(Check::flag(&scope, "matchable", built.matchable(), true));
        if !built.matchable() {
            continue;
        }
        #[allow(clippy::cast_precision_loss, reason = "hypothesis counts are far below 2^53")]
        report.push(Check::relative(
            &scope,
            "n_hyp",
            built.hypotheses(&params).len() as f64,
            theirs as f64,
            NATIVE_COUNT,
        ));
    }
    Ok(())
}

/// The reference's frames for both sides of a pair at the pair's own `t`, or `None` with the skip
/// already recorded.
pub fn sides(
    collection: &Collection,
    pair: &PairFixture,
    report: &mut StageReport,
) -> Result<Option<(Frames, Frames, super::pairs::MdUsed)>> {
    let scope = pair.scope();
    if !pair.has("md_used.json") || !pair.has("scales.json") {
        report.skip(&scope, "no md_used.json in the dump (level min)");
        return Ok(None);
    }
    let used = pair.md_used()?;
    let (Some(a), Some(b)) = (collection.fragment(&pair.a), collection.fragment(&pair.b)) else {
        report.skip(&scope, "a fragment of the pair is not in the collection");
        return Ok(None);
    };
    let (Some(fa), Some(fb)) = (
        reference_frames(a, used.t, used.surface_points)?,
        reference_frames(b, used.t, used.surface_points)?,
    ) else {
        report.skip(&scope, "the dump has no breakline arrays at the pair's own t");
        return Ok(None);
    };
    Ok(Some((fa, fb, used)))
}

/// Every fragment of the collection, preprocessed by the port itself (native mode).
///
/// Built once and shared by every pair: a fragment of a 55-pair collection would otherwise be
/// segmented ten times over.
fn native_fragments(
    collection: &Collection,
    report: &mut StageReport,
) -> Result<BTreeMap<String, Fragment>> {
    let mut out = BTreeMap::new();
    for fragment in &collection.fragments {
        let Some(source) = &fragment.source else {
            report.skip(&fragment.name, "no source file (pass --input DIR)");
            continue;
        };
        let (fr, _) =
            Fragment::load_or_build(source, collection.target_faces, &fragment.name, None)?;
        out.insert(fragment.name.clone(), fr);
    }
    Ok(out)
}

/// How many hypotheses the reference kept for this pair.
///
/// Through the dtype-agnostic reader rather than `read::<i64>`: `hyp.pa` is `int64` in every dump
/// written so far, and a native row that fails because a future sink narrowed it would be
/// reporting the wrong thing.
fn hypotheses_count(pair: &PairFixture) -> Result<usize> {
    Ok(crate::npy::read_indices(pair.file("hyp.pa.npy"))?.len())
}

/// How many of R §1.2's twelve fields the port resolved differently, bit for bit.
fn scales_differing(ours: &Scales, theirs: &Scales) -> usize {
    let mine = [
        ours.t,
        ours.res,
        ours.coarse,
        ours.stage1,
        ours.tight,
        ours.facing,
        ours.gap,
        ours.seam,
        ours.near,
        ours.pen,
        ours.nms,
        ours.icp,
    ];
    let yours = [
        theirs.t,
        theirs.res,
        theirs.coarse,
        theirs.stage1,
        theirs.tight,
        theirs.facing,
        theirs.gap,
        theirs.seam,
        theirs.near,
        theirs.pen,
        theirs.nms,
        theirs.icp,
    ];
    mine.iter().zip(&yours).filter(|(a, b)| a.to_bits() != b.to_bits()).count()
}

/// A merge over two lexicographically ascending `(pa, pb)` lists: how many entries are only in the
/// first, only in the second, and the positions of those in both.
fn merge(pa: &[u32], pb: &[u32], qa: &[u32], qb: &[u32]) -> (usize, usize, Vec<(usize, usize)>) {
    let (mut i, mut j) = (0, 0);
    let (mut only_ours, mut only_theirs) = (0, 0);
    let mut common = Vec::new();
    while i < pa.len() && j < qa.len() {
        let (ours, theirs) = ((pa[i], pb[i]), (qa[j], qb[j]));
        match ours.cmp(&theirs) {
            std::cmp::Ordering::Less => {
                only_ours += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                only_theirs += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                common.push((i, j));
                i += 1;
                j += 1;
            }
        }
    }
    (only_ours + (pa.len() - i), only_theirs + (qa.len() - j), common)
}

/// How many positions of the two hypothesis lists name different frame pairs.
fn positional_differences(ours: &Hypotheses, pa: &[u32], pb: &[u32]) -> usize {
    if ours.len() != pa.len() {
        return ours.len().max(pa.len());
    }
    (0..ours.len()).filter(|&h| ours.pa[h] != pa[h] || ours.pb[h] != pb[h]).count()
}

/// The worst rotation (degrees) and translation (in `t`) between the port's poses and poses built
/// here from R §5.1's formulas, over the frame pairs both sides kept.
#[allow(clippy::many_single_char_names, reason = "R §5.1's own names for the two frame sets")]
fn pose_deviation(
    a: &Frames,
    b: &Frames,
    theirs: &super::pairs::RefHypotheses,
    ours: &Hypotheses,
    common: &[(usize, usize)],
    t: f64,
) -> (f64, f64) {
    let (mut rotation, mut translation) = (0.0_f64, 0.0_f64);
    for &(h, k) in common {
        let (r, tau) = reference_pose(
            a,
            theirs.ia[theirs.pa[k] as usize] as usize,
            b,
            theirs.ib[theirs.pb[k] as usize] as usize,
        );
        // The angle of `Rᵀ R'`: `trace = 1 + 2cos θ`.
        let mut trace = 0.0;
        for (j, row) in r.iter().enumerate() {
            for (i, cell) in row.iter().enumerate() {
                trace += ours.r[h][(j, i)] * cell;
            }
        }
        rotation = rotation.max(sherd_core::matching::icp::pose_gap::angle_from_trace(trace));
        let d = (0..3).map(|i| (ours.tau[h][i] - tau[i]).powi(2)).sum::<f64>().sqrt();
        translation = translation.max(d / t);
    }
    (rotation, translation)
}

/// R §5.1's `R = RA·RBᵀ`, `τ = P_A − R·P_B`, written out a second time from the reference text.
///
/// Deliberately *not* the port's own [`hypotheses::poses`]: this is the independent reading the
/// injected `rotation` and `translation` rows compare against, so it assembles the two frames as
/// explicit column matrices and multiplies them with a plain triple loop rather than sharing any
/// expression with the code under test.
#[allow(clippy::many_single_char_names, reason = "R §5.1's own names for the two frames")]
fn reference_pose(a: &Frames, ka: usize, b: &Frames, kb: usize) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut frame_a = [[0.0; 3]; 3];
    let mut frame_b = [[0.0; 3]; 3];
    for (i, (row_a, row_b)) in frame_a.iter_mut().zip(&mut frame_b).enumerate() {
        *row_a = [a.tangent[ka][i], a.ns[ka][i], a.f[ka][i]];
        *row_b = [-b.tangent[kb][i], b.ns[kb][i], -b.f[kb][i]];
    }
    let mut rot = [[0.0; 3]; 3];
    for (i, row) in rot.iter_mut().enumerate() {
        for (k, cell) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += frame_a[i][j] * frame_b[k][j];
            }
            *cell = sum;
        }
    }
    let mut tau = [0.0; 3];
    for (i, out) in tau.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, cell) in rot[i].iter().enumerate() {
            sum += cell * b.p[kb][j];
        }
        *out = a.p[ka][i] - sum;
    }
    (rot, tau)
}

#[cfg(test)]
mod tests {
    use super::{merge, positional_differences};
    use sherd_core::matching::hypotheses::Hypotheses;

    #[test]
    fn the_merge_finds_what_only_one_side_kept() {
        let (pa, pb) = (vec![0, 0, 1], vec![0, 2, 1]);
        let (qa, qb) = (vec![0, 1, 1], vec![0, 1, 3]);
        // (0,0) common; (0,2) only ours; (1,1) common; (1,3) only theirs.
        let (only_ours, only_theirs, common) = merge(&pa, &pb, &qa, &qb);
        assert_eq!((only_ours, only_theirs), (1, 1));
        assert_eq!(common, vec![(0, 0), (2, 1)]);

        let (only_ours, only_theirs, common) = merge(&pa, &pb, &pa, &pb);
        assert_eq!((only_ours, only_theirs, common.len()), (0, 0, 3));
        assert_eq!(merge(&pa, &pb, &[], &[]), (3, 0, vec![]));
        assert_eq!(merge(&[], &[], &qa, &qb), (0, 3, vec![]));
    }

    #[test]
    fn a_reordered_list_is_the_same_set_and_a_different_order() {
        let ours = Hypotheses { pa: vec![0, 1], pb: vec![5, 4], ..Hypotheses::default() };
        assert_eq!(positional_differences(&ours, &[0, 1], &[5, 4]), 0);
        assert_eq!(positional_differences(&ours, &[1, 0], &[4, 5]), 2);
        assert_eq!(positional_differences(&ours, &[0], &[5]), 2, "a length difference is total");
    }
}
