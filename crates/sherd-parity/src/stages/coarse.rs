//! The coarse stage: one score per hypothesis (R §5.2, D §10.2 row `coarse`).
//!
//! # Injected
//!
//! Everything this stage needs is in the dump: the pair's `t` and `sc.coarse`, both fragments'
//! frames at that `t`, the hypothesis set as `(hyp.ia, hyp.ib, hyp.pa, hyp.pb)`, and — the reason
//! the row can be exact at all — `coarse.idx`, the sixty points the reference's own generator
//! drew. Fed those, the port has no random number left in it, and the score of every hypothesis
//! must be the reference's.
//!
//! The poses are rebuilt here from the reference's own `(pa, pb)` rather than from the port's
//! filter, so that a disagreement in the row above does not reappear as a disagreement in this
//! one: each row of D §10.2 measures its own stage.
//!
//! `cs` is what the row gates — D §10.2 allows `1/60 + 1e-6`, one probe point out of sixty — and
//! `cs exact` is the stronger statement the measurement supports: not one hypothesis of any
//! fixture pair scores differently at all. It is kept as its own row because the two say different
//! things when one of them fails: a single probe point that lands differently is a near-tie on
//! `sc.coarse` or on the `0.7` normal test, while a whole cloud of moved scores is a pose or a
//! tree.
//!
//! # Native
//!
//! There is no native column (D §10.2). The sixty points come out of a generator and PMC-9 lets
//! the port's be a different one, so natively the two implementations score the same hypotheses
//! through different lenses and the numbers are not comparable point by point. What *is* compared
//! natively is what the scores are for: the kept poses of the stages below.

use sherd_core::error::Result;
use sherd_core::matching::coarse::{self, Probe, Target};
use sherd_core::matching::hypotheses;
use sherd_core::spatial::kdtree::PointTree;

use super::Collection;
use crate::npy;
use crate::report::{Check, Mode, StageReport};

/// D §10.2, injected column: one probe point out of the sixty may land differently.
pub const INJECTED_SCORE: f64 = 1.0 / 60.0 + 1e-6;

/// Runs R §5.2 for every pair of the dump and compares it.
pub fn run(collection: &Collection, mode: Mode) -> Result<StageReport> {
    let mut report = StageReport::new("coarse", mode);
    let params = collection.manifest.collection.params;
    for pair in collection.pair_fixtures() {
        let scope = pair.scope();
        if mode == Mode::Native {
            report.skip(
                &scope,
                "D §10.2 has no native column here: the probe comes out of a generator and the \
                 port's is not numpy's (PMC-9)",
            );
            continue;
        }
        let Some((fa, fb, _used)) = super::hypotheses::sides(collection, &pair, &mut report)?
        else {
            continue;
        };
        if !pair.has("coarse.idx.npy") || !pair.has("coarse.cs.npy") || !pair.has("hyp.pa.npy") {
            report.skip(&scope, "no coarse.idx in the dump (level min)");
            continue;
        }
        let theirs_hyp = pair.hypotheses()?;
        let idx = npy::read_indices(pair.file("coarse.idx.npy"))?;
        let theirs = npy::read_f64(pair.file("coarse.cs.npy"))?;
        let sc = pair.scales()?;

        // R §5.2's count rule, and that the draw came from B's subset and from nowhere else.
        report.push(Check::count(
            &scope,
            "probe count",
            (params.coarse_points as usize).min(theirs_hyp.ib.len()) as u64,
            idx.len() as u64,
        ));
        let outside = idx.iter().filter(|i| !theirs_hyp.ib.contains(i)).count();
        report.push(Check::entries(&scope, "probe pool", outside, idx.len()));

        let hyp = hypotheses::poses(
            &fa,
            &fb,
            &theirs_hyp.ia,
            &theirs_hyp.ib,
            &theirs_hyp.pa,
            &theirs_hyp.pb,
        );
        let Some(tree) = PointTree::build(&fa.p) else {
            report.skip(&scope, "the reference's A-side breakline is empty");
            continue;
        };
        let target = Target { points: &fa.p, normals: &fa.ns, tree: &tree };
        let ours = coarse::scores(&target, &Probe::at(&fb, &idx), &hyp, sc.coarse);

        report.push(Check::count(&scope, "n_scored", ours.len() as u64, theirs.len() as u64));
        let (mut worst, mut differing) = (0.0_f64, 0);
        for (a, b) in ours.iter().zip(&theirs) {
            if a.to_bits() != b.to_bits() {
                differing += 1;
                worst = worst.max((a - b).abs());
            }
        }
        report.push(Check::absolute(&scope, "cs", worst, 0.0, INJECTED_SCORE));
        report.push(Check::entries(&scope, "cs exact", differing, ours.len().min(theirs.len())));
    }
    Ok(report)
}
