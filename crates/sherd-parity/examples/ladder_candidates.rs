//! Per-candidate detail behind the `stage1` and `stage2` rows (task C2).
//!
//! The parity harness reports one worst case per pair and one distribution per dump; when a row
//! fails, or when the `chaotic` row moves, the next question is always *which* candidate and what
//! its ladder was doing — how many correspondences each rung had, whether it converged, and how
//! far the same ladder moves when the initial pose moves by one ULP. This prints that, one line
//! per candidate:
//!
//! ```text
//! cargo run --release -p sherd-parity --example ladder_candidates -- DUMP 'a__b' 1
//! ```
//!
//! The third argument is the stage (`1` or `2`); the default is `1`. Candidates outside D §10.2's
//! tolerances are marked `BAD` and get the ULP spread printed beside them. The last line is the
//! time the ladders themselves took — one thread, the ULP probes excluded — which is the number
//! the note compares against the reference's.

use sherd_core::error::Result;
use sherd_core::executor::Engine;
use sherd_core::matching::icp;
use sherd_core::matching::ladder::{self, Rung};
use sherd_core::matching::scales::Scales;
use sherd_core::matching::{coarse, hypotheses};
use sherd_parity::FixtureDir;
use sherd_parity::npy;
use sherd_parity::report::{Mode, StageReport};
use sherd_parity::stages::{Collection, pose_gap};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: ladder_candidates DUMP PAIR [1|2]");
    let want = args.next().expect("a pair scope, as the parity table prints it");
    let stage = args.next().unwrap_or_else(|| "1".to_owned());
    let collection = Collection::open(FixtureDir::new(&dir), None)?;
    let params = collection.manifest.collection.params;
    let mut report = StageReport::new("scratch", Mode::Injected);
    let mut normals = sherd_parity::stages::pairs::NormalCache::default();
    for pair in collection.pair_fixtures().into_iter().filter(|p| p.scope() == want) {
        let Some((fa, fb, used)) =
            sherd_parity::stages::hypotheses::sides(&collection, &pair, &mut report)?
        else {
            continue;
        };
        let Some((a, b)) = sherd_parity::stages::clouds(
            &collection,
            &pair,
            &fa,
            &fb,
            &used,
            &params,
            &mut normals,
            &mut report,
        )?
        else {
            continue;
        };
        let sc = pair.scales()?;
        println!("t {:.4} res {:.4} icp {:.4}", sc.t, sc.res, sc.icp);
        let mut spent = std::time::Duration::ZERO;
        let mut climbed = 0_usize;
        if stage == "1" {
            let theirs = pair.hypotheses()?;
            let hyp = hypotheses::poses(&fa, &fb, &theirs.ia, &theirs.ib, &theirs.pa, &theirs.pb);
            let kept = npy::read_indices(pair.file("nms1.kept.npy"))?;
            let poses = npy::read_transforms(pair.file("s1.T.npy"))?;
            let target = a.brk_full.target();
            let rungs = ladder::stage1_rungs(&b.brk_sub.p, &target);
            let tree = sherd_core::spatial::kdtree::PointTree::build(&a.brk_full.p)
                .expect("a breakline has points");
            let scoring =
                coarse::Target { points: &a.brk_full.p, normals: &a.brk_full.n, tree: &tree };
            for (i, &h) in kept.iter().enumerate() {
                let init = icp::homogeneous(&hyp.r[h as usize], &hyp.tau[h as usize]);
                let started = std::time::Instant::now();
                let out = ladder::climb(Engine::REFERENCE, &rungs, &init, &sc);
                let last = out.last().expect("two rungs");
                let score = ladder::brk_score(
                    Engine::REFERENCE,
                    &scoring,
                    &b.brk_sub.p,
                    &b.brk_sub.n,
                    &last.transform,
                    sc.stage1,
                );
                spent += started.elapsed();
                climbed += 1;
                line(i, &[pose_gap(&last.transform, &poses[i], sc.t)], &out, &rungs, &init, &sc);
                if score > 0.0 {
                    println!("          s1 {score:.6}");
                }
            }
        } else {
            let stage1 = npy::read_transforms(pair.file("s1.T.npy"))?;
            let kept2 = npy::read_indices(pair.file("nms2.kept.npy"))?;
            let reg_target = a.reg.target();
            let frac_target = a.frac.target();
            let rungs = ladder::stage2_rungs(&b.reg.p, &reg_target, &b.frac.p, &frac_target);
            let names = ["s2.T_reg1", "s2.T_reg2", "s2.T_frac1", "s2.T_frac2"];
            let mut theirs = Vec::new();
            for name in names {
                theirs.push(npy::read_transforms(pair.file(&format!("{name}.npy")))?);
            }
            println!("clouds: reg {} frac {}", a.reg.len(), a.frac.len());
            for (i, &k) in kept2.iter().enumerate() {
                let init = stage1[k as usize];
                let started = std::time::Instant::now();
                let out = ladder::climb(Engine::REFERENCE, &rungs, &init, &sc);
                spent += started.elapsed();
                climbed += 1;
                let gaps: Vec<(f64, f64)> = out
                    .iter()
                    .zip(&theirs)
                    .map(|(r, t)| pose_gap(&r.transform, &t[i], sc.t))
                    .collect();
                line(i, &gaps, &out, &rungs, &init, &sc);
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "at most `stage1` ladders")]
        let each = 1e3 * spent.as_secs_f64() / climbed.max(1) as f64;
        println!(
            "{climbed} ladders in {:.3} s on one thread ({each:.1} ms each)",
            spent.as_secs_f64()
        );
    }
    Ok(())
}

/// One candidate's line: its per-rung deviations, what each rung's ICP did, and — when it is
/// outside the tolerances — how far one ULP of the initial pose moves the same ladder.
fn line(
    index: usize,
    gaps: &[(f64, f64)],
    out: &[icp::Registration],
    rungs: &[Rung<'_>],
    init: &nalgebra::Matrix4<f64>,
    sc: &Scales,
) {
    let bad = gaps.iter().any(|&(angle, distance)| angle > 0.05 || distance > 0.01);
    println!(
        "cand {index:3}{}: gaps {:?} corr {:?} iters {:?} conv {:?}{}",
        if bad { " BAD" } else { "" },
        gaps.iter().map(|&(a, d)| format!("{a:.2e}/{d:.2e}")).collect::<Vec<_>>(),
        out.iter().map(|r| r.correspondences).collect::<Vec<_>>(),
        out.iter().map(|r| r.iterations).collect::<Vec<_>>(),
        out.iter().map(|r| r.converged).collect::<Vec<_>>(),
        if bad { format!(" ulp spread {:?}", spread(rungs, init, sc)) } else { String::new() },
    );
}

/// The worst distance between the ladder's answer and its answers from the twelve one-ULP
/// neighbours of `init` — the probe `stages::determined` gates on, printed rather than judged.
fn spread(rungs: &[Rung<'_>], init: &nalgebra::Matrix4<f64>, sc: &Scales) -> (f64, f64) {
    let base = ladder::climb(Engine::REFERENCE, rungs, init, sc)
        .last()
        .expect("a ladder has rungs")
        .transform;
    let mut worst = (0.0_f64, 0.0_f64);
    for i in 0..3 {
        for j in 0..4 {
            let mut near = *init;
            near[(i, j)] = near[(i, j)].next_up();
            let out = ladder::climb(Engine::REFERENCE, rungs, &near, sc);
            let (angle, distance) =
                pose_gap(&out.last().expect("a ladder has rungs").transform, &base, sc.t);
            worst = (worst.0.max(angle), worst.1.max(distance));
        }
    }
    worst
}
