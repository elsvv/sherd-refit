//! Review images (audit §D.1, roadmap item 3): one PNG per join a person has to look at.
//!
//! > *"`render::render_pair(a, b, T)` on the existing splat renderer: three views (down the seam's
//! > mean shell normal, and the two shells), A grey and B orange, the seam's `t/3` voxels … drawn
//! > white, B's fracture samples coloured by their distance to A's fracture surface (green under
//! > `tight`, yellow under the gap limit, red beyond), a caption with the numbers, the tier and the
//! > seed. `review/<a>__<b>.png` for probable **and** confirmed joins."*
//!
//! This module is the half of that which measures; [`render_pair`](crate::render::render_pair) is
//! the half that draws. Nothing here computes a score of its own: the seam is the voxel list
//! [`seam_cells`](crate::matching::verify::seam_cells) forms — the same cells whose count *is*
//! R §6.2's `seam` — and the contact classes are R §6.1's own point-to-surface distances compared
//! against the pair's own `tight` and gap limit. A picture that disagreed with the table under it
//! would be worse than no picture.
//!
//! # What is drawn, and for which pairs
//!
//! One image per **pair**, for the pair's representative candidate (`tiers::representatives`) when
//! its band is [`Tier::Confirmed`] or [`Tier::Probable`] — and, on a run with the tier pass off,
//! for every pair R §6.5 accepted, captioned `accepted` because there is no band to print.
//! `review/<a>__<b>.png`, with `a` and `b` the two file stems in R §4.1's pair order.
//!
//! # Determinism
//!
//! Every image starts its own `rng(seed)` (R §10's seed, which `--seed` sets), so the samples a
//! pair is drawn from depend on the pair and on the seed and on nothing else — not on how many
//! images came before it, not on the thread that drew it, and not on the order rayon happened to
//! finish them in. Two runs of one collection therefore write byte-identical PNGs, which is the
//! gate this step is given.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nalgebra::Matrix4;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::error::{Error, Result};
use crate::executor::{DistBatch, DistReduce, Engine};
use crate::fragment::Fragment;
use crate::matching::pair::{Candidate, Pair};
use crate::matching::scales::Scales;
use crate::matching::verify::{SEAM_VOXEL, Surfaces, seam_cells, seam_points};
use crate::objects::ObjectReport;
use crate::params::Params;
use crate::render::{PAIR_VIEWS, PairEvidence, View, render_pair};
use crate::tiers::{Evidence, Tier, TierReport, representatives};
use crate::types::{FragId, apply_transform};

/// Samples drawn per fragment for a review image.
///
/// A group preview draws 250 000 (R §11.5); a review image draws two fragments in three views and
/// a collection can have eighty probable joins, so it draws fewer. At 100 000 the two sherds are
/// solid at 900 × 700 and the pass costs about a third of a second per image on this machine.
pub const REVIEW_POINTS: usize = 100_000;

/// The directory review images are written into, under the run's output directory.
pub const REVIEW_DIR: &str = "review";

/// Which pairs got a review image, by pair, with the path each one was written to relative to the
/// output directory.
///
/// `report.md`'s per-fragment index links these, which is the whole point of the file names being
/// predictable: a conservator reading the index clicks the row.
pub type ReviewIndex = BTreeMap<(FragId, FragId), String>;

/// `review/<a>__<b>.png` for every pair whose band is confirmed or probable (audit §D.1).
///
/// Returns the files written, in pair order, and the index `report.md` links. Runs after matching
/// and before the fracture BVHs are released, because the contact colouring is R §6.1's own
/// distance against A's fracture tree.
///
/// `probable_top` is `--probable-top`, and it cuts the same rows here as in `report.md`
/// ([`crate::tiers::probable_shown`]): every confirmed join, and the best `probable_top` of the
/// probable band. A run with the tier pass **off** has no band to cut and draws every accepted
/// pair, which is what keeps `--tiers off` the run it was before the flag existed.
#[allow(
    clippy::too_many_arguments,
    reason = "one pass's whole input: what to draw, from what, into where, at which band"
)]
pub fn write_review_images(
    engine: Engine<'_>,
    out_dir: &Path,
    fragments: &[Fragment],
    names: &[String],
    candidates: &[Candidate],
    tiers: Option<&TierReport>,
    objects: Option<&ObjectReport>,
    params: &Params,
    probable_top: usize,
) -> Result<(Vec<PathBuf>, ReviewIndex)> {
    let shown: std::collections::BTreeSet<usize> = if tiers.is_some() {
        crate::tiers::probable_shown(candidates, probable_top).0.into_iter().collect()
    } else {
        std::collections::BTreeSet::new()
    };
    let wanted: Vec<usize> = representatives(candidates)
        .into_iter()
        .filter(|&i| {
            let c = &candidates[i];
            if tiers.is_some() {
                match c.tier {
                    Tier::Confirmed => true,
                    Tier::Probable => shown.contains(&i),
                    Tier::Rejected => false,
                }
            } else {
                c.accepted
            }
        })
        .collect();
    if wanted.is_empty() {
        return Ok((Vec::new(), ReviewIndex::new()));
    }
    let dir = out_dir.join(REVIEW_DIR);
    std::fs::create_dir_all(&dir).map_err(|e| Error::write(&dir, e))?;

    let started = std::time::Instant::now();
    let drawn: Vec<Result<(usize, PathBuf, String)>> = wanted
        .par_iter()
        .map(|&i| {
            let c = &candidates[i];
            let (a, b) = (&fragments[c.a as usize], &fragments[c.b as usize]);
            let file = format!("{}__{}.png", names[c.a as usize], names[c.b as usize]);
            let evidence = pair_evidence(
                engine,
                a,
                b,
                c,
                tiers.and_then(|t| t.evidence.get(i).and_then(Option::as_ref)),
                params,
                tiers.is_some(),
                &names[c.a as usize],
                &names[c.b as usize],
                demotion_of(objects, &names[c.a as usize], &names[c.b as usize]),
            );
            render_pair(a, b, &c.transform, &evidence).write_png(dir.join(&file))?;
            Ok((i, dir.join(&file), format!("{REVIEW_DIR}/{file}")))
        })
        .collect();

    let mut written = Vec::with_capacity(drawn.len());
    let mut index = ReviewIndex::new();
    for row in drawn {
        let (i, path, relative) = row?;
        let c = &candidates[i];
        index.insert((c.a, c.b), relative);
        written.push(path);
    }
    tracing::info!(
        images = written.len(),
        points = REVIEW_POINTS,
        seconds = started.elapsed().as_secs_f64(),
        out = %dir.display(),
        "review images"
    );
    Ok((written, index))
}

/// The sentence roadmap item 4's object pass demoted this pair with, if it demoted it.
///
/// Audit §D.2 asks for the object numbers to be *"in the report and the review image"*, and this
/// is the review image's half of that: the picture a conservator would argue the demotion with
/// carries the demotion's own words. A pair the pass left alone gets no extra line, so a run with
/// nothing demoted draws exactly the images task T2 drew.
fn demotion_of(objects: Option<&ObjectReport>, a: &str, b: &str) -> Option<String> {
    let report = objects?;
    report
        .demotions
        .iter()
        .find(|d| (d.a == a && d.b == b) || (d.a == b && d.b == a))
        .map(|d| format!("OBJECT PASS ({}): {}", d.arm.label(), d.reason))
}

/// The seam, the contact classes, the three views and the caption of one candidate.
#[allow(clippy::too_many_arguments, reason = "one image's whole input, and it is a leaf function")]
fn pair_evidence(
    engine: Engine<'_>,
    a: &Fragment,
    b: &Fragment,
    candidate: &Candidate,
    evidence: Option<&Evidence>,
    params: &Params,
    banded: bool,
    a_name: &str,
    b_name: &str,
    demoted: Option<String>,
) -> PairEvidence {
    let pair = Pair::build(a, b, params);
    let sc = pair.scales;
    let transform = &candidate.transform;
    let (seam, contact, contact_class, views) = match pair.surfaces() {
        Some((sa, sb)) => {
            let voxel = sc.t / SEAM_VOXEL;
            let seam: Vec<[f64; 3]> = seam_cells(&sa, &sb, transform, &sc)
                .into_iter()
                .map(|c| centre_of(c, voxel))
                .collect();
            let (contact, contact_class) = contact_map(engine, &sa, &sb, transform, &sc);
            let views = pair_views(&sa, &sb, transform, &sc);
            (seam, contact, contact_class, views)
        }
        // A fragment whose working mesh has no triangle has no fracture tree and no scores; the
        // image is then the two point clouds and the caption, which is still worth writing.
        None => (Vec::new(), Vec::new(), Vec::new(), fallback_views()),
    };
    PairEvidence {
        views,
        seam,
        contact,
        contact_class,
        caption: caption(candidate, evidence, &sc, params, banded, a_name, b_name, demoted),
        points: REVIEW_POINTS,
        seed: params.seed,
    }
}

/// The centre of R §6.2's voxel `cell`, which is indexed by `⌊p / voxel⌋` on a grid at the origin.
#[inline]
#[allow(clippy::cast_precision_loss, reason = "a voxel index of a mesh coordinate is small")]
fn centre_of(cell: [i64; 3], voxel: f64) -> [f64; 3] {
    [(cell[0] as f64 + 0.5) * voxel, (cell[1] as f64 + 0.5) * voxel, (cell[2] as f64 + 0.5) * voxel]
}

/// B's fracture samples at the pose, and how far each is from A's fracture surface.
///
/// The distances are R §6.1's own `d1` — `bounded_distance` over A's fracture BVH with the pair's
/// `facing` window — and the three classes are the two limits R §6.5 judges the join by: under
/// `sc.tight` is green, under the gap limit `sc.gap` is yellow, and anything beyond (a point
/// outside the window included, which comes back `+∞`) is red.
fn contact_map(
    engine: Engine<'_>,
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
) -> (Vec<[f64; 3]>, Vec<u8>) {
    let distances = engine.exec.bounded_distance(&DistBatch {
        points: &b.pf,
        transform,
        scene: a.fracture,
        max_dist: sc.facing,
        reduce: DistReduce::All,
    });
    let points: Vec<[f64; 3]> = b.pf.iter().map(|p| apply_transform(transform, *p)).collect();
    let classes: Vec<u8> = distances
        .iter()
        .map(|&d| {
            if d < sc.tight {
                0
            } else if d < sc.gap {
                1
            } else {
                2
            }
        })
        .collect();
    (points, classes)
}

/// Audit §D.1's three directions: down the seam's mean shell normal, and down the two shells.
///
/// The shell normal of a fragment is the mean of R §3.5.6's shell-margin normals `Nm`, which is
/// the outward normal of the wall the pair shares; B's is rotated into A's frame by the pose. The
/// seam's is the mean of A's breakline macro-normals `brk_ns` over the points R §6.2 counted as
/// seam, so it is the wall *at the join* rather than over the whole sherd. Up is the seam's own
/// direction, which puts the join across the image rather than down it.
fn pair_views(
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
) -> [View; PAIR_VIEWS] {
    let on_seam = seam_points(a, b, transform, sc);
    let seam_normal = mean_unit(on_seam.iter().map(|&i| a.brk_ns[i as usize]));
    let a_shell = mean_unit(a.margin_n.iter().copied());
    let b_shell =
        mean_unit(b.margin_n.iter().map(|n| rotate(transform, *n))).or(seam_normal).or(a_shell);
    let seam_line: Vec<[f64; 3]> = on_seam.iter().map(|&i| a.brk_p[i as usize]).collect();
    let along = crate::tiers::principal_axis(&seam_line);

    let eyes = [
        seam_normal.or(a_shell).unwrap_or([0.0, 0.0, 1.0]),
        a_shell.or(seam_normal).unwrap_or([0.0, 1.0, 0.0]),
        b_shell.unwrap_or([1.0, 0.0, 0.0]),
    ];
    // The three directions are three views only as far as they differ, and how far that is is a
    // property of the pair: on a flat plate whose two shells cancel they can coincide, and on the
    // terracotta they are 9-83 degrees apart (the note's measurement). Logged rather than
    // asserted, because a degenerate pair still deserves its picture.
    let deg = |u: [f64; 3], v: [f64; 3]| {
        (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0).acos().to_degrees()
    };
    tracing::debug!(
        seam_vs_a = deg(eyes[0], eyes[1]),
        seam_vs_b = deg(eyes[0], eyes[2]),
        a_vs_b = deg(eyes[1], eyes[2]),
        "review views"
    );
    eyes.map(|eye| View { eye, up: up_for(eye, along) })
}

/// The views of a pair one of whose fragments has no surface at all.
fn fallback_views() -> [View; PAIR_VIEWS] {
    [
        View { eye: [0.0, 0.0, 1.0], up: [0.0, 1.0, 0.0] },
        View { eye: [0.0, 1.0, 0.0], up: [0.0, 0.0, 1.0] },
        View { eye: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] },
    ]
}

/// The roll of a view: the seam's own direction when it is not along the line of sight, and any
/// direction that is not otherwise.
fn up_for(eye: [f64; 3], along: Option<[f64; 3]>) -> [f64; 3] {
    let fallback = if eye[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    match along {
        Some(u) if (u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]).abs() < 0.99 => u,
        _ => fallback,
    }
}

/// The mean of a set of directions, normalised; `None` when the set is empty or cancels out.
fn mean_unit(directions: impl Iterator<Item = [f64; 3]>) -> Option<[f64; 3]> {
    let mut sum = [0.0_f64; 3];
    let mut n = 0_usize;
    for d in directions {
        for k in 0..3 {
            sum[k] += d[k];
        }
        n += 1;
    }
    if n == 0 {
        return None;
    }
    let norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    if norm.partial_cmp(&1e-12) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    Some([sum[0] / norm, sum[1] / norm, sum[2] / norm])
}

/// `R·n`, the rotation block alone — how R §6.2 moves a macro normal.
fn rotate(t: &Matrix4<f64>, n: [f64; 3]) -> [f64; 3] {
    [
        t[(0, 0)] * n[0] + t[(0, 1)] * n[1] + t[(0, 2)] * n[2],
        t[(1, 0)] * n[0] + t[(1, 1)] * n[1] + t[(1, 2)] * n[2],
        t[(2, 0)] * n[0] + t[(2, 1)] * n[1] + t[(2, 2)] * n[2],
    ]
}

/// Audit §D.1's caption: *"the numbers, the tier and the seed"*, plus the legend the colours need.
///
/// Three lines, drawn in the port's own upper-case bitmap font (PMC-20), so the text is written in
/// the alphabet that font has rather than in one it would draw as boxes.
#[allow(clippy::too_many_arguments, reason = "one caption's whole input, and it is a leaf")]
fn caption(
    candidate: &Candidate,
    evidence: Option<&Evidence>,
    sc: &Scales,
    params: &Params,
    banded: bool,
    a_name: &str,
    b_name: &str,
    demoted: Option<String>,
) -> Vec<String> {
    let s = &candidate.scores;
    let band = if banded {
        candidate.tier.label().to_uppercase()
    } else if candidate.accepted {
        "ACCEPTED".to_owned()
    } else {
        "REFUSED".to_owned()
    };
    // Task S3: the wide second placement is the one a margin can always be read against, so it is
    // the one the caption prints where it exists; `margin` falls back to R §5.7's own.
    let margin = evidence
        .and_then(|e| e.wide_margin.or(e.margin))
        .map_or_else(|| "NONE".to_owned(), |m| format!("{m:.2}"));
    let support = evidence.map_or_else(|| "-".to_owned(), |e| e.support.to_string());
    // And the arm that answered for a confirmed join, which is what a conservator holding the
    // image wants to know about the word in the band field.
    let arm = evidence.and_then(|e| e.arm.as_deref()).unwrap_or("-").to_uppercase();
    let slide =
        evidence.and_then(|e| e.slide_t).map_or_else(|| "-".to_owned(), |x| format!("{x:.2E}"));
    let mut lines = vec![
        format!(
            "A={a_name} GREY | B={b_name} ORANGE | {band} | ARM {arm} | MARGIN {margin} | \
             SUPPORT {support} | SEED {}",
            params.seed
        ),
        format!(
            "SCORE {:.2} | SEAM {:.1} T | TIGHT {:.2} | GAP {:.4} T (LIMIT {:.4}) | NORMAL AGR. \
             {:.3} | PEN {:.4} | SLIDE {slide} T",
            candidate.score(),
            s.seam,
            s.tight,
            s.gap,
            s.gap_limit,
            s.cont_n,
            s.pen,
        ),
        format!(
            "B FRACTURE SAMPLES: GREEN < TIGHT {:.4} | YELLOW < GAP {:.4} | RED BEYOND. WHITE = \
             SEAM VOXELS OF T/{:.0} = {:.4}",
            sc.tight,
            sc.gap,
            SEAM_VOXEL,
            sc.t / SEAM_VOXEL,
        ),
    ];
    // Task S2: one more line, and only where the collection has colour to put on it — a
    // colour-less collection's review image is the image it was.
    if let Some(colour) = evidence.and_then(|e| e.colour.as_ref()) {
        let de = colour.frac_delta_e.map_or_else(|| "-".to_owned(), |x| format!("{x:.1}"));
        let hist = colour.shell_hist.map_or_else(|| "-".to_owned(), |x| format!("{x:.2}"));
        lines.push(format!(
            "COLOUR: CLAY BODY DE76 {de} | SKIN HISTOGRAM DISTANCE {hist} | NOT PART OF THE BAND"
        ));
    }
    if let Some(sentence) = demoted {
        lines.push(sentence.to_uppercase());
    }
    lines
}
