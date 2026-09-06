//! Surface, fracture and margin samples, and the runtime `MatchData` built on them
//! (R §3.5.1–3.5.2, R §3.5.6, R §3.6, R §10).
//!
//! The breakline of [`breakline`](super::breakline) says where a fragment's fracture *ends*; these
//! arrays say what its surfaces *are*. Three area-weighted samples, drawn once per fragment and
//! cached beside the mesh:
//!
//! * **`S`, `sp`** — 20 000 points over the whole surface with the face each came from. The
//!   penetration test of R §6.4 pushes these through the other fragment's working mesh, and the
//!   shell margin is selected out of them.
//! * **`Pf`, `fp`** — the fracture samples, `clip(⌊150·fracture_area/t²⌋, 5000, 12000)` of them,
//!   over the fracture faces alone. Every contact score of R §6.1 and both fine ICPs of R §5.6
//!   run on these.
//! * **`margin_idx`** — the shell samples in a band around the break, `0.12 t < d_brk < 1.5 t`,
//!   thinned to 6 000. They carry the shell continuity test of R §6.3 and half of the coarse ICP
//!   cloud of R §3.6.
//!
//! # The three rules that are easy to get wrong
//!
//! **The fracture count is a density, not a number.** `150` samples per `t²` of fracture area
//! describes a big sherd and a small one equally finely; the clamps keep the ICP affordable at
//! the top and meaningful at the bottom. Since `tight` and `gap` are measured against the other
//! fragment's *triangles* rather than against its samples, the count sets no floor under the
//! scores — what it buys is the ICP, which averages over that many correspondences.
//!
//! **The margin band is a band.** The inner edge (`0.12 t`) is there because crease faces
//! misclassified as shell sit right at the breakline and would otherwise dominate the
//! nearest-neighbour test; the outer edge (`1.5 t`) is there because shell far from the break says
//! nothing about whether two fragments meet. Neither is resolution-floored, which is **PMC-5** —
//! kept as it is for parity, and flagged in R §12 as a threshold to revisit.
//!
//! **`d_brk` is measured to the whole breakline, not to the hypothesis subset.**
//!
//! # Random numbers (PMC-9, D §7)
//!
//! The reference draws from one numpy `default_rng(seed)` per fragment, in this order: `n`
//! uniforms to pick the surface faces, `n` for `u`, `n` for `v`; then the same three for the
//! fracture samples; then, only when the margin is larger than `margin_points`, one
//! `choice(margin, 6000, replace=False)`. This port keeps the order *inside* each sampler and
//! gives each sampler its own stream ([`crate::rng::Draw`]), so that rebuilding the arrays at
//! another `t` or another `surface_points` (R §4.2, R §8) cannot move a sampler that did not
//! change. The uniforms themselves are numpy's construction — `(word >> 11)·2⁻⁵³` — over ChaCha8's
//! words rather than PCG64's, so the port draws *the same estimator on a different sample* and
//! native parity is statistical (D §10.2).
//!
//! # `f32`, and where the narrowing happens
//!
//! `S` and `Pf` are `f32` in the cache (D §4.2), so they are narrowed as soon as they are drawn
//! and everything derived from them — `d_brk`, the margin band, `Pm` — is computed from the
//! narrowed values. That is the rule [`WorkingMesh::from_parts`](crate::types::WorkingMesh) obeys
//! for the mesh, and for the same reason: a fragment read back from the cache and the same
//! fragment computed from its file must be the same fragment, so every stored array is a function
//! of the other stored arrays and of nothing wider.

use std::borrow::Cow;

use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::Fragment;
use super::breakline::{Breaklines, BrkParams};
use crate::mesh::geometry::{FaceGeometry, face_geometry, pairwise_sum};
use crate::rng::{self, Draw};
use crate::spatial::kdtree::PointTree;
use crate::types::{Cloud, FaceLabel};
use crate::vec3::Vec3f;

/// Whole-surface samples per fragment (R §1.1 `surface_points`).
pub const SURFACE_POINTS: u32 = 20_000;
/// Fracture samples per `t²` of fracture area (R §1.1 `frac_per_t2`).
pub const FRAC_PER_T2: f64 = 150.0;
/// Lower clamp of the fracture sample count (R §1.1 `min_frac_points`).
pub const MIN_FRAC_POINTS: u32 = 5_000;
/// Upper clamp of the fracture sample count (R §1.1 `max_frac_points`).
pub const MAX_FRAC_POINTS: u32 = 12_000;
/// Shell-margin samples kept after thinning (R §1.1 `margin_points`).
pub const MARGIN_POINTS: u32 = 6_000;
/// Points in the cloud the two coarse stage-2 ICPs run on; 0 means all of them (R §3.6).
pub const REG_POINTS: u32 = 6_000;
/// Inner edge of the shell margin, in wall thicknesses (R §3.5.6; PMC-5, no `res` floor).
pub const MARGIN_INNER: f64 = 0.12;
/// Outer edge of the shell margin, in wall thicknesses (R §3.5.6; PMC-5, no `res` floor).
pub const MARGIN_OUTER: f64 = 1.5;

/// The knobs R §3.5.1–3.5.2 and §3.5.6 depend on: the wall thickness they are measured in, the
/// seed, and the four counts.
///
/// This is the sampled half of the reference's `md_params` (R §3.7's `mdp_*`); the other half —
/// the two annulus radii and the voxel — belongs to the breaklines and travels as
/// [`BrkParams`](super::breakline::BrkParams). A cache whose `md_params` differ from the run's has
/// these arrays recomputed and nothing else, which is R §3.7's rule.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleParams {
    /// Wall thickness `t` the margin band is measured in and the fracture density divides by.
    pub t: f64,
    /// Seed of every draw (R §10).
    pub seed: u64,
    /// Whole-surface samples.
    pub surface_points: u32,
    /// Fracture samples per `t²`.
    pub frac_per_t2: f64,
    /// Lower clamp of the fracture sample count.
    pub min_frac_points: u32,
    /// Upper clamp of the fracture sample count.
    pub max_frac_points: u32,
    /// Shell-margin samples kept.
    pub margin_points: u32,
}

impl SampleParams {
    /// The shipped knobs at a wall thickness, with the default seed.
    pub fn at(t: f64) -> Self {
        Self {
            t,
            seed: 0,
            surface_points: SURFACE_POINTS,
            frac_per_t2: FRAC_PER_T2,
            min_frac_points: MIN_FRAC_POINTS,
            max_frac_points: MAX_FRAC_POINTS,
            margin_points: MARGIN_POINTS,
        }
    }
}

impl Default for SampleParams {
    /// The shipped knobs at `t = 0`, which describes no fragment — [`SampleParams::at`] is the
    /// constructor with a meaning.
    fn default() -> Self {
        Self::at(0.0)
    }
}

/// The sampled match arrays of one fragment (R §3.5.1–3.5.2, §3.5.6).
///
/// `s`/`sp` and `pf`/`fp` are parallel: one point and the face it was drawn from.
/// `margin_idx` indexes `s`, ascending.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Samples {
    /// What the arrays were built with (R §3.7).
    pub params: SampleParams,
    /// `S`: whole-surface samples (R §3.5.1).
    pub s: Vec<Vec3f>,
    /// `sp`: the face each surface sample came from.
    pub sp: Vec<u32>,
    /// `Pf`: fracture samples (R §3.5.2).
    pub pf: Vec<Vec3f>,
    /// `fp`: the face each fracture sample came from.
    pub fp: Vec<u32>,
    /// `margin_idx`: the shell-margin subset of `s`, ascending (R §3.5.6).
    pub margin_idx: Vec<u32>,
}

impl Samples {
    /// Number of whole-surface samples.
    #[inline]
    pub fn n_surface(&self) -> usize {
        self.s.len()
    }

    /// Number of fracture samples.
    #[inline]
    pub fn n_fracture(&self) -> usize {
        self.pf.len()
    }

    /// Number of shell-margin samples kept.
    #[inline]
    pub fn n_margin(&self) -> usize {
        self.margin_idx.len()
    }

    /// True when the fragment has no fracture sample at all — R §5's first exit.
    #[inline]
    pub fn has_fracture(&self) -> bool {
        !self.pf.is_empty()
    }

    /// The surface samples as `f64`, which is what a [`PointTree`] over them wants.
    pub fn surface_f64(&self) -> Vec<[f64; 3]> {
        self.s.iter().map(|p| p.to_f64()).collect()
    }

    /// The fracture samples as `f64`.
    pub fn fracture_f64(&self) -> Vec<[f64; 3]> {
        self.pf.iter().map(|p| p.to_f64()).collect()
    }
}

/// R §3.5.1, §3.5.2 and §3.5.6 for one labelled working mesh.
///
/// `v` and `geom` are the mesh's `f64` vertices and per-face arrays — the ones derived from the
/// **narrowed** working mesh (D §4.1), as everywhere else in R §3.4–3.5. `brk_points` is the whole
/// breakline of R §3.5.3, not the hypothesis subset: `d_brk` is measured to all of it.
pub fn build(
    v: &[[f64; 3]],
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    labels: &[FaceLabel],
    brk_points: &[[f64; 3]],
    params: SampleParams,
) -> Samples {
    // --- R §3.5.1: the whole surface -----------------------------------------------------------
    let all: Vec<u32> =
        (0..u32::try_from(f.len()).expect("the face count fits in u32")).collect::<Vec<u32>>();
    let mut rng = rng::seeded_for(params.seed, Draw::Surface);
    let (s, sp) =
        sample_on_faces(v, f, &geom.areas, &all, params.surface_points as usize, &mut rng);
    let s: Vec<Vec3f> = s.iter().map(|&p| Vec3f::from_f64(p)).collect();

    // --- R §3.5.2: the fracture ----------------------------------------------------------------
    let fracture: Vec<u32> = (0..f.len())
        .filter(|&i| labels[i].is_fracture())
        .map(|i| u32::try_from(i).expect("the face count fits in u32"))
        .collect();
    let n_frac = fracture_count(masked_area(&geom.areas, &fracture), params);
    let mut rng = rng::seeded_for(params.seed, Draw::Fracture);
    let (pf, fp) = sample_on_faces(v, f, &geom.areas, &fracture, n_frac, &mut rng);
    let pf: Vec<Vec3f> = pf.iter().map(|&p| Vec3f::from_f64(p)).collect();

    // --- R §3.5.6: the shell margin ------------------------------------------------------------
    // Measured on the *narrowed* samples, so that `margin_idx` is a function of the arrays the
    // cache holds and of nothing wider (module documentation).
    let surface: Vec<[f64; 3]> = s.iter().map(|p| p.to_f64()).collect();
    let d_brk = breakline_distance(&surface, brk_points);
    let margin = margin_indices(&sp, labels, &d_brk, params);
    let mut rng = rng::seeded_for(params.seed, Draw::Margin);
    let margin_idx = subsample(&margin, params.margin_points as usize, &mut rng);

    Samples { params, s, sp, pf, fp, margin_idx }
}

/// R §3.5.2's count rule: `int(clip(frac_per_t2 · fracture_area / t², min, max))`.
///
/// numpy clips first and truncates afterwards, and both bounds are integers, so the order does not
/// matter — except at the edges the reference never reaches: a `t` of zero makes the density
/// infinite (clipped to the upper bound) or, with no fracture at all, a NaN, where numpy's `int()`
/// would raise and this returns the lower bound. Neither can happen on a fragment R §3.1–3.3
/// produced.
pub fn fracture_count(fracture_area: f64, params: SampleParams) -> usize {
    let raw = params.frac_per_t2 * fracture_area / (params.t * params.t);
    if raw.is_nan() {
        return params.min_frac_points as usize;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped between two u32 bounds first, so the cast is exact"
    )]
    {
        raw.clamp(f64::from(params.min_frac_points), f64::from(params.max_frac_points)) as usize
    }
}

/// `sherd_refit.geometry.sample_on_faces`: `n` area-weighted random points on the faces of
/// `faces`, and the face each came from.
///
/// ```text
/// p    = A[faces] / ΣA[faces]
/// pick = faces[ searchsorted(cumsum(p)/cumsum(p)[-1], u, 'right') ]     # n uniforms
/// u, v = rng.random(n), rng.random(n);  where u + v > 1: (u, v) ← (1 − u, 1 − v)
/// P    = V[F[pick,0]] + u·(V[F[pick,1]] − V[F[pick,0]]) + v·(V[F[pick,2]] − V[F[pick,0]])
/// ```
///
/// The three draws are taken in that order and each in index order, which is the reference's
/// consumption of its generator (R §10). An empty selection, or `n = 0`, samples nothing.
///
/// A selection whose total area is not positive cannot occur after R §3.1's cleaning — numpy would
/// raise on it, since the probabilities would not sum to one — and is drawn uniformly here rather
/// than dividing by zero.
pub fn sample_on_faces(
    v: &[[f64; 3]],
    f: &[[u32; 3]],
    areas: &[f64],
    faces: &[u32],
    n: usize,
    rng: &mut ChaCha8Rng,
) -> (Vec<[f64; 3]>, Vec<u32>) {
    if faces.is_empty() || n == 0 {
        return (Vec::new(), Vec::new());
    }
    let cdf = cumulative_weights(areas, faces);

    // Draw first, map afterwards: the stream is consumed in one sequential pass whatever the
    // thread count, and the mapping is pure (D §7).
    let picks: Vec<u32> = (0..n)
        .map(|_| {
            let u = rng::unit_f64(rng);
            faces[cdf.partition_point(|&c| c <= u).min(faces.len() - 1)]
        })
        .collect();
    let first: Vec<f64> = (0..n).map(|_| rng::unit_f64(rng)).collect();
    let second: Vec<f64> = (0..n).map(|_| rng::unit_f64(rng)).collect();

    let points = picks
        .iter()
        .zip(first.iter().zip(&second))
        .map(|(&face, (&bary_u, &bary_v))| {
            // The fold: a draw outside the triangle is reflected into it, which keeps the
            // distribution uniform over the triangle rather than over the parallelogram.
            let (alpha, beta) =
                if bary_u + bary_v > 1.0 { (1.0 - bary_u, 1.0 - bary_v) } else { (bary_u, bary_v) };
            let tri = f[face as usize];
            let (origin, along_u, along_v) =
                (v[tri[0] as usize], v[tri[1] as usize], v[tri[2] as usize]);
            [
                origin[0] + alpha * (along_u[0] - origin[0]) + beta * (along_v[0] - origin[0]),
                origin[1] + alpha * (along_u[1] - origin[1]) + beta * (along_v[1] - origin[1]),
                origin[2] + alpha * (along_u[2] - origin[2]) + beta * (along_v[2] - origin[2]),
            ]
        })
        .collect();
    (points, picks)
}

/// The normalised cumulative area of a face selection, as numpy's `choice` builds it: `p = A/ΣA`
/// with `ΣA` numpy's pairwise sum, then a sequential `cumsum`, then a division by its last entry.
fn cumulative_weights(areas: &[f64], faces: &[u32]) -> Vec<f64> {
    let total = masked_area(areas, faces);
    #[allow(clippy::cast_precision_loss, reason = "a face count, far below 2^53")]
    let weight = |i: u32| {
        if total > 0.0 && total.is_finite() {
            areas[i as usize] / total
        } else {
            1.0 / faces.len() as f64
        }
    };
    let mut cdf = Vec::with_capacity(faces.len());
    let mut acc = 0.0;
    for &i in faces {
        acc += weight(i);
        cdf.push(acc);
    }
    let last = *cdf.last().expect("a non-empty selection");
    if last > 0.0 {
        for c in &mut cdf {
            *c /= last;
        }
    }
    cdf
}

/// `A[mask].sum()` — numpy's pairwise summation over the selected faces, which is what R §3.4's
/// `fracture_area` and R §3.5.1's weights are both built from.
pub fn masked_area(areas: &[f64], faces: &[u32]) -> f64 {
    let selected: Vec<f64> = faces.iter().map(|&i| areas[i as usize]).collect();
    pairwise_sum(&selected)
}

/// `d_brk`: the distance from every query point to the nearest breakline point, or `+∞` for every
/// point when the fragment has no breakline (R §3.5.6, `cKDTree.query`).
pub fn breakline_distance(queries: &[[f64; 3]], brk_points: &[[f64; 3]]) -> Vec<f64> {
    let Some(tree) = PointTree::build(brk_points) else {
        return vec![f64::INFINITY; queries.len()];
    };
    queries.par_iter().map(|q| tree.nearest_distance(q).1).collect()
}

/// R §3.5.6's margin: `np.where(¬frac[sp] ∧ 0.12 t < d_brk < 1.5 t)`, ascending.
///
/// Strict on both sides, and neither bound is floored by `res` (PMC-5). A sample with no
/// breakline to measure against has `d_brk = ∞`, which fails the outer test, so a fragment
/// without a breakline has no margin — the reference's behaviour, by the same arithmetic.
pub fn margin_indices(
    sp: &[u32],
    labels: &[FaceLabel],
    d_brk: &[f64],
    params: SampleParams,
) -> Vec<u32> {
    let (inner, outer) = (MARGIN_INNER * params.t, MARGIN_OUTER * params.t);
    (0..sp.len())
        .filter(|&i| !labels[sp[i] as usize].is_fracture() && d_brk[i] > inner && d_brk[i] < outer)
        .map(|i| u32::try_from(i).expect("the sample count fits in u32"))
        .collect()
}

/// `_subsample`: a seeded random subset of at most `n` entries, **sorted**, so that the order of
/// the underlying samples is preserved.
///
/// Fewer entries than asked for, or `n = 0`, returns the input untouched and draws nothing — which
/// is the reference's early return, and is why a small margin consumes no random numbers.
pub fn subsample(idx: &[u32], n: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
    if n == 0 || idx.len() <= n {
        return idx.to_vec();
    }
    let mut chosen: Vec<u32> =
        rng::without_replacement(idx.len(), n, rng).iter().map(|&k| idx[k as usize]).collect();
    chosen.sort_unstable();
    chosen
}

/// The runtime half of the reference's `MatchData` (R §3.6): everything the matcher reads that is
/// *derived* from the cached arrays and therefore never stored.
///
/// Built per `(fragment, t)` and shared behind an `Arc` by the pair loop (D §5). The KD-trees are
/// built here rather than cached, which is the point of the split: the arrays are the expensive
/// half and they come off the disk, the trees cost milliseconds.
///
/// A pair is matched at `t_pair = min(t_A, t_B)` (R §4.2), so the fragment whose own `t` is
/// `t_pair` uses its cached arrays and the other rebuilds both halves at `t_pair`.
/// [`MatchData::at`] does that decision; a rebuild is identical to a from-scratch build.
#[derive(Debug)]
pub struct MatchData<'a> {
    /// The fragment these arrays describe.
    pub fragment: &'a Fragment,
    /// The wall thickness they were built at — the fragment's own, or a pair's `t_pair`.
    pub t: f64,
    /// R §3.5.3–3.5.5's breakline, the fragment's own when `t` is its own.
    pub brk: Cow<'a, Breaklines>,
    /// R §3.5.1–3.5.6's samples, likewise.
    pub samples: Cow<'a, Samples>,
    /// `SN = FN[sp]`: the face normal at each surface sample.
    pub surface_normals: Vec<Vec3f>,
    /// `S_frac = frac[sp]`: whether each surface sample sits on the fracture.
    pub surface_fracture: Vec<bool>,
    /// `Nf = FN[fp]`: the face normal at each fracture sample.
    pub fracture_normals: Vec<Vec3f>,
    /// `brk_t = ns × f`: the tangent at each breakline point (R §3.6).
    pub brk_tangent: Vec<Vec3f>,
    /// `brk_dih`: the dihedral at each breakline point, in degrees (R §3.6).
    pub brk_dih: Vec<f64>,
    /// `Pm`, `Nm`: the shell-margin points and their normals.
    pub margin: Cloud,
    /// `pc_reg`: the cloud the two coarse stage-2 ICPs run on (R §3.6).
    pub pc_reg: Cloud,
    /// `pc_frac`: the fracture samples with their normals.
    pub pc_frac: Cloud,
    /// `pc_brk`: the hypothesis subset of the breakline, with `ns`.
    pub pc_brk: Cloud,
    /// `pc_brk_full`: the whole breakline, with `ns`.
    pub pc_brk_full: Cloud,
    /// A KD-tree over the whole breakline (R §6.2's seam test); `None` when there is none.
    pub kd_brk: Option<PointTree>,
    /// A KD-tree over the margin points (R §6.3's continuity test); `None` when there are none.
    pub kd_margin: Option<PointTree>,
    /// `frac_area`: the fragment's fracture area (R §3.4), which R §6.1's `contact` scales by.
    pub frac_area: f64,
}

impl<'a> MatchData<'a> {
    /// R §3.6 at a fragment's own wall thickness, from its cached arrays.
    pub fn own(fragment: &'a Fragment) -> Self {
        Self::build(fragment, fragment.thick, REG_POINTS as usize)
    }

    /// R §4.2: the arrays at `t`, cached when `t` is the fragment's own and rebuilt otherwise.
    ///
    /// `reg_points` is R §1.1's knob; `0` puts every fracture and margin point into `pc_reg`.
    pub fn at(fragment: &'a Fragment, t: f64, reg_points: usize) -> Self {
        Self::build(fragment, t, reg_points)
    }

    fn build(fragment: &'a Fragment, t: f64, reg_points: usize) -> Self {
        let (brk, samples) = arrays_at(fragment, t);
        let normals = &fragment.mesh.face_normals;
        let surface_normals: Vec<Vec3f> = samples.sp.iter().map(|&i| normals[i as usize]).collect();
        let surface_fracture: Vec<bool> =
            samples.sp.iter().map(|&i| fragment.labels[i as usize].is_fracture()).collect();
        let fracture_normals: Vec<Vec3f> =
            samples.fp.iter().map(|&i| normals[i as usize]).collect();

        let margin = Cloud {
            p: samples.margin_idx.iter().map(|&i| samples.s[i as usize]).collect(),
            n: samples.margin_idx.iter().map(|&i| surface_normals[i as usize]).collect(),
        };
        let pc_reg = registration_cloud(&samples.pf, &fracture_normals, &margin, reg_points);
        let pc_frac = Cloud { p: samples.pf.clone(), n: fracture_normals.clone() };
        let pc_brk = Cloud {
            p: brk.sub.iter().map(|&i| brk.p[i as usize]).collect(),
            n: brk.sub.iter().map(|&i| brk.ns[i as usize]).collect(),
        };
        let pc_brk_full = Cloud { p: brk.p.clone(), n: brk.ns.clone() };
        let kd_brk = PointTree::build(&brk.points_f64());
        let kd_margin =
            PointTree::build(&margin.p.iter().map(|p| p.to_f64()).collect::<Vec<[f64; 3]>>());

        Self {
            brk_tangent: brk.tangents(),
            brk_dih: brk.dihedrals(),
            frac_area: fragment.frac_area,
            fragment,
            t,
            brk,
            samples,
            surface_normals,
            surface_fracture,
            fracture_normals,
            margin,
            pc_reg,
            pc_frac,
            pc_brk,
            pc_brk_full,
            kd_brk,
            kd_margin,
        }
    }

    /// R §5's first exit: a fragment with no fracture sample or no breakline cannot be matched.
    #[inline]
    pub fn matchable(&self) -> bool {
        self.samples.has_fracture() && !self.brk.is_empty()
    }
}

/// The arrays at `t`: the fragment's own when they were built at `t`, rebuilt otherwise (R §4.2).
fn arrays_at(fragment: &Fragment, t: f64) -> (Cow<'_, Breaklines>, Cow<'_, Samples>) {
    let want_brk = BrkParams { t, ..fragment.brk.params };
    let want_md = SampleParams { t, ..fragment.samples.params };
    if fragment.brk.params == want_brk && fragment.samples.params == want_md {
        return (Cow::Borrowed(&fragment.brk), Cow::Borrowed(&fragment.samples));
    }
    let v64: Vec<[f64; 3]> = fragment.mesh.v.iter().map(|p| p.to_f64()).collect();
    let geom = face_geometry(&v64, &fragment.mesh.f);
    let brk = super::breakline::build(&v64, &fragment.mesh.f, &geom, &fragment.labels, want_brk);
    let samples =
        build(&v64, &fragment.mesh.f, &geom, &fragment.labels, &brk.points_f64(), want_md);
    (Cow::Owned(brk), Cow::Owned(samples))
}

/// R §3.6's `pc_reg`: a prefix of the fracture samples and a prefix of the margin, in proportion,
/// adding up to `reg_points`.
///
/// Both are i.i.d. area-weighted draws, so a prefix of each is still an area-weighted sample and
/// the split keeps their proportion. The rounding is numpy's round-half-to-even, and the fracture
/// side keeps at least one point so that a pose is still constrained across the break.
fn registration_cloud(pf: &[Vec3f], nf: &[Vec3f], margin: &Cloud, reg_points: usize) -> Cloud {
    let (mut take_f, mut take_m) = (pf.len(), margin.len());
    if reg_points > 0 && take_f + take_m > reg_points {
        #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
        let share = (take_f as f64) * (reg_points as f64) / ((take_f + take_m) as f64);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a rounded count below reg_points"
        )]
        let rounded = round_half_even(share) as usize;
        take_f = rounded.max(1).min(pf.len());
        take_m = reg_points.saturating_sub(take_f).min(margin.len());
    }
    let mut p = pf[..take_f].to_vec();
    p.extend_from_slice(&margin.p[..take_m]);
    let mut n = nf[..take_f].to_vec();
    n.extend_from_slice(&margin.n[..take_m]);
    Cloud { p, n }
}

/// numpy's `round`: half-way cases go to the even neighbour, which is what `f64::round_ties_even`
/// is. Named here because R §3.6 names it, and so that the one place it matters is greppable.
#[inline]
fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(test)]
mod tests {
    use super::{
        MARGIN_INNER, MARGIN_OUTER, MatchData, REG_POINTS, SampleParams, Samples,
        breakline_distance, build, fracture_count, margin_indices, masked_area, registration_cloud,
        round_half_even, sample_on_faces, subsample,
    };
    use crate::mesh::geometry::face_geometry;
    use crate::rng::{Draw, seeded, seeded_for};
    use crate::types::{Cloud, FaceLabel};
    use crate::vec3::Vec3f;

    /// A flat unit square in the `z = 0` plane, two triangles, total area 1.
    fn square() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        (
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    /// Every sample lands inside the triangle its index names, and the two triangles are hit in
    /// proportion to their areas.
    #[test]
    fn samples_are_area_weighted_and_land_on_their_face() {
        // The second triangle is made four times the first by stretching the square.
        let v = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-3.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let geom = face_geometry(&v, &faces);
        assert!((geom.areas[1] / geom.areas[0] - 4.0).abs() < 1e-12);

        let mut rng = seeded_for(0, Draw::Surface);
        let (points, pick) = sample_on_faces(&v, &faces, &geom.areas, &[0, 1], 20_000, &mut rng);
        assert_eq!(points.len(), 20_000);
        assert_eq!(pick.len(), 20_000);

        #[allow(clippy::cast_precision_loss, reason = "a sample count")]
        let share = pick.iter().filter(|&&k| k == 1).count() as f64 / 20_000.0;
        assert!((share - 0.8).abs() < 0.01, "{share}");

        for (point, &face) in points.iter().zip(&pick) {
            assert!(point[2].abs() < 1e-12, "the samples stay in the plane: {point:?}");
            // Barycentric coordinates of the point in its own triangle, both in [0, 1].
            let tri = faces[face as usize];
            let (alpha, beta) =
                barycentric(*point, v[tri[0] as usize], v[tri[1] as usize], v[tri[2] as usize]);
            assert!(alpha >= -1e-12 && beta >= -1e-12 && alpha + beta <= 1.0 + 1e-12);
        }
    }

    /// The coordinates of `query` in the frame `(second − first, third − first)`.
    fn barycentric(
        query: [f64; 3],
        first: [f64; 3],
        second: [f64; 3],
        third: [f64; 3],
    ) -> (f64, f64) {
        let e1 = [second[0] - first[0], second[1] - first[1]];
        let e2 = [third[0] - first[0], third[1] - first[1]];
        let rel = [query[0] - first[0], query[1] - first[1]];
        let det = e1[0] * e2[1] - e1[1] * e2[0];
        ((rel[0] * e2[1] - rel[1] * e2[0]) / det, (e1[0] * rel[1] - e1[1] * rel[0]) / det)
    }

    /// The same seed gives the same samples; another draw site gives different ones.
    #[test]
    fn the_draw_is_seeded_and_separated_by_purpose() {
        let (v, f) = square();
        let geom = face_geometry(&v, &f);
        let of = |draw| {
            let mut rng = seeded_for(0, draw);
            sample_on_faces(&v, &f, &geom.areas, &[0, 1], 64, &mut rng)
        };
        assert_eq!(of(Draw::Surface).0, of(Draw::Surface).0);
        assert_ne!(of(Draw::Surface).0, of(Draw::Fracture).0, "PMC-9: one stream per draw site");
    }

    /// An empty selection or a zero count samples nothing, and neither divides by anything.
    #[test]
    fn an_empty_selection_samples_nothing() {
        let (v, f) = square();
        let geom = face_geometry(&v, &f);
        let mut rng = seeded(0);
        assert_eq!(sample_on_faces(&v, &f, &geom.areas, &[], 100, &mut rng).0.len(), 0);
        assert_eq!(sample_on_faces(&v, &f, &geom.areas, &[0], 0, &mut rng).0.len(), 0);
    }

    /// R §3.5.2's clip, at both clamps and in between.
    #[test]
    fn the_fracture_count_is_a_density_between_two_clamps() {
        let p = SampleParams::at(2.0);
        // 150 · area / t² with t = 2: area 100 gives 3750 -> the lower clamp.
        assert_eq!(fracture_count(100.0, p), 5000);
        // area 400 gives 15000 -> the upper clamp.
        assert_eq!(fracture_count(400.0, p), 12000);
        // area 200 gives 7500, and the truncation is towards zero.
        assert_eq!(fracture_count(200.0, p), 7500);
        assert_eq!(fracture_count(200.001, p), 7500, "int(), not round()");
        assert_eq!(fracture_count(0.0, p), 5000, "no fracture: the clamp, and no samples anyway");
        assert_eq!(fracture_count(0.0, SampleParams::at(0.0)), 5000, "0/0 is not a panic");
    }

    /// The margin is a band on the shell, strict on both sides.
    #[test]
    fn the_margin_is_the_shell_inside_the_band() {
        let params = SampleParams::at(10.0);
        let labels = vec![FaceLabel::Shell, FaceLabel::Fracture];
        let sp = vec![0, 0, 0, 0, 1];
        let d = vec![
            MARGIN_INNER * 10.0,        // exactly on the inner edge: out
            MARGIN_INNER * 10.0 + 1e-9, // just inside: in
            MARGIN_OUTER * 10.0 - 1e-9, // just inside the outer edge: in
            MARGIN_OUTER * 10.0,        // exactly on it: out
            5.0,                        // in the band, but a fracture sample: out
        ];
        assert_eq!(margin_indices(&sp, &labels, &d, params), vec![1, 2]);

        // No breakline at all: every distance is infinite and nothing is in the band.
        let inf = vec![f64::INFINITY; 5];
        assert!(margin_indices(&sp, &labels, &inf, params).is_empty());
    }

    #[test]
    fn a_fragment_without_a_breakline_measures_infinity() {
        let d = breakline_distance(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], &[]);
        assert_eq!(d, vec![f64::INFINITY; 2]);
        let d = breakline_distance(&[[0.0, 0.0, 3.0]], &[[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]]);
        assert!((d[0] - 3.0).abs() < 1e-12);
    }

    /// The thinning keeps a sorted subset, draws nothing when it does not have to, and is seeded.
    #[test]
    fn the_margin_thinning_is_sorted_seeded_and_lazy() {
        let idx: Vec<u32> = (0..100).collect();
        let mut rng = seeded(0);
        assert_eq!(subsample(&idx, 100, &mut rng), idx, "no draw when it already fits");
        assert_eq!(subsample(&idx, 0, &mut rng), idx, "0 means no thinning (R §3.5.6)");

        let mut rng = seeded(0);
        let taken = subsample(&idx, 10, &mut rng);
        assert_eq!(taken.len(), 10);
        assert!(taken.windows(2).all(|w| w[0] < w[1]), "sorted and distinct: {taken:?}");
        assert!(taken.iter().all(|k| idx.contains(k)));
        let mut rng = seeded(0);
        assert_eq!(subsample(&idx, 10, &mut rng), taken, "the same seed, the same subset");
    }

    #[test]
    fn the_registration_split_keeps_the_proportion_and_the_count() {
        let cloud = |n: usize| Cloud { p: vec![Vec3f::ZERO; n], n: vec![Vec3f::ZERO; n] };
        let pf = vec![Vec3f::ZERO; 4000];
        let nf = vec![Vec3f::ZERO; 4000];
        let margin = cloud(6000);
        let reg = registration_cloud(&pf, &nf, &margin, 6000);
        assert_eq!(reg.len(), 6000);
        assert_eq!(reg.n.len(), 6000);
        // 4000 of 10000 at 6000 points: 2400 fracture, 3600 margin.
        assert_eq!(reg.p.len() - 3600, 2400);

        // Below the budget nothing is dropped.
        let small = registration_cloud(&pf[..100], &nf[..100], &cloud(200), 6000);
        assert_eq!(small.len(), 300);
        // reg_points = 0 keeps everything.
        assert_eq!(registration_cloud(&pf, &nf, &margin, 0).len(), 10_000);
        // A cloud that is almost all margin still keeps one fracture point.
        let one = registration_cloud(&pf[..1], &nf[..1], &cloud(1_000_000), 6000);
        assert_eq!(one.len(), 6000);
    }

    #[test]
    fn rounding_is_numpys_half_to_even() {
        assert!((round_half_even(0.5) - 0.0).abs() < 1e-12);
        assert!((round_half_even(1.5) - 2.0).abs() < 1e-12);
        assert!((round_half_even(2.5) - 2.0).abs() < 1e-12);
        assert!((round_half_even(-0.5) - 0.0).abs() < 1e-12);
        assert!((round_half_even(-1.5) + 2.0).abs() < 1e-12);
        assert!((round_half_even(2.4) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn the_masked_area_is_the_selection() {
        let (v, f) = square();
        let geom = face_geometry(&v, &f);
        assert!((masked_area(&geom.areas, &[0, 1]) - 1.0).abs() < 1e-12);
        assert!((masked_area(&geom.areas, &[0]) - 0.5).abs() < 1e-12);
        assert!(masked_area(&geom.areas, &[]).abs() < 1e-12);
    }

    /// The whole of R §3.5 on a labelled square: the surface samples cover both triangles, the
    /// fracture samples stay on the fracture one, and the margin is the shell inside the band.
    #[test]
    fn the_arrays_hang_together_on_a_labelled_square() {
        let (v, f) = square();
        let geom = face_geometry(&v, &f);
        let labels = vec![FaceLabel::Shell, FaceLabel::Fracture];
        // A breakline point in the middle of the shared edge.
        let brk = vec![[0.5, 0.5, 0.0]];
        let params = SampleParams { surface_points: 5000, ..SampleParams::at(1.0) };
        let md = build(&v, &f, &geom, &labels, &brk, params);

        assert_eq!(md.n_surface(), 5000);
        assert_eq!(md.sp.len(), 5000);
        assert!(md.sp.contains(&0) && md.sp.contains(&1));
        // 150 · 0.5 / 1 = 75, clamped up to the minimum.
        assert_eq!(md.n_fracture(), 5000);
        assert!(md.fp.iter().all(|&i| i == 1), "the fracture samples stay on the fracture");
        assert!(md.has_fracture());

        // Every margin sample is a shell sample inside the band around (0.5, 0.5).
        assert!(!md.margin_idx.is_empty());
        assert!(md.margin_idx.windows(2).all(|w| w[0] < w[1]));
        for &i in &md.margin_idx {
            assert_eq!(md.sp[i as usize], 0, "shell only");
            let p = md.s[i as usize].to_f64();
            let d = ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2)).sqrt();
            assert!(d > MARGIN_INNER && d < MARGIN_OUTER, "{d}");
        }
        // And the same parameters give the same arrays, twice.
        assert_eq!(build(&v, &f, &geom, &labels, &brk, params), md);
    }

    /// A mesh with no fracture face samples the surface and nothing else.
    #[test]
    fn a_fragment_that_never_broke_has_no_fracture_samples() {
        let (v, f) = square();
        let geom = face_geometry(&v, &f);
        let labels = vec![FaceLabel::Shell; 2];
        let params = SampleParams { surface_points: 100, ..SampleParams::at(1.0) };
        let md = build(&v, &f, &geom, &labels, &[], params);
        assert_eq!(md.n_surface(), 100);
        assert_eq!(md.n_fracture(), 0);
        assert!(!md.has_fracture());
        assert!(md.margin_idx.is_empty(), "no breakline, no band");
    }

    /// `MatchData` on the slab of the repository's fixture: every derived array is the right
    /// length and the clouds hang together.
    #[test]
    fn the_runtime_arrays_are_derived_from_the_cached_ones() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/slab/input/pieceA.ply");
        let fr = crate::fragment::Fragment::from_mesh_file(&path, 200_000).expect("the slab loads");
        let md = MatchData::own(&fr);

        assert!((md.t - fr.thick).abs() < 1e-12);
        assert!(
            matches!(md.samples, std::borrow::Cow::Borrowed(_)),
            "its own t: the cached arrays"
        );
        assert_eq!(md.surface_normals.len(), md.samples.n_surface());
        assert_eq!(md.surface_fracture.len(), md.samples.n_surface());
        assert_eq!(md.fracture_normals.len(), md.samples.n_fracture());
        assert_eq!(md.margin.len(), md.samples.n_margin());
        assert_eq!(md.pc_frac.len(), md.samples.n_fracture());
        assert_eq!(md.pc_brk.len(), md.brk.sub.len());
        assert_eq!(md.pc_brk_full.len(), md.brk.len());
        assert_eq!(md.brk_tangent.len(), md.brk.len());
        assert_eq!(md.brk_dih.len(), md.brk.len());
        assert_eq!(md.pc_reg.len(), REG_POINTS as usize);
        assert!(md.kd_brk.is_some() && md.kd_margin.is_some());
        assert!(md.matchable());
        assert!((md.frac_area - fr.fracture_area()).abs() < 1e-12);
        // The margin points are the surface samples the index names.
        for (k, &i) in md.samples.margin_idx.iter().enumerate() {
            assert_eq!(md.margin.p[k], md.samples.s[i as usize]);
        }

        // At another `t` both halves are rebuilt, and the rebuild is a from-scratch build.
        let other = MatchData::at(&fr, fr.thick * 0.9, REG_POINTS as usize);
        assert!(matches!(other.samples, std::borrow::Cow::Owned(_)));
        assert!((other.samples.params.t - fr.thick * 0.9).abs() < 1e-12);
        assert_ne!(other.samples.margin_idx, md.samples.margin_idx, "another band, another margin");
        assert_eq!(other.samples.s, md.samples.s, "the surface draw does not depend on t");
    }

    /// The empty case of every accessor, so that a fragment with nothing on it cannot panic.
    #[test]
    fn an_empty_sample_set_answers_every_question() {
        let s = Samples::default();
        assert_eq!(s.n_surface(), 0);
        assert_eq!(s.n_fracture(), 0);
        assert_eq!(s.n_margin(), 0);
        assert!(!s.has_fracture());
        assert!(s.surface_f64().is_empty());
        assert!(s.fracture_f64().is_empty());
        assert_eq!(s.params, SampleParams::at(0.0));
    }
}
