//! Per-fragment object features (audit §D.2, D §11, roadmap items 4 and 6).
//!
//! Item 4 wants to split a mixed collection into vessels before it matches anything: cross-object
//! pairs are 91 % of `mixed_all` (`notes/2026-09-06-scale-pairs.md` §4.3), so a cheap rigid
//! invariant per fragment is worth more than any seam search. The audit's rule for what may then
//! *veto* a pair is a measurement and not a guess — "object features veto only where their
//! measured AUC exceeds 0.8, otherwise they report" — so this module computes the features and
//! nothing else. It still decides nothing: task M1 measured every one of them on every collection
//! that carries real object ids and the best AUC is **0.740**, under audit §D.2's own 0.800 rule,
//! so the shortlist that may demote a join is empty and these numbers *report*
//! ([`crate::objects`]).
//!
//! # Where they are computed, and where they live
//!
//! Task O1 moved the pass into R §3's preprocessing and the answer into the cache
//! ([`CACHE_VERSION`](crate::CACHE_VERSION) 6, audit §D.2's own number): every fragment carries
//! its [`Features`] from the moment it is built, a warm run reads them back with the rest of
//! R §3, and a cache written before they existed is refused by its version and rebuilt. The pass
//! costs **0.15 s of `mixed_ABG`'s 3.2 s cold preprocessing** (24 fragments, measured) and nothing
//! at all on a warm cache. `sherd-refit-rs segment --features FILE` still writes task M1's table,
//! and now writes the fragments' own stored numbers rather than a second computation of them.
//!
//! # The seven features, and how each is defined
//!
//! | feature | what it is | why it might separate two vessels |
//! |---|---|---|
//! | `thick` | R §3.2's wall, already on the fragment | two pots are thrown at different walls |
//! | `thick_mode` / `rim` | R §3.2's plain ray mode over the wall; `rim` is the reference's own `thick_mode > 1.15 · thick` | a rim or a collar is thicker than the body it belongs to |
//! | `shell_radius` | radius of a sphere fitted to the **outer** shell samples | scale-pairs §4.3's second invariant: 17–60 units over pots A–J |
//! | `frac_rough` | RMS of the fracture samples to a local plane fitted in a `0.5 t` ball, in `t` | temper and fabric: a coarse-tempered body breaks rougher than a fine one |
//! | `axis_*` | the cylinder-axis of the shell normals, and the circle those samples make about it | a wheel-thrown pot has an axis; its diameter is the classic object evidence |
//! | `lab_*` | mean and spread of the source file's vertex colours in CIE Lab | fabric colour, where the scanner gives it |
//!
//! # What is deliberately approximate, and what that costs
//!
//! * **The sphere fit is algebraic, not geometric.** It minimises `Σ(|p−c|² − r²)²`, which is one
//!   4×4 solve rather than an iteration, and it is biased towards larger radii on a patch that
//!   covers little of the sphere. Every fragment of every set is such a patch, so the bias is
//!   common to all of them and a *comparison* between two fragments — which is all a veto is —
//!   survives it. [`Features::shell_rms`] is reported beside the radius so that a fit which said
//!   nothing can be told from one that did.
//! * **The rim edge is not segmented.** R §3.4 labels a face shell or fracture and nothing else,
//!   so there is no rim boundary to fit a circle to. What [`Features::rim_diameter`] reports is
//!   the diameter of the fragment's own arc about its fitted axis, which *is* the conservator's
//!   rim diameter when the fragment is a rim sherd and is a body diameter when it is not. The
//!   `rim` flag says which of the two the number is.
//! * **The axis is the cylinder axis of the shell normals**, the unit `d` minimising `Σ(n·d)²`.
//!   It is exact on a cylinder, near enough on a pot's body, and wrong on a base or a sharply
//!   curved shoulder — which is what [`Features::axis_residual`] measures, and why the audit gates
//!   the rim spike on that residual rather than on the diameter.
//!
//! Every fit refuses rather than guesses: fewer points than its own minimum, or a singular normal
//! matrix, gives `None`, and the aggregate drops that fragment from that feature's table instead
//! of feeding it a number nothing supports.

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use serde::{Deserialize, Serialize};

use crate::fragment::Fragment;
use crate::fragment::samples::MatchData;
use crate::mesh::geometry::median;
use crate::spatial::kdtree::PointTree;
use crate::types::FaceLabel;
use crate::vec3::Vec3f;

/// Fewest points any fit here will accept.
///
/// The same number R §6.1 calls [`MIN_FACING`](crate::matching::verify::MIN_FACING) and for the
/// same reason: twenty samples of a surface is a sliver of one triangle's worth, and a fit over
/// fewer describes the sliver.
pub const MIN_FIT_POINTS: usize = 20;

/// Neighbourhood the local plane of [`Features::frac_rough`] is fitted in, in wall thicknesses
/// (audit §D.2: "RMS of `Pf` to local plane fits at `0.5 t`").
pub const ROUGHNESS_BALL_T: f64 = 0.5;

/// Fewest neighbours a roughness ball needs before its plane is fitted.
///
/// A plane has three degrees of freedom; six points is the smallest neighbourhood whose residual
/// is a measurement of the surface rather than of the fit.
pub const MIN_PLANE_POINTS: usize = 6;

/// R §3.2's rim rule, as `Fragment::from_mesh_file` already logs it: the plain ray mode more than
/// this much above the wall is a rim or a collar (`fragment/mod.rs`, R §3.2).
pub const RIM_RATIO: f64 = 1.15;

/// Bins per Lab axis in [`ColourStats::hist`] — 4×4×4 = 64 (task S2, the brief's own number).
pub const HIST_BINS: usize = 4;

/// Low edge of the histogram window, per Lab axis (`L`, `a`, `b`).
///
/// The grid is **fixed and collection-independent** on purpose: a histogram whose edges move with
/// the set it was computed on is not comparable between two fragments of two collections, and the
/// whole point of the bins is a distance between two fragments. The window is the fired-earthenware
/// range measured on the sets this project has — `L` from a dark grey-brown body to a pale buff
/// slip, `a` from a green paint to a strongly red-firing fabric, `b` from neutral to yellow-brown —
/// and a value outside it falls in the end bin rather than being dropped.
///
/// Measured, on the four vertex clouds task S2 §2 reads: the three `mix3` vessels have their
/// median `(L, a, b)` at (33.9, 3.2, 9.2), (38.5, 15.3, 17.6) and (41.9, 8.5, 19.6) and the
/// monochrome pingsdorf control at (53.4, 27.4, 31.5) — four different cells of this grid, which is
/// what a coarse histogram has to do before it can be a distance.
pub const HIST_LO: [f64; 3] = [20.0, -10.0, -5.0];

/// High edge of the same window.
pub const HIST_HI: [f64; 3] = [80.0, 30.0, 35.0];

/// The colour of one side of a fragment — R §3.4's shell faces or its fracture faces — as the
/// source file spells it.
///
/// Three statements about the same multiset of vertex colours, in CIE Lab: where it sits
/// ([`lab_mean`](ColourStats::lab_mean)), how far it spreads ([`lab_mad`](ColourStats::lab_mad),
/// the median absolute deviation, which a painted band does not drag the way a standard deviation
/// does) and what shape it has ([`hist`](ColourStats::hist), the coarse 4×4×4 grid above). The mean
/// is the robust statement about a fracture face, which is bare clay; the histogram is the
/// statement a painted skin needs, where a mean over a cream ground and a blue-green band is a
/// colour neither of them is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColourStats {
    /// Mean CIE Lab over this side's vertices.
    pub lab_mean: [f64; 3],
    /// Median absolute deviation from the per-channel median, channel by channel.
    pub lab_mad: [f64; 3],
    /// Counts in the fixed 4×4×4 Lab grid, `L` slowest and `b` fastest.
    pub hist: Vec<u32>,
    /// How many vertices this side carried.
    pub points: usize,
}

impl ColourStats {
    /// The three statements over one side's Lab values, or `None` when the side has none.
    ///
    /// `None` and never a zero: a fragment whose file has no colours, and a fragment R §3.4 gave
    /// no fracture face at all, both have to read as *unavailable* rather than as black.
    #[must_use]
    pub fn of(labs: &[[f64; 3]]) -> Option<Self> {
        if labs.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss, reason = "vertex counts are far below 2^53")]
        let n = labs.len() as f64;
        let mut sum = [0.0_f64; 3];
        for lab in labs {
            for c in 0..3 {
                sum[c] += lab[c];
            }
        }
        let lab_mean = [sum[0] / n, sum[1] / n, sum[2] / n];
        let lab_mad = std::array::from_fn(|c| {
            let mut values: Vec<f64> = labs.iter().map(|lab| lab[c]).collect();
            let m = median(&values);
            for v in &mut values {
                *v = (*v - m).abs();
            }
            median(&values)
        });
        let mut hist = vec![0_u32; HIST_BINS * HIST_BINS * HIST_BINS];
        for lab in labs {
            hist[hist_index(*lab)] += 1;
        }
        Some(Self { lab_mean, lab_mad, hist, points: labs.len() })
    }

    /// CIE76 between two sides' mean Lab — the distance task S1 §5 measured colour's first AUC on.
    #[must_use]
    pub fn delta_e(&self, other: &Self) -> f64 {
        let d: [f64; 3] = std::array::from_fn(|c| self.lab_mean[c] - other.lab_mean[c]);
        ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt()
    }

    /// Total variation between the two normalised histograms: `0` for two identical shapes, `1`
    /// for two that share no bin.
    ///
    /// Total variation and not `χ²` because the counts are vertex counts and two fragments of one
    /// vessel differ in size by two orders of magnitude on these sets (S1 §2); normalising first
    /// and then summing half the absolute difference is the one distance that reads the *shape*
    /// and nothing about how many vertices the scanner wrote. Its known weakness is the bin edge —
    /// two colours a just-noticeable difference apart on opposite sides of one count as fully
    /// different — which is why the mean is reported beside it rather than replaced by it.
    #[must_use]
    pub fn hist_distance(&self, other: &Self) -> Option<f64> {
        let total = |h: &[u32]| f64::from(h.iter().copied().fold(0_u32, u32::saturating_add));
        let (mine, theirs) = (total(&self.hist), total(&other.hist));
        if self.hist.len() != other.hist.len() || mine == 0.0 || theirs == 0.0 {
            return None;
        }
        let sum: f64 = self
            .hist
            .iter()
            .zip(&other.hist)
            .map(|(&x, &y)| (f64::from(x) / mine - f64::from(y) / theirs).abs())
            .sum();
        Some(sum / 2.0)
    }
}

/// Which cell of the fixed 4×4×4 grid one Lab value falls in, `L` slowest and `b` fastest.
fn hist_index(lab: [f64; 3]) -> usize {
    let mut at = 0_usize;
    for c in 0..3 {
        #[allow(clippy::cast_precision_loss, reason = "HIST_BINS is 4")]
        let bins = HIST_BINS as f64;
        let t = (lab[c] - HIST_LO[c]) / (HIST_HI[c] - HIST_LO[c]) * bins;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped into 0..HIST_BINS before the cast"
        )]
        let bin = t.floor().clamp(0.0, bins - 1.0) as usize;
        at = at * HIST_BINS + bin;
    }
    at
}

/// The source file's vertices and colours, carried from R §3.1's read to R §3.4's labels.
///
/// [`Features::with_colour`] answers *what colour is this fragment*; the shell/fracture split needs
/// the same vertices again **after** the segmentation exists, and the segmentation runs on the
/// working mesh, which R §3.3's decimation has already stripped of colour. So the file's own
/// vertices are held — as `f32`, because their only use is a nearest-face lookup — for the length
/// of one `from_mesh_file_named` and dropped as soon as the split is taken. A file with no colours
/// holds nothing and pays nothing, which is what keeps every SfS++ collection exactly where it was.
#[derive(Clone, Debug, Default)]
pub struct RawColours {
    points: Vec<Vec3f>,
    colours: Vec<[u8; 3]>,
}

impl RawColours {
    /// The mesh's vertices and colours, or an empty carrier when the file has none.
    #[must_use]
    pub fn of(mesh: &crate::Mesh) -> Self {
        let Some(colours) = mesh.colors.as_ref() else { return Self::default() };
        if colours.is_empty() || colours.len() != mesh.v.len() {
            return Self::default();
        }
        Self {
            points: mesh.v.iter().copied().map(Vec3f::from_f64).collect(),
            colours: colours.clone(),
        }
    }

    /// True when there is nothing to split.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colours.is_empty()
    }
}

/// The file's vertex colours split by R §3.4's labels: `(shell, fracture)`.
///
/// # How a file vertex is given a label it never had
///
/// R §3.4 labels *faces of the working mesh*, which is the decimated, Taubin-smoothed mesh; the
/// colours are on the vertices of the file. The bridge is the nearest working-mesh **face
/// centroid**: the two meshes describe one surface, they are in one frame, and the smoothing moves
/// a vertex by well under one `res`. The error this makes is a band about one `res` wide along the
/// rim where shell meets fracture — and a fracture face is half a wall deep, five to seven `res`
/// on these scans (S1 §1.2), so the band is a boundary effect and not a confusion of the two
/// populations. It is measured rather than argued: S2 §2 compares the fracture side's mean against
/// the clay body the generator painted it with.
///
/// Seed-independent by construction. Everything else in [`Features`] is measured on R §3.5's draw
/// and moves with `--seed`; this is measured on the labels and the file, so a warm cache can carry
/// it across a seed change the way it already carries [`Features::lab_mean`].
#[must_use]
pub fn split_colour(
    raw: &RawColours,
    centroids: &[[f64; 3]],
    labels: &[FaceLabel],
) -> (Option<ColourStats>, Option<ColourStats>) {
    if raw.is_empty() || centroids.is_empty() || labels.len() != centroids.len() {
        return (None, None);
    }
    let Some(tree) = PointTree::build(centroids) else { return (None, None) };
    let mut seen: std::collections::BTreeMap<[u8; 3], [f64; 3]> = std::collections::BTreeMap::new();
    let mut shell: Vec<[f64; 3]> = Vec::new();
    let mut fracture: Vec<[f64; 3]> = Vec::new();
    for (point, &rgb) in raw.points.iter().zip(&raw.colours) {
        let face = tree.nearest(&point.to_f64()) as usize;
        let lab = *seen.entry(rgb).or_insert_with(|| srgb_to_lab(rgb));
        if labels[face].is_fracture() {
            fracture.push(lab);
        } else {
            shell.push(lab);
        }
    }
    (ColourStats::of(&shell), ColourStats::of(&fracture))
}

/// Every feature of one fragment, `None` where the fragment does not support the fit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Features {
    /// The fragment's name, so the file is readable without the collection beside it.
    pub name: String,
    /// R §3.2's wall thickness.
    pub thick: f64,
    /// R §3.2's plain ray mode.
    pub thick_mode: f64,
    /// `thick_mode > 1.15 · thick`: the reference's own rim/collar flag.
    pub rim: bool,
    /// Radius of the sphere fitted to the outer shell samples.
    pub shell_radius: Option<f64>,
    /// RMS residual of that fit, in `t`.
    pub shell_rms: Option<f64>,
    /// How many samples the second (outer-only) fit ran on.
    pub shell_points: usize,
    /// RMS of the fracture samples to their local planes, in `t`.
    pub frac_rough: Option<f64>,
    /// How many fracture samples had a full enough ball to contribute.
    pub frac_rough_points: usize,
    /// `λ_min / (λ_min + λ_mid + λ_max)` of `Σ n nᵀ` over the shell normals: 0 on a perfect
    /// cylinder, 1/3 on an isotropic patch.
    pub axis_residual: Option<f64>,
    /// Diameter of the circle the shell samples make about the fitted axis.
    pub axis_diameter: Option<f64>,
    /// RMS residual of that circle, in `t`.
    pub axis_rms: Option<f64>,
    /// [`Features::axis_diameter`] when [`Features::rim`] is set, `None` otherwise — the number a
    /// conservator would call the rim diameter.
    pub rim_diameter: Option<f64>,
    /// Mean CIE Lab of the source file's vertex colours, or `None` when the file has none.
    pub lab_mean: Option<[f64; 3]>,
    /// Standard deviation of the same, channel by channel.
    pub lab_spread: Option<[f64; 3]>,
    /// How many vertices carried a colour.
    pub colour_points: usize,
    /// Number of distinct RGB triples among them — 1 is a flat paint job, not a fabric.
    pub colour_distinct: usize,
    /// The colour of R §3.4's **shell** faces alone, or `None` where the file has none (task S2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shell_colour: Option<ColourStats>,
    /// The colour of R §3.4's **fracture** faces alone — the clay body, where a sherd is bare.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frac_colour: Option<ColourStats>,
}

impl Features {
    /// Every feature of one fragment except the colours, which need the source file
    /// ([`Features::with_colour`]).
    ///
    /// The samples are the fragment's own — R §3.5's draw at the run's seed — so two seeds give
    /// two slightly different tables, which is the same honesty R §13's bands have.
    pub fn of(fragment: &Fragment) -> Self {
        let md = MatchData::own(fragment);
        let shell: Vec<usize> = (0..md.samples.n_surface())
            .filter(|&i| !md.surface_fracture[i])
            .filter(|&i| md.surface_normals[i].norm_squared() > 0.0)
            .collect();
        let points: Vec<[f64; 3]> =
            shell.iter().map(|&i| md.samples.s[i].to_f64()).collect::<Vec<_>>();
        let normals: Vec<[f64; 3]> =
            shell.iter().map(|&i| md.surface_normals[i].to_f64()).collect::<Vec<_>>();
        let t = fragment.thick.max(f64::MIN_POSITIVE);

        let (shell_radius, shell_rms, shell_points) = outer_shell_sphere(&points, &normals, t);
        let (frac_rough, frac_rough_points) = fracture_roughness(&md.samples.fracture_f64(), t);
        let (axis_residual, axis_diameter, axis_rms) = axis_fit(&points, &normals, t);
        let rim = fragment.thick_mode > RIM_RATIO * fragment.thick;

        Self {
            name: fragment.name.clone(),
            thick: fragment.thick,
            thick_mode: fragment.thick_mode,
            rim,
            shell_radius,
            shell_rms,
            shell_points,
            frac_rough,
            frac_rough_points,
            axis_residual,
            axis_diameter,
            axis_rms,
            rim_diameter: if rim { axis_diameter } else { None },
            lab_mean: None,
            lab_spread: None,
            colour_points: 0,
            colour_distinct: 0,
            shell_colour: None,
            frac_colour: None,
        }
    }

    /// Fills the four colour fields from the fragment's source mesh, **as the file spells it**.
    ///
    /// The working mesh has no colours — decimation drops them (R §3.3) — so the fabric has to be
    /// read from the source. Preprocessing takes it through
    /// [`io::load_mesh_with`](crate::io::load_mesh_with), before `clean` merges duplicate
    /// vertices, which is the vertex multiset M1 §4's table was measured over; a file without
    /// colours leaves the fields as they are.
    ///
    /// The Lab of each distinct RGB triple is computed once and looked up after that. The
    /// accumulation order is the vertex order either way, so the sums are the same bits as the
    /// straight loop M1 measured with — what the memo buys is a museum scan with 500 000 vertices
    /// and 30 000 distinct colours, where the transform would otherwise run half a million times.
    pub fn with_colour(mut self, mesh: &crate::Mesh) -> Self {
        let Some(colours) = mesh.colors.as_ref() else { return self };
        if colours.is_empty() {
            return self;
        }
        let mut sum = [0.0_f64; 3];
        let mut sum2 = [0.0_f64; 3];
        let mut seen: std::collections::BTreeMap<[u8; 3], [f64; 3]> =
            std::collections::BTreeMap::new();
        for &rgb in colours {
            let lab = *seen.entry(rgb).or_insert_with(|| srgb_to_lab(rgb));
            for c in 0..3 {
                sum[c] += lab[c];
                sum2[c] += lab[c] * lab[c];
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "vertex counts are far below 2^53")]
        let n = colours.len() as f64;
        let mean = [sum[0] / n, sum[1] / n, sum[2] / n];
        let spread = std::array::from_fn(|c| (sum2[c] / n - mean[c] * mean[c]).max(0.0).sqrt());
        self.lab_mean = Some(mean);
        self.lab_spread = Some(spread);
        self.colour_points = colours.len();
        self.colour_distinct = seen.len();
        self
    }

    /// The colour fields of `from`, carried onto a freshly computed geometric table.
    ///
    /// [`Fragment::rebuild_features`](crate::fragment::Fragment::rebuild_features) is the caller:
    /// R §3.5's samples move when the seed does and every geometric feature moves with them, but
    /// the fabric is a property of the file and re-reading a 500 000-vertex scan to learn it again
    /// would be the most expensive thing a warm run did. The two [`ColourStats`] of task S2 are
    /// carried for the same reason and one stronger: they need R §3.4's labels beside the file's
    /// own vertices, and a warm run has thrown those vertices away.
    #[must_use]
    pub fn with_colour_of(mut self, from: &Self) -> Self {
        self.lab_mean = from.lab_mean;
        self.lab_spread = from.lab_spread;
        self.colour_points = from.colour_points;
        self.colour_distinct = from.colour_distinct;
        self.shell_colour.clone_from(&from.shell_colour);
        self.frac_colour.clone_from(&from.frac_colour);
        self
    }
}

/// Audit §D.2's table for a whole collection, in R §2's order.
///
/// `colour` decides whether each fragment's source file is read again for its vertex colours:
/// R §3.3's decimation drops them, so a Lab mean cannot come from the working mesh, and re-reading
/// 164 scans is the most expensive thing this pass can do. A file that cannot be read again is
/// logged and keeps its four colour fields empty — a feature table is not worth failing a
/// preprocessing pass for.
pub fn table(fragments: &[Fragment], colour: bool) -> Vec<Features> {
    use rayon::prelude::*;
    fragments
        .par_iter()
        .map(|fragment| {
            // Since task O1 the fragment carries its own table from preprocessing, colours and
            // all, and the cache holds it; recomputing here would answer with the same numbers at
            // the price of a second `MatchData`. `colour` then only decides whether a fragment
            // whose *stored* table has no colours is worth a second read of its file — which is
            // the case for a cache written by a build that could not read one.
            let features = fragment.features.clone().unwrap_or_else(|| Features::of(fragment));
            if !colour || features.colour_points > 0 {
                return features;
            }
            match crate::io::read_mesh(&fragment.source.path) {
                Ok(mesh) => features.with_colour(&mesh),
                Err(error) => {
                    tracing::warn!(
                        fragment = fragment.name,
                        %error,
                        "could not read the source file again for its colours"
                    );
                    features
                }
            }
        })
        .collect()
}

/// [`table`] written as pretty JSON, one array of rows.
pub fn write_table(rows: &[Features], path: &std::path::Path) -> crate::Result<()> {
    let json = serde_json::to_string_pretty(rows)
        .map_err(|e| crate::Error::write(path, std::io::Error::other(e)))?;
    std::fs::write(path, json).map_err(|e| crate::Error::write(path, e))
}

/// The sphere of the **outer** shell: fit once to every shell sample, keep the samples whose
/// normal points away from that centre, fit again.
///
/// One pass would answer with the mean of the two shells, whose radii differ by a wall; the second
/// pass is what makes the number comparable with scale-pairs §4.3's "radius of curvature of the
/// outer shell". Where the split leaves too few points — a fragment that is nearly all inner
/// surface — the first fit stands, and `shell_points` says which of the two happened.
fn outer_shell_sphere(
    points: &[[f64; 3]],
    normals: &[[f64; 3]],
    t: f64,
) -> (Option<f64>, Option<f64>, usize) {
    let Some((centre, radius)) = sphere_fit(points) else { return (None, None, 0) };
    let outer: Vec<[f64; 3]> = points
        .iter()
        .zip(normals)
        .filter(|(p, n)| {
            let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
            n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0.0
        })
        .map(|(p, _)| *p)
        .collect();
    let (used, centre, radius) = if outer.len() >= MIN_FIT_POINTS {
        match sphere_fit(&outer) {
            Some((c, r)) => (outer, c, r),
            None => (points.to_vec(), centre, radius),
        }
    } else {
        (points.to_vec(), centre, radius)
    };
    let rms = rms(used.iter().map(|p| distance(*p, centre) - radius)) / t;
    (Some(radius), Some(rms), used.len())
}

/// Centre and radius of the algebraic least-squares sphere through `points`.
///
/// `|p − c|² = r²` expands to `2 p·c + (r² − |c|²) = |p|²`, which is linear in the four unknowns
/// `(c, k)`; the normal equations are one 4×4 solve. `None` when the design matrix is singular —
/// fewer than [`MIN_FIT_POINTS`] points, or points on a plane through the origin of the fit.
fn sphere_fit(points: &[[f64; 3]]) -> Option<([f64; 3], f64)> {
    if points.len() < MIN_FIT_POINTS {
        return None;
    }
    let mut ata = Matrix4::<f64>::zeros();
    let mut atb = Vector4::<f64>::zeros();
    for p in points {
        let row = Vector4::new(2.0 * p[0], 2.0 * p[1], 2.0 * p[2], 1.0);
        let rhs = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        ata += row * row.transpose();
        atb += row * rhs;
    }
    let x = ata.lu().solve(&atb)?;
    let centre = [x[0], x[1], x[2]];
    let squared = x[3] + centre[0] * centre[0] + centre[1] * centre[1] + centre[2] * centre[2];
    (squared > 0.0 && squared.is_finite()).then(|| (centre, squared.sqrt()))
}

/// Audit §D.2's fracture roughness: the RMS distance of every fracture sample to a plane fitted to
/// its own neighbours inside a [`ROUGHNESS_BALL_T`] ball, in `t`.
///
/// The point itself is excluded from its plane, so the number is a *prediction* residual rather
/// than a fit residual and does not shrink towards zero as the ball empties. A sample whose ball
/// holds fewer than [`MIN_PLANE_POINTS`] neighbours contributes nothing and is counted out.
fn fracture_roughness(pf: &[[f64; 3]], t: f64) -> (Option<f64>, usize) {
    let Some(tree) = PointTree::build(pf) else { return (None, 0) };
    let radius = ROUGHNESS_BALL_T * t;
    let mut residuals = Vec::with_capacity(pf.len());
    let mut near = Vec::new();
    for (i, p) in pf.iter().enumerate() {
        tree.within_into(p, radius, &mut near);
        let neighbours: Vec<[f64; 3]> =
            near.iter().filter(|&&j| j as usize != i).map(|&j| pf[j as usize]).collect();
        if neighbours.len() < MIN_PLANE_POINTS {
            continue;
        }
        let Some((centre, normal)) = plane_fit(&neighbours) else { continue };
        let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
        residuals.push(d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2]);
    }
    if residuals.len() < MIN_FIT_POINTS {
        return (None, residuals.len());
    }
    let n = residuals.len();
    (Some(rms(residuals.into_iter()) / t), n)
}

/// Centroid and unit normal of the total-least-squares plane through `points` — the eigenvector of
/// the covariance with the smallest eigenvalue.
fn plane_fit(points: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3])> {
    let centre = centroid(points);
    let mut cov = Matrix3::<f64>::zeros();
    for p in points {
        let d = Vector3::new(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]);
        cov += d * d.transpose();
    }
    let normal = smallest_eigenvector(&cov)?;
    Some((centre, normal))
}

/// The cylinder axis of the shell normals and the circle the shell samples make about it.
///
/// A surface of revolution's normals are perpendicular to its axis, so the axis is the unit `d`
/// minimising `Σ(n·d)²` — the smallest eigenvector of `Σ n nᵀ`. The residual reported beside it is
/// that eigenvalue as a share of the trace: **0** when every normal is exactly perpendicular to
/// one direction (a cylinder), **1/3** when the normals point everywhere (a base, a sphere, a
/// sculptural sherd), and it is the number audit §D.2's rim spike is gated on.
///
/// The diameter comes from a circle fitted to the samples projected onto the plane through the
/// origin with normal `d`, which is the same algebraic least squares as [`sphere_fit`] one
/// dimension down.
fn axis_fit(
    points: &[[f64; 3]],
    normals: &[[f64; 3]],
    t: f64,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    if points.len() < MIN_FIT_POINTS {
        return (None, None, None);
    }
    let mut scatter = Matrix3::<f64>::zeros();
    for n in normals {
        let v = Vector3::new(n[0], n[1], n[2]);
        scatter += v * v.transpose();
    }
    let trace = scatter.trace();
    let Some(axis) = smallest_eigenvector(&scatter) else { return (None, None, None) };
    let residual = if trace > 0.0 {
        let d = Vector3::new(axis[0], axis[1], axis[2]);
        Some(((d.transpose() * scatter * d)[(0, 0)] / trace).max(0.0))
    } else {
        None
    };
    // Two unit vectors spanning the plane the axis is normal to; the pair is deterministic
    // because `off_axis` picks the smallest component of the axis, never a random vector.
    let u = unit(cross(axis, off_axis(axis)));
    let v = cross(axis, u);
    let flat: Vec<[f64; 3]> =
        points.iter().map(|p| [dot(*p, u), dot(*p, v), 0.0]).collect::<Vec<_>>();
    let Some((centre, radius)) = circle_fit(&flat) else { return (residual, None, None) };
    let rms = rms(flat.iter().map(|p| distance(*p, centre) - radius)) / t;
    (residual, Some(2.0 * radius), Some(rms))
}

/// [`sphere_fit`] in the plane: the same normal equations with the third coordinate dropped.
fn circle_fit(points: &[[f64; 3]]) -> Option<([f64; 3], f64)> {
    if points.len() < MIN_FIT_POINTS {
        return None;
    }
    let mut ata = Matrix3::<f64>::zeros();
    let mut atb = Vector3::<f64>::zeros();
    for p in points {
        let row = Vector3::new(2.0 * p[0], 2.0 * p[1], 1.0);
        let rhs = p[0] * p[0] + p[1] * p[1];
        ata += row * row.transpose();
        atb += row * rhs;
    }
    let x = ata.lu().solve(&atb)?;
    let centre = [x[0], x[1], 0.0];
    let squared = x[2] + centre[0] * centre[0] + centre[1] * centre[1];
    (squared > 0.0 && squared.is_finite()).then(|| (centre, squared.sqrt()))
}

/// The eigenvector of a symmetric 3×3 belonging to its smallest eigenvalue, unit and with a
/// deterministic sign (its largest component is made positive).
///
/// The sign matters because the axis is used to build a basis: an eigen decomposition is free to
/// return either `d` or `−d`, and a fragment must not get a different circle fit for that.
fn smallest_eigenvector(m: &Matrix3<f64>) -> Option<[f64; 3]> {
    if !m.iter().all(|x| x.is_finite()) {
        return None;
    }
    let eigen = m.symmetric_eigen();
    let mut best = 0;
    for i in 1..3 {
        if eigen.eigenvalues[i] < eigen.eigenvalues[best] {
            best = i;
        }
    }
    let column = eigen.eigenvectors.column(best);
    let mut v = [column[0], column[1], column[2]];
    let mut largest = 0;
    for i in 1..3 {
        if v[i].abs() > v[largest].abs() {
            largest = i;
        }
    }
    if v[largest] < 0.0 {
        v = [-v[0], -v[1], -v[2]];
    }
    unit(v).into()
}

/// The coordinate axis least aligned with `v` — the same `off_axis` `kernels/umeyama.wgsl` uses to
/// complete a basis, and here for the same reason: a cross product with it can never be zero.
fn off_axis(v: [f64; 3]) -> [f64; 3] {
    let mut smallest = 0;
    for i in 1..3 {
        if v[i].abs() < v[smallest].abs() {
            smallest = i;
        }
    }
    let mut e = [0.0; 3];
    e[smallest] = 1.0;
    e
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = dot(v, v).sqrt();
    if n > 0.0 { [v[0] / n, v[1] / n, v[2] / n] } else { [0.0, 0.0, 1.0] }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    dot(d, d).sqrt()
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
    let n = points.len() as f64;
    let mut c = [0.0; 3];
    for p in points {
        for i in 0..3 {
            c[i] += p[i];
        }
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

fn rms(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut n = 0_usize;
    for v in values {
        sum += v * v;
        n += 1;
    }
    #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
    if n == 0 { 0.0 } else { (sum / n as f64).sqrt() }
}

/// sRGB bytes to CIE Lab under D65, the colour space a fabric comparison wants.
///
/// RGB distances are not perceptual and are dominated by the scanner's exposure; Lab's `L` is
/// lightness and `a`/`b` are the two opponent axes, so a red fabric and a buff one are far apart
/// in `a` whatever the lamp did. The transfer function is the sRGB standard's, not a 2.2 power.
fn srgb_to_lab(rgb: [u8; 3]) -> [f64; 3] {
    let linear = |c: u8| {
        let c = f64::from(c) / 255.0;
        if c <= 0.040_45 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let [red, green, blue] = [linear(rgb[0]), linear(rgb[1]), linear(rgb[2])];
    // sRGB -> XYZ (D65), then XYZ -> Lab. Each row of the published matrix is divided by its own
    // sum, which is that channel's D65 white point (0.95047, 1.0, 1.08883) to the seven digits the
    // standard quotes: without the division white lands 3e-3 off neutral in `a`, which is a real
    // colour difference on a scale where a fabric comparison reads whole Lab units.
    let xyz = [
        (0.412_456_4 * red + 0.357_576_1 * green + 0.180_437_5 * blue) / 0.950_470,
        (0.212_672_9 * red + 0.715_152_2 * green + 0.072_175_0 * blue) / 1.000_000_1,
        (0.019_333_9 * red + 0.119_192_0 * green + 0.950_304_1 * blue) / 1.088_830,
    ];
    let bend = |v: f64| {
        if v > 216.0 / 24_389.0 { v.cbrt() } else { (841.0_f64 / 108.0).mul_add(v, 4.0 / 29.0) }
    };
    let [fx, fy, fz] = xyz.map(bend);
    [116.0f64.mul_add(fy, -16.0), 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

#[cfg(test)]
mod tests {
    use super::{
        ColourStats, Features, HIST_BINS, MIN_FIT_POINTS, RawColours, axis_fit, circle_fit,
        fracture_roughness, hist_index, plane_fit, sphere_fit, split_colour, srgb_to_lab,
    };
    use crate::types::FaceLabel;
    use crate::vec3::Vec3f;
    use approx::assert_relative_eq;

    /// Task S2: the split is decided by the nearest labelled face and by nothing else, and a side
    /// with no vertex is `None` rather than a black one.
    #[test]
    fn the_colour_split_follows_the_labels() {
        // Two faces a long way apart, one shell and one fracture, and four vertices near them.
        let centroids = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]];
        let labels = [FaceLabel::Shell, FaceLabel::Fracture];
        let raw = RawColours {
            points: vec![
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(-1.0, 0.0, 0.0),
                Vec3f::new(99.0, 0.0, 0.0),
                Vec3f::new(101.0, 0.0, 0.0),
            ],
            colours: vec![[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
        };
        let (shell, fracture) = split_colour(&raw, &centroids, &labels);
        let shell = shell.expect("two vertices sit on the shell face");
        let fracture = fracture.expect("two sit on the fracture face");
        assert_eq!((shell.points, fracture.points), (2, 2));
        assert_relative_eq!(shell.lab_mean[0], 100.0, epsilon = 1e-9, max_relative = 1e-9);
        assert_relative_eq!(fracture.lab_mean[0], 0.0, epsilon = 1e-9);
        // White against black is the whole of Lab's lightness axis, and the two histograms share
        // no bin.
        assert_relative_eq!(shell.delta_e(&fracture), 100.0, epsilon = 1e-9);
        assert_relative_eq!(shell.hist_distance(&fracture).expect("both have points"), 1.0);
        assert_relative_eq!(shell.hist_distance(&shell).expect("both have points"), 0.0);

        // A file with no colours splits into nothing at all, which is the SfS++ case.
        let (a, b) = split_colour(&RawColours::default(), &centroids, &labels);
        assert!(a.is_none() && b.is_none(), "no colour is unavailable, not zero");
    }

    /// The 4x4x4 grid is fixed, clamps at both ends, and orders `L` slowest.
    #[test]
    fn the_lab_histogram_grid_is_fixed_and_clamped() {
        assert_eq!(hist_index([-1000.0, -1000.0, -1000.0]), 0, "below every edge is bin 0");
        let last = HIST_BINS * HIST_BINS * HIST_BINS - 1;
        assert_eq!(hist_index([1000.0, 1000.0, 1000.0]), last, "above every edge is the last bin");
        // One step in `b` moves one bin; one step in `L` moves sixteen.
        assert_eq!(hist_index([21.0, -9.0, -4.0]), 0);
        assert_eq!(hist_index([21.0, -9.0, 6.0]), 1);
        assert_eq!(hist_index([21.0, 1.0, -4.0]), HIST_BINS);
        assert_eq!(hist_index([36.0, -9.0, -4.0]), HIST_BINS * HIST_BINS);
        // Every bin of a 64-bin histogram is counted exactly once over the whole cube.
        let labs: Vec<[f64; 3]> = (0..HIST_BINS)
            .flat_map(|i| {
                (0..HIST_BINS).flat_map(move |j| {
                    (0..HIST_BINS).map(move |k| {
                        let at = |n: usize| f64::from(u32::try_from(n).expect("0..HIST_BINS"));
                        [15.0f64.mul_add(at(i), 25.0), 10.0f64.mul_add(at(j), -5.0), 10.0 * at(k)]
                    })
                })
            })
            .collect();
        let stats = ColourStats::of(&labs).expect("64 values");
        assert!(stats.hist.iter().all(|&n| n == 1), "one value per bin: {:?}", stats.hist);
        assert!(ColourStats::of(&[]).is_none(), "no values is no statistic");
    }

    /// The MAD is the median absolute deviation and not a standard deviation: one outlier in nine
    /// moves it by nothing.
    #[test]
    fn the_colour_mad_is_a_median_and_not_a_mean() {
        let mut labs: Vec<[f64; 3]> = (0..9).map(|k| [50.0 + f64::from(k % 3), 0.0, 0.0]).collect();
        let tight = ColourStats::of(&labs).expect("nine values");
        labs.push([500.0, 0.0, 0.0]);
        let with_outlier = ColourStats::of(&labs).expect("ten values");
        assert_relative_eq!(tight.lab_mad[0], 1.0);
        assert_relative_eq!(with_outlier.lab_mad[0], 1.0, epsilon = 1e-12);
        assert!(with_outlier.lab_mean[0] > tight.lab_mean[0] + 40.0, "the mean does move");
    }

    /// Points on a known sphere give back its centre and its radius; a plane gives nothing useful
    /// and says so through the residual rather than by pretending.
    #[test]
    fn the_sphere_fit_recovers_a_sphere() {
        let mut points = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                let (u, v) = (f64::from(i) * 0.2, f64::from(j) * 0.25);
                points.push([
                    7.0 + 5.0 * u.sin() * v.cos(),
                    -3.0 + 5.0 * u.sin() * v.sin(),
                    2.0 + 5.0 * u.cos(),
                ]);
            }
        }
        let (centre, radius) = sphere_fit(&points).expect("144 points on a sphere");
        assert_relative_eq!(radius, 5.0, epsilon = 1e-9);
        assert_relative_eq!(centre[0], 7.0, epsilon = 1e-9);
        assert_relative_eq!(centre[1], -3.0, epsilon = 1e-9);
        assert_relative_eq!(centre[2], 2.0, epsilon = 1e-9);
        assert!(sphere_fit(&points[..MIN_FIT_POINTS - 1]).is_none(), "too few points is a refusal");
    }

    /// A circle in the plane, and the same refusal below the minimum.
    #[test]
    fn the_circle_fit_recovers_a_circle() {
        let points: Vec<[f64; 3]> = (0..40)
            .map(|i| {
                let a = f64::from(i) * 0.15;
                [1.5 + 12.0 * a.cos(), -0.5 + 12.0 * a.sin(), 0.0]
            })
            .collect();
        let (centre, radius) = circle_fit(&points).expect("40 points on a circle");
        assert_relative_eq!(radius, 12.0, epsilon = 1e-9);
        assert_relative_eq!(centre[0], 1.5, epsilon = 1e-9);
        assert_relative_eq!(centre[1], -0.5, epsilon = 1e-9);
    }

    /// The plane fit returns the normal of the plane its points lie on, whichever way it is
    /// spanned.
    #[test]
    fn the_plane_fit_returns_the_planes_normal() {
        let points: Vec<[f64; 3]> =
            (0..6).flat_map(|i| (0..6).map(move |j| [f64::from(i), f64::from(j), 4.0])).collect();
        let (centre, normal) = plane_fit(&points).expect("a grid spans a plane");
        assert_relative_eq!(centre[2], 4.0, epsilon = 1e-12);
        assert_relative_eq!(normal[2].abs(), 1.0, epsilon = 1e-9);
    }

    /// A cylinder's normals have a zero-eigenvalue direction: the residual is 0 and the diameter
    /// is the cylinder's. An isotropic sphere of normals has residual 1/3 and no axis to speak of.
    #[test]
    fn the_axis_fit_finds_a_cylinder_and_reports_a_sphere_as_isotropic() {
        let (mut points, mut normals) = (Vec::new(), Vec::new());
        for i in 0..60 {
            for k in 0..4 {
                let a = f64::from(i) * 0.1;
                let (c, s) = (a.cos(), a.sin());
                points.push([9.0 * c, 9.0 * s, f64::from(k)]);
                normals.push([c, s, 0.0]);
            }
        }
        let (residual, diameter, rms) = axis_fit(&points, &normals, 1.0);
        assert!(residual.expect("a residual") < 1e-12, "{residual:?}");
        assert_relative_eq!(diameter.expect("a diameter"), 18.0, epsilon = 1e-6);
        assert!(rms.expect("an rms") < 1e-6);

        let mut iso_p = Vec::new();
        let mut iso_n = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let (u, v) = (f64::from(i) * 0.157, f64::from(j) * 0.314);
                let n = [u.sin() * v.cos(), u.sin() * v.sin(), u.cos()];
                iso_p.push([3.0 * n[0], 3.0 * n[1], 3.0 * n[2]]);
                iso_n.push(n);
            }
        }
        let (residual, _, _) = axis_fit(&iso_p, &iso_n, 1.0);
        assert!(residual.expect("a residual") > 0.2, "{residual:?}");
    }

    /// Roughness is a length in `t`: a plane is 0, and a surface with a known ripple reads the
    /// ripple. The ball is `0.5 t`, so `t` sets both the neighbourhood and the unit.
    #[test]
    fn roughness_is_zero_on_a_plane_and_positive_on_a_ripple() {
        let flat: Vec<[f64; 3]> = (0..40)
            .flat_map(|i| (0..40).map(move |j| [0.1 * f64::from(i), 0.1 * f64::from(j), 0.0]))
            .collect();
        let (rough, n) = fracture_roughness(&flat, 2.0);
        assert!(n > 1000, "every sample has a full ball");
        assert!(rough.expect("a plane has a roughness") < 1e-9);

        let ripple: Vec<[f64; 3]> = flat
            .iter()
            .map(|p| [p[0], p[1], 0.05 * (p[0] * 7.0).sin() * (p[1] * 7.0).cos()])
            .collect();
        let (rough, _) = fracture_roughness(&ripple, 2.0);
        assert!(rough.expect("a ripple has a roughness") > 1e-3, "{rough:?}");
    }

    /// Lab of the three colours whose values are standard: white, black and a mid grey.
    #[test]
    fn lab_puts_white_at_a_hundred_and_black_at_zero() {
        let white = srgb_to_lab([255, 255, 255]);
        assert_relative_eq!(white[0], 100.0, epsilon = 1e-9);
        assert_relative_eq!(white[1], 0.0, epsilon = 1e-9);
        assert_relative_eq!(white[2], 0.0, epsilon = 1e-9);
        assert_relative_eq!(srgb_to_lab([0, 0, 0])[0], 0.0, epsilon = 1e-12);
        let grey = srgb_to_lab([119, 119, 119]);
        assert!((49.0..51.0).contains(&grey[0]), "{grey:?}");
        assert_relative_eq!(grey[1], 0.0, epsilon = 1e-9);
    }

    /// A file with one constant colour reports a spread of zero and one distinct triple — which is
    /// exactly what both synthetic sets are (audit §C.4), and the reason colour has no AUC to
    /// measure on any collection that carries object ids.
    #[test]
    fn a_constant_colour_reports_no_spread() {
        let mesh = crate::Mesh {
            v: vec![[0.0; 3]; 4],
            f: vec![[0, 1, 2]],
            colors: Some(vec![[180, 120, 90]; 4]),
        };
        let f = Features { name: "x".into(), ..Features::default() }.with_colour(&mesh);
        assert_eq!(f.colour_points, 4);
        assert_eq!(f.colour_distinct, 1);
        assert_eq!(f.lab_spread, Some([0.0, 0.0, 0.0]));
        let mean = f.lab_mean.expect("a mean");
        assert!(mean[1] > 0.0 && mean[2] > 0.0, "a terracotta is warm in both opponent axes");

        let bare = crate::Mesh::new(vec![[0.0; 3]; 4], vec![[0, 1, 2]]);
        assert_eq!(Features::default().with_colour(&bare).lab_mean, None);
    }
}
