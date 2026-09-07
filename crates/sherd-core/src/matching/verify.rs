//! Verifying one candidate pose (R §6).
//!
//! The ICP ladder of R §5.6 hands over a pose and its own opinion of it — `fitness` and
//! `inlier_rmse`, both of which say only how many source points found a target inside the last
//! rung's radius. Neither is evidence that two sherds broke apart from each other. R §6 is: five
//! independent measurements of the placed pair, each answering a different way for a candidate to
//! be wrong.
//!
//! | score | what it asks | how a false pair fails it |
//! |---|---|---|
//! | `tight`, `gap`, `contact` (§6.1) | how close the two **fracture surfaces** actually are | two flat regions laid on each other touch on a band, not over the surface |
//! | `seam` (§6.2) | how much of the **breakline** the two share, in wall thicknesses | a coincidental fit shares a point, not a curve |
//! | `cont`, `cont_n` (§6.3) | does the **outer shell** continue across the join | a step, or a shell facing the other way |
//! | `pen`, `pen_depth` (§6.4) | does either fragment sit **inside** the other | the pose that maximises contact often does |
//! | [`accept`] (§6.5) | all five at once | any one of them |
//!
//! # Point-to-surface, not point-to-point
//!
//! §6.1 measures each fragment's fracture samples against the other's fracture **triangles**
//! ([`RayScene`]), never against its sample cloud. Two independent samples of one surface never
//! land on each other, so the point-to-point form had a floor equal to the sample spacing —
//! `≈ 0.5/√density`, which reached 0.075 t on a large pot A sherd against the terracotta's 0.043 t
//! — and that floor alone kept every true join of the thin-walled pots away from the gap
//! threshold. Against the triangles the only floor left is the mesh resolution, which
//! [`Scales`] already carries.
//!
//! The scene is the **fracture faces alone**
//! ([`Fragment::fracture_scene`](crate::fragment::Fragment::fracture_scene)): against the whole
//! mesh a fragment laid flat on its neighbour's outer shell would score perfect contact.
//!
//! # What each measurement costs, and how the port pays it
//!
//! Every score here is a query against a BVH or a KD-tree, and the thresholds are what make them
//! affordable. The fracture distance is only ever compared against something at or below
//! `sc.facing`, so it runs through [`RayScene::bounded_distance`] and a point whose surface is
//! further away costs a pruned traversal instead of a full one (E4: 2.0–3.3× faster than the
//! unbounded call, exact inside the window). The penetration test rejects on the mesh's AABB
//! before it casts a parity ray, and computes a distance only for the points that are actually
//! inside (D §6.5) — with the one exception R §6.4's `pen_depth` forces: when *nothing* is inside,
//! the deepest excursion is the smallest positive distance, and that is found by a running
//! minimum whose window shrinks with the best distance so far.

use nalgebra::Matrix4;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::fragment::samples::MatchData;
use crate::matching::coarse::NORMAL_AGREE;
use crate::matching::scales::Scales;
use crate::spatial::bvh::RayScene;
use crate::spatial::kdtree::PointTree;

/// R §6.1's minimum number of facing points: below this the fracture scores are the refusal
/// `tight = 0`, `gap = 1`, `contact = 0`.
///
/// Twenty samples of a fracture surface is nothing — a sliver of one triangle's worth — and a
/// median over fewer is noise rather than a gap.
pub const MIN_FACING: usize = 20;
/// R §6.3's minimum number of margin points near the seam, above which the continuity is measured.
pub const MIN_NEAR: usize = 20;
/// R §6.2's voxel, in wall thicknesses: the seam is counted in cubes of `t/3`.
pub const SEAM_VOXEL: f64 = 3.0;

/// Every number R §6 produces for one candidate, under the keys the report writes (R §6.5).
///
/// A struct rather than a map (D §4.1): the report serialises it to exactly the Python's JSON
/// keys, and the two optional ones — `partial` and `pen_unavailable` — appear only when they are
/// set, as flags worth `1.0`, which is how the reference writes them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    /// Fraction of **A**'s facing fracture samples within `sc.tight` of B's fracture surface.
    #[serde(rename = "tightA")]
    pub tight_a: f64,
    /// The same for B's samples against A's surface.
    #[serde(rename = "tightB")]
    pub tight_b: f64,
    /// `min(tightA, tightB)` — the side that fits worse is the fit.
    pub tight: f64,
    /// Median distance of A's facing samples to B's fracture surface, in `t`.
    #[serde(rename = "gapA")]
    pub gap_a: f64,
    /// The same for B.
    #[serde(rename = "gapB")]
    pub gap_b: f64,
    /// `max(gapA, gapB)`, in `t`.
    pub gap: f64,
    /// Area of A in contact with B, in `t²`.
    #[serde(rename = "contactA")]
    pub contact_a: f64,
    /// The same for B.
    #[serde(rename = "contactB")]
    pub contact_b: f64,
    /// `min(contactA, contactB)`, in `t²`.
    pub contact: f64,
    /// Length of the shared seam, in `t` (R §6.2).
    pub seam: f64,
    /// `sc.gap / t` — the pair's own gap limit, carried so the report can print it.
    pub gap_limit: f64,
    /// `sc.tight / t` — the distance `tight` counted, likewise.
    pub tight_delta: f64,
    /// Median step height of the outer shell across the seam, in `t` (R §6.3).
    pub cont: f64,
    /// Median agreement of the two shells' normals across the seam (R §6.3).
    pub cont_n: f64,
    /// Fraction of surface samples of either fragment inside the other (R §6.4).
    pub pen: f64,
    /// Deepest excursion of either fragment into the other, in `t` (R §6.4).
    pub pen_depth: f64,
    /// Set when a fragment is not watertight and the penetration test could not run.
    #[serde(
        default,
        skip_serializing_if = "is_not_set",
        serialize_with = "as_one",
        deserialize_with = "from_one"
    )]
    pub pen_unavailable: bool,
    /// Set when only the cheap half of R §6 was computed, which also makes [`accept`] refuse the
    /// candidate.
    #[serde(
        default,
        skip_serializing_if = "is_not_set",
        serialize_with = "as_one",
        deserialize_with = "from_one"
    )]
    pub partial: bool,
    /// R §5.4's re-score of the stage-1 pose this candidate was refined from.
    #[serde(default)]
    pub brk: f64,
    /// The best stage-1 re-score of the whole pair (R §5.7).
    #[serde(default)]
    pub brk_best: f64,
}

impl Default for Scores {
    /// The refusal: every score at the value that fails [`accept`].
    ///
    /// This is R §5.4's partial candidate with its two limits left at zero — `_partial_scores`
    /// fills those in from the pair's own [`Scales`], which [`Scores::partial`] does here.
    fn default() -> Self {
        Self {
            tight_a: 0.0,
            tight_b: 0.0,
            tight: 0.0,
            gap_a: 1.0,
            gap_b: 1.0,
            gap: 1.0,
            contact_a: 0.0,
            contact_b: 0.0,
            contact: 0.0,
            seam: 0.0,
            gap_limit: 0.0,
            tight_delta: 0.0,
            cont: 1.0,
            cont_n: -1.0,
            pen: 0.0,
            pen_depth: 0.0,
            pen_unavailable: false,
            partial: false,
            brk: 0.0,
            brk_best: 0.0,
        }
    }
}

impl Scores {
    /// R §5.7's ranking key: `seam · tight`.
    ///
    /// A product rather than a sum, because a candidate needs *both*: a long seam with no contact
    /// is two curves that happen to run together, and tight contact along nothing is a corner
    /// resting on a surface.
    #[inline]
    pub fn score(&self) -> f64 {
        self.seam * self.tight
    }

    /// R §5.4's `_partial_scores`: the refusal, with the pair's two limits filled in and `brk`
    /// set, for a candidate cut before stage 2.
    ///
    /// Every field the report prints is present — a partial candidate still appears in it — and
    /// the ones that were never computed are at the values [`accept`] refuses.
    pub fn partial(sc: &Scales, brk: f64) -> Self {
        Self { partial: true, brk, brk_best: brk, ..Self::default().with_limits(sc) }
    }

    /// The two `Scales`-derived fields R §6 reports beside the scores (`Scales::limits`).
    #[must_use]
    fn with_limits(mut self, sc: &Scales) -> Self {
        self.gap_limit = sc.gap / sc.t;
        self.tight_delta = sc.tight / sc.t;
        self
    }
}

/// `1.0` when the flag is set — the reference writes both optional scores as numbers.
#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde's `serialize_with` signature")]
fn as_one<S: Serializer>(flag: &bool, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(f64::from(u8::from(*flag)))
}

/// The reader of [`as_one`]: any non-zero number is the flag.
fn from_one<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    Ok(f64::deserialize(deserializer)? != 0.0)
}

/// Whether an optional flag is left out of the JSON.
#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde's `skip_serializing_if` signature")]
fn is_not_set(flag: &bool) -> bool {
    !*flag
}

/// One fragment as R §6 measures it: two BVHs, four point sets and the margin's tree.
///
/// Built once per pair and read once per candidate — a stage-2 pair verifies ten poses against
/// exactly these arrays, and neither the BVHs (19–41 ms each) nor the KD-tree may be rebuilt for
/// each of them.
///
/// The points are `f64` because every score is computed in `f64`; the meshes behind the two scenes
/// are `f32`, which is what Open3D's `RaycastingScene` is too (PMC-15, R §0).
#[derive(Debug)]
pub struct Surfaces<'a> {
    /// A BVH over this fragment's fracture faces (R §6.1).
    pub fracture: &'a RayScene,
    /// A BVH over its whole working mesh (R §6.4); `None` leaves the penetration test unavailable.
    pub mesh: Option<&'a RayScene>,
    /// R §3.3.2's verdict: whether a signed distance against `mesh` can be trusted.
    pub watertight: bool,
    /// R §3.4's fracture area, which `contact` scales by.
    pub frac_area: f64,
    /// `Pf`: the fracture samples.
    pub pf: Vec<[f64; 3]>,
    /// `S`: the whole-surface samples the penetration test casts from.
    pub s: Vec<[f64; 3]>,
    /// `brk_P`: every breakline point (R §6.2 uses the whole curve, not the subset).
    pub brk_p: Vec<[f64; 3]>,
    /// `brk_ns`: the shell macro-normal at each of them.
    pub brk_ns: Vec<[f64; 3]>,
    /// `Pm`: the shell-margin points (R §6.3).
    pub margin_p: Vec<[f64; 3]>,
    /// `Nm`: the shell normal at each of them.
    pub margin_n: Vec<[f64; 3]>,
    /// A KD-tree over `margin_p`, which R §6.3 queries from the other fragment's margin.
    pub margin_tree: Option<PointTree>,
}

impl<'a> Surfaces<'a> {
    /// The view a pair takes of one of its two [`MatchData`] (R §3.6, R §4.2).
    ///
    /// `None` when the fragment's working mesh has no triangle, which is the one case
    /// [`RayScene`] refuses; nothing downstream can score such a fragment either.
    pub fn of(md: &MatchData<'a>) -> Option<Self> {
        let fragment = md.fragment;
        Some(Self::new(
            fragment.fracture_scene()?,
            fragment.surface_scene(),
            fragment.watertight,
            md.frac_area,
            md.samples.fracture_f64(),
            md.samples.surface_f64(),
            md.brk.points_f64(),
            md.brk.ns.iter().map(|n| n.to_f64()).collect(),
            md.margin.p.iter().map(|p| p.to_f64()).collect(),
            md.margin.n.iter().map(|n| n.to_f64()).collect(),
        ))
    }

    /// A side from arrays the caller already holds — the parity harness's entry point, which
    /// feeds R §6 the *reference's* own samples and mesh (D §10.2's injected column).
    #[allow(clippy::too_many_arguments, reason = "R §6 reads exactly these ten things")]
    pub fn new(
        fracture: &'a RayScene,
        mesh: Option<&'a RayScene>,
        watertight: bool,
        frac_area: f64,
        pf: Vec<[f64; 3]>,
        s: Vec<[f64; 3]>,
        brk_p: Vec<[f64; 3]>,
        brk_ns: Vec<[f64; 3]>,
        margin_p: Vec<[f64; 3]>,
        margin_n: Vec<[f64; 3]>,
    ) -> Self {
        let margin_tree = PointTree::build(&margin_p);
        Self {
            fracture,
            mesh,
            watertight,
            frac_area,
            pf,
            s,
            brk_p,
            brk_ns,
            margin_p,
            margin_n,
            margin_tree,
        }
    }
}

/// R §6.1's nine numbers, kept apart from the rest because R §5.6's early rejection computes them
/// on their own and then hands them to [`verify`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FractureScores {
    /// `tightA`, `tightB`, `tight`.
    pub tight: [f64; 3],
    /// `gapA`, `gapB`, `gap`, in `t`.
    pub gap: [f64; 3],
    /// `contactA`, `contactB`, `contact`, in `t²`.
    pub contact: [f64; 3],
}

/// R §6.1: the tight-contact fraction, the median gap and the contact area, both ways round.
///
/// `d1` is B's fracture samples measured against A's fracture surface and `d2` is A's against B's;
/// the `A` scores are `d2`'s (A's own points), which is the reference's pairing and the one that
/// makes `tightA` a statement about A.
pub fn fracture_scores(
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
) -> FractureScores {
    let inverse = pose_inverse(transform);
    let d1 = surface_distances(&b.pf, transform, a.fracture, sc.facing);
    let d2 = surface_distances(&a.pf, &inverse, b.fracture, sc.facing);
    let (tight_a, gap_a, contact_a) = one_side(&d2, a.frac_area, sc);
    let (tight_b, gap_b, contact_b) = one_side(&d1, b.frac_area, sc);
    FractureScores {
        tight: [tight_a, tight_b, tight_a.min(tight_b)],
        gap: [gap_a, gap_b, gap_a.max(gap_b)],
        contact: [contact_a, contact_b, contact_a.min(contact_b)],
    }
}

/// One fragment's three fracture scores from its own point-to-surface distances.
fn one_side(d: &[f64], area: f64, sc: &Scales) -> (f64, f64, f64) {
    let mut facing: Vec<f64> = d.iter().copied().filter(|&d| d < sc.facing).collect();
    if facing.len() < MIN_FACING {
        return (0.0, 1.0, 0.0);
    }
    #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
    let tight = facing.iter().filter(|&&d| d < sc.tight).count() as f64 / facing.len() as f64;
    let gap = median(&mut facing) / sc.t;
    #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
    let touching = d.iter().filter(|&&d| d < 2.0 * sc.tight).count() as f64 / d.len() as f64;
    (tight, gap, touching * area / (sc.t * sc.t))
}

/// The distance from each moved point to `scene`, or `+∞` beyond `max_dist` (R §6.1).
///
/// The window is `sc.facing`, and every use of the result is a comparison against a threshold at
/// or below it — `sc.tight`, `2·sc.tight` and `sc.facing` itself — so a point whose surface is
/// further away needs no number, only the knowledge that it has none.
fn surface_distances(
    points: &[[f64; 3]],
    transform: &Matrix4<f64>,
    scene: &RayScene,
    max_dist: f64,
) -> Vec<f64> {
    #[allow(clippy::cast_possible_truncation, reason = "the scene is f32, as Open3D's is")]
    let window = max_dist as f32;
    points
        .iter()
        .map(|p| {
            scene
                .bounded_distance(narrow(apply(transform, *p)), window)
                .map_or(f64::INFINITY, f64::from)
        })
        .collect()
}

/// R §6.2: the length of A's breakline, in `t`, covered by B's with agreeing shell normals.
///
/// The tree is over the **moved** B breakline and is queried from A's, which is the direction that
/// makes the answer a length of *A*'s curve: the voxels counted are A's own points, in a grid
/// fixed to the world origin, three to a wall thickness. A candidate that lays a corner of B on a
/// point of A shares one voxel; a join shares a run of them.
pub fn seam_score(
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
) -> f64 {
    let moved: Vec<[f64; 3]> = b.brk_p.iter().map(|p| apply(transform, *p)).collect();
    let normals: Vec<[f64; 3]> = b.brk_ns.iter().map(|n| rotate(transform, *n)).collect();
    let Some(tree) = PointTree::build(&moved) else { return 0.0 };
    let voxel = sc.t / SEAM_VOXEL;
    let mut cells: Vec<[i64; 3]> = Vec::new();
    for (point, normal) in a.brk_p.iter().zip(&a.brk_ns) {
        // R §6.2 is `dA, jA = cKDTree(...).query(A.brk_P)` — unbounded — and then `dA < sc.seam`.
        // `nearest_below` is that pair of steps, computed through a bounded traversal that is
        // provably the same answer, ties and radius boundary included (see its documentation).
        let Some((j, _)) = tree.nearest_below(point, sc.seam) else { continue };
        if dot(*normal, normals[j as usize]) > NORMAL_AGREE {
            cells.push(cell(*point, voxel));
        }
    }
    if cells.is_empty() {
        return 0.0;
    }
    cells.sort_unstable();
    cells.dedup();
    #[allow(clippy::cast_precision_loss, reason = "a breakline is thousands of points")]
    let count = cells.len() as f64;
    count / SEAM_VOXEL
}

/// R §6.3: the median step height of the outer shell across the seam, in `t`, and the median
/// agreement of the two shells' normals.
///
/// `(1.0, −1.0)` — a step of a whole wall and shells facing opposite ways — when either fragment
/// has no margin or fewer than [`MIN_NEAR`] of B's margin points land near A's. That is the
/// refusal, and [`accept`] reads it as one.
pub fn continuity_scores(
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
) -> (f64, f64) {
    let (Some(tree), false) = (a.margin_tree.as_ref(), b.margin_p.is_empty()) else {
        return (1.0, -1.0);
    };
    let mut steps = Vec::new();
    let mut agreements = Vec::new();
    for (point, normal) in b.margin_p.iter().zip(&b.margin_n) {
        let moved = apply(transform, *point);
        // R §6.3's `dm, jm = A.tree_margin.query(PBm)` — unbounded — and then `dm < sc.near`.
        let Some((j, _)) = tree.nearest_below(&moved, sc.near) else { continue };
        let (near_point, near_normal) = (a.margin_p[j as usize], a.margin_n[j as usize]);
        let delta = [moved[0] - near_point[0], moved[1] - near_point[1], moved[2] - near_point[2]];
        steps.push(dot(delta, near_normal).abs());
        agreements.push(dot(rotate(transform, *normal), near_normal));
    }
    if steps.len() <= MIN_NEAR {
        return (1.0, -1.0);
    }
    (median(&mut steps) / sc.t, median(&mut agreements))
}

/// R §6.4: the fraction of either fragment's surface samples inside the other, and the deepest
/// excursion in `t`.
///
/// Both fragments must be watertight for the question to have an answer; when one is not, the
/// scores are `0` and `pen_unavailable` is set, which is the reference's own refusal and is
/// *not* a pass — R §6.5 reads `pen = 0` as "no penetration found", and a fragment with holes
/// simply never contributes one.
pub fn penetration_scores(
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
) -> (f64, f64, bool) {
    let (Some(mesh_a), Some(mesh_b)) = (a.mesh, b.mesh) else { return (0.0, 0.0, true) };
    if !(a.watertight && b.watertight) {
        return (0.0, 0.0, true);
    }
    let inverse = pose_inverse(transform);
    let (pen_b, min_b) = one_penetration(&b.s, transform, mesh_a, sc.pen);
    let (pen_a, min_a) = one_penetration(&a.s, &inverse, mesh_b, sc.pen);
    (pen_a.max(pen_b), (-min_a).max(-min_b) / sc.t, false)
}

/// One direction of R §6.4: the moved points against one mesh, as `(fraction deeper than `pen`,
/// smallest signed distance)`.
///
/// The signed distance is Open3D's — the closest-point distance with the sign of the ray parity —
/// and this computes exactly the two statistics R §6.4 reads from it, not the array itself. Three
/// prunings do that cheaply and change no answer:
///
/// * a point outside the mesh's AABB is outside the mesh, with no ray cast at all;
/// * a point outside the mesh needs no distance, because its signed distance is positive and both
///   statistics are about the negative ones;
/// * when *nothing* is inside, the smallest signed distance is the smallest positive distance, and
///   a running minimum finds it with a window that shrinks as it goes.
fn one_penetration(
    points: &[[f64; 3]],
    transform: &Matrix4<f64>,
    scene: &RayScene,
    pen: f64,
) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }
    let (lo, hi) = scene.aabb();
    let moved: Vec<[f32; 3]> = points.iter().map(|p| narrow(apply(transform, *p))).collect();
    let mut deepest = f64::NEG_INFINITY;
    let mut deeper_than_pen = 0_usize;
    for point in &moved {
        let outside_box = (0..3).any(|k| point[k] < lo[k] || point[k] > hi[k]);
        if outside_box || !scene.inside(*point) {
            continue;
        }
        let distance = f64::from(scene.distance(*point));
        deepest = deepest.max(distance);
        if distance > pen {
            deeper_than_pen += 1;
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "sample counts are far below 2^53")]
    let fraction = deeper_than_pen as f64 / points.len() as f64;
    if deepest > f64::NEG_INFINITY {
        return (fraction, -deepest);
    }
    // Nothing is inside: R §6.4's `min(sd)` is the closest the two surfaces come, and each query
    // only has to beat the best distance so far.
    let mut best = f32::MAX;
    for point in &moved {
        if let Some(distance) = scene.bounded_distance(*point, best) {
            best = best.min(distance);
        }
    }
    (fraction, f64::from(best))
}

/// R §6: every score of one pose.
///
/// `full = false` computes only the cheap half — the fracture contact and the seam — and marks the
/// rest as not computed (`cont = 1`, `cont_n = −1`, `pen = 0`, `partial = 1`), which also makes
/// [`accept`] refuse the candidate. That is R §5.6's early rejection, off by default.
///
/// `frac` passes in fracture scores already computed for this very pose, which is the only reason
/// the early rejection is cheaper than the thing it replaces.
pub fn verify(
    a: &Surfaces<'_>,
    b: &Surfaces<'_>,
    transform: &Matrix4<f64>,
    sc: &Scales,
    full: bool,
    frac: Option<FractureScores>,
) -> Scores {
    let frac = frac.unwrap_or_else(|| fracture_scores(a, b, transform, sc));
    let mut scores = Scores {
        tight_a: frac.tight[0],
        tight_b: frac.tight[1],
        tight: frac.tight[2],
        gap_a: frac.gap[0],
        gap_b: frac.gap[1],
        gap: frac.gap[2],
        contact_a: frac.contact[0],
        contact_b: frac.contact[1],
        contact: frac.contact[2],
        seam: seam_score(a, b, transform, sc),
        ..Scores::default()
    }
    .with_limits(sc);
    if !full {
        scores.partial = true;
        return scores;
    }
    let (cont, cont_n) = continuity_scores(a, b, transform, sc);
    scores.cont = cont;
    scores.cont_n = cont_n;
    let (pen, pen_depth, unavailable) = penetration_scores(a, b, transform, sc);
    scores.pen = pen;
    scores.pen_depth = pen_depth;
    scores.pen_unavailable = unavailable;
    scores
}

/// R §6.5: all five limits at once.
///
/// `gap` is reported in units of `t`, so it is compared against the pair's own limit rather than
/// against `p.max_gap` — the two differ whenever the resolution floor of R §1.2 binds.
pub fn accept(s: &Scores, p: &crate::params::Params, sc: &Scales) -> bool {
    s.tight >= p.min_tight
        && s.gap * sc.t <= sc.gap
        && s.pen <= p.max_pen
        && s.seam >= p.min_seam
        && s.cont_n >= p.min_cont_n
}

/// numpy's `median`: the middle of an odd sample, the mean of the two middle ones of an even one.
///
/// The slice is sorted in place, which is what makes this take `&mut`; every caller owns its copy.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { values[n / 2] } else { f64::midpoint(values[n / 2 - 1], values[n / 2]) }
}

/// `R·p + τ` — the reference's `apply_transform` for one point.
#[inline]
fn apply(t: &Matrix4<f64>, p: [f64; 3]) -> [f64; 3] {
    [
        t[(0, 0)] * p[0] + t[(0, 1)] * p[1] + t[(0, 2)] * p[2] + t[(0, 3)],
        t[(1, 0)] * p[0] + t[(1, 1)] * p[1] + t[(1, 2)] * p[2] + t[(1, 3)],
        t[(2, 0)] * p[0] + t[(2, 1)] * p[1] + t[(2, 2)] * p[2] + t[(2, 3)],
    ]
}

/// `R·n` — the reference's `n @ T[:3, :3].T`, which is not renormalised.
#[inline]
fn rotate(t: &Matrix4<f64>, n: [f64; 3]) -> [f64; 3] {
    [
        t[(0, 0)] * n[0] + t[(0, 1)] * n[1] + t[(0, 2)] * n[2],
        t[(1, 0)] * n[0] + t[(1, 1)] * n[1] + t[(1, 2)] * n[2],
        t[(2, 0)] * n[0] + t[(2, 1)] * n[1] + t[(2, 2)] * n[2],
    ]
}

/// `np.linalg.inv(T)`: the inverse of the full 4×4 by LU with partial pivoting.
///
/// R §6.1 and R §6.4 both move A's points *backwards* through the pose, and the reference writes
/// that as `np.linalg.inv(T)` — LAPACK's `dgetrf` + `dgetri`, a factorisation of the whole matrix.
/// The port used to substitute `[Rᵀ | −Rᵀτ]`, which is the inverse of the *rotation* rather than of
/// the matrix, and the substitution is not free: a pose that has climbed two ICP ladders is
/// orthonormal only to about 2.6e-14, so `Rᵀ` is not `R⁻¹` at that level, and the error is then
/// multiplied by the point's distance from the origin — up to 885 units on these scans.
///
/// Measured over the 2 239 stage-2 poses of the eight fixture dumps and the clouds they are applied
/// to (`notes/2026-09-07-x-phase1c-findings.md` §3): the transpose puts a point **4.6e-13 t** from
/// where `np.linalg.inv` puts it at the median, 2.6e-12 t at p99 and 6.2e-11 t at the worst, while
/// this factorisation is at **2.7e-14 t / 1.5e-13 t / 1.2e-11 t** — seventeen times closer at the
/// median and at p99, on the same poses.
///
/// What is left is LAPACK's kernel against this loop, not an algorithm difference: both tails
/// belong to candidates whose ICP diverged and whose `‖τ‖` runs to 1.8e5 units, where the 4×4's
/// condition number reaches 3.3e10 and any two implementations of the same factorisation part
/// company. The loop is written out rather than delegated so that it is the same arithmetic on
/// every machine (D §7) — nalgebra's `Matrix4::try_inverse` is a cofactor expansion, which is a
/// different algorithm from the reference's.
fn pose_inverse(t: &Matrix4<f64>) -> Matrix4<f64> {
    let mut m = [[0.0_f64; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = t[(i, j)];
        }
    }
    // `dgetf2`: partial pivoting on the largest remaining `|column|`, first index on a tie, then
    // the rank-1 update of the trailing block. `piv[k]` is the original row now sitting at `k`.
    let mut piv = [0_usize, 1, 2, 3];
    for k in 0..4 {
        let mut best = k;
        for i in k + 1..4 {
            if m[i][k].abs() > m[best][k].abs() {
                best = i;
            }
        }
        if best != k {
            m.swap(k, best);
            piv.swap(k, best);
        }
        let pivot = m[k][k];
        if pivot == 0.0 {
            continue;
        }
        let (top, rest) = m.split_at_mut(k + 1);
        let row_k = &top[k];
        for row in rest {
            row[k] /= pivot;
            let factor = row[k];
            for c in k + 1..4 {
                row[c] -= factor * row_k[c];
            }
        }
    }
    // `dgetri`: one forward and one back substitution per column of `P·I`.
    let mut out = Matrix4::zeros();
    for col in 0..4 {
        let mut y = [0.0_f64; 4];
        for i in 0..4 {
            let mut acc = 0.0;
            for j in 0..i {
                acc += m[i][j] * y[j];
            }
            y[i] = f64::from(piv[i] == col) - acc;
        }
        let mut x = [0.0_f64; 4];
        for i in (0..4).rev() {
            let mut acc = 0.0;
            for j in i + 1..4 {
                acc += m[i][j] * x[j];
            }
            x[i] = (y[i] - acc) / m[i][i];
        }
        for (r, value) in x.into_iter().enumerate() {
            out[(r, col)] = value;
        }
    }
    out
}

/// The dot product of two `f64` triples, in the reference's `einsum` order.
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A point narrowed to the `f32` the BVH is built in — Open3D narrows its queries the same way.
#[inline]
#[allow(clippy::cast_possible_truncation, reason = "the scene is f32, as Open3D's is")]
fn narrow(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

/// R §6.2's voxel index of a point: `⌊p / voxel⌋` per axis, on a grid fixed to the world origin.
#[inline]
#[allow(clippy::cast_possible_truncation, reason = "a mesh coordinate over t/3 fits an i64")]
fn cell(p: [f64; 3], voxel: f64) -> [i64; 3] {
    [(p[0] / voxel).floor() as i64, (p[1] / voxel).floor() as i64, (p[2] / voxel).floor() as i64]
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "the test geometry is exact: 0, 1 and 4 are answers, not fits")]
mod tests {
    use super::{
        FractureScores, Scores, Surfaces, accept, continuity_scores, fracture_scores, median,
        penetration_scores, pose_inverse, seam_score, verify,
    };
    use crate::matching::scales::Scales;
    use crate::params::Params;
    use crate::spatial::bvh::RayScene;
    use crate::vec3::{Vec3f, vec3};
    use nalgebra::Matrix4;

    /// A square in the plane `x = 0`, spanning `[0, 2]²` in `y` and `z`, as two triangles.
    fn wall() -> (Vec<Vec3f>, Vec<[u32; 3]>) {
        let v = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 2.0, 0.0),
            vec3(0.0, 2.0, 2.0),
            vec3(0.0, 0.0, 2.0),
        ];
        (v, vec![[0, 1, 2], [0, 2, 3]])
    }

    /// An axis-aligned closed box, outward normals, twelve triangles.
    fn box_mesh(lo: [f32; 3], hi: [f32; 3]) -> (Vec<Vec3f>, Vec<[u32; 3]>) {
        let v = vec![
            vec3(lo[0], lo[1], lo[2]),
            vec3(hi[0], lo[1], lo[2]),
            vec3(hi[0], hi[1], lo[2]),
            vec3(lo[0], hi[1], lo[2]),
            vec3(lo[0], lo[1], hi[2]),
            vec3(hi[0], lo[1], hi[2]),
            vec3(hi[0], hi[1], hi[2]),
            vec3(lo[0], hi[1], hi[2]),
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    /// A 10 × 10 grid of points on the wall above.
    fn wall_samples() -> Vec<[f64; 3]> {
        (0..10)
            .flat_map(|i| {
                (0..10).map(move |j| [0.0, 0.2 + f64::from(i) * 0.2, 0.2 + f64::from(j) * 0.2])
            })
            .collect()
    }

    /// A translation along `x`.
    fn shifted(dx: f64) -> Matrix4<f64> {
        let mut t = Matrix4::identity();
        t[(0, 3)] = dx;
        t
    }

    /// A side with nothing but a fracture scene and its samples — everything R §6.1 reads and
    /// nothing else.
    fn fracture_side(scene: &RayScene, pf: Vec<[f64; 3]>, area: f64) -> Surfaces<'_> {
        Surfaces::new(
            scene,
            None,
            false,
            area,
            pf,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// R §6.1 on two coincident walls: perfect contact, no gap, the whole area in contact — and
    /// each of the three thresholds moves the answer where it should.
    #[test]
    fn the_fracture_scores_measure_two_surfaces_against_each_other() {
        let (vertices, faces) = wall();
        let scene = RayScene::of_mesh(&vertices, &faces).expect("two triangles");
        let side_a = fracture_side(&scene, wall_samples(), 4.0);
        let side_b = fracture_side(&scene, wall_samples(), 4.0);
        // t = 1, res = 0: tight = 0.01, facing = 0.3, gap limit = 0.03.
        let sc = Scales::for_pair(&Params::default(), 1.0, 0.0);

        let s = fracture_scores(&side_a, &side_b, &Matrix4::identity(), &sc);
        assert!(s.tight.iter().all(|&t| (t - 1.0).abs() < 1e-12), "{s:?}");
        assert!(s.gap.iter().all(|&g| g < 1e-6), "{s:?}");
        // `contact` is the fraction in contact times the fracture area over `t²`.
        assert!(s.contact.iter().all(|&c| (c - 4.0).abs() < 1e-9), "{s:?}");

        // Inside `tight` (0.01 t): still perfect contact, and the gap is the offset itself.
        let s = fracture_scores(&side_a, &side_b, &shifted(0.005), &sc);
        assert!((s.tight[2] - 1.0).abs() < 1e-12);
        assert!((s.gap[2] - 0.005).abs() < 1e-6, "{:?}", s.gap);

        // Past `2·tight`: no tight contact and no contact area, but still facing, so the gap is
        // measured rather than refused.
        let s = fracture_scores(&side_a, &side_b, &shifted(0.05), &sc);
        assert_eq!(s.tight[2], 0.0);
        assert_eq!(s.contact[2], 0.0);
        assert!((s.gap[2] - 0.05).abs() < 1e-6, "{:?}", s.gap);

        // Past `facing` (0.3 t): fewer than twenty facing points, and R §6.1's refusal.
        let s = fracture_scores(&side_a, &side_b, &shifted(0.5), &sc);
        assert_eq!((s.tight[2], s.gap[2], s.contact[2]), (0.0, 1.0, 0.0));
    }

    /// The `A` scores are A's own points against B's surface: a fragment whose samples all fit is
    /// not saved by a partner whose samples do not.
    #[test]
    fn the_two_sides_of_the_fracture_score_are_not_the_same_number() {
        let (vertices, faces) = wall();
        let scene = RayScene::of_mesh(&vertices, &faces).expect("two triangles");
        // A's samples sit on the wall; B's sit 0.05 off it, so B's own distances are 0.05 and A's
        // are 0 — the scores must land on opposite sides.
        let off: Vec<[f64; 3]> = wall_samples().iter().map(|p| [0.05, p[1], p[2]]).collect();
        let side_a = fracture_side(&scene, wall_samples(), 4.0);
        let side_b = fracture_side(&scene, off, 4.0);
        let sc = Scales::for_pair(&Params::default(), 1.0, 0.0);

        let s = fracture_scores(&side_a, &side_b, &Matrix4::identity(), &sc);
        assert!((s.tight[0] - 1.0).abs() < 1e-12, "A's samples are on B's wall: {s:?}");
        assert_eq!(s.tight[1], 0.0, "B's samples are 0.05 t off A's wall: {s:?}");
        assert_eq!(s.tight[2], 0.0, "and `tight` is the worse of the two");
        assert!((s.gap[2] - 0.05).abs() < 1e-6, "`gap` is the larger of the two: {:?}", s.gap);
    }

    /// R §6.2 counts `t/3` voxels of A's own breakline that B's covers with an agreeing normal.
    #[test]
    fn the_seam_counts_voxels_of_the_shared_curve() {
        let curve: Vec<[f64; 3]> = (0..30).map(|k| [0.0, f64::from(k) * 0.1, 0.0]).collect();
        let up = vec![[0.0, 0.0, 1.0]; 30];
        let side = |p: Vec<[f64; 3]>, n: Vec<[f64; 3]>| {
            let (v, f) = wall();
            (v, f, p, n)
        };
        let (vertices, faces) = wall();
        let scene = RayScene::of_mesh(&vertices, &faces).expect("two triangles");
        let make = |p: &Vec<[f64; 3]>, n: &Vec<[f64; 3]>| {
            Surfaces::new(
                &scene,
                None,
                false,
                0.0,
                Vec::new(),
                Vec::new(),
                p.clone(),
                n.clone(),
                Vec::new(),
                Vec::new(),
            )
        };
        let _ = side(curve.clone(), up.clone());
        let a = make(&curve, &up);
        let b = make(&curve, &up);
        // t = 3: the voxel is 1.0 and the curve spans y ∈ [0, 2.9] — three voxels, so `seam = 1`.
        let sc = Scales::for_pair(&Params::default(), 3.0, 0.0);
        assert!((seam_score(&a, &b, &Matrix4::identity(), &sc) - 1.0).abs() < 1e-12);

        // Beyond `sc.seam` (0.12 t = 0.36) nothing is covered.
        assert_eq!(seam_score(&a, &b, &shifted(0.5), &sc), 0.0);

        // Inside it but with the shells facing opposite ways: also nothing. That is what makes the
        // seam a seam rather than a proximity.
        let down = vec![[0.0, 0.0, -1.0]; 30];
        let flipped = make(&curve, &down);
        assert_eq!(seam_score(&a, &flipped, &Matrix4::identity(), &sc), 0.0);

        // A longer curve is a longer seam: twice the points over twice the span, twice the voxels.
        let long: Vec<[f64; 3]> = (0..60).map(|k| [0.0, f64::from(k) * 0.1, 0.0]).collect();
        let long_up = vec![[0.0, 0.0, 1.0]; 60];
        let a2 = make(&long, &long_up);
        let b2 = make(&long, &long_up);
        assert!((seam_score(&a2, &b2, &Matrix4::identity(), &sc) - 2.0).abs() < 1e-12);
    }

    /// R §6.3 is the step across the seam and the agreement of the two shells' normals.
    #[test]
    fn the_continuity_measures_the_step_and_the_normal_agreement() {
        let (vertices, faces) = wall();
        let scene = RayScene::of_mesh(&vertices, &faces).expect("two triangles");
        let grid: Vec<[f64; 3]> = (0..5)
            .flat_map(|i| (0..5).map(move |j| [f64::from(i) * 0.1, f64::from(j) * 0.1, 0.0]))
            .collect();
        let up = vec![[0.0, 0.0, 1.0]; 25];
        let margin = |p: Vec<[f64; 3]>, n: Vec<[f64; 3]>| {
            Surfaces::new(
                &scene,
                None,
                false,
                0.0,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                p,
                n,
            )
        };
        let a = margin(grid.clone(), up.clone());
        let b = margin(grid.clone(), up.clone());
        let sc = Scales::for_pair(&Params::default(), 1.0, 0.0);

        // Coincident margins: no step, perfect agreement.
        let (cont, cont_n) = continuity_scores(&a, &b, &Matrix4::identity(), &sc);
        assert!(cont.abs() < 1e-12 && (cont_n - 1.0).abs() < 1e-12);

        // B's shell raised by 0.05 t along its own normal: that is the step, and the normals still
        // agree — a smooth wall at the wrong height.
        let mut step = Matrix4::identity();
        step[(2, 3)] = 0.05;
        let (cont, cont_n) = continuity_scores(&a, &b, &step, &sc);
        assert!((cont - 0.05).abs() < 1e-12, "{cont}");
        assert!((cont_n - 1.0).abs() < 1e-12);

        // B turned over: the step is still nothing and the agreement is −1, which R §6.5 refuses.
        let mut flip = Matrix4::identity();
        flip[(1, 1)] = -1.0;
        flip[(2, 2)] = -1.0;
        let (_, cont_n) = continuity_scores(&a, &b, &flip, &sc);
        assert!((cont_n + 1.0).abs() < 1e-12, "{cont_n}");

        // Fewer than twenty margin points near the seam is the refusal, whatever they say.
        let few = margin(grid[..10].to_vec(), up[..10].to_vec());
        assert_eq!(continuity_scores(&a, &few, &Matrix4::identity(), &sc), (1.0, -1.0));
        // And so is no margin at all on either side.
        let none = margin(Vec::new(), Vec::new());
        assert_eq!(continuity_scores(&a, &none, &Matrix4::identity(), &sc), (1.0, -1.0));
        assert_eq!(continuity_scores(&none, &a, &Matrix4::identity(), &sc), (1.0, -1.0));
    }

    /// R §6.4 counts the surface samples of either fragment that sit inside the other, and the
    /// deepest of them.
    #[test]
    fn the_penetration_finds_what_is_inside_the_other_fragment() {
        fn build<'a>(
            fracture: &'a RayScene,
            mesh: &'a RayScene,
            s: Vec<[f64; 3]>,
            watertight: bool,
        ) -> Surfaces<'a> {
            Surfaces::new(
                fracture,
                Some(mesh),
                watertight,
                0.0,
                Vec::new(),
                s,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }
        let (va, fa) = box_mesh([0.0; 3], [10.0; 3]);
        let (vb, fb) = box_mesh([20.0; 3], [30.0; 3]);
        let mesh_a = RayScene::of_mesh(&va, &fa).expect("twelve triangles");
        let mesh_b = RayScene::of_mesh(&vb, &fb).expect("twelve triangles");
        let (vw, fw) = wall();
        let fracture = RayScene::of_mesh(&vw, &fw).expect("two triangles");
        let sc = Scales::for_pair(&Params::default(), 1.0, 0.0);

        // Two of B's four samples land inside A; none of A's land inside B.
        let inside_a = vec![[4.0, 5.0, 6.0], [1.0, 5.5, 6.5], [50.0, 5.0, 5.0], [-5.0, 5.0, 5.0]];
        let outside_b = vec![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
        let a = build(&fracture, &mesh_a, outside_b, true);
        let b = build(&fracture, &mesh_b, inside_a, true);
        let (pen, depth, unavailable) = penetration_scores(&a, &b, &Matrix4::identity(), &sc);
        assert!(!unavailable);
        assert!((pen - 0.5).abs() < 1e-12, "two of four: {pen}");
        assert!((depth - 4.0).abs() < 1e-5, "the deepest sample is 4 from the wall: {depth}");

        // Why the vote is over *three* rays: a box's triangulation puts a diagonal across each
        // face, and a point whose ray hits exactly on one is counted twice or not at all. This
        // sample sits on the `z = 10` face's diagonal (`x = y`) and is still found inside, because
        // the other two rays are not degenerate. One ray — Open3D's own rule — would lose it.
        let diagonal = build(&fracture, &mesh_b, vec![[5.0, 5.0, 9.0]], true);
        let (pen, _, _) = penetration_scores(&a, &diagonal, &Matrix4::identity(), &sc);
        assert!((pen - 1.0).abs() < 1e-12, "the majority of three still finds it: {pen}");

        // With nothing inside either mesh, `pen` is zero and `pen_depth` is the negated distance
        // to the nearest surface — R §6.4's `−min(sd)` where every `sd` is positive.
        let far = build(&fracture, &mesh_b, vec![[-4.0, 5.0, 5.0], [-9.0, 5.0, 5.0]], true);
        let (pen, depth, _) = penetration_scores(&a, &far, &Matrix4::identity(), &sc);
        assert_eq!(pen, 0.0);
        assert!((depth + 4.0).abs() < 1e-5, "the closest sample is 4 outside: {depth}");

        // A fragment that is not watertight has no signed distance, and R §6.4 says so.
        let holed = build(&fracture, &mesh_b, vec![[5.0, 5.0, 5.0]], false);
        assert_eq!(penetration_scores(&a, &holed, &Matrix4::identity(), &sc), (0.0, 0.0, true));
    }

    /// R §6.5 refuses on any one of its five limits, and the two flags refuse on their own.
    #[test]
    fn the_acceptance_rule_refuses_on_any_one_limit() {
        let p = Params::default();
        let sc = Scales::for_pair(&p, 1.0, 0.0);
        // Exactly on every limit: `gap · t ≤ sc.gap` and the other four are `≥`, so this passes.
        let good = Scores {
            tight: p.min_tight,
            gap: sc.gap / sc.t,
            pen: p.max_pen,
            seam: p.min_seam,
            cont_n: p.min_cont_n,
            ..Scores::default()
        };
        assert!(accept(&good, &p, &sc));

        let step = 1e-9;
        assert!(!accept(&Scores { tight: p.min_tight - step, ..good }, &p, &sc));
        assert!(!accept(&Scores { gap: good.gap + step, ..good }, &p, &sc));
        assert!(!accept(&Scores { pen: p.max_pen + step, ..good }, &p, &sc));
        assert!(!accept(&Scores { seam: p.min_seam - step, ..good }, &p, &sc));
        assert!(!accept(&Scores { cont_n: p.min_cont_n - step, ..good }, &p, &sc));
        // The refusal `Scores::default` and R §5.4's partial candidate both fail it.
        assert!(!accept(&Scores::default(), &p, &sc));
        assert!(!accept(&Scores::partial(&sc, 0.9), &p, &sc));
    }

    /// R §5.4's partial candidate carries the pair's limits and its own `brk`, and R §5.7's key is
    /// the product of the two scores that matter.
    #[test]
    fn a_partial_candidate_is_the_refusal_with_the_pairs_limits_on_it() {
        let p = Params::default();
        let sc = Scales::for_pair(&p, 2.0, 0.0);
        let s = Scores::partial(&sc, 0.42);
        assert!(s.partial);
        assert_eq!((s.brk, s.brk_best), (0.42, 0.42));
        assert!((s.gap_limit - sc.gap / sc.t).abs() < 1e-15);
        assert!((s.tight_delta - sc.tight / sc.t).abs() < 1e-15);
        assert_eq!(s.score(), 0.0);
        assert_eq!(Scores { seam: 7.0, tight: 0.5, ..Scores::default() }.score(), 3.5);
    }

    /// The scores serialise under the reference's own JSON keys, and the two optional ones appear
    /// only when they are set — as the numbers the reference writes, not as booleans.
    #[test]
    fn the_scores_carry_the_pythons_json_keys() {
        let s = Scores { tight_a: 0.5, gap_b: 0.25, cont_n: 0.9, ..Scores::default() };
        let json = serde_json::to_value(s).expect("Scores serialises");
        let object = json.as_object().expect("an object");
        assert_eq!(object["tightA"], 0.5);
        assert_eq!(object["gapB"], 0.25);
        assert_eq!(object["cont_n"], 0.9);
        assert!(!object.contains_key("partial"), "an unset flag is left out");
        assert!(!object.contains_key("pen_unavailable"));
        assert_eq!(object.len(), 18, "R §6.5 lists eighteen keys plus the two optional ones");

        let flagged = Scores { partial: true, pen_unavailable: true, ..s };
        let json = serde_json::to_value(flagged).expect("Scores serialises");
        assert_eq!(json["partial"], 1.0);
        assert_eq!(json["pen_unavailable"], 1.0);
        let back: Scores = serde_json::from_value(json).expect("Scores deserialises");
        assert_eq!(back, flagged);

        // And the reference's own dump, which carries neither optional key nor `brk_best`, reads.
        let text = r#"{"brk": 0.2, "cont": 0.02, "cont_n": 0.99, "contact": 7.0,
            "contactA": 7.0, "contactB": 7.2, "gap": 0.001, "gapA": 0.0009, "gapB": 0.001,
            "gap_limit": 0.034, "pen": 0.0, "pen_depth": 0.011, "seam": 21.0, "tight": 0.84,
            "tightA": 0.9, "tightB": 0.84, "tight_delta": 0.011}"#;
        let s: Scores = serde_json::from_str(text).expect("the dump's own scores read");
        assert_eq!(s.seam, 21.0);
        assert_eq!(s.brk_best, 0.0, "the dump writes it later; the reader defaults it");
        assert!(!s.partial && !s.pen_unavailable);
    }

    /// [`verify`] is the five measurements together, and `full = false` is the half R §5.6's early
    /// rejection computes — with the rest at the values [`accept`] refuses.
    #[test]
    fn the_cheap_half_is_marked_and_cannot_be_accepted() {
        let (vertices, faces) = wall();
        let scene = RayScene::of_mesh(&vertices, &faces).expect("two triangles");
        let side_a = fracture_side(&scene, wall_samples(), 4.0);
        let side_b = fracture_side(&scene, wall_samples(), 4.0);
        let sc = Scales::for_pair(&Params::default(), 1.0, 0.0);

        let s = verify(&side_a, &side_b, &Matrix4::identity(), &sc, false, None);
        assert!(s.partial);
        assert!((s.tight - 1.0).abs() < 1e-12, "the cheap half is still measured");
        assert_eq!((s.cont, s.cont_n, s.pen, s.pen_depth), (1.0, -1.0, 0.0, 0.0));
        assert!(!accept(&s, &Params::default(), &sc));

        // Fracture scores computed once are not computed again: the same numbers come back.
        let frac = fracture_scores(&side_a, &side_b, &Matrix4::identity(), &sc);
        let reused = verify(&side_a, &side_b, &Matrix4::identity(), &sc, false, Some(frac));
        assert_eq!(reused, s);
        assert_eq!(
            FractureScores { tight: [1.0; 3], ..frac }.tight,
            [1.0; 3],
            "the struct is plain data"
        );

        // The full pass has no `partial` and, with no mesh on either side, no penetration either.
        let s = verify(&side_a, &side_b, &Matrix4::identity(), &sc, true, None);
        assert!(!s.partial && s.pen_unavailable);
    }

    /// [`pose_inverse`] is `np.linalg.inv` on a real stage-2 pose, to the last few bits.
    ///
    /// The matrix is the first stage-2 candidate of the terracotta pair `021__104`
    /// (`s2.T_frac2.npy`) and the expected inverse is what `np.linalg.inv` returns for it on the
    /// reference's own numpy — a pose with `‖τ‖ = 321` and a condition number of 1.0e5, which is
    /// an ordinary one for this stage rather than a constructed corner. The gate is 4 ULP of each
    /// entry's own magnitude: LAPACK's blocked kernels and the loop in [`pose_inverse`] are two
    /// implementations of one factorisation and they do not agree bit for bit.
    ///
    /// The transpose this function replaced misses the translation column of this very pose by
    /// 5.7e-13 units, where the factorisation is inside an ULP of numpy's own answer.
    #[test]
    fn the_inverse_is_numpys_on_a_real_stage_two_pose() {
        let t = Matrix4::<f64>::from_row_slice(&[
            -0.322_610_103_922_126_47,
            0.936_147_816_701_064,
            -0.139_821_264_953_462_98,
            -34.038_201_472_338_27,
            -0.888_398_054_254_096_7,
            -0.350_445_684_862_542_96,
            -0.296_541_260_465_994_7,
            -297.533_026_859_910_65,
            -0.326_606_212_501_986_46,
            0.028_549_732_871_869_956,
            0.944_729_217_664_011_6,
            114.448_926_720_524_33,
            0.0,
            0.0,
            0.0,
            1.0,
        ]);
        let want = Matrix4::<f64>::from_row_slice(&[
            -0.322_610_103_922_126_2,
            -0.888_398_054_254_094_1,
            -0.326_606_212_501_984_8,
            -237.929_099_371_881_42,
            0.936_147_816_701_063_6,
            -0.350_445_684_862_542_24,
            0.028_549_732_871_869_994,
            -75.671_863_659_729_35,
            -0.139_821_264_953_462_95,
            -0.296_541_260_465_994_44,
            0.944_729_217_664_008_5,
            -201.113_328_205_070_05,
            0.0,
            0.0,
            0.0,
            1.0,
        ]);
        let got = pose_inverse(&t);
        for i in 0..4 {
            for j in 0..4 {
                let bound = 4.0 * f64::EPSILON * want[(i, j)].abs().max(1.0);
                assert!(
                    (got[(i, j)] - want[(i, j)]).abs() <= bound,
                    "({i},{j}): {} against numpy's {}",
                    got[(i, j)],
                    want[(i, j)]
                );
            }
        }
    }

    /// The pose inverse is the inverse, and the median is numpy's.
    #[test]
    fn the_two_small_helpers_are_what_they_claim() {
        let angle: f64 = 0.7;
        let mut t = Matrix4::identity();
        t[(0, 0)] = angle.cos();
        t[(0, 1)] = -angle.sin();
        t[(1, 0)] = angle.sin();
        t[(1, 1)] = angle.cos();
        t[(0, 3)] = 3.0;
        t[(1, 3)] = -4.0;
        t[(2, 3)] = 5.0;
        let product = t * pose_inverse(&t);
        for i in 0..4 {
            for j in 0..4 {
                let want = f64::from(u8::from(i == j));
                assert!((product[(i, j)] - want).abs() < 1e-15, "({i},{j}) = {}", product[(i, j)]);
            }
        }

        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5, "an even count averages the middle two");
        assert_eq!(median(&mut [7.0]), 7.0);
        assert!(median(&mut []).is_nan());
    }
}
