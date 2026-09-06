//! Shell/fracture segmentation of the working mesh (R §3.4).
//!
//! A sherd's surface is two things: the vessel's own skin — glazed, smooth, curved with the pot —
//! and the fracture where it broke away. Only the fracture is worth matching, so every face of the
//! working mesh is labelled [`Shell`](FaceLabel::Shell) or [`Fracture`](FaceLabel::Fracture)
//! before anything else runs.
//!
//! The test is the opposite wall. From a face on the vessel's skin, a ray fired inwards crosses
//! the wall and lands on the *other* side of it, roughly one wall thickness away, on a face whose
//! normal points the way the ray travels. From a face on a fracture surface it does not: the
//! fracture cuts across the wall, so an inward ray runs along the wall, out of the open side, or
//! into geometry at the wrong distance. Firing seven rays in a 15° cone rather than one is what
//! makes the test survive a noisy decimated surface, and a majority of them must agree.
//!
//! The raw vote is noisy at the shell/fracture boundary, so four cleanup passes follow it, in this
//! order: a majority filter over `t/4` balls, island removal by area, a boundary refinement that
//! grows the shell back into the fracture band while the surface still looks like the shell it
//! grew from, and one more island removal. R §3.4 fixes the order and every constant; the numbers
//! themselves were measured against the Structure-from-Sherds++ surface ground truth
//! (`tools/eval_segmentation.py`) before being frozen — see [`SegParams`].
//!
//! # The smooth fields live on a grid, not on the faces
//!
//! Three quantities are needed per face and are expensive per face: the smoothed normal `NS` that
//! aims the cone, the majority vote, and the shell's reference normal. All three are evaluated on
//! the centroids of a `t/8` voxel grid (R §3.4.1) and looked up per face through `near`, the index
//! of the nearest grid representative. On a 150 000-face terracotta that is ~38 000 evaluations
//! instead of 150 000, and it also smooths: a grid point's ball is a fixed metric neighbourhood,
//! not a face-count one, so a sliver triangle and a large one get the same answer.
//!
//! # What this port does differently
//!
//! * **PMC-4.** The reference's `rep` comes out of an Open3D `unordered_map` and is in hash order;
//!   this port sorts it ascending. Only `near`-indexed values are read downstream, so the labels
//!   are unchanged — but a fixture's `seg.rep` and `seg.near` cannot be compared entry by entry
//!   with this port's, only as the map `face -> representative face` they define.
//! * **Summation order.** `scipy.spatial.cKDTree.query_ball_point(..., return_sorted=False)`
//!   returns a ball in an unspecified order and the reference sums the neighbourhood in that
//!   order; [`crate::spatial::kdtree::PointTree`] returns it ascending by index. The reference's
//!   own sums are therefore not reproducible bit for bit and this port's are; the difference is
//!   round-off in `NS`, far below the 15° cone or the 25° growth rule.

use rayon::prelude::*;

use crate::mesh::adjacency::{FaceAdjacency, face_adjacency};
use crate::mesh::components::drop_small_components;
use crate::mesh::geometry::{FaceGeometry, pairwise_sum};
use crate::spatial::bvh::RayScene;
use crate::spatial::kdtree::PointTree;
use crate::types::FaceLabel;

/// Rays in the cone of R §3.4.3, the first of them straight down `−NS`.
pub const CONE_RAYS: usize = 7;
/// Half-angle of that cone, in degrees.
pub const CONE_ANGLE_DEG: f64 = 15.0;
/// How far the ray origin is pushed off the surface, along the **raw** face normal (PMC-1).
pub const RAY_OFFSET: f64 = 1e-3;
/// A hit face "looks back along the ray" when its normal agrees with the ray direction this well.
pub const LOOKS_BACK_COS: f64 = 0.7;
/// A hit nearer than this fraction of `t` is the face's own neighbourhood, not the far wall.
pub const MIN_HIT_T: f64 = 0.1;
/// The window, in wall thicknesses, a hit must fall in to count as the opposite wall.
pub const WALL_WINDOW: (f64, f64) = (0.5, 1.8);
/// Voxel side of the coarse grid, in wall thicknesses (R §3.4.1).
pub const GRID_T: f64 = 1.0 / 8.0;
/// Radius of the majority filter's balls, in wall thicknesses (R §3.4.4).
pub const MAJORITY_T: f64 = 0.25;
/// Radius of the shell reference normal's balls, in wall thicknesses (R §3.4.7).
pub const REFERENCE_T: f64 = 0.5;
/// Fracture islands below this area, in `t²`, become shell (R §3.4.6, R §3.4.8).
pub const MIN_FRACTURE_T2: f64 = 0.5;
/// Shell islands below this area, in `t²`, become fracture (R §3.4.6).
pub const MIN_SHELL_T2: f64 = 2.0;
/// Growth passes of the boundary refinement (R §3.4.7).
pub const MAX_GROWTH_PASSES: usize = 60;

/// The knobs of R §1.3, with the defaults the pipeline ships.
///
/// Every one of them was measured against the Structure-from-Sherds++ surface ground truth on pots
/// A and B at full resolution and C, G and H decimated. Only the vote count earned a second value:
/// asking 4 of 7 rays instead of 5 once the mesh is coarser than `0.1·t` raises the precision of
/// the fracture mask on every pot (A 0.524 → 0.546, B 0.483 → 0.502, C 0.610 → 0.631,
/// G 0.474 → 0.497, H 0.520 → 0.544) and cuts the shell area wrongly called fracture by a fifth,
/// for one to three points of recall. The other three candidates did not survive measurement:
/// judging the hit face by its smoothed normal moves nothing (< 0.005 on all five pots), widening
/// the smoothing radius to `max(t/3, 3·res)` makes it worse on all three decimated pots, and the
/// resolution-dependent boundary angle is a no-op because after three Taubin iterations the
/// shell's own normal noise is 1–10°, never enough to clear the 25° already in use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegParams {
    /// The smoothing radius is `max(t/3, smooth_res·res)`; at 0 it is `t/3`.
    pub smooth_res: f64,
    /// Cone rays out of seven that must reach the far wall.
    pub votes: u32,
    /// … on a mesh coarser than `coarse_at` wall thicknesses.
    pub votes_coarse: u32,
    /// Where "coarse" starts, in wall thicknesses per edge.
    pub coarse_at: f64,
    /// Judge the hit face by its smoothed normal instead of its raw one.
    pub smoothed_hit_normal: bool,
    /// Shell growth stops beyond this angle from the shell's own reference normal.
    pub boundary_angle: f64,
    /// … or beyond the shell's measured normal noise plus 15°, whichever is larger.
    pub boundary_angle_auto: bool,
}

impl Default for SegParams {
    fn default() -> Self {
        Self {
            smooth_res: 0.0,
            votes: 5,
            votes_coarse: 4,
            coarse_at: 0.1,
            smoothed_hit_normal: false,
            boundary_angle: 25.0,
            boundary_angle_auto: false,
        }
    }
}

/// The labels and the diagnostics R §3.4 produces for one working mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Segmentation {
    /// One label per face, in face order.
    pub labels: Vec<FaceLabel>,
    /// Fracture area over total area **before** any cleanup — the reference's `raw_fraction`,
    /// reported so that a fragment whose raw vote and final mask disagree wildly is visible.
    pub raw_fraction: f64,
    /// Fracture area over total area of the final mask.
    pub fracture_fraction: f64,
    /// Fracture area of the final mask.
    pub fracture_area: f64,
    /// Total area of the working mesh.
    pub area: f64,
    /// Votes this mesh's resolution asked for (4 or 5).
    pub votes: u32,
    /// `max(t/3, smooth_res·res)`.
    pub smooth_radius: f64,
    /// The growth angle actually used, in degrees.
    pub boundary_angle: f64,
}

/// Every intermediate array R §3.4 passes from one step to the next.
///
/// This is what the parity harness compares stage by stage when the final agreement is short of
/// its gate: the fixture dump carries `seg.rep`, `seg.near`, `seg.NS`, `seg.good`,
/// `seg.frac_raw`, `seg.frac_majority`, `seg.frac_islands`, `seg.ref` and `seg.has_ref`, and each
/// one of them names a different way for the segmentation to go wrong.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SegTrace {
    /// Representative face of each occupied voxel, **ascending** (PMC-4).
    pub rep: Vec<u32>,
    /// Index into `rep` of the representative nearest to each face.
    pub near: Vec<u32>,
    /// The smoothed normal of R §3.4.2, per face.
    pub ns: Vec<[f64; 3]>,
    /// Cone votes per face, 0–7.
    pub good: Vec<u8>,
    /// `¬shell`, before the majority filter.
    pub frac_raw: Vec<bool>,
    /// After the majority filter of R §3.4.4.
    pub frac_majority: Vec<bool>,
    /// After the two island removals of R §3.4.6.
    pub frac_islands: Vec<bool>,
    /// The fixed shell reference normal of R §3.4.7, per face.
    pub reference: Vec<[f64; 3]>,
    /// Whether that ball held any shell face at all.
    pub has_ref: Vec<bool>,
}

/// The `t/8` voxel grid of R §3.4.1.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoarseGrid {
    /// Representative face of each occupied voxel: the **lowest-indexed** face in it, and the list
    /// sorted ascending (PMC-4; the reference leaves it in Open3D hash order).
    pub rep: Vec<u32>,
    /// For each face, the index *into `rep`* of the representative whose centroid is nearest.
    pub near: Vec<u32>,
}

/// R §3.4.1: voxel-downsample the face centroids and index every face by its nearest
/// representative.
///
/// The voxel of a point is `⌊(c − min_bound)/spacing⌋` per axis with `min_bound = C.min − 1`,
/// which is Open3D's `voxel_down_sample_and_trace` with the bounds the reference passes it. Open3D
/// appends the points of a voxel in increasing index order and the reference takes the first of
/// each list, so the representative of a voxel is its lowest-indexed face; that rule is exact
/// here, only the *order of the voxels* differs (PMC-4).
pub fn coarse_grid(centroids: &[[f64; 3]], spacing: f64) -> CoarseGrid {
    if centroids.is_empty() || !spacing.is_finite() || spacing <= 0.0 {
        return CoarseGrid { rep: Vec::new(), near: vec![0; centroids.len()] };
    }
    let mut min_bound = [f64::INFINITY; 3];
    for c in centroids {
        for k in 0..3 {
            min_bound[k] = min_bound[k].min(c[k]);
        }
    }
    for m in &mut min_bound {
        *m -= 1.0;
    }

    // (voxel, face) sorted: the faces of one voxel end up adjacent and ascending, so the first of
    // each run is the reference's `l[0]`. Sorting rather than hashing is what makes `rep`
    // reproducible across machines (D §7).
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a voxel index outside i64 needs a mesh larger than 9e18 voxels across"
    )]
    let mut cells: Vec<([i64; 3], u32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let v = [
                ((c[0] - min_bound[0]) / spacing).floor() as i64,
                ((c[1] - min_bound[1]) / spacing).floor() as i64,
                ((c[2] - min_bound[2]) / spacing).floor() as i64,
            ];
            (v, u32::try_from(i).expect("face count fits in u32"))
        })
        .collect();
    cells.sort_unstable();
    let mut rep: Vec<u32> = Vec::new();
    for (i, cell) in cells.iter().enumerate() {
        if i == 0 || cells[i - 1].0 != cell.0 {
            rep.push(cell.1);
        }
    }
    rep.sort_unstable();

    let points: Vec<[f64; 3]> = rep.iter().map(|&r| centroids[r as usize]).collect();
    let near = match PointTree::build(&points) {
        Some(tree) => centroids.par_iter().map(|c| tree.nearest(c)).collect(),
        None => vec![0; centroids.len()],
    };
    CoarseGrid { rep, near }
}

/// The area-weighted mean normal over each query point's ball, and whether that ball was empty.
///
/// `sherd_refit.geometry.ball_matrix` followed by `smoothed_normals`: `m = Σ A·FN / Σ A` over the
/// neighbours, then `m / |m|`, with the reference's two `1e-12` floors so that an empty ball
/// yields a zero vector rather than a NaN. `normals` and `areas` are indexed like the tree's
/// points, which is how R §3.4.7 smooths over the *shell* faces alone.
pub fn ball_normals(
    tree: &PointTree,
    queries: &[[f64; 3]],
    normals: &[[f64; 3]],
    areas: &[f64],
    radius: f64,
) -> (Vec<[f64; 3]>, Vec<bool>) {
    let out: Vec<([f64; 3], bool)> = queries
        .par_iter()
        .map(|q| {
            let ball = tree.within(q, radius);
            let mut m = [0.0_f64; 3];
            let mut wsum = 0.0_f64;
            for &j in &ball {
                let (n, a) = (normals[j as usize], areas[j as usize]);
                for k in 0..3 {
                    m[k] += n[k] * a;
                }
                wsum += a;
            }
            let den = wsum.max(1e-12);
            for v in &mut m {
                *v /= den;
            }
            let norm = ((m[0] * m[0] + m[1] * m[1]) + m[2] * m[2]).sqrt().max(1e-12);
            ([m[0] / norm, m[1] / norm, m[2] / norm], !ball.is_empty())
        })
        .collect();
    (out.iter().map(|o| o.0).collect(), out.iter().map(|o| o.1).collect())
}

/// R §3.4.3, the shell test: how many of the seven cone rays reach the far wall from behind.
///
/// Returns `good`, the vote count per face; the caller compares it with `votes`. `hit_normals` is
/// what the hit face is judged by — the raw face normals by default, the smoothed ones when
/// [`SegParams::smoothed_hit_normal`] is set.
///
/// Every arithmetic detail is the reference's, and two of them matter:
///
/// * the ray's origin and direction are computed in `f64` and narrowed to `f32` once, at the end
///   (`np.concatenate([...]).astype(np.float32)`), because Open3D's scene is `f32`;
/// * `dh > 0.1·t` and `0.5 < dh/t < 1.8` are evaluated in `f32`. `dh` is a `float32` array and `t`
///   a Python float, which numpy 2's NEP 50 treats as a weak scalar: it is cast down to `float32`
///   and the comparison happens there. The dot product against the hit face's normal, by contrast,
///   is `f64` — it is `np.einsum` over the `f64` normals and the `f64` directions.
///
/// A face whose smoothed normal is the zero vector (an empty ball, which cannot happen on a real
/// mesh because a representative's own face is always within its own radius) casts no ray and
/// votes zero, where the reference would build a NaN basis and let every hit test fail. Same
/// answer, no NaNs.
#[allow(clippy::cast_possible_truncation, reason = "the rays are f32, as Open3D's are")]
pub fn classify_faces(
    scene: &RayScene,
    geom: &FaceGeometry,
    ns: &[[f64; 3]],
    thick: f64,
    hit_normals: &[[f64; 3]],
) -> Vec<u8> {
    let n_faces = geom.len();
    let angle = CONE_ANGLE_DEG.to_radians();
    let (cos_a, sin_a) = (angle.cos(), angle.sin());
    let min_hit = (MIN_HIT_T * thick) as f32;
    let thick_f32 = thick as f32;

    (0..n_faces)
        .into_par_iter()
        .map(|i| {
            let n = ns[i];
            let Some((e1, e2)) = cone_basis(n) else { return 0 };
            let fnv = geom.normals[i];
            let c = geom.centroids[i];
            let origin = [
                (c[0] - fnv[0] * RAY_OFFSET) as f32,
                (c[1] - fnv[1] * RAY_OFFSET) as f32,
                (c[2] - fnv[2] * RAY_OFFSET) as f32,
            ];
            let mut good = 0_u8;
            for k in 0..CONE_RAYS {
                let d = if k == 0 {
                    [-n[0], -n[1], -n[2]]
                } else {
                    #[allow(clippy::cast_precision_loss, reason = "k <= 6")]
                    let phi =
                        2.0 * std::f64::consts::PI * ((k - 1) as f64) / ((CONE_RAYS - 1) as f64);
                    let (cp, sp) = (phi.cos(), phi.sin());
                    [
                        -(cos_a * n[0]) + sin_a * (cp * e1[0] + sp * e2[0]),
                        -(cos_a * n[1]) + sin_a * (cp * e1[1] + sp * e2[1]),
                        -(cos_a * n[2]) + sin_a * (cp * e1[2] + sp * e2[2]),
                    ]
                };
                let dir = [d[0] as f32, d[1] as f32, d[2] as f32];
                let Some((prim, dh)) = scene.first_hit(origin, dir) else { continue };
                if !dh.is_finite() || prim as usize >= n_faces || dh <= min_hit {
                    continue;
                }
                let ratio = dh / thick_f32;
                if ratio <= WALL_WINDOW.0 as f32 || ratio >= WALL_WINDOW.1 as f32 {
                    continue;
                }
                let h = hit_normals[prim as usize];
                let al = (h[0] * d[0] + h[1] * d[1]) + h[2] * d[2];
                if al > LOOKS_BACK_COS {
                    good += 1;
                }
            }
            good
        })
        .collect()
}

/// The orthonormal pair the cone is built on: `e1 = normalise(n × a)`, `e2 = n × e1`, with
/// `a = [1,0,0]` unless `n` is already close to it.
///
/// `None` when `n` is not a usable direction — the zero vector of an empty ball, or a non-finite
/// coordinate.
fn cone_basis(n: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    if !n.iter().all(|v| v.is_finite()) {
        return None;
    }
    let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let e1 = cross(n, a);
    let len = ((e1[0] * e1[0] + e1[1] * e1[1]) + e1[2] * e1[2]).sqrt();
    if len <= 0.0 {
        return None;
    }
    let e1 = [e1[0] / len, e1[1] / len, e1[2] / len];
    Some((e1, cross(n, e1)))
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// R §3.4.4: a face is fracture when more than half the area within `t/4` of its representative is.
fn majority_filter(
    tree: &PointTree,
    rep_points: &[[f64; 3]],
    areas: &[f64],
    frac: &[bool],
    radius: f64,
    near: &[u32],
) -> Vec<bool> {
    let per_rep: Vec<bool> = rep_points
        .par_iter()
        .map(|q| {
            let ball = tree.within(q, radius);
            let mut wet = 0.0_f64;
            let mut total = 0.0_f64;
            for &j in &ball {
                let a = areas[j as usize];
                total += a;
                if frac[j as usize] {
                    wet += a;
                }
            }
            wet > 0.5 * total
        })
        .collect();
    near.iter().map(|&r| per_rep[r as usize]).collect()
}

/// R §3.4.7: grow the shell back into the fracture band while the face still looks like the shell
/// it grew from.
///
/// The reference normal is *fixed*: it is computed once from the shell as R §3.4.6 left it, and
/// never updated as the shell grows, so the growth cannot drift around a curve. Each pass flips
/// every eligible fracture face that touches the shell, simultaneously — the candidate set is read
/// from the mask at the start of the pass and written at the end, which is what a numpy fancy
/// assignment does.
pub fn refine_boundary(
    frac: &mut [bool],
    adj: &FaceAdjacency,
    normals: &[[f64; 3]],
    reference: &[[f64; 3]],
    has_ref: &[bool],
    angle_deg: f64,
) {
    let cos_grow = angle_deg.to_radians().cos();
    let mut candidate = vec![false; frac.len()];
    let mut cand: Vec<u32> = Vec::new();
    for _ in 0..MAX_GROWTH_PASSES {
        cand.clear();
        for i in 0..adj.len() {
            let (a, b) = (adj.fa[i] as usize, adj.fb[i] as usize);
            if frac[a] != frac[b] {
                let f = if frac[a] { a } else { b };
                if !candidate[f] {
                    candidate[f] = true;
                    cand.push(u32::try_from(f).expect("face count fits in u32"));
                }
            }
        }
        if cand.is_empty() {
            break;
        }
        cand.sort_unstable();
        let mut flipped = false;
        for &f in &cand {
            let f = f as usize;
            candidate[f] = false;
            let (n, r) = (normals[f], reference[f]);
            if has_ref[f] && (n[0] * r[0] + n[1] * r[1]) + n[2] * r[2] > cos_grow {
                frac[f] = false;
                flipped = true;
            }
        }
        if !flipped {
            break;
        }
    }
}

/// R §3.4 end to end for one working mesh.
///
/// `scene` is the BVH over that same mesh, `geom` its `f64` face geometry, `thick` the wall of
/// R §3.2 and `res` the median edge length; the labels come back in face order.
pub fn segment_faces(
    scene: &RayScene,
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    thick: f64,
    res: f64,
    sp: &SegParams,
) -> Segmentation {
    run(scene, f, geom, thick, res, sp, None)
}

/// [`segment_faces`], keeping every intermediate array for the parity harness.
pub fn segment_faces_traced(
    scene: &RayScene,
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    thick: f64,
    res: f64,
    sp: &SegParams,
) -> (Segmentation, SegTrace) {
    let mut trace = SegTrace::default();
    let seg = run(scene, f, geom, thick, res, sp, Some(&mut trace));
    (seg, trace)
}

#[allow(clippy::too_many_lines, reason = "R §3.4's eight steps, in the reference's order")]
fn run(
    scene: &RayScene,
    f: &[[u32; 3]],
    geom: &FaceGeometry,
    thick: f64,
    res: f64,
    sp: &SegParams,
    mut trace: Option<&mut SegTrace>,
) -> Segmentation {
    let n = geom.len();
    let area = geom.total_area();
    if n == 0 || !thick.is_finite() || thick <= 0.0 {
        return Segmentation {
            labels: vec![FaceLabel::Shell; n],
            area,
            votes: sp.votes,
            boundary_angle: sp.boundary_angle,
            ..Segmentation::default()
        };
    }
    let centroids = &geom.centroids;
    let areas = &geom.areas;
    let normals = &geom.normals;

    // --- 3.4.1 the grid ------------------------------------------------------------------------
    let grid = coarse_grid(centroids, thick * GRID_T);
    let rep_points: Vec<[f64; 3]> = grid.rep.iter().map(|&r| centroids[r as usize]).collect();
    let tree = PointTree::build(centroids).expect("a non-empty mesh has centroids");

    // --- 3.4.2 the smoothed normals ------------------------------------------------------------
    let radius = (thick / 3.0).max(sp.smooth_res * res);
    let (ns_g, _) = ball_normals(&tree, &rep_points, normals, areas, radius);
    let ns: Vec<[f64; 3]> = grid.near.iter().map(|&r| ns_g[r as usize]).collect();

    // --- 3.4.3 the cone vote -------------------------------------------------------------------
    let votes = if res > sp.coarse_at * thick { sp.votes_coarse } else { sp.votes };
    let hit_normals: &[[f64; 3]] = if sp.smoothed_hit_normal { &ns } else { normals };
    let good = classify_faces(scene, geom, &ns, thick, hit_normals);
    let mut frac: Vec<bool> = good.iter().map(|&g| u32::from(g) < votes).collect();
    let raw_fraction = fraction_of(masked_area(areas, &frac), area);
    if let Some(t) = trace.as_mut() {
        t.rep.clone_from(&grid.rep);
        t.near.clone_from(&grid.near);
        t.ns.clone_from(&ns);
        t.good.clone_from(&good);
        t.frac_raw.clone_from(&frac);
    }

    // --- 3.4.4 the majority filter -------------------------------------------------------------
    frac = majority_filter(&tree, &rep_points, areas, &frac, thick * MAJORITY_T, &grid.near);
    if let Some(t) = trace.as_mut() {
        t.frac_majority.clone_from(&frac);
    }

    // --- 3.4.5–3.4.6 islands -------------------------------------------------------------------
    let adj = face_adjacency(f);
    drop_small_components(&mut frac, true, MIN_FRACTURE_T2 * thick * thick, &adj, areas);
    drop_small_components(&mut frac, false, MIN_SHELL_T2 * thick * thick, &adj, areas);
    if let Some(t) = trace.as_mut() {
        t.frac_islands.clone_from(&frac);
    }

    // --- 3.4.7 the boundary growth -------------------------------------------------------------
    let shell: Vec<u32> = (0..n)
        .filter(|&i| !frac[i])
        .map(|i| u32::try_from(i).expect("face count fits in u32"))
        .collect();
    let mut boundary_angle = sp.boundary_angle;
    if sp.boundary_angle_auto && !shell.is_empty() {
        let noise: Vec<f64> = shell
            .iter()
            .map(|&i| {
                let (a, b) = (normals[i as usize], ns[i as usize]);
                ((a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]).clamp(-1.0, 1.0).acos().to_degrees()
            })
            .collect();
        boundary_angle = boundary_angle.max(crate::mesh::geometry::median(&noise) + 15.0);
    }
    let (reference, has_ref) = if shell.is_empty() {
        (vec![[0.0; 3]; n], vec![false; n])
    } else {
        let shell_points: Vec<[f64; 3]> = shell.iter().map(|&i| centroids[i as usize]).collect();
        let shell_normals: Vec<[f64; 3]> = shell.iter().map(|&i| normals[i as usize]).collect();
        let shell_areas: Vec<f64> = shell.iter().map(|&i| areas[i as usize]).collect();
        let shell_tree = PointTree::build(&shell_points).expect("a non-empty shell");
        let (ref_g, has_g) = ball_normals(
            &shell_tree,
            &rep_points,
            &shell_normals,
            &shell_areas,
            thick * REFERENCE_T,
        );
        (
            grid.near.iter().map(|&r| ref_g[r as usize]).collect(),
            grid.near.iter().map(|&r| has_g[r as usize]).collect(),
        )
    };
    if let Some(t) = trace.as_mut() {
        t.reference.clone_from(&reference);
        t.has_ref.clone_from(&has_ref);
    }
    refine_boundary(&mut frac, &adj, normals, &reference, &has_ref, boundary_angle);

    // --- 3.4.8 one more island pass ------------------------------------------------------------
    drop_small_components(&mut frac, true, MIN_FRACTURE_T2 * thick * thick, &adj, areas);

    let fracture_area = masked_area(areas, &frac);
    Segmentation {
        labels: frac
            .iter()
            .map(|&is| if is { FaceLabel::Fracture } else { FaceLabel::Shell })
            .collect(),
        raw_fraction,
        fracture_fraction: fraction_of(fracture_area, area),
        fracture_area,
        area,
        votes,
        smooth_radius: radius,
        boundary_angle,
    }
}

/// `A[mask].sum()`: numpy's pairwise sum over the selected faces, so that the fracture fraction is
/// the reference's to the last bit.
fn masked_area(areas: &[f64], mask: &[bool]) -> f64 {
    let selected: Vec<f64> = areas.iter().zip(mask).filter_map(|(&a, &m)| m.then_some(a)).collect();
    pairwise_sum(&selected)
}

/// `part / total`, with a mesh of zero total area reporting 0 rather than a NaN. Nothing in R
/// produces one — `remove_degenerate_triangles` runs first — so this is a guard, not a rule.
fn fraction_of(part: f64, total: f64) -> f64 {
    if total > 0.0 { part / total } else { 0.0 }
}

/// The area-weighted agreement between two label arrays over the same faces (D §10.2).
///
/// `Σ A[i] over faces where the labels agree / Σ A[i]`; an empty mesh agrees perfectly with
/// itself.
pub fn label_agreement(a: &[FaceLabel], b: &[FaceLabel], areas: &[f64]) -> f64 {
    let total = pairwise_sum(areas);
    if total <= 0.0 {
        return 1.0;
    }
    let agree: Vec<f64> = (0..areas.len()).filter(|&i| a[i] == b[i]).map(|i| areas[i]).collect();
    pairwise_sum(&agree) / total
}

#[cfg(test)]
mod tests {
    use super::{
        CONE_RAYS, SegParams, classify_faces, coarse_grid, label_agreement, segment_faces,
        segment_faces_traced,
    };
    use crate::mesh::Mesh;
    use crate::mesh::geometry::face_geometry;
    use crate::spatial::bvh::RayScene;
    use crate::types::FaceLabel;

    /// A closed slab: a `w × w × t` box whose every face is a grid of quads of side ≈ `edge`, so
    /// the mesh has a real resolution and a real wall.
    ///
    /// The two large faces are the shell — an inward ray crosses the wall and lands on the other
    /// side of it, one `t` away — and the four sides are the fracture band, because an inward ray
    /// from a side runs the whole `w` across the slab and lands far outside the `(0.5, 1.8)·t`
    /// window. That is exactly the shape a sherd has, at a size a unit test can afford.
    #[allow(clippy::many_single_char_names, reason = "a box's corners and its two axes")]
    fn slab(w: f64, t: f64, edge: f64) -> Mesh {
        let mut v: Vec<[f64; 3]> = Vec::new();
        let mut f: Vec<[u32; 3]> = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a test helper")]
        let mut quad = |a: [f64; 3], du: [f64; 3], dv: [f64; 3]| {
            let len = |d: [f64; 3]| ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt();
            let (nu, nv) = ((len(du) / edge).round() as usize, (len(dv) / edge).round() as usize);
            let base = u32::try_from(v.len()).unwrap();
            for i in 0..=nu {
                for j in 0..=nv {
                    #[allow(clippy::cast_precision_loss, reason = "small grids")]
                    let (s, r) = (i as f64 / nu as f64, j as f64 / nv as f64);
                    v.push([
                        a[0] + du[0] * s + dv[0] * r,
                        a[1] + du[1] * s + dv[1] * r,
                        a[2] + du[2] * s + dv[2] * r,
                    ]);
                }
            }
            let at = |i: usize, j: usize| base + u32::try_from(i * (nv + 1) + j).unwrap();
            for i in 0..nu {
                for j in 0..nv {
                    f.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                    f.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
                }
            }
        };
        let (h, ht) = (w / 2.0, t / 2.0);
        quad([-h, -h, ht], [w, 0.0, 0.0], [0.0, w, 0.0]);
        quad([-h, -h, -ht], [0.0, w, 0.0], [w, 0.0, 0.0]);
        quad([-h, -h, -ht], [w, 0.0, 0.0], [0.0, 0.0, t]);
        quad([-h, h, -ht], [0.0, 0.0, t], [w, 0.0, 0.0]);
        quad([-h, -h, -ht], [0.0, 0.0, t], [0.0, w, 0.0]);
        quad([h, -h, -ht], [0.0, w, 0.0], [0.0, 0.0, t]);
        Mesh::new(v, f)
    }

    /// `(mesh, geometry, scene, res)` for a slab of 24 × 24 × 6 at half a millimetre per edge:
    /// 12 edges across the wall, which is what R §3.3's face budget aims for.
    fn test_slab() -> (Mesh, crate::mesh::geometry::FaceGeometry, RayScene, f64) {
        let m = slab(36.0, 6.0, 0.5);
        let geom = face_geometry(&m.v, &m.f);
        let scene = RayScene::new(&m.v, &m.f).expect("a closed slab");
        let res = crate::mesh::geometry::median_edge(&m.v, &m.f);
        (m, geom, scene, res)
    }

    #[test]
    fn the_fracture_band_of_a_closed_slab_is_its_sides() {
        let t = 6.0;
        let (m, geom, scene, res) = test_slab();
        assert!(res < 0.1 * t, "res {res} must stay under 0.1·t for the five-vote rule");
        let seg = segment_faces(&scene, &m.f, &geom, t, res, &SegParams::default());

        assert_eq!(seg.labels.len(), m.f.len());
        assert_eq!(seg.votes, 5, "res is under 0.1·t here");
        assert!((seg.smooth_radius - t / 3.0).abs() < 1e-12);
        assert!((seg.boundary_angle - 25.0).abs() < 1e-12);
        assert!(seg.area > 0.0 && seg.fracture_area > 0.0);

        // The sides carry 4·w·t of the total 2·w² + 4·w·t.
        let side = 4.0 * 36.0 * t / (2.0 * 36.0 * 36.0 + 4.0 * 36.0 * t);
        assert!(
            (seg.fracture_fraction - side).abs() < 0.005,
            "fracture fraction {} against the slab's sides at {side}",
            seg.fracture_fraction
        );

        // Face by face: every face on a side is fracture and every face on a large side is shell.
        // On this machine the disagreement is exactly zero — the rim, where the cone straddles
        // both, is caught by the majority filter and the boundary growth — and the gate is left at
        // half a per cent so that a `f32` ray grazing an edge on another platform is not a failure.
        let wrong: f64 = (0..m.f.len())
            .filter(|&i| (geom.normals[i][2].abs() > 0.5) != (seg.labels[i] == FaceLabel::Shell))
            .map(|i| geom.areas[i])
            .sum();
        assert!(
            wrong / seg.area < 0.005,
            "{:.4} of the area is on the wrong side",
            wrong / seg.area
        );
    }

    #[test]
    fn the_vote_counts_the_rays_that_reach_the_far_wall() {
        let t = 6.0;
        let (m, geom, scene, _) = test_slab();
        // With the raw face normals as the cone axis, a face well inside a large side sees the
        // opposite wall with all seven rays; a face on a rim sees nothing, because the slab is
        // 24 across and 4·t is far outside the (0.5, 1.8)·t window.
        let good = classify_faces(&scene, &geom, &geom.normals, t, &geom.normals);
        assert_eq!(good.len(), m.f.len());
        let mut inside = 0;
        for (i, &votes) in good.iter().enumerate() {
            let c = geom.centroids[i];
            let flat = geom.normals[i][2].abs() > 0.5;
            if flat && c[0].abs() < 36.0 / 2.0 - 2.0 * t && c[1].abs() < 36.0 / 2.0 - 2.0 * t {
                assert_eq!(
                    votes as usize, CONE_RAYS,
                    "a shell face must see the far wall with every ray"
                );
                inside += 1;
            }
            if !flat {
                assert_eq!(votes, 0, "a rim face sees no wall inside the window");
            }
        }
        assert!(inside > 100, "the test looked at {inside} interior faces");
    }

    #[test]
    fn the_grid_takes_the_lowest_indexed_face_of_each_voxel_and_sorts_them() {
        // Four points, two of them in the same voxel of side 1.
        let c = vec![[0.1, 0.0, 0.0], [5.0, 0.0, 0.0], [0.3, 0.0, 0.0], [2.5, 0.0, 0.0]];
        let grid = coarse_grid(&c, 1.0);
        assert_eq!(grid.rep, vec![0, 1, 3], "face 2 shares face 0's voxel, and rep is ascending");
        // `near` indexes into `rep`, and face 2 is nearest to representative 0.
        assert_eq!(grid.near, vec![0, 1, 0, 2]);

        let empty = coarse_grid(&[], 1.0);
        assert!(empty.rep.is_empty() && empty.near.is_empty());
        // A spacing of zero cannot bucket anything; the grid degrades rather than dividing by it.
        assert!(coarse_grid(&c, 0.0).rep.is_empty());
    }

    #[test]
    fn the_trace_carries_every_stage_of_the_cleanup() {
        let t = 6.0;
        let (m, geom, scene, res) = test_slab();
        let (seg, trace) = segment_faces_traced(&scene, &m.f, &geom, t, res, &SegParams::default());
        assert_eq!(trace.near.len(), m.f.len());
        assert_eq!(trace.ns.len(), m.f.len());
        assert_eq!(trace.good.len(), m.f.len());
        assert_eq!(trace.frac_raw.len(), m.f.len());
        assert_eq!(trace.frac_majority.len(), m.f.len());
        assert_eq!(trace.frac_islands.len(), m.f.len());
        assert_eq!(trace.reference.len(), m.f.len());
        assert_eq!(trace.has_ref.len(), m.f.len());
        assert!(trace.rep.windows(2).all(|w| w[0] < w[1]), "rep is ascending and unique");
        assert!(trace.rep.len() <= m.f.len());
        // The raw fraction is the raw mask's, not the final one's.
        let raw: Vec<f64> =
            trace.frac_raw.iter().zip(&geom.areas).filter_map(|(&f, &a)| f.then_some(a)).collect();
        assert!((seg.raw_fraction - raw.iter().sum::<f64>() / geom.total_area()).abs() < 1e-12);
        assert_eq!(seg.labels.len(), m.f.len());
    }

    #[test]
    fn the_agreement_is_area_weighted() {
        let a = vec![FaceLabel::Shell, FaceLabel::Fracture, FaceLabel::Shell];
        let b = vec![FaceLabel::Shell, FaceLabel::Shell, FaceLabel::Shell];
        assert!((label_agreement(&a, &a, &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-15);
        // Face 1 disagrees and carries 2 of the 6 units of area.
        assert!((label_agreement(&a, &b, &[1.0, 2.0, 3.0]) - 4.0 / 6.0).abs() < 1e-15);
        assert!((label_agreement(&[], &[], &[]) - 1.0).abs() < 1e-15);
    }
}
