//! Open3D's `registration_icp`, as R §7 freezes it (point-to-point and point-to-plane).
//!
//! Every refinement in the pipeline goes through this one function: the two breakline rungs of
//! R §5.4, the four registration and fracture rungs of R §5.6, and R §9's full-resolution rungs
//! later. It is deliberately a transcription rather than an implementation of ICP in general —
//! the reference is a call into Open3D, so what the port has to reproduce is *Open3D's* ICP,
//! down to which loop it breaks out of:
//!
//! ```text
//! T ← T0;  P ← T0·S
//! (fit, rmse, C) ← corr(P)
//! for it in 1..=max_iter:
//!     U ← update(P, C);  T ← U·T;  P ← U·P
//!     (fit', rmse', C) ← corr(P)
//!     if |fit' − fit| < 1e-6 and |rmse' − rmse| < 1e-6: break
//!     fit, rmse ← fit', rmse'
//! ```
//!
//! The update is applied to the *already transformed* source points rather than recomputed from
//! the source each time, and the returned `(fitness, rmse)` are always the ones measured **after**
//! the last update — both are Open3D's, and both are visible in the numbers.
//!
//! # The five details that decide whether a pose comes out the same
//!
//! * **The correspondence radius is a strict `<` on the squared distance.** Open3D hands FLANN
//!   `max_correspondence_distance²` and FLANN's `KNNRadiusResultSet` drops a candidate with
//!   `dist >= radius`. Measured against Open3D 0.19 (`notes/2026-09-07-c2-icp.md` §2): a source
//!   point exactly `d` from its nearest target is **not** a correspondence, and one an ulp closer
//!   is. The bound is on the distance to the *nearest* target, so a nearer target outside the
//!   radius cannot hide a farther one inside it: there is at most one correspondence per source
//!   point, and it is the nearest.
//! * **`fitness` divides by the number of source points, `inlier_rmse` by the number of
//!   correspondences**, and `error²` is a sum of *squared* distances — the same numbers FLANN
//!   returns, never re-derived from a square root.
//! * **The summation order is the source order.** Open3D accumulates the correspondence set and
//!   `error²` under `#pragma omp parallel`, merging thread-private partials in the order the
//!   threads reach the critical section, and the same is true of the 6×6 normal equations
//!   (`ComputeJTJandJTr`). With `OMP_NUM_THREADS=1` — which is how the fixtures were dumped
//!   (D §10.1) — that is a plain sequential sum over ascending source index, and that is what this
//!   module does on every thread count (D §7).
//! * **The point-to-plane update composes Euler angles, not an exponential map.** `U = Rz(x₂) ·
//!   Ry(x₁) · Rx(x₀)` with the translation taken as-is. On a 0.05 rad step the two differ by
//!   ~3e-4, which single iterations of R §13's gates can see.
//! * **The point-to-point update is Eigen's `umeyama` without scaling**, whose mean is a
//!   *multiplication* by `1/n` rather than a division — the same shape of detail as the coarse
//!   score's `k/60` (step C1), kept here because it costs nothing to be right.
//!
//! # `f32` and `f64` (D §7, experiment E5)
//!
//! The pose is `f64` everywhere and so is the 6×6 solve; what [`Precision`] selects is the
//! arithmetic of the **point loops** — the transform applied to the moving cloud, the residual,
//! the Jacobian row and the accumulation into the normal equations — because that is the half a
//! GPU executor runs in `f32` (D §6.5) and the half whose cost is linear in the cloud size.
//! [`Assembly`] selects where those normal equations are assembled.
//!
//! **Experiment E5 ran both knobs over the injected fixtures and neither is parity-safe**
//! (`notes/2026-09-07-c2-icp.md` §6). `f32` point loops keep half the candidates within 2e-6 t of
//! the reference and put the other tail far outside D §10.2: on terracotta the stage-2 pose is
//! 1.9e-6 t out at the median and 0.26 t at p99, and on pot B — whose `t` is 3.58 rather than
//! terracotta's 38.8, so the same absolute `f32` step is ten times as many wall thicknesses — the
//! *median* is already 0.115 t. The GPU executor of D §6.5 therefore cannot run these ladders in
//! `f32`; what E5 settles is that the design's expectation of 1e-5 t was optimistic by four orders
//! of magnitude. [`Numerics::REFERENCE`] is the default and is what every parity row is measured
//! under.

use nalgebra::{Matrix3, Matrix4, Vector3};

use crate::spatial::kdtree::PointTree;
use crate::types::Cloud;

/// Open3D's `ICPConvergenceCriteria::relative_fitness_` (R §7).
pub const RELATIVE_FITNESS: f64 = 1e-6;
/// Open3D's `ICPConvergenceCriteria::relative_rmse_` (R §7).
pub const RELATIVE_RMSE: f64 = 1e-6;

/// Which estimator computes the per-iteration update (R §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimation {
    /// Umeyama without scaling, over the correspondence pairs; target normals unused (R §5.4).
    PointToPoint,
    /// Gauss–Newton on the point-to-plane residual, using the target normals (R §5.6).
    PointToPlane,
}

/// The scalar the point loops run in (D §7, experiment E5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// The reference's: every point loop in `f64`.
    #[default]
    F64,
    /// The GPU executor's: points, residuals, Jacobian rows and their accumulation in `f32`,
    /// the pose and the 6×6 solve still `f64` (D §6.5).
    F32,
}

/// Where the normal equations of the point-to-plane step are assembled (D §7).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Assembly {
    /// R §7 verbatim: in the meshes' own coordinates.
    #[default]
    World,
    /// D §7: about the target centroid `c`, with the update re-expressed about the origin.
    ///
    /// The *linear* system is unchanged by this — `ω` is the same and `τ = τ' + (I − R)c` — but
    /// the finite update is not: `World` composes `p ↦ Rp + τ` and `Centred` composes
    /// `p ↦ c + R(p − c) + τ'`, two different non-linear extensions of the same Gauss–Newton step
    /// that differ by `(R − I − ω̂)c = O(|ω|²·|c|)` per iteration.
    ///
    /// D §7 estimated that at 1e-6 t and called it f32-safe. **Measured, it is 1e-2 t**: the
    /// terracotta clouds sit `|c| ≈ 100–150` scan units from the origin (2.5–3.8 t) and the first
    /// stage-2 rung takes `|ω| ≈ 0.1` rad steps for thirty iterations without converging, so
    /// `|ω|²·|c|` is ≈ 0.03 t per iteration and the two assemblies end a rung 0.044 t apart at the
    /// median and 0.61 t apart at p90 (`notes/2026-09-07-c2-icp.md` §6). The knob stays because it
    /// is the instrument that measured that; it is not a drop-in for the GPU path, which would
    /// have to translate *both clouds* by `−c` — a rigid change of frame the whole of R §7 is
    /// equivariant under — rather than re-parameterise the Jacobian alone.
    Centred,
}

/// The two numerical choices of one ICP run, together (D §7).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Numerics {
    /// The scalar of the point loops.
    pub precision: Precision,
    /// The frame the normal equations are assembled in.
    pub assembly: Assembly,
}

impl Numerics {
    /// What the reference does, and what every parity row of D §10.2 is measured under.
    pub const REFERENCE: Self = Self { precision: Precision::F64, assembly: Assembly::World };
    /// What D §6.5 proposed for the GPU executor: `f32` point loops on the centred system.
    /// Experiment E5 measured both halves of it outside D §10.2 (see the module documentation),
    /// so this is the combination the note argues *against*, kept as its instrument.
    pub const GPU: Self = Self { precision: Precision::F32, assembly: Assembly::Centred };
}

/// One call of `registration_icp` (R §7).
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Which update to compute.
    pub estimation: Estimation,
    /// `max_correspondence_distance`: no source point pairs with a target further than this.
    pub max_correspondence_distance: f64,
    /// `ICPConvergenceCriteria::max_iteration_`. Zero evaluates the initial pose and returns it,
    /// which is how the parity harness reads a fitness and an RMSE off a pose it did not compute.
    pub max_iteration: usize,
    /// D §7's two knobs.
    pub numerics: Numerics,
}

impl Options {
    /// A rung of R §5.4's ladder: point-to-point at `distance`, under the reference numerics.
    pub fn point_to_point(distance: f64, max_iteration: usize) -> Self {
        Self {
            estimation: Estimation::PointToPoint,
            max_correspondence_distance: distance,
            max_iteration,
            numerics: Numerics::REFERENCE,
        }
    }

    /// A rung of R §5.6's ladder: point-to-plane at `distance`, under the reference numerics.
    pub fn point_to_plane(distance: f64, max_iteration: usize) -> Self {
        Self {
            estimation: Estimation::PointToPlane,
            max_correspondence_distance: distance,
            max_iteration,
            numerics: Numerics::REFERENCE,
        }
    }

    /// The same options under other numerics (experiment E5).
    #[must_use]
    pub fn with(self, numerics: Numerics) -> Self {
        Self { numerics, ..self }
    }
}

/// What one `registration_icp` call returns (R §7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Registration {
    /// The accumulated pose, mapping the source into the target's frame.
    pub transform: Matrix4<f64>,
    /// `|C| / n_source`, measured after the last update.
    pub fitness: f64,
    /// `sqrt(Σ_C |P_i − Q_j|² / |C|)`, measured after the last update.
    pub inlier_rmse: f64,
    /// How many correspondences the last search found.
    pub correspondences: usize,
    /// How many updates were applied before the loop stopped.
    pub iterations: usize,
    /// True when the loop stopped on R §7's convergence test rather than on `max_iteration`.
    pub converged: bool,
}

impl Registration {
    /// A pose that was never iterated: what an empty target or an empty source returns.
    fn unmoved(transform: Matrix4<f64>) -> Self {
        Self {
            transform,
            fitness: 0.0,
            inlier_rmse: 0.0,
            correspondences: 0,
            iterations: 0,
            converged: false,
        }
    }
}

/// A target cloud with the KD-tree the correspondence search runs on.
///
/// Open3D builds a `KDTreeFlann` inside every `registration_icp` call; the port builds one per
/// cloud and reuses it across a ladder's rungs and across a pair's candidates, which is a large
/// part of why the stage is faster than the reference (D §6.6). The tree is over the same
/// coordinates the search compares, so nothing here narrows or widens between the two.
#[derive(Debug)]
pub struct IcpTarget {
    points: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
    tree: Option<PointTree>,
    centroid: [f64; 3],
}

impl IcpTarget {
    /// A target from its points and (for point-to-plane) their unit normals.
    ///
    /// `normals` may be empty for a point-to-point target; a point-to-plane run needs one per
    /// point and [`IcpTarget::has_normals`] says whether it has them.
    pub fn new(points: Vec<[f64; 3]>, normals: Vec<[f64; 3]>) -> Self {
        let tree = PointTree::build(&points);
        let centroid = centroid_of(&points);
        Self { points, normals, tree, centroid }
    }

    /// [`IcpTarget::new`] with every coordinate first rounded through `f32`.
    ///
    /// This is the cloud a GPU executor would hold (D §6.3), and it is what experiment E5 measures
    /// the `f32` path against: narrowing the *data* is a separate question from running the point
    /// loops in `f32`, and the two are separated here so that the note can say which one moved the
    /// pose.
    pub fn narrowed(points: &[[f64; 3]], normals: &[[f64; 3]]) -> Self {
        Self::new(
            points.iter().map(narrow_point).collect(),
            normals.iter().map(narrow_point).collect(),
        )
    }

    /// The target for a given precision: [`IcpTarget::new`] in `f64`, [`IcpTarget::narrowed`] in
    /// `f32`.
    pub fn for_precision(points: &[[f64; 3]], normals: &[[f64; 3]], precision: Precision) -> Self {
        match precision {
            Precision::F64 => Self::new(points.to_vec(), normals.to_vec()),
            Precision::F32 => Self::narrowed(points, normals),
        }
    }

    /// The port's own `f32` cloud (R §3.6, D §4.1), widened once for the matcher.
    pub fn from_cloud(cloud: &Cloud) -> Self {
        Self::new(
            cloud.p.iter().map(|p| p.to_f64()).collect(),
            cloud.n.iter().map(|n| n.to_f64()).collect(),
        )
    }

    /// The target points.
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// The target normals, empty when the target carries none.
    pub fn normals(&self) -> &[[f64; 3]] {
        &self.normals
    }

    /// True when there is a normal per point — the precondition of a point-to-plane run.
    pub fn has_normals(&self) -> bool {
        self.normals.len() == self.points.len() && !self.points.is_empty()
    }

    /// The mean of the target points, which [`Assembly::Centred`] assembles about.
    pub fn centroid(&self) -> [f64; 3] {
        self.centroid
    }

    /// Number of target points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True when the target holds no point (and therefore no tree).
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// A cloud's points, widened to the `f64` the matcher works in (D §4.1).
pub fn cloud_points(cloud: &Cloud) -> Vec<[f64; 3]> {
    cloud.p.iter().map(|p| p.to_f64()).collect()
}

/// R §7's `registration_icp`.
///
/// `source` is the moving cloud in its own frame, `init` the pose that places it in the target's,
/// and the result's `transform` is the refined one. Neither the source's normals nor its order
/// matter to the answer, but the order fixes the summation order and therefore the last bits
/// (D §7).
pub fn register(
    source: &[[f64; 3]],
    target: &IcpTarget,
    init: &Matrix4<f64>,
    options: &Options,
) -> Registration {
    if source.is_empty() || target.is_empty() {
        return Registration::unmoved(*init);
    }
    let precision = options.numerics.precision;
    let mut transform = *init;
    // Open3D transforms the source by `init` before the first correspondence search (and skips
    // the call when `init` is the identity, which is the same arithmetic: `1·p + 0 + 0 + 0` is `p`
    // exactly). In `f32` this is also where the moving cloud is narrowed.
    let mut moved = source.to_vec();
    transform_points(&mut moved, init, precision);

    let mut corres: Vec<(u32, u32)> = Vec::with_capacity(source.len());
    let (mut fitness, mut rmse) =
        correspondences(&moved, target, options.max_correspondence_distance, &mut corres);
    let mut iterations = 0;
    let mut converged = false;
    for _ in 0..options.max_iteration {
        let update = match options.estimation {
            Estimation::PointToPoint => point_to_point(&moved, target, &corres, precision),
            Estimation::PointToPlane => point_to_plane(&moved, target, &corres, options.numerics),
        };
        transform = update * transform;
        transform_points(&mut moved, &update, precision);
        iterations += 1;
        let (next_fitness, next_rmse) =
            correspondences(&moved, target, options.max_correspondence_distance, &mut corres);
        converged = (next_fitness - fitness).abs() < RELATIVE_FITNESS
            && (next_rmse - rmse).abs() < RELATIVE_RMSE;
        fitness = next_fitness;
        rmse = next_rmse;
        if converged {
            break;
        }
    }
    Registration {
        transform,
        fitness,
        inlier_rmse: rmse,
        correspondences: corres.len(),
        iterations,
        converged,
    }
}

/// R §7's `corr`: the nearest target within `distance` of each source point, in source order.
///
/// Returns `(fitness, inlier_rmse)` and fills `out` with `(source, target)` index pairs.
fn correspondences(
    moved: &[[f64; 3]],
    target: &IcpTarget,
    distance: f64,
    out: &mut Vec<(u32, u32)>,
) -> (f64, f64) {
    out.clear();
    let Some(tree) = target.tree.as_ref() else { return (0.0, 0.0) };
    if distance <= 0.0 || distance.is_nan() {
        // Open3D's `GetRegistrationResultAndCorrespondences` returns an empty result for a
        // non-positive radius rather than searching with it.
        return (0.0, 0.0);
    }
    let radius_squared = distance * distance;
    let mut error_squared = 0.0;
    for (i, point) in moved.iter().enumerate() {
        let Some((j, d2)) = tree.nearest_within_squared(point, radius_squared) else { continue };
        // FLANN's radius test is strict, and Open3D measured: a point exactly at the radius is
        // not a correspondence (see the module documentation).
        if d2 >= radius_squared {
            continue;
        }
        error_squared += d2;
        out.push((u32::try_from(i).unwrap_or(u32::MAX), j));
    }
    if out.is_empty() {
        return (0.0, 0.0);
    }
    #[allow(clippy::cast_precision_loss, reason = "cloud sizes are far below 2^53")]
    let (n_corres, n_source) = (out.len() as f64, moved.len() as f64);
    (n_corres / n_source, (error_squared / n_corres).sqrt())
}

/// R §7's point-to-point update: Eigen's `umeyama` without scaling.
fn point_to_point(
    moved: &[[f64; 3]],
    target: &IcpTarget,
    corres: &[(u32, u32)],
    precision: Precision,
) -> Matrix4<f64> {
    match precision {
        Precision::F64 => umeyama::<f64>(moved, target, corres),
        Precision::F32 => umeyama::<f32>(moved, target, corres),
    }
}

/// R §7's point-to-plane update: the 6×6 normal equations, an LDLT solve and an Euler composition.
fn point_to_plane(
    moved: &[[f64; 3]],
    target: &IcpTarget,
    corres: &[(u32, u32)],
    numerics: Numerics,
) -> Matrix4<f64> {
    if corres.is_empty() || !target.has_normals() {
        return Matrix4::identity();
    }
    let centre = match numerics.assembly {
        Assembly::World => [0.0; 3],
        Assembly::Centred => target.centroid,
    };
    let (jtj, jtr) = match numerics.precision {
        Precision::F64 => normal_equations::<f64>(moved, target, corres, centre),
        Precision::F32 => normal_equations::<f32>(moved, target, corres, centre),
    };
    let mut rhs = [0.0; 6];
    for (out, value) in rhs.iter_mut().zip(&jtr) {
        *out = -value;
    }
    // A system this port cannot solve is one Open3D would have solved into NaNs; an identity
    // update leaves the pose where it was and lets the convergence test end the rung.
    let Some(x) = solve_ldlt(&jtj, &rhs) else { return Matrix4::identity() };
    let rotation = euler_zyx(x[0], x[1], x[2]);
    let mut translation = Vector3::new(x[3], x[4], x[5]);
    if numerics.assembly == Assembly::Centred {
        // The solve is about `centre`, the pose is about the origin: `p ↦ c + R(p − c) + τ'`.
        let c = Vector3::new(centre[0], centre[1], centre[2]);
        translation += c - rotation * c;
    }
    homogeneous(&rotation, &translation)
}

/// The scalar of one point loop (D §7): `f64` for the reference path, `f32` for the GPU's.
///
/// Deliberately minimal — the loops below need six operations and two conversions, and a wider
/// trait would invite arithmetic the WGSL kernels of D §6.5 cannot mirror.
trait Real:
    Copy
    + Default
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    /// The scalar nearest `x`.
    fn narrow(x: f64) -> Self;
    /// The `f64` this scalar stands for, exactly.
    fn widen(self) -> f64;
}

impl Real for f64 {
    #[inline]
    fn narrow(x: f64) -> Self {
        x
    }
    #[inline]
    fn widen(self) -> f64 {
        self
    }
}

impl Real for f32 {
    #[inline]
    #[allow(clippy::cast_possible_truncation, reason = "the point of Precision::F32 (D §7)")]
    fn narrow(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn widen(self) -> f64 {
        f64::from(self)
    }
}

/// `[x, y, z]` rounded through `f32` and back.
fn narrow_point(p: &[f64; 3]) -> [f64; 3] {
    [f32::narrow(p[0]).widen(), f32::narrow(p[1]).widen(), f32::narrow(p[2]).widen()]
}

/// A point through a 4×4, in the given precision.
fn transform_points(points: &mut [[f64; 3]], m: &Matrix4<f64>, precision: Precision) {
    match precision {
        Precision::F64 => apply::<f64>(points, m),
        Precision::F32 => apply::<f32>(points, m),
    }
}

fn apply<T: Real>(points: &mut [[f64; 3]], m: &Matrix4<f64>) {
    let mut a = [[T::default(); 4]; 3];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = T::narrow(m[(i, j)]);
        }
    }
    for point in points {
        let p = [T::narrow(point[0]), T::narrow(point[1]), T::narrow(point[2])];
        for (i, row) in a.iter().enumerate() {
            point[i] = (row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3]).widen();
        }
    }
}

/// The mean of a point set, summed in index order (D §7).
fn centroid_of(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0; 3];
    }
    let mut sum = [0.0; 3];
    for p in points {
        for (out, value) in sum.iter_mut().zip(p) {
            *out += value;
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "cloud sizes are far below 2^53")]
    let n = points.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Eigen's `umeyama(src, dst, with_scaling = false)` over the correspondence pairs (R §7).
///
/// The mean is `sum · (1/n)` as Eigen writes it: `one_over_n` is computed once and multiplied in,
/// which is not the same double as a division by `n` — the same shape of detail as the coarse
/// score's `k/60` (step C1). The covariance's `1/n` is folded into each term here, where Eigen
/// hands it to the matrix product as its `alpha` and applies it per accumulated panel; the two
/// differ only in rounding, and cannot differ in more than that, because `Σ` and `cΣ` have the
/// same singular vectors for any `c > 0` and only `U` and `Vᵀ` leave this function. The measured
/// stage-1 rows agree with that: 0.000 deg at the median over all 71 591 fixture candidates.
fn umeyama<T: Real>(moved: &[[f64; 3]], target: &IcpTarget, corres: &[(u32, u32)]) -> Matrix4<f64> {
    if corres.is_empty() {
        return Matrix4::identity();
    }
    #[allow(clippy::cast_precision_loss, reason = "correspondence counts are far below 2^53")]
    let one_over_n = T::narrow(1.0 / corres.len() as f64);
    let (mut sum_p, mut sum_q) = ([T::default(); 3], [T::default(); 3]);
    for &(i, j) in corres {
        let (p, q) = (&moved[i as usize], &target.points[j as usize]);
        for k in 0..3 {
            sum_p[k] = sum_p[k] + T::narrow(p[k]);
            sum_q[k] = sum_q[k] + T::narrow(q[k]);
        }
    }
    let mut mean_p = [T::default(); 3];
    let mut mean_q = [T::default(); 3];
    for k in 0..3 {
        mean_p[k] = sum_p[k] * one_over_n;
        mean_q[k] = sum_q[k] * one_over_n;
    }
    let mut sigma = [[T::default(); 3]; 3];
    for &(i, j) in corres {
        let (p, q) = (&moved[i as usize], &target.points[j as usize]);
        let mut a = [T::default(); 3];
        let mut b = [T::default(); 3];
        for k in 0..3 {
            a[k] = T::narrow(p[k]) - mean_p[k];
            b[k] = (T::narrow(q[k]) - mean_q[k]) * one_over_n;
        }
        for (r, row) in sigma.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = *cell + b[r] * a[c];
            }
        }
    }
    let mut m = Matrix3::zeros();
    for (r, row) in sigma.iter().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            m[(r, c)] = cell.widen();
        }
    }
    // `SVD::new` sorts the singular values descending, which is what puts the reflection fix on
    // the smallest of them, as Eigen's `JacobiSVD` does.
    let svd = m.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else { return Matrix4::identity() };
    let mut d = Matrix3::identity();
    if u.determinant() * v_t.determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rotation = u * d * v_t;
    let mean_p = Vector3::new(mean_p[0].widen(), mean_p[1].widen(), mean_p[2].widen());
    let mean_q = Vector3::new(mean_q[0].widen(), mean_q[1].widen(), mean_q[2].widen());
    homogeneous(&rotation, &(mean_q - rotation * mean_p))
}

/// R §7's `JTJ` and `JTr`, accumulated in source order with weight 1 (D §7).
///
/// `centre` is subtracted from the source point in the *rotational* half of the Jacobian row only
/// — the residual is a difference and does not move — which is D §6.5's kernel line for line.
#[allow(clippy::many_single_char_names, reason = "p, q, n and c are R §7's own names")]
fn normal_equations<T: Real>(
    moved: &[[f64; 3]],
    target: &IcpTarget,
    corres: &[(u32, u32)],
    centre: [f64; 3],
) -> ([[f64; 6]; 6], [f64; 6]) {
    let c = [T::narrow(centre[0]), T::narrow(centre[1]), T::narrow(centre[2])];
    let mut jtj = [[T::default(); 6]; 6];
    let mut jtr = [T::default(); 6];
    for &(i, j) in corres {
        let p = &moved[i as usize];
        let q = &target.points[j as usize];
        let n = &target.normals[j as usize];
        let p = [T::narrow(p[0]), T::narrow(p[1]), T::narrow(p[2])];
        let q = [T::narrow(q[0]), T::narrow(q[1]), T::narrow(q[2])];
        let n = [T::narrow(n[0]), T::narrow(n[1]), T::narrow(n[2])];
        let pc = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let row = [
            pc[1] * n[2] - pc[2] * n[1],
            pc[2] * n[0] - pc[0] * n[2],
            pc[0] * n[1] - pc[1] * n[0],
            n[0],
            n[1],
            n[2],
        ];
        let r = (p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1] + (p[2] - q[2]) * n[2];
        for k in 0..6 {
            for l in 0..=k {
                jtj[k][l] = jtj[k][l] + row[k] * row[l];
            }
            jtr[k] = jtr[k] + row[k] * r;
        }
    }
    let mut a = [[0.0; 6]; 6];
    for k in 0..6 {
        for l in 0..=k {
            a[k][l] = jtj[k][l].widen();
            a[l][k] = a[k][l];
        }
    }
    (a, jtr.map(Real::widen))
}

/// Eigen's pivot order for a symmetric 6×6 `LDLT`, as a permutation of the original indices.
///
/// `ldlt_inplace` chooses each pivot with
/// `mat.diagonal().tail(size - k).cwiseAbs().maxCoeff(&index)` and then **swaps** that entry into
/// position `k`. Two details of that decide the permutation on a tie, and neither is a descending
/// sort:
///
/// * `maxCoeff` keeps the first index it sees, because its visitor tests `value > res` — but the
///   indices it is walking are the *current* ones, and every earlier step has already moved entries
///   around by transposition;
/// * the swap is a transposition rather than a rotation, so the entry displaced out of position `k`
///   lands where the pivot came from, not one place further down.
///
/// A selection sort by transpositions is not stable: on the diagonal `[3, 3, 5]` Eigen swaps
/// positions 0 and 2 and ends at the original indices `[2, 1, 0]`, while a stable descending sort
/// gives `[2, 0, 1]`. Two exactly equal diagonal entries of a real 6×6 normal-equations matrix are
/// measure-zero and nothing on the fixtures has ever produced one — but the two orders factorise
/// different matrices when one appears, so this reproduces Eigen's loop rather than a sort that
/// agrees with it almost everywhere.
fn eigen_pivots(a: &[[f64; 6]; 6]) -> [usize; 6] {
    let mut perm = [0_usize, 1, 2, 3, 4, 5];
    for k in 0..6 {
        let mut best = k;
        for i in k + 1..6 {
            // Strictly greater: `maxCoeff` keeps the first of several equal maxima.
            if a[perm[i]][perm[i]].abs() > a[perm[best]][perm[best]].abs() {
                best = i;
            }
        }
        perm.swap(k, best);
    }
    perm
}

/// Eigen's `LDLT` on a symmetric 6×6, which is the solver Open3D's point-to-plane step uses.
///
/// Eigen pivots on the largest remaining `|diagonal|`, and its left-looking factorisation never
/// touches a trailing diagonal entry before that entry has been chosen — `mat.coeffRef(k, k)` is
/// decremented at step `k`, *after* `k` has been selected, and the rank-1 update below it touches
/// the column and not the diagonal. The values `ldlt_inplace` compares are therefore the original
/// diagonal's, and the permutation can be computed up front instead of interleaved with the
/// elimination ([`eigen_pivots`]). What follows is then the plain unpivoted factorisation on the
/// permuted matrix, and the solve applies Eigen's pseudo-inverse of `D`: a component whose pivot is
/// not above the smallest normal double is set to zero rather than divided by, which is what keeps
/// a rank-deficient system — two flat surfaces have three unconstrained degrees of freedom — from
/// returning infinities.
///
/// Returns `None` when the result is not finite, which R §7 does not describe because Eigen's
/// solve always "succeeds"; an identity update is the caller's answer to it.
#[allow(clippy::many_single_char_names, reason = "the names of a linear solve: A x = b, LDLᵀ")]
fn solve_ldlt(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    let perm = eigen_pivots(a);
    let mut m = [[0.0; 6]; 6];
    for (k, row) in m.iter_mut().enumerate() {
        for (l, cell) in row.iter_mut().enumerate() {
            *cell = a[perm[k]][perm[l]];
        }
    }
    for k in 0..6 {
        if k > 0 {
            let mut temp = [0.0; 6];
            for j in 0..k {
                temp[j] = m[j][j] * m[k][j];
            }
            let mut diagonal = m[k][k];
            for j in 0..k {
                diagonal -= m[k][j] * temp[j];
            }
            m[k][k] = diagonal;
            for row in m.iter_mut().skip(k + 1) {
                let mut below = row[k];
                for j in 0..k {
                    below -= row[j] * temp[j];
                }
                row[k] = below;
            }
        }
        let pivot = m[k][k];
        if pivot != 0.0 {
            for row in m.iter_mut().skip(k + 1) {
                row[k] /= pivot;
            }
        }
    }
    let mut y = [0.0; 6];
    for (k, out) in y.iter_mut().enumerate() {
        *out = b[perm[k]];
    }
    for i in 0..6 {
        let mut value = y[i];
        for j in 0..i {
            value -= m[i][j] * y[j];
        }
        y[i] = value;
    }
    for i in 0..6 {
        let pivot = m[i][i];
        y[i] = if pivot.abs() > f64::MIN_POSITIVE { y[i] / pivot } else { 0.0 };
    }
    for i in (0..6).rev() {
        let mut value = y[i];
        for j in i + 1..6 {
            value -= m[j][i] * y[j];
        }
        y[i] = value;
    }
    let mut x = [0.0; 6];
    for (k, &value) in y.iter().enumerate() {
        x[perm[k]] = value;
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// R §7's `Rz(γ)·Ry(β)·Rx(α)`.
///
/// **This is not the expression Open3D evaluates, and R §12.1 carries the row that licenses it.**
/// `utility::TransformationMatrixFromPoseVector` writes
/// `(AngleAxisd(x₂, UnitZ()) * AngleAxisd(x₁, UnitY()) * AngleAxisd(x₀, UnitX())).matrix()`, and
/// Eigen's `operator*` on two `AngleAxis` converts both to quaternions, multiplies those, and
/// converts the product to a matrix once at the end. The two are the same rotation and they are
/// not the same arithmetic: measured over a sweep of the angles an ICP update produces, the worst
/// entry differs by 3.3e-16 — **1.5 ULP of 1** — which moves the furthest sample of these scans
/// (885 units from the origin) by 4.4e-13 units, **1.9e-13 t** on the thinnest wall of the
/// benchmark (`notes/2026-09-07-x-phase1c-findings.md` §4, and the test below).
///
/// The matrix product is kept because it is the form R §7 states and because the difference is an
/// ULP either way; what changed is that R §12.1 now says so instead of R §7 implying the two are
/// the same expression.
fn euler_zyx(alpha: f64, beta: f64, gamma: f64) -> Matrix3<f64> {
    let (sa, ca) = alpha.sin_cos();
    let (sb, cb) = beta.sin_cos();
    let (sg, cg) = gamma.sin_cos();
    let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, ca, -sa, 0.0, sa, ca);
    let ry = Matrix3::new(cb, 0.0, sb, 0.0, 1.0, 0.0, -sb, 0.0, cb);
    let rz = Matrix3::new(cg, -sg, 0.0, sg, cg, 0.0, 0.0, 0.0, 1.0);
    rz * ry * rx
}

/// The 3×3 rotation of a pose, named so that a crate without a `nalgebra` dependency of its own
/// can still build one (the CLI's `gpu-check`).
pub type Rotation = Matrix3<f64>;

/// The translation of a pose, named for the same reason.
pub type Translation = Vector3<f64>;

/// A 4×4 from a rotation and a translation.
pub fn homogeneous(rotation: &Matrix3<f64>, translation: &Vector3<f64>) -> Matrix4<f64> {
    let mut m = Matrix4::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation);
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(translation);
    m
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "a fitness of exactly zero is the assertion")]

    use super::{
        Assembly, IcpTarget, Numerics, Options, Precision, eigen_pivots, euler_zyx, homogeneous,
        register, solve_ldlt,
    };
    use nalgebra::{Matrix3, Matrix4, Vector3};

    fn diagonal(d: [f64; 6]) -> [[f64; 6]; 6] {
        let mut a = [[0.0; 6]; 6];
        for (k, value) in d.into_iter().enumerate() {
            a[k][k] = value;
        }
        a
    }

    /// The pivot order is Eigen's selection sort by transpositions, not a stable sort.
    ///
    /// On distinct diagonal entries the two agree and this is a descending sort by `|d|`. On a tie
    /// they do not, and the case is written out because it is the one the port used to get wrong:
    /// Eigen swaps the largest entry into place, which sends the entry it displaced to the *end* of
    /// the tail rather than one step down it.
    #[test]
    fn the_pivot_order_is_eigens_transpositions_and_not_a_stable_sort() {
        assert_eq!(eigen_pivots(&diagonal([1.0, 5.0, 3.0, -9.0, 2.0, 4.0])), [3, 1, 5, 2, 4, 0]);
        assert_eq!(eigen_pivots(&diagonal([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])), [0, 1, 2, 3, 4, 5]);
        // `[3, 3, 5, 0, 0, 0]`: Eigen swaps positions 0 and 2, so the original index 0 ends last
        // among the three; a stable descending sort would have given `[2, 0, 1, 3, 4, 5]`.
        assert_eq!(eigen_pivots(&diagonal([3.0, 3.0, 5.0, 0.0, 0.0, 0.0])), [2, 1, 0, 3, 4, 5]);
        // All equal: no transposition fires, because `maxCoeff` keeps the first of equal maxima.
        assert_eq!(eigen_pivots(&diagonal([2.0; 6])), [0, 1, 2, 3, 4, 5]);
        // The magnitude decides, not the sign.
        assert_eq!(eigen_pivots(&diagonal([-7.0, 1.0, 0.0, 0.0, 0.0, 0.0]))[0], 0);
    }

    /// A bumpy patch of surface: `n×n` points of `z = f(x, y)` with the exact unit normals.
    fn patch(n: usize, step: f64, offset: f64) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let mut points = Vec::with_capacity(n * n);
        let mut normals = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                #[allow(clippy::cast_precision_loss, reason = "a small grid index")]
                let (x, y) = (offset + i as f64 * step, offset + j as f64 * step);
                let z = 0.3 * (0.8 * x).sin() + 0.2 * (1.1 * y).cos();
                let (dx, dy) = (0.24 * (0.8 * x).cos(), -0.22 * (1.1 * y).sin());
                let norm = (dx * dx + dy * dy + 1.0).sqrt();
                points.push([x, y, z]);
                normals.push([-dx / norm, -dy / norm, 1.0 / norm]);
            }
        }
        (points, normals)
    }

    /// A rigid transform from an axis-angle and a translation.
    fn rigid(axis: [f64; 3], angle: f64, translation: [f64; 3]) -> Matrix4<f64> {
        let axis = nalgebra::Unit::new_normalize(Vector3::new(axis[0], axis[1], axis[2]));
        let rotation = nalgebra::Rotation3::from_axis_angle(&axis, angle);
        homogeneous(
            rotation.matrix(),
            &Vector3::new(translation[0], translation[1], translation[2]),
        )
    }

    /// The worst distance between two poses over a set of points, and the angle between them.
    fn pose_gap(a: &Matrix4<f64>, b: &Matrix4<f64>, points: &[[f64; 3]]) -> (f64, f64) {
        let mut ra = Matrix3::zeros();
        let mut rb = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                ra[(i, j)] = a[(i, j)];
                rb[(i, j)] = b[(i, j)];
            }
        }
        let trace = (ra.transpose() * rb).trace();
        let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
        let mut worst = 0.0_f64;
        for p in points {
            let q = Vector3::new(p[0], p[1], p[2]);
            let pa = ra * q + a.fixed_view::<3, 1>(0, 3);
            let pb = rb * q + b.fixed_view::<3, 1>(0, 3);
            worst = worst.max((pa - pb).norm());
        }
        (angle, worst)
    }

    /// Point-to-point recovers a known rigid transform exactly: the same cloud, moved, comes back
    /// to where it was in one iteration, and the ICP's own fitness says every point matched.
    #[test]
    fn point_to_point_recovers_a_known_rigid_transform() {
        let (points, _) = patch(20, 0.5, -5.0);
        let target = IcpTarget::new(points.clone(), Vec::new());
        let truth = rigid([0.3, -0.7, 0.6], 0.04, [0.11, -0.07, 0.05]);
        // The source is the target seen from the other frame, so `truth` is the answer.
        let inverse = truth.try_inverse().expect("a rigid transform inverts");
        let source: Vec<[f64; 3]> = points
            .iter()
            .map(|p| {
                let q = inverse * nalgebra::Vector4::new(p[0], p[1], p[2], 1.0);
                [q[0], q[1], q[2]]
            })
            .collect();

        let out =
            register(&source, &target, &Matrix4::identity(), &Options::point_to_point(1.0, 20));
        let (angle, distance) = pose_gap(&out.transform, &truth, &source);
        assert!(angle < 1e-9, "{angle:e} degrees");
        assert!(distance < 1e-9, "{distance:e}");
        assert!((out.fitness - 1.0).abs() < 1e-15, "{}", out.fitness);
        assert!(out.inlier_rmse < 1e-9, "{:e}", out.inlier_rmse);
        assert!(out.converged && out.iterations <= 3, "{out:?}");
    }

    /// Point-to-plane on a pair of planes converges to the plane: the offset and the tilt go, and
    /// the three degrees of freedom the geometry does not constrain stay where they were.
    ///
    /// The system here is genuinely rank-3 — sliding a plane inside itself and turning it about
    /// its own normal change no residual — which is the case Eigen's pivoted LDLT survives and a
    /// plain Cholesky does not.
    #[test]
    fn point_to_plane_converges_to_a_plane_pair() {
        let mut points = Vec::new();
        let mut normals = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                #[allow(clippy::cast_precision_loss, reason = "a small grid index")]
                let (x, y) = (-5.0 + f64::from(i) * 0.25, -5.0 + f64::from(j) * 0.25);
                points.push([x, y, 0.0]);
                normals.push([0.0, 0.0, 1.0]);
            }
        }
        let target = IcpTarget::new(points.clone(), normals);
        // The source plane sits 0.35 above and tilted by 0.05 rad about x.
        let placement = rigid([1.0, 0.0, 0.0], 0.05, [0.0, 0.0, 0.35]);
        let source: Vec<[f64; 3]> = points
            .iter()
            .map(|p| {
                let q = placement * nalgebra::Vector4::new(p[0] + 0.13, p[1] - 0.07, p[2], 1.0);
                [q[0], q[1], q[2]]
            })
            .collect();

        let out =
            register(&source, &target, &Matrix4::identity(), &Options::point_to_plane(2.0, 30));
        let mut worst = 0.0_f64;
        for p in &source {
            let q = out.transform * nalgebra::Vector4::new(p[0], p[1], p[2], 1.0);
            worst = worst.max(q[2].abs());
        }
        assert!(worst < 1e-9, "the source plane is {worst:e} off the target plane");
        assert!(out.converged, "{out:?}");
        assert!((out.fitness - 1.0).abs() < 1e-15);
    }

    /// The ladder narrows: a wide radius pulls a badly placed cloud in, a narrow one refines it,
    /// and the fitness rises as the correspondence set fills.
    #[test]
    fn a_narrower_radius_keeps_fewer_correspondences() {
        let (points, normals) = patch(30, 0.3, -4.0);
        let target = IcpTarget::new(points.clone(), normals);
        let init = rigid([0.0, 0.0, 1.0], 0.05, [0.35, -0.2, 0.15]);

        let wide = register(&points, &target, &init, &Options::point_to_plane(2.0, 30));
        let narrow = register(&points, &target, &init, &Options::point_to_plane(0.05, 30));
        assert!((wide.fitness - 1.0).abs() < 1e-12, "{}", wide.fitness);
        assert!(narrow.fitness < wide.fitness, "{} {}", narrow.fitness, wide.fitness);
        assert!(narrow.correspondences < points.len());
        // Refining the wide answer at the narrow radius is the ladder, and it lands on the truth.
        let refined =
            register(&points, &target, &wide.transform, &Options::point_to_plane(0.05, 30));
        let (angle, distance) = pose_gap(&refined.transform, &Matrix4::identity(), &points);
        assert!(angle < 1e-6 && distance < 1e-6, "{angle:e} {distance:e}");
    }

    /// A source point exactly `d` from its nearest target is not a correspondence, and one an ulp
    /// closer is — Open3D's strict radius test, measured against Open3D 0.19.
    #[test]
    fn the_correspondence_radius_is_strict() {
        let target = IcpTarget::new(vec![[0.0; 3], [100.0, 0.0, 0.0]], Vec::new());
        let d = 1.3;
        let at = |x: f64| {
            register(
                &[[x, 0.0, 0.0]],
                &target,
                &Matrix4::identity(),
                &Options { max_iteration: 0, ..Options::point_to_point(d, 0) },
            )
            .fitness
        };
        assert!((at(f64::from_bits(d.to_bits() - 1)) - 1.0).abs() < 1e-15, "an ulp inside");
        assert!(at(d).abs() < 1e-15, "exactly at the radius is a miss");
        assert!(at(f64::from_bits(d.to_bits() + 1)).abs() < 1e-15, "an ulp outside");
    }

    /// An empty correspondence set, an empty cloud and a non-positive radius all leave the pose
    /// alone rather than dividing by zero.
    #[test]
    fn nothing_to_match_leaves_the_pose_alone() {
        let (points, normals) = patch(10, 0.4, -2.0);
        let target = IcpTarget::new(points.clone(), normals);
        let init = rigid([0.0, 1.0, 0.0], 0.2, [50.0, 0.0, 0.0]);

        for options in [Options::point_to_point(0.01, 30), Options::point_to_plane(0.01, 30)] {
            let out = register(&points, &target, &init, &options);
            assert_eq!(out.transform, init, "no correspondence, no update");
            assert!(out.fitness == 0.0 && out.inlier_rmse == 0.0, "{out:?}");
        }
        let out = register(&points, &target, &init, &Options::point_to_plane(-1.0, 30));
        assert_eq!(out.transform, init);
        let empty = IcpTarget::new(Vec::new(), Vec::new());
        assert!(empty.is_empty() && !empty.has_normals() && empty.points().is_empty());
        assert_eq!(
            register(&points, &empty, &init, &Options::point_to_plane(1.0, 30)).transform,
            init
        );
        assert_eq!(
            register(&[], &target, &init, &Options::point_to_plane(1.0, 30)).transform,
            init
        );
    }

    /// `Assembly::Centred` is a change of variables of the same Gauss–Newton step, so in `f64` it
    /// lands on the same pose to the linearisation error D §7 quotes, and `Precision::F32` stays
    /// far inside D §10.2's 0.05° / 0.01 t.
    #[test]
    fn the_numerics_knobs_move_the_pose_by_far_less_than_the_gate() {
        let (points, normals) = patch(30, 0.3, 400.0);
        let init = rigid([0.2, 0.5, 0.8], 0.03, [0.2, -0.15, 0.1]);
        let reference = IcpTarget::new(points.clone(), normals.clone());
        let base = register(&points, &reference, &init, &Options::point_to_plane(1.0, 30));

        for numerics in [
            Numerics { precision: Precision::F64, assembly: Assembly::Centred },
            Numerics { precision: Precision::F32, assembly: Assembly::World },
            Numerics::GPU,
        ] {
            let target = IcpTarget::for_precision(&points, &normals, numerics.precision);
            let out =
                register(&points, &target, &init, &Options::point_to_plane(1.0, 30).with(numerics));
            let (angle, distance) = pose_gap(&out.transform, &base.transform, &points);
            assert!(angle < 0.05, "{numerics:?}: {angle:e} degrees");
            assert!(distance < 0.01, "{numerics:?}: {distance:e}");
        }
    }

    /// Eigen's own composition, written out: three `AngleAxis` as quaternions, one Hamilton
    /// product each, and `Quaternion::toRotationMatrix` at the end.
    ///
    /// This is what `utility::TransformationMatrixFromPoseVector` evaluates, and it is transcribed
    /// here — the half-angle sines, Eigen's `quat_product` term order, and its `1 − (tyy + tzz)`
    /// matrix formulas — so that the deviation R §12.1 licenses can be measured rather than
    /// asserted.
    #[allow(clippy::many_single_char_names, reason = "a quaternion's components are w, x, y, z")]
    fn eigen_quaternion_zyx(alpha: f64, beta: f64, gamma: f64) -> Matrix3<f64> {
        // `Quaternion(AngleAxis)`: w = cos(θ/2), vec = sin(θ/2)·axis.
        let axis_angle = |half: f64, axis: usize| {
            let (s, c) = half.sin_cos();
            let mut q = [c, 0.0, 0.0, 0.0];
            q[axis + 1] = s;
            q
        };
        // Eigen's `quat_product`, term for term.
        let product = |a: [f64; 4], b: [f64; 4]| {
            [
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
                a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
            ]
        };
        let qz = axis_angle(0.5 * gamma, 2);
        let qy = axis_angle(0.5 * beta, 1);
        let qx = axis_angle(0.5 * alpha, 0);
        let q = product(product(qz, qy), qx);
        let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
        let (tx, ty, tz) = (2.0 * x, 2.0 * y, 2.0 * z);
        let (twx, twy, twz) = (tx * w, ty * w, tz * w);
        let (txx, txy, txz) = (tx * x, ty * x, tz * x);
        let (tyy, tyz, tzz) = (ty * y, tz * y, tz * z);
        Matrix3::new(
            1.0 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1.0 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1.0 - (txx + tyy),
        )
    }

    /// R §7's matrix product against Open3D's quaternion composition (defect D8, PMC-18).
    ///
    /// The angles an ICP update produces run from a few tenths of a radian on the first iteration
    /// of a coarse rung down to 1e-12 on the last, so the sweep covers eleven decades and both
    /// signs, and the worst is reported in the units the parity rows use: a displacement of the
    /// furthest sample of these scans, 885 units from the origin, in wall thicknesses of the
    /// thinnest benchmark wall (2.36 on `pot_G`).
    #[test]
    fn the_quaternion_composition_and_the_matrix_product_agree_to_a_few_ulp() {
        const RADIUS: f64 = 885.0;
        const THINNEST_WALL: f64 = 2.36;
        let mut worst_entry = 0.0_f64;
        let mut worst_move = 0.0_f64;
        for k in 0..11 {
            let scale = 10.0_f64.powi(-k);
            for (i, j, l) in
                [(1, 2, 3), (7, -3, 11), (-5, 9, -2), (13, 17, -19), (1, -1, 1), (31, -7, 23)]
            {
                let (a, b, g) = (
                    f64::from(i) * 0.1 * scale,
                    f64::from(j) * 0.1 * scale,
                    f64::from(l) * 0.1 * scale,
                );
                let matrix = euler_zyx(a, b, g);
                let quaternion = eigen_quaternion_zyx(a, b, g);
                let delta = matrix - quaternion;
                worst_entry = worst_entry.max(delta.abs().max());
                // The furthest a point at `RADIUS` can be moved by the difference of the two
                // rotations is the largest singular value of the difference; its Frobenius norm
                // bounds that and needs no decomposition.
                worst_move = worst_move.max(delta.norm() * RADIUS);
            }
        }
        println!(
            "D8: worst entry {:e} ({:.1} ULP of 1), worst move {:e} units = {:e} t",
            worst_entry,
            worst_entry / f64::EPSILON,
            worst_move,
            worst_move / THINNEST_WALL
        );
        assert!(worst_entry < 4.0 * f64::EPSILON, "worst entry {worst_entry:e}");
        assert!(
            worst_move / THINNEST_WALL < 1e-12,
            "worst displacement {:e} t",
            worst_move / THINNEST_WALL
        );
    }

    /// The Euler composition is `Rz·Ry·Rx` and not any other order, and the 6×6 solve is a solve.
    #[test]
    fn the_update_composes_z_then_y_then_x() {
        let r = euler_zyx(0.3, 0.0, 0.0);
        assert!((r[(1, 1)] - 0.3_f64.cos()).abs() < 1e-15);
        assert!((r[(2, 1)] - 0.3_f64.sin()).abs() < 1e-15);
        let composed = euler_zyx(0.11, 0.22, 0.33);
        let expected =
            euler_zyx(0.0, 0.0, 0.33) * euler_zyx(0.0, 0.22, 0.0) * euler_zyx(0.11, 0.0, 0.0);
        assert!((composed - expected).abs().max() < 1e-14);

        // A well-conditioned system: the solve reproduces the right-hand side.
        let mut a = [[0.0; 6]; 6];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                #[allow(clippy::cast_precision_loss, reason = "small indices")]
                let (x, y) = ((i + 1) as f64, (j + 1) as f64);
                *cell = 1.0 / (x + y) + if i == j { 3.0 } else { 0.0 };
            }
        }
        let b = [1.0, -2.0, 0.5, 4.0, -1.5, 0.25];
        let x = solve_ldlt(&a, &b).expect("a positive-definite 6×6");
        for i in 0..6 {
            let mut row = 0.0;
            for j in 0..6 {
                row += a[i][j] * x[j];
            }
            assert!((row - b[i]).abs() < 1e-12, "row {i}: {row} against {}", b[i]);
        }

        // A singular one: the unconstrained components come back as zeros, not as infinities.
        let mut singular = [[0.0; 6]; 6];
        singular[0][0] = 2.0;
        singular[1][1] = 4.0;
        let x = solve_ldlt(&singular, &[6.0, 8.0, 1.0, 0.0, 0.0, 0.0]).expect("finite");
        assert!((x[0] - 3.0).abs() < 1e-15 && (x[1] - 2.0).abs() < 1e-15);
        assert!(x[2..].iter().all(|v| *v == 0.0), "{x:?}");
    }
}
