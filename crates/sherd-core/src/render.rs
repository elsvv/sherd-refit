//! Preview images (R §11.5): a software point renderer — no GPU, no display, no font stack.
//!
//! Open3D's offscreen renderer does not exist on macOS, so the reference draws its previews with
//! numpy: every fragment is sampled on its faces, the samples are shaded by their face normal and
//! splatted into a z-buffer as 3×3 blocks, and the views are laid side by side into one PNG. This
//! module is that, line by line, because the parity harness compares the two images **pixel for
//! pixel** and a preview is the only stage of the pipeline whose output is an image rather than a
//! number.
//!
//! # The four arithmetic details a pixel depends on
//!
//! * **The rotation into view space is a matmul and therefore fused.** `(V − centre) @ R` and
//!   `N @ R` are numpy `(n,3) @ (3,3)` products, which reach OpenBLAS's `dgemm` and accumulate with
//!   a fused multiply-add; `Nn · light` is a `(n,3) @ (3,)` product, which does **not**
//!   ([`types::apply_transform_fused`](crate::types::apply_transform_fused) has the measurement).
//!   Getting one of the two wrong moves a point across a pixel boundary about once in a hundred
//!   thousand, which is exactly the size of the difference this module exists to avoid.
//! * **`np.round` is round-half-to-**even**,** not half-away-from-zero: `f64::round_ties_even`,
//!   never `f64::round`.
//! * **The z-buffer is `f32` and the comparison is `f64`.** A candidate depth is compared against
//!   the *narrowed* value already stored, then stored narrowed itself.
//! * **Within one of the nine offsets, the last of the deepest points wins.** The reference's
//!   `np.lexsort((dd, lin))` is stable and its `last` mask takes the final entry of each pixel's
//!   run, so a tie between two points at the same depth goes to the later one in concatenation
//!   order.
//!
//! # The label (PMC-20)
//!
//! `render_views` draws a caption at `(10, 10)` in white. The reference draws it with PIL's
//! `ImageDraw.text` and `ImageFont.load_default()`, which in Pillow 12 is an anti-aliased
//! FreeType face; reproducing it would mean embedding that font and FreeType's rasteriser. The
//! port draws the same text with [`FONT`], its own 5×7 bitmap font, upper-case only. The pixels
//! under the caption are therefore the one part of a preview the two implementations do not share,
//! and the parity harness measures how large that part is instead of asserting it away: it
//! compares against the reference's own **unlabelled** render and reports the labelled region's
//! size as its own row.
//!
//! Filled in by phase-1d step D2.

use std::path::Path;

use nalgebra::Matrix4;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::error::{Error, Result};

/// R §11.5's ten fragment colours, in order.
pub const PALETTE: [[f64; 3]; 10] = [
    [0.78, 0.78, 0.78],
    [0.95, 0.55, 0.25],
    [0.40, 0.70, 0.95],
    [0.50, 0.85, 0.50],
    [0.90, 0.80, 0.40],
    [0.80, 0.50, 0.90],
    [0.35, 0.85, 0.85],
    [0.90, 0.45, 0.55],
    [0.60, 0.60, 0.95],
    [0.75, 0.60, 0.40],
];

/// The names R §11.5's caption gives those ten colours.
pub const COLOUR_NAMES: [&str; 10] =
    ["grey", "orange", "blue", "green", "yellow", "purple", "cyan", "pink", "indigo", "tan"];

/// The background grey every preview starts from.
pub const BACKGROUND: f32 = 0.16;

/// The direction R §11.5 lights a preview from, before normalisation.
pub const LIGHT: [f64; 3] = [0.3, 0.4, 1.0];

/// The fraction of the shorter image axis the whole point set is scaled to fill.
pub const FILL: f64 = 0.9;

/// How a splat is coloured: one colour for the whole point set, or one per point.
///
/// The reference tiles the group colour into an `(n, 3)` array; keeping the uniform case as a
/// single triple is the same arithmetic and saves 6 MB per fragment on a 250 000-sample preview.
#[derive(Clone, Debug)]
pub enum Paint {
    /// Every point the same colour — R §11.5's `PALETTE[i mod 10]`.
    Uniform([f64; 3]),
    /// One colour per point — the segmentation preview's grey-or-red.
    PerPoint(Vec<[f64; 3]>),
}

impl Paint {
    /// The colour of one point.
    #[inline]
    pub fn colour(&self, i: usize) -> [f64; 3] {
        match self {
            Self::Uniform(c) => *c,
            Self::PerPoint(c) => c[i],
        }
    }
}

/// One fragment's contribution to a preview: where its samples are, which way they face and what
/// colour they are.
#[derive(Clone, Debug)]
pub struct Splat {
    /// The samples, already in world coordinates.
    pub points: Vec<[f64; 3]>,
    /// Their face normals, already rotated into world coordinates.
    pub normals: Vec<[f64; 3]>,
    /// Their colour.
    pub paint: Paint,
}

/// A direction to look along and a direction to call up (R §11.5's `(eye_dir, up)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// `eye_dir`: the camera's `z`, before normalisation.
    pub eye: [f64; 3],
    /// `up`: what fixes the roll.
    pub up: [f64; 3],
}

/// An 8-bit RGB image, row major, as `Image.fromarray` holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `3 · width · height` bytes, red first.
    pub pixels: Vec<u8>,
}

impl Rgb {
    /// A new image filled with one colour.
    pub fn filled(width: usize, height: usize, colour: [u8; 3]) -> Self {
        let mut pixels = Vec::with_capacity(3 * width * height);
        for _ in 0..width * height {
            pixels.extend_from_slice(&colour);
        }
        Self { width, height, pixels }
    }

    /// The three bytes of one pixel.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let at = 3 * (y * self.width + x);
        [self.pixels[at], self.pixels[at + 1], self.pixels[at + 2]]
    }

    /// Writes one pixel.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, colour: [u8; 3]) {
        let at = 3 * (y * self.width + x);
        self.pixels[at..at + 3].copy_from_slice(&colour);
    }

    /// `np.concatenate(imgs, 1)`: the images side by side, left to right.
    ///
    /// Every image must have the same height, which every caller guarantees by construction.
    pub fn strip(images: &[Self]) -> Self {
        let height = images.first().map_or(0, |i| i.height);
        let width: usize = images.iter().map(|i| i.width).sum();
        let mut out = Self { width, height, pixels: vec![0; 3 * width * height] };
        let mut x0 = 0;
        for image in images {
            for y in 0..height {
                for x in 0..image.width {
                    out.set(x0 + x, y, image.pixel(x, y));
                }
            }
            x0 += image.width;
        }
        out
    }

    /// Writes the image as a PNG.
    pub fn write_png(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a preview is a few thousand pixels wide"
        )]
        let buffer = image::RgbImage::from_raw(
            self.width as u32,
            self.height as u32,
            self.pixels.clone(),
        )
        .ok_or_else(|| Error::write(path, "the pixel buffer does not match the image size"))?;
        buffer.save(path).map_err(|e| Error::write(path, e))
    }
}

/// R §11.5's `principal_views`: four views along the point set's own axes.
///
/// `X = V − mean`, then the eigenvectors of `XᵀX` ascending by eigenvalue: `u` is the direction
/// the points vary in *least* — the sherd's own normal — `e2` the middle one and `e1` the largest.
/// The views are `(u, e2)`, `(−u, e2)`, `(e1 + 0.35u, u)` and `(e2 + 0.35u, u)`.
///
/// **PMC-10**: an eigenvector's sign is whatever the library returns, and numpy's `eigh` and
/// `nalgebra`'s `SymmetricEigen` need not agree. The port fixes it by convention — the entry of
/// largest magnitude is made positive, ties going to the earlier axis — which makes the port's own
/// previews reproducible without making them the reference's. The parity harness therefore
/// compares pixels only at the reference's own views, and reports the port's own views beside
/// them.
pub fn principal_views(points: &[[f64; 3]]) -> [View; 4] {
    let axes = principal_axes(points);
    let (u, e2, e1) = (axes[0], axes[1], axes[2]);
    let combine =
        |a: [f64; 3], b: [f64; 3], k: f64| [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
    [
        View { eye: u, up: e2 },
        View { eye: [-u[0], -u[1], -u[2]], up: e2 },
        View { eye: combine(e1, u, 0.35), up: u },
        View { eye: combine(e2, u, 0.35), up: u },
    ]
}

/// The three eigenvectors of the centred scatter matrix, **ascending** by eigenvalue.
///
/// The mean is numpy's own axis-0 reduction, row by row
/// ([`column_mean`](crate::mesh::geometry::column_mean); V4-D10 — this comment used to claim
/// [`pairwise_sum`](crate::mesh::geometry::pairwise_sum) was numpy's answer for this shape, and it
/// is not). `XᵀX` is then accumulated with `pairwise_sum` per entry, which is neither the
/// reference's `X.T @ X` — a blocked `dgemm` — nor anything numpy does here, so the last bits of
/// the scatter matrix are the port's own. PMC-10 covers the consequence.
pub fn principal_axes(points: &[[f64; 3]]) -> [[f64; 3]; 3] {
    use crate::mesh::geometry::{column_mean, pairwise_sum};

    if points.is_empty() {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let mean = column_mean(points);
    let mut scatter = nalgebra::Matrix3::zeros();
    for i in 0..3 {
        for j in i..3 {
            let terms: Vec<f64> =
                points.iter().map(|p| (p[i] - mean[i]) * (p[j] - mean[j])).collect();
            let value = pairwise_sum(&terms);
            scatter[(i, j)] = value;
            scatter[(j, i)] = value;
        }
    }
    let eigen = nalgebra::SymmetricEigen::new(scatter);
    let mut order = [0_usize, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    order.map(|k| {
        let mut v =
            [eigen.eigenvectors[(0, k)], eigen.eigenvectors[(1, k)], eigen.eigenvectors[(2, k)]];
        // PMC-10's convention: the largest component positive, ties to the earlier axis.
        let lead = (0..3).max_by(|&a, &b| v[a].abs().total_cmp(&v[b].abs())).unwrap_or(0);
        if v[lead] < 0.0 {
            v = [-v[0], -v[1], -v[2]];
        }
        v
    })
}

/// R §11.5's `render_views`: every view of one point set, side by side.
///
/// `width` and `height` are one view's; the strip is `views.len()` of them. The framing — centre,
/// extent and scale — is computed once over **all** the points and shared by every view, which is
/// what makes the views comparable with each other.
pub fn render_views(meshes: &[Splat], views: &[View], width: usize, height: usize) -> Rgb {
    let frame = Frame::of(meshes, width, height);
    // One view is one image over its own z-buffer, and the strip is assembled by index, so the
    // views are `rayon`'s to spread and no pixel depends on which thread drew it (D §7). It is
    // worth spreading because R §11.5 is the largest wall-clock item outside the matcher once
    // R §11.4's meshes are written in parallel: 3.37 s of synthetic 20's 19.50 at 45c93da, on
    // 2.25 core-seconds of work (`notes/2026-09-07-e2-tuning.md` §8).
    let images: Vec<Rgb> = views.par_iter().map(|v| splat(meshes, *v, &frame)).collect();
    Rgb::strip(&images)
}

/// The framing R §11.5 shares between the views of one image.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// `(min + max) / 2` of every point.
    pub centre: [f64; 3],
    /// `‖max − min‖`.
    pub extent: f64,
    /// `0.9 · min(W, H) / max(extent, 1e-9)`.
    pub scale: f64,
    /// One view's width in pixels.
    pub width: usize,
    /// One view's height in pixels.
    pub height: usize,
}

impl Frame {
    /// The framing of a point set at one image size.
    pub fn of(meshes: &[Splat], width: usize, height: usize) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for mesh in meshes {
            for p in &mesh.points {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        if !lo[0].is_finite() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
        let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let extent = (span[0] * span[0] + span[1] * span[1] + span[2] * span[2]).sqrt();
        #[allow(clippy::cast_precision_loss, reason = "an image is a few thousand pixels wide")]
        let short = width.min(height) as f64;
        Self { centre, extent, scale: FILL * short / extent.max(1e-9), width, height }
    }
}

/// R §11.5's `_splat`: one view of one point set into one image.
pub fn splat(meshes: &[Splat], view: View, frame: &Frame) -> Rgb {
    let (width, height) = (frame.width, frame.height);
    let rotation = view_basis(view);
    let light = unit(LIGHT);

    let background = quantise(BACKGROUND);
    let mut image = Rgb::filled(width, height, background);
    let mut zbuffer = vec![f32::NEG_INFINITY; width * height];

    // The nine offsets are nine independent passes that share the z-buffer, and they run in the
    // reference's order because a later pass sees what an earlier one wrote.
    let mut pass_best: Vec<(f64, [f64; 3])> = vec![(0.0, [0.0; 3]); width * height];
    let mut pass_id: Vec<u16> = vec![u16::MAX; width * height];
    let mut touched: Vec<usize> = Vec::new();
    let mut pass = 0_u16;
    for dx in [-1_i64, 0, 1] {
        for dy in [-1_i64, 0, 1] {
            touched.clear();
            for mesh in meshes {
                for (i, point) in mesh.points.iter().enumerate() {
                    let centred = [
                        point[0] - frame.centre[0],
                        point[1] - frame.centre[1],
                        point[2] - frame.centre[2],
                    ];
                    let projected = matmul_fused(centred, &rotation);
                    let Some((px, py)) = pixel_of(projected, frame) else { continue };
                    let (x, y) = (px + dx, py + dy);
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "the mask keeps 1 <= px < W-1, so px + dx is inside"
                    )]
                    let lin = y as usize * width + x as usize;
                    let normal = matmul_fused(mesh.normals[i], &rotation);
                    // `Nn @ light` is a matrix-vector product, which numpy does not fuse.
                    let lambert =
                        normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];
                    let mut shade = lambert.clamp(0.0, 1.0) * 0.75 + 0.25;
                    if normal[2] < 0.0 {
                        shade *= 0.5;
                    }
                    let base = mesh.paint.colour(i);
                    let colour = [base[0] * shade, base[1] * shade, base[2] * shade];
                    let depth = projected[2];
                    // The stable lexsort keeps the *last* of the deepest points in this pass.
                    if pass_id[lin] != pass {
                        pass_id[lin] = pass;
                        pass_best[lin] = (depth, colour);
                        touched.push(lin);
                    } else if depth >= pass_best[lin].0 {
                        pass_best[lin] = (depth, colour);
                    }
                }
            }
            for &lin in &touched {
                let (depth, colour) = pass_best[lin];
                if depth > f64::from(zbuffer[lin]) {
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "the reference's z-buffer is float32"
                    )]
                    {
                        zbuffer[lin] = depth as f32;
                    }
                    let (x, y) = (lin % width, lin / width);
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "the reference's image is float32"
                    )]
                    image.set(
                        x,
                        y,
                        [
                            quantise(colour[0] as f32)[0],
                            quantise(colour[1] as f32)[0],
                            quantise(colour[2] as f32)[0],
                        ],
                    );
                }
            }
            pass += 1;
        }
    }
    image
}

/// R §11.5's pixel of one projected point, or `None` when the reference's mask drops it.
#[inline]
fn pixel_of(projected: [f64; 3], frame: &Frame) -> Option<(i64, i64)> {
    #[allow(clippy::cast_precision_loss, reason = "an image is a few thousand pixels wide")]
    let (w, h) = (frame.width as f64, frame.height as f64);
    // `np.round` is round-half-to-even; `f64::round` is not.
    let x = (projected[0] * frame.scale + w / 2.0).round_ties_even();
    let y = (-projected[1] * frame.scale + h / 2.0).round_ties_even();
    if !(x.is_finite() && y.is_finite()) {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "`astype(int)` truncates an already integral double; the range test follows"
    )]
    let (x, y) = (x as i64, y as i64);
    #[allow(clippy::cast_possible_wrap, reason = "an image is a few thousand pixels wide")]
    let inside = x >= 1 && x < frame.width as i64 - 1 && y >= 1 && y < frame.height as i64 - 1;
    inside.then_some((x, y))
}

/// `(np.clip(v, 0, 1) * 255).astype(np.uint8)` — the reference's narrowing, in `f32` and
/// truncating.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to 0..=255 first, and numpy truncates"
)]
fn quantise(v: f32) -> [u8; 3] {
    let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
    [byte, byte, byte]
}

/// R §11.5's camera basis: `z = unit(eye)`, `x = unit(up × z)`, `y = z × x`, as the columns of `R`.
///
/// A degenerate `up × z` — an `up` parallel to the view direction — falls back to `[1,0,0] × z`,
/// which is the reference's own escape.
pub fn view_basis(view: View) -> [[f64; 3]; 3] {
    let z = unit(view.eye);
    let mut x = cross(view.up, z);
    if norm(x) < 1e-6 {
        x = cross([1.0, 0.0, 0.0], z);
    }
    let x = unit(x);
    let y = cross(z, x);
    // Rows of the returned array are the rows of `R = np.stack([x, y, z], 1)`.
    [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
}

/// `p @ M` for a 3-vector and a 3×3, with numpy's fused accumulation (`(n,3) @ (3,3)` is `dgemm`).
#[inline]
fn matmul_fused(p: [f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    let column = |j: usize| p[2].mul_add(m[2][j], p[1].mul_add(m[1][j], p[0] * m[0][j]));
    [column(0), column(1), column(2)]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// One fragment's samples turned into a [`Splat`] at a pose (R §11.5's group preview).
///
/// `normals` are the working mesh's face normals and `faces` the face each sample landed on, so
/// the sample normal is its face's, rotated by the pose exactly as `FN[pick] @ T[:3,:3].T` rotates
/// it.
pub fn placed_splat(
    points: &[[f64; 3]],
    faces: &[u32],
    normals: &[[f64; 3]],
    pose: &Matrix4<f64>,
    paint: Paint,
) -> Splat {
    use crate::types::{apply_transform_fused, rotate_fused};
    Splat {
        points: points.iter().map(|&p| apply_transform_fused(pose, p)).collect(),
        normals: faces.iter().map(|&f| rotate_fused(pose, normals[f as usize])).collect(),
        paint,
    }
}

// ---------------------------------------------------------------------------------------------
// The label (PMC-20)

/// The port's own 5×7 bitmap font, one entry per printable ASCII code from `0x20` to `0x5F`.
///
/// Each glyph is five columns and each column a bit mask whose bit `r` is row `r` from the top.
/// Lower-case letters are drawn with the upper-case glyph — a caption on a debug preview, not a
/// typesetting engine — and anything outside the table is drawn as a hollow box.
pub const FONT: [[u8; 5]; 64] = [
    [0x00, 0x00, 0x00, 0x00, 0x00], // space
    [0x00, 0x00, 0x5F, 0x00, 0x00], // !
    [0x00, 0x07, 0x00, 0x07, 0x00], // "
    [0x14, 0x7F, 0x14, 0x7F, 0x14], // #
    [0x24, 0x2A, 0x7F, 0x2A, 0x12], // $
    [0x23, 0x13, 0x08, 0x64, 0x62], // %
    [0x36, 0x49, 0x55, 0x22, 0x50], // &
    [0x00, 0x05, 0x03, 0x00, 0x00], // '
    [0x00, 0x1C, 0x22, 0x41, 0x00], // (
    [0x00, 0x41, 0x22, 0x1C, 0x00], // )
    [0x14, 0x08, 0x3E, 0x08, 0x14], // *
    [0x08, 0x08, 0x3E, 0x08, 0x08], // +
    [0x00, 0x50, 0x30, 0x00, 0x00], // ,
    [0x08, 0x08, 0x08, 0x08, 0x08], // -
    [0x00, 0x60, 0x60, 0x00, 0x00], // .
    [0x20, 0x10, 0x08, 0x04, 0x02], // /
    [0x3E, 0x51, 0x49, 0x45, 0x3E], // 0
    [0x00, 0x42, 0x7F, 0x40, 0x00], // 1
    [0x42, 0x61, 0x51, 0x49, 0x46], // 2
    [0x21, 0x41, 0x45, 0x4B, 0x31], // 3
    [0x18, 0x14, 0x12, 0x7F, 0x10], // 4
    [0x27, 0x45, 0x45, 0x45, 0x39], // 5
    [0x3C, 0x4A, 0x49, 0x49, 0x30], // 6
    [0x01, 0x71, 0x09, 0x05, 0x03], // 7
    [0x36, 0x49, 0x49, 0x49, 0x36], // 8
    [0x06, 0x49, 0x49, 0x29, 0x1E], // 9
    [0x00, 0x36, 0x36, 0x00, 0x00], // :
    [0x00, 0x56, 0x36, 0x00, 0x00], // ;
    [0x08, 0x14, 0x22, 0x41, 0x00], // <
    [0x14, 0x14, 0x14, 0x14, 0x14], // =
    [0x00, 0x41, 0x22, 0x14, 0x08], // >
    [0x02, 0x01, 0x51, 0x09, 0x06], // ?
    [0x32, 0x49, 0x79, 0x41, 0x3E], // @
    [0x7E, 0x11, 0x11, 0x11, 0x7E], // A
    [0x7F, 0x49, 0x49, 0x49, 0x36], // B
    [0x3E, 0x41, 0x41, 0x41, 0x22], // C
    [0x7F, 0x41, 0x41, 0x22, 0x1C], // D
    [0x7F, 0x49, 0x49, 0x49, 0x41], // E
    [0x7F, 0x09, 0x09, 0x09, 0x01], // F
    [0x3E, 0x41, 0x49, 0x49, 0x7A], // G
    [0x7F, 0x08, 0x08, 0x08, 0x7F], // H
    [0x00, 0x41, 0x7F, 0x41, 0x00], // I
    [0x20, 0x40, 0x41, 0x3F, 0x01], // J
    [0x7F, 0x08, 0x14, 0x22, 0x41], // K
    [0x7F, 0x40, 0x40, 0x40, 0x40], // L
    [0x7F, 0x02, 0x0C, 0x02, 0x7F], // M
    [0x7F, 0x04, 0x08, 0x10, 0x7F], // N
    [0x3E, 0x41, 0x41, 0x41, 0x3E], // O
    [0x7F, 0x09, 0x09, 0x09, 0x06], // P
    [0x3E, 0x41, 0x51, 0x21, 0x5E], // Q
    [0x7F, 0x09, 0x19, 0x29, 0x46], // R
    [0x46, 0x49, 0x49, 0x49, 0x31], // S
    [0x01, 0x01, 0x7F, 0x01, 0x01], // T
    [0x3F, 0x40, 0x40, 0x40, 0x3F], // U
    [0x1F, 0x20, 0x40, 0x20, 0x1F], // V
    [0x3F, 0x40, 0x38, 0x40, 0x3F], // W
    [0x63, 0x14, 0x08, 0x14, 0x63], // X
    [0x07, 0x08, 0x70, 0x08, 0x07], // Y
    [0x61, 0x51, 0x49, 0x45, 0x43], // Z
    [0x00, 0x7F, 0x41, 0x41, 0x00], // [
    [0x02, 0x04, 0x08, 0x10, 0x20], // backslash
    [0x00, 0x41, 0x41, 0x7F, 0x00], // ]
    [0x04, 0x02, 0x01, 0x02, 0x04], // ^
    [0x40, 0x40, 0x40, 0x40, 0x40], // _
];

/// The glyph of one character: `|` and lower case have their own mapping, the rest is [`FONT`].
fn glyph(c: char) -> [u8; 5] {
    let c = c.to_ascii_uppercase();
    if c == '|' {
        return [0x00, 0x00, 0x7F, 0x00, 0x00];
    }
    let code = c as u32;
    if (0x20..0x60).contains(&code) {
        FONT[(code - 0x20) as usize]
    } else {
        [0x7F, 0x41, 0x41, 0x41, 0x7F]
    }
}

/// The caption R §11.5 draws at `(10, 10)`, in white and in the port's own font (PMC-20).
///
/// Characters that would run off the right edge are dropped rather than wrapped, which is what
/// PIL's `text` does with a single line.
pub fn draw_label(image: &mut Rgb, x0: usize, y0: usize, text: &str) {
    let mut x = x0;
    for c in text.chars() {
        let glyph = glyph(c);
        for (column, bits) in glyph.into_iter().enumerate() {
            for row in 0..7 {
                if bits & (1 << row) != 0 {
                    let (px, py) = (x + column, y0 + row);
                    if px < image.width && py < image.height {
                        image.set(px, py, [255, 255, 255]);
                    }
                }
            }
        }
        x += 6;
        if x >= image.width {
            break;
        }
    }
}

/// R §11.5's caption for one group: `name=colour | name=colour | …`.
pub fn group_label(names: &[String]) -> String {
    names
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{n}={}", COLOUR_NAMES[i % COLOUR_NAMES.len()]))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::{
        BACKGROUND, Frame, Paint, Rgb, Splat, View, draw_label, group_label, principal_views,
        render_views, splat, view_basis,
    };

    fn one_point(p: [f64; 3], n: [f64; 3]) -> Splat {
        Splat { points: vec![p], normals: vec![n], paint: Paint::Uniform([1.0, 0.0, 0.0]) }
    }

    /// The background is `0.16` narrowed to `f32`, multiplied by 255 in `f32` and truncated —
    /// which is 40, not 41. A `f64` intermediate or a rounding would give the other answer.
    #[test]
    fn the_background_is_the_references_forty() {
        let image = render_views(&[], &[View { eye: [0.0, 0.0, 1.0], up: [0.0, 1.0, 0.0] }], 8, 6);
        assert_eq!(image.pixel(0, 0), [40, 40, 40]);
        assert!((BACKGROUND * 255.0 - 40.8).abs() < 1e-3);
    }

    /// A single point lands on a 3×3 block of pixels centred on its rounded position, and the
    /// shade is the reference's `clip(N·light, 0, 1)·0.75 + 0.25`.
    #[test]
    fn one_point_splats_a_three_by_three_block() {
        let mesh = one_point([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let frame = Frame::of(std::slice::from_ref(&mesh), 21, 21);
        let image = splat(&[mesh], View { eye: [0.0, 0.0, 1.0], up: [0.0, 1.0, 0.0] }, &frame);
        let mut lit = 0;
        for y in 0..21 {
            for x in 0..21 {
                if image.pixel(x, y) != [40, 40, 40] {
                    lit += 1;
                }
            }
        }
        assert_eq!(lit, 9, "the splat is one 3x3 block");
        // The point looks straight at the camera; `light` is `unit(0.3, 0.4, 1.0)`, so
        // `N·light = 0.9113`, shade = 0.9335, red = 0.9335 -> 238.
        let light_z = 1.0 / (0.3_f64.mul_add(0.3, 0.4 * 0.4) + 1.0).sqrt();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0..=255")]
        let expected = ((light_z.clamp(0.0, 1.0) * 0.75 + 0.25) as f32 * 255.0) as u8;
        assert_eq!(image.pixel(10, 10), [expected, 0, 0]);
    }

    /// A nearer point wins the pixel whichever order the two are given in, and the z-buffer that
    /// decides it is `f32`.
    #[test]
    fn the_deeper_point_wins_the_pixel() {
        let near = Splat {
            points: vec![[0.0, 0.0, 1.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            paint: Paint::Uniform([1.0, 0.0, 0.0]),
        };
        let far = Splat {
            points: vec![[0.0, 0.0, -1.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            paint: Paint::Uniform([0.0, 0.0, 1.0]),
        };
        let view = View { eye: [0.0, 0.0, 1.0], up: [0.0, 1.0, 0.0] };
        let frame = Frame::of(&[near.clone(), far.clone()], 21, 21);
        let a = splat(&[near.clone(), far.clone()], view, &frame);
        let b = splat(&[far, near], view, &frame);
        assert_eq!(a.pixel(10, 10), b.pixel(10, 10));
        assert!(a.pixel(10, 10)[0] > a.pixel(10, 10)[2], "the near red point is on top");
    }

    /// The basis is orthonormal and its third column is the view direction, including in the
    /// degenerate case the reference guards with `[1, 0, 0] x z`.
    #[test]
    fn the_camera_basis_is_orthonormal_even_when_up_is_the_view_direction() {
        for view in [
            View { eye: [0.0, 0.0, 1.0], up: [0.0, 1.0, 0.0] },
            View { eye: [0.0, 1.0, 0.0], up: [0.0, 1.0, 0.0] },
            View { eye: [1.0, 0.2, 0.4], up: [0.0, 0.0, 1.0] },
        ] {
            let r = view_basis(view);
            for j in 0..3 {
                for k in 0..3 {
                    let dot: f64 = (0..3).map(|i| r[i][j] * r[i][k]).sum();
                    let expected = f64::from(u8::from(j == k));
                    assert!((dot - expected).abs() < 1e-12, "{view:?}: {j}.{k} = {dot}");
                }
            }
        }
    }

    /// The four views are the reference's, and the axes come back ascending: `u` is the direction
    /// a flat slab varies in least.
    #[test]
    fn principal_views_look_along_the_flattest_axis() {
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push([f64::from(i), 0.1 * f64::from(j), 0.01 * f64::from(i * j % 3)]);
            }
        }
        let views = principal_views(&points);
        assert_eq!(views.len(), 4);
        assert!(views[0].eye[2].abs() > 0.9, "u is the z axis: {:?}", views[0].eye);
        let bits = |v: [f64; 3]| v.map(f64::to_bits);
        assert_eq!(
            bits(views[1].eye),
            bits([-views[0].eye[0], -views[0].eye[1], -views[0].eye[2]])
        );
        assert_eq!(bits(views[1].up), bits(views[0].up));
        assert_eq!(bits(views[2].up), bits(views[0].eye));
    }

    /// The caption writes white pixels where the glyphs are and leaves the rest alone.
    #[test]
    fn the_label_draws_where_the_glyphs_are() {
        let mut image = Rgb::filled(120, 30, [40, 40, 40]);
        draw_label(&mut image, 10, 10, "AB=grey");
        // 'A' has its top-left corner clear and its second row set (the apex).
        assert_eq!(image.pixel(10, 10), [40, 40, 40]);
        assert_eq!(image.pixel(10, 11), [255, 255, 255]);
        // Nothing is drawn above or left of the origin.
        assert_eq!(image.pixel(9, 9), [40, 40, 40]);
        assert_eq!(group_label(&["a".to_owned(), "b".to_owned()]), "a=grey | b=orange");
    }

    /// Two renders of the same input are the same bytes — D §7's determinism, for an image.
    #[test]
    fn rendering_is_deterministic() {
        let meshes: Vec<Splat> = (0_u32..3)
            .map(|k| {
                let mut points = Vec::new();
                let mut normals = Vec::new();
                for i in 0..40 {
                    let a = f64::from(i) * 0.37 + f64::from(k);
                    points.push([a.sin() * 5.0, a.cos() * 5.0, 0.3 * f64::from(k)]);
                    normals.push([a.cos(), a.sin(), 0.5]);
                }
                Splat { points, normals, paint: Paint::Uniform(super::PALETTE[k as usize]) }
            })
            .collect();
        let all: Vec<[f64; 3]> = meshes.iter().flat_map(|m| m.points.iter().copied()).collect();
        let views = principal_views(&all);
        let a = render_views(&meshes, &views, 90, 70);
        let b = render_views(&meshes, &views, 90, 70);
        assert_eq!(a, b);
        assert_eq!(a.width, 4 * 90);
        assert_eq!(a.height, 70);

        // And so is the PNG: D §7's rule is about the *files* a run writes, so the encoder has to
        // be a function of the pixels as well.
        let dir = std::env::temp_dir().join(format!("sherd-render-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let (first, second) = (dir.join("a.png"), dir.join("b.png"));
        a.write_png(&first).expect("the first PNG is written");
        b.write_png(&second).expect("the second PNG is written");
        assert_eq!(
            std::fs::read(&first).expect("read"),
            std::fs::read(&second).expect("read"),
            "two encodings of one image are the same bytes"
        );
        // The caption changes the file, which is what makes the comparison above worth making.
        let mut labelled = a;
        draw_label(&mut labelled, 10, 10, "AB");
        let third = dir.join("c.png");
        labelled.write_png(&third).expect("the labelled PNG is written");
        assert_ne!(std::fs::read(&first).expect("read"), std::fs::read(&third).expect("read"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
