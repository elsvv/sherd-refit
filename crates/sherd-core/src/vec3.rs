//! `Vec3f` — the one 3-vector used by the geometry, the buffers and (from phase 2) the WGSL
//! kernels.
//!
//! D §3 asks for an own newtype rather than a library vector so that a slice of them can be
//! uploaded to the GPU as-is: `#[repr(C)]`, exactly twelve bytes, no padding, `bytemuck::Pod`.
//! Coordinates are `f32` because that is what both executors compute in (D §7); the poses and
//! the ICP solves stay `f64` and live in [`crate::types::Pose`].

// bytemuck's derives expand to `unsafe impl Pod` / `unsafe impl Zeroable` in this module. That is
// the only unsafe code in the crate, and it is sound because the struct is `#[repr(C)]`, holds
// three `f32` and therefore has no padding and no invalid bit pattern.
#![allow(unsafe_code)]

use std::ops::{Add, AddAssign, Div, Index, Mul, Neg, Sub, SubAssign};

/// A point or a direction in the scan's units.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3f {
    /// First coordinate.
    pub x: f32,
    /// Second coordinate.
    pub y: f32,
    /// Third coordinate.
    pub z: f32,
}

/// Shorthand constructor: `vec3(1.0, 0.0, 0.0)`.
#[inline]
pub const fn vec3(x: f32, y: f32, z: f32) -> Vec3f {
    Vec3f { x, y, z }
}

impl Vec3f {
    /// The origin, and the zero direction.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    /// Builds a vector from its three coordinates.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The same value in all three coordinates.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Reads a vector from an array, the layout the mesh readers produce.
    #[inline]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }

    /// The three coordinates as an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Narrows the `f64` coordinates the readers and the fixtures carry.
    ///
    /// The narrowing is the point: the working mesh, the samples and both executors are `f32`
    /// (D §7); only the poses stay `f64`.
    #[inline]
    #[allow(clippy::cast_possible_truncation, reason = "f64 -> f32 is the documented contract")]
    pub fn from_f64(a: [f64; 3]) -> Self {
        Self { x: a[0] as f32, y: a[1] as f32, z: a[2] as f32 }
    }

    /// Widens to `f64`, for the pose solves and for comparisons against the fixtures.
    #[inline]
    pub fn to_f64(self) -> [f64; 3] {
        [f64::from(self.x), f64::from(self.y), f64::from(self.z)]
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Cross product, right-handed: `x × y = z`.
    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    /// Squared Euclidean length; the form the distance tests use, to avoid a square root.
    #[inline]
    pub fn norm_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[inline]
    pub fn norm(self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Squared distance to another point.
    #[inline]
    pub fn distance_squared(self, o: Self) -> f32 {
        (self - o).norm_squared()
    }

    /// Distance to another point.
    #[inline]
    pub fn distance(self, o: Self) -> f32 {
        (self - o).norm()
    }

    /// The unit vector in the same direction, or `None` for a vector of length zero — the
    /// reference drops degenerate normals rather than substituting a direction (R §3.3).
    #[inline]
    pub fn normalized(self) -> Option<Self> {
        let n = self.norm();
        if n > 0.0 && n.is_finite() { Some(self / n) } else { None }
    }

    /// Component-wise minimum, for bounding boxes.
    #[inline]
    pub fn min(self, o: Self) -> Self {
        Self { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) }
    }

    /// Component-wise maximum, for bounding boxes.
    #[inline]
    pub fn max(self, o: Self) -> Self {
        Self { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) }
    }

    /// True when all three coordinates are finite; scans do contain NaN vertices.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl From<[f32; 3]> for Vec3f {
    #[inline]
    fn from(a: [f32; 3]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vec3f> for [f32; 3] {
    #[inline]
    fn from(v: Vec3f) -> Self {
        v.to_array()
    }
}

impl Index<usize> for Vec3f {
    type Output = f32;

    /// Panics on an index above 2.
    #[inline]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3f index out of range: {i}"),
        }
    }
}

impl Add for Vec3f {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
}

impl Sub for Vec3f {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}

impl Mul<f32> for Vec3f {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl Mul<Vec3f> for f32 {
    type Output = Vec3f;
    #[inline]
    fn mul(self, v: Vec3f) -> Vec3f {
        v * self
    }
}

impl Div<f32> for Vec3f {
    type Output = Self;
    #[inline]
    fn div(self, s: f32) -> Self {
        Self { x: self.x / s, y: self.y / s, z: self.z / s }
    }
}

impl Neg for Vec3f {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl AddAssign for Vec3f {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl SubAssign for Vec3f {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

#[cfg(test)]
mod tests {
    use super::{Vec3f, vec3};
    use approx::assert_relative_eq;

    #[test]
    fn layout_is_gpu_ready() {
        assert_eq!(size_of::<Vec3f>(), 12);
        assert_eq!(align_of::<Vec3f>(), 4);
        let v = [vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)];
        let bytes: &[u8] = bytemuck::cast_slice(&v);
        assert_eq!(bytes.len(), 24);
        let back: &[f32] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn dot_and_cross_are_right_handed() {
        let (x, y, z) = (vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
        assert_relative_eq!(x.dot(y), 0.0);
        assert_relative_eq!(x.dot(x), 1.0);
        assert_eq!(x.cross(y), z);
        assert_eq!(y.cross(z), x);
        assert_eq!(z.cross(x), y);
        assert_eq!(y.cross(x), -z);
    }

    #[test]
    fn norms_and_distances() {
        let v = vec3(3.0, 4.0, 12.0);
        assert_relative_eq!(v.norm_squared(), 169.0);
        assert_relative_eq!(v.norm(), 13.0);
        assert_relative_eq!(v.distance(Vec3f::ZERO), 13.0);
        assert_relative_eq!(v.distance_squared(vec3(3.0, 4.0, 11.0)), 1.0);
        let u = v.normalized().expect("non-zero vector normalises");
        assert_relative_eq!(u.norm(), 1.0, epsilon = 1e-6);
        assert!(Vec3f::ZERO.normalized().is_none());
        assert!(vec3(f32::NAN, 0.0, 0.0).normalized().is_none());
    }

    #[test]
    fn arithmetic_and_conversions() {
        let a = vec3(1.0, 2.0, 3.0);
        let b = vec3(0.5, 0.5, 0.5);
        assert_eq!(a + b, vec3(1.5, 2.5, 3.5));
        assert_eq!(a - b, vec3(0.5, 1.5, 2.5));
        assert_eq!(a * 2.0, vec3(2.0, 4.0, 6.0));
        assert_eq!(2.0 * a, a * 2.0);
        assert_eq!(a / 2.0, vec3(0.5, 1.0, 1.5));
        let mut c = a;
        c += b;
        c -= b;
        assert_eq!(c, a);
        assert_eq!(Vec3f::from_array(a.to_array()), a);
        assert_eq!(Vec3f::from_f64(a.to_f64()), a);
        assert_eq!(Vec3f::splat(1.0), vec3(1.0, 1.0, 1.0));
        assert_relative_eq!(a[0], 1.0);
        assert_relative_eq!(a[2], 3.0);
    }

    #[test]
    fn bounds_helpers() {
        let a = vec3(1.0, -2.0, 3.0);
        let b = vec3(-1.0, 2.0, 0.0);
        assert_eq!(a.min(b), vec3(-1.0, -2.0, 0.0));
        assert_eq!(a.max(b), vec3(1.0, 2.0, 3.0));
        assert!(a.is_finite());
        assert!(!vec3(0.0, f32::INFINITY, 0.0).is_finite());
    }
}
