//! Turning a scan into the working mesh (R §3.3).
//!
//! The order is the reference's: clean (duplicate vertices, degenerate and duplicate faces,
//! unreferenced vertices), keep the largest connected component, decimate to the adaptive face
//! budget when the mesh is above it, Taubin-smooth, then derive the per-face arrays and `res`.
//! Filled in by plan step S3.

pub mod adjacency;
pub mod clean;
pub mod components;
pub mod decimate;
pub mod geometry;
pub mod taubin;
