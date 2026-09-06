//! Quadric decimation to the adaptive face budget (R §3.3).
//!
//! Experiment E1 (`docs/superpowers/notes/2026-09-06-e1-decimation.md`) chose `meshopt`:
//! `meshopt::simplify(&indices, &VertexDataAdapter::new(f32 positions, stride 12, offset 0),
//! target_faces * 3, 1e9, SimplifyOptions::Regularize, None)` — `Regularize` on (without it the
//! crate misses the `res` gate on 13 of 14 meshes), `LockBorder` off, `ErrorAbsolute` off,
//! `Prune` off. It keeps R §3.3's guard: decimate only when `n_faces > target`.
//!
//! The returned index buffer references the original vertices and no vertex ever moves, so the
//! port keeps its `f64` coordinates through decimation. Measured: 1.73 s over the 14 benchmark
//! meshes against Open3D's 34.70 s, segmentation agreement 0.9801–0.9935 (gate ≥ 0.97).
//! The collapse order differs from Open3D's, which is why native-mode parity for the working
//! mesh is statistical (D §10.2, D §13.1). Filled in by plan step S3.
