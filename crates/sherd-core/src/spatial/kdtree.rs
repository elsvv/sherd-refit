//! Radius-bounded nearest neighbours on the CPU (D §6.2, experiment E3).
//!
//! `kiddo::ImmutableKdTree<f32, 3>` with `nearest_n::<SquaredEuclidean>(1).within(r²)`: 79–251 ns
//! per query on the benchmark clouds, zero errors against brute force over ~0.9 M bounded
//! queries, and one build per cloud instead of one grid per ICP ladder rung. Used by the seam and
//! continuity tests, `near`, `d_brk` and the margin queries. Filled in in phase 1b.
