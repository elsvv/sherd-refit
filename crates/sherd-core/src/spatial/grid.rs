//! Uniform hash grid for radius-bounded nearest neighbours (D §6.2).
//!
//! E3 measured it at 0.4–2.0× `kiddo` on the CPU — never the ≥ 3× D §3 expected — so on the CPU
//! the KD-tree wins and this structure exists for the GPU, where the same cell layout is what the
//! WGSL kernel traverses: cell size `r`, keys hashed into `next_pow2(2n)` slots, the 27
//! neighbouring cells visited in a fixed order, ties resolved by the lowest index. Written with
//! the GPU kernels in phase 2b; the CPU build and a brute-force test come earlier if the layout
//! needs pinning down.
