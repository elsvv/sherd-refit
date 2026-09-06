//! PLY reader (R §3.1) — ASCII, binary little-endian and binary big-endian, `float` or `double`
//! coordinates, `vertex_indices` or `vertex_index` lists, optional `red`/`green`/`blue`
//! (roadmap item 4 keeps the colours), polygons triangulated as a fan.
//!
//! Crate: `ply-rs-bw` through its typed `PropertyAccess` interface rather than `DefaultElement`,
//! which E2 measured as bit-identical to Open3D on all eleven PLY variants of the benchmark and
//! reads a 25 MB scan in 0.057 s. Filled in by plan step S2.
