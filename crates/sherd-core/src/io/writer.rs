//! PLY writer (R §11.2–11.4).
//!
//! Binary little-endian, `uchar` RGB when colours are present, the same header and the same
//! element order as Open3D's `write_triangle_mesh(write_ascii=False)`, so that the Rust outputs
//! are byte-comparable with the reference's. E2 verified `ply-rs-bw`'s writer reproduces such a
//! file byte for byte, including the streaming, element-by-element mode the merged assembly mesh
//! needs (D §5: no merged mesh is ever held in memory). Filled in by plan step S2.
