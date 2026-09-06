//! OFF reader (R §3.1) — the one own reader in this module.
//!
//! No crate on crates.io reads OFF (E2 checked `mesh-loader`, `tobj`, `meshx` and searched the
//! index), so this is roughly forty lines: the `OFF` magic, the counts line, `n` vertex lines,
//! `m` face lines each `k i0 i1 … [r g b [a]]`, comments after `#`, and a fan triangulation for
//! `k > 3`. Filled in by plan step S2.
