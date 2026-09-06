//! Spatial structures (D §6.2; experiments E3/E4).
//!
//! `docs/superpowers/notes/2026-09-06-e3e4-spatial.md` settled the CPU side: `parry3d` for the
//! BVH (closest point, bounded closest point, rays) and `kiddo`'s `ImmutableKdTree<f32, 3>` for
//! radius-bounded nearest neighbours — the own flattened BVH D §3 had costed at ~1.5 weeks is
//! not needed, and the own hash grid stays a GPU-only structure. The one piece of own code is
//! the inside test: parry's pseudo-normal test was wrong on 29 of 30 000 queries on a closed
//! terracotta mesh and on 202 of 30 000 on a non-watertight sherd, so the inside test is ray
//! parity over the BVH's leaves (three axis rays, majority — PMC-7), which matched Open3D on all
//! 270 000 queries. Filled in in phase 1b.

pub mod bvh;
pub mod grid;
pub mod kdtree;
