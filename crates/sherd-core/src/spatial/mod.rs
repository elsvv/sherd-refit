//! Spatial structures (D §6.2; experiments E3/E4).
//!
//! `docs/superpowers/notes/2026-09-06-e3e4-spatial.md` settled the CPU side: `parry3d` for the
//! BVH (closest point, bounded closest point, rays) and `kiddo`'s `ImmutableKdTree<f32, 3>` for
//! radius-bounded nearest neighbours — the own flattened BVH D §3 had costed at ~1.5 weeks is
//! not needed, and the own hash grid stays a GPU-only structure. The one piece of own code is
//! the inside test: parry's pseudo-normal test was wrong on 29 of 30 000 queries on a closed
//! terracotta mesh and on 202 of 30 000 on a non-watertight sherd, so the inside test is ray
//! parity over the BVH's leaves (three axis rays, majority — PMC-7), which matched Open3D on all
//! 270 000 queries.
//!
//! Step B1 filled in what R §3 needs: [`bvh::RayScene`] (first-hit rays for the thickness of
//! R §3.2 and the seven-ray cone of R §3.4.3, and the closest face the parity harness transfers
//! labels along) and [`kdtree::PointTree`] (the nearest representative and the radius balls of
//! R §3.4.1–3.4.7). The inside test and the bounded closest point join them with the penetration
//! and contact scores of R §6 in phase 1c; [`grid`] stays a GPU structure (phase 2b).

pub mod bvh;
pub mod grid;
pub mod kdtree;
