//! The ICP ladder (R §5.4–5.6, R §7): point-to-point on breaklines, then point-to-plane on the
//! fracture and registration clouds, over a ladder of shrinking correspondence radii, each rung
//! capped at `max_iter` with a per-candidate `done` flag exactly as the sequential reference has
//! it. Normal equations are assembled around the target centroid and the update re-expressed
//! about the origin, which is what makes the `f32` GPU path safe (D §7). Filled in in phase 1c.
