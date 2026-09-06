//! GLB reader (R §3.1 is silent: the reference does not read GLB; D §9 wants it for the desktop
//! app) via `gltf` with `default-features = false, features = ["utils"]`, read through
//! `Gltf::from_slice_without_validation` with node transforms applied and `COLOR_0` taken as
//! normalised `u8` or as `f32`. Filled in by plan step S2.
