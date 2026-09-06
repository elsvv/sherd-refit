//! Wall thickness `t` and the face budget (R §3.2–3.3).
//!
//! Rays are cast inwards from sampled surface points, the hit distances are histogrammed and the
//! mode of the histogram is the wall thickness; `t` then fixes the adaptive face budget and every
//! `k·t` threshold of R §1.2. D §10.2 compares `t` and `thick_mode` against the reference within
//! one histogram bin (injected) or 2 % (native). Filled in by plan step S4.
