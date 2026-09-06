//! A fragment: everything derived from one file (R §3).
//!
//! Thickness (§3.2) sets the face budget and every scale-free threshold; the working mesh (§3.3)
//! is what the segmentation (§3.4) labels; the breaklines (§3.5.5) and the match arrays
//! (§3.5–3.6) are what the matcher compares. All of it is cached per fragment (§3.7, D §4.2), so
//! a rerun on the same collection starts at the matching stage.
//!
//! Steps S3 and S4 fill in thickness, the cache and the working-mesh assembly; segmentation,
//! breaklines and the match arrays follow in phase 1b.

pub mod breakline;
pub mod cache;
pub mod features;
pub mod samples;
pub mod segment;
pub mod thickness;
