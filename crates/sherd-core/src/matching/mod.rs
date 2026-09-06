//! Matching one pair of fragments (R §4–§7).
//!
//! The chain is: pair scales from `t_pair` and `res_pair` (§1.2, §4.1), hypotheses from paired
//! breakline frames (§5.1), the coarse breakline score (§5.2), non-maximum suppression (§5.3),
//! the breakline ICP ladder and the re-score (§5.4), full ICP on the fracture and registration
//! clouds (§5.5–5.6), verification — tight contact, gap, seam, continuity, penetration (§6) —
//! and the accept/reject rule (§6.5). Filled in in phase 1c.

pub mod coarse;
pub mod hypotheses;
pub mod icp;
pub mod nms;
pub mod pair;
pub mod scales;
pub mod screen;
pub mod verify;
