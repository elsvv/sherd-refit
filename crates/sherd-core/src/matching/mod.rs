//! Matching one pair of fragments (R §4–§7).
//!
//! The chain is: pair scales from `t_pair` and `res_pair` (§1.2, §4.1), hypotheses from paired
//! breakline frames (§5.1), the coarse breakline score (§5.2), non-maximum suppression (§5.3),
//! the breakline ICP ladder and the re-score (§5.4), full ICP on the fracture and registration
//! clouds (§5.5–5.6), verification — tight contact, gap, seam, continuity, penetration (§6) —
//! and the accept/reject rule (§6.5).
//!
//! Step C1 fills in the first half: [`scales`] (§1.2, §4.2), [`hypotheses`] (§5.1), [`coarse`]
//! (§5.2), [`nms`] (§5.3) and the [`pair`] that drives them. The ICP ladder, the verification and
//! the ranking are step C2's.

pub mod coarse;
pub mod hypotheses;
pub mod icp;
pub mod nms;
pub mod pair;
pub mod scales;
pub mod screen;
pub mod verify;
