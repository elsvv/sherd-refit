//! Matching one pair of fragments (R §4–§7).
//!
//! The chain is: pair scales from `t_pair` and `res_pair` (§1.2, §4.1), hypotheses from paired
//! breakline frames (§5.1), the coarse breakline score (§5.2), non-maximum suppression (§5.3),
//! the breakline ICP ladder and the re-score (§5.4), full ICP on the fracture and registration
//! clouds (§5.5–5.6), verification — tight contact, gap, seam, continuity, penetration (§6) —
//! and the accept/reject rule (§6.5).
//!
//! Step C1 filled in the first half: [`scales`] (§1.2, §4.2), [`hypotheses`] (§5.1), [`coarse`]
//! (§5.2), [`nms`] (§5.3) and the [`pair`] that drives them. Step C2 added the refinement:
//! Open3D's ICP as R §7 freezes it ([`icp`]) and the two ladders it drives ([`ladder`]) — R §5.4's
//! breakline rungs with their re-score, R §5.5's suppression, and the four registration and
//! fracture rungs of R §5.6. The verification of R §6 and the ranking of R §5.7 follow.

pub mod coarse;
pub mod hypotheses;
pub mod icp;
pub mod ladder;
pub mod nms;
pub mod pair;
pub mod scales;
pub mod screen;
pub mod verify;
