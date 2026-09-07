//! Global assembly (R §8): from accepted pairwise joins to placed fragments.
//!
//! Joins are taken greedily in score order, each one checked against the group it would extend —
//! penetration against every placed member, agreement with the alternative paths the group already
//! implies — and the result is one pose per fragment per group.
//!
//! * [`groups`] is what a group is: [`Piece`], the assembly's whole view of a fragment;
//!   [`Grouping`], the poses and memberships while they are grown; and R §8.2's [`recenter`].
//! * [`consistency`] is the two tests a join must pass — R §6.4's penetration at the pair's own
//!   depth, and R §8's `10° / 0.5 t` on two poses of the same fragment.
//! * [`greedy`] is the loop itself and R §8's four return values ([`Assembly`]).
//! * [`constraints`] is roadmap item 3's file format, typed and not yet read by anything.
//!
//! Step D1 filled all of this in, with the `assembly` row of D §10.2 behind it.
//!
//! [`Piece`]: groups::Piece
//! [`Grouping`]: groups::Grouping
//! [`recenter`]: groups::recenter
//! [`Assembly`]: greedy::Assembly

pub mod consistency;
pub mod constraints;
pub mod greedy;
pub mod groups;

pub use consistency::{ROT_TOL_DEG, TRANS_TOL_T};
pub use greedy::{Assembly, Rejected, Rejection, assemble};
pub use groups::{Grouping, Piece, recenter};
