//! Global assembly (R §8): from accepted pairwise joins to placed fragments.
//!
//! Joins are taken greedily in score order, each one checked against the group it would extend —
//! penetration against every placed member, cycle consistency against the alternative paths —
//! and the result is one pose per fragment per group. Filled in in phase 1d.

pub mod consistency;
pub mod constraints;
pub mod greedy;
pub mod groups;
