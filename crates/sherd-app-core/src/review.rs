//! What survives a decision (A §8.4): which groups of a fresh assembly are still the refined ones.
//!
//! R §9's refinement reads the source scans and takes tens of seconds; R §8's assembly takes a
//! tenth of one. A reviewer who accepts a join therefore gets a new assembly at once — and would
//! lose every pose R §9 had already found, if the two were not put back together.
//!
//! They can be, because refinement is a property of a **group**: R §9 walks a group's used joins
//! and moves nothing else. A group of the new assembly whose members and whose joins among those
//! members are the ones a refined group had is the same group, placed by the same walk over the
//! same joins, and its refined poses are still its poses. Anything else — a member gained, a
//! member lost, the same members held together differently — is a group R §9 has not seen.
//!
//! The rule is a pure function over two [`AssemblyDto`]s so that it is tested without a worker,
//! and so that the session that uses it has one thing to get right instead of two.

use std::collections::BTreeSet;

use crate::protocol::{AssemblyDto, JoinDto};

/// A group's identity for the rule above: who is in it, and which joins hold it together.
///
/// Both sets, and not the members alone: two fragments can end up in one group through another
/// join entirely, and the poses R §9 found by walking the old join are not the poses of the new
/// one. Names and not indices, because a decision and an assembly are both by name (A §8.1).
type Shape = (BTreeSet<String>, BTreeSet<(String, String)>);

/// The shape of the group `members`, given the assembly's whole join list.
///
/// A join is kept only when **both** its fragments are in the group, and is stored with its two
/// names in order: the same join can be written `a-b` by one reassembly and `b-a` by the next,
/// because which fragment a candidate calls first is the matcher's business and not the group's.
fn shape(members: &[String], joins: &[JoinDto]) -> Shape {
    let members: BTreeSet<String> = members.iter().cloned().collect();
    let inside = joins
        .iter()
        .filter(|j| members.contains(&j.a) && members.contains(&j.b))
        .map(|j| if j.a <= j.b { (j.a.clone(), j.b.clone()) } else { (j.b.clone(), j.a.clone()) })
        .collect();
    (members, inside)
}

/// `fresh`, with every group the `baseline` had already refined put back at the poses it was
/// refined to (A §8.4).
///
/// A group of `fresh` whose [`Shape`] equals that of a **refined** group of `baseline` takes that
/// group's poses and is `refined`; every other group keeps `fresh`'s poses and is not. `joins`
/// and `unplaced` are `fresh`'s throughout: they describe the assembly that was just built, not
/// the one it is being compared with.
///
/// A singleton is never refined — R §9 has no join to walk in it — so a baseline group of one is
/// not matched even if it says it is, and a fresh group of one keeps its pose.
#[must_use]
pub fn merge_refined(baseline: &AssemblyDto, mut fresh: AssemblyDto) -> AssemblyDto {
    let refined: BTreeSet<Shape> = baseline
        .groups
        .iter()
        .filter(|g| g.refined && g.members.len() > 1)
        .map(|g| shape(&g.members, &baseline.joins))
        .collect();
    // The shapes are taken in one pass, before anything is written back: a group's shape is read
    // from the whole assembly's join list, which the writing below does not touch.
    let keep: Vec<bool> = fresh
        .groups
        .iter()
        .map(|g| g.members.len() > 1 && refined.contains(&shape(&g.members, &fresh.joins)))
        .collect();
    for (n, &keep) in keep.iter().enumerate() {
        fresh.groups[n].refined = keep;
        if !keep {
            continue;
        }
        for member in fresh.groups[n].members.clone() {
            if let Some(&pose) = baseline.poses.get(&member) {
                fresh.poses.insert(member, pose);
            }
        }
    }
    fresh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{GroupDto, UnplacedDto};

    fn pose(x: f64) -> [[f64; 4]; 4] {
        [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    }

    fn join(a: &str, b: &str) -> JoinDto {
        JoinDto { a: a.to_owned(), b: b.to_owned() }
    }

    fn group(members: &[&str], refined: bool) -> GroupDto {
        GroupDto { members: members.iter().map(|&m| m.to_owned()).collect(), refined }
    }

    /// A refined `{A, B}` held together by `A-B`, at poses 1 and 2.
    fn baseline() -> AssemblyDto {
        AssemblyDto {
            groups: vec![group(&["A", "B"], true), group(&["C"], false)],
            poses: [("A", 1.0), ("B", 2.0), ("C", 3.0)]
                .into_iter()
                .map(|(n, x)| (n.to_owned(), pose(x)))
                .collect(),
            joins: vec![join("A", "B")],
            unplaced: Vec::new(),
        }
    }

    /// A §8.4: the same members held together by the same joins are the same group, and R §9's
    /// poses are still its poses — whichever way round the reassembly happens to name the join.
    #[test]
    fn a_group_with_the_same_members_and_joins_keeps_the_poses_it_was_refined_to() {
        let fresh = AssemblyDto {
            groups: vec![group(&["A", "B"], false), group(&["C"], false)],
            poses: [("A", 9.0), ("B", 9.0), ("C", 9.0)]
                .into_iter()
                .map(|(n, x)| (n.to_owned(), pose(x)))
                .collect(),
            joins: vec![join("B", "A")],
            unplaced: vec![UnplacedDto {
                a: "C".into(),
                b: "A".into(),
                reason: "penetrates A (0.004)".into(),
            }],
        };
        let merged = merge_refined(&baseline(), fresh);
        assert!(merged.groups[0].refined);
        assert_eq!(merged.poses["A"], pose(1.0));
        assert_eq!(merged.poses["B"], pose(2.0));
        // the singleton is `fresh`'s, pose and all: R §9 never refined it
        assert!(!merged.groups[1].refined);
        assert_eq!(merged.poses["C"], pose(9.0));
        assert_eq!(merged.joins, [join("B", "A")], "the joins are the new assembly's");
        assert_eq!(merged.unplaced.len(), 1, "and so is what did not get placed");
    }

    /// A member gained is a walk R §9 has not made: the whole group is the new assembly's.
    #[test]
    fn a_group_that_gained_a_member_is_not_refined() {
        let fresh = AssemblyDto {
            groups: vec![group(&["A", "B", "C"], false)],
            poses: [("A", 9.0), ("B", 9.0), ("C", 9.0)]
                .into_iter()
                .map(|(n, x)| (n.to_owned(), pose(x)))
                .collect(),
            joins: vec![join("A", "B"), join("B", "C")],
            unplaced: Vec::new(),
        };
        let merged = merge_refined(&baseline(), fresh);
        assert!(!merged.groups[0].refined);
        assert_eq!(merged.poses["A"], pose(9.0));
    }

    /// The same two fragments through another join are placed by another walk, so the poses R §9
    /// found for the old one say nothing about them.
    #[test]
    fn the_same_members_held_together_differently_are_not_refined() {
        let fresh = AssemblyDto {
            groups: vec![group(&["A", "B"], false)],
            poses: [("A", 9.0), ("B", 9.0)]
                .into_iter()
                .map(|(n, x)| (n.to_owned(), pose(x)))
                .collect(),
            // the pair is the same pair, but this assembly took a join to C with it and lost it
            joins: vec![join("A", "B"), join("A", "C")],
            unplaced: Vec::new(),
        };
        // `A-C` is not inside the group, so the shape is unchanged and the group IS refined
        assert!(merge_refined(&baseline(), fresh.clone()).groups[0].refined);

        // whereas a group of three reduced to two that never had `A-B` at all is not
        let other = AssemblyDto { joins: vec![join("A", "C"), join("C", "B")], ..fresh };
        assert!(!merge_refined(&baseline(), other).groups[0].refined);
    }

    /// A baseline that refined nothing changes nothing, and neither does a first reassembly.
    #[test]
    fn nothing_refined_leaves_the_fresh_assembly_exactly_as_it_came() {
        let fresh = AssemblyDto {
            groups: vec![group(&["A", "B"], false)],
            poses: [("A", 9.0)].into_iter().map(|(n, x)| (n.to_owned(), pose(x))).collect(),
            joins: vec![join("A", "B")],
            unplaced: Vec::new(),
        };
        assert_eq!(merge_refined(&AssemblyDto::default(), fresh.clone()), fresh);
    }
}
