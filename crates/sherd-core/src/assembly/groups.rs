//! Groups of placed fragments (R §8): what a group *is*, the wall it is measured in, the order
//! the groups come back in, and R §8.2's recentring.
//!
//! A group is a set of fragments the greedy pass of [`greedy`](super::greedy) has joined, plus one
//! world pose each; [`Grouping`] is that state while it is being grown, and R §8's `poses`,
//! `groups` and `group_of` are its three fields under the reference's own names. Roadmap item 4
//! adds the consensus features that tell one vessel from another (D §11); phase 1 has membership
//! and nothing else.

use nalgebra::Matrix4;

use crate::mesh::geometry::{column_mean, median};
use crate::spatial::bvh::RayScene;
use crate::types::{FragId, apply_transform_fused};

/// R §8's view of one fragment: everything the assembly reads of it and nothing else.
///
/// The assembly does not touch a fragment's breakline, its fracture samples or its scores — it
/// asks three questions of a fragment (how thick is its wall, how coarse is its mesh, is that mesh
/// closed) and moves one point set through one pose. Taking those as data rather than as a
/// `&Fragment` is what lets the parity harness build a [`Piece`] out of the reference's own dumped
/// arrays, which is the whole of D §10.2's injected column for this stage.
///
/// # `s_pen` and PMC-8
///
/// The reference rebuilds every fragment's match arrays at the **collection median** `t_med` with
/// `surface_points = 15000` just for the assembly, so the penetration test below sees a different
/// point set from the one R §6.4 saw. PMC-8 lets the port hand in the fragment's own cached
/// 20 000-point sample at its own `t` instead. Both are one field of this struct, so the choice
/// belongs to the caller and neither the greedy pass nor the penetration test knows which it got.
#[derive(Debug)]
pub struct Piece<'a> {
    /// R §3.2's wall thickness — the fragment's own, never the collection's median.
    ///
    /// This is what [`Grouping::thickness`] takes the median of and what R §1.2's `pen` distance
    /// is resolved from; the median of a collection holding a 2.4 mm wall and a 13.5 mm one
    /// describes neither.
    pub thick: f64,
    /// R §3.3's `res`, the working mesh's median edge length (the second term of R §1.2).
    pub res: f64,
    /// R §3.3.2's verdict. Both fragments of a pair must be closed for a signed distance to mean
    /// anything, and R §8's penetration test returns `0` when either is not.
    pub watertight: bool,
    /// A BVH over the whole working mesh — the reference's `Fragment.scene`.
    ///
    /// `None` for a fragment whose working mesh has no triangle, which reads the same way as an
    /// open mesh: no penetration can be found against it.
    pub mesh: Option<&'a RayScene>,
    /// `S_pen`: the surface samples the penetration test casts, and the points R §8.2 recentres on.
    pub s_pen: &'a [[f64; 3]],
}

/// The assembly's state: one world pose per placed fragment, and the groups they fall into.
///
/// The three fields are the reference's `poses`, `group_of` and `groups`, kept together because
/// every operation touches all three: placing a fragment writes a pose, a group index and a
/// membership entry, and a check reads all three back.
#[derive(Clone, Debug)]
pub struct Grouping {
    /// `poses[n]`, or `None` while `n` is unplaced. Indexed by [`FragId`], so the map the
    /// reference keys by name is a vector here and the iteration order below is R §2's.
    poses: Vec<Option<Matrix4<f64>>>,
    /// `group_of[n]`: which group `n` was placed into.
    group_of: Vec<Option<usize>>,
    /// `groups`: each group's members, in the order they were placed.
    groups: Vec<Vec<FragId>>,
    /// The order `poses` was filled in — the reference's dict is insertion-ordered and R §11.1
    /// writes `transforms.json` from it (V4-D5).
    order: Vec<FragId>,
}

impl Grouping {
    /// An empty assembly over `n` fragments: nothing placed, no group.
    pub fn new(n: usize) -> Self {
        Self {
            poses: vec![None; n],
            group_of: vec![None; n],
            groups: Vec::new(),
            order: Vec::new(),
        }
    }

    /// How many fragments the collection has.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// True when the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// `n in poses` — whether the fragment has been placed.
    #[inline]
    pub fn placed(&self, n: FragId) -> bool {
        self.poses[n as usize].is_some()
    }

    /// `poses[n]`, or `None` while `n` is unplaced.
    #[inline]
    pub fn pose(&self, n: FragId) -> Option<Matrix4<f64>> {
        self.poses[n as usize]
    }

    /// `group_of[n]`, or `None` while `n` is unplaced.
    #[inline]
    pub fn group_of(&self, n: FragId) -> Option<usize> {
        self.group_of[n as usize]
    }

    /// The members of one group, in placement order.
    #[inline]
    pub fn members(&self, g: usize) -> &[FragId] {
        &self.groups[g]
    }

    /// The groups, in the order they were seeded.
    #[inline]
    pub fn groups(&self) -> &[Vec<FragId>] {
        &self.groups
    }

    /// The fragments in the order they were placed: the reference's `poses` is a dict and R §11.1
    /// writes `transforms.json` in its insertion order.
    #[inline]
    pub fn order(&self) -> &[FragId] {
        &self.order
    }

    /// R §8's seed: a new group holding `a` at the identity and `b` at `transform`.
    ///
    /// The seed is placed without any check. Nothing has been placed for it to disagree with, and
    /// the reference makes the same choice: the best remaining join whose two fragments are both
    /// unplaced starts a group on its own authority.
    pub fn seed(&mut self, a: FragId, b: FragId, transform: Matrix4<f64>) -> usize {
        let g = self.groups.len();
        self.groups.push(vec![a, b]);
        self.poses[a as usize] = Some(Matrix4::identity());
        self.poses[b as usize] = Some(transform);
        self.order.push(a);
        self.order.push(b);
        self.group_of[a as usize] = Some(g);
        self.group_of[b as usize] = Some(g);
        g
    }

    /// R §8: `new` joins the group `placed` is in, at the world pose `transform`.
    pub fn place(&mut self, placed: FragId, new: FragId, transform: Matrix4<f64>) {
        let g = self.group_of[placed as usize].expect("the anchor of a placement is placed");
        self.poses[new as usize] = Some(transform);
        self.group_of[new as usize] = Some(g);
        self.groups[g].push(new);
        self.order.push(new);
    }

    /// R §8's `group_thickness`: the median wall over a set of fragments.
    ///
    /// The wall a pose disagreement is measured in is the *group's*, not the collection's — the
    /// reference's own reason being that a collection may hold pots of very different walls and
    /// its median describes none of them. `extra` is R §8's `groups[g] + [new]`: the candidate
    /// member is counted before it is placed.
    pub fn thickness(&self, pieces: &[Piece<'_>], g: usize, extra: Option<FragId>) -> f64 {
        let mut walls: Vec<f64> =
            self.groups[g].iter().map(|&n| pieces[n as usize].thick).collect();
        if let Some(n) = extra {
            walls.push(pieces[n as usize].thick);
        }
        median(&walls)
    }

    /// R §8's tail: every fragment still unplaced becomes a singleton group at the identity.
    ///
    /// The iteration is over [`FragId`] in order, which is `list(md.keys())` — R §2's discovery
    /// order — on the reference's side.
    pub fn add_singletons(&mut self) {
        for n in 0..self.poses.len() {
            if self.poses[n].is_some() {
                continue;
            }
            let g = self.groups.len();
            let id = u32::try_from(n).expect("a collection has fewer than 2^32 fragments");
            self.poses[n] = Some(Matrix4::identity());
            self.group_of[n] = Some(g);
            self.order.push(id);
            self.groups.push(vec![id]);
        }
    }

    /// R §8's last line: `groups.sort(key=lambda g: -len(g))`, a **stable** sort by size.
    ///
    /// Stability is the whole content of the line: two groups of the same size come back in the
    /// order they were seeded, which for singletons is R §2's order. `group_of` is not renumbered,
    /// because the reference does not renumber it either and nothing reads it afterwards.
    pub fn sort_by_size(&mut self) {
        self.groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    }

    /// Every pose, by [`FragId`]; unplaced fragments come back at the identity.
    ///
    /// Only called after [`add_singletons`](Self::add_singletons), where nothing is unplaced.
    pub fn into_poses(self) -> Vec<Matrix4<f64>> {
        self.poses.into_iter().map(|t| t.unwrap_or_else(Matrix4::identity)).collect()
    }
}

/// R §8.2's `recenter`: each group translated so that its own centroid sits at the origin.
///
/// Cosmetic — it keeps the numbers in `transforms.json` small — and it applies to **every** group,
/// singletons included. The centroid is over every tenth surface sample of every member, moved
/// through that member's pose, which is `md[n].S[::10]` on the reference's side and therefore the
/// same `s_pen` the penetration test above casts.
///
/// The mean is numpy's: `pts.mean(0)` reduces a `(N, 3)` array along its first axis, which numpy
/// walks **row by row** — a running sum per column, not [`pairwise_sum`], measured on the
/// fixtures' own arrays (V4-D10) and factored out as [`column_mean`]. The points are moved with
/// [`apply_transform_fused`], which is the reference's `P @ T[:3,:3].T + T[:3,3]`. Only the
/// translation moves; the rotation block is untouched.
pub fn recenter(
    poses: &[Matrix4<f64>],
    pieces: &[Piece<'_>],
    groups: &[Vec<FragId>],
) -> Vec<Matrix4<f64>> {
    let mut out = poses.to_vec();
    for g in groups {
        let mut points = Vec::new();
        for &n in g {
            let pose = poses[n as usize];
            for point in pieces[n as usize].s_pen.iter().step_by(10) {
                points.push(apply_transform_fused(&pose, *point));
            }
        }
        if points.is_empty() {
            continue;
        }
        let centre = column_mean(&points);
        for &n in g {
            for (i, c) in centre.into_iter().enumerate() {
                out[n as usize][(i, 3)] -= c;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the medians and translations here are exact")]

    use super::{Grouping, Piece, recenter};
    use nalgebra::Matrix4;

    fn piece(thick: f64, s: &[[f64; 3]]) -> Piece<'_> {
        Piece { thick, res: 1.0, watertight: true, mesh: None, s_pen: s }
    }

    #[test]
    fn a_group_thickness_is_the_median_of_its_members_own_walls() {
        let points: Vec<[f64; 3]> = Vec::new();
        let pieces = [piece(1.0, &points), piece(5.0, &points), piece(9.0, &points)];
        let mut g = Grouping::new(3);
        g.seed(0, 1, Matrix4::identity());
        assert_eq!(g.thickness(&pieces, 0, None), 3.0, "the mean of the two middle walls");
        assert_eq!(
            g.thickness(&pieces, 0, Some(2)),
            5.0,
            "the new member is counted before it is placed"
        );
    }

    #[test]
    fn singletons_are_added_in_fragment_order_and_the_sort_is_stable() {
        let mut g = Grouping::new(5);
        g.seed(3, 1, Matrix4::identity());
        g.add_singletons();
        assert_eq!(g.groups(), [vec![3, 1], vec![0], vec![2], vec![4]]);
        g.sort_by_size();
        assert_eq!(
            g.groups(),
            [vec![3, 1], vec![0], vec![2], vec![4]],
            "equal sizes keep their order"
        );
        assert!(g.placed(4) && g.pose(4) == Some(Matrix4::identity()));
    }

    /// R §8.2 moves a group's centroid to the origin and leaves its rotation alone.
    #[test]
    fn recentring_puts_every_groups_centroid_at_the_origin() {
        // Eleven points so that the `[::10]` stride takes exactly the first and the eleventh.
        let mut a = vec![[0.0, 0.0, 0.0]; 11];
        a[0] = [2.0, 4.0, 6.0];
        a[10] = [4.0, 8.0, 12.0];
        let b = vec![[100.0, 0.0, 0.0]; 11];
        let pieces = [piece(1.0, &a), piece(1.0, &b)];
        let poses = [Matrix4::identity(), Matrix4::identity()];

        let out = recenter(&poses, &pieces, &[vec![0], vec![1]]);
        // Group 0's centroid is the mean of (2,4,6) and (4,8,12).
        assert_eq!(out[0][(0, 3)], -3.0);
        assert_eq!(out[0][(1, 3)], -6.0);
        assert_eq!(out[0][(2, 3)], -9.0);
        // Group 1's two sampled points coincide, so its centroid is that point.
        assert_eq!(out[1][(0, 3)], -100.0);
        assert_eq!(out[1].fixed_view::<3, 3>(0, 0), Matrix4::identity().fixed_view::<3, 3>(0, 0));

        // One group of both: the centroid is the mean over all four sampled points.
        let out = recenter(&poses, &pieces, &[vec![0, 1]]);
        assert_eq!(out[0][(0, 3)], -(2.0 + 4.0 + 100.0 + 100.0) / 4.0);
        assert_eq!(out[1][(0, 3)], -(2.0 + 4.0 + 100.0 + 100.0) / 4.0);
    }
}
