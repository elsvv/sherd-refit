//! R §8's greedy assembly on hand-made candidate graphs.
//!
//! The parity harness runs R §8 on the eight fixture dumps, where every join is a real candidate
//! and every number came out of R §6; these tests do the opposite, and that is what they are for.
//! Each graph is four or five joins with poses chosen so that exactly one branch of the loop can
//! fire, so a failure names the branch: a chain that must grow in one direction, a triangle whose
//! third edge closes a loop it disagrees with, two objects that must stay apart, and a join whose
//! fragment would sit inside one already placed.
//!
//! The poses are translations and quarter turns rather than anything a matcher would produce —
//! `10 t` apart is a disagreement whatever the geometry — so the assertions are on the *decisions*
//! and on the sentences the report prints, which is what R §8 contributes.

#![allow(
    clippy::float_cmp,
    reason = "these poses are whole numbers R §8 copies rather than computes"
)]

use nalgebra::Matrix4;
use sherd_core::Params;
use sherd_core::assembly::constraints::{Constraints, Veto, resolve};
use sherd_core::assembly::{Piece, Rejection, assemble, assemble_under, recenter};
use sherd_core::executor::CPU;
use sherd_core::matching::pair::{Candidate, Gate};
use sherd_core::matching::verify::Scores;
use sherd_core::spatial::bvh::RayScene;
use sherd_core::tiers::Tier;
use sherd_core::types::FragId;

/// A join with the given ranking score (R §5.7's `seam · tight`) and a pure translation.
fn join(a: FragId, b: FragId, tau: [f64; 3], score: f64) -> Candidate {
    pose_join(a, b, translation(tau), score)
}

/// A join at an arbitrary pose.
fn pose_join(a: FragId, b: FragId, transform: Matrix4<f64>, score: f64) -> Candidate {
    let scores = Scores { seam: score, tight: 1.0, ..Scores::default() };
    Candidate { a, b, transform, scores, accepted: true, tier: Tier::Probable }
}

fn translation(tau: [f64; 3]) -> Matrix4<f64> {
    let mut m = Matrix4::identity();
    for (i, t) in tau.into_iter().enumerate() {
        m[(i, 3)] = t;
    }
    m
}

/// A quarter turn about `z`, with a translation.
fn quarter_turn(tau: [f64; 3]) -> Matrix4<f64> {
    let mut m = translation(tau);
    m[(0, 0)] = 0.0;
    m[(0, 1)] = -1.0;
    m[(1, 0)] = 1.0;
    m[(1, 1)] = 0.0;
    m
}

/// `n` fragments with a unit wall, no mesh and no samples: everything but the penetration test.
fn bare(n: usize) -> Vec<Piece<'static>> {
    (0..n)
        .map(|_| Piece { thick: 1.0, res: 0.01, watertight: true, mesh: None, s_pen: &[] })
        .collect()
}

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("frag_{i}")).collect()
}

/// The pairs of the joins that were used, in the order they were taken.
fn used_pairs(candidates: &[Candidate], used: &[usize]) -> Vec<(FragId, FragId)> {
    used.iter().map(|&i| (candidates[i].a, candidates[i].b)).collect()
}

/// A closed unit cube at `[0, 1]³`, and the interior points a penetration test casts.
fn unit_cube() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let v = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let f = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    (v, f)
}

/// Twenty-seven points well inside that cube — every one at least `0.13` from a face, which is
/// twice R §1.2's `pen` at a unit wall.
///
/// The coordinates are deliberately irregular. A cube is *two* triangles per face, and an
/// axis-aligned ray from a point whose coordinates are equal or sum to one leaves through one of
/// those diagonals, where a crossing count is not a function of the geometry — the case
/// [`RayScene::inside`]'s majority of three exists for, and one an axis-aligned box can present on
/// two of the three rays at once. No pair of the values below is equal and no pair sums to one, so
/// every ray leaves through the interior of a triangle and the inside test is exact.
fn cube_interior() -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for x in [0.13, 0.22, 0.31] {
        for y in [0.44, 0.53, 0.62] {
            for z in [0.71, 0.77, 0.83] {
                out.push([x, y, z]);
            }
        }
    }
    out
}

/// A chain `0–1–2–3` grows into one group, in score order, with no rejection.
#[test]
fn a_chain_grows_from_its_strongest_join() {
    let pieces = bare(4);
    let candidates = vec![
        join(0, 1, [10.0, 0.0, 0.0], 3.0),
        join(1, 2, [10.0, 0.0, 0.0], 2.0),
        join(2, 3, [10.0, 0.0, 0.0], 1.0),
    ];
    let out = assemble(&CPU, &pieces, &candidates, &Params::default());

    assert_eq!(out.groups, [vec![0, 1, 2, 3]]);
    assert_eq!(used_pairs(&candidates, &out.used), [(0, 1), (1, 2), (2, 3)]);
    assert!(out.rejected.is_empty(), "{:?}", out.rejected);
    // The seed is at the identity and every link composes onto the one before it.
    for (n, x) in [(0, 0.0), (1, 10.0), (2, 20.0), (3, 30.0)] {
        assert_eq!(out.poses[n][(0, 3)], x, "fragment {n}");
    }
}

/// A triangle whose third edge disagrees: the edge closes a loop, so it is judged against the
/// poses already placed rather than allowed to move anything.
#[test]
fn a_loop_closing_edge_that_disagrees_is_rejected_and_moves_nothing() {
    let pieces = bare(3);
    let candidates = vec![
        join(0, 1, [10.0, 0.0, 0.0], 3.0),
        join(1, 2, [10.0, 0.0, 0.0], 2.0),
        // 0 → 2 the long way round: a quarter turn, and 20 units off where the chain puts it.
        pose_join(0, 2, quarter_turn([40.0, 0.0, 0.0]), 1.0),
    ];
    let out = assemble(&CPU, &pieces, &candidates, &Params::default());

    assert_eq!(out.groups, [vec![0, 1, 2]]);
    assert_eq!(used_pairs(&candidates, &out.used), [(0, 1), (1, 2)]);
    assert_eq!(out.rejected.len(), 1);
    assert_eq!(out.rejected[0].candidate, 2);
    let Rejection::InconsistentWithAssembled { angle, distance } = out.rejected[0].reason else {
        panic!("a loop-closing edge is judged against the assembly: {:?}", out.rejected[0]);
    };
    assert!((angle - 90.0).abs() < 1e-9, "{angle}");
    assert!(distance > 0.5, "{distance}");
    assert_eq!(
        out.rejected[0].reason.message(&names(3)),
        "inconsistent with the assembled poses (90.0 deg, 20.00 t)"
    );
    // 2 is where the chain put it, not where the rejected edge wanted it.
    assert_eq!(out.poses[2][(0, 3)], 20.0);

    // The same triangle with a loop-closing edge that *agrees* keeps it as a used join.
    let agreeing = vec![candidates[0], candidates[1], join(0, 2, [20.0, 0.0, 0.0], 1.0)];
    let out = assemble(&CPU, &pieces, &agreeing, &Params::default());
    assert_eq!(used_pairs(&agreeing, &out.used), [(0, 1), (1, 2), (0, 2)]);
    assert!(out.rejected.is_empty(), "{:?}", out.rejected);
}

/// Two objects and a fragment belonging to neither: three groups, biggest first, singleton last.
#[test]
fn two_objects_stay_apart_and_the_odd_fragment_is_a_singleton() {
    let pieces = bare(6);
    let candidates = vec![
        join(0, 1, [10.0, 0.0, 0.0], 4.0),
        join(1, 2, [10.0, 0.0, 0.0], 3.0),
        join(3, 4, [10.0, 0.0, 0.0], 2.0),
    ];
    let out = assemble(&CPU, &pieces, &candidates, &Params::default());

    assert_eq!(out.groups, [vec![0, 1, 2], vec![3, 4], vec![5]]);
    assert_eq!(used_pairs(&candidates, &out.used), [(0, 1), (1, 2), (3, 4)]);
    assert!(out.rejected.is_empty(), "{:?}", out.rejected);
    // The second object is seeded at its own identity, not somewhere near the first.
    assert_eq!(out.poses[3], Matrix4::identity());
    assert_eq!(out.poses[5], Matrix4::identity(), "an unplaced fragment stays at the identity");
}

/// **R §8 never puts a join between two of its own groups**, so `Rejection::MergesGroups` — and
/// with it audit §D.2 (c)'s merge, which replaces that rejection — is unreachable.
///
/// Task O1 measured the branch firing **zero times** over the eight development sets at seeds 0–4
/// and went looking for the reason. It is R §8's own greedy rule: a group grows until no remaining
/// join touches it, and a second group is seeded only in a pass where *nothing* could be placed —
/// a pass that scans every remaining join. So at the moment a second group is seeded, every join
/// that survives has both fragments unplaced, and the next join to touch either of them extends a
/// group rather than spanning two. A join whose placement R §8 refuses is removed and cannot span
/// anything either. The groups are the connected components of the joins R §8 did not refuse.
///
/// This is the measurement rather than the argument: four thousand random graphs of six fragments,
/// with poses that make roughly a third of the placements disagree, and **not one** of them ever
/// reaches the branch. The property asserted is the stronger one — no accepted join of the list
/// ends with its two fragments in different groups.
#[test]
fn r_8s_groups_are_components_so_no_join_ever_spans_two_of_them() {
    const N: u32 = 8;
    let pieces = bare(N as usize);
    let params = Params::default();
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut merges_seen = 0_usize;
    let mut cross_group_joins = 0_usize;
    let mut graphs_with_two_groups = 0_usize;
    let mut refusals = 0_usize;
    for _ in 0..4_000 {
        let mut candidates = Vec::new();
        for a in 0..N {
            for b in (a + 1)..N {
                if next() % 4 != 0 {
                    continue; // a quarter of the pairs have an accepted candidate
                }
                #[allow(clippy::cast_precision_loss, reason = "a small pseudo-random score")]
                let score = ((next() % 1000) as f64) / 100.0;
                // Poses far enough apart that a loop-closing edge or a second path routinely
                // disagrees, which is what exercises R §8's two refusals.
                #[allow(clippy::cast_precision_loss, reason = "a small pseudo-random offset")]
                let tau = [((next() % 40) as f64) - 20.0, 0.0, 0.0];
                candidates.push(join(a, b, tau, score));
            }
        }
        let out = assemble(&CPU, &pieces, &candidates, &params);
        merges_seen += out.merges;
        refusals += out.rejected.len();
        if out.groups.iter().filter(|g| g.len() > 1).count() > 1 {
            graphs_with_two_groups += 1;
        }
        assert!(
            !out.rejected.iter().any(|r| r.reason == Rejection::MergesGroups),
            "R §8 reached a branch this test says is unreachable"
        );
        let mut group_of = vec![usize::MAX; N as usize];
        for (g, members) in out.groups.iter().enumerate() {
            for &n in members {
                group_of[n as usize] = g;
            }
        }
        for c in &candidates {
            if group_of[c.a as usize] != group_of[c.b as usize] {
                cross_group_joins += 1;
            }
        }
    }
    assert_eq!(merges_seen, 0, "and no merge, because nothing ever proposes one");
    assert!(graphs_with_two_groups > 100, "the sample did produce multi-group assemblies");
    assert!(refusals > 100, "and it did exercise R §8's own refusals");
    assert_eq!(
        cross_group_joins, 0,
        "not one accepted join of four thousand graphs ends with its two fragments in different \
         groups: R §8's groups are the connected components of the joins it kept"
    );
}

/// Groups of equal size come back in the order they were seeded, and that order is the candidate
/// list's — R §8's sort is stable and so is `best_per_pair`'s.
#[test]
fn equal_scores_and_equal_sizes_are_broken_by_the_candidate_order() {
    let pieces = bare(4);
    let first = vec![join(2, 3, [1.0, 0.0, 0.0], 5.0), join(0, 1, [1.0, 0.0, 0.0], 5.0)];
    assert_eq!(
        assemble(&CPU, &pieces, &first, &Params::default()).groups,
        [vec![2, 3], vec![0, 1]]
    );

    let second = vec![first[1], first[0]];
    assert_eq!(
        assemble(&CPU, &pieces, &second, &Params::default()).groups,
        [vec![0, 1], vec![2, 3]]
    );
}

/// Only accepted candidates count, and a pair is represented by the **first** of its candidates at
/// the top score.
#[test]
fn a_pair_is_represented_by_its_first_best_accepted_candidate() {
    let pieces = bare(2);
    let mut rejected_by_verification = join(0, 1, [99.0, 0.0, 0.0], 9.0);
    rejected_by_verification.accepted = false;
    let candidates = vec![
        rejected_by_verification,
        join(0, 1, [10.0, 0.0, 0.0], 4.0),
        join(0, 1, [20.0, 0.0, 0.0], 4.0),
        join(0, 1, [30.0, 0.0, 0.0], 1.0),
    ];
    let out = assemble(&CPU, &pieces, &candidates, &Params::default());
    assert_eq!(out.accepted, [1], "the strict `>` keeps the first candidate at the best score");
    assert_eq!(out.poses[1][(0, 3)], 10.0);
}

/// A join whose fragment would sit inside one already placed is rejected, and the weaker join that
/// then proposes the same fragment somewhere else is rejected against it.
///
/// Three unit cubes. `0–1` places 1 clear of 0. `0–2` would put 2 exactly where 1 is, which is
/// R §6.4's penetration against a *group member that is not the anchor* — the check R §6.5 cannot
/// make, because the pair `0–2` never sees fragment 1. The weaker `1–2` then puts 2 somewhere
/// harmless, but a stronger accepted join says 2 belongs elsewhere, so it is refused too.
#[test]
fn a_penetrating_join_is_rejected_and_so_is_the_weaker_join_that_disagrees_with_it() {
    let (v, f) = unit_cube();
    let scene = RayScene::new(&v, &f).expect("a closed cube is a scene");
    let interior = cube_interior();
    let pieces: Vec<Piece<'_>> = (0..3)
        .map(|_| Piece {
            thick: 1.0,
            res: 0.01,
            watertight: true,
            mesh: Some(&scene),
            s_pen: &interior,
        })
        .collect();
    let candidates = vec![
        join(0, 1, [1.5, 0.0, 0.0], 3.0),
        join(0, 2, [1.5, 0.0, 0.0], 2.0),
        join(1, 2, [1.5, 0.0, 0.0], 1.0),
    ];
    let out = assemble(&CPU, &pieces, &candidates, &Params::default());

    assert_eq!(out.groups, [vec![0, 1], vec![2]]);
    assert_eq!(used_pairs(&candidates, &out.used), [(0, 1)]);
    assert_eq!(out.rejected.len(), 2, "{:?}", out.rejected);

    assert_eq!(out.rejected[0].candidate, 1);
    let Rejection::Penetrates { other, pen } = out.rejected[0].reason else {
        panic!("2 lands exactly on 1: {:?}", out.rejected[0]);
    };
    assert_eq!(other, 1);
    assert!((pen - 1.0).abs() < 1e-12, "every sample is inside: {pen}");
    assert_eq!(out.rejected[0].reason.message(&names(3)), "penetrates frag_1 (1.000)");

    assert_eq!(out.rejected[1].candidate, 2);
    assert_eq!(
        out.rejected[1].reason,
        Rejection::InconsistentWithStronger { a: 0, b: 2, angle: 0.0, distance: 1.5 }
    );
    assert_eq!(
        out.rejected[1].reason.message(&names(3)),
        "inconsistent with stronger join frag_0-frag_2 (0.0 deg, 1.50 t)"
    );

    // The same graph with the cubes declared open: R §6.4 has no answer, so nothing penetrates and
    // the join that was refused above is taken.
    let open: Vec<Piece<'_>> = (0..3)
        .map(|_| Piece {
            thick: 1.0,
            res: 0.01,
            watertight: false,
            mesh: Some(&scene),
            s_pen: &interior,
        })
        .collect();
    let out = assemble(&CPU, &open, &candidates, &Params::default());
    assert_eq!(out.groups, [vec![0, 1, 2]]);
    assert_eq!(used_pairs(&candidates, &out.used), [(0, 1), (0, 2)]);
}

/// R §8.2 recentres every group, singletons included, and leaves the rotations alone.
#[test]
fn recentring_moves_each_group_to_its_own_centroid() {
    let interior = cube_interior();
    let pieces: Vec<Piece<'_>> = (0..3)
        .map(|_| Piece { thick: 1.0, res: 0.01, watertight: true, mesh: None, s_pen: &interior })
        .collect();
    let poses =
        [translation([100.0, 0.0, 0.0]), translation([110.0, 0.0, 0.0]), Matrix4::identity()];
    let groups = vec![vec![0, 1], vec![2]];

    let out = recenter(&poses, &pieces, &groups);
    // R §8.2's `S[::10]` takes samples 0, 10 and 20 of the 27.
    let inside = [0, 10, 20].iter().map(|&i| interior[i][0]).sum::<f64>() / 3.0;
    let centre = ((100.0 + inside) + (110.0 + inside)) / 2.0;
    assert!((out[0][(0, 3)] - (100.0 - centre)).abs() < 1e-12);
    assert!((out[1][(0, 3)] - (110.0 - centre)).abs() < 1e-12);
    assert!((out[2][(0, 3)] + inside).abs() < 1e-12, "a singleton is recentred too");
    assert_eq!(out[0].fixed_view::<3, 3>(0, 0), Matrix4::identity().fixed_view::<3, 3>(0, 0));
}

/// Audit §D.1's two ways a constraint reaches R §8: a vetoed pair is refused before any test of
/// R §8's own, and a `must_join` pair is offered first whatever its score.
///
/// The graph is a chain whose strongest join is `0-1`; the constraint names the *weakest*, so the
/// order the loop sees is the assertion. Without a constraint the same graph grows from `0-1`
/// (`a_chain_grows_from_its_strongest_join` above), which is what makes the difference legible.
#[test]
fn a_must_join_pair_seeds_the_assembly_and_a_vetoed_pair_is_refused() {
    let pieces = bare(4);
    let candidates = vec![
        join(0, 1, [10.0, 0.0, 0.0], 3.0),
        join(1, 2, [10.0, 0.0, 0.0], 2.0),
        join(2, 3, [10.0, 0.0, 0.0], 1.0),
    ];
    let names = names(4);
    let file: Constraints = serde_json::from_str(
        r#"{"must_join": [["frag_2", "frag_3"]], "different_object": [["frag_0", "frag_1"]]}"#,
    )
    .expect("the format parses");
    let plan = resolve(&file, &names).expect("the names are the collection's");
    let out = assemble_under(
        &CPU,
        &pieces,
        &candidates,
        &Params::default(),
        Gate::Accepted,
        Some(&plan),
        None,
    );

    // `2-3` was seeded although it is the weakest join, so `1-2` grew onto it and `0-1` never had
    // a chance to be the seed.
    assert_eq!(used_pairs(&candidates, &out.used), [(2, 3), (1, 2)]);
    assert_eq!(out.groups, [vec![2, 3, 1], vec![0]], "0 is refused and stays a singleton");
    assert_eq!(out.rejected.len(), 1, "{:?}", out.rejected);
    assert_eq!(out.rejected[0].reason, Rejection::Constrained(Veto::DifferentObject));
    assert_eq!(
        out.rejected[0].reason.message(&names),
        "refused by constraints.json (`different_object`)",
        "the report says whose decision it was"
    );
}

/// `must_not_join` refuses the same way, and with its own name in the sentence — the pipeline
/// removes such a pair before matching, so this is the belt to that braces.
#[test]
fn must_not_join_refuses_a_candidate_that_reaches_the_assembly() {
    let pieces = bare(3);
    let candidates = vec![join(0, 1, [10.0, 0.0, 0.0], 3.0), join(1, 2, [10.0, 0.0, 0.0], 2.0)];
    let names = names(3);
    let file: Constraints =
        serde_json::from_str(r#"{"must_not_join": [["frag_1", "frag_0"]]}"#).expect("it parses");
    let plan = resolve(&file, &names).expect("the names are the collection's");
    let out = assemble_under(
        &CPU,
        &pieces,
        &candidates,
        &Params::default(),
        Gate::Accepted,
        Some(&plan),
        None,
    );
    assert_eq!(used_pairs(&candidates, &out.used), [(1, 2)]);
    assert_eq!(out.rejected.len(), 1);
    assert_eq!(out.rejected[0].reason, Rejection::Constrained(Veto::MustNotJoin));
}

/// And with no constraints at all, `assemble_under` is `assemble`: same joins, same order, same
/// poses. The off switch of a new behaviour, at the level R §8 sees it.
#[test]
fn no_constraints_is_the_assembly_r_8_always_made() {
    let pieces = bare(4);
    let candidates = vec![
        join(0, 1, [10.0, 0.0, 0.0], 3.0),
        join(1, 2, [10.0, 0.0, 0.0], 2.0),
        join(2, 3, [10.0, 0.0, 0.0], 1.0),
    ];
    let p = Params::default();
    let plain = assemble(&CPU, &pieces, &candidates, &p);
    let under = assemble_under(&CPU, &pieces, &candidates, &p, Gate::Accepted, None, None);
    assert_eq!(plain.used, under.used);
    assert_eq!(plain.groups, under.groups);
    assert_eq!(plain.order, under.order);
    assert_eq!(plain.poses, under.poses);
}
