//! R §8's greedy pass: accepted pairwise joins, taken best first, grown into groups.
//!
//! The rule is one line long — *take the strongest join that extends something already placed, and
//! restart* — and everything below it is the bookkeeping that makes the answer a function of the
//! candidate list and of nothing else: which candidate represents a pair, what order the pairs are
//! walked in, which join a disagreement blames, and what happens to a join whose two fragments are
//! already placed. Every one of those is a tie the reference resolves in a particular way, and this
//! module resolves them the same way (see [`assemble`]).
//!
//! What comes out is R §8's four values: a pose per fragment, the groups, the joins that built the
//! assembly and the accepted joins that were skipped with the reason each was skipped. R §8.1's
//! second pass — off by default — runs the whole of this again on a candidate list with some pairs
//! rematched, so it needs nothing from here but a second call.

use std::collections::BTreeMap;

use nalgebra::Matrix4;

use crate::executor::Executor;
use crate::matching::pair::{Candidate, Gate};
use crate::matching::verify::pose_inverse;
use crate::params::Params;
use crate::types::FragId;

use super::consistency::{agrees, disagreement, penetration};
use super::constraints::{Resolved, Veto};
use super::groups::{Grouping, Piece};
use crate::objects::ObjectParams;
use crate::tiers::Tier;

/// Why an accepted join did not become part of the assembly (R §8).
///
/// The reference carries the reason as an English sentence and `report.json` prints it, so
/// [`Rejection::message`] reproduces those sentences exactly, down to the number of decimals;
/// the variants exist so that the harness and the report can read the *kind* without parsing prose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rejection {
    /// The fragment this join would place lies inside a fragment already in the group.
    Penetrates {
        /// The placed fragment it penetrates.
        other: FragId,
        /// R §6.4's fraction, as measured at the proposed pose.
        pen: f64,
    },
    /// Another accepted join with a **higher** score puts the same fragment somewhere else.
    InconsistentWithStronger {
        /// That join's first fragment.
        a: FragId,
        /// Its second.
        b: FragId,
        /// How far the two poses disagree, in degrees.
        angle: f64,
        /// And in wall thicknesses of the group.
        distance: f64,
    },
    /// A loop-closing join inside one group that does not agree with the poses already placed.
    InconsistentWithAssembled {
        /// How far the two poses disagree, in degrees.
        angle: f64,
        /// And in wall thicknesses of the group.
        distance: f64,
    },
    /// Both fragments are placed, in **different** groups. R §8 does not merge groups.
    MergesGroups,
    /// The same, on a run that *can* merge them (roadmap item 4): this join is not confirmed, and
    /// audit §D.2 (c) gives the merge to a confirmed join alone.
    ///
    /// A separate variant, and a separate sentence, because the two are different facts about the
    /// run: `MergesGroups` says the tool cannot do it, this says the evidence does not justify it.
    MergeNotConfirmed,
    /// A merge (audit §D.2 (c)) that some other join between the two groups contradicts.
    MergeInconsistent {
        /// That join's first fragment.
        a: FragId,
        /// Its second.
        b: FragId,
        /// How far the merged poses and that join disagree, in degrees.
        angle: f64,
        /// And in wall thicknesses of the two groups together.
        distance: f64,
    },
    /// `constraints.json` refuses the pair (audit §D.1): `must_not_join`, or `different_object`.
    ///
    /// Not R §8's own refusal, and it says so: the operator's word is the reason, and the report
    /// prints the key of the list the pair was written in so that the sentence can be traced back
    /// to a line of the file.
    Constrained(crate::assembly::constraints::Veto),
}

impl Rejection {
    /// The reference's own sentence for this rejection, given the collection's fragment names.
    ///
    /// The formats are R §8's: `%.3f` on a penetration fraction, `%.1f` on an angle and `%.2f` on a
    /// distance. Both languages round a decimal representation half to even from the exact binary
    /// value, so the strings agree digit for digit; `report.json` carries them and the parity
    /// harness compares them.
    pub fn message(&self, names: &[String]) -> String {
        let name = |n: FragId| names.get(n as usize).map_or("?", String::as_str);
        match *self {
            Self::Penetrates { other, pen } => format!("penetrates {} ({pen:.3})", name(other)),
            Self::InconsistentWithStronger { a, b, angle, distance } => format!(
                "inconsistent with stronger join {}-{} ({angle:.1} deg, {distance:.2} t)",
                name(a),
                name(b)
            ),
            Self::InconsistentWithAssembled { angle, distance } => {
                format!("inconsistent with the assembled poses ({angle:.1} deg, {distance:.2} t)")
            }
            Self::MergesGroups => "would merge two groups (not supported)".to_owned(),
            Self::MergeNotConfirmed => {
                "would merge two groups (only a confirmed join may)".to_owned()
            }
            Self::MergeInconsistent { a, b, angle, distance } => format!(
                "merging the two groups disagrees with join {}-{} ({angle:.1} deg, {distance:.2} t)",
                name(a),
                name(b)
            ),
            Self::Constrained(veto) => {
                format!("refused by constraints.json (`{}`)", veto.key())
            }
        }
    }
    /// The same sentence with every *measured* number taken out: the kind of refusal and, where
    /// there is one, the fragment or join it names.
    ///
    /// The decision is what R §8 makes; the numbers beside it are statistics of the pose and of the
    /// sample set the penetration test happened to be run on, and PMC-8 lets the port change the
    /// latter. A comparison that has to survive PMC-8 compares this and reports the movement of
    /// the number separately.
    pub fn kind(&self, names: &[String]) -> String {
        let name = |n: FragId| names.get(n as usize).map_or("?", String::as_str);
        match *self {
            Self::Penetrates { other, .. } => format!("penetrates {}", name(other)),
            Self::InconsistentWithStronger { a, b, .. } => {
                format!("inconsistent with stronger join {}-{}", name(a), name(b))
            }
            Self::InconsistentWithAssembled { .. } => {
                "inconsistent with the assembled poses".to_owned()
            }
            Self::MergesGroups => "would merge two groups (not supported)".to_owned(),
            Self::MergeNotConfirmed => {
                "would merge two groups (only a confirmed join may)".to_owned()
            }
            Self::MergeInconsistent { a, b, .. } => {
                format!("merging the two groups disagrees with join {}-{}", name(a), name(b))
            }
            Self::Constrained(veto) => {
                format!("refused by constraints.json (`{}`)", veto.key())
            }
        }
    }

    /// R §6.4's fraction, for a rejection that measured one.
    pub fn penetration(&self) -> Option<f64> {
        match *self {
            Self::Penetrates { pen, .. } => Some(pen),
            _ => None,
        }
    }
}

/// One accepted join that did not make it into the assembly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rejected {
    /// Index into the candidate list [`assemble`] was given.
    pub candidate: usize,
    /// Why.
    pub reason: Rejection,
}

/// R §8's four return values.
#[derive(Clone, Debug)]
pub struct Assembly {
    /// One world pose per fragment, by [`FragId`]; an unplaced fragment is at the identity.
    pub poses: Vec<Matrix4<f64>>,
    /// The groups, sorted by size descending (stable), singletons last.
    pub groups: Vec<Vec<FragId>>,
    /// The joins that built the assembly, in the order they were taken — indices into the
    /// candidate list.
    pub used: Vec<usize>,
    /// The accepted joins that were skipped, in the order they were skipped.
    pub rejected: Vec<Rejected>,
    /// R §8's `accepted`: the best candidate of every pair that had one, in score order — indices
    /// into the candidate list, and the order everything above was decided in.
    pub accepted: Vec<usize>,
    /// The order the poses were filled in: each seed's two fragments, then every placement, then
    /// the singletons in collection order. R §11.1's `transforms.json` is written in it, because
    /// the reference's `poses` is an insertion-ordered dict (V4-D5).
    pub order: Vec<FragId>,
    /// How many times two groups were merged through a confirmed join (audit §D.2 (c)).
    ///
    /// Always `0` without [`ObjectParams::merge`], which is what makes it safe to report
    /// unconditionally: a run of R §8 as the reference wrote it merges nothing.
    pub merges: usize,
}

/// R §8's `best_per_pair`, then its sort: one candidate per pair, strongest pair first.
///
/// Two tie rules live in these four lines and both are the reference's.
///
/// * **Within a pair**, `c.score > best.score` is strict, so the *first* candidate at the top score
///   represents the pair — and R §5.7 already returned that pair's candidates sorted, so it is the
///   pair's own best.
/// * **Between pairs**, `sorted(..., key=-score)` is Python's stable sort over `dict.values()`,
///   whose order is insertion order: pairs tied on score come back in the order the matcher
///   produced them, which is R §4.1's `itertools.combinations` order. `sort_by` in Rust is stable
///   too, and the insertion order is tracked explicitly rather than left to a hash map.
///
/// A third rule joins them when a `constraints.json` is in force: a `must_join` pair sorts **before
/// every other join**, whatever its score, which is audit §D.1's *"seeded first"*. With no
/// constraints the key is constant and a stable sort leaves the two rules above exactly as they
/// are, which is what keeps a run without the file byte for byte the run it was.
fn best_per_pair(
    candidates: &[Candidate],
    gate: Gate,
    constraints: Option<&Resolved>,
) -> Vec<usize> {
    let mut best: BTreeMap<(FragId, FragId), usize> = BTreeMap::new();
    let mut order: Vec<(FragId, FragId)> = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        if !c.admitted(gate) {
            continue;
        }
        match best.entry((c.a, c.b)) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(i);
                order.push((c.a, c.b));
            }
            std::collections::btree_map::Entry::Occupied(mut slot) => {
                if c.score() > candidates[*slot.get()].score() {
                    slot.insert(i);
                }
            }
        }
    }
    let mut accepted: Vec<usize> = order.iter().map(|key| best[key]).collect();
    let forced = |i: usize| {
        let c = &candidates[i];
        u8::from(!constraints.is_some_and(|r| r.forces(c.a, c.b)))
    };
    accepted.sort_by(|&x, &y| {
        forced(x).cmp(&forced(y)).then_with(|| {
            candidates[y]
                .score()
                .partial_cmp(&candidates[x].score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    accepted
}

/// `rel(c, x)`: the transform that maps `x`'s partner into `x`'s frame.
///
/// `c.T` maps `b` into `a`, so it is the answer when `x` is `a` and its inverse when `x` is `b`.
/// The inverse is `np.linalg.inv`'s — the factorisation of the whole 4×4, not the transpose of its
/// rotation block, which is 6.2e-11 t away from it at the worst on a pose that has climbed two ICP
/// ladders (PMC-19, `matching::verify::pose_inverse`).
pub(crate) fn rel(c: &Candidate, x: FragId) -> Matrix4<f64> {
    if x == c.a { c.transform } else { pose_inverse(&c.transform) }
}

/// What one pass of the loop reads and never writes: the collection, the candidate list, R §8's
/// `accepted` order and the parameters.
///
/// Grouped into a struct because [`Inputs::try_place`] needs all four beside the mutable
/// [`Grouping`], and five borrowed arguments in a row is where a transcription of a reference
/// starts to lose its shape.
#[derive(Clone, Copy)]
struct Inputs<'a> {
    exec: &'a dyn Executor,
    pieces: &'a [Piece<'a>],
    candidates: &'a [Candidate],
    accepted: &'a [usize],
    p: &'a Params,
    /// Roadmap item 4's rules, or `None` on a run without them — which is R §8 exactly as the
    /// reference wrote it.
    objects: Option<&'a ObjectParams>,
    constraints: Option<&'a Resolved>,
}

impl Inputs<'_> {
    /// R §8's `try_place`: the world pose `c` implies for `new`, or the reason it may not have one.
    ///
    /// Two checks, in the reference's order and each returning on its **first** failure:
    ///
    /// 1. **Penetration** against every fragment already in the group, in placement order, skipping
    ///    the fragment `c` attaches to — that one is R §6.4's own test, which the candidate has
    ///    already passed to be accepted at all.
    /// 2. **Consistency** with every other accepted join that touches `new` and whose other
    ///    endpoint is already in this group, walked in `accepted` order (strongest first). A
    ///    disagreement only rejects when the *other* join is the stronger of the two; when `c` is
    ///    stronger, the weaker join is left to be rejected on its own turn.
    fn try_place(
        &self,
        grouping: &Grouping,
        here: usize,
        placed: FragId,
        new: FragId,
    ) -> Result<Matrix4<f64>, Rejection> {
        let Self { exec, pieces, candidates, accepted, p, .. } = *self;
        let c = &candidates[accepted[here]];
        let anchor = grouping.pose(placed).expect("the anchor of a placement is placed");
        let t_new = anchor * rel(c, placed);
        let g = grouping.group_of(placed).expect("a placed fragment is in a group");

        for &other in grouping.members(g) {
            if other == placed {
                continue;
            }
            let inverse = pose_inverse(&grouping.pose(other).expect("a group member is placed"));
            let t_rel = inverse * t_new;
            let pen = penetration(exec, &pieces[other as usize], &pieces[new as usize], &t_rel, p);
            if pen > p.max_pen {
                return Err(Rejection::Penetrates { other, pen });
            }
        }

        let tg = grouping.thickness(pieces, g, Some(new));
        for (there, &j) in accepted.iter().enumerate() {
            if there == here {
                continue;
            }
            let c2 = &candidates[j];
            if new != c2.a && new != c2.b {
                continue;
            }
            let other = if c2.a == new { c2.b } else { c2.a };
            if grouping.group_of(other) != Some(g) {
                continue;
            }
            let t_alt = grouping.pose(other).expect("a group member is placed") * rel(c2, other);
            let d = pose_inverse(&t_alt) * t_new;
            let (angle, distance) = disagreement(&d, tg);
            if !agrees(angle, distance) && c2.score() > c.score() {
                return Err(Rejection::InconsistentWithStronger {
                    a: c2.a,
                    b: c2.b,
                    angle,
                    distance,
                });
            }
        }
        Ok(t_new)
    }

    /// Audit §D.2 (c): the two groups this join spans, and the world shift that brings one onto
    /// the other — or the reason they may not be merged.
    ///
    /// R §8 as the reference wrote it refuses such a join outright
    /// ([`Rejection::MergesGroups`]). Roadmap item 4 replaces that refusal with the same two tests
    /// R §8 already applies to a *placement*, applied to a whole group, plus the operator's word:
    ///
    /// 1. **`different_object`** naming any cross-group pair refuses the merge, before any test of
    ///    R §8's own, so the report says whose decision it was. `must_not_join` deliberately does
    ///    **not**: it says two sherds do not join, and two sherds of one pot that do not touch are
    ///    in one group all the same.
    /// 2. **Penetration across both groups**, every member of one against every member of the
    ///    other, at the poses the merge would give them — R §6.4's own fraction at R §6.4's own
    ///    depth. The joined pair itself is skipped: R §6.4 has already tested it, which is how the
    ///    candidate came to be accepted.
    /// 3. **Consistency with every cross-group join the gate admits**, at R §8's own tolerances
    ///    (10°, 0.5 `t` of the merged wall). Audit §D.2 words this as "every cross-group *accepted*
    ///    join"; what it reads here is the list R §8 is walking, because R §8's own consistency
    ///    test reads that list and a merge refused by a join the gate threw out would be refused
    ///    by evidence the assembly is not allowed to use.
    ///
    /// **Which group moves.** The larger one stays and the smaller is shifted onto it; on a tie
    /// the earlier-seeded group stays. Deterministic, and it keeps the frame of the group that has
    /// the most fragments in it — which is the frame `transforms.json` already carries for most of
    /// the collection.
    fn try_merge(
        &self,
        grouping: &Grouping,
        here: usize,
    ) -> Result<(usize, usize, Matrix4<f64>), Rejection> {
        let Self { exec, pieces, candidates, accepted, p, constraints, .. } = *self;
        let candidate = accepted[here];
        let c = &candidates[candidate];
        if c.tier != Tier::Confirmed {
            return Err(Rejection::MergeNotConfirmed);
        }
        let ga = grouping.group_of(c.a).expect("a placed fragment is in a group");
        let gb = grouping.group_of(c.b).expect("a placed fragment is in a group");
        let (na, nb) = (grouping.members(ga).len(), grouping.members(gb).len());
        let (into, from) = if nb > na || (nb == na && gb < ga) { (gb, ga) } else { (ga, gb) };

        // The shift that makes the join hold: whichever endpoint is in the group that stays keeps
        // its world pose, and the other group is moved so that `c` is satisfied at it.
        let (pose_a, pose_b) = (
            grouping.pose(c.a).expect("a placed fragment has a pose"),
            grouping.pose(c.b).expect("a placed fragment has a pose"),
        );
        let shift = if ga == into {
            pose_a * c.transform * pose_inverse(&pose_b)
        } else {
            pose_b * pose_inverse(&(pose_a * c.transform))
        };

        let joined = [c.a, c.b];
        for &x in grouping.members(into) {
            for &y in grouping.members(from) {
                if let Some(veto @ Veto::DifferentObject) = constraints.and_then(|r| r.veto(x, y)) {
                    return Err(Rejection::Constrained(veto));
                }
                if joined.contains(&x) && joined.contains(&y) {
                    continue;
                }
                let moved = shift * grouping.pose(y).expect("a group member is placed");
                let t_rel = pose_inverse(&grouping.pose(x).expect("placed")) * moved;
                let pen = penetration(exec, &pieces[x as usize], &pieces[y as usize], &t_rel, p);
                if pen > p.max_pen {
                    return Err(Rejection::Penetrates { other: x, pen });
                }
            }
        }

        let walls: Vec<f64> = grouping
            .members(into)
            .iter()
            .chain(grouping.members(from))
            .map(|&n| pieces[n as usize].thick)
            .collect();
        let tg = crate::mesh::geometry::median(&walls);
        for (there, &j) in accepted.iter().enumerate() {
            if there == here {
                continue;
            }
            let c2 = &candidates[j];
            let (g2a, g2b) = (grouping.group_of(c2.a), grouping.group_of(c2.b));
            // One endpoint in each of the two groups, either way round.
            let x = match (g2a, g2b) {
                (Some(x), Some(y)) if x == into && y == from => c2.a,
                (Some(x), Some(y)) if x == from && y == into => c2.b,
                _ => continue,
            };
            let y = if x == c2.a { c2.b } else { c2.a };
            let t_alt = grouping.pose(x).expect("placed") * rel(c2, x);
            let moved = shift * grouping.pose(y).expect("placed");
            let (angle, distance) = disagreement(&(pose_inverse(&t_alt) * moved), tg);
            if !agrees(angle, distance) {
                return Err(Rejection::MergeInconsistent { a: c2.a, b: c2.b, angle, distance });
            }
        }
        Ok((into, from, shift))
    }
}

/// R §8: the whole greedy assembly.
///
/// `pieces` is the collection, indexed by [`FragId`]; `candidates` is every candidate of every
/// pair, in the order the matcher produced them (R §4.1's pair order, R §5.7's ranking inside each
/// pair). Only the accepted ones are looked at, and only the best of each pair.
///
/// # The loop, and why its shape matters
///
/// ```text
/// remaining = accepted                       # strongest join first
/// while remaining:
///     for c in a snapshot of remaining:
///         both endpoints placed  -> take it as a loop-closing edge, or reject it; keep scanning
///         neither placed         -> skip it; keep scanning
///         one placed             -> try to place the other; on success RESTART the scan
///     nothing was placed         -> seed a new group from the first join with both ends free
/// ```
///
/// The restart is what makes it greedy rather than a single pass: placing a fragment changes what
/// the joins below it mean, so the scan goes back to the strongest one. The snapshot is the
/// reference's `for c in list(remaining)`, and it is visible behaviour — a join removed during a
/// pass does not shorten that pass.
///
/// Every fragment left unplaced becomes a singleton group, and the groups come back sorted by size
/// (stable). R §8.2's recentring is deliberately *not* applied here: R §9's refinement runs between
/// the two, and the pipeline recentres once, afterwards.
pub fn assemble(
    exec: &dyn Executor,
    pieces: &[Piece<'_>],
    candidates: &[Candidate],
    p: &Params,
) -> Assembly {
    assemble_with(exec, pieces, candidates, p, Gate::Accepted)
}

/// R §8 over the candidates one [`Gate`] admits.
///
/// [`Gate::Accepted`] is [`assemble`] — R §8 as the reference wrote it. [`Gate::Confirmed`] is
/// audit §D.1's rule: *"the assembly is built from confirmed joins only; probable joins are listed
/// and rendered, never placed"*. Nothing else in the pass changes, and that is deliberate — the
/// tier decides which joins R §8 may see, and R §8's own greedy rule, its ties, its penetration
/// and consistency tests and the sentences it writes for a refusal stay exactly as they are. A
/// join listed under "confirmed joins not used" was therefore refused by R §8 and not by the tier.
pub fn assemble_with(
    exec: &dyn Executor,
    pieces: &[Piece<'_>],
    candidates: &[Candidate],
    p: &Params,
    gate: Gate,
) -> Assembly {
    assemble_under(exec, pieces, candidates, p, gate, None, None)
}

/// [`assemble_with`] under an operator's `constraints.json` as well (audit §D.1, roadmap item 3).
///
/// The constraints reach R §8 in exactly two places and nowhere else. A pair `must_not_join` or
/// `different_object` names is **refused**, with [`Rejection::Constrained`] and before any test of
/// R §8's own, so it appears in the report as a join the operator refused rather than one the
/// geometry did; and a pair `must_join` names is offered to the loop **first**, whatever its score,
/// which is what makes the group grow from it. Everything between those two — the greedy rule, the
/// ties, the penetration and consistency tests, the sentences — is R §8's, untouched.
///
/// `must_not_join` has already removed its pairs before matching by the time this runs, so the
/// refusal here is the second half of a belt and braces: a candidate for such a pair can only
/// reach R §8 through a candidate list assembled by something other than the pipeline.
pub fn assemble_under(
    exec: &dyn Executor,
    pieces: &[Piece<'_>],
    candidates: &[Candidate],
    p: &Params,
    gate: Gate,
    constraints: Option<&Resolved>,
    objects: Option<&ObjectParams>,
) -> Assembly {
    let started = std::time::Instant::now();
    let accepted = best_per_pair(candidates, gate, constraints);
    let inputs = Inputs { exec, pieces, candidates, accepted: &accepted, p, objects, constraints };
    let mut grouping = Grouping::new(pieces.len());
    let mut used: Vec<usize> = Vec::new();
    let mut rejected: Vec<Rejected> = Vec::new();
    let mut merges = 0_usize;
    let mut remaining: Vec<usize> = (0..accepted.len()).collect();

    while !remaining.is_empty() {
        let mut progressed = false;
        for here in remaining.clone() {
            let candidate = accepted[here];
            let c = &candidates[candidate];
            if let Some(veto) = constraints.and_then(|r| r.veto(c.a, c.b)) {
                remaining.retain(|&k| k != here);
                rejected.push(Rejected { candidate, reason: Rejection::Constrained(veto) });
                continue;
            }
            let (a_in, b_in) = (grouping.placed(c.a), grouping.placed(c.b));
            if a_in && b_in {
                remaining.retain(|&k| k != here);
                let ga = grouping.group_of(c.a);
                if ga == grouping.group_of(c.b) {
                    // A loop-closing edge: it adds no fragment, so it is kept only if the poses
                    // already placed say the same thing it does.
                    let g = ga.expect("a placed fragment is in a group");
                    let d = pose_inverse(&(grouping.pose(c.a).expect("placed") * c.transform))
                        * grouping.pose(c.b).expect("placed");
                    let (angle, distance) = disagreement(&d, grouping.thickness(pieces, g, None));
                    if agrees(angle, distance) {
                        used.push(candidate);
                    } else {
                        rejected.push(Rejected {
                            candidate,
                            reason: Rejection::InconsistentWithAssembled { angle, distance },
                        });
                    }
                } else if gate != Gate::Confirmed || inputs.objects.is_none_or(|o| !o.merge) {
                    // R §8's own refusal, unchanged: a run without roadmap item 4, and a run
                    // whose candidates have no band at all (`--tiers off`), cannot merge, and
                    // saying "only a confirmed join may" of a run that has no confirmed joins to
                    // speak of would be a worse sentence than the one the reference wrote.
                    rejected.push(Rejected { candidate, reason: Rejection::MergesGroups });
                } else {
                    // Audit §D.2 (c): a confirmed join between two groups merges them, under the
                    // penetration test across both and consistency with every cross-group join
                    // the gate admits.
                    match inputs.try_merge(&grouping, here) {
                        Err(reason) => rejected.push(Rejected { candidate, reason }),
                        Ok((into, from, shift)) => {
                            grouping.merge(into, from, &shift);
                            used.push(candidate);
                            merges += 1;
                            // A merge changes what every join below it means, exactly as a
                            // placement does, so the greedy rule's restart applies to it too.
                            progressed = true;
                            break;
                        }
                    }
                }
                continue;
            }
            if !a_in && !b_in {
                continue;
            }
            let (placed, new) = if a_in { (c.a, c.b) } else { (c.b, c.a) };
            let outcome = inputs.try_place(&grouping, here, placed, new);
            remaining.retain(|&k| k != here);
            match outcome {
                Err(reason) => rejected.push(Rejected { candidate, reason }),
                Ok(t_new) => {
                    grouping.place(placed, new, t_new);
                    used.push(candidate);
                    progressed = true;
                    break;
                }
            }
        }
        if progressed || remaining.is_empty() {
            continue;
        }
        // Nothing in the list touches what is already placed: start a group from the strongest
        // join whose two fragments are both still free. The seed is placed unchecked — there is
        // nothing yet for it to contradict.
        let Some(&here) = remaining.iter().find(|&&k| {
            let c = &candidates[accepted[k]];
            !grouping.placed(c.a) && !grouping.placed(c.b)
        }) else {
            break;
        };
        remaining.retain(|&k| k != here);
        let candidate = accepted[here];
        let c = &candidates[candidate];
        grouping.seed(c.a, c.b, c.transform);
        used.push(candidate);
    }

    grouping.add_singletons();
    // A no-op unless a merge emptied one; a run without roadmap item 4 never has one to drop.
    grouping.drop_empty_groups();
    grouping.sort_by_size();
    let groups = grouping.groups().to_vec();
    tracing::info!(
        groups = groups.len(),
        sizes = ?groups.iter().map(Vec::len).collect::<Vec<usize>>(),
        used = used.len(),
        rejected = rejected.len(),
        accepted = accepted.len(),
        merges,
        seconds = started.elapsed().as_secs_f64(),
        "assembly"
    );
    let order = grouping.order().to_vec();
    Assembly { poses: grouping.into_poses(), groups, used, rejected, accepted, order, merges }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these poses are whole numbers the merge copies")]

    use nalgebra::Matrix4;

    use super::{Grouping, Inputs, Piece, Rejection, rel};
    use crate::assembly::constraints::{Constraints, Resolved, Veto, resolve};
    use crate::executor::CPU;
    use crate::matching::pair::Candidate;
    use crate::matching::verify::Scores;
    use crate::objects::ObjectParams;
    use crate::params::Params;
    use crate::tiers::Tier;
    use crate::types::FragId;

    /// Audit §D.2 (c)'s merge is tested here and not in `tests/assembly_graphs.rs` because R §8's
    /// own loop **cannot reach it**: a group grows until no remaining join touches it, so its
    /// members are a connected component of the joins R §8 kept and no join ever spans two groups.
    /// `r_8s_groups_are_components_so_no_join_ever_spans_two_of_them` measures that over four
    /// thousand random graphs, and task O1 measured `Rejection::MergesGroups` firing zero times
    /// over the eight development sets at seeds 0–4. The decision the branch would make is still
    /// worth stating exactly, so it is stated against a [`Grouping`] built by hand.
    fn translation(tau: [f64; 3]) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        for (i, t) in tau.into_iter().enumerate() {
            m[(i, 3)] = t;
        }
        m
    }

    fn join(a: FragId, b: FragId, tau: [f64; 3], tier: Tier) -> Candidate {
        Candidate {
            a,
            b,
            transform: translation(tau),
            scores: Scores { seam: 1.0, tight: 1.0, ..Scores::default() },
            accepted: true,
            tier,
        }
    }

    fn bare(n: usize) -> Vec<Piece<'static>> {
        (0..n)
            .map(|_| Piece { thick: 1.0, res: 0.01, watertight: true, mesh: None, s_pen: &[] })
            .collect()
    }

    /// Two groups — 0-1 and 2-3, each ten units along `x` — and a spanning join 1-2.
    fn two_groups(candidates: &[Candidate]) -> Grouping {
        let mut grouping = Grouping::new(4);
        grouping.seed(candidates[0].a, candidates[0].b, candidates[0].transform);
        grouping.seed(candidates[1].a, candidates[1].b, candidates[1].transform);
        grouping
    }

    fn inputs<'a>(
        pieces: &'a [Piece<'a>],
        candidates: &'a [Candidate],
        accepted: &'a [usize],
        params: &'a Params,
        rules: Option<&'a ObjectParams>,
        constraints: Option<&'a Resolved>,
    ) -> Inputs<'a> {
        Inputs { exec: &CPU, pieces, candidates, accepted, p: params, objects: rules, constraints }
    }

    /// A confirmed join between two groups gives the shift that brings one onto the other, the
    /// larger group keeps its frame, and the join holds at the merged poses.
    #[test]
    fn a_confirmed_join_merges_the_smaller_group_onto_the_larger() {
        let pieces = bare(4);
        let params = Params::default();
        let rules = ObjectParams::default();
        let candidates = vec![
            join(0, 1, [10.0, 0.0, 0.0], Tier::Confirmed),
            join(2, 3, [10.0, 0.0, 0.0], Tier::Confirmed),
            join(1, 2, [10.0, 0.0, 0.0], Tier::Confirmed),
        ];
        let accepted = [0, 1, 2];
        let mut grouping = two_groups(&candidates);
        let inputs = inputs(&pieces, &candidates, &accepted, &params, Some(&rules), None);

        let (into, from, shift) = inputs.try_merge(&grouping, 2).expect("the merge is allowed");
        // Equal sizes, so the earlier-seeded group stays.
        assert_eq!((into, from), (0, 1));
        grouping.merge(into, from, &shift);
        assert_eq!(grouping.members(0), [0, 1, 2, 3]);
        assert!(grouping.members(1).is_empty());
        assert_eq!(grouping.pose(0).expect("placed"), Matrix4::identity(), "the frame that stays");
        assert_eq!(grouping.pose(2).expect("placed")[(0, 3)], 20.0, "1-2 is another ten along x");
        assert_eq!(grouping.pose(3).expect("placed")[(0, 3)], 30.0);
        // The join it merged through holds exactly: B's world pose is A's through `c.transform`.
        let c = &candidates[2];
        let implied = grouping.pose(c.a).expect("placed") * rel(c, c.a);
        assert_eq!(implied, grouping.pose(c.b).expect("placed"));
    }

    /// The three refusals, each with its own reason: the join is not confirmed, another
    /// cross-group join contradicts the merged poses, and `different_object` names a pair the
    /// merge would put in one object.
    #[test]
    fn a_merge_is_refused_by_the_band_by_a_contradiction_and_by_the_operator() {
        let pieces = bare(4);
        let params = Params::default();
        let rules = ObjectParams::default();
        let names: Vec<String> = (0..4).map(|i| format!("frag_{i}")).collect();

        // 1. The spanning join is only probable.
        let mut candidates = vec![
            join(0, 1, [10.0, 0.0, 0.0], Tier::Confirmed),
            join(2, 3, [10.0, 0.0, 0.0], Tier::Confirmed),
            join(1, 2, [10.0, 0.0, 0.0], Tier::Probable),
        ];
        let accepted = [0, 1, 2];
        let grouping = two_groups(&candidates);
        let refused = inputs(&pieces, &candidates, &accepted, &params, Some(&rules), None)
            .try_merge(&grouping, 2)
            .expect_err("a probable join may not merge");
        assert_eq!(refused, Rejection::MergeNotConfirmed);
        assert_eq!(
            refused.message(&names),
            "would merge two groups (only a confirmed join may)",
            "and R §8's own sentence is left to the run that cannot merge at all"
        );

        // 2. A second cross-group join that puts fragment 3 five hundred walls away.
        candidates[2].tier = Tier::Confirmed;
        candidates.push(join(0, 3, [10.0, 500.0, 0.0], Tier::Confirmed));
        let accepted = [0, 1, 2, 3];
        let refused = inputs(&pieces, &candidates, &accepted, &params, Some(&rules), None)
            .try_merge(&grouping, 2)
            .expect_err("the merge contradicts the other cross-group join");
        let Rejection::MergeInconsistent { a, b, distance, .. } = refused else {
            panic!("expected a contradiction, got {refused:?}");
        };
        assert_eq!((a, b), (0, 3), "the sentence names the join that refused it");
        assert!(distance > 100.0, "{distance}");

        // 3. `different_object`, before any test of R §8's own.
        candidates.pop();
        let accepted = [0, 1, 2];
        let file: Constraints =
            serde_json::from_str(r#"{"version": 1, "different_object": [["frag_0", "frag_3"]]}"#)
                .expect("a constraints file");
        let plan = resolve(&file, &names).expect("the names resolve");
        let refused = inputs(&pieces, &candidates, &accepted, &params, Some(&rules), Some(&plan))
            .try_merge(&grouping, 2)
            .expect_err("the operator refused it");
        assert_eq!(refused, Rejection::Constrained(Veto::DifferentObject));

        // `must_not_join` deliberately does **not** refuse a merge: it says two sherds do not
        // join, and two sherds of one pot that do not touch belong to one object all the same.
        let file: Constraints =
            serde_json::from_str(r#"{"version": 1, "must_not_join": [["frag_0", "frag_3"]]}"#)
                .expect("a constraints file");
        let plan = resolve(&file, &names).expect("the names resolve");
        assert!(
            inputs(&pieces, &candidates, &accepted, &params, Some(&rules), Some(&plan))
                .try_merge(&grouping, 2)
                .is_ok(),
            "`must_not_join` is about a seam, not about a vessel"
        );
    }
}
