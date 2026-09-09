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
use super::groups::{Grouping, Piece};

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
fn best_per_pair(candidates: &[Candidate], gate: Gate) -> Vec<usize> {
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
    accepted.sort_by(|&x, &y| {
        candidates[y]
            .score()
            .partial_cmp(&candidates[x].score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    accepted
}

/// `rel(c, x)`: the transform that maps `x`'s partner into `x`'s frame.
///
/// `c.T` maps `b` into `a`, so it is the answer when `x` is `a` and its inverse when `x` is `b`.
/// The inverse is `np.linalg.inv`'s — the factorisation of the whole 4×4, not the transpose of its
/// rotation block, which is 6.2e-11 t away from it at the worst on a pose that has climbed two ICP
/// ladders (PMC-19, `matching::verify::pose_inverse`).
fn rel(c: &Candidate, x: FragId) -> Matrix4<f64> {
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
        let Self { exec, pieces, candidates, accepted, p } = *self;
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
    let started = std::time::Instant::now();
    let accepted = best_per_pair(candidates, gate);
    let inputs = Inputs { exec, pieces, candidates, accepted: &accepted, p };
    let mut grouping = Grouping::new(pieces.len());
    let mut used: Vec<usize> = Vec::new();
    let mut rejected: Vec<Rejected> = Vec::new();
    let mut remaining: Vec<usize> = (0..accepted.len()).collect();

    while !remaining.is_empty() {
        let mut progressed = false;
        for here in remaining.clone() {
            let candidate = accepted[here];
            let c = &candidates[candidate];
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
                } else {
                    rejected.push(Rejected { candidate, reason: Rejection::MergesGroups });
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
    grouping.sort_by_size();
    let groups = grouping.groups().to_vec();
    tracing::info!(
        groups = groups.len(),
        sizes = ?groups.iter().map(Vec::len).collect::<Vec<usize>>(),
        used = used.len(),
        rejected = rejected.len(),
        accepted = accepted.len(),
        seconds = started.elapsed().as_secs_f64(),
        "assembly"
    );
    let order = grouping.order().to_vec();
    Assembly { poses: grouping.into_poses(), groups, used, rejected, accepted, order }
}
