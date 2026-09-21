//! `Review` (A §2.2, A §8): a long-lived session over a finished run.
//!
//! Matching is 94 % of a run and R §8's assembly is a tenth of a second of it (A §3), so a
//! decision about one join costs a reassembly — *provided* the collection and the saved match are
//! already in memory. Loading them is what the other two jobs would have to do again for every
//! click, and it is the whole reason this job outlives one question: the fragments and
//! `match.state` are read once, and then every [`Request`] is answered from them.
//!
//! The session writes nothing. `decisions.json` and `assembly.json` are the host's (A §2.1), and
//! it is the host that files the [`Event::Assembly`] that comes back.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::Receiver;

use sherd_core::assembly::{Piece, recenter};
use sherd_core::collection;
use sherd_core::executor::Engine;
use sherd_core::fragment::Fragment;
use sherd_core::memory::Budget;
use sherd_core::session::{self, MatchState, Reassembled};
use sherd_core::{FragId, pipeline};

use super::run::{MATCH_STATE_FILE, assembly_dto, pose_matrix};
use super::{Context, Failure};
use crate::decisions::{Decision, DecisionsFile, Verdict, to_constraints};
use crate::host::ASSEMBLY_FILE;
use crate::protocol::{
    AssemblyDto, Event, FailKind, PairDetailDto, Request, ReviewJob, UnplacedDto,
};
use crate::{atomic, review};

/// What the window is told when a decision was accepted but R §8 did not build with the pair and
/// nothing — not the assembly's own refusals, not the constraints report — says why.
///
/// It should not happen: `must_join` is offered to R §8 first and a refusal is recorded. It is
/// here because an empty cell in the reviewer's list would be worse than a plain sentence.
const UNEXPLAINED: &str = "не поставлен";

/// Serves one review session until [`Request::Close`] or the end of the input (A §8).
///
/// The engine is the CPU one whatever the machine has: a draft the reviewer is still moving must
/// not depend on a device being free, and `write_reviewed` records the *match's* backend (A §3.2)
/// rather than this process's, so nothing downstream is misreported by the choice.
pub(crate) fn review(
    job: &ReviewJob,
    requests: Receiver<Request>,
    context: &Context,
) -> Result<Event, Failure> {
    let mut session = Session::open(job, context)?;
    // The iterator blocks for the next line and ends when the host's side of the pipe is gone,
    // which is A §2.1's other way of closing a session.
    for request in requests {
        let answered = match request {
            Request::Close => break,
            // The input reader raises D §5's flag itself; nothing of a cancel reaches this loop.
            Request::Cancel => Ok(()),
            Request::Reassemble { decisions } => session.on_reassemble(context, &decisions),
            Request::PairDetail { a, b, pose } => session.on_pair_detail(context, &a, &b, &pose),
            Request::Refine { decisions } => session.on_refine(context, &decisions),
        };
        if let Err(failure) = answered {
            // A cancel ends the job, as it ends any job (A §2.2); everything else is one
            // request's problem and the session stays open to answer the next.
            if failure.kind == FailKind::Cancelled {
                return Err(failure);
            }
            context.emitter.emit(&Event::RequestFailed { message: failure.message });
        }
    }
    Ok(Event::Done { counts: None, engine: None, params: None })
}

/// Everything the session holds between two questions.
struct Session {
    /// The collection, preprocessed through `cache/` exactly as the run preprocessed it.
    fragments: Vec<Fragment>,
    /// A §3.1's saved match: the candidate list R §8 is run over again.
    state: MatchState,
    /// The assembly the answers are merged against (A §8.4) — the run's own at first, and then
    /// whatever the last `Refine` produced.
    baseline: AssemblyDto,
    /// What R §9 may hold when it reads the source scans.
    budget: Budget,
}

impl Session {
    /// Loads the collection and the match, and says [`Event::Ready`].
    ///
    /// A `match.state` that cannot be read is [`FailKind::Protocol`] with the engine's own
    /// sentence, and so is a collection that is no longer the one the run matched: A §10 keeps
    /// such a run viewable and turns review off, which is a failure of *this job* and not of the
    /// run's folder.
    fn open(job: &ReviewJob, context: &Context) -> Result<Self, Failure> {
        if let Err(already) = pipeline::set_threads(job.workers) {
            tracing::debug!("the thread pool is built already: {already}");
        }
        let dir = job.workspace.join("runs").join(&job.run_id);
        let entries = collection::discover_excluding(&job.input, &job.excluded)?;
        let budget = job.memory_gb.map_or_else(Budget::default_for_machine, Budget::gigabytes);
        let fragments = session::load_fragments(
            &entries,
            job.target_faces,
            Some(&job.workspace.join("cache")),
            budget,
            job.seed,
            &context.watch,
        )?;
        let state = MatchState::load(&dir.join(MATCH_STATE_FILE))
            .map_err(|e| Failure::new(FailKind::Protocol, e.to_string()))?;
        if !fragments.iter().map(|f| f.name.as_str()).eq(state.names.iter().map(String::as_str)) {
            return Err(Failure::new(
                FailKind::Protocol,
                format!(
                    "the {} fragments in the input are not the {} this run matched — the input \
                     changed; run the collection again",
                    fragments.len(),
                    state.names.len()
                ),
            ));
        }
        // A run whose assembly was never filed, or filed and then damaged, is still reviewable:
        // it only means nothing counts as refined yet (A §8.4).
        let baseline =
            atomic::read_json::<AssemblyDto>(&dir.join(ASSEMBLY_FILE)).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "no baseline assembly: no group counts as refined");
                AssemblyDto::default()
            });
        context
            .emitter
            .emit(&Event::Ready { fragments: fragments.len(), candidates: state.candidates.len() });
        Ok(Self { fragments, state, baseline, budget })
    }

    /// R §8 under `decisions`, and what the window draws of it — before the refined groups of the
    /// baseline are merged back in, which is the caller's next step.
    ///
    /// The [`Reassembled`] comes back with it because `Refine` needs the un-recentred poses and
    /// the joins R §8 took, and reassembling twice for one request would be a second run of the
    /// only expensive thing here.
    fn reassemble(
        &self,
        context: &Context,
        decisions: &DecisionsFile,
    ) -> Result<(Reassembled, AssemblyDto), Failure> {
        let (constraints, dropped) =
            to_constraints(decisions, &self.state.names).map_err(|e| super::app_failure(&e))?;
        if !dropped.is_empty() {
            context.emitter.emit(&Event::Dropped { decisions: dropped.clone() });
        }
        let done = session::reassemble(
            Engine::REFERENCE,
            &self.fragments,
            &self.state,
            constraints.as_ref(),
        )?;
        let mut dto =
            assembly_dto(&self.state.names, &done.assembly.groups, &done.poses, &done.used, false);
        dto.unplaced = self.unplaced(&done, &decisions.decisions, &dropped);
        Ok((done, dto))
    }

    /// Answers a `Reassemble` (A §8.2).
    fn on_reassemble(&self, context: &Context, decisions: &DecisionsFile) -> Result<(), Failure> {
        let (_, fresh) = self.reassemble(context, decisions)?;
        let merged = review::merge_refined(&self.baseline, fresh);
        context.emitter.emit(&Event::Assembly(merged));
        Ok(())
    }

    /// Answers a `Refine` (A §8.4): R §9 over the groups this decision set left unrefined, and
    /// nothing else — the rest keep the poses they were refined to.
    ///
    /// R §9 is given the reassembly's own **un-recentred** poses, with every already-refined
    /// group put back at its refined ones. Those are recentred, and are put through R §8.2's
    /// recentring again at the end; that is harmless, because a group whose centroid is already
    /// at the origin is translated by zero.
    fn on_refine(&mut self, context: &Context, decisions: &DecisionsFile) -> Result<(), Failure> {
        let (done, fresh) = self.reassemble(context, decisions)?;
        let merged = review::merge_refined(&self.baseline, fresh);
        let index: BTreeMap<&str, usize> =
            self.state.names.iter().enumerate().map(|(n, name)| (name.as_str(), n)).collect();

        let mut poses = done.assembly.poses.clone();
        for group in merged.groups.iter().filter(|g| g.refined) {
            for member in &group.members {
                if let (Some(&id), Some(pose)) =
                    (index.get(member.as_str()), merged.poses.get(member))
                    && let Some(slot) = poses.get_mut(id)
                {
                    *slot = pose_matrix(pose);
                }
            }
        }
        let unrefined: Vec<Vec<FragId>> = merged
            .groups
            .iter()
            .zip(&done.assembly.groups)
            .filter(|(group, members)| !group.refined && members.len() > 1)
            .map(|(_, members)| members.clone())
            .collect();
        let refined = session::refine_poses(
            Engine::REFERENCE,
            &self.fragments,
            &unrefined,
            &poses,
            &done.used,
            &self.state.params,
            self.budget,
            &context.watch,
        )?;

        let samples: Vec<Vec<[f64; 3]>> =
            self.fragments.iter().map(|f| f.samples.surface_f64()).collect();
        // `mesh: None`: R §8.2's recentring reads `s_pen` and nothing else, and a BVH per
        // fragment is the one thing in a `Piece` that costs anything to build.
        let pieces: Vec<Piece<'_>> = self
            .fragments
            .iter()
            .zip(&samples)
            .map(|(f, s)| Piece {
                thick: f.thick,
                res: f.res(),
                watertight: f.watertight,
                mesh: None,
                s_pen: s,
            })
            .collect();
        let recentred = recenter(&refined, &pieces, &done.assembly.groups);

        let mut dto =
            assembly_dto(&self.state.names, &done.assembly.groups, &recentred, &done.used, true);
        dto.unplaced = merged.unplaced;
        self.baseline.clone_from(&dto);
        context.emitter.emit(&Event::Assembly(dto));
        Ok(())
    }

    /// Answers a `PairDetail` (A §8.3): the seam of one placement, as numbers.
    fn on_pair_detail(
        &self,
        context: &Context,
        a: &str,
        b: &str,
        pose: &[[f64; 4]; 4],
    ) -> Result<(), Failure> {
        let find = |name: &str| {
            self.fragments.iter().find(|f| f.name == name).ok_or_else(|| {
                Failure::new(FailKind::Input, format!("`{name}` is not in this collection"))
            })
        };
        let view = sherd_core::review::seam_view(
            Engine::REFERENCE,
            find(a)?,
            find(b)?,
            &pose_matrix(pose),
            &self.state.params,
        );
        context.emitter.emit(&Event::PairDetail(PairDetailDto {
            a: a.to_owned(),
            b: b.to_owned(),
            contact: view.contact.iter().map(drawn).collect(),
            contact_class: view.contact_class,
            seam: view.seam.iter().map(drawn).collect(),
            tight: view.tight,
            gap: view.gap,
        }));
        Ok(())
    }

    /// Every accepted decision R §8 did not build with, and why (A §8.4).
    ///
    /// The reason is the engine's own, in the order it is worth reading: the assembly's refusal
    /// of that very join first (it penetrates, it disagrees with a stronger one), then what the
    /// constraints report says became of the line, and only then [`UNEXPLAINED`]. A decision that
    /// was dropped for naming a fragment that is gone is not listed here — [`Event::Dropped`] has
    /// already said so, and saying it twice would read as two different problems.
    fn unplaced(
        &self,
        done: &Reassembled,
        decisions: &[Decision],
        dropped: &[Decision],
    ) -> Vec<UnplacedDto> {
        let name = |id: FragId| self.state.names[id as usize].as_str();
        let joined: BTreeSet<(&str, &str)> =
            done.used.iter().map(|&(a, b)| pair_key(name(a), name(b))).collect();
        let gone: BTreeSet<(&str, &str)> = dropped.iter().map(|d| pair_key(&d.a, &d.b)).collect();
        decisions
            .iter()
            .filter(|d| d.verdict == Verdict::Accept)
            .filter(|d| {
                let pair = pair_key(&d.a, &d.b);
                !joined.contains(&pair) && !gone.contains(&pair)
            })
            .map(|d| UnplacedDto {
                a: d.a.clone(),
                b: d.b.clone(),
                reason: self.refusal(done, &d.a, &d.b).unwrap_or_else(|| UNEXPLAINED.to_owned()),
            })
            .collect()
    }

    /// The engine's sentence about why this pair is not in the assembly, if it has one.
    fn refusal(&self, done: &Reassembled, a: &str, b: &str) -> Option<String> {
        let wanted = pair_key(a, b);
        let name = |id: FragId| self.state.names[id as usize].as_str();
        let refused = done.assembly.rejected.iter().find(|r| {
            done.candidates
                .get(r.candidate)
                .is_some_and(|c| pair_key(name(c.a), name(c.b)) == wanted)
        });
        if let Some(refused) = refused {
            return Some(refused.reason.message(&self.state.names));
        }
        done.constraints
            .as_ref()?
            .entries
            .iter()
            .find(|e| pair_key(&e.a, &e.b) == wanted)
            .map(|e| e.outcome.clone())
    }
}

/// A §8.1's key: the pair, the same whichever way round the two names were written.
fn pair_key<'n>(a: &'n str, b: &'n str) -> (&'n str, &'n str) {
    if a <= b { (a, b) } else { (b, a) }
}

/// One point as the window draws it: `f32`, because a `THREE.BufferAttribute` is a
/// `Float32Array` and five thousand contact points are half the JSON this way.
#[allow(clippy::cast_possible_truncation, reason = "a drawing coordinate, deliberately f32")]
fn drawn(point: &[f64; 3]) -> [f32; 3] {
    [point[0] as f32, point[1] as f32, point[2] as f32]
}
