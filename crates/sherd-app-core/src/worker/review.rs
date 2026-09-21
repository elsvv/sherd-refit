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
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use nalgebra::Matrix4;
use sherd_core::assembly::{Piece, recenter};
use sherd_core::collection;
use sherd_core::executor::Engine;
use sherd_core::fragment::Fragment;
use sherd_core::memory::Budget;
use sherd_core::pipeline::RunOptions;
use sherd_core::session::{self, MatchState, Reassembled};
use sherd_core::{FragId, pipeline};

use super::run::{MATCH_STATE_FILE, assembly_dto, pose_matrix};
use super::{Context, Failure, io_failure};
use crate::decisions::{Decision, DecisionsFile, Verdict, to_constraints};
use crate::host::ASSEMBLY_FILE;
use crate::protocol::{
    AssemblyDto, Event, ExportWhat, FailKind, PairDetailDto, Request, ReviewJob, UnplacedDto,
};
use crate::{atomic, review};

/// What the window is told when a decision was accepted but R §8 did not build with the pair and
/// nothing — not the assembly's own refusals, not the constraints report — says why.
///
/// It should not happen: `must_join` is offered to R §8 first and a refusal is recorded. It is
/// here because an empty cell in the reviewer's list would be worse than a plain sentence.
const UNEXPLAINED: &str = "не поставлен";

/// What an export needs at its destination before a single mesh, in bytes (A §9.1).
///
/// A `Tables` export writes no mesh and no picture, so its size is `report.json`'s — tens of
/// megabytes on a collection of 54 000 candidates, and a few hundred kilobytes on most; and a
/// `Folder` export writes all of that as well, which is why this is also its floor. A flat 64 MB
/// is not a measurement; it is the smallest number that is certainly enough, which is all the
/// check is for: A §10 asks that an export refuse a full disk before it starts writing, not that
/// it predict its own output.
const TABLES_ESTIMATE: u64 = 64 << 20;

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
            Request::Export { decisions, what, dest } => {
                session.on_export(context, &decisions, what, &dest)
            }
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
    /// The input folder, as the job named it. An [`Request::Export`] needs it twice: R §11's
    /// writers read the **original scans** from it, and `README.txt` is titled after it.
    input: PathBuf,
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
        Ok(Self { input: job.input.clone(), fragments, state, baseline, budget })
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

    /// What [`Session::on_refine`] and [`Session::on_export`] both start from: R §8 under
    /// `decisions`, with every group the baseline had already refined put back at the poses it
    /// was refined to (A §8.4).
    fn merged(
        &self,
        context: &Context,
        decisions: &DecisionsFile,
    ) -> Result<(Reassembled, AssemblyDto), Failure> {
        let (done, fresh) = self.reassemble(context, decisions)?;
        Ok((done, review::merge_refined(&self.baseline, fresh)))
    }

    /// Answers a `Refine` (A §8.4).
    fn on_refine(&mut self, context: &Context, decisions: &DecisionsFile) -> Result<(), Failure> {
        let (done, merged) = self.merged(context, decisions)?;
        self.refine(context, &done, merged)?;
        Ok(())
    }

    /// R §9 over the groups this decision set left unrefined, and nothing else — the rest keep
    /// the poses they were refined to. The world poses it settles on become the new baseline and
    /// are said as [`Event::Assembly`]; they are returned because an export writes exactly them.
    ///
    /// R §9 is given the reassembly's own **un-recentred** poses, with every already-refined
    /// group put back at its refined ones. R §8.2's recentring at the end is then applied to the
    /// groups this reassembly brought and to no others: an already-refined group is already
    /// centred, and centring it a second time moves it by the rounding error of a centroid that
    /// is only nearly zero — which A §8.4 forbids, because such a group must come back bit for
    /// bit as it was refined.
    fn refine(
        &mut self,
        context: &Context,
        done: &Reassembled,
        merged: AssemblyDto,
    ) -> Result<Vec<Matrix4<f64>>, Failure> {
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
        // What this reassembly brought, singletons included: the groups whose poses are the fresh
        // ones and therefore the only groups either R §9 or R §8.2 may move here.
        let fresh_groups: Vec<Vec<FragId>> = merged
            .groups
            .iter()
            .zip(&done.assembly.groups)
            .filter(|(group, _)| !group.refined)
            .map(|(_, members)| members.clone())
            .collect();
        // R §9 walks joins, so a group of one has nothing for it to do.
        let unrefined: Vec<Vec<FragId>> =
            fresh_groups.iter().filter(|members| members.len() > 1).cloned().collect();
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
        let recentred = recenter(&refined, &pieces, &fresh_groups);

        let mut dto =
            assembly_dto(&self.state.names, &done.assembly.groups, &recentred, &done.used, true);
        dto.unplaced = merged.unplaced;
        self.baseline.clone_from(&dto);
        context.emitter.emit(&Event::Assembly(dto));
        Ok(recentred)
    }

    /// Answers an `Export` (A §9.1): R §11's writers and `export/` over the reviewed assembly,
    /// into a folder the reviewer chose.
    ///
    /// The three refusals come **before** R §9 runs, although what they guard is the writing.
    /// Refinement is tens of seconds and changes neither the grouping the estimate is made from
    /// nor the folder it would be written to, so a destination that is not empty, an input that
    /// has gone away or a disk with no room is said at once instead of half a minute later —
    /// and, because a refusal must leave the review as it found it, refining first would also
    /// move the window's assembly for an export that never happened.
    fn on_export(
        &mut self,
        context: &Context,
        decisions: &DecisionsFile,
        what: ExportWhat,
        dest: &Path,
    ) -> Result<(), Failure> {
        let (done, merged) = self.merged(context, decisions)?;
        empty_dest(dest)?;
        let needs = self.estimate(what, &done)?;
        let room = room_at(dest)?;
        if room < needs {
            return Err(Failure::new(
                FailKind::Disk,
                format!(
                    "{} has {} MB free and this export needs about {} MB",
                    dest.display(),
                    mb(room),
                    mb(needs)
                ),
            ));
        }
        let poses = self.refine(context, &done, merged)?;
        let options = self.options(what, context);
        let written = session::write_reviewed(
            dest,
            &self.input,
            &self.fragments,
            &self.state,
            &done,
            &poses,
            &options,
        )?;
        // What is on the disk and not what the writers believe they wrote: a file the window
        // reports the size of is one it could go and open.
        let bytes = written
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .map(|meta| meta.len())
            .fold(0_u64, u64::saturating_add);
        context.emitter.emit(&Event::Exported {
            dest: dest.to_owned(),
            files: written.iter().map(|path| relative(dest, path)).collect(),
            bytes,
        });
        Ok(())
    }

    /// A §9.1's two kinds as the engine's output switches.
    ///
    /// `params` are the **match's** and not this process's defaults: `report.md` prints the
    /// thresholds the candidates were judged by, and those were settled by the run that matched
    /// them. The backend label is [`session::write_reviewed`]'s own business (A §3.2: it records
    /// the match's), and `watch` is the worker's, so that R §11's `output` stage reaches the
    /// window exactly as a run's stages do (A §3.4).
    fn options(&self, what: ExportWhat, context: &Context) -> RunOptions {
        let (write_meshes, placed_all, merged_meshes, preview) = match what {
            ExportWhat::Folder { placed_all, merged_meshes, previews } => {
                (true, placed_all, merged_meshes, previews)
            }
            ExportWhat::Tables => (false, false, false, false),
        };
        RunOptions {
            params: self.state.params.clone(),
            preview,
            write_meshes,
            placed_all,
            merged_meshes,
            // The engine's switches do not separate the scene from the placed meshes (A §9.1),
            // so «Папка результата» is the whole of what a run writes and `Tables` is none of it.
            viewer: write_meshes,
            memory: self.budget,
            watch: context.watch.clone(),
            ..RunOptions::default()
        }
    }

    /// What this export is expected to need at its destination, in bytes — and, on the way, the
    /// check that the scans it is going to read are still where the run found them (A §9.1).
    ///
    /// A `Folder` export reads the **original** files twice over: `placed/*.ply` re-places every
    /// fragment of an assembled group at full resolution, and `scene.glb` decimates the whole
    /// collection. So every source must be readable, whatever gets placed, and the estimate is
    /// the placed ones' own bytes and a half — the tables, the report and the scene together are
    /// a fraction of one full-resolution mesh, and the PLY writer's output is the input's order
    /// of magnitude. Never less than [`TABLES_ESTIMATE`], though: a `Folder` export writes
    /// everything a `Tables` one writes and then some, and a reassembly that placed nothing at
    /// all would otherwise ask for no room and still write a report and a `scene.glb`.
    fn estimate(&self, what: ExportWhat, done: &Reassembled) -> Result<u64, Failure> {
        let ExportWhat::Folder { placed_all, .. } = what else {
            return Ok(TABLES_ESTIMATE);
        };
        let mut placed = vec![placed_all; self.fragments.len()];
        for members in done.assembly.groups.iter().filter(|group| group.len() > 1) {
            for &id in members {
                if let Some(slot) = placed.get_mut(id as usize) {
                    *slot = true;
                }
            }
        }
        let mut sources = 0_u64;
        for (fragment, &placed) in self.fragments.iter().zip(&placed) {
            let path = &fragment.source.path;
            let size = std::fs::metadata(path)
                .map_err(|e| {
                    Failure::new(
                        FailKind::Input,
                        format!(
                            "{}: {e} — an export writes the original scans and this one is gone",
                            path.display()
                        ),
                    )
                })?
                .len();
            if placed {
                sources = sources.saturating_add(size);
            }
        }
        Ok(sources.saturating_add(sources / 2).max(TABLES_ESTIMATE))
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

/// A §9.1: an export writes into a folder that is empty, or one that is not there yet.
///
/// Nothing is overwritten and nothing is added to somebody else's folder. `write_reviewed` writes
/// forty files under names as ordinary as `report.md` and `README.txt`, and a reviewer who picked
/// «Документы» by mistake would not get the originals back.
fn empty_dest(dest: &Path) -> Result<(), Failure> {
    if dest.is_file() {
        return Err(Failure::new(
            FailKind::Disk,
            format!("{} is a file; an export needs a folder", dest.display()),
        ));
    }
    let mut entries = match std::fs::read_dir(dest) {
        Ok(entries) => entries,
        // Not there at all is exactly right: the writers create it.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_failure(dest, &e)),
    };
    if entries.next().is_some() {
        return Err(Failure::new(
            FailKind::Disk,
            format!("{} is not empty; an export needs a new or empty folder", dest.display()),
        ));
    }
    Ok(())
}

/// Free space on the filesystem `dest` will be written to (A §10's `disk`: space is checked
/// before an export, not discovered halfway through one).
///
/// Measured at the nearest folder of the path that exists, because the destination itself
/// usually does not yet and `statvfs` has nothing to answer about a path that is not there.
fn room_at(dest: &Path) -> Result<u64, Failure> {
    let existing = dest.ancestors().find(|path| path.exists()).ok_or_else(|| {
        Failure::new(FailKind::Disk, format!("{}: no part of this path exists", dest.display()))
    })?;
    fs4::available_space(existing).map_err(|e| io_failure(existing, &e))
}

/// Bytes as a message prints them: whole megabytes, rounded up, so that «0 MB» is said only of a
/// disk that has nothing left at all.
fn mb(bytes: u64) -> u64 {
    bytes.div_ceil(1 << 20)
}

/// One written file as [`Event::Exported`] lists it: relative to `dest`, with `/` between the
/// parts whatever the platform wrote.
///
/// The window counts these and prints them; it opens `dest` and never a file, so a separator that
/// means «folder» to a reader everywhere is worth more here than the platform's own.
fn relative(dest: &Path, path: &Path) -> String {
    // Every file `write_reviewed` reports is under `out_dir`; one that is not is listed whole,
    // because a root or a drive letter rebuilt out of components is a path to nowhere.
    let Ok(inside) = path.strip_prefix(dest) else {
        return path.display().to_string();
    };
    inside.components().map(|part| part.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A §9.1's destination rule, which is the one refusal a reviewer meets by accident: a folder
    /// that is not there yet is what an export wants, and anything already in one is somebody's.
    #[test]
    fn an_export_takes_a_folder_that_is_new_or_empty_and_nothing_else() {
        // A lib test has no `CARGO_TARGET_TMPDIR`; the name is this test's own, so a second run
        // starts from the same clean folder whatever the first one left behind.
        let root = std::env::temp_dir().join("sherd-export-dest");
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();

        let missing = root.join("not-yet");
        assert!(empty_dest(&missing).is_ok(), "the writers create it themselves");
        let empty = root.join("empty");
        std::fs::create_dir(&empty).unwrap();
        assert!(empty_dest(&empty).is_ok());

        std::fs::write(empty.join("report.md"), "#").unwrap();
        let refused = empty_dest(&empty).expect_err("a folder with something in it");
        assert_eq!(refused.kind, FailKind::Disk);
        assert!(refused.message.contains("not empty"), "{}", refused.message);

        let file = root.join("report.md");
        std::fs::write(&file, "#").unwrap();
        let refused = empty_dest(&file).expect_err("a file is not a folder");
        assert!(refused.message.contains("is a file"), "{}", refused.message);

        // And there is room on the disk the tests are running from, which is the other half of
        // the check: `statvfs` answers about a path that does not exist yet by way of its parent.
        assert!(room_at(&missing).unwrap() > 0);
    }

    /// [`Event::Exported`]'s list is read by the window and by whoever the window shows it to:
    /// one shape on every platform, and relative to the folder that was picked.
    #[test]
    fn a_written_file_is_listed_relative_to_the_destination_with_forward_slashes() {
        let dest = Path::new("/tmp/karas/exports/2026-09-21_1200_folder");
        assert_eq!(relative(dest, &dest.join("README.txt")), "README.txt");
        assert_eq!(
            relative(dest, &dest.join("placed").join("FY234012.ply")),
            "placed/FY234012.ply"
        );
        // A path from somewhere else is listed as it is rather than silently truncated.
        assert_eq!(relative(dest, Path::new("/etc/hosts")), "/etc/hosts");
        // Whole megabytes, rounded up: one byte is «1 MB free» and not «0».
        assert_eq!((mb(0), mb(1), mb(1 << 20), mb((1 << 20) + 1)), (0, 1, 1, 2));
    }
}
