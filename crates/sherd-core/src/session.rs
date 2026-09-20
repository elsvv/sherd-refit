//! A match that outlives its run (A §3): saved by `run_with`, read back by a review session, and
//! reassembled under the reviewer's decisions without matching a pair again.
//!
//! Matching is 94 % of a run (997 of 1066 s on the 155-fragment `karas`); R §8's assembly is
//! 0.14 s of it. A decision about one join therefore costs a reassembly, provided the candidate
//! list the assembly was built from is still there — which is what [`MatchState`] is.

use std::path::Path;

use nalgebra::Matrix4;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::assembly::constraints::{self, Constraints};
use crate::assembly::{Assembly, Piece, recenter};
use crate::collection::Entry;
use crate::error::{Error, Result};
use crate::executor::Engine;
use crate::fragment::Fragment;
use crate::matching::pair::Candidate;
use crate::memory::Budget;
use crate::objects::ObjectReport;
use crate::params::Params;
use crate::pipeline;
use crate::progress::Watch;
use crate::tiers::TierReport;
use crate::types::FragId;

/// The `format` tag of a state file.
pub const STATE_FORMAT: &str = "sherd-match-state";
/// The version this build writes and the only one it reads.
pub const STATE_VERSION: u32 = 1;

/// The candidate list and the tier report **as they stood before the last
/// `constraints::apply`**, with what R §8 needs to be run over them again.
///
/// Before, not after: `apply` promotes a `must_join` in place and the object round demotes in
/// place, so the lists a finished run holds already carry that run's constraints. Reassembling
/// under *other* constraints has to start from the match's own opinion.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchState {
    /// The collection's names, in R §2's order; a [`FragId`](crate::FragId) indexes them.
    pub names: Vec<String>,
    /// Every threshold the match ran with.
    pub params: Params,
    /// The collection's median wall thickness (R §4.1).
    pub thickness: f64,
    /// The collection's median working-mesh edge.
    pub resolution: f64,
    /// Pairs matched.
    pub pairs: usize,
    /// Pairs R §4.1's wall-ratio filter skipped.
    pub skipped_pairs: usize,
    /// D §4.3's `backend` label of the run that matched — `cpu`, or `gpu:Apple M2 Pro`.
    pub backend: String,
    /// Every candidate of every pair, in pair order, pinned poses last.
    pub candidates: Vec<Candidate>,
    /// The tier report over exactly that list, when the tier pass ran.
    pub tiers: Option<TierReport>,
}

/// The file's shape: the tag and the version first, so that a reader can refuse before it parses
/// 35 MB of candidates it would misread.
#[derive(Serialize)]
struct FileRef<'a> {
    format: &'a str,
    version: u32,
    names: &'a [String],
    params: &'a Params,
    thickness: f64,
    resolution: f64,
    pairs: usize,
    skipped_pairs: usize,
    backend: &'a str,
    candidates: &'a [Candidate],
    tiers: Option<&'a TierReport>,
}

/// The two fields a reader checks before it trusts the rest.
#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
}

/// The rest, once the header is the one this build writes.
#[derive(Deserialize)]
struct FileOwned {
    names: Vec<String>,
    params: Params,
    thickness: f64,
    resolution: f64,
    pairs: usize,
    skipped_pairs: usize,
    backend: String,
    candidates: Vec<Candidate>,
    tiers: Option<TierReport>,
}

impl MatchState {
    /// Writes the state as JSON (A §3.2).
    ///
    /// JSON and not a compact binary format: `Scores`, `Evidence`, `Probes` and `Params` are
    /// `report.json`'s own types and use `skip_serializing_if`, which a format that is not
    /// self-describing cannot read back. `serde_json`'s `float_roundtrip` makes every `f64` exact.
    ///
    /// # Errors
    ///
    /// [`Error::Write`] when the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = FileRef {
            format: STATE_FORMAT,
            version: STATE_VERSION,
            names: &self.names,
            params: &self.params,
            thickness: self.thickness,
            resolution: self.resolution,
            pairs: self.pairs,
            skipped_pairs: self.skipped_pairs,
            backend: &self.backend,
            candidates: &self.candidates,
            tiers: self.tiers.as_ref(),
        };
        let json =
            serde_json::to_vec(&file).map_err(|e| Error::write(path, std::io::Error::other(e)))?;
        std::fs::write(path, json).map_err(|e| Error::write(path, e))
    }

    /// Reads a state back, bit for bit what [`save`](Self::save) was given.
    ///
    /// # Errors
    ///
    /// [`Error::State`] for a file that is not a state, is of another version, or does not parse.
    pub fn load(path: &Path) -> Result<Self> {
        let refuse = |message: String| Error::State { path: path.to_owned(), message };
        let bytes = std::fs::read(path).map_err(|e| refuse(e.to_string()))?;
        let header: Header = serde_json::from_slice(&bytes)
            .map_err(|e| refuse(format!("not a match state: {e}")))?;
        if header.format != STATE_FORMAT {
            return Err(refuse(format!("not a match state: format `{}`", header.format)));
        }
        if header.version != STATE_VERSION {
            return Err(refuse(format!(
                "match state version {} — this build reads version {STATE_VERSION}; run the \
                 collection again",
                header.version
            )));
        }
        let f: FileOwned = serde_json::from_slice(&bytes).map_err(|e| refuse(e.to_string()))?;
        Ok(Self {
            names: f.names,
            params: f.params,
            thickness: f.thickness,
            resolution: f.resolution,
            pairs: f.pairs,
            skipped_pairs: f.skipped_pairs,
            backend: f.backend,
            candidates: f.candidates,
            tiers: f.tiers,
        })
    }
}

/// The collection's fragments through R §3.7's cache — what `run_with` itself starts from, for a
/// session that starts from a saved match instead of a run.
///
/// # Errors
///
/// The first fragment that fails to preprocess fails the call, as it fails a run.
pub fn load_fragments(
    entries: &[Entry],
    target_faces: usize,
    cache_dir: Option<&Path>,
    budget: Budget,
    seed: u64,
    watch: &Watch,
) -> Result<Vec<Fragment>> {
    pipeline::preprocess_collection(entries, target_faces, cache_dir, budget, seed, watch)
}

/// What [`reassemble`] produced: R §8's result and the lists it was built from.
#[derive(Debug)]
pub struct Reassembled {
    /// R §8's assembly after the object round.
    pub assembly: Assembly,
    /// The candidate list the assembly read — the state's, with each pinned pair's candidates
    /// replaced by its pinned one, and with what `apply` and the object round did to the bands.
    pub candidates: Vec<Candidate>,
    /// The tier report over exactly that list.
    pub tiers: Option<TierReport>,
    /// What each constraint did.
    pub constraints: Option<constraints::Report>,
    /// Roadmap item 4's report.
    pub objects: Option<ObjectReport>,
    /// One world pose per fragment after R §8.2's recentring. **Not refined** (R §9).
    pub poses: Vec<Matrix4<f64>>,
    /// The joins the assembly took, in the order it took them.
    pub used: Vec<(FragId, FragId)>,
}

/// R §8 over a saved match, under `constraints`, without matching anything (A §3.1).
///
/// The result is the one a **full run under the same constraints** would assemble from, given the
/// same match. A full run does not match a pair that `must_join` pins with a pose: the pair's only
/// candidate is the pinned one, appended after every matched candidate. This does the same to the
/// saved list — the pair's matched candidates and their tier rows go, the pinned one is appended.
/// When the pinned pose is bit for bit the pose of a candidate the match scored, that candidate's
/// scores and evidence are kept and R §6 is not run again; any other pose is scored by
/// `pinned_candidate`, as the full run would score it.
///
/// The tier pass is not repeated. A decision changes what is placed, not what the engine believed
/// about the pairs nobody decided on.
///
/// # Errors
///
/// [`Error::State`] when `fragments` is not the collection the state was matched on, and whatever
/// `constraints::resolve` refuses (an unknown name, a pair in two lists, a pose that is not rigid).
pub fn reassemble(
    engine: Engine<'_>,
    fragments: &[Fragment],
    state: &MatchState,
    constraints: Option<&Constraints>,
) -> Result<Reassembled> {
    if !fragments.iter().map(|f| f.name.as_str()).eq(state.names.iter().map(String::as_str)) {
        return Err(Error::State {
            path: std::path::PathBuf::new(),
            message: "the fragments are not the collection this match was made on".to_owned(),
        });
    }
    let params = &state.params;
    let plan = constraints.map(|file| constraints::resolve(file, &state.names)).transpose()?;

    let mut candidates = state.candidates.clone();
    let mut tiers = state.tiers.clone();
    if let Some(plan) = &plan {
        for forced in &plan.forced {
            let Some(pose) = forced.pose else { continue };
            let pair = constraints::key(forced.a, forced.b);
            let scored = candidates
                .iter()
                .position(|c| constraints::key(c.a, c.b) == pair && c.transform == pose);
            let pinned = match scored {
                Some(i) => Candidate { accepted: true, ..candidates[i] },
                None => {
                    pipeline::pinned_candidate(engine, fragments, forced.a, forced.b, &pose, params)
                }
            };
            let row = scored
                .and_then(|i| tiers.as_ref().map(|t| (t.evidence[i].clone(), t.probes[i].clone())));
            let keep: Vec<bool> =
                candidates.iter().map(|c| constraints::key(c.a, c.b) != pair).collect();
            retain(&mut candidates, &keep);
            if let Some(report) = tiers.as_mut() {
                retain(&mut report.tiers, &keep);
                retain(&mut report.evidence, &keep);
                retain(&mut report.probes, &keep);
                let (evidence, probes) = row.unwrap_or((None, None));
                report.tiers.push(pinned.tier);
                report.evidence.push(evidence);
                report.probes.push(probes);
            }
            candidates.push(pinned);
        }
    }

    let samples: Vec<Vec<[f64; 3]>> = fragments.iter().map(|f| f.samples.surface_f64()).collect();
    let scenes: Vec<_> = fragments.par_iter().map(Fragment::surface_scene_arc).collect();
    let piece = |with_mesh: bool| -> Vec<Piece<'_>> {
        fragments
            .iter()
            .zip(&samples)
            .zip(&scenes)
            .map(|((f, s), scene)| Piece {
                thick: f.thick,
                res: f.res(),
                watertight: f.watertight,
                mesh: if with_mesh { scene.as_deref() } else { None },
                s_pen: s,
            })
            .collect()
    };
    let pieces = piece(true);
    let mut stages = pipeline::StageLog::default();
    let pipeline::Assembled { assembly, constraints: report, objects } = pipeline::assemble_stage(
        engine.exec,
        fragments,
        &state.names,
        &pieces,
        &mut candidates,
        tiers.as_mut(),
        plan.as_ref(),
        params,
        None,
        &mut stages,
    );
    let used = assembly.used.iter().map(|&i| (candidates[i].a, candidates[i].b)).collect();
    let poses = recenter(&assembly.poses, &piece(false), &assembly.groups);
    Ok(Reassembled { assembly, candidates, tiers, constraints: report, objects, poses, used })
}

/// `Vec::retain` by a mask computed once, so that three parallel vectors lose the same rows.
fn retain<T>(items: &mut Vec<T>, keep: &[bool]) {
    let mut at = 0;
    items.retain(|_| {
        at += 1;
        keep[at - 1]
    });
}
