//! The wire between the window's process and the engine's (A §2.2): one JSON object per line, a
//! [`Job`] in, [`Event`]s out, and [`Request`]s in while the job runs.
//!
//! Every enum here is serde-tagged rather than positional, and every worker opens with
//! [`Event::Hello`] carrying [`PROTOCOL`]: the two processes are two builds of the app that a user
//! can get out of step (a half-finished update, a stale worker left running), and a tag plus a
//! version number is what lets the newer of them say so instead of mis-reading the other.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sherd_core::Params;
use sherd_core::assembly::constraints::Constraints;
use sherd_core::matching::verify::Scores;
use sherd_core::objects::ObjectParams;
use sherd_core::pipeline::RunOptions;
use sherd_core::report::FragmentStats;
use sherd_core::tiers::{Evidence, Thresholds, Tier};

pub use crate::decisions::{Decision, DecisionsFile};
pub use crate::run::{EngineInfo, FailKind, RunCounts, StageTime};

/// The protocol this build speaks. A window that reads another number fails the job with
/// [`FailKind::Protocol`] rather than guessing (A §2.2).
pub const PROTOCOL: u32 = 1;

/// What a worker is asked to do — exactly one per process (A §2.1).
#[allow(
    clippy::large_enum_variant,
    reason = "`RunJob` is large because `Params` is, and exactly one `Job` crosses the pipe in a \
              worker's life: boxing would save that single move and cost every arm a deref"
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "job", rename_all = "snake_case")]
pub enum Job {
    /// What this build can run on: the backend lines, and the device self-test when asked for it.
    Info {
        /// The adapter to open, or `None` for the one `--backend auto` would pick.
        adapter: Option<String>,
        /// Whether to run the self-test, which opens the device and costs a second.
        selftest: bool,
    },
    /// Preprocess a collection into the workspace's cache (A §5).
    Prepare(PrepareJob),
    /// One run of the pipeline into `runs/<run_id>/` (A §7).
    Run(RunJob),
    /// A review session over a finished run (A §8): the fragments and the saved match loaded
    /// once, then [`Request`]s answered until the window says [`Request::Close`].
    Review(ReviewJob),
}

/// The `Prepare` job's sheet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrepareJob {
    /// The workspace folder: `cache/` and `fragments/` are written under it.
    pub workspace: PathBuf,
    /// The input folder, resolved by the host (A §4: the workspace stores it relative).
    pub input: PathBuf,
    /// Scans the workspace leaves out (R §2's naming is unaffected: M1.6).
    pub excluded: BTreeSet<String>,
    /// Faces of the working mesh, `RunOptions::target_faces`.
    pub target_faces: usize,
    /// R §10's seed, so a second `Prepare` of the same folder is the same cache.
    pub seed: u64,
    /// The memory budget in gigabytes, or `None` for [`sherd_core::memory::Budget`]'s own guess.
    pub memory_gb: Option<f64>,
    /// Worker threads; 0 is rayon's default.
    pub workers: usize,
}

/// The `Run` job's sheet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunJob {
    /// The workspace folder; the run's own folder is `runs/<run_id>/` under it.
    pub workspace: PathBuf,
    /// The input folder, resolved by the host.
    pub input: PathBuf,
    /// The run's id, which is also its folder's name (A §4).
    pub run_id: String,
    /// Scans the workspace leaves out.
    pub excluded: BTreeSet<String>,
    /// The launch sheet (A §7.4).
    pub spec: RunSpec,
    /// What the reviewer's decisions came to (A §8.2), or `None` when there are none.
    pub constraints: Option<Constraints>,
}

/// The `Review` job's sheet (A §8.4). A session, not a run: nothing new is matched and nothing is
/// written into the run's folder — the worker answers questions about a match that already exists.
///
/// It carries the same four preprocessing numbers a [`PrepareJob`] does, because a session that
/// finds a fragment missing from `cache/` preprocesses it exactly as a run would, and a working
/// mesh built at another face budget or another seed is not the one the match was made on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReviewJob {
    /// The workspace folder; the run's own folder is `runs/<run_id>/` under it.
    pub workspace: PathBuf,
    /// The input folder, resolved by the host.
    pub input: PathBuf,
    /// The run being reviewed — the folder [`crate::worker::MATCH_STATE_FILE`] is read from.
    pub run_id: String,
    /// Scans the workspace leaves out.
    pub excluded: BTreeSet<String>,
    /// Faces of the working mesh, as the run used.
    pub target_faces: usize,
    /// R §10's seed, as the run used.
    pub seed: u64,
    /// The memory budget in gigabytes, or `None` for [`sherd_core::memory::Budget`]'s own guess.
    pub memory_gb: Option<f64>,
    /// Worker threads; 0 is rayon's default.
    pub workers: usize,
}

/// What the window can say to a worker that is already on a job.
///
/// Everything but [`Request::Cancel`] belongs to a review session (A §8): a `Prepare` and a `Run`
/// read their input once and answer nothing else. Not `Eq`, because a decision carries a pose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
pub enum Request {
    /// Raise D §5's flag. The job ends at its next unit of work.
    Cancel,
    /// R §8 again under these decisions (A §8.2), without matching a pair. Answered with
    /// [`Event::Assembly`], preceded by [`Event::Dropped`] when a decision names a fragment the
    /// collection no longer has.
    Reassemble {
        /// The decisions as they stand — the whole list, because the window owns undo and redo.
        decisions: DecisionsFile,
    },
    /// The seam of one placement, measured but not drawn (A §2.2). Answered with
    /// [`Event::PairDetail`].
    PairDetail {
        /// The fragment the pose maps *into*.
        a: String,
        /// The fragment the pose moves.
        b: String,
        /// `T`: `p_a = T · p_b`, row-major — the same matrix a [`CandidateRow`] carries.
        pose: [[f64; 4]; 4],
    },
    /// R §9 over the groups these decisions left unrefined (A §8.4). Answered with
    /// [`Event::Assembly`], in which every group of two or more is `refined`.
    Refine {
        /// The decisions the assembly to refine is built from.
        decisions: DecisionsFile,
    },
    /// Write the reviewed assembly with the engine's own writers (A §9.1). Answered with
    /// [`Event::Assembly`] — the refinement below becomes the session's baseline, so the window's
    /// state is the state that was exported — and then [`Event::Exported`].
    Export {
        /// The decisions the assembly to export is built from.
        decisions: DecisionsFile,
        /// Which of A §9.1's two kinds.
        what: ExportWhat,
        /// Where to write it: a folder that does not exist yet, or an empty one.
        dest: PathBuf,
    },
    /// The session is over: the worker answers [`Event::Done`] and exits.
    Close,
}

/// A §9.1's two kinds of export, which are the two shapes of `RunOptions` the engine's writers
/// take: everything a `sherd-refit-rs run` writes, or the same without a single mesh.
///
/// The engine's switches do not separate `scene.glb` from `placed/`, which is why this is two
/// kinds with three opt-ins and not five checkboxes: `write_meshes` turns the lot on, and the
/// three that are left are the ones a run's own flags have (`--placed-all`, `--merged-meshes`,
/// and R §11.5's previews).
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportWhat {
    /// «Папка результата»: the tables, the report, `transforms.*`, `README.txt`, `placed/*.ply`,
    /// `scene.glb` and `viewer.html` — `write_meshes` and `viewer` on, plus the three below.
    Folder {
        /// `--placed-all`: a `placed/<name>.ply` for every fragment, not only the assembled ones.
        placed_all: bool,
        /// `--merged-meshes`: R §11.4's `assembly_<k>.ply`, one per group of two or more.
        merged_meshes: bool,
        /// R §11.5's group previews.
        previews: bool,
    },
    /// «Только таблицы и отчёт»: the same folder with every mesh and every picture switched off —
    /// seconds and megabytes instead of minutes and gigabytes, for a colleague who wants numbers.
    Tables,
}

/// The launch sheet's executor (A §7.4), which is `sherd_core::Backend` with serde on it — the
/// engine's own enum has none, deliberately, because it is not part of any file it writes.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendChoice {
    /// GPU when there is one that works, CPU otherwise.
    Auto,
    /// The CPU executor, whatever the machine has.
    Cpu,
    /// The GPU executor; a machine without one fails the run.
    Gpu,
}

/// Which of A §7.4's three buttons the sheet came from. It changes nothing the engine reads — the
/// thresholds below are the whole truth — and is carried so the history can say what was asked.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    /// «Стандарт»: the command line's defaults, threshold for threshold.
    Standard,
    /// «Тщательный»: the defaults with the agreement arm on.
    Thorough,
    /// «Свой»: the user moved something.
    Custom,
}

/// The subset of `RunOptions` the window sets (A §7.4); everything else is `RunOptions::default()`.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSpec {
    /// Which button the sheet came from.
    pub preset: Preset,
    /// The executor asked for.
    pub backend: BackendChoice,
    /// The adapter, when the user picked one by name.
    pub adapter: Option<String>,
    /// The GPU memory budget in gigabytes, or `None` for the backend's own guess.
    pub gpu_memory_gb: Option<f64>,
    /// R §10's seed.
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub seed: u64,
    /// Faces of the working mesh.
    pub target_faces: usize,
    /// Whether the confidence tier runs at all (`Params::tiers`).
    pub tiers: bool,
    /// Independent agreeing paths the tier's support arm wants (`Thresholds::agree_seeds`).
    pub agree_seeds: u32,
    /// R §4.1's wall-ratio filter.
    pub thick_ratio: f64,
    /// R §6.5's tight-contact fraction.
    pub min_tight: f64,
    /// R §6.5's median gap, in `t`.
    pub max_gap: f64,
    /// R §6.5's penetration fraction.
    pub max_pen: f64,
    /// R §6.5's shortest shared seam, in `t`.
    pub min_seam: f64,
    /// Worker threads; 0 is rayon's default.
    pub workers: usize,
    /// The memory budget in gigabytes, or `None` for the machine's.
    pub memory_gb: Option<f64>,
}

impl Default for RunSpec {
    fn default() -> Self {
        let (p, t, o) = (Params::default(), Thresholds::default(), RunOptions::default());
        Self {
            preset: Preset::Standard,
            backend: BackendChoice::Auto,
            adapter: None,
            gpu_memory_gb: None,
            seed: p.seed,
            target_faces: o.target_faces,
            tiers: true,
            agree_seeds: t.agree_seeds,
            thick_ratio: p.thick_ratio,
            min_tight: p.min_tight,
            max_gap: p.max_gap,
            max_pen: p.max_pen,
            min_seam: p.min_seam,
            workers: 0,
            memory_gb: None,
        }
    }
}

impl RunSpec {
    /// Every threshold of the run: the CLI's defaults (`crates/sherd-cli/src/main.rs`, `params`)
    /// with the launch sheet's eleven on top (A §7.4).
    pub fn params(&self) -> Params {
        Params {
            seed: self.seed,
            thick_ratio: self.thick_ratio,
            min_tight: self.min_tight,
            max_gap: self.max_gap,
            max_pen: self.max_pen,
            min_seam: self.min_seam,
            tiers: self
                .tiers
                .then(|| Thresholds { agree_seeds: self.agree_seeds, ..Thresholds::default() }),
            objects: Some(ObjectParams::default()),
            ..Params::default()
        }
    }

    /// The engine's `Backend` for the choice.
    pub fn backend(&self) -> sherd_core::Backend {
        match self.backend {
            BackendChoice::Auto => sherd_core::Backend::Auto,
            BackendChoice::Cpu => sherd_core::Backend::Cpu,
            BackendChoice::Gpu => sherd_core::Backend::Gpu,
        }
    }
}

/// What a worker says. One line each, in the order they happened.
#[allow(
    clippy::large_enum_variant,
    reason = "`Done` is the large one because it carries the run's resolved `Params`, and it is \
              said once, last; the variants that do arrive in numbers are `Stage` and `Progress`"
)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// Always first, before the job line is even read: who is on the other end.
    Hello {
        /// [`PROTOCOL`].
        protocol: u32,
        /// `sherd-core`'s version.
        core_version: String,
        /// The algorithm reference it implements.
        algo_ref: String,
        /// The commit it was built from.
        commit: String,
    },
    /// The answer to [`Job::Info`].
    Info {
        /// What `sherd-refit-rs info` prints about this build's executors.
        backends: Vec<String>,
        /// The self-test's lines, empty when it was not asked for.
        selftest: Vec<String>,
    },
    /// A stage has begun. Sent once per stage, on its first report: the set of stages varies with
    /// the options (`screen`, `second_pass`), so the window owns the strip and lights it by name.
    Stage {
        /// R §11.2's stage name.
        name: String,
    },
    /// D §5's progress, throttled (`worker::WorkerProgress`).
    Progress {
        /// Which stage.
        stage: String,
        /// Units finished.
        done: usize,
        /// Units in the stage.
        total: usize,
    },
    /// One fragment of a `Prepare` is preprocessed and its display mesh is written (A §5).
    FragmentReady(FragmentInfo),
    /// A review session has the collection and the match in memory and will answer requests
    /// (A §8): the window may open the screen. Said once, by [`Job::Review`] and nothing else.
    Ready {
        /// Fragments loaded.
        fragments: usize,
        /// Candidates the saved match holds.
        candidates: usize,
    },
    /// What was assembled, as the window draws it (A §8). A run says it once, at its end; a
    /// review session says it again for every `Reassemble` and every `Refine`.
    Assembly(AssemblyDto),
    /// The seam of the placement a `PairDetail` asked about (A §8.3).
    PairDetail(PairDetailDto),
    /// Decisions a `Reassemble` could not carry, because a fragment they name is not in the
    /// collection any more (A §8.5). Said before the [`Event::Assembly`] that ignores them, so
    /// that the window can tell the reviewer what was quietly left out.
    Dropped {
        /// The decisions that were not applied, as they stand in the file.
        decisions: Vec<Decision>,
    },
    /// One request of a session could not be answered — a pose that is not rigid, a pair the
    /// collection does not have. The session stays open: the next request is still served, which
    /// is what tells this apart from [`Event::Failed`].
    RequestFailed {
        /// Why, in the engine's words.
        message: String,
    },
    /// An [`Request::Export`] is written and closed (A §9.1). The session stays open: an export
    /// is one more question about the assembly, not the end of the review.
    Exported {
        /// The folder it was written into, as the window shows it and reveals it.
        dest: PathBuf,
        /// Every file written, relative to `dest` and with `/` between the parts whatever the
        /// platform — a list the window counts and prints, not a path it opens.
        files: Vec<String>,
        /// What they came to on disk, in bytes.
        #[cfg_attr(feature = "ts", ts(type = "number"))]
        bytes: u64,
    },
    /// The job is over and it went well. The three blocks are a run's; `Info` fills none of them.
    Done {
        /// What the run found.
        counts: Option<RunCounts>,
        /// What ran it.
        engine: Option<EngineInfo>,
        /// Every threshold it resolved to.
        #[cfg_attr(feature = "ts", ts(type = "Record<string, unknown> | null"))]
        params: Option<Params>,
    },
    /// The job is over and it did not. Always the last line (A §10).
    Failed {
        /// Why, by class.
        kind: FailKind,
        /// Why, in the engine's words.
        message: String,
    },
}

/// One preprocessed fragment, as the window's collection table shows it (A §5).
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FragmentInfo {
    /// The fragment's name (R §2).
    pub name: String,
    /// Its file's name in the input folder.
    pub file: String,
    /// Bytes, for the input snapshot the host keeps (A §4).
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub size: u64,
    /// Modification time, milliseconds since the epoch.
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub mtime_ms: i64,
    /// R §11.3's fragment row.
    #[cfg_attr(feature = "ts", ts(as = "FragmentStatsTs"))]
    pub stats: FragmentStats,
    /// What the window should say about it in red.
    pub warnings: Vec<Warning>,
    /// Whether the scan carries per-vertex colour, which decides how its mesh is drawn.
    pub coloured: bool,
    /// Faces of the display mesh that was written for it.
    pub display_faces: usize,
}

/// [`sherd_core::report::FragmentStats`], field for field, for the TypeScript bindings only: the
/// engine's type cannot derive `TS` from here, and A §2.2's wire contract must still have one
/// source. [`crate::view`]'s tests compare the two field by field, so a field added to the
/// engine's row and forgotten here fails the app's own test suite rather than the window.
#[cfg(feature = "ts")]
#[derive(Debug, Default, serde::Serialize, ts_rs::TS)]
#[ts(export, rename = "FragmentStats")]
pub struct FragmentStatsTs {
    /// The fragment's name in the collection.
    name: String,
    /// Faces of the working mesh.
    #[ts(type = "number")]
    faces: u64,
    /// Faces of the file, after R §3.1's cleaning and before the largest-component pass.
    #[ts(type = "number")]
    orig_faces: u64,
    /// Vertices of the same.
    #[ts(type = "number")]
    orig_vertices: u64,
    /// R §3.2's wall thickness.
    thickness: f64,
    /// R §3.2's unfiltered ray mode.
    thickness_mode: f64,
    /// R §3.3's `res`.
    resolution: f64,
    /// R §3.3.2's verdict.
    watertight: bool,
    /// The working mesh's bounding-box side lengths.
    extent: [f64; 3],
    /// Total area of the working mesh.
    area: f64,
    /// Fracture area over total area (R §3.4).
    fracture_area_fraction: f64,
}

/// [`sherd_core::matching::verify::Scores`], field for field, for the TypeScript bindings only.
///
/// The same arrangement and the same reason as [`FragmentStatsTs`], and the same test keeping it
/// honest: A §8.3's scores panel puts each of these numbers against its threshold and colours it,
/// and a `Record<string, number>` is not a contract that can be written against — a renamed key
/// would reach the window as a blank cell instead of a compile error.
///
/// The two flags carry the engine's own JSON shape: left out unless they are set, and written as
/// `1.0` when they are (`verify::as_one`), which is why they are numbers here and not booleans.
#[cfg(feature = "ts")]
#[derive(Debug, Default, serde::Serialize, ts_rs::TS)]
#[ts(export, rename = "Scores")]
pub struct ScoresTs {
    /// Fraction of A's facing fracture samples within the tight limit of B's surface (R §6.1).
    #[serde(rename = "tightA")]
    tight_a: f64,
    /// The same for B's samples against A's.
    #[serde(rename = "tightB")]
    tight_b: f64,
    /// `min(tightA, tightB)` — the side that fits worse is the fit.
    tight: f64,
    /// Median distance of A's facing samples to B's fracture surface, in `t`.
    #[serde(rename = "gapA")]
    gap_a: f64,
    /// The same for B.
    #[serde(rename = "gapB")]
    gap_b: f64,
    /// `max(gapA, gapB)`, in `t`.
    gap: f64,
    /// Area of A in contact with B, in `t²`.
    #[serde(rename = "contactA")]
    contact_a: f64,
    /// The same for B.
    #[serde(rename = "contactB")]
    contact_b: f64,
    /// `min(contactA, contactB)`, in `t²`.
    contact: f64,
    /// Length of the shared seam, in `t` (R §6.2).
    seam: f64,
    /// The pair's own gap limit, in `t` — what `gap` is shown against.
    gap_limit: f64,
    /// The distance `tight` counted, in `t`.
    tight_delta: f64,
    /// Median step height of the outer shell across the seam, in `t` (R §6.3).
    cont: f64,
    /// Median agreement of the two shells' normals across the seam (R §6.3).
    cont_n: f64,
    /// Fraction of surface samples of either fragment inside the other (R §6.4).
    pen: f64,
    /// Deepest excursion of either fragment into the other, in `t` (R §6.4).
    pen_depth: f64,
    /// `1` when a fragment is not watertight and R §6.4 could not run; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pen_unavailable: Option<f64>,
    /// `1` when only the cheap half of R §6 was computed; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    partial: Option<f64>,
    /// R §5.4's re-score of the stage-1 pose this candidate was refined from.
    brk: f64,
    /// The best stage-1 re-score of the whole pair (R §5.7).
    brk_best: f64,
}

/// [`sherd_core::tiers::Evidence`], field for field, for the TypeScript bindings only.
///
/// This is what A §8.3's «Почему не подтверждён сам» is written from — `failed` above all, which
/// the window turns into sentences — and what says why a confirmed join is confirmed (`arm`). As
/// [`ScoresTs`], with a test against the engine's own keys.
///
/// Every field the engine leaves out of the JSON when it has nothing to say is optional here, and
/// for the same reason: a run that probed no rival writes no `margin`, and a window that read
/// `0` there would show a number the engine never measured.
#[cfg(feature = "ts")]
#[derive(Debug, Default, serde::Serialize, ts_rs::TS)]
#[ts(export, rename = "Evidence")]
pub struct EvidenceTs {
    /// `score / rival_score` over a second placement that scores.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    margin: Option<f64>,
    /// How far that second placement puts the sherd, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    rival_moved_t: Option<f64>,
    /// The same ratio over the wide second placement (R §5.6's full list, then stage 1).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    wide_margin: Option<f64>,
    /// How far the wide second placement puts the sherd from **this** candidate, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    wide_rival_moved_t: Option<f64>,
    /// The same distance from the pair's best candidate, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    wide_rival_pair_t: Option<f64>,
    /// Where the wide second placement was found.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional, type = "\"stage2\" | \"stage1\"")]
    wide_rival_source: Option<String>,
    /// Whether R §6.5 would accept that second placement itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    wide_rival_accepted: Option<bool>,
    /// Distinct placements the pair's kept list makes.
    #[ts(type = "number")]
    placements: u64,
    /// Worst rotation over the twelve one-ULP neighbours, in degrees.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    determined_deg: Option<f64>,
    /// The same in translation, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    determined_t: Option<f64>,
    /// How far the pose stays away after a ±0.5 t push along the seam, in `t`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    slide_t: Option<f64>,
    /// Smallest `tight` over the three draws.
    resample_tight_min: f64,
    /// Largest `gap` over the three draws, in `t`.
    resample_gap_max: f64,
    /// How many of the three draws R §6.5 accepts.
    resample_accept: u32,
    /// Independent accepted joins that agree with this placement.
    support: u32,
    /// The tier's tests this candidate failed; left out on a confirmed join. A §8.3 turns each of
    /// these into a sentence (`state/reasons.ts`).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    failed: Option<Vec<String>>,
    /// Which distinguishing arm confirmed this candidate — `support`, `margin` or `research`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    arm: Option<String>,
    /// How many of this candidate's re-searches landed on its placement, and how many were run.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    research: Option<[u32; 2]>,
    /// Task S2's colour agreement across the seam — reported, never gated.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    colour: Option<ColourAgreementTs>,
}

/// [`sherd_core::tiers::ColourAgreement`] for the bindings, as [`EvidenceTs`] carries it.
#[cfg(feature = "ts")]
#[derive(Debug, Default, serde::Serialize, ts_rs::TS)]
#[ts(export, rename = "ColourAgreement")]
pub struct ColourAgreementTs {
    /// CIE76 between the two fragments' fracture-face mean Lab.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    frac_delta_e: Option<f64>,
    /// Total variation between the two fragments' shell-face Lab histograms, `0`…`1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    shell_hist: Option<f64>,
}

/// Something worth telling the reviewer about one fragment (A §5).
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "warning", rename_all = "snake_case")]
pub enum Warning {
    /// Its wall is far from the collection's — R §4.1 will refuse most of its pairs.
    ThicknessOutlier {
        /// This fragment's wall.
        thickness: f64,
        /// The collection's median.
        median: f64,
    },
    /// R §3.3.2 says the mesh is open, so R §6.4's penetration test cannot run on it.
    NotWatertight,
}

/// What was assembled, by name: the window has the meshes already and needs only the poses.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AssemblyDto {
    /// The groups, largest first, as the engine laid them out.
    pub groups: Vec<GroupDto>,
    /// Every placed fragment's pose, row-major, in its group's frame.
    pub poses: BTreeMap<String, [[f64; 4]; 4]>,
    /// The joins that hold the groups together.
    pub joins: Vec<JoinDto>,
    /// Pairs that were candidates and did not make it, with the reason to show.
    pub unplaced: Vec<UnplacedDto>,
}

/// One assembled group.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupDto {
    /// Its fragments, by name.
    pub members: Vec<String>,
    /// Whether R §8.2's refinement has run on it.
    pub refined: bool,
}

/// One join of a group.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinDto {
    /// One fragment.
    pub a: String,
    /// The other.
    pub b: String,
}

/// A pair the assembly did not use, and why.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnplacedDto {
    /// One fragment.
    pub a: String,
    /// The other.
    pub b: String,
    /// What to tell the reviewer — the tier's refusal, or the conflict that dropped it.
    pub reason: String,
}

/// The seam of one placement, as the «Ревью» screen draws it (A §2.2's `PairDetail`, A §8.3):
/// [`sherd_core::review::SeamView`] on the wire.
///
/// `f32` triples and not `f64`: every number here goes straight into a `THREE.BufferAttribute`,
/// which is `Float32Array`, and B's fracture samples are five thousand points — half the JSON for
/// the same picture. The two limits stay `f64` because they are printed, not drawn.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PairDetailDto {
    /// The fragment the pose maps into, drawn at the identity.
    pub a: String,
    /// The fragment the pose moves.
    pub b: String,
    /// B's fracture samples at the pose, in A's frame (R §3.5.2).
    pub contact: Vec<[f32; 3]>,
    /// One class per contact point: 0 under the tight limit, 1 under the gap limit, 2 beyond
    /// (R §6.1, R §6.5) — the green, yellow and red of the engine's own review images.
    pub contact_class: Vec<u8>,
    /// Centres of R §6.2's seam voxels, in A's frame.
    pub seam: Vec<[f32; 3]>,
    /// The pair's tight limit, in the meshes' own length unit (R §1.2).
    pub tight: f64,
    /// The pair's gap limit, in the same unit.
    pub gap: f64,
}

/// One row of `candidates.json` (A §4), the index the review screen works from: a row carries
/// everything A §8.3 puts on the screen for one candidate, so that opening a join reads a few
/// hundred kilobytes and never the 54 000 rows of `report.json`.
///
/// By name and not by [`sherd_core::FragId`]: the index outlives the collection's numbering, and
/// A §8.1's decisions are about names too.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateRow {
    /// The fragment the pose maps *into* (R §4.1's first name).
    pub a: String,
    /// The fragment the pose moves.
    pub b: String,
    /// `T`: `p_a = T · p_b`, row-major — the matrix the viewer puts on **b**.
    pub pose: [[f64; 4]; 4],
    /// R §5.7's ranking key, `seam · tight`: what the queue sorts on.
    pub score: f64,
    /// Every number R §6 produced, for the scores panel against its thresholds.
    #[cfg_attr(feature = "ts", ts(as = "ScoresTs"))]
    pub scores: Scores,
    /// The band it ended in — after the constraints and the object round, which is the band the
    /// run acted on.
    #[cfg_attr(feature = "ts", ts(type = "\"confirmed\" | \"probable\" | \"rejected\""))]
    pub tier: Tier,
    /// Whether R §8 built with this pair. Several candidates of a pair routinely converge on one
    /// placement, so the flag is the pair's and not one pose's.
    pub used: bool,
    /// The tier pass's evidence, which is what A §8.3 says «почему не подтверждено» from; `None`
    /// where the pass did not run or never probed this candidate.
    #[cfg_attr(feature = "ts", ts(as = "Option<EvidenceTs>"))]
    pub evidence: Option<Evidence>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_and_an_event_are_one_tagged_json_line_each() {
        let job = Job::Info { adapter: None, selftest: false };
        let line = serde_json::to_string(&job).unwrap();
        assert_eq!(line, r#"{"job":"info","adapter":null,"selftest":false}"#);
        assert_eq!(serde_json::from_str::<Job>(&line).unwrap(), job);

        let event = Event::Progress { stage: "matching".into(), done: 7, total: 190 };
        assert_eq!(
            serde_json::to_string(&event).unwrap(),
            r#"{"event":"progress","stage":"matching","done":7,"total":190}"#
        );
        assert_eq!(serde_json::to_string(&Request::Cancel).unwrap(), r#"{"request":"cancel"}"#);
    }

    /// A §2.2: a review session's requests and answers cross the same pipe as everything else —
    /// one tagged line each, and each one readable back into the value it was written from.
    #[test]
    fn a_review_sessions_requests_and_events_are_one_tagged_line_each() {
        let round = |request: &Request| {
            let line = serde_json::to_string(request).unwrap();
            assert_eq!(&serde_json::from_str::<Request>(&line).unwrap(), request);
            line
        };
        let pose = [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(
            round(&Request::PairDetail { a: "pieceA".into(), b: "pieceB".into(), pose }),
            r#"{"request":"pair_detail","a":"pieceA","b":"pieceB","pose":[[1.0,0.0,0.0,0.5],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]}"#
        );
        assert_eq!(
            round(&Request::Reassemble { decisions: DecisionsFile::default() }),
            r#"{"request":"reassemble","decisions":{"version":1,"decisions":[]}}"#
        );
        assert_eq!(
            round(&Request::Refine { decisions: DecisionsFile::default() }),
            r#"{"request":"refine","decisions":{"version":1,"decisions":[]}}"#
        );
        // A §9.1: the two kinds are one tagged object inside the request, so that a window one
        // version older reads `kind` and says «этого экспорта я не знаю» instead of guessing.
        assert_eq!(
            round(&Request::Export {
                decisions: DecisionsFile::default(),
                what: ExportWhat::Folder {
                    placed_all: false,
                    merged_meshes: true,
                    previews: false
                },
                dest: PathBuf::from("/tmp/out"),
            }),
            r#"{"request":"export","decisions":{"version":1,"decisions":[]},"what":{"kind":"folder","placed_all":false,"merged_meshes":true,"previews":false},"dest":"/tmp/out"}"#
        );
        assert_eq!(
            round(&Request::Export {
                decisions: DecisionsFile::default(),
                what: ExportWhat::Tables,
                dest: PathBuf::from("/tmp/out"),
            }),
            r#"{"request":"export","decisions":{"version":1,"decisions":[]},"what":{"kind":"tables"},"dest":"/tmp/out"}"#
        );
        assert_eq!(round(&Request::Close), r#"{"request":"close"}"#);

        let ready = Event::Ready { fragments: 155, candidates: 54_000 };
        let line = serde_json::to_string(&ready).unwrap();
        assert_eq!(line, r#"{"event":"ready","fragments":155,"candidates":54000}"#);
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), ready);
        let detail = Event::PairDetail(PairDetailDto {
            a: "pieceA".into(),
            b: "pieceB".into(),
            contact: vec![[1.0, 2.0, 3.0]],
            contact_class: vec![0],
            seam: Vec::new(),
            tight: 0.35,
            gap: 1.5,
        });
        let line = serde_json::to_string(&detail).unwrap();
        assert!(line.starts_with(r#"{"event":"pair_detail","a":"pieceA""#), "{line}");
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), detail);
        assert_eq!(
            serde_json::to_string(&Event::Dropped { decisions: Vec::new() }).unwrap(),
            r#"{"event":"dropped","decisions":[]}"#
        );
        assert_eq!(
            serde_json::to_string(&Event::RequestFailed { message: "no such pair".into() })
                .unwrap(),
            r#"{"event":"request_failed","message":"no such pair"}"#
        );
        let exported = Event::Exported {
            dest: PathBuf::from("/tmp/out"),
            files: vec!["placed/pieceA.ply".to_owned()],
            bytes: 2_048,
        };
        let line = serde_json::to_string(&exported).unwrap();
        assert_eq!(
            line,
            r#"{"event":"exported","dest":"/tmp/out","files":["placed/pieceA.ply"],"bytes":2048}"#
        );
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), exported);
    }

    /// The keys of a serialised value, in the order serde wrote them.
    #[cfg(feature = "ts")]
    fn keys(value: &serde_json::Value) -> Vec<String> {
        value.as_object().expect("an object").keys().cloned().collect()
    }

    /// The mirrors exist because the engine's types cannot derive `TS` from here (A §2.2: the
    /// wire contract has one source, and it is the Rust). This is what keeps them honest — with
    /// every optional field of the engine's own set, so that one left out of the mirror fails
    /// here and not in the window.
    #[cfg(feature = "ts")]
    #[test]
    fn the_typescript_mirrors_of_a_candidates_scores_and_evidence_have_the_engines_keys() {
        use sherd_core::measure::RivalSource;
        use sherd_core::tiers::ColourAgreement;

        let real = Scores { pen_unavailable: true, partial: true, ..Scores::default() };
        let mirror =
            ScoresTs { pen_unavailable: Some(1.0), partial: Some(1.0), ..ScoresTs::default() };
        assert_eq!(
            keys(&serde_json::to_value(real).unwrap()),
            keys(&serde_json::to_value(mirror).unwrap())
        );

        // Named field by field rather than built from a `Default`, which `Evidence` has none of:
        // a field added to the engine's row stops this test compiling, which is the point.
        let real = Evidence {
            margin: Some(6.2),
            rival_moved_t: Some(9.1),
            wide_margin: Some(2.4),
            wide_rival_moved_t: Some(8.2),
            wide_rival_pair_t: Some(8.2),
            wide_rival_source: Some(RivalSource::Stage2),
            wide_rival_accepted: Some(false),
            placements: 3,
            determined_deg: Some(1e-3),
            determined_t: Some(1e-4),
            slide_t: Some(0.5),
            resample_tight_min: 0.31,
            resample_gap_max: 0.02,
            resample_accept: 2,
            support: 1,
            failed: vec!["cont_n 0.8485 < 0.9".to_owned()],
            arm: Some("support".to_owned()),
            research: Some([1, 2]),
            colour: Some(ColourAgreement { frac_delta_e: Some(3.4), shell_hist: Some(0.12) }),
        };
        let mirror = EvidenceTs {
            margin: Some(6.2),
            rival_moved_t: Some(9.1),
            wide_margin: Some(2.4),
            wide_rival_moved_t: Some(8.2),
            wide_rival_pair_t: Some(8.2),
            wide_rival_source: Some("stage2".to_owned()),
            wide_rival_accepted: Some(false),
            placements: 3,
            determined_deg: Some(1e-3),
            determined_t: Some(1e-4),
            slide_t: Some(0.5),
            resample_tight_min: 0.31,
            resample_gap_max: 0.02,
            resample_accept: 2,
            support: 1,
            failed: Some(vec!["cont_n 0.8485 < 0.9".to_owned()]),
            arm: Some("support".to_owned()),
            research: Some([1, 2]),
            colour: Some(ColourAgreementTs { frac_delta_e: Some(3.4), shell_hist: Some(0.12) }),
        };
        let (real, mirror) =
            (serde_json::to_value(real).unwrap(), serde_json::to_value(mirror).unwrap());
        assert_eq!(keys(&real), keys(&mirror));
        // And the one nested object, whose keys the window reads the same way.
        assert_eq!(keys(&real["colour"]), keys(&mirror["colour"]));
        // The word the engine writes for a source is the word the binding's union offers.
        assert_eq!(real["wide_rival_source"], mirror["wide_rival_source"]);
    }

    /// A §7.4: «Стандарт» is the CLI's defaults, threshold for threshold.
    #[test]
    fn the_standard_preset_is_the_command_lines_defaults() {
        use sherd_core::objects::ObjectParams;
        use sherd_core::tiers::Thresholds;
        let cli = Params {
            tiers: Some(Thresholds::default()),
            objects: Some(ObjectParams::default()),
            ..Params::default()
        };
        assert_eq!(RunSpec::default().params(), cli);
        let thorough = RunSpec { agree_seeds: 2, ..RunSpec::default() }.params();
        assert_eq!(thorough.tiers.unwrap().agree_seeds, 2);
        assert_eq!(RunSpec { tiers: false, ..RunSpec::default() }.params().tiers, None);
    }
}
