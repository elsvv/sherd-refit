//! `sherd-refit-rs` — the command line front end of the Rust core.
//!
//! The subcommands are D §9's: `run` and `segment` mirror the Python's, flag for flag, and
//! `parity`, `bench` and `info` are new. Phase 1a implemented `info`, `segment` up to the working
//! mesh and `parity` for the stages the port computes; step B1 added R §3.4's shell/fracture
//! labels to `segment` and its own `parity` row, step B2 R §3.5's breaklines and theirs, step
//! B3 the sampled match arrays and the `samples` row, step C1 the first three pair rows —
//! `hypotheses`, `coarse` and `nms` — step C2 the two refinement rows and step C3 the last two,
//! `verify` and `candidates`, which is the whole of `match_pair`. Step D1 opened phase 1d with the
//! `assembly` row — R §8's groups, its used joins, its rejections and R §8.2's recentring — and
//! step D2 the last two, `refine` and `outputs`. Step D3 turned `run` and `bench` on: `run` is the
//! reference's own subcommand, flag for flag (R §1.4), and `bench` is a run with the previews off,
//! timed against D §10.3.

use std::path::PathBuf;

mod gpu;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use sherd_core::fragment::cache;
use sherd_core::memory::Budget;
use sherd_core::objects::ObjectParams;
use sherd_core::tiers::Thresholds;
use sherd_core::{
    ALGO_REF, Backend, CACHE_VERSION, CORE_VERSION, GIT_COMMIT, Params, collection, pipeline,
};
use sherd_parity::FixtureDir;
use sherd_parity::report::{Mode, StageReport};
use sherd_parity::stages::{Collection, Stage};

/// Fracture-surface reassembly of 3D-scanned ceramic fragments.
#[derive(Debug, Parser)]
#[command(name = "sherd-refit-rs", version, about, long_about = None)]
struct Cli {
    /// Log level: repeat for more (`-v` info, `-vv` debug, `-vvv` trace).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
#[allow(
    clippy::large_enum_variant,
    reason = "one of these is parsed once, at startup, and then read; `run` carries R §1.4's \
              twenty-one parameter flags and roadmap item 3's eleven, and clap's derive cannot \
              box a variant's `Args` type"
)]
enum Command {
    /// Assemble a collection of fragments (R §2–§11).
    Run(RunArgs),

    /// Preprocess every fragment and write the fragment cache (R §3.1–3.7).
    ///
    /// Reads every mesh of INPUT in the reference's collection order, cleans it, keeps the largest
    /// component, measures the wall thickness, decimates to the adaptive face budget, smooths,
    /// labels every face shell or fracture, traces the breakline and its frames, draws the
    /// surface, fracture and shell-margin samples, and writes `<OUT>/cache/<name>.sherd`. A second
    /// run over the same files reuses those caches.
    ///
    /// Writes `<OUT>/preview_segmentation.png` as well, which is the tail of the reference's
    /// own `segment_only`: every fragment drawn at the identity with its fracture faces in red.
    Segment(SegmentArgs),

    /// Run the port's stages against a Python fixture dump and report D §10.2's tolerances.
    Parity(ParityArgs),

    /// Time the pipeline against the gates of D §10.3.
    Bench(BenchArgs),

    /// Feed identical batches to both executors and report where they disagree (D §10.4 layer 3).
    GpuCheck(GpuCheckArgs),

    /// Print what this build is: versions, algorithm reference, backends, and what the GPU on
    /// this machine actually does.
    Info(InfoArgs),
}

/// Arguments of `info`.
///
/// The self-test opens a device and runs D §6.8's four checks, which takes about a second, so
/// `--no-selftest` is there for a scripted `info` that only wants the versions. Everything else
/// this command prints is free.
#[derive(Debug, Args)]
struct InfoArgs {
    /// Which GPU adapter to test, by index or by a substring of its name (D §9).
    #[arg(long, value_name = "NAME|INDEX")]
    gpu_adapter: Option<String>,
    /// Only list the adapters; do not open one or run the self-test.
    #[arg(long)]
    no_selftest: bool,
}

/// Arguments of `gpu-check`: D §10.4 layer 3's cross-check harness.
///
/// The table is read at the **production policy** (audit §A.2.4): the kernel rows are R §5.2's
/// coarse score and R §5.4's two stage-1 breakline rungs, the stage-2 rows read `cpu by policy`
/// because R §5.6's ten candidates never reach the device, the translation row is read at the
/// cloud, and R §6.1's bounded distance and R §6.4's inside test read `delegated` permanently
/// (phase 2c was measured and decided against, D §12). `--force-device` puts every rung on the
/// device instead, which is task W's kernel measurement and not the criterion.
#[derive(Debug, Args)]
struct GpuCheckArgs {
    /// Which stages to compare.
    #[arg(long, value_enum, default_value_t = GpuStage::All)]
    stage: GpuStage,
    /// A collection to form the batches from (R §2's discovery); without it, and without
    /// `--fixture`, only the self-test kernels run.
    #[arg(long, value_name = "DIR")]
    set: Option<PathBuf>,
    /// A parity fixture dump to form the batches from, instead of `--set`.
    #[arg(long, value_name = "DIR")]
    fixture: Option<PathBuf>,
    /// How many of the collection's matchable pairs to sweep, in R §4.1's own pair order.
    ///
    /// One pair is enough to exercise every batch shape and cheap enough to run at a console; the
    /// cross-check of D §10.4 layer 3 wants more, because a tie that flips is rare per pair.
    #[arg(long, default_value_t = 1, value_name = "N")]
    pairs: usize,
    /// Also probe the twelve one-ULP neighbours of each candidate's own initial pose (D §10.2's
    /// `chaotic` row, task C2 §5).
    ///
    /// Every candidate's ladder is re-climbed on the CPU from the twelve initial poses one ULP
    /// from its own; a candidate whose own answer moves further than the row's tolerance under
    /// one of them is excluded from the pose rows and counted. It costs thirteen CPU ladders.
    ///
    /// The **other** half of the same test needs no flag and is always applied: a candidate whose
    /// `f64` control — the same rungs, on the CPU, from the pose the device itself starts from —
    /// is already outside the pose rows has a ladder that amplifies any `f32` input, and a
    /// worst case over it measures the ladder rather than the kernel. So `--chaos` can only
    /// *widen* the excused set: the run without it is the stricter of the two.
    #[arg(long)]
    chaos: bool,
    /// Force every batch to the device instead of applying the executor's own size thresholds.
    ///
    /// **Off by default since the audit's §A.2.4**, which restated D §12's 2b exit criterion at the
    /// *production policy*: what a `--backend gpu` run of this collection actually sends to the
    /// device is R §5.2's coarse score and R §5.4's two stage-1 breakline rungs, and R §5.6's
    /// stage 2 is the CPU's by policy (`sherd_gpu::icp::STAGE2_ON_DEVICE`). Those are the rows the
    /// criterion is stated over, and the ones this command exits on.
    ///
    /// With the flag the table is task W's: every rung forced onto the device, including the ones
    /// no run would ever put there. That is a measurement of the *kernel* — useful, and not the
    /// criterion. A small collection then has rows that compare the CPU with itself, which is why
    /// forcing was the default while phase 2b was measuring the kernels.
    #[arg(long)]
    force_device: bool,
    /// Which GPU adapter to use, by index or by a substring of its name (D §9).
    #[arg(long, value_name = "NAME|INDEX")]
    gpu_adapter: Option<String>,
}

/// `gpu-check --stage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum GpuStage {
    /// R §5.2's coarse score.
    Coarse,
    /// One rung of R §7's ICP.
    Icp,
    /// R §6.1's bounded point-to-surface distance.
    Distance,
    /// R §6.4's inside test.
    Inside,
    /// All four, and the self-test kernels above them.
    All,
}

/// Arguments of `run`: every flag of `sherd_refit/cli.py`, with its name and its default (R §1.4),
/// plus the four the port adds (D §9).
#[derive(Debug, Args)]
#[allow(clippy::struct_excessive_bools, reason = "the reference's own switches, one field each")]
struct RunArgs {
    /// Directory of fragment files (`.ply`, `.obj`, `.stl`, `.off`).
    input: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Working-mesh face budget per fragment.
    #[arg(long, default_value_t = 200_000)]
    target_faces: u32,
    /// Parallel workers; the reference's process count, and here the size of the one rayon pool
    /// (default: one per core minus one, which is `cli.py`'s own default).
    #[arg(long)]
    workers: Option<usize>,
    /// Threads per matching worker. One process here, so this sizes the same pool `--workers`
    /// does and wins when both are given (default: one per core minus one, the same number
    /// `--workers` defaults to and the one `cli.py` resolves an unset flag to).
    #[arg(long)]
    threads: Option<usize>,
    /// Candidates refined with full ICP per pair.
    #[arg(long, default_value_t = Params::default().stage2)]
    candidates: u32,
    /// Hypotheses refined with breakline ICP per pair.
    #[arg(long, default_value_t = Params::default().stage1)]
    stage1: u32,
    /// t, voxel the breakline is thinned to before frames are paired.
    #[arg(long, default_value_t = Params::default().brk_voxel)]
    brk_voxel: f64,
    /// Degrees, tolerance on |dih_A + dih_B - 180| for a hypothesis.
    #[arg(long, default_value_t = Params::default().dihedral_tol)]
    dihedral_tol: f64,
    /// Min tight-contact fraction to accept a join.
    #[arg(long, default_value_t = Params::default().min_tight)]
    min_tight: f64,
    /// k for the max median fracture gap; the pair's limit is max(k t, m res).
    #[arg(long, default_value_t = Params::default().max_gap)]
    max_gap: f64,
    /// Max penetrating surface fraction.
    #[arg(long, default_value_t = Params::default().max_pen)]
    max_pen: f64,
    /// Min seam length (in t).
    #[arg(long, default_value_t = Params::default().min_seam)]
    min_seam: f64,
    /// Skip a pair whose wall thicknesses differ by more than this factor.
    #[arg(long, default_value_t = Params::default().thick_ratio)]
    thick_ratio: f64,
    /// Partners kept per fragment by the partner search (0 disables it).
    #[arg(long, default_value_t = Params::default().screen_top_k)]
    screen_top_k: u32,
    /// Breakline points per fragment used by the partner search.
    #[arg(long, default_value_t = Params::default().screen_points)]
    screen_points: u32,
    /// The partner search is skipped below this many pairs.
    #[arg(long, default_value_t = Params::default().screen_min_pairs)]
    screen_min_pairs: u32,
    /// Partners of each unplaced fragment to match again with a larger budget (0 disables it).
    #[arg(long, default_value_t = Params::default().second_pass_top)]
    second_pass_top: u32,
    /// Hypotheses refined in the second pass.
    #[arg(long, default_value_t = Params::default().second_pass_stage1)]
    second_pass_stage1: u32,
    /// Candidates fully verified in the second pass.
    #[arg(long, default_value_t = Params::default().second_pass_stage2)]
    second_pass_candidates: u32,
    /// Skip stage 2 when the pair's best stage-1 breakline score is below this.
    #[arg(long, default_value_t = Params::default().stage1_floor)]
    stage1_floor: f64,
    /// Skip the fracture-only ICPs and the costly verification below this tight-contact fraction.
    #[arg(long, default_value_t = Params::default().early_reject_tight)]
    early_reject_tight: f64,
    /// Points in the cloud the two coarse stage-2 ICPs run on (0: all).
    #[arg(long, default_value_t = Params::default().reg_points)]
    reg_points: u32,
    /// Shell-margin points kept per fragment for ICP and the continuity test.
    #[arg(long, default_value_t = Params::default().margin_points)]
    margin_points: u32,
    /// Whole-surface samples per fragment (penetration test and shell margin).
    #[arg(long, default_value_t = Params::default().surface_points)]
    surface_points: u32,
    /// Fracture samples per t^2 of fracture area.
    #[arg(long, default_value_t = Params::default().frac_per_t2)]
    frac_density: f64,
    /// Seed of every draw R §10 lists: the three per-fragment samplers, the coarse probe and the
    /// partner search. R §9's cap and R §11.5's previews keep the literal 0 the reference gives
    /// them. A collection matched at another seed places differently on the sets R §13 measures
    /// a spread on, which is the point of the flag.
    #[arg(long, default_value_t = Params::default().seed)]
    seed: u64,
    /// Do not write R §11.5's previews.
    #[arg(long)]
    no_preview: bool,
    /// Skip full-resolution refinement.
    #[arg(long)]
    no_refine: bool,
    /// Do not write placed/merged meshes.
    #[arg(long)]
    no_meshes: bool,
    /// Executor: `auto`, `cpu` or `gpu` (D §6.8).
    #[arg(long, default_value_t = Backend::Auto)]
    backend: Backend,
    /// Which GPU adapter to use, by index or by a substring of its name (D §9).
    #[arg(long, value_name = "NAME|INDEX")]
    gpu_adapter: Option<String>,
    /// Gigabytes the kernels may hold on the device at once (D §1, D §6.3); 0 removes the bound.
    /// A batch that does not fit is answered by the CPU and counted, never dropped.
    #[arg(long, value_name = "GB")]
    gpu_memory: Option<f64>,
    /// Neither read nor write the fragment cache (R §3.7).
    #[arg(long)]
    no_cache: bool,
    /// Recompute every fragment and overwrite its cache, even when the cache is valid.
    #[arg(long)]
    force: bool,
    /// Gigabytes concurrent scans may hold during preprocessing (D §5, D §9); 0 removes the
    /// bound, and the default is half of the machine's physical memory.
    #[arg(long, value_name = "GB")]
    memory_budget: Option<f64>,
    /// Roadmap item 3's confidence tier (audit §D.1): `on` — the default — or `off`.
    ///
    /// With the tier on, every accepted candidate is probed once more — the margin to the pair's
    /// second placement, the twelve one-ULP neighbours of the pose, two independent redraws of
    /// R §3.5's samples, a ±0.5 t push along the seam, and how many independent joins of the
    /// collection agree with the placement — and lands in one of three bands. **R §8 assembles
    /// from the confirmed band and from nothing else**; the probable band is listed in
    /// `report.md` and never placed. The pass costs about half a run again.
    ///
    /// `--tiers off` is the off switch every new behaviour has: no probe, no band, R §8's own
    /// `accepted` gate, and every output byte for byte the bytes it was before the tier existed.
    #[arg(long, default_value_t = Switch::On, value_name = "on|off")]
    tiers: Switch,
    /// Tight contact a confirmed join needs (M1 §3; R §6.5 ships 0.25).
    #[arg(long, default_value_t = Thresholds::default().min_tight)]
    tier_min_tight: f64,
    /// `t`; median fracture gap a confirmed join may not exceed (M1 §3; R §6.5's own limit is
    /// 0.02-0.03 t).
    #[arg(long, default_value_t = Thresholds::default().max_gap_t)]
    tier_max_gap: f64,
    /// `t`; shortest seam a confirmed join must share (M1 §3; R §6.5 ships 3).
    #[arg(long, default_value_t = Thresholds::default().min_seam)]
    tier_min_seam: f64,
    /// Shell-normal agreement across the seam a confirmed join needs (M1 §3; R §6.5 ships 0.8).
    #[arg(long, default_value_t = Thresholds::default().min_cont_n)]
    tier_min_cont_n: f64,
    /// Penetrating surface fraction a confirmed join may not exceed (M1 §3; R §6.5 ships 0.005).
    #[arg(long, default_value_t = Thresholds::default().max_pen)]
    tier_max_pen: f64,
    /// `t`; how far the pose may stay from where it started after a +/-0.5 t push along the seam.
    #[arg(long, default_value_t = Thresholds::default().max_slide_t)]
    tier_max_slide: f64,
    /// Factor a confirmed join must beat the pair's second placement by, when the margin is the
    /// arm that confirms it.
    #[arg(long, default_value_t = Thresholds::default().min_margin)]
    tier_margin: f64,
    /// Independent agreeing joins a confirmed join needs, when the support count is the arm that
    /// confirms it; 0 makes that arm always true, which disables the disjunction.
    #[arg(long, default_value_t = Thresholds::default().min_support)]
    tier_support: u32,
    /// Degrees; worst rotation over the pose's twelve one-ULP neighbours a confirmed join may
    /// show. Off by default: M1 measured the whole range at 1.6e-14 to 4.1e-7 degrees, so there is
    /// no threshold in it and the number is reported instead.
    #[arg(long, value_name = "DEG")]
    tier_max_determined_deg: Option<f64>,
    /// How many of the three sample draws must accept a confirmed join. Off by default: requiring
    /// all three costs six of M1's 136 confirmed joins and removes no false one.
    #[arg(long, value_name = "N")]
    tier_resample_accept: Option<u32>,
    /// Write roadmap step 7's tier measurement of every accepted candidate to FILE
    /// (audit §D.1, `sherd_core::measure`).
    ///
    /// One extra pass after R §8's assembly — the margin to the pair's second placement, the
    /// twelve one-ULP neighbours of the pose, two independent redraws of R §3.5's samples, the
    /// slide along the seam, and the support count — and one extra file. Without the flag the run
    /// is unchanged, file for file and byte for byte.
    #[arg(long, value_name = "FILE")]
    measure: Option<PathBuf>,
    /// Read roadmap item 3's operator constraints from FILE (audit §D.1,
    /// `sherd_core::assembly::constraints`).
    ///
    /// `constraints.json` v1 carries four lists. `must_not_join` takes a pair out of the run
    /// before it is matched and refuses any candidate for it in the assembly; `must_join` matches
    /// the pair with R §8.1's larger budget and promotes its best probable candidate to confirmed,
    /// offering it to the assembly before every other join — or, with a 4x4 `pose`, skips matching
    /// and places at that pose; `same_object` and `different_object` are roadmap item 4's
    /// evidence, and `different_object` already vetoes a join. Every name is checked against the
    /// collection and an unknown one fails the run.
    ///
    /// A constraint never edits a score. Without the flag the run is unchanged, byte for byte.
    #[arg(long, value_name = "FILE")]
    constraints: Option<PathBuf>,
    /// Roadmap item 4's object separation (audit §D.2): `on` — the default — or `off`.
    ///
    /// With it on, every assembled group is reported as an object with the consensus its members
    /// agree on — median and MAD of the wall, the shell radius, the fracture roughness and the rim
    /// diameter — and the members that consensus does not fit; two confirmed joins into one group
    /// that disagree about a fragment are both demoted to probable; and a confirmed join between
    /// two groups may merge them, under the penetration test across both groups and consistency
    /// with every cross-group join the assembly's gate admits.
    ///
    /// **No feature vetoes anything on the shipped settings.** Audit §D.2's own rule is that a
    /// feature may veto only where its measured AUC exceeds 0.800, and task M1 §4 measured the
    /// best of them at 0.740 on the collections with real object ids, so `--object-demote` ships
    /// empty and every number is reported instead.
    ///
    /// `--objects off` is the off switch: no consensus, no demotion, no merge, no `## Objects`
    /// section, and every output byte for byte the bytes it was.
    #[arg(long, default_value_t = Switch::On, value_name = "on|off")]
    objects: Switch,
    /// Features whose deviation from an object's consensus demotes a join to probable, comma
    /// separated (`thick`, `thick_mode`, `shell_radius`, `frac_rough`, `axis_diameter`,
    /// `rim_diameter`, `lab_L`, `lab_a`, `lab_b`).
    ///
    /// **Empty by default, on the measurement**: M1 §4 found no feature reaching audit §D.2's own
    /// AUC of 0.800 on any collection with real object ids. Turning one on without a table that
    /// justifies it is exactly what the audit's quality principles forbid.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    object_demote: Vec<String>,
    /// How many MADs from its object's median a fragment may sit before the consensus rejects it.
    #[arg(long, default_value_t = ObjectParams::default().k_mad, value_name = "K")]
    object_k_mad: f64,
    /// Fewest members an object needs before its consensus may reject one of them.
    #[arg(long, default_value_t = ObjectParams::default().min_members, value_name = "N")]
    object_min_members: usize,
    /// Audit §D.2 (b): two confirmed joins into one group that disagree about a fragment are both
    /// demoted to probable.
    ///
    /// **Off by default, on the measurement.** Task T1 measured 0 false joins in the confirmed
    /// tier over the eight development sets at seeds 0-4, so a contradiction has no false join to
    /// catch here; switching the arm on takes mixed_ABG seed 0 from 11 confirmed joins to 2, every
    /// one of them a correct join R §8 had already reconciled on its own.
    #[arg(long, default_value_t = Switch::Off, value_name = "on|off")]
    object_disagreement: Switch,
    /// Audit §D.2 (c): a confirmed join between two groups merges them. `off` restores R §8's own
    /// refusal, "would merge two groups (not supported)".
    #[arg(long, default_value_t = Switch::On, value_name = "on|off")]
    object_merge: Switch,
    /// Write audit §D.1's review images to `<OUT>/review/<a>__<b>.png`
    /// (`sherd_core::review`).
    ///
    /// One PNG per confirmed or probable join: three views of the two fragments at the candidate's
    /// pose, A grey and B orange, the seam's `t/3` voxels white, B's fracture samples coloured by
    /// their distance to A's fracture surface, and a caption with the scores, the band, the margin
    /// and the seed. `report.md`'s per-fragment index links them. Deterministic: two runs of one
    /// collection write the same bytes.
    #[arg(long)]
    review_images: bool,
}

impl RunArgs {
    /// R §1.4's CLI-to-`Params` mapping: `--candidates → stage2`, `--frac-density → frac_per_t2`,
    /// `--second-pass-candidates → second_pass_stage2`, and every other flag to the field of the
    /// same name. Everything R §1.1 has no flag for keeps its default.
    fn params(&self) -> Params {
        Params {
            stage1: self.stage1,
            stage2: self.candidates,
            brk_voxel: self.brk_voxel,
            dihedral_tol: self.dihedral_tol,
            min_tight: self.min_tight,
            max_gap: self.max_gap,
            max_pen: self.max_pen,
            min_seam: self.min_seam,
            thick_ratio: self.thick_ratio,
            early_reject_tight: self.early_reject_tight,
            stage1_floor: self.stage1_floor,
            second_pass_top: self.second_pass_top,
            second_pass_stage1: self.second_pass_stage1,
            second_pass_stage2: self.second_pass_candidates,
            screen_top_k: self.screen_top_k,
            screen_points: self.screen_points,
            screen_min_pairs: self.screen_min_pairs,
            margin_points: self.margin_points,
            reg_points: self.reg_points,
            surface_points: self.surface_points,
            frac_per_t2: self.frac_density,
            seed: self.seed,
            tiers: match self.tiers {
                Switch::Off => None,
                Switch::On => Some(Thresholds {
                    min_tight: self.tier_min_tight,
                    max_gap_t: self.tier_max_gap,
                    min_seam: self.tier_min_seam,
                    min_cont_n: self.tier_min_cont_n,
                    max_pen: self.tier_max_pen,
                    max_slide_t: self.tier_max_slide,
                    min_margin: self.tier_margin,
                    min_support: self.tier_support,
                    max_determined_deg: self.tier_max_determined_deg,
                    min_resample_accept: self.tier_resample_accept,
                }),
            },
            objects: match self.objects {
                Switch::Off => None,
                Switch::On => Some(ObjectParams {
                    demote: self.demote_set(),
                    k_mad: self.object_k_mad,
                    min_members: self.object_min_members,
                    disagreement: self.object_disagreement == Switch::On,
                    merge: self.object_merge == Switch::On,
                }),
            },
            ..Params::default()
        }
    }

    /// `--object-demote`'s list, with an unknown name refused rather than ignored.
    ///
    /// A typo that quietly became "no feature" is the same failure `constraints.json`'s name
    /// validation exists to prevent, and here it would silently turn a demotion rule off.
    fn demote_set(&self) -> sherd_core::objects::FeatureSet {
        self.object_demote
            .iter()
            .map(|name| {
                sherd_core::objects::FeatureKey::parse(name).unwrap_or_else(|| {
                    let known: Vec<&str> =
                        sherd_core::objects::FeatureKey::ALL.iter().map(|k| k.key()).collect();
                    clap::Error::raw(
                        clap::error::ErrorKind::InvalidValue,
                        format!(
                            "--object-demote: `{name}` is not a feature; the nine are {}\n",
                            known.join(", ")
                        ),
                    )
                    .exit()
                })
            })
            .collect()
    }
}

/// A flag that is on or off by name, so that `--tiers off` reads the way the brief writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Switch {
    /// The behaviour is computed.
    On,
    /// The behaviour is skipped, and the run is the run it was without it.
    Off,
}

impl std::fmt::Display for Switch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::On => "on",
            Self::Off => "off",
        })
    }
}

/// Arguments of `segment`.
#[derive(Debug, Args)]
struct SegmentArgs {
    /// Directory of fragment files (`.ply`, `.obj`, `.stl`, `.off`).
    input: PathBuf,
    /// Output directory; the caches go to `<OUT>/cache`.
    #[arg(long)]
    out: PathBuf,
    /// Working-mesh face budget per fragment (R §3.3 caps its adaptive budget with this).
    #[arg(long, default_value_t = 200_000)]
    target_faces: u32,
    /// Parallel workers, as the reference's `segment` takes them (default: one per core minus
    /// one).
    #[arg(long)]
    workers: Option<usize>,
    /// Worker threads. One process here, so this sizes the same pool `--workers` does and wins
    /// when both are given (default: one per core minus one; an explicit 0 means one per core).
    #[arg(long)]
    threads: Option<usize>,
    /// Recompute every fragment and overwrite its cache, even when the cache is valid.
    #[arg(long)]
    force: bool,
    /// Neither read nor write the fragment cache.
    #[arg(long)]
    no_cache: bool,
    /// Gigabytes concurrent scans may hold (D §5, D §9); 0 removes the bound, and the default is
    /// half of the machine's physical memory.
    #[arg(long, value_name = "GB")]
    memory_budget: Option<f64>,
    /// Write audit §D.2's per-fragment object features to FILE
    /// (`sherd_core::fragment::features`).
    ///
    /// Preprocessing only — the wall, the outer-shell sphere, the fracture roughness, the axis of
    /// rotation and its residual, the rim flag and its diameter — and this is the one pass the
    /// audit's plan lets run on a collection above 27 fragments, because it is linear in the
    /// fragment count and matches nothing.
    #[arg(long, value_name = "FILE")]
    features: Option<PathBuf>,
    /// Also read each source file again for its vertex colours (Lab mean and spread).
    ///
    /// Off by default because the working mesh has no colours — decimation drops them — so the
    /// only way to the fabric is a second full read of every scan, which is the most expensive
    /// thing `--features` can do and is worth nothing on a file with bare `v x y z` lines.
    #[arg(long, requires = "features")]
    features_colour: bool,
}

/// D §9's `--memory-budget GB`: the flag when it is given, half of physical memory when it is not.
///
/// `0` (or a negative number) is the escape hatch that removes the bound entirely, which is what
/// a machine with a known-good amount of memory and a very large scan wants.
fn budget(gb: Option<f64>) -> Budget {
    gb.map_or_else(Budget::default_for_machine, Budget::gigabytes)
}

/// The size of the one `rayon` pool (D §5), from `--threads` and `--workers` as every subcommand
/// resolves them.
///
/// `--threads` is the port's own flag and wins; `--workers` is the reference's process count and
/// sizes the pool when it is the only one given; an unset pair is `cli.py`'s own
/// `max(1, cpu_count() - 1)`. `run`, `segment` and `bench` share this function rather than three
/// copies of the same two lines, because two of the three copies had already drifted apart
/// (V4-D8 on `run` and `segment`, V5-D3 on `bench`).
fn pool_threads(threads: Option<usize>, workers: Option<usize>) -> usize {
    threads.or(workers).unwrap_or_else(pipeline::default_workers)
}

/// R §4.2's block schedule, which reads `--workers` alone: `block_size(workers, n_pairs)` decides
/// whether the pairs are walked one at a time or in 3×3 blocks, so an unset flag has to resolve to
/// the reference's own number and not to the machine's core count.
fn schedule_workers(workers: Option<usize>) -> usize {
    workers.unwrap_or_else(pipeline::default_workers)
}

/// Arguments of `parity`.
#[derive(Debug, Args)]
struct ParityArgs {
    /// A fixture dump written by `tools/dump_fixtures.py` (D §10.1).
    #[arg(long)]
    fixtures: PathBuf,
    /// The collection the dump was made from; needed by native mode and by any stage the dump
    /// itself does not carry (levels `slim` and `min`, D §10.1).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Stage to compare: `load`, `thickness`, `working-mesh`, `segmentation`, `breakline`,
    /// `samples`, `hypotheses`, `coarse`, `nms`, `stage1`, `stage2`, `verify`, `candidates`,
    /// `assembly`, `refine`, `outputs`, or `all`. Repeatable.
    #[arg(long, default_value = "all")]
    stage: Vec<String>,
    /// Feed each stage the Python stage's own inputs instead of the port's upstream results
    /// (D §10.2's injected column). Without it the stages run natively.
    #[arg(long)]
    injected: bool,
    /// Print every comparison, not only the per-stage summary.
    #[arg(long)]
    details: bool,
    /// Re-hash every file of the dump and compare against the manifest.
    #[arg(long)]
    verify_checksums: bool,
}

/// Arguments of `bench`: a run with the previews and the meshes off, timed against D §10.3.
///
/// `--workers` and `--threads` mean here exactly what they mean on `run`, and are resolved by the
/// same two lines. They used not to be (V5-D3): `bench` passed `threads.unwrap_or(0)` to the pool
/// and `workers: 0` to the pipeline, so an unset flag gave it ten threads and a block schedule
/// computed at ten where `run` uses nine. No result of the eight development sets moves —
/// `block_size(9, n) == block_size(10, n)` on every one of their pair counts — but the tool D §10.3
/// names for its gates has to be the tool that produced its numbers.
#[derive(Debug, Args)]
struct BenchArgs {
    /// Directory of fragment files to time the pipeline on.
    input: PathBuf,
    /// Where the run's outputs go; `report.json` carries the same timings.
    #[arg(long)]
    out: PathBuf,
    /// Working-mesh face budget per fragment.
    #[arg(long, default_value_t = 200_000)]
    target_faces: u32,
    /// Parallel workers, as on `run`: the reference's process count, which also fixes R §4.2's
    /// block schedule (default: one per core minus one, `cli.py`'s own default).
    #[arg(long)]
    workers: Option<usize>,
    /// Threads per matching worker, as on `run`: one process here, so this sizes the same pool
    /// `--workers` does and wins when both are given (default: one per core minus one).
    #[arg(long)]
    threads: Option<usize>,
    /// D §10.3's wall-clock gate for this set, in seconds; without it the timings are only
    /// reported.
    #[arg(long)]
    gate: Option<f64>,
    /// Also write R §11.4's meshes, which D §10.3's gates do not include.
    #[arg(long)]
    meshes: bool,
    /// Neither read nor write the fragment cache; D §10.3's gates are stated for a warm one.
    #[arg(long)]
    no_cache: bool,
    /// Executor: `auto`, `cpu` or `gpu` (D §6.8). D §10.3's GPU gates are phase 2d's, and this is
    /// the flag they will be measured with; `gpu` fails rather than falling back, so a timing
    /// cannot be attributed to the wrong backend.
    #[arg(long, default_value_t = Backend::Cpu)]
    backend: Backend,
    /// Which GPU adapter to use, by index or by a substring of its name (D §9).
    #[arg(long, value_name = "NAME|INDEX")]
    gpu_adapter: Option<String>,
    /// Gigabytes the kernels may hold on the device at once (D §1); 0 removes the bound.
    #[arg(long, value_name = "GB")]
    gpu_memory: Option<f64>,
    /// Seed of every draw R §10 lists, as on `run` — a timing at another seed is a timing of
    /// another set of sampled arrays, and D §10.3's gates are stated at 0.
    #[arg(long, default_value_t = Params::default().seed)]
    seed: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    match cli.command {
        Command::Run(args) => run(&args),
        Command::Segment(args) => segment(&args),
        Command::Parity(args) => parity(&args),
        Command::Bench(args) => bench(&args),
        Command::GpuCheck(args) => gpu_check(&args),
        Command::Info(args) => {
            info(&args);
            Ok(())
        }
    }
}

fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(format!("sherd={level}")));
    tracing_subscriber::fmt().with_env_filter(filter).with_target(false).init();
}

/// Prints what this build is; the same three strings go into `report.json`'s `engine` key.
///
/// Since task G4 it also answers the question an operator actually has in front of a new machine —
/// *is the GPU worth asking for here?* — by opening the adapter, running D §6.8's self-test and
/// printing **both** ratios with their names on them: the kernel's, on an idle device, and the
/// matching stage's, which is the one `--backend auto` is allowed to read.
fn info(args: &InfoArgs) {
    println!("sherd-refit-rs {CORE_VERSION}");
    println!("  algorithm reference: {ALGO_REF}");
    println!("  cache version:       {CACHE_VERSION}");
    // The same four fields every report's `engine` block carries (D §4.3), so that a file and the
    // binary that wrote it can be matched up without reading JSON.
    println!("  git commit:          {GIT_COMMIT}");
    let mut backends = gpu::info_lines().into_iter();
    println!("  backends:            {}", backends.next().unwrap_or_default());
    for line in backends {
        println!("                       {line}");
    }
    let threads = std::thread::available_parallelism().map_or(0, std::num::NonZero::get);
    println!("  cores available:     {threads}");
    println!("  default seed:        {}", Params::default().seed);
    // D §7's "two processes on one adapter" row, in one sentence, where an operator will meet it.
    // Task H1 measured the mechanism and the port now detects it (`--backend gpu` refuses a
    // readback the device did not write and answers that batch on the CPU), but a refused batch is
    // a batch the two backends answered differently, so the rule still stands for reproducibility.
    for line in [
        "  gpu, two processes:  a `--backend gpu` run wants the adapter to itself. macOS aborts",
        "                       command buffers when two processes drive one GPU, wgpu reports",
        "                       neither the abort nor the fence it resolves as success, and the",
        "                       port answers such a batch on the CPU and counts it `corrupt` in",
        "                       the run's device lines (task H1, D §7).",
        "  gpu, when to use it: `--backend gpu` is an opt-in — for measuring the kernels, and for",
        "                       a discrete adapter. `auto` is the CPU by policy until the matching",
        "                       stage has been measured on one: on an integrated part the device",
        "                       and the cores share one envelope and the stage is 1.07-1.43x,",
        "                       under D §6.8's 1.5x (audit §A.2.4, D §6.8).",
    ] {
        println!("{line}");
    }
    if args.no_selftest {
        println!("  gpu self-test:       skipped (--no-selftest)");
        return;
    }
    let mut lines = gpu::selftest_lines(args.gpu_adapter.as_deref()).into_iter();
    println!("  gpu self-test:       {}", lines.next().unwrap_or_default());
    for line in lines {
        println!("                     {line}");
    }
}

/// R §3.1–3.5 for a whole collection, with the cache of R §3.7 (plan steps S4, B1, B2 and B3).
#[allow(clippy::cast_precision_loss, reason = "counts printed in a table")]
fn segment(args: &SegmentArgs) -> Result<()> {
    let threads = pool_threads(args.threads, args.workers);
    if let Err(e) = pipeline::set_threads(threads) {
        bail!("--threads {threads}: {e}");
    }
    let entries = collection::discover(&args.input)
        .with_context(|| format!("scanning {}", args.input.display()))?;
    if entries.is_empty() {
        bail!("{}: no .ply, .obj, .stl or .off file", args.input.display());
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    // `--force` recomputes by writing to a cache directory it first empties of the names it is
    // about to write; `--no-cache` neither reads nor writes.
    if args.force && !args.no_cache {
        for entry in &entries {
            let path = cache::cache_path(&args.out, &entry.name);
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("removing {}", path.display()))?;
            }
        }
    }
    let out = if args.no_cache { None } else { Some(args.out.as_path()) };

    let started = std::time::Instant::now();
    // `segment` has no `--seed`: R §10's seed is a *matching* run's, and a cache written at one
    // seed simply has R §3.5's three arrays recomputed by the run that wants another (R §3.7).
    let results = pipeline::preprocess(
        &entries,
        args.target_faces as usize,
        out,
        budget(args.memory_budget),
        Params::default().seed,
    );
    let wall = started.elapsed().as_secs_f64();

    println!(
        "{:<28} {:>9} {:>10} {:>9} {:>8} {:>7} {:>8} {:>7} {:>6} {:>7} {:>7} {:>6} {:>6}",
        "fragment",
        "faces",
        "from",
        "t",
        "res",
        "t/res",
        "fracture",
        "brk",
        "sub",
        "frac pts",
        "margin",
        "closed",
        "cache"
    );
    let mut failed = 0;
    let mut total = 0.0;
    for (entry, result) in entries.iter().zip(&results) {
        match result {
            Ok(p) => {
                let fr = &p.fragment;
                total += p.seconds;
                println!(
                    "{:<28} {:>9} {:>10} {:>9.3} {:>8.3} {:>7.1} {:>8.3} {:>7} {:>6} {:>7} {:>7} \
                     {:>6} {:>6}",
                    fr.name,
                    fr.n_faces(),
                    fr.n_orig_faces,
                    fr.thick,
                    fr.res(),
                    fr.thick / fr.res().max(1e-9),
                    fr.fracture_fraction(),
                    fr.brk.len(),
                    fr.brk.sub.len(),
                    fr.samples.n_fracture(),
                    fr.samples.n_margin(),
                    fr.watertight,
                    if p.cached { "hit" } else { "miss" }
                );
            }
            Err(e) => {
                failed += 1;
                println!("{:<28} {e}", entry.name);
            }
        }
    }
    println!(
        "{} fragments, {failed} failed, {wall:.2} s wall ({total:.2} s of work)",
        entries.len()
    );
    if !args.no_cache {
        println!("caches in {}", args.out.join("cache").display());
    }
    println!(
        "R §3.1-3.7 for every fragment: the working mesh, the labels, the breaklines and the \
         match arrays. Matching (R §4-§6) is phase 1c."
    );
    if failed > 0 {
        bail!("{failed} of {} fragments could not be preprocessed", entries.len());
    }
    // R's `segment_only` ends here: the caches, then the segmentation preview (V4-D7).
    let fragments: Vec<sherd_core::fragment::Fragment> =
        results.into_iter().map(|r| r.expect("no fragment failed").fragment).collect();
    if let Some(path) = &args.features {
        write_features(&fragments, path, args.features_colour)?;
    }
    for file in pipeline::write_segmentation_preview(&args.out, &fragments)? {
        println!("{}", file.display());
    }
    Ok(())
}

/// Audit §D.2's per-fragment feature table, as `segment --features FILE` writes it.
///
/// The colours are optional and separate because they are the only part that reads a file again:
/// R §3.3's decimation drops the vertex colours, so a Lab mean has to come from the source scan.
/// `mixed_all` and `synthetic_170` are 164 scans each, which is why this is a flag rather than
/// the default.
fn write_features(
    fragments: &[sherd_core::fragment::Fragment],
    path: &std::path::Path,
    colour: bool,
) -> Result<()> {
    use sherd_core::fragment::features;

    let rows = features::table(fragments, colour);
    let coloured = rows.iter().filter(|f| f.lab_mean.is_some()).count();
    features::write_table(&rows, path).with_context(|| format!("writing {}", path.display()))?;
    println!(
        "{} feature rows in {} ({} with vertex colours)",
        rows.len(),
        path.display(),
        coloured
    );
    Ok(())
}

/// D §5's Ctrl-C: a flag the pipeline checks between units of work.
///
/// The handler does one thing — raise the flag — and returns; the run then stops at its next
/// fragment or its next pair, having finished the one it was on. That is what makes it safe on the
/// GPU path as well as on the CPU: batches already on the device run to completion (a dispatch is
/// bounded by D §6.4's chunking), the submitting thread retires every command buffer it submitted,
/// and nothing is left half-written or half-mapped.
///
/// A **second** Ctrl-C is the operator saying they meant it, and it aborts. The first one can take
/// as long as the pair in flight — up to a couple of seconds on a large collection — and an
/// operator who has waited through that is entitled to a harder stop.
fn watch_signals() -> sherd_core::progress::Watch {
    let cancel = sherd_core::progress::Cancel::new();
    let flag = cancel.clone();
    let handler = ctrlc::set_handler(move || {
        if flag.is_cancelled() {
            eprintln!("interrupted again: stopping now");
            std::process::exit(130);
        }
        eprintln!(
            "interrupted: finishing the units in flight, then stopping (Ctrl-C again to abort)"
        );
        flag.cancel();
    });
    if let Err(e) = handler {
        // A handler that could not be installed is worth a line and not a failure: the run is
        // still correct, it just cannot be stopped politely.
        tracing::warn!(error = %e, "could not install the Ctrl-C handler; the run cannot be cancelled");
    }
    sherd_core::progress::Watch::cancelled_by(cancel)
}

/// R §2–§11 for a whole collection: the reference's `sherd-refit run`, flag for flag.
#[allow(clippy::cast_precision_loss, reason = "counts and seconds printed in a table")]
fn run(args: &RunArgs) -> Result<()> {
    // The pool is sized *before* the backend is resolved: D §6.8's self-test times the same batch
    // on the CPU, over rayon, and a rayon call initialises the global pool at its default size —
    // after which `set_threads` can only fail. The ratio the self-test reports is then measured
    // against the pool the run will actually use, which is the ratio `Backend::Auto` wants.
    let threads = pool_threads(args.threads, args.workers);
    if let Err(e) = pipeline::set_threads(threads) {
        bail!("--threads {threads}: {e}");
    }
    let resolved = gpu::resolve(args.backend, args.gpu_adapter.as_deref(), args.gpu_memory)?;
    tracing::info!(backend = %resolved.backend, "{}", resolved.reason);
    let options = pipeline::RunOptions {
        target_faces: args.target_faces as usize,
        params: args.params(),
        keep_per_pair: pipeline::KEEP_PER_PAIR,
        preview: !args.no_preview,
        refine: !args.no_refine,
        write_meshes: !args.no_meshes,
        cache: !args.no_cache,
        workers: schedule_workers(args.workers),
        backend: resolved.backend,
        adapter: resolved.adapter.clone(),
        memory: budget(args.memory_budget),
        watch: watch_signals(),
        measure: args.measure.clone(),
        constraints: match &args.constraints {
            Some(path) => Some(sherd_core::assembly::constraints::load(path)?),
            None => None,
        },
        review_images: args.review_images,
    };
    if args.force && !args.no_cache {
        clear_caches(&args.input, &args.out)?;
    }
    let started = std::time::Instant::now();
    if args.backend == Backend::Gpu {
        println!("{}", resolved.reason);
    }
    let summary = pipeline::run_with(&args.input, &args.out, &options, resolved.engine)
        .with_context(|| format!("assembling {}", args.input.display()))?;
    let wall = started.elapsed().as_secs_f64();

    println!(
        "{} fragments, {} pairs ({} skipped by --thick-ratio{}), {} candidates, {} accepted",
        summary.names.len(),
        summary.pairs,
        summary.skipped_pairs,
        match summary.screened {
            Some((seen, kept)) => format!(", {seen} screened to {kept}"),
            None => String::new(),
        },
        summary.candidates.len(),
        summary.accepted()
    );
    for (k, group) in summary.groups.iter().enumerate() {
        if group.len() > 1 {
            let members: Vec<&str> =
                group.iter().map(|&n| summary.names[n as usize].as_str()).collect();
            println!("  group {k}: {}", members.join(", "));
        }
    }
    let alone: Vec<&str> = summary
        .groups
        .iter()
        .filter(|g| g.len() == 1)
        .map(|g| summary.names[g[0] as usize].as_str())
        .collect();
    if !alone.is_empty() {
        println!("  not assembled: {}", alone.join(", "));
    }
    print_timings(&summary.timings, summary.memory.as_ref(), wall);
    for line in resolved.device_lines() {
        println!("  {line}");
    }
    println!("{} files in {}", summary.written.len(), args.out.display());
    Ok(())
}

/// D §10.3's timing gate: a run with the previews and the meshes off, timed stage by stage.
fn bench(args: &BenchArgs) -> Result<()> {
    // The two functions `run` uses, so that the timed run is the run (V5-D3).
    let threads = pool_threads(args.threads, args.workers);
    if let Err(e) = pipeline::set_threads(threads) {
        bail!("--threads {threads}: {e}");
    }
    // As on `run`, and for the same reason: the self-test times a rayon batch, and a rayon call
    // initialises the global pool at its default size.
    let resolved = gpu::resolve(args.backend, args.gpu_adapter.as_deref(), args.gpu_memory)?;
    tracing::info!(backend = %resolved.backend, "{}", resolved.reason);
    let options = pipeline::RunOptions {
        target_faces: args.target_faces as usize,
        params: Params { seed: args.seed, ..Params::default() },
        preview: false,
        write_meshes: args.meshes,
        cache: !args.no_cache,
        workers: schedule_workers(args.workers),
        backend: resolved.backend,
        adapter: resolved.adapter.clone(),
        ..pipeline::RunOptions::default()
    };
    let started = std::time::Instant::now();
    let summary = pipeline::run_with(&args.input, &args.out, &options, resolved.engine)
        .with_context(|| format!("timing {}", args.input.display()))?;
    let wall = started.elapsed().as_secs_f64();
    println!(
        "{}: {} fragments, {} pairs, {} accepted, {} assembled group(s)",
        args.input.display(),
        summary.names.len(),
        summary.pairs,
        summary.accepted(),
        summary.assembled().count()
    );
    print_timings(&summary.timings, summary.memory.as_ref(), wall);
    for line in resolved.device_lines() {
        println!("  {line}");
    }
    match args.gate {
        Some(gate) if wall > gate => {
            bail!("{wall:.1} s wall is over the gate of {gate:.1} s")
        }
        Some(gate) => println!("within the gate of {gate:.1} s"),
        None => {}
    }
    Ok(())
}

/// The per-stage table both `run` and `bench` print; the same numbers `report.json` carries.
///
/// Since task H3 the peak resident set of each stage stands beside its seconds (audit §B.3). It is
/// a **sampled** number — `report.json`'s `memory` block is the same one — so it is printed to the
/// megabyte and never to more precision than a 100 ms sampler can claim. A platform whose resident
/// set cannot be read prints the seconds alone.
fn print_timings(
    timings: &sherd_core::report::Timings,
    memory: Option<&sherd_core::report::MemoryReport>,
    wall: f64,
) {
    const MIB: u64 = 1024 * 1024;
    for (stage, seconds) in timings {
        match memory.and_then(|m| m.stages.get(stage)) {
            Some(peak) => println!("  {stage:<12} {seconds:>8.2} s {:>7} MiB", peak / MIB),
            None => println!("  {stage:<12} {seconds:>8.2} s"),
        }
    }
    match memory {
        Some(m) => println!("  {:<12} {wall:>8.2} s {:>7} MiB peak", "wall", m.peak_rss / MIB),
        None => println!("  {:<12} {wall:>8.2} s", "wall"),
    }
}

/// `gpu-check`: D §10.4 layer 3's cross-check table (D §6.8's kernels, then the four batches).
fn gpu_check(args: &GpuCheckArgs) -> Result<()> {
    let stage = match args.stage {
        GpuStage::Coarse => "coarse",
        GpuStage::Icp => "icp",
        GpuStage::Distance => "distance",
        GpuStage::Inside => "inside",
        GpuStage::All => "all",
    };
    let (lines, failed) = gpu::check(
        stage,
        args.set.as_deref(),
        args.fixture.as_deref(),
        args.gpu_adapter.as_deref(),
        args.pairs.max(1),
        args.chaos,
        args.force_device,
    )?;
    for line in &lines {
        println!("{line}");
    }
    // One line for the header, so the row count is the table's own.
    let rows = lines.len().saturating_sub(1);
    println!("\n{rows} row(s), {failed} failed");
    if failed > 0 {
        bail!("{failed} of {rows} gpu-check rows failed");
    }
    Ok(())
}

/// `--force`: removes the cache files of the collection about to be run, so every fragment is
/// recomputed and rewritten.
fn clear_caches(input: &std::path::Path, out: &std::path::Path) -> Result<()> {
    let entries =
        collection::discover(input).with_context(|| format!("scanning {}", input.display()))?;
    for entry in &entries {
        let path = cache::cache_path(out, &entry.name);
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
    }
    Ok(())
}

/// Reads a fixture dump, runs the requested stages against it and prints D §10.2's table.
fn parity(args: &ParityArgs) -> Result<()> {
    let dir = FixtureDir::new(&args.fixtures);
    let collection = Collection::open(dir, args.input.as_deref())
        .with_context(|| format!("reading the fixture in {}", args.fixtures.display()))?;
    let manifest = &collection.manifest;

    println!("fixture:    {}", args.fixtures.display());
    println!("  commit:   {}{}", manifest.commit, if manifest.dirty { " (dirty)" } else { "" });
    println!("  level:    {}", manifest.level);
    println!("  open3d:   {}, numpy {}", manifest.open3d, manifest.numpy);
    println!("  files:    {} in {} fragments", manifest.files.len(), manifest.pairs.names.len());
    println!("  pairs:    {}", manifest.pairs.pairs.len());
    println!("  faces:    --target-faces {}", collection.target_faces);
    println!(
        "  params:   {}",
        if manifest.uses_default_params() {
            "the defaults of R §1.1".to_owned()
        } else {
            "not the defaults — the comparison must use the dump's own values".to_owned()
        }
    );
    match &args.input {
        Some(input) => println!("  input:    {}", input.display()),
        None => println!("  input:    none given (native mode will skip)"),
    }
    println!(
        "  icp:      {:?} point loops, {:?} assembly (the reference's, which is what D §10.2's \
         rows are stated for)",
        collection.icp.precision, collection.icp.assembly,
    );

    if args.verify_checksums {
        let bad = collection.dir.verify_checksums().context("verifying the fixture's checksums")?;
        if bad.is_empty() {
            println!("  checksums: all {} files match the manifest", manifest.files.len());
        } else {
            for path in &bad {
                println!("  MISMATCH: {path}");
            }
            bail!("{} of {} files do not match the manifest", bad.len(), manifest.files.len());
        }
    }

    let stages = requested_stages(&args.stage)?;
    let mode = if args.injected { Mode::Injected } else { Mode::Native };
    let reports = collection.run_all(&stages, mode).context("running the stages")?;

    println!();
    println!("{}", StageReport::summary_header());
    for report in &reports {
        println!("{}", report.summary_line());
    }

    if args.details {
        for report in &reports {
            println!();
            println!("--- {} ({}) ---", report.stage, report.mode);
            println!("{}", StageReport::detail_header());
            for check in &report.checks {
                println!("{}", check.line());
            }
        }
    }

    let skipped: usize = reports.iter().map(|r| r.skips.len()).sum();
    if skipped > 0 {
        println!();
        for report in &reports {
            for skip in &report.skips {
                println!("skipped {} in {}: {}", skip.scope, report.stage, skip.reason);
            }
        }
    }

    let failures: Vec<&sherd_parity::Check> =
        reports.iter().flat_map(StageReport::failures).collect();
    if !failures.is_empty() {
        println!();
        println!("{}", StageReport::detail_header());
        for check in &failures {
            println!("{}", check.line());
        }
        let total: usize = reports.iter().map(|r| r.checks.len()).sum();
        bail!("{} of {total} comparisons outside their tolerance", failures.len());
    }
    Ok(())
}

/// `--stage` values, in pipeline order and without duplicates; `all` is every stage this build
/// can run.
fn requested_stages(requested: &[String]) -> Result<Vec<Stage>> {
    let mut wanted = Vec::new();
    for name in requested {
        if name == "all" {
            wanted.extend(Stage::ALL);
            continue;
        }
        let stage = Stage::parse(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown stage `{name}`; this build compares {} or `all`",
                Stage::ALL.map(Stage::as_str).join(", ")
            )
        })?;
        wanted.push(stage);
    }
    let ordered: Vec<Stage> = Stage::ALL.into_iter().filter(|s| wanted.contains(s)).collect();
    if ordered.is_empty() {
        bail!("no stage requested");
    }
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::{
        Backend, Cli, ObjectParams, Params, Thresholds, pipeline, pool_threads, requested_stages,
        schedule_workers,
    };
    use clap::{CommandFactory, Parser};
    use sherd_parity::stages::Stage;

    #[test]
    fn the_command_line_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_binary_is_named_for_the_transition() {
        assert_eq!(Cli::command().get_name(), "sherd-refit-rs");
    }

    #[test]
    fn stages_come_back_in_pipeline_order_without_duplicates() {
        assert_eq!(requested_stages(&["all".to_owned()]).unwrap(), Stage::ALL.to_vec());
        assert_eq!(
            requested_stages(&["working-mesh".to_owned(), "load".to_owned()]).unwrap(),
            vec![Stage::Load, Stage::WorkingMesh]
        );
        assert_eq!(
            requested_stages(&["load".to_owned(), "load".to_owned()]).unwrap(),
            vec![Stage::Load]
        );
        assert_eq!(
            requested_stages(&["segmentation".to_owned()]).unwrap(),
            vec![Stage::Segmentation]
        );
        assert_eq!(requested_stages(&["breakline".to_owned()]).unwrap(), vec![Stage::Breakline]);
        assert_eq!(requested_stages(&["samples".to_owned()]).unwrap(), vec![Stage::Samples]);
        assert_eq!(
            requested_stages(&["nms".to_owned(), "hypotheses".to_owned(), "coarse".to_owned()])
                .unwrap(),
            vec![Stage::Hypotheses, Stage::Coarse, Stage::Nms],
            "the pair stages come back in pipeline order too"
        );
        assert_eq!(
            requested_stages(&["stage2".to_owned(), "stage1".to_owned()]).unwrap(),
            vec![Stage::Stage1, Stage::Stage2],
            "and so do the two refinement stages"
        );
        assert_eq!(
            requested_stages(&["candidates".to_owned(), "verify".to_owned()]).unwrap(),
            vec![Stage::Verify, Stage::Candidates],
            "and so do the two verification stages"
        );
        assert_eq!(
            requested_stages(&["assembly".to_owned(), "candidates".to_owned()]).unwrap(),
            vec![Stage::Candidates, Stage::Assembly],
            "R §8's row comes after the pair it is built from"
        );
        assert_eq!(
            requested_stages(&["outputs".to_owned(), "refine".to_owned()]).unwrap(),
            vec![Stage::Refine, Stage::Outputs],
            "and so do the last two rows of D §10.2"
        );
        // A stage this build does not have names the ones it does.
        let err = requested_stages(&["no such stage".to_owned()]).unwrap_err().to_string();
        assert!(err.contains("assembly") && err.contains("outputs"), "{err}");
    }

    #[test]
    fn segment_defaults_to_the_references_face_budget() {
        let cli = Cli::try_parse_from(["sherd-refit-rs", "segment", "in", "--out", "out"]).unwrap();
        match cli.command {
            super::Command::Segment(args) => {
                assert_eq!(args.target_faces, 200_000, "the Python's --target-faces default");
                // `cli.py`'s `segment` declares `--workers default=None` and resolves it in
                // `segment_only` to `max(1, cpu_count() - 1)`; `--threads` is the port's own and
                // wins over it (V4-D7, V4-D8).
                assert_eq!(args.workers, None);
                assert_eq!(args.threads, None);
                assert!(!args.force && !args.no_cache);
            }
            other => panic!("{other:?}"),
        }
    }

    /// `--seed N` reaches `Params.seed` on both subcommands that have it, and defaults to R §10's
    /// own 0 on both (task H3).
    ///
    /// The flag is what R §13's spread is measured with, and the reference has no CLI for it — so
    /// the port's is the only way to ask the question on either side. `segment` deliberately has
    /// no such flag: a cache written at one seed has R §3.5's three arrays recomputed by the run
    /// that wants another (R §3.7), which the fragment cache's own tests assert.
    #[test]
    fn the_seed_flag_reaches_the_parameters_on_run_and_bench() {
        let run = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command {
            super::Command::Run(a) => a.params().seed,
            other => panic!("{other:?}"),
        };
        assert_eq!(run(&["sherd-refit-rs", "run", "in", "--out", "out"]), 0, "R §10's default");
        assert_eq!(run(&["sherd-refit-rs", "run", "in", "--out", "out", "--seed", "4"]), 4);

        let bench = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command {
            super::Command::Bench(a) => a.seed,
            other => panic!("{other:?}"),
        };
        assert_eq!(bench(&["sherd-refit-rs", "bench", "in", "--out", "out"]), 0);
        assert_eq!(bench(&["sherd-refit-rs", "bench", "in", "--out", "out", "--seed", "3"]), 3);
        assert_eq!(Params::default().seed, 0, "and the default is the reference's own");

        // `segment` has none, and saying so is the point: the assertion fails if one is added
        // without a decision about what a cache written at another seed means.
        assert!(
            Cli::try_parse_from(["sherd-refit-rs", "segment", "in", "--out", "out", "--seed", "1"])
                .is_err(),
            "`segment` takes no --seed"
        );
    }

    /// `bench` resolves `--workers` and `--threads` exactly as `run` does (V5-D3).
    ///
    /// The two flags decide two different things — the pool's size and R §4.2's block schedule —
    /// and `bench` used to resolve neither: it passed `threads.unwrap_or(0)` to the pool, which
    /// rayon reads as one thread per core, and a literal `workers: 0` to the pipeline, which
    /// `pipeline::run` reads as `rayon::current_num_threads()`. Both came out ten on this machine
    /// where `run` uses nine. D §10.3 states its gates for `bench`, so the tool that reports them
    /// has to schedule the pairs the way the tool that is being timed does.
    #[test]
    fn bench_resolves_the_two_flags_the_way_run_does() {
        let cli = Cli::try_parse_from(["sherd-refit-rs", "bench", "in", "--out", "out"]).unwrap();
        match cli.command {
            super::Command::Bench(args) => {
                assert_eq!(args.workers, None, "unset, like `run`'s");
                assert_eq!(args.threads, None);
                assert_eq!(pool_threads(args.threads, args.workers), pipeline::default_workers());
                assert_eq!(schedule_workers(args.workers), pipeline::default_workers());
            }
            other => panic!("{other:?}"),
        }
        // `--workers` alone sizes both; `--threads` wins over it for the pool alone; both given,
        // each takes its own; and every one of these answers is `run`'s, computed by `run`'s own
        // two functions.
        assert_eq!(pool_threads(None, Some(4)), 4);
        assert_eq!(schedule_workers(Some(4)), 4);
        assert_eq!(pool_threads(Some(2), None), 2);
        assert_eq!(
            schedule_workers(None),
            pipeline::default_workers(),
            "an unset --workers is the reference's own count, not the machine's core count"
        );
        assert_eq!(pool_threads(Some(2), Some(4)), 2);
        assert_eq!(schedule_workers(Some(4)), 4);
        // R §4.2 reads the second number, and the schedule really is a function of it: pot H's
        // 55 pairs are walked one per block at nine workers and in 3x3 blocks at one. The eight
        // development sets happen not to separate nine from ten (V5-D3), which is why the old
        // `bench` moved no result — but "happens not to" is not a resolution rule.
        assert_ne!(pipeline::block_size(1, 55), pipeline::block_size(9, 55));
    }

    /// Every flag of `sherd_refit/cli.py` is here, spelled the same, and every default is the
    /// reference's — which for the twenty-one that map to [`Params`] means `Params::default()`
    /// exactly (R §1.1, R §1.4).
    #[test]
    #[allow(clippy::float_cmp, reason = "the defaults are literals on both sides")]
    fn run_takes_every_python_flag_at_the_python_default() {
        let cli = Cli::try_parse_from(["sherd-refit-rs", "run", "in", "--out", "out"]).unwrap();
        match cli.command {
            super::Command::Run(args) => {
                assert_eq!(args.target_faces, 200_000, "the Python's --target-faces default");
                assert!(args.workers.is_none() && args.threads.is_none(), "both default to None");
                assert!(!args.no_preview && !args.no_refine && !args.no_meshes);
                assert!(!args.no_cache && !args.force);
                assert_eq!(args.backend, Backend::Auto);
                assert_eq!(
                    args.params(),
                    Params {
                        tiers: Some(Thresholds::default()),
                        objects: Some(ObjectParams::default()),
                        ..Params::default()
                    },
                    "no flag given must leave every threshold of R §1.1 at its default"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// The two off switches together are exactly `Params::default()`.
    ///
    /// The library's default is both passes **off** and `run`'s default is both **on**, and the
    /// difference is deliberate: `Params::default()` is R §1.1's 46 knobs and nothing else, which
    /// is what the parity harness compares against a reference fixture and what `bench` measures
    /// the matcher with, while a museum run wants the band and the objects. `--tiers off
    /// --objects off` puts `run` back on the library's default, and that is the pair of switches
    /// the byte-identity checks use.
    #[test]
    fn tiers_off_is_the_library_default_and_tiers_on_is_the_measured_set() {
        let params = |argv: &[&str]| match Cli::try_parse_from(argv).unwrap().command {
            super::Command::Run(a) => a.params(),
            other => panic!("{other:?}"),
        };
        let base = ["sherd-refit-rs", "run", "in", "--out", "out"];
        let off: Vec<&str> =
            base.iter().copied().chain(["--tiers", "off", "--objects", "off"]).collect();
        assert_eq!(params(&off), Params::default());
        assert_eq!(Params::default().tiers, None, "and the library's own default is off");
        assert_eq!(Params::default().objects, None, "for both of them");

        let on = params(&base).tiers.expect("`run` computes a tier unless told not to");
        assert_eq!(on, Thresholds::default(), "M1 §3's chosen set, flag for flag");
        let objects = params(&base).objects.expect("`run` reads the objects unless told not to");
        assert_eq!(objects, ObjectParams::default(), "M1 §4's verdict, flag for flag");
        assert!(objects.demote.is_empty(), "no feature reached audit §D.2's own AUC of 0.800");
        assert!(!objects.disagreement, "and audit §D.2 (b) removes no false join on this benchmark");

        // The object rules take their own flags, and an unknown feature name is refused rather
        // than dropped -- the same rule `constraints.json` applies to a fragment name.
        let tuned: Vec<&str> = base
            .iter()
            .copied()
            .chain(["--object-demote", "shell_radius,thick", "--object-k-mad", "2.5"])
            .collect();
        let rules = params(&tuned).objects.expect("still on");
        assert!(rules.demote.contains(sherd_core::objects::FeatureKey::ShellRadius));
        assert!(rules.demote.contains(sherd_core::objects::FeatureKey::Thick));
        assert!(!rules.demote.contains(sherd_core::objects::FeatureKey::FracRough));
        assert_eq!(rules.k_mad, 2.5);

        let tuned: Vec<&str> = base
            .iter()
            .copied()
            .chain([
                "--tier-max-gap",
                "0.0064",
                "--tier-support",
                "0",
                "--tier-resample-accept",
                "3",
            ])
            .collect();
        let tuned = params(&tuned).tiers.expect("still on");
        assert!((tuned.max_gap_t - 0.0064).abs() < 1e-12);
        assert_eq!(tuned.min_support, 0, "which disables the support arm of the disjunction");
        assert_eq!(tuned.min_resample_accept, Some(3));
        assert!((tuned.min_tight - 0.35).abs() < 1e-12, "and the rest keep M1's values");
    }

    /// R §1.4's three renamed flags, and one that is not renamed, through the mapping.
    #[test]
    #[allow(clippy::float_cmp, reason = "the values are the literals just passed in")]
    fn run_renames_the_three_flags_r_1_4_names() {
        let cli = Cli::try_parse_from([
            "sherd-refit-rs",
            "run",
            "in",
            "--out",
            "out",
            "--candidates",
            "3",
            "--frac-density",
            "7.5",
            "--second-pass-candidates",
            "11",
            "--stage1",
            "4",
            "--no-preview",
        ])
        .unwrap();
        match cli.command {
            super::Command::Run(args) => {
                let p = args.params();
                assert_eq!(p.stage2, 3, "--candidates -> stage2");
                assert_eq!(p.frac_per_t2, 7.5, "--frac-density -> frac_per_t2");
                assert_eq!(p.second_pass_stage2, 11, "--second-pass-candidates");
                assert_eq!(p.stage1, 4, "and --stage1 keeps its name");
                assert!(args.no_preview);
            }
            other => panic!("{other:?}"),
        }
    }

    /// `--backend gpu` is refused on a build without the `gpu` feature; with it, the flag is
    /// resolved by `gpu::resolve`, which needs a device and is exercised by `gpu-check` instead.
    #[test]
    fn the_backend_flag_parses_and_defaults_to_auto() {
        assert_eq!(Backend::default(), Backend::Auto);
        assert_eq!("gpu".parse::<Backend>().unwrap(), Backend::Gpu);
        assert!("metal".parse::<Backend>().is_err());
        #[cfg(not(feature = "gpu"))]
        {
            let err = super::gpu::resolve(Backend::Gpu, None, None).unwrap_err().to_string();
            assert!(err.contains("without the `gpu` feature"), "{err}");
            assert_eq!(
                super::gpu::resolve(Backend::Auto, None, Some(1.0)).unwrap().backend,
                Backend::Cpu,
                "and `--gpu-memory` is accepted and ignored, so the two builds take the same flags"
            );
        }
    }

    #[test]
    fn parity_takes_the_flags_the_plan_names() {
        let cli = Cli::try_parse_from([
            "sherd-refit-rs",
            "parity",
            "--fixtures",
            "dump",
            "--stage",
            "working-mesh",
            "--injected",
        ])
        .unwrap();
        match cli.command {
            super::Command::Parity(args) => {
                assert_eq!(args.fixtures.to_str(), Some("dump"));
                assert_eq!(args.stage, ["working-mesh"]);
                assert!(args.injected);
                assert!(args.input.is_none());
            }
            other => panic!("{other:?}"),
        }
    }
}
