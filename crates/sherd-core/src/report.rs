//! `transforms.json`, `report.json`, `report.md` and the placed meshes (R §11.1–11.4).
//!
//! Same file names, same JSON keys and the same section order as the reference, so that
//! `tools/evaluate.py` scores a Rust run without knowing which implementation wrote it, plus the
//! additive `engine` key of D §4.3 — the three version constants, the git commit of the build and
//! the backend that actually ran, in **both** `transforms.json` and `report.json`; the Python
//! readers ignore keys they do not know.
//!
//! Three things are worth saying about the fidelity of each file.
//!
//! * **`transforms.json`** is a pose per fragment and nothing else is derived from it, so what has
//!   to agree is the numbers. D §10.2's `outputs` row gates them through the displacement they
//!   give the fragment's own samples, at the `refine` row's own tolerance.
//! * **`report.json`** carries the candidate list, the used and rejected joins and the fragment
//!   statistics. Its `timings` differ between two runs of the same input by construction and are
//!   excluded from every comparison — the reference's own fixture dump nulls them for the same
//!   reason.
//! * **`placed/<name>.ply`** is the one output that is compared **byte for byte**. The mesh is the
//!   original file cleaned as R §3.1 steps 1–2 — all components, not the largest — moved by its
//!   pose and written through [`crate::io::writer`], whose header is Open3D's. That
//!   only works because the pose is applied with
//!   [`apply_transform_fused`]: Open3D's `Transform` is an
//!   Eigen 4×4 product and Eigen fuses, so the unfused product would put a few ULP into every
//!   coordinate and every `double` in the file would differ. The one byte range that is *not* the
//!   reference's is the header comment, which names the writer.
//!
//! Filled in by phase-1d step D2.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use nalgebra::Matrix4;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::assembly::constraints::Report as ConstraintReport;
use crate::error::{Error, Result};
use crate::fragment::Fragment;
use crate::io::writer::PlyStream;
use crate::matching::pair::Candidate;
use crate::matching::verify::Scores;
use crate::memory::{self, Budget, MemorySemaphore, reservation};
use crate::mesh::Mesh;
use crate::objects::{FeatureKey, ObjectReport};
use crate::params::Params;
use crate::review::ReviewIndex;
use crate::tiers::{Evidence, Tier, TierCounts, TierJoin, TierReport};
use crate::types::{FragId, apply_transform_fused};
use crate::{ALGO_REF, CACHE_VERSION, CORE_VERSION, GIT_COMMIT};

/// A JSON object that keeps the order it was built in.
///
/// R §11.1's `fragments` and R §11.2's `timings` are Python dicts written by `json.dump`, and a
/// Python dict iterates in **insertion** order: `transforms.json` comes out in R §8's placement
/// order (the seed's two fragments, then each placement, then the singletons in collection order)
/// and `timings` in the order the stages finished. A `BTreeMap` sorts by key and writes a
/// different file — which is what V4-D5 and V4-D4 found, `FY234007` first instead of last and
/// `assembly, matching, preprocess, refine` instead of `preprocess, matching, assembly, refine`.
///
/// Small on purpose: a lookup is a linear scan, which is what a collection of at most a few
/// hundred fragments wants, and it keeps the dependency list as D §3 has it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ordered<V>(Vec<(String, V)>);

impl<V> Ordered<V> {
    /// An empty object.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// `d[key] = value`: a new key is appended, an existing one keeps its place.
    pub fn insert(&mut self, key: impl Into<String>, value: V) {
        let key = key.into();
        match self.0.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => self.0.push((key, value)),
        }
    }

    /// The value under `key`, or `None`.
    pub fn get(&self, key: &str) -> Option<&V> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Whether the object has that key.
    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// The pairs, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &V)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The keys, in order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }

    /// How many entries the object has.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<V> FromIterator<(String, V)> for Ordered<V> {
    fn from_iter<T: IntoIterator<Item = (String, V)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a, V> IntoIterator for &'a Ordered<V> {
    type Item = (&'a str, &'a V);
    type IntoIter =
        std::iter::Map<std::slice::Iter<'a, (String, V)>, fn(&'a (String, V)) -> (&'a str, &'a V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
}

impl<V> std::ops::Index<&str> for Ordered<V> {
    type Output = V;

    fn index(&self, key: &str) -> &V {
        self.get(key).expect("no such key")
    }
}

impl<V: Serialize> Serialize for Ordered<V> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap as _;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

impl<'de, V: Deserialize<'de>> Deserialize<'de> for Ordered<V> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor<V>(std::marker::PhantomData<V>);

        impl<'de, V: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<V> {
            type Value = Ordered<V>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Ordered<V>, M::Error> {
                let mut out = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((k, v)) = map.next_entry::<String, V>()? {
                    out.push((k, v));
                }
                Ok(Ordered(out))
            }
        }

        deserializer.deserialize_map(Visitor(std::marker::PhantomData))
    }
}

/// R §11.2's `timings`: seconds per stage, in the order the pipeline finished them.
pub type Timings = Ordered<f64>;

/// `timings`' neighbour in `report.json`: the peak resident set of the process, per stage.
///
/// Not the reference's — R §11.2 has no such key — and additive in the same sense `engine` is
/// (D §4.3): a reader that does not know it ignores it, and `sherd-parity`'s `outputs` stage
/// never sees it, because the value it rebuilds for the comparison carries no memory block.
///
/// It exists because the audit's §B.3 asks which stage holds the peak and what a structure that is
/// never freed costs at it, and D §8's table could not answer: it is a model with one whole-run
/// measurement beside it. The numbers are **sampled** at
/// [`SAMPLE_INTERVAL`](crate::memory::RssMonitor) rather than integrated, so like `timings` they
/// are the part of the file two runs of one build legitimately disagree on.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct MemoryReport {
    /// The largest resident set seen over the whole run, in bytes.
    pub peak_rss: u64,
    /// The largest resident set seen while each stage ran, in the order the stages finished, in
    /// bytes. A stage is charged for what was resident during it and not for what the stage
    /// before it was holding at the boundary.
    pub stages: Ordered<u64>,
}

/// One fragment's row of `report.json`'s `fragments` and of R §11.3's fragment table — the
/// reference's `Fragment.stats()`, key for key and in its order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FragmentStats {
    /// The fragment's name in the collection.
    pub name: String,
    /// Faces of the working mesh.
    pub faces: u64,
    /// Faces of the file, after R §3.1's cleaning and before the largest-component pass.
    pub orig_faces: u64,
    /// Vertices of the same.
    pub orig_vertices: u64,
    /// R §3.2's wall thickness.
    pub thickness: f64,
    /// R §3.2's unfiltered ray mode.
    pub thickness_mode: f64,
    /// R §3.3's `res`.
    pub resolution: f64,
    /// R §3.3.2's verdict.
    pub watertight: bool,
    /// The working mesh's bounding-box side lengths.
    pub extent: [f64; 3],
    /// Total area of the working mesh.
    pub area: f64,
    /// Fracture area over total area (R §3.4).
    pub fracture_area_fraction: f64,
}

impl FragmentStats {
    /// The reference's `stats()` for one preprocessed fragment.
    pub fn of(fragment: &Fragment) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in &fragment.mesh.v {
            let p = v.to_f64();
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if !lo[0].is_finite() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        Self {
            name: fragment.name.clone(),
            faces: fragment.n_faces() as u64,
            orig_faces: u64::from(fragment.n_orig_faces),
            orig_vertices: u64::from(fragment.n_orig_vertices),
            thickness: fragment.thick,
            thickness_mode: fragment.thick_mode,
            resolution: fragment.res(),
            watertight: fragment.watertight,
            extent: [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
            area: fragment.area,
            fracture_area_fraction: fragment.fracture_fraction(),
        }
    }
}

/// One fragment's entry in `transforms.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Placement {
    /// The world pose, row major, after R §8.2's recentring.
    pub matrix: [[f64; 4]; 4],
    /// Which group the fragment ended in.
    pub group: usize,
    /// Whether that group has two or more members.
    pub placed: bool,
}

/// `transforms.json` (R §11.1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transforms {
    /// The collection's median wall thickness.
    pub thickness: f64,
    /// Every field of [`Params`].
    pub params: Params,
    /// One entry per fragment, by name, **in R §8's placement order** — the order the
    /// reference's `poses` dict was built in, which is what `json.dump` writes (V4-D5).
    pub fragments: Ordered<Placement>,
    /// The groups, as name lists, in R §8's final order.
    pub groups: Vec<Vec<String>>,
    /// Which build wrote the file (D §4.3; not in the reference).
    ///
    /// The poses in this file are the ones a downstream tool applies, so "which build produced
    /// them, on which backend" belongs here as much as it does in `report.json` — D §4.3 names
    /// both files and only `report.json` had it (V6-D7). Absent from a file the reference wrote,
    /// which is why it is optional on the way in, and last in the struct so that R §11.1's own
    /// four keys keep their order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    /// Roadmap item 3's band for every pair that produced a candidate (audit §D.1), the pair's
    /// best band first; absent from a run with the tier pass off.
    ///
    /// The *set* those bands were decided with is in `params.tiers`, so a reader holding this file
    /// has both halves of the statement: which joins the tool stands behind, and what standing
    /// behind one means.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<TierJoin>>,
}

/// One candidate as `report.json` writes it: the reference's `Candidate.to_json()`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateJson {
    /// The fragment the pose maps into.
    pub a: String,
    /// The fragment the pose moves.
    pub b: String,
    /// The pose, row major.
    #[serde(rename = "T")]
    pub transform: [[f64; 4]; 4],
    /// R §6.5's verdict.
    pub accepted: bool,
    /// `seam · tight`.
    pub score: f64,
    /// Every score of R §6, flattened into the same object.
    #[serde(flatten)]
    pub scores: Scores,
    /// Why R §8 did not use an accepted join — only on `joins_rejected`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reason: Option<String>,
    /// Roadmap item 3's band (audit §D.1); absent from a run with the tier pass off.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tier: Option<Tier>,
    /// What the band was decided on; absent for a candidate R §6.5 refused, which is never probed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub evidence: Option<Evidence>,
}

impl CandidateJson {
    /// One candidate, with the names its two ids stand for.
    pub fn of(candidate: &Candidate, names: &[String]) -> Self {
        Self {
            a: names[candidate.a as usize].clone(),
            b: names[candidate.b as usize].clone(),
            transform: rows(&candidate.transform),
            accepted: candidate.accepted,
            score: candidate.score(),
            scores: candidate.scores,
            reason: None,
            tier: None,
            evidence: None,
        }
    }

    /// The same with R §8's rejection sentence attached.
    pub fn rejected(candidate: &Candidate, names: &[String], reason: &str) -> Self {
        Self { reason: Some(reason.to_owned()), ..Self::of(candidate, names) }
    }

    /// The same with roadmap item 3's band and its evidence, when the run computed one.
    ///
    /// `index` is the candidate's place in the run's own list, which is what
    /// [`TierReport`](crate::tiers::TierReport) is indexed by.
    #[must_use]
    pub fn tiered(mut self, tiers: Option<&TierReport>, index: usize) -> Self {
        if let Some(t) = tiers {
            self.tier = t.tiers.get(index).copied();
            self.evidence = t.evidence.get(index).cloned().flatten();
        }
        self
    }
}

/// D §4.3's additive block: which build wrote the file, and what it ran on.
///
/// D §4.3 asks for "all three plus the git commit and the backend used", in `report.json` **and**
/// `transforms.json`. All five are here, and each answers a question the others cannot (V6-D7).
///
/// * `core_version` moves once a release, `algo_ref` once the algorithm changes and
///   `cache_version` once the cache layout does — so between those three nothing distinguishes two
///   builds, which is what `commit` is for. It is `"unknown"` outside a git checkout.
/// * `backend` is the **resolved** executor, not what `--backend` asked for: `auto` never appears
///   here, because a file that says `auto` says nothing about the arithmetic that produced it.
///   With a device it carries the adapter's own name — `gpu:Apple M2 Pro` — which is the field
///   that makes a GPU-side deviation attributable to a machine rather than to "the GPU".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Engine {
    /// `sherd-core`'s crate version.
    pub core_version: String,
    /// The frozen algorithm this port reproduces.
    pub algo_ref: String,
    /// The layout version of the fragment caches this build reads and writes.
    pub cache_version: u32,
    /// The commit this binary was built from, or `"unknown"`.
    pub commit: String,
    /// The executor that ran, with the adapter's name when it was a device.
    pub backend: String,
    /// R §10's seed, which `--seed` sets (task H3).
    ///
    /// Absent from a file the reference wrote and from one this port wrote before task H3, which
    /// is why it reads back as 0 — the value every such run had.
    #[serde(default)]
    pub seed: u64,
}

impl Engine {
    /// The block for a run on `backend` at `seed`, both already resolved — the backend carries its
    /// adapter (see [`Backend::label`](crate::executor::Backend::label)) and the seed is
    /// `Params::seed` as the run resolved `--seed`.
    ///
    /// The seed is in `params` as well, and is repeated here on purpose: `engine` is the block
    /// that answers "which build produced this file, and how", and the seed is the second half of
    /// that question. A reader holding a pose should not have to walk 46 parameters to find out
    /// which of R §13's draws it is looking at.
    pub fn of(backend: &str, seed: u64) -> Self {
        Self {
            core_version: CORE_VERSION.to_owned(),
            algo_ref: ALGO_REF.to_owned(),
            cache_version: CACHE_VERSION,
            commit: GIT_COMMIT.to_owned(),
            backend: backend.to_owned(),
            seed,
        }
    }
}

/// `report.json` (R §11.2), plus D §4.3's `engine`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReportJson {
    /// The collection's median wall thickness.
    pub thickness: f64,
    /// One entry per fragment, in collection order.
    pub fragments: Vec<FragmentStats>,
    /// The groups, as name lists.
    pub groups: Vec<Vec<String>>,
    /// Every field of [`Params`].
    pub params: Params,
    /// Wall-clock seconds per stage; the one part of the file two runs disagree on.
    ///
    /// `Option` because the reference's *fixture dump* replaces every value with `null` for
    /// exactly that reason (`pipeline._dump_outputs`), and this type has to read the dump's copy
    /// as well as the run's own.
    pub timings: Ordered<Option<f64>>,
    /// The joins R §8 used, in the order it took them.
    pub joins_used: Vec<CandidateJson>,
    /// The accepted joins R §8 refused, with the reason.
    pub joins_rejected: Vec<CandidateJson>,
    /// Every candidate of every pair, in pair order and best first within a pair.
    pub candidates: Vec<CandidateJson>,
    /// Which build wrote the file (D §4.3; not in the reference).
    ///
    /// Absent from a file the reference wrote, which is why it is optional on the way in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    /// Peak resident set per stage (audit §B.3; not in the reference).
    ///
    /// Absent from a file the reference wrote and from the value the parity harness rebuilds,
    /// which is why it is optional and why it is skipped when it is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryReport>,
    /// How many candidates fell in each of roadmap item 3's bands (audit §D.1); absent from a run
    /// with the tier pass off. The set they were decided with is `params.tiers`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<TierCounts>,
    /// Every constraint the run was given and what it did (audit §D.1); absent from a run without
    /// a `constraints.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraints: Option<ConstraintReport>,
    /// Audit §D.2's objects — every group with its consensus, its rim diameter and the members
    /// that consensus rejects — absent from a run with `--objects off`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objects: Option<ObjectReport>,
}

/// A 4×4 as the nested lists both JSON files carry.
pub fn rows(m: &Matrix4<f64>) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = m[(i, j)];
        }
    }
    out
}

/// R §11.1's `transforms.json`.
///
/// `groups` are [`FragId`] lists in R §8's final order; `poses` is one pose per fragment, indexed
/// by id, after R §8.2. The `group` of a fragment is its index in `groups` and `placed` is whether
/// that group has two or more members — which is not the same as "the assembly moved it", and is
/// the reference's definition.
///
/// `order` is the order the entries come out in: the reference builds `fragments` from
/// `poses.items()`, and its `poses` is an insertion-ordered dict written by R §8's greedy loop —
/// the seed's two fragments, then every placement in the order it happened, then the singletons in
/// collection order ([`Assembly::order`](crate::assembly::Assembly::order)). Anything `order`
/// leaves out follows in collection order.
/// `backend` is the resolved executor's label (D §4.3); `None` writes no `engine` key, which is
/// what a harness rebuilding the file for a comparison wants.
#[allow(clippy::too_many_arguments, reason = "R §11.1's four keys, plus how they were produced")]
pub fn write_transforms(
    path: impl AsRef<Path>,
    names: &[String],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
    order: &[FragId],
    thickness: f64,
    params: &Params,
    backend: Option<&str>,
    tiers: Option<&[TierJoin]>,
) -> Result<()> {
    write_json(
        path.as_ref(),
        &transforms(names, poses, groups, order, thickness, params, backend, tiers),
    )
}

/// The value [`write_transforms`] serialises, for callers that want it in memory.
#[allow(clippy::too_many_arguments, reason = "R §11.1's four keys, plus how they were produced")]
pub fn transforms(
    names: &[String],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
    order: &[FragId],
    thickness: f64,
    params: &Params,
    backend: Option<&str>,
    tiers: Option<&[TierJoin]>,
) -> Transforms {
    let mut group_of = vec![0_usize; names.len()];
    let mut placed = vec![false; names.len()];
    for (k, group) in groups.iter().enumerate() {
        for &n in group {
            group_of[n as usize] = k;
            placed[n as usize] = group.len() > 1;
        }
    }
    // `order` is R §8's insertion order, and a fragment the assembly never saw would be missing
    // from the file rather than merely late, so the collection order closes the walk.
    let mut fragments = Ordered::new();
    let rest = 0..u32::try_from(names.len()).expect("fewer than 2^32 fragments");
    for n in order.iter().copied().chain(rest) {
        let n = n as usize;
        if fragments.contains_key(&names[n]) {
            continue;
        }
        fragments.insert(
            names[n].clone(),
            Placement { matrix: rows(&poses[n]), group: group_of[n], placed: placed[n] },
        );
    }
    Transforms {
        thickness,
        params: *params,
        fragments,
        groups: groups
            .iter()
            .map(|g| g.iter().map(|&n| names[n as usize].clone()).collect())
            .collect(),
        engine: backend.map(|b| Engine::of(b, params.seed)),
        tiers: tiers.map(<[TierJoin]>::to_vec),
    }
}

/// Everything R §11.2 and R §11.3 need that is not a [`Params`] or a fragment statistic.
#[derive(Clone, Debug)]
pub struct Outcome<'a> {
    /// One name per [`FragId`], in collection order.
    pub names: &'a [String],
    /// Every candidate of every pair, in pair order.
    pub candidates: &'a [Candidate],
    /// Indices into `candidates` of the joins R §8 used, in the order it took them.
    pub used: &'a [usize],
    /// Indices into `candidates` of the accepted joins R §8 refused, with the reason.
    pub rejected: &'a [(usize, String)],
    /// The groups, in R §8's final order.
    pub groups: &'a [Vec<FragId>],
    /// Roadmap item 3's bands and their evidence, or `None` on a run with the tier pass off — in
    /// which case every section this adds to `report.md` and every key it adds to the two JSON
    /// files is absent, and the outputs are the bytes they were.
    pub tiers: Option<&'a TierReport>,
    /// Every constraint of `constraints.json` and what the run did about it, or `None` on a run
    /// without the file — in which case `## Constraints` and the `constraints` key are absent.
    pub constraints: Option<&'a ConstraintReport>,
    /// Which pairs got a review image, or `None` without `--review-images` — in which case the
    /// per-fragment index has no `image` column.
    pub review: Option<&'a ReviewIndex>,
    /// Roadmap item 4's objects, their consensus and the joins it demoted, or `None` on a run with
    /// `--objects off` — in which case `## Objects` and the `objects` key are absent.
    pub objects: Option<&'a ObjectReport>,
    /// `--probable-top`: how many rows of the probable band `report.md` shows, `0` for all of them
    /// ([`crate::tiers::probable_shown`]).
    ///
    /// `report.json` is not cut by it — the whole band is in the file whatever this says, because
    /// the flag is about what a person is handed and not about what the run found.
    pub probable_top: usize,
}

/// R §11.2's `report.json` and R §11.3's `report.md`, both into `out_dir`.
#[allow(
    clippy::too_many_arguments,
    reason = "R §11.2's own contents, plus how and on what they were produced"
)]
pub fn write_report(
    out_dir: impl AsRef<Path>,
    stats: &[FragmentStats],
    thickness: f64,
    outcome: &Outcome<'_>,
    timings: &Timings,
    params: &Params,
    backend: &str,
    memory: Option<&MemoryReport>,
) -> Result<()> {
    let out_dir = out_dir.as_ref();
    let json = report_json(stats, thickness, outcome, timings, params, backend, memory);
    write_json(&out_dir.join("report.json"), &json)?;
    let markdown = report_markdown(stats, thickness, outcome, timings, params);
    std::fs::write(out_dir.join("report.md"), markdown)
        .map_err(|e| Error::write(out_dir.join("report.md"), e))
}

/// The value [`write_report`] serialises.
#[allow(
    clippy::too_many_arguments,
    reason = "R §11.2's own contents, plus how and on what they were produced"
)]
pub fn report_json(
    stats: &[FragmentStats],
    thickness: f64,
    outcome: &Outcome<'_>,
    timings: &Timings,
    params: &Params,
    backend: &str,
    memory: Option<&MemoryReport>,
) -> ReportJson {
    let names = outcome.names;
    ReportJson {
        thickness,
        fragments: stats.to_vec(),
        groups: outcome
            .groups
            .iter()
            .map(|g| g.iter().map(|&n| names[n as usize].clone()).collect())
            .collect(),
        params: *params,
        timings: timings.iter().map(|(k, &v)| (k.to_owned(), Some(v))).collect(),
        joins_used: outcome
            .used
            .iter()
            .map(|&i| CandidateJson::of(&outcome.candidates[i], names).tiered(outcome.tiers, i))
            .collect(),
        joins_rejected: outcome
            .rejected
            .iter()
            .map(|(i, why)| {
                CandidateJson::rejected(&outcome.candidates[*i], names, why)
                    .tiered(outcome.tiers, *i)
            })
            .collect(),
        candidates: outcome
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| CandidateJson::of(c, names).tiered(outcome.tiers, i))
            .collect(),
        engine: Some(Engine::of(backend, params.seed)),
        memory: memory.cloned(),
        tiers: outcome.tiers.map(|t| t.pair_counts(outcome.candidates)),
        constraints: outcome.constraints.cloned(),
        objects: outcome.objects.cloned(),
    }
}

/// R §11.3's `report.md`, section for section and format for format.
#[allow(clippy::too_many_lines, reason = "R §11.3 is a list of sections, and this is the list")]
pub fn report_markdown(
    stats: &[FragmentStats],
    thickness: f64,
    outcome: &Outcome<'_>,
    timings: &Timings,
    params: &Params,
) -> String {
    let names = outcome.names;
    let mut lines: Vec<String> = vec!["# Reassembly report".to_owned(), String::new()];
    lines.push(format!(
        "Wall thickness (collection median): {thickness:.2} units. All distances below are in \
         units of thickness (t)."
    ));
    lines.push(String::new());
    lines.push(
        "Every distance threshold is `max(k t, m res)`, with `res` the median edge length of the \
         working mesh (column `edge`). The `tight` distance and the `gap` limit a pair was \
         actually judged by are listed per join below; they equal the `k t` form on any mesh with \
         enough edges across the wall."
            .to_owned(),
    );
    lines.push(String::new());
    lines.push("## Fragments".to_owned());
    lines.push(String::new());
    lines.push(
        "| fragment | faces (orig) | thickness | ray mode | thickness/median | edge | edges per \
         t | fracture area % | watertight | extent |"
            .to_owned(),
    );
    lines.push("|---|---|---|---|---|---|---|---|---|---|".to_owned());
    for s in stats {
        let flag = if (s.thickness / thickness - 1.0).abs() < 0.4 { "" } else { " **(differs)**" };
        let res = s.resolution;
        let per_t = if res == 0.0 { 0.0 } else { s.thickness / res };
        let extent = s.extent.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" x ");
        lines.push(format!(
            "| {} | {} ({}) | {:.2}{flag} | {:.2} | {:.2} | {res:.3} | {per_t:.1} | {:.1} | {} | \
             {extent} |",
            s.name,
            s.faces,
            s.orig_faces,
            s.thickness,
            s.thickness_mode,
            s.thickness / thickness,
            100.0 * s.fracture_area_fraction,
            python_bool(s.watertight),
        ));
    }
    lines.push(String::new());
    lines.push("## Assembly".to_owned());
    lines.push(String::new());
    for (k, g) in outcome.groups.iter().enumerate() {
        if g.len() > 1 {
            lines.push(format!("- group {k}: {}", join_names(g, names)));
        }
    }
    let singles: Vec<FragId> =
        outcome.groups.iter().filter(|g| g.len() == 1).map(|g| g[0]).collect();
    if !singles.is_empty() {
        lines.push(format!("- not assembled (no confident join): {}", join_names(&singles, names)));
    }
    lines.extend(constraint_section(outcome));
    lines.extend(object_section(outcome));
    lines.push(String::new());
    lines.push("## Joins used".to_owned());
    lines.push(String::new());
    lines.push(
        "| A | B | score | seam (t) | tight A/B | tight at (t) | gap (t) | gap limit (t) | \
         contact (t²) | shell cont. | normal agr. | penetration |"
            .to_owned(),
    );
    lines.push("|---|---|---|---|---|---|---|---|---|---|---|---|".to_owned());
    for &i in outcome.used {
        let c = &outcome.candidates[i];
        let s = &c.scores;
        lines.push(format!(
            "| {} | {} | {:.2} | {:.1} | {:.2} / {:.2} | {:.3} | {:.3} | {:.3} | {:.1} | {:.3} | \
             {:.2} | {:.4} |",
            names[c.a as usize],
            names[c.b as usize],
            c.score(),
            s.seam,
            s.tight_a,
            s.tight_b,
            s.tight_delta,
            s.gap,
            s.gap_limit,
            s.contact,
            s.cont,
            s.cont_n,
            s.pen,
        ));
    }
    lines.extend(tier_sections(outcome, params));
    if !outcome.rejected.is_empty() {
        lines.push(String::new());
        lines.push(
            if outcome.tiers.is_some() {
                "## Confirmed joins not used"
            } else {
                "## Accepted joins not used"
            }
            .to_owned(),
        );
        lines.push(String::new());
        for (i, why) in outcome.rejected {
            let c = &outcome.candidates[*i];
            lines.push(format!(
                "- {} – {} (score {:.2}): {why}",
                names[c.a as usize],
                names[c.b as usize],
                c.score()
            ));
        }
    }
    lines.push(String::new());
    lines.push("## Best candidate per pair".to_owned());
    lines.push(String::new());
    lines.push(legend(params));
    lines.push(String::new());
    lines.push(
        "| A | B | accepted | score | seam (t) | tight A/B | tight at (t) | gap (t) | gap limit \
         (t) | penetration | normal agr. |"
            .to_owned(),
    );
    lines.push("|---|---|---|---|---|---|---|---|---|---|---|".to_owned());
    for (pair, i) in best_per_pair(outcome) {
        let c = &outcome.candidates[i];
        let s = &c.scores;
        let (pen, cont_n) = if s.partial {
            ("n/a".to_owned(), "n/a".to_owned())
        } else {
            (format!("{:.4}", s.pen), format!("{:.2}", s.cont_n))
        };
        lines.push(format!(
            "| {} | {} | {} | {:.2} | {:.1} | {:.2} / {:.2} | {:.3} | {:.3} | {:.3} | {pen} | \
             {cont_n} |",
            pair.0,
            pair.1,
            if c.accepted { "yes" } else { "no" },
            c.score(),
            s.seam,
            s.tight_a,
            s.tight_b,
            s.tight_delta,
            s.gap,
            s.gap_limit,
        ));
    }
    lines.push(String::new());
    lines.push("## Timing".to_owned());
    lines.push(String::new());
    for (k, v) in timings {
        lines.push(format!("- {k}: {v:.1} s"));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// Audit §D.1's `## Constraints`: every line of `constraints.json` and what the run did with it.
///
/// Nothing at all on a run without the file, which is what keeps such a run's `report.md` byte for
/// byte the one before the flag existed. The section stands directly under `## Assembly` because a
/// reader who has just seen which fragments were placed needs to know at once which of those
/// decisions were the operator's rather than the geometry's.
fn constraint_section(outcome: &Outcome<'_>) -> Vec<String> {
    let Some(report) = outcome.constraints else { return Vec::new() };
    let mut lines: Vec<String> = vec![String::new(), "## Constraints".to_owned(), String::new()];
    let unsatisfied = report.unsatisfied();
    lines.push(format!(
        "`constraints.json` v{} — {} constraint{}, {}. A constraint never edits a score: what it \
         changes is which pairs are matched, which candidates the assembly may build with, and the \
         order it sees them in.",
        report.version,
        report.entries.len(),
        if report.entries.len() == 1 { "" } else { "s" },
        if unsatisfied == 0 {
            "every one satisfied".to_owned()
        } else {
            format!("**{unsatisfied} unsatisfiable**")
        },
    ));
    lines.push(String::new());
    if report.entries.is_empty() {
        lines.push("The file was read and constrains nothing.".to_owned());
        return lines;
    }
    lines.push("| constraint | A | B | satisfied | what it did |".to_owned());
    lines.push("|---|---|---|---|---|".to_owned());
    for e in &report.entries {
        lines.push(format!(
            "| `{}` | {} | {} | {} | {} |",
            e.list,
            e.a,
            e.b,
            if e.satisfied { "yes" } else { "**no**" },
            e.outcome,
        ));
    }
    lines
}

/// Audit §D.2's `## Objects`: every group read as a vessel, with the consensus its members agree
/// on and the ones that consensus does not fit.
///
/// Nothing at all on a run with `--objects off`, which is what keeps such a run's `report.md` byte
/// for byte the one before roadmap item 4 existed. The section stands under `## Constraints` and
/// above the joins, because "which pots are these" is the question a conservator asks of the
/// groups they have just read.
///
/// **A deviation printed here has not refused anything.** M1 §4 measured every geometric feature
/// on every collection with real object ids and the best AUC is 0.740, under audit §D.2's own
/// 0.800 rule; task S2 measured the clay body at 0.985 on `synthetic_mix3_24`, above it, and then
/// measured what letting it demote would do — it removes no false join on any collection this
/// project can measure it on — so the shortlist that may demote is still empty and every row below
/// reports. The `demotes` column says which of them would act if a flag put them on that list.
fn object_section(outcome: &Outcome<'_>) -> Vec<String> {
    let Some(report) = outcome.objects else { return Vec::new() };
    let mut lines: Vec<String> = vec![String::new(), "## Objects".to_owned(), String::new()];
    let rejects: usize = report.objects.iter().map(|o| o.rejects.len()).sum();
    lines.push(format!(
        "{} object{} — one per assembled group — with the median and MAD its members agree on. \
         {} member{} sit{} outside its object's consensus, and {} join{} demoted. A feature may \
         **veto** only where its measured separation exceeds audit §D.2's own AUC of 0.800. Of \
         the geometric features task M1 §4 measured the best at 0.740; of the colours task S2 \
         measured the clay body — the mean Lab of the faces the segmentation calls fracture — at \
         0.985 on a collection of three vessels, the first feature of this project above that \
         bar. Letting it refuse a join removes no false join on any collection measured, so on \
         the shipped settings every number below reports and none of them refuses a join.",
        report.objects.len(),
        if report.objects.len() == 1 { "" } else { "s" },
        rejects,
        if rejects == 1 { "" } else { "s" },
        if rejects == 1 { "s" } else { "" },
        report.demotions.len(),
        if report.demotions.len() == 1 { " was" } else { "s were" },
    ));
    if report.merges > 0 {
        lines.push(String::new());
        lines.push(format!(
            "{} pair{} of groups merged through a confirmed join (audit §D.2 (c)), under the \
             penetration test across both groups and consistency with every cross-group join the \
             gate admits.",
            report.merges,
            if report.merges == 1 { "" } else { "s" },
        ));
    }
    lines.push(String::new());
    lines.push(
        "| object | fragments | joins | consensus (median +/- MAD) | rim | outside it |".to_owned(),
    );
    lines.push("|---|---|---:|---|---|---|".to_owned());
    for o in &report.objects {
        let consensus = o
            .consensus
            .iter()
            .filter(|c| FeatureKey::REPORTED.contains(&c.feature))
            .map(|c| format!("{} {:.3} +/- {:.3} (n={})", c.feature.key(), c.median, c.mad, c.n))
            .collect::<Vec<String>>()
            .join("; ");
        let outside = o
            .rejects
            .iter()
            .map(|r| {
                let d = &r.deviations[0];
                format!(
                    "{} ({} {:.3}, {:.1} MAD{})",
                    r.fragment,
                    d.feature.key(),
                    d.value,
                    d.mads,
                    if d.demoting { ", demotes" } else { "" }
                )
            })
            .collect::<Vec<String>>()
            .join("; ");
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} |",
            o.group,
            if o.fragments.is_empty() { "-".to_owned() } else { o.fragments.join(", ") },
            o.joins,
            if consensus.is_empty() { "-".to_owned() } else { consensus },
            o.rim_diameter.map_or_else(|| "-".to_owned(), |d| format!("{d:.1}")),
            if outside.is_empty() { "-".to_owned() } else { outside },
        ));
    }
    lines.extend(demotion_tables(report));
    lines
}

/// The two lists that follow `## Objects` when a run has anything to put in them: the joins the
/// pass demoted, and what the operator's own two lists say about the objects it built.
fn demotion_tables(report: &ObjectReport) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    if !report.demotions.is_empty() {
        lines.push(String::new());
        lines.push("### Joins the object pass demoted".to_owned());
        lines.push(String::new());
        lines.push(
            "Confirmed joins moved to **probable**, never rejected: the conservator decides, with \
             the numbers beside the review image."
                .to_owned(),
        );
        lines.push(String::new());
        lines.push("| A | B | arm | why |".to_owned());
        lines.push("|---|---|---|---|".to_owned());
        for d in &report.demotions {
            lines.push(format!("| {} | {} | {} | {} |", d.a, d.b, d.arm.label(), d.reason));
        }
    }
    for (heading, pairs, sentence) in [
        (
            "`same_object` pairs the assembly did not put together",
            &report.same_object_split,
            "The operator says these belong to one vessel and the geometry has not joined them; \
             nothing was forced, and a constraint never edits a score.",
        ),
        (
            "`different_object` pairs that ended up in one object",
            &report.different_object_together,
            "Both the join veto and the merge test refuse this, so the list is expected to be \
             empty and exists to prove it.",
        ),
    ] {
        if pairs.is_empty() {
            continue;
        }
        lines.push(String::new());
        lines.push(format!("### {heading}"));
        lines.push(String::new());
        lines.push(sentence.to_owned());
        lines.push(String::new());
        for [a, b] in pairs {
            lines.push(format!("- {a} — {b}"));
        }
    }
    lines
}

/// How many refused pairs the `Rejected` section lists before it stops and points at the table
/// that holds them all.
///
/// A 24-fragment collection refuses two hundred pairs and a conservator reads none of them; the
/// tally above the table says how many there were and which limit each one failed, and R §11.3's
/// own "Best candidate per pair" — which this report still writes in full — is where the rest are.
const REJECTED_LISTED: usize = 25;

/// Roadmap item 3's three sections and the per-fragment index (audit §D.1), or nothing at all when
/// the run had the tier pass off.
///
/// The order is the order a bench reads them in: what the tool stands behind, what it wants a
/// person to look at, what it threw away, and then the same thing again arranged by fragment —
/// audit §C.8(a): *"the most useful output for a bench is not the assembly but, per fragment, its
/// best two or three candidate partners with pictures and numbers"*.
#[allow(clippy::too_many_lines, reason = "four tables, and this is the four tables")]
fn tier_sections(outcome: &Outcome<'_>, params: &Params) -> Vec<String> {
    let Some(report) = outcome.tiers else { return Vec::new() };
    let names = outcome.names;
    let th = &report.thresholds;
    let mut lines: Vec<String> = Vec::new();
    let representatives = crate::tiers::representatives(outcome.candidates);
    let pair_name = |i: usize| {
        let c = &outcome.candidates[i];
        (names[c.a as usize].as_str(), names[c.b as usize].as_str())
    };
    let evidence_of = |i: usize| report.evidence.get(i).and_then(Option::as_ref);
    let optional = |x: Option<f64>, spec: fn(f64) -> String| x.map_or("—".to_owned(), spec);
    let exp = |x: f64| format!("{x:.2e}");
    let two = |x: f64| format!("{x:.2}");

    let of_tier = |want: Tier| -> Vec<usize> {
        let mut found: Vec<usize> = representatives
            .iter()
            .copied()
            .filter(|&i| outcome.candidates[i].tier == want)
            .collect();
        found.sort_by(|&x, &y| {
            outcome.candidates[y]
                .score()
                .partial_cmp(&outcome.candidates[x].score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        found
    };
    // Task S2's colour columns exist only where the collection's files carry colour, which is
    // what keeps a colour-less collection's `report.md` the bytes it was: no evidence, no column.
    let coloured =
        representatives.iter().any(|&i| evidence_of(i).is_some_and(|e| e.colour.is_some()));
    // One row of the evidence table, shared by the confirmed and the probable list.
    let row = |i: usize| -> String {
        let candidate = &outcome.candidates[i];
        let (a, b) = pair_name(i);
        let s = &candidate.scores;
        let e = evidence_of(i);
        let mut line = format!(
            "| {a} | {b} | {:.2} | {:.1} | {:.2} | {:.4} | {:.3} | {:.4} | {} | {} | {} | {} | \
             {} | {} |",
            candidate.score(),
            s.seam,
            s.tight,
            s.gap,
            s.cont_n,
            s.pen,
            optional(e.and_then(|e| e.slide_t), exp),
            optional(e.and_then(|e| e.margin), two),
            e.map_or_else(|| "—".to_owned(), |e| e.support.to_string()),
            e.map_or_else(|| "—".to_owned(), |e| e.placements.to_string()),
            e.map_or_else(|| "—".to_owned(), |e| format!("{}/3", e.resample_accept)),
            optional(e.and_then(|e| e.determined_deg), exp),
        );
        if coloured {
            use std::fmt::Write as _;
            let colour = e.and_then(|e| e.colour.as_ref());
            let _ = write!(
                line,
                " {} | {} |",
                optional(colour.and_then(|c| c.frac_delta_e), |x| format!("{x:.1}")),
                optional(colour.and_then(|c| c.shell_hist), two),
            );
        }
        line
    };
    let head = if coloured {
        "| A | B | score | seam (t) | tight | gap (t) | normal agr. | penetration | slide (t) | \
         margin | support | placements | redraws | determined (deg) | fracture dE | shell hist |"
    } else {
        "| A | B | score | seam (t) | tight | gap (t) | normal agr. | penetration | slide (t) | \
         margin | support | placements | redraws | determined (deg) |"
    };
    let rule = if coloured {
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    } else {
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    };

    let confirmed = of_tier(Tier::Confirmed);
    lines.push(String::new());
    lines.push("## Confirmed joins".to_owned());
    lines.push(String::new());
    lines.push(format!(
        "A **confirmed** join clears R §6.5 and then a stricter set on top of it: tight ≥ {}, gap \
         ≤ {} t, seam ≥ {} t, normal agreement ≥ {}, penetration ≤ {}, the pose returns to within \
         {} t after being pushed half a wall along the seam, and either at least {} independent \
         join{} of the collection agree{} with where it puts the sherd or it beats the pair's \
         second placement by a factor of {}. The assembly above is built from these and from \
         nothing else.",
        th.min_tight,
        th.max_gap_t,
        th.min_seam,
        th.min_cont_n,
        th.max_pen,
        th.max_slide_t,
        th.min_support,
        if th.min_support == 1 { "" } else { "s" },
        if th.min_support == 1 { "s" } else { "" },
        th.min_margin,
    ));
    if coloured {
        lines.push(String::new());
        lines.push(
            "The last two columns are the colour the scans carry, and **no test above reads \
             them**: `fracture dE` is the CIE76 distance between the two sherds' bare clay — the \
             fabric each break shows, which a slip or a paint never covers — and `shell hist` is \
             how differently the two outer surfaces are coloured, on a scale where 0 is the same \
             photograph and 1 shares no colour at all. A large `fracture dE` says these two \
             sherds are unlikely to be one vessel; a large `shell hist` on a small `fracture dE` \
             says they are one vessel seen in two places, which is what a decorated pot looks \
             like."
                .to_owned(),
        );
    }
    lines.push(String::new());
    if confirmed.is_empty() {
        lines.push(
            "No candidate of this collection is confirmed. Every accepted join is listed as \
             probable below, with the test it failed."
                .to_owned(),
        );
    } else {
        lines.push(head.to_owned());
        lines.push(rule.to_owned());
        lines.extend(confirmed.iter().map(|&i| row(i)));
    }

    let (probable, probable_total) =
        crate::tiers::probable_shown(outcome.candidates, outcome.probable_top);
    let cut = probable_total > probable.len();
    lines.push(String::new());
    lines.push("## Probable joins".to_owned());
    lines.push(String::new());
    lines.push(
        "R §6.5 accepted these and the tier above did not confirm them. They are **not placed**: \
         the last column is the test each one failed, and the decision is a conservator's."
            .to_owned(),
    );
    lines.push(String::new());
    if probable.is_empty() {
        lines.push("Every accepted join of this collection is confirmed.".to_owned());
    } else {
        if cut {
            lines.push(format!(
                "**{} of {} shown**, best score first (`--probable-top {}`; `0` shows the whole \
                 band). The rows below the cut are in `report.json`, which is never cut, and they \
                 have no review image.",
                probable.len(),
                probable_total,
                outcome.probable_top,
            ));
            lines.push(String::new());
        }
        lines.push(format!("{head} why not confirmed |"));
        lines.push(format!("{rule}---|"));
        for &i in &probable {
            let why =
                evidence_of(i).map_or_else(|| "not probed".to_owned(), |e| e.failed.join("; "));
            lines.push(format!("{} {why} |", row(i)));
        }
    }

    let rejected = of_tier(Tier::Rejected);
    lines.push(String::new());
    lines.push("## Rejected".to_owned());
    lines.push(String::new());
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for &i in &rejected {
        for why in crate::matching::verify::refusals(&outcome.candidates[i].scores, params) {
            let limit = why.split_whitespace().next().unwrap_or("?").to_owned();
            *tally.entry(limit).or_default() += 1;
        }
    }
    lines.push(format!(
        "{} pair{} produced a candidate R §6.5 refused, for the reason it has today{}",
        rejected.len(),
        if rejected.len() == 1 { "" } else { "s" },
        if tally.is_empty() {
            ".".to_owned()
        } else {
            format!(
                ": {}. A pair can fail more than one limit.",
                tally.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", ")
            )
        },
    ));
    if !rejected.is_empty() {
        lines.push(String::new());
        lines.push(
            "| A | B | score | tight | gap (t) | gap limit (t) | seam (t) | normal agr. | \
             penetration | why |"
                .to_owned(),
        );
        lines.push("|---|---|---:|---:|---:|---:|---:|---:|---:|---|".to_owned());
        for &i in rejected.iter().take(REJECTED_LISTED) {
            let candidate = &outcome.candidates[i];
            let (a, b) = pair_name(i);
            let s = &candidate.scores;
            let why = crate::matching::verify::refusals(s, params);
            let why = if why.is_empty() { "refused by R §6.5".to_owned() } else { why.join("; ") };
            lines.push(format!(
                "| {a} | {b} | {:.2} | {:.2} | {:.4} | {:.4} | {:.1} | {:.3} | {:.4} | {why} |",
                candidate.score(),
                s.tight,
                s.gap,
                s.gap_limit,
                s.seam,
                s.cont_n,
                s.pen,
            ));
        }
        if rejected.len() > REJECTED_LISTED {
            lines.push(String::new());
            lines.push(format!(
                "The {} highest-scoring of {} refused pairs are listed; every one of them is in \
                 **Best candidate per pair** below.",
                REJECTED_LISTED,
                rejected.len()
            ));
        }
    }

    lines.push(String::new());
    lines.push("## Candidates by fragment".to_owned());
    lines.push(String::new());
    lines.push(
        "Every fragment with the partners R §6.5 accepted for it, best band first and best score \
         inside a band — the list to work through on the bench. A pair appears twice, once under \
         each of its two fragments. Partners R §6.5 refused are in **Rejected** above and in \
         **Best candidate per pair** below."
            .to_owned(),
    );
    if cut {
        lines.push(String::new());
        lines.push(format!(
            "The probable partners here are the same {} of {} rows the section above shows.",
            probable.len(),
            probable_total,
        ));
    }
    lines.push(String::new());
    // The image column exists only on a run that wrote images, which is what keeps a run without
    // `--review-images` byte for byte the run before the flag.
    let images = outcome.review;
    let shown: BTreeSet<usize> = probable.iter().copied().collect();
    let head = "| fragment | partner | tier | score | seam (t) | tight | gap (t) | slide (t) | \
                margin | support | why not confirmed |";
    let rule = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|";
    if images.is_some() {
        lines.push(format!("{head} image |"));
        lines.push(format!("{rule}---|"));
    } else {
        lines.push(head.to_owned());
        lines.push(rule.to_owned());
    }
    let mut listed = 0_usize;
    for (index, name) in names.iter().enumerate() {
        let here = u32::try_from(index).unwrap_or(u32::MAX);
        let mut mine: Vec<usize> = representatives
            .iter()
            .copied()
            .filter(|&i| {
                let candidate = &outcome.candidates[i];
                (candidate.a == here || candidate.b == here)
                    && match candidate.tier {
                        Tier::Confirmed => true,
                        Tier::Probable => shown.contains(&i),
                        Tier::Rejected => false,
                    }
            })
            .collect();
        mine.sort_by(|&x, &y| {
            let key = |i: usize| {
                let candidate = &outcome.candidates[i];
                (u8::from(candidate.tier != Tier::Confirmed), -candidate.score())
            };
            key(x).partial_cmp(&key(y)).unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in mine {
            let candidate = &outcome.candidates[i];
            let (into, moved) = pair_name(i);
            let partner = if candidate.a == here { moved } else { into };
            let e = evidence_of(i);
            let why = e.map_or_else(
                || "not probed".to_owned(),
                |e| if e.failed.is_empty() { "—".to_owned() } else { e.failed.join("; ") },
            );
            let row = format!(
                "| {name} | {partner} | {} | {:.2} | {:.1} | {:.2} | {:.4} | {} | {} | {} | {why} |",
                candidate.tier.label(),
                candidate.score(),
                candidate.scores.seam,
                candidate.scores.tight,
                candidate.scores.gap,
                optional(e.and_then(|e| e.slide_t), exp),
                optional(e.and_then(|e| e.margin), two),
                e.map_or_else(|| "—".to_owned(), |e| e.support.to_string()),
            );
            lines.push(match images {
                Some(index) => match index.get(&(candidate.a, candidate.b)) {
                    Some(file) => format!("{row} [png]({file}) |"),
                    None => format!("{row} — |"),
                },
                None => row,
            });
            listed += 1;
        }
    }
    if listed == 0 {
        let empty = "| — | — | — | — | — | — | — | — | — | — | no accepted candidate |";
        lines.push(match images {
            Some(_) => format!("{empty} — |"),
            None => empty.to_owned(),
        });
    }
    lines
}

/// The best candidate of every pair, pairs sorted by name — R §11.3's last table.
///
/// `max(cs, key=score)` on the reference's side returns the **first** maximum, which is the
/// candidate that came back highest ranked from R §5.7; `max_by` in Rust returns the last, so the
/// comparison below is strict.
fn best_per_pair(outcome: &Outcome<'_>) -> Vec<((String, String), usize)> {
    let names = outcome.names;
    let mut by_pair: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (i, c) in outcome.candidates.iter().enumerate() {
        let key = (names[c.a as usize].clone(), names[c.b as usize].clone());
        match by_pair.get(&key) {
            Some(&best) if outcome.candidates[best].score() >= c.score() => {}
            _ => {
                by_pair.insert(key, i);
            }
        }
    }
    by_pair.into_iter().collect()
}

/// R §11.3's legend line, built from the parameters it quotes.
fn legend(p: &Params) -> String {
    use std::fmt::Write as _;

    let f = python_float;
    let mut legend = format!(
        "Acceptance requires tight ≥ {}, gap ≤ max({} t, {} res), penetration ≤ {}, seam ≥ {}, \
         normal agreement ≥ {}; tight counts points within max({} t, {} res).",
        f(p.min_tight),
        f(p.max_gap),
        f(p.gap_res),
        f(p.max_pen),
        f(p.min_seam),
        f(p.min_cont_n),
        f(p.tight_delta),
        f(p.tight_res)
    );
    if p.early_reject_tight > 0.0 {
        let _ = write!(
            legend,
            " n/a = not computed: the candidate was rejected early (tight below {}).",
            f(p.early_reject_tight)
        );
    }
    if p.stage1_floor > 0.0 {
        let _ = write!(
            legend,
            " A pair whose best breakline score after stage 1 stays below {} is not taken to \
             stage 2 at all; its row shows that pose with n/a scores.",
            f(p.stage1_floor)
        );
    }
    legend
}

/// `str(x)` for a Python float, which is what R §11.3's legend interpolates.
///
/// Rust's `{}` and Python's `str` both print the shortest decimal that round-trips, and they agree
/// on every value the legend quotes but one: an **integral** float is `3.0` in Python and `3` in
/// Rust. `min_seam` is 3.0, so the difference shows on every report the pipeline writes, and this
/// is where it is removed — measured by diffing the port's `report.md` against the reference's own
/// on terracotta, where it was the only line left. Values small enough for Python to switch to
/// exponent form (below 1e-4) are outside the parameters this is used on and keep Rust's form.
fn python_float(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn join_names(ids: &[FragId], names: &[String]) -> String {
    ids.iter().map(|&n| names[n as usize].as_str()).collect::<Vec<_>>().join(", ")
}

/// `str(True)` / `str(False)` — the reference's f-string interpolation of a Python bool.
fn python_bool(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

/// R §11.4's `placed/<name>.ply` and `assembly_<k>.ply`.
///
/// The meshes are read one at a time and never two at once, which is D §5's rule and the
/// reference's own: a collection of 164 scans is hundreds of megabytes at full resolution, and the
/// merged file is streamed element by element through [`PlyStream`] rather than concatenated in
/// memory.
///
/// `comment` is the PLY header's comment line. The pipeline passes
/// [`DEFAULT_COMMENT`](crate::io::writer::DEFAULT_COMMENT); the parity harness passes
/// [`OPEN3D_COMMENT`](crate::io::writer::OPEN3D_COMMENT), which is what makes the file
/// byte-comparable with the reference's.
pub fn write_placed_meshes(
    out_dir: impl AsRef<Path>,
    paths: &[PathBuf],
    names: &[String],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
    comment: &str,
    budget: Budget,
) -> Result<Vec<PathBuf>> {
    let out_dir = out_dir.as_ref();
    let placed_dir = out_dir.join("placed");
    std::fs::create_dir_all(&placed_dir).map_err(|e| Error::write(&placed_dir, e))?;
    // The placed meshes are independent files with fixed names, so they are written in parallel
    // and their results collected by index: the same bytes in the same files in the same order,
    // and the first failure in *fragment* order is the one the run reports. E1 §9 measured this
    // stage at 1.6 % of the machine's work but 12 % of the user's wait, because one thread read,
    // transformed and wrote one full-resolution mesh at a time.
    //
    // The semaphore of D §5 step 2 bounds what that costs in memory: `place` holds one original
    // scan, which is the same thing preprocessing reserves for, so the same budget prices it.
    let semaphore = MemorySemaphore::new(budget);
    let placed: Vec<Result<PathBuf>> = (0..paths.len())
        .into_par_iter()
        .map(|n| {
            let path = &paths[n];
            let _permit = semaphore.acquire(memory::scan_faces(path).map_or(0, reservation));
            let mesh = place(path, &poses[n])?;
            let file = placed_dir.join(format!("{}.ply", names[n]));
            crate::io::writer::write_ply_with_comment(&file, &mesh, comment)?;
            Ok(file)
        })
        .collect();
    let mut written = Vec::with_capacity(placed.len());
    for file in placed {
        written.push(file?);
    }
    for (k, group) in groups.iter().enumerate() {
        if group.len() < 2 {
            continue;
        }
        // The members are placed in parallel and merged in group order, which is the order
        // R §11.4's vertex and face blocks are written in and therefore the file's own.
        let loaded: Vec<Result<Mesh>> = group
            .par_iter()
            .map(|&n| {
                let path = &paths[n as usize];
                let _permit = semaphore.acquire(memory::scan_faces(path).map_or(0, reservation));
                place(path, &poses[n as usize])
            })
            .collect();
        let mut members = Vec::with_capacity(loaded.len());
        for mesh in loaded {
            members.push(mesh?);
        }
        let file = out_dir.join(format!("assembly_{k}.ply"));
        write_merged(&file, &members, comment)?;
        written.push(file);
    }
    Ok(written)
}

/// One original mesh, cleaned as R §3.1 steps 1–2 and moved by its pose.
///
/// **Not** reduced to the largest component: R §11.4 writes what the file held, and a fragment
/// that arrives as two shells keeps both.
pub fn place(path: impl AsRef<Path>, pose: &Matrix4<f64>) -> Result<Mesh> {
    let mut mesh = crate::io::load_mesh(path)?;
    for v in &mut mesh.v {
        *v = apply_transform_fused(pose, *v);
    }
    Ok(mesh)
}

/// R §11.4's `assembly_<k>.ply`: the members' placed meshes concatenated, indices offset.
///
/// Colours follow Open3D's `operator+=`: the merged mesh keeps them only while **every** member so
/// far has had them, and the first member without clears them for good.
pub fn write_merged(path: impl AsRef<Path>, members: &[Mesh], comment: &str) -> Result<()> {
    let path = path.as_ref();
    let n_vertices: usize = members.iter().map(Mesh::n_vertices).sum();
    let n_faces: usize = members.iter().map(Mesh::n_faces).sum();
    let colors = !members.is_empty() && members.iter().all(Mesh::has_colors);
    let file = std::fs::File::create(path).map_err(|e| Error::write(path, e))?;
    let mut stream = PlyStream::begin(
        std::io::BufWriter::with_capacity(1 << 20, file),
        n_vertices,
        n_faces,
        colors,
        comment,
    )
    .map_err(|e| Error::write(path, e))?;
    for mesh in members {
        stream.write_vertices(mesh).map_err(|e| Error::write(path, e))?;
    }
    let mut offset = 0_u32;
    for mesh in members {
        stream.write_faces(mesh, offset).map_err(|e| Error::write(path, e))?;
        offset += u32::try_from(mesh.n_vertices()).unwrap_or(u32::MAX);
    }
    stream.finish().map_err(|e| Error::write(path, e))?;
    Ok(())
}

/// `json.dump(..., indent=1)`, which is what both JSON outputs use.
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut buffer = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut serialiser = serde_json::Serializer::with_formatter(&mut buffer, formatter);
    value.serialize(&mut serialiser).map_err(|e| Error::write(path, e))?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| Error::write(path, e))?;
    }
    std::fs::write(path, buffer).map_err(|e| Error::write(path, e))
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateJson, FragmentStats, Outcome, ReportJson, Transforms, report_json,
        report_markdown, transforms, write_merged, write_placed_meshes,
    };
    use crate::io::writer::{DEFAULT_COMMENT, OPEN3D_COMMENT};
    use crate::matching::pair::Candidate;
    use crate::matching::verify::Scores;
    use crate::mesh::Mesh;
    use crate::params::Params;
    use crate::tiers::Tier;
    use nalgebra::Matrix4;

    fn names() -> Vec<String> {
        vec!["one".to_owned(), "two".to_owned(), "three".to_owned()]
    }

    fn candidate(a: u32, b: u32, seam: f64, tight: f64) -> Candidate {
        let scores = Scores { seam, tight, tight_a: tight, tight_b: tight, ..Scores::default() };
        let accepted = seam > 0.0;
        Candidate {
            a,
            b,
            transform: Matrix4::identity(),
            scores,
            accepted,
            tier: Tier::of_accept(accepted),
        }
    }

    fn tetra() -> Mesh {
        Mesh {
            v: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            f: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            colors: None,
        }
    }

    /// `transforms.json` round-trips: every pose, group and `placed` flag comes back as it went
    /// in, and R §11.1's `placed` is "a group of two or more" rather than "the pose is not the
    /// identity".
    #[test]
    fn transforms_round_trip() {
        let mut poses = vec![Matrix4::identity(); 3];
        poses[1][(0, 3)] = 12.5;
        let groups = vec![vec![0_u32, 1], vec![2]];
        // R §8's insertion order: the seed's two, then the singleton — deliberately not the
        // collection order, so that the file's own key order is the thing under test.
        let value = transforms(
            &names(),
            &poses,
            &groups,
            &[1, 0, 2],
            3.75,
            &Params::default(),
            Some("gpu:Apple M2 Pro"),
            None,
        );
        assert_eq!(value.fragments.keys().collect::<Vec<&str>>(), ["two", "one", "three"]);
        let text = serde_json::to_string(&value).expect("transforms serialise");
        let back: Transforms = serde_json::from_str(&text).expect("transforms parse");

        assert!((back.thickness - 3.75).abs() < 1e-12);
        assert_eq!(back.groups, vec![vec!["one", "two"], vec!["three"]]);
        assert_eq!(back.fragments["one"].group, 0);
        assert!(back.fragments["one"].placed);
        assert!(!back.fragments["three"].placed, "a singleton is not placed");
        assert!((back.fragments["two"].matrix[0][3] - 12.5).abs() < 1e-12);
        assert_eq!(back.params, Params::default());
        // Row major, as R §11.1's nested lists are.
        assert_eq!(
            back.fragments["three"].matrix[3].map(f64::to_bits),
            [0.0_f64, 0.0, 0.0, 1.0].map(f64::to_bits)
        );

        // D §4.3's block, in this file as well as in `report.json` (V6-D7): the resolved backend
        // with its adapter, and every version this build carries.
        let engine = back.engine.expect("transforms.json carries D §4.3's engine block");
        assert_eq!(engine.backend, "gpu:Apple M2 Pro");
        assert_eq!(engine.seed, 0, "R §10's seed, as `--seed` resolved it (task H3)");
        assert_eq!(engine.algo_ref, crate::ALGO_REF);
        assert_eq!(engine.core_version, crate::CORE_VERSION);
        assert_eq!(engine.cache_version, crate::CACHE_VERSION);
        assert_eq!(engine.commit, crate::GIT_COMMIT);
        // A file the *reference* wrote has no `engine`, and this type has to read those too.
        let without = serde_json::json!({
            "thickness": 1.0,
            "params": Params::default(),
            "fragments": {},
            "groups": [],
        });
        let back: Transforms =
            serde_json::from_value(without).expect("a reference file parses without `engine`");
        assert!(back.engine.is_none());
        assert!(
            !serde_json::to_string(&back).expect("serialise").contains("engine"),
            "and comes back out without one",
        );
    }

    /// R §11.2's `timings` keeps the order the stages finished in, in both files.
    ///
    /// The reference's is a Python dict and `json.dump` writes it in insertion order; a port that
    /// sorted it by name wrote `assembly, matching, preprocess, refine` where the reference wrote
    /// `preprocess, matching, assembly, refine` (V4-D4). The order is deliberately not
    /// alphabetical here, so that sorting anywhere fails this.
    #[test]
    fn timings_keep_the_order_the_stages_finished_in() {
        let outcome = Outcome {
            names: &names(),
            candidates: &[],
            used: &[],
            rejected: &[],
            groups: &[vec![0], vec![1], vec![2]],
            tiers: None,
            constraints: None,
            review: None,
            objects: None,
            probable_top: 0,
        };
        let timings = super::Timings::from_iter([
            ("preprocess".to_owned(), 16.3),
            ("matching".to_owned(), 12.0),
            ("assembly".to_owned(), 1.8),
            ("refine".to_owned(), 20.3),
        ]);
        let stats = Vec::new();
        let params = Params::default();

        let md = report_markdown(&stats, 3.75, &outcome, &timings, &params);
        let tail: Vec<&str> =
            md.lines().skip_while(|l| *l != "## Timing").skip(2).collect::<Vec<&str>>();
        assert_eq!(
            tail,
            ["- preprocess: 16.3 s", "- matching: 12.0 s", "- assembly: 1.8 s", "- refine: 20.3 s"]
        );

        let stages = ["preprocess", "matching", "assembly", "refine"];
        let json = report_json(&stats, 3.75, &outcome, &timings, &params, "cpu", None);
        assert_eq!(json.timings.keys().collect::<Vec<&str>>(), stages);

        // And serde writes the object in that order rather than the type merely holding it.
        let text = serde_json::to_string(&json).expect("report serialises");
        let at = text.find("\"timings\"").expect("a timings object");
        let at = stages.map(|k| text[at..].find(k).expect("the stage"));
        assert!(at.windows(2).all(|w| w[0] < w[1]), "{text}");
    }

    /// A candidate flattens into `report.json` exactly as `Candidate.to_json()` does: the five
    /// named keys plus every score at the top level, and `reason` only on a rejection.
    #[test]
    fn report_json_flattens_a_candidate_the_way_the_reference_does() {
        let cands =
            [candidate(0, 1, 20.0, 0.5), candidate(0, 1, 4.0, 0.5), candidate(1, 2, 0.0, 0.0)];
        let rejected = [(1_usize, "would merge two groups (not supported)".to_owned())];
        let outcome = Outcome {
            names: &names(),
            candidates: &cands,
            used: &[0],
            rejected: &rejected,
            groups: &[vec![0, 1], vec![2]],
            tiers: None,
            constraints: None,
            review: None,
            objects: None,
            probable_top: 0,
        };
        let stats = Vec::new();
        let timings = super::Timings::from_iter([("matching".to_owned(), 1.25)]);
        let json = report_json(&stats, 3.75, &outcome, &timings, &Params::default(), "cpu", None);
        let text = serde_json::to_string(&json).expect("report serialises");
        let value: serde_json::Value = serde_json::from_str(&text).expect("report parses");

        let used = &value["joins_used"][0];
        assert_eq!(used["a"], "one");
        assert_eq!(used["b"], "two");
        assert_eq!(used["accepted"], true);
        assert!((used["score"].as_f64().expect("a score") - 10.0).abs() < 1e-12);
        assert!((used["seam"].as_f64().expect("seam") - 20.0).abs() < 1e-12);
        assert!(used["reason"].is_null(), "a used join carries no reason");
        assert_eq!(value["joins_rejected"][0]["reason"], "would merge two groups (not supported)");
        assert_eq!(value["candidates"].as_array().expect("candidates").len(), 3);
        assert_eq!(value["engine"]["backend"], "cpu");
        assert_eq!(value["engine"]["algo_ref"], crate::ALGO_REF);
        assert_eq!(value["engine"]["core_version"], crate::CORE_VERSION);
        assert_eq!(value["engine"]["cache_version"], crate::CACHE_VERSION);
        assert_eq!(value["engine"]["commit"], crate::GIT_COMMIT);
        assert_eq!(value["engine"]["seed"], 0);

        // And it parses back into the typed form the harness reads.
        let back: ReportJson = serde_json::from_str(&text).expect("report round-trips");
        assert_eq!(back.joins_used.len(), 1);
        assert!((back.candidates[0].scores.seam - 20.0).abs() < 1e-12);
        let one: CandidateJson = serde_json::from_value(used.clone()).expect("one candidate");
        assert_eq!(one.a, "one");
    }

    /// R §11.3's sections, in order, with the number formats it names.
    #[test]
    fn report_markdown_has_the_sections_and_the_formats() {
        let stats = vec![FragmentStats {
            name: "one".to_owned(),
            faces: 1000,
            orig_faces: 5000,
            orig_vertices: 2600,
            thickness: 3.75,
            thickness_mode: 4.125,
            resolution: 0.25,
            watertight: true,
            extent: [10.4, 20.6, 3.2],
            area: 500.0,
            fracture_area_fraction: 0.135,
        }];
        let cands = [candidate(0, 1, 20.0, 0.5), candidate(1, 2, 3.0, 0.25)];
        let rejected = [(1_usize, "penetrates two (0.127)".to_owned())];
        let outcome = Outcome {
            names: &names(),
            candidates: &cands,
            used: &[0],
            rejected: &rejected,
            groups: &[vec![0, 1], vec![2]],
            tiers: None,
            constraints: None,
            review: None,
            objects: None,
            probable_top: 0,
        };
        let timings = super::Timings::from_iter([("matching".to_owned(), 12.34)]);
        let md = report_markdown(&stats, 3.75, &outcome, &timings, &Params::default());

        let sections: Vec<&str> = md.lines().filter(|l| l.starts_with('#')).collect();
        assert_eq!(
            sections,
            vec![
                "# Reassembly report",
                "## Fragments",
                "## Assembly",
                "## Joins used",
                "## Accepted joins not used",
                "## Best candidate per pair",
                "## Timing",
            ]
        );
        assert!(md.contains("Wall thickness (collection median): 3.75 units."), "{md}");
        assert!(md.contains("| one | 1000 (5000) | 3.75 | 4.12 | 1.00 | 0.250 | 15.0 | 13.5 | True | 10 x 21 x 3 |"), "{md}");
        assert!(md.contains("- group 0: one, two"), "{md}");
        assert!(md.contains("- not assembled (no confident join): three"), "{md}");
        assert!(md.contains("- two – three (score 0.75): penetrates two (0.127)"), "{md}");
        assert!(md.contains("- matching: 12.3 s"), "{md}");
        // A fragment more than 40 % off the median is flagged.
        let mut thin = stats;
        thin[0].thickness = 1.0;
        let md = report_markdown(&thin, 3.75, &outcome, &timings, &Params::default());
        assert!(md.contains("1.00 **(differs)**"), "{md}");
    }

    /// The writer on a placed mesh: `placed/<name>.ply` for every fragment, `assembly_<k>.ply` for
    /// every group of two or more, and the merged file's second member's indices offset.
    #[test]
    fn the_writer_places_every_mesh_and_merges_the_groups() {
        let dir = std::env::temp_dir().join(format!("sherd-report-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let source = dir.join("piece.ply");
        crate::io::writer::write_ply(&source, &tetra()).expect("the source writes");

        let mut poses = vec![Matrix4::identity(); 3];
        poses[1][(0, 3)] = 10.0;
        let paths = vec![source.clone(), source.clone(), source];
        let out = dir.join("out");
        let written = write_placed_meshes(
            &out,
            &paths,
            &names(),
            &poses,
            &[vec![0, 1], vec![2]],
            OPEN3D_COMMENT,
            crate::memory::Budget::default_for_machine(),
        )
        .expect("the placed meshes write");
        assert_eq!(written.len(), 4, "three placed files and one assembly");
        assert!(out.join("placed/one.ply").is_file());
        assert!(out.join("assembly_0.ply").is_file());
        assert!(!out.join("assembly_1.ply").exists(), "a singleton gets no merged file");

        let placed = crate::io::ply::read(out.join("placed/two.ply")).expect("it reads back");
        assert!((placed.v[1][0] - 11.0).abs() < 1e-12, "the pose moved the mesh");
        let merged = crate::io::ply::read(out.join("assembly_0.ply")).expect("it reads back");
        assert_eq!(merged.n_vertices(), 8);
        assert_eq!(merged.f[4], [4, 6, 5], "the second member's indices are offset");

        // The comment is the only thing that separates the port's own file from the reference's.
        let mine = dir.join("mine.ply");
        write_merged(&mine, &[tetra()], DEFAULT_COMMENT).expect("write");
        let theirs = dir.join("theirs.ply");
        write_merged(&theirs, &[tetra()], OPEN3D_COMMENT).expect("write");
        let (a, b) = (std::fs::read(&mine).expect("read"), std::fs::read(&theirs).expect("read"));
        let cut = |v: &[u8]| {
            let end = b"end_header\n";
            let at = v.windows(end.len()).position(|w| w == end).expect("a header") + end.len();
            v[at..].to_vec()
        };
        assert_eq!(cut(&a), cut(&b), "only the header differs");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Roadmap item 3's four sections, and the fact that they are absent without a tier.
    ///
    /// The band is the same `Outcome` with and without `tiers`, so this is the off switch stated
    /// at the level `report.md` is written at: the same call, one field apart, and the sections
    /// appear or they do not.
    #[test]
    fn the_tier_sections_appear_only_when_a_run_computed_a_band() {
        use crate::tiers::{Evidence, Thresholds, TierReport};

        // The pipeline copies `TierReport::tiers` onto the candidates before it writes anything
        // (`pipeline::tier_pass`), so the two agree by construction; the fixture does the same.
        let banded = |a, b, seam, tight, tier| Candidate { tier, ..candidate(a, b, seam, tight) };
        let cands = [
            banded(0, 1, 20.0, 0.5, Tier::Confirmed),
            banded(1, 2, 3.0, 0.25, Tier::Probable),
            banded(0, 2, 0.0, 0.0, Tier::Rejected),
        ];
        let evidence = |failed: Vec<String>| Evidence {
            colour: None,
            margin: Some(4.5),
            rival_moved_t: Some(9.0),
            placements: 2,
            determined_deg: Some(2.7e-14),
            determined_t: Some(1.8e-15),
            slide_t: Some(1.9e-15),
            resample_tight_min: 0.49,
            resample_gap_max: 0.008,
            resample_accept: 3,
            support: 1,
            failed,
        };
        let report = TierReport {
            thresholds: Thresholds::default(),
            tiers: vec![Tier::Confirmed, Tier::Probable, Tier::Rejected],
            evidence: vec![
                Some(evidence(Vec::new())),
                Some(evidence(vec!["tight 0.2500 < 0.35".to_owned()])),
                None,
            ],
            probes: vec![None, None, None],
        };
        let stats = Vec::new();
        let timings = super::Timings::from_iter([("matching".to_owned(), 1.0)]);
        let make = |tiers: Option<&TierReport>| {
            let outcome = Outcome {
                names: &names(),
                candidates: &cands,
                used: &[0],
                rejected: &[],
                groups: &[vec![0, 1], vec![2]],
                tiers,
                constraints: None,
                review: None,
                objects: None,
                probable_top: 0,
            };
            report_markdown(&stats, 3.75, &outcome, &timings, &Params::default())
        };

        let without = make(None);
        for section in ["## Confirmed joins", "## Probable joins", "## Candidates by fragment"] {
            assert!(!without.contains(section), "{section} without a tier");
        }

        let with = make(Some(&report));
        let sections: Vec<&str> = with.lines().filter(|l| l.starts_with("## ")).collect();
        assert_eq!(
            sections,
            [
                "## Fragments",
                "## Assembly",
                "## Joins used",
                "## Confirmed joins",
                "## Probable joins",
                "## Rejected",
                "## Candidates by fragment",
                "## Best candidate per pair",
                "## Timing",
            ],
            "the order a bench reads them in"
        );
        assert!(with.contains("tight ≥ 0.35, gap ≤ 0.015 t, seam ≥ 5 t"), "{with}");
        let confirmed = with
            .split("## Confirmed joins")
            .nth(1)
            .and_then(|t| t.lines().find(|l| l.starts_with("| one | two |")))
            .expect("the confirmed row");
        assert!(confirmed.starts_with("| one | two | 10.00 | 20.0 | 0.50 |"), "{confirmed}");
        assert!(confirmed.contains("| 1.90e-15 | 4.50 | 1 | 2 | 3/3 | 2.70e-14 |"), "{confirmed}");
        assert!(with.contains("tight 0.2500 < 0.35 |"), "the probable row names its refusal");
        assert!(with.contains("1 pair produced a candidate R §6.5 refused"), "{with}");
        // The per-fragment index shows a pair once under each of its two fragments, and never a
        // partner R §6.5 refused.
        let index = with.split("## Candidates by fragment").nth(1).expect("the index");
        let index = index.split("## Best candidate").next().expect("its end");
        assert_eq!(index.matches("| confirmed |").count(), 2);
        assert_eq!(index.matches("| probable |").count(), 2);
        assert_eq!(index.matches("| rejected |").count(), 0);
    }

    /// `--probable-top N`: the Probable section and the per-fragment index show the same best `N`
    /// rows and say how long the band was, and `0` shows all of it.
    ///
    /// The cut is a report decision and nothing else: the three candidates are the same three in
    /// both calls, and `report.json` is written from the same list whatever the flag says.
    #[test]
    fn probable_top_cuts_the_band_in_both_places_and_prints_the_total() {
        use crate::tiers::{Evidence, Thresholds, TierReport};

        let banded =
            |a, b, seam, tight| Candidate { tier: Tier::Probable, ..candidate(a, b, seam, tight) };
        // Scores 20, 10 and 5 -- deliberately out of pair order, so a list that came back sorted
        // by pair rather than by score would fail here.
        let cands = [banded(0, 1, 20.0, 0.5), banded(0, 2, 10.0, 0.5), banded(1, 2, 40.0, 0.5)];
        let evidence = Evidence {
            colour: None,
            margin: None,
            rival_moved_t: None,
            placements: 1,
            determined_deg: None,
            determined_t: None,
            slide_t: Some(1.9e-15),
            resample_tight_min: 0.49,
            resample_gap_max: 0.008,
            resample_accept: 3,
            support: 0,
            failed: vec!["tight 0.5000 < 0.35".to_owned()],
        };
        let report = TierReport {
            thresholds: Thresholds::default(),
            tiers: vec![Tier::Probable; 3],
            evidence: vec![Some(evidence.clone()), Some(evidence.clone()), Some(evidence)],
            probes: vec![None, None, None],
        };
        let timings = super::Timings::from_iter([("matching".to_owned(), 1.0)]);
        let make = |top: usize| {
            let outcome = Outcome {
                names: &names(),
                candidates: &cands,
                used: &[],
                rejected: &[],
                groups: &[vec![0], vec![1], vec![2]],
                tiers: Some(&report),
                constraints: None,
                review: None,
                objects: None,
                probable_top: top,
            };
            report_markdown(&Vec::new(), 3.75, &outcome, &timings, &Params::default())
        };
        let band = |md: &str| {
            md.split("## Probable joins")
                .nth(1)
                .and_then(|t| t.split("## Rejected").next().map(str::to_owned))
                .expect("the section")
        };
        let index = |md: &str| {
            md.split("## Candidates by fragment")
                .nth(1)
                .and_then(|t| t.split("## Best candidate").next().map(str::to_owned))
                .expect("the index")
        };

        let all = make(0);
        assert_eq!(band(&all).matches("| 1.90e-15 |").count(), 3, "every row");
        assert!(!all.contains("shown**"), "nothing is cut, so nothing is said about a cut");
        assert_eq!(index(&all).matches("| probable |").count(), 6, "each pair under both names");

        let cut = make(2);
        let section = band(&cut);
        let rows: Vec<&str> = section.lines().filter(|l| l.contains("| 1.90e-15 |")).collect();
        assert_eq!(rows.len(), 2, "the best two");
        assert!(rows[0].starts_with("| two | three | 20.00 |"), "best score first: {rows:?}");
        assert!(rows[1].starts_with("| one | two | 10.00 |"), "{rows:?}");
        assert!(cut.contains("**2 of 3 shown**"), "the total is printed: {section}");
        assert!(cut.contains("`report.json`, which is never cut"), "where the rest is");
        let index = index(&cut);
        assert_eq!(index.matches("| probable |").count(), 4, "the same two pairs, twice each");
        assert!(
            index.contains("the same 2 of 3 rows the section above shows"),
            "the index says it too: {index}"
        );
        assert!(!index.contains("| one | three |"), "the cut pair is in neither place");
    }
}
