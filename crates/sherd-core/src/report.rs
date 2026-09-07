//! `transforms.json`, `report.json`, `report.md` and the placed meshes (R §11.1–11.4).
//!
//! Same file names, same JSON keys and the same section order as the reference, so that
//! `tools/evaluate.py` scores a Rust run without knowing which implementation wrote it, plus the
//! additive `engine` key that records `core_version`, `algo_ref` and the backend that actually ran
//! (D §4.3); the Python readers ignore keys they do not know.
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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nalgebra::Matrix4;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fragment::Fragment;
use crate::io::writer::PlyStream;
use crate::matching::pair::Candidate;
use crate::matching::verify::Scores;
use crate::memory::{self, Budget, MemorySemaphore, reservation};
use crate::mesh::Mesh;
use crate::params::Params;
use crate::types::{FragId, apply_transform_fused};
use crate::{ALGO_REF, CORE_VERSION};

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
        }
    }

    /// The same with R §8's rejection sentence attached.
    pub fn rejected(candidate: &Candidate, names: &[String], reason: &str) -> Self {
        Self { reason: Some(reason.to_owned()), ..Self::of(candidate, names) }
    }
}

/// D §4.3's additive block: which build wrote the file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Engine {
    /// `sherd-core`'s crate version.
    pub core_version: String,
    /// The frozen algorithm this port reproduces.
    pub algo_ref: String,
    /// The executor that ran.
    pub backend: String,
}

impl Engine {
    /// The block for a run on `backend`.
    pub fn of(backend: &str) -> Self {
        Self {
            core_version: CORE_VERSION.to_owned(),
            algo_ref: ALGO_REF.to_owned(),
            backend: backend.to_owned(),
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
pub fn write_transforms(
    path: impl AsRef<Path>,
    names: &[String],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
    order: &[FragId],
    thickness: f64,
    params: &Params,
) -> Result<()> {
    write_json(path.as_ref(), &transforms(names, poses, groups, order, thickness, params))
}

/// The value [`write_transforms`] serialises, for callers that want it in memory.
pub fn transforms(
    names: &[String],
    poses: &[Matrix4<f64>],
    groups: &[Vec<FragId>],
    order: &[FragId],
    thickness: f64,
    params: &Params,
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
}

/// R §11.2's `report.json` and R §11.3's `report.md`, both into `out_dir`.
pub fn write_report(
    out_dir: impl AsRef<Path>,
    stats: &[FragmentStats],
    thickness: f64,
    outcome: &Outcome<'_>,
    timings: &Timings,
    params: &Params,
    backend: &str,
) -> Result<()> {
    let out_dir = out_dir.as_ref();
    let json = report_json(stats, thickness, outcome, timings, params, backend);
    write_json(&out_dir.join("report.json"), &json)?;
    let markdown = report_markdown(stats, thickness, outcome, timings, params);
    std::fs::write(out_dir.join("report.md"), markdown)
        .map_err(|e| Error::write(out_dir.join("report.md"), e))
}

/// The value [`write_report`] serialises.
pub fn report_json(
    stats: &[FragmentStats],
    thickness: f64,
    outcome: &Outcome<'_>,
    timings: &Timings,
    params: &Params,
    backend: &str,
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
            .map(|&i| CandidateJson::of(&outcome.candidates[i], names))
            .collect(),
        joins_rejected: outcome
            .rejected
            .iter()
            .map(|(i, why)| CandidateJson::rejected(&outcome.candidates[*i], names, why))
            .collect(),
        candidates: outcome.candidates.iter().map(|c| CandidateJson::of(c, names)).collect(),
        engine: Some(Engine::of(backend)),
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
    if !outcome.rejected.is_empty() {
        lines.push(String::new());
        lines.push("## Accepted joins not used".to_owned());
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
    use nalgebra::Matrix4;

    fn names() -> Vec<String> {
        vec!["one".to_owned(), "two".to_owned(), "three".to_owned()]
    }

    fn candidate(a: u32, b: u32, seam: f64, tight: f64) -> Candidate {
        let scores = Scores { seam, tight, tight_a: tight, tight_b: tight, ..Scores::default() };
        Candidate { a, b, transform: Matrix4::identity(), scores, accepted: seam > 0.0 }
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
        let value = transforms(&names(), &poses, &groups, &[1, 0, 2], 3.75, &Params::default());
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
        let json = report_json(&stats, 3.75, &outcome, &timings, &params, "cpu");
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
        };
        let stats = Vec::new();
        let timings = super::Timings::from_iter([("matching".to_owned(), 1.25)]);
        let json = report_json(&stats, 3.75, &outcome, &timings, &Params::default(), "cpu");
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
}
