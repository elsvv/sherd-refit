//! Files for the people who open a result, and for the programs they open it in.
//!
//! The museum's request of 2026-09-11: a conservator has to **recognise** every fragment of an
//! assembly — its number is in its file name — ideally by pointing at it, and has to be able to
//! apply the placement in their own 3D software (Geomagic Wrap, which runs Python) to their own
//! files. R §11's outputs answer neither: `transforms.json` nests the matrices in a JSON tree,
//! `assembly_<k>.ply` merges a group into one mesh and loses the names, and R §8.2 centres every
//! group on the same origin, so two groups opened together lie inside each other.
//!
//! Everything here is **additive**. It reads what R §11 already writes from — the poses after
//! R §8.2, the groups, the candidates and their bands — and changes none of it. A fragment's
//! matrix is the same 4 × 4 in every file: it maps the fragment's **original file** to its place
//! in the assembly, `p' = M · p` on column vectors, in the units of the scan.
//!
//! * [`tables`] — `transforms.csv`, `joins.csv` and `matrices/<name>.txt`;
//! * [`scene`] — `scene.glb`: simplified meshes under one named node per fragment carrying that
//!   matrix, with the groups set apart;
//! * [`viewer`] — `viewer.html`: the same scene in a browser, the fragment's name under the cursor;
//! * [`readme`] — `README.txt`: what is in the folder, in Russian, for the person who opens it.

pub mod readme;
pub mod scene;
pub mod tables;
pub mod viewer;

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use nalgebra::Matrix4;

use crate::error::{Error, Result};
use crate::matching::pair::Candidate;
use crate::review::ReviewIndex;
use crate::tiers::{Tier, representatives};
use crate::types::FragId;

/// What a run hands the exporters: R §11's own inputs, read and never changed.
#[derive(Clone, Copy, Debug)]
pub struct Export<'a> {
    /// A short name for the collection, for titles ([`collection_title`]).
    pub collection: &'a str,
    /// The fragments' names — their file stems — by [`FragId`].
    pub names: &'a [String],
    /// The original files, by [`FragId`].
    pub sources: &'a [PathBuf],
    /// The final poses, `transforms.json`'s matrices, by [`FragId`].
    pub poses: &'a [Matrix4<f64>],
    /// R §8's groups, largest first, the singletons included.
    pub groups: &'a [Vec<FragId>],
    /// Every candidate of the run, with its band.
    pub candidates: &'a [Candidate],
    /// The candidates R §8 built with, as indices into `candidates`.
    pub used: &'a [usize],
    /// Whether the tier pass ran; without it a candidate's band is R §6.5's own verdict.
    pub tiers: bool,
    /// The review images the run drew, by pair.
    pub review: Option<&'a ReviewIndex>,
    /// The collection's median wall thickness, in the units of the scans.
    pub thickness: f64,
}

impl Export<'_> {
    /// The group of every fragment, by [`FragId`]; a fragment in no group gets `usize::MAX`.
    pub fn group_of(&self) -> Vec<usize> {
        let mut group = vec![usize::MAX; self.names.len()];
        for (k, members) in self.groups.iter().enumerate() {
            for &n in members {
                group[n as usize] = k;
            }
        }
        group
    }

    /// Whether each fragment sits in a group of two or more — `transforms.json`'s `placed`.
    pub fn assembled(&self) -> Vec<bool> {
        let mut assembled = vec![false; self.names.len()];
        for members in self.groups.iter().filter(|g| g.len() > 1) {
            for &n in members {
                assembled[n as usize] = true;
            }
        }
        assembled
    }

    /// The name of fragment `n`'s original file, without its directory.
    pub fn source_name(&self, n: usize) -> String {
        self.sources[n]
            .file_name()
            .map_or_else(|| self.names[n].clone(), |s| s.to_string_lossy().into_owned())
    }

    /// The joins a person may want to look at: every pair whose best candidate is confirmed or
    /// probable (with the tier pass off, every pair R §6.5 accepted), strongest band first and
    /// best score first inside it.
    pub fn joins(&self) -> Vec<JoinRow> {
        let used: BTreeSet<(FragId, FragId)> =
            self.used.iter().map(|&i| (self.candidates[i].a, self.candidates[i].b)).collect();
        let group_of = self.group_of();
        let mut rows: Vec<(u8, usize, JoinRow)> = representatives(self.candidates)
            .into_iter()
            .filter_map(|i| {
                let c = &self.candidates[i];
                let (rank, band) = if self.tiers {
                    match c.tier {
                        Tier::Confirmed => (0, Tier::Confirmed.label()),
                        Tier::Probable => (1, Tier::Probable.label()),
                        Tier::Rejected => return None,
                    }
                } else if c.accepted {
                    (1, "accepted")
                } else {
                    return None;
                };
                let (ga, gb) = (group_of[c.a as usize], group_of[c.b as usize]);
                let group =
                    (ga == gb && self.groups.get(ga).is_some_and(|g| g.len() > 1)).then_some(ga);
                let s = &c.scores;
                let row = JoinRow {
                    a: c.a,
                    b: c.b,
                    band,
                    in_assembly: used.contains(&(c.a, c.b)),
                    group,
                    score: c.score(),
                    seam: s.seam,
                    tight: s.tight,
                    gap: s.gap,
                    gap_limit: s.gap_limit,
                    cont_n: s.cont_n,
                    pen: s.pen,
                    review: self.review.and_then(|r| r.get(&(c.a, c.b)).cloned()),
                };
                Some((rank, i, row))
            })
            .collect();
        rows.sort_by(|x, y| {
            x.0.cmp(&y.0).then(y.2.score.total_cmp(&x.2.score)).then(x.1.cmp(&y.1))
        });
        rows.into_iter().map(|(_, _, row)| row).collect()
    }
}

/// One pair of [`Export::joins`], with the numbers `report.md`'s tables print for it.
#[derive(Clone, Debug, PartialEq)]
pub struct JoinRow {
    /// The fragment the pose maps into.
    pub a: FragId,
    /// The fragment the pose moves.
    pub b: FragId,
    /// `confirmed` or `probable`, or `accepted` on a run with the tier pass off.
    pub band: &'static str,
    /// Whether R §8 built the assembly with this pair.
    pub in_assembly: bool,
    /// The group both fragments ended in, when they ended in the same assembled one.
    pub group: Option<usize>,
    /// `seam · tight`.
    pub score: f64,
    /// Seam length, in wall thicknesses.
    pub seam: f64,
    /// The smaller of the two tight-contact fractions.
    pub tight: f64,
    /// Median gap, in wall thicknesses.
    pub gap: f64,
    /// The gap limit the pair was judged by, in wall thicknesses.
    pub gap_limit: f64,
    /// Normal agreement across the seam.
    pub cont_n: f64,
    /// Penetration fraction.
    pub pen: f64,
    /// The review image, relative to the output directory, when one was drawn.
    pub review: Option<String>,
}

/// A short title for the collection at `input`: its directory's name, with the parent's in
/// front when the directory is called `fragments` (the layout the test sets use).
pub fn collection_title(input: &Path) -> String {
    let full = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());
    let parts: Vec<String> = full
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    match parts.as_slice() {
        [.., parent, last] if last.eq_ignore_ascii_case("fragments") => format!("{parent}/{last}"),
        [.., last] => last.clone(),
        [] => "collection".to_owned(),
    }
}

/// A number the way the text files print it: the shortest decimal that reads back as the same
/// `f64`, and `0` for a negative zero (adding `+0.0` turns `-0.0` into `+0.0`).
pub(crate) fn number(x: f64) -> String {
    format!("{}", x + 0.0)
}

/// Writes `text` to `path`, creating the parent directory.
pub(crate) fn write_text(path: &Path, text: &str) -> Result<PathBuf> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| Error::write(path, e))?;
    }
    std::fs::write(path, text).map_err(|e| Error::write(path, e))?;
    Ok(path.to_path_buf())
}

/// UTF-8's byte-order mark, which Excel and Windows Notepad read as "this file is UTF-8" — the
/// tables and the README carry fragment names, and a name may be Cyrillic.
pub(crate) const BOM: &str = "\u{feff}";

#[cfg(test)]
mod tests {
    use super::{collection_title, number};
    use std::path::Path;

    #[test]
    fn numbers_read_back_and_negative_zero_is_zero() {
        assert_eq!(number(-0.0), "0");
        assert_eq!(number(1.0), "1");
        assert_eq!(number(-560.789_876_567_390_4), "-560.7898765673904");
        let x = 0.1 + 0.2;
        assert_eq!(number(x).parse::<f64>().expect("a number").to_bits(), x.to_bits());
    }

    #[test]
    fn a_directory_called_fragments_takes_its_parent_along() {
        assert_eq!(
            collection_title(Path::new("/no/such/test_set/fragments")),
            "test_set/fragments"
        );
        assert_eq!(collection_title(Path::new("/no/such/mixed_all")), "mixed_all");
    }
}
