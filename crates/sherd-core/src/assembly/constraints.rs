//! Operator constraints (roadmap item 3, audit §D.1, D §11): `constraints.json` v1.
//!
//! A conservator knows things the geometry does not — that two sherds are from different vessels,
//! that a join was made on the bench and is certain, that a pair the tool likes is impossible. The
//! file is how they say so, and the four statements it carries are audit §D.1's:
//!
//! ```json
//! { "version": 1,
//!   "must_not_join":     [["FY234021", "FY234094"]],
//!   "must_join":         [["FY234007", "FY234021"],
//!                         {"a": "FY234094", "b": "FY234104", "pose": [[1,0,0,0], [0,1,0,0],
//!                                                                     [0,0,1,0], [0,0,0,1]]}],
//!   "same_object":       [["pot_A_01", "pot_A_02"]],
//!   "different_object":  [["pot_A_01", "pot_B_03"]] }
//! ```
//!
//! * **`must_not_join`** removes the pair *before* matching — it is never scored, so it costs
//!   nothing and can never be placed — and R §8 refuses any candidate for it that reaches it
//!   anyway.
//! * **`must_join`** without a pose matches the pair with R §8.1's second-pass budget and promotes
//!   its best probable candidate to confirmed, seeded first; with a pose, matching is skipped and
//!   the pose is a confirmed candidate. A pair with no candidate at all is reported
//!   **unsatisfiable** and the run still succeeds — the tool says what it could not do rather than
//!   refusing to run.
//! * **`same_object`** and **`different_object`** are roadmap item 4's evidence, carried now so
//!   that the file written today is the file item 4 reads; `different_object` already vetoes a
//!   join in R §8.
//!
//! # What a constraint may and may not do
//!
//! **A constraint never edits a score.** Every number in `report.json` is the number R §5–§6
//! computed; what a constraint changes is which pairs are matched, which candidates R §8 may build
//! with, and the order it sees them in. A pinned pose is verified at the pose it was given, so the
//! report carries R §6's honest opinion of it beside the fact that the operator pinned it.
//!
//! # Validation
//!
//! Every name is checked against the collection and an unknown one is an **error**, not a skip: a
//! typo in a constraint file must not quietly become "no constraint". A pair naming one fragment
//! twice is an error, a pair in two of the four lists is an error, an unsupported `version` is an
//! error, and a `pose` that is not a rigid transform is an error.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use nalgebra::Matrix4;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::matching::pair::Candidate;
use crate::tiers::{Tier, TierReport};
use crate::types::FragId;

/// The only `version` this build reads, and the one it documents.
pub const VERSION: u32 = 1;

/// Two fragments an operator has an opinion about, by name.
///
/// Unordered: `["a", "b"]` and `["b", "a"]` name the same pair. A pair is written by **name**
/// because an operator writes it and [`FragId`]s are an artefact of R §2's discovery order.
pub type NamedPair = [String; 2];

/// The contents of `constraints.json` (audit §D.1).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Constraints {
    /// The file format's version; [`VERSION`] is the only one this build reads.
    pub version: u32,
    /// Pairs the operator asserts *do* join, with an optional pose.
    pub must_join: Vec<MustJoin>,
    /// Pairs the operator asserts do *not* join.
    pub must_not_join: Vec<NamedPair>,
    /// Pairs the operator says are from one object (roadmap item 4).
    pub same_object: Vec<NamedPair>,
    /// Pairs the operator says are from different objects; vetoes a join today.
    pub different_object: Vec<NamedPair>,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            version: VERSION,
            must_join: Vec::new(),
            must_not_join: Vec::new(),
            same_object: Vec::new(),
            different_object: Vec::new(),
        }
    }
}

/// One `must_join` entry: a bare pair, or a pair with the pose to place it at.
///
/// Both spellings are read, because the pair form is what the file looked like before there was a
/// pose to write and what a person types by hand.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum MustJoin {
    /// `["a", "b"]` — match the pair and take its best candidate.
    Pair(NamedPair),
    /// `{"a": …, "b": …, "pose": …}` — with a pose, skip matching and place at it.
    Pinned(Pinned),
}

/// A `must_join` written as an object, so that it can carry a pose.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Pinned {
    /// The fragment the pose maps into (R §0's `a`).
    pub a: String,
    /// The fragment the pose moves.
    pub b: String,
    /// `T`, row major, mapping B's coordinates into A's — R §0's convention and the 4×4 both
    /// output files carry. Absent means "match the pair and find one".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose: Option<[[f64; 4]; 4]>,
}

impl MustJoin {
    /// The two names, whichever spelling was used.
    #[must_use]
    pub fn names(&self) -> (&str, &str) {
        match self {
            Self::Pair(p) => (p[0].as_str(), p[1].as_str()),
            Self::Pinned(p) => (p.a.as_str(), p.b.as_str()),
        }
    }

    /// The pose, when the entry carries one.
    #[must_use]
    pub fn pose(&self) -> Option<&[[f64; 4]; 4]> {
        match self {
            Self::Pair(_) => None,
            Self::Pinned(p) => p.pose.as_ref(),
        }
    }
}

impl Constraints {
    /// True when the file constrains nothing, which is what an absent `constraints.json` means.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.must_join.is_empty()
            && self.must_not_join.is_empty()
            && self.same_object.is_empty()
            && self.different_object.is_empty()
    }
}

/// Reads and parses a `constraints.json`; the names are validated later, against the collection.
pub fn load(path: impl AsRef<Path>) -> Result<Constraints> {
    let path = path.as_ref();
    let text = std::fs::read(path).map_err(|e| Error::read(path, e))?;
    let file: Constraints =
        serde_json::from_slice(&text).map_err(|e| Error::read(path, format!("{e}")))?;
    if file.version != VERSION {
        return Err(Error::read(
            path,
            format!(
                "constraints.json version {} — this build reads version {VERSION}",
                file.version
            ),
        ));
    }
    Ok(file)
}

/// One `must_join` after the names have been resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forced {
    /// The fragment the pose maps into.
    pub a: FragId,
    /// The fragment the pose moves.
    pub b: FragId,
    /// The pinned pose, when the operator gave one — then matching is skipped for the pair.
    pub pose: Option<Matrix4<f64>>,
}

/// Why R §8 refused a candidate on the operator's word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Veto {
    /// `must_not_join`.
    MustNotJoin,
    /// `different_object`.
    DifferentObject,
}

impl Veto {
    /// The key `constraints.json` spells this veto with.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::MustNotJoin => "must_not_join",
            Self::DifferentObject => "different_object",
        }
    }
}

/// [`Constraints`] with every name resolved to a [`FragId`] of the collection.
///
/// Every pair is held with its smaller id first, so a lookup is one comparison and the file's own
/// order of the two names does not matter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Resolved {
    /// `must_not_join`, ordered pairs.
    pub forbidden: BTreeSet<(FragId, FragId)>,
    /// `must_join`, in the file's order — which is the order R §8 seeds them in.
    pub forced: Vec<Forced>,
    /// `same_object`, ordered pairs (roadmap item 4).
    pub same: BTreeSet<(FragId, FragId)>,
    /// `different_object`, ordered pairs.
    pub different: BTreeSet<(FragId, FragId)>,
}

/// The pair key both sets are indexed by: the smaller id first.
#[must_use]
pub fn key(a: FragId, b: FragId) -> (FragId, FragId) {
    if a <= b { (a, b) } else { (b, a) }
}

impl Resolved {
    /// True when nothing is constrained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.forbidden.is_empty()
            && self.forced.is_empty()
            && self.same.is_empty()
            && self.different.is_empty()
    }

    /// Why R §8 may not build with this pair, if it may not.
    #[must_use]
    pub fn veto(&self, a: FragId, b: FragId) -> Option<Veto> {
        let k = key(a, b);
        if self.forbidden.contains(&k) {
            Some(Veto::MustNotJoin)
        } else if self.different.contains(&k) {
            Some(Veto::DifferentObject)
        } else {
            None
        }
    }

    /// Whether `must_not_join` names this pair — the test that removes it before matching.
    #[must_use]
    pub fn forbids(&self, a: FragId, b: FragId) -> bool {
        self.forbidden.contains(&key(a, b))
    }

    /// The `must_join` entry for this pair, if there is one.
    #[must_use]
    pub fn forced_pair(&self, a: FragId, b: FragId) -> Option<&Forced> {
        let k = key(a, b);
        self.forced.iter().find(|f| key(f.a, f.b) == k)
    }

    /// Whether `must_join` names this pair — the test that seeds it first in R §8.
    #[must_use]
    pub fn forces(&self, a: FragId, b: FragId) -> bool {
        self.forced_pair(a, b).is_some()
    }
}

/// Resolves every name against the collection and refuses a file that contradicts itself.
///
/// The errors are audit §D.1's: an unknown name, a pair naming one fragment twice, a pair in two
/// of the four lists, and a `pose` that is not a rigid transform.
pub fn resolve(file: &Constraints, names: &[String]) -> Result<Resolved> {
    let mut index: BTreeMap<&str, FragId> = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        index.insert(name.as_str(), u32::try_from(i).expect("fewer than 2^32 fragments"));
    }
    let lookup = |name: &str| -> Result<FragId> {
        index.get(name).copied().ok_or_else(|| {
            Error::read(
                "constraints.json",
                format!("no fragment named `{name}` in this collection"),
            )
        })
    };
    // Which list each pair was seen in, so that "a pair in two lists" names both of them.
    let mut seen: BTreeMap<(FragId, FragId), &'static str> = BTreeMap::new();
    let mut claim = |k: (FragId, FragId), list: &'static str, a: &str, b: &str| -> Result<()> {
        match seen.insert(k, list) {
            Some(first) if first != list => Err(Error::read(
                "constraints.json",
                format!("the pair `{a}`-`{b}` is in both `{first}` and `{list}`"),
            )),
            _ => Ok(()),
        }
    };
    let pair_of = |p: &NamedPair| -> Result<(FragId, FragId)> {
        let (a, b) = (lookup(&p[0])?, lookup(&p[1])?);
        if a == b {
            return Err(Error::read(
                "constraints.json",
                format!("the pair `{}`-`{}` names one fragment twice", p[0], p[1]),
            ));
        }
        Ok(key(a, b))
    };

    let mut out = Resolved::default();
    for entry in &file.must_join {
        let (a_name, b_name) = entry.names();
        let (a, b) = (lookup(a_name)?, lookup(b_name)?);
        if a == b {
            return Err(Error::read(
                "constraints.json",
                format!("the pair `{a_name}`-`{b_name}` names one fragment twice"),
            ));
        }
        claim(key(a, b), "must_join", a_name, b_name)?;
        let pose = entry.pose().map(|rows| rigid(rows, a_name, b_name)).transpose()?;
        out.forced.push(Forced { a, b, pose });
    }
    for p in &file.must_not_join {
        let k = pair_of(p)?;
        claim(k, "must_not_join", &p[0], &p[1])?;
        out.forbidden.insert(k);
    }
    for p in &file.same_object {
        let k = pair_of(p)?;
        claim(k, "same_object", &p[0], &p[1])?;
        out.same.insert(k);
    }
    for p in &file.different_object {
        let k = pair_of(p)?;
        claim(k, "different_object", &p[0], &p[1])?;
        out.different.insert(k);
    }
    Ok(out)
}

/// What one constraint did in a run — one row of `report.md`'s `## Constraints` and one entry of
/// `report.json`'s `constraints` block.
///
/// Audit §D.1: *"the report lists every constraint and what it did"*. Every constraint gets a row,
/// including the ones that did nothing, because a conservator who wrote a line into the file needs
/// to see it come back.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Applied {
    /// The list the constraint was written in: `must_join`, `must_not_join`, `same_object` or
    /// `different_object`.
    pub list: String,
    /// The first fragment named.
    pub a: String,
    /// The second.
    pub b: String,
    /// Whether the run could do what the line asked.
    pub satisfied: bool,
    /// What it did, in the plain language `report.md` prints.
    pub outcome: String,
}

/// Every constraint of the file and what the run did about it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Report {
    /// The `version` the file declared.
    pub version: u32,
    /// One entry per constraint, `must_join` first and then the file's own order inside each list.
    pub entries: Vec<Applied>,
}

impl Report {
    /// How many constraints the run could not satisfy.
    #[must_use]
    pub fn unsatisfied(&self) -> usize {
        self.entries.iter().filter(|e| !e.satisfied).count()
    }
}

/// Carries out the two constraints that act on the candidate list, and reports on all four.
///
/// * **`must_join`** — the pair's best accepted candidate is promoted to [`Tier::Confirmed`] if it
///   is not already there. `assemble_under` does the other half by offering the pair first. A pair
///   with no accepted candidate is **unsatisfiable** and says so; the run continues.
/// * **`must_not_join`** — nothing to do here, because the pair was removed before matching; the
///   entry records how many candidates reached this point (zero, unless something built the
///   candidate list by hand).
/// * **`same_object`** — recorded and read by nothing yet (roadmap item 4).
/// * **`different_object`** — nothing to do here either; `assemble_under` refuses the pair. The
///   entry records whether there was a join to refuse.
///
/// `banded` says whether the tier pass ran: with it off there is no band to promote to and the
/// constraint is satisfied by R §6.5's own verdict, which is the gate R §8 is then using.
/// `tiers` is kept in step with [`Candidate::tier`] so that `report.json`'s per-candidate band and
/// the band R §8 read are the same value.
pub fn apply(
    resolved: &Resolved,
    names: &[String],
    candidates: &mut [Candidate],
    mut tiers: Option<&mut TierReport>,
    banded: bool,
) -> Report {
    let name = |n: FragId| names.get(n as usize).map_or("?", String::as_str).to_owned();
    let mut entries: Vec<Applied> = Vec::new();

    for forced in &resolved.forced {
        let (satisfied, outcome) = promote(forced, candidates, tiers.as_deref_mut(), banded);
        entries.push(Applied {
            list: "must_join".to_owned(),
            a: name(forced.a),
            b: name(forced.b),
            satisfied,
            outcome,
        });
    }

    let count = |k: (FragId, FragId), candidates: &[Candidate]| {
        candidates.iter().filter(|c| key(c.a, c.b) == k).count()
    };
    for &k in &resolved.forbidden {
        let seen = count(k, candidates);
        entries.push(Applied {
            list: "must_not_join".to_owned(),
            a: name(k.0),
            b: name(k.1),
            satisfied: true,
            outcome: if seen == 0 {
                "the pair was removed before matching and was never scored".to_owned()
            } else {
                format!("{seen} candidate(s) reached the assembly and every one was refused")
            },
        });
    }
    for &k in &resolved.different {
        let seen = count(k, candidates);
        entries.push(Applied {
            list: "different_object".to_owned(),
            a: name(k.0),
            b: name(k.1),
            satisfied: true,
            outcome: if seen == 0 {
                "the pair produced no candidate; nothing to refuse".to_owned()
            } else {
                format!("{seen} candidate(s) were scored and R §8 refused the pair")
            },
        });
    }
    for &k in &resolved.same {
        entries.push(Applied {
            list: "same_object".to_owned(),
            a: name(k.0),
            b: name(k.1),
            satisfied: true,
            outcome: "recorded for the group consensus (roadmap item 4); no join was changed"
                .to_owned(),
        });
    }
    Report { version: VERSION, entries }
}

/// One `must_join`: promote the pair's best accepted candidate to the confirmed band, and say
/// what happened.
///
/// The search for "the pair's best" is R §8's own — a strict `>` keeps the first of a tie, which is
/// the candidate R §5.7 ranked highest.
fn promote(
    forced: &Forced,
    candidates: &mut [Candidate],
    tiers: Option<&mut TierReport>,
    banded: bool,
) -> (bool, String) {
    let k = key(forced.a, forced.b);
    let mut best: Option<usize> = None;
    for (i, c) in candidates.iter().enumerate() {
        if key(c.a, c.b) != k || !c.accepted {
            continue;
        }
        if best.is_none_or(|held| c.score() > candidates[held].score()) {
            best = Some(i);
        }
    }
    let pinned = forced.pose.is_some();
    let Some(i) = best else {
        return (
            false,
            if pinned {
                "unsatisfiable: the pinned pose could not be scored (a fragment has no surface)"
                    .to_owned()
            } else {
                "unsatisfiable: the pair was matched with the second-pass budget and R §6.5 \
                 accepted no candidate"
                    .to_owned()
            },
        );
    };
    let score = candidates[i].score();
    let was = candidates[i].tier;
    if banded && was != Tier::Confirmed {
        candidates[i].tier = Tier::Confirmed;
        if let Some(slot) = tiers.and_then(|report| report.tiers.get_mut(i)) {
            *slot = Tier::Confirmed;
        }
    }
    let head = if pinned {
        "matching skipped; the pinned pose is a confirmed candidate".to_owned()
    } else if !banded {
        format!("matched with the second-pass budget; accepted at score {score:.2}")
    } else if was == Tier::Confirmed {
        format!("matched with the second-pass budget; already confirmed at score {score:.2}")
    } else {
        format!(
            "matched with the second-pass budget; promoted from {} to confirmed at score \
             {score:.2}",
            was.label()
        )
    };
    (true, format!("{head}, and offered to R §8 before every other join"))
}

/// A pinned pose, checked to be a rigid transform before anything is placed at it.
///
/// The tolerances are loose on purpose — a 4×4 that came through a JSON file and a review screen
/// has been through `%.17g` at worst — but a mirrored, scaled or sheared matrix is a mistake that
/// would place a fragment inside its partner and be blamed on the matcher.
fn rigid(rows: &[[f64; 4]; 4], a: &str, b: &str) -> Result<Matrix4<f64>> {
    let bad = |why: &str| {
        Error::read(
            "constraints.json",
            format!("the pose pinned for `{a}`-`{b}` is not a rigid transform: {why}"),
        )
    };
    if !rows.iter().flatten().all(|x| x.is_finite()) {
        return Err(bad("it holds a value that is not finite"));
    }
    let last = rows[3];
    if (last[0].abs() + last[1].abs() + last[2].abs()) > 1e-9 || (last[3] - 1.0).abs() > 1e-9 {
        return Err(bad("its last row is not [0, 0, 0, 1]"));
    }
    let mut m = Matrix4::<f64>::identity();
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            m[(i, j)] = *v;
        }
    }
    let r = m.fixed_view::<3, 3>(0, 0).into_owned();
    let error = (r.transpose() * r - nalgebra::Matrix3::<f64>::identity()).abs().max();
    if error > 1e-6 {
        return Err(bad(&format!("its rotation is not orthonormal (off by {error:.2e})")));
    }
    if r.determinant() <= 0.0 {
        return Err(bad("its rotation is a reflection (the determinant is not positive)"));
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::{Constraints, Resolved, VERSION, load, resolve};

    fn names() -> Vec<String> {
        ["a", "b", "c"].iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn both_spellings_of_must_join_parse_and_a_missing_version_is_the_current_one() {
        let text = r#"{"must_join": [["a", "b"], {"a": "b", "b": "c"}]}"#;
        let file: Constraints = serde_json::from_str(text).expect("the format parses");
        assert_eq!(file.version, VERSION, "an absent version is this build's own");
        assert_eq!(file.must_join[0].names(), ("a", "b"));
        assert_eq!(file.must_join[1].names(), ("b", "c"));
        assert!(file.must_join.iter().all(|m| m.pose().is_none()));
        let resolved = resolve(&file, &names()).expect("the names are the collection's");
        assert!(resolved.forces(1, 0), "either order names the same pair");
        assert!(resolved.veto(0, 1).is_none(), "must_join is not a veto");
    }

    #[test]
    fn an_empty_file_constrains_nothing() {
        let file: Constraints = serde_json::from_str("{}").expect("every key defaults");
        assert!(file.is_empty());
        assert_eq!(resolve(&file, &names()).expect("nothing to resolve"), Resolved::default());
    }

    #[test]
    fn an_unknown_name_is_an_error_and_not_a_skip() {
        let file: Constraints =
            serde_json::from_str(r#"{"must_not_join": [["a", "zz"]]}"#).expect("it parses");
        let error = resolve(&file, &names()).expect_err("`zz` is not in the collection");
        assert!(format!("{error}").contains("zz"), "{error}");
    }

    #[test]
    fn a_pair_in_two_lists_is_an_error_and_the_message_names_both() {
        let file: Constraints =
            serde_json::from_str(r#"{"must_join": [["a", "b"]], "must_not_join": [["b", "a"]]}"#)
                .expect("it parses");
        let error = resolve(&file, &names()).expect_err("a contradiction");
        let text = format!("{error}");
        assert!(text.contains("must_join") && text.contains("must_not_join"), "{text}");
    }

    #[test]
    fn a_pair_naming_one_fragment_twice_is_an_error() {
        let file: Constraints =
            serde_json::from_str(r#"{"same_object": [["a", "a"]]}"#).expect("it parses");
        assert!(resolve(&file, &names()).is_err());
    }

    #[test]
    fn a_pinned_pose_must_be_rigid() {
        let good = r#"{"must_join": [{"a": "a", "b": "b", "pose":
            [[0,-1,0,5],[1,0,0,0],[0,0,1,0],[0,0,0,1]]}]}"#;
        let file: Constraints = serde_json::from_str(good).expect("it parses");
        let resolved = resolve(&file, &names()).expect("a quarter turn and a shift are rigid");
        let pose = resolved.forced[0].pose.expect("the pose came through");
        assert!((pose[(0, 3)] - 5.0).abs() < 1e-12);

        for bad in [
            "[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
            "[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
            "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]]",
        ] {
            let text = format!(r#"{{"must_join": [{{"a": "a", "b": "b", "pose": {bad}}}]}}"#);
            let file: Constraints = serde_json::from_str(&text).expect("it parses");
            assert!(resolve(&file, &names()).is_err(), "{bad} is not rigid");
        }
    }

    #[test]
    fn a_version_this_build_does_not_read_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("sherd-constraints-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("constraints.json");
        std::fs::write(&path, r#"{"version": 2, "must_join": []}"#).expect("written");
        let error = load(&path).expect_err("version 2 is not this build's");
        assert!(format!("{error}").contains("version 2"), "{error}");
        std::fs::write(&path, r#"{"version": 1}"#).expect("written");
        assert!(load(&path).expect("version 1 is").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn different_object_vetoes_and_same_object_does_not() {
        let text = r#"{"same_object": [["a", "b"]], "different_object": [["a", "c"]]}"#;
        let file: Constraints = serde_json::from_str(text).expect("it parses");
        let r = resolve(&file, &names()).expect("both resolve");
        assert!(r.veto(0, 1).is_none(), "same_object is evidence, not a veto");
        assert_eq!(r.veto(2, 0), Some(super::Veto::DifferentObject));
        assert!(r.same.contains(&(0, 1)) && r.different.contains(&(0, 2)));
    }
}
