//! The reviewer's decisions (A §8.1) and what the engine is told about them (A §8.2).

use std::path::Path;

use serde::{Deserialize, Serialize};
use sherd_core::assembly::constraints::Constraints;

use crate::{AppError, Result, atomic};

/// The file's name inside a run folder.
pub const DECISIONS_FILE: &str = "decisions.json";
/// The format this build writes.
pub const DECISIONS_VERSION: u32 = 1;

/// What the reviewer said about a pair.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Place it, at [`Decision::pose`].
    Accept,
    /// Never place this pair — the pair as a whole, which is `must_not_join`'s meaning.
    Reject,
}

/// One decision.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// One fragment, by name.
    pub a: String,
    /// The other.
    pub b: String,
    /// What was decided.
    pub verdict: Verdict,
    /// The accepted candidate's pose, four rows of four, `b` into `a`'s frame **as the candidate
    /// names them**. Carried here so that a decision outlives the `match.state` it was made on.
    #[serde(default)]
    pub pose: Option<[[f64; 4]; 4]>,
    /// The band the candidate was in when it was decided on.
    #[serde(default)]
    pub source: Option<String>,
    /// Made by «Принять все оставшиеся вероятные» (A §8.4), and undone with it.
    #[serde(default)]
    pub bulk: bool,
    /// RFC 3339.
    pub at: String,
    /// The run it was carried over from (A §8.5).
    #[serde(default)]
    pub carried_from: Option<String>,
}

/// `decisions.json`: the decisions as they stand. Undo and redo are the window's.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionsFile {
    /// [`DECISIONS_VERSION`].
    pub version: u32,
    /// One per pair.
    pub decisions: Vec<Decision>,
}

impl Default for DecisionsFile {
    fn default() -> Self {
        Self { version: DECISIONS_VERSION, decisions: Vec::new() }
    }
}

/// A §8.1's key: the pair, in either order.
fn same_pair(d: &Decision, a: &str, b: &str) -> bool {
    (d.a == a && d.b == b) || (d.a == b && d.b == a)
}

/// Whether the collection still holds both fragments the decision is about.
///
/// The one filter A §8.2 and A §8.5 share: `constraints::resolve` fails a whole run on a name it
/// does not know, so a decision about a fragment that has been removed or excluded is neither
/// sent to the engine nor carried into the next run — it is reported instead.
fn known(d: &Decision, names: &[String]) -> bool {
    names.contains(&d.a) && names.contains(&d.b)
}

impl DecisionsFile {
    /// `<dir>/decisions.json`, or no decisions when the run has none yet.
    ///
    /// # Errors
    ///
    /// [`AppError::Json`], [`AppError::Version`].
    pub fn load_or_default(dir: &Path) -> Result<Self> {
        let path = dir.join(DECISIONS_FILE);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let file: Self = atomic::read_json(&path)?;
        if file.version > DECISIONS_VERSION {
            return Err(AppError::Version {
                path,
                found: file.version,
                expected: DECISIONS_VERSION,
            });
        }
        Ok(file)
    }

    /// Writes `<dir>/decisions.json`.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn save(&self, dir: &Path) -> Result<()> {
        atomic::write_json(&dir.join(DECISIONS_FILE), self)
    }

    /// Decides a pair, replacing whatever was decided about it before.
    pub fn set(&mut self, decision: Decision) {
        self.clear(&decision.a, &decision.b);
        self.decisions.push(decision);
    }

    /// Takes a pair's decision back.
    pub fn clear(&mut self, a: &str, b: &str) {
        self.decisions.retain(|d| !same_pair(d, a, b));
    }

    /// This file as the *next* run's (A §8.5): every decision whose two fragments `names` still
    /// holds, marked with the run it came from and with `at`, the moment it was carried.
    ///
    /// The decisions left out are [`to_constraints`]'s `dropped` — the same filter, so that what
    /// the new run's file holds and what its engine is told cannot disagree.
    ///
    /// `at` is the carry, not the click: this entry is new in this run's file, while
    /// [`Decision::carried_from`] keeps where the reviewer's own decision was made and when it
    /// can be read there.
    #[must_use]
    pub fn carried(&self, names: &[String], from_run: &str, at: &str) -> Self {
        Self {
            version: DECISIONS_VERSION,
            decisions: self
                .decisions
                .iter()
                .filter(|d| known(d, names))
                .map(|d| Decision {
                    carried_from: Some(from_run.to_owned()),
                    at: at.to_owned(),
                    ..d.clone()
                })
                .collect(),
        }
    }
}

/// The engine's `constraints.json` for these decisions, and the decisions that could not be
/// carried because a fragment they name is not in `names`.
///
/// Built as the documented JSON and parsed by the engine's own deserialiser, so that this crate
/// depends on the file format the README describes and not on the shape of the engine's enums.
///
/// # Errors
///
/// [`AppError::Json`] if the engine refuses the JSON, which would be a bug here.
pub fn to_constraints(
    file: &DecisionsFile,
    names: &[String],
) -> Result<(Option<Constraints>, Vec<Decision>)> {
    let (kept, dropped): (Vec<&Decision>, Vec<&Decision>) =
        file.decisions.iter().partition(|d| known(d, names));
    let dropped: Vec<Decision> = dropped.into_iter().cloned().collect();
    if kept.is_empty() {
        return Ok((None, dropped));
    }
    let must_join: Vec<serde_json::Value> = kept
        .iter()
        .filter(|d| d.verdict == Verdict::Accept)
        .map(|d| match d.pose {
            Some(pose) => serde_json::json!({ "a": d.a, "b": d.b, "pose": pose }),
            None => serde_json::json!([d.a, d.b]),
        })
        .collect();
    let must_not_join: Vec<serde_json::Value> = kept
        .iter()
        .filter(|d| d.verdict == Verdict::Reject)
        .map(|d| serde_json::json!([d.a, d.b]))
        .collect();
    let json = serde_json::json!({
        "version": sherd_core::assembly::constraints::VERSION,
        "must_join": must_join,
        "must_not_join": must_not_join,
    });
    let constraints = serde_json::from_value(json)
        .map_err(|source| AppError::Json { path: DECISIONS_FILE.into(), source })?;
    Ok((Some(constraints), dropped))
}

#[cfg(test)]
mod tests {
    use super::*;

    const POSE: [[f64; 4]; 4] =
        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, -2.25], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]];

    fn decision(a: &str, b: &str, verdict: Verdict) -> Decision {
        Decision {
            a: a.to_owned(),
            b: b.to_owned(),
            verdict,
            pose: (verdict == Verdict::Accept).then_some(POSE),
            source: Some("probable".to_owned()),
            bulk: false,
            at: "2026-09-20T14:31:07+03:00".to_owned(),
            carried_from: None,
        }
    }

    /// A §8.1: the key is the unordered pair, and a later decision about it replaces the earlier.
    #[test]
    fn a_pair_has_one_decision_whichever_way_round_it_is_named() {
        let mut file = DecisionsFile::default();
        file.set(decision("pieceA", "pieceB", Verdict::Reject));
        file.set(decision("pieceB", "pieceA", Verdict::Accept));
        assert_eq!(file.decisions.len(), 1);
        assert_eq!(file.decisions[0].verdict, Verdict::Accept);
        file.clear("pieceA", "pieceB");
        assert!(file.decisions.is_empty());
    }

    /// A §8.2: accept is `must_join` with the pose, reject is `must_not_join`, and a decision about
    /// a fragment that is gone is dropped and **said** — `constraints::resolve` fails the whole
    /// run on an unknown name, so the filter is the app's duty.
    #[test]
    fn decisions_become_the_engines_constraints_and_the_orphans_are_reported() {
        let mut file = DecisionsFile::default();
        file.set(decision("pieceA", "pieceB", Verdict::Accept));
        file.set(decision("pieceA", "pieceC", Verdict::Reject));
        file.set(decision("pieceA", "gone", Verdict::Reject));
        let names = ["pieceA", "pieceB", "pieceC"].map(str::to_owned);
        let (constraints, dropped) = to_constraints(&file, &names).unwrap();
        let json = serde_json::to_value(constraints.expect("two decisions survive")).unwrap();
        assert_eq!(json["version"], 1);
        assert_eq!(json["must_join"][0]["a"], "pieceA");
        assert_eq!(json["must_join"][0]["pose"][1][3], -2.25);
        assert_eq!(json["must_not_join"], serde_json::json!([["pieceA", "pieceC"]]));
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].b, "gone");
        // and the engine itself accepts what was built
        sherd_core::assembly::constraints::resolve(&serde_json::from_value(json).unwrap(), &names)
            .expect("the engine resolves it");
    }

    #[test]
    fn no_decisions_are_no_constraints() {
        assert_eq!(to_constraints(&DecisionsFile::default(), &[]).unwrap().0, None);
    }
}
