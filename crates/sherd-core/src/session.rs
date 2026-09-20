//! A match that outlives its run (A §3): saved by `run_with`, read back by a review session, and
//! reassembled under the reviewer's decisions without matching a pair again.
//!
//! Matching is 94 % of a run (997 of 1066 s on the 155-fragment `karas`); R §8's assembly is
//! 0.14 s of it. A decision about one join therefore costs a reassembly, provided the candidate
//! list the assembly was built from is still there — which is what [`MatchState`] is.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::matching::pair::Candidate;
use crate::params::Params;
use crate::tiers::TierReport;

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
