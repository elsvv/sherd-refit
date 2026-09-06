//! `manifest.json`: what a fixture dump contains and what each file should hash to (D §10.1).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sherd_core::Params;

/// The manifest at the root of a fixture dump.
///
/// Unknown keys are kept out of the way rather than rejected: the Python sink may grow fields,
/// and an older dump must stay readable.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    /// What the run was given: files, parameters, output options.
    pub collection: Collection,
    /// The commit of the Python the dump was made from — `9d4b9d3` for the frozen algorithm.
    pub commit: String,
    /// Whether that working tree had uncommitted changes.
    #[serde(default)]
    pub dirty: bool,
    /// The collection order, the pair order and the medians the pipeline derived from them.
    pub pairs: Pairs,
    /// Verbosity the dump was written at: `full`, `slim` or `min`.
    #[serde(default)]
    pub level: String,
    /// Open3D version of the reference run.
    #[serde(default)]
    pub open3d: String,
    /// numpy version of the reference run.
    #[serde(default)]
    pub numpy: String,
    /// scipy version of the reference run.
    #[serde(default)]
    pub scipy: String,
    /// Python version of the reference run.
    #[serde(default)]
    pub python: String,
    /// Platform string of the machine that produced the dump.
    #[serde(default)]
    pub platform: String,
    /// Every file of the dump, keyed by its path relative to the dump root, with `/` separators.
    pub files: BTreeMap<String, FileEntry>,
}

/// The collection the reference was run on.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Collection {
    /// File names, in the reference's discovery order (R §2).
    pub files: Vec<String>,
    /// Input directory as given on the command line.
    pub input_dir: String,
    /// Candidates kept per pair.
    pub keep_per_pair: u32,
    /// The thresholds of the run (R §1.1).
    pub params: Params,
    /// Whether preview images were rendered.
    #[serde(default)]
    pub preview: bool,
    /// Whether the assembly was refined at full resolution (R §9).
    #[serde(default)]
    pub refine: bool,
    /// The face budget the run was given; 0 means the adaptive budget of R §3.3.
    #[serde(default)]
    pub target_faces: u32,
    /// Whether the placed meshes were written out.
    #[serde(default)]
    pub write_meshes: bool,
}

/// The collection order and the pair order — the two orders every seeded draw depends on.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pairs {
    /// Fragment names, in collection order.
    pub names: Vec<String>,
    /// The pairs the pipeline matched, in its own order.
    pub pairs: Vec<[String; 2]>,
    /// Pairs skipped before matching, with the reason the reference gave (R §4.1, R §4.3).
    #[serde(default)]
    pub skipped: Vec<serde_json::Value>,
    /// Median `res` over the collection.
    #[serde(default)]
    pub resolution_median: f64,
    /// Median wall thickness over the collection.
    #[serde(default)]
    pub thickness_median: f64,
}

/// One file of the dump.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FileEntry {
    /// SHA-256 of the file's bytes, lower-case hexadecimal.
    pub sha256: String,
    /// Size in bytes.
    pub size: u64,
    /// Array shape, for `.npy` files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<u64>>,
    /// numpy dtype, for `.npy` files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
}

impl Manifest {
    /// The fragment names in collection order.
    pub fn fragment_names(&self) -> &[String] {
        &self.pairs.names
    }

    /// True when the run used the defaults of R §1.1 — the usual case, and the one where a Rust
    /// run can be started from `Params::default()` rather than from the dump's own values.
    pub fn uses_default_params(&self) -> bool {
        self.collection.params == Params::default()
    }

    /// Total size of the dump in bytes, as recorded in the manifest.
    pub fn total_size(&self) -> u64 {
        self.files.values().map(|f| f.size).sum()
    }
}
