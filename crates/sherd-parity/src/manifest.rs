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
    /// The face cap the run was given, or 0 for [`NO_CAP`] — see [`Collection::face_cap`].
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

/// The cap [`Collection::face_cap`] reports when a manifest names none.
///
/// `u32::MAX` and not `usize::MAX`, so that it survives the round trip through
/// `Fragment::target_faces`, which is a `u32` because it is part of the cache key (R §3.7).
pub const NO_CAP: usize = u32::MAX as usize;

// `face_cap` narrows a u32 to a usize; every target this workspace builds for has at least 32 bits
// of it, and a 16-bit one would silently truncate a face count rather than fail.
const _: () = assert!(usize::BITS >= u32::BITS, "a face cap must fit in a usize");

impl Collection {
    /// The upper bound of R §3.3's budget, `int(clip(600·ΣA0/t², 50000, target_faces))`.
    ///
    /// **0 and a missing `target_faces` key mean the same thing and mean it explicitly**
    /// (finding F2 of the phase-1a verification): no cap, i.e. R §3.3's adaptive budget on its
    /// own, floored at `MIN_FACES`. `serde` resolves an absent key to 0, so the two cannot be told
    /// apart in the first place, and there is deliberately no third reading:
    ///
    /// * handing the 0 to `face_budget` straight through would give
    ///   `clip(raw, 50000, 0) == 0` — numpy applies the floor first, so the cap wins — and
    ///   decimate every fragment of the dump to nothing. That is what this used to do.
    /// * hiding a default of 200 000 here would be a *guess* at what the reference ran with, and
    ///   it would be silently wrong for a dump made at any other cap.
    ///
    /// Every committed manifest carries 200 000 and takes neither path.
    pub fn face_cap(&self) -> usize {
        match self.target_faces {
            0 => NO_CAP,
            cap => cap as usize,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{Collection, NO_CAP};
    use sherd_core::Params;

    fn collection_with(target_faces: u32) -> Collection {
        Collection {
            files: Vec::new(),
            input_dir: "in".to_owned(),
            keep_per_pair: 5,
            params: Params::default(),
            preview: false,
            refine: false,
            target_faces,
            write_meshes: false,
        }
    }

    /// Finding F2: there are exactly two readings of `target_faces`, and 0 is the same one as an
    /// absent key.
    #[test]
    fn a_zero_or_missing_face_cap_is_the_adaptive_budget() {
        assert_eq!(collection_with(200_000).face_cap(), 200_000);
        assert_eq!(collection_with(20_000).face_cap(), 20_000);
        assert_eq!(collection_with(0).face_cap(), NO_CAP);

        // A manifest with no `target_faces` key at all: `serde` gives 0, which is the same reading.
        let mut json = serde_json::to_value(collection_with(200_000)).expect("serialises");
        json.as_object_mut().expect("an object").remove("target_faces");
        let without: Collection = serde_json::from_value(json).expect("deserialises");
        assert_eq!(without.target_faces, 0);
        assert_eq!(without.face_cap(), NO_CAP);
    }

    /// What the sentinel means where it is used: R §3.3's floor still applies, the cap never
    /// binds, and the 0 this replaces would have decimated the fragment to nothing.
    #[test]
    fn the_uncapped_budget_is_still_floored() {
        use sherd_core::mesh::decimate::{MIN_FACES, face_budget};

        assert_eq!(face_budget(1.0, 1.0, NO_CAP), MIN_FACES);
        assert_eq!(face_budget(1e6, 1.0, NO_CAP), 600_000_000);
        assert_eq!(face_budget(1e6, 1.0, 0), 0, "the trap F2 found");
    }
}
