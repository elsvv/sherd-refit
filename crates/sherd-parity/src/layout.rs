//! Where things sit in a fixture dump (D §10.1).
//!
//! ```text
//! DIR/manifest.json
//! DIR/fragments/<name>/   load.* thick.* mesh.* seg.* md.*
//! DIR/pairs/<a>__<b>/     scales.json hyp.* coarse.* nms1.* s1.* nms2.* s2.* result.*
//! DIR/assembly/           md_t_median.json poses.json groups.json used.json rejected.json md/
//! DIR/refine/             <name>.npy, per-join transforms, poses_final.json
//! DIR/outputs/            transforms.json report.json
//! ```

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use sherd_core::error::{Error, Result};

use crate::manifest::Manifest;

/// A fixture dump on disk.
#[derive(Clone, Debug)]
pub struct FixtureDir {
    root: PathBuf,
}

impl FixtureDir {
    /// Names a dump; nothing is read until [`FixtureDir::load_manifest`] is called.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The dump's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// A path inside the dump, from a manifest key (`/` separated, as the Python writes them).
    pub fn path(&self, relative: &str) -> PathBuf {
        let mut p = self.root.clone();
        for part in relative.split('/').filter(|s| !s.is_empty() && *s != ".") {
            p.push(part);
        }
        p
    }

    /// `DIR/manifest.json`.
    pub fn manifest_path(&self) -> PathBuf {
        self.path("manifest.json")
    }

    /// `DIR/fragments/<name>`.
    pub fn fragment_dir(&self, name: &str) -> PathBuf {
        self.root.join("fragments").join(name)
    }

    /// `DIR/pairs/<a>__<b>` — the pair directory name is the two fragment names joined by two
    /// underscores, in the order the pipeline matched them.
    pub fn pair_dir(&self, a: &str, b: &str) -> PathBuf {
        self.root.join("pairs").join(format!("{a}__{b}"))
    }

    /// `DIR/assembly`.
    pub fn assembly_dir(&self) -> PathBuf {
        self.root.join("assembly")
    }

    /// `DIR/refine`.
    pub fn refine_dir(&self) -> PathBuf {
        self.root.join("refine")
    }

    /// `DIR/outputs`.
    pub fn outputs_dir(&self) -> PathBuf {
        self.root.join("outputs")
    }

    /// Reads and parses `manifest.json`.
    pub fn load_manifest(&self) -> Result<Manifest> {
        let path = self.manifest_path();
        let bytes = std::fs::read(&path).map_err(|e| Error::fixture(&path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::fixture(&path, e))
    }

    /// Re-hashes every file the manifest lists and returns the keys that do not match — an empty
    /// vector means the dump is intact. A file that is missing or unreadable counts as a
    /// mismatch, so a truncated dump cannot pass silently.
    pub fn verify_checksums(&self) -> Result<Vec<String>> {
        let manifest = self.load_manifest()?;
        let mut bad = Vec::new();
        for (relative, entry) in &manifest.files {
            let path = self.path(relative);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    if bytes.len() as u64 != entry.size || hex_sha256(&bytes) != entry.sha256 {
                        bad.push(relative.clone());
                    }
                }
                Err(_) => bad.push(relative.clone()),
            }
        }
        Ok(bad)
    }
}

/// SHA-256 of a byte slice as lower-case hexadecimal, the form the manifest stores.
pub fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::{FixtureDir, hex_sha256};
    use std::path::Path;

    #[test]
    fn paths_follow_the_dump_layout() {
        let d = FixtureDir::new("/fx");
        assert_eq!(d.root(), Path::new("/fx"));
        assert_eq!(d.manifest_path(), Path::new("/fx/manifest.json"));
        assert_eq!(d.fragment_dir("pieceA"), Path::new("/fx/fragments/pieceA"));
        assert_eq!(d.pair_dir("pieceA", "pieceB"), Path::new("/fx/pairs/pieceA__pieceB"));
        assert_eq!(d.assembly_dir(), Path::new("/fx/assembly"));
        assert_eq!(d.refine_dir(), Path::new("/fx/refine"));
        assert_eq!(d.outputs_dir(), Path::new("/fx/outputs"));
        assert_eq!(
            d.path("fragments/pieceA/mesh.V.npy"),
            Path::new("/fx/fragments/pieceA/mesh.V.npy")
        );
    }

    #[test]
    fn hashes_are_the_hex_the_manifest_stores() {
        // The empty string's SHA-256, as `hashlib.sha256(b"").hexdigest()` prints it.
        assert_eq!(
            hex_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_missing_manifest_names_the_file() {
        let d = FixtureDir::new("/no/such/fixture");
        let err = d.load_manifest().unwrap_err();
        assert!(err.to_string().contains("manifest.json"), "{err}");
    }
}
