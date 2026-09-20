//! What the input folder held when a run was made, and how that differs from now (A §4, «Stale»).
//!
//! Names, sizes and modification times, never content: the input is gigabytes, and a file changed
//! under the same size and mtime is still caught by R §3.7's cache validation at the next
//! `Prepare`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::{AppError, Result};

/// One scan as the snapshot saw it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStamp {
    /// The fragment's name (R §2) — what every decision, cache and display mesh is keyed by.
    pub name: String,
    /// The file's name in the input folder.
    pub file: String,
    /// Bytes.
    pub size: u64,
    /// Modification time, milliseconds since the epoch; 0 where the file system has none.
    pub mtime_ms: i64,
}

/// The input as it stood.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSnapshot {
    /// Every scan of the folder, the excluded ones included, in R §2's order.
    pub files: Vec<FileStamp>,
    /// The workspace's exclusions at the time.
    pub excluded: BTreeSet<String>,
}

/// How two snapshots differ, by fragment name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleDiff {
    /// Scans that are there now and were not.
    pub added: Vec<String>,
    /// Scans that were there and are not.
    pub removed: Vec<String>,
    /// Scans whose size or modification time changed.
    pub changed: Vec<String>,
    /// Fragments excluded since.
    pub excluded_added: Vec<String>,
    /// Fragments taken back in since.
    pub excluded_removed: Vec<String>,
}

impl StaleDiff {
    /// Whether nothing differs — the run is current.
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// The input folder now.
///
/// # Errors
///
/// [`AppError::Core`] when the folder cannot be listed, [`AppError::Io`] for a file's metadata.
pub fn scan(input: &Path, excluded: &BTreeSet<String>) -> Result<InputSnapshot> {
    let mut files = Vec::new();
    for entry in sherd_core::collection::discover(input)? {
        let meta = std::fs::metadata(&entry.path).map_err(|e| AppError::io(&entry.path, e))?;
        let mtime_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .and_then(|d| i64::try_from(d.as_millis()).ok())
            .unwrap_or(0);
        files.push(FileStamp {
            name: entry.name,
            file: entry
                .path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
            size: meta.len(),
            mtime_ms,
        });
    }
    Ok(InputSnapshot { files, excluded: excluded.clone() })
}

/// What changed from `then` to `now`.
pub fn diff(then: &InputSnapshot, now: &InputSnapshot) -> StaleDiff {
    let by_name = |s: &InputSnapshot| -> BTreeMap<String, (u64, i64)> {
        s.files.iter().map(|f| (f.name.clone(), (f.size, f.mtime_ms))).collect()
    };
    let (old, new) = (by_name(then), by_name(now));
    StaleDiff {
        added: new.keys().filter(|n| !old.contains_key(*n)).cloned().collect(),
        removed: old.keys().filter(|n| !new.contains_key(*n)).cloned().collect(),
        changed: new
            .iter()
            .filter(|(n, stamp)| old.get(*n).is_some_and(|was| was != *stamp))
            .map(|(n, _)| n.clone())
            .collect(),
        excluded_added: now.excluded.difference(&then.excluded).cloned().collect(),
        excluded_removed: then.excluded.difference(&now.excluded).cloned().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str, files: &[(&str, &[u8])]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-snap-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in files {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
        dir
    }

    /// A §4 «Stale»: names, sizes, mtimes and the exclusions — and what changed is *said*.
    #[test]
    fn what_changed_in_the_input_is_named() {
        let input = dir("diff", &[("a.ply", b"1"), ("b.ply", b"22"), ("c.ply", b"333")]);
        let then = scan(&input, &BTreeSet::new()).unwrap();
        assert_eq!(then.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
        assert!(diff(&then, &then).is_empty());

        std::fs::remove_file(input.join("a.ply")).unwrap();
        std::fs::write(input.join("b.ply"), b"4444").unwrap();
        std::fs::write(input.join("d.ply"), b"5").unwrap();
        let now = scan(&input, &BTreeSet::from(["c".to_owned()])).unwrap();
        let d = diff(&then, &now);
        assert_eq!(
            (d.added, d.removed, d.changed),
            (vec!["d".to_owned()], vec!["a".to_owned()], vec!["b".to_owned()])
        );
        assert_eq!(d.excluded_added, ["c"]);
        assert!(d.excluded_removed.is_empty());
    }
}
