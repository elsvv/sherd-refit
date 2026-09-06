//! Discovering a collection: which files, in which order, under which names (R §2).
//!
//! The order matters far beyond tidiness. It fixes the pair order of R §4.1, the group seeding
//! order of R §8 and, through them, every seeded draw of the run, so a port that discovers the
//! same directory in a different order does not reproduce the reference even when every stage is
//! exact. The rule, verbatim from `sherd_refit.pipeline.find_meshes`:
//!
//! ```text
//! for ext in (".ply", ".obj", ".stl", ".off"):
//!     files += glob(dir/*ext) + glob(dir/*EXT)
//! return sorted(set(files))
//! ```
//!
//! — so the extension loop and the case pair only decide *which* files are found; `sorted(set(…))`
//! then orders them by their full path as a string. Names are the file stem, unless two files
//! share a stem, in which case *both* take the basename with `.` replaced by `_`.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::io::MeshFormat;

/// One entry of a collection: the file and the name every report and output is keyed by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The mesh file.
    pub path: PathBuf,
    /// The fragment's name (R §2).
    pub name: String,
}

/// Every mesh file of a directory, in the reference's order (R §2).
///
/// Only the four extensions of `sherd_refit.fragment.MESH_EXT` are discovered — `.glb` is
/// readable but is never picked up by a scan, because the reference does not pick it up either.
/// The sort is over the full path as a string, which is what `sorted(set(...))` does on the paths
/// `glob` returns.
pub fn find_meshes(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    let entries = std::fs::read_dir(dir).map_err(|e| Error::read(dir, e))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| Error::read(dir, e))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        // `glob("*.ply")` and `glob("*.PLY")` between them accept exactly the two pure cases,
        // and `MeshFormat::from_path` lower-cases the extension, which additionally accepts
        // `.Ply`. The difference can only appear on a case-sensitive filesystem holding a mixed
        // -case extension; such a file is a fragment by any reading, and rejecting it would be a
        // silent skip rather than a difference in results.
        if MeshFormat::from_path(&path).is_ok_and(MeshFormat::is_discovered) {
            files.push(path);
        }
    }
    files.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    files.dedup();
    Ok(files)
}

/// The fragment name of each file: its stem, or the basename with `.` replaced by `_` when two
/// files in the collection share a stem (R §2).
pub fn fragment_names(files: &[PathBuf]) -> Vec<String> {
    let stems: Vec<String> = files
        .iter()
        .map(|f| f.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned()))
        .collect();
    files
        .iter()
        .zip(&stems)
        .map(|(f, stem)| {
            if stems.iter().filter(|s| *s == stem).count() == 1 {
                stem.clone()
            } else {
                f.file_name().map_or_else(String::new, |s| s.to_string_lossy().replace('.', "_"))
            }
        })
        .collect()
}

/// [`find_meshes`] and [`fragment_names`] together, in collection order.
///
/// Fewer than two files is R §2's error case for a run; it is not rejected here, because
/// `segment` on a single fragment is a perfectly reasonable thing to ask for and the pipeline is
/// where the pair count matters.
pub fn discover(dir: impl AsRef<Path>) -> Result<Vec<Entry>> {
    let files = find_meshes(dir)?;
    let names = fragment_names(&files);
    Ok(files.into_iter().zip(names).map(|(path, name)| Entry { path, name }).collect())
}

#[cfg(test)]
mod tests {
    use super::{discover, find_meshes, fragment_names};
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sherd-collection-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    }

    #[test]
    fn discovery_is_sorted_by_path_and_ignores_everything_else() {
        let dir = scratch("order");
        for f in ["b.ply", "a.obj", "b.STL", "d.off", "notes.txt", "mesh.glb", "z.ply"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::create_dir_all(dir.join("sub.ply")).unwrap();
        let files = find_meshes(&dir).unwrap();
        let got: Vec<String> =
            files.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        // `sorted(set(...))` over the full paths, byte by byte — so `b.STL` comes before
        // `b.ply` (`S` < `p`) rather than the extension loop deciding anything — and neither the
        // `.txt`, nor the `.glb`, nor the directory called `sub.ply` is a fragment.
        assert_eq!(got, ["a.obj", "b.STL", "b.ply", "d.off", "z.ply"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_directory_names_itself() {
        let err = find_meshes("/no/such/collection").unwrap_err().to_string();
        assert!(err.contains("/no/such/collection"), "{err}");
    }

    #[test]
    fn names_are_stems_until_two_stems_collide() {
        let files: Vec<PathBuf> =
            ["/c/a.ply", "/c/b.obj", "/c/b.ply"].iter().map(PathBuf::from).collect();
        assert_eq!(fragment_names(&files), ["a", "b_obj", "b_ply"]);

        let files: Vec<PathBuf> = ["/c/one.ply", "/c/two.ply"].iter().map(PathBuf::from).collect();
        assert_eq!(fragment_names(&files), ["one", "two"]);

        // A stem that itself contains dots keeps them; only a collision rewrites the name.
        let files: Vec<PathBuf> = ["/c/frag.001.ply"].iter().map(PathBuf::from).collect();
        assert_eq!(fragment_names(&files), ["frag.001"]);
    }

    #[test]
    fn discover_pairs_each_file_with_its_name() {
        let dir = scratch("discover");
        std::fs::write(dir.join("x.ply"), b"x").unwrap();
        std::fs::write(dir.join("x.obj"), b"x").unwrap();
        let entries = discover(&dir).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "x_obj");
        assert_eq!(entries[1].name, "x_ply");
        assert!(entries[0].path.ends_with("x.obj"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
