//! A workspace (A §4): a plain folder holding `sherd-workspace.json`, one collection, its cache,
//! its display meshes and the history of its runs. The scans themselves are linked, never copied
//! — 155 of them are 10 GB.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{AppError, Result, atomic};

/// The file that makes a folder a workspace.
pub const WORKSPACE_FILE: &str = "sherd-workspace.json";
/// The file an open workspace holds a lock on.
pub const LOCK_FILE: &str = "sherd-workspace.lock";
/// The format this build writes, and the newest it reads.
pub const WORKSPACE_VERSION: u32 = 1;
/// How many folders up from the workspace the input may be and still be remembered by a relative
/// path: beside the workspace, or beside its parent — a project folder moved as a whole.
const MAX_CLIMB: usize = 2;

/// Where the scans are.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRef {
    /// As the user picked it.
    pub absolute: PathBuf,
    /// From the workspace folder, when there is such a path: tried first, so that a workspace and
    /// its scans moved together — another disk, a colleague's machine — still open.
    pub relative: Option<PathBuf>,
}

/// `sherd-workspace.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceFile {
    /// [`WORKSPACE_VERSION`].
    pub version: u32,
    /// The input folder, once one is linked.
    #[serde(default)]
    pub input: Option<InputRef>,
    /// Fragments left out of every run, by name (A §5.1). The files are untouched.
    #[serde(default)]
    pub excluded: BTreeSet<String>,
    /// The last launch sheet, as the app wrote it; the workspace does not read it.
    #[serde(default)]
    pub last_spec: Option<serde_json::Value>,
}

/// An open workspace. Holding one holds its lock; dropping it releases the lock.
#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
    file: WorkspaceFile,
    _lock: File,
}

impl Workspace {
    /// Makes `root` a workspace. The folder may exist; it may not already be a workspace.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`]; [`AppError::Locked`] when it is one already and is open elsewhere.
    pub fn create(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root).map_err(|e| AppError::io(root, e))?;
        if root.join(WORKSPACE_FILE).exists() {
            return Self::open(root);
        }
        let lock = lock(root)?;
        let file = WorkspaceFile {
            version: WORKSPACE_VERSION,
            input: None,
            excluded: BTreeSet::new(),
            last_spec: None,
        };
        let ws = Self { root: root.to_owned(), file, _lock: lock };
        ws.save()?;
        Ok(ws)
    }

    /// Opens a workspace and takes its lock.
    ///
    /// # Errors
    ///
    /// [`AppError::NotAWorkspace`], [`AppError::Locked`], [`AppError::Version`] for a file a newer
    /// app wrote, [`AppError::Json`].
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join(WORKSPACE_FILE);
        if !path.is_file() {
            return Err(AppError::NotAWorkspace { path: root.to_owned() });
        }
        let lock = lock(root)?;
        let file: WorkspaceFile = atomic::read_json(&path)?;
        if file.version > WORKSPACE_VERSION {
            return Err(AppError::Version {
                path,
                found: file.version,
                expected: WORKSPACE_VERSION,
            });
        }
        Ok(Self { root: root.to_owned(), file, _lock: lock })
    }

    /// The folder.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The file as it stands.
    pub fn file(&self) -> &WorkspaceFile {
        &self.file
    }

    /// The input folder if it can be found now: by its relative path first, then its absolute one.
    /// `None` both when no input is linked and when it is linked but missing (A §5, «input
    /// missing»); [`file`](Self::file) tells the two apart.
    pub fn input(&self) -> Option<PathBuf> {
        let input = self.file.input.as_ref()?;
        let relative = input.relative.as_ref().map(|r| self.root.join(r));
        relative
            .into_iter()
            .chain(std::iter::once(input.absolute.clone()))
            .find(|p| p.is_dir())
            .and_then(|p| p.canonicalize().ok())
    }

    /// Links the input folder.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`] when `dir` is not a folder that can be resolved.
    pub fn set_input(&mut self, dir: &Path) -> Result<()> {
        let absolute = dir.canonicalize().map_err(|e| AppError::io(dir, e))?;
        let root = self.root.canonicalize().map_err(|e| AppError::io(&self.root, e))?;
        // Only a *nearby* input gets a relative path. Two folders that share nothing but the
        // file system's root — a workspace at home, scans on another volume — also have one, and
        // after the workspace moved it would point at whatever happens to be there.
        let relative = relative_to(&root, &absolute)
            .filter(|r| r.components().filter(|c| *c == Component::ParentDir).count() <= MAX_CLIMB);
        self.file.input = Some(InputRef { absolute, relative });
        self.save()
    }

    /// Leaves a fragment out of every run, or takes it back in (A §5.1).
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn set_excluded(&mut self, name: &str, excluded: bool) -> Result<()> {
        if excluded {
            self.file.excluded.insert(name.to_owned());
        } else {
            self.file.excluded.remove(name);
        }
        self.save()
    }

    /// Remembers the launch sheet.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`].
    pub fn set_last_spec(&mut self, spec: serde_json::Value) -> Result<()> {
        self.file.last_spec = Some(spec);
        self.save()
    }

    /// `cache/`: R §3.7's fragment cache, shared by every run.
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    /// `fragments/`: thumbnails, display meshes and `index.json`.
    pub fn fragments_dir(&self) -> PathBuf {
        self.root.join("fragments")
    }

    /// `runs/`.
    pub fn runs_dir(&self) -> PathBuf {
        self.root.join("runs")
    }

    /// `runs/<id>/`.
    pub fn run_dir(&self, id: &str) -> PathBuf {
        self.runs_dir().join(id)
    }

    /// `exports/`.
    pub fn exports_dir(&self) -> PathBuf {
        self.root.join("exports")
    }

    fn save(&self) -> Result<()> {
        atomic::write_json(&self.root.join(WORKSPACE_FILE), &self.file)
    }
}

/// The OS's advisory lock on [`LOCK_FILE`]: released when the process ends, however it ends, so
/// there is no stale lock to explain to anyone.
fn lock(root: &Path) -> Result<File> {
    let path = root.join(LOCK_FILE);
    let file = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| AppError::io(&path, e))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => Err(AppError::Locked { path: root.to_owned() }),
        Err(std::fs::TryLockError::Error(e)) => Err(AppError::io(&path, e)),
    }
}

/// `target` as seen from `base`, both absolute; `None` when they share no root (two Windows
/// drives).
fn relative_to(base: &Path, target: &Path) -> Option<PathBuf> {
    let (base, target): (Vec<Component<'_>>, Vec<Component<'_>>) =
        (base.components().collect(), target.components().collect());
    let shared = base.iter().zip(&target).take_while(|(a, b)| a == b).count();
    if shared == 0 {
        return None;
    }
    let mut out = PathBuf::new();
    for _ in shared..base.len() {
        out.push("..");
    }
    for part in &target[shared..] {
        out.push(part);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-ws-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_workspace_is_a_folder_that_reopens_as_it_was_left() {
        let root = scratch("reopen").join("karas");
        let scans = scratch("reopen-scans");
        {
            let mut ws = Workspace::create(&root).unwrap();
            ws.set_input(&scans).unwrap();
            ws.set_excluded("FY234009", true).unwrap();
        }
        let ws = Workspace::open(&root).unwrap();
        assert_eq!(ws.input().unwrap(), scans.canonicalize().unwrap());
        assert!(ws.file().excluded.contains("FY234009"));
        assert_eq!(ws.run_dir("2026-09-20_1412"), root.join("runs").join("2026-09-20_1412"));
    }

    /// A §4: the input is found by its path relative to the workspace first, so that a workspace
    /// and its scans moved together to another disk still open.
    #[test]
    fn the_input_is_found_again_after_both_folders_moved_together() {
        let base = scratch("moved");
        let (root, scans) = (base.join("a/karas"), base.join("a/scans"));
        std::fs::create_dir_all(&scans).unwrap();
        {
            let mut ws = Workspace::create(&root).unwrap();
            ws.set_input(&scans).unwrap();
        }
        std::fs::rename(base.join("a"), base.join("b")).unwrap();
        let ws = Workspace::open(&base.join("b/karas")).unwrap();
        assert_eq!(ws.input().unwrap(), base.join("b/scans").canonicalize().unwrap());
    }

    #[test]
    fn a_second_open_of_the_same_workspace_is_refused() {
        let root = scratch("locked").join("karas");
        let _first = Workspace::create(&root).unwrap();
        assert!(matches!(Workspace::open(&root), Err(AppError::Locked { .. })));
    }

    #[test]
    fn a_folder_that_is_not_a_workspace_and_a_newer_workspace_are_both_refused() {
        let empty = scratch("not-one");
        assert!(matches!(Workspace::open(&empty), Err(AppError::NotAWorkspace { .. })));
        let newer = scratch("newer");
        std::fs::write(newer.join(WORKSPACE_FILE), r#"{"version":99,"excluded":[]}"#).unwrap();
        assert!(matches!(Workspace::open(&newer), Err(AppError::Version { found: 99, .. })));
    }
}
