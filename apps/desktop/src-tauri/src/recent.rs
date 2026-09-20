//! The workspaces this machine opened last (A §4: «recent workspaces … live in the OS's
//! app-config directory, not in a workspace»), which is the welcome screen's list.
//!
//! Only the path and the moment are written down. The folder's name and whether it is still there
//! are worked out at every read, because a list on disk that claims a workspace exists is a lie as
//! soon as someone moves a folder — and A §5's «вход недоступен» row already says the app must be
//! able to show a row for something that is gone rather than drop it silently.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sherd_app_core::{AppError, atomic};

/// The file, inside Tauri's `app_config_dir`.
pub(crate) const RECENT_FILE: &str = "recent.json";

/// How many are kept. Enough for the welcome screen's list without a scrollbar.
const KEEP: usize = 12;

/// One row of the welcome screen (A §7.4).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct RecentEntry {
    /// The workspace folder.
    pub(crate) path: String,
    /// The folder's own name, which is what the app calls the workspace.
    pub(crate) name: String,
    /// When it was last opened, as [`sherd_app_core::run::timestamp`] writes a time.
    pub(crate) opened_at: String,
    /// Whether the folder is there now. A row that is gone stays in the list, greyed: a workspace
    /// on an unplugged disk is not a workspace the user wants forgotten.
    pub(crate) available: bool,
}

/// What `recent.json` holds: only what cannot be worked out again.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Remembered {
    /// The workspace folder, canonical.
    path: PathBuf,
    /// When it was last opened.
    opened_at: String,
}

/// The list, newest first.
///
/// A missing, damaged or hand-edited `recent.json` reads as no recent workspaces. It is a
/// convenience file with no state of the user's work in it, and refusing to show the welcome
/// screen over it would be the worst of both.
pub(crate) fn load(config_dir: &Path) -> Vec<RecentEntry> {
    remembered(config_dir)
        .into_iter()
        .map(|entry| {
            let name = entry
                .path
                .file_name()
                .map_or_else(|| entry.path.to_string_lossy(), OsStr::to_string_lossy)
                .into_owned();
            let available = entry.path.is_dir();
            RecentEntry {
                path: entry.path.to_string_lossy().into_owned(),
                name,
                opened_at: entry.opened_at,
                available,
            }
        })
        .collect()
}

/// Puts `root` at the top of the list, as opened at `now`, and keeps the newest [`KEEP`].
///
/// Nothing is returned: the list is a convenience, and a workspace that opened must not fail to
/// open because the app's own config folder is read-only. A failure is logged and dropped.
pub(crate) fn touch(config_dir: &Path, root: &Path, now: &str) {
    // Canonical, so that the same workspace reached by two paths — a symlink, `/var` against
    // `/private/var` on macOS — is one row and not two.
    let root = root.canonicalize().unwrap_or_else(|_| root.to_owned());
    let mut list = remembered(config_dir);
    list.retain(|entry| entry.path != root);
    list.insert(0, Remembered { path: root, opened_at: now.to_owned() });
    list.truncate(KEEP);
    let written = std::fs::create_dir_all(config_dir)
        .map_err(|e| AppError::io(config_dir, e))
        .and_then(|()| atomic::write_json(&config_dir.join(RECENT_FILE), &list));
    if let Err(error) = written {
        tracing::warn!(%error, "the list of recent workspaces could not be written");
    }
}

/// The file as it stands, or nothing at all.
fn remembered(config_dir: &Path) -> Vec<Remembered> {
    let path = config_dir.join(RECENT_FILE);
    if !path.is_file() {
        return Vec::new();
    }
    atomic::read_json(&path).unwrap_or_else(|error| {
        tracing::warn!(%error, "the list of recent workspaces could not be read; starting empty");
        Vec::new()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recent_list_is_newest_first_without_duplicates_and_says_what_is_gone() {
        let dir = std::env::temp_dir().join(format!("sherd-recent-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let (a, b) = (dir.join("ws/a"), dir.join("ws/b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        touch(&dir, &a, "2026-09-20T10:00:00+03:00");
        touch(&dir, &b, "2026-09-20T11:00:00+03:00");
        touch(&dir, &a, "2026-09-20T12:00:00+03:00");
        std::fs::remove_dir_all(&b).unwrap();
        let list = load(&dir);
        assert_eq!(list.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!((list[0].available, list[1].available), (true, false));
        assert_eq!(list[0].opened_at, "2026-09-20T12:00:00+03:00");
        std::fs::remove_dir_all(&dir).ok();
    }
}
