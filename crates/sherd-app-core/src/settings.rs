//! What belongs to the machine and not to a workspace (A §11).
//!
//! A workspace is a folder someone may copy onto a colleague's laptop, and three of the launch
//! sheet's answers have no business travelling with it: which executor this computer has, how
//! much memory the user is willing to let a run take, how many threads, and where Blender is
//! installed here. Those are properties of the computer, exactly as A §6's calibration is, so
//! they live beside it in the app's config folder and follow no folder anywhere.
//!
//! Nothing here is the user's work. A file that is missing, damaged or written by a newer build
//! is [`Settings::default`] and a line in the log — the app opens with its defaults rather than
//! refusing to start over a preferences file, which is A §10's rule for derived data on the side.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::protocol::BackendChoice;
use crate::{AppError, Result, atomic};

/// The settings' file name in the app's config folder (A §2.1: host-side code writes it).
pub const SETTINGS_FILE: &str = "settings.json";

/// The format this build writes.
pub const SETTINGS_VERSION: u32 = 1;

/// More threads than any machine this app runs on has, by a wide margin.
///
/// A number out of a file, not out of the window: the sheet offers a slider, and a hand-edited
/// `settings.json` asking for four billion threads would have `rayon::ThreadPoolBuilder` try to
/// make them. The check is not a policy about hardware — it is the line past which the number
/// cannot have been meant, and past it the file is read as damaged.
const MAX_WORKERS: usize = 1024;

/// What the app remembers about this computer (A §11).
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// [`SETTINGS_VERSION`].
    pub version: u32,
    /// The executor a launch sheet starts on (A §7.4), when the workspace has no sheet of its own
    /// to repeat.
    pub backend: BackendChoice,
    /// The memory budget in gigabytes, or `None` for D §5's default — half of what the machine
    /// has. A §11's «лимит памяти».
    pub memory_gb: Option<f64>,
    /// Worker threads; `0` is rayon's own default, which is every core.
    pub workers: usize,
    /// Where Blender is on this machine (A §9.2), when it is not where
    /// [`crate::blender::find_blender`] looks.
    pub blender_path: Option<PathBuf>,
}

impl Default for Settings {
    /// Every answer left to the machine: `auto` for the backend, the engine's own budget, every
    /// core, and Blender wherever it installed itself. This is what a first start reads, and what
    /// a damaged file falls back to.
    fn default() -> Self {
        Self {
            version: SETTINGS_VERSION,
            backend: BackendChoice::Auto,
            memory_gb: None,
            workers: 0,
            blender_path: None,
        }
    }
}

impl Settings {
    /// The settings at `path`, or [`Settings::default`] when there are none to read.
    ///
    /// Infallible on purpose, as [`crate::eta::Calibration::load`] is and for the same reason:
    /// this is a preferences file, not the user's work, and no state of it is worth a window that
    /// will not open. A file that does not parse, one a newer build wrote, and one holding
    /// numbers no slider could have produced are each logged and replaced by the defaults.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let Ok(bytes) = std::fs::read(path) else {
            return Self::default();
        };
        match serde_json::from_slice::<Self>(&bytes) {
            Ok(file) if file.version > SETTINGS_VERSION => {
                tracing::warn!(
                    path = %path.display(),
                    found = file.version,
                    expected = SETTINGS_VERSION,
                    "the settings were written by a newer build; this one starts from its defaults"
                );
                Self::default()
            }
            Ok(file) if !file.is_usable() => {
                tracing::warn!(
                    path = %path.display(),
                    "the settings hold a limit no sheet could have asked for; starting again"
                );
                Self::default()
            }
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "the settings do not parse; starting from the defaults"
                );
                Self::default()
            }
        }
    }

    /// Writes them to `path`, atomically, making the config folder when it is not there yet.
    ///
    /// # Errors
    ///
    /// [`AppError::Io`] when the folder cannot be made or the file cannot be written,
    /// [`AppError::Json`] when they cannot be serialised.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(folder) = path.parent()
            && !folder.as_os_str().is_empty()
        {
            std::fs::create_dir_all(folder).map_err(|source| AppError::io(folder, source))?;
        }
        atomic::write_json(path, self)
    }

    /// Whether the two numbers could have come off A §11's own controls.
    ///
    /// A budget that is not a positive, finite number of gigabytes and a thread count past
    /// [`MAX_WORKERS`] are the two ways a file edited by hand can reach the engine as something
    /// it will act on: `Budget::gigabytes` reads a zero or a NaN as «unbounded», which is the
    /// opposite of what a limit means, and `set_threads` would try to build the pool.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.memory_gb.is_none_or(|gb| gb.is_finite() && gb > 0.0) && self.workers <= MAX_WORKERS
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{MAX_WORKERS, SETTINGS_VERSION, Settings};
    use crate::protocol::BackendChoice;

    /// A folder of this test's own, emptied first.
    fn dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-settings-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A §11: the settings survive the app being closed, and nothing about them is worth a window
    /// that will not open — a file that is not there, does not parse, was written by a newer
    /// build or holds a limit no control could produce is the defaults and a line in the log.
    #[test]
    fn the_settings_come_back_as_they_were_written_and_a_damaged_file_is_the_defaults() {
        let dir = dir("roundtrip");
        let path = dir.join("settings.json");

        // Nothing written yet: the defaults, and the folder is left alone.
        assert_eq!(Settings::load(&path), Settings::default());
        assert!(!path.exists());

        let mine = Settings {
            version: SETTINGS_VERSION,
            backend: BackendChoice::Cpu,
            memory_gb: Some(12.5),
            workers: 6,
            blender_path: Some(PathBuf::from("/Applications/Blender.app")),
        };
        mine.save(&path).unwrap();
        assert_eq!(Settings::load(&path), mine);

        // Written into a folder that does not exist yet — the app's first start.
        let deeper = dir.join("config").join("sherd").join("settings.json");
        mine.save(&deeper).unwrap();
        assert_eq!(Settings::load(&deeper), mine);

        // Damaged, and a newer build's: both are the defaults rather than a refusal.
        std::fs::write(&path, b"{not json at all").unwrap();
        assert_eq!(Settings::load(&path), Settings::default());
        let newer = Settings { version: SETTINGS_VERSION + 1, ..mine.clone() };
        newer.save(&path).unwrap();
        assert_eq!(Settings::load(&path), Settings::default());

        // And the two numbers a hand edit can make nonsense of.
        for wrong in [
            Settings { memory_gb: Some(0.0), ..mine.clone() },
            Settings { memory_gb: Some(-4.0), ..mine.clone() },
            Settings { workers: MAX_WORKERS + 1, ..mine.clone() },
        ] {
            assert!(!wrong.is_usable(), "{wrong:?}");
            wrong.save(&path).unwrap();
            assert_eq!(Settings::load(&path), Settings::default());
        }
        // A NaN is not JSON at all — serde writes `null` for it — so it arrives as «no limit»
        // and is usable; the check is here for the file that says `"memory_gb": 0`.
        assert!(Settings { memory_gb: None, ..mine }.is_usable());

        std::fs::remove_dir_all(&dir).ok();
    }
}
