//! A §11's «Настройки»: the answers that belong to this computer, read and written by the shell.
//!
//! The file itself and every rule about it are `sherd_app_core::settings` — this is where the
//! shell keeps it (A §4: the app's config folder, beside A §6's calibration and the recent list)
//! and the two operations the window's commands are made of. Nothing here is a workspace's: a
//! folder carried to a colleague's laptop must not bring a memory limit chosen for this machine,
//! nor a path to a Blender that is not installed there.
//!
//! The commands themselves are [`crate::commands::settings_get`] and
//! [`crate::commands::settings_set`], with every other one: `main.rs` hands Tauri a single list
//! of `commands::…`, and `src/ipc/contract.test.ts` reads that list to check the window against
//! it.

use std::path::PathBuf;

use sherd_app_core::AppError;
use sherd_app_core::settings::{SETTINGS_FILE, SETTINGS_VERSION, Settings};
use tauri::AppHandle;

use crate::commands::config_dir;
use crate::error::CommandError;

/// Where the settings live on this machine, or `None` when it has no config folder at all.
///
/// An `Option` and not a `Result`, because the two callers want opposite things from the same
/// failure: [`current`] answers the defaults, and only [`write`] — which is being asked to
/// remember something — has anything to refuse.
fn path(app: &AppHandle) -> Option<PathBuf> {
    match config_dir(app) {
        Ok(dir) => Some(dir.join(SETTINGS_FILE)),
        Err(error) => {
            tracing::warn!(
                message = %error.message,
                "this machine has no config folder; the app runs on its default settings"
            );
            None
        }
    }
}

/// The settings as they stand, whatever state the file is in (A §11).
///
/// Infallible, like [`Settings::load`] itself and for the same reason: these are preferences, and
/// no state of them is worth a window that will not open or a «Подготовить» that will not start.
/// Read afresh every time rather than cached — the file is a few hundred bytes, it is read at
/// most once per command, and a copy in memory is one more thing that could disagree with the
/// disk (A §5's rule about second opinions, applied to the shell's own state).
pub(crate) fn current(app: &AppHandle) -> Settings {
    path(app).as_deref().map(Settings::load).unwrap_or_default()
}

/// Writes them and answers with what was written.
///
/// A limit the engine could not act on is refused before anything is written, exactly as
/// [`crate::commands::review_apply`] refuses a `decisions.json` it could not read back: a file
/// this build would fall back to the defaults over is not one worth writing.
///
/// # Errors
///
/// [`CommandError`] of kind `json` when the settings hold a number no control could have
/// produced, or `io` when this machine has no config folder or the file cannot be written.
pub(crate) fn write(app: &AppHandle, settings: Settings) -> Result<Settings, CommandError> {
    if !settings.is_usable() {
        return Err(CommandError::malformed(format!(
            "{SETTINGS_FILE}: a memory limit of {:?} GB and {} threads is not a setting this \
             build would read back",
            settings.memory_gb, settings.workers
        )));
    }
    // Stamped with this build's format and not with whatever arrived: the shell owns the file,
    // and a `version` from the window would be the one field of it the window could damage.
    let settings = Settings { version: SETTINGS_VERSION, ..settings };
    let path = path(app).ok_or_else(|| {
        CommandError::from(AppError::io(
            PathBuf::from(SETTINGS_FILE),
            std::io::Error::other("this machine has no config folder to keep the settings in"),
        ))
    })?;
    settings.save(&path)?;
    Ok(settings)
}
