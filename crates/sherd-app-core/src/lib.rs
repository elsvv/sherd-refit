//! The desktop app without its window (A §2): the workspace on disk, the protocol between the
//! window's process and the engine's, the worker that speaks it, and the host that keeps a run's
//! files. Nothing here depends on Tauri, which is what lets all of it be tested headless.

pub mod atomic;
pub mod workspace;

use std::path::PathBuf;

/// What this crate fails with.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// A file or folder of the workspace could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The path.
        path: PathBuf,
        /// The OS's word.
        source: std::io::Error,
    },
    /// A state file does not parse.
    #[error("{path}: {source}")]
    Json {
        /// The file.
        path: PathBuf,
        /// serde's word.
        source: serde_json::Error,
    },
    /// The folder holds no `sherd-workspace.json`.
    #[error("{path} is not a sherd-refit workspace")]
    NotAWorkspace {
        /// The folder.
        path: PathBuf,
    },
    /// Another running app has the workspace open (A §10).
    #[error("{path} is open in another window of the app")]
    Locked {
        /// The workspace.
        path: PathBuf,
    },
    /// A state file written by a newer app.
    #[error(
        "{path} was written by a newer version of the app (format {found}, this build reads {expected})"
    )]
    Version {
        /// The file.
        path: PathBuf,
        /// Its version.
        found: u32,
        /// Ours.
        expected: u32,
    },
    /// The engine's own error.
    #[error(transparent)]
    Core(#[from] sherd_core::Error),
    /// The worker could not be started or spoken to.
    #[error("the engine process: {0}")]
    Worker(String),
}

impl AppError {
    /// An [`AppError::Io`] about `path`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io { path: path.into(), source }
    }
}

/// `Result` with [`AppError`].
pub type Result<T> = std::result::Result<T, AppError>;
