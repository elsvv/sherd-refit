//! What a command answers when it cannot do what the window asked (A §10).
//!
//! The window never sees a Rust error. Every failure crosses the IPC boundary as a `kind` the
//! frontend branches on and a `message` it may show beside its own sentence: A §10's table pairs
//! each kind with a wording and an action («Показать лог», «Повторить на CPU»), and those
//! wordings are translations in `ru.json`/`en.json`, not the engine's English.

use serde::Serialize;
use sherd_app_core::AppError;

/// A failed command, as the window receives it (A §10).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct CommandError {
    /// One of `locked`, `not_a_workspace`, `version`, `io`, `json`, `engine`, `worker`, `busy`,
    /// `no_workspace`. A closed set, because the frontend switches on it.
    pub(crate) kind: String,
    /// The reason in words. Shown as detail under the translated sentence, never instead of it.
    pub(crate) message: String,
}

impl CommandError {
    /// A failure of `kind` with `message`.
    fn of(kind: &str, message: impl Into<String>) -> Self {
        Self { kind: kind.to_owned(), message: message.into() }
    }

    /// A job is on the worker, and what was asked would need it too (A §5: one worker at a time).
    pub(crate) fn busy() -> Self {
        Self::of("busy", "a job is running")
    }

    /// The command needs an open workspace and there is none — the window is on the welcome
    /// screen, or it asked after `workspace_close`.
    pub(crate) fn no_workspace() -> Self {
        Self::of("no_workspace", "no workspace is open")
    }

    /// A lock of this process is poisoned: a thread panicked while holding the workspace or the
    /// job slot. A §10 has no kind for «this app is damaged», and `worker` is the nearest — the
    /// only thread that holds these locks besides a command is the one driving a job.
    pub(crate) fn poisoned(what: &str) -> Self {
        Self::of("worker", format!("the app's {what} is no longer usable after a panic"))
    }

    /// A review command arrived with no session open (A §8) — the window left the mode, or a run
    /// took the worker. `worker` for [`poisoned`](Self::poisoned)'s reason: the closed set has no
    /// kind of its own for it, and what it is about is the engine process that is not there.
    pub(crate) fn no_session() -> Self {
        Self::of("worker", "no review session is open")
    }

    /// The window sent something the shell will not write into the workspace — a `decisions.json`
    /// this build could not read back. `json` because that is the kind A §10 pairs with «файл
    /// повреждён», and a file that would be is exactly what this refuses to make.
    pub(crate) fn malformed(message: impl Into<String>) -> Self {
        Self::of("json", message)
    }
}

impl From<AppError> for CommandError {
    /// A §10: each variant keeps its own kind, so the window can offer the right action; the
    /// engine's own errors all arrive as `engine`, because what the user can do about them is
    /// read the message.
    fn from(error: AppError) -> Self {
        let kind = match &error {
            AppError::Io { .. } => "io",
            AppError::Json { .. } => "json",
            AppError::NotAWorkspace { .. } => "not_a_workspace",
            AppError::Locked { .. } => "locked",
            AppError::Version { .. } => "version",
            AppError::Core(_) => "engine",
            AppError::Worker(_) => "worker",
        };
        Self::of(kind, error.to_string())
    }
}

impl From<tauri::Error> for CommandError {
    /// The shell's own failures — no config directory on this machine, a workspace path the asset
    /// protocol's globs cannot express — are all about a path, which is what `io` means here.
    fn from(error: tauri::Error) -> Self {
        Self::of("io", error.to_string())
    }
}
