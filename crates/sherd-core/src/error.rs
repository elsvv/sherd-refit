//! The crate's error type.
//!
//! The library never prints and never exits: every failure that the reference reports as a
//! skipped file, an unusable cache or a rejected mesh comes back as an [`Error`] and the caller
//! (the CLI, the pipeline, later the desktop app) decides what to say about it.

use std::path::{Path, PathBuf};

/// Everything `sherd-core` can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A mesh file could not be read or does not parse.
    #[error("{path}: {message}")]
    Read {
        /// The file that failed.
        path: PathBuf,
        /// What went wrong, in the reader's words.
        message: String,
    },

    /// A file whose extension is not one of the formats of R §2.
    #[error("{path}: unsupported mesh format `{extension}`")]
    UnsupportedFormat {
        /// The file that was offered.
        path: PathBuf,
        /// Its extension, lower-cased, or the empty string when it has none.
        extension: String,
    },

    /// An output file could not be written.
    #[error("{path}: {message}")]
    Write {
        /// The file that failed.
        path: PathBuf,
        /// What went wrong.
        message: String,
    },

    /// A mesh has no faces left after cleaning and largest-component extraction (R §3.1).
    #[error("fragment `{name}`: no faces left after cleaning")]
    EmptyMesh {
        /// The fragment's name.
        name: String,
    },

    /// A cache file exists but cannot be used; the caller recomputes the fragment (R §3.7).
    #[error("cache {path}: {message}")]
    Cache {
        /// The cache file.
        path: PathBuf,
        /// Why it was rejected — a version mismatch, a changed source, a broken header.
        message: String,
    },

    /// A parity fixture is missing a stage, or holds something other than what the stage expects
    /// (D §10.1).
    #[error("fixture {path}: {message}")]
    Fixture {
        /// The fixture file or directory.
        path: PathBuf,
        /// What is missing or unexpected.
        message: String,
    },

    /// An I/O failure with no file to blame.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Builds a [`Error::Read`] from anything that can describe itself.
    pub fn read(path: impl AsRef<Path>, message: impl std::fmt::Display) -> Self {
        Self::Read { path: path.as_ref().to_path_buf(), message: message.to_string() }
    }

    /// Builds a [`Error::Write`] from anything that can describe itself.
    pub fn write(path: impl AsRef<Path>, message: impl std::fmt::Display) -> Self {
        Self::Write { path: path.as_ref().to_path_buf(), message: message.to_string() }
    }

    /// Builds a [`Error::Cache`] from anything that can describe itself.
    pub fn cache(path: impl AsRef<Path>, message: impl std::fmt::Display) -> Self {
        Self::Cache { path: path.as_ref().to_path_buf(), message: message.to_string() }
    }

    /// Builds a [`Error::Fixture`] from anything that can describe itself.
    pub fn fixture(path: impl AsRef<Path>, message: impl std::fmt::Display) -> Self {
        Self::Fixture { path: path.as_ref().to_path_buf(), message: message.to_string() }
    }
}

/// The crate's result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn messages_name_the_file() {
        let e = Error::read("/tmp/a.ply", "unexpected end of header");
        assert_eq!(e.to_string(), "/tmp/a.ply: unexpected end of header");
        let e = Error::cache("/tmp/a.sherd", "cache_version 0, expected 1");
        assert_eq!(e.to_string(), "cache /tmp/a.sherd: cache_version 0, expected 1");
        let e = Error::EmptyMesh { name: "frag_001".into() };
        assert_eq!(e.to_string(), "fragment `frag_001`: no faces left after cleaning");
    }

    #[test]
    fn io_errors_convert() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        let e: Error = io.into();
        assert!(matches!(e, Error::Io(_)));
    }
}
