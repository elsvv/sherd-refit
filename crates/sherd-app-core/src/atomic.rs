//! State files are replaced whole or not at all (A §2.1): `decisions.json` is written on every
//! click of a review, and a crash between two clicks must leave the previous file, not half of
//! the next one.

use std::io::Write;
use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{AppError, Result};

/// Writes `value` as pretty JSON to a temporary neighbour of `path` and renames it over `path`.
///
/// # Errors
///
/// [`AppError::Io`] when the folder cannot be written.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_vec_pretty(value)
        .map_err(|source| AppError::Json { path: path.to_owned(), source })?;
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(format!(".{}.tmp", std::process::id()));
    let temporary = path.with_file_name(name);
    let written = (|| {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&json)?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)
    })();
    written.map_err(|source| {
        std::fs::remove_file(&temporary).ok();
        AppError::io(path, source)
    })
}

/// Reads a JSON state file.
///
/// # Errors
///
/// [`AppError::Io`] when it cannot be read, [`AppError::Json`] when it does not parse.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|source| AppError::io(path, source))?;
    serde_json::from_slice(&bytes)
        .map_err(|source| AppError::Json { path: path.to_owned(), source })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_replaced_whole_and_leaves_no_temporary_behind() {
        let dir = std::env::temp_dir().join(format!("sherd-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        write_json(&path, &vec![1, 2, 3]).unwrap();
        write_json(&path, &vec![4]).unwrap();
        assert_eq!(read_json::<Vec<u32>>(&path).unwrap(), vec![4]);
        let left: Vec<_> =
            std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, [std::ffi::OsString::from("state.json")]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
