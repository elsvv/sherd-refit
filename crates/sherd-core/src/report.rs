//! `transforms.json`, `report.json` and `report.md` (R §11.1, R §6.5).
//!
//! Same file names, same JSON keys and same section order as the reference, plus the additive
//! `engine` key that records `core_version`, `algo_ref` and the backend that actually ran
//! (D §4.3); the Python readers ignore unknown keys. Filled in in phase 1d.
