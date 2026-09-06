//! The Rust side of the parity harness (D §10.1): `--dump-fixtures DIR` writes the same layout of
//! `.npy` and JSON files at the same stage boundaries as the Python sink
//! (`sherd_refit/fixture.py`), so that `tools/compare_fixtures.py REF CAND` can judge a Rust run
//! against a Python one. Reading fixtures — the injected mode — lives in the `sherd-parity`
//! crate. Filled in by plan step S4.
