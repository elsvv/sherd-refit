//! Operator constraints (roadmap item 3, D §11): a `constraints.json` of `{must_join,
//! must_not_join}`. `must_not_join` removes pairs before matching and rejects candidates;
//! `must_join` forces a pair through the second-pass budget and accepts its best candidate that
//! passes the *probable* thresholds. Phase 1 fixes the file format only.
