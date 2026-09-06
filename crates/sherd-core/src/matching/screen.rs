//! The optional screening pass (R §4.3): a cheap breakline-only score over all pairs that keeps
//! `screen_top_k` partners per fragment before the expensive matching runs, skipped below
//! `screen_min_pairs`. Off by default. Filled in in phase 1c.
