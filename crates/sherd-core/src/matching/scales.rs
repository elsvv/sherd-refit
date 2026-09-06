//! `Scales` (R §1.2): every distance the matcher uses, resolved once per pair from
//! `t_pair = min(t_A, t_B)` and `res_pair = max(res_A, res_B)` by `f(k, m) = max(k·t, m·res)`,
//! so the two-term rule of [`crate::Params`] lives in exactly one place. Filled in in phase 1c.
