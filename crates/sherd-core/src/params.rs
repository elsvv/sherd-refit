//! [`Params`] — every threshold of the matcher, in one place (R §1.1).
//!
//! Field names, types, order and defaults are the Python `sherd_refit.matching.Params`'s. They
//! are serialised under exactly those names, because the parity harness compares this struct
//! against the `collection.params` block of a Python fixture manifest (D §10.1) and because the
//! CLI exposes each of them as a flag (R §1.4).
//!
//! Every distance threshold comes as a pair `(k, m)` and resolves to `max(k·t_pair, m·res_pair)`
//! (R §1.2): `k·t` is the scale-free part — a wall thickness means the same thing on a 39-unit
//! terracotta relief and on a 3.5 mm pot wall — and `m·res` is the resolution floor that stops
//! the pipeline from demanding a precision the mesh cannot carry. The `*_res` floors are set
//! just below `k/0.058`, the coarsest working mesh the terracotta reference set produces, so on
//! that set no floor binds. See `docs/superpowers/notes/2026-09-05-thin-walls.md`.

use serde::{Deserialize, Serialize};

/// Thresholds of the matcher; `Params::default()` is the reference's default run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    /// Degrees; tolerance on `|dih_A + dih_B − 180|` when a hypothesis is formed (R §5.1).
    pub dihedral_tol: f64,
    /// `t`; breakline proximity for the coarse score (R §5.2).
    pub coarse_delta: f64,
    /// B breakline points used by the coarse score (R §5.2).
    pub coarse_points: u32,
    /// Hypotheses refined with breakline ICP (R §5.3).
    pub stage1: u32,
    /// Candidates refined with full ICP (R §5.5).
    pub stage2: u32,
    /// `t`; breakline proximity when stage-1 poses are re-scored (R §5.4).
    pub stage1_delta: f64,
    /// `t`; distance for the tight-contact fraction (R §6.1).
    pub tight_delta: f64,
    /// `t`; fracture points considered to face the other fragment (R §6.1).
    pub facing_delta: f64,
    /// `t`; breakline proximity counted as a shared seam (R §6.2).
    pub seam_delta: f64,
    /// `t`; shell-margin radius of the continuity test (R §6.3).
    pub near_delta: f64,
    /// `t`; penetration depth counted (R §6.4).
    pub pen_delta: f64,
    /// `t`; translation radius of the non-maximum suppression, no resolution floor (R §5.3).
    pub nms_delta: f64,
    /// `t`; finest rung of the ICP ladder — the whole ladder scales with it (R §5.4–5.6).
    pub icp_delta: f64,
    /// Working-mesh edges; floor under `coarse_delta`.
    pub coarse_res: f64,
    /// Working-mesh edges; floor under `stage1_delta`.
    pub stage1_res: f64,
    /// Working-mesh edges; floor under `tight_delta`.
    pub tight_res: f64,
    /// Working-mesh edges; floor under `facing_delta` (chosen so that it never binds, R §1.1).
    pub facing_res: f64,
    /// Working-mesh edges; floor under `max_gap`.
    pub gap_res: f64,
    /// Working-mesh edges; floor under `seam_delta`.
    pub seam_res: f64,
    /// Working-mesh edges; floor under `near_delta`.
    pub near_res: f64,
    /// Working-mesh edges; floor under `pen_delta`.
    pub pen_res: f64,
    /// Working-mesh edges; floor under `icp_delta`.
    pub icp_res: f64,
    /// Fraction; a candidate needs at least this much tight contact (R §6.5).
    pub min_tight: f64,
    /// `t`; median gap a candidate may not exceed (R §6.5).
    pub max_gap: f64,
    /// Fraction of surface samples allowed to sit inside the other fragment (R §6.5, R §7).
    pub max_pen: f64,
    /// `t`; shortest seam a join must share (R §6.5).
    pub min_seam: f64,
    /// Cosine; continuity of the shell across the seam (R §6.5).
    pub min_cont_n: f64,
    /// Fraction; above 0 skips the fracture-only ICPs and the costly verification below it
    /// (R §5.6, off by default).
    pub early_reject_tight: f64,
    /// Fraction; above 0 skips stage 2 when the pair's best stage-1 breakline score is below it
    /// (R §5.4, off by default).
    pub stage1_floor: f64,
    /// A pair whose walls differ by more than this ratio is not matched at all (R §4.1).
    pub thick_ratio: f64,
    /// Partners kept per fragment by the screening pass; 0 disables screening (R §4.3).
    pub screen_top_k: u32,
    /// Breakline points per fragment the screening pass uses (R §4.3).
    pub screen_points: u32,
    /// Screening is skipped below this many pairs (R §4.3).
    pub screen_min_pairs: u32,
    /// Partners of each unplaced fragment rematched with a larger budget; 0 disables the second
    /// pass (R §8.1).
    pub second_pass_top: u32,
    /// `stage1` for that second pass (R §8.1).
    pub second_pass_stage1: u32,
    /// `stage2` for that second pass (R §8.1).
    pub second_pass_stage2: u32,
    /// Shell-margin points kept for the `pc_reg` ICP and the continuity test (R §3.5.6).
    pub margin_points: u32,
    /// Points in the cloud the two coarse stage-2 ICPs run on; 0 means all (R §3.6).
    pub reg_points: u32,
    /// Whole-surface samples per fragment, for the penetration test and the shell margin
    /// (R §3.5.1).
    pub surface_points: u32,
    /// `t`; faces this close to the breakline are left out of the macro normals (R §3.5.4).
    pub macro_inner: f64,
    /// `t`; outer radius of the macro-normal neighbourhood (R §3.5.4).
    pub macro_outer: f64,
    /// `t`; voxel the breakline is thinned to before frames are paired (R §3.5.5).
    pub brk_voxel: f64,
    /// Fracture samples per `t²` of fracture area (R §3.5.2).
    pub frac_per_t2: f64,
    /// Lower bound on the number of fracture samples (R §3.5.2).
    pub min_frac_points: u32,
    /// Upper bound on the number of fracture samples (R §3.5.2).
    pub max_frac_points: u32,
    /// Seed of every draw the pipeline makes (R §10).
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            dihedral_tol: 25.0,
            coarse_delta: 0.15,
            coarse_points: 60,
            stage1: 250,
            stage2: 10,
            stage1_delta: 0.06,
            tight_delta: 0.01,
            facing_delta: 0.3,
            seam_delta: 0.12,
            near_delta: 0.5,
            pen_delta: 0.06,
            nms_delta: 0.5,
            icp_delta: 0.04,
            coarse_res: 2.3,
            stage1_res: 0.9,
            tight_res: 0.15,
            facing_res: 1.0,
            gap_res: 0.45,
            seam_res: 1.8,
            near_res: 4.0,
            pen_res: 0.9,
            icp_res: 0.6,
            min_tight: 0.25,
            max_gap: 0.03,
            max_pen: 0.005,
            min_seam: 3.0,
            min_cont_n: 0.8,
            early_reject_tight: 0.0,
            stage1_floor: 0.0,
            thick_ratio: 2.5,
            screen_top_k: 0,
            screen_points: 150,
            screen_min_pairs: 200,
            second_pass_top: 0,
            second_pass_stage1: 400,
            second_pass_stage2: 40,
            margin_points: 6000,
            reg_points: 6000,
            surface_points: 20000,
            macro_inner: 0.15,
            macro_outer: 0.60,
            brk_voxel: 0.5,
            frac_per_t2: 150.0,
            min_frac_points: 5000,
            max_frac_points: 12000,
            seed: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Params;

    #[test]
    fn round_trips_through_json_under_the_python_names() {
        let p = Params::default();
        let json = serde_json::to_value(p).expect("Params serialises");
        let object = json.as_object().expect("an object");
        assert_eq!(object.len(), 46, "R §1.1 lists 46 parameters");
        assert_eq!(object["dihedral_tol"], 25.0);
        assert_eq!(object["stage1"], 250);
        assert_eq!(object["seed"], 0);
        let back: Params = serde_json::from_value(json).expect("Params deserialises");
        assert_eq!(back, p);
    }

    #[test]
    fn an_unknown_parameter_is_an_error() {
        let mut json = serde_json::to_value(Params::default()).expect("Params serialises");
        json.as_object_mut().expect("an object").insert("no_such_knob".into(), 1.into());
        assert!(serde_json::from_value::<Params>(json).is_err());
    }
}
