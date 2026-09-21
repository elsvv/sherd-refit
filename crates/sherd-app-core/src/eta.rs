//! What this machine takes per pair (A §6).
//!
//! The launch sheet promises a number of minutes before anything has started, and the only honest
//! source for it is what *this* computer did last time: the same collection on a laptop and on a
//! workstation are an hour apart, and no constant compiled into the app can know which one it is
//! running on. So every finished run leaves behind two measurements — the seconds one pair of
//! fragments cost, and what the stages after matching cost as a fraction of it — and the next
//! launch sheet multiplies them by the pairs it is about to do.
//!
//! The average is exponential with weight ½ rather than a plain mean: a machine gets a new GPU,
//! a collection is replaced by a coarser one, and the estimate should follow within a run or two
//! instead of being dragged back by every run since the app was installed. Until the first run
//! finishes there is nothing measured, and [`Calibration::estimate`] says so with `None` — A §6
//! asks the sheet to offer «оценка появится через минуту» rather than to invent a figure.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::run::RunCounts;
use crate::{AppError, Result, atomic};

/// The calibration's file name in the app's config folder (A §2.1: host-side code writes it).
pub const CALIBRATION_FILE: &str = "calibration.json";

/// The format this build writes.
pub const CALIBRATION_VERSION: u32 = 1;

/// The stage the whole estimate is measured against: A §6 records every other stage as a ratio to
/// it, because matching is 94 % of a run's wall time and the only one that scales with the pairs.
const MATCHING: &str = "matching";

/// Preprocessing is per fragment and mostly cached (A §5), so its seconds are neither a function
/// of the pairs nor a stable fraction of matching: it is left out of the ratios on purpose.
const PREPROCESS: &str = "preprocess";

/// What the last runs on this machine measured — enough to say how long the next one will take.
///
/// Derived data, cheap to relearn: nothing here is the user's, and A §6 lets a damaged or missing
/// file start again from `karas`'s published ratios rather than fail anything.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// [`CALIBRATION_VERSION`].
    pub version: u32,
    /// Finished runs that taught it something. Zero means nothing is measured yet.
    pub runs: u32,
    /// Seconds one pair costs in matching on this machine.
    pub pair_seconds: f64,
    /// Every stage after matching, as a fraction of matching's seconds; keyed by R §11.2's stage
    /// name. A §6's `karas` figures until a run of this machine's own replaces them.
    pub ratios: BTreeMap<String, f64>,
}

impl Default for Calibration {
    /// A §6: with no history, `karas`'s ratios — tiers 3.3 %, refine 1.5 %, output 1.6 % of
    /// matching — so that the *shape* of a run is right from the first estimate even though its
    /// scale is not yet known.
    fn default() -> Self {
        Self {
            version: CALIBRATION_VERSION,
            runs: 0,
            pair_seconds: 0.0,
            ratios: [("tiers", 0.033), ("refine", 0.015), ("output", 0.016)]
                .into_iter()
                .map(|(stage, ratio)| (stage.to_owned(), ratio))
                .collect(),
        }
    }
}

impl Calibration {
    /// Takes what a finished run measured into the average (A §6).
    ///
    /// A run teaches nothing unless it matched at least one pair and spent a positive, finite
    /// number of seconds doing it — a cancelled run, a run whose whole matching came out of the
    /// cache, and a `counts` a worker filled with nonsense all leave the calibration as it was
    /// rather than poison every later estimate.
    ///
    /// The first run is taken whole; every run after it counts for half against everything known
    /// before. A stage missing from `timings` keeps whatever was known about it: the run simply
    /// says nothing about that stage.
    pub fn learn(&mut self, counts: &RunCounts) {
        let Some(matching) = counts
            .timings
            .iter()
            .find(|timing| timing.stage == MATCHING)
            .map(|timing| timing.seconds)
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        else {
            return;
        };
        if counts.pairs == 0 {
            return;
        }
        #[allow(clippy::cast_precision_loss, reason = "a run's pair count is far below 2^53")]
        let pairs = counts.pairs as f64;

        let first = self.runs == 0;
        // The exponential average, at weight ½ — see the module's note on why it is not a mean.
        let blend =
            |known: f64, measured: f64| if first { measured } else { (known + measured) / 2.0 };

        self.pair_seconds = blend(self.pair_seconds, matching / pairs);
        for timing in &counts.timings {
            if timing.stage == MATCHING || timing.stage == PREPROCESS {
                continue;
            }
            if !timing.seconds.is_finite() || timing.seconds < 0.0 {
                continue;
            }
            let measured = timing.seconds / matching;
            let known = self.ratios.entry(timing.stage.clone()).or_insert(measured);
            *known = blend(*known, measured);
        }
        self.runs = self.runs.saturating_add(1);
    }

    /// Seconds a run of `pairs` pairs will take, or `None` while nothing has been measured (A §6:
    /// the sheet then says an estimate will appear a minute into the run instead of guessing).
    pub fn estimate(&self, pairs: usize) -> Option<f64> {
        if self.runs == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss, reason = "a run's pair count is far below 2^53")]
        let pairs = pairs as f64;
        let after_matching: f64 = self.ratios.values().sum();
        Some(pairs * self.pair_seconds * (1.0 + after_matching))
    }

    /// Reads the calibration, or starts again from [`Calibration::default`].
    ///
    /// Infallible on purpose: a run must not be refused because a derived file on the side is
    /// missing (the first launch), truncated by a crash, or written by a newer build. Anything
    /// unreadable is said once through `tracing` and then forgotten.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let Ok(bytes) = std::fs::read(path) else {
            return Self::default();
        };
        match serde_json::from_slice::<Self>(&bytes) {
            Ok(file) if file.version > CALIBRATION_VERSION => {
                tracing::warn!(
                    path = %path.display(),
                    found = file.version,
                    expected = CALIBRATION_VERSION,
                    "the calibration was written by a newer build; learning it again"
                );
                Self::default()
            }
            Ok(file) if !file.is_usable() => {
                tracing::warn!(
                    path = %path.display(),
                    "the calibration holds numbers no run could have measured; learning it again"
                );
                Self::default()
            }
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "the calibration does not parse; learning it again"
                );
                Self::default()
            }
        }
    }

    /// Writes the calibration whole or not at all, creating the config folder if this is the
    /// app's first run (A §2.1: host-side files are written atomically).
    ///
    /// # Errors
    ///
    /// [`AppError::Io`] when the folder cannot be created or written.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(folder) = path.parent()
            && !folder.as_os_str().is_empty()
        {
            std::fs::create_dir_all(folder).map_err(|source| AppError::io(folder, source))?;
        }
        atomic::write_json(path, self)
    }

    /// Whether the numbers could have come off a clock: a NaN or a negative would travel straight
    /// into the sheet's «осталось ≈ …» and out again as nonsense.
    fn is_usable(&self) -> bool {
        self.pair_seconds.is_finite()
            && self.pair_seconds >= 0.0
            && self.ratios.values().all(|ratio| ratio.is_finite() && *ratio >= 0.0)
    }
}

/// Pairs a run of `fragments` scans can have at most: `n·(n−1)/2`.
///
/// An upper bound and not the count: R §4.1's wall-ratio filter skips pairs whose thicknesses
/// cannot belong together, and how many that is nobody knows before matching starts — which is
/// why the launch sheet says «до N пар» and the estimate built on it is a ceiling.
#[must_use]
pub fn pairs_upper_bound(fragments: usize) -> usize {
    fragments.saturating_mul(fragments.saturating_sub(1)) / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::StageTime;

    /// `karas`'s own run, as R §11.2 wrote it down.
    fn karas() -> RunCounts {
        RunCounts {
            pairs: 11_097,
            timings: [
                ("preprocess", 0.1),
                ("matching", 996.9),
                ("tiers", 33.2),
                ("assembly", 0.1),
                ("refine", 15.3),
            ]
            .into_iter()
            .map(|(stage, seconds)| StageTime { stage: stage.to_owned(), seconds })
            .collect(),
            ..RunCounts::default()
        }
    }

    /// A §6: what one pair costs is matching's seconds over the pairs, and every later stage is
    /// a fraction of matching — the first run is believed whole.
    #[test]
    fn a_finished_run_says_what_a_pair_costs_and_what_the_later_stages_add() {
        let mut calibration = Calibration::default();
        calibration.learn(&karas());
        assert_eq!(calibration.runs, 1);
        assert!((calibration.pair_seconds - 0.089_84).abs() < 1e-4, "{calibration:?}");
        assert!((calibration.ratios["tiers"] - 0.033_3).abs() < 1e-4, "{calibration:?}");
        assert!((calibration.ratios["refine"] - 15.3 / 996.9).abs() < 1e-9, "{calibration:?}");
        // a stage the run never reported keeps what A §6 published about `karas`
        assert!((calibration.ratios["output"] - 0.016).abs() < 1e-12, "{calibration:?}");
    }

    /// The average is exponential at weight ½, so a machine that has become twice as slow is met
    /// half way rather than ignored or believed outright.
    #[test]
    fn a_second_run_moves_the_average_half_way_towards_it() {
        let mut calibration = Calibration::default();
        calibration.learn(&karas());
        let first = calibration.pair_seconds;

        let mut slow = karas();
        for timing in &mut slow.timings {
            timing.seconds *= 2.0;
        }
        calibration.learn(&slow);

        assert_eq!(calibration.runs, 2);
        assert!((calibration.pair_seconds - (first + first * 2.0) / 2.0).abs() < 1e-12);
        // the ratios are unchanged: every stage doubled with matching
        assert!((calibration.ratios["tiers"] - 33.2 / 996.9).abs() < 1e-12);
    }

    /// A §6: no estimate before the first run; after it, the pairs at the measured rate plus what
    /// the stages after matching add.
    #[test]
    fn an_estimate_needs_a_run_behind_it() {
        let mut calibration = Calibration::default();
        assert_eq!(calibration.estimate(11_097), None);

        calibration.learn(&karas());
        let after_matching: f64 = calibration.ratios.values().sum();
        let want = 11_097.0 * calibration.pair_seconds * (1.0 + after_matching);
        let got = calibration.estimate(11_097).expect("a run has been learnt");
        assert!((got - want).abs() < 1e-9, "{got} != {want}");
        assert!(got > 996.9, "the later stages add to matching, they do not replace it");
    }

    /// A run that measured nothing usable — cancelled before matching, or a worker's nonsense —
    /// must leave the calibration exactly as it was rather than divide by zero into it.
    #[test]
    fn a_run_with_nothing_to_measure_teaches_nothing() {
        let mut taught = Calibration::default();
        taught.learn(&karas());

        let mut no_pairs = taught.clone();
        no_pairs.learn(&RunCounts { pairs: 0, ..karas() });
        assert_eq!(no_pairs, taught);

        let mut no_matching = taught.clone();
        let timings = karas().timings.into_iter().filter(|t| t.stage != MATCHING).collect();
        no_matching.learn(&RunCounts { timings, ..karas() });
        assert_eq!(no_matching, taught);

        let mut zero_matching = taught.clone();
        let timings = vec![StageTime { stage: MATCHING.to_owned(), seconds: 0.0 }];
        zero_matching.learn(&RunCounts { timings, ..karas() });
        assert_eq!(zero_matching, taught);

        let mut fresh = Calibration::default();
        fresh.learn(&RunCounts { pairs: 0, ..karas() });
        assert_eq!(fresh, Calibration::default());
    }

    /// The file is derived data on the side: missing or damaged, it starts again instead of
    /// stopping a run.
    #[test]
    fn a_missing_or_damaged_file_is_the_published_default() {
        let folder = std::env::temp_dir().join(format!("sherd-eta-{}", std::process::id()));
        std::fs::remove_dir_all(&folder).ok();
        let path = folder.join(CALIBRATION_FILE);
        assert_eq!(Calibration::load(&path), Calibration::default());

        let mut taught = Calibration::default();
        taught.learn(&karas());
        taught.save(&path).expect("the folder is created on the way");
        assert_eq!(Calibration::load(&path), taught);

        std::fs::write(&path, "garbage").expect("the folder is there now");
        assert_eq!(Calibration::load(&path), Calibration::default());

        std::fs::remove_dir_all(&folder).ok();
    }

    /// The launch sheet's «до N пар» (R §4.1 skips some of them by wall ratio).
    #[test]
    fn the_pairs_of_a_collection_are_bounded_by_every_two_of_it() {
        assert_eq!(pairs_upper_bound(155), 11_935);
        assert_eq!(pairs_upper_bound(0), 0);
        assert_eq!(pairs_upper_bound(1), 0);
        assert_eq!(pairs_upper_bound(2), 1);
    }
}
