//! One comparison, one stage, one table (D §10.2).
//!
//! Every stage runner produces [`Check`]s: a measured value, the reference's value, how far apart
//! they are and how far apart they are allowed to be. The distance and the tolerance are always in
//! the *same* unit — a fraction for a relative tolerance, wall thicknesses for a pose, histogram
//! bins for the thickness — so `deviation / tolerance` is comparable across quantities and the
//! table can report the worst one. That is the same shape `tools/compare_fixtures.py` prints on
//! the Python side, deliberately: the two tools are read side by side.
//!
//! A tolerance of zero means an exact comparison. Nothing is rounded into passing: a check with
//! tolerance zero and a non-zero deviation has an infinite ratio and prints as `exact`.

use std::fmt;

/// Which of D §10.2's two columns a stage was run under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The Rust stage ran on the Python stage's own inputs.
    Injected,
    /// The Rust stage ran on Rust's own upstream results.
    Native,
}

impl Mode {
    /// The word the table prints.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Injected => "injected",
            Self::Native => "native",
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a deviation is measured in — printed beside the numbers so a table row is readable
/// without the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// `|measured − reference| / |reference|`.
    Relative,
    /// The quantity's own units.
    Absolute,
    /// Bins of the reference's own thickness histogram (R §3.2).
    Bins,
    /// Units of the last place of an `f32` — how two readers of the same file differ.
    Ulps,
    /// A count of entries that differ.
    Entries,
    /// The two values are equal or they are not.
    Identical,
}

impl Unit {
    /// The suffix the table prints after a deviation.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Relative => "rel",
            Self::Absolute => "abs",
            Self::Bins => "bins",
            Self::Ulps => "ulp32",
            Self::Entries => "entries",
            Self::Identical => "",
        }
    }
}

/// One comparison between the port and the reference.
#[derive(Clone, Debug)]
pub struct Check {
    /// Fragment (or pair) the check belongs to.
    pub scope: String,
    /// What was compared: `res`, `faces`, `t`, …
    pub quantity: &'static str,
    /// What the port produced.
    pub measured: f64,
    /// What the reference produced.
    pub reference: f64,
    /// The distance between them, in [`unit`](Check::unit).
    pub deviation: f64,
    /// The largest distance D §10.2 allows; zero means exact.
    pub tolerance: f64,
    /// What `deviation` and `tolerance` are measured in.
    pub unit: Unit,
}

impl Check {
    /// A relative comparison: `|a − b| / |b|` against `tolerance`.
    pub fn relative(
        scope: impl Into<String>,
        quantity: &'static str,
        measured: f64,
        reference: f64,
        tolerance: f64,
    ) -> Self {
        let deviation = if reference == 0.0 {
            (measured - reference).abs()
        } else {
            (measured - reference).abs() / reference.abs()
        };
        Self {
            scope: scope.into(),
            quantity,
            measured,
            reference,
            deviation,
            tolerance,
            unit: Unit::Relative,
        }
    }

    /// An absolute comparison in the quantity's own units.
    pub fn absolute(
        scope: impl Into<String>,
        quantity: &'static str,
        measured: f64,
        reference: f64,
        tolerance: f64,
    ) -> Self {
        Self {
            scope: scope.into(),
            quantity,
            measured,
            reference,
            deviation: (measured - reference).abs(),
            tolerance,
            unit: Unit::Absolute,
        }
    }

    /// An exact comparison of two numbers — every bit, no tolerance.
    pub fn exact(
        scope: impl Into<String>,
        quantity: &'static str,
        measured: f64,
        reference: f64,
    ) -> Self {
        Self {
            scope: scope.into(),
            quantity,
            measured,
            reference,
            deviation: if measured.to_bits() == reference.to_bits() {
                0.0
            } else {
                (measured - reference).abs().max(f64::MIN_POSITIVE)
            },
            tolerance: 0.0,
            unit: Unit::Absolute,
        }
    }

    /// Two values that must simply be equal (a flag, a count).
    pub fn identical<T: PartialEq + Into<f64>>(
        scope: impl Into<String>,
        quantity: &'static str,
        measured: T,
        reference: T,
    ) -> Self {
        let same = measured == reference;
        Self {
            scope: scope.into(),
            quantity,
            measured: measured.into(),
            reference: reference.into(),
            deviation: if same { 0.0 } else { 1.0 },
            tolerance: 0.0,
            unit: Unit::Identical,
        }
    }

    /// Two counts that must be equal — faces, vertices, a face budget.
    ///
    /// The widening to `f64` happens here and nowhere else: every count the fixtures carry is far
    /// below 2^53, and doing it once keeps the stage runners free of casts.
    #[allow(clippy::cast_precision_loss, reason = "counts are exact in f64 well past 2^53")]
    pub fn count(
        scope: impl Into<String>,
        quantity: &'static str,
        measured: u64,
        reference: u64,
    ) -> Self {
        Self::identical(scope, quantity, measured as f64, reference as f64)
    }

    /// A flag that must match.
    pub fn flag(
        scope: impl Into<String>,
        quantity: &'static str,
        measured: bool,
        reference: bool,
    ) -> Self {
        Self::identical(
            scope,
            quantity,
            f64::from(u8::from(measured)),
            f64::from(u8::from(reference)),
        )
    }

    /// A count of entries that differ out of `total`, which must be zero.
    pub fn entries(
        scope: impl Into<String>,
        quantity: &'static str,
        differing: usize,
        total: usize,
    ) -> Self {
        #[allow(clippy::cast_precision_loss, reason = "counts are exact well past 2^53")]
        Self {
            scope: scope.into(),
            quantity,
            measured: differing as f64,
            reference: total as f64,
            deviation: differing as f64,
            tolerance: 0.0,
            unit: Unit::Entries,
        }
    }

    /// True when the check is inside its tolerance.
    pub fn passed(&self) -> bool {
        self.deviation <= self.tolerance
    }

    /// `deviation / tolerance`, the number the table sorts and reports on. Infinite when an exact
    /// comparison failed.
    pub fn ratio(&self) -> f64 {
        if self.tolerance == 0.0 {
            if self.deviation == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            self.deviation / self.tolerance
        }
    }

    /// The row a detail listing prints.
    pub fn line(&self) -> String {
        let status = if self.passed() { "ok" } else { "FAIL" };
        let tol = if self.tolerance == 0.0 {
            "exact".to_owned()
        } else {
            format!("{:.3e} {}", self.tolerance, self.unit.suffix())
        };
        format!(
            "{:<28} {:<14} {:>16.9} {:>16.9} {:>11.3e} {:>14} {status}",
            self.scope, self.quantity, self.measured, self.reference, self.deviation, tol
        )
    }
}

/// Why a fragment was not compared at all.
#[derive(Clone, Debug)]
pub struct Skip {
    /// Fragment (or pair) that was skipped.
    pub scope: String,
    /// What was missing.
    pub reason: String,
}

/// Every check of one stage in one mode.
#[derive(Clone, Debug)]
pub struct StageReport {
    /// The stage's name, as D §10.2's table names it.
    pub stage: &'static str,
    /// Which column of that table was applied.
    pub mode: Mode,
    /// Every comparison made.
    pub checks: Vec<Check>,
    /// Fragments the stage could not compare, with the reason.
    pub skips: Vec<Skip>,
}

impl StageReport {
    /// An empty report for a stage.
    pub fn new(stage: &'static str, mode: Mode) -> Self {
        Self { stage, mode, checks: Vec::new(), skips: Vec::new() }
    }

    /// Adds a check.
    pub fn push(&mut self, check: Check) {
        self.checks.push(check);
    }

    /// Records that a fragment could not be compared.
    pub fn skip(&mut self, scope: impl Into<String>, reason: impl Into<String>) {
        self.skips.push(Skip { scope: scope.into(), reason: reason.into() });
    }

    /// Checks that failed.
    pub fn failures(&self) -> impl Iterator<Item = &Check> {
        self.checks.iter().filter(|c| !c.passed())
    }

    /// How many checks failed.
    pub fn n_failed(&self) -> usize {
        self.failures().count()
    }

    /// The check that came closest to its tolerance (or furthest past it).
    pub fn worst(&self) -> Option<&Check> {
        self.checks
            .iter()
            .max_by(|a, b| a.ratio().partial_cmp(&b.ratio()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// True when nothing failed. A stage with no checks at all has not passed anything and says
    /// so through [`StageReport::status`]; it does not fail the run.
    pub fn passed(&self) -> bool {
        self.n_failed() == 0
    }

    /// `PASS`, `FAIL` or `SKIP`.
    pub fn status(&self) -> &'static str {
        if !self.passed() {
            "FAIL"
        } else if self.checks.is_empty() {
            "SKIP"
        } else {
            "PASS"
        }
    }

    /// The summary row of the table.
    pub fn summary_line(&self) -> String {
        let worst = self.worst().map_or_else(
            || "-".to_owned(),
            |c| {
                let r = c.ratio();
                if !r.is_finite() {
                    "inf".to_owned()
                } else if r == 0.0 || r >= 0.01 {
                    format!("{r:.2}")
                } else {
                    // Headroom of several orders of magnitude is the interesting case in the
                    // injected column; `0.00` would hide the difference between exact and nearly.
                    format!("{r:.1e}")
                }
            },
        );
        format!(
            "{:<14} {:<9} {:>7} {:>7} {:>8} {:>7} {:>6}",
            self.stage,
            self.mode.as_str(),
            self.checks.len(),
            self.n_failed(),
            worst,
            self.skips.len(),
            self.status()
        )
    }

    /// The header the summary rows sit under.
    pub fn summary_header() -> String {
        format!(
            "{:<14} {:<9} {:>7} {:>7} {:>8} {:>7} {:>6}",
            "stage", "mode", "checks", "failed", "worst/tol", "skipped", "status"
        )
    }

    /// The header a detail listing sits under.
    pub fn detail_header() -> String {
        format!(
            "{:<28} {:<14} {:>16} {:>16} {:>11} {:>14} {}",
            "fragment", "quantity", "rust", "reference", "deviation", "tolerance", "status"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Check, Mode, StageReport, Unit};

    #[test]
    fn a_relative_check_measures_the_fraction_of_the_reference() {
        let c = Check::relative("frag", "res", 1.5, 1.0, 0.5);
        assert!(
            (c.deviation - 0.5).abs() < 1e-15 && c.passed(),
            "a deviation equal to the tolerance passes"
        );
        assert!((c.ratio() - 1.0).abs() < 1e-12);
        let c = Check::relative("frag", "res", 1.2, 1.0, 0.10);
        assert!(!c.passed());
        assert!(c.line().contains("FAIL"));
        // A zero reference falls back to the absolute difference rather than dividing by zero.
        let c = Check::relative("frag", "res", 0.5, 0.0, 1.0);
        assert!(c.deviation.is_finite() && c.passed());
    }

    #[test]
    fn an_exact_check_has_no_headroom() {
        let c = Check::exact("frag", "area", 2.0, 2.0);
        assert!(c.passed() && c.ratio() < 1e-15);
        let c = Check::exact("frag", "area", 2.0, 2.000_000_000_000_001);
        assert!(!c.passed(), "one ULP is a failure when the comparison is exact");
        assert!(c.ratio().is_infinite());
        assert!(c.line().contains("exact"));

        // A difference too small to show up in the subtraction still counts as a difference.
        let c = Check::exact("frag", "t", f64::from_bits(1), 0.0);
        assert!(!c.passed());
    }

    #[test]
    fn identical_and_entries_are_pass_or_fail() {
        let c = Check::identical("frag", "watertight", 1.0, 1.0);
        assert!(c.passed());
        assert_eq!(c.unit, Unit::Identical);
        let c = Check::identical("frag", "watertight", 1.0, 0.0);
        assert!(!c.passed());
        let c = Check::entries("frag", "V0", 0, 1000);
        assert!(c.passed());
        let c = Check::entries("frag", "V0", 3, 1000);
        assert!(!c.passed() && (c.deviation - 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_report_finds_its_worst_check_and_its_status() {
        let mut r = StageReport::new("working mesh", Mode::Native);
        assert_eq!(r.status(), "SKIP", "a stage with nothing to compare is not a pass");
        r.push(Check::relative("a", "res", 1.02, 1.0, 0.10));
        r.push(Check::relative("b", "res", 1.09, 1.0, 0.10));
        assert_eq!(r.status(), "PASS");
        assert_eq!(r.worst().unwrap().scope, "b");
        assert_eq!(r.n_failed(), 0);
        r.push(Check::relative("c", "res", 1.5, 1.0, 0.10));
        assert_eq!(r.status(), "FAIL");
        assert_eq!(r.n_failed(), 1);
        assert_eq!(r.failures().next().unwrap().scope, "c");
        r.skip("d", "no source file");
        assert_eq!(r.skips.len(), 1);
        assert!(r.summary_line().contains("FAIL"));
        assert!(StageReport::summary_header().contains("worst/tol"));
        assert!(StageReport::detail_header().contains("reference"));
        assert_eq!(Mode::Injected.to_string(), "injected");
    }
}
