//! Roadmap item 4, step 10 (audit §D.2, D §12 row 10): which vessel a group of sherds is, and
//! what the collection itself says about a join once the groups exist.
//!
//! A mixed collection is the museum's real case and the port's own baseline says what it costs:
//! `mixed_ABG` under R §6.5 placed **three cross-object joins** at group purity 0.864
//! (`notes/2026-09-08-z-phase1e-findings.md` §6). Audit §D.2 answers that with four things — a
//! per-fragment feature table, a per-group consensus over it, cycle consistency read as *evidence*
//! rather than only as a filter, and group merging through a confirmed join — and with one rule
//! above all of them: **a feature may veto only where its measured AUC exceeds 0.8, otherwise it
//! reports.**
//!
//! # What the measurement decided, before a line of this module was written
//!
//! Task M1 (`notes/2026-09-09-m1-measure.md` §4) computed every feature of
//! [`Features`](crate::fragment::features::Features) on every collection that carries real object
//! ids — `mixed_ABG` (the development test), `mixed_all` (the acceptance set) and the pooled
//! `pots_A_H` — and the best AUC of any of them is **0.740** (`shell_radius` on `mixed_ABG`), then
//! 0.695, 0.694 and 0.663. Under audit §D.2's own 0.800 rule **no feature earns a veto**, so
//! [`ObjectParams::demote`] ships **empty**: the consensus is computed, printed per object and
//! drawn in the review image, and it demotes nothing until a collection arrives on which some
//! feature separates. The mechanism is here and it is off, which is not the same thing as absent —
//! `--object-demote shell_radius` is one flag away, and the note that turns it on will have to
//! carry the table that justifies it.
//!
//! # What task S2 changed, and what it did not
//!
//! The collection M1 was waiting for arrived: `synthetic_mix3_24` carries three photographed
//! vessels with object ids, and on it the **clay body** — the mean Lab of the faces R §3.4 labels
//! fracture ([`FeatureKey::FracLabA`]) — separates same-object from different-object pairs with
//! **AUC 0.985** (0.990 on `synthetic_mix3_60`), the first feature of this project above audit
//! §D.2's own bar. Two things follow, and they are not the same thing.
//!
//! * **The colours join the consensus.** [`FeatureKey::ALL`] gained the six split-colour channels
//!   and [`FeatureKey::REPORTED`] the three of the clay body, so `## Objects` now prints what
//!   colour each assembled vessel is and which of its members do not match it. A collection with
//!   no colour prints no such row.
//! * **They still do not veto.** `--object-demote frac_lab_a` was measured on the tree that added
//!   it (`notes/2026-09-11-s2-colour.md` §4): on `synthetic_mix3_24` seed 0 it demotes **four
//!   correct joins**, 11 confirmed down to 7, and removes **no** false one — because a group of
//!   one vessel agrees on its clay body to a MAD of 0.025 Lab units and a `k·MAD` rule divides by
//!   that. [`COLOUR_MAD_FLOOR`] is the fix, measured on the three single-object controls, and with
//!   it the same flag costs nothing; but a rule that costs nothing and removes nothing is not a
//!   rule a default should carry, so [`ObjectParams::demote`] ships empty for the second time and
//!   for a different reason. The rule that uses colour is task S3's, chosen on the whole evidence
//!   table, and [`Evidence::colour`](crate::tiers::Evidence::colour) is the column it will read.
//!
//! A `k·MAD` rule is also the wrong shape for these quantities and M1 measured that too: on
//! `mixed_ABG` a 2·MAD gate on `thick` removes 48 % of the pairs and **43 % of the adjacent ones**,
//! because that set's three pots are the twins `notes/2026-09-06-scale-pairs.md` §4.3 already
//! named. MAD is a poor scale for a feature whose within-object spread is a fifth of its
//! between-object spread. That is why the demotion, when it is switched on, is a *demotion*:
//! audit §D.2's own wording is "demoted to probable, **not rejected** — the conservator decides,
//! with the numbers in the review image".
//!
//! # The three things that do act
//!
//! | | what it does | why it is defensible on this evidence |
//! |---|---|---|
//! | **the consensus** | reports median and MAD per feature per object, and names the members it would reject | no threshold: a number a conservator reads |
//! | **mutual disagreement** | two joins into one group that disagree about where a fragment goes, **both confirmed**, are both demoted | a contradiction between two certainties is evidence against both, and it needs no threshold beyond R §8's own tolerances |
//! | **group merging** | a **confirmed** join between two groups merges them, under the penetration test across both and consistency with every cross-group join the gate admits | R §8 refuses such a join outright today (`Rejection::MergesGroups`); the merge is the same two tests R §8 already applies to a placement, applied to a group |
//!
//! # The off switch
//!
//! [`Params::objects`](crate::params::Params::objects) is `None` by default, and with it `None`
//! nothing here runs: no consensus, no demotion, no merge, no `## Objects` section, and every
//! output is the byte it was before this module existed. `--objects off` is the flag.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::assembly::consistency::{agrees, disagreement};
use crate::assembly::constraints::{Resolved, key as pair_key};
use crate::assembly::greedy::{Assembly, rel};
use crate::assembly::groups::Piece;
use crate::fragment::Fragment;
use crate::fragment::features::Features;
use crate::matching::pair::Candidate;
use crate::matching::verify::pose_inverse;
use crate::mesh::geometry::median;
use crate::tiers::Tier;
use crate::types::FragId;

/// Fewest Lab units a colour consensus is allowed to call a spread ([`FeatureKey::mad_floor`]).
///
/// **3.0, on the controls and not on a preference.** The number has to clear the widest spread a
/// collection that is *one vessel* shows, because everything above it is read as a second vessel.
/// Task S2 §3 measured that spread on the three single-object controls: the four real museum scans
/// of `input/test_fragments_1` agree on their clay body's `a` to **0.39** Lab units,
/// `synthetic_pingsdorf_20` to **3.32** and `synthetic_pingsdorf_60` to **6.61** — the last two
/// being the generator's own per-fragment tint, which a real vessel does not have. With `k_mad` at
/// its default 3 the effective limit is `3 x 3 = 9` Lab units, above every one of them; below it
/// no monochrome collection is touched, and `V049`'s clay body is 11.4 units from `V012`'s.
pub const COLOUR_MAD_FLOOR: f64 = 3.0;

/// Fewest members the consensus a fragment is measured **against** must have.
///
/// A fragment is compared with its object *without itself* (see [`objects`]), so a group of `n`
/// members offers a consensus of `n − 1` and this constant is a rule about a group of four.
///
/// Two is not enough, and the terracotta measures why: the MAD of a pair is half their difference,
/// so a third fragment 1.2 units of wall away from two that differ by 0.2 reads as **11.7 MAD**
/// and every collection of three would report an outlier. Three is the smallest set whose median
/// absolute deviation is a spread rather than a subtraction.
pub const MIN_CONSENSUS_MEMBERS: usize = 3;

/// One feature of [`Features`] a group consensus can be taken over.
///
/// The list is M1 §4's own columns minus the two residuals (`shell_rms`, `axis_rms`), which
/// describe how well a fit worked rather than what it found, and minus `colour_points` and
/// `colour_distinct`, which are counts of the file. `rim_diameter` is in it because audit §D.2
/// asks for it by name, although M1 measured R §3.2's rim flag firing on **2 of 517** fragments
/// across the whole benchmark, so it is almost always absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKey {
    /// R §3.2's wall thickness.
    Thick,
    /// R §3.2's plain ray mode.
    ThickMode,
    /// Radius of the sphere fitted to the outer shell samples.
    ShellRadius,
    /// RMS of the fracture samples to their local planes, in `t`.
    FracRough,
    /// Diameter of the circle the shell samples make about the fitted axis.
    AxisDiameter,
    /// The same, on a fragment R §3.2 flagged as a rim or a collar.
    RimDiameter,
    /// CIE Lab lightness of the source file's vertex colours.
    LabL,
    /// The green–red opponent axis.
    LabA,
    /// The blue–yellow opponent axis.
    LabB,
    /// Lightness of R §3.4's **shell** faces alone (task S2).
    ShellLabL,
    /// The green–red axis of the same.
    ShellLabA,
    /// The blue–yellow axis of the same.
    ShellLabB,
    /// Lightness of R §3.4's **fracture** faces alone — the clay body.
    FracLabL,
    /// The green–red axis of the same.
    FracLabA,
    /// The blue–yellow axis of the same.
    FracLabB,
}

impl FeatureKey {
    /// Every feature a consensus is taken over, in the order an object's row prints them.
    pub const ALL: [Self; 15] = [
        Self::Thick,
        Self::ThickMode,
        Self::ShellRadius,
        Self::FracRough,
        Self::AxisDiameter,
        Self::RimDiameter,
        Self::LabL,
        Self::LabA,
        Self::LabB,
        Self::ShellLabL,
        Self::ShellLabA,
        Self::ShellLabB,
        Self::FracLabL,
        Self::FracLabA,
        Self::FracLabB,
    ];

    /// The features an object's row reports a member's deviation on whether or not anything may
    /// demote it.
    ///
    /// M1 §5.7's own list — `shell_radius`, the best AUC on the development test (0.740), and
    /// `thick`, the best on the acceptance set (0.663) — plus the two audit §D.2 names in the same
    /// sentence as those: the fracture roughness ("temper and fabric") and the rim diameter. Task
    /// S2 added the **clay body**: the mean Lab of the faces R §3.4 labels fracture, which is the
    /// one colour statement a break makes and the only object feature this project has measured
    /// above audit §D.2's own AUC of 0.800 (0.985 on `synthetic_mix3_24`, 0.990 on
    /// `synthetic_mix3_60` for `frac_lab_a`). It **reports**: what it would cost to let it veto is
    /// in `notes/2026-09-11-s2-colour.md` §4, and the answer is four correct joins for no false
    /// one.
    ///
    /// A collection whose files carry no colour has no consensus on these three and prints no row
    /// for them, which is what keeps every SfS++ collection's `## Objects` section the bytes it
    /// was.
    pub const REPORTED: [Self; 7] = [
        Self::Thick,
        Self::ShellRadius,
        Self::FracRough,
        Self::RimDiameter,
        Self::FracLabL,
        Self::FracLabA,
        Self::FracLabB,
    ];

    /// The name the CLI, `report.md` and `report.json` spell this feature with.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Thick => "thick",
            Self::ThickMode => "thick_mode",
            Self::ShellRadius => "shell_radius",
            Self::FracRough => "frac_rough",
            Self::AxisDiameter => "axis_diameter",
            Self::RimDiameter => "rim_diameter",
            Self::LabL => "lab_L",
            Self::LabA => "lab_a",
            Self::LabB => "lab_b",
            Self::ShellLabL => "shell_lab_L",
            Self::ShellLabA => "shell_lab_a",
            Self::ShellLabB => "shell_lab_b",
            Self::FracLabL => "frac_lab_L",
            Self::FracLabA => "frac_lab_a",
            Self::FracLabB => "frac_lab_b",
        }
    }

    /// The smallest spread a consensus over this feature may be measured against.
    ///
    /// Zero for every geometric feature, which leaves [`Consensus::mads`] exactly the test it was:
    /// a wall thickness is in the collection's own units and nothing outside the collection knows
    /// how large one is, so its only scale is the one its members give it.
    ///
    /// [`COLOUR_MAD_FLOOR`] for every Lab channel, and that is a measurement rather than a taste.
    /// A `k·MAD` rule divides by the spread, so a group whose members agree closely has *no*
    /// scale: task S2 measured `V012`'s eight sherds agreeing on their clay body to a MAD of
    /// **0.025** Lab units, which turned a difference of 0.29 — invisible to an eye, a quarter of
    /// one just-noticeable difference — into **11.8 MAD** and demoted four correct joins on
    /// `synthetic_mix3_24` seed 0 while removing no false one. A Lab channel, unlike a wall, comes
    /// with an absolute scale: one unit is about the smallest difference a person sees. Below the
    /// floor the group is one colour, and the eye's scale is used instead of a spread that is not
    /// there.
    #[must_use]
    pub const fn mad_floor(self) -> f64 {
        match self {
            Self::LabL
            | Self::LabA
            | Self::LabB
            | Self::ShellLabL
            | Self::ShellLabA
            | Self::ShellLabB
            | Self::FracLabL
            | Self::FracLabA
            | Self::FracLabB => COLOUR_MAD_FLOOR,
            _ => 0.0,
        }
    }

    /// The feature by name, for the CLI's `--object-demote` list.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.key().eq_ignore_ascii_case(name))
    }

    /// This feature of one fragment, or `None` where the fragment does not support the fit.
    #[must_use]
    pub fn of(self, f: &Features) -> Option<f64> {
        match self {
            Self::Thick => Some(f.thick),
            Self::ThickMode => Some(f.thick_mode),
            Self::ShellRadius => f.shell_radius,
            Self::FracRough => f.frac_rough,
            Self::AxisDiameter => f.axis_diameter,
            Self::RimDiameter => f.rim_diameter,
            Self::LabL => f.lab_mean.map(|l| l[0]),
            Self::LabA => f.lab_mean.map(|l| l[1]),
            Self::LabB => f.lab_mean.map(|l| l[2]),
            Self::ShellLabL => f.shell_colour.as_ref().map(|c| c.lab_mean[0]),
            Self::ShellLabA => f.shell_colour.as_ref().map(|c| c.lab_mean[1]),
            Self::ShellLabB => f.shell_colour.as_ref().map(|c| c.lab_mean[2]),
            Self::FracLabL => f.frac_colour.as_ref().map(|c| c.lab_mean[0]),
            Self::FracLabA => f.frac_colour.as_ref().map(|c| c.lab_mean[1]),
            Self::FracLabB => f.frac_colour.as_ref().map(|c| c.lab_mean[2]),
        }
    }
}

/// A set of [`FeatureKey`]s, as one word.
///
/// A bitmask and not a `Vec` because [`Params`](crate::params::Params) is `Copy` and every stage
/// of the pipeline passes it by value; a set of fifteen flags does not need an allocation to make
/// that stop being true. It serialises as the list of names `report.json` prints and
/// `--object-demote` accepts, so the wire form is the readable one either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureSet(u16);

impl FeatureSet {
    /// The empty set — M1 §4's shortlist.
    #[must_use]
    pub const fn none() -> Self {
        Self(0)
    }

    /// True when nothing is in it, and therefore when nothing can demote.
    ///
    /// Takes `&self` so that `serde`'s `skip_serializing_if` can call it.
    #[must_use]
    #[allow(clippy::trivially_copy_pass_by_ref, reason = "serde's skip_serializing_if")]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Whether this feature is in the set.
    #[must_use]
    pub fn contains(self, key: FeatureKey) -> bool {
        self.0 & Self::bit(key) != 0
    }

    /// The set with this feature added.
    #[must_use]
    pub fn with(mut self, key: FeatureKey) -> Self {
        self.0 |= Self::bit(key);
        self
    }

    /// Its members, in [`FeatureKey::ALL`]'s order.
    pub fn iter(self) -> impl Iterator<Item = FeatureKey> {
        FeatureKey::ALL.into_iter().filter(move |&k| self.contains(k))
    }

    fn bit(key: FeatureKey) -> u16 {
        let at = FeatureKey::ALL.iter().position(|&k| k == key).expect("every key is in ALL");
        1_u16 << at
    }
}

impl FromIterator<FeatureKey> for FeatureSet {
    fn from_iter<I: IntoIterator<Item = FeatureKey>>(keys: I) -> Self {
        keys.into_iter().fold(Self::none(), Self::with)
    }
}

impl Serialize for FeatureSet {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.iter())
    }
}

impl<'de> Deserialize<'de> for FeatureSet {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Vec::<FeatureKey>::deserialize(d)?.into_iter().collect())
    }
}

/// What roadmap item 4 is allowed to do, and how much of it (audit §D.2).
///
/// `None` on [`Params::objects`](crate::params::Params::objects) is the off switch; this struct is
/// the on position, and every field is a flag on `run`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectParams {
    /// The features whose `k·MAD` deviation demotes a join to [`Tier::Probable`].
    ///
    /// **Empty by default, on the measurement and not on a preference**: M1 §4 found the best AUC
    /// of any feature on any collection with real object ids to be 0.740, and audit §D.2's own
    /// rule is 0.800. Every feature is still computed, reported per object and named in the review
    /// image; none of them demotes.
    #[serde(default, skip_serializing_if = "FeatureSet::is_empty")]
    pub demote: FeatureSet,
    /// How many MADs from its object's median a fragment may sit before the consensus rejects it.
    pub k_mad: f64,
    /// Fewest members the consensus a fragment is compared with must have
    /// ([`MIN_CONSENSUS_MEMBERS`]); an object of `n` fragments offers `n − 1`, so this is a rule
    /// about a group of `min_members + 1`.
    pub min_members: usize,
    /// Audit §D.2 (b): two joins into one group that disagree about a fragment, both confirmed,
    /// are both demoted.
    ///
    /// **Off by default, on the measurement** — see [`ObjectParams::default`].
    pub disagreement: bool,
    /// Audit §D.2 (c): a confirmed join between two groups may merge them, replacing
    /// [`Rejection::MergesGroups`](crate::assembly::greedy::Rejection::MergesGroups).
    pub merge: bool,
}

impl Default for ObjectParams {
    /// What roadmap item 4 does on the shipped settings, and why each of the three arms is where
    /// it is.
    ///
    /// * **`demote` is empty** — M1 §4: no feature reaches audit §D.2's own AUC of 0.800 on any
    ///   collection with real object ids.
    /// * **`disagreement` is off** — on the measurement, and this is where that measurement lives;
    ///   everything else that quotes it cites this doc. The arm is implemented exactly as audit
    ///   §D.2 (b) words it and it is a **pure recall loss on this benchmark**: the confirmed tier
    ///   has **0 false joins** over the eight development sets at seeds 0–4 (task T1, and still 0
    ///   today), so there is no false join for a contradiction to catch. What it costs, measured
    ///   on this tree by task C (`quality_gate.py --sets mixed_ABG --object-disagreement on`,
    ///   commit `cce1d66`): `mixed_ABG` goes from **8 / 7 / 7 / 7 / 9** confirmed joins at seeds
    ///   0–4 to **2 / 3 / 2 / 4 / 0** — **27 of 38 demoted, every one of them correct, and no
    ///   false join removed**. Every contradiction it finds is between joins R §8 has already
    ///   reconciled — R §8 refuses the odd one out itself, with `InconsistentWithAssembled`.
    ///   `--object-disagreement on` is the flag, and the day a false join survives the tier is the
    ///   day to measure it again. (Task O1 measured 11 → 2 at seed 0 on its own tree; the
    ///   denominator moved when task G stopped confirming a join whose penetration cannot be
    ///   measured, so the figure is re-measured here rather than carried.)
    /// * **`merge` is on** — it needs no threshold, it can only *add* a join R §8 refused for a
    ///   reason that was never about the geometry, and every one it adds has passed the same two
    ///   tests R §8 applies to a placement.
    fn default() -> Self {
        Self {
            demote: FeatureSet::none(),
            // Inert while `demote` is empty. 3 rather than 2 because M1 measured what the two cost
            // on `mixed_ABG`: at 2·MAD the best feature removes 43 % of the adjacent pairs, at
            // 3·MAD 37 % — both far too much, and the more conservative of two bad numbers is the
            // one a default should carry.
            k_mad: 3.0,
            min_members: MIN_CONSENSUS_MEMBERS,
            disagreement: false,
            merge: true,
        }
    }
}

/// One feature's consensus over the members of one object: audit §D.2's "median and MAD per
/// feature over members".
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Consensus {
    /// Which feature.
    pub feature: FeatureKey,
    /// The median over the members that have it.
    pub median: f64,
    /// The median absolute deviation from that median.
    pub mad: f64,
    /// How many members contributed — a feature no member supports has no row at all.
    pub n: usize,
}

impl Consensus {
    /// How far one value sits from this consensus, in MADs.
    ///
    /// `None` when the consensus has no scale to measure with — a MAD of zero, which is what a
    /// group of identical values gives and what M1 §4 measured for `lab_a` and `lab_b` on every
    /// SfS++ collection (one flat grey per file). A zero MAD would make every deviation infinite
    /// and every member an outlier, so the honest answer is that this feature cannot say.
    ///
    /// The scale is the MAD or [`FeatureKey::mad_floor`], whichever is larger. That floor is zero
    /// for every geometric feature — this is the test it always was for them — and
    /// [`COLOUR_MAD_FLOOR`] for a Lab channel, where a group can agree so closely that its spread
    /// stops being a scale at all.
    #[must_use]
    pub fn mads(&self, value: f64) -> Option<f64> {
        let scale = self.mad.max(self.feature.mad_floor());
        (scale > 0.0 && scale.is_finite()).then(|| (value - self.median).abs() / scale)
    }
}

/// Every feature's consensus over one set of fragments.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroupFeatures {
    /// How many fragments the consensus was taken over.
    pub members: usize,
    /// One row per feature at least one member supports, in [`FeatureKey::ALL`]'s order.
    pub consensus: Vec<Consensus>,
}

impl GroupFeatures {
    /// The median and MAD of every feature over these fragments.
    ///
    /// The median is [`crate::mesh::geometry::median`] — the codebase's own, numpy's convention,
    /// the mean of the two middle values on an even count — so a consensus and a report agree
    /// about what "median" means. A fragment with no table contributes to nothing.
    #[must_use]
    pub fn of<'a>(members: impl IntoIterator<Item = &'a Features>) -> Self {
        let rows: Vec<&Features> = members.into_iter().collect();
        let mut consensus = Vec::new();
        for feature in FeatureKey::ALL {
            let values: Vec<f64> =
                rows.iter().filter_map(|f| feature.of(f)).filter(|v| v.is_finite()).collect();
            if values.is_empty() {
                continue;
            }
            let m = median(&values);
            let spread: Vec<f64> = values.iter().map(|v| (v - m).abs()).collect();
            consensus.push(Consensus { feature, median: m, mad: median(&spread), n: values.len() });
        }
        Self { members: rows.len(), consensus }
    }

    /// This object's consensus for one feature, if it has one.
    #[must_use]
    pub fn get(&self, feature: FeatureKey) -> Option<&Consensus> {
        self.consensus.iter().find(|c| c.feature == feature)
    }
}

/// How far one fragment sits from the consensus of the object it is in, on one feature.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Deviation {
    /// Which feature.
    pub feature: FeatureKey,
    /// The fragment's own value.
    pub value: f64,
    /// The object's median **without this fragment**.
    pub median: f64,
    /// How many MADs away that is.
    pub mads: f64,
    /// Whether this feature is on [`ObjectParams::demote`] and therefore acted.
    pub demoting: bool,
}

/// A member its own object's consensus does not fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reject {
    /// The fragment.
    pub fragment: String,
    /// Every feature it exceeds `k·MAD` on, worst first.
    pub deviations: Vec<Deviation>,
}

/// One assembled group, read as an object (audit §D.2: *"every group is reported as an object with
/// its consensus, its rim diameter if any, and the fragments the consensus rejects"*).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Object {
    /// The group's index in the assembly's own order.
    pub group: usize,
    /// Its members, in placement order.
    pub fragments: Vec<String>,
    /// The joins R §8 used to build it.
    pub joins: usize,
    /// Median and MAD per feature over the members ([`GroupFeatures`]).
    pub consensus: Vec<Consensus>,
    /// The consensus rim diameter, on an object at least one of whose members R §3.2 flagged as a
    /// rim or a collar.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rim_diameter: Option<f64>,
    /// The members the consensus rejects, whether or not the feature that rejects them is allowed
    /// to demote anything.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejects: Vec<Reject>,
}

/// Which of audit §D.2's two arms demoted a join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    /// The per-group consensus: the fragment is more than `k·MAD` from its object on a shortlisted
    /// feature. Never fires on the shipped defaults, whose shortlist is empty.
    Consensus,
    /// Audit §D.2 (b): two confirmed joins into one group that disagree about a fragment.
    Disagreement,
}

impl Arm {
    /// The word `report.md` prints.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Consensus => "consensus",
            Self::Disagreement => "disagreement",
        }
    }
}

/// One join the object pass moved from [`Tier::Confirmed`] to [`Tier::Probable`], and why.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Demotion {
    /// The fragment the pose maps into.
    pub a: String,
    /// The fragment the pose moves.
    pub b: String,
    /// Which arm demoted it.
    pub arm: Arm,
    /// The sentence the report prints and the evidence carries.
    pub reason: String,
}

/// What one object pass decided: the objects, the demotions, and what the two operator lists said.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObjectReport {
    /// One entry per group of the final assembly.
    pub objects: Vec<Object>,
    /// Every join the pass demoted, in the order it demoted them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub demotions: Vec<Demotion>,
    /// `same_object` pairs the assembly put in **two** objects — the operator's word and the
    /// geometry's answer disagreeing, which is a thing a conservator has to see.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub same_object_split: Vec<[String; 2]>,
    /// `different_object` pairs that ended up in **one** object. The veto and the merge test both
    /// refuse this, so the list is expected to be empty and exists to prove it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub different_object_together: Vec<[String; 2]>,
    /// How many groups merged through a confirmed join during the assembly (audit §D.2 (c)).
    pub merges: usize,
}

/// Audit §D.2's demotions over one finished assembly: the consensus arm and the disagreement arm.
///
/// Returns one entry per **pair**, keyed the way
/// [`constraints::key`](crate::assembly::constraints::key) keys a pair, with the sentence the
/// demotion carries. [`crate::pipeline::run`] applies each to *every* candidate of that pair and
/// then runs R §8 **once** more — one round, not to a fixed point, because a second round's
/// demotions would be computed on the groups the first round's demotions built and the answer
/// would then depend on how many rounds were run.
///
/// **Per pair and not per candidate**, which is the one place this departs from how task T1 reads
/// a band. A pair routinely has several candidates converging on one placement and all of them
/// confirmed (44 candidates for 11 pairs on `mixed_ABG` seed 0, measured); demoting only the
/// representative would leave R §8's `best_per_pair` free to pick the next one and place the join
/// anyway. What the arm says is about the two sherds, so it is said about the pair.
///
/// A join is only ever moved from [`Tier::Confirmed`] to [`Tier::Probable`]. Nothing here rejects
/// a candidate, and nothing here edits a score.
#[must_use]
pub fn demotions(
    fragments: &[Fragment],
    names: &[String],
    candidates: &[Candidate],
    assembly: &Assembly,
    pieces: &[Piece<'_>],
    constraints: Option<&Resolved>,
    p: &ObjectParams,
) -> Vec<((FragId, FragId), Demotion)> {
    let mut out: Vec<((FragId, FragId), Demotion)> = Vec::new();
    let name = |n: FragId| names.get(n as usize).cloned().unwrap_or_default();
    let mut group_of: Vec<Option<usize>> = vec![None; fragments.len()];
    for (g, members) in assembly.groups.iter().enumerate() {
        for &n in members {
            group_of[n as usize] = Some(g);
        }
    }

    // --- the consensus arm (audit §D.2's per-group consensus) ----------------------------------
    if !p.demote.is_empty() {
        for (g, members) in assembly.groups.iter().enumerate() {
            if members.len() <= p.min_members.max(2) {
                continue;
            }
            for &x in members {
                let Some(mine) = fragments[x as usize].features.as_ref() else { continue };
                let others = GroupFeatures::of(
                    members
                        .iter()
                        .filter(|&&y| y != x)
                        .filter_map(|&y| fragments[y as usize].features.as_ref()),
                );
                let far = exceeded(&others, mine, p);
                if far.is_empty() {
                    continue;
                }
                for &i in &assembly.used {
                    let c = &candidates[i];
                    if (c.a != x && c.b != x) || c.tier != Tier::Confirmed {
                        continue;
                    }
                    // The operator's word outranks a measured deviation: two fragments they have
                    // declared to be one vessel are not separated by a median.
                    if constraints.is_some_and(|r| r.same.contains(&pair_key(c.a, c.b))) {
                        continue;
                    }
                    let worst = far[0];
                    out.push((
                        pair_key(c.a, c.b),
                        Demotion {
                            a: name(c.a),
                            b: name(c.b),
                            arm: Arm::Consensus,
                            reason: format!(
                                "{} is {:.1} MAD from object {g}'s {} ({:.4} against {:.4})",
                                name(x),
                                worst.mads,
                                worst.feature.key(),
                                worst.value,
                                worst.median
                            ),
                        },
                    ));
                }
            }
        }
    }

    // --- the mutual-disagreement arm (audit §D.2 (b)) ------------------------------------------
    if p.disagreement {
        out.extend(disagreements(names, candidates, assembly, pieces, &group_of));
    }

    out.sort_by_key(|(key, _)| *key);
    out.dedup_by_key(|(key, _)| *key);
    out
}

/// The features a fragment exceeds `k·MAD` on inside its own object, worst first.
fn exceeded(others: &GroupFeatures, mine: &Features, p: &ObjectParams) -> Vec<Deviation> {
    let mut far: Vec<Deviation> = Vec::new();
    for feature in p.demote.iter() {
        let (Some(c), Some(value)) = (others.get(feature), feature.of(mine)) else { continue };
        if c.n < p.min_members.max(2) {
            continue;
        }
        let Some(mads) = c.mads(value) else { continue };
        if mads > p.k_mad {
            far.push(Deviation { feature, value, median: c.median, mads, demoting: true });
        }
    }
    far.sort_by(|x, y| y.mads.total_cmp(&x.mads));
    far
}

/// Audit §D.2 (b): *"when two accepted joins into one group disagree about a fragment, demote both
/// unless one is confirmed"*.
///
/// # How "unless one is confirmed" is read, and why it can only be read that way
///
/// Two joins into one group that contradict each other are evidence that at least one of them is
/// wrong, and nothing in R §6 says which. When exactly one of the two is confirmed, the
/// contradiction is explained by the weaker one and the confirmed one stands; when **both** are
/// confirmed, there is nothing to choose between them and both fall to probable. Read the other
/// way round — "either of them is confirmed" — the rule fires only when both are probable already,
/// and demoting a probable join to probable is nothing at all: the sentence would be inert. So
/// this is the only reading under which audit §D.2 (b) does anything, and it is the safe one.
///
/// The joins compared are the ones the assembly's own gate admitted (`assembly.accepted`, one per
/// pair, strongest first), and the tolerance is R §8's own — 10° and 0.5 `t` of the group's median
/// wall, through [`disagreement`] and [`agrees`] — because a merge or a placement is refused by
/// exactly that test and a demotion must not use a second, different one.
fn disagreements(
    names: &[String],
    candidates: &[Candidate],
    assembly: &Assembly,
    pieces: &[Piece<'_>],
    group_of: &[Option<usize>],
) -> Vec<((FragId, FragId), Demotion)> {
    let name = |n: FragId| names.get(n as usize).cloned().unwrap_or_default();
    let pose = |n: FragId| assembly.poses.get(n as usize).copied();
    let walls: Vec<f64> = assembly
        .groups
        .iter()
        .map(|g| median(&g.iter().map(|&n| pieces[n as usize].thick).collect::<Vec<f64>>()))
        .collect();
    let mut out: Vec<((FragId, FragId), Demotion)> = Vec::new();

    for (u, &i) in assembly.accepted.iter().enumerate() {
        for &j in assembly.accepted.iter().skip(u + 1) {
            let (c1, c2) = (&candidates[i], &candidates[j]);
            // Exactly one shared fragment: two joins that share both are the same pair, and two
            // that share none say nothing about each other.
            let Some(x) = shared(c1, c2) else { continue };
            let (p1, p2) = (other_end(c1, x), other_end(c2, x));
            let (Some(g), Some(gp1), Some(gp2)) =
                (group_of[x as usize], group_of[p1 as usize], group_of[p2 as usize])
            else {
                continue;
            };
            if gp1 != g || gp2 != g {
                continue;
            }
            let (Some(t1), Some(t2)) = (pose(p1), pose(p2)) else { continue };
            let (x1, x2) = (t1 * rel(c1, p1), t2 * rel(c2, p2));
            let (angle, distance) = disagreement(&(pose_inverse(&x1) * x2), walls[g]);
            if agrees(angle, distance) {
                continue;
            }
            if !(c1.tier == Tier::Confirmed && c2.tier == Tier::Confirmed) {
                continue;
            }
            for (k, mine) in [(i, c1), (j, c2)] {
                let theirs = if k == i { c2 } else { c1 };
                out.push((
                    pair_key(mine.a, mine.b),
                    Demotion {
                        a: name(mine.a),
                        b: name(mine.b),
                        arm: Arm::Disagreement,
                        reason: format!(
                            "disagrees with the confirmed join {}-{} about {} ({angle:.1} deg, \
                             {distance:.2} t) and neither is the stronger evidence",
                            name(theirs.a),
                            name(theirs.b),
                            name(x)
                        ),
                    },
                ));
            }
        }
    }
    out
}

/// The one fragment two joins have in common, or `None` when they have none or both.
fn shared(c1: &Candidate, c2: &Candidate) -> Option<FragId> {
    let mine = [c1.a, c1.b];
    let theirs = [c2.a, c2.b];
    let common: Vec<FragId> = mine.into_iter().filter(|n| theirs.contains(n)).collect();
    (common.len() == 1).then(|| common[0])
}

/// The endpoint of a join that is not `x`.
fn other_end(c: &Candidate, x: FragId) -> FragId {
    if c.a == x { c.b } else { c.a }
}

/// Audit §D.2's per-group report: every group as an object, with its consensus and the members
/// that consensus rejects.
///
/// The consensus printed is over the **whole** group; a member's deviation is measured against the
/// group **without it**, because "does this sherd belong to this pot" is a question about the
/// other members and a fragment that pulls the median towards itself has answered it for free.
#[must_use]
pub fn objects(
    fragments: &[Fragment],
    names: &[String],
    candidates: &[Candidate],
    assembly: &Assembly,
    p: &ObjectParams,
) -> Vec<Object> {
    let name = |n: FragId| names.get(n as usize).cloned().unwrap_or_default();
    let mut out = Vec::with_capacity(assembly.groups.len());
    for (g, members) in assembly.groups.iter().enumerate() {
        let all = GroupFeatures::of(
            members.iter().filter_map(|&n| fragments[n as usize].features.as_ref()),
        );
        let rim = members
            .iter()
            .filter_map(|&n| fragments[n as usize].features.as_ref())
            .any(|f| f.rim)
            .then(|| all.get(FeatureKey::RimDiameter).map(|c| c.median))
            .flatten();
        let mut rejects = Vec::new();
        if members.len() > p.min_members.max(2) {
            for &x in members {
                let Some(mine) = fragments[x as usize].features.as_ref() else { continue };
                let others = GroupFeatures::of(
                    members
                        .iter()
                        .filter(|&&y| y != x)
                        .filter_map(|&y| fragments[y as usize].features.as_ref()),
                );
                let deviations = reported_deviations(&others, mine, p);
                if !deviations.is_empty() {
                    rejects.push(Reject { fragment: name(x), deviations });
                }
            }
        }
        let inside: BTreeSet<FragId> = members.iter().copied().collect();
        let joins = assembly
            .used
            .iter()
            .filter(|&&i| inside.contains(&candidates[i].a) && inside.contains(&candidates[i].b))
            .count();
        out.push(Object {
            group: g,
            fragments: members.iter().map(|&n| name(n)).collect(),
            joins,
            consensus: all.consensus,
            rim_diameter: rim,
            rejects,
        });
    }
    out
}

/// Every feature the fragment exceeds `k·MAD` on — the ones that demote **and** the ones that only
/// report, which is the whole of audit §D.2's "otherwise they report".
fn reported_deviations(
    others: &GroupFeatures,
    mine: &Features,
    p: &ObjectParams,
) -> Vec<Deviation> {
    let mut far: Vec<Deviation> = Vec::new();
    for feature in FeatureKey::REPORTED.into_iter().chain(p.demote.iter()) {
        if far.iter().any(|d| d.feature == feature) {
            continue;
        }
        let (Some(c), Some(value)) = (others.get(feature), feature.of(mine)) else { continue };
        if c.n < p.min_members.max(2) {
            continue;
        }
        let Some(mads) = c.mads(value) else { continue };
        if mads > p.k_mad {
            far.push(Deviation {
                feature,
                value,
                median: c.median,
                mads,
                demoting: p.demote.contains(feature),
            });
        }
    }
    far.sort_by(|x, y| y.mads.total_cmp(&x.mads));
    far
}

/// What the two operator lists have to say about the objects the assembly built (audit §D.1's
/// *"item 4's consensus and the group purity reporting read them"*).
#[must_use]
pub fn operator_view(
    names: &[String],
    assembly: &Assembly,
    constraints: Option<&Resolved>,
) -> (Vec<[String; 2]>, Vec<[String; 2]>) {
    let Some(r) = constraints else { return (Vec::new(), Vec::new()) };
    let name = |n: FragId| names.get(n as usize).cloned().unwrap_or_default();
    let mut group_of: BTreeMap<FragId, usize> = BTreeMap::new();
    for (g, members) in assembly.groups.iter().enumerate() {
        for &n in members {
            group_of.insert(n, g);
        }
    }
    let split = r
        .same
        .iter()
        .filter(|(a, b)| group_of.get(a) != group_of.get(b))
        .map(|&(a, b)| [name(a), name(b)])
        .collect();
    let together = r
        .different
        .iter()
        .filter(|(a, b)| group_of.get(a).is_some_and(|g| group_of.get(b) == Some(g)))
        .map(|&(a, b)| [name(a), name(b)])
        .collect();
    (split, together)
}

#[cfg(test)]
mod tests {
    use super::{
        COLOUR_MAD_FLOOR, Consensus, FeatureKey, FeatureSet, GroupFeatures, MIN_CONSENSUS_MEMBERS,
        ObjectParams, exceeded,
    };
    use crate::fragment::features::{ColourStats, Features};
    use approx::assert_relative_eq;

    fn feature(thick: f64, radius: Option<f64>) -> Features {
        Features { thick, thick_mode: thick, shell_radius: radius, ..Features::default() }
    }

    fn body(a: f64) -> Features {
        Features {
            frac_colour: Some(ColourStats {
                lab_mean: [40.0, a, 15.0],
                lab_mad: [1.0, 1.0, 1.0],
                hist: vec![0; 64],
                points: 1_000,
            }),
            ..Features::default()
        }
    }

    /// Task S2: a group that agrees on its colour has no spread to divide by, and the floor is
    /// what stops a quarter of a just-noticeable difference from reading as eleven MADs.
    ///
    /// The four values below are `V012`'s own clay body on `synthetic_mix3_24` rounded to two
    /// places; without the floor the odd one out is 11.8 MAD from the other three and every
    /// confirmed join it touches is demoted, which is what the note measured and what the
    /// shortlist is empty because of.
    #[test]
    fn a_colour_consensus_is_never_measured_against_a_spread_it_does_not_have() {
        let group = GroupFeatures::of([&body(3.62), &body(3.65), &body(3.39)]);
        let c = group.get(FeatureKey::FracLabA).expect("three members carry a clay body");
        assert!(c.mad < 0.1, "the group agrees to a MAD of {:.3} Lab units", c.mad);
        let mads = c.mads(3.33).expect("the floor is a scale");
        assert_relative_eq!(mads, (3.62 - 3.33) / COLOUR_MAD_FLOOR, epsilon = 1e-12);
        assert!(mads < ObjectParams::default().k_mad, "0.29 Lab units refuses nothing");
        // A second vessel's clay body is another matter: `V049` sits at 15.1.
        assert!(
            c.mads(15.13).expect("the floor is a scale") > ObjectParams::default().k_mad,
            "11.5 Lab units is a different pot"
        );
        // Every geometric feature keeps the test it always had: its own spread, or nothing.
        assert_relative_eq!(FeatureKey::Thick.mad_floor(), 0.0);
        let flat = Consensus { feature: FeatureKey::Thick, median: 3.0, mad: 0.0, n: 4 };
        assert!(flat.mads(9.0).is_none(), "a zero MAD on a wall is still a refusal to answer");
    }

    /// M1 §4's verdict is the shipped default, and it is the one thing about this module a later
    /// step must not change without a new table: the shortlist is empty, so nothing demotes.
    #[test]
    fn the_shipped_shortlist_is_empty_because_no_feature_reached_audit_d2s_own_auc() {
        let p = ObjectParams::default();
        assert!(p.demote.is_empty(), "M1 §4: the best AUC on a set with real object ids is 0.740");
        assert!(p.merge, "the one arm that needs no threshold and can only add a join is on");
        assert!(
            !p.disagreement,
            "the arm removes no false join -- there are none -- and costs 27 of mixed_ABG's 38 \
             confirmed joins over seeds 0-4 (task C's re-measurement)"
        );
        assert_eq!(p.min_members, MIN_CONSENSUS_MEMBERS);

        // And with it empty, a fragment ten walls away from its object demotes nothing.
        let others =
            GroupFeatures::of([&feature(3.0, None), &feature(3.2, None), &feature(3.4, None)]);
        assert!(exceeded(&others, &feature(30.0, None), &p).is_empty());
    }

    /// The consensus is a median and a MAD, over the members that have the feature.
    #[test]
    fn a_consensus_is_the_median_and_the_mad_of_the_members_that_have_the_feature() {
        let rows = [
            feature(3.0, Some(100.0)),
            feature(3.4, None),
            feature(3.6, Some(110.0)),
            feature(9.0, Some(104.0)),
        ];
        let g = GroupFeatures::of(&rows);
        assert_eq!(g.members, 4);
        let thick = g.get(FeatureKey::Thick).expect("every fragment has a wall");
        assert_eq!(thick.n, 4);
        // median(3.0, 3.4, 3.6, 9.0) = 3.5; |Δ| = 0.5, 0.1, 0.1, 5.5; median = 0.3.
        assert_relative_eq!(thick.median, 3.5, epsilon = 1e-12);
        assert_relative_eq!(thick.mad, 0.3, epsilon = 1e-12);
        let radius = g.get(FeatureKey::ShellRadius).expect("three of the four fitted");
        assert_eq!(radius.n, 3, "the fragment whose sphere did not fit is not counted");
        assert_relative_eq!(radius.median, 104.0, epsilon = 1e-12);
        assert!(g.get(FeatureKey::LabL).is_none(), "no member carries a colour");
    }

    /// A consensus with no spread cannot say how far anything is from it, and says so rather than
    /// calling every member an outlier — on every feature whose units only the collection knows.
    ///
    /// Task S2 split that sentence in two. A wall thickness with a zero MAD still refuses. A Lab
    /// channel does not have to: its units are the eye's, so [`COLOUR_MAD_FLOOR`] answers instead
    /// — which on M1 §4's flat-grey SfS++ collections is still *no deviation at all*, because
    /// every member is the same grey, and on a real outlier is a real number.
    #[test]
    fn a_zero_mad_refuses_the_question_instead_of_answering_infinity() {
        let flat_wall = Consensus { feature: FeatureKey::Thick, median: 3.0, mad: 0.0, n: 7 };
        assert_eq!(flat_wall.mads(3.0), None);
        assert_eq!(flat_wall.mads(50.0), None, "a wall has no scale outside its collection");

        let flat_grey = Consensus { feature: FeatureKey::LabA, median: 0.0, mad: 0.0, n: 7 };
        assert_relative_eq!(flat_grey.mads(0.0).expect("the eye's scale"), 0.0);
        assert_relative_eq!(
            flat_grey.mads(50.0).expect("the eye's scale"),
            50.0 / COLOUR_MAD_FLOOR,
            epsilon = 1e-12
        );

        let real = Consensus { feature: FeatureKey::Thick, median: 3.5, mad: 0.25, n: 7 };
        assert_relative_eq!(real.mads(4.0).expect("a scale"), 2.0, epsilon = 1e-12);
    }

    /// A consensus of two is a subtraction and not a spread, so nothing is measured against one —
    /// which is the terracotta's three fragments, where a 2-member MAD called the third an
    /// eleven-sigma outlier of a collection that is one pot.
    #[test]
    fn a_consensus_of_two_measures_nothing() {
        let p = ObjectParams {
            demote: FeatureSet::none().with(FeatureKey::Thick),
            ..Default::default()
        };
        let pair = GroupFeatures::of([&feature(38.808, None), &feature(39.024, None)]);
        assert_eq!(pair.get(FeatureKey::Thick).expect("a row").n, 2);
        assert!(
            exceeded(&pair, &feature(40.177, None), &p).is_empty(),
            "11.7 MAD from a two-member median is arithmetic, not an outlier"
        );
    }

    /// With a feature on the shortlist the arm fires, and it fires on the member's distance from
    /// the **other** members — not from a median it helped to set.
    #[test]
    fn a_shortlisted_feature_measures_the_member_against_the_rest_of_its_object() {
        let p = ObjectParams {
            demote: FeatureSet::none().with(FeatureKey::Thick),
            k_mad: 3.0,
            ..Default::default()
        };
        let others =
            GroupFeatures::of([&feature(3.0, None), &feature(3.2, None), &feature(3.4, None)]);
        // median 3.2, MAD 0.2: 4.0 is four MADs out, 3.6 is two.
        assert!(exceeded(&others, &feature(3.6, None), &p).is_empty());
        let far = exceeded(&others, &feature(4.0, None), &p);
        assert_eq!(far.len(), 1);
        assert_eq!(far[0].feature, FeatureKey::Thick);
        assert_relative_eq!(far[0].mads, 4.0, epsilon = 1e-9);
        assert!(far[0].demoting);
    }

    /// Every name the CLI accepts round-trips, and an unknown one is refused rather than ignored.
    #[test]
    fn every_feature_parses_from_the_name_it_prints() {
        for k in FeatureKey::ALL {
            assert_eq!(FeatureKey::parse(k.key()), Some(k), "{}", k.key());
        }
        assert_eq!(FeatureKey::parse("SHELL_RADIUS"), Some(FeatureKey::ShellRadius));
        assert_eq!(FeatureKey::parse("wall"), None);
    }
}
