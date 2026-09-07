//! Operator constraints (roadmap item 3, D §11): a `constraints.json` of `{must_join,
//! must_not_join}`.
//!
//! `must_not_join` removes pairs before matching and rejects candidates; `must_join` forces a pair
//! through the second-pass budget and accepts its best candidate that passes the *probable*
//! thresholds. **Phase 1 fixes the file format and nothing else**: nothing in the port reads a
//! [`Constraints`] yet, and [`assemble`](super::greedy::assemble) does not take one. The type is
//! here so that the format is settled while R §8 is being written rather than after it, and so
//! that a file written today is still readable when the behaviour lands.
//!
//! ```json
//! { "must_join":     [["Pot_A_Piece_01", "Pot_A_Piece_03"]],
//!   "must_not_join": [["Pot_A_Piece_02", "Pot_A_Piece_05"]] }
//! ```
//!
//! A pair is written by **name**, in either order, because an operator writes it and [`FragId`]s
//! are an artefact of R §2's discovery order.
//!
//! [`FragId`]: crate::types::FragId

use serde::{Deserialize, Serialize};

/// Two fragments an operator has an opinion about, by name.
///
/// Unordered: `["a", "b"]` and `["b", "a"]` name the same pair, and [`Constraints::says`] compares
/// both ways round.
pub type NamedPair = [String; 2];

/// The contents of `constraints.json` (roadmap item 3, D §11).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct Constraints {
    /// Pairs the operator asserts *do* join.
    pub must_join: Vec<NamedPair>,
    /// Pairs the operator asserts do *not* join.
    pub must_not_join: Vec<NamedPair>,
}

/// What the operator said about one pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing — the pair is decided by R §5–§8 alone.
    #[default]
    Unconstrained,
    /// The pair must be joined.
    MustJoin,
    /// The pair must not be joined.
    MustNotJoin,
}

impl Constraints {
    /// What the file says about the pair `(a, b)`, in either order.
    ///
    /// `must_not_join` wins a contradiction: a file that says both things about one pair is a
    /// mistake, and refusing a join is the recoverable half of it.
    pub fn says(&self, a: &str, b: &str) -> Verdict {
        let holds = |pairs: &[NamedPair]| {
            pairs.iter().any(|p| (p[0] == a && p[1] == b) || (p[0] == b && p[1] == a))
        };
        if holds(&self.must_not_join) {
            Verdict::MustNotJoin
        } else if holds(&self.must_join) {
            Verdict::MustJoin
        } else {
            Verdict::Unconstrained
        }
    }

    /// True when the file constrains nothing, which is what an absent `constraints.json` means.
    pub fn is_empty(&self) -> bool {
        self.must_join.is_empty() && self.must_not_join.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{Constraints, Verdict};

    #[test]
    fn a_pair_is_unordered_and_a_refusal_wins_a_contradiction() {
        let text = r#"{"must_join": [["a", "b"], ["c", "d"]], "must_not_join": [["b", "a"]]}"#;
        let c: Constraints = serde_json::from_str(text).expect("the format parses");
        assert_eq!(c.says("a", "b"), Verdict::MustNotJoin, "the refusal wins");
        assert_eq!(c.says("d", "c"), Verdict::MustJoin, "either order names the same pair");
        assert_eq!(c.says("a", "c"), Verdict::Unconstrained);
        assert!(!c.is_empty());
    }

    #[test]
    fn an_empty_file_and_a_missing_key_both_constrain_nothing() {
        let c: Constraints = serde_json::from_str("{}").expect("both keys default");
        assert!(c.is_empty());
        assert_eq!(c.says("a", "b"), Verdict::Unconstrained);
        assert_eq!(serde_json::to_string(&c).unwrap(), r#"{"must_join":[],"must_not_join":[]}"#);
    }
}
