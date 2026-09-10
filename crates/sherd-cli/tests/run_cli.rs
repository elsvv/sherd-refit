//! `sherd-refit-rs run` end to end (plan step D3).
//!
//! The gate the plan states for the run is that **two runs are byte-identical** — every output
//! except the wall-clock fields — so that is what this does: run the binary twice into two
//! directories and compare every file it wrote. Two files carry a moving part and it is the same
//! one in both: R §11.2's `timings` in `report.json`, compared by key set rather than by value, and
//! R §11.3's `## Timing` section of `report.md`, compared by stage name. Everything else — the
//! poses, the meshes and the PNGs — is compared byte for byte.
//!
//! The committed slab pair is what runs everywhere. The terracotta scans of
//! `input/test_fragments_1/fragments` are R §13's own gate and live outside git, so the second test
//! runs on demand: `cargo test -p sherd-cli -- --ignored`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sherd-run-it-{}-{tag}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// The one tier flag the slab fixture has to be run with when a test needs a **confirmed** join.
///
/// `fixtures/slab` is a curved wall that genuinely fits itself in two places — at the seam and
/// 88° round — and R §6.5 accepts both (`slab_pair::the_curved_slab_has_a_second_fit`, and
/// `report.json` puts the second placement 13.96 t away at a score the winner beats 4.5×). Task
/// R1's `--tier-rival-refused` is precisely the rule that refuses a margin over a second placement
/// the search itself would have believed, and a two-fragment collection has no third sherd for the
/// support arm, so under the shipped rule this fixture confirms **nothing**. That is the right
/// answer about this slab and the wrong fixture for a test whose subject is something else, so the
/// tests below that need a confirmed join to look at turn that one conjunct off and say so here.
/// Every other conjunct, and every other test in this file, is the default rule.
const AMBIGUOUS: [&str; 2] = ["--tier-rival-refused", "off"];

/// Runs `run INPUT --out OUT` and returns its standard output.
fn run(input: &Path, out: &Path) -> String {
    run_with(input, out, &[])
}

/// The same with extra flags, for the switches whose gate is that they change nothing.
fn run_with(input: &Path, out: &Path, extra: &[&str]) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sherd-refit-rs"));
    command.args(["run", &input.to_string_lossy(), "--out", &out.to_string_lossy()]);
    command.args(extra);
    let result = command.output().expect("the binary runs");
    assert!(result.status.success(), "run failed: {}", String::from_utf8_lossy(&result.stderr));
    String::from_utf8_lossy(&result.stdout).into_owned()
}

/// Every file of a run below `out`, keyed by its path relative to it — the caches excluded, since
/// `segment_cli.rs` already gates those and the second run reads rather than writes them.
fn outputs(out: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(dir: &Path, prefix: &str, into: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).expect("an output directory") {
            let path = entry.expect("a directory entry").path();
            let name = path.file_name().expect("a name").to_string_lossy().into_owned();
            let key = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
            if path.is_dir() {
                if name != "cache" {
                    walk(&path, &key, into);
                }
            } else {
                into.insert(key, std::fs::read(&path).expect("an output file"));
            }
        }
    }
    let mut files = BTreeMap::new();
    walk(out, "", &mut files);
    files
}

/// `report.json` with its two sampled fields reduced to their key sets — the only fields of that
/// file two runs of the same input may differ in (D §7).
///
/// `timings` is a wall clock. `memory` (audit §B.3, task H3) is a resident set read at 100 ms:
/// both are measurements *of* the run rather than results of it, and a run that reproduced either
/// to the byte would be reporting a constant. The stage **names** are compared, because the two
/// runs must still have walked the same stages in the same order.
fn report_without_timings(bytes: &[u8]) -> serde_json::Value {
    let mut value: serde_json::Value =
        serde_json::from_slice(bytes).expect("report.json is valid JSON");
    let timings = value["timings"].as_object().expect("a timings object");
    let keys: Vec<String> = timings.keys().cloned().collect();
    value["timings"] = serde_json::json!(keys);
    if let Some(memory) = value.get("memory") {
        let stages = memory["stages"].as_object().expect("a stages object");
        let keys: Vec<String> = stages.keys().cloned().collect();
        value["memory"] = serde_json::json!(keys);
    }
    value
}

/// `report.md` up to R §11.3's `## Timing` heading, plus the names of the stages under it — the
/// same exemption as above, for the file that prints those seconds to one decimal.
fn markdown_without_timings(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let (before, timing) = text.split_once("## Timing").expect("R §11.3's last section");
    let stages: Vec<&str> = timing
        .lines()
        .filter_map(|line| line.strip_prefix("- "))
        .filter_map(|line| line.split_once(':').map(|(stage, _)| stage))
        .collect();
    format!("{before}## Timing\n{}", stages.join("\n"))
}

fn two_runs_agree(input: &Path) {
    let first_dir = scratch("first");
    let second_dir = scratch("second");
    let summary = run(input, &first_dir);
    run(input, &second_dir);

    let a = outputs(&first_dir);
    let b = outputs(&second_dir);
    assert!(!a.is_empty(), "the run wrote nothing");
    assert_eq!(
        a.keys().collect::<Vec<&String>>(),
        b.keys().collect::<Vec<&String>>(),
        "the two runs wrote the same files"
    );
    for (name, first) in &a {
        let second = &b[name];
        if name == "report.json" {
            assert_eq!(
                report_without_timings(first),
                report_without_timings(second),
                "report.json must agree outside its timings and its sampled memory"
            );
            continue;
        }
        if name == "report.md" {
            assert_eq!(
                markdown_without_timings(first),
                markdown_without_timings(second),
                "report.md must agree outside its `## Timing` seconds"
            );
            continue;
        }
        assert_eq!(first.len(), second.len(), "{name}: size");
        assert!(first == second, "{name}: the two runs must write byte-identical files");
    }

    // R §11's five writers all ran, and the report says what the assembly did.
    for wanted in ["transforms.json", "report.json", "report.md", "preview_segmentation.png"] {
        assert!(a.contains_key(wanted), "{wanted} is missing from {:?}", a.keys());
    }
    assert!(a.keys().any(|k| k.starts_with("placed/")), "R §11.4's placed meshes: {:?}", a.keys());
    assert!(summary.contains("candidates"), "the summary counts what the matcher found: {summary}");
    assert!(
        summary.contains("preprocess") && summary.contains("wall"),
        "and prints R §11.2's per-stage timings: {summary}"
    );

    std::fs::remove_dir_all(&first_dir).ok();
    std::fs::remove_dir_all(&second_dir).ok();
}

#[test]
fn two_runs_on_the_slab_write_byte_identical_outputs() {
    two_runs_agree(&repo_root().join("fixtures/slab/input"));
}

/// Task M1's off switch: `--measure` adds one file and moves nothing.
///
/// The audit's rule for every new behaviour is that switching it off reproduces today's outputs
/// byte for byte, and the measurement pass is the easiest case to get wrong — it runs inside the
/// pipeline, on the fragments and the candidates the run has just produced. So the gate is stated
/// the other way round as well: a run **with** `--measure` must write the same bytes as a run
/// without it, because the pass reads and does not write.
#[test]
fn a_measured_run_writes_the_same_outputs_and_one_more_file() {
    let input = repo_root().join("fixtures/slab/input");
    let plain_dir = scratch("plain");
    let measured_dir = scratch("measured");
    let dump = measured_dir.join("..").join(format!("measure-{}.json", std::process::id()));

    run(&input, &plain_dir);
    run_with(&input, &measured_dir, &["--measure", &dump.to_string_lossy()]);

    let plain = outputs(&plain_dir);
    let measured = outputs(&measured_dir);
    assert_eq!(
        plain.keys().collect::<Vec<&String>>(),
        measured.keys().collect::<Vec<&String>>(),
        "--measure writes its file where it was asked to and adds nothing to the output directory"
    );
    for (name, bytes) in &plain {
        let other = &measured[name];
        if name == "report.json" {
            assert_eq!(report_without_timings(bytes), report_without_timings(other), "{name}");
        } else if name == "report.md" {
            assert_eq!(markdown_without_timings(bytes), markdown_without_timings(other), "{name}");
        } else {
            assert!(bytes == other, "{name}: --measure must not move a byte of the run");
        }
    }

    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&dump).expect("the measurement file")).unwrap();
    let rows = json["rows"].as_array().expect("a rows array");
    let accepted = json["accepted"].as_u64().expect("an accepted count");
    assert_eq!(rows.len() as u64, accepted, "one row per accepted candidate");
    assert!(accepted > 0, "the slab pair has an accepted candidate to measure");
    for row in rows {
        // Every probe answered, and the three redraws are the run's own plus two.
        assert!(row["determined_deg"].as_f64().expect("a determinedness").is_finite());
        assert!(row["slide_t"].as_f64().expect("a slide").is_finite());
        assert_eq!(row["resamples"].as_array().expect("two redraws").len(), 2);
        assert!(row["placements"].as_u64().expect("a placement count") >= 1);
    }

    std::fs::remove_file(&dump).ok();
    std::fs::remove_dir_all(&plain_dir).ok();
    std::fs::remove_dir_all(&measured_dir).ok();
}

/// Roadmap item 3's off switch, and the rule it turns on.
///
/// Two things at once, because they are one statement. `--tiers off` writes exactly the file set
/// and the key set the run wrote before the tier existed — no band on a candidate, no `tiers`
/// block in either JSON file, no new section in `report.md` — which is the off switch audit §E
/// requires of every new behaviour. `--tiers on`, the default, writes a band everywhere and
/// **R §8 assembles from the confirmed band alone**: every join the assembly used has to carry it.
#[test]
fn the_tier_is_an_off_switch_and_the_assembly_is_built_from_the_confirmed_band() {
    let input = repo_root().join("fixtures/slab/input");
    let off_dir = scratch("tiers-off");
    let on_dir = scratch("tiers-on");
    run_with(&input, &off_dir, &["--tiers", "off"]);
    run_with(&input, &on_dir, &AMBIGUOUS);

    let read = |dir: &Path, name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).expect(name)).expect("valid JSON")
    };
    let off_report = read(&off_dir, "report.json");
    let off_transforms = read(&off_dir, "transforms.json");
    assert!(off_report.get("tiers").is_none(), "no tier block with the pass off");
    assert!(off_report["params"].get("tiers").is_none(), "and no tier set in the parameters");
    assert!(off_transforms.get("tiers").is_none(), "nor in transforms.json");
    for candidate in off_report["candidates"].as_array().expect("candidates") {
        assert!(candidate.get("tier").is_none(), "no band on a candidate with the pass off");
    }
    let off_markdown = std::fs::read_to_string(off_dir.join("report.md")).expect("report.md");
    for section in ["## Confirmed joins", "## Probable joins", "## Candidates by fragment"] {
        assert!(!off_markdown.contains(section), "{section} is not written with the pass off");
    }

    let on_report = read(&on_dir, "report.json");
    let on_transforms = read(&on_dir, "transforms.json");
    assert_eq!(on_report["params"]["tiers"]["min_tight"], 0.35, "M1 §3's chosen set");
    assert_eq!(on_report["params"]["tiers"]["max_gap_t"], 0.015);
    assert!(on_report["tiers"]["confirmed"].as_u64().expect("a confirmed count") >= 1);
    assert!(!on_transforms["tiers"].as_array().expect("a band per pair").is_empty());
    let used = on_report["joins_used"].as_array().expect("joins_used");
    assert!(!used.is_empty(), "the slab pair is confirmed and placed");
    for join in used {
        assert_eq!(join["tier"], "confirmed", "R §8 builds from the confirmed band alone");
    }
    for candidate in on_report["candidates"].as_array().expect("candidates") {
        let tier = candidate["tier"].as_str().expect("a band on every candidate");
        assert!(matches!(tier, "confirmed" | "probable" | "rejected"), "{tier}");
        // A refused candidate is never probed, and a probed one carries every quantity the band
        // was decided on.
        if tier == "rejected" {
            assert!(candidate.get("evidence").is_none());
        } else {
            let e = &candidate["evidence"];
            assert!(e["slide_t"].as_f64().expect("a slide").is_finite());
            assert!(e["support"].as_u64().is_some());
            assert!(e["resample_accept"].as_u64().expect("the draws") >= 1);
            assert_eq!(
                e.get("failed").is_none(),
                tier == "confirmed",
                "a confirmed join failed no test and a probable one names the test it failed"
            );
        }
    }
    let on_markdown = std::fs::read_to_string(on_dir.join("report.md")).expect("report.md");
    for section in [
        "## Confirmed joins",
        "## Probable joins",
        "## Rejected",
        "## Candidates by \
                     fragment",
    ] {
        assert!(on_markdown.contains(section), "{section} is missing");
    }

    std::fs::remove_dir_all(&off_dir).ok();
    std::fs::remove_dir_all(&on_dir).ok();
}

/// The two runs of [`the_object_pass_is_an_off_switch_and_reports_a_consensus_without_vetoing`],
/// compared file for file: the same names, the same bytes, and the object pass's own section,
/// block and stage the only differences.
fn same_run_apart_from_the_objects(off_dir: &Path, on_dir: &Path) {
    let (a, b) = (outputs(off_dir), outputs(on_dir));
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "the same files");
    for (name, off_bytes) in &a {
        let on_bytes = &b[name];
        if name == "report.md" {
            // Every line except the `## Objects` section itself, which is the one thing the pass
            // adds to the file.
            let strip = |text: &str| -> String {
                let mut inside = false;
                let mut kept = String::new();
                for line in text.lines() {
                    // `## Timing` is a wall clock and the object pass is one more row in it.
                    if line == "## Timing" {
                        break;
                    }
                    if line == "## Objects" {
                        inside = true;
                    } else if inside && line.starts_with("## ") {
                        inside = false;
                    }
                    if !inside {
                        kept.push_str(line);
                        kept.push('\n');
                    }
                }
                kept
            };
            let (x, y) = (String::from_utf8_lossy(off_bytes), String::from_utf8_lossy(on_bytes));
            assert_eq!(
                strip(&x),
                strip(&y),
                "report.md above `## Timing` and without the `## Objects` section"
            );
        } else if std::path::Path::new(name).extension().is_some_and(|e| e == "json") {
            let parse = |bytes: &[u8]| -> serde_json::Value {
                if name == "report.json" {
                    report_without_timings(bytes)
                } else {
                    serde_json::from_slice(bytes).expect("valid JSON")
                }
            };
            let mut x = parse(off_bytes);
            let mut y = parse(on_bytes);
            for value in [&mut x, &mut y] {
                if let Some(map) = value.as_object_mut() {
                    map.remove("objects");
                    // The pass is one more stage, so it is one more key in the two sampled
                    // blocks; `report_without_timings` has already reduced both to their names.
                    for block in ["timings", "memory"] {
                        if let Some(names) = map.get_mut(block).and_then(|v| v.as_array_mut()) {
                            names.retain(|n| n != "objects");
                        }
                    }
                }
                if let Some(map) = value["params"].as_object_mut() {
                    map.remove("objects");
                }
            }
            assert_eq!(x, y, "{name} apart from the object block and the object stage");
        } else {
            assert_eq!(off_bytes, on_bytes, "{name}");
        }
    }
}

/// Roadmap item 4's off switch, and what it turns on (audit §D.2, task O1).
///
/// Two things at once, because they are one statement. `--objects off` writes exactly the file set
/// and the key set task T2's run wrote — no `## Objects` section, no `objects` block in
/// `report.json`, no object rules in `params` — and every other byte of the run is the same as the
/// default's, which is what makes the switch an off switch rather than a second algorithm.
/// `--objects on`, the default, writes one object per group with the consensus its members agree
/// on, and **demotes nothing**: task M1 §4 measured no feature reaching audit §D.2's own AUC of
/// 0.800 on any collection with real object ids, so the shortlist is empty and every number
/// reports.
#[test]
fn the_object_pass_is_an_off_switch_and_reports_a_consensus_without_vetoing() {
    let input = repo_root().join("fixtures/slab/input");
    let off_dir = scratch("objects-off");
    let on_dir = scratch("objects-on");
    run_with(&input, &off_dir, &["--objects", "off"]);
    run(&input, &on_dir);

    let off = report_of(&off_dir);
    assert!(off.get("objects").is_none(), "no object block with the pass off");
    assert!(off["params"].get("objects").is_none(), "and no object rules in the parameters");
    let off_markdown = std::fs::read_to_string(off_dir.join("report.md")).expect("report.md");
    assert!(!off_markdown.contains("## Objects"), "and no section");

    // Nothing else moved: the two runs are the same run apart from the section and the block.
    same_run_apart_from_the_objects(&off_dir, &on_dir);

    let on = report_of(&on_dir);
    assert_eq!(on["params"]["objects"]["k_mad"], 3.0);
    assert_eq!(on["params"]["objects"]["min_members"], 3);
    assert_eq!(on["params"]["objects"]["merge"], true, "the arm that needs no threshold");
    assert_eq!(
        on["params"]["objects"]["disagreement"], false,
        "O1 §2: the arm removes no false join and costs correct ones"
    );
    assert!(
        on["params"]["objects"].get("demote").is_none(),
        "M1 §4's shortlist is empty, so the key is not even written"
    );
    let objects = on["objects"]["objects"].as_array().expect("one object per group");
    assert_eq!(
        objects.len(),
        on["groups"].as_array().expect("groups").len(),
        "every group is reported as an object"
    );
    assert_eq!(on["objects"]["merges"], 0, "the slab has one pair and nothing to merge");
    assert!(on["objects"].get("demotions").is_none(), "and nothing to demote");
    let consensus = objects[0]["consensus"].as_array().expect("a consensus");
    assert!(
        consensus.iter().any(|c| c["feature"] == "thick"),
        "the wall is in every object's consensus"
    );
    for row in consensus {
        assert!(row["median"].as_f64().expect("a median").is_finite());
        assert!(row["mad"].as_f64().expect("a MAD") >= 0.0);
    }
    let on_markdown = std::fs::read_to_string(on_dir.join("report.md")).expect("report.md");
    assert!(on_markdown.contains("## Objects"));
    assert!(on_markdown.contains("| object | fragments | joins |"));

    std::fs::remove_dir_all(&off_dir).ok();
    std::fs::remove_dir_all(&on_dir).ok();
}

/// `different_object` refuses a **merge** as well as a join, and `same_object` is reported rather
/// than acted on (audit §D.1's *"item 4's consensus and the group purity reporting read them"*).
///
/// The slab is one pair, so the merge itself cannot be exercised here; what can is that the two
/// lists reach the object report at all and that neither of them moves a pose.
#[test]
fn the_two_object_constraints_reach_the_object_report_without_moving_a_pose() {
    let input = repo_root().join("fixtures/slab/input");
    let plain = scratch("objects-plain");
    let same = scratch("objects-same");
    run_with(&input, &plain, &AMBIGUOUS);
    let file = constraints_file(
        "objects-same",
        r#"{"version": 1, "same_object": [["pieceA", "pieceB"]]}"#,
    );
    run_with(
        &input,
        &same,
        &["--tier-rival-refused", "off", "--constraints", &file.to_string_lossy()],
    );

    let report = report_of(&same);
    assert_eq!(report["constraints"]["entries"][0]["list"], "same_object");
    assert_eq!(report["constraints"]["entries"][0]["satisfied"], true);
    // The pair is in one group, so the operator and the geometry agree and the split list is empty.
    assert!(
        report["objects"].get("same_object_split").is_none(),
        "the assembly put them together, which is what the operator said"
    );
    assert_eq!(
        report_of(&plain)["fragments"],
        report["fragments"],
        "a constraint never edits a score, and `same_object` never moves a pose"
    );
    std::fs::remove_dir_all(&plain).ok();
    std::fs::remove_dir_all(&same).ok();
    std::fs::remove_file(&file).ok();
}

/// `measure`'s "different placement" is the parity harness's, value for value.
///
/// `sherd_core::measure` cannot import the constant — `sherd-parity` sits above it — and the two
/// have to be the same number for audit §D.1's sentence ("placements more than one wall apart,
/// `candidates.rs`'s `SAME_PLACEMENT_T`") to mean what it says. `sherd-cli` is the crate that sees
/// both, so the tie is asserted here.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the two constants are one literal written twice; the equality is the assertion"
)]
fn the_margin_and_the_parity_row_call_the_same_thing_a_different_placement() {
    assert_eq!(
        sherd_core::measure::SAME_PLACEMENT_T,
        sherd_parity::stages::candidates::SAME_PLACEMENT_T
    );
}

/// R §13's own gate, on the set it is stated for: exactly the joins 021–094 and 094–104, with 007
/// left unplaced. The scans are not in the repository, so this runs on demand.
#[test]
#[ignore = "needs input/test_fragments_1/fragments, which is not in the repository"]
fn the_terracotta_assembles_the_two_joins_of_r_13() {
    let input = repo_root().join("input/test_fragments_1/fragments");
    assert!(input.is_dir(), "{} is missing", input.display());
    let out = scratch("terracotta");
    run(&input, &out);
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("report.json")).expect("report.json"))
            .expect("valid JSON");
    let joins: Vec<(String, String)> = report["joins_used"]
        .as_array()
        .expect("joins_used")
        .iter()
        .map(|j| (j["a"].as_str().expect("a").to_owned(), j["b"].as_str().expect("b").to_owned()))
        .collect();
    assert_eq!(
        joins,
        [
            ("FY234021_reduced".to_owned(), "FY234094_reduced".to_owned()),
            ("FY234094_reduced".to_owned(), "FY234104_reduced".to_owned()),
        ],
        "R §13: the joins used are exactly these two"
    );
    for join in report["joins_used"].as_array().expect("joins_used") {
        assert_eq!(join["pen"].as_f64(), Some(0.0), "R §13: both penetrations are 0");
        assert!(join["tight"].as_f64().expect("tight") >= 0.27, "R §13: tight of both ≥ 0.27");
    }
    let groups = report["groups"].as_array().expect("groups");
    assert_eq!(groups.len(), 2, "one group of three and one singleton");
    assert_eq!(groups[1], serde_json::json!(["FY234007_reduced"]), "R §13: 007 is unplaced");
    std::fs::remove_dir_all(&out).ok();
}

// ---------------------------------------------------------------------------------------------
// Task T2: review images and constraints (audit §D.1)

/// Writes a `constraints.json` beside a scratch directory and returns its path.
fn constraints_file(tag: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sherd-constraints-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join("constraints.json");
    std::fs::write(&path, body).expect("the constraints file");
    path
}

/// `report.json` of a finished run.
fn report_of(out: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(out.join("report.json")).expect("report.json"))
        .expect("valid JSON")
}

/// Audit §E's gate for the review images: **deterministic between two runs**.
///
/// The sampler is seeded per image rather than per pass, so nothing about the image depends on how
/// many pairs came before it or on which thread rayon gave it; two runs must write the same PNG
/// bytes. The rest of the run is compared too, because an image pass that moved a pose would be a
/// worse defect than a non-deterministic image.
#[test]
fn review_images_are_the_same_bytes_twice_and_the_index_links_them() {
    let input = repo_root().join("fixtures/slab/input");
    let first = scratch("review-a");
    let second = scratch("review-b");
    run_with(&input, &first, &["--review-images"]);
    run_with(&input, &second, &["--review-images"]);

    let a = outputs(&first);
    let b = outputs(&second);
    let images: Vec<&String> = a.keys().filter(|k| k.starts_with("review/")).collect();
    assert_eq!(
        images,
        ["review/pieceA__pieceB.png"],
        "one image per confirmed or probable join, named after the pair"
    );
    for name in &images {
        assert!(a[*name] == b[*name], "{name}: the two runs must write the same bytes");
        assert!(a[*name].len() > 1000, "{name}: an empty PNG is not an image");
    }
    let markdown = String::from_utf8(a["report.md"].clone()).expect("report.md is UTF-8");
    assert!(
        markdown.contains("[png](review/pieceA__pieceB.png)"),
        "the per-fragment index links the image"
    );

    // And the run itself did not move: everything but the images is what a run without the flag
    // writes.
    let plain_dir = scratch("review-none");
    run(&input, &plain_dir);
    let plain = outputs(&plain_dir);
    assert!(
        plain.keys().all(|k| !k.starts_with("review/")),
        "no flag, no images: {:?}",
        plain.keys()
    );
    assert!(
        plain["transforms.json"] == a["transforms.json"],
        "the images are an output, not an input: the poses do not move"
    );
    for dir in [&first, &second, &plain_dir] {
        std::fs::remove_dir_all(dir).ok();
    }
}

/// `must_not_join` takes the pair out of the run **before matching**, and the report says so.
///
/// The slab has exactly one pair, so "removed before matching" is visible as a candidate list that
/// is empty rather than merely as an assembly that placed nothing.
#[test]
fn must_not_join_removes_the_pair_before_matching_and_the_report_lists_it() {
    let input = repo_root().join("fixtures/slab/input");
    let out = scratch("must-not-join");
    let file =
        constraints_file("not", r#"{"version": 1, "must_not_join": [["pieceB", "pieceA"]]}"#);
    run_with(&input, &out, &["--constraints", &file.to_string_lossy()]);

    let report = report_of(&out);
    assert_eq!(
        report["candidates"].as_array().expect("candidates").len(),
        0,
        "the pair was never matched, so it produced no candidate"
    );
    assert_eq!(report["joins_used"].as_array().expect("joins_used").len(), 0);
    let entries = report["constraints"]["entries"].as_array().expect("the constraints block");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["list"], "must_not_join");
    assert_eq!(entries[0]["satisfied"], serde_json::json!(true));
    assert!(
        entries[0]["outcome"].as_str().expect("an outcome").contains("before matching"),
        "{:?}",
        entries[0]
    );
    let markdown =
        String::from_utf8(std::fs::read(out.join("report.md")).expect("report.md")).unwrap();
    assert!(markdown.contains("## Constraints"), "report.md lists every constraint");
    assert!(markdown.contains("`must_not_join`"), "and names the list it was written in");
    std::fs::remove_dir_all(&out).ok();
    std::fs::remove_dir_all(file.parent().expect("a parent")).ok();
}

/// A `must_join` carrying a 4x4: **matching is skipped and the pose is a confirmed candidate**.
///
/// The pose pinned is the one an unconstrained run found for the pair, so the assembly must place
/// it; what the test asserts is that it got there without being matched — one candidate for the
/// pair instead of the five R §5.7 keeps — and that the band is confirmed.
#[test]
fn a_pinned_pose_places_without_matching() {
    let input = repo_root().join("fixtures/slab/input");
    let found_dir = scratch("pinned-source");
    run_with(&input, &found_dir, &AMBIGUOUS);
    let found = report_of(&found_dir);
    let candidates = found["candidates"].as_array().expect("candidates");
    assert!(candidates.len() > 1, "the matcher keeps several candidates for this pair");
    let pose = found["joins_used"][0]["T"].clone();
    assert!(pose.is_array(), "the unconstrained run placed the pair");

    let out = scratch("pinned");
    let body = serde_json::json!({
        "version": 1,
        "must_join": [{"a": "pieceA", "b": "pieceB", "pose": pose}],
    });
    let file = constraints_file("pin", &body.to_string());
    run_with(&input, &out, &["--constraints", &file.to_string_lossy()]);

    let report = report_of(&out);
    let pinned = report["candidates"].as_array().expect("candidates");
    assert_eq!(pinned.len(), 1, "matching was skipped: the pinned pose is the only candidate");
    assert_eq!(pinned[0]["tier"], "confirmed", "audit §D.1: the pose is a confirmed candidate");
    assert_eq!(pinned[0]["T"], found["joins_used"][0]["T"], "and it is the pose that was pinned");
    assert_eq!(report["joins_used"].as_array().expect("joins_used").len(), 1, "so R §8 placed it");
    let entry = &report["constraints"]["entries"][0];
    assert_eq!(entry["satisfied"], serde_json::json!(true));
    assert!(
        entry["outcome"].as_str().expect("an outcome").contains("matching skipped"),
        "{entry:?}"
    );
    for dir in [&found_dir, &out] {
        std::fs::remove_dir_all(dir).ok();
    }
    std::fs::remove_dir_all(file.parent().expect("a parent")).ok();
}

/// `different_object` vetoes a join that R §6 accepted, and says whose decision it was.
#[test]
fn different_object_vetoes_a_join_the_geometry_accepted() {
    let input = repo_root().join("fixtures/slab/input");
    let out = scratch("different-object");
    let file =
        constraints_file("diff", r#"{"version": 1, "different_object": [["pieceA", "pieceB"]]}"#);
    run_with(
        &input,
        &out,
        &["--tier-rival-refused", "off", "--constraints", &file.to_string_lossy()],
    );

    let report = report_of(&out);
    assert!(
        report["candidates"].as_array().expect("candidates").iter().any(|c| c["accepted"] == true),
        "the pair was still matched and still accepted: a constraint never edits a score"
    );
    assert_eq!(report["joins_used"].as_array().expect("joins_used").len(), 0, "and not placed");
    let refused = report["joins_rejected"].as_array().expect("joins_rejected");
    assert_eq!(refused.len(), 1);
    assert_eq!(
        refused[0]["reason"], "refused by constraints.json (`different_object`)",
        "{refused:?}"
    );
    std::fs::remove_dir_all(&out).ok();
    std::fs::remove_dir_all(file.parent().expect("a parent")).ok();
}

/// An unknown fragment name is an **error**, not a skip (audit §D.1).
#[test]
fn an_unknown_name_in_the_constraints_file_fails_the_run() {
    let input = repo_root().join("fixtures/slab/input");
    let out = scratch("unknown-name");
    let file =
        constraints_file("unknown", r#"{"version": 1, "must_not_join": [["pieceA", "pieceZ"]]}"#);
    let result = Command::new(env!("CARGO_BIN_EXE_sherd-refit-rs"))
        .args(["run", &input.to_string_lossy(), "--out", &out.to_string_lossy()])
        .args(["--constraints", &file.to_string_lossy()])
        .output()
        .expect("the binary runs");
    assert!(!result.status.success(), "an unknown name must fail the run");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("pieceZ"), "the message names the fragment: {stderr}");
    std::fs::remove_dir_all(&out).ok();
    std::fs::remove_dir_all(file.parent().expect("a parent")).ok();
}

/// Audit §E's two terracotta gates for the constraints file. The scans are not in the repository,
/// so this runs on demand.
///
/// * `must_not_join [021, 094]` leaves 094–104 as the only join, and the report says the pair was
///   removed before matching.
/// * `must_join [007, 021]` is **unsatisfiable** — 007 has no accepted partner at any budget
///   (R §13) — and the run still succeeds and says so.
#[test]
#[ignore = "needs input/test_fragments_1/fragments, which is not in the repository"]
fn the_terracotta_honours_its_two_constraints() {
    let input = repo_root().join("input/test_fragments_1/fragments");
    assert!(input.is_dir(), "{} is missing", input.display());

    let refused = scratch("terracotta-refused");
    let file = constraints_file(
        "terracotta-not",
        r#"{"version": 1,
            "must_not_join": [["FY234021_reduced", "FY234094_reduced"]]}"#,
    );
    run_with(&input, &refused, &["--constraints", &file.to_string_lossy()]);
    let report = report_of(&refused);
    let joins: Vec<(String, String)> = report["joins_used"]
        .as_array()
        .expect("joins_used")
        .iter()
        .map(|j| (j["a"].as_str().expect("a").to_owned(), j["b"].as_str().expect("b").to_owned()))
        .collect();
    assert_eq!(
        joins,
        [("FY234094_reduced".to_owned(), "FY234104_reduced".to_owned())],
        "with 021-094 refused, 094-104 is the only join left"
    );
    assert!(
        report["candidates"]
            .as_array()
            .expect("candidates")
            .iter()
            .all(|c| !(c["a"] == "FY234021_reduced" && c["b"] == "FY234094_reduced")),
        "and the pair was never matched"
    );
    let entry = &report["constraints"]["entries"][0];
    assert_eq!(entry["list"], "must_not_join");
    assert_eq!(entry["satisfied"], serde_json::json!(true));

    let forced = scratch("terracotta-forced");
    let file2 = constraints_file(
        "terracotta-join",
        r#"{"version": 1, "must_join": [["FY234007_reduced", "FY234021_reduced"]]}"#,
    );
    run_with(&input, &forced, &["--constraints", &file2.to_string_lossy()]);
    let report = report_of(&forced);
    let entry = &report["constraints"]["entries"][0];
    assert_eq!(entry["list"], "must_join");
    assert_eq!(entry["satisfied"], serde_json::json!(false), "{entry:?}");
    assert!(entry["outcome"].as_str().expect("an outcome").contains("unsatisfiable"), "{entry:?}");

    for dir in [&refused, &forced] {
        std::fs::remove_dir_all(dir).ok();
    }
    std::fs::remove_dir_all(file.parent().expect("a parent")).ok();
    std::fs::remove_dir_all(file2.parent().expect("a parent")).ok();
}
