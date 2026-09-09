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

/// Runs `run INPUT --out OUT` and returns its standard output.
fn run(input: &Path, out: &Path) -> String {
    let result = Command::new(env!("CARGO_BIN_EXE_sherd-refit-rs"))
        .args(["run", &input.to_string_lossy(), "--out", &out.to_string_lossy()])
        .output()
        .expect("the binary runs");
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
