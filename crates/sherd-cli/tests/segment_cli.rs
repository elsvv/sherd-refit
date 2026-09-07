//! `sherd-refit-rs segment` end to end (plan step S4).
//!
//! The exit criterion the plan states for the cache is that **two runs produce byte-identical
//! caches**, so that is what these tests do: run the binary twice, over two output directories,
//! and compare the files. The committed slab pair is what runs everywhere; the terracotta scans of
//! `input/test_fragments_1/fragments` are the set the plan names, and the same test runs over them
//! under `--ignored` on a machine that has the data (they are 25 MB scans and are not in git).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sherd-segment-it-{}-{tag}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// Runs `segment INPUT --out OUT` and returns its standard output.
fn segment(input: &Path, out: &Path) -> String {
    let result = Command::new(env!("CARGO_BIN_EXE_sherd-refit-rs"))
        .args(["segment", &input.to_string_lossy(), "--out", &out.to_string_lossy()])
        .output()
        .expect("the binary runs");
    assert!(result.status.success(), "segment failed: {}", String::from_utf8_lossy(&result.stderr));
    String::from_utf8_lossy(&result.stdout).into_owned()
}

/// Every cache file of a run, as `(name, bytes)`, sorted by name.
fn caches(out: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(out.join("cache"))
        .expect("the cache directory")
        .map(|e| {
            let path = e.expect("a directory entry").path();
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&path).expect("a cache file"),
            )
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

fn two_runs_agree(input: &Path) {
    let first_dir = scratch("first");
    let second_dir = scratch("second");

    let first_out = segment(input, &first_dir);
    let second_out = segment(input, &second_dir);

    let a = caches(&first_dir);
    let b = caches(&second_dir);
    assert!(!a.is_empty(), "the run wrote no cache");
    assert_eq!(
        a.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        b.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "the two runs cached the same fragments"
    );
    for ((name, x), (_, y)) in a.iter().zip(&b) {
        assert_eq!(x.len(), y.len(), "{name}: cache size");
        assert!(x == y, "{name}: the two runs must produce byte-identical caches");
    }

    // Both runs computed everything (nothing was carried over between the two directories) and
    // the summary says what the subcommand does and does not do.
    for out in [&first_out, &second_out] {
        assert!(out.contains("miss"), "a cold run computes: {out}");
        assert!(out.contains("fracture"), "the table carries R §3.4's label fraction: {out}");
        assert!(out.contains("brk"), "and R §3.5's breakline counts: {out}");
        assert!(out.contains("margin"), "and R §3.5.6's shell margin: {out}");
        assert!(out.contains("match arrays"), "the summary says where it stops: {out}");
    }

    // A third run over the first directory must hit every cache and change nothing.
    let before = caches(&first_dir);
    let third_out = segment(input, &first_dir);
    assert!(third_out.contains("hit"), "a warm run reads the cache: {third_out}");
    assert!(!third_out.contains("miss"), "and computes nothing: {third_out}");
    assert_eq!(caches(&first_dir), before, "a warm run must not rewrite the caches");

    std::fs::remove_dir_all(&first_dir).ok();
    std::fs::remove_dir_all(&second_dir).ok();
}

#[test]
fn two_runs_on_the_slab_produce_byte_identical_caches() {
    two_runs_agree(&repo_root().join("fixtures/slab/input"));
}

/// The plan's own wording: two runs of `segment` on the terracotta. The scans live outside git, so
/// this runs on demand: `cargo test -p sherd-cli -- --ignored`.
#[test]
#[ignore = "needs input/test_fragments_1/fragments, which is not in the repository"]
fn two_runs_on_the_terracotta_produce_byte_identical_caches() {
    let input = repo_root().join("input/test_fragments_1/fragments");
    assert!(input.is_dir(), "{} is missing", input.display());
    two_runs_agree(&input);
}

/// R's `segment_only` ends with `write_previews(...)`, and so does the port's `segment` (V4-D7).
///
/// The file is the segmentation preview of a collection where every fragment is its own group:
/// two views of 1 400 × 600, the fracture faces in red. Byte-identical between two runs, like the
/// caches — the renderer draws from a seeded sampler.
#[test]
fn segment_writes_the_segmentation_preview() {
    let first = scratch("preview-first");
    let second = scratch("preview-second");
    let input = repo_root().join("fixtures/slab/input");

    let out = segment(&input, &first);
    let png = first.join("preview_segmentation.png");
    assert!(png.is_file(), "segment must write {}: {out}", png.display());
    assert!(out.contains("preview_segmentation.png"), "and name it: {out}");

    segment(&input, &second);
    assert_eq!(
        std::fs::read(&png).expect("the preview"),
        std::fs::read(second.join("preview_segmentation.png")).expect("the preview"),
        "two runs must draw the same preview"
    );
    // A PNG of the reference's own size: `render_views` lays the two views out side by side.
    let bytes = std::fs::read(&png).expect("the preview");
    assert_eq!(&bytes[1..4], b"PNG");
    let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    assert_eq!((width, height), (2800, 600), "two 1400x600 views of R §11.5");

    std::fs::remove_dir_all(&first).ok();
    std::fs::remove_dir_all(&second).ok();
}

#[test]
fn info_names_the_algorithm_reference_the_caches_carry() {
    let out = Command::new(env!("CARGO_BIN_EXE_sherd-refit-rs"))
        .arg("info")
        .output()
        .expect("the binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success());
    assert!(text.contains(sherd_core::ALGO_REF), "{text}");
    assert!(text.contains(&sherd_core::CACHE_VERSION.to_string()), "{text}");
}
