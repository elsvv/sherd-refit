//! Stamps the git commit of the build into the binary (D §4.3, defect V6-D7).
//!
//! D §4.3 asks every report to carry "the git commit" beside `core_version`, `algo_ref` and
//! `cache_version`, and the three constants alone cannot say which build wrote a file: the crate
//! version moves once a release, the algorithm reference once an algorithm changes, and everything
//! between them is invisible. The commit is the only field that separates two builds of one phase.
//!
//! It is read here rather than at run time on purpose. A report says what wrote it, and a binary
//! that asked `git` at run time would report whatever repository it happened to be standing in.
//!
//! **What the stamp claims is exactly "`HEAD` when this crate was compiled".** It does not claim
//! the tree was clean: a build script cannot see an edit in another crate of the workspace without
//! being re-run by it, so a `-dirty` suffix here would be a promise this file cannot keep, and a
//! promise that is wrong occasionally is worse than one that is never made.
//!
//! **Nothing about this is allowed to fail a build.** A source tarball, a vendored crate, a CI
//! checkout without `.git`, a machine with no `git` on `PATH` — every one is a legitimate way to
//! build this crate, and each yields `unknown`, which is a truthful answer to "which commit".
use std::process::Command;

fn main() {
    // `-C` the crate's own directory, so the answer is this repository's and not the caller's.
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR")])
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
            .filter(|s| !s.is_empty())
    };

    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=SHERD_GIT_COMMIT={commit}");

    // Rebuild when the commit moves. `HEAD` alone is not enough — on a branch it holds a symbolic
    // ref whose *content* does not change when you commit — so the reflog, which is appended on
    // every commit and every checkout, is watched beside it. `--git-path` resolves both through
    // git, which is what makes this right inside a linked worktree, where `.git` is a file.
    for path in ["HEAD", "logs/HEAD"] {
        if let Some(resolved) = git(&["rev-parse", "--git-path", path]) {
            println!("cargo:rerun-if-changed={resolved}");
        }
    }
    println!("cargo:rerun-if-env-changed=SHERD_GIT_COMMIT");
}
