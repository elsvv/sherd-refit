//! Running the port's stages against a fixture dump (D §10.2).
//!
//! A [`Collection`] is one dump plus, optionally, the directory the reference read: the dump
//! carries every stage boundary, and the source files are what native mode needs, since native
//! mode by definition starts from the file rather than from the reference's arrays. Each stage
//! runner then produces a [`StageReport`](crate::report::StageReport) of individual
//! [`Check`](crate::report::Check)s.
//!
//! Plan step S4 filled in the first three rows of D §10.2's table — `load`, `thickness` and
//! `working mesh` — step B1 the fourth, `segmentation`, and step B2 the fifth, `breakline`.
//! `hypotheses` and the rest follow their stages in phases 1b–1d, as new modules beside these.

pub mod breakline;
pub mod load;
pub mod segmentation;
pub mod thickness;
pub mod working_mesh;

use std::path::{Path, PathBuf};

use sherd_core::error::{Error, Result};
use sherd_core::mesh::Mesh;

use crate::layout::FixtureDir;
use crate::manifest::Manifest;
use crate::npy;
use crate::report::{Mode, StageReport};

/// A stage of D §10.2's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// R §3.1 — read, clean, largest component.
    Load,
    /// R §3.2 — the wall thickness and the plain ray mode.
    Thickness,
    /// R §3.3 — the decimated, smoothed mesh and everything derived from it.
    WorkingMesh,
    /// R §3.4 — shell against fracture, face by face.
    Segmentation,
    /// R §3.5.3–3.5.5 — the breakline points, their frames and the hypothesis subset.
    Breakline,
}

impl Stage {
    /// Every stage this build can run, in pipeline order.
    pub const ALL: [Self; 5] =
        [Self::Load, Self::Thickness, Self::WorkingMesh, Self::Segmentation, Self::Breakline];

    /// The name the command line and the table use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Thickness => "thickness",
            Self::WorkingMesh => "working-mesh",
            Self::Segmentation => "segmentation",
            Self::Breakline => "breakline",
        }
    }

    /// The stage of that name.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|stage| stage.as_str() == s)
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A cleaned mesh as R §3.1 leaves it: `(V0, F0)`.
pub type Original = (Vec<[f64; 3]>, Vec<[u32; 3]>);

/// Where a [`FragmentFixture::original`] came from — the distinction finding F3 asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OriginalSource {
    /// `load.V0` / `load.F0` of the dump: the reference's own arrays.
    Dump,
    /// Read from the source file by this port, because the dump does not carry them.
    Recomputed,
}

impl OriginalSource {
    /// Why an injected comparison cannot be made from arrays of this provenance, or `None` when
    /// it can.
    pub fn injected_skip_reason(self) -> Option<&'static str> {
        match self {
            Self::Dump => None,
            Self::Recomputed => Some(
                "no load.V0 in the dump (level slim or min): injected mode would run on the \
                 port's own (V0, F0), which is native mode under another name",
            ),
        }
    }
}

/// One fragment of a dump: where its stage boundaries are, and which file it was made from.
#[derive(Clone, Debug)]
pub struct FragmentFixture {
    /// The fragment's name in the collection (R §2).
    pub name: String,
    /// `DIR/fragments/<name>`.
    pub dir: PathBuf,
    /// The source mesh, when the input directory was given and holds it.
    pub source: Option<PathBuf>,
}

impl FragmentFixture {
    /// A file inside the fragment's fixture directory.
    pub fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// True when the dump carries that file (the `slim` and `min` levels of D §10.1 leave some
    /// out).
    pub fn has(&self, name: &str) -> bool {
        self.file(name).is_file()
    }

    /// R §3.1's `(V0, F0)` as the *dump* holds them, or `None` at the `slim` and `min` levels,
    /// which do not carry them.
    ///
    /// This is the only source an injected comparison may use. Injected means "the Rust stage ran
    /// on the Python stage's own inputs" (D §10.2); arrays the port computed itself are not that,
    /// however likely they are to be equal.
    pub fn dumped_original(&self) -> Result<Option<Original>> {
        if !self.has("load.V0.npy") || !self.has("load.F0.npy") {
            return Ok(None);
        }
        let v = npy::read_points(self.file("load.V0.npy"))?;
        let f = npy::read_triangles(self.file("load.F0.npy"))?;
        Ok(Some((v, f)))
    }

    /// R §3.1's `(V0, F0)`: the cleaned largest component the rest of R §3 is built on, from the
    /// dump when it is there and from the source file otherwise, saying **which**.
    ///
    /// Finding F3 of the phase-1a verification: the recomputation used to be silent, and was
    /// justified by "the `load` stage compares the two and the comparison is exact" — which is
    /// true only at the `full` level. At `slim` and `min` the load stage has no `load.V0` to
    /// compare against and skips that check, so on a `slim` dump the fallback was feeding the
    /// port's own arrays into an *injected* comparison with nothing pinning them to the
    /// reference's. Callers now decide: native mode may recompute (it starts from the file
    /// anyway), injected mode may not.
    pub fn original(&self) -> Result<Option<(Original, OriginalSource)>> {
        if let Some(dumped) = self.dumped_original()? {
            return Ok(Some((dumped, OriginalSource::Dump)));
        }
        let Some(source) = &self.source else { return Ok(None) };
        let mut mesh = sherd_core::io::load_mesh(source)?;
        sherd_core::mesh::components::largest_component(&mut mesh);
        Ok(Some(((mesh.v, mesh.f), OriginalSource::Recomputed)))
    }

    /// The working mesh the reference produced (`mesh.V`, `mesh.F`).
    pub fn working(&self) -> Result<Option<Mesh>> {
        if !self.has("mesh.V.npy") || !self.has("mesh.F.npy") {
            return Ok(None);
        }
        let v = npy::read_points(self.file("mesh.V.npy"))?;
        let f = npy::read_triangles(self.file("mesh.F.npy"))?;
        Ok(Some(Mesh::new(v, f)))
    }
}

/// A fixture dump, resolved against the collection it was made from.
#[derive(Clone, Debug)]
pub struct Collection {
    /// The dump.
    pub dir: FixtureDir,
    /// Its manifest.
    pub manifest: Manifest,
    /// The directory holding the source meshes, when the caller gave one.
    pub input: Option<PathBuf>,
    /// The face cap the reference ran with, from
    /// [`Collection::face_cap`](crate::manifest::Collection::face_cap) — so
    /// [`NO_CAP`](crate::manifest::NO_CAP) when the manifest names none.
    pub target_faces: usize,
    /// The fragments, in collection order (R §2).
    pub fragments: Vec<FragmentFixture>,
}

impl Collection {
    /// Reads a dump's manifest and pairs each fragment with its source file.
    ///
    /// `input` may be `None`: the injected comparisons of `load`, `thickness` and `working mesh`
    /// all run off the dump alone as long as it was written at the `full` level. Native mode and
    /// a `slim` dump need the files, and every stage that does says so by skipping.
    pub fn open(dir: FixtureDir, input: Option<&Path>) -> Result<Self> {
        let manifest = dir.load_manifest()?;
        let names = manifest.pairs.names.clone();
        let files = manifest.collection.files.clone();
        if !files.is_empty() && files.len() != names.len() {
            return Err(Error::fixture(
                dir.manifest_path(),
                format!("{} files but {} fragment names", files.len(), names.len()),
            ));
        }
        let fragments = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let source = input.and_then(|dir| {
                    let file = files.get(i).map_or_else(|| name.clone(), Clone::clone);
                    let path = dir.join(file);
                    path.is_file().then_some(path)
                });
                FragmentFixture { name: name.clone(), dir: dir.fragment_dir(name), source }
            })
            .collect();
        let target_faces = manifest.collection.face_cap();
        Ok(Self { dir, manifest, input: input.map(Path::to_path_buf), target_faces, fragments })
    }

    /// Runs one stage in one mode.
    pub fn run(&self, stage: Stage, mode: Mode) -> Result<StageReport> {
        match stage {
            Stage::Load => load::run(self, mode),
            Stage::Thickness => thickness::run(self, mode),
            Stage::WorkingMesh => working_mesh::run(self, mode),
            Stage::Segmentation => segmentation::run(self, mode),
            Stage::Breakline => breakline::run(self, mode),
        }
    }

    /// Runs several stages in one mode, in pipeline order.
    pub fn run_all(&self, stages: &[Stage], mode: Mode) -> Result<Vec<StageReport>> {
        stages.iter().map(|&stage| self.run(stage, mode)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Collection, OriginalSource, Stage};
    use crate::layout::FixtureDir;
    use std::path::{Path, PathBuf};

    pub(crate) fn slab_dump() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/dump")
    }

    pub(crate) fn slab_input() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
    }

    /// A throwaway dump built from the slab's: its manifest, optionally edited, plus only the
    /// named per-fragment files.
    ///
    /// This is how a test asks what the harness does when a dump does *not* carry something —
    /// a `slim` level with no `load.V0`, a manifest with no `target_faces` — without needing a
    /// second fixture in the repository.
    pub(crate) fn partial_slab_dump(
        tag: &str,
        keep: &[&str],
        edit: impl FnOnce(&mut serde_json::Value),
    ) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sherd-parity-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a scratch dump directory");

        let text = std::fs::read_to_string(slab_dump().join("manifest.json"))
            .expect("the committed manifest reads");
        let mut manifest: serde_json::Value =
            serde_json::from_str(&text).expect("the manifest is JSON");
        edit(&mut manifest);
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string(&manifest).expect("the manifest serialises"),
        )
        .expect("the manifest is written");

        for name in ["pieceA", "pieceB"] {
            let out = dir.join("fragments").join(name);
            std::fs::create_dir_all(&out).expect("a fragment directory");
            for file in keep {
                let from = slab_dump().join("fragments").join(name).join(file);
                if from.is_file() {
                    std::fs::copy(&from, out.join(file)).expect("the fixture file copies");
                }
            }
        }
        dir
    }

    #[test]
    fn stage_names_round_trip() {
        for stage in Stage::ALL {
            assert_eq!(Stage::parse(stage.as_str()), Some(stage));
            assert_eq!(stage.to_string(), stage.as_str());
        }
        assert_eq!(Stage::parse("hypotheses"), None, "R §5.1, not this build");
    }

    /// Finding F2: `target_faces = 0` and a manifest with no `target_faces` key are the same
    /// thing, and that thing is R §3.3's adaptive budget with no cap on it.
    ///
    /// The old reading handed the 0 straight to `face_budget`, where numpy's floor-first clip made
    /// it `clip(raw, 50000, 0) == 0` and decimated every fragment to nothing: 6 of the slab's 24
    /// native comparisons failed at deviation 1.0. The native stage below is the proof that the
    /// new reading is not merely different but right — on this collection the adaptive budget is
    /// under 200 000 anyway, so removing the cap changes no answer.
    #[test]
    fn a_manifest_without_a_face_cap_means_the_adaptive_budget() {
        use crate::manifest::NO_CAP;

        let dir = partial_slab_dump("no-cap", &["mesh.stats.json"], |m| {
            m["collection"]
                .as_object_mut()
                .expect("collection is an object")
                .remove("target_faces");
        });
        let c = Collection::open(FixtureDir::new(&dir), Some(&slab_input())).unwrap();
        assert_eq!(c.target_faces, NO_CAP, "an absent key is the sentinel, not 0 and not 200 000");

        let r = c.run(Stage::WorkingMesh, crate::report::Mode::Native).unwrap();
        let failures: Vec<String> = r.failures().map(crate::report::Check::line).collect();
        assert_eq!(r.status(), "PASS", "{failures:?}");
        assert_eq!(r.checks.len(), 8);

        // And an explicit 0 reads the same way as an absent key.
        let dir = partial_slab_dump("zero-cap", &[], |m| {
            m["collection"]["target_faces"] = serde_json::json!(0);
        });
        let c = Collection::open(FixtureDir::new(&dir), None).unwrap();
        assert_eq!(c.target_faces, NO_CAP);
    }

    /// Finding F3: a dump that does not carry `load.V0` cannot support an injected comparison of
    /// anything derived from it, and the harness says so instead of recomputing the arrays.
    #[test]
    fn injected_mode_skips_a_fragment_whose_v0_is_not_in_the_dump() {
        let dir = partial_slab_dump(
            "slim",
            &["thick.t.json", "thick.thick_mode.json", "mesh.stats.json"],
            |m| m["level"] = serde_json::json!("slim"),
        );
        let c = Collection::open(FixtureDir::new(&dir), Some(&slab_input())).unwrap();
        // The arrays would have been recomputed happily, which is the trap.
        assert_eq!(
            c.fragments[0].original().unwrap().map(|(_, source)| source),
            Some(OriginalSource::Recomputed)
        );
        assert!(c.fragments[0].dumped_original().unwrap().is_none());

        let r = c.run(Stage::Thickness, crate::report::Mode::Injected).unwrap();
        assert!(r.checks.is_empty(), "nothing may be compared without the reference's own V0");
        assert_eq!(r.skips.len(), 2);
        assert!(r.skips[0].reason.contains("slim or min"), "{}", r.skips[0].reason);
        assert_eq!(r.status(), "SKIP");

        // Native mode never refuses for this reason: it starts from the file by definition. (This
        // partial dump has no `thick.t_hit` either, so it skips for that instead — a different
        // sentence, which is the point.)
        let r = c.run(Stage::Thickness, crate::report::Mode::Native).unwrap();
        assert!(
            r.skips.iter().all(|s| !s.reason.contains("slim or min")),
            "{:?}",
            r.skips.iter().map(|s| s.reason.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_collection_pairs_fragments_with_their_files() {
        let c = Collection::open(FixtureDir::new(slab_dump()), Some(&slab_input())).unwrap();
        assert_eq!(c.target_faces, 200_000);
        assert_eq!(c.fragments.len(), 2);
        assert_eq!(c.fragments[0].name, "pieceA");
        assert!(c.fragments[0].source.as_ref().unwrap().ends_with("pieceA.ply"));
        assert!(c.fragments[0].has("mesh.V.npy"));
        assert!(!c.fragments[0].has("seg.no_such.npy"));

        // Without an input directory the fixture still resolves; only native mode loses.
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        assert!(c.fragments[1].source.is_none());
        let ((v, f), source) = c.fragments[1].original().unwrap().expect("the slab dump is `full`");
        assert_eq!(source, OriginalSource::Dump);
        assert!(!v.is_empty() && !f.is_empty());
        assert!(c.fragments[1].working().unwrap().is_some());
    }
}
