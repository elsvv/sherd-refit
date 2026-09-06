//! Running the port's stages against a fixture dump (D §10.2).
//!
//! A [`Collection`] is one dump plus, optionally, the directory the reference read: the dump
//! carries every stage boundary, and the source files are what native mode needs, since native
//! mode by definition starts from the file rather than from the reference's arrays. Each stage
//! runner then produces a [`StageReport`](crate::report::StageReport) of individual
//! [`Check`](crate::report::Check)s.
//!
//! Plan step S4 fills in the first three rows of D §10.2's table — `load`, `thickness` and
//! `working mesh`, which is everything the port computes so far. `segmentation`, `breakline`,
//! `hypotheses` and the rest follow their stages in phases 1b–1d, as new modules beside these.

pub mod load;
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
}

impl Stage {
    /// Every stage this build can run, in pipeline order.
    pub const ALL: [Self; 3] = [Self::Load, Self::Thickness, Self::WorkingMesh];

    /// The name the command line and the table use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Thickness => "thickness",
            Self::WorkingMesh => "working-mesh",
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

    /// R §3.1's `(V0, F0)`: the cleaned largest component the rest of R §3 is built on.
    ///
    /// Taken from the dump when it is there (`full` level), and otherwise recomputed from the
    /// source file — which is legitimate for both modes, because the `load` stage compares the
    /// two and the comparison is exact.
    pub fn original(&self) -> Result<Option<Original>> {
        if self.has("load.V0.npy") && self.has("load.F0.npy") {
            let v = npy::read_points(self.file("load.V0.npy"))?;
            let f = npy::read_triangles(self.file("load.F0.npy"))?;
            return Ok(Some((v, f)));
        }
        let Some(source) = &self.source else { return Ok(None) };
        let mut mesh = sherd_core::io::load_mesh(source)?;
        sherd_core::mesh::components::largest_component(&mut mesh);
        Ok(Some((mesh.v, mesh.f)))
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
    /// The face cap the reference ran with (`collection.target_faces`).
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
        let target_faces = usize::try_from(manifest.collection.target_faces).unwrap_or(200_000);
        Ok(Self { dir, manifest, input: input.map(Path::to_path_buf), target_faces, fragments })
    }

    /// Runs one stage in one mode.
    pub fn run(&self, stage: Stage, mode: Mode) -> Result<StageReport> {
        match stage {
            Stage::Load => load::run(self, mode),
            Stage::Thickness => thickness::run(self, mode),
            Stage::WorkingMesh => working_mesh::run(self, mode),
        }
    }

    /// Runs several stages in one mode, in pipeline order.
    pub fn run_all(&self, stages: &[Stage], mode: Mode) -> Result<Vec<StageReport>> {
        stages.iter().map(|&stage| self.run(stage, mode)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Collection, Stage};
    use crate::layout::FixtureDir;
    use std::path::{Path, PathBuf};

    pub(crate) fn slab_dump() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/dump")
    }

    pub(crate) fn slab_input() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/slab/input")
    }

    #[test]
    fn stage_names_round_trip() {
        for stage in Stage::ALL {
            assert_eq!(Stage::parse(stage.as_str()), Some(stage));
            assert_eq!(stage.to_string(), stage.as_str());
        }
        assert_eq!(Stage::parse("segmentation"), None, "phase 1b, not this build");
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
        let (v, f) = c.fragments[1].original().unwrap().expect("the slab dump is `full`");
        assert!(!v.is_empty() && !f.is_empty());
        assert!(c.fragments[1].working().unwrap().is_some());
    }
}
