//! What the three pair stages read: `DIR/pairs/<a>__<b>/`, and the reference's frames at the
//! pair's own `t` (D §10.1).
//!
//! A pair is matched at `t_pair = min(t_A, t_B)` (R §4.2), so one of the two fragments is *not* at
//! its own wall thickness and its match arrays were rebuilt. The dump keys those rebuilds by
//! `(fragment, t, surface_points)` rather than by pair — `fragments/<name>/md_t/t…_sp…/` — so
//! [`reference_frames`] resolves which directory a given pair actually used by reading the
//! candidates' own `md.params.json`, rather than by reproducing Python's `%.9g` in a path name.
//!
//! Everything here is the reference's own array. That is the whole content of the word
//! *injected*: the port's stage runs on the Python stage's inputs, so a difference is the stage's
//! and not an inheritance from the stage above it (D §10.2).

use std::path::PathBuf;

use sherd_core::error::{Error, Result};
use sherd_core::matching::hypotheses::Frames;
use sherd_core::matching::scales::Scales;

use super::{Collection, FragmentFixture};
use crate::npy;

/// One pair directory of a dump.
#[derive(Clone, Debug)]
pub struct PairFixture {
    /// The first fragment's name, as `itertools.combinations` ordered it (R §4.1).
    pub a: String,
    /// The second fragment's name.
    pub b: String,
    /// `DIR/pairs/<a>__<b>`.
    pub dir: PathBuf,
}

impl PairFixture {
    /// The scope name the table prints: `a__b`, as the dump directory is called.
    pub fn scope(&self) -> String {
        format!("{}__{}", self.a, self.b)
    }

    /// A file inside the pair's directory.
    pub fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// True when the dump carries that file (level `min` drops the whole `pair` group).
    pub fn has(&self, name: &str) -> bool {
        self.file(name).is_file()
    }

    /// `md_used.json`: which arrays the pair used, and at which `t`.
    pub fn md_used(&self) -> Result<MdUsed> {
        npy::read_json_as(self.file("md_used.json"))
    }

    /// `scales.json`: R §1.2 as the reference resolved it for this pair.
    pub fn scales(&self) -> Result<Scales> {
        npy::read_json_as(self.file("scales.json"))
    }

    /// The hypothesis inputs: the two subsets and the pairs the dihedral filter kept.
    pub fn hypotheses(&self) -> Result<RefHypotheses> {
        Ok(RefHypotheses {
            ia: npy::read_indices(self.file("hyp.ia.npy"))?,
            ib: npy::read_indices(self.file("hyp.ib.npy"))?,
            pa: npy::read_indices(self.file("hyp.pa.npy"))?,
            pb: npy::read_indices(self.file("hyp.pb.npy"))?,
        })
    }
}

/// `md_used.json` — the pair's `t`, both fragments' own `t` and `res`, and the sample count the
/// arrays were drawn with.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct MdUsed {
    /// `t_pair = min(t_a, t_b)`.
    pub t: f64,
    /// A's own wall thickness.
    pub t_a: f64,
    /// B's own wall thickness.
    pub t_b: f64,
    /// A's working-mesh resolution.
    pub res_a: f64,
    /// B's working-mesh resolution.
    pub res_b: f64,
    /// The `surface_points` the arrays were drawn with — part of the `md_t` key.
    pub surface_points: u64,
}

/// The reference's own hypothesis inputs and output indices (R §5.1).
///
/// Every stage that reads these indexes the frames with them, so [`RefHypotheses::describes`] is
/// the reader check the writer cannot make: a dump whose arrays do not fit together is skipped
/// with a reason rather than indexed off the end.
#[derive(Clone, Debug)]
pub struct RefHypotheses {
    /// `hyp.ia`: A's breakline subset, in the reference's own order (PMC-4).
    pub ia: Vec<u32>,
    /// `hyp.ib`: B's.
    pub ib: Vec<u32>,
    /// `hyp.pa`: position into `ia` of each kept frame pair.
    pub pa: Vec<u32>,
    /// `hyp.pb`: position into `ib`.
    pub pb: Vec<u32>,
}

impl RefHypotheses {
    /// True when `ia`/`ib` index those two breaklines and `pa`/`pb` index `ia`/`ib`.
    pub fn describes(&self, a: &Frames, b: &Frames) -> bool {
        self.pa.len() == self.pb.len()
            && self.ia.iter().all(|&i| (i as usize) < a.len())
            && self.ib.iter().all(|&i| (i as usize) < b.len())
            && self.pa.iter().all(|&i| (i as usize) < self.ia.len())
            && self.pb.iter().all(|&i| (i as usize) < self.ib.len())
    }
}

impl Collection {
    /// The dump's pairs, in the order the reference matched them (`pairs.json`).
    ///
    /// A pair whose directory is not there — level `min`, or a pair skipped after the manifest was
    /// written — is left out, so a caller can iterate without checking.
    pub fn pair_fixtures(&self) -> Vec<PairFixture> {
        self.manifest
            .pairs
            .pairs
            .iter()
            .map(|names| PairFixture {
                a: names[0].clone(),
                b: names[1].clone(),
                dir: self.dir.pair_dir(&names[0], &names[1]),
            })
            .filter(|pair| pair.dir.is_dir())
            .collect()
    }

    /// The fragment of that name.
    pub fn fragment(&self, name: &str) -> Option<&FragmentFixture> {
        self.fragments.iter().find(|f| f.name == name)
    }
}

/// The reference's breakline frames for one fragment at wall thickness `t` (R §3.5.3–3.5.5,
/// R §3.6).
///
/// Returns `None` when the dump has no arrays at that `t` — a `min`-level dump, or a fragment
/// whose rebuild directory is missing — and the caller then skips the pair rather than comparing
/// against something else.
///
/// `md.brk_t` and `md.brk_dih` are read from the dump rather than recomputed from `ns`, `nf` and
/// `f`: they are R §3.6's own derived arrays, and injecting them keeps this stage's residual the
/// hypothesis construction's rather than the breakline stage's.
pub fn reference_frames(
    fragment: &FragmentFixture,
    t: f64,
    surface_points: u64,
) -> Result<Option<Frames>> {
    let Some(dir) = arrays_at(fragment, t, surface_points)? else {
        return Ok(None);
    };
    let file = |name: &str| dir.join(name);
    for name in ["md.brk_P.npy", "md.brk_ns.npy", "md.brk_f.npy", "md.brk_t.npy"] {
        if !file(name).is_file() {
            return Ok(None);
        }
    }
    let frames = Frames {
        p: npy::read_points(file("md.brk_P.npy"))?,
        ns: npy::read_points(file("md.brk_ns.npy"))?,
        f: npy::read_points(file("md.brk_f.npy"))?,
        tangent: npy::read_points(file("md.brk_t.npy"))?,
        dih: npy::read_f64(file("md.brk_dih.npy"))?,
        sub: npy::read_indices(file("md.brk_sub.npy"))?,
    };
    let n = frames.p.len();
    if frames.ns.len() != n
        || frames.f.len() != n
        || frames.tangent.len() != n
        || frames.dih.len() != n
    {
        return Err(Error::fixture(
            dir.join("md.brk_P.npy"),
            "the breakline arrays of the dump have different lengths",
        ));
    }
    Ok(Some(frames))
}

/// Which directory of the dump holds this fragment's arrays at `(t, surface_points)`: its own,
/// or one of its `md_t/` rebuilds.
///
/// The match is on `md.params.json`'s `t` **bit for bit**, because that is what the reference
/// itself compares (`arrays["params"] != want`) when it decides whether to rebuild.
fn arrays_at(fragment: &FragmentFixture, t: f64, surface_points: u64) -> Result<Option<PathBuf>> {
    let mut candidates = vec![fragment.dir.clone()];
    let rebuilds = fragment.dir.join("md_t");
    if let Ok(entries) = std::fs::read_dir(&rebuilds) {
        let mut dirs: Vec<PathBuf> =
            entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        candidates.extend(dirs);
    }
    for dir in candidates {
        let params = dir.join("md.params.json");
        if !params.is_file() {
            continue;
        }
        let value = npy::read_json(&params)?;
        let their_t = npy::field_f64(&value, "t", &params)?;
        let their_points = npy::field_u64(&value, "surface_points", &params)?;
        if their_t.to_bits() == t.to_bits() && their_points == surface_points {
            return Ok(Some(dir));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the dump's own doubles, compared as identities")]

    use super::reference_frames;
    use crate::layout::FixtureDir;
    use crate::stages::Collection;
    use crate::stages::tests::slab_dump;

    #[test]
    fn the_pair_directory_and_both_sides_arrays_resolve() {
        let c = Collection::open(FixtureDir::new(slab_dump()), None).unwrap();
        let pairs = c.pair_fixtures();
        assert_eq!(pairs.len(), 1);
        let pair = &pairs[0];
        assert_eq!(pair.scope(), "pieceA__pieceB");
        assert!(pair.has("hyp.pa.npy") && pair.has("coarse.cs.npy"));

        let used = pair.md_used().unwrap();
        assert_eq!(used.t, used.t_b, "the slab pair is matched at pieceB's wall");
        assert!(used.t < used.t_a);
        assert_eq!(used.surface_points, 20000);

        let sc = pair.scales().unwrap();
        assert_eq!(sc.t, used.t);
        assert_eq!(sc.res, used.res_a.max(used.res_b));

        // pieceA is not at its own `t`, so its arrays come out of `md_t/`; pieceB's are its own.
        let a = c.fragment(&pair.a).unwrap();
        let b = c.fragment(&pair.b).unwrap();
        let fa = reference_frames(a, used.t, used.surface_points).unwrap().unwrap();
        let fb = reference_frames(b, used.t, used.surface_points).unwrap().unwrap();
        assert!(!fa.is_empty() && !fb.is_empty());
        assert_eq!(fa.ns.len(), fa.p.len());
        assert!(fa.sub.iter().all(|&i| (i as usize) < fa.len()));

        // pieceA at its *own* thickness is a different directory of the dump, and it is there too.
        assert_ne!(
            super::arrays_at(a, used.t_a, used.surface_points).unwrap(),
            super::arrays_at(a, used.t, used.surface_points).unwrap(),
            "the pair's `t` is not pieceA's own, so it reads the `md_t` rebuild"
        );
        let own = reference_frames(a, used.t_a, used.surface_points).unwrap().unwrap();
        // The *curve* of R §3.5.3 is the boundary of the fracture mask and does not depend on `t`
        // at all; what a rebuild moves is the frames on it (R §3.5.4's macro radii are `0.15 t`
        // and `0.60 t`) and the subset (R §3.5.5's voxel is `0.5 t`).
        assert_eq!(own.p, fa.p);
        assert_ne!(own.ns, fa.ns, "a rebuild at another `t` moves the macro normals");

        // The hypothesis indices of the dump address the arrays of the dump.
        let hyp = pair.hypotheses().unwrap();
        assert!(hyp.describes(&fa, &fb));
        assert!(!hyp.describes(&fb, &fa) || fa.len() == fb.len());
        let broken = super::RefHypotheses { pa: vec![u32::MAX], ..hyp.clone() };
        assert!(!broken.describes(&fa, &fb), "a `pa` off the end of `ia`");
        let broken = super::RefHypotheses { ia: vec![u32::MAX], ..hyp };
        assert!(!broken.describes(&fa, &fb), "an `ia` off the end of the breakline");

        // A thickness nothing was built at resolves to nothing rather than to the nearest one.
        assert!(reference_frames(a, 1.0, used.surface_points).unwrap().is_none());
        assert!(reference_frames(a, used.t, 12345).unwrap().is_none());
    }
}
