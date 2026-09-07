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

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use sherd_core::error::{Error, Result};
use sherd_core::fragment::samples::registration_split;
use sherd_core::matching::hypotheses::Frames;
use sherd_core::matching::icp::IcpTarget;
use sherd_core::matching::scales::Scales;
use sherd_core::matching::verify::Surfaces;
use sherd_core::mesh::geometry::face_geometry;
use sherd_core::spatial::bvh::RayScene;
use sherd_core::vec3::Vec3f;

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

/// One of R §3.6's point clouds, as the reference built it.
#[derive(Clone, Debug, Default)]
pub struct RefCloud {
    /// The points.
    pub p: Vec<[f64; 3]>,
    /// One unit normal per point.
    pub n: Vec<[f64; 3]>,
}

impl RefCloud {
    /// The cloud as an ICP target, with its tree.
    pub fn target(&self) -> IcpTarget {
        IcpTarget::new(self.p.clone(), self.n.clone())
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.p.len()
    }

    /// True when the cloud holds no point.
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }
}

/// The four clouds R §5.4–5.6 register, rebuilt from one fragment's dumped arrays (R §3.6).
///
/// Everything here is the reference's own: `brk_P` and `brk_ns` come from the dump, `Pf` and `S`
/// come from the dump, the normals are `FN[fp]` and `FN[sp[margin_idx]]` over the dump's own
/// working mesh, and the `pc_reg` prefixes follow R §3.6's split. The only arithmetic the port
/// contributes is the face normals, which step S3 measured bit-identical to numpy's on this input.
#[derive(Clone, Debug, Default)]
pub struct RefClouds {
    /// `pc_brk_full`: the whole breakline with its shell normals (stage 1's target).
    pub brk_full: RefCloud,
    /// `pc_brk`: the breakline subset (stage 1's source).
    pub brk_sub: RefCloud,
    /// `pc_reg`: fracture prefix plus shell-margin prefix (stage 2's first two rungs).
    pub reg: RefCloud,
    /// `pc_frac`: the fracture samples (stage 2's last two rungs).
    pub frac: RefCloud,
}

/// The reference's own face normals `FN`, computed from the dump's own working mesh.
///
/// Returns `None` when the dump carries no working mesh for that fragment (level `min`).
pub fn face_normals(fragment: &FragmentFixture) -> Result<Option<Vec<[f64; 3]>>> {
    let Some(mesh) = fragment.working()? else { return Ok(None) };
    Ok(Some(face_geometry(&mesh.v, &mesh.f).normals))
}

/// R §3.6's clouds for one fragment at the pair's `t`, from the dump's own arrays.
///
/// `frames` is what [`reference_frames`] returned for the same fragment and `t`; `normals` is
/// [`face_normals`] for it, cached by the caller because it costs a pass over the working mesh and
/// every pair the fragment takes part in needs it.
pub fn reference_clouds(
    fragment: &FragmentFixture,
    frames: &Frames,
    normals: &[[f64; 3]],
    t: f64,
    surface_points: u64,
    reg_points: usize,
) -> Result<Option<RefClouds>> {
    let Some(dir) = arrays_at(fragment, t, surface_points)? else { return Ok(None) };
    let file = |name: &str| dir.join(name);
    for name in ["md.Pf.npy", "md.fp.npy", "md.S.npy", "md.sp.npy", "md.margin_idx.npy"] {
        if !file(name).is_file() {
            return Ok(None);
        }
    }
    let pf = npy::read_points(file("md.Pf.npy"))?;
    let fp = npy::read_indices(file("md.fp.npy"))?;
    let s = npy::read_points(file("md.S.npy"))?;
    let sp = npy::read_indices(file("md.sp.npy"))?;
    let margin_idx = npy::read_indices(file("md.margin_idx.npy"))?;
    let faces = normals.len();
    if pf.len() != fp.len() || s.len() != sp.len() {
        return Err(Error::fixture(file("md.Pf.npy"), "a sample array and its face ids differ"));
    }
    if fp.iter().chain(&sp).any(|&f| (f as usize) >= faces)
        || margin_idx.iter().any(|&i| (i as usize) >= s.len())
        || frames.sub.iter().any(|&i| (i as usize) >= frames.len())
    {
        return Err(Error::fixture(
            file("md.fp.npy"),
            "the dump's sample indices do not address its own mesh",
        ));
    }

    let nf: Vec<[f64; 3]> = fp.iter().map(|&f| normals[f as usize]).collect();
    let pm: Vec<[f64; 3]> = margin_idx.iter().map(|&i| s[i as usize]).collect();
    let nm: Vec<[f64; 3]> = margin_idx.iter().map(|&i| normals[sp[i as usize] as usize]).collect();
    let (take_f, take_m) = registration_split(pf.len(), pm.len(), reg_points);
    let mut reg = RefCloud { p: pf[..take_f].to_vec(), n: nf[..take_f].to_vec() };
    reg.p.extend_from_slice(&pm[..take_m]);
    reg.n.extend_from_slice(&nm[..take_m]);

    let at = |source: &[[f64; 3]]| -> Vec<[f64; 3]> {
        frames.sub.iter().map(|&i| source[i as usize]).collect()
    };
    Ok(Some(RefClouds {
        brk_full: RefCloud { p: frames.p.clone(), n: frames.ns.clone() },
        brk_sub: RefCloud { p: at(&frames.p), n: at(&frames.ns) },
        reg,
        frac: RefCloud { p: pf, n: nf },
    }))
}

/// One fragment's working mesh as R §6 reads it: the two BVHs, the face normals and the two
/// numbers the scores need — all from the dump's own `mesh.V`, `mesh.F` and `seg.frac_final`.
///
/// The reference builds exactly these: `Fragment.scene` over the whole working mesh (R §6.4) and
/// `Fragment.frac_scene` over the fracture faces alone (R §6.1), both on `V.astype(float32)`,
/// which is what [`RayScene`] holds too. `frac_area` is `A[frac].sum()` over the dump's own mesh.
#[derive(Debug)]
pub struct RefGeometry {
    /// A BVH over the whole working mesh; `None` when the dump carries no mesh for the fragment.
    pub scene: Option<RayScene>,
    /// A BVH over the fracture faces alone.
    pub fracture: Option<RayScene>,
    /// `FN`, the face normals of that mesh.
    pub normals: Vec<[f64; 3]>,
    /// R §3.4's `fracture_area`.
    pub frac_area: f64,
    /// R §3.3.2's verdict, from `mesh.watertight.json`.
    pub watertight: bool,
    /// How many edges of that mesh are used by a number of faces other than two.
    ///
    /// R §3.3.2's `closed_enough` accepts up to 0.2 % of them, so `watertight` can be true on a
    /// mesh with holes — and on such a mesh a signed distance has no definition, which is what
    /// the `verify` stage's `pen` row has to know about (PMC-7).
    pub n_boundary: u32,
}

impl RefGeometry {
    /// Reads one fragment's mesh, labels and watertightness out of the dump.
    ///
    /// Returns `None` when the dump carries no working mesh (level `min`). A dump without
    /// `seg.frac_final` has no fracture mask and therefore no fracture scene, which the caller
    /// reports as a skip rather than scoring against the whole mesh.
    pub fn of(fragment: &FragmentFixture) -> Result<Option<Self>> {
        let Some(mesh) = fragment.working()? else { return Ok(None) };
        let geom = face_geometry(&mesh.v, &mesh.f);
        let v32: Vec<Vec3f> = mesh.v.iter().map(|p| Vec3f::from_f64(*p)).collect();
        let (fracture, frac_area) = if fragment.has("seg.frac_final.npy") {
            let frac = crate::npy::read_bool(fragment.file("seg.frac_final.npy"))?;
            if frac.len() != mesh.f.len() {
                return Err(Error::fixture(
                    fragment.file("seg.frac_final.npy"),
                    "the fracture mask does not describe the dump's own mesh",
                ));
            }
            // `float(self.A[self.frac].sum())` is numpy's **pairwise** sum, and this is the
            // reference's own areas over the reference's own mask: summing it left to right would
            // put the harness's arithmetic into a number the injected `verify` row then compares.
            let area = sherd_core::fragment::segment::masked_area(&geom.areas, &frac);
            (RayScene::of_subset(&v32, &mesh.f, |i| frac[i]), area)
        } else {
            (None, 0.0)
        };
        let (watertight, n_boundary) = if fragment.has("mesh.watertight.json") {
            let path = fragment.file("mesh.watertight.json");
            let value = crate::npy::read_json(&path)?;
            let boundary = crate::npy::field_u64(&value, "n_boundary", &path)?;
            (
                crate::npy::field_bool(&value, "watertight", &path)?,
                u32::try_from(boundary).unwrap_or(u32::MAX),
            )
        } else {
            (false, u32::MAX)
        };
        Ok(Some(Self {
            scene: RayScene::of_mesh(&v32, &mesh.f),
            fracture,
            normals: geom.normals,
            frac_area,
            watertight,
            n_boundary,
        }))
    }
}

/// R §6's view of one fragment, built entirely from the dump: the reference's own samples at the
/// pair's `t`, its own breakline, its own margin and its own two meshes.
///
/// Returns `None` when the dump has no arrays at that `(t, surface_points)` or no fracture scene.
pub fn reference_surfaces<'a>(
    fragment: &FragmentFixture,
    geometry: &'a RefGeometry,
    frames: &Frames,
    t: f64,
    surface_points: u64,
) -> Result<Option<Surfaces<'a>>> {
    let (Some(dir), Some(fracture)) = (arrays_at(fragment, t, surface_points)?, &geometry.fracture)
    else {
        return Ok(None);
    };
    let file = |name: &str| dir.join(name);
    for name in ["md.Pf.npy", "md.S.npy", "md.sp.npy", "md.margin_idx.npy"] {
        if !file(name).is_file() {
            return Ok(None);
        }
    }
    let pf = npy::read_points(file("md.Pf.npy"))?;
    let s = npy::read_points(file("md.S.npy"))?;
    let sp = npy::read_indices(file("md.sp.npy"))?;
    let margin_idx = npy::read_indices(file("md.margin_idx.npy"))?;
    if s.len() != sp.len()
        || margin_idx.iter().any(|&i| (i as usize) >= s.len())
        || sp.iter().any(|&f| (f as usize) >= geometry.normals.len())
    {
        return Err(Error::fixture(
            file("md.margin_idx.npy"),
            "the dump's sample indices do not address its own mesh",
        ));
    }
    let margin_p: Vec<[f64; 3]> = margin_idx.iter().map(|&i| s[i as usize]).collect();
    let margin_n: Vec<[f64; 3]> =
        margin_idx.iter().map(|&i| geometry.normals[sp[i as usize] as usize]).collect();
    Ok(Some(Surfaces::new(
        fracture,
        geometry.scene.as_ref(),
        geometry.watertight,
        geometry.frac_area,
        pf,
        s,
        frames.p.clone(),
        frames.ns.clone(),
        margin_p,
        margin_n,
    )))
}

/// The [`RefGeometry`] of every fragment of a dump, built once and kept.
///
/// A BVH over a 150 000-face mesh costs 40 ms and a fragment of a ten-fragment collection takes
/// part in nine pairs; the fracture scene is built once too, and the reference caches it the same
/// way (`Fragment.frac_scene` is a lazily-built property).
/// The entries are behind an [`Arc`] rather than borrowed out of the map, because a pair needs
/// *both* of its fragments' geometries alive at once and the cache would otherwise hand out one
/// borrow at a time.
#[derive(Debug, Default)]
pub struct GeometryCache {
    cached: BTreeMap<String, Option<Arc<RefGeometry>>>,
}

impl GeometryCache {
    /// The geometry of one fragment, read on first use.
    pub fn get(&mut self, fragment: &FragmentFixture) -> Result<Option<Arc<RefGeometry>>> {
        if !self.cached.contains_key(&fragment.name) {
            let geometry = RefGeometry::of(fragment)?.map(Arc::new);
            self.cached.insert(fragment.name.clone(), geometry);
        }
        Ok(self.cached.get(&fragment.name).and_then(Clone::clone))
    }
}

/// The face normals of every fragment of a dump, computed once and kept.
///
/// A collection of ten fragments takes part in forty-five pairs, and recomputing `FN` from a
/// 150 000-face working mesh for each of them would dominate the stage.
#[derive(Debug, Default)]
pub struct NormalCache {
    cached: BTreeMap<String, Option<Vec<[f64; 3]>>>,
}

impl NormalCache {
    /// `FN` for a fragment, computed on first use.
    pub fn get(&mut self, fragment: &FragmentFixture) -> Result<Option<&Vec<[f64; 3]>>> {
        if !self.cached.contains_key(&fragment.name) {
            let normals = face_normals(fragment)?;
            self.cached.insert(fragment.name.clone(), normals);
        }
        Ok(self.cached.get(&fragment.name).and_then(Option::as_ref))
    }
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
