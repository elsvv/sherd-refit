//! `Prepare` (A §2.2, A §3.5): the collection preprocessed into the workspace's cache — which is
//! the first stage of any run, so a run then starts at matching — and, from the same pass, what
//! the window shows of the input: a thumbnail, a display mesh and the warnings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use sherd_core::export::scene::{self, DisplayMesh};
use sherd_core::fragment::Fragment;
use sherd_core::memory::{self, Budget, MemorySemaphore};
use sherd_core::render::{self, Paint, Splat};
use sherd_core::report::FragmentStats;

use super::{Context, Failure, app_failure, io_failure};
use crate::protocol::{Event, FailKind, FragmentInfo, PrepareJob, Warning};
use crate::{atomic, snapshot};

/// `fragments/index.json`.
pub const INDEX_FILE: &str = "index.json";
/// The README's own threshold: a wall more than 40 % off the collection's median is worth a word.
pub const THICKNESS_OUTLIER: f64 = 0.4;

/// The thumbnail's pixels: one cell of A §7.3's grid, and nothing larger — 155 of them are read
/// every time the window opens.
const THUMB: (usize, usize) = (320, 240);
/// The clay a thumbnail is splatted in. Lighter than `scene.glb`'s clay material, because a
/// splat carries no light of its own and must read against the window's pale grid.
const CLAY: [f64; 3] = [0.80, 0.62, 0.46];
/// Shell, in the segmentation mesh (R §3.4).
const SHELL: [u8; 3] = [204, 204, 204];
/// Fracture, in the segmentation mesh — «Излом красным» (A §7.3).
const FRACTURE: [u8; 3] = [230, 51, 51];

/// Preprocesses the collection into `cache/` and writes what the window draws of it.
///
/// The cache is the point: A §5 starts a `Prepare` by itself whenever the input is linked or
/// found changed, so that the run the reviewer then asks for begins at matching. The three files
/// per fragment fall out of the same pass (A §3.5) and are skipped for a fragment whose scan has
/// not moved since the last `Prepare` wrote them.
pub(crate) fn prepare(job: &PrepareJob, context: &Context) -> Result<Event, Failure> {
    if let Err(already) = sherd_core::pipeline::set_threads(job.workers) {
        tracing::debug!("the thread pool is built already: {already}");
    }
    let entries = sherd_core::collection::discover_excluding(&job.input, &job.excluded)?;
    if entries.len() < 2 {
        // R §2's own words, so the window says what `sherd-refit-rs run` would have said.
        let message = format!("need at least two mesh files, found {}", entries.len());
        return Err(Failure::new(FailKind::TooFew, message));
    }
    let budget = budget(job.memory_gb);
    let fragments = sherd_core::session::load_fragments(
        &entries,
        job.target_faces,
        Some(&job.workspace.join("cache")),
        budget,
        job.seed,
        &context.watch,
    )?;

    let snapshot = snapshot::scan(&job.input, &job.excluded).map_err(|e| app_failure(&e))?;
    let stamps: BTreeMap<&str, &snapshot::FileStamp> =
        snapshot.files.iter().map(|f| (f.name.as_str(), f)).collect();
    let dir = job.workspace.join("fragments");
    std::fs::create_dir_all(&dir).map_err(|e| io_failure(&dir, &e))?;
    let index_path = dir.join(INDEX_FILE);
    let previous: BTreeMap<String, FragmentInfo> =
        atomic::read_json::<Vec<FragmentInfo>>(&index_path)
            .unwrap_or_default()
            .into_iter()
            .map(|info| (info.name.clone(), info))
            .collect();

    // The same share of the display budget by surface area that `scene.glb` gives each fragment.
    let areas: Vec<f64> =
        fragments.iter().map(|f| f.mesh.face_areas.iter().map(|&a| f64::from(a)).sum()).collect();
    let targets = scene::face_targets(&areas, scene::DEFAULT_FACES);
    let walls: Vec<f64> = fragments.iter().map(|f| f.thick).collect();
    let median = sherd_core::mesh::geometry::median(&walls);

    let semaphore = MemorySemaphore::new(budget);
    let done = AtomicUsize::new(0);
    let total = fragments.len();
    let prepared: Vec<Result<FragmentInfo, Failure>> = fragments
        .par_iter()
        .enumerate()
        .map(|(n, fragment)| {
            if context.cancel.is_cancelled() {
                return Err(sherd_core::Error::Cancelled.into());
            }
            let name = fragment.name.as_str();
            let Some(stamp) = stamps.get(name) else {
                let gone = format!("{name}: the scan left the input folder while it was read");
                return Err(Failure::new(FailKind::Input, gone));
            };
            let (glb, seg, png) = files(&dir, name);
            let stats = FragmentStats::of(fragment);
            let unmoved = previous
                .get(name)
                .filter(|p| p.size == stamp.size && p.mtime_ms == stamp.mtime_ms)
                // The same scan preprocessed at another face budget is another working mesh, and
                // `<name>.seg.glb` is drawn from the working mesh: its row says whether it moved.
                .filter(|p| p.stats == stats)
                .filter(|_| glb.is_file() && seg.is_file() && png.is_file());
            let (coloured, display_faces) = if let Some(kept) = unmoved {
                (kept.coloured, kept.display_faces)
            } else {
                write_display(fragment, &entries[n].path, targets[n], &dir, &semaphore)?
            };
            let info = FragmentInfo {
                name: name.to_owned(),
                file: stamp.file.clone(),
                size: stamp.size,
                mtime_ms: stamp.mtime_ms,
                stats,
                warnings: warnings(fragment, median),
                coloured,
                display_faces,
            };
            context.emitter.emit(&Event::FragmentReady(info.clone()));
            context.watch.advance("display", done.fetch_add(1, Ordering::Relaxed) + 1, total);
            Ok(info)
        })
        .collect();

    let mut index = Vec::with_capacity(prepared.len());
    for fragment in prepared {
        index.push(fragment?);
    }
    atomic::write_json(&index_path, &index).map_err(|e| app_failure(&e))?;
    Ok(Event::Done { counts: None, engine: None, params: None })
}

/// The memory the pass may hold: what the sheet asked for, or what the machine suggests.
fn budget(gb: Option<f64>) -> Budget {
    gb.map_or_else(Budget::default_for_machine, Budget::gigabytes)
}

/// The three files one prepared fragment has in `fragments/` (A §3.5).
fn files(dir: &Path, name: &str) -> (PathBuf, PathBuf, PathBuf) {
    (
        dir.join(format!("{name}.glb")),
        dir.join(format!("{name}.seg.glb")),
        dir.join(format!("{name}.png")),
    )
}

/// Reads the source once more (A §3.5) and writes the fragment's display mesh, its segmentation
/// mesh and its thumbnail; answers with what `index.json` has to remember about the first.
///
/// The read is under the semaphore a scan's own preprocessing was under, because it is the same
/// file at the same size and two of them at once cost what two of them cost there.
fn write_display(
    fragment: &Fragment,
    source: &Path,
    target: usize,
    dir: &Path,
    semaphore: &MemorySemaphore,
) -> Result<(bool, usize), Failure> {
    let name = fragment.name.as_str();
    let (glb, seg, png) = files(dir, name);
    let permit = semaphore.acquire(memory::scan_faces(source).map_or(0, memory::reservation));
    // `load_mesh` and not `read_mesh`: the cleaned original, which is what `scene.glb` is
    // simplified from — so a fragment looks the same in the app and in an export (A §3.5).
    let mesh = sherd_core::io::load_mesh(source)?;
    let display = scene::display_mesh(&mesh, target);
    drop(mesh);
    drop(permit);
    std::fs::write(&glb, scene::fragment_glb(name, &display)).map_err(|e| io_failure(&glb, &e))?;
    std::fs::write(&seg, scene::fragment_glb(name, &segmentation_mesh(fragment, target)))
        .map_err(|e| io_failure(&seg, &e))?;
    thumbnail(fragment, &png)?;
    Ok((display.colors.is_some(), display.faces.len()))
}

/// What A §7.3 shows in red beside a fragment, in the order it reads them.
fn warnings(fragment: &Fragment, median: f64) -> Vec<Warning> {
    let mut warnings = Vec::new();
    let thickness = fragment.thick;
    if median > 0.0 && (thickness / median - 1.0).abs() > THICKNESS_OUTLIER {
        warnings.push(Warning::ThicknessOutlier { thickness, median });
    }
    if !fragment.watertight {
        warnings.push(Warning::NotWatertight);
    }
    warnings
}

/// The thumbnail: the fragment's own surface samples (R §3.5.1), which are uniform over the
/// surface — a display mesh's vertices are not — through the renderer the engine's previews use.
fn thumbnail(fragment: &Fragment, path: &Path) -> Result<(), Failure> {
    let points = fragment.samples.surface_f64();
    let normals: Vec<[f64; 3]> = fragment
        .samples
        .sp
        .iter()
        .map(|&face| fragment.mesh.face_normals[face as usize].to_f64())
        .collect();
    let views = render::principal_views(&points);
    let splat = Splat { points, normals, paint: Paint::Uniform(CLAY) };
    render::render_views(&[splat], &views[..1], THUMB.0, THUMB.1)
        .write_png(path)
        .map_err(|e| io_failure(path, &e))
}

/// The working mesh with R §3.4's labels as vertex colours: a vertex is red when most of the
/// faces around it are fracture. Vertex colours, because a simplifier keeps vertices and makes
/// new faces.
fn segmentation_mesh(fragment: &Fragment, target: usize) -> DisplayMesh {
    let mesh = &fragment.mesh;
    let mut votes = vec![(0_u32, 0_u32); mesh.v.len()];
    for (face, label) in mesh.f.iter().zip(&fragment.labels) {
        for &v in face {
            let slot = &mut votes[v as usize];
            slot.1 += 1;
            slot.0 += u32::from(label.is_fracture());
        }
    }
    let colors =
        votes.iter().map(|&(red, all)| if 2 * red > all { FRACTURE } else { SHELL }).collect();
    let as_mesh = sherd_core::Mesh {
        v: mesh.v.iter().map(|p| p.to_f64()).collect(),
        f: mesh.f.clone(),
        colors: Some(colors),
    };
    scene::display_mesh(&as_mesh, target)
}
