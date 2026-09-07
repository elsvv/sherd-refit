//! How many of R §5.2's probe queries can be answered without the KD-tree (task E2, step 3).
//!
//! ```text
//! cargo run --release -p sherd-parity --example coarse_probe -- INPUT_DIR [MAX_PAIRS]
//! ```
//!
//! For every pair it takes the hypotheses of R §5.1 and the sixty probe points of R §5.2, moves
//! the probe by each pose, and classifies the query three ways:
//!
//! * **hit** — `PointTree::nearest_below` finds a breakline point within `sc.coarse`;
//! * **box reject** — the moved point is outside A's breakline bounding box widened by
//!   `sc.coarse`, so no point can be within it and six comparisons say so;
//! * **cell reject** — it is inside that box, but a 64³ occupancy grid over the box says the
//!   cells that could hold a neighbour are all empty.
//!
//! The first two are what `PointTree`'s own filter is built on
//! (`notes/2026-09-07-e2-tuning.md` §4). The third is measured here and **not** built: on
//! synthetic 20 the grid rejects a further 13.3 % of all queries, on pot H a further 9.2 %, and
//! at the cell counts a curve through a 64³ grid actually needs — 27 to 125 bit probes per query
//! — that does not pay for itself against a `kiddo` descent E1 measured at 99 ns.

use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use sherd_core::error::Result;
use sherd_core::fragment::Fragment;
use sherd_core::matching::pair::Pair;
use sherd_core::params::Params;

/// Cells per axis of the occupancy grid the third classification uses.
const GRID: u32 = 64;

static TOTAL: AtomicU64 = AtomicU64::new(0);
static HIT: AtomicU64 = AtomicU64::new(0);
static OUT_BOX: AtomicU64 = AtomicU64::new(0);
static OUT_CELL: AtomicU64 = AtomicU64::new(0);

/// A dense occupancy grid over a point cloud's bounding box.
struct Grid {
    lo: [f64; 3],
    hi: [f64; 3],
    cell: f64,
    dims: [usize; 3],
    bits: Vec<u64>,
}

impl Grid {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the quotient is clamped into 0..GRID before it is used as an index"
    )]
    fn of(points: &[[f64; 3]]) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for point in points {
            for ((low, high), &c) in lo.iter_mut().zip(&mut hi).zip(point) {
                *low = low.min(c);
                *high = high.max(c);
            }
        }
        let ext: [f64; 3] = std::array::from_fn(|k| hi[k] - lo[k]);
        let cell =
            (ext.iter().copied().fold(0.0_f64, f64::max) / f64::from(GRID)).max(f64::MIN_POSITIVE);
        let dims: [usize; 3] =
            std::array::from_fn(|k| ((ext[k] / cell) as usize + 1).clamp(1, GRID as usize));
        let mut grid =
            Self { lo, hi, cell, dims, bits: vec![0; dims[0] * dims[1] * dims[2] / 64 + 1] };
        for point in points {
            let c: [usize; 3] = std::array::from_fn(|k| {
                (((point[k] - grid.lo[k]) / grid.cell) as usize).min(grid.dims[k] - 1)
            });
            let at = grid.index(c);
            grid.bits[at / 64] |= 1 << (at % 64);
        }
        grid
    }

    fn index(&self, c: [usize; 3]) -> usize {
        (c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]
    }

    fn occupied(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn cells(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// True when the moved point is outside the box widened by `radius`.
    fn beyond_the_box(&self, q: &[f64; 3], radius: f64) -> bool {
        (0..3).any(|k| q[k] < self.lo[k] - radius || q[k] > self.hi[k] + radius)
    }

    /// True when every cell that could hold a neighbour within `radius` is empty.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the quotient is clamped into the grid before it is used as an index"
    )]
    fn empty_around(&self, q: &[f64; 3], radius: f64) -> bool {
        let range: [(usize, usize); 3] = std::array::from_fn(|k| {
            let last = i64::try_from(self.dims[k]).unwrap_or(i64::MAX) - 1;
            let lo = (((q[k] - radius - self.lo[k]) / self.cell) as i64 - 1).clamp(0, last);
            let hi = (((q[k] + radius - self.lo[k]) / self.cell) as i64 + 1).clamp(0, last);
            (usize::try_from(lo).unwrap_or(0), usize::try_from(hi).unwrap_or(0))
        });
        for x in range[0].0..=range[0].1 {
            for y in range[1].0..=range[1].1 {
                for z in range[2].0..=range[2].1 {
                    let at = self.index([x, y, z]);
                    if self.bits[at / 64] & (1 << (at % 64)) != 0 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn classify(name: &str, pair: &Pair<'_>, params: &Params) {
    let Some(tree) = pair.a.kd_brk.as_ref() else { return };
    let hyp = pair.hypotheses(params);
    if hyp.is_empty() {
        return;
    }
    let probe = pair.probe(params);
    let points = &pair.frames_a.p;
    let delta = pair.scales.coarse;
    let grid = Grid::of(points);
    println!(
        "pair {name}: brk {} pts, hyp {}, delta {delta:.4}, cell {:.4}, occupancy {}/{}",
        points.len(),
        hyp.len(),
        grid.cell,
        grid.occupied(),
        grid.cells()
    );

    (0..hyp.len()).into_par_iter().for_each(|h| {
        let (rot, tau) = (&hyp.r[h], &hyp.tau[h]);
        let (mut total, mut hit, mut out_box, mut out_cell) = (0_u64, 0_u64, 0_u64, 0_u64);
        for point in &probe.q {
            total += 1;
            let moved: [f64; 3] = std::array::from_fn(|r| {
                rot[(r, 0)] * point[0] + rot[(r, 1)] * point[1] + rot[(r, 2)] * point[2] + tau[r]
            });
            if grid.beyond_the_box(&moved, delta) {
                out_box += 1;
            } else if grid.empty_around(&moved, delta) {
                out_cell += 1;
            }
            if tree.nearest_below(&moved, delta).is_some() {
                hit += 1;
            }
        }
        TOTAL.fetch_add(total, Ordering::Relaxed);
        HIT.fetch_add(hit, Ordering::Relaxed);
        OUT_BOX.fetch_add(out_box, Ordering::Relaxed);
        OUT_CELL.fetch_add(out_cell, Ordering::Relaxed);
    });
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: coarse_probe INPUT_DIR [MAX_PAIRS]");
    let max_pairs: usize = args.next().map_or(usize::MAX, |s| s.parse().expect("a number"));
    let params = Params::default();
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("an input directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "ply" | "obj"))
        })
        .collect();
    files.sort();

    let mut fragments = Vec::new();
    for path in &files {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("f").to_owned();
        let (fragment, _) = Fragment::load_or_build(path, 200_000, &name, None)?;
        fragments.push(fragment);
    }

    let mut done = 0;
    'pairs: for i in 0..fragments.len() {
        for j in i + 1..fragments.len() {
            if done >= max_pairs {
                break 'pairs;
            }
            let (a, b) = (&fragments[i], &fragments[j]);
            if Pair::skipped(a, b, &params) {
                continue;
            }
            let pair = Pair::build(a, b, &params);
            if !pair.matchable() {
                continue;
            }
            classify(&format!("{}__{}", a.name, b.name), &pair, &params);
            done += 1;
        }
    }

    #[allow(clippy::cast_precision_loss, reason = "counts far under 2^53")]
    let (total, hit, out_box, out_cell) = (
        TOTAL.load(Ordering::Relaxed) as f64,
        HIT.load(Ordering::Relaxed) as f64,
        OUT_BOX.load(Ordering::Relaxed) as f64,
        OUT_CELL.load(Ordering::Relaxed) as f64,
    );
    println!("\npairs {done}, queries {total:.0}");
    println!("hits          {hit:.0}  ({:.2} %)", 100.0 * hit / total);
    println!("box rejects   {out_box:.0}  ({:.2} %)", 100.0 * out_box / total);
    println!("cell rejects  {out_cell:.0}  ({:.2} %)", 100.0 * out_cell / total);
    println!(
        "residue for the tree: {:.2} % after the box, {:.2} % after the box and the grid",
        100.0 * (total - out_box) / total,
        100.0 * (total - out_box - out_cell) / total
    );
    Ok(())
}
