"""End-to-end pipeline: preprocess (parallel) -> match pairs (parallel) -> assemble -> refine -> outputs."""
from __future__ import annotations

import glob
import itertools
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np

from .assembly import assemble, recenter
from .fragment import MESH_EXT, Fragment, MatchData, match_arrays, md_params
from .geometry import apply_transform, sample_on_faces
from .matching import Candidate, Params, match_pair, screen_pair, top_partners
from .render import PALETTE, principal_views, render_views
from .report import write_placed_meshes, write_report, write_transforms

log = logging.getLogger("sherd_refit")


def find_meshes(input_dir: str) -> list[str]:
    files = []
    for ext in MESH_EXT:
        files += glob.glob(os.path.join(input_dir, f"*{ext}")) + glob.glob(os.path.join(input_dir, f"*{ext.upper()}"))
    return sorted(set(files))


# ---------------------------------------------------------------- workers (module level: picklable)

def _set_threads(workers: int):
    """Give each worker process a fair share of the cores (inherited by spawned workers)."""
    per = max(1, (os.cpu_count() or 2) // max(1, workers))
    os.environ["SHERD_REFIT_THREADS"] = str(per)
    os.environ.setdefault("OMP_NUM_THREADS", str(per))


@contextmanager
def _worker_env(per: int):
    """Give the worker processes `per` threads each and one OpenMP thread, then put the
    environment back: they thread over candidates themselves, so Open3D's own OpenMP would only
    multiply with that."""
    env = {k: os.environ.get(k) for k in ("SHERD_REFIT_THREADS", "OMP_NUM_THREADS")}
    os.environ["SHERD_REFIT_THREADS"] = str(per)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _match_workers(workers: int, n_jobs: int, threads: int | None = None,
                   screening: bool = False) -> tuple[int, int, int]:
    """(processes, threads per process, fragments per block) for the matching stage, keeping
    processes x threads ~ cores.

    The screening pass is the opposite case: its jobs are a tenth of a second each and there are
    thousands of them, so every worker takes one thread and the pairs are grouped into blocks that
    fit its `MatchData` cache.

    One process per pair leaves most of the machine idle whenever there are fewer pairs than
    cores, and even with more pairs than cores the pairs finish at very different times, so the
    tail runs on a handful of processes.  A pair parallelises well inside one process (stage 2 is
    3.1x faster on 4 threads, 5.9x on 10), so the pairs that cannot fill the machine on their own
    get several threads each.  Open3D's ICP is OpenMP-parallel as well, but it scales worse than
    threading over candidates (2.9x on 10 cores against 5.9x), so the worker processes run it
    single-threaded and spend their budget on candidates instead.
    """
    cores = os.cpu_count() or 2
    procs = max(1, min(workers, n_jobs))
    block = max(1, MD_LRU_MAX // 2)
    if screening:
        return procs, 1, block
    if threads is None and n_jobs > 4 * workers:
        # Enough pairs to fill the machine on their own.  Threading inside a pair then only
        # competes with the other processes for the same cores, so each worker takes one pair and
        # one thread.  Grouping the pairs into blocks pays for the `MatchData` cache but costs
        # load balance, and it only becomes worth it once there are many more blocks than
        # workers: pot H (55 pairs, 11 fragments, 10 blocks over 9 workers) ran 688 s blocked
        # against 505 s unblocked, because the last worker had two blocks to do and the rest none.
        return procs, 1, block if n_jobs >= 16 * workers else 1
    per = max(1, int(threads)) if threads else max(1, round(cores / procs))
    return procs, per, 1


def _init_worker(level):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s [worker] %(message)s", datefmt="%H:%M:%S", force=True)


def fragment_names(files: list[str]) -> dict[str, str]:
    """Unique fragment name per file: the stem, or the full basename when stems collide (x.ply + x.obj)."""
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    out = {}
    for f, s in zip(files, stems):
        out[f] = s if stems.count(s) == 1 else os.path.basename(f).replace(".", "_")
    return out


def _md_kw(p: Params) -> dict:
    """The `match_arrays` knobs taken from `Params` (everything except `t`)."""
    return dict(seed=p.seed, surface_points=p.surface_points, frac_per_t2=p.frac_per_t2,
                min_frac_points=p.min_frac_points, max_frac_points=p.max_frac_points,
                margin_points=p.margin_points, macro_inner=p.macro_inner, macro_outer=p.macro_outer,
                brk_voxel=p.brk_voxel)


def _preprocess_one(args):
    """Segment one fragment and store its matching arrays beside the segmentation.

    A cache whose geometry is still valid but whose matching arrays were built with other
    settings only has the arrays recomputed (a fraction of a second), not the segmentation.
    """
    path, name, cache_path, target_faces, md_kw = args
    if os.path.exists(cache_path):
        try:
            fr = Fragment.load(cache_path)
            if fr.cache_valid_for(path, target_faces) and fr.name == name:
                if fr.md is not None and fr.md["params"] == md_params(fr.thick, **md_kw):
                    return cache_path
                log.info("%s: matching arrays are stale, recomputing them", fr.name)
                fr.save(cache_path, md=match_arrays(fr, fr.thick, **md_kw))
                return cache_path
            log.info("%s: cache is stale (settings or file changed), recomputing", fr.name)
        except Exception as e:      # unreadable cache: recompute
            log.warning("%s: cannot read cache (%s), recomputing", cache_path, e)
    fr = Fragment.from_mesh_file(path, target_faces=target_faces, name=name)
    fr.save(cache_path, md=match_arrays(fr, fr.thick, **md_kw))
    return cache_path


def _match_data(fr: Fragment, t: float, p: Params, surface_points: int | None = None) -> MatchData:
    kw = _md_kw(p)
    if surface_points is not None:
        kw["surface_points"] = surface_points
    return MatchData(fr, t, **kw)


# Per-worker cache of the fragments a pair job needs.  With thousands of pairs the same fragment
# is asked for over and over, and `_pair_blocks` orders the jobs so that consecutive ones share
# most of their fragments; six entries is enough to serve a block of three against a block of
# three, and costs about 40 MB per fragment held.
MD_LRU_MAX = 6
_MD_LRU: "OrderedDict[tuple, MatchData]" = OrderedDict()


def _cached_match_data(cache_path: str, t: float, p: Params) -> MatchData:
    """`MatchData` for one fragment at one wall thickness, from the worker's LRU if it is there.

    A miss still reuses the loaded `Fragment` if the same file is in the cache under another `t`:
    the mesh, its face geometry and its raycasting scene do not depend on the thickness.
    """
    key = (cache_path, t)
    md = _MD_LRU.pop(key, None)
    if md is None:
        fr = next((m.fr for (path, _), m in _MD_LRU.items() if path == cache_path), None)
        md = _match_data(fr if fr is not None else Fragment.load(cache_path), t, p)
    _MD_LRU[key] = md
    while len(_MD_LRU) > MD_LRU_MAX:
        _MD_LRU.popitem(last=False)
    return md


def _screen_block(args):
    """Score a group of pairs with the cheap breakline screen.

    Each fragment's `MatchData` is built at its *own* wall thickness rather than at the pair's, so
    that the cached arrays always serve and a worker never rebuilds anything: the screen only
    ranks partners, and the thresholds it applies (`Scales`) still come from the pair.
    """
    jobs, params_dict, points = args
    p = Params(**params_dict)
    out = []
    for k, ca, cb, ta, tb in jobs:
        A = _cached_match_data(ca, ta, p)
        B = _cached_match_data(cb, tb, p)
        out.append((k, screen_pair(A, B, p, points=points)))
    return out


def _match_block(args):
    """Match a group of pairs in one worker.  Both `MatchData` of a pair are built at the pair's
    own wall thickness, `min(t_A, t_B)`: the thicker of the two is often a rim, and the wall is
    the thinner one.  Returns (pair index, candidates) so the caller can restore the pair order."""
    jobs, params_dict, keep, n_threads = args
    p = Params(**params_dict)
    out = []
    for k, ca, cb, t_pair in jobs:
        A = _cached_match_data(ca, t_pair, p)
        B = _cached_match_data(cb, t_pair, p)
        out.append((k, [c.to_json() for c in match_pair(A, B, p, keep=keep, n_threads=n_threads)]))
    return out


def _pair_blocks(names: list[str], pairs: list[tuple], block: int) -> list[list[int]]:
    """Pair indices grouped so that one group touches at most `2 * block` fragments.

    Building a fragment's `MatchData` costs a fraction of a second, which is nothing against a
    pair but a large share of a cheap screening pass over thousands of pairs.  Cutting the
    collection into blocks of `block` fragments and handing a worker all the pairs between two
    blocks turns `2` builds per pair into `2 * block / (block + 1)` at worst, and into none at all
    once the LRU holds the block.
    """
    blk = {n: i // block for i, n in enumerate(names)}
    groups: dict[tuple, list[int]] = {}
    for k, (a, b) in enumerate(pairs):
        key = (min(blk[a], blk[b]), max(blk[a], blk[b]))
        groups.setdefault(key, []).append(k)
    return [groups[key] for key in sorted(groups)]


# ---------------------------------------------------------------- pipeline

def run(input_dir: str, out_dir: str, target_faces: int = 200000, workers: int | None = None, params: Params | None = None,
        preview: bool = True, refine: bool = True, write_meshes: bool = True, keep_per_pair: int = 5, threads: int | None = None):
    p = params or Params()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    _set_threads(workers)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache"); os.makedirs(cache_dir, exist_ok=True)
    timings = {}
    files = find_meshes(input_dir)
    if len(files) < 2:
        raise SystemExit(f"need at least two mesh files in {input_dir}, found {len(files)}")
    log.info("%d fragments in %s; %d workers", len(files), input_dir, workers)

    # 1. preprocessing
    t0 = time.time()
    name_of = fragment_names(files)
    jobs = [(f, name_of[f], os.path.join(cache_dir, name_of[f] + ".npz"), target_faces, _md_kw(p)) for f in files]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=_init_worker, initargs=(log.getEffectiveLevel(),)) as ex:
        caches = list(ex.map(_preprocess_one, jobs))
    frags, cache_of = {}, {}
    for c in caches:
        fr = Fragment.load(c); frags[fr.name] = fr; cache_of[fr.name] = c
    names = list(frags)
    thick = float(np.median([fr.thick for fr in frags.values()]))
    for fr in frags.values():
        if abs(fr.thick / thick - 1) > 0.4:
            log.warning("%s: thickness %.2f differs from collection median %.2f by more than 40%%", fr.name, fr.thick, thick)
    timings["preprocess"] = time.time() - t0
    res = float(np.median([fr.res for fr in frags.values()]))
    log.info("preprocessing done in %.1fs; collection thickness %.2f, working-mesh edge %.3f (%.1f edges per t)",
             timings["preprocess"], thick, res, thick / max(res, 1e-9))

    # 2. pairwise matching
    t0 = time.time()
    pairs, skipped = [], []
    for a, b in itertools.combinations(names, 2):
        ratio = frags[a].thick / frags[b].thick
        if ratio > p.thick_ratio or ratio < 1 / p.thick_ratio:
            skipped.append((a, b))            # walls too different to be one object
        else:
            pairs.append((a, b))
    if skipped:
        log.info("%d pairs skipped because wall thickness differs by more than %.1fx", len(skipped), p.thick_ratio)

    # 2a. partner search: on a large collection almost every pair is between fragments that never
    # touched, and the coarse breakline stage on a capped subsample settles that in a tenth of a
    # second per pair instead of the tens of seconds a full pair job costs
    if p.screen_top_k > 0 and len(pairs) >= p.screen_min_pairs:
        t0 = time.time()
        procs, _, block = _match_workers(workers, len(pairs), 1, screening=True)
        blocks = _pair_blocks(names, pairs, block)
        jobs = [([(k, cache_of[pairs[k][0]], cache_of[pairs[k][1]],
                   frags[pairs[k][0]].thick, frags[pairs[k][1]].thick) for k in b],
                 asdict(p), p.screen_points) for b in blocks]
        score = {}
        with _worker_env(1), ProcessPoolExecutor(max_workers=procs, initializer=_init_worker,
                                                 initargs=(log.getEffectiveLevel(),)) as ex:
            for res in ex.map(_screen_block, jobs):
                for k, s in res:
                    score[pairs[k]] = s
        keep = top_partners(score, names, p.screen_top_k)
        pairs = [pr for pr in pairs if pr in keep]
        timings["screen"] = time.time() - t0
        log.info("partner search: %d pairs screened in %.1fs, %d kept (top %d per fragment, %.1fx fewer)",
                 len(score), timings["screen"], len(pairs), p.screen_top_k, len(score) / max(1, len(pairs)))
    elif len(pairs) >= p.screen_min_pairs:
        log.info("%d pairs to match; --screen-top-k K would keep only the K best-scoring partners per "
                 "fragment and cut that by several times, at a cost in true joins measured in "
                 "docs/superpowers/notes/2026-09-06-scale-pairs.md", len(pairs))
    t0 = time.time()                    # the partner search has its own entry in `timings`
    per_pair: list[list[Candidate]] = [[] for _ in pairs]
    if workers > 1 and pairs:
        procs, per, block = _match_workers(workers, len(pairs), threads)
        blocks = _pair_blocks(names, pairs, block)
        log.info("matching %d pairs in %d processes x %d threads, %d job%s of up to %d pairs",
                 len(pairs), procs, per, len(blocks), "" if len(blocks) == 1 else "s", max(len(b) for b in blocks))
        def t_of(k):
            return min(frags[pairs[k][0]].thick, frags[pairs[k][1]].thick)

        # inside a block, pairs that share a wall thickness come together: a pair is matched at
        # `min(t_A, t_B)`, so its two `MatchData` are keyed by that thickness and consecutive jobs
        # with the same one reuse the thinner fragment's outright
        jobs = [([(k, cache_of[pairs[k][0]], cache_of[pairs[k][1]], t_of(k)) for k in sorted(b, key=lambda k: (t_of(k), k))],
                 asdict(p), keep_per_pair, per) for b in blocks]
        with _worker_env(per), ProcessPoolExecutor(max_workers=procs, initializer=_init_worker,
                                                   initargs=(log.getEffectiveLevel(),)) as ex:
            for res in ex.map(_match_block, jobs):
                for k, cs in res:
                    per_pair[k] = [Candidate.from_json(d) for d in cs]
    else:
        # in-process: this process's OpenMP was configured at start-up and cannot be changed any
        # more, so leave the parallelism inside Open3D's ICP where it already is
        jobs = [(k, cache_of[a], cache_of[b], min(frags[a].thick, frags[b].thick)) for k, (a, b) in enumerate(pairs)]
        for k, cs in _match_block((jobs, asdict(p), keep_per_pair, 1)):
            per_pair[k] = [Candidate.from_json(d) for d in cs]
    cands: list[Candidate] = [c for cs in per_pair for c in cs]      # back in pair order
    timings["matching"] = time.time() - t0
    log.info("matching done in %.1fs: %d candidates, %d accepted", timings["matching"], len(cands), sum(c.accepted for c in cands))

    # 3. assembly
    t0 = time.time()
    md = {n: _match_data(frags[n], thick, p, surface_points=15000) for n in names}
    poses, groups, used, rejected = assemble(md, cands, p)
    timings["assembly"] = time.time() - t0

    # 4. full-resolution refinement + outputs
    paths = {n: frags[n].path for n in names}
    if refine and any(len(g) > 1 for g in groups):
        from .refine import refine_joins
        t0 = time.time()
        poses = refine_joins(frags, paths, poses, groups, used, p)
        timings["refine"] = time.time() - t0
    poses = recenter(poses, md, groups)
    t0 = time.time()
    write_transforms(os.path.join(out_dir, "transforms.json"), poses, groups, thick, asdict(p))
    write_report(out_dir, [frags[n].stats() for n in names], thick, cands, poses, groups, used, rejected, timings, asdict(p))
    if write_meshes:
        write_placed_meshes(out_dir, paths, poses, groups)
    if preview:
        write_previews(out_dir, frags, poses, groups)
    timings["output"] = time.time() - t0
    log.info("outputs written to %s", out_dir)
    return poses, groups, used, cands


def write_previews(out_dir, frags, poses, groups, n_points=250000):
    rng = np.random.default_rng(0)
    for k, g in enumerate(groups):
        if len(g) < 2:
            continue
        ms = []
        for i, n in enumerate(g):
            fr = frags[n]
            P, pick = sample_on_faces(fr.V, fr.F, fr.A, np.ones(len(fr.F), bool), n_points, rng)
            T = poses[n]
            ms.append((apply_transform(T, P), fr.FN[pick] @ T[:3, :3].T, np.tile(PALETTE[i % len(PALETTE)], (len(P), 1))))
        V = np.concatenate([m[0] for m in ms])
        render_views(ms, os.path.join(out_dir, f"preview_{k}.png"), principal_views(V), W=900, H=700,
                     label=" | ".join(f"{n}={['grey','orange','blue','green','yellow','purple','cyan','pink','indigo','tan'][i % 10]}" for i, n in enumerate(g)))
    # segmentation preview of every fragment (fracture in red)
    ms = []
    for i, n in enumerate(frags):
        fr = frags[n]
        P, pick = sample_on_faces(fr.V, fr.F, fr.A, np.ones(len(fr.F), bool), n_points // 2, rng)
        C = np.full((len(P), 3), 0.8); C[fr.frac[pick]] = [0.9, 0.2, 0.2]
        off = np.zeros(3); off[0] = i * 1.3 * (fr.V.max(0) - fr.V.min(0))[0]
        Pc = P - P.mean(0) + off
        ms.append((Pc, fr.FN[pick], C))
    V = np.concatenate([m[0] for m in ms])
    render_views(ms, os.path.join(out_dir, "preview_segmentation.png"), principal_views(V)[:2], W=1400, H=600, label=" ".join(frags))


def segment_only(input_dir: str, out_dir: str, target_faces: int = 200000, workers: int | None = None):
    """Preprocess and segment every fragment, write caches and the segmentation preview."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    _set_threads(workers)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache"); os.makedirs(cache_dir, exist_ok=True)
    files = find_meshes(input_dir)
    name_of = fragment_names(files)
    jobs = [(f, name_of[f], os.path.join(cache_dir, name_of[f] + ".npz"), target_faces, _md_kw(Params())) for f in files]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=_init_worker, initargs=(log.getEffectiveLevel(),)) as ex:
        caches = list(ex.map(_preprocess_one, jobs))
    frags = {}
    for c in caches:
        fr = Fragment.load(c); frags[fr.name] = fr
    for fr in frags.values():
        log.info("%s: thickness %.2f, edge %.3f (%.1f per t), fracture area %.1f%%",
                 fr.name, fr.thick, fr.res, fr.thick / max(fr.res, 1e-9), 100 * fr.fracture_area / fr.area)
    write_previews(out_dir, frags, {n: np.eye(4) for n in frags}, [[n] for n in frags])
    return frags
