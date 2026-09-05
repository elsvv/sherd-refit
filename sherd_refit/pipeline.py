"""End-to-end pipeline: preprocess (parallel) -> match pairs (parallel) -> assemble -> refine -> outputs."""
from __future__ import annotations

import glob
import itertools
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from .assembly import assemble, recenter
from .fragment import MESH_EXT, Fragment, MatchData, load_mesh
from .geometry import apply_transform, sample_on_faces
from .matching import Candidate, Params, match_pair
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


def _match_workers(workers: int, n_jobs: int, threads: int | None = None) -> tuple[int, int]:
    """(processes, threads per process) for the matching stage, keeping processes x threads ~ cores.

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
    per = max(1, int(threads)) if threads else max(1, round(cores / procs))
    return procs, per


def _init_worker(level):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s [worker] %(message)s", datefmt="%H:%M:%S", force=True)


def fragment_names(files: list[str]) -> dict[str, str]:
    """Unique fragment name per file: the stem, or the full basename when stems collide (x.ply + x.obj)."""
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    out = {}
    for f, s in zip(files, stems):
        out[f] = s if stems.count(s) == 1 else os.path.basename(f).replace(".", "_")
    return out


def _preprocess_one(args):
    path, name, cache_path, target_faces = args
    if os.path.exists(cache_path):
        try:
            fr = Fragment.load(cache_path)
            if fr.cache_valid_for(path, target_faces) and fr.name == name:
                return cache_path
            log.info("%s: cache is stale (settings or file changed), recomputing", fr.name)
        except Exception as e:      # unreadable cache: recompute
            log.warning("%s: cannot read cache (%s), recomputing", cache_path, e)
    fr = Fragment.from_mesh_file(path, target_faces=target_faces, name=name)
    fr.save(cache_path)
    return cache_path


def _match_data(fr: Fragment, t: float, p: Params, surface_points: int | None = None) -> MatchData:
    return MatchData(fr, t, seed=p.seed, surface_points=p.surface_points if surface_points is None else surface_points,
                     frac_per_t2=p.frac_per_t2, min_frac_points=p.min_frac_points,
                     max_frac_points=p.max_frac_points, margin_points=p.margin_points)


def _match_one(args):
    """Match one pair.  Both `MatchData` are built at the pair's own wall thickness, `min(t_A,
    t_B)`: the thicker of the two is often a rim, and the wall is the thinner one."""
    ca, cb, params_dict, keep, n_threads = args
    p = Params(**params_dict)
    fa, fb = Fragment.load(ca), Fragment.load(cb)
    t_pair = min(fa.thick, fb.thick)
    A = _match_data(fa, t_pair, p); B = _match_data(fb, t_pair, p)
    return [c.to_json() for c in match_pair(A, B, p, keep=keep, n_threads=n_threads)]


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
    jobs = [(f, name_of[f], os.path.join(cache_dir, name_of[f] + ".npz"), target_faces) for f in files]
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
    cands: list[Candidate] = []
    if workers > 1 and pairs:
        procs, per = _match_workers(workers, len(pairs), threads)
        log.info("matching %d pairs in %d processes x %d threads", len(pairs), procs, per)
        jobs = [(cache_of[a], cache_of[b], asdict(p), keep_per_pair, per) for a, b in pairs]
        env = {k: os.environ.get(k) for k in ("SHERD_REFIT_THREADS", "OMP_NUM_THREADS")}
        # the workers thread over candidates themselves, so their ICP gets one OpenMP thread each
        os.environ["SHERD_REFIT_THREADS"] = str(per)
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            with ProcessPoolExecutor(max_workers=procs, initializer=_init_worker, initargs=(log.getEffectiveLevel(),)) as ex:
                for res in ex.map(_match_one, jobs):
                    cands += [Candidate.from_json(d) for d in res]
        finally:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    else:
        # in-process: this process's OpenMP was configured at start-up and cannot be changed any
        # more, so leave the parallelism inside Open3D's ICP where it already is
        for a, b in pairs:
            cands += [Candidate.from_json(d) for d in _match_one((cache_of[a], cache_of[b], asdict(p), keep_per_pair, 1))]
    timings["matching"] = time.time() - t0
    log.info("matching done in %.1fs: %d candidates, %d accepted", timings["matching"], len(cands), sum(c.accepted for c in cands))

    # 3. assembly
    t0 = time.time()
    md = {n: _match_data(frags[n], thick, p, surface_points=15000) for n in names}
    poses, groups, used, rejected = assemble(md, cands, p)
    timings["assembly"] = time.time() - t0

    # 4. full-resolution refinement + outputs
    meshes = {n: load_mesh(frags[n].path) for n in names} if (refine or write_meshes) else {}
    if refine and any(len(g) > 1 for g in groups):
        from .refine import refine_joins
        t0 = time.time()
        poses = refine_joins(frags, meshes, poses, groups, used, p)
        timings["refine"] = time.time() - t0
    poses = recenter(poses, md, groups)
    t0 = time.time()
    write_transforms(os.path.join(out_dir, "transforms.json"), poses, groups, thick, asdict(p))
    write_report(out_dir, [frags[n].stats() for n in names], thick, cands, poses, groups, used, rejected, timings, asdict(p))
    if write_meshes:
        write_placed_meshes(out_dir, meshes, poses, groups)
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
    jobs = [(f, name_of[f], os.path.join(cache_dir, name_of[f] + ".npz"), target_faces) for f in files]
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
