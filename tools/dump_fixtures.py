#!/usr/bin/env python3
"""Run the reference pipeline with the fixture sink on and write a parity fixture.

    tools/dump_fixtures.py INPUT_DIR OUT_DIR [--level full|slim|min] [--target-faces N] ...

`OUT_DIR` receives the stage dumps described in the design document's §10.1 (see
``docs/superpowers/notes/2026-09-06-p0-fixtures.md`` for the file-by-file inventory) plus a
``manifest.json`` holding the versions, the parameters and a SHA-256 of every file.  The pipeline's
own outputs (cache, `report.md`, placed meshes) go to ``OUT_DIR/_run``, which the manifest ignores.

The dump is deterministic: two runs over the same input produce byte-identical files, which
``--verify-determinism`` checks by running twice into two directories and diffing the hashes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args, repo=None):
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def build_params(overrides: list[str]):
    from sherd_refit.matching import Params
    kw = {}
    names = {f.name: f.type for f in fields(Params)}
    for item in overrides or []:
        k, _, v = item.partition("=")
        k = k.strip().replace("-", "_")
        if k not in names:
            raise SystemExit(f"unknown Params field: {k}")
        cur = getattr(Params(), k)
        kw[k] = type(cur)(float(v)) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v
    return Params(**kw)


def dir_size(path: str) -> int:
    total = 0
    for base, _, names in os.walk(path):
        for n in names:
            p = os.path.join(base, n)
            if os.path.isfile(p):
                total += os.path.getsize(p)
    return total


def dump(input_dir: str, out_dir: str, level: str = "full", target_faces: int = 200000,
         workers: int | None = None, params=None, preview: bool = False, meshes: bool = False,
         refine: bool = True, keep_cache: bool = False) -> dict:
    """Run the pipeline once with the sink pointed at `out_dir`; returns the manifest."""
    from sherd_refit import fixture, pipeline
    out_dir = os.path.abspath(out_dir)
    run_dir = os.path.join(out_dir, "_run")
    os.makedirs(run_dir, exist_ok=True)
    os.environ[fixture.ENV_DIR] = out_dir
    os.environ[fixture.ENV_LEVEL] = level
    t0 = time.time()
    try:
        pipeline.run(input_dir, run_dir, target_faces=target_faces, workers=workers, params=params,
                     preview=preview, refine=refine, write_meshes=meshes)
    finally:
        os.environ.pop(fixture.ENV_DIR, None)
        os.environ.pop(fixture.ENV_LEVEL, None)
    wall = time.time() - t0
    if not keep_cache:
        shutil.rmtree(os.path.join(run_dir, "cache"), ignore_errors=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import numpy
    import open3d
    import scipy
    extra = dict(
        commit=_git("rev-parse", "HEAD", repo=repo),
        dirty=bool(_git("status", "--porcelain", "--untracked-files=no", repo=repo)),
        level=level,
        python=platform.python_version(),
        numpy=numpy.__version__, scipy=scipy.__version__, open3d=open3d.__version__,
        platform=f"{platform.system()}-{platform.machine()}",
    )
    for name in ("collection", "pairs"):
        path = os.path.join(out_dir, name + ".json")
        if os.path.exists(path):
            with open(path) as f:
                extra[name] = json.load(f)
    man = fixture.write_manifest(out_dir, extra)
    print(f"{len(man['files'])} files, {dir_size(out_dir) / 1e6:.1f} MB total "
          f"({sum(v['size'] for v in man['files'].values()) / 1e6:.1f} MB of fixtures), {wall:.1f}s")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--level", default="full", choices=("full", "slim", "min"),
                    help="full: everything; slim: without the cleaned original mesh; "
                         "min: the reduced set of D 10.1 for large collections")
    ap.add_argument("--target-faces", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="override one `Params` field (repeatable)")
    ap.add_argument("--preview", action="store_true", help="also write the preview PNGs")
    ap.add_argument("--meshes", action="store_true", help="also write the placed meshes")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--keep-cache", action="store_true", help="keep OUT_DIR/_run/cache")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty OUT_DIR")
    ap.add_argument("--omp-threads", type=int, default=1, metavar="N",
                    help="OMP_NUM_THREADS for the run (default 1).  Open3D's point-to-plane ICP "
                         "sums its normal equations with an OpenMP reduction whose order depends "
                         "on the thread count, so the full-resolution refinement of a multi-threaded "
                         "run moves by ~1e-15 between runs; one thread makes the dump byte-reproducible.")
    ap.add_argument("--verify-determinism", action="store_true",
                    help="dump twice (into OUT_DIR and OUT_DIR.check) and compare every hash")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.omp_threads > 0:                       # before open3d is imported anywhere
        os.environ["OMP_NUM_THREADS"] = str(a.omp_threads)
    if os.path.isdir(a.out_dir) and os.listdir(a.out_dir):
        if not a.force:
            raise SystemExit(f"{a.out_dir} is not empty (use --force)")
        shutil.rmtree(a.out_dir)
    p = build_params(a.param)
    kw = dict(level=a.level, target_faces=a.target_faces, workers=a.workers, params=p,
              preview=a.preview, meshes=a.meshes, refine=not a.no_refine, keep_cache=a.keep_cache)
    man = dump(a.input_dir, a.out_dir, **kw)
    if a.verify_determinism:
        other = a.out_dir.rstrip("/") + ".check"
        shutil.rmtree(other, ignore_errors=True)
        man2 = dump(a.input_dir, other, **kw)
        bad = [k for k in set(man["files"]) | set(man2["files"])
               if man["files"].get(k, {}).get("sha256") != man2["files"].get(k, {}).get("sha256")]
        if bad:
            print(f"NOT deterministic: {len(bad)} file(s) differ, e.g. {bad[:5]}")
            return 1
        print(f"deterministic: {len(man['files'])} files byte-identical across two runs")
        shutil.rmtree(other, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
