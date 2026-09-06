#!/usr/bin/env python3
"""Write the synthetic slab pair of `tests/test_synthetic.py` to a directory.

    tools/make_slab.py OUT_DIR

The two complementary halves and their ground-truth poses come from the test module itself, so
the committed fixture under ``fixtures/slab/`` and the assertions the test makes describe exactly
the same geometry.  `ground_truth.json` uses the format the synthetic generator and the SfS++
staging tool share (`matrix` maps the fragment's file coordinates into the assembled frame).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PIECES = ((-1, "pieceA", 1), (1, "pieceB", 2))


def _test_module():
    """`tests/test_synthetic.py` loaded by path (the test directory is not a package)."""
    path = os.path.join(ROOT, "tests", "test_synthetic.py")
    spec = importlib.util.spec_from_file_location("sherd_refit_test_synthetic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_slab(out_dir: str) -> dict:
    ts = _test_module()
    os.makedirs(out_dir, exist_ok=True)
    from sherd_refit.geometry import apply_transform
    gt = {"units": "arbitrary", "source": "tests/test_synthetic.py build_piece()",
          "license": "same as this repository",
          "object_of": {}, "fragments": {}, "unknown": [], "adjacency": [["pieceA", "pieceB"]],
          "thickness": float(ts.THICK), "edge": float(ts.EDGE)}
    for side, name, seed in PIECES:
        V, F, _, _ = ts.build_piece(side, seed=seed)
        T = ts.rigid(seed)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(apply_transform(T, V)),
                                      o3d.utility.Vector3iVector(F))
        path = os.path.join(out_dir, f"{name}.ply")
        if not o3d.io.write_triangle_mesh(path, m, write_ascii=False):
            raise SystemExit(f"could not write {path}")
        # the pose that maps the file's coordinates back into the assembled frame
        gt["fragments"][name] = {"matrix": np.linalg.inv(T).tolist(), "faces": int(len(F))}
        gt["object_of"][name] = "slab"
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=1, sort_keys=True)
    return gt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    gt = write_slab(a.out_dir)
    for n, d in sorted(gt["fragments"].items()):
        print(f"{n}: {d['faces']} faces -> {os.path.join(a.out_dir, n + '.ply')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
