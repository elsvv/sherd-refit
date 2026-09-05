#!/usr/bin/env python3
"""Score the shell/fracture segmentation against the Structure-from-Sherds++ surface ground truth.

    python tools/eval_segmentation.py CACHE_DIR [--surfaces DIR] [--thick-frac 0.3]

``CACHE_DIR`` is the ``cache`` folder of a ``sherd-refit run`` or ``sherd-refit segment`` output,
holding one ``<fragment>.npz`` per fragment.  ``--surfaces`` points at the dataset's ``Surfaces``
folder, which ships two point sets per piece, ``<piece>_Surface_0.xyz`` and ``_Surface_1.xyz``:
the inner and the outer wall of the intact sherd, in the same file coordinates as the mesh.

A working-mesh face is **shell** in the ground truth when its centroid lies within
``thick_frac * t`` of either point set, and **fracture** otherwise.  The numbers printed are for
the fracture class, area-weighted:

``precision``  of the area we call fracture, how much really is fracture
``recall``     of the area that really is fracture, how much we find
``over``       shell area wrongly called fracture, as a fraction of the fragment's area -- the
               quantity that actually hurts, because the matcher aligns whatever is labelled
               fracture and long arcs of intact shell give it a better fit than the truth.

``coverage`` is the share of the fragment's area within reach of the point sets at all; a low
value means the ground truth is too sparse to trust for that piece.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sherd_refit.fragment import Fragment      # noqa: E402


def surface_points(surfaces_dir: str, name: str) -> np.ndarray | None:
    """The two shell point sets of one piece, concatenated.  `name` is the mesh stem."""
    stem = name
    for suffix in ("_Mesh_DS", "_Mesh"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    files = sorted(glob.glob(os.path.join(surfaces_dir, f"{stem}_Surface_*.xyz")))
    if not files:
        return None
    P = [np.loadtxt(f, usecols=(0, 1, 2)) for f in files]
    P = [x.reshape(-1, 3) for x in P if x.size]
    return np.concatenate(P) if P else None


def score_fragment(fr: Fragment, S: np.ndarray, thick_frac: float) -> dict:
    d, _ = cKDTree(S).query(fr.C)
    r = thick_frac * fr.thick
    gt_shell = d < r
    gt_frac = ~gt_shell
    A = fr.A
    pred = fr.frac
    inter = float(A[pred & gt_frac].sum())
    return dict(name=fr.name, thick=fr.thick, res=fr.res, area=float(A.sum()),
                pred_frac=float(A[pred].sum() / A.sum()), gt_frac=float(A[gt_frac].sum() / A.sum()),
                precision=inter / float(A[pred].sum()) if A[pred].sum() else float("nan"),
                recall=inter / float(A[gt_frac].sum()) if A[gt_frac].sum() else float("nan"),
                over=float(A[pred & gt_shell].sum() / A.sum()),
                under=float(A[~pred & gt_frac].sum() / A.sum()),
                coverage=float(A[d < 3 * r].sum() / A.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("--surfaces", default=os.path.expanduser(
        "~/@dev/structure-from-sherds-pp/Dataset/SfS_pp/Surfaces"))
    ap.add_argument("--thick-frac", type=float, default=0.3, help="shell band half-width, in wall thicknesses")
    ap.add_argument("--quiet", action="store_true", help="only the per-pot summary line")
    a = ap.parse_args(argv)

    rows = []
    for f in sorted(glob.glob(os.path.join(a.cache_dir, "*.npz"))):
        fr = Fragment.load(f)
        S = surface_points(a.surfaces, fr.name)
        if S is None:
            print(f"{fr.name}: no surface ground truth, skipped", file=sys.stderr)
            continue
        rows.append(score_fragment(fr, S, a.thick_frac))
    if not rows:
        return 1
    if not a.quiet:
        print("%-26s %6s %6s %8s %8s %9s %7s %7s %8s" %
              ("fragment", "t", "res", "pred %", "gt %", "precision", "recall", "over %", "coverage"))
        for r in rows:
            print("%-26s %6.2f %6.3f %8.1f %8.1f %9.3f %7.3f %7.1f %8.3f" %
                  (r["name"], r["thick"], r["res"], 100 * r["pred_frac"], 100 * r["gt_frac"],
                   r["precision"], r["recall"], 100 * r["over"], r["coverage"]))
    w = np.array([r["area"] for r in rows])
    def wm(k):
        return float(np.average([r[k] for r in rows], weights=w))
    print("%-26s pieces %2d  precision %.3f  recall %.3f  over %.1f %%  pred %.1f %%  gt %.1f %%  coverage %.3f" %
          (os.path.basename(os.path.dirname(os.path.abspath(a.cache_dir))), len(rows),
           wm("precision"), wm("recall"), 100 * wm("over"), 100 * wm("pred_frac"), 100 * wm("gt_frac"),
           wm("coverage")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
