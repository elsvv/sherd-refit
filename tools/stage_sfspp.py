#!/usr/bin/env python3
"""Stage the Structure-from-Sherds++ dataset into the sherd-refit input layout.

Creates one folder per pot (``input/sfspp/pot_<X>/``) plus a mixed collection
(``input/sfspp/mixed_all/``) holding every piece of all ten pots.  Each folder
gets a ``ground_truth.json`` in the format shared with the synthetic generator::

    {"units": "mm", "source": ..., "license": ...,
     "object_of":  {fragment_name: object_id},
     "fragments":  {fragment_name: {"matrix": [[...4x4...]]}},
     "unknown":    [fragment_name, ...],
     "adjacency":  [[name_a, name_b], ...]}

``matrix`` is the world pose: it maps the fragment's own file coordinates into
the assembled frame.  That is the dataset's own convention for
``Ground Truth/Pot_<X>_Piece_<n>_T.txt``, verified numerically (see
``--verify``): with the matrices applied forward, every pair the adjacency
graph calls neighbouring comes within 0.05 mm, while non-neighbouring pairs of
the same pot stay tens of millimetres apart.

Dataset quirks handled here:

* mesh files are zero padded (``Pot_A_Piece_01_Mesh.obj``), ground-truth files
  are not (``Pot_A_Piece_1_T.txt``), and pots C..J use a ``_Mesh_DS.obj``
  (decimated) suffix;
* several pots ship more meshes than the ground truth covers.  Those pieces
  have no ``_T.txt`` and are either an all-zero row of the adjacency matrix or
  beyond its size.  They are still staged (they belong to the collection) but
  are listed under ``unknown`` and never take part in the adjacency list.

Usage::

    python tools/stage_sfspp.py                      # stage everything
    python tools/stage_sfspp.py --pots A B C         # only some pots
    python tools/stage_sfspp.py --mode copy          # copy instead of symlink
    python tools/stage_sfspp.py --verify             # re-run the GT sanity check
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

DEFAULT_DATASET = "/Users/vaceslaveliseev/@dev/structure-from-sherds-pp/Dataset/SfS_pp"
DEFAULT_OUT = "input/sfspp"
POTS = list("ABCDEFGHIJ")

SOURCE = ("Structure-from-Sherds++ dataset (SfS_pp), Yoo, Liu, Arshad, Kim, Kim, Aloimonos, "
          "Fermuller, Joo, Kim and Hong, 'Structure-From-Sherds++: Robust Incremental 3D Reassembly "
          "of Axially Symmetric Pots from Unordered and Mixed Fragment Collections', arXiv:2502.13986, 2025. "
          "https://sj-yoo.info/sfs/ — meshes from Dataset/SfS_pp/Mesh, poses from Dataset/SfS_pp/Ground Truth.")
LICENSE = "CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International), per the Structure-from-Sherds repository LICENSE."


def mesh_files(dataset, pot):
    """Return {piece_index: path} for one pot, in piece order."""
    d = os.path.join(dataset, "Mesh")
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.startswith(f"Pot_{pot}_Piece_"):
            continue
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".obj", ".ply", ".stl", ".off"):
            continue
        idx = int(stem.split("_Piece_")[1].split("_")[0])
        out[idx] = os.path.join(d, f)
    return dict(sorted(out.items()))


def ground_truth_pose(dataset, pot, idx):
    p = os.path.join(dataset, "Ground Truth", f"Pot_{pot}_Piece_{idx}_T.txt")
    if not os.path.exists(p):
        return None
    T = np.loadtxt(p)
    if T.shape != (4, 4):
        raise ValueError(f"{p}: expected a 4x4 matrix, got {T.shape}")
    return T


def adjacency_matrix(dataset, pot):
    p = os.path.join(dataset, "Ground Truth", f"Pot_{pot}_simple_graph.txt")
    G = np.loadtxt(p)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"{p}: expected a square matrix, got {G.shape}")
    return G


def collect(dataset, pot):
    """Names, poses, unknown pieces and adjacency for one pot."""
    files = mesh_files(dataset, pot)
    G = adjacency_matrix(dataset, pot)
    names, poses, unknown = {}, {}, []
    for idx, path in files.items():
        name = os.path.splitext(os.path.basename(path))[0]
        names[idx] = name
        T = ground_truth_pose(dataset, pot, idx)
        if T is None:
            unknown.append(name)
        else:
            poses[name] = T
    adjacency = []
    n = G.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if not G[i, j]:
                continue
            a, b = names.get(i + 1), names.get(j + 1)
            if a is None or b is None:
                raise ValueError(f"Pot_{pot}: adjacency names piece {i+1}/{j+1} but no such mesh")
            if a in poses and b in poses:
                adjacency.append([a, b])
    return files, names, poses, unknown, adjacency


def write_ground_truth(path, object_of, poses, unknown, adjacency):
    data = dict(units="mm", source=SOURCE, license=LICENSE,
                object_of=object_of,
                fragments={n: dict(matrix=np.asarray(T).tolist()) for n, T in poses.items()},
                unknown=sorted(unknown),
                adjacency=[sorted(p) for p in adjacency])
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def place(src, dst, mode):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.realpath(src), dst)


def stage(dataset, out_root, pots, mode, mixed):
    mixed_files, mixed_object_of, mixed_poses, mixed_unknown, mixed_adj = [], {}, {}, [], []
    for pot in pots:
        files, names, poses, unknown, adjacency = collect(dataset, pot)
        d = os.path.join(out_root, f"pot_{pot}")
        os.makedirs(d, exist_ok=True)
        for idx, src in files.items():
            place(src, os.path.join(d, os.path.basename(src)), mode)
        object_of = {names[i]: f"Pot_{pot}" for i in files}
        write_ground_truth(os.path.join(d, "ground_truth.json"), object_of, poses, unknown, adjacency)
        print(f"pot_{pot}: {len(files)} meshes, {len(poses)} with ground truth, "
              f"{len(unknown)} unknown, {len(adjacency)} adjacent pairs -> {d}")
        mixed_files += list(files.values())
        mixed_object_of.update(object_of)
        mixed_poses.update(poses)
        mixed_unknown += unknown
        mixed_adj += adjacency
    if mixed:
        d = os.path.join(out_root, "mixed_all")
        os.makedirs(d, exist_ok=True)
        for src in mixed_files:
            place(src, os.path.join(d, os.path.basename(src)), mode)
        write_ground_truth(os.path.join(d, "ground_truth.json"), mixed_object_of, mixed_poses, mixed_unknown, mixed_adj)
        print(f"mixed_all: {len(mixed_files)} meshes, {len(mixed_poses)} with ground truth, "
              f"{len(mixed_unknown)} unknown, {len(mixed_adj)} adjacent pairs -> {d}")
    # The poses of different pots live in unrelated frames, so the mixed set's
    # ground truth is only meaningful pot by pot; the evaluator uses object_of
    # to keep the comparison inside one pot.


def verify(dataset, pots, samples=20000, seed=0):
    """Check the pose convention: forward-transformed neighbours must touch."""
    import open3d as o3d
    rng = np.random.default_rng(seed)
    print(f"{'pot':>4} {'adjacent pairs':>15} {'median min-dist':>16} {'worst':>8} | "
          f"{'non-adjacent':>13} {'median min-dist':>16} {'closest':>8}")
    for pot in pots:
        files, names, poses, unknown, adjacency = collect(dataset, pot)
        P = {}
        for idx, path in files.items():
            name = names[idx]
            if name not in poses:
                continue
            v = np.asarray(o3d.io.read_triangle_mesh(path).vertices)
            if len(v) > samples:
                v = v[rng.choice(len(v), samples, replace=False)]
            T = poses[name]
            P[name] = v @ T[:3, :3].T + T[:3, 3]
        def mindist(a, b):
            pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[a]))
            pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[b]))
            return float(np.min(np.asarray(pa.compute_point_cloud_distance(pb))))
        adj_d = [mindist(a, b) for a, b in adjacency]
        known = sorted(P)
        adj_set = {tuple(sorted(p)) for p in adjacency}
        non = [(a, b) for i, a in enumerate(known) for b in known[i + 1:] if (a, b) not in adj_set]
        non_d = [mindist(a, b) for a, b in non]
        print(f"{pot:>4} {len(adj_d):>15} {np.median(adj_d):>16.3f} {max(adj_d):>8.3f} | "
              f"{len(non_d):>13} {np.median(non_d):>16.3f} {min(non_d):>8.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="path to Dataset/SfS_pp")
    ap.add_argument("--out", default=DEFAULT_OUT, help="staging root (default: input/sfspp)")
    ap.add_argument("--pots", nargs="+", default=POTS, help="pot letters to stage")
    ap.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    ap.add_argument("--no-mixed", action="store_true", help="do not build the mixed_all collection")
    ap.add_argument("--verify", action="store_true", help="only check the ground-truth pose convention")
    a = ap.parse_args(argv)
    pots = [p.upper().replace("POT_", "") for p in a.pots]
    if not os.path.isdir(a.dataset):
        ap.error(f"dataset not found: {a.dataset}")
    if a.verify:
        verify(a.dataset, pots)
        return 0
    stage(a.dataset, a.out, pots, a.mode, not a.no_mixed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
