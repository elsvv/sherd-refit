#!/usr/bin/env python3
"""Add R §11.4's meshes and R §11.5's previews to an existing parity fixture dump.

    tools/dump_outputs.py DUMP_DIR INPUT_DIR [--keep-ply DIR]

``tools/dump_fixtures.py`` runs the pipeline with ``preview=False`` and ``write_meshes=False``:
neither output is on the algorithm's critical path and both are large, so the dump carries the
poses that produce them and not the files themselves.  Step D2 needs them, because the port has to
reproduce `placed/<name>.ply` **byte for byte** and `preview_<k>.png` **pixel for pixel**, so this
tool produces them from the dump that already exists rather than by re-running an eight-hour sweep.

Nothing here re-derives anything the reference computed.  The fragments are rebuilt out of the
dump's own arrays — ``mesh.V``, ``mesh.F``, ``seg.frac_final``, ``mesh.stats.json`` — with ``FN``
and ``A`` from the reference's own ``face_geometry`` over those arrays; the poses and the groups
come from the dump's own ``outputs/transforms.json``; and the two writers called are
``sherd_refit.report.write_placed_meshes`` and the body of ``sherd_refit.pipeline.write_previews``,
transcribed here only so that the sampled indices and uniforms can be dumped on the way past.  The
generator is consumed in exactly the pipeline's order, so the samples are the ones the pipeline
would have drawn.

Written into ``DUMP_DIR/outputs``:

``placed.sha256.json``
    ``{name: {sha256, size, vertices, faces, colors, members}}`` for every ``placed/<name>.ply``
    and every ``assembly_<k>.ply``, plus a ``sources`` block holding the SHA-256 of every
    fragment's mesh *as the reference read and cleaned it*, before any pose is applied.  The meshes
    themselves are hundreds of megabytes and are not kept unless ``--keep-ply DIR`` says where to
    put them; a hash is what a byte-for-byte gate needs.  The source hashes are what say whether a
    byte-for-byte gate is *possible*: D §10.2's ``load`` row measures Open3D's OBJ reader one
    ``f32`` ULP away from the port's on pot_A, pot_B, pot_C, pot_G and pot_H (Assimp's
    ``fast_atof``), and a placed mesh cannot be byte-identical when its input is not.
``preview_<k>.png`` / ``preview_<k>.nolabel.png``
    The pipeline's own preview, and the same render with no caption.  The port compares against the
    second: the caption is PIL's font and PMC-20 gives it up.
``preview_<k>.meta.json``
    ``{width, height, label, n_points, views, members}`` — the views are ``principal_views``'
    answer, which PMC-10 makes library-defined, so the port renders at the reference's own views.
``preview_<k>.<name>.pick.npy`` / ``.u.npy`` / ``.v.npy``
    The face each sample landed on and the two uniforms behind it, **before** the ``u + v > 1``
    fold — the same form D §10.2's `samples` row already compares points through (defect D6).
``preview_index.json``
    The list of previews written, so the harness knows what to look for.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OMP_NUM_THREADS", "1")


class Frag:
    """The parts of a `Fragment` that `write_previews` reads, out of a dump."""

    def __init__(self, name, V, F, frac, thick, res):
        from sherd_refit.geometry import face_geometry
        self.name = name
        self.V, self.F, self.frac = V, F, frac
        self.FN, self.A, self.C = face_geometry(V, F)
        self.thick, self.res = thick, res


def load_fragments(dump: str) -> dict[str, Frag]:
    man = json.load(open(os.path.join(dump, "manifest.json")))
    out = {}
    for name in man["pairs"]["names"]:
        d = os.path.join(dump, "fragments", name)
        stats = json.load(open(os.path.join(d, "mesh.stats.json")))
        out[name] = Frag(name,
                         np.load(os.path.join(d, "mesh.V.npy")),
                         np.load(os.path.join(d, "mesh.F.npy")),
                         np.load(os.path.join(d, "seg.frac_final.npy")),
                         float(stats["thick"]), float(stats["res"]))
    return out, man


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_meshes(dump: str, input_dir: str, man: dict, poses: dict, groups: list, keep: str | None):
    """`write_placed_meshes` into a scratch directory, hashing what it wrote."""
    import shutil
    import tempfile
    from sherd_refit.report import write_placed_meshes

    names = man["pairs"]["names"]
    files = man["collection"]["files"] or [n + ".ply" for n in names]
    paths = {n: os.path.join(input_dir, f) for n, f in zip(names, files)}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"source mesh not found: {missing[0]}")

    scratch = keep or tempfile.mkdtemp(prefix="sherd-placed-")
    os.makedirs(scratch, exist_ok=True)
    write_placed_meshes(scratch, paths, poses, groups)

    files_index = {}
    for n in names:
        p = os.path.join(scratch, "placed", f"{n}.ply")
        files_index[f"placed/{n}.ply"] = dict(mesh_entry(p), members=[n])
    for k, g in enumerate(groups):
        if len(g) > 1:
            p = os.path.join(scratch, f"assembly_{k}.ply")
            files_index[f"assembly_{k}.ply"] = dict(mesh_entry(p), members=list(g))
    index = dict(files=files_index,
                 sources={n: source_hash(paths[n]) for n in names})
    with open(os.path.join(dump, "outputs", "placed.sha256.json"), "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)
    if keep is None:
        shutil.rmtree(scratch, ignore_errors=True)
    return index


def source_hash(path: str) -> str:
    """SHA-256 of the mesh `load_mesh` leaves behind: `V` as float64 and `F` as int32, C order.

    The one number that says whether the two implementations are even reading the same mesh, and
    therefore whether a byte-for-byte comparison of what they *write* is a statement about the
    writer.  Vertex colours are deliberately outside it: the placed file carries them and the
    comparison of the file itself covers them.
    """
    from sherd_refit.fragment import load_mesh
    m = load_mesh(path)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(m.vertices), dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(np.asarray(m.triangles), dtype=np.int32).tobytes())
    return h.hexdigest()


def mesh_entry(path: str) -> dict:
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    return dict(sha256=sha256_of(path), size=os.path.getsize(path),
                vertices=len(m.vertices), faces=len(m.triangles),
                colors=bool(len(m.vertex_colors)))


def write_previews(dump: str, frags: dict, poses: dict, groups: list, n_points: int):
    """`pipeline.write_previews`, with the samples dumped as they are drawn."""
    from sherd_refit.geometry import apply_transform, sample_on_faces
    from sherd_refit.render import PALETTE, principal_views, render_views

    out = os.path.join(dump, "outputs")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    written = []

    def dump_samples(tag, name, pick, u, v):
        np.save(os.path.join(out, f"{tag}.{name}.pick.npy"), pick.astype(np.uint32))
        np.save(os.path.join(out, f"{tag}.{name}.u.npy"), u)
        np.save(os.path.join(out, f"{tag}.{name}.v.npy"), v)

    def emit(tag, ms, views, W, H, label, meta):
        render_views(ms, os.path.join(out, f"{tag}.png"), views, W=W, H=H, label=label)
        render_views(ms, os.path.join(out, f"{tag}.nolabel.png"), views, W=W, H=H, label=None)
        meta.update(width=W, height=H, label=label,
                    views=[[list(map(float, e)), list(map(float, u))] for e, u in views])
        with open(os.path.join(out, f"{tag}.meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        written.append(tag)

    for k, g in enumerate(groups):
        if len(g) < 2:
            continue
        ms = []
        for i, n in enumerate(g):
            fr = frags[n]
            P, pick, u, v = sample_on_faces(fr.V, fr.F, fr.A, np.ones(len(fr.F), bool), n_points,
                                            rng, return_uv=True)
            dump_samples(f"preview_{k}", n, pick, u, v)
            T = poses[n]
            ms.append((apply_transform(T, P), fr.FN[pick] @ T[:3, :3].T,
                       np.tile(PALETTE[i % len(PALETTE)], (len(P), 1))))
        V = np.concatenate([m[0] for m in ms])
        label = " | ".join(
            f"{n}={['grey','orange','blue','green','yellow','purple','cyan','pink','indigo','tan'][i % 10]}"
            for i, n in enumerate(g))
        emit(f"preview_{k}", ms, principal_views(V), 900, 700, label,
             dict(members=list(g), n_points=int(n_points), kind="group", group=int(k)))

    ms = []
    for i, n in enumerate(frags):
        fr = frags[n]
        P, pick, u, v = sample_on_faces(fr.V, fr.F, fr.A, np.ones(len(fr.F), bool), n_points // 2,
                                        rng, return_uv=True)
        dump_samples("preview_segmentation", n, pick, u, v)
        C = np.full((len(P), 3), 0.8)
        C[fr.frac[pick]] = [0.9, 0.2, 0.2]
        off = np.zeros(3)
        off[0] = i * 1.3 * (fr.V.max(0) - fr.V.min(0))[0]
        ms.append((P - P.mean(0) + off, fr.FN[pick], C))
    V = np.concatenate([m[0] for m in ms])
    emit("preview_segmentation", ms, principal_views(V)[:2], 1400, 600, " ".join(frags),
         dict(members=list(frags), n_points=int(n_points // 2), kind="segmentation"))

    with open(os.path.join(out, "preview_index.json"), "w") as f:
        json.dump(written, f, indent=1)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("input_dir")
    ap.add_argument("--n-points", type=int, default=250000,
                    help="the pipeline's own preview sample count (default 250000)")
    ap.add_argument("--keep-ply", default=None,
                    help="keep the placed meshes in this directory instead of hashing and dropping them")
    ap.add_argument("--no-meshes", action="store_true", help="previews only")
    ap.add_argument("--no-previews", action="store_true", help="meshes only")
    args = ap.parse_args()

    dump = os.path.abspath(args.dump)
    tr_path = os.path.join(dump, "outputs", "transforms.json")
    if not os.path.exists(tr_path):
        raise SystemExit(f"{tr_path}: the dump carries no outputs/transforms.json")
    tr = json.load(open(tr_path))
    poses = {n: np.asarray(v["matrix"], dtype=np.float64) for n, v in tr["fragments"].items()}
    groups = tr["groups"]

    frags, man = load_fragments(dump)
    if not args.no_meshes:
        index = write_meshes(dump, args.input_dir, man, poses, groups, args.keep_ply)
        print(f"{len(index['files'])} meshes hashed into outputs/placed.sha256.json")
    if not args.no_previews:
        written = write_previews(dump, frags, poses, groups, args.n_points)
        print(f"{len(written)} previews written: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
