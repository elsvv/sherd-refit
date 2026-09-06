"""Regenerate the IO test data under `crates/sherd-core/tests/data` (Rust plan step S2).

Two halves that have to stay in step, so one script does both:

1. **the fixtures** — a dozen small files covering every format and every variant the Rust readers
   claim to handle: binary little- and big-endian PLY, ASCII PLY with CRLF line ends and
   properties that must be skipped, polygons under the alternative property name `vertex_index`, a
   mesh that needs every cleaning pass and has two components, OBJ with vertex colours and
   `f v/vt/vn` faces, binary and ASCII STL, COFF and plain OFF, and GLB carrying `COLOR_0` as
   normalised `u8` and as `f32` under a node transform. Whatever Open3D can write is written by
   Open3D; the rest is hand-built and then read back with Open3D.
2. **the expectations** — `io_expected.json`: for every fixture and every benchmark file, what
   Open3D's reader returns raw, after `fragment.load_mesh`'s three cleaning passes and after
   `fragment.largest_component` — counts, SHA-256 of the arrays in file order, and
   order-independent statistics for the files Open3D cannot read exactly (its `fast_atof` is not
   correctly rounded, and its vertex order for OBJ, STL, OFF and GLB is Assimp's, not the file's).

Nothing in the Rust tests is a hand-written number; they read this JSON. Rerun this script after
touching a reader, and commit the diff only if you can explain it.

Usage, from the repository root with the venv active:

    OMP_NUM_THREADS=1 python tools/make_io_expectations.py

Benchmark files under `input/` that are not in this checkout are skipped with a note, and the Rust
test skips them too, so a checkout without the data still runs green.
"""

import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sherd_refit.fragment import largest_component  # noqa: E402  (the reference's own function)

DATA = "crates/sherd-core/tests/data"
OUT = os.path.join(ROOT, DATA)


def make_fixtures():
    """Writes the small committed fixtures."""
    os.makedirs(OUT, exist_ok=True)

    # A tetrahedron with four coloured vertices; small, closed, and every face non-degenerate.
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    C = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=np.uint8)

    def mesh(v=V, f=F, c=C):
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        if c is not None:
            m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(c, dtype=np.float64) / 255.0)
        return m

    # 1. binary little-endian PLY, exactly Open3D's own output layout (double xyz + uchar rgb).
    o3d.io.write_triangle_mesh(f"{OUT}/tetra_binary_le.ply", mesh(), write_ascii=False,
                               compressed=False, write_vertex_normals=False, print_progress=False)

    # 1b. the same, without colours: the writer has to reproduce this header too.
    o3d.io.write_triangle_mesh(f"{OUT}/tetra_binary_le_plain.ply", mesh(c=None), write_ascii=False,
                               compressed=False, write_vertex_normals=False, print_progress=False)

    # 2. ASCII PLY with CRLF line ends, a mid-header comment, float coordinates, an unused `alpha`
    #    property and `list uchar int` faces.
    rows = []
    for i in range(4):
        rows.append(f"{V[i,0]:.1f} {V[i,1]:.1f} {V[i,2]:.1f} {C[i,0]} {C[i,1]} {C[i,2]} 255")
    faces = [f"3 {a} {b} {c}" for a, b, c in F]
    ascii_ply = "\r\n".join([
        "ply", "format ascii 1.0", "comment made by hand for the sherd-refit S2 tests",
        "element vertex 4", "property float x", "property float y", "property float z",
        "comment the colours follow", "property uchar red", "property uchar green",
        "property uchar blue", "property uchar alpha",
        "element face 4", "property list uchar int vertex_indices", "end_header",
        *rows, *faces]) + "\r\n"
    open(f"{OUT}/tetra_ascii_crlf.ply", "w", newline="").write(ascii_ply)

    # 3. binary big-endian PLY, float xyz, no colours, `list uchar int` faces.
    hdr = ("ply\nformat binary_big_endian 1.0\nelement vertex 4\n"
           "property float x\nproperty float y\nproperty float z\n"
           "element face 4\nproperty list uchar int vertex_indices\nend_header\n")
    buf = bytearray(hdr.encode())
    for i in range(4):
        buf += struct.pack(">3f", *V[i])
    for a, b, c in F:
        buf += struct.pack(">B3i", 3, int(a), int(b), int(c))
    open(f"{OUT}/tetra_binary_be.ply", "wb").write(bytes(buf))

    # 4. ASCII PLY with polygons: a triangle, a quad and a pentagon under the alternative property
    #    name `vertex_index`, with normals in between that the reader must skip.
    pv = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[2,0,0],[3,0,0],[3,1,0],[2.5,1.5,0],[2,1,0]],
                  dtype=np.float64)
    poly = ["ply", "format ascii 1.0", "element vertex 9",
            "property float x", "property float y", "property float z",
            "property float nx", "property float ny", "property float nz",
            "element face 3", "property list uchar int vertex_index", "end_header"]
    for p in pv:
        poly.append(f"{p[0]} {p[1]} {p[2]} 0 0 1")
    poly += ["3 0 1 2", "4 0 1 2 3", "5 4 5 6 7 8"]
    open(f"{OUT}/polygons_ascii.ply", "w").write("\n".join(poly) + "\n")

    # 5. a mesh that needs every cleaning step: vertex 4 duplicates vertex 0 exactly, face 3 is
    #    degenerate after the merge, vertex 5 is unreferenced, and faces 4-5 are a second, smaller
    #    component that only shares a vertex with the first (so an edge-connected clustering must
    #    still separate them).
    mv = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,0],[9,9,9],[2,2,2],[3,2,2],[2,3,2]],
                  dtype=np.float64)
    mf = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3],[0,4,1],[6,7,8],[6,8,7]], dtype=np.int32)
    mc = np.array([[10,20,30],[40,50,60],[70,80,90],[100,110,120],[130,140,150],
                   [160,170,180],[190,200,210],[220,230,240],[250,240,230]], dtype=np.uint8)
    lines = ["ply", "format ascii 1.0", "element vertex 9",
             "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green", "property uchar blue",
             "element face 7", "property list uchar int vertex_indices", "end_header"]
    for p, c in zip(mv, mc):
        lines.append(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}")
    for f in mf:
        lines.append(f"3 {f[0]} {f[1]} {f[2]}")
    open(f"{OUT}/messy_ascii.ply", "w").write("\n".join(lines) + "\n")

    # 6. OBJ with per-vertex colours and `f v/vt/vn` face references.
    obj = ["# hand-made for the sherd-refit S2 tests", "mtllib missing.mtl"]
    for p, c in zip(V, C):
        obj.append(f"v {p[0]} {p[1]} {p[2]} {c[0]/255:.6f} {c[1]/255:.6f} {c[2]/255:.6f}")
    obj += ["vt 0 0", "vt 1 0", "vt 0 1", "vn 0 0 1", "usemtl nothing"]
    for a, b, c in F:
        obj.append(f"f {a+1}/1/1 {b+1}/2/1 {c+1}/3/1")
    open(f"{OUT}/tetra.obj", "w").write("\n".join(obj) + "\n")

    # 7. STL, binary and ASCII, written by Open3D (it needs normals for STL).
    ms = mesh(c=None); ms.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"{OUT}/tetra_binary.stl", ms, write_ascii=False, print_progress=False)
    # Open3D cannot write ASCII STL ("not supported yet"), so that one is hand-written.
    def facet(a, b, c):
        n = np.cross(b - a, c - a); n = n / np.linalg.norm(n)
        out = [f"facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}", "  outer loop"]
        for p in (a, b, c):
            out.append(f"    vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        return "\n".join(out + ["  endloop", "endfacet"])
    open(f"{OUT}/tetra_ascii.stl", "w").write(
        "solid tetra\n" + "\n".join(facet(V[a], V[b], V[c]) for a, b, c in F) + "\nendsolid tetra\n")

    # 8. OFF: Open3D writes COFF with four integer colour components.
    o3d.io.write_triangle_mesh(f"{OUT}/tetra_coff.off", mesh(), write_ascii=True, print_progress=False)
    # and a plain OFF with a comment and a quad face, which Open3D fan-triangulates.
    open(f"{OUT}/quad_comment.off", "w").write(
        "OFF\n# a comment line\n\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")

    # 9. GLB with COLOR_0, once as normalised u8 and once as f32, under a node transform.
    def glb(path, colors, comp_type, normalized):
        pos = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        idx = np.array([0,2,1, 0,1,3, 0,3,2, 1,2,3], dtype=np.uint16)
        col = np.asarray(colors)
        blobs = [pos.tobytes(), idx.tobytes(), col.tobytes()]
        offs, cur, bin_ = [], 0, b""
        for b in blobs:
            pad = (-len(bin_)) % 4
            bin_ += b"\0" * pad
            offs.append(len(bin_))
            bin_ += b
        bin_ += b"\0" * ((-len(bin_)) % 4)
        acc = [
            dict(bufferView=0, componentType=5126, count=4, type="VEC3",
                 min=pos.min(0).tolist(), max=pos.max(0).tolist()),
            dict(bufferView=1, componentType=5123, count=12, type="SCALAR"),
            dict(bufferView=2, componentType=comp_type, count=4, type="VEC4",
                 normalized=normalized),
        ]
        js = dict(asset=dict(version="2.0"), scene=0,
                  scenes=[dict(nodes=[0])],
                  nodes=[dict(mesh=0, translation=[10.0, 0.0, 0.0], scale=[2.0, 2.0, 2.0])],
                  meshes=[dict(primitives=[dict(attributes={"POSITION": 0, "COLOR_0": 2},
                                                indices=1, mode=4)])],
                  buffers=[dict(byteLength=len(bin_))],
                  bufferViews=[dict(buffer=0, byteOffset=offs[i], byteLength=len(blobs[i]))
                               for i in range(3)],
                  accessors=acc)
        jb = json.dumps(js, separators=(",", ":")).encode()
        jb += b" " * ((-len(jb)) % 4)
        out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(bin_))
        out += struct.pack("<II", len(jb), 0x4E4F534A) + jb
        out += struct.pack("<II", len(bin_), 0x004E4942) + bin_
        open(path, "wb").write(out)

    glb(f"{OUT}/tetra_color_u8.glb",
        np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255],[128,128,128,255]], dtype=np.uint8),
        5121, True)
    glb(f"{OUT}/tetra_color_f32.glb",
        np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[128/255,128/255,128/255,1]], dtype=np.float32),
        5126, False)


FIXTURES = [
    (f"{DATA}/tetra_binary_le.ply", "ply", True),
    (f"{DATA}/tetra_binary_le_plain.ply", "ply", True),
    (f"{DATA}/tetra_ascii_crlf.ply", "ply", True),
    (f"{DATA}/tetra_binary_be.ply", "ply", True),
    (f"{DATA}/polygons_ascii.ply", "ply", True),
    (f"{DATA}/messy_ascii.ply", "ply", True),
    (f"{DATA}/tetra.obj", "obj", True),
    (f"{DATA}/tetra_binary.stl", "stl", True),
    (f"{DATA}/tetra_ascii.stl", "stl", True),
    (f"{DATA}/tetra_coff.off", "off", True),
    (f"{DATA}/quad_comment.off", "off", True),
    (f"{DATA}/tetra_color_u8.glb", "glb", False),
    (f"{DATA}/tetra_color_f32.glb", "glb", False),
]

BENCH = [
    ("input/test_fragments_1/fragments/FY234007_reduced.ply", "ply", True),
    ("input/test_fragments_1/fragments/FY234021_reduced.ply", "ply", True),
    ("input/test_fragments_1/fragments/FY234094_reduced.ply", "ply", True),
    ("input/test_fragments_1/fragments/FY234104_reduced.ply", "ply", True),
    ("input/synthetic_pingsdorf_20/fragments/frag_000.ply", "ply", True),
    ("input/synthetic_pingsdorf_20/fragments/frag_003.ply", "ply", True),
    ("input/sfspp/pot_A/Pot_A_Piece_01_Mesh.obj", "obj", False),
    ("input/sfspp/pot_A/Pot_A_Piece_02_Mesh.obj", "obj", False),
    ("input/sfspp/pot_C/Pot_C_Piece_01_Mesh_DS.obj", "obj", False),
    ("input/sfspp/pot_B/Pot_B_Piece_01_Mesh.obj", "obj", False),
    ("input/source_models/049_kelch.glb", "glb", False),
    ("input/source_models/025_zylinderhalsgefaess.glb", "glb", False),
]


def to_u8(c):
    """Open3D's own float colour -> uchar conversion (measured: clamp, then round half up)."""
    return np.floor(np.clip(c * 255.0, 0.0, 255.0) + 0.5).astype(np.uint8)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(m):
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.triangles, dtype=np.int64)
    d = dict(vertices=int(len(V)), faces=int(len(F)))
    if len(V):
        d["v_sha256"] = sha(V)
        d["bbox_min"] = V.min(0).tolist()
        d["bbox_max"] = V.max(0).tolist()
        d["centroid"] = V.mean(0).tolist()
    if len(F):
        d["f_sha256"] = sha(F.astype(np.uint32))
        e0 = V[F[:, 1]] - V[F[:, 0]]
        e1 = V[F[:, 2]] - V[F[:, 0]]
        d["area"] = float(0.5 * np.linalg.norm(np.cross(e0, e1), axis=1).sum())
        d["index_sum"] = int(F.sum())
    if m.has_vertex_colors():
        C = to_u8(np.asarray(m.vertex_colors, dtype=np.float64))
        d["colors"] = True
        d["c_sha256"] = sha(C)
        d["c_mean"] = (C.astype(np.float64).mean(0)).tolist()
    else:
        d["colors"] = False
    return d


def clean(m):
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    return m


def entry(rel, fmt, exact):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"  MISSING {rel}")
        return None
    t0 = time.time()
    m = o3d.io.read_triangle_mesh(path)
    dt = time.time() - t0
    e = dict(path=rel, format=fmt, exact=exact, raw=stats(m))
    m = clean(m)
    e["clean"] = stats(m)
    e["largest_component"] = stats(largest_component(m))
    print(f"  {rel}: raw {e['raw']['vertices']}v/{e['raw']['faces']}f -> clean "
          f"{e['clean']['vertices']}v/{e['clean']['faces']}f -> lc "
          f"{e['largest_component']['vertices']}v/{e['largest_component']['faces']}f  ({dt:.3f}s)")
    return e


def dump_expectations():
    out = dict(open3d_version=o3d.__version__,
               note="generated by scratchpad/rust/S2/dump_reference.py; "
                    "colours are Open3D's float colours through its own uchar conversion",
               fixtures=[], benchmarks=[])
    print("fixtures:")
    for rel, fmt, exact in FIXTURES:
        e = entry(rel, fmt, exact)
        if e:
            out["fixtures"].append(e)
    print("benchmarks:")
    for rel, fmt, exact in BENCH:
        e = entry(rel, fmt, exact)
        if e:
            out["benchmarks"].append(e)
    p = os.path.join(ROOT, DATA, "io_expected.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", p, os.path.getsize(p), "bytes")



if __name__ == "__main__":
    make_fixtures()
    dump_expectations()
