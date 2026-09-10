#!/usr/bin/env python3
"""Merge several single-vessel synthetic collections into one mixed collection.

    python tools/merge_collections.py --out DIR OBJECT=SRC_DIR [OBJECT=SRC_DIR ...]

Each ``SRC_DIR`` is an output directory of ``tools/make_synthetic.py`` -- ``fragments/*.ply``
beside a ``ground_truth.json``.  The fragments are copied into ``DIR/fragments`` under the name
``<OBJECT>_<original stem>.ply`` and one ``ground_truth.json`` is written in the format
``tools/evaluate.py`` reads, with ``object_of`` naming the vessel each fragment came from.

Why a separate file rather than a mode of the generator.  ``make_synthetic.py`` breaks *one*
watertight solid: the voxel grid, the Voronoi seeds, the contact areas and the wear are all one
vessel's.  A mixed collection is not one break, it is several breaks that arrived in the same
crate, and the only thing that has to be decided when they are put together is the naming and
the object ids.  Keeping the two apart also keeps the generator byte-reproducible: nothing here
touches a mesh, it copies bytes.

Poses.  Every vessel keeps the matrices of its own assembled frame, exactly as
``tools/stage_sfspp.py`` does for ``mixed_ABG`` and ``mixed_all``: a pose is only ever compared
with another pose of the same object (``evaluate.py`` buckets a cross-object join before it looks
at any matrix), so there is no common frame to invent and inventing one would be a claim the data
does not make.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def load(src):
    with open(os.path.join(src, "ground_truth.json")) as fh:
        return json.load(fh)


def merge(out, parts):
    """``parts`` is a list of ``(object_id, source_dir)``. Returns the merged ground truth."""
    frag_dir = os.path.join(out, "fragments")
    os.makedirs(frag_dir, exist_ok=True)
    for f in os.listdir(frag_dir):
        if f.endswith(".ply"):
            os.remove(os.path.join(frag_dir, f))
    for f in os.listdir(out):
        if f.startswith("preview_") and f.endswith(".png"):
            os.remove(os.path.join(out, f))

    object_of, fragments, adjacency, unknown, missing, objects = {}, {}, [], [], [], {}
    units, wall = None, None
    for obj, src in parts:
        gt = load(src)
        if units is None:
            units = gt.get("units")
        elif gt.get("units") != units:
            raise ValueError(f"{src}: units {gt.get('units')!r} but {units!r} elsewhere")
        if wall is None:
            wall = gt.get("wall_thickness")
        elif gt.get("wall_thickness") != wall:
            raise ValueError(f"{src}: wall_thickness {gt.get('wall_thickness')} but {wall} elsewhere")

        def rename(name, obj=obj):
            return f"{obj}_{name}"

        for name, v in sorted(gt["fragments"].items()):
            new = rename(name)
            fragments[new] = v
            object_of[new] = obj
            shutil.copyfile(os.path.join(src, "fragments", name + ".ply"),
                            os.path.join(frag_dir, new + ".ply"))
        adjacency += [sorted([rename(a), rename(b)]) for a, b in gt.get("adjacency", [])]
        unknown += [rename(n) for n in gt.get("unknown", [])]
        missing += [rename(n) for n in gt.get("missing", [])]
        objects[obj] = {k: gt[k] for k in ("source", "license", "author", "body_colour",
                                           "body_colour_rule") if k in gt}
        objects[obj]["fragments"] = len(gt["fragments"])
        objects[obj]["adjacent_pairs"] = len(gt.get("adjacency", []))
        # the generator's own previews are the only picture of this set that shows the colours a
        # fragment carries -- the binary's segmentation preview is a shell/fracture map by design
        for f in sorted(os.listdir(src)):
            if f.startswith("preview_") and f.endswith(".png"):
                shutil.copyfile(os.path.join(src, f),
                                os.path.join(out, f"preview_{obj}_{f[len('preview_'):]}"))

    merged = {"units": units, "object_of": object_of, "fragments": fragments,
              "unknown": sorted(unknown), "adjacency": sorted(adjacency),
              "missing": sorted(missing), "wall_thickness": wall, "objects": objects}
    with open(os.path.join(out, "ground_truth.json"), "w") as fh:
        json.dump(merged, fh, indent=1)
    return merged


def write_readme(out, parts, merged):
    sizes = [os.path.getsize(os.path.join(out, "fragments", f))
             for f in os.listdir(os.path.join(out, "fragments")) if f.endswith(".ply")]
    lines = [
        f"# {os.path.basename(out)}",
        "",
        "Mixed synthetic benchmark: several vessels broken separately by `tools/make_synthetic.py`",
        "and put in one crate by `tools/merge_collections.py`. Every fragment carries the source",
        "scan's own photograph on its shell and its vessel's clay-body colour on the fracture faces,",
        "so colour is measurable evidence here and not the one flat terracotta of the earlier sets.",
        "",
        "## Objects",
        "",
        "| object | source scan | fragments | adjacent pairs | clay body (sRGB) |",
        "|---|---|---|---|---|",
    ]
    for obj, _ in parts:
        o = merged["objects"][obj]
        body = o.get("body_colour")
        lines.append("| `%s` | %s | %d | %d | %s |" % (
            obj, o.get("source", "?"), o["fragments"], o["adjacent_pairs"],
            "-" if body is None else "%d, %d, %d" % tuple(body)))
    lines += [
        "",
        "## Result",
        "",
        "| | |",
        "|---|---|",
        f"| fragment files | {len(merged['fragments'])} |",
        f"| objects | {len(merged['objects'])} |",
        f"| adjacent pairs | {len(merged['adjacency'])} |",
        f"| wall thickness t | {merged['wall_thickness']} mm |",
        f"| file size | {min(sizes)/1e3:.0f} / {sorted(sizes)[len(sizes)//2]/1e3:.0f} /"
        f" {max(sizes)/1e3:.0f} kB (min/median/max) |",
        "",
        "## Ground truth",
        "",
        "`ground_truth.json` is the format `tools/evaluate.py` reads: `object_of` names the vessel a",
        "fragment came from, `fragments` gives the 4x4 that maps the coordinates stored in the PLY",
        "back into **that vessel's own** assembled frame, and `adjacency` lists the pairs that share",
        "a fracture surface. There is no frame shared between the vessels and none is needed: a join",
        "between two objects is wrong before any pose is looked at.",
        "",
        "## Pictures",
        "",
        "`preview_<object>_fragments.png` shows six of that vessel's fragments in their stored",
        "poses with the colours they carry, and `preview_<object>_assembled.png` the vessel put back",
        "together by its ground truth. They come from the per-vessel runs and are the only picture",
        "of these sets that shows colour: the binary's `preview_segmentation.png` paints shell grey",
        "and fracture red, which is what it is for.",
        "",
    ]
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("parts", nargs="+", metavar="OBJECT=SRC_DIR",
                   help="object id and the make_synthetic.py output directory it comes from")
    a = p.parse_args(argv)
    parts = []
    for spec in a.parts:
        if "=" not in spec:
            p.error(f"expected OBJECT=SRC_DIR, got {spec!r}")
        obj, src = spec.split("=", 1)
        parts.append((obj, src))
    os.makedirs(a.out, exist_ok=True)
    merged = merge(a.out, parts)
    write_readme(a.out, parts, merged)
    print("merged %d objects, %d fragments, %d adjacent pairs -> %s"
          % (len(merged["objects"]), len(merged["fragments"]), len(merged["adjacency"]), a.out))


if __name__ == "__main__":
    main()
