"""Report and output writing."""
from __future__ import annotations

import json
import os

import numpy as np
import open3d as o3d

from .matching import Candidate


def write_transforms(path, poses, groups, thickness, params):
    group_of = {n: k for k, g in enumerate(groups) for n in g}
    placed = {n for g in groups if len(g) > 1 for n in g}
    data = dict(thickness=thickness, params=params,
                fragments={n: dict(matrix=np.asarray(T).tolist(), group=group_of[n], placed=n in placed) for n, T in poses.items()},
                groups=groups)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def write_report(out_dir, frag_stats, thickness, cands, poses, groups, used, rejected, timings, params):
    by_pair = {}
    for c in cands:
        by_pair.setdefault((c.a, c.b), []).append(c)
    lines = ["# Reassembly report", ""]
    lines.append(f"Wall thickness (collection median): {thickness:.2f} units. All distances below are in units of thickness (t).")
    lines.append("")
    lines.append("## Fragments")
    lines.append("")
    lines.append("| fragment | faces (orig) | thickness | thickness/median | fracture area % | watertight | extent |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in frag_stats:
        flag = "" if abs(s["thickness"] / thickness - 1) < 0.4 else " **(differs)**"
        lines.append(f"| {s['name']} | {s['faces']} ({s['orig_faces']}) | {s['thickness']:.2f}{flag} | {s['thickness']/thickness:.2f} | "
                     f"{100*s['fracture_area_fraction']:.1f} | {s['watertight']} | {' x '.join(f'{x:.0f}' for x in s['extent'])} |")
    lines.append("")
    lines.append("## Assembly")
    lines.append("")
    for k, g in enumerate(groups):
        if len(g) > 1:
            lines.append(f"- group {k}: {', '.join(g)}")
    single = [g[0] for g in groups if len(g) == 1]
    if single:
        lines.append(f"- not assembled (no confident join): {', '.join(single)}")
    lines.append("")
    lines.append("## Joins used")
    lines.append("")
    lines.append("| A | B | score | seam (t) | tight A/B | gap (t) | contact (t²) | shell cont. | normal agr. | penetration |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in used:
        s = c.scores
        lines.append(f"| {c.a} | {c.b} | {c.score:.2f} | {s['seam']:.1f} | {s['tightA']:.2f} / {s['tightB']:.2f} | {s['gap']:.3f} | "
                     f"{s['contact']:.1f} | {s['cont']:.3f} | {s['cont_n']:.2f} | {s['pen']:.4f} |")
    if rejected:
        lines.append("")
        lines.append("## Accepted joins not used")
        lines.append("")
        for c, why in rejected:
            lines.append(f"- {c.a} – {c.b} (score {c.score:.2f}): {why}")
    lines.append("")
    lines.append("## Best candidate per pair")
    lines.append("")
    legend = "Acceptance requires tight ≥ {min_tight}, gap ≤ {max_gap}, penetration ≤ {max_pen}, seam ≥ {min_seam}, normal agreement ≥ {min_cont_n}.".format(**params)
    if params.get("early_reject_tight", 0.0) > 0:
        legend += " n/a = not computed: the candidate was rejected early (tight below {early_reject_tight}).".format(**params)
    lines.append(legend)
    lines.append("")
    lines.append("| A | B | accepted | score | seam (t) | tight A/B | gap (t) | penetration | normal agr. |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for (a, b), cs in sorted(by_pair.items()):
        c = max(cs, key=lambda c: c.score); s = c.scores
        partial = s.get("partial", 0.0) > 0
        pen_txt = "n/a" if partial else f"{s['pen']:.4f}"
        cn_txt = "n/a" if partial else f"{s['cont_n']:.2f}"
        lines.append(f"| {a} | {b} | {'yes' if c.accepted else 'no'} | {c.score:.2f} | {s['seam']:.1f} | {s['tightA']:.2f} / {s['tightB']:.2f} | "
                     f"{s['gap']:.3f} | {pen_txt} | {cn_txt} |")
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    for k, v in timings.items():
        lines.append(f"- {k}: {v:.1f} s")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(dict(thickness=thickness, fragments=frag_stats, groups=groups, params=params, timings=timings,
                       joins_used=[c.to_json() for c in used],
                       joins_rejected=[dict(**c.to_json(), reason=why) for c, why in rejected],
                       candidates=[c.to_json() for c in cands]), f, indent=1)


def write_placed_meshes(out_dir, meshes, poses, groups):
    """Write each original mesh in its placed pose and one merged mesh per group with >= 2 fragments."""
    placed_dir = os.path.join(out_dir, "placed")
    os.makedirs(placed_dir, exist_ok=True)
    files = {}
    for n, m in meshes.items():
        mm = o3d.geometry.TriangleMesh(m)
        mm.transform(poses[n])
        p = os.path.join(placed_dir, f"{n}.ply")
        o3d.io.write_triangle_mesh(p, mm, write_ascii=False, compressed=False, write_vertex_normals=False, print_progress=False)
        files[n] = p
    for k, g in enumerate(groups):
        if len(g) < 2:
            continue
        merged = o3d.geometry.TriangleMesh()
        for n in g:
            mm = o3d.geometry.TriangleMesh(meshes[n]); mm.transform(poses[n]); merged += mm
        o3d.io.write_triangle_mesh(os.path.join(out_dir, f"assembly_{k}.ply"), merged, write_ascii=False, compressed=False,
                                   write_vertex_normals=False, print_progress=False)
    return files
