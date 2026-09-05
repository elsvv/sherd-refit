#!/usr/bin/env python3
"""Break a digitized vessel into realistic fragments with ground truth.

The source mesh must be a closed shell of a real vessel (a photogrammetry or laser scan whose
interior surface was captured, so that the solid enclosed by the mesh is the clay itself).
The object is voxelized, cut by a noise-warped weighted Voronoi diagram, and each cell is turned
back into a watertight coloured mesh with worn fracture edges and a random rigid pose.

    python tools/make_synthetic.py SOURCE_MESH --out DIR --fragments N [options]

Outputs into DIR: ``fragments/*.ply``, ``ground_truth.json``, ``README.md``,
``preview_assembled.png`` and ``preview_fragments.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TERRACOTTA = np.array([0.72, 0.42, 0.29])
TAUBIN_ITERS = 6      # marching-cubes staircase removal; more than this pulls fracture faces apart


# --------------------------------------------------------------------------------------- loading

def load_solid(path):
    """Load any mesh format as one watertight trimesh (largest component, scene transforms applied)."""
    import trimesh
    obj = trimesh.load(path, force='scene')
    mesh = obj.to_geometry() if hasattr(obj, 'to_geometry') else obj
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    mesh.remove_unreferenced_vertices()
    return mesh


def wall_thickness(mesh, rng, n=20000):
    """Mode of the inward ray-hit distance, the same estimator the pipeline uses."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, np.int32))))
    normals, centres = mesh.face_normals, mesh.triangles_center
    idx = rng.choice(len(centres), min(n, len(centres)), replace=False)
    rays = np.hstack([(centres[idx] - normals[idx] * 1e-5).astype(np.float32),
                      (-normals[idx]).astype(np.float32)])
    hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    hit = hit[np.isfinite(hit)]
    if len(hit) < 100:
        return float(min(mesh.extents) / 10.0)
    counts, edges = np.histogram(hit, bins=200, range=(0.0, float(np.percentile(hit, 90))))
    k = int(counts.argmax())
    return float(0.5 * (edges[k] + edges[k + 1]))


# ---------------------------------------------------------------------------------- voxelization

def occupancy(mesh, voxel, pad=3, cache=None, verbose=True):
    """Solid voxels of the mesh interior. Returns (linear indices, dims, origin)."""
    import open3d as o3d
    lo = mesh.vertices.min(0) - pad * voxel
    hi = mesh.vertices.max(0) + pad * voxel
    dims = (np.ceil((hi - lo) / voxel).astype(np.int64) + 1)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z['dims'], dims) and np.allclose(z['origin'], lo, atol=1e-6):
            if verbose:
                print(f"  occupancy: {len(z['lin'])/1e6:.2f}M solid voxels (cached)")
            return z['lin'], dims, lo
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, np.int32))))
    nx, ny, nz = (int(d) for d in dims)
    gx, gy = np.meshgrid(lo[0] + np.arange(nx) * voxel, lo[1] + np.arange(ny) * voxel, indexing='ij')
    plane = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    slab = max(1, int(6e6 // len(plane)))
    out, t0 = [], time.time()
    for k0 in range(0, nz, slab):
        ks = np.arange(k0, min(k0 + slab, nz))
        pts = np.empty((len(plane) * len(ks), 3), np.float32)
        for i, k in enumerate(ks):
            s = slice(i * len(plane), (i + 1) * len(plane))
            pts[s, :2] = plane
            pts[s, 2] = lo[2] + k * voxel
        occ = scene.compute_occupancy(o3d.core.Tensor(pts)).numpy().reshape(len(ks), nx, ny)
        for i, k in enumerate(ks):
            xi, yi = np.nonzero(occ[i])
            if len(xi):
                out.append(((xi.astype(np.int64) * ny + yi) * nz + k))
    lin = np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)
    if verbose:
        print(f"  occupancy: {len(lin)/1e6:.2f}M solid voxels of {nx*ny*nz/1e6:.0f}M "
              f"grid {nx}x{ny}x{nz} in {time.time()-t0:.1f}s")
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        np.savez_compressed(cache, lin=lin, dims=dims, origin=lo)
    return lin, dims, lo


# ------------------------------------------------------------------------------------------ noise

def coherent_noise(rng, pts, lo, size, wavelength, octaves=3, gain=0.5):
    """Smooth multi-octave value noise with unit RMS, sampled at ``pts`` (world units)."""
    from scipy.ndimage import map_coordinates
    total = np.zeros(len(pts), np.float32)
    amp, wl = 1.0, float(wavelength)
    for _ in range(octaves):
        n = np.maximum(np.ceil(size / wl).astype(np.int64) + 4, 5)
        grid = rng.standard_normal(tuple(int(v) for v in n)).astype(np.float32)
        layer = np.empty(len(pts), np.float32)
        for a in range(0, len(pts), 4_000_000):
            b = min(a + 4_000_000, len(pts))
            coords = ((pts[a:b] - lo) / wl + 1.5).T.astype(np.float64)
            layer[a:b] = map_coordinates(grid, coords, order=3, mode='nearest').astype(np.float32)
        total += amp * layer
        amp *= gain
        wl *= 0.5
    s = float(total.std())
    return total / s if s > 1e-9 else total


# -------------------------------------------------------------------------------------- fracture

def voronoi_labels(pts, seeds, weights, chunk=200_000):
    """Multiplicatively weighted nearest-seed assignment: argmin_i w_i * |x - s_i|."""
    w2 = (weights ** 2).astype(np.float32)
    s = seeds.astype(np.float32)
    s2 = (s ** 2).sum(1).astype(np.float32)
    labels = np.empty(len(pts), np.int16)
    for a in range(0, len(pts), chunk):
        b = min(a + chunk, len(pts))
        x = pts[a:b]
        d2 = (x ** 2).sum(1)[:, None] - 2.0 * (x @ s.T) + s2[None, :]
        np.multiply(d2, w2[None, :], out=d2)
        labels[a:b] = d2.argmin(1)
    return labels


def blue_noise_seeds(coords, n, rng, r_vox):
    """Greedy dart throwing over solid voxels: n seeds with a soft minimum spacing ``r_vox``."""
    from scipy.spatial import cKDTree
    cand = coords[rng.choice(len(coords), min(len(coords), 60 * n), replace=False)].astype(np.float64)
    r = float(r_vox)
    chosen = [cand[0]]
    tree = None
    for p in cand[1:]:
        if len(chosen) >= n:
            break
        if tree is None or len(chosen) % 16 == 0:
            tree = cKDTree(np.asarray(chosen))
        if tree.query(p)[0] >= r:
            chosen.append(p)
    while len(chosen) < n:                                        # top up if spacing was too strict
        chosen.append(cand[rng.integers(len(cand))])
    return np.asarray(chosen[:n])


def break_object(mesh, n_fragments, wall, voxel, seed, wear, cache=None, verbose=True):
    """Voxelize and cut into ``n_fragments`` cells. Returns a dict with the label grid and geometry."""
    rng = np.random.default_rng(seed)
    lin, dims, origin = occupancy(mesh, voxel, cache=cache, verbose=verbose)
    nx, ny, nz = (int(d) for d in dims)
    coords = np.empty((len(lin), 3), np.int32)
    coords[:, 0] = lin // (ny * nz)
    coords[:, 1] = (lin // nz) % ny
    coords[:, 2] = lin % nz
    world = (coords.astype(np.float32) * voxel + origin.astype(np.float32))

    # cell size drives seed spacing and how much warping the diagram tolerates before cells split
    solid_volume = len(lin) * voxel ** 3
    cell = float((2.0 * solid_volume / wall / max(n_fragments, 1)) ** 0.5)   # mean cell width (mm)

    seeds_vox = blue_noise_seeds(coords, n_fragments, rng, 0.62 * cell / voxel)
    seeds = seeds_vox.astype(np.float32) * voxel + origin.astype(np.float32)
    weights = np.ones(len(seeds), np.float32)
    big = rng.random(len(seeds)) < 0.15                       # 15 % of cells claim ~3x the area
    weights[big] = 1.0 / np.sqrt(3.0)

    amp_big = min(1.5 * wall, 0.22 * cell)
    wl_big = max(3.5 * wall, 0.9 * cell)
    size = np.asarray(dims, float) * voxel
    warp = world.copy()
    for axis in range(3):
        warp[:, axis] += amp_big * coherent_noise(rng, world, origin, size, wl_big, octaves=2)
        warp[:, axis] += 0.30 * wall * coherent_noise(rng, world, origin, size, 1.2 * wall, octaves=2)
    if verbose:
        print(f"  cells: {n_fragments} seeds, mean width {cell:.1f} mm, "
              f"warp {amp_big:.1f} mm @ {wl_big:.0f} mm + {0.30*wall:.1f} mm @ {1.2*wall:.0f} mm")

    labels = voronoi_labels(warp, seeds, weights)
    del warp

    grid = np.full((nx, ny, nz), -1, np.int16)
    grid.reshape(-1)[lin] = labels
    return dict(grid=grid, coords=coords, labels=labels, origin=origin, voxel=voxel, dims=dims,
                wall=wall, rng=rng, wear=wear, n_seeds=n_fragments)


def contact_areas(grid, voxel):
    """Shared 6-connected voxel-face area between every pair of labels, in mm^2."""
    counts = {}
    for axis in range(3):
        a_sl = [slice(None)] * 3
        b_sl = [slice(None)] * 3
        a_sl[axis], b_sl[axis] = slice(0, -1), slice(1, None)
        step = 64 if axis != 0 else 64
        n0 = grid.shape[0]
        for i0 in range(0, n0, step):
            i1 = min(i0 + step + (1 if axis == 0 else 0), n0)
            blk = grid[i0:i1]
            a = blk[tuple(a_sl)] if axis != 0 else blk[:-1]
            b = blk[tuple(b_sl)] if axis != 0 else blk[1:]
            m = (a >= 0) & (b >= 0) & (a != b)
            if not m.any():
                continue
            u, v = a[m].astype(np.int64), b[m].astype(np.int64)
            lo_, hi_ = np.minimum(u, v), np.maximum(u, v)
            key, c = np.unique(lo_ * 100000 + hi_, return_counts=True)
            for k, cc in zip(key.tolist(), c.tolist()):
                counts[k] = counts.get(k, 0) + cc
    return {(k // 100000, k % 100000): c * voxel * voxel for k, c in counts.items()}


# ------------------------------------------------------------------------------ mesh per fragment

def extract_fragment(state, label, edge_target, smooth_sigma=0.8):
    """Turn one Voronoi cell into a watertight coloured open3d mesh in the assembled frame."""
    import open3d as o3d
    from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label as cc_label
    from skimage.measure import marching_cubes

    grid, voxel, wall = state['grid'], state['voxel'], state['wall']
    sel = state['coords'][state['labels'] == label]
    if len(sel) < 200:
        return None
    pad = 4
    lo_i = np.maximum(sel.min(0) - pad, 0)
    hi_i = np.minimum(sel.max(0) + pad + 1, np.asarray(grid.shape))
    sub = grid[lo_i[0]:hi_i[0], lo_i[1]:hi_i[1], lo_i[2]:hi_i[2]]
    frag = sub == label
    if not frag.any():
        return None

    # keep the largest connected component (the warp can shed a few crumbs)
    cc, ncc = cc_label(frag)
    if ncc > 1:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        frag = cc == sizes.argmax()

    other = (sub >= 0) & ~frag
    air = sub < 0
    struct = np.zeros((3, 3, 3), bool)
    struct[1, 1, :] = struct[1, :, 1] = struct[:, 1, 1] = True

    keep = frag
    d_other = distance_transform_edt(~other).astype(np.float32) * voxel if other.any() else None
    wear_i = float(state['rng'].uniform(0.0, state['wear']))
    if wear_i > 1e-6 and d_other is not None:
        crease = binary_dilation(other, struct) & binary_dilation(air, struct) & frag
        if crease.any():
            d_crease = distance_transform_edt(~crease).astype(np.float32) * voxel
        else:
            d_crease = np.full(frag.shape, 1e6, np.float32)
        depth = wear_i * wall * (0.01 + 0.99 * np.exp(-d_crease / (0.06 * wall)))
        keep = frag & (d_other > depth)
        cc, ncc = cc_label(keep)
        if ncc > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            keep = cc == sizes.argmax()
        if keep.sum() < 0.3 * frag.sum():
            keep = frag

    field = np.pad(keep.astype(np.float32), 2, mode='constant')
    field = gaussian_filter(field, smooth_sigma)
    if field.max() <= 0.5 or field.min() >= 0.5:
        return None
    verts, faces, _, _ = marching_cubes(field, level=0.5, spacing=(voxel, voxel, voxel))
    verts = verts + (lo_i - 2).astype(np.float64) * voxel + state['origin']

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(faces.astype(np.int32)))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if _signed_volume(mesh) < 0:
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
    if TAUBIN_ITERS:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERS)

    tri = np.asarray(mesh.triangles)
    v = np.asarray(mesh.vertices)
    med_edge = float(np.median(np.linalg.norm(v[tri[:, 0]] - v[tri[:, 1]], axis=1)))
    if med_edge < 0.75 * edge_target and len(tri) > 5000:
        target = max(2000, int(len(tri) * (med_edge / edge_target) ** 2))
        simple = mesh.simplify_quadric_decimation(target)
        if is_watertight(simple) or not is_watertight(mesh):
            mesh = simple
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, lo_i, d_other


def is_watertight(mesh):
    """Every undirected edge used by exactly two triangles.

    Open3D's own ``is_watertight`` takes minutes on meshes of this size, so this does the same
    edge-manifold test in numpy; marching cubes already guarantees consistent orientation.
    """
    t = np.asarray(mesh.triangles)
    if len(t) == 0:
        return False
    n = int(np.asarray(mesh.vertices).shape[0])
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e.sort(axis=1)
    _, c = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1], return_counts=True)
    return bool((c == 2).all())


def _signed_volume(mesh):
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def colour_fragment(mesh, state, lo_i, d_other, rng):
    """Terracotta with low-frequency mottling; fracture faces a touch lighter."""
    from scipy.ndimage import map_coordinates
    v = np.asarray(mesh.vertices)
    size = np.asarray(state['dims'], float) * state['voxel']
    mott = coherent_noise(rng, v.astype(np.float32), state['origin'], size,
                          12.0 * state['wall'], octaves=2)
    base = TERRACOTTA[None, :] * (1.0 + 0.10 * np.clip(mott, -2.5, 2.5)[:, None])
    if d_other is not None:
        idx = ((v - state['origin']) / state['voxel'] - lo_i).T.astype(np.float64)
        dv = map_coordinates(d_other, idx, order=1, mode='nearest')
        base[dv < 1.6 * state['voxel']] *= 1.04
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(base, 0, 1))
    return mesh


# ------------------------------------------------------------------------------------------ poses

def random_pose(rng, mesh, spread):
    """Move the mesh to a random pose. Returns the 4x4 that maps file coords back to assembled."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(_rand_quat(rng)).as_matrix()
    v = np.asarray(mesh.vertices)
    c = 0.5 * (v.min(0) + v.max(0))
    t = rng.uniform(-spread, spread, 3)
    mesh.vertices = o3d.utility.Vector3dVector((v - c) @ R.T + t)
    mesh.compute_vertex_normals()
    gt = np.eye(4)
    gt[:3, :3] = R.T
    gt[:3, 3] = c - R.T @ t
    return gt


def _rand_quat(rng):
    u1, u2, u3 = rng.random(3)
    return np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                     np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])


# ---------------------------------------------------------------------------------------- preview

def preview(paths_gt, out_png, voxel_stride=6, label=None):
    from sherd_refit.render import render_views, PALETTE
    pts, nrm, col = [], [], []
    for i, (mesh, gt) in enumerate(paths_gt):
        v = np.asarray(mesh.vertices)
        n = np.asarray(mesh.vertex_normals)
        v = v @ gt[:3, :3].T + gt[:3, 3]
        n = n @ gt[:3, :3].T
        s = slice(None, None, voxel_stride)
        pts.append(v[s]); nrm.append(n[s])
        col.append(np.tile(PALETTE[i % len(PALETTE)], (len(v[s]), 1)))
    render_views([(np.concatenate(pts), np.concatenate(nrm), np.concatenate(col))], out_png,
                 views=[([0, 0, 1], [0, 1, 0]), ([1, 0, 0], [0, 1, 0]), ([0.6, 0.8, 0.4], [0, 1, 0])],
                 W=640, H=560, label=label)


def preview_singles(meshes, out_png, label=None):
    from sherd_refit.render import render_views
    import PIL.Image as Image
    tiles = []
    tmp = out_png + '.tmp.png'
    for i, m in enumerate(meshes):
        v = np.asarray(m.vertices); n = np.asarray(m.vertex_normals)
        c = np.asarray(m.vertex_colors) if len(m.vertex_colors) else np.tile(TERRACOTTA, (len(v), 1))
        s = slice(None, None, max(1, len(v) // 60000))
        render_views([(v[s], n[s], c[s])], tmp, views=[([0, 0, 1], [0, 1, 0]), ([0, 0, -1], [0, 1, 0])],
                     W=380, H=340, label=f'{i}')
        tiles.append(np.asarray(Image.open(tmp)))
    if os.path.exists(tmp):
        os.remove(tmp)
    rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    im = Image.fromarray(np.concatenate(rows, 0))
    if label:
        import PIL.ImageDraw as ImageDraw
        ImageDraw.Draw(im).text((10, 10), label, fill=(255, 255, 255))
    im.save(out_png)


# ------------------------------------------------------------------------------------------- main

def build(args):
    global o3d
    import open3d as o3d
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    mesh = load_solid(args.source)
    t0 = wall_thickness(mesh, np.random.default_rng(0))
    scale = args.wall / t0
    mesh.apply_scale(scale)
    print(f"source {os.path.basename(args.source)}: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"t={t0:.4f} -> scaled x{scale:.2f} to t={args.wall:.1f} mm, extent {np.round(mesh.extents,1)} mm")
    if not mesh.is_watertight:
        print("  WARNING: source is not watertight; the occupancy test may leak")

    cache = os.path.join(os.path.dirname(os.path.abspath(args.source)),
                         '.voxcache_%s_w%.2f_v%.2f.npz' % (
                             os.path.splitext(os.path.basename(args.source))[0], args.wall, args.voxel))
    state = break_object(mesh, args.fragments, args.wall, args.voxel, args.seed, args.wear,
                         cache=cache, verbose=True)

    print("  contact areas ...")
    contacts = contact_areas(state['grid'], args.voxel)

    frag_dir = os.path.join(args.out, 'fragments')
    os.makedirs(frag_dir, exist_ok=True)
    for f in os.listdir(frag_dir):
        if f.endswith('.ply'):
            os.remove(os.path.join(frag_dir, f))

    present, meshes, gts, stats = [], {}, {}, {}
    min_area = 1.5 * args.wall ** 2
    t_extract = time.time()
    for lab in range(state['n_seeds']):
        r = extract_fragment(state, lab, args.edge)
        if r is None:
            continue
        m, lo_i, d_other = r
        area = m.get_surface_area()
        if area < min_area or len(m.triangles) < 200:
            continue
        m = colour_fragment(m, state, lo_i, d_other, rng)
        wt = is_watertight(m)
        meshes[lab] = m
        present.append(lab)
        stats[lab] = dict(faces=len(m.triangles), verts=len(m.vertices), area=float(area), watertight=wt,
                          extent=np.ptp(np.asarray(m.vertices), axis=0).tolist())
        if len(present) % 10 == 0:
            print(f"    {len(present)} fragments extracted ({time.time()-t_extract:.0f}s)")
    print(f"  extracted {len(present)} fragments in {time.time()-t_extract:.0f}s")

    n_missing = int(round(args.missing * len(present)))
    missing = sorted(rng.choice(present, n_missing, replace=False).tolist()) if n_missing else []
    kept = [l for l in present if l not in set(missing)]

    names = {l: 'frag_%03d' % i for i, l in enumerate(sorted(present))}
    gt_frag, object_of, fragments_out = {}, {}, []
    for l in sorted(kept):
        m = meshes[l]
        gt = random_pose(rng, m, args.spread)
        o3d.io.write_triangle_mesh(os.path.join(frag_dir, names[l] + '.ply'), m, write_ascii=False)
        gt_frag[names[l]] = {'matrix': gt.tolist()}
        object_of[names[l]] = 'main'
        gts[names[l]] = gt
        fragments_out.append((names[l], m, gt))

    intruder_info = None
    if args.intruders > 0 and args.intruder_mesh:
        intruder_info = add_intruders(args, rng, gt_frag, object_of, gts, fragments_out, frag_dir,
                                      float(mesh.volume))

    adjacency = []
    keep_set = set(kept)
    thresh = 0.5 * args.wall ** 2
    for (a, b), area in sorted(contacts.items()):
        if a in keep_set and b in keep_set and area >= thresh:
            adjacency.append([names[a], names[b]])

    gt = {'units': 'mm', 'source': args.source_url, 'license': args.license, 'author': args.author,
          'object_of': object_of, 'fragments': gt_frag, 'unknown': [],
          'adjacency': adjacency, 'missing': [names[l] for l in missing],
          'wall_thickness': float(args.wall)}
    with open(os.path.join(args.out, 'ground_truth.json'), 'w') as fh:
        json.dump(gt, fh, indent=1)

    elapsed = time.time() - t_start
    write_readme(args, gt, stats, names, kept, missing, contacts, elapsed, scale, mesh, intruder_info)

    if not args.no_preview:
        print("  previews ...")
        preview([(m, g) for _, m, g in fragments_out], os.path.join(args.out, 'preview_assembled.png'),
                label=f'{os.path.basename(args.out)}: {len(fragments_out)} fragments, assembled by ground truth')
        pick = [m for _, m, _ in fragments_out[:6]]
        preview_singles(pick, os.path.join(args.out, 'preview_fragments.png'),
                        label='six fragments in their stored (randomised) poses')

    n_wt = sum(1 for s in stats.values() if s['watertight'])
    print(f"done: {len(gt_frag)} files, watertight {n_wt}/{len(stats)}, "
          f"{len(adjacency)} adjacent pairs, {len(missing)} missing, {elapsed:.0f}s -> {args.out}")
    return gt


def add_intruders(args, rng, gt_frag, object_of, gts, fragments_out, frag_dir, main_volume):
    im = load_solid(args.intruder_mesh)
    t0 = wall_thickness(im, np.random.default_rng(0))
    im.apply_scale(args.wall / t0)
    cache = os.path.join(os.path.dirname(os.path.abspath(args.intruder_mesh)),
                         '.voxcache_%s_w%.2f_v%.2f.npz' % (
                             os.path.splitext(os.path.basename(args.intruder_mesh))[0], args.wall, args.voxel))
    # break the other vessel into pieces of the same size as the main ones, or the intruders would
    # be recognisable by size alone
    n_cut = max(args.intruders * 2, int(round(args.fragments * float(im.volume) / max(main_volume, 1e-9))))
    print(f"  intruders: cutting {args.intruders} of {n_cut} pieces from "
          f"{os.path.basename(args.intruder_mesh)}")
    st = break_object(im, n_cut, args.wall, args.voxel, args.seed + 1, args.wear, cache=cache, verbose=True)
    made = 0
    for lab in rng.permutation(st['n_seeds']):
        if made >= args.intruders:
            break
        r = extract_fragment(st, int(lab), args.edge)
        if r is None:
            continue
        m, lo_i, d_other = r
        if m.get_surface_area() < 1.5 * args.wall ** 2:
            continue
        m = colour_fragment(m, st, lo_i, d_other, rng)
        name = 'intruder_%02d' % made
        g = random_pose(rng, m, args.spread)
        o3d.io.write_triangle_mesh(os.path.join(frag_dir, name + '.ply'), m, write_ascii=False)
        gt_frag[name] = {'matrix': g.tolist()}
        object_of[name] = 'intruder'
        gts[name] = g
        fragments_out.append((name, m, g))
        made += 1
    return dict(source=args.intruder_mesh, count=made)


def write_readme(args, gt, stats, names, kept, missing, contacts, elapsed, scale, mesh, intruder_info):
    areas = np.array([stats[l]['area'] for l in kept]) if kept else np.zeros(1)
    faces = np.array([stats[l]['faces'] for l in kept]) if kept else np.zeros(1)
    n_wt = sum(1 for l in kept if stats[l]['watertight'])
    sizes = [os.path.getsize(os.path.join(args.out, 'fragments', f))
             for f in os.listdir(os.path.join(args.out, 'fragments')) if f.endswith('.ply')]
    lines = [
        f"# {os.path.basename(args.out)}",
        "",
        "Synthetic benchmark generated by `tools/make_synthetic.py` from a scan of a real vessel.",
        "",
        "## Source object",
        "",
        f"- file: `{args.source}`",
        f"- url: {args.source_url}",
        f"- author: {args.author}",
        f"- license: {args.license}",
        f"- scaled x{scale:.3f} so the wall is {args.wall:.1f} mm; "
        f"extent after scaling {np.round(mesh.extents, 1).tolist()} mm",
        "",
        "## Parameters",
        "",
        "| parameter | value |",
        "|---|---|",
        f"| fragments requested | {args.fragments} |",
        f"| seed | {args.seed} |",
        f"| wall thickness t | {args.wall} mm |",
        f"| voxel | {args.voxel} mm ({args.wall/args.voxel:.0f} voxels across the wall) |",
        f"| target edge length | {args.edge} mm |",
        f"| wear | {args.wear} (per-fragment strength drawn from U(0, wear)) |",
        f"| missing fraction | {args.missing} |",
        f"| intruders | {args.intruders}"
        + (f" from `{intruder_info['source']}`" if intruder_info else "") + " |",
        f"| pose spread | +/-{args.spread:.0f} mm |",
        "",
        "## Result",
        "",
        "| | |",
        "|---|---|",
        f"| fragment files | {len(gt['fragments'])} |",
        f"| of which intruders | {sum(1 for v in gt['object_of'].values() if v == 'intruder')} |",
        f"| removed as missing | {len(missing)} |",
        f"| watertight | {n_wt}/{len(kept)} |",
        f"| adjacent pairs (contact >= 0.5 t^2) | {len(gt['adjacency'])} |",
        f"| surface area per fragment | {areas.min():.0f} / {np.median(areas):.0f} / {areas.max():.0f} mm^2"
        f" (min/median/max) = {areas.min()/args.wall**2:.1f} / {np.median(areas)/args.wall**2:.1f}"
        f" / {areas.max()/args.wall**2:.1f} t^2 |",
        f"| faces per fragment | {faces.min():.0f} / {np.median(faces):.0f} / {faces.max():.0f} |",
        f"| file size | {min(sizes)/1e3:.0f} / {np.median(sizes)/1e3:.0f} / {max(sizes)/1e3:.0f} kB |",
        f"| generation time | {elapsed:.0f} s |",
        "",
        "## Ground truth",
        "",
        "`ground_truth.json` gives, per fragment, the 4x4 matrix that maps the coordinates stored in the",
        "PLY file back into the assembled frame. `adjacency` lists the pairs that share a fracture surface",
        "of at least 0.5 t^2, measured on the voxel labelling before wear. `missing` names the fragments",
        "that were generated and then withheld. Intruders come from a different vessel; their matrices are",
        "relative to that vessel's own assembled frame and are meaningless for this assembly.",
        "",
        "## Method",
        "",
        "The scan is a closed shell whose interior is the clay, so voxelizing its interior gives the vessel",
        "wall directly. Fragments are the cells of a multiplicatively weighted Voronoi diagram (15 % of the",
        "seeds are weighted to claim about three times the area) evaluated on positions displaced by a coherent",
        "noise field, which makes the fracture lines wavy at the scale of a few wall thicknesses and bumpy at",
        "the scale of the wall. Wear erodes each cell near the crease where its fracture meets the shell,",
        "with a per-fragment strength drawn uniformly from 0 to `--wear` x t and an exponential falloff of",
        "0.06 t into the fracture face, so the arris is chipped while the fracture face itself stays close to",
        "its neighbour. Surfaces come from marching cubes on a Gaussian-smoothed occupancy field, Taubin",
        "smoothing and quadric decimation to the target edge length.",
        "",
    ]
    with open(os.path.join(args.out, 'README.md'), 'w') as fh:
        fh.write('\n'.join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', help='source mesh (PLY/OBJ/STL/GLB), a closed shell of a real vessel')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--fragments', type=int, required=True, help='number of Voronoi cells')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wall', type=float, default=8.0, help='target wall thickness in mm; the source is scaled to it')
    p.add_argument('--voxel', type=float, default=0.6, help='voxel size in mm')
    p.add_argument('--edge', type=float, default=0.6, help='target mesh edge length in mm')
    p.add_argument('--wear', type=float, default=0.25,
                   help='deepest chip at the fracture arris, as a fraction of t, for the most worn '
                        'fragment; each fragment draws its own strength from U(0, wear)')
    p.add_argument('--missing', type=float, default=0.05, help='fraction of fragments to withhold')
    p.add_argument('--intruders', type=int, default=0)
    p.add_argument('--intruder-mesh', default=None)
    p.add_argument('--spread', type=float, default=500.0, help='random translation range in mm')
    p.add_argument('--source-url', default='', help='recorded in ground_truth.json')
    p.add_argument('--license', default='', help='recorded in ground_truth.json')
    p.add_argument('--author', default='', help='recorded in ground_truth.json')
    p.add_argument('--no-preview', action='store_true')
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    build(args)


if __name__ == '__main__':
    main()
