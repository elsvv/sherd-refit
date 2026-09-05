"""Tiny software renderer (shaded point splats with a z-buffer). Open3D's offscreen renderer is
not available on macOS, and this needs no GPU or display."""
from __future__ import annotations

import numpy as np

PALETTE = np.array([[0.78, 0.78, 0.78], [0.95, 0.55, 0.25], [0.40, 0.70, 0.95], [0.50, 0.85, 0.50], [0.90, 0.80, 0.40],
                    [0.80, 0.50, 0.90], [0.35, 0.85, 0.85], [0.90, 0.45, 0.55], [0.60, 0.60, 0.95], [0.75, 0.60, 0.40]])


def _splat(V, N, C, eye_dir, up, W, H, center, scale, light):
    z = np.asarray(eye_dir, float); z = z / np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0, 0], z)
    x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    P = (V - center) @ R
    Nn = N @ R
    shade = np.clip(Nn @ light, 0, 1) * 0.75 + 0.25
    shade[Nn[:, 2] < 0] *= 0.5
    col = C * shade[:, None]
    px = np.round(P[:, 0] * scale + W / 2).astype(int)
    py = np.round(-P[:, 1] * scale + H / 2).astype(int)
    ok = (px >= 1) & (px < W - 1) & (py >= 1) & (py < H - 1)
    img = np.full((H, W, 3), 0.16, np.float32)
    zb = np.full((H, W), -np.inf, np.float32)
    depth = P[:, 2]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            xx, yy, dd, cc = px[ok] + dx, py[ok] + dy, depth[ok], col[ok]
            lin = yy * W + xx
            o2 = np.lexsort((dd, lin))
            lin2 = lin[o2]
            last = np.ones(len(lin2), bool); last[:-1] = lin2[1:] != lin2[:-1]
            sel = o2[last]
            better = dd[sel] > zb[yy[sel], xx[sel]]
            sel = sel[better]
            zb[yy[sel], xx[sel]] = dd[sel]
            img[yy[sel], xx[sel]] = cc[sel]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_views(meshes, out_png, views=None, W=1000, H=800, label=None):
    """meshes: list of (V, N, C) arrays. views: list of (eye_dir, up). Writes a horizontal strip."""
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
    if views is None:
        views = [([0, 0, 1], [0, 1, 0]), ([0, 0, -1], [0, 1, 0]), ([1, 0.2, 0.4], [0, 0, 1])]
    V = np.concatenate([m[0] for m in meshes]); N = np.concatenate([m[1] for m in meshes]); C = np.concatenate([m[2] for m in meshes])
    center = 0.5 * (V.min(0) + V.max(0))
    ext = np.linalg.norm(V.max(0) - V.min(0))
    scale = 0.9 * min(W, H) / max(ext, 1e-9)
    light = np.array([0.3, 0.4, 1.0]); light /= np.linalg.norm(light)
    imgs = [_splat(V, N, C, e, u, W, H, center, scale, light) for e, u in views]
    im = Image.fromarray(np.concatenate(imgs, 1))
    if label:
        ImageDraw.Draw(im).text((10, 10), label, fill=(255, 255, 255))
    im.save(out_png)
    return out_png


def principal_views(V):
    """Views along the dominant normal axis of the point set (top, bottom, two obliques)."""
    X = V - V.mean(0)
    w, ev = np.linalg.eigh(X.T @ X)
    u, e1, e2 = ev[:, 0], ev[:, 2], ev[:, 1]
    return [(u, e2), (-u, e2), (e1 + 0.35 * u, u), (e2 + 0.35 * u, u)]
