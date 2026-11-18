"""Skin texture for the warped basemesh from the Tripo colours: each texel takes the colour of the closest
point on the Tripo surface; away from the head (or on the T-shirt) it falls back to the MPFB skin,
recoloured to match the Tripo neck.

python tripo_tex.py TEXELS_NPZ TRIPO_IN_MPFB_NPZ TRIPO_BASECOLOR WARP_NPZ BASE_SKIN OUT_PNG
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def bilinear(img, uv):
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * (w - 1), 0, w - 1.001)
    y = np.clip((1 - uv[:, 1]) * (h - 1), 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def closest_bary(p, a, b, c):
    """Barycentric coordinates (w_a, w_b, w_c) of the point on triangle abc closest to p, all arrays (n, 3) (Ericson, vectorised)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp_ = p - c; d5, d6 = (ab * cp_).sum(1), (ac * cp_).sum(1)
    vc = d1 * d4 - d3 * d2; vb = d5 * d2 - d1 * d6; va = d3 * d6 - d5 * d4
    n = len(p); out = np.zeros((n, 3), np.float64)
    done = np.zeros(n, bool)
    def put(m, w):
        m = m & ~done; out[m] = w[m] if w.ndim == 2 else w; done[m] = True
    one = lambda i: np.tile(np.eye(3)[i], (n, 1))
    put((d1 <= 0) & (d2 <= 0), one(0))
    put((d3 >= 0) & (d4 <= d3), one(1))
    put((d6 >= 0) & (d5 <= d6), one(2))
    with np.errstate(all='ignore'):
        v = d1 / (d1 - d3); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - v, v, 0 * v], 1))
        w = d2 / (d2 - d6); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - w, 0 * w, w], 1))
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6)); put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.stack([0 * w2, 1 - w2, w2], 1))
        den = 1 / (va + vb + vc); v2, w3 = vb * den, vc * den
        put(np.ones(n, bool), np.stack([1 - v2 - w3, v2, w3], 1))
    return out


def main():
    texels, tripo, basecolor, warp, base_skin, out = sys.argv[1:7]
    tx = np.load(texels)
    pos, have = tx['pos'], tx['have']
    m = np.load(tripo)
    tex = np.asarray(Image.open(basecolor).convert('RGB'), np.float32) / 255
    steps = int(os.environ.get('BAKE_STEPS', 12))        # samples per triangle edge: a low-poly Tripo head needs more, or the bake turns to mosaic
    g = [(i, j) for i in range(steps + 1) for j in range(steps + 1 - i)]
    Bc = np.array([[(steps - i - j) / steps, i / steps, j / steps] for i, j in g], np.float32)
    P = np.einsum('gk,tkd->tgd', Bc, m['co'][m['tris']].astype(np.float32)).reshape(-1, 3)
    UV = np.einsum('gk,tkd->tgd', Bc, m['tuv'].astype(np.float32)).reshape(-1, 2)
    tree = cKDTree(P)

    ys, xs = np.where(have)
    q = pos[ys, xs]
    w = np.load(warp)
    chin = float(w['z_hi']) + 0.015
    above = q[:, 2] > chin - 0.14                                     # nothing below the collar takes a Tripo colour
    dist = np.full(len(q), np.inf, np.float32)
    col = np.zeros((len(q), 3), np.float32)
    dist[above], j = tree.query(q[above], workers=-1)
    if os.environ.get('TEX_EXACT') == '1':
        # TEX_EXACT=1: the nearest sample only picks the triangle; the colour comes from the exact closest point on it (the samples are
        # further apart than a texel, so their own UVs make the bake blocky)
        tri = j // len(g)
        T3 = m['co'][m['tris']].astype(np.float64)[tri]; U3 = m['tuv'].astype(np.float64)[tri]
        qa = q[above].astype(np.float64)
        bw = closest_bary(qa, T3[:, 0], T3[:, 1], T3[:, 2])
        dist[above] = np.linalg.norm(qa - np.einsum('nk,nkd->nd', bw, T3), axis=1)
        col[above] = bilinear(tex, np.einsum('nk,nkd->nd', bw, U3))
    else:
        col[above] = bilinear(tex, UV[j])

    mx, mn = col.max(1), col.min(1)
    sat = (mx - mn) / (mx + 1e-6)
    shirt = (sat < 0.12) & (mx > 0.45) & (os.environ.get('SHIRT', '1') == '1')   # SHIRT=0: pale highlights of a fair face stay Tripo's
    near = np.clip((0.008 - dist) / 0.004, 0, 1)                      # full within 4 mm, none past 8 mm
    # NECK_REACH (m): below the chin MPFB's neck is not fitted to Tripo's and lies further off; with the 8 mm reach
    # it turned to the MPFB skin along a sharp line under the jaw. There the nearest Tripo point is still the neck: reach further
    NR = float(os.environ.get('NECK_REACH', 0))
    if NR > 0:
        under = q[:, 2] < chin
        near = np.where(under, np.clip((NR - dist) / (0.3 * NR), 0, 1), near)
    zramp = np.clip((q[:, 2] - (chin - 0.11)) / 0.03, 0, 1)          # neck fades out towards the collar
    wt = near * zramp * (~shirt)
    # the scalp sits inside Tripo's hair shell: take the hair colour from further away (painted-on hair)
    # only above the eyes or behind the ears (nape): the shadow under the jaw is dark too
    hair = (mx < 0.3) & (dist < 0.03) & ((q[:, 2] > chin + 0.075) | ((q[:, 1] > -0.03) & (q[:, 2] > chin - 0.02)))
    wt = np.maximum(wt, hair.astype(np.float32))

    base = np.asarray(Image.open(base_skin).convert('RGB').resize(have.shape[::-1]), np.float32) / 255
    bcol = base[ys, xs]
    # recolour the MPFB skin: per-channel gain from texels on the neck/jaw that are clearly skin
    ref = (wt > 0.9) & (q[:, 2] < chin) & (sat > 0.15)
    gain = np.median(col[ref], 0) / np.median(bcol[ref], 0)
    print('neck texels', int(ref.sum()), 'gain', gain.round(3))

    W = np.zeros(have.shape, np.float32)
    W[ys, xs] = wt
    # soft seam in UV space, normalised inside the islands: a plain blur took in the zeros outside each island, so every island's
    # border fell back to the MPFB skin, a thin light line along each UV seam (e.g. the head/neck seam under the jaw)
    Hv = have.astype(np.float32)
    W = gaussian_filter(W, 3) / np.maximum(gaussian_filter(Hv, 3), 1e-6) * Hv
    res = np.clip(base * gain, 0, 1)
    C = res.copy()
    C[ys, xs] = col
    res = C * W[..., None] + res * (1 - W[..., None])
    # pad each UV island: the texels just outside it held the MPFB skin, and texture filtering drew that across the islands'
    # seams as a thin light line (e.g. under the jaw). PAD texels outside take the nearest island texel's colour
    from scipy.ndimage import distance_transform_edt
    PAD = int(os.environ.get('UV_PAD', 8))
    if PAD > 0:
        dist_out, (iy, ix) = distance_transform_edt(~have, return_indices=True)
        gut = (~have) & (dist_out <= PAD)
        res[gut] = res[iy[gut], ix[gut]]
    Image.fromarray((np.clip(res, 0, 1) * 255).round().astype(np.uint8)).save(out)
    print('texels', len(q), 'from tripo', int((wt > 0.5).sum()), 'median dist mm', round(float(np.median(dist[wt > 0.5]) * 1000), 2))


if __name__ == '__main__':
    main()
