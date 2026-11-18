"""Fit the MPFB basemesh head onto a Tripo head mesh: one smooth Gaussian-RBF warp (the same form fit.py writes,
so apply.py uses it as is), driven by the MediaPipe landmarks plus dense closest points on the Tripo skin.

python tripo_fit.py CORR_NPZ TRIPO_DIR OUT_NPZ [--centers 700 --sigma 0.013 --dense 3 --lap 10]
TRIPO_DIR holds tripo_mesh.npz, tripo_basecolor.png, lm3d.npy (tripo_export.py, tripo_lm.py),
and base_uvmesh.npz (export_uv.py on the unwarped base: vertex UVs and the eyeballs).
Also writes OUT_DIR/tripo_in_mpfb.npz: the Tripo vertices moved into the basemesh frame (for baking).
Dense targets skip the hair (dark texels), the eye openings (lids follow the eye landmarks only) and
the mouth-cavity and eye-socket UV islands. Each eyeball centre moves with the mean of its eye corners and lid middles.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpfb'))
from fit import OVAL, NOSE_TIP, CHIN, umeyama, farthest, smoothstep, phi  # noqa: E402

EYE_IDS = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
           263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466}


def vertex_normals(x, tris):
    fn = np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])
    n = np.zeros_like(x)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def surface_samples(co, tris, tuv, tex, steps=7):
    """points on every Tripo triangle (barycentric grid), with face normal and texture colour."""
    g = [(i, j) for i in range(steps + 1) for j in range(steps + 1 - i)]
    B = np.array([[(steps - i - j) / steps, i / steps, j / steps] for i, j in g])   # (G,3)
    V = co[tris]                                                                     # (T,3,3)
    P = np.einsum('gk,tkd->tgd', B, V).reshape(-1, 3)
    fn = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    N = np.repeat(fn, len(B), 0)
    UV = np.einsum('gk,tkd->tgd', B, tuv).reshape(-1, 2)
    h, w = tex.shape[:2]
    px = np.clip((UV[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    py = np.clip(((1 - UV[:, 1]) * (h - 1)).round().astype(int), 0, h - 1)
    return P, N, tex[py, px, :3].astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corr')
    ap.add_argument('tripo')
    ap.add_argument('out')
    ap.add_argument('--sigma', type=float, default=0.013)
    ap.add_argument('--centers', type=int, default=700)
    ap.add_argument('--mu', type=float, default=1e-4)
    ap.add_argument('--anchor', type=float, default=0.3)
    ap.add_argument('--lap', type=float, default=10.0)
    ap.add_argument('--dense', type=float, default=3.0, help='total weight of the dense term, relative to the landmarks')
    ap.add_argument('--eye-weight-fit', type=float, default=20.0)
    ap.add_argument('--iters', default='0.02,0.012,0.008,0.006,0.004,0.004', help='max closest-point distance (m) per iteration')
    ap.add_argument('--eyeball', type=float, default=50.0, help='weight of eyeball centre = mean lid displacement')
    ap.add_argument('--interior', type=float, default=2.0, help='weight of: points inside the mouth (helpers, cavity) move like the lips')
    ap.add_argument('--outline', type=float, default=0.0, help='>0: head vertices outside the target face outline (the jaw part of the MediaPipe outline, after front_morph.py) are pushed back onto it, at this weight')
    ap.add_argument('--hair-value', type=float, default=0.28, help='texels darker than this (max channel) count as hair')
    a = ap.parse_args()

    c = np.load(a.corr)
    co, trip, bary, ids, tris = c['co'], c['trip'], c['bary'], c['ids'], c['tris']
    src_all = (co[trip] * bary[:, :, None]).sum(1)
    # corr.npz ties the upper lid's landmarks to the socket: re-tie them to the lid's margin, the lowest skin at their place along
    # the eye, so the lid's arch follows the target
    for corner_a, corner_b, top, upper in ((33, 133, 159, (246, 161, 160, 158, 157, 173)), (263, 362, 386, (466, 388, 387, 385, 384, 398))):
        ka, kb, kt = [list(ids).index(i) for i in (corner_a, corner_b, top)]
        pa, pb, pt = src_all[ka], src_all[kb], src_all[kt]
        lid = (co[:, 2] > min(pa[2], pb[2]) - 0.001) & (co[:, 2] < pt[2] + 0.006) & (co[:, 1] < pt[1] + 0.004) & c['in_body']
        for f, i in zip((1 / 7, 2 / 7, 3 / 7, 4.5 / 7, 5.5 / 7, 6.3 / 7), upper):
            if i not in ids:
                continue
            x = pa[0] + (pb[0] - pa[0]) * f
            near = lid & (np.abs(co[:, 0] - x) < 0.0012) & (co[:, 2] >= (pa[2] + (pb[2] - pa[2]) * f) - 0.0005)
            if near.any():
                cand_ = co[near]
                src_all[list(ids).index(i)] = cand_[np.argmin(cand_[:, 2])]
    ref = np.load(os.path.join(a.tripo, 'lm3d.npy'))
    keep = [k for k, i in enumerate(ids) if i < 468 and (i not in OVAL or i == CHIN) and not np.isnan(ref[i, 0])]
    lm = ids[keep]
    src = src_all[keep]
    s, R, t = umeyama(ref[lm], src)
    # corr.npz ties a few upper-lid landmarks to the socket inside (~27 mm behind the lid): drop outliers, refit
    good = np.linalg.norm((s * (R @ ref[lm].T)).T + t - src, axis=1) < 0.01
    err_ = np.linalg.norm((s * (R @ ref[lm].T)).T + t - src, axis=1)
    print("dropped landmarks", lm[~good].tolist(), "err mm", (err_[~good] * 1000).round(1).tolist())
    lm, src = lm[good], src[good]
    s, R, t = umeyama(ref[lm], src)
    if os.environ.get('KEEP_LEVEL') == '1':
        # the head's own lean / turn kept (level_head.sh made it level): only the nod (about x) from the fit. MPFB's landmarks
        # are a little lopsided, and the free fit leaned one levelled test head 1.07 degrees again.
        pitch = np.arctan2(R[2, 1], R[1, 1])
        c_, s_ = np.cos(pitch), np.sin(pitch)
        R = np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]])
        rc, sc_ = ref[lm] - ref[lm].mean(0), src - src.mean(0)
        s = float(((R @ rc.T).T * sc_).sum() / (rc ** 2).sum())
        t = src.mean(0) - s * (R @ ref[lm].mean(0))
        print('KEEP_LEVEL: nod only, %.2f deg' % np.degrees(pitch))
    tgt = (s * (R @ ref[lm].T)).T + t
    before = np.linalg.norm(tgt - src, axis=1)

    m = np.load(os.path.join(a.tripo, 'tripo_mesh.npz'))
    tco = (s * (R @ m['co'].T)).T + t
    np.savez(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'tripo_in_mpfb.npz'), co=tco, tris=m['tris'], tuv=m['tuv'])
    tex = np.array(Image.open(os.path.join(a.tripo, 'tripo_basecolor.png')).convert('RGB'))
    P, N, col = surface_samples(tco, m['tris'], m['tuv'], tex)
    skin = col.max(1) > a.hair_value
    tree = cKDTree(P)
    print('tripo samples', len(P), 'skin', int(skin.sum()))

    nose = src[list(lm).index(NOSE_TIP)]
    chin = src[list(lm).index(CHIN)]
    body = np.where(c['in_body'])[0]
    head = body[co[body, 2] > chin[2] - 0.07]
    hc = co[head].mean(0)
    centers = farthest(co[head], a.centers)
    anchors = head[(co[head, 1] > hc[1] + 0.04) | (co[head, 2] > nose[2] + 0.11)][::3]
    z_lo, z_hi = chin[2] - 0.06, chin[2] - 0.015

    # eye openings: an ellipse around each eye outline, a little wider than the lids
    eye_zone = np.zeros(len(co), bool)
    for side in ((33, 133, 159, 145), (362, 263, 386, 374)):
        pts = src[[list(lm).index(i) for i in side]]
        ctr = pts.mean(0)
        hw = np.linalg.norm(pts[0] - pts[1]) / 2
        d = co - ctr
        eye_zone |= ((d[:, 0] / (hw * 1.25)) ** 2 + (d[:, 2] / (hw * 0.9)) ** 2 < 1) & (np.abs(d[:, 1]) < 0.02)
    bu = np.load(os.path.join(a.tripo, 'base_uvmesh.npz'))
    vu = np.zeros((len(co), 2))
    vu[bu['tv'].ravel()] = bu['tuv'].reshape(-1, 2)
    inner = ((vu[:, 0] > 0.74) & (vu[:, 1] < 0.14)) | ((vu[:, 0] < 0.17) & (vu[:, 1] > 0.5) & (vu[:, 1] < 0.7))
    cand = head[~eye_zone[head] & ~inner[head]]
    print('dense candidates', len(cand), 'of head', len(head))
    eye_co = bu['eye_co']
    eye_src, eye_tgt = [], []
    for sgn, side in ((-1, (33, 133, 159, 145)), (1, (263, 362, 386, 374))):   # corners + lid middles
        k = [list(lm).index(i) for i in side if i in lm]
        ctr = eye_co[eye_co[:, 0] * sgn > 0].mean(0)
        eye_src.append(ctr)
        eye_tgt.append(ctr + (tgt[k] - src[k]).mean(0))
    eye_src, eye_tgt = np.array(eye_src), np.array(eye_tgt)

    def basis(x):
        return smoothstep(x[:, 2], z_lo, z_hi)[:, None] * phi(x, centers, a.sigma)

    pos = {v: k for k, v in enumerate(head)}
    nbr = [set() for _ in head]
    for tri in tris:
        if all(v in pos for v in tri):
            for x in tri:
                nbr[pos[x]].update(pos[y] for y in tri if y != x)
    Bh = basis(co[head])
    L = Bh - np.stack([Bh[list(n)].mean(0) if n else Bh[k] for k, n in enumerate(nbr)])
    Bc = basis(co[cand])
    Ba = basis(co[anchors])
    lw = np.sqrt(np.where(np.isin(lm, list(EYE_IDS)), a.eye_weight_fit, 1.0))[:, None]
    Bl = basis(src)
    Be = basis(eye_src)
    # inside the head nothing pulls on the warp: helper/cavity points (teeth, tongue sit there) follow the lips
    lips = [list(lm).index(i) for i in (0, 17, 13, 14, 61, 291, 37, 267, 84, 314) if i in lm]
    lip_d = (tgt[lips] - src[lips]).mean(0)
    mouth_c = src[lips].mean(0)
    near_mouth = np.linalg.norm((co - mouth_c) * np.array([1, 0.6, 1]), axis=1) < 0.05
    interior = np.where(near_mouth & (~c['in_body'] | inner))[0][::2]
    Bi = basis(co[interior])
    print('interior points', len(interior), 'lip move mm', (lip_d * 1000).round(1))

    # the target outline: from the temple side down round the chin; verts outside it in the front plane come back onto it.
    # The neck under the chin is behind it (larger y) and stays out of this
    JAW = (132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361)
    out_tree = None
    if a.outline > 0:
        plane = np.load(os.path.join(a.tripo, 'lm_plane.npy'))[list(JAW)]
        plane[:, 1] = ref[CHIN, 1]                               # the camera's depth there means nothing: at the chin's
        jaw_t = (s * (R @ plane.T)).T + t
        P2 = jaw_t[:, [0, 2]]
        # OUTLINE_X: narrower than the portrait's, for a slimmer face; OUTLINE_CHIN: this much more at the chin, fading to none at the outline's ends
        P2[:, 0] = P2[8, 0] + (P2[:, 0] - P2[8, 0]) * (float(os.environ.get('OUTLINE_X', 1)) - float(os.environ.get('OUTLINE_CHIN', 0)) * (1 - np.abs(np.arange(len(P2)) - 8) / 8))
        dense_line = np.vstack([P2[k] + (P2[k + 1] - P2[k]) * f for k in range(len(P2) - 1) for f in np.linspace(0, 1, 12, endpoint=False)] + [P2[-1]])
        out_tree = cKDTree(dense_line)
        top_z = P2[[0, -1], 1].mean()
        closed = np.vstack([P2, [P2[-1, 0], top_z + 0.2], [P2[0, 0], top_z + 0.2]]).astype(np.float32)
        y_chin = jaw_t[8, 1]
        chin_z = P2[8, 1]

    def outside(cur):
        '''indices into head of the vertices outside the target outline, with their targets'''
        h = cur[head]
        reach = (h[:, 2] < top_z) & (h[:, 1] < y_chin + 0.075) & ((h[:, 2] > chin_z - 0.006) | (h[:, 1] < y_chin + 0.03))
        # under the jaw line, further back than the chin's underside, it is neck
        far = (h[:, 1] > y_chin + 0.02 + 0.8 * np.clip(h[:, 2] - chin_z, 0, 1)) & (h[:, 2] < chin_z + 0.03)
        idx = np.where(reach & ~far)[0]
        ins = np.array([cv2.pointPolygonTest(closed, (float(x), float(z)), False) for x, z in h[idx][:, [0, 2]]])
        idx = idx[ins < 0]
        tg = h[idx].copy()
        tg[:, [0, 2]] = dense_line[out_tree.query(h[idx][:, [0, 2]])[1]]
        return idx, tg

    out_idx, out_tgt = np.zeros(0, int), np.zeros((0, 3))

    def solve(sel, dtgt):
        wd = np.sqrt(a.dense * len(src) / max(len(sel), 1))
        wo = np.sqrt(a.outline * len(src) / max(len(out_idx), 1))
        A = np.vstack([lw * Bl, np.sqrt(a.eyeball) * Be, np.sqrt(a.interior * len(src) / max(len(interior), 1)) * Bi, wd * Bc[sel], np.sqrt(a.anchor) * Ba, np.sqrt(a.mu) * np.eye(len(centers)), np.sqrt(a.lap) * L, wo * Bh[out_idx]])
        B = np.vstack([lw * (tgt - src), np.sqrt(a.eyeball) * (eye_tgt - eye_src),
                       np.sqrt(a.interior * len(src) / max(len(interior), 1)) * np.tile(lip_d, (len(interior), 1)), wd * (dtgt - co[cand[sel]]), np.zeros((len(anchors), 3)),
                       np.zeros((len(centers), 3)), np.zeros((len(head), 3)), wo * (out_tgt - co[head[out_idx]])])
        return np.linalg.lstsq(A, B, rcond=None)[0]

    W = solve(np.zeros(0, int), np.zeros((0, 3)))
    for it, maxd in enumerate(float(x) for x in a.iters.split(',')):
        cur = co + basis(co) @ W
        nrm = vertex_normals(cur, tris)[cand]
        dist, j = tree.query(cur[cand])
        ok = (dist < maxd) & skin[j] & ((nrm * N[j]).sum(1) > 0.5)
        sel = np.where(ok)[0]
        if out_tree is not None:                                 # stop the closest-point term holding a vertex out there
            out_idx, out_tgt = outside(cur)
            ok &= ~np.isin(cand, head[out_idx])
            sel = np.where(ok)[0]
            print('outside the outline', len(out_idx), 'max mm', round(float(np.linalg.norm(cur[head[out_idx]] - out_tgt, axis=1).max() * 1000) if len(out_idx) else 0, 1))
        print('iter', it, 'maxd mm', maxd * 1000, 'matched', len(sel), '/', len(cand), 'median gap mm', round(float(np.median(dist[ok]) * 1000), 2))
        W = solve(sel, P[j[sel]])
    cur = co + basis(co) @ W
    dist, j = tree.query(cur[cand])
    ok = skin[j] & ((vertex_normals(cur, tris)[cand] * N[j]).sum(1) > 0.5) & (dist < 0.01)
    after = np.linalg.norm(src + Bl @ W - tgt, axis=1)
    eye_k = np.isin(lm, list(EYE_IDS))
    eye_scale = []
    for ca, cb in ((33, 133), (362, 263)):
        ka, kb = list(lm).index(ca), list(lm).index(cb)
        wa = src + Bl @ W
        eye_scale.append(float(np.linalg.norm(wa[ka] - wa[kb]) / np.linalg.norm(src[ka] - src[kb])))
    np.savez(a.out, eye_scale=np.array(eye_scale), centers=centers, W=W, sigma=a.sigma, z_lo=z_lo, z_hi=z_hi, scale=s, rot=R, trans=t)
    np.save(a.out.replace('.npz', '_gap.npy'), np.stack([cand[ok], dist[ok]], 1))
    print('eyeball move mm', ((eye_tgt - eye_src) * 1000).round(1).tolist(), 'got', ((Be @ W) * 1000).round(1).tolist())
    print(json.dumps({'landmarks': len(lm), 'tripo_to_mpfb_scale': s, 'eye_scale': eye_scale,
                      'before_mm_median': float(np.median(before) * 1000), 'after_mm_median': float(np.median(after) * 1000),
                      'eye_after_mm_median': float(np.median(after[eye_k]) * 1000),
                      'surface_gap_mm_median': float(np.median(dist[ok]) * 1000), 'surface_gap_mm_p90': float(np.percentile(dist[ok], 90) * 1000),
                      'max_disp_mm': float(np.linalg.norm(Bh @ W, axis=1).max() * 1000)}, indent=1))


if __name__ == '__main__':
    main()
