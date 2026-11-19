"""Round the eyes, the skin takes the input's colour as seen from the camera the lid lines were fitted in (projection), not from the
closest point in 3D.   any face, after run.sh with LID_FIT (it reads the camera lid_arap.py used)
python lid_project.py OUT_DIR SKIN_IN SKIN_OUT

Why: the bake (tripo_tex.py) gives each texel the colour of the closest point on the input surface. MPFB's lids
curl in towards the eyeball more than the input's flat painted eye, so just above the upper margin the closest input point was the
painted eye itself: a band of iris and white colours sat on the lid above the fitted margin, and lash_erase.py smeared it into a
dull grey band. The margins were fitted to the input's eye outline as seen from this camera, so here a texel takes the colour of the
input surface on the same camera ray: from that view the lids show exactly what the input texture shows there.
Which texels: within an ellipse round each eye (x: EX * the ball's radius, up EU, down ED; full inside FULL of it), facing the
camera, and not hidden from it (by the skin in front or the eyeball, a sphere); the input hit within MAXD of the texel along the
ray. The weight is softened in UV (sigma 2 texels). Prints the texels replaced per eye.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

O, src, dst = sys.argv[1:4]
E = lambda k, d: float(os.environ.get(k, d))
EX, EU, ED, FULL, MAXD, SS = E('EX', 1.45), E('EU', 0.85), E('ED', 0.75), E('FULL', 0.7), E('MAXD', 0.006), int(E('SS', 2))

cam = json.load(open(f'{O}/lidfit/warp/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1] * SS; F = (RES / 2) / np.tan(cam['fov'] / 2)


def to_cam(P):
    """pixel x, pixel y, depth along the view axis"""
    q = (P - Cc) @ Rc
    d = -q[..., 2]
    return np.stack([RES / 2 + F * q[..., 0] / d, RES / 2 - F * q[..., 1] / d, d], -1)


def raster(S, attr, box):
    """z-buffer of screen-space triangles S (T,3,3: px, py, depth) over box (x0, y0, x1, y1); attr (T,3,k) interpolated
    perspective-correctly. Returns depth (H,W) (inf where empty) and attr (H,W,k)."""
    x0, y0, x1, y1 = box
    H, W = y1 - y0, x1 - x0
    Z = np.full((H, W), np.inf); A = np.zeros((H, W, attr.shape[2]))
    for s, a in zip(S, attr):
        if (s[:, 2] <= 0).any():
            continue
        bx0, by0 = max(int(np.floor(s[:, 0].min())), x0), max(int(np.floor(s[:, 1].min())), y0)
        bx1, by1 = min(int(np.ceil(s[:, 0].max())) + 1, x1), min(int(np.ceil(s[:, 1].max())) + 1, y1)
        if bx0 >= bx1 or by0 >= by1:
            continue
        gx, gy = np.meshgrid(np.arange(bx0, bx1) + 0.5, np.arange(by0, by1) + 0.5)
        (ax, ay), (bx_, by_), (cx, cy) = s[0, :2], s[1, :2], s[2, :2]
        den = (by_ - cy) * (ax - cx) + (cx - bx_) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l0 = ((by_ - cy) * (gx - cx) + (cx - bx_) * (gy - cy)) / den
        l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        iw = l0 / s[0, 2] + l1 / s[1, 2] + l2 / s[2, 2]          # 1/depth, linear on screen
        z = 1 / iw
        sub = Z[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
        win = inside & (z < sub)
        if not win.any():
            continue
        sub[win] = z[win]
        val = (l0[..., None] * a[0] / s[0, 2] + l1[..., None] * a[1] / s[1, 2] + l2[..., None] * a[2] / s[2, 2]) * z[..., None]
        A[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0][win] = val[win]
    return Z, A


t = np.load(f'{O}/texels.npz'); pos, nrm, have = t['pos'], t['nrm'], t['have']
u = np.load(f'{O}/uvmesh.npz'); V, tv, EC = u['co'], u['tv'], u['eye_co']
m = np.load(f'{O}/tripo_in_mpfb.npz')
tex = np.asarray(Image.open(f'{O}/tripo_basecolor.png').convert('RGB'), np.float32) / 255
img = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255
Wt = np.zeros(have.shape, np.float32)
dbg_in = np.zeros(have.shape, bool)
# the approved outlines in this camera (SS times its pixels): per pixel, how far inside an opening it lies (px), 0 outside
ln_ = [l for l in open(f'{O}/lidfit/check_lines.txt').read().split('\n') if l.strip()]
OUTLINE = np.zeros((RES, RES), np.uint8)
for k_ in range(0, len(ln_), 3):
    a_, b_ = map(float, ln_[k_].split()); up_ = np.array(ln_[k_ + 1].split(), float); lo_ = np.array(ln_[k_ + 2].split(), float)
    xx_ = np.linspace(a_, b_, len(up_))
    cv2.fillPoly(OUTLINE, [(np.r_[np.c_[xx_, up_], np.c_[xx_[::-1], lo_[::-1]]] * SS).round().astype(np.int32)], 1)
OUTLINE = cv2.distanceTransform(OUTLINE, cv2.DIST_L2, 5)
Pc = np.zeros(img.shape, np.float32)
ys, xs = np.nonzero(have)
q = pos[ys, xs]; qn = nrm[ys, xs]
for sgn in (1, -1):
    e = EC[np.sign(EC[:, 0]) == sgn]
    # the ball as a sphere fitted to its back half (the box's longest side put the sphere's front 5 mm before the cornea
    # and hid the lids resting on the ball)
    b = e[e[:, 1] > np.median(e[:, 1])]
    s_ = np.linalg.lstsq(np.c_[2 * b, np.ones(len(b))], (b ** 2).sum(1), rcond=None)[0]
    c = s_[:3]; r = float(np.sqrt(s_[3] + c @ c))
    dx, dz = (q[:, 0] - c[0]) / (EX * r), (q[:, 2] - c[2]) / np.where(q[:, 2] > c[2], EU * r, ED * r)
    rho = np.hypot(dx, dz)
    zone = (rho < 1) & (q[:, 1] < c[1] + r)                      # in front of the ball's back
    if not zone.any():
        continue
    k = np.where(zone)[0]
    Q = to_cam(q[k])
    view = Cc - q[k]; view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing = (qn[k] * view).sum(1) > 0.05
    box = (int(Q[:, 0].min()) - 2, int(Q[:, 1].min()) - 2, int(Q[:, 0].max()) + 3, int(Q[:, 1].max()) + 3)
    # occluders: our skin, and the eyeball as a sphere
    S = to_cam(V[tv])
    near_t = \
        (S[..., 0].max(1) >= box[0]) & (S[..., 0].min(1) <= box[2]) & (S[..., 1].max(1) >= box[1]) & (S[..., 1].min(1) <= box[3])
    Zs, _ = raster(S[near_t], np.zeros((near_t.sum(), 3, 1)), box)
    gx, gy = np.meshgrid(np.arange(box[0], box[2]) + 0.5, np.arange(box[1], box[3]) + 0.5)
    d = (Rc @ np.stack([(gx - RES / 2) / F, -(gy - RES / 2) / F, -np.ones_like(gx)], 0).reshape(3, -1)).T   # per-pixel ray, depth 1
    oc = Cc - c; bq = (d * oc).sum(1); cq = (oc * oc).sum() - r * r; disc = bq * bq - (d * d).sum(1) * cq
    tb = np.where(disc > 0, (-bq - np.sqrt(np.maximum(disc, 0))) / (d * d).sum(1), np.inf).reshape(gx.shape)
    Zocc = np.minimum(Zs, tb)
    # the input's colour along each ray
    T = to_cam(m['co'][m['tris']])
    near_tr = (T[..., 0].max(1) >= box[0]) & (T[..., 0].min(1) <= box[2]) & (T[..., 1].max(1) >= box[1]) & (T[..., 1].min(1) <= box[3])
    Zt, UVt = raster(T[near_tr], m['tuv'][near_tr], box)
    ix = np.clip(Q[:, 0].astype(int) - box[0], 0, gx.shape[1] - 1); iy = np.clip(Q[:, 1].astype(int) - box[1], 0, gx.shape[0] - 1)
    visible = Q[:, 2] <= Zocc[iy, ix] + 0.0004
    hit = np.isfinite(Zt[iy, ix]) & (np.abs(Zt[iy, ix] - Q[:, 2]) < MAXD)
    uv = UVt[iy, ix]
    h, w = tex.shape[:2]
    col = tex[np.clip(((1 - uv[:, 1]) * (h - 1)).round().astype(int), 0, h - 1), np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)]
    wgt = np.clip((1 - rho[k]) / (1 - FULL), 0, 1) * (facing & visible & hit)
    # skin never takes what the input texture shows inside its eye opening: a texel of ours that projects inside the approved outline (where
    # the input shows the eyeball) by more than IN_MM is left to the nearest texel outside it (the hidden-texel pass below). The colour
    # test after this caught only the purest white; mixed white stayed on the lids
    inside_px = OUTLINE[np.clip(Q[:, 1].astype(int), 0, RES - 1), np.clip(Q[:, 0].astype(int), 0, RES - 1)]
    in_eye = inside_px * (Q[:, 2] / F * 1000) > E('IN_MM', 0.1)
    if os.environ.get('SCL_DEBUG'):
        dbg_in[ys[k][in_eye & (wgt > 0)], xs[k][in_eye & (wgt > 0)]] = True
    wgt = wgt * ~in_eye
    Wt[ys[k], xs[k]] = np.maximum(Wt[ys[k], xs[k]], wgt)
    Pc[ys[k][wgt > 0], xs[k][wgt > 0]] = col[wgt > 0]
    print('LID_PROJECT eye %+d: zone texels %d, facing %d, visible %d, hit %d, replaced %d' % (sgn, len(k), facing.sum(), visible.sum(), hit.sum(), (wgt > 0.5).sum()))
# the lids' under-edges and anything else in the zones the camera does not see (they face down or hide behind the margin) kept the
# closest-point bake: the input's painted white, a pale band along the margin seen from straight ahead. They take the
# colour of the nearest projected texel in 3D within REACH_HID, so the rim's colour wraps round the margin.
from scipy.spatial import cKDTree
got = Wt > 0.5
# where our lid lies over what the input texture shows as the eyeball (its opening a little wider than ours here and there, mostly at the
# corners), the lid took the white (the eye's white baked onto the lid). Projected texels of the eyeball's white (light, nearly
# grey: chroma under SCL_C, L over SCL_L in Lab) count as not projected, and take the nearest projected skin colour below.
lab_p = cv2.cvtColor(np.clip(Pc * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
chroma = np.hypot(lab_p[..., 1] - 128, lab_p[..., 2] - 128); Lp = lab_p[..., 0] * 100 / 255
sclera = got & (chroma < E('SCL_C', 9)) & (Lp > E('SCL_L', 55))
print('LID_PROJECT eyeball white on the lids: %d texels' % sclera.sum())
if os.environ.get('SCL_DEBUG'):
    np.save(f'{O}/lid_sclera_mask.npy', sclera | dbg_in)
Wt[sclera] = 0; got &= ~sclera
gy, gx = np.nonzero(got)
if len(gy):
    tree = cKDTree(pos[gy, gx])
    zone_all = np.zeros(have.shape, bool)
    for sgn in (1, -1):
        e = EC[np.sign(EC[:, 0]) == sgn]; bb = e[e[:, 1] > np.median(e[:, 1])]
        s2 = np.linalg.lstsq(np.c_[2 * bb, np.ones(len(bb))], (bb ** 2).sum(1), rcond=None)[0]
        c2 = s2[:3]; r2 = float(np.sqrt(s2[3] + c2 @ c2))
        dz = (pos[..., 2] - c2[2]) / np.where(pos[..., 2] > c2[2], EU * r2, ED * r2)
        zone_all |= have & (np.hypot((pos[..., 0] - c2[0]) / (EX * r2), dz) < FULL) & (pos[..., 1] < c2[1] + r2)
    hy, hx = np.nonzero(zone_all & ~got)
    d_, j_ = tree.query(pos[hy, hx], workers=-1)
    ok = d_ < E('REACH_HID', 0.005)   # 1.5 mm left the lid's inner side further back with the input's painted white
    Pc[hy[ok], hx[ok]] = Pc[gy[j_[ok]], gx[j_[ok]]]; Wt[hy[ok], hx[ok]] = 1.0
    print('LID_PROJECT hidden texels given the nearest projected colour: %d' % ok.sum())
if os.environ.get('SRC_DEBUG'):   # which route each texel's colour took: green projected, blue the nearest projected, red the bake
    dimg = np.asarray(Image.open(src).convert('RGB')).copy()
    zt = np.zeros(have.shape, bool)
    for sgn in (1, -1):
        e = EC[np.sign(EC[:, 0]) == sgn]; bb = e[e[:, 1] > np.median(e[:, 1])]
        s2 = np.linalg.lstsq(np.c_[2 * bb, np.ones(len(bb))], (bb ** 2).sum(1), rcond=None)[0]
        c2 = s2[:3]; r2 = float(np.sqrt(s2[3] + c2 @ c2))
        zt |= have & (np.linalg.norm(pos - c2, axis=-1) < r2 + 0.006)
    dimg[zt] = (255, 0, 0); dimg[Wt > 0] = (0, 0, 255); dimg[got] = (0, 255, 0)
    Image.fromarray(dimg).save(os.environ['SRC_DEBUG'])
# soften the weight in UV, normalised inside the islands (a plain blur pulled the weight down along every island border)
Hv = have.astype(np.float32)
Ws = gaussian_filter(Wt, 2) / np.maximum(gaussian_filter(Hv, 2), 1e-6) * Hv * (Wt > 0)
out = img * (1 - Ws[..., None]) + Pc * Ws[..., None]
Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8)).save(dst)
np.save(f'{O}/lid_project_w.npy', Ws)
print('LID_PROJECT wrote', dst)
