"""The input texture's painted eyes onto the eyeball texture, projected from the lid-fit camera (the same view the lids were fitted in).   any face
python eye_project_cam.py OUT_DIR EYE_NPZ EYE_IN.png EYE_OUT.png

Why: the base mesh's iris (its radial pattern, a hard black pupil) read as a doll's eye next to the input's.
How, per eyeball texel of the front half (facing the camera): its 3D point projected into lidfit/warp/v0_tripo.png.
 - inside the approved outline (lidfit/check_lines.txt) shrunk by EDGE mm: the input's colour as it is, except its painted catchlight
   (light pixels inside the iris circle) which the viewer adds itself;
 - elsewhere (under the lids, past the corners) a model of the same eye: inside the iris circle (fitted to the input's dark iris
   pixels away from the lids) the median colour of the visible iris at that radius; outside it the median visible white;
 - blended over BLEND mm inside the shrunk outline. The back half and the alpha stay as they were.
"""
import json
import os
import sys

import cv2
import numpy as np

O, eye_npz, src, dst = sys.argv[1:5]
E = lambda k, d: float(os.environ.get(k, d))
EDGE, BLEND, NEAR, SS = E('EDGE', 0.15), E('BLEND', 0.25), E('NEAR', 0.8), 1
# KEEP=1 (the radial model blurred the iris's brown and the pupil's black): the input's eye kept as painted. The colours are read
# bilinearly (nearest pixel stretched the input's 88 px iris to ~200 texels in blocks); the painted catchlight is filled from the
# input pixels round it (inpainting) instead of the radial median, which flattened the pupil's rim where the catchlight lay
# over it; the pupil is not redrawn as a disc. The catchlight still goes (the viewer adds its own).
KEEP = os.environ.get('KEEP', '0') == '1'

cam = json.load(open(f'{O}/lidfit/warp/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1]; F = (RES / 2) / np.tan(cam['fov'] / 2)
timg = cv2.imread(f'{O}/lidfit/warp/v0_tripo.png')
lab = cv2.cvtColor(timg, cv2.COLOR_BGR2LAB).astype(np.float32)
Lt = lab[..., 0] * 100 / 255; Ct = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
ln = [l for l in open(f'{O}/lidfit/check_lines.txt').read().split('\n') if l.strip()]
outlines = []
for k in range(0, len(ln), 3):
    x0, x1 = map(float, ln[k].split()); up = np.array(ln[k + 1].split(), float); lo = np.array(ln[k + 2].split(), float)
    xx = np.linspace(x0, x1, len(up))
    m = np.zeros(timg.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.r_[np.c_[xx, up], np.c_[xx[::-1], lo[::-1]]].round().astype(np.int32)], 1)
    outlines.append(((x0 + x1) / 2, m))
mm_px = np.linalg.norm(Cc - np.array(cam['centre'])) / F * 1000

# per eye in the picture: inside depth (mm), the iris circle, its radial colours, the white's colour, the catchlight
eyes = []
At = lab[..., 1] - 128
IRIS_L, SKIN_A = E('IRIS_L', 25), E('SKIN_A', 8)
for cx_, m in outlines:
    inside_mm = cv2.distanceTransform(m, cv2.DIST_L2, 5) * mm_px
    ok = inside_mm > EDGE + 0.2                                       # well away from the lids for the measurements
    # the iris: the largest dark, not reddish blob (on a dark-eyed test head Lt < 40 took the shaded white and the pink corners too, so the
    # circle came out twice the size, 84 px against 40, and the white's colour was read from the corners)
    # light irises (a brown iris at Lt 26-41 against a white at 75, with a* 10-17) fell outside Lt < IRIS_L and At < 9: only the
    # pupil was found and the iris was then taken for skin. The limit follows the eye: 0.8 of the Otsu split of the lightness inside
    # the outline (iris against white), never under IRIS_L; a light iris may be as reddish as IRIS_A_LIGHT
    otsu = cv2.threshold(np.clip(Lt[ok] * 2.55, 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 2.55
    T_iris = max(IRIS_L, 0.8 * otsu); A_iris = 9 if T_iris <= IRIS_L else E('IRIS_A_LIGHT', 20)
    # the skin test's a* limit sits above the white's own redness (a pinkish white at a* 5-9 was taken for skin): the bright
    # pixels' median a* + 4 when that white is pinkish (median a* over SKIN_A - 2), else SKIN_A
    bright = ok & (Lt > T_iris + 15)
    a_w = float(np.median(At[bright])) if bright.sum() > 20 else 0.0
    A_skin = a_w + 4 if a_w > SKIN_A - 2 else SKIN_A                    # only a pinkish white moves the limit
    print('EYE_PROJECT eye at x %.0f: iris Lt < %.0f (a* < %.0f), skin a* >= %.1f' % (cx_, T_iris, A_iris, A_skin))
    cand = (ok & (Lt < T_iris) & (At < A_iris)).astype(np.uint8)
    nlab, lbl, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    iris = lbl == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) if nlab > 1 else ok & (Lt < 40)
    ys, xs = np.nonzero(iris)
    # the iris circle: its dark pixels' extent per row near its middle row, and its centre column
    cyi = int(np.median(ys)); cxi = float(np.median(xs))
    rows = [r for r in range(cyi - 6, cyi + 7) if (iris[r]).any()]
    half = np.median([(np.nonzero(iris[r])[0].max() - np.nonzero(iris[r])[0].min()) / 2 for r in rows])
    cxi = np.median([(np.nonzero(iris[r])[0].max() + np.nonzero(iris[r])[0].min()) / 2 for r in rows])
    # the vertical centre: the lowest iris pixel minus the radius (the top is under the lid)
    low = np.max(ys[np.abs(xs - cxi) < 0.3 * half])
    cyi = low - half; cy_seen = cyi   # cy_seen: the iris centre as painted (before the circle is tucked under the lower lid)
    R = half
    # an iris that reaches the lower lid goes on under it: the lowest dark pixel is measured EDGE + 0.2 mm inside
    # the outline, so the circle stopped a few px above the lid and, with the white painted outside the iris, a white strip showed
    # under it. When the iris reaches within TUCK_NEAR mm of the lower line, its circle's bottom goes TUCK x R below the line
    lowline = np.nonzero(m[:, int(round(cxi))])[0].max()
    if (lowline - low) * mm_px < E('TUCK_NEAR', 0.6):
        cyi = lowline + E('TUCK', 0.12) * R - R
        print('EYE_PROJECT iris reaches the lower lid (%d px above the line): centre y %.1f -> %.1f' % (lowline - low, low - half, cyi))
    yy, xx_ = np.mgrid[0:RES, 0:RES]
    rr = np.hypot(xx_ - cxi, yy - cyi) / R
    # the lids' wet rims and the pink corners inside the outline (skin colour was baked into the white: 11-15% of
    # the outline was skin-coloured, painted onto the white below the iris): reddish (Lab a* >= SKIN_A), outside the iris, and
    # joined to the outline's edge. They are taken out of the outline, so the white and iris round them fill their place
    sk = ((m > 0) & (At >= A_skin) & (Lt >= 12) & (rr > 1.0)).astype(np.uint8)
    rim_ = (m > 0) & (cv2.erode(m, np.ones((5, 5), np.uint8)) == 0)
    n_, lb_ = cv2.connectedComponents(sk, connectivity=8)
    touch = np.unique(lb_[rim_ & (sk > 0)]); skin_in = np.isin(lb_, touch[touch > 0])
    skin_in = cv2.morphologyEx(skin_in.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    print('EYE_PROJECT eye at x %.0f: skin inside the outline %d px of %d (%.0f%%) left out' % (cx_, skin_in.sum(), (m > 0).sum(), 100 * skin_in.sum() / max((m > 0).sum(), 1)))
    m = (m > 0) & ~skin_in
    m = m.astype(np.uint8)
    inside_mm = cv2.distanceTransform(m, cv2.DIST_L2, 5) * mm_px
    # the input's painted catchlight: light, or bluish (Lab b* under -3: its reflection is a navy-white blob over the pupil). The
    # viewer adds a highlight that follows the light, so the painted one goes
    Bt = lab[..., 2] - 128
    # a light iris (up to Lt 41) is not a catchlight: the limit stays T_iris + 20 or more
    catch = (rr < 1.0) & ((Lt > max(E('CATCH_L', 55), T_iris + 20)) | (Bt < E('CATCH_B', -3))) & (m > 0)
    dil_ = int(E('CATCH_DIL', 7)); catch = cv2.dilate(catch.astype(np.uint8), np.ones((dil_, dil_), np.uint8)) > 0
    vis_iris = (inside_mm > EDGE) & (rr < 1.0) & ~catch
    bins = np.clip((rr * 20).astype(int), 0, 29)
    prof = np.zeros((30, 3)); 
    for b_ in range(30):
        sel = vis_iris & (bins == b_)
        if sel.sum() > 5:
            prof[b_] = np.median(timg[sel], 0)
    for b_ in range(30):                                              # empty bins take the nearest filled one
        if not prof[b_].any():
            f_ = [j for j in range(30) if prof[j].any()]
            prof[b_] = prof[min(f_, key=lambda j: abs(j - b_))]
    # the pupil: under the painted catchlight in the input, so the radial colours missed it (a flat brown patch). Its radius from the
    # dark pixels (L < 15) seen inside the iris (90th percentile of their distance, kept within 0.25..0.5 of the iris), its colour
    # their median; inside it the profile is the pupil, with a soft edge
    dark = (rr < 0.7) & (Lt < 15) & (inside_mm > EDGE)
    rp = float(np.clip(np.percentile(rr[dark], 75), 0.25, float(os.environ.get('PUPIL_MAX', 0.36)))) if dark.sum() > 10 else 0.32   # the 90th percentile took the dark limbal ring too (0.43-0.48, a pupil cut flat by the lid)
    pupil_c = np.median(timg[dark], 0) if dark.sum() > 10 else np.array([20, 20, 25])
    for b_ in range(30):
        r_mid = (b_ + 0.5) / 20
        k_ = float(np.clip((r_mid - rp) / 0.06 + 0.5, 0, 1))
        prof[b_] = pupil_c * (1 - k_) + prof[b_] * k_ if r_mid < rp + 0.06 else prof[b_]
    print('EYE_PROJECT pupil radius %.2f of the iris, colour %s' % (rp, pupil_c.round().tolist()))
    eyes_rp = rp
    white = (inside_mm > EDGE + 0.1) & (rr > 1.1) & (Lt > T_iris + 15) & (At < A_skin - 2) & (Ct < 14)   # the visible white as it is shaded (Lt > 50 kept only its brightest few)
    white_c = timg[white].mean(0) if white.sum() > 20 else np.array([215, 215, 220])
    valid = (inside_mm > EDGE) & (True if KEEP else ~catch)
    from scipy.ndimage import distance_transform_edt
    dout, (ny_, nx_) = distance_transform_edt(~valid, return_indices=True)
    eyes.append(dict(cx=cx_, inside=inside_mm, c=(cxi, cyi), c_seen=(cxi, cy_seen), R=R, prof=prof, white=white_c, catch=catch, dout=dout * mm_px, ny=ny_, nx=nx_, rp=eyes_rp, pupil_c=pupil_c))
    print('EYE_PROJECT eye at x %.0f: iris centre %.1f,%.1f r %.1f px (%.2f mm), white %s, catchlight px %d' % (cx_, cxi, cyi, R, R * mm_px, white_c.round().tolist(), catch.sum()))

if KEEP:   # the catchlights filled from the pixels round them, then read bilinearly
    # the catchlight lay over the pupil's upper part: inpainted from the iris round it, the pupil came out a small blob with its top
    # brown. Within the pupil's circle (radius rp, its colour from the input's dark pixels) the catchlight's place takes the pupil's
    # colour, with a 1 px soft edge; only the rest is inpainted
    cm = np.zeros(timg.shape[:2], np.uint8); yy_, xx_ = np.mgrid[0:timg.shape[0], 0:timg.shape[1]]
    timg = timg.astype(np.float32)
    for e in eyes:
        # the pupil's own circle, fitted to the edge of the input's dark pupil where the catchlight leaves it (the iris's centre and the
        # clipped radius drew a square-cornered patch past the painted pupil)
        rr0 = np.hypot(xx_ - e['c'][0], yy_ - e['c'][1]) / e['R']
        dk = ((Lt < 18) & (rr0 < 0.6) & (e['inside'] > EDGE)).astype(np.uint8)
        dk = cv2.morphologyEx(dk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        edge_ = (dk > 0) & (cv2.erode(dk, np.ones((3, 3), np.uint8)) == 0) & (cv2.dilate(e['catch'].astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
        ey_, ex_ = np.nonzero(edge_)
        pc = (e['c'][0], e['c'][1]); rp_px = e['rp'] * e['R']
        if len(ex_) > 12:   # algebraic circle fit: x^2 + y^2 + D x + E y + F = 0
            A_ = np.c_[ex_, ey_, np.ones(len(ex_))]; b_ = -(ex_ ** 2 + ey_ ** 2).astype(float)
            Dd, Ee, Ff = np.linalg.lstsq(A_, b_, rcond=None)[0]; pc = (-Dd / 2, -Ee / 2); rp_px = float(np.sqrt(max(pc[0] ** 2 + pc[1] ** 2 - Ff, 1)))
        # the fit is kept only when the pupil's edge is well seen: a big painted catchlight over most of the pupil left 14 edge px
        # and a circle 0.18 of the iris off to one side, and 50 px gave a lopsided blob. Otherwise a disc of PUPIL_R x the iris
        # radius at the painted iris centre
        ok_fit = (len(ex_) >= E('PUPIL_EDGE_MIN', 60) and 0.3 <= rp_px / e['R'] <= 0.55
                  and np.hypot(pc[0] - e['c_seen'][0], pc[1] - e['c_seen'][1]) < 0.35 * e['R'])
        if not ok_fit:
            pc = e['c_seen']; rp_px = E('PUPIL_R', 0.4) * e['R']
        print('EYE_PROJECT KEEP: pupil circle at %.1f,%.1f r %.1f px (%.2f of the iris; %d edge px%s)' % (pc[0], pc[1], rp_px, rp_px / e['R'], len(ex_), '' if ok_fit else '; fit not trusted, a plain disc'))
        rr_px = np.hypot(xx_ - pc[0], yy_ - pc[1])
        # the whole disc, not only where the catchlight was: the catchlight's square-cornered cut left steps against the
        # remaining painted pupil
        a_ = np.clip(rp_px + 0.7 - rr_px, 0, 1)
        timg = timg * (1 - a_[..., None]) + e['pupil_c'] * a_[..., None]
        cm |= (e['catch'] & (rr_px > rp_px + 0.5)).astype(np.uint8)
    timg = cv2.inpaint(np.clip(timg, 0, 255).astype(np.uint8), cm, 5, cv2.INPAINT_TELEA)
    for e in eyes: e['catch'] = np.zeros_like(e['catch'])
    print('EYE_PROJECT KEEP: catchlight px inpainted %d' % cm.sum())

def bilinear(im, x, y):
    x = np.clip(x, 0, im.shape[1] - 1.001); y = np.clip(y, 0, im.shape[0] - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    f = im.astype(np.float32)
    return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy) + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)


# the eyeball texels: rasterise the mesh's UV triangles to texel positions
d = np.load(eye_npz); co, tris, tuv = d['co'], d['tris'], d['tuv']
img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
H, W = img.shape[:2]
pos = np.full((H, W, 3), np.nan); nrm = np.zeros((H, W, 3))
fn = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
for t, uvt, n in zip(tris, tuv, fn):
    px_ = np.c_[uvt[:, 0] * W, (1 - uvt[:, 1]) * H]
    x0_, x1_ = int(np.floor(px_[:, 0].min())), int(np.ceil(px_[:, 0].max())); y0_, y1_ = int(np.floor(px_[:, 1].min())), int(np.ceil(px_[:, 1].max()))
    gx, gy = np.meshgrid(np.arange(max(x0_, 0), min(x1_ + 1, W)) + 0.5, np.arange(max(y0_, 0), min(y1_ + 1, H)) + 0.5)
    (ax, ay), (bx, by), (cx2, cy2) = px_
    den = (by - cy2) * (ax - cx2) + (cx2 - bx) * (ay - cy2)
    if abs(den) < 1e-12: continue
    l0 = ((by - cy2) * (gx - cx2) + (cx2 - bx) * (gy - cy2)) / den; l1 = ((cy2 - ay) * (gx - cx2) + (ax - cx2) * (gy - cy2)) / den; l2 = 1 - l0 - l1
    ins = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
    if not ins.any(): continue
    P = l0[..., None] * co[t[0]] + l1[..., None] * co[t[1]] + l2[..., None] * co[t[2]]
    iy, ix = (gy - 0.5).astype(int)[ins], (gx - 0.5).astype(int)[ins]
    old = pos[iy, ix]
    # where two surfaces share a texel keep the front-most one (smaller y: the face looks to -y)
    put = np.isnan(old[:, 1]) | (P[ins][:, 1] < old[:, 1])
    pos[iy[put], ix[put]] = P[ins][put]; nrm[iy[put], ix[put]] = n
ys, xs = np.nonzero(~np.isnan(pos[..., 0]))
Pt = pos[ys, xs]; Nt = nrm[ys, xs]
q = (Pt - Cc) @ Rc; dep = -q[:, 2]
px = np.c_[RES / 2 + F * q[:, 0] / dep, RES / 2 - F * q[:, 1] / dep]
view = Cc - Pt; view /= np.linalg.norm(view, axis=1, keepdims=True)
front = (Nt * view).sum(1) > 0.0   # facing the camera (the side of the ball in the corners too; a front-half cut left them the base white)
out = img.copy()
ch = out[..., :3].astype(np.float32)
n_set = 0
for e in eyes:
    side = np.abs(px[:, 0] - e['cx']) < np.abs(px[:, 0] - [o_[0] for o_ in outlines if o_[0] != e['cx']][0])
    sel = front & side & (px[:, 0] >= 0) & (px[:, 0] < RES) & (px[:, 1] >= 0) & (px[:, 1] < RES)
    k = np.flatnonzero(sel)
    ix, iy = px[k, 0].astype(int), px[k, 1].astype(int)
    rr = np.hypot(px[k, 0] - e['c'][0], px[k, 1] - e['c'][1]) / e['R']
    model = np.where((rr < 1.0)[:, None], e['prof'][np.clip((rr * 20).astype(int), 0, 29)], e['white'])
    # the iris edge softened over 4% of the radius
    ew = np.clip((rr - 0.98) / 0.06, 0, 1)[:, None]
    model = np.where((rr < 1.04)[:, None], e['prof'][np.clip((rr * 20).astype(int), 0, 29)] * (1 - ew) + e['white'] * ew, e['white'])
    tri_c = bilinear(timg, px[k, 0] - 0.5, px[k, 1] - 0.5) if KEEP else timg[iy, ix].astype(np.float32)
    w = np.clip((e['inside'][iy, ix] - EDGE) / BLEND, 0, 1) * ~e['catch'][iy, ix]
    # outside the shrunk outline: the nearest input colour inside it, fading into the model over NEAR mm (a straight hand-over
    # drew the outline's edge across the iris)
    near_c = timg[e['ny'][iy, ix], e['nx'][iy, ix]].astype(np.float32)
    t_ = np.clip(e['dout'][iy, ix] / NEAR, 0, 1)[:, None]
    fill = near_c * (1 - t_) + model * t_
    fill = np.where(e['catch'][iy, ix][:, None], model, fill)   # the catchlight's place: the eye's own radial colours
    # the pupil round: within its radius (+ the soft edge) the model only, whatever the input texture shows there (the catchlight's cut-out
    # left it the shape of the catchlight)
    in_p = np.clip((e['rp'] + 0.1 - rr) / 0.06, 0, 1)[:, None] * (0 if KEEP else 1)
    w = w * (1 - in_p[:, 0])
    # outside the iris only the white (WHITE_ONLY, default on): the input texture shows
    # pink spots, lid shadows and the wet rims onto the white, and taking out only the skin joined to the lids left a red blob by
    # the inner corner. Past the iris's edge (softened over 6% of its radius) the eye's own white colour, whatever the input texture shows
    if E('WHITE_ONLY', 1):
        o_ = np.clip((rr - 0.98) / 0.06, 0, 1)
        w = w * (1 - o_); fill = fill * (1 - o_[:, None]) + model * o_[:, None]
    col = tri_c * w[:, None] + fill * (1 - w[:, None])
    ch[ys[k], xs[k]] = col
    n_set += len(k)
out[..., :3] = np.clip(ch, 0, 255).astype(np.uint8)
cv2.imwrite(dst, out)
print('EYE_PROJECT texels set %d, wrote %s' % (n_set, dst))
