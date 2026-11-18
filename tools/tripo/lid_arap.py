"""The lids' visible edges put on the Tripo head's own eye outline by an as-rigid-as-possible deformation.   any face
python lid_arap.py OUT_DIR          (cwd: tripo/; run.sh calls it with LID_FIT=1; CHECK=1 only draws what is extracted)

The view: lid_render.py VIEW=tripo, the viewer's "Tripo と同じ" camera. The target: MediaPipe's eye contour on the Tripo head
rendered from it, one smooth line per lid (upper: cubic, lower: a quadratic Bezier bow), the ends on the corners. A person
approves these lines first (run.sh; CHECK=1 writes lidfit/check.jpg, check_corners.jpg and check_lines.txt).

Why ARAP (Codex, 2026-11-09, consult/lid_corner_answer.md): moving lid vertices up and down only could not shorten MPFB's
longer outer corner; squeezing the slit shut tore the corner into fragments (lid_fit.py, 30 rounds, max gap stuck at 3.57 mm).
How:
1. The margin, once, on the mesh before any move: Human's vertices facing out next to ones facing the eyeball, round each
   eyeball, upper / lower by the line joining its x extremes; per 0.8 mm of width the front-most one (the edge one sees). Its
   two ends are the corners.
2. Each margin vertex gets its place on the target line by its share of the margin's width in the image (the corners go to the
   target's corners, so the corner moves sideways too); in 3D the point on that pixel's ray nearest the vertex (its depth kept).
3. The lid region (within REGION of the margin, front of the eyeball's centre + BACK, the lid's thickness included) is solved
   as-rigid-as-possible (local rotations / global least squares, cotangent weights clamped positive) against its shape before
   the move, the margin vertices held softly on their targets, the region's border fixed. The targets are reached in STEPS
   steps; after each, vertices inside the eyeball (radius + CLEAR) go out along the radius. Folds are counted every step.
4. Then the visible opening is rendered; where the eyeball shows outside the lines (past a corner the slit between the lids is
   still open), the skin bounding the slit above and below is drawn to its middle height, column by column, and the region
   solved again (COVER rounds). Bringing that skin in front of the ball along its view ray (tried first) changed nothing: it
   was a slit, not skin behind the ball. Moving the margin
   targets by the gap left per column (tried first) made it worse every round (p90 0.90 -> 2.56 mm, folds 6 -> 23).
Writes OUT_DIR/lid_delta.npz (seam_apply.py format; the lashes follow their nearest moved lid vertex) and lidfit/arap_<k>.jpg.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mpfb'))
from lm import detect  # noqa: E402

O = os.path.abspath(sys.argv[1])
E_ = lambda k, d: float(os.environ.get(k, d))
REGION, BACK, CLEAR = E_('REGION', 15) / 1000, E_('BACK', 8) / 1000, E_('CLEAR', 0.15) / 1000
STEPS, ITERS, COVER, W_H = int(E_('STEPS', 6)), int(E_('ITERS', 8)), int(E_('COVER', 4)), E_('W_H', 1000.0)
MAXGAP = E_('MAXGAP', 0.3)
CHECK = os.environ.get('CHECK')
T = os.path.dirname(os.path.abspath(__file__))
L = f'{O}/lidfit'; W = f'{L}/warp'
os.makedirs(W, exist_ok=True)
B = os.environ.get('BLENDER', '/Applications/Blender.app/Contents/MacOS/Blender')
M = f'{T}/../mpfb'


def run(cmd, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=600)
    if r.returncode != 0 or 'Traceback' in r.stdout + r.stderr:
        print((r.stdout + r.stderr)[-3000:]); raise SystemExit('failed: ' + cmd)


def render(tag, tripo=False):
    run(f'{B} -b {W}/cur.blend -P {T}/lid_render.py -- {W}/{tag} {O}/tripo_in_mpfb.npz {O}/tripo_basecolor.png',
        env={'MASK_ONLY': '1', 'VIEW': 'tripo', **({'TRIPO_TOO': '1'} if tripo else {})})


# ---- the camera and the approved lines ----
run(f'cp {O}/warped_e0.blend {W}/cur.blend')
render('v0', tripo=True)
cam = json.load(open(f'{W}/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1]; F = (RES / 2) / np.tan(cam['fov'] / 2)
MM_PX = np.linalg.norm(Cc - np.array(cam['centre'])) / F


def project(P):
    q = (P - Cc) @ Rc
    return np.c_[RES / 2 + F * q[:, 0] / -q[:, 2], RES / 2 - F * q[:, 1] / -q[:, 2]]


def ray(px):
    d = (Rc @ np.c_[(px[:, 0] - RES / 2) / F, -(px[:, 1] - RES / 2) / F, -np.ones(len(px))].T).T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


lm = detect([f'{W}/v0_tripo.png'])[0][:, :2]
RINGS = (((33, 246, 161, 160, 159, 158, 157, 173, 133), (33, 7, 163, 144, 145, 153, 154, 155, 133)),
         ((362, 398, 384, 385, 386, 387, 388, 466, 263), (362, 382, 381, 380, 374, 373, 390, 249, 263)))


def arc(P, deg):
    w = np.ones(len(P)); w[[0, -1]] = 10
    return np.polyfit(P[:, 0], P[:, 1], deg, w=w)


def bow(P):
    """One bow from corner to corner whose deepest point may sit anywhere along it (a quadratic Bezier, its control point
    fitted): the lower lid rises fast by the inner corner, slowly by the outer, and never turns over."""
    from scipy.optimize import minimize
    P = P[np.argsort(P[:, 0])]; a, b = P[0], P[-1]
    t = np.linspace(0, 1, 400)[:, None]
    curve = lambda c: (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b
    cost = lambda c: (np.min(np.linalg.norm(curve(c)[None] - P[1:-1, None], axis=2), 1) ** 2).sum()
    c = minimize(cost, np.array([(a[0] + b[0]) / 2, 2 * P[1:-1, 1].mean() - (a[1] + b[1]) / 2]), method='Nelder-Mead').x
    c[0] = np.clip(c[0], a[0], b[0])     # the control point between the ends, or the curve runs past a corner and back
    C = curve(c)
    return lambda x: np.interp(x, C[:, 0], C[:, 1])


SCLERA_FRAC = float(os.environ.get("SCLERA_FRAC", 0.25))   # 0.5 stopped 1.5 mm short of the white's tip (the slit is thin and shaded there)


def caruncle_end(up, lo, img):
    """MediaPipe's inner corner is the caruncle's tip, but the base mesh has no caruncle: inside the opening there the eyeball's
    white showed as a wedge where Tripo painted pink. Walk from that tip towards the eye's middle along the
    opening; the corner is where SCLERA_FRAC of the column between the lids is sclera (light, little colour) for 3 pixels running.
    up / lo: the ring's points, inner corner last. Returns the new inner corner, or None."""
    tip, other = up[-1], up[0]
    step = np.sign(other[0] - tip[0])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / [180, 255, 255]
    fu0, fl0 = arc(up, 2), arc(lo, 2)
    run_ = 0
    for k in range(1, int(abs(other[0] - tip[0]) * 0.4)):
        x = tip[0] + step * k
        a, b = sorted((np.polyval(fu0, x), np.polyval(fl0, x)))
        if b - a < 2:
            continue
        ys = np.arange(int(a + 0.2 * (b - a)), int(b - 0.2 * (b - a)) + 1)
        px = hsv[ys, int(round(x))]
        sclera = ((px[:, 1] < 0.20) & (px[:, 2] > 0.55)).mean()
        run_ = run_ + 1 if sclera > SCLERA_FRAC else 0
        if run_ == 3:
            xc = x - step * 2
            return np.array([xc, (np.polyval(fu0, xc) + np.polyval(fl0, xc)) / 2])
    return None


tripo_img = cv2.imread(f'{W}/v0_tripo.png')


def outer_tip(up, lo, img):
    """MediaPipe's outer corner can sit past and below the white's tip, on the line where the lash line and the waterline run on
    (on one test head the left eye's outer corner came out 1.1 mm low). The corner goes to the tip of Tripo's painted white: of the
    white pixels (Lab chroma < 9, L > 45 of 100) within 30 px of the corner, the outermost along the eye's axis (the median of the
    outer 3%). up / lo: the ring's points, outer corner first. Returns the new corner or None."""
    o, i = up[0], up[-1]
    d = (o - i) / np.linalg.norm(o - i)
    lab_ = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_ = lab_[..., 0] * 100 / 255; C_ = np.hypot(lab_[..., 1] - 128, lab_[..., 2] - 128)
    y0_, y1_, x0_, x1_ = int(o[1]) - 30, int(o[1]) + 31, int(o[0]) - 30, int(o[0]) + 31
    yy, xx_ = np.mgrid[y0_:y1_, x0_:x1_]
    m = (C_[y0_:y1_, x0_:x1_] < 9) & (L_[y0_:y1_, x0_:x1_] > 45)
    # only the white of this eye: inside the band between the two lines' chords, not skin highlights out on the cheek
    P = np.c_[xx_[m], yy[m]].astype(float)
    if len(P) < 20:
        return None
    t = (P - i) @ d
    sel = t >= np.percentile(t, 97)
    tip = np.median(P[sel], 0)
    return tip
eyes = []
for up_ids, lo_ids in RINGS:
    up, lo = lm[list(up_ids)], lm[list(lo_ids)]
    if os.environ.get('OUTER_TIP', '0') == '1':   # off: near the corner the white, the pink and the skin overlap in colour (chroma 12-20 for all three on one test head)
        outer_first = up_ids[0] in (33, 263)                 # 33 / 263: MediaPipe's outer corners
        u_, l_ = (up, lo) if outer_first else (up[::-1], lo[::-1])
        tip = outer_tip(u_, l_, tripo_img)
        if tip is not None and np.linalg.norm(tip - u_[0]) > 2:
            dax = (u_[0] - u_[-1]) / np.linalg.norm(u_[0] - u_[-1])
            keep_u = ((u_ - tip) @ dax) < 0; keep_l = ((l_ - tip) @ dax) < 0
            print('target outer corner: %.0f,%.0f -> white tip %.0f,%.0f (%.1f px)' % (*u_[0], *tip, np.linalg.norm(tip - u_[0])))
            u_ = np.vstack([tip, u_[keep_u]]); l_ = np.vstack([tip, l_[keep_l]])
        up, lo = (u_, l_) if outer_first else (u_[::-1], l_[::-1])
    # a person's correction of the corners at the check (lidfit/corner_adjust.txt: "EYE CORNER DX DY" per line, EYE 0 = the
    # screen-left eye, CORNER outer|inner, px in this camera; e.g. an outer corner that came out too low)
    adj_f = f'{L}/corner_adjust.txt'
    if os.path.exists(adj_f):
        ei_ = len(eyes)
        for row in open(adj_f):
            row = row.split('#')[0].split()
            if len(row) == 4 and int(row[0]) == ei_:
                outer_first = up_ids[0] in (33, 263)
                which = 0 if (row[1] == 'outer') == outer_first else -1
                pt = up[which] + [float(row[2]), float(row[3])]
                dax = (up[which] - up[-1 - which]) / np.linalg.norm(up[which] - up[-1 - which])
                sx = np.sign(dax[0])                         # beyond the new corner along x: dropped (along the axis kept a point past it)
                # points past it, and its old neighbours within NEAR px (they belonged to the old corner and bent the curves), go
                near_ = float(os.environ.get('ADJ_NEAR', 12))
                keep_u = (up[:, 0] - pt[0]) * sx < -near_; keep_l = (lo[:, 0] - pt[0]) * sx < -near_
                keep_u[-1 - which] = True; keep_l[-1 - which] = True
                up, lo = up.copy(), lo.copy()
                up[which] = pt; lo[which] = pt
                keep_u[which] = True; keep_l[which] = True   # the moved corner itself (which may be -1)
                up, lo = up[keep_u], lo[keep_l]
                print('target %s corner of eye %d moved by %s px by hand -> %.0f,%.0f' % (row[1], ei_, row[2:], *pt))
    if os.environ.get('CARUNCLE', '0') == '1':   # off by default: on one test head it cut 3.7 mm where the caruncle is ~1 mm (shaded white read as not white), so the white shrank and the eye looked small
        inner_last = up_ids[-1] in (133, 362)                # 133 / 362: MediaPipe's inner corners
        if not inner_last:
            up, lo = up[::-1], lo[::-1]
        cc = caruncle_end(up, lo, tripo_img)
        if cc is not None:
            keep_u = (up[:, 0] - cc[0]) * np.sign(up[0, 0] - cc[0]) > 0
            keep_l = (lo[:, 0] - cc[0]) * np.sign(lo[0, 0] - cc[0]) > 0
            print('target inner corner: caruncle %.0f,%.0f -> sclera %.0f,%.0f (%.1f px)' % (*up[-1], *cc, np.linalg.norm(cc - up[-1])))
            up, lo = np.vstack([up[keep_u], cc]), np.vstack([lo[keep_l], cc])
        if not inner_last:
            up, lo = up[::-1], lo[::-1]
    cu = arc(up, 3)
    xx = np.linspace(up[:, 0].min(), up[:, 0].max(), 50)
    if np.ptp(np.sign(np.polyval(np.polyder(cu, 2), xx))) > 0:
        cu = arc(up, 2)
    x0, x1 = sorted((up[0, 0], up[-1, 0]))
    fu0, fl0 = (lambda c_: lambda x: np.polyval(c_, x))(cu), bow(lo)
    if os.environ.get('INSET', '1') == '1':
        # MediaPipe's contour runs on the lash line above and under the waterline below: inside it Tripo painted the lid's grey rim
        # (above) and the pink waterline (below), and the eyeball only shows within those. Fitting the opening to the contour bared
        # the eyeball over both and the eyes looked wide open. Per eye, measured in the picture:
        # du = rows from the upper line down to the iris (lightness falls below 3/4 of the rim's), over the middle of the eye;
        # dl = rows from the lower line up to where it is no longer pink (a* 5 under the waterline's) or reaches the iris.
        # PER COLUMN (the white under the lid line should stay eyeball white, not turn to skin): one inset measured over the
        # iris and used across the whole width covered the white Tripo painted right under the line by the corners. Now each column
        # goes in only to where Tripo's eyeball begins: the white (chroma < 9, L > 30 of 100) or the iris (L < 28); below, past the
        # pink waterline to the white (chroma < 9, L > 45) or the iris (L < 35). Smoothed along the eye (median, then mean, 9 px).
        labi = cv2.cvtColor(tripo_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        Li = labi[..., 0] * 100 / 255; Ci = np.hypot(labi[..., 1] - 128, labi[..., 2] - 128)
        cols = np.arange(int(np.ceil(x0)), int(np.floor(x1)) + 1)
        du_c, dl_c = np.zeros(len(cols)), np.zeros(len(cols))
        for i_, x in enumerate(cols):
            yu = int(round(fu0(np.array([x]))[0])); yl = int(round(fl0(np.array([x]))[0]))
            j = next((j for j in range(0, 15) if yu + j < yl and ((Ci[yu + j, x] < 9 and Li[yu + j, x] > 30) or Li[yu + j, x] < 28)), 0)
            du_c[i_] = j
            j = next((j for j in range(0, 15) if yl - j > yu and ((Ci[yl - j, x] < 9 and Li[yl - j, x] > 45) or Li[yl - j, x] < 35)), 0)
            dl_c[i_] = j
        from scipy.ndimage import median_filter, uniform_filter1d
        du_c = np.clip(uniform_filter1d(median_filter(du_c, 9, mode='nearest'), 9, mode='nearest'), 0, 12)
        dl_c = np.clip(uniform_filter1d(median_filter(dl_c, 9, mode='nearest'), 9, mode='nearest'), 0, 12)
        print('target inset per column: upper median %.1f (max %.1f) px, lower median %.1f (max %.1f) px' % (np.median(du_c), du_c.max(), np.median(dl_c), dl_c.max()))
        def inset(fu_, fl_, cols_, du_, dl_):
            def amt(x, d_, d_other):
                x = np.asarray(x, float)
                dd = np.interp(x, cols_, d_); do = np.interp(x, cols_, d_other)
                hgt = np.maximum(fl_(x) - fu_(x), 0)
                return np.minimum(dd, 0.4 * hgt * dd / np.maximum(dd + do, 1e-6))   # by the corners at most 40% of the opening
            return (lambda x: fu_(x) + amt(x, du_, dl_)), (lambda x: fl_(x) - amt(x, dl_, du_))
        fu0, fl0 = inset(fu0, fl0, cols, du_c, dl_c)
        # one smooth curve each again (the per-column offsets wobble): upper a cubic, lower the bow, through the inset points with the
        # corners held
        xs_ = np.linspace(x0, x1, 41)
        Pu = np.c_[xs_, fu0(xs_)]; Pl = np.c_[xs_, fl0(xs_)]
        cu2 = arc(Pu, 3)
        if np.ptp(np.sign(np.polyval(np.polyder(cu2, 2), xs_))) > 0:
            cu2 = arc(Pu, 2)
        fl0 = bow(Pl)
        # the cubic misses the corners by a pixel or two: a straight correction puts both its ends on the lower line's ends
        e0 = float(fl0(np.array([x0]))[0] - np.polyval(cu2, x0)); e1 = float(fl0(np.array([x1]))[0] - np.polyval(cu2, x1))
        fu0 = (lambda c_, a_, b_: lambda x: np.polyval(c_, x) + a_ + (b_ - a_) * (np.asarray(x, float) - x0) / (x1 - x0))(cu2, e0, e1)
    # a person's lift of the upper line at the check (UP_RAISE px at the middle, 0 at the corners, 4t(1-t);
    # e.g. when the upper line should sit a little higher)
    ur_ = E_('UP_RAISE', 0)
    if ur_:
        fu0 = (lambda f_, a_, b_: lambda x: f_(x) - ur_ * 4 * np.clip((np.asarray(x, float) - a_) / (b_ - a_), 0, 1) * (1 - np.clip((np.asarray(x, float) - a_) / (b_ - a_), 0, 1)))(fu0, x0, x1)
    eyes.append(dict(fu=fu0, fl=fl0, x0=x0, x1=x1))
    print('target eye: px %.0f..%.0f' % (x0, x1))

mk = cv2.imread(f'{W}/v0_mask.png')
eyeball_px = (mk[..., 2] > 200) & (mk[..., 0] > 200) & (mk[..., 1] < 80)
if CHECK:
    tri_img = cv2.imread(f'{W}/v0_tripo.png'); ours = np.full_like(tri_img, 200); ours[eyeball_px] = (255, 0, 255)
    for e in eyes:
        xx = np.linspace(e['x0'], e['x1'], 60)
        for im in (tri_img, ours):
            for f_ in (e['fu'], e['fl']):
                cv2.polylines(im, [np.c_[xx, f_(xx)].round().astype(np.int32)], False, (0, 130, 255), 2)
    ys_, xs_ = np.nonzero(eyeball_px)
    y0_, y1_, x0_, x1_ = max(ys_.min() - 60, 0), ys_.max() + 60, max(xs_.min() - 60, 0), xs_.max() + 60
    cv2.imwrite(f'{L}/check.jpg', cv2.resize(np.vstack([tri_img[y0_:y1_, x0_:x1_], ours[y0_:y1_, x0_:x1_]]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    tiles = []
    for e in eyes:
        for xc in (e['x0'], e['x1']):
            yc = int(e['fu'](np.array([xc]))[0]); xc = int(xc)
            tiles.append(cv2.resize(tri_img[yc - 25:yc + 25, xc - 35:xc + 35], None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(f'{L}/check_corners.jpg', np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    with open(f'{L}/check_lines.txt', 'w') as fh:
        for e in eyes:
            xx = np.linspace(e['x0'], e['x1'], 25)
            fh.write('%d %d\n' % (round(e['x0']), round(e['x1'])))
            fh.write(' '.join('%d' % round(v) for v in e['fu'](xx)) + '\n' + ' '.join('%d' % round(v) for v in e['fl'](xx)) + '\n')
    print('CHECK wrote', f'{L}/check.jpg')
    raise SystemExit

# ---- the mesh before any move ----
run(f'{B} --background {W}/cur.blend --python {M}/export_uv.py -- {W}/uv.npz')
u = np.load(f'{W}/uv.npz')
V0, N0, EC, tv = u['co'].astype(np.float64), u['nrm'].astype(np.float64), u['eye_co'], u['tv']
EDG = np.unique(np.sort(np.r_[tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]], 1), axis=0)
nV = len(V0)
balls = {}
for s_ in (1, -1):
    b_ = EC[np.sign(EC[:, 0]) == s_]
    bc = np.linalg.lstsq(np.c_[2 * b_, np.ones(len(b_))], (b_ ** 2).sum(1), rcond=None)[0]
    balls[s_] = (bc[:3], np.sqrt(bc[3] + (bc[:3] ** 2).sum()))

# the margin and its targets' order
handles = []                        # (vertex, eye index, 'u'|'l')
region = np.zeros(nV, bool)
for ei, e in enumerate(eyes):
    s_ = 1 if project(balls[1][0][None])[0, 0] < project(balls[-1][0][None])[0, 0] else -1
    s_ = s_ if ei == 0 else -s_      # eyes[0] is the screen-left eye
    c, r = balls[s_]
    d = V0 - c; dist = np.linalg.norm(d, axis=1)
    near = dist < r + 0.004
    tow = (N0 * d).sum(1) / np.maximum(dist, 1e-9)
    inn, out = near & (tow < 0), near & (tow >= 0) & (V0[:, 1] < c[1])
    rim = np.unique(np.r_[EDG[out[EDG[:, 0]] & inn[EDG[:, 1]], 0], EDG[out[EDG[:, 1]] & inn[EDG[:, 0]], 1]])
    a_, b_ = V0[rim[np.argmin(V0[rim, 0])]], V0[rim[np.argmax(V0[rim, 0])]]
    line = a_[2] + (b_[2] - a_[2]) * (V0[rim, 0] - a_[0]) / (b_[0] - a_[0])
    for part, sel in (('u', rim[V0[rim, 2] >= line - 1e-5]), ('l', rim[V0[rim, 2] <= line + 1e-5])):
        bins = np.round((V0[sel, 0] - a_[0]) / 0.0008).astype(int)
        for bn in np.unique(bins):
            k = sel[bins == bn]
            handles.append((int(k[np.argmin(V0[k, 1])]), ei, part))
    eyes[ei]['inn'] = inn
    dmin = np.min(np.linalg.norm(V0[:, None] - V0[rim][None], axis=2), 1) if len(rim) < 400 else None
    region |= (dmin < REGION) & (V0[:, 1] < c[1] + BACK)
    eyes[ei]['side'] = s_
hv = np.array([h[0] for h in handles])
hv, first = np.unique(hv, return_index=True)          # a corner vertex in both chains: once
hinfo = [handles[i] for i in first]
print('ARAP margin vertices %d, region %d' % (len(hv), region.sum()))


def targets(P, extra=None):
    """3D targets of the margin vertices for the current positions P (their depth kept)."""
    px = project(P[hv])
    Tt = np.zeros((len(hv), 3))
    for ei, e in enumerate(eyes):
        for part in 'ul':
            m_ = np.array([(h[1] == ei and h[2] == part) or (h[1] == ei and hv[j] in corner_v[ei]) for j, h in enumerate(hinfo)])
            if not m_.any():
                continue
            xs = px0[m_, 0]
            f = (xs - xs.min()) / max(np.ptp(xs), 1e-9)          # share of the margin's width, from its own state before the move
            xt = e['x0'] + f * (e['x1'] - e['x0'])
            yt = (e['fu'] if part == 'u' else e['fl'])(xt)
            tp = np.c_[xt, yt]
            if extra is not None:
                tp = tp + extra[m_]
            dr = ray(tp)
            t = ((P[hv[m_]] - Cc) * dr).sum(1)
            Tt[m_] = Cc + dr * t[:, None]
            # a target inside the eyeball: where that ray first meets the ball (radius + 0.5 mm), in front of it
            c, r = balls[e['side']]
            oc = Cc - c; bq = (dr * oc).sum(1); cq = (oc ** 2).sum() - (r + 0.0005) ** 2
            disc = bq ** 2 - cq
            ins = (np.linalg.norm(Tt[m_] - c, axis=1) < r + 0.0005) & (disc > 0)
            tf = -bq - np.sqrt(np.maximum(disc, 0))
            sub = Tt[m_]; sub[ins] = Cc + dr[ins] * tf[ins, None]; Tt[m_] = sub
    return Tt


px0 = project(V0[hv])
corner_v = []
for ei in range(len(eyes)):
    idx = [j for j, h in enumerate(hinfo) if h[1] == ei]
    xs = px0[idx, 0]
    corner_v.append({hv[idx[int(np.argmin(xs))]], hv[idx[int(np.argmax(xs))]]})

# ---- ARAP over the region ----
in_tri = np.zeros(nV, bool); in_tri[tv.ravel()] = True
region &= in_tri                                      # MPFB's helper points belong to no face: left alone (solved alone they went to the origin and took the lashes with them)
free = np.flatnonzero(region)
fid = -np.ones(nV, int); fid[free] = np.arange(len(free))
tri_r = tv[region[tv].any(1)]
def cot_weights(P):
    w = {}
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u_ = P[tri_r[:, a]] - P[tri_r[:, c]]; v_ = P[tri_r[:, b]] - P[tri_r[:, c]]
        cot = (u_ * v_).sum(1) / np.maximum(np.linalg.norm(np.cross(u_, v_), axis=1), 1e-12)
        for i, j, cc in zip(tri_r[:, a], tri_r[:, b], cot):
            key = (min(i, j), max(i, j)); w[key] = w.get(key, 0) + 0.5 * cc
    keys = np.array(list(w.keys())); vals = np.maximum(np.array(list(w.values())), 1e-3)
    return keys, vals
EK, EW = cot_weights(V0)
inreg = region[EK[:, 0]] | region[EK[:, 1]]
EK, EW = EK[inreg], EW[inreg]
nb = [[] for _ in range(nV)]
for (i, j), w_ in zip(EK, EW):
    nb[i].append((j, w_)); nb[j].append((i, w_))
h_free = fid[hv]
# the system: L (free x free) + W_H at the margin vertices; fixed neighbours go to the right side
rows, cols_, vals = [], [], []
for (i, j), w_ in zip(EK, EW):
    for a, b in ((i, j), (j, i)):
        if fid[a] >= 0:
            rows.append(fid[a]); cols_.append(fid[a]); vals.append(w_)
            if fid[b] >= 0:
                rows.append(fid[a]); cols_.append(fid[b]); vals.append(-w_)
Lm = sp.csr_matrix((vals, (rows, cols_)), shape=(len(free), len(free)))
def solver(hidx):
    Hm = sp.csr_matrix((np.full(len(hidx), W_H), (fid[hidx], fid[hidx])), shape=(len(free), len(free)))
    return factorized((Lm + Hm + sp.identity(len(free)) * 1e-9).tocsc())
solve_m = solver(hv)


def arap(P, Tt, iters=ITERS, hidx=None, solve=None):
    hidx = hv if hidx is None else hidx
    solve = solve_m if solve is None else solve
    P = P.copy()
    for _ in range(iters):
        R = np.tile(np.eye(3), (nV, 1, 1))
        S = np.zeros((nV, 3, 3))
        for (i, j), w_ in zip(EK, EW):
            e0 = V0[i] - V0[j]; e1 = P[i] - P[j]
            o = w_ * np.outer(e0, e1); S[i] += o; S[j] += o
        for i in free:
            U, _, Vt = np.linalg.svd(S[i])
            Ri = Vt.T @ U.T
            if np.linalg.det(Ri) < 0:
                Vt[-1] *= -1; Ri = Vt.T @ U.T
            R[i] = Ri
        rhs = np.zeros((len(free), 3))
        for (i, j), w_ in zip(EK, EW):
            e0 = V0[i] - V0[j]
            if fid[i] >= 0:
                rhs[fid[i]] += 0.5 * w_ * (R[i] + R[j]) @ e0
                if fid[j] < 0:
                    rhs[fid[i]] += w_ * P[j]
            if fid[j] >= 0:
                rhs[fid[j]] -= 0.5 * w_ * (R[i] + R[j]) @ e0
                if fid[i] < 0:
                    rhs[fid[j]] += w_ * P[i]
        rhs[fid[hidx]] += W_H * Tt
        P[free] = np.c_[solve(rhs[:, 0]), solve(rhs[:, 1]), solve(rhs[:, 2])]
    # nothing inside the eyeballs
    for s_, (c, r) in balls.items():
        dd = P[free] - c; dist = np.linalg.norm(dd, axis=1)
        inside = (dist < r + CLEAR) & (np.sign(P[free, 0]) == s_)
        P[free[inside]] = c + dd[inside] / dist[inside, None] * (r + CLEAR)
    return P


tri_n = lambda P: np.cross(P[tv[:, 1]] - P[tv[:, 0]], P[tv[:, 2]] - P[tv[:, 0]])
n_ref = tri_n(V0); a_ref = np.linalg.norm(n_ref, axis=1)
def folds(P):
    n = tri_n(P); a = np.linalg.norm(n, axis=1)
    reg_t = region[tv].any(1)
    flip = ((n * n_ref).sum(1) < 0) & reg_t
    crush = (a < 0.15 * a_ref) & reg_t
    return int(flip.sum()), int(crush.sum())


def measure(tag):
    """the visible opening, column by column, against the lines: per margin vertex the px gap in its column; the max gap mm."""
    mk = cv2.imread(f'{W}/{tag}_mask.png')
    opening = (mk[..., 2] > 200) & (mk[..., 0] > 200) & (mk[..., 1] < 80)
    gaps, pic = [], cv2.imread(f'{W}/v0_tripo.png')
    cnt, _ = cv2.findContours(opening.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(pic, cnt, -1, (255, 0, 255), 1)
    col_gap = {}
    for ei, e in enumerate(eyes):
        xx = np.linspace(e['x0'], e['x1'], 60)
        for f_ in (e['fu'], e['fl']):
            cv2.polylines(pic, [np.c_[xx, f_(xx)].round().astype(np.int32)], False, (0, 200, 0), 1)
        span = e['x1'] - e['x0']; cy = int((e['fu'](np.array([(e['x0'] + e['x1']) / 2]))[0] + e['fl'](np.array([(e['x0'] + e['x1']) / 2]))[0]) / 2)
        for c_ in range(int(e['x0'] - 0.2 * span), int(e['x1'] + 0.2 * span)):
            rows = np.flatnonzero(opening[cy - 120:cy + 120, c_])
            inside = e['x0'] <= c_ <= e['x1']
            if inside:
                tu, tl = e['fu'](np.array([c_]))[0], e['fl'](np.array([c_]))[0]
                if len(rows):
                    mu, ml = cy - 120 + rows.min(), cy - 120 + rows.max()
                    col_gap[(ei, c_)] = (tu - mu, tl - ml)
                    gaps += [abs(tu - mu), abs(tl - ml)]
                else:
                    gaps += [abs(tl - tu)]          # the column should be open and is not
            elif len(rows):
                gaps += [rows.max() - rows.min() + 1]   # open past a corner
    ys_, xs_ = np.nonzero(opening)
    crop = pic[max(ys_.min() - 80, 0):ys_.max() + 80, max(xs_.min() - 80, 0):xs_.max() + 80]
    return np.array(gaps) * MM_PX * 1000, col_gap, crop


def apply_and_render(P, tag):
    total = P - V0
    z = np.zeros((1, 3))
    np.savez(f'{W}/step.npz', delta=total, pos1=z, off1=z, pos2=z, off2=z)
    run(f'cp {O}/warped_e0.blend {W}/cur.blend')
    run(f'{B} --background {W}/cur.blend --python {T}/seam_apply.py -- {W}/step.npz {W}/cur.blend')
    render(tag)


# ---- reach the targets in steps ----
P = V0.copy()
T_full = targets(V0)
# the lid's back edge (the vertices facing the eyeball next to each margin vertex) goes with it: onto the same view ray, on the
# ball (+ INNER_GAP). Without it a raised margin left the lid's thickness hanging below it, seen from the camera above as a grey
# band between the lashes and the eyeball (as marked in a review). INNER=0 turns it off.
hv_all, T_all = hv, T_full
if os.environ.get('INNER', '1') == '1':
    nbr_ = {}
    for a_e, b_e in EDG:
        nbr_.setdefault(a_e, []).append(b_e); nbr_.setdefault(b_e, []).append(a_e)
    in_v, in_t = [], []
    hv_set = set(hv.tolist())
    pxT = project(T_full)
    for j, (h, ei, part) in enumerate(hinfo):
        c, r = balls[eyes[ei]['side']]
        dr = ray(pxT[j][None])[0]
        oc = Cc - c; bq = (dr * oc).sum(); cq = (oc ** 2).sum() - (r + E_('INNER_GAP', 0.3) / 1000) ** 2
        disc = bq ** 2 - cq
        if disc <= 0:
            continue
        tgt = Cc + dr * (-bq - np.sqrt(disc))
        for nb in nbr_.get(hv[j], []):
            if eyes[ei]['inn'][nb] and nb not in hv_set and region[nb]:
                in_v.append(nb); in_t.append(tgt)
    if in_v:
        in_v, ix_ = np.unique(np.array(in_v), return_index=True)
        in_t = np.array(in_t)[ix_]
        hv_all = np.r_[hv, in_v]; T_all = np.r_[T_full, in_t]
        print('ARAP lid back edge: %d vertices held behind the margin, on the ball' % len(in_v))
solve_all = solver(hv_all)
for k in range(1, STEPS + 1):
    a = k / STEPS
    P = arap(P, V0[hv_all] * (1 - a) + T_all * a, hidx=hv_all, solve=solve_all)
    fl, cr = folds(P)
    print('step %d/%d: margin to target max %.2f mm; flipped %d, crushed %d' % (k, STEPS, np.linalg.norm(P[hv] - T_full, axis=1).max() * 1000, fl, cr))
# ---- cover: the eyeball seen outside the lines means skin behind it there; that skin comes forward along its own view ray
# to the ball's front (0.5 mm out), so the picture keeps its place and the ball is hidden (COVER rounds) ----
outside_lines = np.ones((RES, RES), np.uint8)
for e in eyes:
    xx = np.linspace(e['x0'], e['x1'], 80)
    cv2.fillPoly(outside_lines, [np.r_[np.c_[xx, e['fu'](xx)], np.c_[xx[::-1], e['fl'](xx[::-1])]].round().astype(np.int32)], 0)
outside_lines = cv2.erode(outside_lines, np.ones((3, 3), np.uint8))      # a pixel's slack along the lines
skin_v = np.ones(nV, bool)
for s_, (c, r) in balls.items():
    dd = V0 - c; dist = np.linalg.norm(dd, axis=1)
    skin_v &= ~((dist < r + 0.004) & ((N0 * dd).sum(1) / np.maximum(dist, 1e-9) < 0))   # not the lids' inner faces
for rd in range(COVER + 1):
    apply_and_render(P, f'a{rd}')
    g, col_gap, crop = measure(f'a{rd}')
    cv2.imwrite(f'{L}/arap_{rd}.jpg', cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    mk = cv2.imread(f'{W}/a{rd}_mask.png')
    shown = (mk[..., 2] > 200) & (mk[..., 0] > 200) & (mk[..., 1] < 80) & (outside_lines > 0)
    fl, cr = folds(P)
    print('round %d: |gap| mm median %.2f p90 %.2f max %.2f; eyeball outside the lines %d px; flipped %d, crushed %d' % (rd, np.median(g), np.percentile(g, 90), g.max(), shown.sum(), fl, cr))
    if rd == COVER or shown.sum() == 0:
        break
    # zip: past a corner the slit between the lids is still open; per column of it, the skin just above its top and just below
    # its bottom go to its middle height there (each along its own depth), the rest follows as-rigid-as-possible
    px = project(P)
    pxi = px.round().astype(int)
    okp = (pxi[:, 0] >= 0) & (pxi[:, 0] < RES) & (pxi[:, 1] >= 0) & (pxi[:, 1] < RES) & region & ~np.isin(np.arange(nV), hv)
    cov, tgt = [], []
    for c_ in np.unique(np.nonzero(shown)[1]):
        rows = np.flatnonzero(shown[:, c_])
        top, bot = rows.min(), rows.max(); mid = (top + bot) / 2
        sel = okp & (np.abs(px[:, 0] - c_) < 1.0) & (((px[:, 1] > top - 4) & (px[:, 1] <= top + 1)) | ((px[:, 1] >= bot - 1) & (px[:, 1] < bot + 4)))
        for v in np.flatnonzero(sel):
            cov.append(v); tgt.append([px[v, 0], mid])
    if not cov:
        print('  zip: nothing found next to the slit'); break
    cov, keep = np.unique(np.array(cov), return_index=True)
    tgt = np.array(tgt)[keep]
    dr = ray(tgt); t = ((P[cov] - Cc) * dr).sum(1)
    Tc = Cc + dr * t[:, None]
    keep_c = ~np.isin(cov, hv_all); cov, Tc = cov[keep_c], Tc[keep_c]
    hidx = np.r_[hv_all, cov]
    P = arap(P, np.r_[P[hv_all], Tc], hidx=hidx, solve=solver(hidx))
    print('  zip: %d vertices beside the slit drawn to its middle' % len(cov))
total = P - V0
mv_ = np.flatnonzero(np.linalg.norm(total, axis=1) > 1e-5)
np.savez(f'{O}/lid_delta.npz', delta=total, pos1=V0[mv_], off1=total[mv_], pos2=np.full((1, 3), 1e3), off2=np.zeros((1, 3)))
print('folds: flipped %d, crushed %d' % folds(P))
print('wrote', f'{O}/lid_delta.npz', 'max mm %.2f' % (np.abs(total).max() * 1000))
