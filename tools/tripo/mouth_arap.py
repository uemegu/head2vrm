"""The lips onto the Tripo head's own mouth as seen from the lid-fit camera, by an as-rigid-as-possible deformation.   any face
python mouth_arap.py OUT_DIR          (cwd: tripo/; after lid_arap.py; CHECK=1 only draws what is extracted and stops)

Why: seam_fit.py puts the lip seam on Tripo's lines in a straight-on orthographic view, so depth is free; from the
viewer's camera (10.4 deg above) the mouth corners came 1.8 mm low and both lips 0.6-1.3 mm too full: the corners stood further
forward than Tripo's. Here the targets are 3D: MediaPipe's lip contour (outer and inner) on the Tripo head rendered from the
lid-fit camera (lidfit/warp/v0_tripo.png), each point where its pixel's ray meets the Tripo surface.
Handles: each landmark's point on the base mesh (its corr.npz triangle, the vertex with the largest weight); one whose current
place is more than DROP mm from its target is left out (a bad correspondence). The region within REGION mm of the mouth's centre
is solved as-rigid-as-possible against its shape before (cotangent weights clamped positive), the border fixed, in STEPS steps.
A person approves the lines first: mouthfit/check.jpg, check_lines.txt; `md5 -q mouthfit/check_lines.txt > mouthfit/approved`.
Writes OUT_DIR/mouth_delta.npz (seam_apply.py format).
"""
import json
import os
import sys

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mpfb'))
from lm import detect  # noqa: E402

O = os.path.abspath(sys.argv[1])
E_ = lambda k, d: float(os.environ.get(k, d))
REGION, DROP, STEPS, ITERS, W_H = E_('REGION', 20) / 1000, E_('DROP', 4) / 1000, int(E_('STEPS', 4)), int(E_('ITERS', 6)), E_('W_H', 1000)
CORR = os.environ.get('CORR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mpfb', os.environ.get('BASE', 'base_asian'), 'corr.npz'))
L = f'{O}/mouthfit'; os.makedirs(L, exist_ok=True)
CHECK = os.environ.get('CHECK')

cam = json.load(open(f'{O}/lidfit/warp/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1]; F = (RES / 2) / np.tan(cam['fov'] / 2)
MM_PX = np.linalg.norm(Cc - np.array(cam['centre'])) / F * 1000


def project(P):
    q = (P - Cc) @ Rc
    return np.c_[RES / 2 + F * q[:, 0] / -q[:, 2], RES / 2 - F * q[:, 1] / -q[:, 2]]


def ray(px):
    d = (Rc @ np.c_[(px[:, 0] - RES / 2) / F, -(px[:, 1] - RES / 2) / F, -np.ones(len(px))].T).T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


OUTER_U = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_L = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
INNER_U = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
INNER_L = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
LINES = (OUTER_U, OUTER_L, INNER_U, INNER_L)
IDS = sorted({i for l in LINES for i in l})

img = cv2.imread(f'{O}/lidfit/warp/v0_tripo.png')
lm = detect([f'{O}/lidfit/warp/v0_tripo.png'])[0][:, :2]
# the seam: MediaPipe's inner points of a closed, smiling mouth sit low, on the lower lip (on one test head ~1 mm under the dark line between the
# lips). Each inner point but the corners goes to the darkest row (lightness blurred 2 px) within SEAM_UP px above / SEAM_DN below
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
SEAM_UP, SEAM_DN = int(E_('SEAM_UP', 14)), int(E_('SEAM_DN', 4))
for i in sorted(set(INNER_U + INNER_L) - {78, 308}):
    x, y = int(round(lm[i, 0])), int(round(lm[i, 1]))
    col = gray[y - SEAM_UP:y + SEAM_DN + 1, x]
    lm[i, 1] = y - SEAM_UP + int(np.argmin(col))
# the lower lip's outer edge: MediaPipe's points can sit in the skin below it (on one test head: the viewer's right half, 8-12 px low where the lip
# is less red). Along each point's normal (corners excepted) the edge is where the redness (Lab a*, blurred 2 px) falls fastest,
# within EDGE_IN px inside / EDGE_OUT px outside; a point is moved only when that is more than EDGE_MIN px away (EDGE=0 turns it off)
a_ = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32), (0, 0), 2)
EDGE_IN, EDGE_OUT, EDGE_MIN = int(E_('EDGE_IN', 25)), int(E_('EDGE_OUT', 10)), E_('EDGE_MIN', 4)
# the mouth corners: MediaPipe stopped one corner short of the dark crease and put the other on it (on one test head: the viewer's left 15 px
# inside, the right on it). Both by one rule (CORNER=0 turns it off): from the inner corner, follow the darkest row (lightness,
# blurred 1 px, within 4 px of the last row) outward for CORNER_REACH px; the outer corner is the darkest point of that path
if E_('CORNER', 1):
    gl = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32), (0, 0), 1)
    for oc, ic, sg in ((61, 78, -1), (291, 308, 1)):
        x, y = int(round(lm[ic, 0])), int(round(lm[ic, 1])); path = []
        for k in range(int(E_('CORNER_REACH', 40))):
            xx = x + sg * k; y = y - 4 + int(np.argmin(gl[y - 4:y + 5, xx])); path.append((gl[y, xx], xx, y))
        _, cx, cy = min(path)
        # the inner corner keeps MediaPipe's gap to the outer one (in x), on the same dark path
        gap = int(round(abs(lm[ic, 0] - lm[oc, 0]))); _, ix, iy = min(path, key=lambda p: abs(p[1] - (cx - sg * gap)))
        print('  mouth corner %d: (%d, %d) -> (%d, %d); inner %d: (%d, %d) -> (%d, %d)' % (oc, lm[oc, 0], lm[oc, 1], cx, cy, ic, lm[ic, 0], lm[ic, 1], ix, iy))
        lm[oc] = (cx, cy); lm[ic] = (ix, iy)
if E_('EDGE', 1):
    base = lm.copy(); S_ = np.arange(-EDGE_IN, EDGE_OUT + 1)
    for k in range(1, len(OUTER_L) - 1):
        i = OUTER_L[k]; t_ = base[OUTER_L[k + 1]] - base[OUTER_L[k - 1]]
        n_ = np.array([-t_[1], t_[0]]) / np.linalg.norm(t_); n_ = n_ if n_[1] > 0 else -n_
        prof = np.array([cv2.getRectSubPix(a_, (1, 1), tuple((base[i] + s * n_).astype(np.float32)))[0, 0] for s in S_])
        s_ = S_[np.argmax(-np.gradient(prof))]
        if abs(s_) > EDGE_MIN:
            lm[i] = base[i] + s_ * n_; print('  lower lip edge %d: moved %+d px along its normal' % (i, s_))
# the lower lip's outer edge, symmetric (LIP_SYM, default on; it came out clearly lopsided): on a male test head's pale
# lips the redness edge is weak by the corners, and one point (146) stayed on the seam while its mirror (375) sat 12 px below, so the
# line dropped straight down on one side and bulged on the other. Each point is placed by its share t of the half width (the lip's
# centre at 17 to its own corner, so the corners stay) and its depth d below the line between the corners; both halves' (t, d) and
# 17's are fitted by d = a(1 - t^2) + b(1 - t^4) (d = 0 at the corners), the farthest point is dropped once if it lies more than
# LIP_SYM_OUT px off, and each pair goes to the mean of its two t on the fitted curve.
if E_('LIP_SYM', 1):
    c0, c1 = lm[61], lm[291]; mx = lm[17, 0]
    yc = lambda x: c0[1] + (c1[1] - c0[1]) * (x - c0[0]) / (c1[0] - c0[0])
    side = lambda i: (mx - c0[0]) if lm[i, 0] < mx else (c1[0] - mx)
    ids = [i for i in OUTER_L[1:-1]]
    t_ = {i: abs(lm[i, 0] - mx) / side(i) for i in ids}; d_ = {i: lm[i, 1] - yc(lm[i, 0]) for i in ids}
    A = lambda t: np.c_[1 - t ** 2, 1 - t ** 4]
    use = list(ids)
    for _ in range(2):
        T = np.array([t_[i] for i in use]); D = np.array([d_[i] for i in use])
        ab = np.linalg.lstsq(A(T), D, rcond=None)[0]; r = D - A(T) @ ab
        k = int(np.argmax(abs(r)))
        if abs(r[k]) <= E_('LIP_SYM_OUT', 10) or _: break
        print('  lower lip: %d left out of the fit (%+.0f px off)' % (use[k], r[k])); use.pop(k)
    half = len(OUTER_L) // 2
    for a_i, b_i in [(OUTER_L[k], OUTER_L[-1 - k]) for k in range(1, half)] + [(17, 17)]:
        tm = (t_[a_i] + t_[b_i]) / 2; d = float((A(np.array([tm])) @ ab)[0])
        for i in {a_i, b_i}:
            x = mx if i == 17 else mx + (-1 if lm[i, 0] < mx else 1) * tm * side(i)
            print('  lower lip %d: (%d, %d) -> (%d, %d)' % (i, lm[i, 0], lm[i, 1], x, yc(x) + d))
            lm[i] = (x, yc(x) + d)
if CHECK:
    pic = img.copy()
    for l, col in zip(LINES, ((0, 130, 255), (0, 130, 255), (255, 0, 255), (255, 0, 255))):
        cv2.polylines(pic, [lm[list(l)].round().astype(np.int32)], False, col, 1)
    for i in IDS:
        cv2.circle(pic, tuple(lm[i].round().astype(int)), 2, (0, 255, 0), -1)
    P_ = lm[IDS]; x0, y0 = (P_.min(0) - 40).astype(int); x1, y1 = (P_.max(0) + 40).astype(int)
    crop = pic[y0:y1, x0:x1]
    cv2.imwrite(f'{L}/check.jpg', cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
    with open(f'{L}/check_lines.txt', 'w') as fh:
        for i in IDS:
            fh.write('%d %d %d\n' % (i, round(lm[i, 0]), round(lm[i, 1])))
    print('CHECK wrote', f'{L}/check.jpg')
    raise SystemExit
if open(f'{L}/check_lines.txt').read() and os.popen(f'md5 -q {L}/check_lines.txt').read().strip() != (open(f'{L}/approved').read().strip() if os.path.exists(f'{L}/approved') else ''):
    print('MOUTH_CHECK waiting: approve', f'{L}/check.jpg'); sys.exit(3)
pts = {int(r.split()[0]): np.array([float(v) for v in r.split()[1:]]) for r in open(f'{L}/check_lines.txt') if r.strip()}

# the targets in 3D: each approved pixel's ray onto the Tripo surface (the nearest hit)
tm = np.load(f'{O}/tripo_in_mpfb.npz'); TV = tm['co'][tm['tris']]
e1, e2 = TV[:, 1] - TV[:, 0], TV[:, 2] - TV[:, 0]


def hit(o, d):
    p = np.cross(d, e2); det = (e1 * p).sum(1); ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0); s = o - TV[:, 0]
    u = (s * p).sum(1) * inv; q = np.cross(s, e1); v = (q @ d) * inv; t = (e2 * q).sum(1) * inv
    g = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return o + d * t[g].min() if g.any() else None


ids = [i for i in IDS if i in pts]
dirs = ray(np.array([pts[i] for i in ids]))
tg = {i: hit(Cc, d) for i, d in zip(ids, dirs)}

# the handles on the base mesh
c = np.load(CORR)
cid = {int(i): k for k, i in enumerate(c['ids'])}
u = np.load(os.environ.get('UVMESH', f'{O}/uvmesh.npz'))
V0 = u['co'].astype(np.float64); tv = u['tv']
EDG = np.unique(np.sort(np.r_[tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]], 1), axis=0)
nV = len(V0)
hv, T = [], []
for i in ids:
    if i not in cid or tg[i] is None:
        continue
    k = cid[i]; vtx = int(c['trip'][k][np.argmax(c['bary'][k])])
    if np.linalg.norm(V0[vtx] - tg[i]) > DROP:
        print('  dropped landmark %d: %.1f mm off' % (i, np.linalg.norm(V0[vtx] - tg[i]) * 1000)); continue
    hv.append(vtx); T.append(tg[i])
hv, first = np.unique(np.array(hv), return_index=True); T = np.array(T)[first]
# the outer lip line (ONTO_LINE, default on): which base vertex a landmark lands on is good across the line but not
# along it. By the corners the base's 146/375/185 sit 20-26 px nearer the corner than Tripo's, and pulling them along the line
# squeezed the corners (crumpled when the mouth opens) and the lower lip's border (the mouth's inside showed through under it).
# Each outer handle but the corners goes to the nearest point of its approved line (a spline through the line's points) from where
# it is now, on the Tripo surface along that pixel's ray: it moves across the line only.
if E_('ONTO_LINE', 1):
    from scipy.interpolate import CubicSpline
    vtx_of = {i: int(c['trip'][cid[i]][np.argmax(c['bary'][cid[i]])]) for i in ids if i in cid}
    for line in (OUTER_U, OUTER_L):
        Q = np.array([pts[i] for i in line]); t_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        cs = CubicSpline(t_, Q); dense = cs(np.linspace(0, t_[-1], 2000))
        for i in line[1:-1]:
            if i not in vtx_of or vtx_of[i] not in hv: continue
            v_ = vtx_of[i]; p_ = project(V0[[v_]])[0]
            q_ = dense[np.argmin(np.linalg.norm(dense - p_, axis=1))]
            h_ = hit(Cc, ray(q_[None])[0])
            if h_ is not None:
                T[list(hv).index(v_)] = h_
    print('ONTO_LINE outer handles moved onto their line from where they are')
centre = T.mean(0)
# the corners rise by the base's own smile keys first (SMILE, default on): pulled by one handle, the corner's knot
# of small faces folded (31 flipped) and, with the mouth open, crumpled. Per corner, the smile key that moves it most is set so the
# corner's height (in this camera) reaches its target's; the ARAP below starts from, and keeps the shape of, that smiled face
V_orig = V0.copy()
if E_('SMILE', 0):   # off: one test head flips 26 with it, 13 without (the key also moves the corner 14 px outward, which the fit undoes)
    kk0 = np.load(os.environ.get('MOUTH_KEYS', f'{O}/mouth_keys.npz'))
    for lmk in (61, 291):
        if lmk not in cid: continue
        cv_ = int(c['trip'][cid[lmk]][np.argmax(c['bary'][cid[lmk]])])
        if cv_ not in hv: continue
        key = max(('mouthSmileLeft', 'mouthSmileRight'), key=lambda k_: np.linalg.norm(kk0[k_][cv_]))
        y0_ = project(V0[[cv_]])[0, 1]; y1_ = project(V0[[cv_]] + kk0[key][[cv_]])[0, 1]; yt_ = project(T[[list(hv).index(cv_)]])[0, 1]
        w_ = float(np.clip((yt_ - y0_) / (y1_ - y0_), 0, E_('SMILE_MAX', 0.5))) if abs(y1_ - y0_) > 1e-6 else 0.0
        V0 = V0 + w_ * kk0[key]
        print('  corner %d: %s %.3f (corner %.1f px -> %.1f, target %.1f)' % (lmk, key, w_, y0_, project(V0[[cv_]])[0, 1], yt_))
# or (CORNER_SIGMA mm, default 4) the corner's move spread smoothly round it before the fit (a Gaussian of the 3D distance), so the
# knot of small faces moves together and the jaw-open key still opens it cleanly; the fit then starts from, and keeps, that shape
if E_('CORNER_SIGMA', 4) > 0 and not E_('SMILE', 0):
    sg_ = E_('CORNER_SIGMA', 4) / 1000
    for lmk in (61, 291):
        if lmk not in cid: continue
        cv_ = int(c['trip'][cid[lmk]][np.argmax(c['bary'][cid[lmk]])])
        if cv_ not in hv: continue
        mv = T[list(hv).index(cv_)] - V0[cv_]
        V0 = V0 + np.exp(-((V0 - V0[cv_]) ** 2).sum(1) / (2 * sg_ ** 2))[:, None] * mv
        print('  corner %d: its %.2f mm move spread over sigma %.1f mm' % (lmk, np.linalg.norm(mv) * 1000, sg_ * 1000))
# each mouth corner moves as a block: every vertex within CORNER_R mm of the corner's vertex takes the corner's move. The corner is
# a knot of small faces where the two lips and the mouth's inside meet; pulling its one vertex folded them (on one test head: the
# right corner rises 23 px, 31 faces flipped round the corners)
CORNER_R = E_('CORNER_R', 0) / 1000   # flips on one test head: 2.5 mm 21, 4 mm 10, 5 mm 8; but the block slid through the cheek (open mouth): off
in_tri0 = np.zeros(len(V0), bool); in_tri0[tv.ravel()] = True
corner_blk = {}
for lmk in (61, 291):
    if lmk in cid:
        cv_ = int(c['trip'][cid[lmk]][np.argmax(c['bary'][cid[lmk]])])
        if cv_ in hv and CORNER_R > 0:
            mv = T[list(hv).index(cv_)] - V0[cv_]
            blk = np.flatnonzero(in_tri0 & (np.linalg.norm(V0 - V0[cv_], axis=1) < CORNER_R))
            corner_blk[lmk] = blk
            new = np.setdiff1d(blk, hv); hv = np.r_[hv, new]; T = np.r_[T, V0[new] + mv]
            print('  corner %d: %d vertices moved as one, %.2f mm' % (lmk, len(blk), np.linalg.norm(mv) * 1000))
# REGION round the centre, but always reaching MARGIN past the farthest handle (on one test head the corners 61/291 and the lower
# lip's 146/375 lay 21-22 mm out, outside a 20 mm region: they were never moved, so the approved corners did not take effect)
REGION = max(REGION, np.linalg.norm(V0[hv] - centre, axis=1).max() + E_("MARGIN", 6) / 1000)
region = np.linalg.norm(V0 - centre, axis=1) < REGION
print("MOUTH region radius %.1f mm" % (REGION * 1000))
in_tri = np.zeros(nV, bool); in_tri[tv.ravel()] = True
region &= in_tri
print('MOUTH handles %d, region %d, handle-to-target mm: median %.2f max %.2f' % (len(hv), region.sum(), np.median(np.linalg.norm(V0[hv] - T, axis=1)) * 1000, np.linalg.norm(V0[hv] - T, axis=1).max() * 1000))

free = np.flatnonzero(region); fid = -np.ones(nV, int); fid[free] = np.arange(len(free))
tri_r = tv[region[tv].any(1)]
w = {}
for a_, b_, c_ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    u_ = V0[tri_r[:, a_]] - V0[tri_r[:, c_]]; v_ = V0[tri_r[:, b_]] - V0[tri_r[:, c_]]
    cot = (u_ * v_).sum(1) / np.maximum(np.linalg.norm(np.cross(u_, v_), axis=1), 1e-12)
    for i_, j_, cc in zip(tri_r[:, a_], tri_r[:, b_], cot):
        key = (min(i_, j_), max(i_, j_)); w[key] = w.get(key, 0) + 0.5 * cc
EK = np.array(list(w.keys())); EW = np.maximum(np.array(list(w.values())), 1e-3)
rows, cols, vals = [], [], []
for (i_, j_), w_ in zip(EK, EW):
    for a_, b_ in ((i_, j_), (j_, i_)):
        if fid[a_] >= 0:
            rows.append(fid[a_]); cols.append(fid[a_]); vals.append(w_)
            if fid[b_] >= 0:
                rows.append(fid[a_]); cols.append(fid[b_]); vals.append(-w_)
Lm = sp.csr_matrix((vals, (rows, cols)), shape=(len(free), len(free)))


def fit(hv, T):
    """ARAP from V0 with handles hv pulled to T in STEPS steps"""
    hf = fid[hv]; keep_h = hf >= 0; hv, T, hf = hv[keep_h], T[keep_h], hf[keep_h]
    Hm = sp.csr_matrix((np.full(len(hv), W_H), (hf, hf)), shape=(len(free), len(free)))
    solve = factorized((Lm + Hm + sp.identity(len(free)) * 1e-9).tocsc())

    def arap(P, Tt):
        P = P.copy()
        for _ in range(ITERS):
            S = np.zeros((nV, 3, 3))
            for (i_, j_), w_ in zip(EK, EW):
                o_ = w_ * np.outer(V0[i_] - V0[j_], P[i_] - P[j_]); S[i_] += o_; S[j_] += o_
            R = np.tile(np.eye(3), (nV, 1, 1))
            for i_ in free:
                U, _, Vt = np.linalg.svd(S[i_]); Ri = Vt.T @ U.T
                if np.linalg.det(Ri) < 0:
                    Vt[-1] *= -1; Ri = Vt.T @ U.T
                R[i_] = Ri
            rhs = np.zeros((len(free), 3))
            for (i_, j_), w_ in zip(EK, EW):
                e0 = V0[i_] - V0[j_]
                if fid[i_] >= 0:
                    rhs[fid[i_]] += 0.5 * w_ * (R[i_] + R[j_]) @ e0
                    if fid[j_] < 0: rhs[fid[i_]] += w_ * P[j_]
                if fid[j_] >= 0:
                    rhs[fid[j_]] -= 0.5 * w_ * (R[i_] + R[j_]) @ e0
                    if fid[i_] < 0: rhs[fid[j_]] += w_ * P[i_]
            rhs[hf] += W_H * Tt
            P[free] = np.c_[solve(rhs[:, 0]), solve(rhs[:, 1]), solve(rhs[:, 2])]
        return P

    P = V0.copy()
    for k in range(1, STEPS + 1):
        a = k / STEPS
        P = arap(P, V0[hv] * (1 - a) + T * a)
        print('step %d/%d: handles to target max %.2f mm' % (k, STEPS, np.linalg.norm(P[hv] - T, axis=1).max() * 1000))
    return P


P = fit(hv, T)
# the seam as one gentle curve (SEAM_SMOOTH, default on; like the eyes, a smooth curve, not a jagged one):
# the landmark handles pin only a dozen vertices of the two lips' meeting rows, and the rows between them waved. Both rows are
# found on the mesh (the upper one through the inner upper landmarks' vertices, the lower one through the vertex facing each of
# them that the jaw-open key moves down), and every vertex of them goes, at its depth after the first pass, to a smoothing spline
# y(x) through the approved seam points and the outer corners as seen from this camera. A second pass then holds the outer lip
# landmarks and these rows (the inner landmarks are replaced by the rows).
if E_('SEAM_SMOOTH', 1):
    from scipy.interpolate import UnivariateSpline
    from scipy.sparse.csgraph import dijkstra
    kk = np.load(os.environ.get('MOUTH_KEYS', f'{O}/mouth_keys.npz'))
    Jy = (kk['jawOpen'] @ Rc)[:, 1]
    vt = lambda i: int(c['trip'][cid[i]][np.argmax(c['bary'][cid[i]])])
    sp_ids = [i for i in (61,) + INNER_U + INNER_L + (291,) if i in pts]
    sx = np.array([pts[i][0] for i in sp_ids]); sy = np.array([pts[i][1] for i in sp_ids])
    sw = np.array([5.0 if i in (61, 291) else 1.0 for i in sp_ids])
    o_ = np.argsort(sx, kind='stable'); sx, sy, sw = sx[o_], sy[o_], sw[o_]
    sx = sx + np.arange(len(sx)) * 1e-6   # strictly increasing
    curve = UnivariateSpline(sx, sy, w=sw, k=3, s=len(sx) * E_('SEAM_SIGMA', 1.0) ** 2)
    G = sp.coo_matrix((np.linalg.norm(V0[EDG[:, 0]] - V0[EDG[:, 1]], axis=1), (EDG[:, 0], EDG[:, 1])), shape=(nV, nV))

    def row(chain, ok):
        ok = ok.copy(); ok[chain] = True
        g = G.tocsr().multiply(ok[:, None]).multiply(ok[None, :]); g = g + g.T
        out = [chain[0]]
        for a_, b_ in zip(chain[:-1], chain[1:]):
            _, pr = dijkstra(g, indices=a_, return_predecessors=True)
            pth = [b_]
            while pth[-1] != a_ and pth[-1] >= 0: pth.append(pr[pth[-1]])
            out += pth[::-1][1:]
        return out
    c61, c291 = vt(61), vt(291)
    up_lm = [vt(i) for i in INNER_U[1:-1] if i in cid and Jy[vt(i)] > -E_('UP_J', 6) / 1000]
    lo_lm = []
    for i_ in up_lm:
        cand = np.flatnonzero(in_tri & (Jy < Jy[i_] - 0.004) & (np.linalg.norm(V0 - V0[i_], axis=1) < 0.004))
        if len(cand):
            j_ = int(cand[np.argmin(np.linalg.norm(V0[cand] - V0[i_], axis=1))])
            if j_ not in lo_lm: lo_lm.append(j_)
    rowU = row([c61] + up_lm + [c291], in_tri & (Jy > -0.013))
    rowL = row([c61] + lo_lm + [c291], in_tri & (Jy < -0.009))
    rim = np.array(sorted(set(rowU + rowL) - {c61, c291}))
    # near the corners the rows are left free (CORNER_FREE px): pinning both rows and the corner squeezed the faces between (on one test head:
    # 24 flipped round the right corner, which rises 23 px)
    pr_ = project(V0[rim]); cf_ = E_('CORNER_FREE', 25)   # flips on one test head: 0 px 23, 12 px 14, 25 px 2
    near_c = ~((np.linalg.norm(pr_ - pts[61], axis=1) > cf_) & (np.linalg.norm(pr_ - pts[291], axis=1) > cf_))
    rim_c = rim[near_c]; rim = rim[~near_c]
    rim = np.setdiff1d(rim, np.concatenate([np.zeros(0, int)] + list(corner_blk.values())))   # the corner blocks keep their move
    qc = (P[rim] - Cc) @ Rc; dep = -qc[:, 2]
    xr = RES / 2 + F * qc[:, 0] / dep; yr = curve(xr)
    rimT = Cc + (Rc @ np.stack([(xr - RES / 2) / F * dep, -(yr - RES / 2) / F * dep, -dep])).T
    yr0 = RES / 2 - F * qc[:, 1] / dep
    print('SEAM_SMOOTH rows: upper %d, lower %d vertices; off the curve before: median %.1f max %.1f px' % (len(rowU), len(rowL), np.median(np.abs(yr0 - yr)), np.abs(yr0 - yr).max()))
    # the curve and the rows for lip_paint.py (mouthfit/seam_curve.npz; written here so lip_paint.py can read it)
    if not CHECK:
        xs_c = np.linspace(pts[61][0], pts[291][0], 200)
        np.savez(f'{O}/mouthfit/seam_curve.npz', x=xs_c, y=curve(xs_c), rowU=np.array(rowU), rowL=np.array(rowL))
    # the free rows by the corners onto the curve too, as a change of the rest shape rather than handles (CORNER_ROWS, default on;
    # the mouth corners looked wrong): the base's corner vertex sat 12-15 px below its rows' next vertices, and the corner's move
    # carried that droop along, so the rows by the corner stood up to 14 px above the curve and the corner read as an upward hook.
    # Each such row vertex's way onto the curve (at its depth after the first pass; the corners themselves stay) is spread round it
    # (a Gaussian-weighted mean, CORNER_ROW_SIGMA mm) and added to V0; the second pass keeps that shape, the rows there still free
    def corner_rows(P):
        qk = (P[rim_c] - Cc) @ Rc; dk = -qk[:, 2]; xk = RES / 2 + F * qk[:, 0] / dk; yk0 = RES / 2 - F * qk[:, 1] / dk
        tk = Cc + (Rc @ np.stack([(xk - RES / 2) / F * dk, -(curve(xk) - RES / 2) / F * dk, -dk])).T
        src = np.r_[rim_c, [c61, c291]]; mvk = np.r_[tk - P[rim_c], np.zeros((2, 3))]
        sgk = E_('CORNER_ROW_SIGMA', 1.2) / 1000
        gk = np.exp(-((V0[:, None] - V0[src][None]) ** 2).sum(2) / (2 * sgk ** 2))   # nV x len(src)
        corr_ = (gk @ mvk) / np.maximum(gk.sum(1), 1.0)[:, None]
        corr_[~region] = 0
        print('SEAM_SMOOTH corner rows: %d vertices onto the curve, off it before: max %.1f px; spread over %.1f mm, %d vertices moved > 0.1 mm'
              % (len(rim_c), np.abs(yk0 - curve(xk)).max(), sgk * 1000, int((np.linalg.norm(corr_, axis=1) > 1e-4).sum())))
        return corr_
    CR_ROUNDS = int(E_('CORNER_ROWS', 3)) if len(rim_c) else 0   # rounds: the second pass gives back part of each correction
    if CR_ROUNDS: V0 = V0 + corner_rows(P)
    inner = set(vt(i) for i in INNER_U + INNER_L if i in cid) - {c61, c291}
    keep = np.array([h not in inner and h not in set(rim) for h in hv])
    # an outer lip handle (not a corner) whose target lies within MIN_LIP px of the seam curve pins the lip's edge onto its seam
    # (e.g. 409, the upper lip's edge by the right corner, 1 px off it): its faces are squeezed flat and fold; it is left free
    pT = project(T); thin = (np.abs(pT[:, 1] - curve(pT[:, 0])) < E_('MIN_LIP', 5)) & ~np.isin(hv, [c61, c291])
    if thin.any(): print('SEAM_SMOOTH outer handles on the seam left free:', int(thin.sum()))
    keep &= ~thin
    hv2 = np.r_[hv[keep], rim]; T2 = np.r_[T[keep], rimT]
    P = fit(hv2, T2)
    for _ in range(CR_ROUNDS - 1):   # what the fit gave back, corrected again
        V0 = V0 + corner_rows(P); P = fit(hv2, T2)
    hv = hv2
    # the faces just beside the rows (BAND mm) evened out (BAND_IT rounds of moving each to its neighbours' mean, the rows and the
    # handles held): with the rows on one curve, the lower lip's top strip showed its uneven faces as light wedges in the upper
    # lip's shadow
    band_ = np.zeros(nV, bool); dr = np.min(np.linalg.norm(P[:, None] - P[np.array(rowU + rowL)][None], axis=2), axis=1)
    fwd_ = (u['nrm'] @ Rc)[:, 2] > E_('BAND_FACING', 0.3)   # only the lips' outer faces (the mouth's inside folded, 41 flips)
    band_ = region & fwd_ & (dr < E_('BAND', 1.5) / 1000); band_[hv] = False; band_[rowU + rowL] = False
    nb = [[] for _ in range(nV)]
    for a_, b_ in EDG:
        nb[a_].append(b_); nb[b_].append(a_)
    bi = np.flatnonzero(band_)
    for _ in range(int(E_('BAND_IT', 0))):   # off: one test head flips 18-41 with it (the knot of small faces by the seam)
        P[bi] = 0.5 * P[bi] + 0.5 * np.array([P[nb[i_]].mean(0) for i_ in bi])
    if E_('BAND_IT', 0): print('SEAM_SMOOTH band evened: %d vertices' % len(bi))
# the mouth's inside kept behind the skin (INSIDE_BACK, default on): the lower lip's border went up and back, and the
# lip's inner side (faces turned away from the camera) came through the skin under the lip, a small pentagon in the viewer. Any
# vertex of the region on a face turned away that sits in front of the camera-facing skin at its pixel goes back along its ray to
# BACK_MM behind that skin; then once more, as moving one uncovers the next.
if E_('INSIDE_BACK', 1):
    def zbuf(Pp, faces):
        S_ = np.stack([project(Pp[faces[:, k]]) for k in range(3)], 1)
        q_ = (Pp - Cc) @ Rc; dep_ = -q_[:, 2]; Sd = dep_[faces]
        box_ = np.floor(S_.reshape(-1, 2).min(0)).astype(int) - 2, np.ceil(S_.reshape(-1, 2).max(0)).astype(int) + 2
        Z_ = np.full((box_[1][1] - box_[0][1], box_[1][0] - box_[0][0]), np.inf)
        for s2, z3 in zip(S_, Sd):
            a0, b0 = np.floor(s2.min(0)).astype(int); a1, b1 = np.ceil(s2.max(0)).astype(int) + 1
            px_, py_ = np.meshgrid(np.arange(a0, a1) + .5, np.arange(b0, b1) + .5)
            (ax, ay), (bx, by), (cx, cy) = s2
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12: continue
            l0 = ((by - cy) * (px_ - cx) + (cx - bx) * (py_ - cy)) / den; l1 = ((cy - ay) * (px_ - cx) + (ax - cx) * (py_ - cy)) / den
            l2 = 1 - l0 - l1; ins = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            z = 1 / (l0 / z3[0] + l1 / z3[1] + l2 / z3[2])
            sub = Z_[b0 - box_[0][1]:b1 - box_[0][1], a0 - box_[0][0]:a1 - box_[0][0]]; w_ = ins & (z < sub); sub[w_] = z[w_]
        return Z_, box_
    for rnd in range(2):
        fn = np.cross(P[tv[:, 1]] - P[tv[:, 0]], P[tv[:, 2]] - P[tv[:, 0]]); fz = (fn @ Rc)[:, 2]
        reg_f = region[tv].all(1)
        Zs, bx_ = zbuf(P, tv[reg_f & (fz > 0)])
        away = np.zeros(nV, bool); away[tv[reg_f & (fz < 0)].ravel()] = True
        cand = np.flatnonzero(away & region)
        pc = project(P[cand]); dep = -((P[cand] - Cc) @ Rc)[:, 2]
        ix = np.clip(pc[:, 0].astype(int) - bx_[0][0], 0, Zs.shape[1] - 1); iy = np.clip(pc[:, 1].astype(int) - bx_[0][1], 0, Zs.shape[0] - 1)
        zs = Zs[iy, ix]; front = np.isfinite(zs) & (dep < zs + E_('BACK_MM', 0.5) / 1000)
        # only where skin lies in front of the inside nearby (the lips' own rims, where inside meets outside, stay)
        front &= np.min(np.linalg.norm(P[cand][:, None] - P[np.array(rowU + rowL)][None], axis=2), axis=1) > 0.002 if E_('SEAM_SMOOTH', 1) else front
        far_c = (np.linalg.norm(pc - pts[61], axis=1) > E_('CORNER_FREE', 25)) & (np.linalg.norm(pc - pts[291], axis=1) > E_('CORNER_FREE', 25))
        # only under the lower lip: between the corners and below the seam (on one test head the region reached the nostrils, whose
        # inner faces turn away too, and the left corner, where it raised a tick)
        ux_ = np.array([pts[i][0] for i in OUTER_U]); uy_ = np.array([pts[i][1] for i in OUTER_U])
        mouth_ = (pc[:, 0] > pts[61][0] + E_('CORNER_FREE', 25)) & (pc[:, 0] < pts[291][0] - E_('CORNER_FREE', 25)) & (pc[:, 1] > np.interp(pc[:, 0], ux_, uy_)) & (pc[:, 1] > curve(pc[:, 0]) + 10)   # below the seam: above it a rim moved back left a sliver
        far_c &= mouth_
        sel = front & far_c; k_ = cand[sel]
        if not len(k_): break
        rays = P[k_] - Cc; rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        cosz = -(rays @ Rc)[:, 2]
        P[k_] = Cc + rays * ((zs[sel] + E_('BACK_MM', 0.5) / 1000) / cosz)[:, None]
        print("INSIDE_BACK round %d: %d vertices of the inside put behind the skin" % (rnd, len(k_)))
        if os.environ.get("INSIDE_DEBUG"): print(np.c_[k_, project(P[k_]).round(0), ((zs[sel] - dep[sel]) * 1000).round(2)])
n_ref = np.cross(V_orig[tv[:, 1]] - V_orig[tv[:, 0]], V_orig[tv[:, 2]] - V_orig[tv[:, 0]]); n_new = np.cross(P[tv[:, 1]] - P[tv[:, 0]], P[tv[:, 2]] - P[tv[:, 0]])
reg_t = region[tv].any(1)
print('folds: flipped %d, crushed %d' % (int((((n_ref * n_new).sum(1) < 0) & reg_t).sum()), int(((np.linalg.norm(n_new, axis=1) < 0.15 * np.linalg.norm(n_ref, axis=1)) & reg_t).sum())))
total = P - V_orig
mv_ = np.flatnonzero(np.linalg.norm(total, axis=1) > 1e-5)
np.savez(f'{O}/mouth_delta.npz', delta=total, pos1=V_orig[mv_], off1=total[mv_], pos2=np.full((1, 3), 1e3), off2=np.zeros((1, 3)))
print('wrote', f'{O}/mouth_delta.npz', 'max mm %.2f' % (np.abs(total).max() * 1000))
