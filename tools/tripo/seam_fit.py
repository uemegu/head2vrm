"""Bring the mesh's lid margins and lip seam, and the bangs' inner edge, to the portrait. tripo_fit.py drops the lid and inner-lip
points (corr.npz ties them to the socket), so the mesh's opening keeps MPFB's shape: a tall, blunt-cornered eye and a straight upper
lip margin, while face_paint.py paints the portrait's lines beside them. The real margins are where the head's UV island meets the
socket and mouth islands. Each branch of them goes along the same landmark curve that face_paint.py paints, by arc length, and the skin
follows within a few mm on its own side of the opening.

python seam_fit.py PORTRAIT OUT_DIR CORR_NPZ [HAIR_NPZ]
Writes OUT_DIR/seam.npz (world offsets of Human's vertices, the margins for the lashes) and, with HAIR_NPZ, OUT_DIR/bang.npz
(offsets of the hair's vertices in the base frame, for hair_graft.py).
"""
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mpfb'))
from fit import phi, smoothstep  # noqa: E402

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mpfb', 'models', 'face_landmarker.task')
portrait, out, corr = sys.argv[1:4]
hair_npz = sys.argv[4] if len(sys.argv) > 4 else None
img = cv2.cvtColor(cv2.imread(portrait), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), num_faces=1)
with vision.FaceLandmarker.create_from_options(opts) as det:
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
px = np.array([[p.x * W, p.y * H] for p in r.face_landmarks[0]])[:468]

# face_paint.py's mapping: the same points, so the margins land on the painted lines
w = np.load(f'{out}/warp.npz')
s, R, t = float(w['scale']), w['rot'], w['trans']
tgt = (s * (R @ np.load(f'{out}/lm3d.npy')[:468].T)).T + t
c = np.load(corr)
co0, trip, bary, ids = c['co'], c['trip'], c['bary'], c['ids']
warped = co0 + smoothstep(co0[:, 2], float(w['z_lo']), float(w['z_hi']))[:, None] * phi(co0, w['centers'], float(w['sigma'])) @ w['W']
pts = tgt.copy()
for k, i in enumerate(ids):
    if i < 468:
        p = (warped[trip[k]] * bary[k][:, None]).sum(0)
        if np.linalg.norm(p - tgt[i]) < 0.008:
            pts[i] = p
good = ~np.isnan(pts[:, 0])
to_base = RBFInterpolator(px[good], pts[good][:, [0, 2]], kernel='thin_plate_spline', smoothing=1e-5, neighbors=120)
# SEAM_TGT=1: the lid and lip rings aim at Tripo's own 3D landmarks. Without it, a landmark within 8 mm of the warped mesh
# is replaced by the mesh's own point, so the margins stay where the fit left them (on one test head: lip middle 2 mm high, corners 2-3 mm
# low, a flat mouth instead of Tripo's smile).
if os.environ.get('SEAM_TGT'):
    pts = tgt.copy()

u = np.load(f'{out}/uvmesh.npz')
V, tv, tuv = u['co'], u['tv'], u['tuv']
n = len(tv)
seen, a_, b_ = {}, [], []
for tri in range(n):
    for k in range(3):
        key = (int(tv[tri, k]), round(float(tuv[tri, k, 0]), 5), round(float(tuv[tri, k, 1]), 5))
        if key in seen:
            a_.append(tri); b_.append(seen[key])
        else:
            seen[key] = tri
_, lab = connected_components(coo_matrix((np.ones(len(a_)), (a_, b_)), shape=(n, n)), directed=False)


def island(uv_box):
    (u0, v0), (u1, v1) = uv_box
    c_ = tuv.mean(1)
    return np.bincount(lab[(c_[:, 0] > u0) & (c_[:, 0] < u1) & (c_[:, 1] > v0) & (c_[:, 1] < v1)]).argmax()


HEAD = island(((0.8, 0.4), (0.95, 0.6)))
head_v = set(tv[lab == HEAD].ravel().tolist())


def branches(isl):
    """the seam of island isl with the head as (upper, lower) vertex paths from the x-min corner to the x-max corner"""
    tris = tv[lab == isl]
    S = head_v & set(tris.ravel().tolist())
    nb = {v: set() for v in S}
    for tri in tris:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if tri[a] in S and tri[b] in S:
                nb[tri[a]].add(int(tri[b])); nb[tri[b]].add(int(tri[a]))
    # the seam is one cycle; a diagonal between two seam vertices is left out by walking only degree-2 steps, shortest turn first
    start = min(S, key=lambda v: V[v, 0])
    path, prev, cur = [start], None, start
    while True:
        opts_ = [v for v in nb[cur] if v != prev and (v not in path or (v == start and len(path) > 3))]
        if not opts_:
            break
        nxt = min(opts_, key=lambda v: np.linalg.norm(V[v] - V[cur]))
        if nxt == start:
            break
        path.append(nxt); prev, cur = cur, nxt
    end = int(np.argmax(V[path, 0]))
    a, b = path[:end + 1], path[end:] + [path[0]]
    b = b[::-1]
    return (a, b) if V[a, 2].mean() >= V[b, 2].mean() else (b, a)


def along(P, f):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    d /= d[-1]
    return np.c_[[np.interp(f, d, P[:, k]) for k in range(P.shape[1])]].T


def frac(P):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return d / d[-1]


LIPS = ((78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308), (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308))
EYE_L = ((33, 246, 161, 160, 159, 158, 157, 173, 133), (33, 7, 163, 144, 145, 153, 154, 155, 133))
EYE_R = ((133 + 229, 398, 384, 385, 386, 387, 388, 466, 263), (362, 382, 381, 380, 374, 373, 390, 249, 263))
eye_co = u['eye_co']
delta = np.zeros_like(V)
anchors = []
for name, uvb, rings, reach in (
        ('lips', ((0.76, 0.01), (0.92, 0.14)), LIPS, (0.006, 0.004)),
        ('eyeA', ((0.01, 0.54), (0.08, 0.64)), EYE_L if pts[33, 0] < 0 else EYE_R, (0.008, 0.005)),
        ('eyeB', ((0.085, 0.54), (0.155, 0.64)), EYE_R if pts[33, 0] < 0 else EYE_L, (0.008, 0.005))):
    up, lo = branches(island(uvb))
    upper_t = pts[list(rings[0])]
    lower_t = pts[list(rings[1])]
    if upper_t[0, 0] > upper_t[-1, 0]:
        upper_t, lower_t = upper_t[::-1], lower_t[::-1]
    if name != 'lips' and os.environ.get('SEAM_EYE_NPZ'):
        # the lid margins' targets given directly (an external fitting loop: Tripo's painted opening, measured on a close render and
        # corrected by what the mesh shows): polylines in (x, z), ascending x, keyed by the eye's side
        E_ = np.load(os.environ['SEAM_EYE_NPZ'])
        side_ = 'pos' if upper_t[:, 0].mean() > 0 else 'neg'
        for key_, ring_ in (('upper', 'u'), ('lower', 'l')):
            T_ = E_[f'{side_}_{key_}']
            old_ = upper_t if ring_ == 'u' else lower_t
            o_ = np.argsort(old_[:, 0])
            new_ = np.c_[T_[:, 0], np.interp(T_[:, 0], old_[o_, 0], old_[o_, 1]), T_[:, 1]]
            if ring_ == 'u':
                upper_t = new_
            else:
                lower_t = new_
    if name != 'lips' and os.environ.get('EYE_OPEN'):         # the upper lid's arch above the corners' line, raised (a Tripo eye can be smaller than its portrait's)
        chord = np.interp(upper_t[:, 0], upper_t[[0, -1], 0], upper_t[[0, -1], 2])
        upper_t[:, 2] = chord + (upper_t[:, 2] - chord) * float(os.environ['EYE_OPEN'])
        chord = np.interp(lower_t[:, 0], lower_t[[0, -1], 0], lower_t[[0, -1], 2])
        lower_t[:, 2] = chord + (lower_t[:, 2] - chord) * float(os.environ.get('EYE_OPEN_LOW', 1))
    if name != 'lips' and os.environ.get('EYE_FAN'):          # the upper lid as a fan: from a pointed inner corner it rises quickly and round, then a long slope to the tail
        T_, H_ = np.array(((0, .05, .12, .2, .39, .6, .83, .93, 1), (0, .18, .45, .72, 1, .97, .72, .42, 0)))
        inner_first = abs(upper_t[0, 0]) < abs(upper_t[-1, 0])
        p0, p1 = (upper_t[0], upper_t[-1]) if inner_first else (upper_t[-1], upper_t[0])
        tt = np.linspace(0, 1, 41)
        peak = float((upper_t[:, 2] - np.interp(upper_t[:, 0], upper_t[[0, -1], 0], upper_t[[0, -1], 2])).max())
        fine = p0[None] + (p1 - p0)[None] * tt[:, None]
        fine[:, 2] += peak * np.interp(tt, T_, H_)
        y_old = upper_t[np.argsort(upper_t[:, 0]), 1]
        fine[:, 1] = np.interp(fine[:, 0], np.sort(upper_t[:, 0]), y_old)
        upper_t = fine if inner_first else fine[::-1]
    if name != 'lips' and os.environ.get('EYE_TILT'):         # degrees: the tail up, the inner corner a little down, about the eye's middle
        xc = (upper_t[0, 0] + upper_t[-1, 0]) / 2
        k = np.tan(np.radians(float(os.environ['EYE_TILT'])))
        upper_t[:, 2] += (np.abs(upper_t[:, 0]) - abs(xc)) * k
        lower_t[:, 2] += (np.abs(lower_t[:, 0]) - abs(xc)) * k
    if name == 'lips':
        # the mouth's corners stay on the mesh and the painted seam comes to them (seam.npz's pins, read by face_paint.py): moving a corner to the
        # landmarks' uneven height folded the skin beside it
        # SEAM_CORNERS=1: the corners go to the landmarks too (a plain Tripo head: its own landmarks, no portrait's uneven
        # ones). On one test head the fit left the mesh corners ~2 mm below Tripo's, and pinning the seam's ends there flattened its smile.
        for ring, path in ((upper_t, up), (lower_t, lo)):
            if os.environ.get('SEAM_CORNERS'):
                break
            f_ = frac(ring[:, [0, 2]])[:, None]
            ring += (V[path[0]] - ring[0]) * (1 - f_) + (V[path[-1]] - ring[-1]) * f_
        L0, L1 = (V[up[0]] - pts[78])[[0, 2]], (V[up[-1]] - pts[308])[[0, 2]]
        if pts[78, 0] > pts[308, 0]:
            L0, L1 = (V[up[-1]] - pts[78])[[0, 2]], (V[up[0]] - pts[308])[[0, 2]]
        pin_ids = list(LIPS[0]) + list(LIPS[1]) + [61, 291]
        order = -1 if pts[78, 0] > pts[308, 0] else 1
        pin_xz = np.vstack([upper_t[::order][:, [0, 2]], lower_t[::order][:, [0, 2]], pts[61, [0, 2]] + L0, pts[291, [0, 2]] + L1])
    d_up, d_lo = np.zeros((len(up), 3)), np.zeros((len(lo), 3))
    for path, ring, d_ in ((up, upper_t, d_up), (lo, lower_t, d_lo)):
        q = along(ring[:, [0, 2]], frac(V[path][:, [0, 2]]))
        d_[:, [0, 2]] = q - V[path][:, [0, 2]]
    if name != 'lips' and os.environ.get('SEAM_INNER'):      # only the inner corner and the lower lid: the upper lid and the outer corner stay as the fit left them. LID_DROP (mm): the upper margin comes down onto
        ramp_ = lambda v, a_, b_: (lambda t_: t_ * t_ * (3 - 2 * t_))(np.clip((v - a_) / (b_ - a_), 0, 1))   # where the lid meets the eyeball, or its inner side shows as a dark band over the iris
        for path, d_, a_, b_ in ((up, d_up, 0.06, 0.28), (lo, d_lo, 0.55, 0.95)):
            f_ = frac(V[path][:, [0, 2]])
            f_ = f_ if abs(V[path[0], 0]) < abs(V[path[-1], 0]) else 1 - f_
            d_ *= (1 - ramp_(f_, a_, b_))[:, None]
            if path is up:
                d_[:, 2] -= float(os.environ.get('LID_DROP', 0)) / 1000 * np.maximum(np.sin(np.pi * f_), 0) ** 0.8
    if name == 'lips':                                       # the upper margin sits just in front of the lower one where they meet
        y_lo = along(V[lo][:, [0, 1]], frac(V[up][:, [0, 2]]))[:, 1]
        d_up[:, 1] = (y_lo - 0.0003) - V[up, 1]
    else:                                                    # the margins stay on the eyeball's outside
        side = eye_co[np.sign(eye_co[:, 0]) == np.sign(V[up[0], 0])]
        centre = np.linalg.lstsq(np.c_[2 * side, np.ones(len(side))], (side ** 2).sum(1), rcond=None)[0][:3]
        for path, d_ in ((up, d_up), (lo, d_lo)):                # each point keeps its own clearance from the eye's centre
            p_ = V[path] + d_
            keep_r = np.linalg.norm(V[path] - centre, axis=1)
            flat = np.linalg.norm((p_ - centre)[:, [0, 2]], axis=1)
            y_s = centre[1] - np.sqrt(np.maximum(keep_r ** 2 - flat ** 2, 0))
            d_[:, 1] += np.minimum(y_s - p_[:, 1], 0)            # only forward (-y)
    for k_ in (0, -1):                                       # one corner point, whichever branch it is read from
        d_lo[k_] = d_up[k_]
    if name == 'lips' and os.environ.get('SEAM_INNER'):
        d_up[:], d_lo[:] = 0, 0
    print('SEAM', name, 'upper move mm max', round(float(np.linalg.norm(d_up, axis=1).max()) * 1000, 2), 'lower', round(float(np.linalg.norm(d_lo, axis=1).max()) * 1000, 2))
    x_u, z_u = V[up, 0], V[up, 2]
    x_l, z_l = V[lo, 0], V[lo, 2]
    mid = lambda x: (np.interp(x, x_u, z_u) + np.interp(x, x_l, z_l)) / 2
    is_up = V[:, 2] > mid(V[:, 0])
    lo_x, hi_x = x_u.min() - 0.012, x_u.max() + 0.012
    box = (V[:, 0] > lo_x) & (V[:, 0] < hi_x) & (np.abs(V[:, 2] - mid(V[:, 0])) < 0.02) & (V[:, 1] < V[up, 1].max() + 0.02)
    for side_up, path, d_, R_ in ((True, up, d_up, reach[0]), (False, lo, d_lo, reach[1])):
        idx = np.where(box & (is_up == side_up))[0]
        P = V[path]
        dist = np.linalg.norm(V[idx][:, None] - P[None], axis=2)
        wt = 1 / np.maximum(dist, 2e-4) ** 4
        val = (wt[..., None] * d_[None]).sum(1) / wt.sum(1, keepdims=True)
        dm = dist.min(1)
        tt = np.clip(dm / R_, 0, 1)
        delta[idx] = val * (1 - tt * tt * (3 - 2 * tt))[:, None]
        delta[path] = d_
    anchors.append(dict(pos=np.vstack([V[up], V[lo]]), off=np.vstack([d_up, d_lo]), upper=np.r_[np.ones(len(up), bool), np.zeros(len(lo), bool)]))
if os.environ.get('EYE_SYM'):
    # A tilted portrait leaves the skin round one eye about 2 mm above the other's. Both sides become one shape: the eyes and lids take the x < 0 side's (its lid is
    # open), the brows the x > 0 side's (its brow is at the portrait's height), crossing between them above the lid. The nose and everything outside stay.
    B_ = np.load(f'{out}/base_uvmesh.npz')['co']
    mi = cKDTree(B_ * [-1, 1, 1]).query(B_)[1]
    P_ = V + delta
    Pm = P_[mi] * [-1, 1, 1]
    neg = V[:, 0] < 0
    left, right = np.where(neg[..., None], P_, Pm), np.where(neg[..., None], Pm, P_)   # each vertex on the x < 0 side's shape and on the x > 0 side's
    z_ = (P_[:, 2] + P_[mi, 2]) / 2
    x_ = (np.abs(P_[:, 0]) + np.abs(P_[mi, 0])) / 2
    y_ = (P_[:, 1] + P_[mi, 1]) / 2
    ramp = lambda v, a_, b_: (lambda t_: t_ * t_ * (3 - 2 * t_))(np.clip((v - a_) / (b_ - a_), 0, 1))
    zt = max(float((an['pos'] + an['off'])[an['upper'], 2].max()) for an in anchors[1:])
    zl = min(float((an['pos'] + an['off'])[~an['upper'], 2].min()) for an in anchors[1:])
    brow = ramp(z_, zt + 0.003, zt + 0.009)[:, None]
    mask = (ramp(x_, 0.007, 0.012) * (1 - ramp(x_, 0.06, 0.072)) * ramp(z_, zl - 0.018, zl - 0.008) * (1 - ramp(z_, zt + 0.024, zt + 0.04)) * (1 - ramp(y_, -0.06, -0.045)))[:, None]
    sym = (left * (1 - brow) + right * brow - P_) * mask
    delta += sym
    for an in anchors[1:]:
        idx_ = cKDTree(V).query(an['pos'])[1]
        an['off'] = delta[idx_]
    print('SEAM symmetry move mm max', round(float(np.linalg.norm(sym, axis=1).max()) * 1000, 2), 'vertices', int((mask[:, 0] > 0).sum()))
np.savez(f'{out}/seam.npz', delta=delta, pin_ids=pin_ids, pin_xz=pin_xz, **{f'{k}{i}': v for i, an in enumerate(anchors) for k, v in an.items()})
print('SEAM skin vertices moved', int((np.linalg.norm(delta, axis=1) > 1e-5).sum()))

if hair_npz:
    # the bangs' inner edge, row by row, where the portrait's hair mass starts (brows and lashes are too thin to survive the opening)
    h = np.load(hair_npz)
    hv = (s * (R @ h['co'].T)).T + t
    lum = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), 1.5).astype(np.float32)
    x50, y50 = px[50].astype(int)
    cheek = np.median(lum[y50 - 15:y50 + 15, x50 - 15:x50 + 15])
    mass = cv2.morphologyEx((lum < 0.62 * cheek).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    cx = int(px[[168, 1, 152], 0].mean())
    y_top, y_bot = int(px[10, 1]), int(px[[33, 263], 1].mean())
    rows = np.arange(y_top, y_bot)
    front = hv[:, 1] < np.nanmax(pts[good][:, 1]) - 0.01
    PX = 2000                                                 # cells per metre, the bangs' silhouette from the front
    x0, z0 = -0.14, float(pts[good][:, 2].min())
    grid = np.zeros((int(0.25 * PX), int(0.28 * PX)), np.uint8)
    for tri in h['tris'][front[h['tris']].all(1)]:
        cv2.fillConvexPoly(grid, ((hv[tri][:, [0, 2]] - [x0, z0]) * PX).astype(np.int32), 1)
    offs = np.zeros(len(hv))
    bang = {}
    for sgn in (-1, 1):
        edge_px = []
        for y in rows:
            line = mass[y, cx::sgn] if sgn > 0 else mass[y, cx::-1]
            k = np.argmax(line)
            edge_px.append(cx + sgn * k if line[k] else np.nan)
        edge_px = np.array(edge_px, np.float32)
        ok = ~np.isnan(edge_px)
        target = to_base(np.c_[np.interp(rows, rows[ok], edge_px[ok]), rows])
        zs, xt = target[:, 1], target[:, 0]
        o_ = np.argsort(zs)
        zs, xt = zs[o_], gaussian_filter1d(xt[o_], 6)
        cur = []
        for z in zs:
            row = grid[int((z - z0) * PX)]
            cells = np.where(row)[0]
            cells = cells[(cells / PX + x0) * sgn > 0]
            cur.append((cells[np.argmin(np.abs(cells / PX + x0))] / PX + x0) if len(cells) else np.nan)
        bang[sgn] = (zs, xt, np.array(cur))
    # one target for both sides, the mean of the portrait's two uneven edges; a low-order curve, since the shell has few vertices and follows every wiggle
    z_all = np.linspace(max(b_[0][0] for b_ in bang.values()), min(b_[0][-1] for b_ in bang.values()), 300)
    t_ = np.mean([np.abs(np.interp(z_all, b_[0], b_[1])) for b_ in bang.values()], 0)
    for sgn in (-1, 1):
        zs, xt, cur = bang[sgn]
        cur = np.interp(z_all, zs[~np.isnan(cur)], cur[~np.isnan(cur)])
        need = np.maximum(np.abs(cur) - t_, 0)                    # inwards only
        need = np.polyval(np.polyfit(z_all, need, 3), z_all).clip(0)
        # nothing at the hairline's top, all by the brows, and back to nothing at the eye's tail so it stays clear
        fz = (z_all - z_all[0]) / np.ptp(z_all)
        need = need * np.clip((1 - fz) / 0.7, 0, 1) ** 1.5 * np.clip(fz / 0.3, 0, 1) ** 1.5
        need = gaussian_filter1d(need, 12)
        bang[sgn] = (z_all, need, gaussian_filter1d(cur, 12))
        print('BANG side', sgn, 'in mm: top', round(float(need[-20] * 1000), 1), 'middle', round(float(need[len(need) // 2] * 1000), 1), 'bottom', round(float(need[20] * 1000), 1), 'max', round(float(need.max() * 1000), 1))
    off = np.zeros_like(hv)
    for sgn, (zs, need, cur) in bang.items():
        m = np.sign(hv[:, 0]) == sgn
        z_ = hv[m, 2]
        nd = np.interp(z_, zs, need, left=0, right=0)
        beyond = np.abs(hv[m, 0]) - np.abs(np.interp(z_, zs, cur))
        tt = np.clip(beyond / 0.03, 0, 1)                     # the edge moves most, gone 3 cm outwards
        off[m, 0] = -sgn * nd * (1 - tt * tt * (3 - 2 * tt)) * front[m]
    np.savez(f'{out}/bang.npz', offset=off)
    print('BANG hair vertices moved', int((np.abs(off[:, 0]) > 1e-5).sum()), 'of', len(hv))
