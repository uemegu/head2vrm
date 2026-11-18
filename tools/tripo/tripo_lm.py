"""MediaPipe landmarks on the flat-lit Tripo views, cast onto the Tripo surface -> 3D landmarks (Blender coords).

python tripo_lm.py OUT_DIR   (reads views.json, tripo_mesh.npz; writes lm3d.npy (478,3), NaN where missed)
The front view gives the points; the two 35-degree views only report how well they agree.
"""
import json, os, sys
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mpfb', 'models', 'face_landmarker.task')
out = sys.argv[1]
m = np.load(f'{out}/tripo_mesh.npz')
V = m['co'][m['tris']]                                 # (T,3,3)


def cast(orig, d):
    """nearest hit of each ray (orig[i], shared dir d) on the mesh; NaN if none (Moller-Trumbore)."""
    e1, e2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    hits = np.full((len(orig), 3), np.nan)
    for i, o in enumerate(orig):
        s = o - V[:, 0]
        u = (s * p).sum(1) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = (e2 * q).sum(1) * inv
        good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if good.any():
            hits[i] = o + d * t[good].min()
    return hits


opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), num_faces=1)
res3d = {}
with vision.FaceLandmarker.create_from_options(opts) as det:
    for cam in json.load(open(f'{out}/views.json')):
        img = cv2.cvtColor(cv2.imread(cam['path']), cv2.COLOR_BGR2RGB)
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
        if not r.face_landmarks:
            print('no face', cam['angle']); continue
        uv = np.array([[p.x, p.y] for p in r.face_landmarks[0]])
        M = np.array(cam['matrix_world']); S = cam['ortho_scale']
        right, up, back = M[:3, 0], M[:3, 1], M[:3, 2]
        orig = M[:3, 3] + np.outer((uv[:, 0] - 0.5) * S, right) + np.outer((0.5 - uv[:, 1]) * S, up)
        res3d[cam['angle']] = cast(orig, -back)
        if cam['angle'] == 0:                  # compose_skin.py's jaw outline: rays past the jaw can miss the mesh, so no depth (front_morph.py replaces it with the portrait's)
            np.save(os.path.join(sys.argv[1], 'lm_plane.npy'), orig)
        dbg = img.copy()
        for x, y in uv * cam['size']:
            cv2.circle(dbg, (int(x), int(y)), 2, (255, 0, 0), -1)
        cv2.imwrite(cam['path'].replace('.png', '_lm.jpg'), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
P = res3d[0]
# PLUG (hairgraft/defan.py): inside the eye openings cut out of the skin a ray has no skin to land on (eyeball, or the skull's
# inside): the depth comes from the smooth surface fitted to the skin round each opening
fixed = 0
if os.environ.get('PLUG'):
    for cx, cz, A, B, *k in np.load(os.environ['PLUG']):
        dx, dz = P[:, 0] - cx, P[:, 2] - cz
        r2 = (dx / A) ** 2 + (dz / B) ** 2
        f = np.array(k) @ np.stack([np.ones_like(dx), dx, dz, dx ** 2, dx * dz, dz ** 2])
        m = (r2 <= 1) | ((r2 < 2.5) & (P[:, 1] - f > 0.03))     # cut faces reach past the ellipse: there only the rays that went deep
        P[m, 1] = f[m]
        fixed += int(m.sum())
# a ray just past the jaw lands on the neck far behind: depth from the neighbours' plane
ok = ~np.isnan(P[:, 0])
width = np.ptp(P[ok, 0])
Q = P.copy()
for i in np.where(ok)[0]:
    r = np.hypot(P[:, 0] - P[i, 0], P[:, 2] - P[i, 2])
    nb = np.where(ok & (r < 0.25 * width) & (r > 0))[0]
    if len(nb) < 8 or P[i, 1] - np.median(P[nb, 1]) < 0.4 * width:
        continue
    nb = nb[P[nb, 1] < np.percentile(P[nb, 1], 70)]
    Q[i, 1] = np.linalg.lstsq(np.c_[P[nb, 0], P[nb, 2], np.ones(len(nb))], P[nb, 1], rcond=None)[0] @ [P[i, 0], P[i, 2], 1]
    fixed += 1
P = Q
# the jaw line: a ray just inside the chin's silhouette can slip past it onto the neck behind (on a test head with its eyes turned,
# the chin 152 lay 23 cm back and tripo_fit.py dropped it). A jaw point far behind its jaw neighbours takes their depth.
JAW_LINE = (172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397)
for k, i in enumerate(JAW_LINE[1:-1], 1):
    nbs = [JAW_LINE[k - 1], JAW_LINE[k + 1]]
    if np.isnan(P[i, 1]) or any(np.isnan(P[n, 1]) for n in nbs):
        continue
    d = np.mean([P[n, 1] for n in nbs])
    if P[i, 1] - d > 0.05:
        print('jaw depth fixed', i, round(float(P[i, 1]), 3), '->', round(float(d), 3))
        P[i, 1] = d
# a jaw-line ray can graze past the face's lower edge onto a separate neck part behind it (on a part-split test head:
# the chin 152 hit the neck 0.19 behind the chin's front, and its neighbours 148/377 half that), so neighbours cannot vouch
# for each other. The silhouette point is where the ray is tangent: take the frontmost surface point within a hair of the ray.
g = np.array([[(6 - i - j) / 6, i / 6, j / 6] for i in range(7) for j in range(7 - i)])
SP = np.einsum('gk,tkd->tgd', g, V).reshape(-1, 3)
for i in (58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288):
    if np.isnan(P[i, 1]):
        continue
    near = np.hypot(SP[:, 0] - P[i, 0], SP[:, 2] - P[i, 2]) < 0.008
    if near.any() and SP[near, 1].min() < P[i, 1] - 0.02:
        print('jaw silhouette', i, round(float(P[i, 1]), 3), '->', round(float(SP[near, 1].min()), 3))
        P[i, 1] = SP[near, 1].min()
print('front hits', int((~np.isnan(P[:, 0])).sum()), 'depth from plug', fixed)
np.save(f'{out}/lm3d.npy', P)
for a in (-35, 35):
    if a in res3d:
        d = np.linalg.norm(res3d[a][:468] - P[:468], axis=1) * 1000
        print('agree', a, 'median (tripo units x1000)', round(float(np.nanmedian(d)), 2))
