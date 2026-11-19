"""A clean eyeliner drawn along the approved upper lid line, instead of keeping what the input texture shows there.   any face
python eyeliner.py OUT_DIR SKIN_IN SKIN_OUT        (after lash_erase.py with BAND_LO=0, so the painted roots are gone)

Why: the band of the painted texture kept along the margin (lash_erase's BAND_LO) held the painted lashes' roots,
and its top was cut at a height interpolated from a dozen margin vertices: a ragged brown strip. With the line's shape
known, a liner drawn programmatically is cleaner than one patched to match the original painting.
How: the margins were fitted to the approved lines (lidfit/check_lines.txt) as seen from the lid-fit camera (lidfit/warp/v0_cam.json).
Each skin texel near an upper lid (3D: within REACH above the eyeball's centre band, in front of the ball) is projected into that
camera; its distance above the upper line, in mm at the eye, sets the liner's cover: full below the line (the lid's under-edge
seen from the front) and up to the width w(s), fading over SOFT. w(s) runs from W_IN at the inner corner to W_OUT at the outer
one (s = 0..1 along the line, smoothstep), tapering to nothing over the last TAPER of either end.
PROFILE=1 (default; one line did not read as a double lid): two bands from the input's measured order, the rim (RIM_RGB, to RIM_IN..RIM_OUT mm)
then the fold line (LINE_RGB, LN_IN..LN_OUT mm more, crisp top CRISP). PROFILE=0: one line W_IN..W_OUT.
Colour: LINER_RGB (sRGB; default the darkest of the painted lash line measured on one test head: 60,40,36), opacity ALPHA.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

O, src, dst = sys.argv[1:4]
E = lambda k, d: float(os.environ.get(k, d))
W_IN, W_OUT, SOFT, TAPER, ALPHA = E('W_IN', 0.25), E('W_OUT', 0.6), E('SOFT', 0.25), E('TAPER', 0.08), E('ALPHA', 0.9)
REACH = E('REACH', 0.006)
RGB = np.array([float(v) for v in os.environ.get('LINER_RGB', '60,40,36').split(',')])
PROFILE = os.environ.get('PROFILE', '1') == '1'
RIM_RGB = np.array([float(v) for v in os.environ.get('RIM_RGB', '118,88,82').split(',')])
LINE_RGB = np.array([float(v) for v in os.environ.get('LINE_RGB', '98,70,64').split(',')])
RIM_IN, RIM_OUT, LN_IN, LN_OUT, CRISP = E('RIM_IN', 0.3), E('RIM_OUT', 0.5), E('LN_IN', 0.25), E('LN_OUT', 0.5), E('CRISP', 0.06)

cam = json.load(open(f'{O}/lidfit/warp/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1]; F = (RES / 2) / np.tan(cam['fov'] / 2)
lines = open(f'{O}/lidfit/check_lines.txt').read().split('\n')
eyes = []
for k in range(0, len(lines) - 2, 3):
    if not lines[k].strip():
        continue
    x0, x1 = map(float, lines[k].split())
    up = np.array(lines[k + 1].split(), float)
    eyes.append((x0, x1, np.linspace(x0, x1, len(up)), up))


def project(P):
    q = (P - Cc) @ Rc
    return np.c_[RES / 2 + F * q[:, 0] / -q[:, 2], RES / 2 - F * q[:, 1] / -q[:, 2]], -q[:, 2]


t = np.load(f'{O}/texels.npz'); pos, have = t['pos'], t['have']
u = np.load(f'{O}/uvmesh.npz'); EC = u['eye_co']
img = np.asarray(Image.open(src).convert('RGB'), np.float32)
ys, xs = np.nonzero(have); q = pos[ys, xs]
cover = np.zeros(len(q), np.float32)
colour = np.tile(RGB, (len(q), 1)).astype(np.float32)
for sgn in (1, -1):
    b = EC[np.sign(EC[:, 0]) == sgn]; bb = b[b[:, 1] > np.median(b[:, 1])]
    s_ = np.linalg.lstsq(np.c_[2 * bb, np.ones(len(bb))], (bb ** 2).sum(1), rcond=None)[0]
    c = s_[:3]; r = float(np.sqrt(s_[3] + c @ c))
    near = (np.abs(q[:, 0] - c[0]) < 1.3 * r) & (q[:, 2] > c[2] - 0.2 * r) & (q[:, 2] < c[2] + 0.75 * r) & (q[:, 1] < c[1])
    near &= np.linalg.norm(q - c, axis=1) < r + REACH
    k = np.flatnonzero(near)
    px, depth = project(q[k])
    mm_px = depth / F * 1000                                  # mm per pixel at each texel
    cx = project(c[None])[0][0, 0]
    e = min(eyes, key=lambda e_: abs((e_[0] + e_[1]) / 2 - cx))   # this eye's line
    x0, x1, lx, ly = e
    inside = (px[:, 0] >= x0) & (px[:, 0] <= x1)
    yl = np.interp(px[:, 0], lx, ly)
    above = (yl - px[:, 1]) * mm_px                           # mm above the line (image y grows down)
    # s: 0 at the inner corner .. 1 at the outer one (the inner corner is the end nearer the nose: the image centre)
    s = (px[:, 0] - x0) / (x1 - x0)
    if abs(x0 - RES / 2) < abs(x1 - RES / 2):
        pass                                                  # x0 is the inner corner
    else:
        s = 1 - s
    ss = s * s * (3 - 2 * s)
    ends = np.clip(s / TAPER, 0, 1) ** 0.7 * np.clip((1 - s) / TAPER, 0, 1) ** 0.7
    if PROFILE:
        # the input's own order above the eyeball's edge (medians over the eye's width, on one test head): a grey-brown rim to ~0.5 mm, then a darker
        # line to ~1.0 mm (what reads as the double lid's fold), then the lashes. Drawn as two bands with these widths
        bt = (RIM_IN + (RIM_OUT - RIM_IN) * ss) * ends                  # the rim band's top
        lt = bt + (LN_IN + (LN_OUT - LN_IN) * ss) * ends                # the line's top
        rim_c = np.clip((bt - above) / 0.08 + 0.5, 0, 1) * np.clip((above + 0.05) / 0.08 + 0.5, 0, 1)
        line_c = np.clip((lt - above) / CRISP + 0.5, 0, 1) * np.clip((above - bt) / 0.15 + 0.5, 0, 1)
        cv = np.maximum(rim_c, line_c) * inside
        colk = np.where(line_c[:, None] > rim_c[:, None], LINE_RGB, RIM_RGB)
    else:
        w = (W_IN + (W_OUT - W_IN) * ss) * ends
        cv = np.clip((w - above) / SOFT + 0.5, 0, 1) * inside
        colk = np.tile(RGB, (len(cv), 1))
    cv[above < -1.5] = 0                                      # far below the line: not the lid (the socket's inside)
    upd = cv > cover[k]
    cover[k[upd]] = cv[upd]; colour[k[upd]] = colk[upd]
    print('EYELINER eye %+d: texels near %d, covered %d (> 0.5: %d)' % (sgn, len(k), int((cv > 0).sum()), int((cv > 0.5).sum())))
a = (cover * ALPHA)[:, None]
img[ys, xs] = img[ys, xs] * (1 - a) + colour * a
Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8)).save(dst)
print('EYELINER wrote', dst)
