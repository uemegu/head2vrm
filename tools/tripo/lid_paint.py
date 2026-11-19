"""The lid skin next to the lashes made programmatically from the approved lid lines, instead of patching the painted texture.   any face
python lid_paint.py OUT_DIR SKIN_IN SKIN_OUT      (after lid_project.py; eyeliner.py draws the line after it; lash_new.py the lashes)

Why: the input texture can paint the lashes densely over the 2 mm of lid above the margin (and sparser ones under the lower lid),
so there is no skin under them to recover. Erasing and filling (lash_erase.py, then lid_clean.py: smear, flat patches, grey) only
ever invented colour. So it is made programmatically from the known shape, not by matching the original.
How, per eye, in the lid-fit camera (lidfit/warp/v0_cam.json) against the approved lines (lidfit/check_lines.txt):
  upper lid: texels from the upper line up to UP_H mm (and EXT mm past either corner) get the clean skin found UP_H..UP_H+1 mm above
  the line at their x (median per column, smoothed along x), darkened towards the margin by SHADE at the line (fading over SHADE_H mm),
  plus the source's own fine grain (its difference from a 2-texel blur, clipped to +-GRAIN, so a painted lash adds no more than that);
  blended into the original over the top BLEND mm and past the corners.
  lower lid: the same from KEEP_LO mm (the pink waterline stays) down to LO_H mm, the clean skin LO_H..LO_H+1 mm under the line.
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

O, src, dst = sys.argv[1:4]
E = lambda k, d: float(os.environ.get(k, d))
UP_H, LO_H, KEEP_LO, EXT, BLEND = E('UP_H', 4.5), E('LO_H', 2.2), E('KEEP_LO', 0.45), E('EXT', 3.5), E('BLEND', 1.0)
CAR, EXT_IN = E('CAR', 2.5), E('EXT_IN', 0.4)
SHADE, SHADE_H, GRAIN = E('SHADE', 0.9), E('SHADE_H', 2.0), E('GRAIN', 4)

cam = json.load(open(f'{O}/lidfit/warp/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1]; F = (RES / 2) / np.tan(cam['fov'] / 2)
ln = [l for l in open(f'{O}/lidfit/check_lines.txt').read().split('\n') if l.strip()]
eyes = []
for k in range(0, len(ln), 3):
    x0, x1 = map(float, ln[k].split())
    up, lo = np.array(ln[k + 1].split(), float), np.array(ln[k + 2].split(), float)
    eyes.append((x0, x1, np.linspace(x0, x1, len(up)), up, lo))

t = np.load(f'{O}/texels.npz'); pos, nrm, have = t['pos'], t['nrm'], t['have']
EC = np.load(f'{O}/uvmesh.npz')['eye_co']
img = cv2.imread(src).astype(np.float32)
grain = img - cv2.GaussianBlur(img, (0, 0), 2)
# no grain from the painted lashes: where a thin dark line is (black-hat), none (their ghosts were left as dirt)
Lg = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0]
bh = cv2.morphologyEx(Lg, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
grain *= (cv2.dilate((bh > 4).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)[..., None]
out = img.copy()
for sgn in (1, -1):
    b = EC[np.sign(EC[:, 0]) == sgn]; bb = b[b[:, 1] > np.median(b[:, 1])]
    s_ = np.linalg.lstsq(np.c_[2 * bb, np.ones(len(bb))], (bb ** 2).sum(1), rcond=None)[0]
    c = s_[:3]; r = float(np.sqrt(s_[3] + c @ c))
    near = have & (np.abs(pos[..., 0] - c[0]) < 1.6 * r) & (np.abs(pos[..., 2] - c[2]) < 1.0 * r) & (pos[..., 1] < c[1])
    view = Cc - pos; facing = (nrm * view).sum(-1) > 0
    ys, xs = np.nonzero(near & facing)
    q = (pos[ys, xs] - Cc) @ Rc; d = -q[:, 2]
    px, py = RES / 2 + F * q[:, 0] / d, RES / 2 - F * q[:, 1] / d
    mm = d / F * 1000
    cx = RES / 2 + F * ((c - Cc) @ Rc)[0] / -((c - Cc) @ Rc)[2]
    x0, x1, lx, lu, ll = min(eyes, key=lambda e_: abs((e_[0] + e_[1]) / 2 - cx))
    pxc = np.clip(px, x0, x1)
    past = np.maximum(x0 - px, px - x1) * mm                    # mm past the nearer corner (negative inside)
    # 1 inside, fading to 0 EXT mm past the outer corner (the painted lashes sweep out there) but only EXT_IN past the inner one: reaching
    # past it, the lower band took the red under the inner corner as its skin and left a pink spot by the nose
    inner_left = abs(x0 - RES / 2) < abs(x1 - RES / 2)
    past_inner = (px < x0) if inner_left else (px > x1)
    side = np.clip(1 - np.maximum(past, 0) / np.where(past_inner, EXT_IN, EXT), 0, 1)
    above = (np.interp(pxc, lx, lu) - py) * mm                  # mm above the upper line
    below = (py - np.interp(pxc, lx, ll)) * mm                  # mm below the lower line
    col = img[ys, xs]; gr = np.clip(grain[ys, xs], -GRAIN, GRAIN)
    # past the inner corner the caruncle's pink painted in the texture fell on the skin by the nose, a pink spot: within CAR mm past
    # that corner and near its height, texels redder than the skin round them take that skin's colour
    inner_x = x0 if abs(x0 - RES / 2) < abs(x1 - RES / 2) else x1
    ci = np.argmin(np.abs(lx - inner_x)); cy = (lu[ci] + ll[ci]) / 2
    dpx = (px - inner_x) * (1 if inner_x == x1 else -1)            # px past the inner corner (towards the nose)
    zone_c = (dpx * mm > -0.3) & (dpx * mm < CAR) & (np.abs(py - cy) * mm < 1.6)
    ring = (dpx * mm > CAR) & (dpx * mm < CAR + 1.2) & (np.abs(py - cy) * mm < 1.6)
    if zone_c.any() and ring.sum() > 10:
        skin_c = np.median(col[ring], 0)
        lab_c = cv2.cvtColor(col[None].clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[0].astype(np.float32)
        a_skin = cv2.cvtColor(skin_c[None, None].astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0, 1].astype(np.float32)
        wc = np.clip((lab_c[:, 1] - a_skin - 2) / 4, 0, 1) * zone_c
        col = col * (1 - wc[:, None]) + (skin_c + gr) * wc[:, None]          # the source the lid bands blend from too
        out[ys, xs] = col
        print('LID_PAINT eye %+d caruncle spot: texels %d' % (sgn, int((wc > 0.5).sum())))
    for name, h, H, keep in (('upper', above, UP_H, 0.0), ('lower', below, LO_H, KEEP_LO)):
        # the clean skin just beyond the band, per column of the image (2 px bins), smoothed along x
        ref = (h > H) & (h < H + 1.0) & (past < EXT)
        bins = ((px - x0) / 2).astype(int)
        nb = int((x1 - x0) / 2) + int(EXT / np.median(mm) / 2) * 2 + 10
        bins_c = np.clip(bins + nb // 2 - int((x1 - x0) / 4), 0, nb - 1)
        S = np.zeros((nb, 3)); C = np.zeros(nb)
        for ch in range(3):
            np.add.at(S[:, ch], bins_c[ref], col[ref, ch])
        np.add.at(C, bins_c[ref], 1)
        S = gaussian_filter1d(S, 6, axis=0); C = gaussian_filter1d(C, 6)
        skin = S[bins_c] / np.maximum(C[bins_c], 1e-6)[:, None]
        okc = C[bins_c] > 1e-3
        shade = 1 - (1 - SHADE) * np.clip(1 - (h - keep) / SHADE_H, 0, 1)
        synth = skin * shade[:, None] + gr
        w = np.clip((H - h) / BLEND, 0, 1) * np.clip((h - keep) / 0.15, 0, 1) * side * okc
        w = w * (h > keep - 0.2)
        sel = w > 0
        out[ys[sel], xs[sel]] = col[sel] * (1 - w[sel, None]) + synth[sel] * w[sel, None]
        print('LID_PAINT eye %+d %s: texels %d (full %d), clean-skin samples %d' % (sgn, name, sel.sum(), (w > 0.99).sum(), ref.sum()))
cv2.imwrite(dst, np.clip(out, 0, 255).round().astype(np.uint8))
print('LID_PAINT wrote', dst)
