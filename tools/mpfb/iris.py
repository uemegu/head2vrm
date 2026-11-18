"""Colour the MPFB eye texture's iris like the reference image's iris.

python iris.py REF_IMAGE EYE_TEXTURE OUT_PNG
The reference iris is unrolled to polar coordinates from its left and right sides only (the lids cover the
top and the bottom), catchlights are dropped, and its colour per radius (pupil, inner iris, outer iris, limbal ring) recolours the
texture's own high-resolution iris pattern (the reference iris is only ~18 px in radius).
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from lm import detect

ref_path, eye_path, out_path = sys.argv[1:4]
ref = np.asarray(Image.open(ref_path).convert('RGB')).astype(np.float32) / 255
m = detect([ref_path])[0]
NR, NA = 64, 64                         # radial / angular samples
HALF = np.radians(32)                   # sector half-width around the horizontal
strips = []
for c, ring in ((468, range(469, 473)), (473, range(474, 478))):
    ctr = m[c, :2]
    rad = np.mean([np.linalg.norm(m[i, :2] - ctr) for i in ring])
    r = np.linspace(0, 1.08, NR)[:, None] * rad
    for base, flip in ((0.0, False), (np.pi, True)):       # right side, left side (mirrored to match)
        a = base + np.linspace(-HALF, HALF, NA)[None, :]
        x, y = ctr[0] + r * np.cos(a), ctr[1] + r * np.sin(a)
        st = np.stack([ndimage.map_coordinates(ref[:, :, k], [y, x], order=1) for k in range(3)], -1)
        strips.append(st[:, ::-1] if flip else st)
    print('iris radius px', round(float(rad), 1))
strip = np.mean(strips, 0)               # NR x NA x 3
# catchlights: far brighter than the median at that radius -> the median
lum = strip.mean(2)
med = np.median(strip, 1, keepdims=True)
bright = lum > np.median(lum, 1, keepdims=True) + 0.12
strip = np.where(bright[:, :, None], np.nan, strip)
profile = np.nanmedian(strip, 1)          # NR x 3: colour by radius

eye = np.asarray(Image.open(eye_path).convert('RGBA')).astype(np.float32) / 255
rgb, alpha = eye[:, :, :3], eye[:, :, 3:]
mx, mn = rgb.max(2), rgb.min(2)
sat = (mx - mn) / np.maximum(mx, 1e-3)
lab, n = ndimage.label(ndimage.binary_closing(sat > 0.35, iterations=3))
sizes = ndimage.sum(np.ones_like(sat), lab, range(1, n + 1))
out = (rgb * np.array([1.25, 1.25, 1.28])).clip(0, 1)   # a clean, bright white of the eye (the viewer adds the lid shadow)
H, W = sat.shape
yy, xx = np.mgrid[0:H, 0:W]
for lbl in np.argsort(sizes)[::-1][:2] + 1:
    ys, xs = np.where(lab == lbl)
    cy, cx, R = ys.mean(), xs.mean(), np.sqrt(len(ys) / np.pi)
    S = float(os.environ.get('IRIS_SCALE', '1.15'))   # a larger iris than MPFB's, so the upper lid covers its top
    d0 = np.hypot(yy - cy, xx - cx) / R
    d = d0 / S
    tex_lum = rgb.mean(2)
    bins0 = np.round((d0 / 1.08 * (NR - 1)).clip(0, NR - 1)).astype(int)
    inside = d0 < 1.08
    prof = ndimage.mean(tex_lum, bins0 * inside + (~inside) * NR, range(NR))
    lum_s = ndimage.map_coordinates(tex_lum, [cy + (yy - cy) / S, cx + (xx - cx) / S], order=1)
    ri = (d / 1.08 * (NR - 1)).clip(0, NR - 1)
    detail = (lum_s / np.maximum(np.take(prof, np.round(ri).astype(int)), 1e-3)).clip(0.5, 1.6) ** 0.8
    col = np.stack([np.interp(ri, np.arange(NR), profile[:, k]) for k in range(3)], -1) * detail[:, :, None]
    pupil = np.clip((0.36 - d) / 0.05, 0, 1)[:, :, None]   # about 0.36 of the iris radius, as in the reference
    col = col * (1 - pupil) + 0.025 * pupil
    w = np.clip((1.06 - d) / 0.08, 0, 1)[:, :, None]
    out = out * (1 - w) + col * w
    print('texture iris', round(cx), round(cy), 'radius', round(R))
Image.fromarray((np.concatenate([out.clip(0, 1), alpha], 2) * 255).astype(np.uint8)).save(out_path)
