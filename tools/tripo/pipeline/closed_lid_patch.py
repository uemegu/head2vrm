"""The closed lids' skin without the painted lash tips, as a small patch the viewer blends in by how far each eye is closed
(closed-lid.js). python closed_lid_patch.py OUT_DIR SKIN.png ASSETS_DIR NAME   -> ASSETS_DIR/NAME.png, NAME.json

Why: long lashes painted on the lid's skin texture are stretched into stray lines when the eye closes. The skin that covers the
eye when it is closed is the lid 3-7 mm above the open eye (closed_lid_vis.py: every camera's closed render, the pixels inside
that camera's open eye, and the lid skin that moved over MOVED texels between open and closed), and the long painted lashes reach into it. The patch: those texels (grown by GROW px), the dark
band's skin replaced by the local skin colour (a normalised blur of the texels that
are not dark), the rest as it is; alpha the band, feathered. One rect per eye (its 3D side from texels.npz) for the viewer.
"""
import json, os, sys
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
O, skin, adir, name = sys.argv[1:5]
E = lambda k, d: float(os.environ.get(k, d))
GROW, SIG, DARK, MOVED = int(E('GROW', 3)), E('SIGMA', 5), E('DARK', 0.9), E('MOVED', 4)
D = f'{O}/closedlid'
img = cv2.imread(skin).astype(np.float32) / 255; H, W = img.shape[:2]
rd = lambda f: cv2.imread(f, cv2.IMREAD_UNCHANGED)[..., 2::-1]
band = np.zeros((H, W), np.uint8)
for cam in ('front', 'up', 'down', 'left', 'right'):
    a, b = rd(f'{D}/uv_open_{cam}.exr'), rd(f'{D}/uv_closed_{cam}.exr')
    lid = (a[..., 2] < 0.5) & (b[..., 2] > 0.5)
    # and the lid skin that moves on closing (over MOVED texels between the two renders), above the open eye too
    moved = (a[..., 2] > 0.5) & (b[..., 2] > 0.5) & (np.hypot(*(a[..., :2] - b[..., :2]).transpose(2, 0, 1)) * W > MOVED)
    # the upper lid only: in the columns over the open eye, above its lower edge (the lower lid and the corners keep theirs)
    hole = a[..., 2] < 0.5; rows = np.arange(hole.shape[0])[:, None]
    bottom = np.where(hole.any(0), hole.shape[0] - 1 - np.argmax(hole[::-1], 0), -1)
    lid |= moved & (rows < bottom[None, :])
    band[((1 - b[lid, 1]) * H).astype(int).clip(0, H - 1), (b[lid, 0] * W).astype(int).clip(0, W - 1)] = 1
band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(band)
band = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 200]).astype(np.uint8)
grown = cv2.dilate(band, np.ones((2 * GROW + 1,) * 2, np.uint8))
ys, xs = np.nonzero(grown); y0, y1, x0, x1 = ys.min() - 16, ys.max() + 17, xs.min() - 16, xs.max() + 17
c = img[y0:y1, x0:x1]; g = grown[y0:y1, x0:x1] > 0
L = c.mean(2)
# the local skin: a normalised blur of the light texels, twice (the second without the first's dark ones)
ok = g.astype(np.float32)          # the lid's own skin only (not the eye or brow beside it in the texture)
for _ in range(2):
    ref = cv2.GaussianBlur(c * ok[..., None], (0, 0), SIG) / np.maximum(cv2.GaussianBlur(ok, (0, 0), SIG), 1e-4)[..., None]
    ok = ((L >= DARK * ref.mean(2)) & g).astype(np.float32)
# the band all smooth skin: the faint tips (a few % darker) slipped through a darkness test and still read as lines
w = np.ones_like(L)
out = c + (ref - c) * w[..., None]
alpha = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
rgba = np.dstack([out[..., ::-1], alpha]) if False else np.dstack([out, alpha])   # BGR + A for cv2
cv2.imwrite(f'{adir}/{name}.png', (np.clip(rgba, 0, 1) * 255).round().astype(np.uint8))
# per eye: the band's parts, and which side of the head they lie on
t = np.load(f'{O}/texels.npz'); pos, have = t['pos'], t['have']
eyes = []
n, lab, st, _ = cv2.connectedComponentsWithStats(grown)
for i in range(1, n):
    m = lab == i; yy, xx = np.nonzero(m)
    h = have[yy, xx]; side = float(np.sign(np.median(pos[yy[h], xx[h], 0]))) if h.any() else 0.0
    eyes.append({'rect': [int(xx.min()) / W, int(yy.min()) / H, (int(xx.max()) + 1) / W, (int(yy.max()) + 1) / H], 'side': side, 'texels': int(m.sum())})
# one rect per side (a side's band can come in pieces)
by = {}
for e in eyes:
    r = by.setdefault(e['side'], {'rect': list(e['rect']), 'side': e['side'], 'texels': 0})
    r['rect'] = [min(r['rect'][0], e['rect'][0]), min(r['rect'][1], e['rect'][1]), max(r['rect'][2], e['rect'][2]), max(r['rect'][3], e['rect'][3])]
    r['texels'] += e['texels']
eyes = list(by.values())
import time
meta = {'image': f'{name}.png?v={int(time.time())}', 'rect': [x0 / W, y0 / H, x1 / W, y1 / H], 'eyes': eyes}
json.dump(meta, open(f'{adir}/{name}.json', 'w'), indent=1)
print('CLOSED_PATCH', meta['rect'], [(e['side'], e['texels']) for e in eyes], 'strokes darkened texels %.1f%%' % (100 * (w[g] > 0.5).mean()))
cv2.imwrite(f'{O}/closedlid/patch_check.png', cv2.resize(np.concatenate([(c * 255).astype(np.uint8), (np.clip(out, 0, 1) * 255).astype(np.uint8), cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)], 1), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
