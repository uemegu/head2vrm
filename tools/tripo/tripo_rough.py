"""Roughness map (glTF: G channel) for the baked input skin: painted-on hair matte, lips glossy, skin in between.
Without a map the viewer puts its pore roughness (0.3-0.75) and skin sheen on the painted hair too.

python tripo_rough.py TEXELS_NPZ SKIN_PNG WARP_NPZ OUT_PNG [size]
"""
import os, sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

texels, skin, warp, out = sys.argv[1:5]
size = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
tx = np.load(texels)
pos, have = tx['pos'], tx['have']
col = np.asarray(Image.open(skin).convert('RGB').resize(have.shape[::-1]), np.float32) / 255
chin = float(np.load(warp)['z_hi']) + 0.015
mx, mn = col.max(2), col.min(2)
sat = (mx - mn) / (mx + 1e-6)
x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]

rough = np.full(have.shape, 0.55, np.float32)
dark = np.clip((0.4 - mx) / 0.15, 0, 1) * (z > chin + 0.02)          # hair, brows, lash line: matte
rough = rough + (1.0 - rough) * dark
# lips: more saturated and redder than the skin around them (skin sat ~0.29, r-g ~0.16)
box = (np.abs(x) < 0.03) & (z > chin + 0.015) & (z < chin + 0.055) & (y < -0.105)
lip = box * np.clip((sat - 0.33) / 0.08, 0, 1) * np.clip((col[..., 0] - col[..., 1] - 0.17) / 0.05, 0, 1)
if os.environ.get('LIP_MASK'):   # LIP_MASK=npy (lip_paint.py's lip_mask.npy): the painted lips instead of the colour test
    lm = np.load(os.environ['LIP_MASK']).astype(np.float32)
    lip = np.clip(np.asarray(Image.fromarray(lm).resize(have.shape[::-1], Image.BILINEAR)) if lm.shape != have.shape else lm, 0, 1)
rough = rough + (0.35 - rough) * lip
lips = lip > 0.5
rough = gaussian_filter(rough, 2)
g = (np.clip(rough, 0, 1) * 255).round().astype(np.uint8)
img = np.zeros(have.shape + (3,), np.uint8)
img[..., 1] = g
Image.fromarray(img).resize((size, size), Image.BILINEAR).save(out)
print('hair-matte texels', int((dark > 0.5).sum()), 'lip texels', int(lips.sum()))
