"""Paint the skin under the hair (scalp_mask.py) the hair's root colour, so the gaps in the Tripo hair show dark, not skin.
python scalp_paint.py OUT_DIR SKIN_IN.png SKIN_OUT.png   (OUT_DIR: scalp_mask.npy, texels.npz)
ROOT_RGB sRGB (default 34,27,23: a little above the hair texture's median 20,18,17), FADE texels of soft edge at 4k."""
import os, sys, shutil
import numpy as np
from PIL import Image
from scipy import ndimage
O, src, dst = sys.argv[1:4]
img = np.asarray(Image.open(src).convert('RGB'), np.float32)
m = np.load(f'{O}/scalp_mask.npy')
have = np.load(f'{O}/texels.npz')['have'][::4, ::4]
m = ndimage.binary_closing(m > 0, iterations=2) & have          # fill single missed texels
m = ndimage.binary_opening(m, iterations=1)
m4 = np.kron(m.astype(np.float32), np.ones((4, 4), np.float32))
a = ndimage.gaussian_filter(m4, float(os.environ.get('FADE', 8)))
a = np.clip(a * 1.6, 0, 1)[..., None]
root = np.array([float(v) for v in os.environ.get('ROOT_RGB', '34,27,23').split(',')], np.float32)
rng = np.random.default_rng(1)
noise = ndimage.gaussian_filter(rng.normal(0, 1, img.shape[:2]).astype(np.float32), 1.0)[..., None] * 18
col = np.clip(root + noise, 0, 255)
out = img * (1 - a) + col * a
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(dst)
print('SCALP_PAINT texels', int(m4.sum()))
