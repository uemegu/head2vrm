"""The skin-coloured scraps in a Tripo hair texture (bits of face/ears baked into the hair part) -> hair.
python hair_skin_out.py IN.glb OUT.glb [CHECK.jpg]
Skin: warm (R-B > 28, R > 60) or shaded skin ((R-B) > 0.28 R, R > 32); grown 3 px. Filled with hair from the nearest unmasked pixel along
a few offsets (keeps the strand texture), then darkened toward the hair's median so no bright copy stands out."""
import io, sys
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'look'))
import os
import glb_tex
def skin_test(rgb):
    """sRGB 0..255 (..., 3) -> bool. With SKIN_STATS and HAIR_STATS (hair/parts.py: mean and spread of the lightness and the warmth
    (R-B)/(R+10) of the head's and the hair's colours): each pixel goes to the nearer of the two, distances in each one's own spread
    (dark brown hair is as warm as skin, shaded skin as dark as light hair; one fixed test fails one or the other).
    Without them the fixed test for dark hair."""
    if os.environ.get('SKIN_STATS') and os.environ.get('HAIR_STATS'):
        L = rgb.mean(-1); w = (rgb[..., 0] - rgb[..., 2]) / (rgb[..., 0] + 10)
        def d(st):
            mL, sL, mw, sw = (float(v) for v in st.split(','))
            return ((L - mL) / sL) ** 2 + ((w - mw) / sw) ** 2 + 2 * np.log(sL * sw)
        return d(os.environ['SKIN_STATS']) < d(os.environ['HAIR_STATS'])
    R, B = rgb[..., 0], rgb[..., 2]
    return ((R - B > 28) & (R > 60)) | (((R - B) > 0.28 * R) & (R > 32))
src, dst = sys.argv[1:3]
j, b = glb_tex.load(src)
name = j['images'][0]['name']
im = np.asarray(Image.open(io.BytesIO(glb_tex.image(j, b, name))).convert('RGB')).astype(np.float32)
R, G, B = im[..., 0], im[..., 1], im[..., 2]
skin = skin_test(im)
skin = ndimage.binary_opening(skin, iterations=1)
mask = ndimage.binary_dilation(skin, iterations=3)
hair = ~mask & (im.max(2) > 8)          # not the black padding between islands
print('HAIR_SKIN_OUT skin px', int(skin.sum()), 'masked', int(mask.sum()), f'{mask.mean()*100:.1f}%')
out = im.copy()
todo = mask.copy()
H, W = mask.shape
for r in (24, 48, 96, 160, 256, 400):
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        dy, dx = int(round(r * np.sin(ang))), int(round(r * np.cos(ang)))
        ys, xs = np.nonzero(todo)
        y2, x2 = np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)
        ok = hair[y2, x2]
        out[ys[ok], xs[ok]] = im[y2[ok], x2[ok]]
        todo[ys[ok], xs[ok]] = False
    if not todo.any(): break
med = np.median(im[hair], axis=0)
out[todo] = med
print('HAIR_SKIN_OUT left to median', int(todo.sum()), 'hair median', med.round(1))
# soften the patch edges
soft = ndimage.gaussian_filter(mask.astype(np.float32), 1.5)[..., None]
res = im * (1 - soft) + out * soft
res = np.where(mask[..., None], out, res)
buf = io.BytesIO(); Image.fromarray(res.clip(0, 255).astype(np.uint8)).save(buf, 'PNG')
glb_tex.save(dst, j, b, {name: buf.getvalue()})
if len(sys.argv) > 3:
    a = Image.fromarray(im.astype(np.uint8)).resize((1024, 1024)); c = Image.fromarray(res.clip(0, 255).astype(np.uint8)).resize((1024, 1024))
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize((1024, 1024)).convert('RGB')
    S = Image.new('RGB', (3072, 1024)); S.paste(a, (0, 0)); S.paste(m, (1024, 0)); S.paste(c, (2048, 0)); S.save(sys.argv[3], quality=88)
