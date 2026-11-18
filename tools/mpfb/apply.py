"""Apply the fitted warp to every mesh of the character (shape keys re-derived), then save.

blender --background base.blend --python apply.py -- WARP_NPZ OUT_BLEND [strength]
Eyeballs move rigidly with the warp at their centre. The MPFB eyebrows are removed
(the brows come from the baked video colours).
"""
import os
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index('--') + 1:]
w = np.load(args[0])
out = args[1]
strength = float(args[2]) if len(args) > 2 else 1.0
C, W, sigma, z_lo, z_hi = w['centers'], w['W'] * strength, float(w['sigma']), float(w['z_lo']), float(w['z_hi'])


def disp(x):
    t = np.clip((x[:, 2] - z_lo) / (z_hi - z_lo), 0, 1)
    out = np.zeros_like(x)
    for s in range(0, len(x), 20000):
        xs = x[s:s + 20000]
        d2 = ((xs[:, None, :] - C[None]) ** 2).sum(-1)
        out[s:s + 20000] = np.exp(-d2 / (2 * sigma ** 2)) @ W
    return (t * t * (3 - 2 * t))[:, None] * out


def read(coll, n):
    a = np.zeros(n * 3)
    coll.foreach_get('co', a)
    return a.reshape(n, 3)


for obj in [o for o in bpy.data.objects if o.type == 'MESH']:
    if 'eyebrow' in obj.name:
        bpy.data.objects.remove(obj)
        continue
    mw = np.array(obj.matrix_world)
    inv = np.linalg.inv(mw)
    n = len(obj.data.vertices)
    to_w = lambda p: p @ mw[:3, :3].T + mw[:3, 3]
    to_l = lambda p: p @ inv[:3, :3].T + inv[:3, 3]
    base = to_w(read(obj.data.vertices, n))
    rigid = 'high-poly' in obj.name
    if rigid:
        # each eyeball separately
        d = np.zeros_like(base)
        es = w['eye_scale'] if 'eye_scale' in w.files else (1.0, 1.0)
        # MediaPipe 33/133 is the subject's right eye (Blender -X); 362/263 the left (+X)
        for side, sc in ((base[:, 0] <= 0, es[0]), (base[:, 0] > 0, es[1])):
            ctr = base[side].mean(0, keepdims=True)
            sc = 1 + (sc - 1) * strength
            d[side] = disp(ctr) + (base[side] - ctr) * (sc - 1)
        d[:, 2] += float(os.environ.get('EYE_LIFT', '0'))   # raise the eyeballs (metres) so the upper lid covers more iris
    else:
        d = disp(base)
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
    new_keys = []
    for kb in keys[1:]:
        k = to_w(read(kb.data, n))
        dk = d if rigid else disp(k)  # eye meshes have no shape keys
        new_keys.append(to_l(k + dk))
    newbase = to_l(base + d)
    obj.data.vertices.foreach_set('co', newbase.ravel())
    if keys:
        keys[0].data.foreach_set('co', newbase.ravel())
        for kb, k in zip(keys[1:], new_keys):
            kb.data.foreach_set('co', k.ravel())
    obj.data.update()
    print('WARPED', obj.name, round(float(np.linalg.norm(d, axis=1).max() * 1000), 2), 'mm')

# Close the lips a little: subtract part of the mouth-open key from every mesh that has it
# (body, teeth, tongue move together), keeping the other keys relative to the new basis.
# Lowering the upper lids a little (EYE_CLOSE, a fraction of the blink keys) works the same way, with + sign.
def bake_key(names, amount, label):
    for obj in [o for o in bpy.data.objects if o.type == 'MESH' and o.data.shape_keys]:
        keys = obj.data.shape_keys.key_blocks
        if not all(nm in keys for nm in names):
            continue
        n = len(obj.data.vertices)
        base = read(keys[0].data, n)
        delta = sum(read(keys[nm].data, n) - base for nm in names)
        for kb in keys:
            kb.data.foreach_set('co', (read(kb.data, n) + amount * delta).ravel())
        obj.data.vertices.foreach_set('co', (base + amount * delta).ravel())
        obj.data.update()
        print(label, obj.name, amount)


close = float(os.environ.get('MOUTH_CLOSE', '0'))
if close:
    bake_key(['aa'], -close, 'MOUTH')
eye_close = float(os.environ.get('EYE_CLOSE', '0'))
if eye_close:
    bake_key(['blink_l', 'blink_r'], eye_close, 'EYES')

bpy.ops.wm.save_as_mainfile(filepath=out)
