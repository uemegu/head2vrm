"""Hair from a second GLB (HAIR_GLB): the bald input head and its haired version of the same model. The two GLBs
may differ by a scale and an offset (e.g. each exported at its own unit height). The haired GLB's other parts (its skin parts: face pieces, neck) are fitted onto
the bald head by a similarity transform (scale, rotation, offset; iterative closest points from several starting scales, the best
kept), and its hair part is moved by the same transform into the bald head part's frame.

blender --background --python hair_align.py -- HEAD_GLB HEAD_PARTS_JSON HAIR_GLB HAIR_PARTS_JSON OUT_HAIR_GLB
  HEAD_PARTS_JSON / HAIR_PARTS_JSON: parts.py's output for each GLB (which part is the head, which the hair).
Stops when the fit is poor (median distance over ALIGN_MAX of the head's height): the two GLBs are then not the same model.
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, kdtree

head_glb, head_js, hair_glb, hair_js, out = sys.argv[sys.argv.index('--') + 1:]
ALIGN_MAX = float(os.environ.get('ALIGN_MAX', 0.01))
hj, rj = json.load(open(head_js)), json.load(open(hair_js))
if rj['hair'] is None:
    raise SystemExit('HAIR_ALIGN no hair part in %s (set HAIR_PART)' % hair_glb)


def load(glb):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    return {int(o.name.split('_')[-1]) if o.name.split('_')[-1].isdigit() else i: o
            for i, o in enumerate(o for o in bpy.data.objects if o.type == 'MESH')}


world = lambda o: np.array([o.matrix_world @ v.co for v in o.data.vertices])

parts = load(head_glb)
head = parts[hj['head']]
tgt = world(head); head_mw = np.array(head.matrix_world)
height = tgt[:, 2].max() - tgt[:, 2].min()

parts = load(hair_glb)
# only the skin parts (parts.py's skin share): the inner hair layer, hair clips and eyeballs are not on the bald head's surface
skin_parts = [r['part'] for r in rj['parts'] if r['skin'] is not None and r['skin'] >= 0.5 and r['part'] != rj['hair']]
if not skin_parts:
    skin_parts = [k for k, o in parts.items() if k != rj['hair'] and len(o.data.vertices) >= 60]
src = np.concatenate([world(parts[k]) for k in skin_parts])
print('HAIR_ALIGN fitted by parts', skin_parts)

tree = kdtree.KDTree(len(tgt))
for i, p in enumerate(tgt):
    tree.insert(p, i)
tree.balance()
near = lambda Y: np.array([tree.find(p)[1:] for p in Y], dtype=object)


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / Xc.var(0).sum()
    return s, R, my - s * R @ mx


best = None
sub = src[np.linspace(0, len(src) - 1, min(len(src), 1500)).astype(int)]
s0 = height / (src[:, 2].max() - src[:, 2].min())
for k in (0.8, 0.9, 1.0, 1.1, 1.25, 1.4):
    s, R, t = s0 * k, np.eye(3), tgt.mean(0) - s0 * k * src.mean(0)
    for _ in range(60):
        Y = s * sub @ R.T + t; nd = near(Y); idx = nd[:, 0].astype(int); d = nd[:, 1].astype(float)
        keep = d <= np.percentile(d, 85)
        s, R, t = umeyama(sub[keep], tgt[idx[keep]])
    d = near(s * sub @ R.T + t)[:, 1].astype(float); med = float(np.median(d))
    if best is None or med < best[0]:
        best = (med, s, R, t)
med, s, R, t = best
print('HAIR_ALIGN scale %.4f, rotation %.2f deg, offset %s; median distance %.4f (%.2f%% of the head height)'
      % (s, np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))), np.round(t, 4).tolist(), med, 100 * med / height))
if med > ALIGN_MAX * height:
    raise SystemExit('HAIR_ALIGN the haired GLB does not sit on the head (median %.2f%% of its height, limit %.2f%%): '
                     'not the same model?' % (100 * med / height, 100 * ALIGN_MAX))

# the hair: world -> fitted onto the head -> the head part's own frame (as parts.py writes a hair part found in the head's GLB)
h = parts[rj['hair']]
M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t
L = Matrix((np.linalg.inv(head_mw) @ M @ np.array(h.matrix_world)).tolist())
for v in h.data.vertices:
    v.co = L @ v.co
h.parent = None; h.matrix_world = Matrix(head_mw.tolist())
for o in list(bpy.data.objects):
    if o is not h:
        bpy.data.objects.remove(o)
bpy.ops.object.select_all(action='DESELECT'); h.select_set(True); bpy.context.view_layer.objects.active = h
bpy.ops.export_scene.gltf(filepath=out, use_selection=True, export_format='GLB', export_image_format='AUTO')
print('HAIR_ALIGN hair part %d of %s -> %s' % (rj['hair'], os.path.basename(hair_glb), out))
