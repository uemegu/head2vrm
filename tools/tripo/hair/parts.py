"""Which parts of a Tripo GLB are the head and which the hair, by what their textures show (not by size: in one GLB the hair is the
largest part, in another the face). -> OUT_DIR/parts.json, and OUT_DIR/hair.glb when a hair part is found.

blender --background --python parts.py -- GLB OUT_DIR
env HEAD_PART: auto (default) or a part number; HAIR_PART: auto (default), none, or a part number.

Each part with MIN_VERTS vertices or more: the share of its surface whose texture is skin (the hair_cut.py test, sampled at 7 points
on every face), and the faces' median colour. The head is the part with the most skin; the hair is the largest other part that
reaches the head's top and is either under HAIR_MAX_SKIN skin or darker than HAIR_DARKER of the head (brown hair is skin-coloured
by the test: one test head's hair 0.71 skin, but its median lightness 120 against the face's 207). Blond or white hair may need HAIR_PART (its top within 15% of the head's height of the head's top). Small parts (eyes, teeth, scraps) are left out.
"""
import json
import os
import sys

import bmesh
import bpy
import numpy as np

glb, out = sys.argv[sys.argv.index('--') + 1:]
MIN_VERTS = int(os.environ.get('MIN_VERTS', 300))
HAIR_MAX_SKIN = float(os.environ.get('HAIR_MAX_SKIN', 0.35))
HAIR_DARKER = float(os.environ.get('HAIR_DARKER', 0.75))
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
parts = {int(o.name.split('_')[-1]) if o.name.split('_')[-1].isdigit() else i: o
         for i, o in enumerate(o for o in bpy.data.objects if o.type == 'MESH')}
BARY = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])


def skin_share(o):
    """(share of faces whose texture is skin-coloured, the faces' median sRGB colour)"""
    img = next((n.image for m in o.data.materials if m and m.node_tree for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image), None)
    if img is None:
        return 0.0, [0, 0, 0], None
    W, H = img.size
    px = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(H, W, 4)[..., :3]
    srgb = (np.where(px <= 0.0031308, px * 12.92, 1.055 * np.power(np.clip(px, 0, None), 1 / 2.4) - 0.055) * 255
            if img.colorspace_settings.name != 'sRGB' else px * 255)
    R, B = srgb[..., 0], srgb[..., 2]
    skin = ((R - B > 28) & (R > 60)) | (((R - B) > 0.28 * R) & (R > 32))
    bm = bmesh.new(); bm.from_mesh(o.data); uvl = bm.loops.layers.uv.active
    if uvl is None:
        return 0.0, [0, 0, 0], None
    hits, cols = [], []
    for f in bm.faces:
        uv = np.array([l[uvl].uv[:] for l in f.loops])[:3]
        s = BARY @ uv
        x = np.clip((s[:, 0] % 1 * W).astype(int), 0, W - 1); y = np.clip((s[:, 1] % 1 * H).astype(int), 0, H - 1)
        hits.append(skin[y, x].mean()); cols.append(srgb[y[0], x[0]].tolist())
    bm.free()
    return (float(np.mean(hits)), np.median(cols, 0).round(1).tolist(), feat_stats(np.array(cols))) if hits else (0.0, [0, 0, 0], None)


def feat_stats(rgb):
    """[mean L, sd L, mean warmth, sd warmth] of the part's colours; hair_cut.py / hair_skin_out.py tell skin from hair with them"""
    L = rgb.mean(1); w = (rgb[:, 0] - rgb[:, 2]) / (rgb[:, 0] + 10)
    return [round(float(L.mean()), 2), round(float(L.std()) + 1, 2), round(float(w.mean()), 4), round(float(w.std()) + 0.01, 4)]


rows = []
for pid, o in sorted(parts.items()):
    nv = len(o.data.vertices)
    co = np.array([o.matrix_world @ v.co for v in o.data.vertices])
    r = {'part': pid, 'name': o.name, 'verts': nv, 'zmin': float(co[:, 2].min()), 'zmax': float(co[:, 2].max()),
         'skin': None, 'rgb': None}
    if nv >= MIN_VERTS:
        r['skin'], r['rgb'], r['stats'] = skin_share(o)
    rows.append(r)
big = [r for r in rows if r['skin'] is not None]
if not big:
    raise SystemExit(f'PARTS no part with {MIN_VERTS}+ vertices in {glb}')

hp = os.environ.get('HEAD_PART', 'auto')
head = max(big, key=lambda r: r['skin']) if hp == 'auto' else next(r for r in rows if r['part'] == int(hp))
hr = os.environ.get('HAIR_PART', 'auto')
hair = None
if hr == 'auto':
    reach = head['zmax'] - 0.15 * (head['zmax'] - head['zmin'])
    L = lambda r: float(np.mean(r['rgb']))
    # hair: little skin colour, or much darker than the face (brown hair passes the skin-colour test)
    cand = [r for r in big if r is not head and r['zmax'] >= reach and (r['skin'] < HAIR_MAX_SKIN or L(r) < HAIR_DARKER * L(head))]
    hair = max(cand, key=lambda r: r['verts']) if cand else None
elif hr != 'none':
    hair = next(r for r in rows if r['part'] == int(hr))

for r in rows:
    tag = 'HEAD' if r is head else 'HAIR' if r is hair else ''
    sk = '-' if r['skin'] is None else '%.2f' % r['skin']
    print('PARTS part %3d  verts %5d  skin %s  colour %s  z %.3f..%.3f  %s' % (r['part'], r['verts'], sk, r['rgb'] or '-', r['zmin'], r['zmax'], tag))
if hair:
    h, f = parts[hair['part']], parts[head['part']]
    # the hair in the head part's frame (the parts may be placed differently in the GLB): the levelling and the fit then move both alike
    if not np.allclose(np.array(h.matrix_world), np.array(f.matrix_world)):
        M = f.matrix_world.inverted() @ h.matrix_world
        for v in h.data.vertices:
            v.co = M @ v.co
        h.parent = None; h.matrix_world = f.matrix_world.copy()
        print('PARTS hair moved into the head part\'s frame')
    for o in list(bpy.data.objects):
        if o is not h:
            bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action='DESELECT'); h.select_set(True); bpy.context.view_layer.objects.active = h
    bpy.ops.export_scene.gltf(filepath=os.path.join(out, 'hair.glb'), use_selection=True, export_format='GLB', export_image_format='AUTO')
    print('PARTS hair part %d -> %s' % (hair['part'], os.path.join(out, 'hair.glb')))
else:
    print('PARTS no hair part (HAIR_PART=%s)' % hr)
json.dump({'head': head['part'], 'hair': hair['part'] if hair else None, 'skin_rgb': head['rgb'], 'hair_rgb': hair['rgb'] if hair else None,
           'skin_stats': head['stats'], 'hair_stats': hair['stats'] if hair else None, 'parts': rows}, open(os.path.join(out, 'parts.json'), 'w'), indent=1)
