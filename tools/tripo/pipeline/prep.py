"""Part-split bald Tripo head (GLB) -> head.glb: one mesh, one 4096 atlas, every part's texture copied
pixel for pixel (no rescale; the face keeps its 2048 texels). Parts not listed are dropped.
blender -b -P prep.py -- GLB OUT_DIR PART:X0,Y0 [PART:X0,Y0 ...]     (X0,Y0: the cell's corner in the atlas, 0..1, v up)
"""
import os, sys
import bpy
import numpy as np
args = sys.argv[sys.argv.index('--') + 1:]
glb, out, cells = args[0], args[1], args[2:]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
parts = {int(o.name.split('_')[-1]): o for o in bpy.data.objects if o.type == 'MESH'}
S = 4096
atlas = np.zeros((S, S, 4), np.float32); atlas[..., 3] = 1
used = []
for c in cells:
    pid, xy = c.split(':'); pid = int(pid); x0, y0 = map(float, xy.split(','))
    o = parts[pid]
    img = next(n for n in o.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE').image
    w, h = img.size
    a = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
    X, Y = int(S * x0), int(S * y0)
    assert X + w <= S and Y + h <= S
    atlas[Y:Y + h, X:X + w] = a
    uv = o.data.uv_layers.active.data
    arr = np.zeros(len(uv) * 2); uv.foreach_get('uv', arr); arr = arr.reshape(-1, 2)
    arr = np.clip(arr, 0, 1) * [w / S, h / S] + [x0, y0]
    uv.foreach_set('uv', arr.ravel())
    used.append(o)
    print('PREP part', pid, len(o.data.vertices), (w, h), 'at', X, Y)
for pid, o in parts.items():
    if o not in used:
        print('PREP dropped part', pid, len(o.data.vertices))
        bpy.data.objects.remove(o)
timg = bpy.data.images.new('head', S, S, alpha=True)
timg.pixels.foreach_set(atlas.ravel())
timg.filepath_raw = os.path.join(out, 'head_atlas.png'); timg.file_format = 'PNG'; timg.save()
mat = used[0].data.materials[0]
next(n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE').image = timg
mat.name = 'head'
for o in used:
    o.data.materials.clear(); o.data.materials.append(mat)
bpy.ops.object.select_all(action='DESELECT')
for o in used: o.select_set(True)
bpy.context.view_layer.objects.active = used[0]
bpy.ops.object.join()
used[0].name = 'head'
bpy.ops.export_scene.gltf(filepath=os.path.join(out, 'head.glb'), use_selection=True, export_format='GLB')
print('PREP head.glb', len(used[0].data.vertices))
