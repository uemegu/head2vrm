"""Drop the faces of a Tripo hair part whose texture is skin (temple, sideburn and ear bits Tripo put in the hair part);
painting them hair-coloured left flat dark flaps by the temples (2026-11-11). Skin as in hair_skin_out.py.
blender --background --python hair_cut.py -- IN.glb OUT.glb   (CUT: share of skin samples to drop a face, default 0.5)"""
import os, sys
import bpy, bmesh
import numpy as np
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
src, dst = sys.argv[sys.argv.index('--') + 1:]
CUT = float(os.environ.get('CUT', 0.5))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
o = next(o for o in bpy.data.objects if o.type == 'MESH')
img = next(n.image for n in o.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE')
W, H = img.size
px = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(H, W, 4)[..., :3]
srgb = np.where(px <= 0.0031308, px * 12.92, 1.055 * np.power(np.clip(px, 0, None), 1 / 2.4) - 0.055) * 255 if img.colorspace_settings.name != 'sRGB' else px * 255
skin = skin_test(srgb)
bm = bmesh.new(); bm.from_mesh(o.data)
uvl = bm.loops.layers.uv.active
bary = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
drop = []
for f in bm.faces:
    uv = np.array([l[uvl].uv[:] for l in f.loops])
    if len(uv) != 3: uv = uv[:3]
    s = bary @ uv
    x = np.clip((s[:, 0] * W).astype(int), 0, W - 1); y = np.clip((s[:, 1] * H).astype(int), 0, H - 1)   # Blender pixels: row 0 at the bottom
    if skin[y, x].mean() >= CUT: drop.append(f)
print('HAIR_CUT faces', len(bm.faces), 'dropped', len(drop))
bmesh.ops.delete(bm, geom=drop, context='FACES')
bm.to_mesh(o.data); o.data.update()
bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.context.view_layer.objects.active = o
bpy.ops.export_scene.gltf(filepath=dst, use_selection=True, export_format='GLB', export_image_format='AUTO')
